from __future__ import annotations

"""MCI World Model v4.5.0 — ROS2Bridge 原型
============================================

手术机器人桥接预研——ROS2 节点收发 JointState ↔ RobotWorldState，
实现从 ROS2 topic 接收关节状态 → CEWM 推演 → 发布预测状态的闭环。

核心能力:
    ROS2Bridge — ROS2↔CEWM 桥接原型
    - start() / stop() — 启动/停止桥接
    - on_joint_state(msg) — 接收 JointState
    - publish_prediction(state) — 发布预测
    - spin_once() — 单次事件循环

设计原则:
    - 零硬依赖: rclpy 为可选依赖，不安装时降级为模拟模式
    - 协议优先: 定义清晰的 ROS2↔CEWM 转换协议
    - 可测试: 不依赖真实 ROS2 环境，所有逻辑可模拟

重要声明:
    Phase 3 交付的是桥接原型和架构验证，不是生产级 ROS2 节点。
    生产部署需要: QoS 配置、DDS 安全、实时线程、硬件在环测试。
    ROS2 为可选依赖: pip install mci-world-model[ros2]
"""


import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from mci_world_model.sdk._robot_state import RobotWorldState

logger = logging.getLogger(__name__)

# ── ROS2 可选依赖 ──
_rclpy_available: bool = False

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState as JointStateMsg

    _rclpy_available = True
except ImportError:
    pass


# =============================================================================
# ROS2BridgeConfig — 桥接配置
# =============================================================================


@dataclass
class ROS2BridgeConfig:
    """ROS2 桥接配置。

    Attributes:
        node_name: ROS2 节点名
        subscribe_topic: 订阅的 JointState topic
        publish_topic: 发布预测的 topic
        queue_size: 消息队列大小
        frame_id: TF frame ID
        simulation_mode: 是否使用模拟模式（无真实 ROS2）
        qos_reliability: QoS 可靠性策略 ('reliable' / 'best_effort')
        qos_durability: QoS 持久性策略 ('transient_local' / 'volatile')
        qos_deadline_ms: QoS deadline (毫秒, 0=不限制)
    """

    node_name: str = "cewm_bridge"
    subscribe_topic: str = "/joint_states"
    publish_topic: str = "/cewm_prediction"
    queue_size: int = 10
    frame_id: str = "base_link"
    simulation_mode: bool = True
    # v4.5.0: QoS 配置预留 (生产部署需 RELIABLE + TRANSIENT_LOCAL)
    qos_reliability: str = "best_effort"
    qos_durability: str = "volatile"
    qos_deadline_ms: int = 0


# =============================================================================
# BridgeState — 桥接状态
# =============================================================================


@dataclass
class BridgeState:
    """桥接状态快照。

    Attributes:
        is_running: 是否正在运行
        messages_received: 接收消息数
        messages_published: 发布消息数
        last_receive_time: 最后接收时间
        last_publish_time: 最后发布时间
        current_robot_state: 当前机器人状态
    """

    is_running: bool = False
    messages_received: int = 0
    messages_published: int = 0
    last_receive_time: float = 0.0
    last_publish_time: float = 0.0
    current_robot_state: RobotWorldState | None = None


# =============================================================================
# ROS2Bridge — ROS2↔CEWM 桥接
# =============================================================================


class ROS2Bridge:
    """ROS2↔CEWM 桥接原型——在 ROS2 JointState 和 CEWM RobotWorldState 之间转换。

    用法 (模拟模式):
        >>> bridge = ROS2Bridge(simulation_mode=True)
        >>> bridge.start()
        >>> # 模拟接收 JointState
        >>> bridge.on_joint_state({
        ...     "name": ["j1", "j2", "j3", "j4", "j5", "j6"],
        ...     "position": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ...     "velocity": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        ...     "effort": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ... })
        >>> state = bridge.current_state
        >>> bridge.stop()

    用法 (真实 ROS2):
        >>> bridge = ROS2Bridge(simulation_mode=False)
        >>> bridge.start()  # 初始化 rclpy + 创建 node
        >>> bridge.spin()   # 阻塞式事件循环
    """

    def __init__(
        self,
        config: ROS2BridgeConfig | None = None,
        simulation_mode: bool | None = None,
        world_model: Any = None,
        on_prediction: Callable[[RobotWorldState], None] | None = None,
    ):
        """
        Args:
            config: 桥接配置
            simulation_mode: 是否使用模拟模式（覆盖 config 中的设置）
            world_model: MCIWorldModel 实例（可选，用于自动 cewm_step）
            on_prediction: 预测回调函数
        """
        self._config = config or ROS2BridgeConfig()
        if simulation_mode is not None:
            self._config.simulation_mode = simulation_mode

        self._world_model = world_model
        self._on_prediction = on_prediction
        self._state = BridgeState()
        self._lock = threading.Lock()

        # ROS2 相关（仅在真实模式下使用）
        self._node: Any = None
        self._subscriber: Any = None
        self._publisher: Any = None

        # 模拟模式回调
        self._sim_callbacks: list[Callable[[dict], None]] = []  # type: ignore

    @property
    def current_state(self) -> RobotWorldState | None:
        """当前机器人状态。"""
        with self._lock:
            return self._state.current_robot_state

    @property
    def is_running(self) -> bool:
        """是否正在运行。"""
        with self._lock:
            return self._state.is_running

    @property
    def config(self) -> ROS2BridgeConfig:
        """当前配置。"""
        return self._config

    def start(self) -> None:
        """启动桥接。"""
        with self._lock:
            if self._state.is_running:
                logger.warning("ROS2Bridge 已在运行")
                return
            self._state.is_running = True

        if self._config.simulation_mode:
            logger.info("ROS2Bridge 启动 (模拟模式)")
        else:
            self._start_ros2()

    def stop(self) -> None:
        """停止桥接。"""
        with self._lock:
            self._state.is_running = False

        if not self._config.simulation_mode and self._node is not None:
            self._stop_ros2()

        logger.info("ROS2Bridge 已停止")

    def on_joint_state(self, msg: dict | Any) -> None:  # type: ignore
        """接收 JointState 消息并转换为 RobotWorldState。

        Args:
            msg: JointState 消息（dict 或 ROS2 JointState 消息）
        """
        robot_state = self._convert_joint_state(msg)

        with self._lock:
            self._state.messages_received += 1
            self._state.last_receive_time = time.monotonic()
            self._state.current_robot_state = robot_state

        # 通知模拟回调
        for cb in self._sim_callbacks:
            try:
                cb(msg if isinstance(msg, dict) else {"raw": True})
            except Exception as e:
                logger.warning("模拟回调异常: %s", e)

        # 如果有 world_model，自动推演
        if self._world_model is not None and robot_state is not None:
            self._auto_predict(robot_state)

    def publish_prediction(self, state: RobotWorldState) -> None:
        """发布预测状态到 ROS2 topic。

        Args:
            state: 预测的机器人状态
        """
        with self._lock:
            self._state.messages_published += 1
            self._state.last_publish_time = time.monotonic()

        if not self._config.simulation_mode and self._publisher is not None:
            self._publish_ros2(state)

        if self._on_prediction is not None:
            try:
                self._on_prediction(state)
            except Exception as e:
                logger.warning("预测回调异常: %s", e)

    def spin_once(self, timeout_sec: float = 0.1) -> None:
        """单次事件循环（仅真实 ROS2 模式）。

        Args:
            timeout_sec: 超时时间
        """
        if not self._config.simulation_mode and _rclpy_available:
            try:
                rclpy.spin_once(self._node, timeout_sec=timeout_sec)
            except Exception as e:
                logger.warning("spin_once 异常: %s", e)

    def register_callback(self, callback: Callable[[dict], None]) -> None:  # type: ignore
        """注册模拟模式回调。

        Args:
            callback: 回调函数，接收 JointState dict
        """
        self._sim_callbacks.append(callback)

    def bridge_state(self) -> BridgeState:
        """获取桥接状态快照。"""
        with self._lock:
            return BridgeState(
                is_running=self._state.is_running,
                messages_received=self._state.messages_received,
                messages_published=self._state.messages_published,
                last_receive_time=self._state.last_receive_time,
                last_publish_time=self._state.last_publish_time,
                current_robot_state=self._state.current_robot_state,
            )

    # ── 内部方法 ──

    def _convert_joint_state(self, msg: dict | Any) -> RobotWorldState | None:  # type: ignore
        """将 JointState 消息转换为 RobotWorldState。"""
        try:
            if isinstance(msg, dict):
                positions = np.array(msg.get("position", []), dtype=np.float64)
                velocities = np.array(msg.get("velocity", []), dtype=np.float64)
                efforts = np.array(msg.get("effort", []), dtype=np.float64)
            else:
                # ROS2 JointState 消息
                positions = np.array(msg.position, dtype=np.float64)
                velocities = np.array(msg.velocity, dtype=np.float64)
                efforts = np.array(msg.effort, dtype=np.float64)

            n = len(positions)
            if n == 0:
                return None

            return RobotWorldState(
                joint_positions=positions,
                joint_velocities=velocities if len(velocities) == n else np.zeros(n),
                joint_efforts=efforts if len(efforts) == n else np.zeros(n),
                n_joints=n,
            )
        except Exception as e:
            logger.error("JointState 转换失败: %s", e)
            return None

    def _auto_predict(self, state: RobotWorldState) -> None:
        """使用 world_model 自动推演。"""
        try:
            result = self._world_model.cewm_step(observation=state)
            predicted = result.get("state")
            if isinstance(predicted, RobotWorldState):
                self.publish_prediction(predicted)
        except Exception as e:
            logger.warning("自动推演失败: %s", e)

    def _start_ros2(self) -> None:
        """启动真实 ROS2 节点。"""
        if not _rclpy_available:
            logger.error("rclpy 不可用，切换到模拟模式")
            self._config.simulation_mode = True
            return

        try:
            if not rclpy.ok():
                rclpy.init()

            self._node = Node(self._config.node_name)

            # 创建订阅者
            self._subscriber = self._node.create_subscription(
                JointStateMsg,
                self._config.subscribe_topic,
                self._ros2_callback,
                self._config.queue_size,
            )

            # 创建发布者
            self._publisher = self._node.create_publisher(
                JointStateMsg,
                self._config.publish_topic,
                self._config.queue_size,
            )

            logger.info(
                "ROS2Bridge 启动 (真实模式): sub=%s, pub=%s",
                self._config.subscribe_topic,
                self._config.publish_topic,
            )
        except Exception as e:
            logger.error("ROS2 初始化失败: %s，切换到模拟模式", e)
            self._config.simulation_mode = True

    def _stop_ros2(self) -> None:
        """停止真实 ROS2 节点。"""
        try:
            if self._node is not None:
                self._node.destroy_node()
                self._node = None
        except Exception as e:
            logger.warning("ROS2 节点销毁异常: %s", e)

    def _ros2_callback(self, msg: Any) -> None:
        """ROS2 订阅回调。"""
        self.on_joint_state(msg)

    def _publish_ros2(self, state: RobotWorldState) -> None:
        """发布 ROS2 JointState 消息。"""
        if not _rclpy_available or self._publisher is None:
            return

        try:
            msg = JointStateMsg()
            msg.header.stamp = self._node.get_clock().now().to_msg()
            msg.header.frame_id = self._config.frame_id
            msg.name = [f"joint_{i}" for i in range(state.n_joints)]
            msg.position = state.joint_positions.tolist() if state.joint_positions is not None else []
            msg.velocity = state.joint_velocities.tolist() if state.joint_velocities is not None else []
            msg.effort = state.joint_efforts.tolist() if state.joint_efforts is not None else []
            self._publisher.publish(msg)
        except Exception as e:
            logger.warning("ROS2 发布失败: %s", e)

    def __repr__(self) -> str:
        state = self.bridge_state()
        mode = "sim" if self._config.simulation_mode else "ros2"
        return (
            f"ROS2Bridge(mode={mode}, running={state.is_running}, "
            f"rx={state.messages_received}, tx={state.messages_published})"
        )
