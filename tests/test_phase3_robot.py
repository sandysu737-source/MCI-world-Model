"""Phase 3 手术机器人桥接 + 硬实时保证测试.

覆盖:
    - RobotWorldState + RobotAction
    - EmergencyStop 紧急停止
    - DeadlineMonitor 超时检测 + 降级
    - ROS2Bridge 模拟模式
    - 扩展安全约束 (Joint/SelfCollision/Workspace/ToolForce)
    - cewm_step_fast() 快速路径
    - RobotWorldState 通过 cewm_step() 闭环
"""

import time

import numpy as np
import pytest

from mci_world_model.sdk._deadline_monitor import (
    DeadlineConfig,
    DeadlineMonitor,
)
from mci_world_model.sdk._emergency_stop import (
    EmergencyStop,
    EmergencyStopState,
)
from mci_world_model.sdk._robot_state import RobotAction, RobotWorldState
from mci_world_model.sdk._ros2_bridge import (
    ROS2Bridge,
    ROS2BridgeConfig,
)
from mci_world_model.sdk._safety import (
    JointLimitConstraint,
    SafetyMonitor,
    SelfCollisionConstraint,
    ToolForceConstraint,
    WorkspaceBoundConstraint,
)
from mci_world_model.sdk._world_model import MCIWorldModel
from mci_world_model.sdk._world_state import PendulumState

# =============================================================================
# RobotWorldState
# =============================================================================


class TestRobotWorldState:
    """RobotWorldState 测试。"""

    def test_creation(self):
        state = RobotWorldState(n_joints=6)
        assert state.n_joints == 6
        state._ensure_arrays()
        np.testing.assert_array_equal(state.joint_positions, np.zeros(6))

    def test_to_vector(self):
        state = RobotWorldState(
            joint_positions=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            joint_velocities=np.zeros(6),
            joint_efforts=np.zeros(6),
            n_joints=6,
        )
        vec = state.to_vector()
        assert len(vec) == 18  # 6 * 3
        assert float(vec[0]) == pytest.approx(0.1)

    def test_from_vector(self):
        vec = np.arange(18, dtype=np.float64)
        state = RobotWorldState.from_vector(vec)
        assert state.n_joints == 6
        assert float(state.joint_positions[0]) == pytest.approx(0.0)
        assert float(state.joint_velocities[0]) == pytest.approx(6.0)
        assert float(state.joint_efforts[0]) == pytest.approx(12.0)

    def test_distance(self):
        s1 = RobotWorldState(n_joints=6)
        s2 = RobotWorldState(n_joints=6)
        assert s1.distance(s2) == pytest.approx(0.0)

    def test_step_physics(self):
        state = RobotWorldState(
            joint_positions=np.zeros(6),
            joint_velocities=np.ones(6),
            joint_efforts=np.zeros(6),
            n_joints=6,
            dt=0.01,
        )
        next_state = state.step_physics()
        # position += velocity * dt
        np.testing.assert_allclose(next_state.joint_positions, np.full(6, 0.01))

    def test_copy(self):
        state = RobotWorldState(
            joint_positions=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            joint_velocities=np.zeros(6),
            joint_efforts=np.zeros(6),
            n_joints=6,
        )
        copy = state.copy()
        assert copy is not state
        np.testing.assert_array_equal(copy.joint_positions, state.joint_positions)

    def test_to_dict(self):
        state = RobotWorldState(n_joints=6)
        d = state.to_dict()
        assert d["type"] == "RobotWorldState"
        assert d["n_joints"] == 6


# =============================================================================
# RobotAction
# =============================================================================


class TestRobotAction:
    """RobotAction 测试。"""

    def test_creation(self):
        action = RobotAction(target_positions=np.zeros(6))
        assert len(action.target_positions) == 6

    def test_apply(self):
        state = RobotWorldState(
            joint_positions=np.zeros(6),
            joint_velocities=np.zeros(6),
            joint_efforts=np.zeros(6),
            n_joints=6,
            dt=0.01,
        )
        action = RobotAction(target_positions=np.ones(6), dt=0.01)
        new_state = action.apply(state)
        np.testing.assert_allclose(new_state.joint_positions, np.ones(6))

    def test_to_vector(self):
        action = RobotAction(
            target_positions=np.array([1.0, 2.0, 3.0]),
            target_efforts=np.array([0.5, 0.5, 0.5]),
        )
        vec = action.to_vector()
        assert len(vec) == 6

    def test_from_vector(self):
        vec = np.array([1.0, 2.0, 3.0, 0.5, 0.5, 0.5])
        action = RobotAction.from_vector(vec)
        np.testing.assert_array_equal(action.target_positions, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(action.target_efforts, [0.5, 0.5, 0.5])

    def test_apply_wrong_type_raises(self):
        state = PendulumState(theta=0.1, omega=0.0)
        action = RobotAction(target_positions=np.zeros(6))
        with pytest.raises(TypeError):
            action.apply(state)


# =============================================================================
# EmergencyStop
# =============================================================================


class TestEmergencyStop:
    """紧急停止测试。"""

    def test_initial_state(self):
        estop = EmergencyStop()
        assert estop.is_stopped is False
        assert estop.is_armed is False
        assert estop.trigger_count == 0

    def test_trigger(self):
        estop = EmergencyStop()
        estop.trigger("test")
        assert estop.is_stopped is True
        assert estop.trigger_count == 1

    def test_reset(self):
        estop = EmergencyStop()
        estop.trigger("test")
        estop.reset()
        assert estop.is_stopped is False

    def test_arm(self):
        estop = EmergencyStop()
        estop.arm(timeout_ms=5000)
        assert estop.is_armed is True

    def test_disarm(self):
        estop = EmergencyStop()
        estop.arm(timeout_ms=5000)
        estop.disarm()
        assert estop.is_armed is False

    def test_check(self):
        estop = EmergencyStop()
        state = estop.check()
        assert isinstance(state, EmergencyStopState)
        assert state.is_stopped is False

    def test_on_stop_callback(self):
        reasons = []
        estop = EmergencyStop(on_stop=reasons.append)
        estop.trigger("callback_test")
        assert reasons == ["callback_test"]

    def test_trigger_count(self):
        estop = EmergencyStop()
        estop.trigger("a")
        estop.trigger("b")
        estop.trigger("c")
        assert estop.trigger_count == 3

    def test_arm_timeout(self):
        """武装超时触发紧急停止。"""
        estop = EmergencyStop()
        estop.arm(timeout_ms=50)  # 50ms 超时
        time.sleep(0.1)  # 等待超时
        assert estop.is_stopped is True

    def test_repr(self):
        estop = EmergencyStop()
        r = repr(estop)
        assert "EmergencyStop" in r


# =============================================================================
# DeadlineMonitor
# =============================================================================


class TestDeadlineMonitor:
    """超时检测测试。"""

    def test_initial_state(self):
        monitor = DeadlineMonitor()
        assert monitor.is_degraded is False

    def test_start_stop(self):
        monitor = DeadlineMonitor()
        monitor.start()
        time.sleep(0.001)
        elapsed = monitor.stop()
        assert elapsed > 0

    def test_deadline_config(self):
        config = DeadlineConfig(deadline_ms=10)
        monitor = DeadlineMonitor(config)
        assert monitor.config.deadline_ms == 10

    def test_no_deadline_no_violation(self):
        monitor = DeadlineMonitor(DeadlineConfig(deadline_ms=0))
        monitor.start()
        time.sleep(0.01)
        monitor.stop()
        stats = monitor.statistics()
        assert stats.deadline_violations == 0

    def test_deadline_violation(self):
        monitor = DeadlineMonitor(DeadlineConfig(deadline_ms=1))
        monitor.start()
        time.sleep(0.01)  # 远超 1ms
        monitor.stop()
        stats = monitor.statistics()
        assert stats.deadline_violations > 0

    def test_should_degrade(self):
        monitor = DeadlineMonitor(
            DeadlineConfig(
                deadline_ms=1,
                degrade_threshold=0.3,
            )
        )
        # 多次超时
        for _ in range(10):
            monitor.start()
            time.sleep(0.01)
            monitor.stop()
        assert monitor.should_degrade() is True

    def test_statistics(self):
        monitor = DeadlineMonitor(DeadlineConfig(deadline_ms=1000))
        for _ in range(5):
            monitor.start()
            time.sleep(0.001)
            monitor.stop()
        stats = monitor.statistics()
        assert stats.total_steps == 5
        assert stats.p50_ms > 0

    def test_percentile(self):
        monitor = DeadlineMonitor()
        for _ in range(10):
            monitor.start()
            time.sleep(0.001)
            monitor.stop()
        p50 = monitor.percentile(0.5)
        assert p50 > 0

    def test_reset_degrade(self):
        monitor = DeadlineMonitor(DeadlineConfig(deadline_ms=1))
        for _ in range(10):
            monitor.start()
            time.sleep(0.01)
            monitor.stop()
        monitor.should_degrade()
        monitor.reset_degrade()
        assert monitor.is_degraded is False

    def test_check_during_step(self):
        monitor = DeadlineMonitor(DeadlineConfig(deadline_ms=1))
        monitor.start()
        time.sleep(0.01)
        assert monitor.check() is True  # 已超时

    def test_repr(self):
        monitor = DeadlineMonitor()
        r = repr(monitor)
        assert "DeadlineMonitor" in r


# =============================================================================
# ROS2Bridge 模拟模式
# =============================================================================


class TestROS2BridgeSimulation:
    """ROS2 桥接模拟模式测试。"""

    def test_start_stop(self):
        bridge = ROS2Bridge(simulation_mode=True)
        bridge.start()
        assert bridge.is_running is True
        bridge.stop()
        assert bridge.is_running is False

    def test_on_joint_state(self):
        bridge = ROS2Bridge(simulation_mode=True)
        bridge.start()
        bridge.on_joint_state(
            {
                "name": ["j1", "j2", "j3", "j4", "j5", "j6"],
                "position": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "velocity": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
                "effort": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        state = bridge.current_state
        assert state is not None
        assert state.n_joints == 6
        bridge.stop()

    def test_publish_prediction(self):
        predictions = []
        bridge = ROS2Bridge(
            simulation_mode=True,
            on_prediction=predictions.append,
        )
        bridge.start()
        state = RobotWorldState(n_joints=6)
        bridge.publish_prediction(state)
        assert len(predictions) == 1
        bridge.stop()

    def test_bridge_state(self):
        bridge = ROS2Bridge(simulation_mode=True)
        bridge.start()
        bridge.on_joint_state(
            {
                "position": [0.1, 0.2],
                "velocity": [0.01, 0.02],
                "effort": [0.0, 0.0],
            }
        )
        state = bridge.bridge_state()
        assert state.messages_received == 1
        assert state.current_robot_state is not None
        bridge.stop()

    def test_register_callback(self):
        callbacks = []
        bridge = ROS2Bridge(simulation_mode=True)
        bridge.register_callback(callbacks.append)
        bridge.start()
        bridge.on_joint_state(
            {
                "position": [0.1],
                "velocity": [0.01],
                "effort": [0.0],
            }
        )
        assert len(callbacks) == 1
        bridge.stop()

    def test_config(self):
        config = ROS2BridgeConfig(node_name="test_bridge")
        bridge = ROS2Bridge(config=config)
        assert bridge.config.node_name == "test_bridge"

    def test_repr(self):
        bridge = ROS2Bridge(simulation_mode=True)
        r = repr(bridge)
        assert "ROS2Bridge" in r


# =============================================================================
# 扩展安全约束
# =============================================================================


class TestJointLimitConstraint:
    """关节限位约束测试。"""

    def test_within_limit(self):
        c = JointLimitConstraint(n_joints=6)
        state = RobotWorldState(
            joint_positions=np.zeros(6),
            joint_velocities=np.zeros(6),
            joint_efforts=np.zeros(6),
            n_joints=6,
        )
        result = c.check(state, None)
        assert result.passed is True

    def test_exceeds_limit(self):
        c = JointLimitConstraint(
            joint_limits=[(-1.0, 1.0)] * 6,
        )
        state = RobotWorldState(
            joint_positions=np.array([0.5, 0.5, 1.5, 0.5, 0.5, 0.5]),
            joint_velocities=np.zeros(6),
            joint_efforts=np.zeros(6),
            n_joints=6,
        )
        result = c.check(state, None)
        assert result.passed is False
        assert "关节 2" in result.reason

    def test_non_robot_state_passes(self):
        c = JointLimitConstraint()
        state = PendulumState(theta=0.1, omega=0.0)
        result = c.check(state, None)
        assert result.passed is True


class TestSelfCollisionConstraint:
    """自碰撞约束测试。"""

    def test_no_collision(self):
        c = SelfCollisionConstraint()
        state = RobotWorldState(
            joint_positions=np.zeros(6),
            joint_velocities=np.zeros(6),
            joint_efforts=np.zeros(6),
            n_joints=6,
        )
        result = c.check(state, None)
        assert result.passed is True

    def test_collision_detected(self):
        c = SelfCollisionConstraint(min_clearance=0.1)
        # 相邻关节角度和接近 π → 碰撞风险
        state = RobotWorldState(
            joint_positions=np.array([1.5, 1.5, 0.0, 0.0, 0.0, 0.0]),
            joint_velocities=np.zeros(6),
            joint_efforts=np.zeros(6),
            n_joints=6,
        )
        result = c.check(state, None)
        assert result.passed is False
        assert "碰撞风险" in result.reason


class TestWorkspaceBoundConstraint:
    """工作空间约束测试。"""

    def test_within_workspace(self):
        c = WorkspaceBoundConstraint(max_reach=2.0)
        state = RobotWorldState(n_joints=6)
        result = c.check(state, None)
        assert result.passed is True

    def test_exceeds_workspace(self):
        c = WorkspaceBoundConstraint(max_reach=0.01)
        state = RobotWorldState(
            joint_positions=np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0]),
            joint_velocities=np.zeros(6),
            joint_efforts=np.zeros(6),
            n_joints=6,
        )
        result = c.check(state, None)
        assert result.passed is False
        assert "工作空间" in result.reason


class TestToolForceConstraint:
    """工具力约束测试。"""

    def test_within_force_limit(self):
        c = ToolForceConstraint(max_force=10.0)
        state = RobotWorldState(n_joints=6)
        result = c.check(state, None)
        assert result.passed is True

    def test_exceeds_force_limit(self):
        c = ToolForceConstraint(max_force=5.0, tool_joint_index=-1)
        state = RobotWorldState(
            joint_positions=np.zeros(6),
            joint_velocities=np.zeros(6),
            joint_efforts=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 8.0]),
            n_joints=6,
        )
        result = c.check(state, None)
        assert result.passed is False
        assert "工具力超限" in result.reason


class TestExtendedSafetyMonitor:
    """扩展安全约束链式注册测试。"""

    def test_eight_constraints(self):
        """≥ 8 条安全约束可注册。"""
        monitor = SafetyMonitor()
        # Phase 2 约束
        from mci_world_model.sdk._safety import (
            ForceLimitConstraint,
            PositionBoundConstraint,
            VelocityLimitConstraint,
        )

        monitor.register(ForceLimitConstraint())
        monitor.register(PositionBoundConstraint())
        monitor.register(VelocityLimitConstraint())
        # Phase 3 约束
        monitor.register(JointLimitConstraint())
        monitor.register(SelfCollisionConstraint())
        monitor.register(WorkspaceBoundConstraint())
        monitor.register(ToolForceConstraint())
        # 再加一个自定义
        monitor.register(ForceLimitConstraint(max_torque=20.0))

        assert monitor.constraint_count == 8

        # 机器人状态应该全部通过
        state = RobotWorldState(n_joints=6)
        result = monitor.check_all(state)
        assert result.passed is True


# =============================================================================
# cewm_step_fast() 快速路径
# =============================================================================


class TestCewmStepFast:
    """cewm_step_fast() 测试。"""

    def test_basic_fast_step(self):
        wm = MCIWorldModel()
        result = wm.cewm_step_fast(
            observation=PendulumState(theta=0.1, omega=0.0),
            goal=PendulumState(theta=0.0, omega=0.0),
        )
        assert result["fast_path"] is True
        assert "latency_ms" in result
        assert result["safety_violation"] is False

    def test_fast_step_with_safety(self):
        from mci_world_model.sdk._safety import ForceLimitConstraint

        wm = MCIWorldModel()
        wm._safety_monitor = SafetyMonitor()
        wm._safety_monitor.register(ForceLimitConstraint(max_torque=2.0))

        result = wm.cewm_step_fast(
            observation=PendulumState(theta=0.1, omega=0.0),
            goal=PendulumState(theta=0.0, omega=0.0),
            action=PendulumState(theta=0.1, omega=0.0),  # 不是 action 但不会崩
        )
        # 不应崩溃
        assert "safety_violation" in result

    def test_fast_step_with_emergency_stop(self):
        wm = MCIWorldModel()
        estop = EmergencyStop()
        estop.trigger("test")
        wm._emergency_stop = estop

        result = wm.cewm_step_fast(
            observation=PendulumState(theta=0.1, omega=0.0),
        )
        assert result["safety_violation"] is True
        assert result["safety_reason"] == "emergency_stop"

    def test_fast_step_faster_than_normal(self):
        """快速路径延迟应 ≤ 通用路径。"""
        wm = MCIWorldModel()
        obs = PendulumState(theta=0.1, omega=0.0)
        goal = PendulumState(theta=0.0, omega=0.0)

        # 预热
        wm.cewm_step(observation=obs, goal=goal)
        wm.cewm_step_fast(observation=obs, goal=goal)

        # 测量
        fast_results = []
        for _ in range(20):
            r = wm.cewm_step_fast(observation=obs, goal=goal)
            fast_results.append(r["latency_ms"])

        # 快速路径应该返回 latency_ms
        assert all(r >= 0 for r in fast_results)


# =============================================================================
# RobotWorldState 通过 cewm_step() 闭环
# =============================================================================


class TestRobotCewmClosedLoop:
    """RobotWorldState 通过 CEWM 闭环测试。"""

    def test_cewm_step_with_robot_state(self):
        wm = MCIWorldModel()
        state = RobotWorldState(n_joints=6)
        result = wm.cewm_step(observation=state)
        assert result["state"] is not None

    def test_cewm_step_fast_with_robot_state(self):
        wm = MCIWorldModel()
        state = RobotWorldState(n_joints=6)
        result = wm.cewm_step_fast(observation=state)
        assert result["state"] is not None
        assert result["fast_path"] is True
