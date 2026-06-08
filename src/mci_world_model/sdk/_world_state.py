"""
MCI World Model v3.2.0 — WorldState 抽象基类
==============================================

WorldState 是 MCI 世界模型"认知环"的核心数据结构——被建模世界的内在表征。

与 CausalWorldModelState 的关键区别:
    CausalWorldModelState: 描述"推理过程"（几条因果边、多少确认/抑制）
    WorldState:           描述"世界本身"（摆角多少度、小车在什么位置）

设计原则:
    - 通用性: 不含任何领域术语（无"血糖""临床""NRS2002"等）
    - 最小接口: 只定义 JEPA Encoder/Predictor/Cost 需要的四个核心操作
    - 单摆验证: PendulumState 作为最简物理世界验证 WorldState 契约完备性

子类化指南:
    任何领域要接入 MCI 世界模型，只需:
    1. 继承 WorldState，实现四个抽象方法
    2. 实现 from_signals() 类方法（物理信号→状态）
    3. 用单摆验证四环闭环后，再接入具体领域
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


# =============================================================================
# WorldState — 抽象基类
# =============================================================================


class WorldState(ABC):
    """世界状态的抽象基类——独立于推理过程的世界内在表征。

    四个核心契约:
        to_vector()   →  将世界状态编码为固定维度向量（给 JEPA Encoder）
        from_vector() →  从向量解码为世界状态（给 JEPA Decoder）
        distance()    →  两个世界状态的距离（给 Cost 模块评估预测误差）
        copy()        →  深拷贝（给 Actor 做 what-if 推演）
    """

    @abstractmethod
    def to_vector(self) -> np.ndarray:
        """将世界状态编码为固定维度向量。

        Returns:
            1D numpy float64 数组，维度固定。
            此向量是 JEPA Encoder 的输入，进入潜空间预测。
        """
        ...

    @classmethod
    @abstractmethod
    def from_vector(cls, vec: np.ndarray) -> WorldState:
        """从向量解码为世界状态。

        JEPA Decoder 的输出，从潜空间重建世界状态。
        """
        ...

    @abstractmethod
    def distance(self, other: WorldState) -> float:
        """两个世界状态之间的距离度量。

        用于 Cost 模块评估预测精度:
            cost = distance(s_predicted, s_ground_truth)

        Returns:
            非负浮点数，0 表示相同状态。
        """
        ...

    @abstractmethod
    def copy(self) -> WorldState:
        """深拷贝当前世界状态。

        用于 Actor 做 what-if 推演:
            state_copy = state.copy()
            state_copy = action.apply(state_copy)  # 不污染原状态
        """
        ...

    def to_dict(self) -> dict:
        """序列化为字典。子类应覆盖此方法提供更有意义的输出。"""
        return {"type": self.__class__.__name__}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# =============================================================================
# PendulumState — 单摆物理状态（最简世界验证器）
# =============================================================================


@dataclass
class PendulumState(WorldState):
    """单摆物理状态——MCI 世界模型的最简架构验证器。

    状态空间 S = (theta, omega):
        theta: 摆角 (rad), 范围 [-π, π], 0 = 垂直向下平衡点
        omega: 角速度 (rad/s), 正值 = 顺时针

    物理定律（无外力、小角度近似可忽略 sin 的非线性）:
        θ_{t+1} = θ_t + ω_t · dt
        ω_{t+1} = ω_t − (g/L) · sin(θ_t) · dt

    为什么选择单摆？
    1. 状态空间最小: 仅 2 维 (θ, ω)，可直接可视化
    2. 物理定律精确: 确定性微分方程，ground truth 无歧义
    3. 闭环完整: 感知(传感器读数)→认知(PendulumState)→预测(JEPA)→行动(推力)→反馈
    4. 可验证性强: 能量守恒、周期可计算、预测精度可量化

    使用:
        >>> state = PendulumState(theta=0.5, omega=0.0)
        >>> vec = state.to_vector()          # np.array([0.5, 0.0])
        >>> next_state = state.step_physics() # 自由演化一步
        >>> state.distance(next_state)       # 状态变化量
    """

    theta: float  # 摆角 (rad)
    omega: float  # 角速度 (rad/s)

    # 物理常量（全局共享，不在状态空间中）
    g: float = 9.81  # 重力加速度 (m/s²)
    L: float = 1.0  # 摆长 (m)
    dt: float = 0.01  # 时间步长 (s)

    # ── WorldState 核心契约实现 ──

    def to_vector(self) -> np.ndarray:
        """编码为 2 维向量 [theta, omega]."""
        return np.array([self.theta, self.omega], dtype=np.float64)

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> PendulumState:
        """从 2 维向量解码。"""
        return cls(theta=float(vec[0]), omega=float(vec[1]))

    def distance(self, other: WorldState) -> float:
        """欧氏距离在 (θ, ω) 空间中的度量。

        注意: θ 是角度，需要处理周期性。
        """
        if not isinstance(other, PendulumState):
            return float("inf")
        # 角度距离: 取最短弧（处理 -π 和 π 是同一位置）
        d_theta = _angular_distance(self.theta, other.theta)
        d_omega = abs(self.omega - other.omega)
        return float(np.sqrt(d_theta**2 + d_omega**2))

    def copy(self) -> PendulumState:
        """深拷贝。"""
        return PendulumState(
            theta=self.theta,
            omega=self.omega,
            g=self.g,
            L=self.L,
            dt=self.dt,
        )

    # ── 物理演化 ──

    def step_physics(self) -> PendulumState:
        """按照物理定律演化一步（无外力自然演化）。

        Returns:
            新 PendulumState，dt 秒后的状态。
        """
        theta_next = self.theta + self.omega * self.dt
        omega_next = self.omega - (self.g / self.L) * np.sin(self.theta) * self.dt
        return PendulumState(
            theta=theta_next,
            omega=omega_next,
            g=self.g,
            L=self.L,
            dt=self.dt,
        )

    # ── 从物理信号构建 ──

    @classmethod
    def from_signals(cls, signals: list) -> PendulumState:
        """从物理信号列表构建单摆状态（感知→认知的桥接）。

        期望信号组合:
            - ENCODER_POSITION × 1: 摆角 (rad)
            - IMU_9AXIS × 1: 角速度 (rad/s)，从 gyro_z 提取

        即使只提供部分信号，未提供的维度默认为 0。

        Args:
            signals: PhysicalSignal 列表

        Returns:
            PendulumState 或默认状态
        """
        theta = 0.0
        omega = 0.0

        for sig in signals:
            modality = getattr(sig, "modality", None)
            sub_type = getattr(sig, "sub_type", None)
            if modality is None or sub_type is None:
                continue

            # 使用字符串比较避免跨模块导入枚举（大小写不敏感）
            modality_str = str(modality).lower()
            sub_type_str = str(sub_type).lower()

            if "proprioception" in modality_str:
                if "encoder_position" in sub_type_str:
                    try:
                        theta = float(sig.value)
                    except (TypeError, ValueError):
                        pass
                elif "imu_9axis" in sub_type_str:
                    val = sig.value
                    if isinstance(val, (list, tuple)) and len(val) >= 3:
                        try:
                            omega = float(val[2])  # gyro_z
                        except (TypeError, ValueError):
                            pass

        return cls(theta=theta, omega=omega)

    # ── 序列化 ──

    def to_dict(self) -> dict:
        return {
            "type": "PendulumState",
            "theta": round(self.theta, 6),
            "omega": round(self.omega, 6),
            "g": self.g,
            "L": self.L,
            "dt": self.dt,
        }


# =============================================================================
# Action — 抽象动作基类（v3.2.0 Task 2）
# =============================================================================


class Action(ABC):
    """抽象动作基类——施加于世界状态的干预操作。

    WorldState 描述"世界是什么"，Action 描述"对世界做什么"。
    两者组合构成 JEPA 预测器的完整输入:
        predict(state: WorldState, action: Action, n_steps: int) → list[WorldState]

    子类化指南:
        任何领域要接入 MCI 世界模型的动作系统，只需:
        1. 继承 Action，实现 apply() 方法
        2. apply() 返回施加动作后的新 WorldState（不修改原状态）
        3. 用 PendulumAction 验证动作闭环后，再接入具体领域
    """

    @abstractmethod
    def apply(self, state: WorldState) -> WorldState:
        """对世界状态施加动作，返回新状态。

        不修改原状态（函数式语义）。

        此方法的 ground truth 用于训练 JEPA 预测器的监督信号:
            s_pred = predictor.predict(s, a)  # JEPA 预测
            s_true = a.apply(s)               # ground truth
            loss = distance(s_pred, s_true)
        """
        ...

    def to_dict(self) -> dict:
        return {"type": self.__class__.__name__}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@dataclass
class PendulumAction(Action):
    """单摆动作——施加外力力矩。

    动作空间 A: 1 维连续值
        torque: 外力力矩 (N·m)，正值 = 顺时针推，负值 = 逆时针推

    物理效果（Euler 积分，叠加到自然演化上）:
        α = torque / (m · L²)       # 角加速度，设 m=1, L=1 → α = torque
        ω_{t+1} = ω_t + [− (g/L) · sin(θ_t) + α] · dt
        θ_{t+1} = θ_t + ω_t · dt

    使用:
        >>> state = PendulumState(theta=0.0, omega=0.0)
        >>> push = PendulumAction(torque=5.0)
        >>> next_state = push.apply(state)  # 顺时针推后状态
        >>> assert next_state.omega > 0
    """

    torque: float = 0.0  # 外力力矩 (N·m)
    dt: float = 0.01  # 动作持续时间 (s)，默认覆盖 PendulumState.dt

    def apply(self, state: WorldState) -> PendulumState:
        """施加力矩到 PendulumState，返回新状态。"""
        if not isinstance(state, PendulumState):
            raise TypeError(f"PendulumAction 只能作用于 PendulumState，收到 {type(state).__name__}")
        # 角加速度: α = torque / (m·L²)，设定 m=1kg → α = torque / L²
        alpha = self.torque / (state.L**2)
        theta_next = state.theta + state.omega * self.dt
        omega_next = state.omega + (-(state.g / state.L) * np.sin(state.theta) + alpha) * self.dt
        return PendulumState(
            theta=theta_next,
            omega=omega_next,
            g=state.g,
            L=state.L,
            dt=state.dt,  # 保持原物理时间步长
        )

    def to_dict(self) -> dict:
        return {
            "type": "PendulumAction",
            "torque": self.torque,
            "dt": self.dt,
        }


# =============================================================================
# 工具函数
# =============================================================================


def _angular_distance(a: float, b: float) -> float:
    """两个角之间的最短弧距离 (rad)。

    例: _angular_distance(3.14, -3.14) ≈ 0.0 (同一位置)
    """
    diff = abs(a - b) % (2 * np.pi)
    return float(min(diff, 2 * np.pi - diff))
