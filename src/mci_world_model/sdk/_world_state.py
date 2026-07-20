from __future__ import annotations

"""
MCI World Model v4.6.0 — WorldState 抽象基类
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


from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import logging

logger = logging.getLogger(__name__)
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

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。子类应覆盖此方法提供更有意义的输出。"""
        return {"type": self.__class__.__name__}

    def causal_edges(self) -> list[tuple[str, str]]:
        """声明此状态类型的因果边。

        FIX-C5: WorldState 自描述因果结构，避免在 _cewm_state_change 中硬编码。
        子类可覆盖以提供领域特定的因果边。

        Returns:
            因果边列表 [(cause, effect), ...]
        """
        # 默认: 基于 to_vector() 维度生成相邻项因果边
        vec = self.to_vector()
        return [(f"dim_{i}", f"dim_{i + 1}") for i in range(len(vec) - 1)]

    def causal_query(self) -> str:
        """返回用于 JEPA 预测的因果查询字符串。

        FIX-C2: 替代 str(state) 作为 jepa_predict 的查询。
        返回有意义的因果描述而非 Python 对象 repr。

        Returns:
            因果查询字符串（如 "pendulum", "cart"）
        """
        return self.__class__.__name__.lower().replace("state", "")

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
    def from_signals(cls, signals: list[Any]) -> PendulumState:
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "PendulumState",
            "theta": round(self.theta, 6),
            "omega": round(self.omega, 6),
            "g": self.g,
            "L": self.L,
            "dt": self.dt,
        }

    # ── FIX-C2/C5: 因果结构自描述 ──

    def causal_edges(self) -> list[tuple[str, str]]:
        """单摆因果边: theta ↔ omega 双向因果。"""
        return [("theta", "omega"), ("omega", "theta")]

    def causal_query(self) -> str:
        """JEPA 因果查询: 'pendulum'。"""
        return "pendulum"


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

    def to_vector(self) -> np.ndarray:
        """编码动作为向量（默认实现，子类应覆盖）。

        v4.4.1: 动作空间泛化——动作可编码为 N 维向量。
        """
        return np.array([], dtype=np.float64)

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> Action:
        """从向量解码动作（默认实现，子类应覆盖）。

        v4.4.1: 动作空间泛化——向量可解码为动作。
        """
        raise NotImplementedError(f"{cls.__name__} 未实现 from_vector")

    def to_dict(self) -> dict[str, Any]:
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

    def to_vector(self) -> np.ndarray:
        """编码动作为向量。"""
        return np.array([self.torque], dtype=np.float64)

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> PendulumAction:
        """从向量解码动作。"""
        return cls(torque=float(vec[0]))

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "PendulumAction",
            "torque": self.torque,
            "dt": self.dt,
        }


# =============================================================================
# CartState — 小车移动状态 (v4.4.0 Phase 0)
# =============================================================================


@dataclass
class CartState(WorldState):
    """小车移动状态——CEWM 架构泛化的第二种 WorldState 验证器。

    状态空间 S = (x, v):
        x: 位置 (m), 范围无限制
        v: 速度 (m/s), 正值 = 向右

    物理定律（无外力、无摩擦）:
        x_{t+1} = x_t + v_t · dt
        v_{t+1} = v_t

    为什么选择小车？
    1. 与 PendulumState 结构不同：位置/速度 vs 角度/角速度
    2. 物理定律更简单：无重力非线性项，Euler 积分为精确解
    3. 验证 Protocol 抽象：证明 cewm_step() 不依赖 Pendulum 特有属性

    使用:
        >>> state = CartState(x=0.0, v=1.0)
        >>> vec = state.to_vector()          # np.array([0.0, 1.0])
        >>> next_state = state.step_physics() # 自由演化一步
        >>> assert next_state.x > state.x    # 向右移动
    """

    x: float  # 位置 (m)
    v: float  # 速度 (m/s)

    # 物理常量
    dt: float = 0.01  # 时间步长 (s)

    # ── WorldState 核心契约实现 ──

    def to_vector(self) -> np.ndarray:
        """编码为 2 维向量 [x, v]."""
        return np.array([self.x, self.v], dtype=np.float64)

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> CartState:
        """从 2 维向量解码。"""
        return cls(x=float(vec[0]), v=float(vec[1]))

    def distance(self, other: WorldState) -> float:
        """欧氏距离在 (x, v) 空间中的度量。"""
        if not isinstance(other, CartState):
            # 跨类型：使用向量 L2 距离
            if hasattr(other, "to_vector"):
                return float(np.linalg.norm(self.to_vector() - other.to_vector()))
            return float("inf")
        dx = abs(self.x - other.x)
        dv = abs(self.v - other.v)
        return float(np.sqrt(dx**2 + dv**2))

    def copy(self) -> CartState:
        """深拷贝。"""
        return CartState(x=self.x, v=self.v, dt=self.dt)

    # ── 物理演化 ──

    def step_physics(self) -> CartState:
        """按照物理定律演化一步（无外力自然演化）。"""
        x_next = self.x + self.v * self.dt
        v_next = self.v  # 无摩擦时速度不变
        return CartState(x=x_next, v=v_next, dt=self.dt)

    # ── 序列化 ──

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "CartState",
            "x": round(self.x, 6),
            "v": round(self.v, 6),
            "dt": self.dt,
        }

    # ── FIX-C2/C5: 因果结构自描述 ──

    def causal_edges(self) -> list[tuple[str, str]]:
        """小车因果边: x ↔ v 双向因果。"""
        return [("x", "v"), ("v", "x")]

    def causal_query(self) -> str:
        """JEPA 因果查询: 'cart'。"""
        return "cart"

    def __repr__(self) -> str:
        return f"CartState(x={self.x:.4f}, v={self.v:.4f})"


# =============================================================================
# CartAction — 小车力动作 (v4.4.0 Phase 0)
# =============================================================================


@dataclass
class CartAction(Action):
    """小车动作——施加外力。

    动作空间 A: 1 维连续值
        force: 外力 (N), 正值 = 向右推, 负值 = 向左推

    物理效果（Euler 积分，F=ma, m=1）:
        a = force / m       # 加速度，m=1 → a = force
        v_{t+1} = v_t + a · dt
        x_{t+1} = x_t + v_t · dt

    使用:
        >>> state = CartState(x=0.0, v=0.0)
        >>> push = CartAction(force=5.0)
        >>> next_state = push.apply(state)  # 向右推后状态
        >>> assert next_state.v > 0
    """

    force: float = 0.0  # 外力 (N)
    dt: float = 0.01  # 动作持续时间 (s)

    def to_vector(self) -> np.ndarray:
        """编码动作为向量。"""
        return np.array([self.force], dtype=np.float64)

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> CartAction:
        """从向量解码动作。"""
        return cls(force=float(vec[0]))

    def apply(self, state: WorldState) -> CartState:
        """施加力到 CartState，返回新状态。"""
        if not isinstance(state, CartState):
            raise TypeError(f"CartAction 只能作用于 CartState，收到 {type(state).__name__}")
        # F = ma, m=1 → a = force
        acceleration = self.force
        v_next = state.v + acceleration * self.dt
        x_next = state.x + state.v * self.dt
        return CartState(x=x_next, v=v_next, dt=state.dt)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "CartAction",
            "force": self.force,
            "dt": self.dt,
        }

    def __repr__(self) -> str:
        return f"CartAction(force={self.force:.2f})"


# =============================================================================
# DoublePendulumState — 双摆物理状态 (v4.4.1 Phase 1)
# =============================================================================


@dataclass
class DoublePendulumState(WorldState):
    """双摆物理状态——4D 状态空间验证器。

    状态空间 S = (theta1, omega1, theta2, omega2):
        theta1: 第一级摆角 (rad)
        omega1: 第一级角速度 (rad/s)
        theta2: 第二级摆角 (rad)
        omega2: 第二级角速度 (rad/s)

    动作空间: 2D (torque1, torque2)

    使用:
        >>> state = DoublePendulumState(theta1=0.5, omega1=0.0, theta2=0.3, omega2=0.0)
        >>> vec = state.to_vector()  # 4D 向量
    """

    theta1: float  # 第一级摆角 (rad)
    omega1: float  # 第一级角速度 (rad/s)
    theta2: float  # 第二级摆角 (rad)
    omega2: float  # 第二级角速度 (rad/s)

    g: float = 9.81
    L1: float = 1.0
    L2: float = 1.0
    dt: float = 0.01

    def to_vector(self) -> np.ndarray:
        return np.array([self.theta1, self.omega1, self.theta2, self.omega2], dtype=np.float64)

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> DoublePendulumState:
        return cls(
            theta1=float(vec[0]),
            omega1=float(vec[1]),
            theta2=float(vec[2]),
            omega2=float(vec[3]),
        )

    def distance(self, other: WorldState) -> float:
        if not isinstance(other, DoublePendulumState):
            if hasattr(other, "to_vector"):
                return float(np.linalg.norm(self.to_vector() - other.to_vector()))
            return float("inf")
        d1 = abs(self.theta1 - other.theta1)
        d2 = abs(self.omega1 - other.omega1)
        d3 = abs(self.theta2 - other.theta2)
        d4 = abs(self.omega2 - other.omega2)
        return float(np.sqrt(d1**2 + d2**2 + d3**2 + d4**2))

    def copy(self) -> DoublePendulumState:
        return DoublePendulumState(
            theta1=self.theta1,
            omega1=self.omega1,
            theta2=self.theta2,
            omega2=self.omega2,
            g=self.g,
            L1=self.L1,
            L2=self.L2,
            dt=self.dt,
        )

    def step_physics(self) -> DoublePendulumState:
        """自由演化一步（无外力）。"""
        d_theta1 = self.omega1
        d_omega1 = -(self.g / self.L1) * np.sin(self.theta1)
        d_theta2 = self.omega2
        d_omega2 = -(self.g / self.L2) * np.sin(self.theta2)
        return DoublePendulumState(
            theta1=self.theta1 + d_theta1 * self.dt,
            omega1=self.omega1 + d_omega1 * self.dt,
            theta2=self.theta2 + d_theta2 * self.dt,
            omega2=self.omega2 + d_omega2 * self.dt,
            g=self.g,
            L1=self.L1,
            L2=self.L2,
            dt=self.dt,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "DoublePendulumState",
            "theta1": round(self.theta1, 6),
            "omega1": round(self.omega1, 6),
            "theta2": round(self.theta2, 6),
            "omega2": round(self.omega2, 6),
        }

    def __repr__(self) -> str:
        return (
            f"DoublePendulumState(θ1={self.theta1:.3f}, ω1={self.omega1:.3f}, "
            f"θ2={self.theta2:.3f}, ω2={self.omega2:.3f})"
        )


# =============================================================================
# DoublePendulumAction — 双摆力矩动作 (v4.4.1 Phase 1)
# =============================================================================


@dataclass
class DoublePendulumAction(Action):
    """双摆动作——施加两个力矩。

    动作空间 A: 2 维连续值
        torque1: 第一级力矩 (N·m)
        torque2: 第二级力矩 (N·m)
    """

    torque1: float = 0.0
    torque2: float = 0.0
    dt: float = 0.01

    def apply(self, state: WorldState) -> DoublePendulumState:
        if not isinstance(state, DoublePendulumState):
            raise TypeError("DoublePendulumAction 只能作用于 DoublePendulumState")
        m1, m2 = 1.0, 1.0
        alpha1 = self.torque1 / (m1 * state.L1**2)
        alpha2 = self.torque2 / (m2 * state.L2**2)
        d_omega1 = -(state.g / state.L1) * np.sin(state.theta1) + alpha1
        d_omega2 = -(state.g / state.L2) * np.sin(state.theta2) + alpha2
        return DoublePendulumState(
            theta1=state.theta1 + state.omega1 * self.dt,
            omega1=state.omega1 + d_omega1 * self.dt,
            theta2=state.theta2 + state.omega2 * self.dt,
            omega2=state.omega2 + d_omega2 * self.dt,
            g=state.g,
            L1=state.L1,
            L2=state.L2,
            dt=state.dt,
        )

    def to_vector(self) -> np.ndarray:
        return np.array([self.torque1, self.torque2], dtype=np.float64)

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> DoublePendulumAction:
        return cls(torque1=float(vec[0]), torque2=float(vec[1]))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "DoublePendulumAction", "torque1": self.torque1, "torque2": self.torque2}

    def __repr__(self) -> str:
        return f"DoublePendulumAction(τ1={self.torque1:.2f}, τ2={self.torque2:.2f})"


# =============================================================================
# MultimodalWorldState — 多模态世界状态 (v3.3.0)
# =============================================================================


@dataclass
class MultimodalWorldState(WorldState):
    """多模态世界状态 —— 融合本体感觉 + 视觉 + 音频 + 热感应。

    v3.3.0: 扩展 WorldState 以支持多模态感知输入。
    每个模态字段可选（None 表示该模态未提供）。

    Example:
        >>> state = MultimodalWorldState(
        ...     proprioception=np.array([0.5, 1.0]),
        ...     vision=np.array([0.1, 0.2, 0.3]),
        ... )
        >>> vec = state.to_vector()
    """

    proprioception: np.ndarray | None = None
    vision: np.ndarray | None = None
    audio: np.ndarray | None = None
    thermal: np.ndarray | None = None
    fused: np.ndarray | None = None
    modality_confidences: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    # v4.4.1: 保留模态结构元数据，供 from_vector() 往返保真
    _modality_layout: dict[str, Any] = field(default_factory=dict, repr=False)

    def _get_vector_parts(self) -> list[np.ndarray]:
        """收集所有非空模态向量。"""
        parts: list[np.ndarray] = []
        for arr in (self.proprioception, self.vision, self.audio, self.thermal, self.fused):
            if arr is not None:
                parts.append(np.asarray(arr, dtype=np.float64).flatten())
        return parts

    def to_vector(self) -> np.ndarray:
        """拼接所有模态向量为统一向量，同时记录布局信息。

        v4.4.1: 在 _modality_layout 中记录每个模态在拼接向量中的位置，
        供 from_vector() 恢复模态结构。
        """
        parts: list[np.ndarray] = []
        layout: dict[str, tuple[int, int]] = {}
        offset = 0

        for name, arr in [
            ("proprioception", self.proprioception),
            ("vision", self.vision),
            ("audio", self.audio),
            ("thermal", self.thermal),
            ("fused", self.fused),
        ]:
            if arr is not None:
                flat = np.asarray(arr, dtype=np.float64).flatten()
                layout[name] = (offset, offset + len(flat))
                parts.append(flat)
                offset += len(flat)

        # 保存布局信息
        self._modality_layout = layout
        # v4.5.0: 同步到类级别缓存，供 from_vector() 自动保真
        MultimodalWorldState._last_layout = layout

        if not parts:
            return np.zeros(1, dtype=np.float64)
        return np.concatenate(parts)

    # v4.5.0: 类级别布局缓存，供 from_vector() 自动保真 (非 dataclass 字段)
    _last_layout: ClassVar[dict] = {}  # type: ignore

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> MultimodalWorldState:
        """从向量解码为 MultimodalWorldState。

        v4.5.0: 自动保真——优先使用 to_vector() 时缓存的布局信息。
        如果缓存存在且维度匹配，自动还原各模态；否则退化为 fused 字段。
        保真度目标 ≥ 90% (KPI I-3)。
        """
        vec = np.asarray(vec, dtype=np.float64)

        # 尝试使用缓存的布局信息
        if cls._last_layout and len(vec) >= sum(end - start for start, end in cls._last_layout.values()):
            try:
                return cls.from_vector_with_layout(vec, cls._last_layout)
            except Exception as e:
                logger.warning("吞异常", exc_info=True)
        return cls(fused=vec)

    @classmethod
    def from_vector_with_layout(cls, vec: np.ndarray, layout: dict[str, tuple[int, int]]) -> MultimodalWorldState:
        """从向量+布局信息解码，保真还原各模态。

        v4.4.1: 往返保真方法。

        用法:
            >>> state = MultimodalWorldState(proprioception=np.array([1.0, 2.0]), vision=np.array([0.5]))
            >>> vec = state.to_vector()  # 同时更新 _modality_layout
            >>> layout = state._modality_layout
            >>> restored = MultimodalWorldState.from_vector_with_layout(vec, layout)
            >>> assert restored.active_modalities() == state.active_modalities()
        """
        vec = np.asarray(vec, dtype=np.float64)
        kwargs: dict[str, Any] = {}
        for name, (start, end) in layout.items():
            if end <= len(vec):
                kwargs[name] = vec[start:end]
        return cls(**kwargs)

    def distance(self, other: WorldState) -> float:
        """多模态状态距离：向量 L2 距离。

        v4.4.1: 使用 padding 策略代替截断——短向量用零填充到长向量维度。
        """
        self_vec = self.to_vector()
        other_vec = other.to_vector()

        # padding 策略：短向量用零填充
        max_len = max(len(self_vec), len(other_vec))
        if max_len == 0:
            return 0.0
        self_padded = np.zeros(max_len, dtype=np.float64)
        other_padded = np.zeros(max_len, dtype=np.float64)
        self_padded[: len(self_vec)] = self_vec
        other_padded[: len(other_vec)] = other_vec

        return float(np.linalg.norm(self_padded - other_padded))

    def copy(self) -> MultimodalWorldState:
        """深拷贝。"""
        return MultimodalWorldState(
            proprioception=self.proprioception.copy() if self.proprioception is not None else None,
            vision=self.vision.copy() if self.vision is not None else None,
            audio=self.audio.copy() if self.audio is not None else None,
            thermal=self.thermal.copy() if self.thermal is not None else None,
            fused=self.fused.copy() if self.fused is not None else None,
            modality_confidences=dict(self.modality_confidences),
            timestamp=self.timestamp,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "MultimodalWorldState",
            "proprioception": self.proprioception.tolist() if self.proprioception is not None else None,
            "vision": self.vision.tolist() if self.vision is not None else None,
            "audio": self.audio.tolist() if self.audio is not None else None,
            "thermal": self.thermal.tolist() if self.thermal is not None else None,
            "fused": self.fused.tolist() if self.fused is not None else None,
            "modality_confidences": self.modality_confidences,
            "timestamp": self.timestamp,
        }

    def active_modalities(self) -> list[str]:
        """返回当前激活的模态列表。"""
        modalities = []
        if self.proprioception is not None:
            modalities.append("proprioception")
        if self.vision is not None:
            modalities.append("vision")
        if self.audio is not None:
            modalities.append("audio")
        if self.thermal is not None:
            modalities.append("thermal")
        if self.fused is not None:
            modalities.append("fused")
        return modalities

    @classmethod
    def from_signals(cls, signals: list[Any]) -> MultimodalWorldState:
        """从 PhysicalSignal 列表构建多模态世界状态。"""
        state = cls()
        for sig in signals:
            modality_val = str(getattr(sig, "modality", "")).lower()
            value = getattr(sig, "value", None)
            if value is None:
                continue
            try:
                vec = np.asarray(value, dtype=np.float64).flatten()
            except (TypeError, ValueError):
                continue

            if "proprioception" in modality_val:
                state.proprioception = vec
            elif "vision" in modality_val:
                state.vision = vec
            elif "audition" in modality_val:
                state.audio = vec
            elif "tactition" in modality_val:
                state.thermal = vec  # 触觉映射到热感应
        return state

    def __repr__(self) -> str:
        mods = ", ".join(self.active_modalities())
        return f"MultimodalWorldState(modalities=[{mods}])"


# =============================================================================
# 工具函数
# =============================================================================


def _angular_distance(a: float, b: float) -> float:
    """两个角之间的最短弧距离 (rad)。

    例: _angular_distance(3.14, -3.14) ≈ 0.0 (同一位置)
    """
    diff = abs(a - b) % (2 * np.pi)
    return float(min(diff, 2 * np.pi - diff))
