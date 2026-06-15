"""MCI World Model v5.1.0 — GeneralizedPhysicsPredictor 通用物理预测器
========================================================================

CEWM 架构泛化的核心引擎——通过注册动力学函数支持任意维度的物理系统。

v5.1.0 新增 4 个物理系统:
    - spring_mass: 弹簧-质量阻尼系统 (2D: [x, v])
    - projectile: 抛体运动 (4D: [x, y, vx, vy])
    - fluid_flow: 流体管道系统 (3D: [P, Q, T])
    - double_pendulum: 双摆 (4D: 已有，小角度近似)

设计理念:
    不为每个物理系统写一个预测器类，而是用一个通用预测器 + 可插拔动力学函数。
    Pendulum/Cart/DoublePendulum/SpringMass/Projectile/FluidFlow 共用同一个 GeneralizedPhysicsPredictor。

核心能力:
    - register_dynamics(name, fn): 注册自定义动力学函数
    - predict(state, action, n_steps): 通用 Euler 积分多步预测
    - evaluate(dataset): 在测试数据集上评估精度

动力学函数接口:
    dynamics_fn(state_vec: np.ndarray, action_vec: np.ndarray) -> np.ndarray
    输入: 当前状态向量 + 动作向量
    输出: 状态导数 (d_state/dt)

注册示例:
    >>> from mci_world_model.sdk._generalized_physics import GeneralizedPhysicsPredictor
    >>> pred = GeneralizedPhysicsPredictor()
    >>> pred.register_dynamics("pendulum", pendulum_dynamics)
    >>> pred.register_dynamics("cart", cart_dynamics)
    >>> # 然后使用 pred.predict(state, action, n_steps)
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from mci_world_model.sdk._action_conditioned_predictor import ActionConditionedPredictor

if TYPE_CHECKING:
    from mci_world_model.sdk._world_state import Action, WorldState

logger = logging.getLogger(__name__)

# 动力学函数类型签名
DynamicsFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


# =============================================================================
# 内置动力学函数
# =============================================================================


def pendulum_dynamics(state_vec: np.ndarray, action_vec: np.ndarray) -> np.ndarray:
    """单摆动力学: [theta, omega] → [d_theta/dt, d_omega/dt]。

    物理定律:
        d_theta/dt = omega
        d_omega/dt = -(g/L) * sin(theta) + torque / (m*L^2)

    参数默认: g=9.81, L=1.0, m=1.0
    """
    theta, omega = float(state_vec[0]), float(state_vec[1])
    torque = float(action_vec[0]) if len(action_vec) > 0 else 0.0

    g, L, m = 9.81, 1.0, 1.0
    d_theta = omega
    d_omega = -(g / L) * np.sin(theta) + torque / (m * L**2)
    return np.array([d_theta, d_omega], dtype=np.float64)


def cart_dynamics(state_vec: np.ndarray, action_vec: np.ndarray) -> np.ndarray:
    """小车动力学: [x, v] → [dx/dt, dv/dt]。

    物理定律 (F=ma, m=1):
        dx/dt = v
        dv/dt = force / m = force
    """
    _x, v = float(state_vec[0]), float(state_vec[1])
    force = float(action_vec[0]) if len(action_vec) > 0 else 0.0

    m = 1.0
    d_x = v
    d_v = force / m
    return np.array([d_x, d_v], dtype=np.float64)


def double_pendulum_dynamics(state_vec: np.ndarray, action_vec: np.ndarray) -> np.ndarray:
    """双摆动力学: [theta1, omega1, theta2, omega2] → [d_theta1/dt, d_omega1/dt, d_theta2/dt, d_omega2/dt]。

    简化模型: 两级摆，第二级挂载在第一级摆球上。
    使用小角度近似线性化，便于验证。

    参数默认: g=9.81, L1=L2=1.0, m1=m2=1.0
    """
    theta1, omega1, theta2, omega2 = (
        float(state_vec[0]),
        float(state_vec[1]),
        float(state_vec[2]),
        float(state_vec[3]),
    )
    torque1 = float(action_vec[0]) if len(action_vec) > 0 else 0.0
    torque2 = float(action_vec[1]) if len(action_vec) > 1 else 0.0

    g, L1, L2 = 9.81, 1.0, 1.0
    m1, m2 = 1.0, 1.0

    # 简化动力学（小角度近似）
    d_theta1 = omega1
    d_omega1 = -(g / L1) * np.sin(theta1) + torque1 / (m1 * L1**2)
    d_theta2 = omega2
    d_omega2 = -(g / L2) * np.sin(theta2) + torque2 / (m2 * L2**2)

    return np.array([d_theta1, d_omega1, d_theta2, d_omega2], dtype=np.float64)


# =============================================================================
# v5.1.0 新增物理系统
# =============================================================================


def spring_mass_dynamics(state_vec: np.ndarray, action_vec: np.ndarray) -> np.ndarray:
    """弹簧-质量阻尼系统: [x, v] → [dx/dt, dv/dt]。

    物理定律 (简谐运动 + 阻尼 + 外力):
        dx/dt = v
        dv/dt = -(k/m)*x - (c/m)*v + F_ext/m

    参数默认: k=10.0 (弹簧刚度), m=1.0 (质量), c=0.5 (阻尼系数)
    F_ext: 外部控制力 (action)

    固有频率: ω_n = sqrt(k/m) = sqrt(10) ≈ 3.16 rad/s
    阻尼比: ζ = c/(2*sqrt(k*m)) = 0.5/(2*sqrt(10)) ≈ 0.079 (欠阻尼)
    """
    x, v = float(state_vec[0]), float(state_vec[1])
    F_ext = float(action_vec[0]) if len(action_vec) > 0 else 0.0

    k, m, c = 10.0, 1.0, 0.5
    d_x = v
    d_v = -(k / m) * x - (c / m) * v + F_ext / m
    return np.array([d_x, d_v], dtype=np.float64)


def projectile_dynamics(state_vec: np.ndarray, action_vec: np.ndarray) -> np.ndarray:
    """抛体运动: [x, y, vx, vy] → [dx/dt, dy/dt, dvx/dt, dvy/dt]。

    物理定律:
        dx/dt = vx
        dy/dt = vy
        dvx/dt = -drag * vx * |v| / m + F_x / m   (空气阻力 + 推力)
        dvy/dt = -g - drag * vy * |v| / m + F_y / m

    参数默认: g=9.81, drag=0.01 (低空气阻力), m=1.0
    action: [F_x, F_y] 推力分量

    无推力时为标准抛物线 (加阻力修正)。
    """
    _x, _y, vx, vy = (
        float(state_vec[0]),
        float(state_vec[1]),
        float(state_vec[2]),
        float(state_vec[3]),
    )
    F_x = float(action_vec[0]) if len(action_vec) > 0 else 0.0
    F_y = float(action_vec[1]) if len(action_vec) > 1 else 0.0

    g, drag, m = 9.81, 0.01, 1.0
    speed = math.sqrt(float(vx**2 + vy**2))

    d_x = vx
    d_y = vy
    d_vx = -drag * vx * speed / m + F_x / m
    d_vy = -g - drag * vy * speed / m + F_y / m
    return np.array([d_x, d_y, d_vx, d_vy], dtype=np.float64)


def fluid_flow_dynamics(state_vec: np.ndarray, action_vec: np.ndarray) -> np.ndarray:
    """流体管道系统: [P, Q, T] → [dP/dt, dQ/dt, dT/dt]。

    简化一维管道模型:
        P: 压力 (Pa)
        Q: 体积流量 (m³/s)
        T: 温度 (K)

    物理定律 (简化):
        dP/dt = -k_p * P + R * Q + P_in   (压力: 管道弹性 + 流量贡献 + 入口压力)
        dQ/dt = (P - R_f * Q) / L_f         (流量: 压差驱动 - 摩擦损耗)
        dT/dt = -k_t * (T - T_amb) + Q * T_heat  (温度: 对流散热 + 流量加热)

    参数默认:
        k_p = 1.0 (压力衰减)
        R = 0.5 (流量-压力耦合)
        R_f = 0.8 (摩擦阻力)
        L_f = 1.0 (流体惯性)
        k_t = 0.1 (温度衰减)
        T_amb = 293.15 (环境温度, 20°C)

    action: [P_in, T_heat] 入口压力和加热功率
    """
    P, Q, T = float(state_vec[0]), float(state_vec[1]), float(state_vec[2])
    P_in = float(action_vec[0]) if len(action_vec) > 0 else 0.0
    T_heat = float(action_vec[1]) if len(action_vec) > 1 else 0.0

    k_p, R, R_f, L_f = 1.0, 0.5, 0.8, 1.0
    k_t, T_amb = 0.1, 293.15

    d_P = -k_p * P + R * Q + P_in
    d_Q = (P - R_f * Q) / L_f
    d_T = -k_t * (T - T_amb) + Q * T_heat
    return np.array([d_P, d_Q, d_T], dtype=np.float64)


# =============================================================================
# Euler 步进器
# =============================================================================


def euler_step(
    state_vec: np.ndarray,
    action_vec: np.ndarray,
    dynamics_fn: DynamicsFn,
    dt: float,
) -> np.ndarray:
    """Euler 积分单步。

    s_{t+1} = s_t + dynamics_fn(s_t, a_t) * dt
    """
    d_state = dynamics_fn(state_vec, action_vec)
    return state_vec + d_state * dt


# =============================================================================
# GeneralizedPhysicsPredictor — 通用物理预测器
# =============================================================================


class GeneralizedPhysicsPredictor(ActionConditionedPredictor):
    """通用物理预测器——通过注册动力学函数支持任意物理系统。

    用法:
        >>> pred = GeneralizedPhysicsPredictor()
        >>> pred.register_dynamics("pendulum", pendulum_dynamics, state_dim=2, action_dim=1)
        >>> pred.register_dynamics("cart", cart_dynamics, state_dim=2, action_dim=1)
        >>> pred.register_dynamics("double_pendulum", double_pendulum_dynamics, state_dim=4, action_dim=2)
        >>>
        >>> # 使用 pendulum 后端
        >>> pred.set_backend("pendulum")
        >>> state = PendulumState(theta=0.5, omega=0.0)
        >>> action = PendulumAction(torque=2.0)
        >>> trajectory = pred.predict(state, action, n_steps=10)
    """

    def __init__(self, default_backend: str = "pendulum", dt: float = 0.01):
        super().__init__(name="generalized_physics")
        self._dynamics_registry: dict[str, DynamicsFn] = {}
        self._state_dims: dict[str, int] = {}
        self._action_dims: dict[str, int] = {}
        self._backend: str = default_backend
        self._dt: float = dt

        # 注册内置动力学函数
        self.register_dynamics("pendulum", pendulum_dynamics, state_dim=2, action_dim=1)
        self.register_dynamics("cart", cart_dynamics, state_dim=2, action_dim=1)
        self.register_dynamics("double_pendulum", double_pendulum_dynamics, state_dim=4, action_dim=2)
        # v5.1.0: 新增物理系统
        self.register_dynamics("spring_mass", spring_mass_dynamics, state_dim=2, action_dim=1)
        self.register_dynamics("projectile", projectile_dynamics, state_dim=4, action_dim=2)
        self.register_dynamics("fluid_flow", fluid_flow_dynamics, state_dim=3, action_dim=2)

    @property
    def backend(self) -> str:
        """当前激活的后端名称。"""
        return self._backend

    @property
    def available_backends(self) -> list[str]:
        """所有已注册的后端名称。"""
        return list(self._dynamics_registry.keys())

    def set_backend(self, name: str) -> None:
        """切换预测器后端。"""
        if name not in self._dynamics_registry:
            raise ValueError(f"未知后端 '{name}'，可用: {self.available_backends}")
        self._backend = name

    def register_dynamics(
        self,
        name: str,
        dynamics_fn: DynamicsFn,
        state_dim: int = 2,
        action_dim: int = 1,
    ) -> None:
        """注册自定义动力学函数。

        Args:
            name: 后端名称（如 "my_robot"）
            dynamics_fn: 动力学函数 (state_vec, action_vec) → d_state/dt
            state_dim: 状态维度
            action_dim: 动作维度
        """
        self._dynamics_registry[name] = dynamics_fn
        self._state_dims[name] = state_dim
        self._action_dims[name] = action_dim
        logger.info("注册动力学后端: %s (state_dim=%d, action_dim=%d)", name, state_dim, action_dim)

    def predict(
        self,
        state: WorldState,
        action: Action | None,
        n_steps: int = 1,
    ) -> list[WorldState]:
        """使用 Euler 积分 + 注册的动力学函数做多步预测。

        支持两种模式:
        1. 自动推断后端: 根据 state 类型自动选择
        2. 手动指定后端: 使用 set_backend() 预设
        """
        state_vec = state.to_vector()

        # 动作向量编码
        if action is not None and hasattr(action, "to_vector"):
            action_vec = action.to_vector()
        else:
            # 根据后端确定动作维度，填充零
            action_dim = self._action_dims.get(self._backend, 1)
            action_vec = np.zeros(action_dim, dtype=np.float64)

        # 获取动力学函数
        backend = self._infer_backend(state)
        dynamics_fn = self._dynamics_registry[backend]
        dt = self._dt
        if hasattr(state, "dt"):
            dt = state.dt

        # Euler 积分
        trajectory: list[WorldState] = []
        current_vec = state_vec.copy()

        for _ in range(n_steps):
            current_vec = euler_step(current_vec, action_vec, dynamics_fn, dt)
            next_state = state.__class__.from_vector(current_vec)
            trajectory.append(next_state)

        return trajectory

    def _infer_backend(self, state: WorldState) -> str:
        """根据状态类型推断后端。"""
        # 显式设置的后端优先
        if self._backend in self._dynamics_registry:
            return self._backend

        # 根据状态维度推断
        state_dim = len(state.to_vector())
        for name, dim in self._state_dims.items():
            if dim == state_dim:
                return name

        # 默认返回 pendulum
        return "pendulum"

    def evaluate(
        self,
        dataset: list,
    ) -> dict:
        """在测试数据集上评估预测精度。"""
        distances = []
        for state, action, gt in dataset:
            preds = self.predict(state, action, n_steps=1)
            d = preds[0].distance(gt)
            distances.append(d)
            self._prediction_count += 1

        if not distances:
            return {"avg_distance": 1.0, "n": 0, "predictor": self._name}

        return {
            "avg_distance": round(float(np.mean(distances)), 6),
            "min_distance": round(float(np.min(distances)), 6),
            "max_distance": round(float(np.max(distances)), 6),
            "n": len(distances),
            "predictor": self._name,
            "backend": self._backend,
        }
