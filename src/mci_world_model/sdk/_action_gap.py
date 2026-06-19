"""MCI World Model — Heidegger 参与式距离度量 (ActionGapMetric)

CEWM v3.6.0 新增组件 (N6)：
目标导向的语义距离度量，区别于纯物理欧氏距离。

理论基础：
    1. Heidegger "在世存在" (Being-in-the-world) — 距离由行动代价定义
    2. 参与式度量: distance(state, goal) ≠ physical_dist(state, goal)
    3. 行动空间中的测地线距离

核心区别：
    - 物理距离: ‖state - goal‖₂ (欧氏距离)
    - 行动距离: 从 state 到 goal 所需的最小行动代价

    例: 单摆 θ₁=0.1, goal=π (倒立平衡点)
    - 物理距离: |0.1 - π| ≈ 3.04
    - 行动距离: 需要大推力克服重力势垒 → 远大于 3.04

核心能力：
    - distance(state, goal) — 行动距离
    - physical_distance(state, goal) — 物理距离（对照）
    - energy_barrier(state, goal) — 能量势垒
    - action_cost(state, action, goal) — 单步行动代价
    - reachable(state, goal, budget) — 预算内是否可达
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# 数据类型
# =============================================================================


@dataclass
class ActionCostResult:
    """行动代价评估结果。

    Attributes:
        action_distance: 行动距离（语义距离）
        physical_distance: 物理距离（欧氏距离）
        energy_barrier: 能量势垒
        action_effort: 行动努力度
        is_reachable: 是否在预算内可达
        ratio: 行动距离/物理距离 比值 (>1 表示行动比物理更远)
    """

    action_distance: float = 0.0
    physical_distance: float = 0.0
    energy_barrier: float = 0.0
    action_effort: float = 0.0
    is_reachable: bool = True
    ratio: float = 1.0

    def to_dict(self) -> dict:
        return {
            "action_distance": round(self.action_distance, 4),
            "physical_distance": round(self.physical_distance, 4),
            "energy_barrier": round(self.energy_barrier, 4),
            "action_effort": round(self.action_effort, 4),
            "is_reachable": self.is_reachable,
            "ratio": round(self.ratio, 4),
        }


@dataclass
class ActionGapConfig:
    """度量配置参数。

    Attributes:
        energy_weight: 能量势垒权重 (默认 0.4)
        effort_weight: 行动努力度权重 (默认 0.3)
        base_weight: 基础距离权重 (默认 0.3)
        gravity_factor: 重力修正因子 (默认 1.0)
        budget: 行动预算上限
    """

    energy_weight: float = 0.4
    effort_weight: float = 0.3
    base_weight: float = 0.3
    gravity_factor: float = 1.0
    budget: float = 100.0


# =============================================================================
# ActionGapMetric 主类
# =============================================================================


@dataclass
class ActionGapMetric:
    """Heidegger 参与式距离度量。

    核心主张：
        "距离"不是空间中的度量，而是行动中参与的代价。
        一把椅子在 3 米外，但你需要绕过桌子才能到达 —
        行动距离 > 物理距离。

    对于单摆系统：
        - 物理距离 = |θ₁ - θ₂|（角度差）
        - 行动距离 = 物理距离 + 能量势垒惩罚 + 行动努力度

        能量势垒：
            当目标在重力势阱的另一侧时，需要克服势能峰值。
            V(θ) = -mgl·cos(θ)
            barrier = max(V_along_path) - V(current)

    Example:
        >>> metric = ActionGapMetric()
        >>> from mci_world_model.sdk._world_state import PendulumState
        >>> state = PendulumState(theta=0.1, omega=0.0)
        >>> goal = PendulumState(theta=3.14, omega=0.0)
        >>> result = metric.distance(state, goal)
        >>> assert result.action_distance > result.physical_distance
    """

    config: ActionGapConfig = field(default_factory=ActionGapConfig)
    _call_count: int = 0
    _total_action_dist: float = 0.0
    _total_physical_dist: float = 0.0

    # ── 核心度量 ──

    def distance(self, state: Any, goal: Any, budget: float | None = None) -> ActionCostResult:
        """计算从 state 到 goal 的行动距离。

        综合公式：
            action_distance = w_base × base_dist
                            + w_energy × energy_barrier
                            + w_effort × action_effort

        Args:
            state: 当前状态（支持 PendulumState 或通用 WorldState）
            goal: 目标状态
            budget: 可选行动预算

        Returns:
            ActionCostResult 包含行动距离和物理距离
        """
        phys_dist = self.physical_distance(state, goal)
        energy_bar = self.energy_barrier(state, goal)
        effort = self._action_effort(state, goal)

        cfg = self.config
        action_dist = cfg.base_weight * phys_dist + cfg.energy_weight * energy_bar + cfg.effort_weight * effort

        ratio = action_dist / phys_dist if phys_dist > 1e-8 else 1.0

        use_budget = budget if budget is not None else cfg.budget
        is_reachable = action_dist <= use_budget

        result = ActionCostResult(
            action_distance=action_dist,
            physical_distance=phys_dist,
            energy_barrier=energy_bar,
            action_effort=effort,
            is_reachable=is_reachable,
            ratio=ratio,
        )

        # 统计
        self._call_count += 1
        self._total_action_dist += action_dist
        self._total_physical_dist += phys_dist

        return result

    # ── 分量计算 ──

    def physical_distance(self, state: Any, goal: Any) -> float:
        """计算物理欧氏距离。

        支持：
            1. PendulumState: 使用 θ/ω 空间的角距离
            2. 通用 WorldState: 使用 to_vector() → L2 距离
            3. numpy 数组: 直接 L2 距离
            4. 数值: |a - b|
        """
        if isinstance(state, (int, float)) and isinstance(goal, (int, float)):
            return abs(float(state) - float(goal))

        if isinstance(state, np.ndarray) and isinstance(goal, np.ndarray):
            return float(np.linalg.norm(state - goal))

        # WorldState 接口
        if hasattr(state, "to_vector") and hasattr(goal, "to_vector"):
            vec_s = state.to_vector()
            vec_g = goal.to_vector()
            return float(np.linalg.norm(vec_s - vec_g))

        # PendulumState 特殊处理（角距离）
        if hasattr(state, "theta") and hasattr(goal, "theta"):
            d_theta = self._angular_distance(state.theta, goal.theta)
            d_omega = abs(state.omega - goal.omega)
            return math.sqrt(d_theta**2 + d_omega**2)

        return 0.0

    def energy_barrier(self, state: Any, goal: Any) -> float:
        """计算从 state 到 goal 路径上的能量势垒。

        对单摆系统：
            V(θ) = -mgl·cos(θ)
            barrier = max(V(θ) for θ in path) - V(state)

        物理意义：如果 goal 在重力势阱的另一侧，
        需要额外能量克服势能峰值。
        """
        if not hasattr(state, "theta") or not hasattr(goal, "theta"):
            return 0.0

        g = getattr(state, "g", 9.81)
        mass_l = getattr(state, "L", 1.0)

        # 当前势能
        v_current = -g * mass_l * math.cos(state.theta)

        # 目标势能
        v_goal = -g * mass_l * math.cos(goal.theta)

        # 沿路径的最大势能（路径插值）
        theta_s = state.theta
        theta_g = goal.theta
        n_steps = 20
        v_max = v_current

        for i in range(n_steps + 1):
            t = i / n_steps
            theta_interp = theta_s + t * self._signed_angular_distance(theta_s, theta_g)
            v_interp = -g * mass_l * math.cos(theta_interp)
            v_max = max(v_max, v_interp)

        # 势垒 = 路径最大势能 - 当前势能
        barrier = max(0.0, v_max - v_current)

        # 如果目标在更高的势能位置，还需要额外的势能
        if v_goal > v_current:
            barrier += (v_goal - v_current) * self.config.gravity_factor

        return barrier

    def action_cost(self, state: Any, action: Any, goal: Any) -> float:
        """评估从 state 执行 action 后到 goal 的剩余代价。

        Args:
            state: 当前状态
            action: 执行的动作（对单摆：力矩值）
            goal: 目标状态

        Returns:
            剩余行动距离
        """
        # 模拟执行动作后的状态
        next_state = self._simulate_action(state, action)
        return self.distance(next_state, goal).action_distance

    def reachable(self, state: Any, goal: Any, budget: float) -> bool:
        """判断在给定预算内是否可达。"""
        return self.distance(state, goal, budget=budget).is_reachable

    # ── 统计 ──

    def statistics(self) -> dict:
        """度量统计信息。"""
        return {
            "call_count": self._call_count,
            "avg_action_distance": (self._total_action_dist / self._call_count if self._call_count > 0 else 0.0),
            "avg_physical_distance": (self._total_physical_dist / self._call_count if self._call_count > 0 else 0.0),
            "avg_ratio": (
                (self._total_action_dist / self._total_physical_dist) if self._total_physical_dist > 0 else 1.0
            ),
        }

    def reset_stats(self) -> None:
        """重置统计计数器。"""
        self._call_count = 0
        self._total_action_dist = 0.0
        self._total_physical_dist = 0.0

    # ── 内部方法 ──

    def _action_effort(self, state: Any, goal: Any) -> float:
        """行动努力度：基于状态差异的非线性放大。

        公式：
            effort = phys_dist × (1 + |velocity| / v_max)

        物理意义：当前速度越大，改变方向越困难。
        v4.4.0: 泛化为任意 WorldState，不再硬编码 omega。
        """
        phys_dist = self.physical_distance(state, goal)
        velocity_factor = 1.0

        # 优先检查 omega (PendulumState)，其次 v (CartState)，
        # 最后用 to_vector() 的第二维（如果有）
        if hasattr(state, "omega"):
            v_max = 10.0  # 归一化最大角速度
            velocity_factor = 1.0 + abs(state.omega) / v_max
        elif hasattr(state, "v"):
            v_max = 10.0  # 归一化最大线速度
            velocity_factor = 1.0 + abs(state.v) / v_max
        elif hasattr(state, "to_vector"):
            vec = state.to_vector()
            if len(vec) >= 2:
                v_max = 10.0
                velocity_factor = 1.0 + min(abs(float(vec[1])), v_max) / v_max

        return phys_dist * velocity_factor

    def _simulate_action(self, state: Any, action: Any) -> Any:
        """模拟执行动作后的状态。

        GEN-02 (W-2): 移除 PendulumState 硬编码回退，
        改为通用委托模式。物理模拟应通过注入的 PredictorProtocol 实现。
        """
        # 优先：如果 action 有 apply 方法，委托给它
        if hasattr(action, "apply") and callable(action.apply):
            try:
                return action.apply(state)
            except TypeError:
                pass  # action 不匹配 state 类型，回退

        # 回退 1: 如果 state 有 step_physics 方法，做自然演化
        if hasattr(state, "step_physics") and callable(state.step_physics):
            return state.step_physics()

        # GEN-02: 无可用物理模拟器，返回原状态并记录警告
        logger.warning("_simulate_action: 无可用物理模拟器 (state=%s)，返回原状态", type(state).__name__)
        return state

    @staticmethod
    def _angular_distance(a: float, b: float) -> float:
        """角度最短弧距离。"""
        diff = (b - a + math.pi) % (2 * math.pi) - math.pi
        return abs(diff)

    @staticmethod
    def _signed_angular_distance(a: float, b: float) -> float:
        """有符号角度最短弧距离。"""
        return (b - a + math.pi) % (2 * math.pi) - math.pi
