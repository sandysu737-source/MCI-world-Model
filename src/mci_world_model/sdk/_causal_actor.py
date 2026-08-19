from __future__ import annotations

from typing import Any

"""
MCI World Model v4.6.0 — Causal Actor
======================================

LeCun 六模块架构中的 Actor 模块：基于 Cost 梯度搜索最优干预动作。

核心功能:
- 接收 CausalWorldModelState + CostSignal → 搜索最优动作
- 使用有限差分法估计代价梯度，指导动作搜索
- 输出的 ActionCandidate 可直接应用于 World Model
- 支持三种动作类型: do_intervention / adjust_weight / suggest_edge

状态机: IDLE → SEARCHING → EVALUATING → SELECTING → APPLYING → COMPLETE

用法:
    from mci_world_model.sdk._causal_actor import CausalActor, ActionCandidate

    actor = CausalActor(world_model, cost_module)
    actions = actor.search(state, n_candidates=3)
    if actions:
        new_state = actor.apply(actions[0])
"""


import copy
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# ActionCandidate — 候选动作
# =============================================================================


@dataclass
class ActionCandidate:
    """
    候选动作。

    Attributes:
        action_type: 动作类型
            - "do_intervention": Pearl do-算子干预
            - "adjust_weight": 调整因果边权重
            - "suggest_edge": 建议新增/删除因果边
        target: 目标变量名称
        proposed_value: 建议值（权重/do值）
        expected_cost: 执行后预期代价降低值 (>0 表示代价降低)
        confidence: 执行置信度 [0, 1]
        metadata: 附加信息
    """

    action_type: str
    target: str
    proposed_value: float
    expected_cost: float
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target": self.target,
            "proposed_value": round(self.proposed_value, 6),
            "expected_cost": round(self.expected_cost, 6),
            "confidence": round(self.confidence, 4),
        }


# =============================================================================
# v3.0.5: EnergyGuidedAction — 能量流引导干预动作
# =============================================================================


@dataclass
class EnergyGuidedAction(ActionCandidate):
    """
    v3.0.5: 能量流引导的干预动作。

    继承 ActionCandidate，新增能量干预方向字段，
    可基于五行生克自动推断干预方向（增强/抑制）。

    Attributes:
        energy_direction: 能量干预方向 "enhance" | "suppress" | "neutral"
    """

    energy_direction: str = ""

    @classmethod
    def from_edge(cls: Any, edge: dict[str, Any], energy_core: Any, **kwargs: Any) -> None:
        """
        基于五行生克自动推断干预方向。

        生关系 → 提升权重 (enhance)
        克关系 → 降低权重 (suppress)

        Args:
            edge: 因果边 dict，含 cause_energy/effect_energy/rho
            energy_core: EnergyCore 实例
            **kwargs: 传递给 ActionCandidate 的其他参数

        Returns:
            EnergyGuidedAction 实例
        """
        cause_e = edge.get("cause_energy", "earth")
        effect_e = edge.get("effect_energy", "earth")
        direction = "neutral"
        rho = edge.get("rho", 0.5)
        proposed = kwargs.get("proposed_value", rho)

        if energy_core.get_enhance_relation(cause_e, effect_e):
            direction = "enhance"
            proposed = min(rho + 0.15, 1.0)  # 生 → 提升权重
        elif energy_core.get_suppress_relation(cause_e, effect_e):
            direction = "suppress"
            proposed = max(rho - 0.10, 0.0)  # 克 → 降低权重

        return cls(
            energy_direction=direction,
            proposed_value=proposed,
            **{k: v for k, v in kwargs.items() if k != "proposed_value"},
        )

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["energy_direction"] = self.energy_direction
        return d


# =============================================================================
# CausalActor — 因果动作搜索器
# =============================================================================


class CausalActor:
    """
    基于 Cost 梯度的动作搜索器。

    六态流转：IDLE → SEARCHING → EVALUATING → SELECTING → APPLYING → COMPLETE

    核心流程:
    1. 评估当前状态代价（baseline）
    2. 对每条候选边做有限差分扰动（±δ）→ 重新评估代价
    3. 选择代价降低最大的 Top-K 动作
    4. apply() 在实际状态上执行最优动作
    """

    MAX_ACTIONS: int = 5  # 最多候选动作
    MIN_COST_IMPROVEMENT: float = 0.01  # 最小代价改善阈值
    DEFAULT_DELTA: float = 0.1  # 有限差分扰动幅度

    def __init__(self, world_model: Any, cost_module: Any = None, energy_core: Any = None) -> None:
        """
        Args:
            world_model: MCIWorldModel 实例
            cost_module: EnergyCostModule 实例（可选，None 时使用 world_model._cost_module）
            energy_core: v3.0.4: EnergyCore 实例，用于能量亲和度引导（可选）
        """
        self._wm = world_model
        self._cost = cost_module
        self._energy_core = energy_core  # v3.0.4: 能量亲和度引导
        self._state: str = "IDLE"  # IDLE → SEARCHING → ... → COMPLETE
        self._action_history: list[ActionCandidate] = []

    # -----------------------------------------------------------------
    # 属性
    # -----------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def action_history(self) -> list[dict[str, Any]]:
        return [a.to_dict() for a in self._action_history[-20:]]

    def _get_cost_module(self) -> None:
        """延迟获取 Cost 模块（优先显式传参，其次 world_model 内部）。"""
        if self._cost is not None:
            return self._cost
        if self._wm._cost_module is not None:
            return self._wm._cost_module
        # 降级：创建临时 Cost 模块
        from mci_world_model.sdk._cost_module import EnergyCostModule

        self._cost = EnergyCostModule()
        logger.warning("CausalActor 降级使用临时 CostModule")
        return self._cost

    # -----------------------------------------------------------------
    # 动作搜索
    # -----------------------------------------------------------------

    def search(  # type: ignore
        self,
        state,
        n_candidates: int = 3,
        delta: float | None = None,
    ) -> list[ActionCandidate]:
        """
        基于 Cost 梯度搜索最优干预动作。

        使用有限差分法估计每条因果边上的代价偏导:
            dCost/dρ ≈ [Cost(ρ + δ) - Cost(ρ)] / δ

        Args:
            state: CausalWorldModelState 当前状态
            n_candidates: 返回 Top-K 候选动作
            delta: 有限差分步长（默认 0.1）

        Returns:
            按 expected_cost 降序排列的动作列表
        """
        self._state = "SEARCHING"

        if delta is None:
            delta = self.DEFAULT_DELTA

        if n_candidates < 1:
            self._state = "IDLE"
            return []

        cost_module = self._get_cost_module()  # type: ignore

        # ── 1. 基线代价评估 ──
        baseline = cost_module.evaluate(state)  # type: ignore
        baseline_total = baseline.total

        if not state.causal_edges:
            self._state = "IDLE"
            return []

        # ── 2. 有限差分梯度估计 ──
        self._state = "EVALUATING"
        candidates: list[ActionCandidate] = []

        # 对每条边做扰动评估
        for i, edge in enumerate(state.causal_edges):
            if i >= min(len(state.causal_edges), 20):  # 上限防止过长时间
                break

            rho_original = edge.get("rho", 0.0)
            cause = edge.get("cause", f"node_{i}")
            effect = edge.get("effect", f"effect_{i}")

            # 扰动 +
            perturbed_plus = copy.deepcopy(state)
            perturbed_plus.causal_edges[i]["rho"] = min(rho_original + delta, 1.0)
            cost_plus = cost_module.evaluate(perturbed_plus)  # type: ignore

            # 梯度估计
            grad = (cost_plus.total - baseline_total) / delta

            # 负梯度 → 调整方向使代价降低
            proposed_rho = max(min(rho_original - grad * delta, 1.0), 0.0)
            improvement = -grad * abs(proposed_rho - rho_original)

            if improvement > self.MIN_COST_IMPROVEMENT:
                candidate = ActionCandidate(
                    action_type="adjust_weight",
                    target=f"{cause}→{effect}",
                    proposed_value=round(proposed_rho, 4),
                    expected_cost=round(improvement, 6),
                    confidence=round(min(abs(grad) * 5, 1.0), 4),
                    metadata={
                        "edge_index": i,
                        "original_rho": rho_original,
                        "gradient": round(grad, 6),
                    },
                )

                # v3.0.4: 能量亲和度调整
                if self._energy_core is not None:
                    cause_energy = edge.get("cause_energy", "earth")
                    effect_energy = edge.get("effect_energy", "earth")
                    if self._energy_core.get_enhance_relation(cause_energy, effect_energy):
                        candidate.confidence *= 1.15  # 生关系 → 优先增强
                    elif self._energy_core.get_suppress_relation(cause_energy, effect_energy):
                        candidate.confidence *= 1.10  # 克关系 → 优先抑制

                candidates.append(candidate)

        # ── 3. Top-K 选择 ──
        self._state = "SELECTING"
        candidates.sort(key=lambda a: a.expected_cost, reverse=True)
        selected = candidates[: min(n_candidates, len(candidates))]

        self._state = "COMPLETE"
        return selected

    # -----------------------------------------------------------------
    # 动作执行
    # -----------------------------------------------------------------

    def apply(self, state: Any, action: ActionCandidate) -> None:
        """
        执行动作，返回新状态。

        在 state 的副本上执行动作，不修改原始状态。

        Args:
            state: 当前 CausalWorldModelState
            action: 待执行的动作

        Returns:
            执行后的新 CausalWorldModelState
        """
        self._state = "APPLYING"

        new_state = copy.deepcopy(state)

        if action.action_type == "adjust_weight":
            self._apply_adjust_weight(new_state, action)
        elif action.action_type == "do_intervention":
            self._apply_do_intervention(new_state, action)
        elif action.action_type == "suggest_edge":
            self._apply_suggest_edge(new_state, action)
        else:
            logger.warning("未知动作类型: %s", action.action_type)

        self._action_history.append(action)
        self._state = "COMPLETE"
        return new_state

    # -----------------------------------------------------------------
    # 动作实现（私有）
    # -----------------------------------------------------------------

    def _apply_adjust_weight(self, state: Any, action: ActionCandidate) -> None:
        """调整因果边权重。"""
        edge_idx = action.metadata.get("edge_index", -1)
        if 0 <= edge_idx < len(state.causal_edges):
            state.causal_edges[edge_idx]["rho"] = action.proposed_value
            logger.debug(
                "Actor 调整边权重: %s → %.4f",
                action.target,
                action.proposed_value,
            )

    def _apply_do_intervention(self, state: Any, action: ActionCandidate) -> None:
        """Pearl do-算子干预：设置目标变量为指定值。"""
        # 找到所有以 target 为 effect 的边，冻结其权重
        target = action.target
        for edge in state.causal_edges:
            if edge.get("effect") == target or edge.get("cause") == target:
                edge["rho"] = action.proposed_value
        state.do_interventions.append(
            {
                "do_x": {target: action.proposed_value},
                "expected_cost": action.expected_cost,
                "confidence": action.confidence,
            }
        )
        logger.debug("Actor do-干预: %s = %.4f", target, action.proposed_value)

    def _apply_suggest_edge(self, state: Any, action: ActionCandidate) -> None:
        """建议新增因果边。"""
        cause, effect = action.target.split("→", 1)
        new_edge = {
            "cause": cause.strip(),
            "effect": effect.strip(),
            "rho": action.proposed_value,
            "confidence": action.confidence,
            "verdict": "novel",
            "energy_relation": "neutral",
            "bayes_factor": 0.5,
        }
        state.causal_edges.append(new_edge)
        logger.debug("Actor 新增边: %s → %s (ρ=%.4f)", cause.strip(), effect.strip(), action.proposed_value)

    # -----------------------------------------------------------------
    # 迭代优化（Cost→Actor 闭环）
    # -----------------------------------------------------------------

    def optimize(  # type: ignore
        self,
        state,
        max_iterations: int = 3,
        delta: float | None = None,
    ) -> dict[str, Any]:
        """
        迭代优化：重复 search → apply 直到代价不再降低。

        这是 Cost→Actor 梯度闭环的完整实现。

        Args:
            state: 初始状态
            max_iterations: 最大迭代次数
            delta: 有限差分步长

        Returns:
            {"n_actions": int, "initial_cost": float, "final_cost": float, "actions": [...], "state": new_state}
        """
        cost_module = self._get_cost_module()  # type: ignore

        initial_signal = cost_module.evaluate(state)  # type: ignore
        initial_cost = initial_signal.total

        current_state = copy.deepcopy(state)
        executed_actions: list[dict[str, Any]] = []

        for iteration in range(max_iterations):
            actions = self.search(current_state, n_candidates=1, delta=delta)
            if not actions:
                break

            best = actions[0]
            if best.expected_cost <= self.MIN_COST_IMPROVEMENT:
                break

            current_state = self.apply(current_state, best)  # type: ignore
            executed_actions.append(best.to_dict())

            logger.info(
                "Actor 优化迭代 %d/%d: %s → 代价降低 %.6f",
                iteration + 1,
                max_iterations,
                best.target,
                best.expected_cost,
            )

        final_signal = cost_module.evaluate(current_state)  # type: ignore
        final_cost = final_signal.total

        return {
            "n_actions": len(executed_actions),
            "initial_cost": round(initial_cost, 6),
            "final_cost": round(final_cost, 6),
            "cost_reduction": round(initial_cost - final_cost, 6),
            "actions": executed_actions,
            "state": current_state,
        }
