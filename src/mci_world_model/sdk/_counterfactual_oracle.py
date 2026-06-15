"""MCI World Model v4.4.2 — CounterfactualOracle
===============================================

LLM↔CEWM 反馈闭环的核心组件——让 LLM 可以查询 CEWM 的反事实推演结果，
并将结果融入后续推理，实现"如果选 A 会怎样 → CEWM 推演 → 改为选 B"的决策闭环。

核心能力:
    batch_what_if(scenarios)   — 批量反事实推演
    rank_scenarios(scenarios)  — 推演后按目标函数排序
    query(hypotheses, goal)    — 完整查询流程: 假设→推演→排序→返回

架构:
    LLM 生成假设         CEWM 反事实推演           LLM 再推理
    ═══════════          ═══════════════          ══════════
    "如果方案A..."   →   query_counterfactual() →   "方案A导致指标↓5%"
    "如果方案B..."   →   query_counterfactual() →   "方案B导致指标↑8%"
    "如果方案C..."   →   query_counterfactual() →   "方案C导致指标↑3%"
                                                        ↓
                                                  "推荐方案B，因为..."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mci_world_model.sdk._counterfactual import CounterfactualEngine

logger = logging.getLogger(__name__)


# =============================================================================
# CScenario — 反事实场景
# =============================================================================


@dataclass
class CFScenario:
    """反事实场景——一个"如果...会怎样"的假设。

    Attributes:
        name: 场景名称（如 "方案A"）
        description: 场景描述
        intervention: 干预变量 {变量名: 干预值}
        target: 目标变量名
    """

    name: str
    description: str = ""
    intervention: dict[str, Any] = field(default_factory=dict)
    target: str = ""


# =============================================================================
# CFRanking — 反事实推演排序结果
# =============================================================================


@dataclass
class CFRanking:
    """反事实推演排序结果。

    Attributes:
        scenario: 场景信息
        counterfactual_value: 反事实推演结果值
        factual_value: 事实值（对照）
        effect: 因果效应 (counterfactual - factual)
        rank: 排名（0=最优）
        confidence: 置信度 [0, 1]
        is_uncertain: 是否为低置信度结果
    """

    scenario: CFScenario
    counterfactual_value: Any = None
    factual_value: Any = None
    effect: float = 0.0
    rank: int = -1
    confidence: float = 1.0
    is_uncertain: bool = False


# =============================================================================
# CounterfactualOracle — LLM↔CEWM 反馈 Oracle
# =============================================================================


class CounterfactualOracle:
    """反事实 Oracle——LLM 查询 CEWM 反事实推演的桥接器。

    用法:
        >>> oracle = CounterfactualOracle(world_model=wm)
        >>> scenarios = [
        ...     CFScenario(name="方案A", intervention={"treatment": "A"}),
        ...     CFScenario(name="方案B", intervention={"treatment": "B"}),
        ... ]
        >>> rankings = oracle.rank_scenarios(scenarios, goal="最大化营养指标")
        >>> best = rankings[0]  # 最优方案
        >>> print(f"推荐: {best.scenario.name}, 效应: {best.effect:.3f}")
    """

    def __init__(
        self,
        world_model: Any = None,
        counterfactual_engine: CounterfactualEngine | None = None,
    ):
        """
        Args:
            world_model: MCIWorldModel 实例（用于调用 query_counterfactual）
            counterfactual_engine: 独立的反事实引擎（可选，优先使用）
        """
        self._world_model = world_model
        self._cf_engine = counterfactual_engine
        self._query_count: int = 0

    @property
    def query_count(self) -> int:
        return self._query_count

    def batch_what_if(
        self,
        scenarios: list[CFScenario],
    ) -> list[CFRanking]:
        """批量反事实推演——对每个场景执行反事实查询。

        Args:
            scenarios: 反事实场景列表

        Returns:
            每个场景的推演结果（未排序）
        """
        results: list[CFRanking] = []

        for scenario in scenarios:
            self._query_count += 1

            if self._cf_engine is not None:
                try:
                    cf_result = self._cf_engine.query(
                        do_x=scenario.intervention,
                        target=scenario.target,
                    )
                    cf_value = getattr(cf_result, "counterfactual_value", None)
                    f_value = getattr(cf_result, "factual_value", None)
                    effect = float(getattr(cf_result, "individual_effect", 0.0) or 0.0)

                    results.append(
                        CFRanking(
                            scenario=scenario,
                            counterfactual_value=cf_value,
                            factual_value=f_value,
                            effect=effect,
                            confidence=0.9,
                            is_uncertain=False,
                        )
                    )
                    continue
                except Exception as e:
                    logger.warning("反事实引擎异常: %s，降级为模拟推演", e)

            # 降级: 使用 world_model 的推演能力
            if self._world_model is not None:
                try:
                    result = self._world_model.query_counterfactual(
                        do_x=scenario.intervention,
                        target=scenario.target,
                    )
                    cf_value = result.get("counterfactual_value") if isinstance(result, dict) else None
                    f_value = result.get("factual_value") if isinstance(result, dict) else None
                    effect = float(result.get("individual_effect", 0.0) if isinstance(result, dict) else 0.0)

                    results.append(
                        CFRanking(
                            scenario=scenario,
                            counterfactual_value=cf_value,
                            factual_value=f_value,
                            effect=effect,
                            confidence=0.7,
                            is_uncertain=False,
                        )
                    )
                    continue
                except Exception as e:
                    logger.warning("world_model 反事实查询异常: %s", e)

            # 最终降级: 标记为不确定
            results.append(
                CFRanking(
                    scenario=scenario,
                    counterfactual_value=None,
                    factual_value=None,
                    effect=0.0,
                    confidence=0.0,
                    is_uncertain=True,
                )
            )

        return results

    def rank_scenarios(
        self,
        scenarios: list[CFScenario],
        goal: str = "maximize",
        target_direction: str = "higher_is_better",
    ) -> list[CFRanking]:
        """批量推演 + 排序。

        Args:
            scenarios: 反事实场景列表
            goal: 优化目标描述
            target_direction: 'higher_is_better' 或 'lower_is_better'

        Returns:
            排序后的 CFRanking 列表（最优排第一）
        """
        rankings = self.batch_what_if(scenarios)

        # 排序: higher_is_better → effect 降序; lower_is_better → effect 升序
        reverse = target_direction == "higher_is_better"
        rankings.sort(key=lambda r: r.effect, reverse=reverse)

        # 赋排名
        for i, r in enumerate(rankings):
            r.rank = i

        return rankings

    def query(
        self,
        hypotheses: list[dict],
        goal: str = "maximize",
        target_direction: str = "higher_is_better",
    ) -> dict[str, Any]:
        """完整查询流程: 假设→推演→排序→返回。

        Args:
            hypotheses: 假设列表 [{"name": "A", "intervention": {...}, "target": "..."}]
            goal: 优化目标
            target_direction: 'higher_is_better' 或 'lower_is_better'

        Returns:
            {
                "best_scenario": 最优场景名,
                "best_effect": 最优效应,
                "rankings": [CFRanking.to_dict(), ...],
                "recommendation": 推荐字符串,
                "n_scenarios": int,
            }
        """
        scenarios = [
            CFScenario(
                name=h.get("name", f"假设{i}"),
                description=h.get("description", ""),
                intervention=h.get("intervention", {}),
                target=h.get("target", ""),
            )
            for i, h in enumerate(hypotheses)
        ]

        rankings = self.rank_scenarios(scenarios, goal, target_direction)

        best = rankings[0] if rankings else None
        recommendation = ""
        if best and not best.is_uncertain:
            recommendation = (
                f"基于世界模型反事实推演，推荐 {best.scenario.name}："
                f"效应={best.effect:.3f}，置信度={best.confidence:.1%}"
            )
        elif best and best.is_uncertain:
            recommendation = f"反事实推演结果不确定，{best.scenario.name} 可能较优，但建议结合更多信息决策"

        return {
            "best_scenario": best.scenario.name if best else None,
            "best_effect": best.effect if best else None,
            "rankings": [
                {
                    "name": r.scenario.name,
                    "effect": r.effect,
                    "rank": r.rank,
                    "confidence": r.confidence,
                    "is_uncertain": r.is_uncertain,
                }
                for r in rankings
            ],
            "recommendation": recommendation,
            "n_scenarios": len(scenarios),
        }
