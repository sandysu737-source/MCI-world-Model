from __future__ import annotations

"""MCI World Model — ExperimentDesigner 实验设计器
=================================================

自动设计因果验证实验——基于假设生成器的输出，
设计干预实验来验证或证伪因果假设。

核心能力:
    ExperimentPlan      — 实验计划
    ExperimentDesigner  — 实验设计器

设计原则:
    - 基于 HypothesisGenerator (T19) 的假设
    - 纯 numpy，零外部依赖
"""


import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExperimentPlan:
    """实验计划。

    Attributes:
        plan_id: 计划ID
        hypothesis_id: 关联假设ID
        intervention: 干预方案
        control_group: 对照组方案
        sample_size: 样本量
        expected_effect_size: 期望效应量
        statistical_power: 统计功效
        is_feasible: 是否可行
    """

    plan_id: str
    hypothesis_id: str = ""
    intervention: dict[str, Any] = field(default_factory=dict)
    control_group: dict[str, Any] = field(default_factory=dict)
    sample_size: int = 100
    expected_effect_size: float = 0.5
    statistical_power: float = 0.8
    is_feasible: bool = True


class ExperimentDesigner:
    """实验设计器 — 自动设计因果验证实验。

    用法:
        >>> designer = ExperimentDesigner()
        >>> plan = designer.design(hypothesis_cause="X", hypothesis_effect="Y")
    """

    def __init__(
        self,
        min_sample_size: int = 30,
        target_power: float = 0.8,
        alpha: float = 0.05,
    ):
        if min_sample_size < 10:
            raise ValueError("min_sample_size 必须 ≥ 10")
        self._min_n = min_sample_size
        self._target_power = target_power
        self._alpha = alpha
        self._plans: list[ExperimentPlan] = []
        self._plan_counter: int = 0

    @property
    def plan_count(self) -> int:
        return len(self._plans)

    def design(
        self,
        hypothesis_cause: str,
        hypothesis_effect: str,
        hypothesis_id: str = "",
        prior_effect_size: float = 0.5,
        available_variables: list[str] | None = None,
    ) -> ExperimentPlan:
        """设计因果验证实验。

        Args:
            hypothesis_cause: 假设原因
            hypothesis_effect: 假设结果
            hypothesis_id: 关联假设ID
            prior_effect_size: 先验效应量
            available_variables: 可用变量列表

        Returns:
            ExperimentPlan
        """
        self._plan_counter += 1
        plan_id = f"EXP{self._plan_counter:04d}"

        # 确定干预方案
        intervention = {
            "treatment_variable": hypothesis_cause,
            "action": f"设置 {hypothesis_cause} 为干预值",
            "measure_variable": hypothesis_effect,
        }

        # 确定对照组
        control_group = {
            "treatment_variable": hypothesis_cause,
            "action": f"保持 {hypothesis_cause} 为基线值",
            "measure_variable": hypothesis_effect,
        }

        # 样本量计算 (简化: 基于效应量)
        sample_size = self._compute_sample_size(prior_effect_size)

        # 统计功效
        power = self._compute_power(sample_size, prior_effect_size)

        # 可行性评估
        is_feasible = sample_size >= self._min_n and power >= 0.5 and prior_effect_size > 0.1

        plan = ExperimentPlan(
            plan_id=plan_id,
            hypothesis_id=hypothesis_id,
            intervention=intervention,
            control_group=control_group,
            sample_size=sample_size,
            expected_effect_size=prior_effect_size,
            statistical_power=power,
            is_feasible=is_feasible,
        )

        self._plans.append(plan)
        logger.info("实验设计: %s, 样本量=%d, 功效=%.2f, 可行=%s", plan_id, sample_size, power, is_feasible)
        return plan

    def design_batch(self, hypotheses: list[dict[str, Any]]) -> list[ExperimentPlan]:
        """批量设计实验。

        Args:
            hypotheses: 假设列表 [{cause, effect, id, effect_size}]

        Returns:
            ExperimentPlan 列表
        """
        plans = []
        for h in hypotheses:
            plan = self.design(
                hypothesis_cause=h.get("cause", ""),
                hypothesis_effect=h.get("effect", ""),
                hypothesis_id=h.get("id", ""),
                prior_effect_size=h.get("effect_size", 0.5),
            )
            plans.append(plan)
        return plans

    def _compute_sample_size(self, effect_size: float) -> int:
        """简化样本量计算。

        基于公式: n ≈ (z_α + z_β)² / effect_size²
        简化: n ≈ 16 / effect_size² (for α=0.05, power=0.8)
        """
        if effect_size < 0.01:
            return 10000
        n = int(16.0 / (effect_size**2))
        return max(n, self._min_n)

    def _compute_power(self, sample_size: int, effect_size: float) -> float:
        """简化统计功效计算。"""
        if effect_size < 0.01:
            return 0.05
        # 简化: power ≈ Φ(effect_size * √n / 2 - z_α)
        z = effect_size * np.sqrt(sample_size) / 2.0 - 1.96
        # 标准正态 CDF 近似
        power = 0.5 * (1 + np.tanh(z * 0.7978))
        return float(np.clip(power, 0.05, 0.99))

    def get_feasible_plans(self) -> list[ExperimentPlan]:
        """获取可行的实验计划。"""
        return [p for p in self._plans if p.is_feasible]

    def statistics(self) -> dict[str, Any]:
        feasible = sum(1 for p in self._plans if p.is_feasible)
        return {
            "plan_count": self.plan_count,
            "feasible_count": feasible,
            "feasibility_rate": feasible / max(self.plan_count, 1),
            "avg_sample_size": (float(np.mean([p.sample_size for p in self._plans])) if self._plans else 0.0),
            "target_power": self._target_power,
        }
