"""MCI World Model — HypothesisGenerator 假设生成器
=================================================

从已有因果知识和观测数据中自动生成可测试的因果假设，
支持假设排序、去重和可证伪性评估。

核心能力:
    CausalHypothesis    — 因果假设数据类
    HypothesisGenerator — 假设生成器

设计原则:
    - 基于 ScientificDiscoveryPipeline (T18) 的发现结果
    - 纯 numpy，零外部依赖
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# CausalHypothesis — 因果假设
# =============================================================================


@dataclass
class CausalHypothesis:
    """因果假设。

    Attributes:
        hypothesis_id: 假设ID
        cause: 原因变量
        effect: 结果变量
        mechanism: 因果机制描述
        prior_plausibility: 先验可信度 [0, 1]
        testability: 可测试性 [0, 1]
        falsifiability: 可证伪性 [0, 1]
        novelty: 新颖性 [0, 1]
        source: 假设来源
    """

    hypothesis_id: str
    cause: str = ""
    effect: str = ""
    mechanism: str = ""
    prior_plausibility: float = 0.5
    testability: float = 0.5
    falsifiability: float = 0.5
    novelty: float = 0.5
    source: str = ""


# =============================================================================
# HypothesisGenerator — 假设生成器
# =============================================================================


class HypothesisGenerator:
    """因果假设生成器 — 从因果知识中自动生成可测试假设。

    生成策略:
      - 基于已知因果边: 推断传递效应
      - 基于未连接变量: 生成潜在因果假设
      - 基于反事实: 生成干预假设

    用法:
        >>> gen = HypothesisGenerator()
        >>> gen.add_known_cause("X", "Y")
        >>> gen.add_known_cause("Y", "Z")
        >>> hypotheses = gen.generate()
    """

    def __init__(self, max_hypotheses: int = 50):
        if max_hypotheses < 1:
            raise ValueError("max_hypotheses 必须 ≥ 1")
        self._max_hypotheses = max_hypotheses
        self._known_causes: list[tuple[str, str, float]] = []  # (cause, effect, strength)
        self._variables: set[str] = set()
        self._hypotheses: list[CausalHypothesis] = []
        self._hypothesis_counter: int = 0

    @property
    def hypothesis_count(self) -> int:
        return len(self._hypotheses)

    def add_known_cause(self, cause: str, effect: str, strength: float = 0.5) -> None:
        """添加已知因果关系。"""
        self._known_causes.append((cause, effect, strength))
        self._variables.add(cause)
        self._variables.add(effect)

    def generate(self) -> list[CausalHypothesis]:
        """生成因果假设。

        Returns:
            CausalHypothesis 列表 (按优先级排序)
        """
        hypotheses = []

        # 策略1: 传递因果 (A→B, B→C ⟹ A→C?)
        hypotheses.extend(self._generate_transitive())

        # 策略2: 未连接变量对 (潜在因果)
        hypotheses.extend(self._generate_unconnected())

        # 策略3: 反向因果 (B→A?)
        hypotheses.extend(self._generate_reverse())

        # 去重
        seen = set()
        unique = []
        for h in hypotheses:
            key = (h.cause, h.effect)
            if key not in seen:
                seen.add(key)
                unique.append(h)

        # 排序: 综合得分
        unique.sort(key=lambda h: -(h.prior_plausibility * 0.4 + h.testability * 0.3 + h.novelty * 0.3))

        # 限制数量
        self._hypotheses = unique[: self._max_hypotheses]
        return list(self._hypotheses)

    def _generate_transitive(self) -> list[CausalHypothesis]:
        """传递因果假设生成。"""
        hypotheses = []
        for a, b, s1 in self._known_causes:
            for c, d, s2 in self._known_causes:
                if b == c and a != d:
                    h = CausalHypothesis(
                        hypothesis_id=self._next_id(),
                        cause=a,
                        effect=d,
                        mechanism=f"传递: {a}→{b}→{d}",
                        prior_plausibility=min(s1 * s2, 1.0),
                        testability=0.7,
                        falsifiability=0.8,
                        novelty=0.3,
                        source="transitive",
                    )
                    hypotheses.append(h)
        return hypotheses

    def _generate_unconnected(self) -> list[CausalHypothesis]:
        """未连接变量假设生成。"""
        hypotheses = []
        connected = {(a, b) for a, b, _ in self._known_causes}
        vars_list = list(self._variables)

        for i, v1 in enumerate(vars_list):
            for v2 in vars_list[i + 1 :]:
                if (v1, v2) not in connected and (v2, v1) not in connected:
                    h = CausalHypothesis(
                        hypothesis_id=self._next_id(),
                        cause=v1,
                        effect=v2,
                        mechanism=f"潜在因果: {v1}→{v2}",
                        prior_plausibility=0.3,
                        testability=0.6,
                        falsifiability=0.9,
                        novelty=0.8,
                        source="unconnected",
                    )
                    hypotheses.append(h)
        return hypotheses

    def _generate_reverse(self) -> list[CausalHypothesis]:
        """反向因果假设生成。"""
        hypotheses = []
        for cause, effect, strength in self._known_causes:
            h = CausalHypothesis(
                hypothesis_id=self._next_id(),
                cause=effect,
                effect=cause,
                mechanism=f"反向因果: {effect}→{cause} (与已知相反)",
                prior_plausibility=max(1.0 - strength, 0.1),
                testability=0.5,
                falsifiability=0.9,
                novelty=0.9,
                source="reverse",
            )
            hypotheses.append(h)
        return hypotheses

    def rank_hypotheses(
        self, weight_plausibility: float = 0.4, weight_testability: float = 0.3, weight_novelty: float = 0.3
    ) -> list[CausalHypothesis]:
        """重新排序假设。

        Args:
            weight_plausibility: 可信度权重
            weight_testability: 可测试性权重
            weight_novelty: 新颖性权重

        Returns:
            排序后的假设列表
        """
        self._hypotheses.sort(
            key=lambda h: (
                -(
                    h.prior_plausibility * weight_plausibility
                    + h.testability * weight_testability
                    + h.novelty * weight_novelty
                )
            )
        )
        return list(self._hypotheses)

    def _next_id(self) -> str:
        self._hypothesis_counter += 1
        return f"H{self._hypothesis_counter:04d}"

    def statistics(self) -> dict[str, Any]:
        return {
            "hypothesis_count": self.hypothesis_count,
            "known_causes": len(self._known_causes),
            "variables": len(self._variables),
            "max_hypotheses": self._max_hypotheses,
        }
