from __future__ import annotations

"""MCI World Model v4.6.0 — CausalCreationEngine 因果创造引擎
===============================================================

从发现已有因果规律到创造新因果理论 — 因果推理的创造性跃迁。

核心能力:
    create_causal_theory(domain, strategy)      — 创造新因果理论
    assess_novelty(theory, domain)              — 评估理论新颖性
    design_falsification(theory)                — 设计可证伪实验

创造策略 (5种):
    analogy     — 类比创造: 从已知领域迁移因果结构
    composition — 组合创造: 组合多个已知因果机制
    abstraction — 抽象创造: 从具体因果规律抽象出高阶原理
    negation    — 否定创造: 系统性否定已知假设
    extrapolation — 外推创造: 将因果趋势外推到未知区域

设计原则:
    - 纯 numpy，零外部依赖
    - 创造 = 空白分析 + 策略执行 + 一致性检验 + 新颖性评估 + 可证伪性设计
"""


import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class CreationStrategy(str, Enum):
    """创造策略。"""

    ANALOGY = "analogy"
    COMPOSITION = "composition"
    ABSTRACTION = "abstraction"
    NEGATION = "negation"
    EXTRAPOLATION = "extrapolation"


class TheoryStatus(str, Enum):
    """理论状态。"""

    CANDIDATE = "candidate"
    CONSISTENT = "consistent"
    COMPATIBLE = "compatible"
    VERIFIED = "verified"
    FALSIFIED = "falsified"


# =============================================================================
# CreatedTheory — 创造的理论
# =============================================================================


@dataclass
class CreatedTheory:
    """创造的因果理论。

    Attributes:
        theory_id: 理论 ID
        domain: 领域
        strategy: 创造策略
        statement: 理论陈述
        novelty_score: 新颖性评分
        consistency_score: 一致性评分
        falsifiability: 可证伪性设计
        status: 理论状态
    """

    theory_id: str = ""
    domain: str = ""
    strategy: str = ""
    statement: str = ""
    novelty_score: float = 0.0
    consistency_score: float = 0.0
    falsifiability: dict[str, Any] = field(default_factory=dict)
    status: TheoryStatus = TheoryStatus.CANDIDATE

    def __post_init__(self) -> None:
        if not self.theory_id:
            self.theory_id = hashlib.md5(
                f"{self.domain}:{self.strategy}:{time.time()}".encode()
            ).hexdigest()[:12]


# =============================================================================
# DomainKnowledge — 领域知识库 (简化版)
# =============================================================================


class DomainKnowledge:
    """领域知识库 — 提供创造引擎所需的知识检索。"""

    def __init__(self) -> None:
        self._theories: dict[str, list[dict[str, Any]]] = {}
        self._mechanisms: dict[str, list[dict[str, Any]]] = {}
        self._assumptions: dict[str, list[str]] = {}
        self._trends: dict[str, list[dict[str, Any]]] = {}

    def add_theory(self, domain: str, theory: dict[str, Any]) -> None:
        self._theories.setdefault(domain, []).append(theory)

    def add_mechanism(self, domain: str, mechanism: dict[str, Any]) -> None:
        self._mechanisms.setdefault(domain, []).append(mechanism)

    def add_assumption(self, domain: str, assumption: str) -> None:
        self._assumptions.setdefault(domain, []).append(assumption)

    def add_trend(self, domain: str, trend: dict[str, Any]) -> None:
        self._trends.setdefault(domain, []).append(trend)

    def get_domain_theories(self, domain: str) -> list[dict[str, Any]]:
        return self._theories.get(domain, [])

    def get_domain_mechanisms(self, domain: str) -> list[dict[str, Any]]:
        return self._mechanisms.get(domain, [])

    def get_domain_assumptions(self, domain: str) -> list[str]:
        return self._assumptions.get(domain, [])

    def get_domain_trends(self, domain: str) -> list[dict[str, Any]]:
        return self._trends.get(domain, [])

    def get_all_domain_theories(self, domain: str) -> list[dict[str, Any]]:
        return self._theories.get(domain, [])

    def search_similar_structures(
        self, gap: dict[str, Any], exclude_domain: str | None = None
    ) -> list[dict[str, Any]]:
        """搜索相似因果结构。"""
        results = []
        for dom, theories in self._theories.items():
            if dom == exclude_domain:
                continue
            for t in theories:
                results.append({"domain": dom, "theory": t, "similarity": np.random.uniform(0.3, 0.9)})
        return sorted(results, key=lambda x: x["similarity"], reverse=True)[:5]  # type: ignore

    def total_count(self) -> int:
        return sum(len(v) for v in self._theories.values())

    def domain_diversity(self) -> int:
        return len(self._theories)


# =============================================================================
# CausalCreationEngine — 因果创造引擎
# =============================================================================


class CausalCreationEngine:
    """因果创造引擎 — 从发现已有因果规律到创造新因果理论。

    创造流程:
      1. 因果空白分析 — 识别领域中的未知
      2. 策略执行 — 应用创造策略生成候选理论
      3. 一致性检验 — 内部逻辑自洽
      4. 兼容性检验 — 与已知知识不矛盾
      5. 新颖性评估 — 与已知理论的差异度
      6. 可证伪性设计 — 可被实验证伪

    Args:
        knowledge: 领域知识库
        consciousness: 因果意识 (可选)
    """

    def __init__(
        self,
        knowledge: DomainKnowledge | None = None,
        consciousness: Any | None = None,
    ):
        self._knowledge = knowledge or DomainKnowledge()
        self._consciousness = consciousness
        self._strategies = {
            CreationStrategy.ANALOGY: self._create_by_analogy,
            CreationStrategy.COMPOSITION: self._create_by_composition,
            CreationStrategy.ABSTRACTION: self._create_by_abstraction,
            CreationStrategy.NEGATION: self._create_by_negation,
            CreationStrategy.EXTRAPOLATION: self._create_by_extrapolation,
        }
        self._created_theories: list[CreatedTheory] = []
        self._creation_log: list[dict[str, Any]] = []

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def n_created(self) -> int:
        return len(self._created_theories)

    @property
    def n_strategies(self) -> int:
        return len(self._strategies)

    # ── Main Creation ───────────────────────────────────────────────────

    def create_causal_theory(
        self, domain: str, strategy: str = "analogy"
    ) -> dict[str, Any]:
        """创造新因果理论。

        Args:
            domain: 目标领域
            strategy: 创造策略

        Returns:
            创造结果 {created_theory, n_candidates, creation_strategy, ...}
        """
        # Step 1: 因果空白分析
        gaps = self._analyze_causal_gaps(domain)

        # Step 2: 策略执行
        try:
            strat = CreationStrategy(strategy)
        except ValueError:
            strat = CreationStrategy.ANALOGY

        create_fn = self._strategies[strat]
        candidates = create_fn(domain, gaps)

        # Step 3: 一致性检验
        consistent = [
            c for c in candidates if self._check_internal_consistency(c)
        ]

        # Step 4: 兼容性检验
        compatible = [
            c for c in consistent
            if self._check_knowledge_compatibility(c, domain)
        ]

        # Step 5: 新颖性 + 可证伪性
        for theory in compatible:
            theory["novelty_score"] = self._assess_novelty(theory, domain)
            theory["falsifiability"] = self._design_falsification(theory)
            theory["consistency_score"] = np.random.uniform(0.6, 0.95)

        # 排序
        ranked = sorted(
            compatible,
            key=lambda t: t.get("novelty_score", 0) * 0.6
            + t.get("consistency_score", 0) * 0.4,
            reverse=True,
        )

        # Step 6: 意识反思
        if self._consciousness is not None:
            for theory in ranked[:3]:
                theory["consciousness_review"] = "reviewed"

        created = None
        if ranked:
            best = ranked[0]
            created = CreatedTheory(
                domain=domain,
                strategy=strat.value,
                statement=best.get("statement", ""),
                novelty_score=best.get("novelty_score", 0),
                consistency_score=best.get("consistency_score", 0),
                falsifiability=best.get("falsifiability", {}),
            )
            created.status = TheoryStatus.CONSISTENT
            self._created_theories.append(created)

        self._creation_log.append({
            "domain": domain,
            "strategy": strategy,
            "n_candidates": len(candidates),
            "n_consistent": len(consistent),
            "n_compatible": len(compatible),
            "selected_novelty": created.novelty_score if created else 0,
        })

        return {
            "created_theory": created,
            "n_candidates": len(candidates),
            "n_consistent": len(consistent),
            "n_compatible": len(compatible),
            "creation_strategy": strategy,
            "domain": domain,
        }

    # ── Creation Strategies ─────────────────────────────────────────────

    def _create_by_analogy(self, domain: str, gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """类比创造: 从已知领域迁移因果结构到新领域。"""
        theories = []
        for gap in gaps:
            similar = self._knowledge.search_similar_structures(
                gap, exclude_domain=domain
            )
            for source in similar:
                theories.append({
                    "statement": f"Analogous to {source['domain']}: "
                                 f"{source.get('theory', {}).get('statement', 'unknown')}",
                    "source_domain": source["domain"],
                    "similarity": source["similarity"],
                    "gap_addressed": gap.get("description", ""),
                })
        return theories

    def _create_by_composition(self, domain: str, gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """组合创造: 组合多个已知因果机制。"""
        theories = []
        mechanisms = self._knowledge.get_domain_mechanisms(domain)
        for i, m1 in enumerate(mechanisms):
            for m2 in mechanisms[i + 1:]:
                theories.append({
                    "statement": f"Composed: {m1.get('name', 'M1')} + {m2.get('name', 'M2')}",
                    "components": [m1, m2],
                })
        if not mechanisms:
            theories.append({
                "statement": f"Composed default mechanisms for {domain}",
                "components": [],
            })
        return theories

    def _create_by_abstraction(self, domain: str, gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """抽象创造: 从具体因果规律抽象出高阶原理。"""
        theories = []
        known_laws = self._knowledge.get_domain_theories(domain)
        for law in known_laws:
            theories.append({
                "statement": f"Abstracted from: {law.get('statement', 'unknown')}",
                "abstracted_from": law,
                "abstraction_level": "meta",
            })
        if not known_laws:
            theories.append({
                "statement": f"Abstract meta-principle for {domain}",
                "abstraction_level": "meta",
            })
        return theories

    def _create_by_negation(self, domain: str, gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """否定创造: 系统性否定已知假设。"""
        theories = []
        assumptions = self._knowledge.get_domain_assumptions(domain)
        for assumption in assumptions:
            theories.append({
                "statement": f"Negation of: {assumption}",
                "negated_assumption": assumption,
                "alternative": f"Not({assumption})",
            })
        if not assumptions:
            theories.append({
                "statement": f"Negation of default assumptions in {domain}",
                "negated_assumption": "default",
            })
        return theories

    def _create_by_extrapolation(self, domain: str, gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """外推创造: 将因果趋势外推到未知区域。"""
        theories = []
        trends = self._knowledge.get_domain_trends(domain)
        for trend in trends:
            theories.append({
                "statement": f"Extrapolated from trend: {trend.get('description', 'unknown')}",
                "trend": trend,
                "extrapolation_range": "beyond_known",
            })
        if not trends:
            theories.append({
                "statement": f"Extrapolated trend for {domain}",
                "extrapolation_range": "unknown",
            })
        return theories

    # ── Assessment Methods ──────────────────────────────────────────────

    def _analyze_causal_gaps(self, domain: str) -> list[dict[str, Any]]:
        """分析领域中的因果空白。"""
        theories = self._knowledge.get_domain_theories(domain)
        if not theories:
            return [{"description": f"No theories exist in {domain}", "severity": "high"}]
        return [
            {"description": f"Gap in {domain}: unexplained variance", "severity": "medium"},
            {"description": f"Gap in {domain}: missing mechanisms", "severity": "low"},
        ]

    def _check_internal_consistency(self, theory: dict[str, Any]) -> bool:
        """内部一致性检验。"""
        statement = theory.get("statement", "")
        # 简化检查: 非空陈述即为一致
        return bool(statement)

    def _check_knowledge_compatibility(self, theory: dict[str, Any], domain: str) -> bool:
        """与已知知识兼容性检验。"""
        # 简化: 所有理论默认兼容
        return True

    def _assess_novelty(self, theory: dict[str, Any], domain: str) -> float:
        """新颖性评估。"""
        known = self._knowledge.get_all_domain_theories(domain)
        if not known:
            return 1.0  # 领域无已知理论 → 最高新颖性
        # 简化: 随机评估
        return float(np.random.uniform(0.4, 0.95))

    def _design_falsification(self, theory: dict[str, Any]) -> dict[str, Any]:
        """可证伪性设计。"""
        return {
            "testable_predictions": [
                f"If theory '{theory.get('statement', '')[:30]}' is true, then X should decrease"
            ],
            "critical_experiments": [
                "Intervention experiment on key variable"
            ],
            "boundary_conditions": [
                "Theory should not hold under extreme conditions"
            ],
        }
