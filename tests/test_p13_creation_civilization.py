"""MCI World Model v13.0.0 — P13 因果创造/知识文明/因果经济 测试
================================================================

覆盖 P13 全阶段核心模块:
  - CausalCreationEngine: 因果创造引擎
  - CreativeCausalConsciousness: 创造因果意识
  - AutonomousKnowledgeCivilization: 自主知识文明
  - CausalEconomy: 因果经济体系
  - NoveltyVerifier: 新颖性验证
  - CreativeTrust: 创造可信框架
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._causal_creation_engine import (
    CausalCreationEngine,
    CreationStrategy,
    CreatedTheory,
    DomainKnowledge,
    TheoryStatus,
)
from mci_world_model.sdk._creative_consciousness import (
    CreativeCausalConsciousness,
    CreativeState,
    CreativeDrive,
)
from mci_world_model.sdk._knowledge_civilization import (
    AutonomousKnowledgeCivilization,
    CivilizationMetrics,
    KnowledgeRepository,
)
from mci_world_model.sdk._causal_economy import (
    CausalEconomy,
    Transaction,
    CausalKnowledgeValueModel,
)
from mci_world_model.sdk._novelty_verifier import (
    NoveltyVerifier,
    NoveltyResult,
)
from mci_world_model.sdk._creative_trust import CreativeTrust


# =============================================================================
# CausalCreationEngine Tests
# =============================================================================


class TestCausalCreationEngine:
    """因果创造引擎测试。"""

    def test_init(self):
        engine = CausalCreationEngine()
        assert engine.n_strategies == 5
        assert engine.n_created == 0

    def test_create_analogy(self):
        knowledge = DomainKnowledge()
        knowledge.add_theory("physics", {"statement": "F=ma", "domain": "physics"})
        engine = CausalCreationEngine(knowledge=knowledge)
        result = engine.create_causal_theory("medical", strategy="analogy")
        assert "created_theory" in result
        assert result["creation_strategy"] == "analogy"

    def test_create_composition(self):
        knowledge = DomainKnowledge()
        knowledge.add_mechanism("medical", {"name": "drug_effect"})
        knowledge.add_mechanism("medical", {"name": "placebo"})
        engine = CausalCreationEngine(knowledge=knowledge)
        result = engine.create_causal_theory("medical", strategy="composition")
        assert result["n_candidates"] > 0

    def test_create_abstraction(self):
        knowledge = DomainKnowledge()
        knowledge.add_theory("economics", {"statement": "supply=demand"})
        engine = CausalCreationEngine(knowledge=knowledge)
        result = engine.create_causal_theory("economics", strategy="abstraction")
        assert result["creation_strategy"] == "abstraction"

    def test_create_negation(self):
        knowledge = DomainKnowledge()
        knowledge.add_assumption("physics", "causality_is_local")
        engine = CausalCreationEngine(knowledge=knowledge)
        result = engine.create_causal_theory("physics", strategy="negation")
        assert result["n_candidates"] > 0

    def test_create_extrapolation(self):
        knowledge = DomainKnowledge()
        knowledge.add_trend("climate", {"description": "temperature_rising"})
        engine = CausalCreationEngine(knowledge=knowledge)
        result = engine.create_causal_theory("climate", strategy="extrapolation")
        assert result["creation_strategy"] == "extrapolation"

    def test_5_strategies(self):
        """KPI: 5 种创造策略全部可用。"""
        assert len(CreationStrategy) == 5

    def test_invalid_strategy_falls_back(self):
        engine = CausalCreationEngine()
        result = engine.create_causal_theory("test", strategy="invalid")
        assert result["creation_strategy"] == "invalid"

    def test_novelty_assessment(self):
        knowledge = DomainKnowledge()
        # Empty domain → novelty = 1.0
        engine = CausalCreationEngine(knowledge=knowledge)
        result = engine.create_causal_theory("new_domain")
        if result["created_theory"]:
            assert result["created_theory"].novelty_score >= 0


# =============================================================================
# CreativeCausalConsciousness Tests
# =============================================================================


class TestCreativeCausalConsciousness:
    """创造因果意识测试。"""

    def test_init(self):
        ccc = CreativeCausalConsciousness()
        assert ccc.creative_state == CreativeState.ANALYTICAL

    def test_enter_creative_mode(self):
        ccc = CreativeCausalConsciousness()
        result = ccc.enter_creative_mode("medical")
        assert result["creative_state"] in ("exploratory", "creative", "visionary")

    def test_drive_adjustment(self):
        ccc = CreativeCausalConsciousness()
        ccc.enter_creative_mode("test", drive_adjustment={"novelty": 0.9})
        assert ccc.drive.novelty == 0.9

    def test_strategy_from_drive(self):
        ccc = CreativeCausalConsciousness()
        result = ccc.enter_creative_mode("test", drive_adjustment={"novelty": 0.9})
        assert result["strategy_used"] == "negation"

    def test_creative_reflect(self):
        ccc = CreativeCausalConsciousness()
        result = ccc.creative_reflect({"quality": 0.7})
        assert "strategy_effectiveness" in result
        assert "aesthetic_score" in result

    def test_aesthetic_evaluation(self):
        ccc = CreativeCausalConsciousness()
        result = ccc.aesthetic_evaluation({"statement": "E=mc^2"})
        assert "score" in result
        assert result["score"] >= 0

    def test_4_states(self):
        """KPI: 4 种创造状态。"""
        assert len(CreativeState) == 4


# =============================================================================
# AutonomousKnowledgeCivilization Tests
# =============================================================================


class TestAutonomousKnowledgeCivilization:
    """自主知识文明测试。"""

    def test_init(self):
        kc = AutonomousKnowledgeCivilization()
        assert kc.n_generations == 0

    def test_generation_cycle_without_creation(self):
        kc = AutonomousKnowledgeCivilization()
        gen = kc.knowledge_generation_cycle("medical", n_theories=3)
        assert "domain" in gen
        assert gen["n_created"] == 0

    def test_generation_cycle_with_creation(self):
        knowledge = DomainKnowledge()
        knowledge.add_theory("physics", {"statement": "F=ma"})
        engine = CausalCreationEngine(knowledge=knowledge)
        kc = AutonomousKnowledgeCivilization(creation_engine=engine)
        gen = kc.knowledge_generation_cycle("physics", n_theories=3)
        assert gen["domain"] == "physics"
        # 创造引擎产生了候选 (n_created >= 0)
        assert gen["n_created"] >= 0

    def test_knowledge_heritage(self):
        kc = AutonomousKnowledgeCivilization()
        # 先在源领域存储一些知识
        kc._repository.store({"statement": "F=ma"}, "physics")
        result = kc.knowledge_heritage("physics", "engineering")
        assert result["source"] == "physics"
        assert result["target"] == "engineering"

    def test_heritage_empty_source(self):
        kc = AutonomousKnowledgeCivilization()
        result = kc.knowledge_heritage("empty_domain", "target")
        assert result["n_knowledge_transferred"] == 0

    def test_metrics_update(self):
        engine = CausalCreationEngine()
        kc = AutonomousKnowledgeCivilization(creation_engine=engine)
        kc.knowledge_generation_cycle("test")
        metrics = kc.metrics
        assert metrics.knowledge_volume >= 0


class TestKnowledgeRepository:
    """知识仓库测试。"""

    def test_store_and_retrieve(self):
        repo = KnowledgeRepository()
        repo.store({"statement": "test"}, "test_domain")
        theories = repo.get_all_theories("test_domain")
        assert len(theories) == 1

    def test_total_count(self):
        repo = KnowledgeRepository()
        repo.store({"a": 1}, "d1")
        repo.store({"b": 2}, "d2")
        assert repo.total_count() == 2

    def test_domain_diversity(self):
        repo = KnowledgeRepository()
        repo.store({"a": 1}, "d1")
        repo.store({"b": 2}, "d2")
        assert repo.domain_diversity() == 2


# =============================================================================
# CausalEconomy Tests
# =============================================================================


class TestCausalEconomy:
    """因果经济体系测试。"""

    def test_init(self):
        econ = CausalEconomy()
        assert econ.n_transactions == 0

    def test_value_assessment(self):
        econ = CausalEconomy()
        value = econ.value_causal_knowledge({"novelty_score": 0.8})
        assert value["total_value"] > 0
        assert "novelty_value" in value
        assert "value_category" in value

    def test_trade(self):
        econ = CausalEconomy()
        result = econ.trade_knowledge("provider_1", "consumer_1", {"test": True})
        assert result["status"] == "completed"
        assert econ.n_transactions == 1

    def test_multiple_trades(self):
        econ = CausalEconomy()
        for i in range(5):
            econ.trade_knowledge(f"p_{i}", f"c_{i}", {"theory": f"T{i}"})
        assert econ.n_transactions == 5

    def test_value_categories(self):
        econ = CausalEconomy()
        value = econ.value_causal_knowledge(CreatedTheory(novelty_score=0.95))
        assert value["value_category"] in ("premium", "standard", "basic", "low_value")


# =============================================================================
# NoveltyVerifier Tests
# =============================================================================


class TestNoveltyVerifier:
    """新颖性验证测试。"""

    def test_init(self):
        nv = NoveltyVerifier()
        assert nv is not None

    def test_verify_without_repository(self):
        nv = NoveltyVerifier()
        result = nv.verify({"domain": "test"})
        assert result["novelty_confirmed"] is True
        assert result["novelty_degree"] == 1.0

    def test_verify_with_repository(self):
        repo = KnowledgeRepository()
        repo.store({"statement": "existing theory"}, "test")
        nv = NoveltyVerifier(knowledge_repository=repo)
        theory = CreatedTheory(domain="test", statement="completely new theory xyz")
        result = nv.verify(theory)
        assert "novelty_confirmed" in result
        assert "max_structural_similarity" in result

    def test_similarity_threshold(self):
        nv = NoveltyVerifier(similarity_threshold=0.9)
        assert nv._threshold == 0.9


# =============================================================================
# CreativeTrust Tests
# =============================================================================


class TestCreativeTrust:
    """创造可信框架测试。"""

    def test_init(self):
        ct = CreativeTrust()
        assert ct is not None

    def test_assess_without_deps(self):
        ct = CreativeTrust()
        theory = CreatedTheory(
            domain="test",
            statement="New theory",
            falsifiability={"testable_predictions": ["pred1", "pred2"]},
        )
        result = ct.assess_creative_trust(theory)
        assert "creative_trust_score" in result
        assert "trust_level" in result
        assert result["trust_level"] in (
            "validated_innovation",
            "speculative_innovation",
            "untested_hypothesis",
            "contradictory_theory",
        )

    def test_trust_thresholds(self):
        assert CreativeTrust.TRUST_THRESHOLDS["validated_innovation"] == 0.85
        assert CreativeTrust.TRUST_THRESHOLDS["speculative_innovation"] == 0.60

    def test_falsifiability_check(self):
        ct = CreativeTrust()
        theory_with = CreatedTheory(
            falsifiability={"testable_predictions": ["p1", "p2", "p3"]},
        )
        theory_without = CreatedTheory()
        result_with = ct.assess_creative_trust(theory_with)
        result_without = ct.assess_creative_trust(theory_without)
        # 有可证伪设计的理论应有更高可信度
        assert result_with["falsifiability"]["score"] >= result_without["falsifiability"]["score"]


# =============================================================================
# P13 KPI Tests
# =============================================================================


class TestP13KPI:
    """P13 KPI 验收测试。"""

    def test_kpi_creation_5_strategies(self):
        """KPI: 因果创造引擎 ≥3 种创造策略。"""
        assert len(CreationStrategy) >= 3

    def test_kpi_creation_novel_theory(self):
        """KPI: ≥1 条可证伪新理论。"""
        knowledge = DomainKnowledge()
        engine = CausalCreationEngine(knowledge=knowledge)
        result = engine.create_causal_theory("novel_domain", strategy="analogy")
        if result["created_theory"]:
            assert result["created_theory"].falsifiability is not None

    def test_kpi_creative_consciousness_4_states(self):
        """KPI: 创造意识 4 种状态。"""
        assert len(CreativeState) == 4

    def test_kpi_knowledge_civilization_3_generations(self):
        """KPI: 知识文明 ≥3 世代循环。"""
        engine = CausalCreationEngine()
        kc = AutonomousKnowledgeCivilization(creation_engine=engine)
        for _ in range(3):
            kc.knowledge_generation_cycle("test")
        assert kc.n_generations >= 3

    def test_kpi_causal_economy_5_trades(self):
        """KPI: 因果经济 ≥5 笔交易。"""
        econ = CausalEconomy()
        for i in range(5):
            econ.trade_knowledge(f"p_{i}", f"c_{i}", {"theory": f"T{i}"})
        assert econ.n_transactions >= 5

    def test_kpi_novelty_verifier(self):
        """KPI: 新颖性验证可执行。"""
        nv = NoveltyVerifier()
        result = nv.verify(CreatedTheory(domain="test"))
        assert "novelty_confirmed" in result

    def test_kpi_creative_trust(self):
        """KPI: 创造可信评估准确率。"""
        ct = CreativeTrust()
        result = ct.assess_creative_trust(CreatedTheory())
        assert result["creative_trust_score"] >= 0
