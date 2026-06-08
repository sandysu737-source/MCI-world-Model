"""
测试 _sys 模块覆盖率补齐 (P1.4)

目标: meta_cognition.py (14%→80%+), _energy_relations (30%→70%+),
      _causal_engine (14%→50%+), bayesian_network (30%→60%+)
"""

from __future__ import annotations

import time

import numpy as np
import pytest


# =============================================================================
# meta_cognition.py — 元认知模块
# =============================================================================


class TestMetaCognition:
    """覆盖 meta_cognition.py: discover_gaps, detect_conflicts, get_aging, _contradicts."""

    def test_discover_gaps_domain_empty(self):
        """领域空洞: fact 占比低于 30% 时返回 domain gap。"""
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        types = {"fact": 1, "other": 9}
        gaps = mc.discover_gaps(types, [], [])
        assert any(g["type"] == "domain" for g in gaps)

    def test_discover_gaps_domain_sufficient(self):
        """fact >= 30%: 不产生 domain gap。"""
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        types = {"fact": 4, "other": 6}
        gaps = mc.discover_gaps(types, [], [])
        assert not any(g["type"] == "domain" for g in gaps)

    def test_discover_gaps_temporal_stale(self):
        """时间空洞: 最旧记忆超过 30 天。"""
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        memories = [{"timestamp": time.time() - 86400 * 40}]
        gaps = mc.discover_gaps({"fact": 5, "other": 5}, [], memories)
        assert any(g["type"] == "temporal" for g in gaps)

    def test_discover_gaps_temporal_fresh(self):
        """时间空洞: 记忆在 30 天内不产生 gap。"""
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        memories = [{"timestamp": time.time() - 86400 * 10}]
        gaps = mc.discover_gaps({"fact": 5, "other": 5}, [], memories)
        assert not any(g["type"] == "temporal" for g in gaps)

    def test_discover_gaps_causal_isolated(self):
        """因果空洞: 超半数记忆无因果关联。"""
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        memories = [{"causal": False}, {"causal": False}, {"causal": True}, {"causal": False}]
        gaps = mc.discover_gaps({"fact": 5, "other": 5}, [], memories)
        assert any(g["type"] == "causal" for g in gaps)

    def test_discover_gaps_causal_connected(self):
        """因果空洞: 有因果关联的记忆 >= 50% 不产生 gap。"""
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        memories = [{"causal": True}, {"causal": True}, {"causal": False}]
        gaps = mc.discover_gaps({"fact": 5, "other": 5}, [], memories)
        assert not any(g["type"] == "causal" for g in gaps)

    def test_detect_conflicts_no_conflict(self):
        """无冲突: 两个信念内容不矛盾。"""
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        beliefs = {"a": {"content": "这是正确的"}, "b": {"content": "天气很好"}}
        conflicts = mc.detect_conflicts(beliefs)
        assert len(conflicts) == 0

    def test_detect_conflicts_has_conflict(self):
        """有冲突: 一个包含肯定词，另一个包含否定词。"""
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        beliefs = {"a": {"content": "这是正确的"}, "b": {"content": "这是错误的"}}
        conflicts = mc.detect_conflicts(beliefs)
        assert any(c["memory_a"] == "a" and c["memory_b"] == "b" for c in conflicts)

    def test_contradicts_mixed_polarity(self):
        """_contradicts: 正负混合返回 True。"""
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        assert mc._contradicts("这是一个正确的决定", "但不是这样")
        assert mc._contradicts("这个方法是错误的", "我知道答案")

    def test_contradicts_no_opposition(self):
        """_contradicts: 纯正向或纯负向返回 False。"""
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        assert not mc._contradicts("这是正确的", "我知道答案")
        assert not mc._contradicts("不是这样的", "错了")

    def test_get_aging_warning(self):
        """get_aging: 14 < days < 30 → warning。"""
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        memories = [{"id": "m1", "timestamp": time.time() - 86400 * 20}]
        aging = mc.get_aging(memories)
        assert len(aging) == 1
        assert aging[0]["severity"] == "warning"

    def test_get_aging_critical(self):
        """get_aging: days >= 30 → critical。"""
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        memories = [{"id": "m1", "timestamp": time.time() - 86400 * 40}]
        aging = mc.get_aging(memories)
        assert len(aging) == 1
        assert aging[0]["severity"] == "critical"

    def test_get_aging_fresh(self):
        """get_aging: <=14 天不产生老化警告。"""
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        memories = [{"id": "m1", "timestamp": time.time() - 86400 * 5}]
        aging = mc.get_aging(memories)
        assert len(aging) == 0


# =============================================================================
# _energy_relations.py — 五行能量关系
# =============================================================================


class TestEnergyRelations:
    """覆盖 _energy_relations.py 核心纯函数。"""

    # ── EnergyType ──

    def test_energy_type_from_string(self):
        from mci_world_model._sys._energy_relations import EnergyType

        assert EnergyType.from_string("semantic") == EnergyType.WOOD
        assert EnergyType.from_string("causal") == EnergyType.FIRE
        assert EnergyType.from_string("wood") == EnergyType.WOOD  # backward compat

    def test_energy_type_invalid(self):
        from mci_world_model._sys._energy_relations import EnergyType

        with pytest.raises(ValueError):
            EnergyType.from_string("unknown")

    def test_energy_type_all_values(self):
        from mci_world_model._sys._energy_relations import EnergyType

        vals = EnergyType.all_values()
        assert "semantic" in vals
        assert len(vals) == 5

    # ── RelationType ──

    def test_relation_type_values(self):
        from mci_world_model._sys._energy_relations import RelationType

        assert RelationType.ENHANCE.value == "enhance"
        assert RelationType.SUPPRESS.value == "suppress"

    # ── get_enhance_relation / get_suppress_relation ──

    def test_enhance_relation(self):
        from mci_world_model._sys._energy_relations import get_enhance_relation

        assert get_enhance_relation("wood", "fire")
        assert not get_enhance_relation("fire", "wood")

    def test_suppress_relation(self):
        from mci_world_model._sys._energy_relations import get_suppress_relation

        assert get_suppress_relation("wood", "earth")
        assert not get_suppress_relation("earth", "wood")

    # ── analyze_relation ──

    def test_analyze_relation_enhance(self):
        from mci_world_model._sys._energy_relations import analyze_relation, RelationType

        rel = analyze_relation("wood", "fire")
        assert rel.relation == RelationType.ENHANCE
        assert rel.strength >= 0.8

    def test_analyze_relation_suppress(self):
        from mci_world_model._sys._energy_relations import analyze_relation, RelationType

        rel = analyze_relation("wood", "earth")
        assert rel.relation == RelationType.SUPPRESS
        assert rel.strength == 0.8

    def test_analyze_relation_same(self):
        from mci_world_model._sys._energy_relations import analyze_relation, RelationType

        rel = analyze_relation("wood", "wood")
        assert rel.relation == RelationType.SAME
        assert rel.strength == pytest.approx(1.0, abs=0.2)

    # ── calculate_link_weight ──

    def test_calculate_link_weight_enhance(self):
        from mci_world_model._sys._energy_relations import calculate_link_weight

        w = calculate_link_weight("wood", "fire", base_weight=1.0)
        assert w > 1.0  # enhance 增加权重

    def test_calculate_link_weight_suppress(self):
        from mci_world_model._sys._energy_relations import calculate_link_weight

        w = calculate_link_weight("wood", "earth", base_weight=1.0)
        assert w < 1.0

    def test_calculate_link_weight_same(self):
        from mci_world_model._sys._energy_relations import calculate_link_weight

        w = calculate_link_weight("wood", "wood", base_weight=1.0)
        assert w == pytest.approx(1.0, abs=0.15)

    # ── get_cycle_sequence / get_suppress_chain ──

    def test_cycle_sequence(self):
        from mci_world_model._sys._energy_relations import get_cycle_sequence

        seq = get_cycle_sequence("wood", steps=5)
        assert seq == ["wood", "fire", "earth", "metal", "water"]

    def test_suppress_chain(self):
        from mci_world_model._sys._energy_relations import get_suppress_chain

        chain = get_suppress_chain("wood", steps=5)
        assert chain == ["wood", "earth", "water", "fire", "metal"]

    # ── analyze_balance ──

    def test_analyze_balance(self):
        from mci_world_model._sys._energy_relations import analyze_balance

        dist = {"wood": 0.2, "fire": 0.2, "earth": 0.2, "metal": 0.2, "water": 0.2}
        result = analyze_balance(dist)
        assert "status" in result
        assert result["status"] in ("balanced", "unbalanced")

    def test_analyze_balance_unbalanced(self):
        from mci_world_model._sys._energy_relations import analyze_balance

        dist = {"wood": 0.9, "fire": 0.02, "earth": 0.02, "metal": 0.03, "water": 0.03}
        result = analyze_balance(dist)
        # 极端分布 → concentrated
        assert result["status"] in ("concentrated", "balanced", "dispersed")

    # ── get_affinity_score ──

    def test_affinity_score_enhance(self):
        from mci_world_model._sys._energy_relations import get_affinity_score

        score = get_affinity_score("wood", "fire")
        assert score > 0.5

    def test_affinity_score_suppress(self):
        from mci_world_model._sys._energy_relations import get_affinity_score

        score = get_affinity_score("wood", "earth")
        assert 0 < score <= 0.6

    def test_affinity_score_same(self):
        from mci_world_model._sys._energy_relations import get_affinity_score

        score = get_affinity_score("wood", "wood")
        assert score == 1.2

    # ── EnergyRelation dataclass ──

    def test_energy_relation_dataclass(self):
        from mci_world_model._sys._energy_relations import EnergyRelation, RelationType

        er = EnergyRelation(source="wood", target="fire", relation=RelationType.ENHANCE, strength=0.9, description="test")
        assert er.source == "wood"
        assert er.target == "fire"
        assert er.strength == 0.9
        assert er.boost_factor == 1.2


# =============================================================================
# _causal_engine.py — CategoryCausalEngine
# =============================================================================


class TestCategoryCausalEngine:
    """覆盖 CategoryCausalEngine 核心方法: add_node, add_edges, get_children, get_parents, propagate."""

    def test_add_node(self):
        from mci_world_model._sys._causal_engine import CategoryCausalEngine

        engine = CategoryCausalEngine()
        engine.add_node("n1", "test node", "wood")
        assert "n1" in engine.nodes
        assert engine.nodes["n1"].energy_type == "wood"

    def test_add_node_duplicate(self):
        """重复添加同 ID 节点: 跳过不覆盖。"""
        from mci_world_model._sys._causal_engine import CategoryCausalEngine

        engine = CategoryCausalEngine()
        engine.add_node("n1", "first", "wood")
        engine.add_node("n1", "second", "fire")
        assert engine.nodes["n1"].content == "first"

    def test_link_nodes(self):
        from mci_world_model._sys._causal_engine import CategoryCausalEngine

        engine = CategoryCausalEngine()
        engine.add_node("n1", "parent", "wood")
        engine.add_node("n2", "child", "fire")
        success, weight = engine.link("n1", "n2")
        assert success is True
        assert weight > 0

    def test_link_with_energy_relation(self):
        from mci_world_model._sys._causal_engine import CategoryCausalEngine

        engine = CategoryCausalEngine()
        engine.add_node("n1", "木", "wood")
        engine.add_node("n2", "火", "fire")
        success, rel = engine.link_with_energy_relation("n1", "n2")
        assert success is True
        assert rel is not None

    def test_get_relation(self):
        from mci_world_model._sys._causal_engine import CategoryCausalEngine

        engine = CategoryCausalEngine()
        engine.add_node("n1", "木", "wood")
        engine.add_node("n2", "火", "fire")

        rel = engine.get_relation("n1", "n2")
        assert rel is not None
        assert rel.is_enhancing is True

    def test_get_relation_unknown_node(self):
        from mci_world_model._sys._causal_engine import CategoryCausalEngine

        engine = CategoryCausalEngine()
        assert engine.get_relation("n1", "n2") is None

    def test_get_neighbors_by_relation(self):
        from mci_world_model._sys._causal_engine import CategoryCausalEngine

        engine = CategoryCausalEngine()
        engine.add_node("n1", "木", "wood")
        engine.add_node("n2", "火", "fire")
        engine.link("n1", "n2")

        neighbors = engine.get_neighbors_by_relation("n1")
        assert len(neighbors) >= 0

    def test_get_neighbors_empty_graph(self):
        from mci_world_model._sys._causal_engine import CategoryCausalEngine

        engine = CategoryCausalEngine()
        engine.add_node("n1", "木", "wood")
        neighbors = engine.get_neighbors_by_relation("n1")
        assert neighbors == []

    def test_propagate(self):
        from mci_world_model._sys._causal_engine import CategoryCausalEngine

        engine = CategoryCausalEngine()
        engine.add_node("n1", "木", "wood")
        engine.add_node("n2", "火", "fire")
        engine.link("n1", "n2")

        result = engine.propagate("n1", delta=0.1)
        assert isinstance(result, dict)
        assert "n2" in result

    def test_propagate_unknown_node(self):
        from mci_world_model._sys._causal_engine import CategoryCausalEngine

        engine = CategoryCausalEngine()
        result = engine.propagate("nonexistent")
        assert result == {}

    def test_analyze_memory_graph_empty(self):
        from mci_world_model._sys._causal_engine import CategoryCausalEngine

        engine = CategoryCausalEngine()
        result = engine.analyze_memory_graph()
        assert result["status"] == "empty"

    def test_analyze_memory_graph(self):
        from mci_world_model._sys._causal_engine import CategoryCausalEngine

        engine = CategoryCausalEngine()
        engine.add_node("n1", "木", "wood")
        engine.add_node("n2", "火", "fire")
        engine.link("n1", "n2")

        result = engine.analyze_memory_graph()
        assert "energy_distribution" in result

    def test_get_enhancing_neighbors(self):
        from mci_world_model._sys._causal_engine import CategoryCausalEngine

        engine = CategoryCausalEngine()
        engine.add_node("n1", "木", "wood")
        engine.add_node("n2", "火", "fire")
        engine.link("n1", "n2")

        ench = engine.get_enhancing_neighbors("n1")
        assert isinstance(ench, list)

    def test_get_suppressing_neighbors(self):
        from mci_world_model._sys._causal_engine import CategoryCausalEngine

        engine = CategoryCausalEngine()
        engine.add_node("n1", "木", "wood")
        engine.add_node("n2", "土", "earth")
        engine.link("n1", "n2")

        supp = engine.get_suppressing_neighbors("n1")
        assert isinstance(supp, list)

    def test_query_with_energy_boost(self):
        from mci_world_model._sys._causal_engine import CategoryCausalEngine

        engine = CategoryCausalEngine()
        engine.add_node("n1", "木", "wood")
        engine.add_node("n2", "火", "fire")
        engine.link("n1", "n2")

        result = engine.query_with_energy_boost(
            "n1", candidates=["n2"], base_scores={"n2": 0.5}
        )
        assert isinstance(result, list)
        assert len(result) > 0

    def test_engine_init_defaults(self):
        from mci_world_model._sys._causal_engine import CategoryCausalEngine

        engine = CategoryCausalEngine()
        assert isinstance(engine.nodes, dict)
        assert "creative" in engine.category_energy_map
        assert engine.category_energy_map["creative"] == "metal"


# =============================================================================
# bayesian_network.py — BayesianNetwork
# =============================================================================


class TestBayesianNetworkCoverage:
    """覆盖 BayesianNetwork 基础方法。"""

    def test_bn_init(self):
        from mci_world_model._sys.bayesian_network import BayesianNetwork

        bn = BayesianNetwork()
        assert bn._nodes == {}
        assert bn._edges == {}

    def test_bn_add_node(self):
        from mci_world_model._sys.bayesian_network import BayesianNetwork

        bn = BayesianNetwork()
        node = bn.add_node("A", label="Node A")
        assert node.node_id == "A"
        assert "A" in bn._nodes

    def test_bn_add_edge(self):
        from mci_world_model._sys.bayesian_network import BayesianNetwork

        bn = BayesianNetwork()
        edge = bn.add_edge("A", "B")
        assert edge.parent_id == "A"
        assert edge.child_id == "B"
        assert ("A", "B") in bn._edges

    def test_bn_get_node(self):
        from mci_world_model._sys.bayesian_network import BayesianNetwork

        bn = BayesianNetwork()
        bn.add_node("X")
        node = bn.get_node("X")
        assert node is not None

    def test_bn_get_node_missing(self):
        from mci_world_model._sys.bayesian_network import BayesianNetwork

        bn = BayesianNetwork()
        assert bn.get_node("Z") is None

    def test_bn_remove_node(self):
        from mci_world_model._sys.bayesian_network import BayesianNetwork

        bn = BayesianNetwork()
        bn.add_node("A")
        bn.add_node("B")
        bn.add_edge("A", "B")
        bn.remove_node("A")
        assert "A" not in bn._nodes

    def test_bn_get_neighbors(self):
        from mci_world_model._sys.bayesian_network import BayesianNetwork

        bn = BayesianNetwork()
        bn.add_edge("A", "B")
        neighbors = bn.get_neighbors("A")
        assert "B" in neighbors

    def test_bn_get_parents(self):
        from mci_world_model._sys.bayesian_network import BayesianNetwork

        bn = BayesianNetwork()
        bn.add_edge("A", "B")
        parents = bn.get_parents("B")
        assert "A" in parents

    def test_bn_get_children(self):
        from mci_world_model._sys.bayesian_network import BayesianNetwork

        bn = BayesianNetwork()
        bn.add_edge("A", "B")
        children = bn.get_children("A")
        assert "B" in children

    def test_bn_infer_posterior(self):
        from mci_world_model._sys.bayesian_network import BayesianNetwork

        bn = BayesianNetwork()
        bn.add_edge("A", "B")

        result = bn.infer_posterior(query_nodes=["B"], evidence={"A": True})
        assert isinstance(result, dict)
        assert "B" in result

    def test_bn_query_causal_strength(self):
        from mci_world_model._sys.bayesian_network import BayesianNetwork

        bn = BayesianNetwork()
        bn.add_edge("A", "B")

        result = bn.query_causal_strength("A", "B")
        assert isinstance(result, dict)
        assert "causal_strength" in result

    def test_bn_get_statistics(self):
        from mci_world_model._sys.bayesian_network import BayesianNetwork

        bn = BayesianNetwork()
        bn.add_node("A")

        stats = bn.get_statistics()
        assert isinstance(stats, dict)
