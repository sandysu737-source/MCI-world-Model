"""L1 oracle 测试 — EnergyWeightedCausalGraph 能量加权因果图。

验证能量 Flow 矩阵与 algebra.CausalDAG 的桥接:
- 边权重来自守恒 Flow 矩阵 (非硬编码)
- 传播遵循生克几何衰减
- 与 EnergyCore 矩阵迭代语义一致
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.oracle

from mci_world_model._sys._energy_core import EnergyCore
from mci_world_model.sdk._energy_causal_graph import EnergyWeightedCausalGraph
from mci_world_model._sys._terms import ENERGY_ENHANCE, ENERGY_SUPPRESS


@pytest.fixture
def graph():
    return EnergyWeightedCausalGraph(EnergyCore())


class TestEnergyWeightedGraph:
    """能量加权因果图的结构与传播。"""

    def test_graph_has_all_five_nodes(self, graph):
        assert set(graph.categories) == {"semantic", "causal", "spacetime", "generative", "trust"}

    def test_edges_match_flow_matrix(self, graph):
        """边权重应来自 Flow 矩阵的正向非对角元素。"""
        F = EnergyCore().flow_matrix()
        cats = graph.categories
        idx = {c: i for i, c in enumerate(cats)}
        for src in cats:
            for dst in cats:
                if src == dst:
                    continue
                w_flow = F[idx[dst], idx[src]]
                w_graph = graph.dag.edge_weight(src, dst)
                if w_flow > 0.01:
                    assert abs(w_graph - round(w_flow, 6)) < 1e-9, (
                        f"{src}→{dst}: graph={w_graph} vs flow={w_flow}"
                    )

    def test_propagation_source_included(self, graph):
        """传播结果应包含源节点自身。"""
        eff = graph.propagate("semantic", 1.0)
        assert eff["semantic"] == 1.0

    def test_propagation_geometric_attenuation(self, graph):
        """传播应遵循几何衰减 (生关系 0.15 衰减)。"""
        eff = graph.propagate("semantic", 1.0)
        # semantic 生 causal: causal 应 ≈ 0.15
        assert abs(eff["causal"] - 0.15) < 0.01, (
            f"causal={eff['causal']}, 生关系应≈0.15"
        )

    def test_different_sources_different_propagation(self, graph):
        """不同干预源应产生不同传播模式。"""
        sem = graph.propagation_vector("semantic")
        cau = graph.propagation_vector("causal")
        assert not np.allclose(sem, cau), "不同源传播相同, 无可区分性"

    def test_propagation_consistent_with_matrix_power(self, graph):
        """单步传播应与 Flow 矩阵的正向部分一致。

        CausalDAG.propagate 取最短路径首达, 而 F·δ 是全路径求和。
        对直接邻居 (1跳), 两者应一致。
        """
        F = EnergyCore().flow_matrix()
        cats = graph.categories
        idx = {c: i for i, c in enumerate(cats)}
        eff = graph.propagate("semantic", 1.0)
        # 直接邻居: F[dst, semantic] > 0 的维度
        for dst in cats:
            f_val = F[idx[dst], idx["semantic"]]
            if f_val > 0.01 and dst != "semantic":
                # 直接邻居的图传播 = F 矩阵值
                assert abs(eff[dst] - f_val) < 0.011, (
                    f"{dst}: graph={eff[dst]} vs F={f_val}"
                )


class TestRelationClassification:
    """关系分类 (生/克) 正确性。"""

    def test_generation_relation(self, graph):
        """ENERGY_ENHANCE 的对应边应分类为'生'。"""
        for src, dst in ENERGY_ENHANCE.items():
            assert graph.relation(src, dst) == "生", (
                f"{src}→{dst} 应为'生', 实际 '{graph.relation(src, dst)}'"
            )

    def test_suppress_energy_seizure(self, graph):
        """克关系的能量夺取方向: 克方←被克方 应有边。"""
        # semantic 克 spacetime => 能量 spacetime→semantic (夺取)
        # 所以 semantic 的入边里有 spacetime
        w = graph.dag.edge_weight("spacetime", "semantic")
        assert w > 0, f"克夺取: spacetime→semantic 应有边, w={w}"
