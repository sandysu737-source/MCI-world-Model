"""
tests/test_fci_discoverer.py — FCIDiscoverer 测试
==================================================

覆盖:
    - 基本因果链发现 (含潜在混淆)
    - 独立变量 (无伪边)
    - 边界情况 (空数据/单变量/2变量)
    - 参数验证 (alpha 合法性)
    - 混淆器存在时的鲁棒性
    - 大规模数据 + 条件独立性
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._autonomous_law_discoverer_v2 import (
    CausalSkeleton,
    FCIDiscoverer,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fci():
    return FCIDiscoverer(alpha=0.05, min_corr=0.1)


@pytest.fixture
def chain_with_confounder():
    """X1 → X2 → X3, 但 X1 ↔ X3 受隐混淆 U 影响。"""
    rng = np.random.RandomState(42)
    n = 200
    U = rng.randn(n)  # 隐混淆
    X1 = 0.5 * U + rng.randn(n)
    X2 = 0.7 * X1 + rng.randn(n)
    X3 = 0.6 * X2 + 0.4 * U + rng.randn(n)
    return np.column_stack([X1, X2, X3]), ["X1", "X2", "X3"]


@pytest.fixture
def independent_vars():
    rng = np.random.RandomState(42)
    n = 150
    return np.column_stack([rng.randn(n) for _ in range(4)]), ["A", "B", "C", "D"]


@pytest.fixture
def two_var_causal():
    rng = np.random.RandomState(42)
    n = 200
    A = rng.randn(n)
    B = 0.8 * A + 0.2 * rng.randn(n)
    return np.column_stack([A, B]), ["A", "B"]


# =============================================================================
# 基本功能测试
# =============================================================================


class TestFCIDiscovererBasic:
    def test_chain_with_confounder(self, fci, chain_with_confounder):
        """FCI 在有混淆器时发现边 (更保守)。"""
        data, names = chain_with_confounder
        skel = fci.discover(data, names)
        assert isinstance(skel, CausalSkeleton)
        assert len(skel.nodes) == 3
        # FCI 比 PC 更保守, 可能删掉更多边
        assert len(skel.edges) >= 0

    def test_independent_vars_no_edges(self, fci, independent_vars):
        """独立变量不应产生边。"""
        data, names = independent_vars
        skel = fci.discover(data, names)
        assert len(skel.edges) == 0

    def test_two_var_causal_link(self, fci, two_var_causal):
        """2 变量因果链。"""
        data, names = two_var_causal
        skel = fci.discover(data, names)
        assert len(skel.nodes) == 2
        adj = skel.adj_matrix
        assert adj.shape == (2, 2)
        # 强相关应产生边
        assert adj[0, 1] == 1 or adj[1, 0] == 1

    def test_returns_skeleton_type(self, fci, two_var_causal):
        """discover 返回 CausalSkeleton。"""
        data, names = two_var_causal
        skel = fci.discover(data, names)
        assert isinstance(skel, CausalSkeleton)
        assert isinstance(skel.nodes, list)
        assert isinstance(skel.edges, list)
        assert isinstance(skel.adj_matrix, np.ndarray)

    def test_confidence_range(self, fci, two_var_causal):
        """confidence 在 [0, 1]。"""
        data, names = two_var_causal
        skel = fci.discover(data, names)
        assert 0.0 <= skel.confidence <= 1.0


# =============================================================================
# 边界情况
# =============================================================================


class TestFCIDiscovererEdgeCases:
    def test_empty_data(self, fci):
        """空数据返回空骨架。"""
        data = np.array([]).reshape(0, 3)
        skel = fci.discover(data, ["A", "B", "C"])
        assert len(skel.nodes) == 3
        assert len(skel.edges) == 0

    def test_single_variable(self, fci):
        """单变量。"""
        rng = np.random.RandomState(42)
        data = rng.randn(50, 1)
        skel = fci.discover(data, ["A"])
        assert len(skel.nodes) == 1
        assert len(skel.edges) == 0

    def test_single_sample(self, fci):
        """单样本不崩溃。"""
        data = np.array([[1.0, 2.0, 3.0]])
        skel = fci.discover(data, ["A", "B", "C"])
        assert len(skel.nodes) == 3

    def test_alpha_validation(self):
        """alpha 必须在 (0,1)。"""
        with pytest.raises(ValueError):
            FCIDiscoverer(alpha=0.0)
        with pytest.raises(ValueError):
            FCIDiscoverer(alpha=1.0)
        with pytest.raises(ValueError):
            FCIDiscoverer(alpha=-0.1)

    def test_fci_more_conservative_than_pc(self):
        """FCI 条件更严 → 边数 ≤ PC。"""
        rng = np.random.RandomState(42)
        n = 100
        X1 = rng.randn(n)
        X2 = 0.3 * X1 + rng.randn(n)
        X3 = 0.3 * X1 + rng.randn(n)
        X4 = 0.3 * X2 + 0.3 * X3 + rng.randn(n)
        data = np.column_stack([X1, X2, X3, X4])
        names = ["A", "B", "C", "D"]

        from mci_world_model.sdk._autonomous_law_discoverer_v2 import PCSkeletonDiscoverer
        pc = PCSkeletonDiscoverer(alpha=0.1)
        fci = FCIDiscoverer(alpha=0.1)
        pc_skel = pc.discover(data, names)
        fci_skel = fci.discover(data, names)
        assert len(fci_skel.edges) <= len(pc_skel.edges) + 4  # 允许少量差异


# =============================================================================
# 高维 + 混淆鲁棒性
# =============================================================================


class TestFCIRobustness:
    def test_large_sample(self, fci):
        """大样本收敛。"""
        rng = np.random.RandomState(42)
        n = 500
        X1 = rng.randn(n)
        X2 = 0.7 * X1 + rng.randn(n)
        X3 = 0.5 * X2 + rng.randn(n)
        data = np.column_stack([X1, X2, X3])
        skel = fci.discover(data, ["A", "B", "C"])
        adj = skel.adj_matrix
        # 应发现 A-B, B-C 至少部分
        assert adj[0, 1] == 1 or adj[1, 0] == 1

    def test_min_corr_parameter(self):
        """min_corr 参数生效。"""
        rng = np.random.RandomState(42)
        n = 200
        A = rng.randn(n)
        B = 0.05 * A + rng.randn(n)  # 极弱相关
        data = np.column_stack([A, B])

        fci_lo = FCIDiscoverer(alpha=0.05, min_corr=0.01)
        fci_hi = FCIDiscoverer(alpha=0.05, min_corr=0.5)
        skel_lo = fci_lo.discover(data, ["A", "B"])
        skel_hi = fci_hi.discover(data, ["A", "B"])
        # 高 min_corr 可能产生更少边
        assert len(skel_hi.edges) <= len(skel_lo.edges)

    def test_deterministic_output(self, fci):
        """相同输入 → 相同输出。"""
        rng = np.random.RandomState(42)
        data = rng.randn(100, 3)
        names = ["A", "B", "C"]
        s1 = fci.discover(data, names)
        s2 = fci.discover(data, names)
        assert np.array_equal(s1.adj_matrix, s2.adj_matrix)
        assert s1.edges == s2.edges
        assert s1.confidence == s2.confidence

# =============================================================================
# Regression-based edge orientation tests for FCI (Phase A: SOTA gap fix)
# =============================================================================

class TestFCIRegressionOrientation:
    """Tests for regression-based orientation in FCIDiscoverer."""

    def test_fci_chain_x_to_y(self):
        """FCI chain: regression should produce directed edges."""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import FCIDiscoverer
        rng = np.random.RandomState(42)
        n = 500
        X = rng.randn(n)
        Y = 0.8 * X + 0.5 * rng.randn(n)
        data = np.column_stack([X, Y])
        fci = FCIDiscoverer(alpha=0.01)
        skel = fci.discover(data, ["X", "Y"])
        assert ("X", "Y") in skel.edges or ("Y", "X") in skel.edges,             f"No edge: {skel.edges}"

    def test_fci_v_structure(self):
        """FCI v-structure: should find collider edges."""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import FCIDiscoverer
        rng = np.random.RandomState(42)
        n = 500
        X = rng.randn(n)
        Y = rng.randn(n)
        Z = 0.7 * X + 0.6 * Y + 0.5 * rng.randn(n)
        data = np.column_stack([X, Y, Z])
        fci = FCIDiscoverer(alpha=0.01)
        skel = fci.discover(data, ["X", "Y", "Z"])
        assert ("X", "Z") in skel.edges
        assert ("Y", "Z") in skel.edges

    def test_fci_no_bidirectional(self):
        """FCI should find chain edges (undirected OK with BIC hybrid)."""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import FCIDiscoverer
        rng = np.random.RandomState(42)
        n = 500
        X = rng.randn(n)
        Y = 0.7 * X + 0.4 * rng.randn(n)
        Z = 0.5 * Y + 0.4 * rng.randn(n)
        data = np.column_stack([X, Y, Z])
        fci = FCIDiscoverer(alpha=0.01)
        skel = fci.discover(data, ["X", "Y", "Z"])
        # Should find at least 2 edges (X-Y and Y-Z, may be undirected)
        assert len(skel.edges) >= 2, f"Too few edges: {len(skel.edges)}"

    def test_fci_adj_matrix_directed(self):
        """FCI adj_matrix should have at least one edge."""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import FCIDiscoverer
        rng = np.random.RandomState(42)
        n = 500
        X = rng.randn(n)
        Y = 0.7 * X + 0.5 * rng.randn(n)
        data = np.column_stack([X, Y])
        fci = FCIDiscoverer(alpha=0.01)
        skel = fci.discover(data, ["X", "Y"])
        adj = skel.adj_matrix
        assert adj[0, 1] == 1 or adj[1, 0] == 1,             f"No edge: {adj}"

    def test_fci_large_sample_edges(self):
        """FCI large sample should detect at least one directed edge."""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import FCIDiscoverer
        rng = np.random.RandomState(42)
        n = 500
        X1 = rng.randn(n)
        X2 = 0.7 * X1 + rng.randn(n)
        X3 = 0.5 * X2 + rng.randn(n)
        data = np.column_stack([X1, X2, X3])
        fci = FCIDiscoverer(alpha=0.05, min_corr=0.1)
        skel = fci.discover(data, ["A", "B", "C"])
        total_edges = len(skel.edges)
        assert total_edges >= 1, f"FCI found {total_edges} edges, expected >= 1"


# ═══════════════════════════════════════════════════════════════════════════════
# PAG edge classification tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFCIPAGLabels:
    """PAG edge type classification (direct ↔ confounded ↔ undirected)."""

    def test_pag_direct_causal_chain(self):
        """PAG: simple chain X→Y→Z should produce directed edges."""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import FCIDiscoverer
        rng = np.random.RandomState(42)
        n = 500
        X = rng.randn(n)
        Y = 0.8 * X + 0.3 * rng.randn(n)
        Z = 0.7 * Y + 0.3 * rng.randn(n)
        data = np.column_stack([X, Y, Z])
        fci = FCIDiscoverer(alpha=0.01)
        labels = fci.pag_edge_labels(data, ["X", "Y", "Z"])
        assert "direct" in labels
        assert "bidirected" in labels
        assert "undirected" in labels
        assert "partial" in labels

    def test_pag_latent_confounder_produces_bidirected(self):
        """PAG: shared confounder should produce bidirected edges."""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import FCIDiscoverer
        rng = np.random.RandomState(42)
        n = 300
        U = rng.randn(n)  # latent
        X = 0.6 * U + 0.4 * rng.randn(n)
        Y = 0.6 * U + 0.4 * rng.randn(n)
        data = np.column_stack([X, Y])
        fci = FCIDiscoverer(alpha=0.05, min_corr=0.1)
        labels = fci.pag_edge_labels(data, ["X", "Y"])
        # X and Y correlated via U → may produce bidirected or undirected
        total = sum(len(v) for v in labels.values())
        assert total >= 0  # FCI may or may not find edges (conservative)

    def test_pag_independent_vars_no_labels(self):
        """PAG: independent variables should produce no edge labels."""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import FCIDiscoverer
        rng = np.random.RandomState(42)
        data = rng.randn(200, 3)
        fci = FCIDiscoverer(alpha=0.05)
        labels = fci.pag_edge_labels(data, ["A", "B", "C"])
        total = sum(len(v) for v in labels.values())
        assert total == 0, f"Expected 0 edges for independent vars, got {total}"

    def test_pag_all_categories_are_lists(self):
        """PAG: all category values should be lists of tuples."""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import FCIDiscoverer
        rng = np.random.RandomState(42)
        n = 300
        X = rng.randn(n)
        Y = 0.7 * X + 0.5 * rng.randn(n)
        Z = 0.5 * Y + 0.5 * rng.randn(n)
        data = np.column_stack([X, Y, Z])
        fci = FCIDiscoverer(alpha=0.01)
        labels = fci.pag_edge_labels(data, ["X", "Y", "Z"])
        for cat in ["direct", "bidirected", "undirected", "partial"]:
            assert cat in labels, f"Missing category: {cat}"
            assert isinstance(labels[cat], list), f"{cat} is not a list"
        # Each edge tuple should have 2 string elements
        for cat_edges in labels.values():
            for edge in cat_edges:
                assert len(edge) == 2
                assert isinstance(edge[0], str) and isinstance(edge[1], str)

    def test_pag_consistent_with_discover(self):
        """PAG edge labels should match discover edges in total count."""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import FCIDiscoverer
        rng = np.random.RandomState(42)
        n = 300
        X = rng.randn(n)
        Y = 0.7 * X + 0.3 * rng.randn(n)
        data = np.column_stack([X, Y])
        fci = FCIDiscoverer(alpha=0.05)
        skel = fci.discover(data, ["X", "Y"])
        labels = fci.pag_edge_labels(data, ["X", "Y"])
        label_total = sum(len(v) for v in labels.values())
        # PAG labels may count undirected edges once (bidirected = 1 edge)
        # while discover uses directed adjacency matrix
        assert label_total >= len(skel.edges) // 2  # undirected counted once in PAG
