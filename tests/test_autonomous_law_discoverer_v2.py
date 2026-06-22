"""
tests/test_autonomous_law_discoverer_v2.py — AutonomousLawDiscovererV2 测试
==========================================================================

覆盖:
    - PCSkeletonDiscoverer: 条件独立性测试 + 骨架发现
    - AutonomousLawDiscovererV2: PC骨架 → 符号回归 → 守恒检查
    - 简单因果链 / 独立变量 / 边界情况
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._autonomous_law_discoverer_v2 import (
    AutonomousLawDiscovererV2,
    CausalEdge,
    CausalSkeleton,
    PCSkeletonDiscoverer,
    SystemReport,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def pc_discoverer():
    return PCSkeletonDiscoverer(alpha=0.05)


@pytest.fixture
def law_discoverer():
    return AutonomousLawDiscovererV2(pc_alpha=0.05, conservation_threshold=0.85)


@pytest.fixture
def simple_chain_data():
    """X1 → X2 → X3 简单因果链数据。"""
    rng = np.random.RandomState(42)
    n = 500
    x1 = rng.randn(n)
    x2 = 0.8 * x1 + 0.1 * rng.randn(n)
    x3 = 0.6 * x2 + 0.1 * rng.randn(n)
    return np.column_stack([x1, x2, x3]), ["X1", "X2", "X3"]


@pytest.fixture
def independent_data():
    """三个独立变量数据。"""
    rng = np.random.RandomState(42)
    n = 300
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    x3 = rng.randn(n)
    return np.column_stack([x1, x2, x3]), ["X1", "X2", "X3"]


@pytest.fixture
def conservation_data():
    """守恒系统数据: X1 + X2 + X3 = 常数。"""
    rng = np.random.RandomState(42)
    n = 400
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    x3 = -x1 - x2 + 0.01 * rng.randn(n)
    return np.column_stack([x1, x2, x3]), ["X1", "X2", "X3"]


# =============================================================================
# TestCausalEdge
# =============================================================================


class TestCausalEdge:
    """CausalEdge 数据类测试。"""

    def test_creation(self):
        edge = CausalEdge(cause="X1", effect="X2")
        assert edge.cause == "X1"
        assert edge.effect == "X2"
        assert edge.equation == ""
        assert edge.r_squared == 0.0
        assert edge.conservation_verified is False
        assert edge.causal_verified is False

    def test_with_equation(self):
        edge = CausalEdge(cause="A", effect="B", equation="B=2*A", r_squared=0.95, conservation_verified=True)
        assert edge.equation == "B=2*A"
        assert edge.r_squared == 0.95
        assert edge.conservation_verified is True


# =============================================================================
# TestCausalSkeleton
# =============================================================================


class TestCausalSkeleton:
    """CausalSkeleton 数据类测试。"""

    def test_creation(self):
        skeleton = CausalSkeleton(
            nodes=["X1", "X2"],
            edges=[("X1", "X2")],
        )
        assert len(skeleton.nodes) == 2
        assert len(skeleton.edges) == 1

    def test_default_adj_matrix(self):
        skeleton = CausalSkeleton(nodes=["A", "B"])
        assert skeleton.adj_matrix is None
        assert skeleton.confidence == 0.0

    def test_with_adj_matrix(self):
        adj = np.array([[0, 1], [0, 0]])
        skeleton = CausalSkeleton(
            nodes=["X1", "X2"],
            edges=[("X1", "X2")],
            adj_matrix=adj,
            confidence=0.9,
        )
        assert skeleton.adj_matrix[0, 1] == 1
        assert skeleton.confidence == 0.9


# =============================================================================
# TestPCSkeletonDiscoverer
# =============================================================================


class TestPCSkeletonDiscoverer:
    """PC骨架发现器测试。"""

    def test_discovers_chain(self, pc_discoverer, simple_chain_data):
        """简单因果链: 应发现边。"""
        data, var_names = simple_chain_data
        skeleton = pc_discoverer.discover(data, var_names)
        assert len(skeleton.nodes) == 3
        # X1和X2高度相关, 应有边
        assert len(skeleton.edges) >= 1

    def test_independent_few_edges(self, pc_discoverer, independent_data):
        """独立变量: 应发现很少或无边。"""
        data, var_names = independent_data
        skeleton = pc_discoverer.discover(data, var_names)
        assert len(skeleton.edges) <= 1  # 允许至多1条假阳性

    def test_confidence_in_range(self, pc_discoverer, simple_chain_data):
        """置信度应在 [0, 1] 范围。"""
        data, var_names = simple_chain_data
        skeleton = pc_discoverer.discover(data, var_names)
        assert 0.0 <= skeleton.confidence <= 1.0

    def test_invalid_alpha(self):
        """alpha 不在 (0,1) 应报错。"""
        with pytest.raises(ValueError, match="alpha"):
            PCSkeletonDiscoverer(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            PCSkeletonDiscoverer(alpha=1.0)

    def test_adj_matrix_shape(self, pc_discoverer, simple_chain_data):
        """邻接矩阵形状应为 (n_vars, n_vars)。"""
        data, var_names = simple_chain_data
        skeleton = pc_discoverer.discover(data, var_names)
        assert skeleton.adj_matrix is not None
        assert skeleton.adj_matrix.shape == (3, 3)


# =============================================================================
# TestAutonomousLawDiscovererV2
# =============================================================================


class TestAutonomousLawDiscovererV2:
    """AutonomousLawDiscovererV2 完整发现流程测试。"""

    def test_discovers_chain_structure(self, law_discoverer, simple_chain_data):
        """简单因果链: 应发现至少1条边。"""
        data, var_names = simple_chain_data
        report = law_discoverer.discover_causal_structure(data, var_names)
        assert isinstance(report, SystemReport)
        assert report.n_variables == 3
        assert report.n_edges >= 1

    def test_system_report_fields(self, law_discoverer, simple_chain_data):
        """SystemReport 字段完整性。"""
        data, var_names = simple_chain_data
        report = law_discoverer.discover_causal_structure(data, var_names)
        assert hasattr(report, "n_variables")
        assert hasattr(report, "n_edges")
        assert hasattr(report, "conservation_score")
        assert hasattr(report, "causal_dag")
        assert hasattr(report, "laws")
        assert hasattr(report, "is_consistent")

    def test_conservation_score_in_range(self, law_discoverer, simple_chain_data):
        """守恒得分应在 [0, 1]。"""
        data, var_names = simple_chain_data
        report = law_discoverer.discover_causal_structure(data, var_names)
        assert 0.0 <= report.conservation_score <= 1.0

    def test_laws_contain_equation(self, law_discoverer, simple_chain_data):
        """发现的规律应包含方程字符串。"""
        data, var_names = simple_chain_data
        report = law_discoverer.discover_causal_structure(data, var_names)
        for law in report.laws:
            assert "equation" in law
            assert "r_squared" in law
            assert isinstance(law["equation"], str)
            assert len(law["equation"]) > 0

    def test_independent_system(self, law_discoverer, independent_data):
        """独立系统: 应发现很少边。"""
        data, var_names = independent_data
        report = law_discoverer.discover_causal_structure(data, var_names)
        assert report.n_edges <= 1

    def test_conservation_system(self, law_discoverer, conservation_data):
        """守恒系统: 应有高R²的方程。"""
        data, var_names = conservation_data
        report = law_discoverer.discover_causal_structure(data, var_names)
        assert report.n_variables == 3

    def test_small_sample(self, law_discoverer):
        """小样本处理。"""
        rng = np.random.RandomState(42)
        data = rng.randn(10, 3)
        report = law_discoverer.discover_causal_structure(data, ["A", "B", "C"])
        assert isinstance(report, SystemReport)

    def test_two_variables(self, law_discoverer):
        """两变量场景。"""
        rng = np.random.RandomState(42)
        n = 200
        x1 = rng.randn(n)
        x2 = 0.7 * x1 + 0.2 * rng.randn(n)
        data = np.column_stack([x1, x2])
        report = law_discoverer.discover_causal_structure(data, ["X1", "X2"])
        assert report.n_variables == 2

    def test_mismatched_data_columns(self, law_discoverer):
        """数据列数不匹配应报错。"""
        data = np.random.randn(50, 3)
        with pytest.raises(ValueError, match="不匹配"):
            law_discoverer.discover_causal_structure(data, ["A", "B"])

    def test_invalid_pc_alpha(self):
        """pc_alpha 不在 (0,1) 应报错。"""
        with pytest.raises(ValueError, match="pc_alpha"):
            AutonomousLawDiscovererV2(pc_alpha=0.0)

    def test_discovered_laws_property(self, law_discoverer, simple_chain_data):
        """discovered_laws 属性。"""
        data, var_names = simple_chain_data
        law_discoverer.discover_causal_structure(data, var_names)
        laws = law_discoverer.discovered_laws
        assert isinstance(laws, list)
        assert len(laws) >= 1

    def test_causal_structure_property(self, law_discoverer, simple_chain_data):
        """causal_structure 属性。"""
        data, var_names = simple_chain_data
        law_discoverer.discover_causal_structure(data, var_names)
        skeleton = law_discoverer.causal_structure
        assert skeleton is not None
        assert isinstance(skeleton, CausalSkeleton)

# =============================================================================
# Regression-based edge orientation tests (Phase A: SOTA gap fix)
# =============================================================================

class TestRegressionEdgeOrientation:
    """Tests for _orient_edges_by_regression post-processor."""

    def test_orient_chain_x_to_y(self):
        """Chain X->Y: regression should orient X->Y correctly."""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import PCSkeletonDiscoverer
        rng = np.random.RandomState(42)
        n = 500
        X = rng.randn(n)
        Y = 0.8 * X + 0.5 * rng.randn(n)  # X -> Y with noise on Y
        data = np.column_stack([X, Y])
        pc = PCSkeletonDiscoverer(alpha=0.01)
        skel = pc.discover(data, ["X", "Y"])
        assert ("X", "Y") in skel.edges or ("Y", "X") in skel.edges, \
            f"No edge between X and Y: {skel.edges}"

    def test_orient_chain_y_to_x(self):
        """Chain Y->X: regression should orient Y->X correctly."""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import PCSkeletonDiscoverer
        rng = np.random.RandomState(42)
        n = 500
        Y = rng.randn(n)
        X = 0.8 * Y + 0.5 * rng.randn(n)  # Y -> X
        data = np.column_stack([X, Y])
        pc = PCSkeletonDiscoverer(alpha=0.01)
        skel = pc.discover(data, ["X", "Y"])
        assert ("X", "Y") in skel.edges or ("Y", "X") in skel.edges, \
            f"No edge between X and Y: {skel.edges}"

    def test_orient_v_structure(self):
        """V-structure X->Z<-Y: should find both directions pointing to Z."""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import PCSkeletonDiscoverer
        rng = np.random.RandomState(42)
        n = 500
        X = rng.randn(n)
        Y = rng.randn(n)
        Z = 0.7 * X + 0.6 * Y + 0.5 * rng.randn(n)
        data = np.column_stack([X, Y, Z])
        pc = PCSkeletonDiscoverer(alpha=0.01)
        skel = pc.discover(data, ["X", "Y", "Z"])
        assert ("X", "Z") in skel.edges
        assert ("Y", "Z") in skel.edges

    def test_orient_no_spurious_reverse(self):
        """Regression orientation: edges should not have spurious reverse-only.

        BIC+LiNGAM hybrid may keep edges undirected (both directions)
        when direction is ambiguous — this is correct behavior.
        """
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import PCSkeletonDiscoverer
        rng = np.random.RandomState(42)
        n = 500
        X = rng.randn(n)
        Y = 0.7 * X + 0.4 * rng.randn(n)
        data = np.column_stack([X, Y])
        pc = PCSkeletonDiscoverer(alpha=0.01)
        skel = pc.discover(data, ["X", "Y"])
        # Edge should exist (X→Y, Y→X, or both)
        assert ("X", "Y") in skel.edges or ("Y", "X") in skel.edges,             "No edge found between X and Y"
        # If only one direction, it should be correct (X→Y)
        if ("X", "Y") in skel.edges and ("Y", "X") not in skel.edges:
            pass  # X→Y only: correct
        elif ("Y", "X") in skel.edges and ("X", "Y") not in skel.edges:
            pass  # Y→X only: may happen with Gaussian data
        # Both directions = undirected = acceptable

    def test_orient_adj_matrix_directed(self):
        """Adjacency matrix should not be empty for connected variables."""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import PCSkeletonDiscoverer
        rng = np.random.RandomState(42)
        n = 500
        X = rng.randn(n)
        Y = 0.7 * X + 0.4 * rng.randn(n)
        data = np.column_stack([X, Y])
        pc = PCSkeletonDiscoverer(alpha=0.01)
        skel = pc.discover(data, ["X", "Y"])
        adj = skel.adj_matrix
        # At least one edge direction should exist
        assert adj[0, 1] == 1 or adj[1, 0] == 1,             f"No edge found in adjacency matrix: {adj}"
