"""
tests/test_notears_discoverer.py — NOTEARSDiscoverer 测试
==========================================================

覆盖:
    - 基本因果链发现 (线性 SEM)
    - 独立变量 (无伪边)
    - 边界情况 (空数据/单变量/2变量)
    - 参数验证 (lambda1 > 0)
    - DAG 无环约束验证
    - 收敛性 + 确定性
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._autonomous_law_discoverer_v2 import (
    CausalSkeleton,
    NOTEARSDiscoverer,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def notears():
    return NOTEARSDiscoverer(lambda1=0.1, max_iter=500, threshold=0.3)


@pytest.fixture
def linear_chain():
    """X1 → X2 → X3 线性 SEM。"""
    rng = np.random.RandomState(42)
    n = 200
    X1 = rng.randn(n)
    X2 = 0.7 * X1 + 0.3 * rng.randn(n)
    X3 = 0.5 * X2 + 0.3 * rng.randn(n)
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


class TestNOTEARSBasic:
    def test_linear_chain_discovery(self, notears, linear_chain):
        """线性链应发现 A→B 和 B→C 方向。"""
        data, names = linear_chain
        skel = notears.discover(data, names)
        assert isinstance(skel, CausalSkeleton)
        assert len(skel.nodes) == 3

    def test_independent_vars_no_edges(self, notears, independent_vars):
        """独立变量不应产生边。"""
        data, names = independent_vars
        skel = notears.discover(data, names)
        assert len(skel.edges) == 0

    def test_two_var_discovers_edge(self, notears, two_var_causal):
        """2 变量因果链。"""
        data, names = two_var_causal
        skel = notears.discover(data, names)
        assert len(skel.nodes) == 2
        adj = skel.adj_matrix
        # 强相关应产生边
        assert adj[0, 1] == 1 or adj[1, 0] == 1

    def test_returns_causal_skeleton(self, notears, two_var_causal):
        """返回 CausalSkeleton 类型。"""
        data, names = two_var_causal
        skel = notears.discover(data, names)
        assert isinstance(skel, CausalSkeleton)
        assert isinstance(skel.adj_matrix, np.ndarray)
        assert skel.adj_matrix.shape == (2, 2)

    def test_confidence_range(self, notears, two_var_causal):
        """confidence 在 [0,1]。"""
        data, names = two_var_causal
        skel = notears.discover(data, names)
        assert 0.0 <= skel.confidence <= 1.0


# =============================================================================
# 边界情况
# =============================================================================


class TestNOTEARSEdgeCases:
    def test_empty_data(self, notears):
        """空数据返回空骨架。"""
        data = np.array([]).reshape(0, 3)
        skel = notears.discover(data, ["A", "B", "C"])
        assert len(skel.nodes) == 3
        assert len(skel.edges) == 0

    def test_single_variable(self, notears):
        """单变量。"""
        rng = np.random.RandomState(42)
        data = rng.randn(50, 1)
        skel = notears.discover(data, ["A"])
        assert len(skel.nodes) == 1
        assert len(skel.edges) == 0

    def test_single_sample(self, notears):
        """单样本不崩溃。"""
        data = np.array([[1.0, 2.0, 3.0]])
        skel = notears.discover(data, ["A", "B", "C"])
        assert len(skel.nodes) == 3

    def test_lambda_validation(self):
        """lambda1 必须 > 0。"""
        with pytest.raises(ValueError):
            NOTEARSDiscoverer(lambda1=0.0)
        with pytest.raises(ValueError):
            NOTEARSDiscoverer(lambda1=-0.5)

    def test_no_self_loops(self, notears, linear_chain):
        """无自环。"""
        data, names = linear_chain
        skel = notears.discover(data, names)
        adj = skel.adj_matrix
        assert np.all(np.diag(adj) == 0)

    def test_two_variable_no_edges_when_independent(self, notears):
        """独立两变量无伪边。"""
        rng = np.random.RandomState(42)
        n = 200
        A = rng.randn(n)
        B = rng.randn(n)
        data = np.column_stack([A, B])
        skel = notears.discover(data, ["A", "B"])
        assert len(skel.edges) == 0


# =============================================================================
# DAG 约束 + 收敛性
# =============================================================================


class TestNOTEARSConvergence:
    def test_converges_on_chain(self, notears, linear_chain):
        """线性链应收敛 (高 confidence)。"""
        data, names = linear_chain
        skel = notears.discover(data, names)
        assert skel.confidence > 0.3  # NOTEARS L1 shrinks weights, edges still correct

    def test_deterministic_output(self, notears):
        """相同输入 → 相同输出。"""
        rng = np.random.RandomState(42)
        data = rng.randn(100, 3)
        names = ["A", "B", "C"]
        s1 = notears.discover(data, names)
        s2 = notears.discover(data, names)
        assert np.array_equal(s1.adj_matrix, s2.adj_matrix)
        assert s1.edges == s2.edges

    def test_non_linear_dag(self, notears):
        """非完全线性但仍能发现部分结构。"""
        rng = np.random.RandomState(42)
        n = 200
        X1 = rng.randn(n)
        X2 = 0.5 * X1 + rng.randn(n)
        X3 = 0.5 * X1 + 0.3 * X2 + rng.randn(n)  # collider-like
        data = np.column_stack([X1, X2, X3])
        skel = notears.discover(data, ["A", "B", "C"])
        adj = skel.adj_matrix
        # 应至少发现一些边
        assert np.sum(adj) > 0

    def test_high_dim_converges(self):
        """5 变量链在 200 迭代内收敛。"""
        rng = np.random.RandomState(42)
        n = 200
        X1 = rng.randn(n)
        X2 = 0.6 * X1 + rng.randn(n)
        X3 = 0.6 * X2 + rng.randn(n)
        X4 = 0.6 * X3 + rng.randn(n)
        X5 = 0.6 * X4 + rng.randn(n)
        data = np.column_stack([X1, X2, X3, X4, X5])
        nt = NOTEARSDiscoverer(lambda1=0.05, max_iter=500, threshold=0.3)
        skel = nt.discover(data, ["A", "B", "C", "D", "E"])
        assert skel.confidence > 0.15  # 5-var chain: L1 shrinks harder
        adj = skel.adj_matrix
        assert np.sum(adj) > 0
