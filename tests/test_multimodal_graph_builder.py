"""
tests/test_multimodal_graph_builder.py — MultimodalGraphBuilder 测试
=====================================================================

覆盖:
    - build_from_features: 多模态时序 → 因果边
    - build_cross_modality_edges: 跨模态因果边检测
    - 自回归边（模态内）
    - 阈值/滞后参数
    - 边界条件（空输入/维度不一致/过短时序）
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._multimodal_graph_builder import MultimodalGraphBuilder


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def builder():
    return MultimodalGraphBuilder(min_correlation=0.3, max_lag=3)


@pytest.fixture
def correlated_timeline():
    """vision 和 audio 高度相关的时序。"""
    rng = np.random.RandomState(42)
    T = 20
    base = np.cumsum(rng.randn(T))
    timeline = []
    for t in range(T):
        timeline.append({
            "vision": np.array([base[t] + rng.randn() * 0.1, base[t] * 0.5]),
            "audio": np.array([base[t] + rng.randn() * 0.1, base[t] * 0.3]),
        })
    return timeline


@pytest.fixture
def uncorrelated_timeline():
    """vision 和 audio 不相关的时序。"""
    rng = np.random.RandomState(42)
    T = 20
    timeline = []
    for t in range(T):
        timeline.append({
            "vision": rng.randn(2),
            "audio": rng.randn(2),
        })
    return timeline


# =============================================================================
# TestMultimodalGraphBuilder — 基本构造
# =============================================================================


class TestMultimodalGraphBuilderInit:
    def test_default_params(self):
        b = MultimodalGraphBuilder()
        assert b.min_correlation == 0.3

    def test_custom_params(self, builder):
        assert builder.min_correlation == 0.3

    def test_repr(self, builder):
        r = repr(builder)
        assert "MultimodalGraphBuilder" in r


# =============================================================================
# TestBuildFromFeatures — 主入口
# =============================================================================


class TestBuildFromFeatures:
    def test_empty_timeline(self, builder):
        edges = builder.build_from_features([])
        assert edges == []

    def test_too_short_timeline(self, builder):
        """少于 3 步 → 无因果边。"""
        timeline = [
            {"vision": np.array([1.0, 2.0])},
            {"vision": np.array([1.1, 2.1])},
        ]
        edges = builder.build_from_features(timeline)
        assert edges == []

    def test_correlated_produces_edges(self, builder, correlated_timeline):
        """高相关时序应产出因果边。"""
        edges = builder.build_from_features(correlated_timeline)
        assert len(edges) > 0

    def test_edge_structure(self, builder, correlated_timeline):
        """因果边包含必要字段。"""
        edges = builder.build_from_features(correlated_timeline)
        for edge in edges:
            assert "cause" in edge
            assert "effect" in edge
            assert "correlation" in edge
            assert "lag" in edge
            assert "modality_pair" in edge
            assert "confidence" in edge
            assert "direction" in edge

    def test_correlation_range(self, builder, correlated_timeline):
        """相关系数在 [-1, 1] 范围内。"""
        edges = builder.build_from_features(correlated_timeline)
        for edge in edges:
            assert -1.0 <= edge["correlation"] <= 1.0

    def test_confidence_range(self, builder, correlated_timeline):
        """置信度在 [0, 1] 范围内。"""
        edges = builder.build_from_features(correlated_timeline)
        for edge in edges:
            assert 0.0 <= edge["confidence"] <= 1.0

    def test_single_modality(self, builder):
        """单模态 → 仅有自回归边。"""
        rng = np.random.RandomState(42)
        timeline = [{"vision": np.cumsum(rng.randn(5))[t]} for t in range(5)]
        edges = builder.build_from_features(timeline)
        for edge in edges:
            assert edge["modality_pair"][0] == "vision"
            assert edge["modality_pair"][1] == "vision"

    def test_missing_modality_in_some_steps(self, builder):
        """部分步骤缺失某模态 → 仍能处理。"""
        timeline = [
            {"vision": np.array([1.0]), "audio": np.array([1.0])},
            {"vision": np.array([2.0])},  # audio 缺失
            {"vision": np.array([3.0]), "audio": np.array([3.0])},
            {"vision": np.array([4.0]), "audio": np.array([4.0])},
            {"vision": np.array([5.0]), "audio": np.array([5.0])},
        ]
        edges = builder.build_from_features(timeline)
        # 应该有一些边（至少有 vision 自回归）
        assert isinstance(edges, list)


# =============================================================================
# TestBuildCrossModalityEdges — 跨模态边
# =============================================================================


class TestBuildCrossModalityEdges:
    def test_correlated_modalities(self, builder):
        """高度相关的两个模态应产出跨模态边。"""
        T = 20
        rng = np.random.RandomState(42)
        base = np.cumsum(rng.randn(T))
        features_a = np.column_stack([base, base * 0.5])
        features_b = np.column_stack([base + rng.randn(T) * 0.1, base * 0.3])
        edges = builder.build_cross_modality_edges(
            features_a, features_b, "vision", "audio",
        )
        assert len(edges) > 0
        for edge in edges:
            assert edge["modality_pair"] in [
                ("vision", "audio"),
                ("audio", "vision"),
            ]

    def test_uncorrelated_modalities(self, builder):
        """不相关的模态应产出较少或无边。"""
        rng = np.random.RandomState(42)
        T = 20
        features_a = rng.randn(T, 2)
        features_b = rng.randn(T, 2)
        edges = builder.build_cross_modality_edges(
            features_a, features_b, "vision", "audio",
        )
        # 可能有少量偶然相关，但应少于强相关的情况
        assert len(edges) < 10

    def test_short_series(self, builder):
        """过短序列 → 无边。"""
        features_a = np.array([[1.0], [2.0]])
        features_b = np.array([[1.0], [2.0]])
        edges = builder.build_cross_modality_edges(
            features_a, features_b, "a", "b",
        )
        assert edges == []

    def test_direction_field(self, builder):
        """因果边的 direction 字段格式正确。"""
        T = 15
        base = np.linspace(0, 10, T)
        fa = np.column_stack([base, base * 2])
        fb = np.column_stack([base + 0.1, base * 1.5])
        edges = builder.build_cross_modality_edges(fa, fb, "vision", "audio")
        for edge in edges:
            assert "→" in edge["direction"]

    def test_lag_range(self, builder):
        """滞后值在 [0, max_lag] 范围内。"""
        T = 20
        rng = np.random.RandomState(42)
        base = np.cumsum(rng.randn(T))
        fa = np.column_stack([base, base])
        fb = np.column_stack([base, base])
        edges = builder.build_cross_modality_edges(fa, fb, "a", "b")
        for edge in edges:
            assert 0 <= edge["lag"] <= 3


# =============================================================================
# TestThresholdSensitivity — 阈值敏感性
# =============================================================================


class TestThresholdSensitivity:
    def test_high_threshold_fewer_edges(self):
        """高阈值应产出更少的边。"""
        rng = np.random.RandomState(42)
        T = 20
        base = np.cumsum(rng.randn(T))
        timeline = [
            {"vision": np.array([base[t]]), "audio": np.array([base[t] * 0.5 + rng.randn() * 0.5])}
            for t in range(T)
        ]
        b_low = MultimodalGraphBuilder(min_correlation=0.1)
        b_high = MultimodalGraphBuilder(min_correlation=0.9)
        edges_low = b_low.build_from_features(timeline)
        edges_high = b_high.build_from_features(timeline)
        assert len(edges_low) >= len(edges_high)

    def test_max_lag_zero(self):
        """max_lag=0 → 只检测同步相关。"""
        b = MultimodalGraphBuilder(min_correlation=0.3, max_lag=0)
        T = 15
        base = np.linspace(0, 5, T)
        fa = np.column_stack([base])
        fb = np.column_stack([base])
        edges = b.build_cross_modality_edges(fa, fb, "a", "b")
        for edge in edges:
            assert edge["lag"] == 0


# =============================================================================
# TestHelperMethods — 辅助方法
# =============================================================================


class TestHelperMethods:
    def test_extract_series_missing_modality(self, builder):
        """提取不存在的模态 → None。"""
        timeline = [{"vision": np.array([1.0])} for _ in range(5)]
        result = builder._extract_series(timeline, "audio")
        assert result is None

    def test_extract_series_inconsistent_dim(self, builder):
        """维度不一致 → None。"""
        timeline = [
            {"vision": np.array([1.0, 2.0])},
            {"vision": np.array([1.0])},
            {"vision": np.array([1.0, 2.0])},
        ]
        result = builder._extract_series(timeline, "vision")
        assert result is None

    def test_scalar_correlation_identical(self, builder):
        """完全相同的序列 → 相关=1。"""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        corr = builder._scalar_correlation(a, a)
        assert abs(corr - 1.0) < 1e-10

    def test_scalar_correlation_opposite(self, builder):
        """完全相反的序列 → 相关=-1。"""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = -a
        corr = builder._scalar_correlation(a, b)
        assert abs(corr + 1.0) < 1e-10

    def test_scalar_correlation_short(self, builder):
        """过短序列 → 相关=0。"""
        corr = builder._scalar_correlation(np.array([1.0]), np.array([2.0]))
        assert corr == 0.0

    def test_vector_lagged_correlation_zero_denom(self, builder):
        """常值序列 → 相关=0（分母为零）。"""
        a = np.ones((10, 2))
        b = np.ones((10, 2))
        corr = builder._vector_lagged_correlation(a, b, 0)
        assert corr == 0.0
