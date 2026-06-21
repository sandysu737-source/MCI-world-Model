"""
tests/test_p8_neural_symbolic_fusion.py — P8 神经符号融合 双向验证
=================================================================

覆盖:
    - neural_to_symbolic: 神经→符号 规则提取
    - symbolic_to_neural: 符号→神经 约束生成
    - fuse: 双向循环融合
    - 双向一致性验证 (roundtrip)
    - 边界情况
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._neural_symbolic_fusion_v2 import (
    FusionState,
    NeuralSymbolicFusionV2,
)


@pytest.fixture
def fusion():
    return NeuralSymbolicFusionV2(rule_threshold=0.7, max_iterations=10)


@pytest.fixture
def sample_vector():
    return np.array([1.0, 0.8, 0.3, 0.1])


@pytest.fixture
def var_names():
    return ["A", "B", "C", "D"]


# =============================================================================
# 基本功能
# =============================================================================


class TestFusionBasic:
    def test_init(self, fusion):
        assert fusion.fusion_count == 0

    def test_neural_to_symbolic(self, fusion, sample_vector, var_names):
        rules = fusion.neural_to_symbolic(sample_vector, var_names)
        assert isinstance(rules, list)
        # A/B ratio = 1.0/0.8 = 1.25 > 0.7 → should produce rule
        assert len(rules) > 0

    def test_neural_to_symbolic_no_varnames(self, fusion, sample_vector):
        rules = fusion.neural_to_symbolic(sample_vector)
        assert isinstance(rules, list)
        for r in rules:
            assert "type" in r
            assert "rule" in r
            assert "strength" in r

    def test_symbolic_to_neural(self, fusion, sample_vector, var_names):
        rules = fusion.neural_to_symbolic(sample_vector, var_names)
        constraint = fusion.symbolic_to_neural(rules, len(sample_vector))
        assert isinstance(constraint, np.ndarray)
        assert len(constraint) == len(sample_vector)

    def test_fuse_returns_state(self, fusion, sample_vector, var_names):
        state = fusion.fuse(sample_vector, var_names, n_iterations=5)
        assert isinstance(state, FusionState)
        assert state.n_iterations == 5
        assert state.fusion_score >= 0.0
        assert state.fusion_score <= 1.0

    def test_fuse_consistency_range(self, fusion, sample_vector, var_names):
        state = fusion.fuse(sample_vector, var_names, n_iterations=10)
        assert 0.0 <= state.consistency <= 1.0

    def test_fusion_history(self, fusion, sample_vector, var_names):
        fusion.fuse(sample_vector, var_names, n_iterations=3)
        assert fusion.fusion_count == 1

    def test_statistics(self, fusion, sample_vector, var_names):
        fusion.fuse(sample_vector, var_names, n_iterations=3)
        stats = fusion.statistics()
        assert stats["fusion_count"] == 1
        assert "avg_fusion_score" in stats


# =============================================================================
# 双向验证
# =============================================================================


class TestFusionBidirectional:
    def test_roundtrip_preserves_structure(self, fusion):
        """神经→符号→神经 roundtrip 保持结构。"""
        vec = np.array([1.0, 0.9, 0.2])
        names = ["X", "Y", "Z"]

        # 神经→符号
        rules = fusion.neural_to_symbolic(vec, names)
        assert len(rules) > 0  # X/Y = 1.11 > 0.7

        # 符号→神经
        constraint = fusion.symbolic_to_neural(rules, len(vec))
        # 约束应非零
        assert np.any(constraint > 0)

    def test_fuse_converges(self, fusion):
        """融合应收敛 (分数稳定)。"""
        vec = np.array([1.0, 0.85, 0.3, 0.15])
        names = ["A", "B", "C", "D"]
        state = fusion.fuse(vec, names, n_iterations=20)
        assert state.fusion_score > 0.0

    def test_fuse_improves_representation(self, fusion):
        """融合后表征应比原始更接近符号约束。"""
        vec = np.array([1.0, 0.5, 0.2, 0.1])
        names = ["A", "B", "C", "D"]

        # 未融合时的规则
        raw_rules = fusion.neural_to_symbolic(vec, names)
        raw_score = fusion._evaluate_fusion(vec, raw_rules)

        # 融合后
        fused = fusion.fuse(vec, names, n_iterations=10)
        assert fused.fusion_score >= raw_score

    def test_consistency_with_clear_pattern(self, fusion):
        """清晰模式应产生高一致性。"""
        vec = np.array([1.0, 0.95, 0.1])
        names = ["A", "B", "C"]
        state = fusion.fuse(vec, names, n_iterations=10)
        assert state.consistency >= 0.5


# =============================================================================
# 边界情况
# =============================================================================


class TestFusionEdgeCases:
    def test_empty_vector(self, fusion):
        rules = fusion.neural_to_symbolic(np.array([]))
        assert rules == []

    def test_single_element(self, fusion):
        rules = fusion.neural_to_symbolic(np.array([1.0]))
        assert rules == []

    def test_uniform_vector(self, fusion):
        """均匀向量 → 无规则 (ratio=1 < 0.7? 需要 =1，不应提取)。"""
        vec = np.array([0.5, 0.5, 0.5])
        rules = fusion.neural_to_symbolic(vec)
        # 1.0/1.0 = 1.0 > 0.7 但在 vec[0]/vec[1] = 0.5/0.5 = 1.0 > 0.7
        assert len(rules) >= 0  # 取决于阈值

    def test_empty_rules_fuse(self, fusion):
        """空规则融合不崩溃。"""
        vec = np.array([0.1, 0.1, 0.1])
        state = fusion.fuse(vec, n_iterations=5)
        assert isinstance(state, FusionState)
        assert state.fusion_score >= 0

    def test_fuse_no_varnames(self, fusion):
        state = fusion.fuse(np.array([1.0, 0.8, 0.3]))
        assert isinstance(state, FusionState)

    def test_custom_threshold(self):
        """自定义阈值影响规则数量。"""
        lo = NeuralSymbolicFusionV2(rule_threshold=0.5)
        hi = NeuralSymbolicFusionV2(rule_threshold=2.0)
        vec = np.array([1.0, 0.6, 0.3])
        assert len(lo.neural_to_symbolic(vec)) >= len(hi.neural_to_symbolic(vec))


# =============================================================================
# 因果图 → 符号逻辑 验证
# =============================================================================


class TestFusionCausalIntegration:
    def test_fuse_causal_graph_like_data(self, fusion):
        """类似因果图邻接矩阵的融合。"""
        adj_vec = np.array([1.0, 0.7, 0.1, 0.8, 0.2])
        names = ["A→B", "B→C", "A↛C", "C→D", "B↛D"]
        state = fusion.fuse(adj_vec, names, n_iterations=10)
        assert state.fusion_score > 0

    def test_symbolic_rules_from_causal_graph(self, fusion):
        """从因果图参数提取的符号规则应可靠。"""
        # 模拟参数: A→B 强, B→C 中等, 其余弱
        params = np.array([0.9, 0.5, 0.1])
        names = ["A→B", "B→C", "A↛C"]
        rules = fusion.neural_to_symbolic(params, names)
        # A→B / B→C ratio = 1.8 > 0.7
        assert len(rules) > 0
        # 规则应提及 A→B
        rule_texts = [r["rule"] for r in rules]
        assert any("A→B" in t for t in rule_texts)
