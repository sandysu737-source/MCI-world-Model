"""tests/test_neural_symbolic_fusion_v2.py"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._neural_symbolic_fusion_v2 import (
    FusionState,
    NeuralSymbolicFusionV2,
)


@pytest.fixture
def fusion():
    return NeuralSymbolicFusionV2(rule_threshold=0.7, consistency_threshold=0.6)


class TestNeuralToSymbolic:
    def test_basic_extraction(self, fusion):
        vec = np.array([0.9, 0.3, 0.1])
        rules = fusion.neural_to_symbolic(vec, var_names=["A", "B", "C"])
        # ratio A/B = 3.0 > 0.7 → rule produced
        assert len(rules) >= 1
        assert all(r["type"] == "linear" for r in rules)

    def test_default_var_names(self, fusion):
        vec = np.array([1.0, 0.1])
        rules = fusion.neural_to_symbolic(vec)
        assert len(rules) >= 1
        assert "v0" in rules[0]["rule"] or "v1" in rules[0]["rule"]

    def test_uniform_vector(self, fusion):
        vec = np.array([0.5, 0.5, 0.5])
        rules = fusion.neural_to_symbolic(vec)
        # ratios ≈ 1.0 all > 0.7 → still produce rules
        assert isinstance(rules, list)


class TestSymbolicToNeural:
    def test_basic_projection(self, fusion):
        rules = [{"type": "linear", "rule": "B ≈ 0.8 * A", "strength": 0.8}]
        result = fusion.symbolic_to_neural(rules, target_dim=4)
        assert result.shape == (4,)
        assert result[0] == pytest.approx(0.8)

    def test_empty_rules(self, fusion):
        result = fusion.symbolic_to_neural([], target_dim=3)
        assert result.shape == (3,)
        assert np.allclose(result, 0.0)


class TestFuse:
    def test_basic_fuse(self, fusion):
        vec = np.array([0.9, 0.2, 0.1])
        state = fusion.fuse(vec, n_iterations=5)
        assert isinstance(state, FusionState)
        assert state.n_iterations == 5
        assert state.fusion_score >= 0.0
        assert state.neural_representation is not None

    def test_fuse_increments_count(self, fusion):
        assert fusion.fusion_count == 0
        fusion.fuse(np.array([1.0, 0.5]))
        assert fusion.fusion_count == 1
        fusion.fuse(np.array([1.0, 0.5]))
        assert fusion.fusion_count == 2

    def test_fuse_default_iterations(self, fusion):
        vec = np.array([0.8, 0.3])
        state = fusion.fuse(vec)
        assert state.n_iterations == fusion._max_iterations


class TestStatistics:
    def test_empty_stats(self):
        f = NeuralSymbolicFusionV2()
        stats = f.statistics()
        assert stats["fusion_count"] == 0
        assert stats["avg_fusion_score"] == 0.0

    def test_stats_after_fuse(self, fusion):
        fusion.fuse(np.array([0.9, 0.2]), n_iterations=3)
        stats = fusion.statistics()
        assert stats["fusion_count"] == 1
        assert stats["avg_fusion_score"] > 0.0
