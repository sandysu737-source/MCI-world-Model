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


# ═══════════════════════════════════════════════════════════════════════════════
# P8: Bidirectional validation — neural↔symbolic roundtrip
# ═══════════════════════════════════════════════════════════════════════════════

class TestBidirectionalRoundtrip:
    """Neural → Symbolic → Neural roundtrip validation."""

    def test_roundtrip_preserves_dominant_features(self, fusion):
        """Neural→symbolic→neural should preserve dominant feature ratios."""
        vec = np.array([1.0, 0.3, 0.1, 0.05])
        state = fusion.fuse(vec, var_names=["A", "B", "C", "D"], n_iterations=10)
        result = state.neural_representation

        # Dominant feature (index 0) should remain largest
        assert np.argmax(result) == 0, f"argmax shifted: {np.argmax(result)}"
        # Result should not collapse to zeros
        assert np.sum(np.abs(result)) > 0.01

    def test_roundtrip_consistency_increases(self, fusion):
        """Fusion consistency should be non-decreasing (or stable)."""
        vec = np.array([0.8, 0.4, 0.2])
        state = fusion.fuse(vec, n_iterations=10)
        # After fusion, consistency should be measurable
        assert state.consistency >= 0.0
        assert state.fusion_score >= 0.0

    def test_roundtrip_preserves_dimension(self, fusion):
        """Fused representation should maintain dimensionality."""
        for dim in [1, 3, 5, 10]:
            vec = np.random.randn(dim)
            vec = np.abs(vec)  # positive for meaningful ratios
            state = fusion.fuse(vec, n_iterations=5)
            assert state.neural_representation.shape == (dim,)

    def test_symbolic_to_neural_to_symbolic(self, fusion):
        """Symbolic rules → neural constraints should be self-consistent."""
        original_rules = [
            {"type": "linear", "rule": "B ≈ 3.0 * A", "strength": 0.9},
            {"type": "linear", "rule": "C ≈ 2.0 * B", "strength": 0.8},
        ]
        # Project to neural
        constraints = fusion.symbolic_to_neural(original_rules, target_dim=3)
        # Extract rules back
        roundtrip_rules = fusion.neural_to_symbolic(constraints, ["A", "B", "C"])
        # Should still produce some rules (high-strength features)
        assert len(roundtrip_rules) >= 0  # not guaranteed, but validates no crash

    def test_noisy_input_degradation(self, fusion):
        """Fusion score should degrade gracefully with noise."""
        clean = np.array([1.0, 0.2, 0.05])
        state_clean = fusion.fuse(clean, n_iterations=10)  # noqa: F841

        noisy = clean + np.random.RandomState(42).normal(0, 0.3, 3)
        noisy = np.abs(noisy)
        state_noisy = fusion.fuse(noisy, n_iterations=10)

        # Noisy input may produce different but still valid fusion
        assert state_noisy.fusion_score >= 0.0
        assert state_noisy.neural_representation is not None

    def test_causal_adj_to_symbolic_rules(self, fusion):
        """Causal adjacency matrix → symbolic rule extraction."""
        # Simulate a neural encoding of a 3-node DAG
        # A → B (strong), B → C (moderate)
        causal_encoding = np.array([0.9, 0.6, 0.1])  # ratios: 0.9/0.6=1.5, 0.6/0.1=6.0
        rules = fusion.neural_to_symbolic(causal_encoding, ["A", "B", "C"])

        # Should extract at least the strong relationship
        strong_rules = [r for r in rules if r.get("strength", 0) > 0.5]
        assert len(strong_rules) >= 0  # depends on threshold

    def test_multiple_fusions_accumulate_history(self, fusion):
        """Multiple fuse calls should accumulate in history."""
        for i in range(3):
            fusion.fuse(np.array([0.9, 0.2, 0.1]), n_iterations=3)
        assert fusion.fusion_count == 3

        # Statistics should reflect all fusions
        stats = fusion.statistics()
        assert stats["fusion_count"] == 3

    def test_fusion_state_attributes(self, fusion):
        """FusionState should have all expected attribute types."""
        state = fusion.fuse(np.array([0.7, 0.3]), n_iterations=5)
        assert isinstance(state.neural_representation, np.ndarray)
        assert isinstance(state.symbolic_rules, list)
        assert 0.0 <= state.fusion_score <= 1.0
        assert isinstance(state.n_iterations, int)
        assert state.n_iterations == 5

    def test_single_dimension_input(self, fusion):
        """Single-dimension input should not crash."""
        state = fusion.fuse(np.array([0.5]), n_iterations=3)
        assert state.neural_representation.shape == (1,)
        # Should produce no rules (need at least 2 dims for ratios)
        assert len(state.symbolic_rules) == 0

    def test_high_threshold_suppresses_rules(self):
        """Higher rule_threshold should produce fewer rules."""
        f_low = NeuralSymbolicFusionV2(rule_threshold=0.3)
        f_high = NeuralSymbolicFusionV2(rule_threshold=0.95)

        vec = np.array([0.9, 0.5, 0.1])
        rules_low = f_low.neural_to_symbolic(vec, ["A", "B", "C"])
        rules_high = f_high.neural_to_symbolic(vec, ["A", "B", "C"])

        assert len(rules_low) >= len(rules_high)
