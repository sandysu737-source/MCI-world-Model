"""tests/test_symbol_grounding.py"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._symbol_grounding import (
    GroundingEntry,
    SymbolGroundingLearning,
)


@pytest.fixture
def sgl():
    return SymbolGroundingLearning(similarity_threshold=0.5)


class TestGround:
    def test_new_symbol(self, sgl):
        vec = np.array([1.0, 0.0, 0.0])
        entry = sgl.ground("red", "vision", vec)
        assert isinstance(entry, GroundingEntry)
        assert entry.symbol == "red"
        assert entry.modality == "vision"
        assert entry.examples == 1
        assert entry.grounding_strength == pytest.approx(0.2)

    def test_incremental_update(self, sgl):
        vec1 = np.array([1.0, 0.0, 0.0])
        sgl.ground("red", "vision", vec1)
        vec2 = np.array([0.9, 0.1, 0.0])
        entry = sgl.ground("red", "vision", vec2)
        assert entry.examples == 2
        assert entry.grounding_strength > 0.2

    def test_grounding_count(self, sgl):
        assert sgl.grounded_symbol_count == 0
        sgl.ground("red", "vision", np.array([1.0, 0.0]))
        sgl.ground("hot", "thermal", np.array([0.0, 1.0]))
        assert sgl.grounded_symbol_count == 2


class TestVerifyGrounding:
    def test_matching_vector(self, sgl):
        vec = np.array([1.0, 0.0, 0.0])
        sgl.ground("red", "vision", vec)
        similarity = sgl.verify_grounding("red", vec)
        assert similarity > 0.9

    def test_unknown_symbol(self, sgl):
        similarity = sgl.verify_grounding("blue", np.array([1.0, 0.0]))
        assert similarity == 0.0

    def test_orthogonal_vector(self, sgl):
        sgl.ground("red", "vision", np.array([1.0, 0.0]))
        similarity = sgl.verify_grounding("red", np.array([0.0, 1.0]))
        assert similarity < 0.1


class TestIsGrounded:
    def test_not_grounded_initially(self, sgl):
        # First grounding gives strength=0.2 < 0.5 threshold
        sgl.ground("red", "vision", np.array([1.0, 0.0]))
        assert not sgl.is_grounded("red")

    def test_becomes_grounded(self, sgl):
        vec = np.array([1.0, 0.0])
        for _ in range(5):
            sgl.ground("red", "vision", vec)
        # After 5 examples: strength = min(1.0, 5/5.0) = 1.0 >= 0.5
        assert sgl.is_grounded("red")

    def test_unknown_not_grounded(self, sgl):
        assert not sgl.is_grounded("blue")


class TestGetUngroundedSymbols:
    def test_mixed(self, sgl):
        vec = np.array([1.0, 0.0])
        for _ in range(5):
            sgl.ground("red", "vision", vec)
        sgl.ground("blue", "vision", np.array([0.0, 1.0]))
        ungrounded = sgl.get_ungrounded_symbols(["red", "blue", "green"])
        assert "blue" in ungrounded
        assert "green" in ungrounded
        assert "red" not in ungrounded


class TestGetGrounding:
    def test_existing(self, sgl):
        sgl.ground("red", "vision", np.array([1.0, 0.0]))
        entry = sgl.get_grounding("red")
        assert entry is not None
        assert entry.symbol == "red"

    def test_nonexistent(self, sgl):
        assert sgl.get_grounding("blue") is None


class TestStatistics:
    def test_empty_stats(self):
        s = SymbolGroundingLearning()
        stats = s.statistics()
        assert stats["grounded_symbols"] == 0
        assert stats["avg_grounding_strength"] == 0.0

    def test_stats_after_grounding(self, sgl):
        sgl.ground("red", "vision", np.array([1.0, 0.0]))
        stats = sgl.statistics()
        assert stats["grounded_symbols"] == 1
        assert "red" in stats["symbols"]
