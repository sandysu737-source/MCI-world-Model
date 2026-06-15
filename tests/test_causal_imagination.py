"""tests/test_causal_imagination.py"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._causal_imagination import CausalImaginationEngine, ImaginedWorld


@pytest.fixture
def engine():
    return CausalImaginationEngine(state_dim=5, n_imagination_steps=3)


class TestImagination:
    def test_imagine(self, engine):
        engine.set_current_state(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        worlds = engine.imagine(intervention={0: 0.0})
        assert len(worlds) == 1
        assert isinstance(worlds[0], ImaginedWorld)
        assert worlds[0].plausibility >= 0.0

    def test_imagine_multiple(self, engine):
        engine.set_current_state(np.ones(5))
        worlds = engine.imagine(intervention={1: 10.0}, n_worlds=3)
        assert len(worlds) == 3

    def test_no_state(self, engine):
        worlds = engine.imagine(intervention={0: 5.0})
        assert len(worlds) == 1

    def test_narrative(self, engine):
        engine.set_current_state(np.zeros(5))
        worlds = engine.imagine(intervention={"var_X": 1.0})
        assert len(worlds[0].narrative) > 0


class TestCounterfactuals:
    def test_explore(self, engine):
        engine.set_current_state(np.ones(5))
        worlds = engine.explore_counterfactuals(0, [0.0, 0.5, 1.0])
        assert len(worlds) == 3


class TestCausalMatrix:
    def test_with_matrix(self, engine):
        engine.set_current_state(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
        matrix = np.eye(5) * 0.5
        matrix[0, 1] = 0.8
        engine.set_causal_matrix(matrix)
        worlds = engine.imagine(intervention={0: 5.0})
        assert len(worlds) == 1


class TestStatistics:
    def test_stats(self, engine):
        engine.set_current_state(np.ones(5))
        engine.imagine(intervention={0: 0.0})
        stats = engine.statistics()
        assert stats["imagination_count"] == 1
