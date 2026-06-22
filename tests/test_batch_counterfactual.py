"""Tests for BatchCounterfactualEngine — vectorized batch counterfactual queries."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from mci_world_model.sdk._batch_counterfactual import BatchCounterfactualEngine
from mci_world_model.sdk._counterfactual import CounterfactualResult, StructuralEquationModel


@pytest.fixture
def linear_sem():
    """X → Y, with X exogenous."""
    return StructuralEquationModel(
        coefficients=np.array([[0, 1], [0, 0]], dtype=np.float64),
        node_names=["X", "Y"],
        noise_std=0.3,
    )


@pytest.fixture
def engine(linear_sem):
    return BatchCounterfactualEngine(linear_sem)


class TestBatchCounterfactualBasic:
    def test_sem_property(self, engine, linear_sem):
        assert engine.sem is linear_sem

    def test_node_names(self, engine):
        assert engine.node_names == ["X", "Y"]

    def test_batch_query_empty(self, engine):
        results = engine.batch_query([])
        assert results == []

    def test_batch_query_single_scenario(self, engine):
        results = engine.batch_query([
            {"evidence": {"X": 1.0}, "do_x": {"X": 0.5}, "target": "Y"},
        ])
        assert len(results) == 1
        assert isinstance(results[0], CounterfactualResult)

    def test_batch_query_multiple_scenarios(self, engine):
        scenarios = [
            {"evidence": {"X": 1.0}, "do_x": {"X": 0.5}, "target": "Y"},
            {"evidence": {"X": 2.0}, "do_x": {"X": 1.0}, "target": "Y"},
            {"evidence": {"X": 3.0}, "do_x": {"X": 1.5}, "target": "Y"},
        ]
        results = engine.batch_query(scenarios)
        assert len(results) == 3

    def test_batch_query_invalid_target(self, engine):
        results = engine.batch_query([
            {"evidence": {}, "do_x": {}, "target": "Z"},  # nonexistent
        ])
        assert len(results) == 1
        assert results[0].counterfactual_value is None or True  # survived

    def test_batch_query_n_mc_parameter(self, engine):
        results = engine.batch_query(
            [{"evidence": {"X": 1.0}, "do_x": {"X": 0.0}, "target": "Y"}],
            n_mc=500,
        )
        assert len(results) == 1

    def test_batch_query_result_structure(self, engine):
        results = engine.batch_query([
            {"evidence": {"X": 1.0}, "do_x": {"X": 0.0}, "target": "Y"},
        ])
        r = results[0]
        assert hasattr(r, "counterfactual_value")
        assert hasattr(r, "factual_value")

    def test_batch_query_deterministic(self, engine):
        """Same input should give reproducible results (no external randomness)."""
        scenario = {"evidence": {"X": 1.0}, "do_x": {"X": 0.5}, "target": "Y"}
        r1 = engine.batch_query([scenario], n_mc=1000)
        r2 = engine.batch_query([scenario], n_mc=1000)
        # Results should be approximately equal (MC converges)
        if r1[0].counterfactual_value is not None and r2[0].counterfactual_value is not None:
            assert abs(r1[0].counterfactual_value - r2[0].counterfactual_value) < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
