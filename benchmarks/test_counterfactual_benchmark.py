"""Counterfactual accuracy benchmark.

Tests CEWM CounterfactualEngine on known SEM data.
"""

from __future__ import annotations

import numpy as np

from mci_world_model.sdk._counterfactual import CounterfactualEngine
from mci_world_model.sdk._do_calculus import CausalGraph


def _generate_cf_data(n=200, seed=42):
    rng = np.random.RandomState(seed)
    Z = rng.randn(n)
    X = 0.3 * Z + rng.randn(n)
    Y = 2.0 * X + 0.5 * Z + 0.3 * rng.randn(n)
    graph = CausalGraph(nodes=["X","Y","Z"], edges=[("X","Y"),("Z","X"),("Z","Y")])
    return X, Y, Z, graph


def _make_engine(graph, seed=42):
    return CounterfactualEngine.from_causal_graph(graph, noise_std=0.3, seed=seed)


class TestCounterfactualAccuracy:

    def test_cf_direction_correct(self):
        X, Y, Z, graph = _generate_cf_data(n=300)
        engine = _make_engine(graph)
        assert engine is not None
        r_up = engine.query({"X": 1.0}, {"X": 3.0}, "Y")
        r_down = engine.query({"X": 1.0}, {"X": -1.0}, "Y")
        assert r_up.counterfactual_value > r_down.counterfactual_value

    def test_cf_ate_positive(self):
        X, Y, Z, graph = _generate_cf_data(n=300)
        engine = _make_engine(graph)
        assert engine is not None
        diffs = []
        for i in range(min(15, len(X))):
            r1 = engine.query({"X": float(X[i])}, {"X": float(X[i])+1.0}, "Y")
            r0 = engine.query({"X": float(X[i])}, {"X": float(X[i])}, "Y")
            diffs.append(r1.counterfactual_value - r0.counterfactual_value)
        avg = float(np.mean(diffs))
        print(f"\n  Counterfactual ATE: {avg:.4f}")
        assert avg > 0

    def test_cf_not_none(self):
        X, Y, Z, graph = _generate_cf_data(n=100)
        engine = _make_engine(graph)
        assert engine is not None
        r = engine.query({"X": 0.0}, {"X": 1.0}, "Y")
        assert r is not None
        assert np.isfinite(r.counterfactual_value)
