"""tests/test_differentiable_causal.py"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._differentiable_causal import DifferentiableCausalInference, OptimizationResult


@pytest.fixture
def dci():
    return DifferentiableCausalInference(learning_rate=0.01)


@pytest.fixture
def simple_data():
    rng = np.random.RandomState(42)
    n = 200
    X = rng.randn(n)
    Y = 2.0 * X + 1.0 + 0.1 * rng.randn(n)
    return X, Y


class TestSetData:
    def test_set(self, dci, simple_data):
        X, Y = simple_data
        dci.set_data(treatment=X, outcome=Y)
        assert dci.treatment_effect == 0.0  # before optimization


class TestOptimize:
    def test_converges(self, dci, simple_data):
        X, Y = simple_data
        dci.set_data(treatment=X, outcome=Y)
        result = dci.optimize(n_iterations=200)
        assert isinstance(result, OptimizationResult)
        assert result.final_loss < result.initial_loss

    def test_ate_estimation(self, dci, simple_data):
        X, Y = simple_data
        dci.set_data(treatment=X, outcome=Y)
        dci.optimize(n_iterations=500)
        # 真实 β=2.0, 估计应接近
        assert abs(dci.treatment_effect - 2.0) < 0.5

    def test_with_confounders(self, dci):
        rng = np.random.RandomState(42)
        n = 200
        Z = rng.randn(n)
        X = 0.5 * Z + rng.randn(n) * 0.5
        Y = 1.5 * X + 0.8 * Z + rng.randn(n) * 0.1
        dci.set_data(treatment=X, outcome=Y, confounders=Z.reshape(-1, 1))
        result = dci.optimize(n_iterations=300)
        assert result.final_loss < result.initial_loss

    def test_loss_decreases(self, dci, simple_data):
        X, Y = simple_data
        dci.set_data(treatment=X, outcome=Y)
        dci.optimize(n_iterations=100)
        history = dci.loss_history
        assert len(history) > 0
        assert history[-1] <= history[0]


class TestPredict:
    def test_predict(self, dci, simple_data):
        X, Y = simple_data
        dci.set_data(treatment=X, outcome=Y)
        dci.optimize(n_iterations=50)
        pred = dci.predict()
        assert len(pred) == len(Y)


class TestStatistics:
    def test_stats(self, dci, simple_data):
        X, Y = simple_data
        dci.set_data(treatment=X, outcome=Y)
        stats = dci.statistics()
        assert stats["n_params"] == 3
