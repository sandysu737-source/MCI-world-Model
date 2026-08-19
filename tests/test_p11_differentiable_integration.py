"""
tests/test_p11_differentiable_integration.py — P11 可微因果 + NOTEARS 整合
==========================================================================

覆盖:
    - DifferentiableCausalInference 端到端
    - NOTEARS → DifferentiableCausal 桥接
    - 梯度传播验证
    - 因果参数优化
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._autonomous_law_discoverer_v2 import NOTEARSDiscoverer
from mci_world_model.sdk._differentiable_causal import (
    CausalParameter,
    DifferentiableCausalInference,
    OptimizationResult,
)


def _make_dci():
    rng = np.random.RandomState(42)
    n = 200
    X = rng.randn(n)
    Y = 2.0 * X + 0.3 * rng.randn(n)
    d = DifferentiableCausalInference(learning_rate=0.01)
    d.set_data(treatment=X, outcome=Y)
    return d, X, Y


class TestCausalParameter:
    def test_create(self):
        p = CausalParameter(name="beta", value=1.0)
        assert p.name == "beta"
        assert p.value == 1.0

    def test_step(self):
        p = CausalParameter(name="beta", value=2.0, gradient=0.5, learning_rate=0.1)
        new_val = p.step()
        assert new_val == pytest.approx(1.95)
        assert p.gradient == 0.0

    def test_multiple_steps(self):
        p = CausalParameter(name="w", value=1.0, learning_rate=0.1)
        p.gradient = 0.5
        p.step()
        assert p.value == 0.95
        p.gradient = -0.3
        p.step()
        assert p.value == 0.98


class TestDifferentiableCausal:
    def test_set_data(self):
        d, _, _ = _make_dci()
        effect = d.treatment_effect
        assert isinstance(effect, float)

    def test_compute_loss(self):
        d, _, _ = _make_dci()
        loss = d.compute_loss()
        assert loss > 0

    def test_compute_gradients(self):
        d, _, _ = _make_dci()
        grads = d.compute_gradients()
        assert isinstance(grads, dict)
        assert "beta" in grads

    def test_optimize(self):
        d, _, _ = _make_dci()
        result = d.optimize(n_iterations=50)
        assert isinstance(result, OptimizationResult)
        assert result.n_iterations > 0
        assert result.final_loss > 0

    def test_loss_decreases(self):
        d, _, _ = _make_dci()
        initial_loss = d.compute_loss()
        result = d.optimize(n_iterations=200)
        assert result.final_loss < initial_loss

    def test_treatment_effect_after_optimization(self):
        d, _, _ = _make_dci()
        d.optimize(n_iterations=500)
        effect = d.treatment_effect
        assert abs(effect) > 0.5
        assert abs(effect) < 5.0

    def test_loss_history(self):
        d, _, _ = _make_dci()
        d.optimize(n_iterations=30)
        history = d.loss_history
        assert len(history) > 0

    def test_statistics(self):
        d, _, _ = _make_dci()
        d.optimize(n_iterations=20)
        stats = d.statistics()
        assert "learning_rate" in stats
        assert "n_params" in stats

    def test_predict(self):
        d, X, _ = _make_dci()
        d.optimize(n_iterations=100)
        pred = d.predict(X)
        assert pred.shape == (200,)

    def test_without_confounders(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100)
        Y = 3.0 * X + 0.2 * rng.randn(100)
        d = DifferentiableCausalInference(learning_rate=0.01)
        d.set_data(treatment=X, outcome=Y)
        result = d.optimize(n_iterations=200)
        assert result.final_loss >= 0
        assert result.final_loss < 10.0  # should reduce


class TestNOTEARSIntegration:
    def test_notears_to_differentiable_bridge(self):
        rng = np.random.RandomState(42)
        n = 200
        A = rng.randn(n)
        B = 0.7 * A + 0.3 * rng.randn(n)
        X_data = np.column_stack([A, B])

        nt = NOTEARSDiscoverer(lambda1=0.05, max_iter=200, threshold=0.3)
        skel = nt.discover(X_data, ["A", "B"])
        assert skel.adj_matrix.shape == (2, 2)

        dci = DifferentiableCausalInference(learning_rate=0.01)
        # A causes Y with confounder B
        Y = 2.0 * A + 0.5 * B + 0.2 * rng.randn(n)
        dci.set_data(treatment=A, outcome=Y)
        dci.optimize(n_iterations=200)
        effect = dci.treatment_effect
        assert isinstance(effect, float)

    def test_notears_single_predictor(self):
        rng = np.random.RandomState(42)
        n = 200
        X1 = rng.randn(n)
        X2 = 0.6 * X1 + 0.3 * rng.randn(n)  # causally connected
        Y = 2.5 * X1 + 0.2 * rng.randn(n)
        X = np.column_stack([X1, X2])

        nt = NOTEARSDiscoverer(lambda1=0.05, max_iter=200, threshold=0.3)
        skel = nt.discover(X, ["A", "B"])
        assert np.sum(skel.adj_matrix) > 0

        dci = DifferentiableCausalInference(learning_rate=0.01)
        dci.set_data(treatment=X1, outcome=Y)
        dci.optimize(n_iterations=200)
        effect = dci.treatment_effect
        assert abs(effect) > 0.5

    def test_chain_to_dci(self):
        rng = np.random.RandomState(42)
        n = 200
        A = rng.randn(n)
        B = 0.7 * A + 0.3 * rng.randn(n)
        C = 0.5 * B + 0.3 * rng.randn(n)
        X = np.column_stack([A, B])

        nt = NOTEARSDiscoverer(lambda1=0.05, max_iter=150, threshold=0.3)
        skel = nt.discover(X, ["A", "B"])
        assert np.sum(skel.adj_matrix) > 0

        dci = DifferentiableCausalInference(learning_rate=0.01)
        dci.set_data(treatment=B, outcome=C)
        result = dci.optimize(n_iterations=300)
        effect = dci.treatment_effect
        assert abs(effect) > 0.1
        assert result.final_loss < 10.0


class TestOptimizationResult:
    def test_result_fields(self):
        r = OptimizationResult(
            n_iterations=10,
            initial_loss=1.0,
            final_loss=0.5,
            converged=True,
        )
        assert r.converged
        assert r.final_loss < r.initial_loss
