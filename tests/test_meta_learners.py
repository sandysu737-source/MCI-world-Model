from __future__ import annotations

"""Tests for _meta_learners.py — T-learner and S-learner."""

import numpy as np
import pytest

from mci_world_model.sdk._meta_learners import SLearner, TLearner


@pytest.fixture
def linear_hte_data():
    """X0->Y with heterogeneous treatment effect.

    Ground truth:
      Y = 2.0 + 1.5*X0 + (1.0 + 0.5*X0)*T + noise
    True ATE = mean(1.0 + 0.5*X0) = 1.0 (since X0 ~ N(0,1))
    """
    rng = np.random.RandomState(42)
    n = 500
    X = rng.randn(n, 2)  # X0 informative, X1 noise
    T = rng.binomial(1, 0.5, size=n)
    Y = 2.0 + 1.5 * X[:, 0] + T * (1.0 + 0.5 * X[:, 0]) + 0.3 * rng.randn(n)
    return X, T, Y


class TestTLearner:
    def test_fit_predict_basic(self, linear_hte_data):
        X, T, Y = linear_hte_data
        learner = TLearner()
        learner.fit(X, T, Y)
        cate = learner.predict_cate(X)
        assert cate.shape == (len(X),)
        assert np.isfinite(cate).all()

    def test_ate_within_tolerance(self, linear_hte_data):
        X, T, Y = linear_hte_data
        learner = TLearner()
        learner.fit(X, T, Y)
        ate = learner.estimate_ate(X)
        # True ATE ≈ 1.0; allow wide tolerance due to linear approximation
        assert 0.2 < ate < 1.8, f"ATE={ate:.3f} outside (0.2, 1.8)"

    def test_cate_variance_positive(self, linear_hte_data):
        X, T, Y = linear_hte_data
        learner = TLearner()
        learner.fit(X, T, Y)
        cate = learner.predict_cate(X)
        assert np.std(cate) > 0.0, "CATE should vary across samples"

    def test_cate_ranks_preserved(self, linear_hte_data):
        X, T, Y = linear_hte_data
        learner = TLearner()
        learner.fit(X, T, Y)
        cate = learner.predict_cate(X)
        # Higher X0 should generally have higher CATE
        high_x0 = X[:, 0] > np.median(X[:, 0])
        assert np.mean(cate[high_x0]) > np.mean(cate[~high_x0]), \
            "CATE should be higher for larger X0"

    def test_summary_keys(self, linear_hte_data):
        X, T, Y = linear_hte_data
        learner = TLearner()
        learner.fit(X, T, Y)
        summary = learner.heterogeneous_effect_summary(X)
        for key in ("ate", "cate_mean", "cate_std", "cate_q10", "cate_q90"):
            assert key in summary, f"Missing key: {key}"
        assert summary["cate_q10"] <= summary["cate_q90"]

    def test_not_fitted_raises(self):
        learner = TLearner()
        X = np.random.randn(10, 2)
        with pytest.raises(RuntimeError, match="not fitted"):
            learner.predict_cate(X)

    def test_too_few_samples_raises(self):
        learner = TLearner()
        X = np.random.randn(4, 2)
        T = np.array([0, 0, 0, 1])
        Y = np.random.randn(4)
        with pytest.raises(ValueError, match=">=2 samples"):
            learner.fit(X, T, Y)

    def test_ate_zero_effect(self):
        rng = np.random.RandomState(42)
        n = 200
        X = rng.randn(n, 3)
        T = rng.binomial(1, 0.5, size=n)
        Y = 1.0 + 2.0 * X[:, 0] + 0.3 * rng.randn(n)  # No treatment effect
        learner = TLearner()
        learner.fit(X, T, Y)
        ate = learner.estimate_ate(X)
        assert abs(ate) < 1.0, f"ATE={ate:.3f} should be near zero"


class TestSLearner:
    def test_fit_predict_basic(self, linear_hte_data):
        X, T, Y = linear_hte_data
        learner = SLearner()
        learner.fit(X, T, Y)
        cate = learner.predict_cate(X)
        assert cate.shape == (len(X),)
        assert np.isfinite(cate).all()

    def test_ate_within_tolerance(self, linear_hte_data):
        X, T, Y = linear_hte_data
        learner = SLearner()
        learner.fit(X, T, Y)
        ate = learner.estimate_ate(X)
        assert 0.2 < ate < 1.8, f"ATE={ate:.3f} outside (0.2, 1.8)"

    def test_predict_outcome(self, linear_hte_data):
        X, T, Y = linear_hte_data
        learner = SLearner()
        learner.fit(X, T, Y)
        y_pred = learner.predict_outcome(X, T)
        assert y_pred.shape == Y.shape
        assert np.isfinite(y_pred).all()

    def test_constant_cate_linear(self, linear_hte_data):
        X, T, Y = linear_hte_data
        learner = SLearner()
        learner.fit(X, T, Y)
        cate = learner.predict_cate(X)
        # Linear S-learner: CATE is constant (treatment coefficient)
        assert np.allclose(cate, cate[0]), \
            f"CATE not constant: std={np.std(cate):.6f}"

    def test_summary_keys(self, linear_hte_data):
        X, T, Y = linear_hte_data
        learner = SLearner()
        learner.fit(X, T, Y)
        summary = learner.heterogeneous_effect_summary(X)
        for key in ("ate", "cate_mean", "cate_std", "cate_q10", "cate_q90"):
            assert key in summary, f"Missing key: {key}"

    def test_not_fitted_raises(self):
        learner = SLearner()
        X = np.random.randn(10, 2)
        T = np.random.binomial(1, 0.5, size=10)
        with pytest.raises(RuntimeError, match="not fitted"):
            learner.predict_cate(X)
        with pytest.raises(RuntimeError, match="not fitted"):
            learner.predict_outcome(X, T)

    def test_1d_input(self):
        rng = np.random.RandomState(42)
        n = 100
        X = rng.randn(n)
        T = rng.binomial(1, 0.5, size=n)
        Y = 2.0 * X + 1.0 * T + 0.2 * rng.randn(n)
        learner = SLearner()
        learner.fit(X, T, Y)
        cate = learner.predict_cate(X)
        assert cate.shape == (n,)
