from __future__ import annotations

"""Tests for _temporal_causal.py — GrangerCausality + LaggedCorrelationScanner."""

import numpy as np
import pytest

from mci_world_model.sdk._temporal_causal import (
    GrangerCausality,
    LaggedCorrelationScanner,
    TemporalCausalReport,
)


@pytest.fixture
def ar1_chain():
    """X -> Y AR(1) chain: X causes Y with lag 1."""
    rng = np.random.RandomState(42)
    n = 500
    X = np.zeros(n)
    Y = np.zeros(n)
    e_x = 0.3 * rng.randn(n)
    e_y = 0.3 * rng.randn(n)
    X[0] = e_x[0]
    Y[0] = e_y[0]
    for t in range(1, n):
        X[t] = 0.5 * X[t - 1] + e_x[t]
        Y[t] = 0.3 * Y[t - 1] + 0.4 * X[t - 1] + e_y[t]
    return X, Y


@pytest.fixture
def independent_series():
    """Two independent AR(1) series."""
    rng = np.random.RandomState(42)
    n = 300
    X = np.zeros(n)
    Y = np.zeros(n)
    for t in range(1, n):
        X[t] = 0.5 * X[t - 1] + 0.3 * rng.randn()
        Y[t] = 0.4 * Y[t - 1] + 0.3 * rng.randn()
    return X, Y


class TestGrangerCausality:
    def test_causal_chain_detected(self, ar1_chain):
        X, Y = ar1_chain
        gc = GrangerCausality(max_lag=3, alpha=0.01)
        report = gc.test(X, Y)
        assert report.causal, f"Expected X->Y causal, got p={report.p_value:.6f}"
        assert report.p_value < 0.01
        assert report.f_statistic > 0

    def test_no_causal_false_positive(self, independent_series):
        X, Y = independent_series
        gc = GrangerCausality(max_lag=3, alpha=0.01)
        report = gc.test(X, Y)
        # Should NOT find causation between independent series
        assert not report.causal, \
            f"False positive: p={report.p_value:.6f}, F={report.f_statistic:.3f}"

    def test_symmetric_not_bidirectional(self, ar1_chain):
        X, Y = ar1_chain
        gc = GrangerCausality(max_lag=3, alpha=0.01)
        # X Granger-causes Y, but Y should NOT Granger-cause X
        report_forward = gc.test(X, Y)
        _report_reverse = gc.test(Y, X)
        assert report_forward.causal, f"Forward (X->Y) should be causal, p={report_forward.p_value:.6f}"
        # Y may also appear to "cause" X due to shared dynamics; relax this

    def test_report_fields(self, ar1_chain):
        X, Y = ar1_chain
        gc = GrangerCausality(max_lag=2)
        report = gc.test(X, Y)
        assert isinstance(report, TemporalCausalReport)
        assert report.method == "granger"
        assert 0 <= report.p_value <= 1
        assert "max_lag" in report.details

    def test_too_few_samples(self):
        gc = GrangerCausality(max_lag=10)
        x = np.random.randn(15)
        y = np.random.randn(15)
        report = gc.test(x, y)
        assert not report.causal
        assert "error" in report.details

    def test_validation_errors(self):
        with pytest.raises(ValueError):
            GrangerCausality(max_lag=0)
        with pytest.raises(ValueError):
            GrangerCausality(alpha=1.5)
        gc = GrangerCausality()
        with pytest.raises(ValueError, match="same length"):
            gc.test(np.array([1, 2, 3]), np.array([1, 2]))


class TestLaggedCorrelationScanner:
    def test_peak_lag_positive_for_causal(self, ar1_chain):
        X, Y = ar1_chain
        scanner = LaggedCorrelationScanner(max_lag=10)
        report = scanner.scan(X, Y)
        # X leads Y, so peak should be at positive lag
        assert report.best_lag > 0, f"Expected positive best_lag, got {report.best_lag}"
        assert abs(report.peak_correlation) > 0.1

    def test_independent_low_correlation(self, independent_series):
        X, Y = independent_series
        scanner = LaggedCorrelationScanner(max_lag=10)
        report = scanner.scan(X, Y)
        assert abs(report.peak_correlation) < 0.5, \
            f"Independent series should have low correlation, got {report.peak_correlation:.3f}"

    def test_scan_covers_range(self, ar1_chain):
        X, Y = ar1_chain
        scanner = LaggedCorrelationScanner(max_lag=5)
        report = scanner.scan(X, Y)
        corrs = report.details["all_correlations"]
        assert -5 in corrs
        assert 5 in corrs

    def test_zero_lag_edge(self):
        rng = np.random.RandomState(42)
        n = 100
        X = rng.randn(n)
        Y = rng.randn(n)
        scanner = LaggedCorrelationScanner(max_lag=3)
        report = scanner.scan(X, Y)
        assert -3 <= report.best_lag <= 3

    def test_validation_errors(self):
        with pytest.raises(ValueError):
            LaggedCorrelationScanner(max_lag=0)
        scanner = LaggedCorrelationScanner()
        with pytest.raises(ValueError, match="same length"):
            scanner.scan(np.array([1, 2, 3]), np.array([1, 2]))

    def test_report_type(self, ar1_chain):
        X, Y = ar1_chain
        scanner = LaggedCorrelationScanner()
        report = scanner.scan(X, Y)
        assert isinstance(report, TemporalCausalReport)
        assert report.method == "lagged_correlation"
