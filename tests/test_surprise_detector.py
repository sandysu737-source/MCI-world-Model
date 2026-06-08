"""
tests/test_surprise_detector.py — SurpriseDetector 测试
========================================================

覆盖:
    - compute_surprise: 零惊奇/高惊奇/三维度分解
    - detect_anomalies: 批量异常检测
    - running_statistics: 滚动统计
    - adapt_threshold: 自适应阈值
    - reset / repr
    - 与 PendulumState 集成
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._surprise_detector import SurpriseDetector, SurpriseSignal
from mci_world_model.sdk._world_state import PendulumState


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def detector():
    return SurpriseDetector(threshold=0.3)


@pytest.fixture
def state_a():
    return PendulumState(theta=0.5, omega=1.0)


@pytest.fixture
def state_b():
    return PendulumState(theta=0.6, omega=0.8)


@pytest.fixture
def state_far():
    return PendulumState(theta=2.5, omega=-3.0)


# =============================================================================
# TestComputeSurprise
# =============================================================================


class TestComputeSurprise:
    """compute_surprise 惊奇度计算。"""

    def test_zero_surprise_same_state(self, detector, state_a):
        """相同状态 → score ≈ 0。"""
        sig = detector.compute_surprise(state_a, state_a)
        assert sig.score < 0.01
        assert sig.is_anomaly is False

    def test_high_surprise(self, detector, state_a, state_far):
        """远距离状态 → 高 score。"""
        sig = detector.compute_surprise(state_a, state_far)
        assert sig.score > 0.3

    def test_breakdown_keys(self, detector, state_a, state_b):
        """breakdown 包含三个维度。"""
        sig = detector.compute_surprise(state_a, state_b)
        assert "state_distance" in sig.breakdown
        assert "vector_deviation" in sig.breakdown
        assert "direction_error" in sig.breakdown

    def test_score_in_range(self, detector, state_a, state_far):
        sig = detector.compute_surprise(state_a, state_far)
        assert 0.0 <= sig.score <= 1.0

    def test_anomaly_flag(self, detector, state_a, state_far):
        """超过阈值 → is_anomaly=True。"""
        sig = detector.compute_surprise(state_a, state_far)
        if sig.score >= 0.3:
            assert sig.is_anomaly is True

    def test_symmetric(self, detector, state_a, state_b):
        """distance(predicted, actual) == distance(actual, predicted)。"""
        sig_ab = detector.compute_surprise(state_a, state_b)
        detector.reset()
        sig_ba = detector.compute_surprise(state_b, state_a)
        assert abs(sig_ab.score - sig_ba.score) < 0.01

    def test_direction_error_same_direction(self, detector):
        """同方向向量 → direction_error ≈ 0。"""
        s1 = PendulumState(theta=1.0, omega=2.0)
        s2 = PendulumState(theta=2.0, omega=4.0)
        sig = detector.compute_surprise(s1, s2)
        assert sig.breakdown["direction_error"] < 0.05

    def test_threshold_stored(self, detector, state_a, state_b):
        sig = detector.compute_surprise(state_a, state_b)
        assert sig.threshold == 0.3


# =============================================================================
# TestDetectAnomalies
# =============================================================================


class TestDetectAnomalies:
    """detect_anomalies 批量异常检测。"""

    def test_empty_history(self, detector):
        anomalies = detector.detect_anomalies([])
        assert anomalies == []

    def test_filters_by_threshold(self, detector):
        history = [
            (PendulumState(theta=0.1, omega=0.0), PendulumState(theta=0.1, omega=0.0)),  # 零惊奇
            (PendulumState(theta=0.5, omega=1.0), PendulumState(theta=2.5, omega=-3.0)),  # 高惊奇
        ]
        anomalies = detector.detect_anomalies(history)
        assert len(anomalies) >= 1
        for a in anomalies:
            assert a.is_anomaly is True
            assert a.score >= 0.3

    def test_all_normal(self, detector):
        history = [
            (PendulumState(theta=0.1, omega=0.0), PendulumState(theta=0.1, omega=0.0)),
        ]
        anomalies = detector.detect_anomalies(history)
        assert len(anomalies) == 0


# =============================================================================
# TestRunningStatistics
# =============================================================================


class TestRunningStatistics:
    """running_statistics 滚动统计。"""

    def test_empty_history(self, detector):
        stats = detector.running_statistics()
        assert stats["n"] == 0
        assert stats["mean"] == 0.0

    def test_after_computations(self, detector, state_a, state_b, state_far):
        detector.compute_surprise(state_a, state_b)
        detector.compute_surprise(state_a, state_far)
        stats = detector.running_statistics()
        assert stats["n"] == 2
        assert stats["mean"] > 0
        assert stats["max"] >= stats["min"]

    def test_anomaly_rate(self, detector):
        detector.threshold = 0.5
        # 5 次计算：2 次高惊奇，3 次低惊奇
        detector.compute_surprise(
            PendulumState(theta=0.5, omega=1.0),
            PendulumState(theta=2.5, omega=-3.0),
        )  # high
        detector.compute_surprise(
            PendulumState(theta=0.1, omega=0.0),
            PendulumState(theta=0.1, omega=0.0),
        )  # low
        detector.compute_surprise(
            PendulumState(theta=0.1, omega=0.0),
            PendulumState(theta=0.11, omega=0.01),
        )  # low
        stats = detector.running_statistics()
        assert stats["n"] == 3
        assert 0.0 <= stats["anomaly_rate"] <= 1.0


# =============================================================================
# TestAdaptThreshold
# =============================================================================


class TestAdaptThreshold:
    """adapt_threshold 自适应阈值。"""

    def test_insufficient_data(self, detector):
        """数据不足时保持原阈值。"""
        old = detector.threshold
        detector.adapt_threshold()
        assert detector.threshold == old

    def test_adapts_after_data(self, detector, state_a, state_b, state_far):
        detector.compute_surprise(state_a, state_b)
        detector.compute_surprise(state_a, state_far)
        detector.compute_surprise(state_a, state_a)
        new_threshold = detector.adapt_threshold(n_std=2.0)
        assert 0.01 <= new_threshold <= 0.99

    def test_threshold_clamped(self, detector, state_a):
        """阈值始终在 [0.01, 0.99] 内。"""
        for _ in range(10):
            detector.compute_surprise(state_a, state_a)
        new_t = detector.adapt_threshold(n_std=100.0)
        assert new_t <= 0.99


# =============================================================================
# TestMisc
# =============================================================================


class TestMisc:
    """杂项测试。"""

    def test_reset(self, detector, state_a, state_b):
        detector.compute_surprise(state_a, state_b)
        assert detector.n_observations == 1
        detector.reset()
        assert detector.n_observations == 0

    def test_threshold_setter(self, detector):
        detector.threshold = 0.7
        assert detector.threshold == 0.7
        detector.threshold = -0.5  # clamped to 0
        assert detector.threshold == 0.0
        detector.threshold = 2.0  # clamped to 1
        assert detector.threshold == 1.0

    def test_history_property(self, detector, state_a, state_b):
        detector.compute_surprise(state_a, state_b)
        h = detector.history
        assert len(h) == 1
        assert isinstance(h[0], float)

    def test_repr(self, detector):
        r = repr(detector)
        assert "SurpriseDetector" in r
        assert "threshold" in r

    def test_surprise_signal_fields(self, detector, state_a, state_b):
        sig = detector.compute_surprise(state_a, state_b)
        assert hasattr(sig, "score")
        assert hasattr(sig, "predicted")
        assert hasattr(sig, "actual")
        assert hasattr(sig, "breakdown")
        assert hasattr(sig, "is_anomaly")
        assert hasattr(sig, "threshold")
