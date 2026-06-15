"""tests/test_metacognition_v2.py — MetacognitionV2 测试"""

from __future__ import annotations

import pytest

from mci_world_model.sdk._metacognition_v2 import (
    MetacognitionState,
    MetacognitionV2,
)


@pytest.fixture
def meta():
    return MetacognitionV2(uncertainty_threshold=0.7, confidence_floor=0.3)


class TestMetacognitionState:
    def test_creation(self):
        state = MetacognitionState(confidence=0.8, uncertainty=0.2)
        assert state.confidence == 0.8
        assert state.uncertainty == 0.2

    def test_defaults(self):
        state = MetacognitionState()
        assert state.self_awareness_level == 0
        assert state.capability_boundary == []


class TestMonitor:
    def test_high_confidence(self, meta):
        state = meta.monitor(prediction_confidence=0.9, prediction_error=0.1)
        assert state.confidence == 0.9
        assert state.uncertainty < 0.3

    def test_low_confidence(self, meta):
        state = meta.monitor(prediction_confidence=0.2, prediction_error=0.5)
        assert state.uncertainty > 0.5

    def test_evidence_reduces_load(self, meta):
        state_no_ev = meta.monitor(prediction_confidence=0.5, n_evidence=0)
        state_many_ev = meta.monitor(prediction_confidence=0.5, n_evidence=10)
        assert state_many_ev.cognitive_load <= state_no_ev.cognitive_load

    def test_awareness_levels(self, meta):
        state = meta.monitor(prediction_confidence=0.95, prediction_error=0.01)
        assert state.self_awareness_level >= 3
        state2 = meta.monitor(prediction_confidence=0.05, prediction_error=2.0)
        assert state2.self_awareness_level <= 2


class TestDiagnose:
    def test_no_data(self, meta):
        result = meta.diagnose()
        assert result["diagnosis"] == "no_data"

    def test_healthy(self, meta):
        meta.monitor(prediction_confidence=0.9, prediction_error=0.1)
        result = meta.diagnose()
        assert result["bottleneck"] == "none"

    def test_bottleneck_detected(self, meta):
        meta.monitor(prediction_confidence=0.1, prediction_error=1.0)
        result = meta.diagnose()
        assert result["bottleneck"] in ("high_uncertainty", "low_confidence", "cognitive_overload")

    def test_trend(self, meta):
        meta.monitor(prediction_confidence=0.5)
        meta.monitor(prediction_confidence=0.7)
        meta.monitor(prediction_confidence=0.9)
        result = meta.diagnose()
        assert result["trend"] == "improving"


class TestAssessCapability:
    def test_no_data(self, meta):
        result = meta.assess_capability()
        assert result["capability_level"] == "unknown"
        assert result["safe_to_proceed"] is False

    def test_high_capability(self, meta):
        for _ in range(5):
            meta.monitor(prediction_confidence=0.9, prediction_error=0.05)
        result = meta.assess_capability()
        assert result["capability_level"] == "high"
        assert result["safe_to_proceed"] is True

    def test_low_capability(self, meta):
        for _ in range(5):
            meta.monitor(prediction_confidence=0.2, prediction_error=1.0)
        result = meta.assess_capability()
        assert result["capability_level"] == "low"
        assert result["safe_to_proceed"] is False


class TestAdjust:
    def test_valid_strategy(self, meta):
        result = meta.adjust("increase_evidence")
        assert result["adjusted"] is True

    def test_invalid_strategy(self, meta):
        result = meta.adjust("nonexistent")
        assert result["adjusted"] is False

    def test_all_strategies(self, meta):
        for strategy in ["increase_evidence", "reduce_complexity", "switch_method", "request_help", "fallback_safe"]:
            result = meta.adjust(strategy)
            assert result["adjusted"] is True


class TestMisc:
    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            MetacognitionV2(uncertainty_threshold=0.0)
        with pytest.raises(ValueError):
            MetacognitionV2(uncertainty_threshold=1.0)

    def test_statistics(self, meta):
        meta.monitor(prediction_confidence=0.8)
        stats = meta.statistics()
        assert stats["state_history_count"] == 1

    def test_current_state(self, meta):
        meta.monitor(prediction_confidence=0.8)
        assert meta.current_state is not None
        assert meta.current_state.confidence == 0.8
