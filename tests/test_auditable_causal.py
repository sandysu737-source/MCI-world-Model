"""tests/test_auditable_causal.py — AuditableCausalReasoning 测试"""

from __future__ import annotations

import pytest

from mci_world_model.sdk._auditable_causal import (
    AuditableCausalReasoning,
    AuditStep,
    AuditTrail,
)


@pytest.fixture
def acr():
    return AuditableCausalReasoning(reasoner_id="test_acr")


class TestAuditStep:
    def test_creation(self):
        step = AuditStep(step_id="s1", step_type="hypothesis", description="X causes Y")
        assert step.step_id == "s1"
        assert step.step_type == "hypothesis"


class TestAuditTrail:
    def test_add_step(self):
        trail = AuditTrail()
        step = AuditStep(step_id="s1", step_type="evidence", description="obs")
        trail.add_step(step)
        assert trail.step_count == 1

    def test_complete(self):
        trail = AuditTrail()
        trail.complete("X causes Y")
        assert trail.is_complete is True
        assert trail.conclusion == "X causes Y"
        assert trail.duration >= 0


class TestAuditableCausalReasoning:
    def test_begin(self, acr):
        trail = acr.begin("X causes Y")
        assert isinstance(trail, AuditTrail)
        assert trail.step_count == 1
        assert acr.trail_count == 1

    def test_full_workflow(self, acr):
        trail = acr.begin("X causes Y")
        acr.add_evidence_step(trail, "observed correlation", confidence=0.7)
        acr.add_inference_step(trail, "backdoor adjustment", confidence=0.8)
        acr.add_validation_step(trail, "conservation_check", passed=True)
        acr.conclude(trail, "X causes Y with strength 0.8")
        assert trail.step_count == 5
        assert trail.is_complete is True

    def test_verify_valid_trail(self, acr):
        trail = acr.begin("X causes Y")
        acr.add_evidence_step(trail, "evidence", confidence=0.7)
        acr.add_inference_step(trail, "method", confidence=0.8)
        acr.conclude(trail, "conclusion")
        report = acr.verify_trail(trail)
        assert report["is_valid"] is True

    def test_verify_missing_evidence(self, acr):
        trail = acr.begin("X causes Y")
        acr.add_inference_step(trail, "method", confidence=0.8)
        acr.conclude(trail, "conclusion")
        report = acr.verify_trail(trail)
        assert report["is_valid"] is False
        assert "missing_evidence" in report["issues"]

    def test_verify_missing_inference(self, acr):
        trail = acr.begin("X causes Y")
        acr.add_evidence_step(trail, "evidence", confidence=0.7)
        acr.conclude(trail, "conclusion")
        report = acr.verify_trail(trail)
        assert "missing_inference" in report["issues"]

    def test_get_trail(self, acr):
        trail = acr.begin("X causes Y")
        found = acr.get_trail(trail.trail_id)
        assert found is trail
        assert acr.get_trail("nonexistent") is None

    def test_statistics(self, acr):
        trail = acr.begin("X causes Y")
        acr.add_evidence_step(trail, "evidence")
        acr.conclude(trail, "conclusion")
        stats = acr.statistics()
        assert stats["trail_count"] == 1
        assert stats["completed_trails"] == 1
        assert stats["total_steps"] == 3
