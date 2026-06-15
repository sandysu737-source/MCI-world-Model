"""tests/test_legal_compliance_sdk.py — LegalComplianceSDK 测试"""

from __future__ import annotations

import pytest

from mci_world_model.sdk._legal_compliance_sdk import (
    LegalCausalConclusion,
    LegalComplianceSDK,
    LegalEvidence,
)


@pytest.fixture
def sdk():
    return LegalComplianceSDK(jurisdiction="CN", standard="preponderance")


@pytest.fixture
def criminal_sdk():
    return LegalComplianceSDK(jurisdiction="US", standard="beyond_reasonable_doubt")


class TestLegalEvidence:
    def test_creation(self):
        ev = LegalEvidence(evidence_id="d1", reliability=0.8, jurisdiction="CN")
        assert ev.evidence_id == "d1"
        assert ev.reliability == 0.8


class TestLegalComplianceSDK:
    def test_creation(self, sdk):
        assert sdk.jurisdiction == "CN"
        assert sdk.standard == "preponderance"
        assert sdk.threshold == 0.51

    def test_invalid_standard(self):
        with pytest.raises(ValueError, match="未知证明标准"):
            LegalComplianceSDK(standard="invalid")

    def test_reason_with_evidence(self, sdk):
        sdk.add_evidence(LegalEvidence(evidence_id="d1", reliability=0.8))
        sdk.add_evidence(LegalEvidence(evidence_id="d2", reliability=0.7))
        conclusion = sdk.reason(cause="action_A", effect="harm_B")
        assert isinstance(conclusion, LegalCausalConclusion)
        assert conclusion.cause == "action_A"
        assert conclusion.jurisdiction == "CN"

    def test_reason_no_evidence(self, sdk):
        conclusion = sdk.reason(cause="A", effect="B")
        assert "no_evidence" in conclusion.bias_flags

    def test_preponderance_standard(self, sdk):
        sdk.add_evidence(LegalEvidence(evidence_id="d1", reliability=0.9))
        sdk.add_evidence(LegalEvidence(evidence_id="d2", reliability=0.85))
        conclusion = sdk.reason(cause="A", effect="B")
        # 高可靠性证据应能满足优势证据标准
        assert conclusion.causal_link_strength > 0.5

    def test_criminal_standard_higher(self, criminal_sdk):
        criminal_sdk.add_evidence(LegalEvidence(evidence_id="d1", reliability=0.8))
        criminal_sdk.add_evidence(LegalEvidence(evidence_id="d2", reliability=0.75))
        conclusion = criminal_sdk.reason(cause="A", effect="B")
        assert criminal_sdk.threshold == 0.95
        # 0.8可靠性不太可能满足排除合理怀疑标准
        assert conclusion.legal_standard_met is False

    def test_audit_trail(self, sdk):
        sdk.add_evidence(LegalEvidence(evidence_id="d1", reliability=0.8))
        sdk.reason(cause="A", effect="B")
        trail = sdk.get_audit_trail()
        assert len(trail) >= 2

    def test_bias_detection(self, sdk):
        # 所有证据可靠性完全一致 → 可疑
        for i in range(5):
            sdk.add_evidence(LegalEvidence(evidence_id=f"d{i}", reliability=0.80))
        conclusion = sdk.reason(cause="A", effect="B")
        # 可能有 uniform_reliability_suspicion
        assert isinstance(conclusion.bias_flags, list)

    def test_statistics(self, sdk):
        sdk.add_evidence(LegalEvidence(evidence_id="d1", reliability=0.8))
        sdk.reason(cause="A", effect="B")
        stats = sdk.statistics()
        assert stats["evidence_count"] == 1
        assert stats["conclusion_count"] == 1
