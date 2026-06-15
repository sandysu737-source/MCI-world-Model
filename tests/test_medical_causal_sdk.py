"""tests/test_medical_causal_sdk.py — MedicalCausalSDK 测试"""

from __future__ import annotations

import pytest

from mci_world_model.sdk._medical_causal_sdk import (
    CausalDiagnosis,
    ClinicalEvidence,
    MedicalCausalSDK,
)


@pytest.fixture
def sdk():
    return MedicalCausalSDK(patient_id="P001", strict_mode=True)


@pytest.fixture
def sdk_non_strict():
    return MedicalCausalSDK(patient_id="P002", strict_mode=False)


@pytest.fixture
def good_evidence():
    return [
        ClinicalEvidence(
            evidence_id="e1", evidence_type="lab_result", description="blood test shows drug_X", confidence=0.85
        ),
        ClinicalEvidence(
            evidence_id="e2", evidence_type="observation", description="symptom_Y observed", confidence=0.90
        ),
        ClinicalEvidence(
            evidence_id="e3", evidence_type="vital_sign", description="heart rate elevated", confidence=0.75
        ),
    ]


class TestClinicalEvidence:
    def test_creation(self):
        ev = ClinicalEvidence(evidence_id="e1", confidence=0.8)
        assert ev.evidence_id == "e1"
        assert ev.confidence == 0.8


class TestMedicalCausalSDK:
    def test_add_evidence(self, sdk, good_evidence):
        for ev in good_evidence:
            sdk.add_evidence(ev)
        assert sdk.evidence_count == 3

    def test_diagnose_with_evidence(self, sdk, good_evidence):
        for ev in good_evidence:
            sdk.add_evidence(ev)
        diagnosis = sdk.diagnose(cause="drug_X", effect="symptom_Y")
        assert isinstance(diagnosis, CausalDiagnosis)
        assert diagnosis.cause == "drug_X"
        assert len(diagnosis.evidence_ids) >= 2

    def test_diagnose_no_evidence_strict(self, sdk):
        diagnosis = sdk.diagnose(cause="drug_X", effect="symptom_Y")
        assert diagnosis.is_conclusive is False
        assert diagnosis.confidence == 0.0

    def test_diagnose_no_evidence_non_strict(self, sdk_non_strict):
        diagnosis = sdk_non_strict.diagnose(cause="drug_X", effect="symptom_Y")
        assert diagnosis.is_conclusive is False

    def test_diagnose_insufficient_evidence(self, sdk):
        sdk.add_evidence(ClinicalEvidence(evidence_id="e1", confidence=0.9))
        diagnosis = sdk.diagnose(cause="X", effect="Y")
        assert diagnosis.is_conclusive is False
        assert any("证据不足" in w for w in diagnosis.warnings)

    def test_audit_trail(self, sdk, good_evidence):
        for ev in good_evidence:
            sdk.add_evidence(ev)
        sdk.diagnose(cause="X", effect="Y")
        log = sdk.get_audit_log()
        assert len(log) >= 4  # 3 add + 1 diagnose

    def test_clear_evidence(self, sdk, good_evidence):
        for ev in good_evidence:
            sdk.add_evidence(ev)
        sdk.clear_evidence()
        assert sdk.evidence_count == 0

    def test_statistics(self, sdk, good_evidence):
        for ev in good_evidence:
            sdk.add_evidence(ev)
        sdk.diagnose(cause="X", effect="Y")
        stats = sdk.statistics()
        assert stats["evidence_count"] == 3
        assert stats["diagnosis_count"] == 1
