"""
tests/test_p7_industry_sdks.py — P7 行业 SDK 端到端验证
======================================================

覆盖:
    - MedicalCausalSDK: 证据添加/诊断/审计/边界
    - LegalComplianceSDK: 证据可靠性/证明标准/偏差检测
    - EngineeringSafetySDK: 安全裕度/FMEA/冗余/安全评估
"""

from __future__ import annotations

import pytest

from mci_world_model.sdk._engineering_safety_sdk import (
    EngineeringSafetySDK,
    FMEAItem,
    SafetyParameter,
)
from mci_world_model.sdk._legal_compliance_sdk import (
    LegalComplianceSDK,
    LegalEvidence,
)
from mci_world_model.sdk._medical_causal_sdk import (
    CausalDiagnosis,
    ClinicalEvidence,
    MedicalCausalSDK,
)

# =============================================================================
# MedicalCausalSDK
# =============================================================================


class TestMedicalCausalSDK:
    @pytest.fixture
    def sdk(self):
        return MedicalCausalSDK(patient_id="P001", strict_mode=True)

    def test_init_defaults(self, sdk):
        assert sdk.patient_id == "P001"
        assert sdk.evidence_count == 0

    def test_add_evidence(self, sdk):
        e = ClinicalEvidence(evidence_id="e1", evidence_type="lab_result", confidence=0.9)
        sdk.add_evidence(e)
        assert sdk.evidence_count == 1

    def test_diagnose_conclusive(self, sdk):
        """充足证据 → 确定性诊断。"""
        for i in range(5):
            sdk.add_evidence(ClinicalEvidence(f"e{i}", "lab_result", "drug_X causes symptom_Y", 0.95))
        result = sdk.diagnose("drug_X", "symptom_Y", prior_strength=0.9)
        assert isinstance(result, CausalDiagnosis)
        assert result.is_conclusive
        assert result.confidence > 0.7

    def test_diagnose_insufficient_evidence(self, sdk):
        """证据不足 → 非确定性诊断。"""
        sdk.add_evidence(ClinicalEvidence("e1", "lab_result", "weak signal", 0.7))
        result = sdk.diagnose("drug_X", "symptom_Y")
        assert not result.is_conclusive
        assert result.confidence == 0.0

    def test_diagnose_low_confidence(self, sdk):
        """低置信度证据 → 非确定性。"""
        sdk.add_evidence(ClinicalEvidence("e1", "obs", "x", 0.4))
        sdk.add_evidence(ClinicalEvidence("e2", "obs", "y", 0.3))
        result = sdk.diagnose("X", "Y")
        assert not result.is_conclusive

    def test_audit_log(self, sdk):
        sdk.add_evidence(ClinicalEvidence("e1"))
        sdk.add_evidence(ClinicalEvidence("e2"))
        log = sdk.get_audit_log()
        assert len(log) == 2
        assert log[0]["action"] == "add_evidence"

    def test_clear_evidence(self, sdk):
        sdk.add_evidence(ClinicalEvidence("e1"))
        sdk.clear_evidence()
        assert sdk.evidence_count == 0

    def test_statistics(self, sdk):
        sdk.add_evidence(ClinicalEvidence("e1", confidence=0.9))
        sdk.add_evidence(ClinicalEvidence("e2", confidence=0.85))
        sdk.diagnose("A", "B")
        stats = sdk.statistics()
        assert stats["evidence_count"] == 2
        assert stats["diagnosis_count"] == 1

    def test_non_strict_mode(self):
        """非严格模式仍尝试诊断。"""
        sdk = MedicalCausalSDK(strict_mode=False)
        sdk.add_evidence(ClinicalEvidence("e1"))
        result = sdk.diagnose("X", "Y")
        assert result.cause == "X"

    def test_relevant_evidence_filtering(self):
        """相关证据过滤。"""
        sdk = MedicalCausalSDK()
        sdk.add_evidence(ClinicalEvidence("e1", description="drug_A and symptom_B"))
        sdk.add_evidence(ClinicalEvidence("e2", description="unrelated"))
        sdk.add_evidence(ClinicalEvidence("e3", description="drug_A treatment"))
        result = sdk.diagnose("drug_A", "symptom_B")
        assert len(result.evidence_ids) >= 2


# =============================================================================
# LegalComplianceSDK
# =============================================================================


class TestLegalComplianceSDK:
    @pytest.fixture
    def sdk(self):
        return LegalComplianceSDK(jurisdiction="CN", standard="preponderance")

    def test_init(self, sdk):
        assert sdk.jurisdiction == "CN"
        assert sdk.standard == "preponderance"
        assert sdk.threshold == 0.51

    def test_invalid_standard(self):
        with pytest.raises(ValueError):
            LegalComplianceSDK(standard="invalid")

    def test_standards_have_increasing_thresholds(self):
        """证明标准阈值递增。"""
        t1 = LegalComplianceSDK(standard="preponderance").threshold
        t2 = LegalComplianceSDK(standard="clear_convincing").threshold
        t3 = LegalComplianceSDK(standard="beyond_reasonable_doubt").threshold
        assert t1 < t2 < t3

    def test_no_evidence_returns_empty(self, sdk):
        result = sdk.reason("A", "B")
        assert not result.legal_standard_met
        assert "no_evidence" in result.bias_flags

    def test_reason_met_standard(self, sdk):
        """充分可靠证据 → 满足标准。"""
        sdk.add_evidence(LegalEvidence("doc1", reliability=0.9))
        sdk.add_evidence(LegalEvidence("doc2", reliability=0.7))
        sdk.add_evidence(LegalEvidence("doc3", reliability=0.8))
        result = sdk.reason("action_A", "harm_B", prior_strength=0.8)
        assert result.legal_standard_met
        assert result.causal_link_strength > 0.51

    def test_bias_detection_uniform(self, sdk):
        """过一致性证据 → 偏差标记。"""
        sdk.add_evidence(LegalEvidence("d1", reliability=0.99))
        sdk.add_evidence(LegalEvidence("d2", reliability=0.99))
        sdk.add_evidence(LegalEvidence("d3", reliability=0.99))
        result = sdk.reason("A", "B")
        assert "uniform_reliability_suspicion" in result.bias_flags or "excessive_confidence" in result.bias_flags

    def test_audit_trail(self, sdk):
        sdk.add_evidence(LegalEvidence("d1"))
        sdk.reason("A", "B")
        trail = sdk.get_audit_trail()
        assert len(trail) >= 2

    def test_statistics(self, sdk):
        sdk.add_evidence(LegalEvidence("d1", reliability=0.8))
        sdk.reason("A", "B")
        stats = sdk.statistics()
        assert stats["jurisdiction"] == "CN"
        assert stats["evidence_count"] == 1

    def test_single_evidence_insufficient(self, sdk):
        """单条低可靠性证据 → 不满足。"""
        sdk.add_evidence(LegalEvidence("d1", reliability=0.3))
        result = sdk.reason("A", "B", prior_strength=0.5)
        assert not result.legal_standard_met

    def test_beyond_reasonable_doubt(self):
        """刑事标准需要 0.95。"""
        sdk = LegalComplianceSDK(standard="beyond_reasonable_doubt")
        sdk.add_evidence(LegalEvidence("d1", reliability=0.99))
        sdk.add_evidence(LegalEvidence("d2", reliability=0.99))
        # 即使高可靠性，prior=0.5 时仍不达标
        result = sdk.reason("A", "B", prior_strength=0.5)
        assert not result.legal_standard_met  # ~0.86 < 0.95


# =============================================================================
# EngineeringSafetySDK
# =============================================================================


class TestEngineeringSafetySDK:
    @pytest.fixture
    def sdk(self):
        return EngineeringSafetySDK(system_name="Reactor-1", redundancy_required=True)

    def test_init(self, sdk):
        assert sdk.system_name == "Reactor-1"
        assert sdk.parameter_count == 0

    def test_add_parameter(self, sdk):
        p = SafetyParameter("temp", design_value=80.0, limit_value=120.0)
        sdk.add_parameter(p)
        assert sdk.parameter_count == 1
        # 安全裕度 = (120-80)/120 = 0.333 > 0.2
        assert p.safety_margin > 0.2

    def test_margin_violation(self, sdk):
        """裕度不足 → 不安全。"""
        p = SafetyParameter("pressure", design_value=115.0, limit_value=120.0)
        sdk.add_parameter(p)
        assert p.safety_margin < 0.2
        result = sdk.analyze("high_pressure", "failure")
        assert not result.margin_sufficient
        assert result.safety_assessment == "unsafe"
        assert result.causal_confidence < 0.5

    def test_fmea_integration(self, sdk):
        sdk.add_parameter(SafetyParameter("temp", 80, 120))
        sdk.add_fmea(FMEAItem("valve_stuck", severity=8, occurrence=5, detection=4))
        result = sdk.analyze("high_temp", "failure")
        assert result.fmea_rpn_max == 160  # 8*5*4

    def test_critical_fmea(self, sdk):
        """高 RPN → 条件性安全。"""
        sdk.add_parameter(SafetyParameter("temp", 80, 120))
        sdk.add_fmea(FMEAItem("explosion", severity=10, occurrence=5, detection=5))
        result = sdk.analyze("failure", "explosion")
        assert result.fmea_rpn_max == 250  # >200
        assert result.safety_assessment == "conditional"

    def test_redundancy_required(self, sdk):
        """无冗余 → 条件性。"""
        sdk.add_parameter(SafetyParameter("temp", 80, 120))
        result = sdk.analyze("A", "B")
        assert not result.redundancy_ok
        assert result.safety_assessment == "conditional"

    def test_redundancy_satisfied(self, sdk):
        """有冗余 → 安全。"""
        sdk.add_parameter(SafetyParameter("temp", 80, 120))
        sdk.set_redundancy("critical_path", True)
        result = sdk.analyze("A", "B")
        assert result.redundancy_ok
        assert result.safety_assessment == "safe"

    def test_safe_scenario(self, sdk):
        """完整安全场景。"""
        sdk.add_parameter(SafetyParameter("temp", 80, 120))
        sdk.add_parameter(SafetyParameter("voltage", 220, 400))
        sdk.add_fmea(FMEAItem("minor_leak", severity=3, occurrence=2, detection=3))
        sdk.set_redundancy("path1", True)
        result = sdk.analyze("event", "outcome", causal_evidence_strength=0.9)
        assert result.safety_assessment == "safe"
        assert result.causal_confidence > 0.8

    def test_statistics(self, sdk):
        sdk.add_parameter(SafetyParameter("t", 80, 120))
        sdk.analyze("A", "B")
        stats = sdk.statistics()
        assert stats["parameter_count"] == 1
        assert stats["analysis_count"] == 1
