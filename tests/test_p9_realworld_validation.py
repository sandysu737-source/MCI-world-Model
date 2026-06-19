"""P9 波级集成测试 — 真实世界验证与可信增强
==========================================

P9 "归真": 可信因果增强 + 医疗SDK验证 + 法律合规 + 工程安全
"""

from __future__ import annotations

from mci_world_model.sdk import (
    AuditableCausalReasoning,
    CausalTrustEnhancement,
    EngineeringSafetySDK,
    LegalComplianceSDK,
    MedicalCausalSDK,
    TrustGrade,
    ValidationMethod,
)


class TestP9TrustFramework:
    """P9 可信增强框架测试。"""

    def test_causal_trust_enhancement(self):
        cte = CausalTrustEnhancement()
        report = cte.get_trust_report()
        assert "n_claims" in report

    def test_trust_grades_complete(self):
        assert len(TrustGrade) == 5

    def test_validation_methods_complete(self):
        assert len(ValidationMethod) == 5

    def test_full_trust_workflow(self):
        """完整的信任工作流：注册→验证→证书→验证证书。"""
        cte = CausalTrustEnhancement()
        cte.register_claim("c1", "Drug A causes recovery", {"clinical_trial": "phase3"})
        cte.validate_claim("c1", ValidationMethod.EMPIRICAL)
        cte.validate_claim("c1", ValidationMethod.STATISTICAL)
        cte.validate_claim("c1", ValidationMethod.PEER_REVIEW)
        cert = cte.issue_certificate("c1")
        assert cert["status"] == "issued"
        verify = cte.verify_certificate(cert["certificate_id"])
        assert verify["is_valid"] is True


class TestP9DomainSDKs:
    """P9 领域 SDK 测试。"""

    def test_medical_causal_sdk(self):
        mcs = MedicalCausalSDK()
        assert mcs is not None

    def test_legal_compliance_sdk(self):
        lcs = LegalComplianceSDK()
        assert lcs is not None

    def test_engineering_safety_sdk(self):
        ess = EngineeringSafetySDK()
        assert ess is not None

    def test_auditable_causal(self):
        ac = AuditableCausalReasoning()
        assert ac is not None


class TestP9Integration:
    """P9 集成测试。"""

    def test_p9_kpi_comprehensive(self):
        """P9 综合 KPI。"""
        assert len(TrustGrade) == 5
        assert len(ValidationMethod) >= 5
        cte = CausalTrustEnhancement()
        assert cte.get_trust_report()["n_claims"] == 0
