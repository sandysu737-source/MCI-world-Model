"""P9-P11 增强模块测试 — 可信因果增强/跨域迁移/自主因果意识
==========================================================

覆盖模块:
  P9: CausalTrustEnhancement, TrustCertificate, TrustGrade, ValidationMethod
  P10: CrossDomainCausalTransfer, DomainAdapter, DomainType
  P11: AutonomousCausalConsciousness, CausalSelfModel, ConsciousnessLevel
"""

from __future__ import annotations

from mci_world_model.sdk._causal_consciousness import (
    AutonomousCausalConsciousness,
    CausalSelfModel,
    ConsciousnessLevel,
    SelfModelProperty,
)
from mci_world_model.sdk._causal_trust import (
    CausalTrustEnhancement,
    TrustCertificate,
    TrustGrade,
    ValidationMethod,
)
from mci_world_model.sdk._cross_domain_transfer import (
    CausalKnowledge,
    CrossDomainCausalTransfer,
    DomainAdapter,
    DomainType,
)

# ── P9 可信因果增强 测试 ─────────────────────────────────────


class TestCausalTrustEnhancement:
    """P9 可信因果增强框架测试。"""

    def test_init(self):
        cte = CausalTrustEnhancement()
        report = cte.get_trust_report()
        assert report["n_claims"] == 0
        assert report["n_certificates"] == 0

    def test_register_claim(self):
        cte = CausalTrustEnhancement()
        result = cte.register_claim("claim_1", "Treatment A causes recovery")
        assert result["status"] == "registered"

    def test_validate_claim_empirical(self):
        cte = CausalTrustEnhancement()
        cte.register_claim("claim_1", "Test claim")
        result = cte.validate_claim("claim_1", ValidationMethod.EMPIRICAL)
        assert result["status"] == "validated"
        assert result["grade"] == "basic"

    def test_validate_claim_statistical(self):
        cte = CausalTrustEnhancement()
        cte.register_claim("claim_1", "Test claim")
        cte.validate_claim("claim_1", ValidationMethod.EMPIRICAL)
        result = cte.validate_claim("claim_1", ValidationMethod.STATISTICAL)
        assert result["grade"] == "enhanced"

    def test_validate_claim_formal_proof(self):
        cte = CausalTrustEnhancement()
        cte.register_claim("claim_1", "Test claim")
        result = cte.validate_claim("claim_1", ValidationMethod.FORMAL_PROOF)
        assert result["grade"] == "formal"

    def test_issue_certificate(self):
        cte = CausalTrustEnhancement()
        cte.register_claim("claim_1", "Test claim")
        cte.validate_claim("claim_1", ValidationMethod.EMPIRICAL)
        cte.validate_claim("claim_1", ValidationMethod.STATISTICAL)
        result = cte.issue_certificate("claim_1")
        assert result["status"] == "issued"
        assert "certificate_id" in result
        assert result["grade"] == "enhanced"

    def test_issue_certificate_unverified(self):
        cte = CausalTrustEnhancement()
        cte.register_claim("claim_1", "Test claim")
        result = cte.issue_certificate("claim_1")
        assert result["status"] == "cannot_issue"

    def test_verify_certificate(self):
        cte = CausalTrustEnhancement()
        cte.register_claim("claim_1", "Test claim")
        cte.validate_claim("claim_1", ValidationMethod.EMPIRICAL)
        cert = cte.issue_certificate("claim_1")
        result = cte.verify_certificate(cert["certificate_id"])
        assert result["is_valid"] is True

    def test_formal_proof_check(self):
        cte = CausalTrustEnhancement()
        cte.register_claim("claim_1", "Test claim", {"proof": "derivation_chain"})
        result = cte.formal_proof_check("claim_1")
        assert result["status"] == "proven"
        assert result["grade"] == "formal"

    def test_formal_proof_insufficient(self):
        cte = CausalTrustEnhancement()
        cte.register_claim("claim_1", "Test claim")
        result = cte.formal_proof_check("claim_1")
        assert result["status"] == "insufficient_formal_evidence"

    def test_trust_report(self):
        cte = CausalTrustEnhancement()
        cte.register_claim("c1", "Claim 1")
        cte.register_claim("c2", "Claim 2")
        cte.validate_claim("c1", ValidationMethod.STATISTICAL)
        report = cte.get_trust_report()
        assert report["n_claims"] == 2


class TestTrustCertificate:
    """信任证书数据类测试。"""

    def test_certificate_validity(self):
        cert = TrustCertificate(
            certificate_id="CTC-01",
            claim_id="c1",
            grade=TrustGrade.ENHANCED,
        )
        assert cert.is_valid is True

    def test_certificate_unverified_invalid(self):
        cert = TrustCertificate(
            certificate_id="CTC-02",
            claim_id="c2",
            grade=TrustGrade.UNVERIFIED,
        )
        assert cert.is_valid is False

    def test_trust_grades(self):
        assert len(TrustGrade) == 5
        assert TrustGrade.FORMAL.value == "formal"

    def test_validation_methods(self):
        assert len(ValidationMethod) == 5


# ── P10 跨域因果迁移 测试 ─────────────────────────────────────


class TestCrossDomainTransfer:
    """P10 跨域因果迁移测试。"""

    def test_init(self):
        cdct = CrossDomainCausalTransfer()
        report = cdct.get_transfer_report()
        assert report["n_knowledge"] == 0

    def test_register_knowledge(self):
        cdct = CrossDomainCausalTransfer()
        k = CausalKnowledge(
            knowledge_id="med_1",
            source_domain=DomainType.MEDICAL,
            confidence=0.9,
            n_observations=100,
        )
        result = cdct.register_knowledge(k)
        assert result["status"] == "registered"

    def test_create_adapter(self):
        cdct = CrossDomainCausalTransfer()
        result = cdct.create_adapter(DomainType.MEDICAL, DomainType.FINANCE)
        assert result["status"] == "adapter_created"
        assert result["compatibility"] > 0

    def test_transfer_success(self):
        cdct = CrossDomainCausalTransfer()
        k = CausalKnowledge(
            knowledge_id="med_1",
            source_domain=DomainType.MEDICAL,
            confidence=0.8,
            n_observations=50,
        )
        cdct.register_knowledge(k)
        result = cdct.transfer("med_1", DomainType.FINANCE)
        assert result["status"] == "transferred"
        assert result["fidelity"] > 0

    def test_transfer_verification(self):
        cdct = CrossDomainCausalTransfer()
        k = CausalKnowledge(
            knowledge_id="med_1",
            source_domain=DomainType.MEDICAL,
            confidence=0.8,
        )
        cdct.register_knowledge(k)
        transfer_result = cdct.transfer("med_1", DomainType.FINANCE)
        verify_result = cdct.verify_transfer(transfer_result["transfer_id"])
        assert verify_result["verified"] is True

    def test_detect_emergence(self):
        cdct = CrossDomainCausalTransfer()
        # 注册原始知识和多条迁移知识
        k1 = CausalKnowledge(knowledge_id="med_1", source_domain=DomainType.MEDICAL, confidence=0.8)
        cdct.register_knowledge(k1)
        cdct.transfer("med_1", DomainType.FINANCE)
        cdct.transfer("med_1", DomainType.ENGINEERING)
        # 涌现检测在目标域
        result = cdct.detect_emergence(DomainType.FINANCE)
        assert result["transferred_knowledge"] >= 1

    def test_domain_types(self):
        assert len(DomainType) == 5

    def test_domain_adapter_same_domain(self):
        adapter = DomainAdapter(
            source_domain=DomainType.MEDICAL,
            target_domain=DomainType.MEDICAL,
        )
        assert adapter.compute_compatibility() == 1.0


# ── P11 自主因果意识 测试 ─────────────────────────────────────


class TestAutonomousCausalConsciousness:
    """P11 自主因果意识测试。"""

    def test_init(self):
        acc = AutonomousCausalConsciousness()
        assert acc.level == ConsciousnessLevel.REACTIVE
        assert acc.self_model.self_awareness_score == 0.0

    def test_evolve_consciousness(self):
        acc = AutonomousCausalConsciousness()
        result = acc.evolve_consciousness()
        assert result["status"] == "evolved"
        assert result["to"] == "deliberative"

    def test_evolve_to_target(self):
        acc = AutonomousCausalConsciousness()
        result = acc.evolve_consciousness(ConsciousnessLevel.REFLECTIVE)
        assert result["to"] == "reflective"

    def test_evolve_full_path(self):
        acc = AutonomousCausalConsciousness()
        levels = [
            ConsciousnessLevel.DELIBERATIVE,
            ConsciousnessLevel.REFLECTIVE,
            ConsciousnessLevel.AUTONOMOUS,
            ConsciousnessLevel.TRANSCENDENT,
        ]
        for level in levels:
            acc.evolve_consciousness(level)
        assert acc.level == ConsciousnessLevel.TRANSCENDENT
        assert acc.self_model.self_awareness_score == 1.0

    def test_build_self_model(self):
        acc = AutonomousCausalConsciousness()
        acc.evolve_consciousness()
        result = acc.build_self_model({
            SelfModelProperty.CAUSAL_IDENTITY.value: "medical_causal_agent",
            SelfModelProperty.REASONING_STYLE.value: "bayesian",
        })
        assert result["n_properties"] == 2
        assert result["self_awareness"] > 0

    def test_reflect_on_reasoning(self):
        acc = AutonomousCausalConsciousness()
        acc.evolve_consciousness(ConsciousnessLevel.REFLECTIVE)
        result = acc.reflect_on_reasoning({
            "evidence": "clinical_trial",
            "counterfactual": "placebo_comparison",
            "confidence": 0.85,
        })
        assert result["status"] == "reflected"
        assert result["quality_score"] > 0.5

    def test_reflect_reactive_fails(self):
        acc = AutonomousCausalConsciousness()
        result = acc.reflect_on_reasoning({"step": 1})
        assert result["status"] == "cannot_reflect"

    def test_establish_civilization(self):
        acc = AutonomousCausalConsciousness()
        result = acc.establish_civilization(n_citizens=5)
        assert result["status"] == "civilization_established"
        assert result["n_citizens"] == 5

    def test_consciousness_report(self):
        acc = AutonomousCausalConsciousness()
        acc.evolve_consciousness()
        acc.build_self_model({SelfModelProperty.VALUE_SYSTEM.value: "causal_truth"})
        report = acc.get_consciousness_report()
        assert report["level"] == "deliberative"
        assert "godel_note" in report

    def test_consciousness_levels(self):
        assert len(ConsciousnessLevel) == 5

    def test_self_model_godel(self):
        model = CausalSelfModel(identity="test")
        assert "GÖDEL" in model.godel_note

    def test_self_model_awareness(self):
        model = CausalSelfModel(identity="test", self_awareness_score=0.6)
        assert model.is_self_aware is True
