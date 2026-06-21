"""
tests/test_compliance_engine.py — ComplianceRuleEngine 全面测试
==============================================================

覆盖:
    - 医疗领域规则 (证据充分性/干预安全/患者同意)
    - 法律领域规则 (管辖区/审计轨迹/偏差检测)
    - 工程领域规则 (安全裕度/冗余检查/失效模式)
    - ComplianceRuleEngine (注册/执行/审计/聚合)
    - 边界情况 (空上下文/异常处理/自定义规则)
"""

from __future__ import annotations

import pytest

from mci_world_model.sdk._compliance_engine import (
    ComplianceCheckResult,
    ComplianceLevel,
    ComplianceReport,
    ComplianceRule,
    ComplianceRuleEngine,
    _AuditTrailRule,
    _BiasDetectionRule,
    _EvidenceSufficiencyRule,
    _FailureModeRule,
    _InterventionSafetyRule,
    _JurisdictionRule,
    _PatientConsentRule,
    _RedundancyCheckRule,
    _SafetyMarginRule,
)


@pytest.fixture
def engine():
    return ComplianceRuleEngine(auto_register_defaults=True)


@pytest.fixture
def medical_context():
    return {
        "evidence": [{"type": "lab"}, {"type": "imaging"}],
        "confidence": 0.85,
        "intervention": {"name": "drug_A"},
        "risk_assessment": {"level": "low"},
        "patient_data": {"consent_given": True, "age": 45},
    }


@pytest.fixture
def legal_context():
    return {
        "audit_trail": [
            {"step": "evidence_collection", "data": "collected"},
        ],
        "evidence": [{"reliability": 0.8}, {"reliability": 0.75}],
        "evidence": [{"reliability": 0.8}, {"reliability": 0.75}],
        "jurisdiction": "CN",
        "conclusion": "action_A caused harm_B",
        "confidence": 0.85,
    }


@pytest.fixture
def engineering_context():
    return {
        "system_params": {
            "temp": {"design": 80, "limit": 120},
            "pressure": {"design": 30, "limit": 50},
        },
        "redundancy": {"critical_path": True},
        "fmea": [
            {"failure_mode": "valve_stuck", "rpn": 150, "mitigated": True},
        ],
    }


class TestMedicalRules:
    def test_evidence_rule_domain(self):
        rule = _EvidenceSufficiencyRule()
        assert rule.domain == "medical"

    def test_evidence_sufficient(self):
        rule = _EvidenceSufficiencyRule()
        ctx = {"evidence": [1, 2, 3], "confidence": 0.9}
        report = rule.check(ctx)
        assert report.level == ComplianceLevel.COMPLIANT

    def test_evidence_insufficient(self):
        rule = _EvidenceSufficiencyRule()
        ctx = {"evidence": [1], "confidence": 0.5}
        report = rule.check(ctx)
        assert report.level == ComplianceLevel.NON_COMPLIANT

    def test_evidence_conditional(self):
        rule = _EvidenceSufficiencyRule()
        ctx = {"evidence": [1, 2], "confidence": 0.5}
        report = rule.check(ctx)
        assert report.level == ComplianceLevel.CONDITIONAL

    def test_intervention_safe(self):
        rule = _InterventionSafetyRule()
        ctx = {"intervention": "drug", "risk_assessment": {"level": "low"}}
        report = rule.check(ctx)
        assert report.level == ComplianceLevel.COMPLIANT

    def test_intervention_high_risk(self):
        rule = _InterventionSafetyRule()
        ctx = {"intervention": "surgery", "risk_assessment": {"level": "high"}}
        report = rule.check(ctx)
        assert report.level == ComplianceLevel.NON_COMPLIANT

    def test_intervention_medium_risk(self):
        rule = _InterventionSafetyRule()
        ctx = {"intervention": "therapy", "risk_assessment": {"level": "medium"}}
        report = rule.check(ctx)
        assert report.level == ComplianceLevel.CONDITIONAL

    def test_intervention_no_risk(self):
        rule = _InterventionSafetyRule()
        ctx = {"intervention": "drug"}
        report = rule.check(ctx)
        assert report.level == ComplianceLevel.NON_COMPLIANT

    def test_patient_consent_granted(self):
        rule = _PatientConsentRule()
        ctx = {"patient_data": {"consent_given": True}}
        report = rule.check(ctx)
        assert report.level == ComplianceLevel.COMPLIANT

    def test_patient_consent_denied(self):
        rule = _PatientConsentRule()
        ctx = {"patient_data": {"consent_given": False}}
        report = rule.check(ctx)
        assert report.level == ComplianceLevel.NON_COMPLIANT


class TestLegalRules:
    def test_jurisdiction_present(self):
        rule = _JurisdictionRule()
        ctx = {"jurisdiction": "CN"}
        report = rule.check(ctx)
        assert report.level == ComplianceLevel.COMPLIANT

    def test_jurisdiction_missing(self):
        rule = _JurisdictionRule()
        ctx = {}
        report = rule.check(ctx)
        assert report.level == ComplianceLevel.NON_COMPLIANT

    def test_audit_trail_complete(self):
        rule = _AuditTrailRule()
        ctx = {"audit_trail": ["step1", "step2"]}
        report = rule.check(ctx)
        assert report.level == ComplianceLevel.COMPLIANT

    def test_audit_trail_missing(self):
        rule = _AuditTrailRule()
        ctx = {}
        report = rule.check(ctx)
        assert report.level == ComplianceLevel.NON_COMPLIANT

    def test_bias_detection_ok(self):
        rule = _BiasDetectionRule()
        ctx = {"evidence": [{"reliability": 0.7}], "conclusion": "ok", "evidence": [{"reliability": 0.7}]}
        report = rule.check(ctx)
        assert report.level == ComplianceLevel.COMPLIANT


class TestEngineeringRules:
    def test_safety_margin_sufficient(self):
        rule = _SafetyMarginRule()
        ctx = {"system_params": {"t": {"design": 80, "limit": 120}}}
        report = rule.check(ctx)
        assert report.level == ComplianceLevel.COMPLIANT

    def test_safety_margin_insufficient(self):
        rule = _SafetyMarginRule()
        ctx = {"system_params": {"p": {"design": 115, "limit": 120}}}
        report = rule.check(ctx)
        assert report.level == ComplianceLevel.NON_COMPLIANT

    def test_safety_margin_missing_data(self):
        rule = _SafetyMarginRule()
        ctx = {}
        report = rule.check(ctx)
        assert report.level == ComplianceLevel.UNABLE_TO_ASSESS

    def test_redundancy_ok(self):
        rule = _RedundancyCheckRule()
        ctx = {"redundancy": {"main": True}}
        report = rule.check(ctx)
        assert report.level == ComplianceLevel.COMPLIANT

    def test_redundancy_conditional(self):
        rule = _RedundancyCheckRule()
        ctx = {"redundancy": {}}
        report = rule.check(ctx)
        assert report.level == ComplianceLevel.CONDITIONAL

    def test_failure_mode_mitigated(self):
        rule = _FailureModeRule()
        ctx = {"fmea": [{"failure_mode": "leak", "rpn": 250, "mitigated": True}]}
        report = rule.check(ctx)
        assert report.level == ComplianceLevel.COMPLIANT

    def test_failure_mode_unmitigated(self):
        rule = _FailureModeRule()
        ctx = {"fmea": [{"failure_mode": "explosion", "rpn": 300, "mitigated": False}]}
        report = rule.check(ctx)
        assert report.level == ComplianceLevel.CONDITIONAL


class TestComplianceEngine:
    def test_init_registers_defaults(self, engine):
        assert engine.rule_count == 9

    def test_auto_register_false(self):
        e = ComplianceRuleEngine(auto_register_defaults=False)
        assert e.rule_count == 0

    def test_check_all_domains(self, engine, medical_context):
        result = engine.check(medical_context)
        assert isinstance(result, ComplianceCheckResult)
        assert len(result.reports) == 9

    def test_check_single_domain(self, engine, medical_context):
        result = engine.check(medical_context, domains=["medical"])
        assert len(result.reports) == 3

    def test_medical_context_compliant(self, engine, medical_context):
        result = engine.check(medical_context, domains=["medical"])
        assert result.is_compliant

    def test_legal_context_compliant(self, engine, legal_context):
        result = engine.check(legal_context, domains=["legal"])
        assert result.is_compliant

    def test_engineering_context_compliant(self, engine, engineering_context):
        result = engine.check(engineering_context, domains=["engineering"])
        assert result.is_compliant

    def test_empty_context(self, engine):
        result = engine.check({})
        assert not result.is_compliant

    def test_is_acceptable(self, engine):
        ctx = {"evidence": [1, 2], "confidence": 0.5}
        result = engine.check(ctx, domains=["medical"])
        assert result.is_acceptable

    def test_summary(self, engine):
        result = engine.check({})
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0

    def test_history(self, engine, medical_context):
        engine.check(medical_context, domains=["medical"])
        history = engine.get_history()
        assert len(history) == 3

    def test_history_filter(self, engine, medical_context, engineering_context):
        engine.check(medical_context, domains=["medical"])
        engine.check(engineering_context, domains=["engineering"])
        med = engine.get_history(domain="medical")
        eng = engine.get_history(domain="engineering")
        assert len(med) == 3
        assert len(eng) == 3

    def test_statistics(self, engine, medical_context):
        engine.check(medical_context)
        stats = engine.statistics()
        assert stats["rule_count"] == 9
        assert stats["total_checks"] > 0

    def test_custom_rule(self, engine):
        class CustomRule(ComplianceRule):
            @property
            def domain(self): return "test"
            @property
            def name(self): return "custom_check"
            def check(self, ctx):
                return ComplianceReport(rule_name="custom_check", domain="test",
                                        level=ComplianceLevel.COMPLIANT, reasoning="ok")

        engine.register(CustomRule())
        assert engine.rule_count == 10
        result = engine.check({}, domains=["test"])
        assert result.is_compliant

    def test_rule_exception(self, engine):
        class BrokenRule(ComplianceRule):
            @property
            def domain(self): return "test"
            @property
            def name(self): return "broken"
            def check(self, ctx):
                raise RuntimeError("fail")

        engine.register(BrokenRule())
        result = engine.check({}, domains=["test"])
        assert result.overall_level == ComplianceLevel.UNABLE_TO_ASSESS

    def test_check_domain_nonexistent(self, engine):
        reports = engine.check_domain("nonexistent", {})
        assert reports == []


class TestComplianceResult:
    def test_compliant_result(self):
        r = ComplianceCheckResult(overall_level=ComplianceLevel.COMPLIANT, reports=[])
        assert r.is_compliant
        assert r.is_acceptable

    def test_conditional_result(self):
        r = ComplianceCheckResult(overall_level=ComplianceLevel.CONDITIONAL, reports=[])
        assert not r.is_compliant
        assert r.is_acceptable

    def test_non_compliant_result(self):
        r = ComplianceCheckResult(overall_level=ComplianceLevel.NON_COMPLIANT, reports=[])
        assert not r.is_compliant
        assert not r.is_acceptable

    def test_report_fields(self):
        report = ComplianceReport(rule_name="test", domain="medical",
                                  level=ComplianceLevel.COMPLIANT,
                                  reasoning="all good", remediation="none needed")
        assert report.rule_name == "test"
        assert report.domain == "medical"
        assert report.level == ComplianceLevel.COMPLIANT
