"""
tests/test_compliance_engine.py — ComplianceRuleEngine 测试
===========================================================

覆盖:
    - 医疗合规: 证据充分性 + 干预安全 + 知情同意
    - 法律合规: 管辖区 + 审计轨迹 + 偏差检测
    - 工程合规: 安全裕度 + 冗余检查 + FMEA
    - ComplianceRuleEngine: 注册+检查+聚合
    - 边界: 空上下文/未知领域/自定义规则
"""

from __future__ import annotations

import pytest

from mci_world_model.sdk._compliance_engine import (
    ComplianceCheckResult,
    ComplianceLevel,
    ComplianceReport,
    ComplianceRule,
    ComplianceRuleEngine,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def engine():
    return ComplianceRuleEngine(auto_register_defaults=True)


@pytest.fixture
def medical_context_compliant():
    return {
        "evidence": ["obs1", "obs2", "obs3"],
        "confidence": 0.9,
        "intervention": {"type": "drug", "dose": "low"},
        "risk_assessment": {"level": "low"},
        "patient_data": {"consent_given": True},
    }


@pytest.fixture
def medical_context_non_compliant():
    return {
        "evidence": [],
        "confidence": 0.5,
        "intervention": {"type": "surgery"},
        "risk_assessment": {"level": "high"},
        "patient_data": {"consent_given": False},
    }


@pytest.fixture
def legal_context_compliant():
    return {
        "jurisdiction": "CN",
        "audit_trail": [{"step": 1, "action": "query"}, {"step": 2, "action": "infer"}],
        "evidence": ["e1", "e2"],
        "confidence": 0.85,
        "conclusion": "X causes Y",
    }


@pytest.fixture
def legal_context_non_compliant():
    return {
        "jurisdiction": "",
        "audit_trail": None,
        "evidence": [],
        "confidence": 0.99,
        "conclusion": "X causes Y without evidence",
    }


@pytest.fixture
def engineering_context_compliant():
    return {
        "system_params": {
            "temperature": {"design": 80.0, "limit": 120.0},  # 33% margin
            "pressure": {"design": 5.0, "limit": 8.0},  # 37.5% margin
        },
        "redundancy": {"primary_path": True, "backup_path": True},
        "critical_paths": ["primary_path", "backup_path"],
        "fmea": [
            {"failure": "valve_stuck", "rpn": 100, "mitigated": True},
        ],
    }


@pytest.fixture
def engineering_context_non_compliant():
    return {
        "system_params": {
            "temperature": {"design": 115.0, "limit": 120.0},  # 4.2% margin < 20%
        },
        "redundancy": {},
        "critical_paths": ["primary_path"],
        "fmea": None,
    }


# =============================================================================
# TestMedicalCompliance
# =============================================================================


class TestMedicalCompliance:
    """医疗合规测试。"""

    def test_compliant(self, engine, medical_context_compliant):
        reports = engine.check_domain("medical", medical_context_compliant)
        assert len(reports) == 3
        for r in reports:
            assert r.level == ComplianceLevel.COMPLIANT

    def test_non_compliant(self, engine, medical_context_non_compliant):
        reports = engine.check_domain("medical", medical_context_non_compliant)
        non_compliant = [r for r in reports if r.level == ComplianceLevel.NON_COMPLIANT]
        assert len(non_compliant) >= 1

    def test_evidence_sufficiency_insufficient(self, engine):
        """证据不足 → 不合规。"""
        ctx = {"evidence": ["only_one"], "confidence": 0.9}
        reports = engine.check_domain("medical", ctx)
        evidence_report = [r for r in reports if r.rule_name == "evidence_sufficiency"][0]
        assert evidence_report.level == ComplianceLevel.NON_COMPLIANT

    def test_intervention_without_risk_assessment(self, engine):
        """干预无风险评估 → 不合规。"""
        ctx = {"evidence": ["e1", "e2"], "confidence": 0.8, "intervention": {"type": "drug"}}
        reports = engine.check_domain("medical", ctx)
        safety_report = [r for r in reports if r.rule_name == "intervention_safety"][0]
        assert safety_report.level == ComplianceLevel.NON_COMPLIANT

    def test_no_consent(self, engine):
        """无知情同意 → 不合规。"""
        ctx = {"evidence": ["e1", "e2"], "confidence": 0.8, "patient_data": {"consent_given": False}}
        reports = engine.check_domain("medical", ctx)
        consent_report = [r for r in reports if r.rule_name == "patient_consent"][0]
        assert consent_report.level == ComplianceLevel.NON_COMPLIANT


# =============================================================================
# TestLegalCompliance
# =============================================================================


class TestLegalCompliance:
    """法律合规测试。"""

    def test_compliant(self, engine, legal_context_compliant):
        reports = engine.check_domain("legal", legal_context_compliant)
        for r in reports:
            assert r.level == ComplianceLevel.COMPLIANT

    def test_non_compliant(self, engine, legal_context_non_compliant):
        reports = engine.check_domain("legal", legal_context_non_compliant)
        non_compliant = [r for r in reports if r.level == ComplianceLevel.NON_COMPLIANT]
        assert len(non_compliant) >= 1

    def test_missing_jurisdiction(self, engine):
        """缺少管辖区 → 不合规。"""
        ctx = {"jurisdiction": None}
        reports = engine.check_domain("legal", ctx)
        jur_report = [r for r in reports if r.rule_name == "jurisdiction_applicability"][0]
        assert jur_report.level == ComplianceLevel.NON_COMPLIANT

    def test_unknown_jurisdiction(self, engine):
        """未知管辖区 → 有条件合规。"""
        ctx = {"jurisdiction": "XX"}
        reports = engine.check_domain("legal", ctx)
        jur_report = [r for r in reports if r.rule_name == "jurisdiction_applicability"][0]
        assert jur_report.level == ComplianceLevel.CONDITIONAL

    def test_empty_audit_trail(self, engine):
        """空审计轨迹 → 不合规。"""
        ctx = {"audit_trail": []}
        reports = engine.check_domain("legal", ctx)
        audit_report = [r for r in reports if r.rule_name == "audit_trail"][0]
        assert audit_report.level == ComplianceLevel.NON_COMPLIANT


# =============================================================================
# TestEngineeringCompliance
# =============================================================================


class TestEngineeringCompliance:
    """工程合规测试。"""

    def test_compliant(self, engine, engineering_context_compliant):
        reports = engine.check_domain("engineering", engineering_context_compliant)
        for r in reports:
            assert r.level == ComplianceLevel.COMPLIANT

    def test_non_compliant(self, engine, engineering_context_non_compliant):
        reports = engine.check_domain("engineering", engineering_context_non_compliant)
        non_compliant = [r for r in reports if r.level == ComplianceLevel.NON_COMPLIANT]
        assert len(non_compliant) >= 1

    def test_insufficient_safety_margin(self, engine):
        """安全裕度不足 → 不合规。"""
        ctx = {"system_params": {"temp": {"design": 95.0, "limit": 100.0}}}
        reports = engine.check_domain("engineering", ctx)
        margin_report = [r for r in reports if r.rule_name == "safety_margin"][0]
        assert margin_report.level == ComplianceLevel.NON_COMPLIANT

    def test_missing_fmea(self, engine):
        """缺少FMEA → 不合规。"""
        ctx = {"fmea": None}
        reports = engine.check_domain("engineering", ctx)
        fmea_report = [r for r in reports if r.rule_name == "failure_mode_analysis"][0]
        assert fmea_report.level == ComplianceLevel.NON_COMPLIANT

    def test_high_rpn_unmitigated(self, engine):
        """高RPN未缓解 → 有条件合规。"""
        ctx = {"fmea": [{"failure": "x", "rpn": 300, "mitigated": False}]}
        reports = engine.check_domain("engineering", ctx)
        fmea_report = [r for r in reports if r.rule_name == "failure_mode_analysis"][0]
        assert fmea_report.level == ComplianceLevel.CONDITIONAL


# =============================================================================
# TestComplianceRuleEngine
# =============================================================================


class TestComplianceRuleEngine:
    """ComplianceRuleEngine 整体测试。"""

    def test_default_rules_registered(self, engine):
        assert engine.rule_count == 9  # 3 medical + 3 legal + 3 engineering

    def test_check_medical_domain_compliant(self, engine, medical_context_compliant):
        result = engine.check(medical_context_compliant, domains=["medical"])
        assert isinstance(result, ComplianceCheckResult)
        assert result.is_compliant is True

    def test_check_specific_domains(self, engine):
        ctx = {"evidence": ["e1", "e2"], "confidence": 0.9, "jurisdiction": "CN", "audit_trail": ["step1"]}
        result = engine.check(ctx, domains=["medical", "legal"])
        assert "engineering" not in result.domains_checked

    def test_custom_rule(self, engine):
        """自定义规则注册。"""

        class CustomRule(ComplianceRule):
            @property
            def domain(self):
                return "custom"

            @property
            def name(self):
                return "custom_rule"

            def check(self, context):
                return ComplianceReport(
                    rule_name=self.name, domain=self.domain, level=ComplianceLevel.COMPLIANT, reasoning="always ok"
                )

        engine.register(CustomRule())
        assert engine.rule_count == 10

    def test_statistics(self, engine, medical_context_compliant):
        engine.check(medical_context_compliant)
        stats = engine.statistics()
        assert "rule_count" in stats
        assert "total_checks" in stats
        assert stats["rule_count"] == 9

    def test_history(self, engine, medical_context_compliant):
        engine.check(medical_context_compliant)
        history = engine.get_history()
        assert len(history) > 0
        medical_history = engine.get_history(domain="medical")
        assert all(r.domain == "medical" for r in medical_history)

    def test_non_compliant_result(self, engine, medical_context_non_compliant):
        result = engine.check(medical_context_non_compliant)
        assert result.is_compliant is False
        assert len(result.non_compliant_rules) > 0
        assert len(result.summary) > 0


# =============================================================================
# TestComplianceCheckResult
# =============================================================================


class TestComplianceCheckResult:
    """ComplianceCheckResult 聚合结果测试。"""

    def test_compliant_result(self):
        result = ComplianceCheckResult(
            overall_level=ComplianceLevel.COMPLIANT,
            domains_checked=["medical"],
        )
        assert result.is_compliant is True
        assert result.is_acceptable is True
        assert result.summary == "全部合规"

    def test_non_compliant_result(self):
        result = ComplianceCheckResult(
            overall_level=ComplianceLevel.NON_COMPLIANT,
            non_compliant_rules=["rule_a"],
            conditional_rules=["rule_b"],
            domains_checked=["legal"],
        )
        assert result.is_compliant is False
        assert "rule_a" in result.summary

    def test_conditional_result(self):
        result = ComplianceCheckResult(
            overall_level=ComplianceLevel.CONDITIONAL,
            conditional_rules=["rule_x"],
        )
        assert result.is_compliant is False
        assert result.is_acceptable is True
