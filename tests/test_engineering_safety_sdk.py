"""tests/test_engineering_safety_sdk.py — EngineeringSafetySDK 测试"""

from __future__ import annotations

import pytest

from mci_world_model.sdk._engineering_safety_sdk import (
    EngineeringCausalResult,
    EngineeringSafetySDK,
    FMEAItem,
    SafetyParameter,
)


@pytest.fixture
def sdk():
    return EngineeringSafetySDK(system_name="reactor_01", redundancy_required=True)


class TestSafetyParameter:
    def test_margin_calculation(self):
        p = SafetyParameter("temp", design_value=80.0, limit_value=100.0)
        assert abs(p.safety_margin - 0.2) < 1e-6

    def test_zero_limit(self):
        p = SafetyParameter("zero", design_value=0.0, limit_value=0.0)
        assert p.safety_margin == 0.0


class TestFMEAItem:
    def test_rpn_calculation(self):
        item = FMEAItem("valve_stuck", severity=8, occurrence=3, detection=2)
        assert item.rpn == 48

    def test_default_rpn(self):
        item = FMEAItem("test")
        assert item.rpn == 125  # 5*5*5


class TestEngineeringSafetySDK:
    def test_add_parameter(self, sdk):
        sdk.add_parameter(SafetyParameter("temp", 80.0, 120.0))
        assert sdk.parameter_count == 1

    def test_safe_analysis(self, sdk):
        sdk.add_parameter(SafetyParameter("temp", 80.0, 120.0))  # 33% margin
        sdk.add_parameter(SafetyParameter("pressure", 5.0, 8.0))  # 37.5% margin
        sdk.add_fmea(FMEAItem("valve_stuck", severity=3, occurrence=2, detection=2, mitigated=True))
        sdk.set_redundancy("primary", True)
        sdk.set_redundancy("backup", True)
        result = sdk.analyze(cause="high_temp", effect="failure", causal_evidence_strength=0.8)
        assert isinstance(result, EngineeringCausalResult)
        assert result.safety_assessment == "safe"
        assert result.margin_sufficient is True

    def test_unsafe_margin(self, sdk):
        sdk.add_parameter(SafetyParameter("temp", 95.0, 100.0))  # 5% margin < 20%
        result = sdk.analyze(cause="high_temp", effect="failure")
        assert result.safety_assessment == "unsafe"
        assert result.margin_sufficient is False

    def test_high_rpn_unmitigated(self, sdk):
        sdk.add_parameter(SafetyParameter("temp", 80.0, 120.0))
        sdk.add_fmea(FMEAItem("critical_failure", severity=9, occurrence=5, detection=5))  # RPN=225
        sdk.set_redundancy("primary", True)
        result = sdk.analyze(cause="X", effect="Y")
        assert result.safety_assessment == "conditional"

    def test_no_redundancy(self, sdk):
        sdk.add_parameter(SafetyParameter("temp", 80.0, 120.0))
        sdk.set_redundancy("primary", False)
        result = sdk.analyze(cause="X", effect="Y")
        assert result.redundancy_ok is False
        assert result.safety_assessment == "conditional"

    def test_statistics(self, sdk):
        sdk.add_parameter(SafetyParameter("temp", 80.0, 120.0))
        sdk.analyze(cause="X", effect="Y")
        stats = sdk.statistics()
        assert stats["parameter_count"] == 1
        assert stats["analysis_count"] == 1
