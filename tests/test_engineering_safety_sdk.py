"""End-to-end tests for EngineeringSafetySDK — P7 engineering safety validation."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mci_world_model.sdk._engineering_safety_sdk import (
    SafetyParameter,
    FMEAItem,
    EngineeringCausalResult,
    EngineeringSafetySDK,
)


# ────────────────────────────────────────────────────────────────────
# SafetyParameter tests
# ────────────────────────────────────────────────────────────────────

class TestSafetyParameter:
    def test_basic_creation(self):
        p = SafetyParameter("temp", 80.0, 120.0, "°C")
        assert p.name == "temp"
        assert p.design_value == 80.0
        assert p.limit_value == 120.0
        assert p.unit == "°C"

    def test_margin_calculation(self):
        p = SafetyParameter("pressure", 90.0, 100.0, "bar")
        # margin = |100-90|/|100| = 0.10
        assert p.safety_margin == pytest.approx(0.10, abs=0.001)

    def test_margin_at_limit(self):
        p = SafetyParameter("speed", 100.0, 100.0)
        assert p.safety_margin == 0.0

    def test_margin_zero_limit(self):
        p = SafetyParameter("offset", 5.0, 0.0)
        assert p.safety_margin == 0.0

    def test_margin_high_safety(self):
        p = SafetyParameter("voltage", 12.0, 50.0, "V")
        # margin = |50-12|/50 = 0.76
        assert p.safety_margin == pytest.approx(0.76, abs=0.01)


# ────────────────────────────────────────────────────────────────────
# FMEAItem tests
# ────────────────────────────────────────────────────────────────────

class TestFMEAItem:
    def test_basic_creation(self):
        item = FMEAItem("valve_stuck", "loss of control", 8, 3, 2)
        assert item.failure_mode == "valve_stuck"
        assert item.effect == "loss of control"
        assert item.rpn == 48  # 8*3*2

    def test_rpn_calculation(self):
        item = FMEAItem("sensor_fail", severity=10, occurrence=8, detection=5)
        assert item.rpn == 400  # critical

    def test_rpn_defaults(self):
        item = FMEAItem("minor_glitch")
        assert item.rpn == 125  # 5*5*5
        assert not item.mitigated

    def test_mitigated_flag(self):
        item = FMEAItem("fixed_issue", mitigated=True)
        assert item.mitigated


# ────────────────────────────────────────────────────────────────────
# EngineeringSafetySDK tests
# ────────────────────────────────────────────────────────────────────

class TestEngineeringSafetySDK:
    def test_initialization(self):
        sdk = EngineeringSafetySDK("TurbineControl")
        assert sdk.system_name == "TurbineControl"
        assert sdk.parameter_count == 0

    def test_add_parameter_increments_count(self):
        sdk = EngineeringSafetySDK()
        assert sdk.parameter_count == 0
        sdk.add_parameter(SafetyParameter("temp", 80, 120))
        assert sdk.parameter_count == 1

    def test_safe_scenario_all_checks_pass(self):
        """Full safe scenario: sufficient margin, no critical FMEA, has redundancy."""
        sdk = EngineeringSafetySDK("SafeSystem")
        sdk.add_parameter(SafetyParameter("temp", 80.0, 120.0))  # margin=0.33 > 0.20
        sdk.add_fmea(FMEAItem("minor_leak", severity=3, occurrence=2, detection=2))  # RPN=12
        sdk.set_redundancy("control_path", True)

        result = sdk.analyze("high_temp", "overheat", causal_evidence_strength=0.8)
        assert result.safety_assessment == "safe"
        assert result.margin_sufficient
        assert result.redundancy_ok
        assert result.causal_confidence == pytest.approx(0.8, abs=0.01)

    def test_unsafe_margin_violation(self):
        """Margin below 20% triggers unsafe."""
        sdk = EngineeringSafetySDK("UnsafeSystem")
        sdk.add_parameter(SafetyParameter("temp", 95.0, 100.0))  # margin=0.05 < 0.20
        sdk.set_redundancy("path", True)

        result = sdk.analyze("high_temp", "failure")
        assert result.safety_assessment == "unsafe"
        assert not result.margin_sufficient
        assert result.causal_confidence == pytest.approx(0.5 * 0.3, abs=0.01)

    def test_conditional_unmitigated_critical_fmea(self):
        """Critical FMEA not mitigated → conditional."""
        sdk = EngineeringSafetySDK("CriticalFM")
        sdk.add_parameter(SafetyParameter("temp", 80.0, 120.0))  # margin OK
        sdk.add_fmea(FMEAItem("catastrophic_failure", severity=9, occurrence=5, detection=5))  # RPN=225 > 200
        sdk.set_redundancy("path", True)

        result = sdk.analyze("overload", "explosion")
        assert result.safety_assessment == "conditional"
        assert result.fmea_rpn_max == 225
        assert result.causal_confidence == pytest.approx(0.5 * 0.7, abs=0.01)

    def test_conditional_no_redundancy(self):
        """Missing redundancy on required system → conditional."""
        sdk = EngineeringSafetySDK("NoRedundancy", redundancy_required=True)
        sdk.add_parameter(SafetyParameter("temp", 80.0, 120.0))
        # No FMEA items, no redundancy set

        result = sdk.analyze("high_temp", "failure")
        assert result.safety_assessment == "conditional"
        assert not result.redundancy_ok

    def test_redundancy_not_required(self):
        """When redundancy not required, missing it is fine."""
        sdk = EngineeringSafetySDK("NoRedundNeeded", redundancy_required=False)
        sdk.add_parameter(SafetyParameter("temp", 80.0, 120.0))

        result = sdk.analyze("high_temp", "failure")
        assert result.safety_assessment == "safe"
        assert result.redundancy_ok  # not required → always True

    def test_multiple_parameters_margin_check(self):
        """Only one parameter below margin still triggers unsafe."""
        sdk = EngineeringSafetySDK()
        sdk.add_parameter(SafetyParameter("ok1", 80, 120))    # margin=0.33
        sdk.add_parameter(SafetyParameter("bad1", 98, 100))   # margin=0.02
        sdk.add_parameter(SafetyParameter("ok2", 50, 200))    # margin=0.75
        sdk.set_redundancy("path", True)

        result = sdk.analyze("multi", "cascade")
        assert result.safety_assessment == "unsafe"

    def test_mitigated_fmea_passes(self):
        """Mitigated FMEA above RPN threshold should not trigger conditional."""
        sdk = EngineeringSafetySDK()
        sdk.add_parameter(SafetyParameter("temp", 80, 120))
        sdk.add_fmea(FMEAItem("mitigated_risk", severity=9, occurrence=5, detection=5, mitigated=True))
        sdk.set_redundancy("path", True)

        result = sdk.analyze("normal", "event")
        assert result.safety_assessment == "safe"

    def test_statistics_accumulation(self):
        sdk = EngineeringSafetySDK("StatsSystem")
        sdk.add_parameter(SafetyParameter("a", 1, 2))
        sdk.add_parameter(SafetyParameter("b", 3, 10))
        sdk.add_fmea(FMEAItem("f1"))
        sdk.add_fmea(FMEAItem("f2"))
        sdk.set_redundancy("r1", True)
        sdk.set_redundancy("r2", True)  # both True for safe result

        sdk.analyze("c1", "e1")  # safe
        sdk.analyze("c2", "e2")  # safe

        stats = sdk.statistics()
        assert stats["system_name"] == "StatsSystem"
        assert stats["parameter_count"] == 2
        assert stats["fmea_items"] == 2
        assert stats["redundancy_paths"] == 2
        assert stats["analysis_count"] == 2
        assert stats["safe_count"] == 2

    def test_audit_trail_completeness(self):
        sdk = EngineeringSafetySDK("AuditSys")
        sdk.add_parameter(SafetyParameter("temp", 80, 120))
        sdk.add_fmea(FMEAItem("f1"))
        sdk.set_redundancy("path", True)

        result = sdk.analyze("cause", "effect")
        assert len(result.audit_trail) == 3
        steps = [entry["step"] for entry in result.audit_trail]
        assert steps == ["margin_check", "fmea_check", "redundancy_check"]

    def test_confidence_modifier_safe(self):
        sdk = EngineeringSafetySDK()
        sdk.add_parameter(SafetyParameter("ok", 50, 100))  # margin=0.5 > 0.2
        sdk.set_redundancy("p", True)
        result = sdk.analyze("c", "e", causal_evidence_strength=0.6)
        assert result.causal_confidence == pytest.approx(0.6, abs=0.01)

    def test_confidence_modifier_unsafe(self):
        sdk = EngineeringSafetySDK()
        sdk.add_parameter(SafetyParameter("bad", 98, 100))  # margin=0.02
        sdk.set_redundancy("p", True)
        result = sdk.analyze("c", "e", causal_evidence_strength=0.9)
        assert result.causal_confidence == pytest.approx(0.9 * 0.3, abs=0.01)

    def test_confidence_modifier_conditional(self):
        sdk = EngineeringSafetySDK()
        sdk.add_parameter(SafetyParameter("ok", 50, 100))
        sdk.add_fmea(FMEAItem("critical", severity=8, occurrence=7, detection=5))  # RPN=280
        sdk.set_redundancy("p", True)
        result = sdk.analyze("c", "e", causal_evidence_strength=0.7)
        assert result.causal_confidence == pytest.approx(0.7 * 0.7, abs=0.01)

    def test_redundancy_partial_failure(self):
        """One redundancy path missing → conditional."""
        sdk = EngineeringSafetySDK()
        sdk.add_parameter(SafetyParameter("ok", 50, 100))
        sdk.set_redundancy("path_a", True)
        sdk.set_redundancy("path_b", False)

        result = sdk.analyze("c", "e")
        assert result.safety_assessment == "conditional"
        assert not result.redundancy_ok

    def test_empty_system_safe(self):
        """Empty system (no params, no FMEA) with redundancy should be safe."""
        sdk = EngineeringSafetySDK()
        sdk.set_redundancy("path", True)
        result = sdk.analyze("c", "e")
        # No params → no margin violations → margin_sufficient=True
        # No FMEA → rpn_max=0
        assert result.safety_assessment == "safe"
        assert result.fmea_rpn_max == 0

    def test_rpn_at_threshold(self):
        """RPN exactly at 200 should NOT trigger conditional."""
        sdk = EngineeringSafetySDK()
        sdk.add_parameter(SafetyParameter("ok", 50, 100))
        sdk.add_fmea(FMEAItem("boundary", severity=8, occurrence=5, detection=5))  # RPN=200
        sdk.set_redundancy("p", True)

        result = sdk.analyze("c", "e")
        # RPN=200 is not > 200, so it should be safe
        assert result.safety_assessment == "safe"

    def test_rpn_above_threshold_unmitigated(self):
        """RPN=201 > 200, unmitigated → conditional."""
        sdk = EngineeringSafetySDK()
        sdk.add_parameter(SafetyParameter("ok", 50, 100))
        sdk.add_fmea(FMEAItem("over", severity=8, occurrence=6, detection=5))  # RPN=240
        sdk.set_redundancy("p", True)

        result = sdk.analyze("c", "e")
        assert result.safety_assessment == "conditional"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
