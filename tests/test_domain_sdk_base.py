"""tests/test_domain_sdk_base.py"""

from __future__ import annotations

import pytest

from mci_world_model.sdk._domain_sdk_base import DomainType, MCIDomainSDK


@pytest.fixture
def sdk():
    return MCIDomainSDK()


class TestMedicalReasoning:
    def test_with_evidence(self, sdk):
        sdk.add_domain_evidence("medical", {"confidence": 0.85})
        sdk.add_domain_evidence("medical", {"confidence": 0.9})
        result = sdk.reason("medical", "drug_X", "symptom_Y")
        assert result.domain == DomainType.MEDICAL
        assert result.is_compliant is True

    def test_insufficient_evidence(self, sdk):
        result = sdk.reason("medical", "X", "Y")
        assert result.is_compliant is False


class TestLegalReasoning:
    def test_compliant(self, sdk):
        sdk.add_domain_evidence("legal", {"reliability": 0.8, "jurisdiction": "CN", "audit_trail": ["step1"]})
        sdk.add_domain_evidence("legal", {"reliability": 0.9, "jurisdiction": "CN", "audit_trail": ["step2"]})
        result = sdk.reason("legal", "action_A", "harm_B")
        assert result.domain == DomainType.LEGAL

    def test_no_jurisdiction(self, sdk):
        sdk.add_domain_evidence("legal", {"reliability": 0.8})
        result = sdk.reason("legal", "A", "B")
        assert result.is_compliant is False


class TestEngineeringReasoning:
    def test_safe(self, sdk):
        sdk.add_domain_evidence("engineering", {"safety_margin": 0.3})
        result = sdk.reason("engineering", "high_temp", "failure")
        assert result.domain == DomainType.ENGINEERING
        assert result.is_compliant is True

    def test_unsafe(self, sdk):
        result = sdk.reason("engineering", "X", "Y")
        assert result.is_compliant is False


class TestStatistics:
    def test_stats(self, sdk):
        sdk.add_domain_evidence("medical", {"confidence": 0.9})
        sdk.reason("medical", "X", "Y")
        stats = sdk.statistics()
        assert stats["reason_count"] == 1
