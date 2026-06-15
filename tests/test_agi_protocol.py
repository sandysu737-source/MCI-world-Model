"""tests/test_agi_protocol.py"""

from __future__ import annotations

import pytest

from mci_world_model.sdk._agi_protocol import (
    AGICapability,
    AGIIntegrationProtocol,
    AGIRequest,
    AGIResponse,
)


@pytest.fixture
def protocol():
    return AGIIntegrationProtocol(min_confidence=0.5, audit_enabled=True)


class TestRegisterCapability:
    def test_register_one(self, protocol):
        protocol.register_capability(AGICapability.CAUSAL_REASONING)
        assert "causal_reasoning" in protocol.registered_capabilities

    def test_register_multiple(self, protocol):
        protocol.register_capability(AGICapability.CAUSAL_REASONING)
        protocol.register_capability(AGICapability.COUNTERFACTUAL)
        assert len(protocol.registered_capabilities) == 2

    def test_duplicate_ignored(self, protocol):
        protocol.register_capability(AGICapability.CAUSAL_REASONING)
        protocol.register_capability(AGICapability.CAUSAL_REASONING)
        assert len(protocol.registered_capabilities) == 1


class TestHandleRequest:
    def test_unregistered_capability_fails(self, protocol):
        request = AGIRequest(
            request_id="r1",
            capability=AGICapability.CAUSAL_REASONING,
        )
        response = protocol.handle_request(request)
        assert isinstance(response, AGIResponse)
        assert not response.success
        assert len(response.warnings) > 0

    def test_causal_reasoning_success(self, protocol):
        protocol.register_capability(AGICapability.CAUSAL_REASONING)
        request = AGIRequest(
            request_id="r2",
            capability=AGICapability.CAUSAL_REASONING,
            payload={"hypothesis": "X→Y", "evidence_strength": 0.8},
        )
        response = protocol.handle_request(request)
        assert response.success
        assert response.confidence >= 0.5
        assert response.audit_trail_id != ""

    def test_causal_reasoning_low_confidence(self, protocol):
        protocol.register_capability(AGICapability.CAUSAL_REASONING)
        request = AGIRequest(
            request_id="r3",
            capability=AGICapability.CAUSAL_REASONING,
            payload={"hypothesis": "X→Y", "evidence_strength": 0.3},
        )
        response = protocol.handle_request(request)
        assert not response.success
        assert len(response.warnings) > 0

    def test_counterfactual(self, protocol):
        protocol.register_capability(AGICapability.COUNTERFACTUAL)
        request = AGIRequest(
            request_id="r4",
            capability=AGICapability.COUNTERFACTUAL,
            payload={"plausibility": 0.7},
        )
        response = protocol.handle_request(request)
        assert response.success

    def test_audit_disabled(self):
        p = AGIIntegrationProtocol(min_confidence=0.5, audit_enabled=False)
        p.register_capability(AGICapability.CAUSAL_REASONING)
        request = AGIRequest(
            request_id="r5",
            capability=AGICapability.CAUSAL_REASONING,
            payload={"evidence_strength": 0.8},
        )
        response = p.handle_request(request)
        assert response.audit_trail_id == ""

    def test_no_audit_requested(self, protocol):
        protocol.register_capability(AGICapability.CAUSAL_REASONING)
        request = AGIRequest(
            request_id="r6",
            capability=AGICapability.CAUSAL_REASONING,
            payload={"evidence_strength": 0.8},
            requires_audit=False,
        )
        response = protocol.handle_request(request)
        assert response.audit_trail_id == ""


class TestStatistics:
    def test_empty_stats(self):
        p = AGIIntegrationProtocol()
        stats = p.statistics()
        assert stats["request_count"] == 0
        assert stats["success_rate"] == 0.0

    def test_stats_after_requests(self, protocol):
        protocol.register_capability(AGICapability.CAUSAL_REASONING)
        request = AGIRequest(
            request_id="r1",
            capability=AGICapability.CAUSAL_REASONING,
            payload={"evidence_strength": 0.9},
        )
        protocol.handle_request(request)
        stats = protocol.statistics()
        assert stats["request_count"] == 1
        assert stats["success_count"] == 1
