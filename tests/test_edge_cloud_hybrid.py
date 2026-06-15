"""tests/test_edge_cloud_hybrid.py — EdgeCloudHybrid 测试"""

from __future__ import annotations

import pytest

from mci_world_model.sdk._edge_cloud_hybrid import (
    EdgeCloudHybrid,
    InferenceRequest,
    InferenceResult,
)


@pytest.fixture
def hybrid():
    return EdgeCloudHybrid(edge_capacity=5, edge_latency_ms=10.0, cloud_latency_ms=100.0)


class TestInferenceRequest:
    def test_creation(self):
        req = InferenceRequest(request_id="r1", complexity=3)
        assert req.request_id == "r1"
        assert req.complexity == 3
        assert req.priority == "medium"


class TestInferenceResult:
    def test_creation(self):
        result = InferenceResult(request_id="r1", executed_on="edge", latency_ms=15.0)
        assert result.request_id == "r1"
        assert result.executed_on == "edge"
        assert result.cached is False


class TestEdgeCloudHybrid:
    def test_edge_dispatch(self, hybrid):
        req = InferenceRequest(request_id="r1", complexity=3)
        result = hybrid.dispatch(req)
        assert result.executed_on == "edge"
        assert result.cached is False

    def test_cloud_dispatch(self, hybrid):
        req = InferenceRequest(request_id="r2", complexity=8)
        result = hybrid.dispatch(req)
        assert result.executed_on == "cloud"

    def test_audit_forces_cloud(self, hybrid):
        req = InferenceRequest(request_id="r3", complexity=3, requires_audit=True)
        result = hybrid.dispatch(req)
        assert result.executed_on == "cloud"

    def test_cache_hit(self, hybrid):
        req = InferenceRequest(request_id="r4", query={"q": "test"}, complexity=3)
        hybrid.dispatch(req)
        result = hybrid.dispatch(req)
        assert result.cached is True
        assert result.latency_ms < 10.0

    def test_invalid_edge_capacity(self):
        with pytest.raises(ValueError, match="edge_capacity"):
            EdgeCloudHybrid(edge_capacity=0)

    def test_statistics(self, hybrid):
        hybrid.dispatch(InferenceRequest(request_id="r1", complexity=3))
        hybrid.dispatch(InferenceRequest(request_id="r2", complexity=8))
        stats = hybrid.statistics()
        assert stats["dispatch_count"] == 2
        assert stats["edge_count"] == 1
        assert stats["cloud_count"] == 1

    def test_clear_cache(self, hybrid):
        req = InferenceRequest(request_id="r1", query={"q": "test"}, complexity=3)
        hybrid.dispatch(req)
        hybrid.clear_cache()
        stats = hybrid.statistics()
        assert stats["cache_size"] == 0

    def test_cache_eviction(self):
        h = EdgeCloudHybrid(edge_capacity=5, cache_size=2)
        h.dispatch(InferenceRequest(request_id="r1", query={"q": "a"}, complexity=3))
        h.dispatch(InferenceRequest(request_id="r2", query={"q": "b"}, complexity=3))
        h.dispatch(InferenceRequest(request_id="r3", query={"q": "c"}, complexity=3))
        # Cache size should stay at 2
        stats = h.statistics()
        assert stats["cache_size"] <= 2
