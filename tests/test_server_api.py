"""API 服务层端到端测试。"""

from __future__ import annotations

import json
import threading
import time
import urllib.request

import pytest

pytestmark = pytest.mark.contract

_SERVER_STARTED = False
_SERVER_PORT = 18099


def _ensure_server():
    global _SERVER_STARTED
    if not _SERVER_STARTED:
        import os

        os.environ["MCI_API_KEY"] = "test-key"
        os.environ["MCI_RATE_LIMIT"] = "1000"
        os.environ["MCI_RATE_BURST"] = "1000"
        # 重置安全单例, 确保 from_env() 重新读取配置
        import mci_world_model.server.security as sec_mod

        sec_mod._auth_config = None
        sec_mod._rate_limiter = None
        from mci_world_model.server.app import create_server

        server = create_server(port=_SERVER_PORT)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        time.sleep(1)
        _SERVER_STARTED = True


def _get(path):
    _ensure_server()
    return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{_SERVER_PORT}{path}").read())


def _post(path, data):
    _ensure_server()
    req = urllib.request.Request(
        f"http://127.0.0.1:{_SERVER_PORT}{path}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json", "X-API-Key": "test-key"},
    )
    return json.loads(urllib.request.urlopen(req).read())


class TestHealthEndpoints:
    def test_health(self):
        r = _get("/health")
        assert r["status"] == "alive"

    def test_ready(self):
        r = _get("/ready")
        assert r["ready"] is True

    def test_metrics(self):
        _ensure_server()
        import urllib.request

        raw = urllib.request.urlopen(f"http://127.0.0.1:{_SERVER_PORT}/metrics").read().decode()
        assert "requests_total" in raw or "request_duration" in raw
        assert "mci_uptime" in raw


class TestDiagnoseEndpoint:
    def test_diagnose_conclusive(self):
        r = _post(
            "/api/v1/diagnose",
            {
                "patient_id": "P001",
                "evidence": [
                    {"id": f"E{i}", "type": "lab", "description": "白蛋白", "confidence": 0.85} for i in range(5)
                ],
                "cause": "低白蛋白",
                "effect": "营养不良",
                "prior_strength": 0.5,
            },
        )
        assert r["is_conclusive"] is True
        assert r["confidence"] > 0.7


class TestEnergyEndpoint:
    def test_what_if(self):
        r = _post(
            "/api/v1/energy/what_if",
            {
                "target_energy": "semantic",
                "boost": 1.5,
            },
        )
        assert "results" in r
        assert len(r["results"]) > 0


class TestBatchEndpoints:
    def test_batch_diagnose(self):
        r = _post(
            "/api/v1/diagnose/batch",
            {
                "queries": [
                    {
                        "cause": "A",
                        "effect": "B",
                        "prior_strength": 0.5,
                        "evidence": [{"id": f"E{i}", "description": "A B", "confidence": 0.85} for i in range(5)],
                    },
                    {
                        "cause": "C",
                        "effect": "D",
                        "prior_strength": 0.6,
                        "evidence": [{"id": f"E{i}", "description": "C D", "confidence": 0.9} for i in range(5)],
                    },
                ],
            },
        )
        assert r["count"] == 2
        assert all("confidence" in d for d in r["diagnoses"])
        assert r["diagnoses"][0]["is_conclusive"] is True
