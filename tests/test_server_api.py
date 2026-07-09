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
        headers={"Content-Type": "application/json"},
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
        r = _get("/metrics")
        assert "requests_total" in r


class TestDiagnoseEndpoint:
    def test_diagnose_conclusive(self):
        r = _post("/api/v1/diagnose", {
            "patient_id": "P001",
            "evidence": [
                {"id": f"E{i}", "type": "lab", "description": "白蛋白", "confidence": 0.85}
                for i in range(5)
            ],
            "cause": "低白蛋白",
            "effect": "营养不良",
            "prior_strength": 0.5,
        })
        assert r["is_conclusive"] is True
        assert r["confidence"] > 0.7


class TestEnergyEndpoint:
    def test_what_if(self):
        r = _post("/api/v1/energy/what_if", {
            "target_energy": "semantic",
            "boost": 1.5,
        })
        assert "results" in r
        assert len(r["results"]) > 0
