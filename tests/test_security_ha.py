"""安全层与高可用组件测试 — 认证、限流、断路器、存储外置化。"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import urllib.request

import pytest

pytestmark = pytest.mark.contract

_SERVER_STARTED = False
_SERVER_PORT = 18100


def _ensure_server():
    global _SERVER_STARTED
    if not _SERVER_STARTED:
        os.environ["MCI_API_KEY"] = "secure-test-key"
        os.environ["MCI_RATE_LIMIT"] = "1000"
        os.environ["MCI_RATE_BURST"] = "1000"
        os.environ["MCI_STORAGE_PATH"] = tempfile.mkdtemp(prefix="mci_test_")
        from mci_world_model.server.app import create_server

        server = create_server(port=_SERVER_PORT)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        time.sleep(1)
        _SERVER_STARTED = True


def _post(path, data, headers=None):
    _ensure_server()
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(
        f"http://127.0.0.1:{_SERVER_PORT}{path}",
        data=json.dumps(data).encode(),
        headers=hdrs,
    )
    return urllib.request.urlopen(req)


def _post_raw(path, data, headers=None):
    """返回 (status_code, body_dict), 不抛异常。"""
    try:
        resp = _post(path, data, headers)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get_raw(path, headers=None):
    """GET 请求, 返回 (status_code, body_dict), 不抛异常。"""
    _ensure_server()
    hdrs = {"X-API-Key": "secure-test-key"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(f"http://127.0.0.1:{_SERVER_PORT}{path}", headers=hdrs)
    try:
        resp = urllib.request.urlopen(req)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


# =============================================================================
# 认证测试
# =============================================================================


class TestAuthentication:
    def test_no_auth_returns_401(self):
        """无认证头 → 401。"""
        status, body = _post_raw(
            "/api/v1/diagnose",
            {
                "cause": "A",
                "effect": "B",
                "prior_strength": 0.5,
            },
        )
        assert status == 401
        assert body["error"] == "unauthorized"

    def test_wrong_key_returns_401(self):
        """错误 API Key → 401。"""
        status, _body = _post_raw(
            "/api/v1/diagnose",
            {
                "cause": "A",
                "effect": "B",
                "prior_strength": 0.5,
            },
            headers={"X-API-Key": "wrong-key"},
        )
        assert status == 401

    def test_correct_key_x_api_key(self):
        """正确 X-API-Key → 200。"""
        status, _body = _post_raw(
            "/api/v1/diagnose",
            {
                "cause": "低白蛋白",
                "effect": "营养不良",
                "prior_strength": 0.5,
                "evidence": [{"id": f"E{i}", "description": "白蛋白", "confidence": 0.85} for i in range(5)],
            },
            headers={"X-API-Key": "secure-test-key"},
        )
        assert status == 200

    def test_correct_key_bearer(self):
        """正确 Bearer token → 200。"""
        status, _body = _post_raw(
            "/api/v1/diagnose",
            {
                "cause": "低白蛋白",
                "effect": "营养不良",
                "prior_strength": 0.5,
                "evidence": [{"id": f"E{i}", "description": "白蛋白", "confidence": 0.85} for i in range(5)],
            },
            headers={"Authorization": "Bearer secure-test-key"},
        )
        assert status == 200

    def test_health_no_auth_required(self):
        """健康检查不需要认证。"""
        _ensure_server()
        resp = urllib.request.urlopen(f"http://127.0.0.1:{_SERVER_PORT}/health")
        assert resp.status == 200

    def test_case_insensitive_header(self):
        """HTTP header 大小写不敏感。"""
        status, _body = _post_raw(
            "/api/v1/diagnose",
            {
                "cause": "A",
                "effect": "B",
                "prior_strength": 0.5,
                "evidence": [{"id": "E1", "description": "A B", "confidence": 0.85}],
            },
            headers={"x-api-key": "secure-test-key"},
        )
        assert status == 200


# =============================================================================
# 限流测试
# =============================================================================


class TestRateLimiting:
    def test_burst_limit_returns_429(self):
        """独立 RateLimiter 实例验证突发限流。"""
        from mci_world_model.server.security import RateLimiter

        rl = RateLimiter(rate=0.1, burst=3)  # 极低速率, burst 3
        # 前 3 个通过 (消耗 burst 令牌)
        assert rl.allow("1.2.3.4") is True
        assert rl.allow("1.2.3.4") is True
        assert rl.allow("1.2.3.4") is True
        # 第 4 个被限流 (令牌耗尽, rate=0.1 太慢来不及补充)
        assert rl.allow("1.2.3.4") is False

    def test_rate_limiter_isolates_by_ip(self):
        """不同 IP 各自有独立令牌桶。"""
        from mci_world_model.server.security import RateLimiter

        rl = RateLimiter(rate=0.1, burst=2)
        assert rl.allow("10.0.0.1") is True
        assert rl.allow("10.0.0.1") is True
        # 不同 IP 仍有自己的令牌
        assert rl.allow("10.0.0.2") is True

    def test_rate_limiter_refill(self):
        """令牌随时间补充。"""
        from mci_world_model.server.security import RateLimiter

        rl = RateLimiter(rate=100, burst=1)  # 高速率补充
        assert rl.allow("1.1.1.1") is True  # 消耗唯一令牌
        assert rl.allow("1.1.1.1") is False  # 立即再请求: 被限
        time.sleep(0.05)  # 等待补充
        assert rl.allow("1.1.1.1") is True  # 补充后通过


# =============================================================================
# 断路器测试
# =============================================================================


class TestCircuitBreaker:
    def test_cb_opens_after_consecutive_failures(self):
        """连续 5 次失败后断路器 OPEN。"""
        from mci_world_model.server.security import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=1.0)
        assert cb.state == CircuitBreaker.CLOSED
        for _ in range(5):
            cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN
        assert cb.allow() is False

    def test_cb_half_open_recovery(self):
        """超时后断路器 HALF_OPEN, 成功后 CLOSED。"""
        from mci_world_model.server.security import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN
        time.sleep(0.15)
        assert cb.state == CircuitBreaker.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitBreaker.CLOSED


# =============================================================================
# 存储外置化测试
# =============================================================================


class TestStorage:
    def test_file_storage_round_trip(self):
        """FileStorage 保存 → 加载 round-trip。"""
        from mci_world_model.server.storage import FileStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStorage(tmpdir)
            store.save("key1", {"confidence": 0.9, "effect": "test"})
            loaded = store.load("key1")
            assert loaded == {"confidence": 0.9, "effect": "test"}

    def test_file_storage_list_keys(self):
        """FileStorage 前缀匹配 list_keys。"""
        from mci_world_model.server.storage import FileStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStorage(tmpdir)
            store.save("diag:P1:1", {"a": 1})
            store.save("diag:P1:2", {"a": 2})
            store.save("diag:P2:1", {"a": 3})
            keys = store.list_keys("diag:P1")
            assert len(keys) == 2
            assert all(k.startswith("diag:P1") for k in keys)

    def test_storage_save_load_diagnosis(self):
        """save_diagnosis / load_diagnosis 端到端。"""
        from mci_world_model.server.storage import load_diagnosis, save_diagnosis

        os.environ["MCI_STORAGE_BACKEND"] = "file"
        os.environ["MCI_STORAGE_PATH"] = tempfile.mkdtemp(prefix="mci_test2_")
        # 重置全局存储单例
        import mci_world_model.server.storage as storage_mod

        storage_mod._storage = None

        record_id = save_diagnosis("P001", {"confidence": 0.85})
        loaded = load_diagnosis(record_id)
        assert loaded is not None
        assert loaded["confidence"] == 0.85
        assert loaded["patient_id"] == "P001"
        assert "timestamp" in loaded


# =============================================================================
# 诊断持久化端到端测试
# =============================================================================


class TestDiagnosisPersistence:
    def test_diagnose_returns_record_id(self):
        """POST /diagnose 返回 record_id, 可用 GET 查询。"""
        status, body = _post_raw(
            "/api/v1/diagnose",
            {
                "cause": "低白蛋白",
                "effect": "营养不良",
                "prior_strength": 0.5,
                "evidence": [{"id": f"E{i}", "description": "白蛋白", "confidence": 0.85} for i in range(5)],
                "patient_id": "PERSIST_P001",
            },
            headers={"X-API-Key": "secure-test-key"},
        )
        assert status == 200
        assert "record_id" in body
        record_id = body["record_id"]

        # GET 查询
        status2, record = _get_raw(f"/api/v1/diagnosis/{record_id}")
        assert status2 == 200
        assert record["confidence"] == body["confidence"]
        assert record["patient_id"] == "PERSIST_P001"

    def test_diagnosis_not_found_404(self):
        """查询不存在的 record_id → 404。"""
        status, _body = _get_raw("/api/v1/diagnosis/nonexistent")
        assert status == 404
