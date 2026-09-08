"""WP-03 JSON 解析、有限性与安全响应测试。"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import socket
import tempfile
import threading
from unittest.mock import MagicMock

import pytest

from mci_world_model.server.app import MCIAPIHandler, create_server
from mci_world_model.server.metrics import metrics
from mci_world_model.server.validation import (
    assert_finite_tree,
    assert_json_depth,
    loads_safe_json,
)

pytestmark = pytest.mark.contract

_SERVER = None
_SERVER_THREAD = None
_SERVER_PORT = 0


def _write_identity_map(api_key: str) -> str:
    """创建最小身份映射，保持与生产认证路径一致。"""
    base = tempfile.mkdtemp(prefix="mci_wp03_identity_")
    path = os.path.join(base, "identity-map.json")
    key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    payload = {
        "version": 1,
        "identities": {
            key_hash: {
                "subject": "wp03-subject",
                "tenant_id": "wp03-tenant",
                "patient_ids": ["P001"],
            }
        },
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file)
    os.chmod(path, 0o600)
    os.chmod(base, 0o700)
    return path


@pytest.fixture(scope="module")
def json_server():
    """启动带认证身份的隔离 server，专测解析与响应门禁。"""
    global _SERVER, _SERVER_THREAD, _SERVER_PORT
    if _SERVER is None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            _SERVER_PORT = sock.getsockname()[1]

        import mci_world_model.server.security as security_module

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv("MCI_ENV", "test")
            monkeypatch.setenv("MCI_API_KEY", "wp03-key")
            monkeypatch.setenv("MCI_IDENTITY_MAP_PATH", _write_identity_map("wp03-key"))
            monkeypatch.setenv("MCI_RATE_LIMIT", "1000")
            monkeypatch.setenv("MCI_RATE_BURST", "1000")
            security_module._auth_config = None
            security_module._rate_limiter = None
            security_module.get_auth_config()
            security_module.get_rate_limiter()
            _SERVER = create_server(host="127.0.0.1", port=_SERVER_PORT)
        _SERVER_THREAD = threading.Thread(target=_SERVER.serve_forever, daemon=True)
        _SERVER_THREAD.start()

    yield _SERVER_PORT

    _SERVER.shutdown()
    _SERVER_THREAD.join(timeout=5)
    _SERVER.server_close()
    _SERVER = None
    _SERVER_THREAD = None
    security_module._auth_config = None
    security_module._rate_limiter = None


def _request(port: int, path: str, body: str | None = None) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        headers = {"X-API-Key": "wp03-key"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        connection.request("POST" if body is not None else "GET", path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def test_json_depth_scanner_is_iterative_and_string_aware() -> None:
    assert_json_depth('{"text":"[{}]"}', max_depth=1)
    with pytest.raises(ValueError, match="嵌套深度"):
        assert_json_depth("[" * 65 + "1" + "]" * 65, max_depth=64)


def test_safe_json_rejects_invalid_structure_and_constants() -> None:
    with pytest.raises(ValueError, match="JSON 对象"):
        loads_safe_json(b"[]")
    with pytest.raises(ValueError, match="非标准 JSON"):
        loads_safe_json(b'{"value": NaN}')
    with pytest.raises(ValueError, match="非有限数值"):
        loads_safe_json(b'{"value": 1e309}')


def test_finite_tree_checks_nested_numeric_values() -> None:
    assert_finite_tree({"a": [1, {"b": 2.5}]})
    with pytest.raises(ValueError, match="a\\[1\\]\\.b"):
        assert_finite_tree({"a": [1, {"b": float("inf")}]})


def test_deep_json_returns_400_and_server_remains_available(json_server) -> None:
    deep_body = "[" * 10_000 + "1" + "]" * 10_000
    status, body = _request(json_server, "/api/v1/backdoor", deep_body)
    payload = json.loads(body)

    assert status == 400
    assert payload["error"] == "invalid JSON"
    assert "trace_id" in payload
    health_status, health_body = _request(json_server, "/health")
    assert health_status == 200
    assert json.loads(health_body)["status"] == "alive"


def test_non_finite_request_returns_400_without_payload_echo(json_server) -> None:
    status, body = _request(json_server, "/api/v1/diagnose", '{"patient_id": "P001", "value": NaN}')
    payload = json.loads(body)

    assert status == 400
    assert payload["error"] == "invalid JSON"
    assert b"NaN" not in body
    assert "trace_id" in payload


def test_upstream_nan_does_not_emit_nonstandard_json(json_server, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        MCIAPIHandler,
        "_handle_backdoor",
        lambda self, body: {"ate": float("nan"), "method": "test"},
    )
    status, body = _request(json_server, "/api/v1/backdoor", "{}")
    payload = json.loads(body)

    assert status == 400
    assert payload["error"] == "invalid request"
    assert b"NaN" not in body


def test_send_json_serialization_failure_emits_safe_500(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCI_ENV", "test")
    counter = 'errors_total{endpoint="response_serialization"}'
    before = metrics._counters.get(counter, 0)
    handler = MagicMock()

    MCIAPIHandler._send_json(handler, 200, {"value": float("nan")})

    assert handler.send_response.call_args.args[0] == 500
    body = handler.wfile.write.call_args.args[0]
    assert json.loads(body) == {"error": "response serialization failed"}
    assert metrics._counters.get(counter, 0) == before + 1
