"""WP-12 验证口径与全局状态隔离测试。"""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.contract


def test_auth_config_supports_injection_and_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    security = importlib.import_module("mci_world_model.server.security")
    monkeypatch.setenv("MCI_ENV", "test")
    monkeypatch.setenv("MCI_AUTH_DISABLED", "true")
    security.reset_auth_config()

    injected = security.AuthConfig(api_keys=set(), disabled=True, env="test")
    assert security.set_auth_config(injected) is injected
    assert security.get_auth_config() is injected

    security.reset_auth_config()
    assert security.get_auth_config() is not injected


def test_storage_reset_forces_environment_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    storage = importlib.import_module("mci_world_model.server.storage")
    monkeypatch.setenv("MCI_ENV", "test")
    monkeypatch.delenv("MCI_STORAGE_PATH", raising=False)
    storage.reset_storage()

    with pytest.raises(ValueError, match="MCI_STORAGE_PATH"):
        storage.get_storage()


def test_runtime_state_reset() -> None:
    app = importlib.import_module("mci_world_model.server.app")
    security = importlib.import_module("mci_world_model.server.security")
    storage = importlib.import_module("mci_world_model.server.storage")

    security.get_auth_config()
    security.get_rate_limiter()
    security.get_trusted_proxies()
    security.get_circuit_breaker()
    storage._storage = object()  # type: ignore[assignment]

    app._ready = True
    with app._active_requests_lock:
        app._active_requests = 2
    app.MCIAPIHandler._request_count = 3
    app.reset_runtime_state()

    assert app._ready is False
    assert app._active_requests == 0
    assert app.MCIAPIHandler._request_count == 0
    assert security._auth_config is None
    assert security._rate_limiter is None
    assert security._trusted_proxies is None
    assert security._circuit_breaker is None
    assert storage._storage is None
