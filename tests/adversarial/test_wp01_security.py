"""WP-01 认证与限流身份治理测试。"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import stat
import tempfile

import pytest

from mci_world_model.server.security import (
    AuthConfig,
    IdentityRegistry,
    RateLimiter,
    TrustedProxies,
    resolve_client_ip,
    verify_auth,
)


def _write_identity_map(api_key: str, file_mode: int = 0o600) -> str:
    base = tempfile.mkdtemp(prefix="mci_wp01_identity_")
    path = os.path.join(base, "identity-map.json")
    key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    payload = {
        "version": 1,
        "identities": {
            key_hash: {
                "subject": "subject-1",
                "tenant_id": "tenant-1",
                "patient_ids": ["P001"],
            }
        },
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file)
    os.chmod(path, file_mode)
    os.chmod(base, 0o700)
    return path


class TestAuthConfig:
    def test_strict_config_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MCI_ENV", "production")
        monkeypatch.setenv("MCI_API_KEY", "")
        monkeypatch.delenv("MCI_AUTH_DISABLED", raising=False)
        config = AuthConfig.from_env()
        with pytest.raises(ValueError, match="MCI_API_KEY"):
            config.validate(strict=True)

    def test_disable_only_allowed_in_test(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MCI_ENV", "production")
        monkeypatch.setenv("MCI_AUTH_DISABLED", "true")
        monkeypatch.setenv("MCI_API_KEY", "")
        with pytest.raises(ValueError, match="MCI_ENV=test"):
            AuthConfig.from_env()

        monkeypatch.setenv("MCI_ENV", "test")
        config = AuthConfig.from_env()
        assert config.disabled is True


class TestIdentityRegistry:
    def test_load_and_resolve_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        api_key = "secure-key"
        path = _write_identity_map(api_key)
        registry = IdentityRegistry.from_path(path)
        identity = registry.resolve(api_key)
        assert identity is not None
        assert identity.subject == "subject-1"
        assert identity.tenant_id == "tenant-1"
        assert identity.patient_ids == frozenset({"P001"})

    def test_reject_wide_file_permissions(self) -> None:
        path = _write_identity_map("secure-key", file_mode=0o644)
        with pytest.raises(ValueError, match="权限过宽"):
            IdentityRegistry.from_path(path)

    def test_reject_duplicate_key_hash(self) -> None:
        api_key = "secure-key"
        path = _write_identity_map(api_key)
        key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        identity_entry = """{
          "subject": "subject-1",
          "tenant_id": "tenant-1",
          "patient_ids": ["P001"]
        }"""
        with open(path, "w", encoding="utf-8") as file:
            file.write(
                f'{{"version": 1, "identities": {{"{key_hash}": {identity_entry}, "{key_hash}": {identity_entry}}}}}'
            )
        os.chmod(path, 0o600)

        with pytest.raises(ValueError, match="重复键"):
            IdentityRegistry.from_path(path)


class TestVerifyAuth:
    def test_missing_key_fails_closed(self) -> None:
        config = AuthConfig(api_keys=set(), disabled=False)
        assert verify_auth({"X-API-Key": "anything"}, config) is False

    def test_valid_key_with_identity_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        api_key = "secure-key"
        monkeypatch.setenv("MCI_ENV", "production")
        monkeypatch.setenv("MCI_API_KEY", api_key)
        monkeypatch.setenv("MCI_IDENTITY_MAP_PATH", _write_identity_map(api_key))
        config = AuthConfig.from_env()
        config.validate(strict=True)
        assert verify_auth({"X-API-Key": api_key}, config) is True
        assert verify_auth({"X-API-Key": "wrong-key"}, config) is False


class TestProxyIdentity:
    def test_untrusted_proxy_header_is_ignored(self) -> None:
        headers = {"X-Forwarded-For": "203.0.113.9, 198.51.100.8"}
        proxies = TrustedProxies(networks=())
        assert resolve_client_ip(headers, "127.0.0.1", proxies) == "127.0.0.1"

    def test_trusted_proxy_uses_rightmost_forwarded_ip(self) -> None:
        headers = {
            "x-forwarded-for": "203.0.113.9, 198.51.100.8",
            "x-real-ip": "203.0.113.99",
        }
        proxies = TrustedProxies(networks=(ipaddress.ip_network("127.0.0.0/8"),))
        assert resolve_client_ip(headers, "127.0.0.1", proxies) == "198.51.100.8"

    def test_invalid_forwarded_value_falls_back(self) -> None:
        headers = {"x-forwarded-for": "not-an-ip"}
        proxies = TrustedProxies(networks=(ipaddress.ip_network("127.0.0.0/8"),))
        assert resolve_client_ip(headers, "127.0.0.1", proxies) == "127.0.0.1"

    def test_forged_headers_collapse_to_socket_bucket(self) -> None:
        proxies = TrustedProxies(networks=())
        limiter = RateLimiter(rate=0.0, burst=2)
        for index in range(10):
            headers = {"x-forwarded-for": f"203.0.113.{index}"}
            ip = resolve_client_ip(headers, "127.0.0.1", proxies)
            assert ip == "127.0.0.1"
        assert limiter.allow("127.0.0.1") is True
        assert limiter.allow("127.0.0.1") is True
        assert limiter.allow("127.0.0.1") is False


def test_identity_file_permissions_are_minimal() -> None:
    path = _write_identity_map("secure-key")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(os.path.dirname(path)).st_mode) == 0o700
