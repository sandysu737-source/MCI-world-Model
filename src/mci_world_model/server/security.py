"""API 安全层 — 认证、限流、TLS 支持。

零第三方依赖，使用标准库实现。
生产环境建议前置 Nginx/Traefik 做 TLS 终止和 L7 限流，
本模块提供应用层的纵深防御。
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import os
import secrets
import stat
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Identity:
    """认证 key 对应的服务端身份。"""

    subject: str
    tenant_id: str
    patient_ids: frozenset[str]


class IdentityRegistry:
    """基于 SHA-256(key) 的身份映射。

    映射文件不保存明文 key；文件权限必须不宽于 0600，目录不宽于 0700。
    """

    def __init__(self, identities: dict[str, Identity], source_path: str | None = None) -> None:
        self._identities = identities
        self._source_path = source_path

    @classmethod
    def from_path(cls, path: str | os.PathLike[str]) -> IdentityRegistry:
        """从受控 JSON 文件加载身份映射。"""
        map_path = Path(path)
        if not map_path.is_file():
            raise ValueError(f"身份映射文件不存在: {map_path}")
        dir_mode = stat.S_IMODE(map_path.parent.stat().st_mode)
        file_mode = stat.S_IMODE(map_path.stat().st_mode)
        if dir_mode & 0o077:
            raise ValueError(f"身份映射目录权限过宽: {oct(dir_mode)}")
        if file_mode & 0o077:
            raise ValueError(f"身份映射文件权限过宽: {oct(file_mode)}")

        try:
            payload = json.loads(map_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"身份映射文件无效: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("身份映射文件版本不受支持")
        raw_identities = payload.get("identities")
        if not isinstance(raw_identities, dict):
            raise ValueError("身份映射缺少 identities")

        identities: dict[str, Identity] = {}
        for key_hash, raw_identity in raw_identities.items():
            if not isinstance(key_hash, str) or len(key_hash) != 64:
                raise ValueError("身份映射的 key hash 必须是 64 位 SHA-256")
            try:
                bytes.fromhex(key_hash)
            except ValueError as exc:
                raise ValueError("身份映射的 key hash 不是合法十六进制") from exc
            if not isinstance(raw_identity, dict):
                raise ValueError("身份映射条目必须是对象")
            subject = raw_identity.get("subject")
            tenant_id = raw_identity.get("tenant_id")
            patient_ids = raw_identity.get("patient_ids", [])
            if not isinstance(subject, str) or not subject.strip():
                raise ValueError("身份映射缺少 subject")
            if not isinstance(tenant_id, str) or not tenant_id.strip():
                raise ValueError("身份映射缺少 tenant_id")
            if not isinstance(patient_ids, list) or not all(
                isinstance(item, str) and item.strip() for item in patient_ids
            ):
                raise ValueError("身份映射的 patient_ids 必须是非空字符串列表")
            identities[key_hash.lower()] = Identity(
                subject=subject,
                tenant_id=tenant_id,
                patient_ids=frozenset(patient_ids),
            )
        if not identities:
            raise ValueError("身份映射不能为空")
        return cls(identities=identities, source_path=str(map_path))

    def resolve(self, api_key: str) -> Identity | None:
        """解析 key 对应身份；未注册 key 返回 None。"""
        key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        return self._identities.get(key_hash)

    @property
    def identities(self) -> dict[str, Identity]:
        """返回身份映射只读视图。"""
        return dict(self._identities)


# =============================================================================
# 认证
# =============================================================================


@dataclass
class AuthConfig:
    """认证配置。

    通过环境变量配置:
        MCI_API_KEY: API Key (逗号分隔多个)
        MCI_AUTH_DISABLED: 设为 "true" 禁用认证 (仅开发)
    """

    api_keys: set[str] = field(default_factory=set)
    disabled: bool = False
    env: str = "production"
    identity_registry: IdentityRegistry | None = None

    @classmethod
    def from_env(cls) -> AuthConfig:
        env = os.environ.get("MCI_ENV", "production").strip().lower()
        if env not in {"production", "development", "test"}:
            raise ValueError(f"非法 MCI_ENV: {env}")
        keys_str = os.environ.get("MCI_API_KEY", "")
        keys = {k.strip() for k in keys_str.split(",") if k.strip()}
        disabled_raw = os.environ.get("MCI_AUTH_DISABLED", "").strip().lower()
        if disabled_raw and disabled_raw not in {"true", "false"}:
            raise ValueError("MCI_AUTH_DISABLED 只允许 true/false")
        disabled = disabled_raw == "true"
        registry: IdentityRegistry | None = None
        registry_path = os.environ.get("MCI_IDENTITY_MAP_PATH", "").strip()
        if registry_path:
            registry = IdentityRegistry.from_path(registry_path)
        config = cls(
            api_keys=keys,
            disabled=disabled,
            env=env,
            identity_registry=registry,
        )
        config.validate(strict=False)
        return config

    def validate(self, strict: bool = False) -> None:
        """校验认证配置；strict 用于服务启动路径。"""
        if self.disabled:
            if self.env != "test":
                raise ValueError("MCI_AUTH_DISABLED=true 只允许 MCI_ENV=test")
            return
        if strict and not self.api_keys:
            raise ValueError("MCI_API_KEY 未配置，认证流量必须 fail closed")
        if strict and not self.identity_registry:
            raise ValueError("MCI_IDENTITY_MAP_PATH 未配置，认证流量必须 fail closed")


def verify_auth(headers: Any, config: AuthConfig) -> bool:
    """验证 API 请求认证。

    支持两种方式:
        Authorization: Bearer <api_key>
        X-API-Key: <api_key>

    HTTP header 大小写不敏感 (规范化为小写后查找)。

    Args:
        headers: HTTP 请求头
        config: 认证配置

    Returns:
        True 如果认证通过
    """
    if config.disabled:
        return True
    if not config.api_keys:
        return False

    # 规范化 header 为小写 (HTTP header 大小写不敏感)
    lower_headers = {k.lower(): v for k, v in headers.items()}

    # Bearer token
    auth_header = lower_headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if token and any(secrets.compare_digest(token, key) for key in config.api_keys):
            return True

    # X-API-Key
    api_key = lower_headers.get("x-api-key", "")
    if not api_key or not any(secrets.compare_digest(api_key, key) for key in config.api_keys):
        return False
    return config.identity_registry is not None and config.identity_registry.resolve(api_key) is not None


@dataclass(frozen=True)
class TrustedProxies:
    """可信代理网段集合；未命中时忽略全部转发头。"""

    networks: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]

    @classmethod
    def from_env(cls) -> TrustedProxies:
        raw = os.environ.get("MCI_TRUSTED_PROXIES", "")
        networks = []
        for item in raw.split(","):
            value = item.strip()
            if value:
                networks.append(ipaddress.ip_network(value, strict=False))
        return cls(tuple(networks))

    def contains(self, address: str | None) -> bool:
        if not address:
            return False
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            return False
        return any(parsed in network for network in self.networks)


def resolve_client_ip(
    headers: Any,
    remote_addr: str | None,
    trusted_proxies: TrustedProxies,
) -> str:
    """仅信任直接连接的可信代理，并取转发链最右端有效地址。"""
    if not trusted_proxies.contains(remote_addr):
        return remote_addr or "unknown"
    lower_headers = {str(name).lower(): value for name, value in headers.items()}
    forwarded = str(lower_headers.get("x-forwarded-for", "")).strip()
    if forwarded:
        for candidate in reversed([part.strip() for part in forwarded.split(",")]):
            try:
                ipaddress.ip_address(candidate)
            except ValueError:
                continue
            return candidate
    real_ip = str(lower_headers.get("x-real-ip", "")).strip()
    try:
        ipaddress.ip_address(real_ip)
    except ValueError:
        return remote_addr or "unknown"
    return real_ip


# =============================================================================
# 限流 (Token Bucket per-IP)
# =============================================================================


@dataclass
class _TokenBucket:
    """令牌桶: 每秒填充 rate 个令牌, 容量 burst。"""

    rate: float  # 令牌/秒
    burst: int  # 桶容量
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.time)

    def try_consume(self, now: float, n: int = 1) -> bool:
        """尝试消费 n 个令牌, 返回是否成功。"""
        elapsed = now - self.last_refill
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_refill = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False


class RateLimiter:
    """线程安全的 per-IP 限流器。

    Args:
        rate: 每秒允许的请求数
        burst: 突发容量
        max_ips: 追踪的最大 IP 数 (LRU 淘汰)
    """

    def __init__(self, rate: float = 10.0, burst: int = 20, max_ips: int = 10000) -> None:
        self._rate = rate
        self._burst = burst
        self._max_ips = max_ips
        self._buckets: dict[str, _TokenBucket] = {}
        self._lock = threading.Lock()

    def allow(self, client_ip: str) -> bool:
        """检查该 IP 是否被允许请求。"""
        now = time.time()
        with self._lock:
            if client_ip not in self._buckets:
                # LRU 淘汰: 超过上限时清除最早的
                if len(self._buckets) >= self._max_ips:
                    oldest = next(iter(self._buckets))
                    del self._buckets[oldest]
                self._buckets[client_ip] = _TokenBucket(
                    rate=self._rate,
                    burst=self._burst,
                    tokens=float(self._burst),
                    last_refill=now,
                )
            return self._buckets[client_ip].try_consume(now)

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


# =============================================================================
# 断路器
# =============================================================================


class CircuitBreaker:
    """断路器: 连续失败超过阈值时熔断, 保护下游。

    状态:
        CLOSED: 正常, 请求通过
        OPEN: 熔断, 请求直接拒绝
        HALF_OPEN: 探测, 允许少量请求测试恢复
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max: int = 3,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max = half_open_max
        self._state = self.CLOSED
        self._failures = 0
        self._last_failure_time = 0.0
        self._half_open_count = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.OPEN:
                # 检查是否该转为 half-open
                if time.time() - self._last_failure_time > self._recovery_timeout:
                    self._state = self.HALF_OPEN
                    self._half_open_count = 0
            return self._state

    def allow(self) -> bool:
        """是否允许请求通过。"""
        state = self.state
        if state == self.CLOSED:
            return True
        if state == self.HALF_OPEN:
            with self._lock:
                if self._half_open_count < self._half_open_max:
                    self._half_open_count += 1
                    return True
            return False
        return False  # OPEN

    def record_success(self) -> None:
        with self._lock:
            if self._state == self.HALF_OPEN:
                self._state = self.CLOSED
                logger.info("断路器恢复: HALF_OPEN → CLOSED")
            self._failures = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()
            if self._state == self.HALF_OPEN:
                self._state = self.OPEN
                logger.warning("断路器熔断: HALF_OPEN → OPEN (探测失败)")
            elif self._failures >= self._failure_threshold:
                self._state = self.OPEN
                logger.warning("断路器熔断: CLOSED → OPEN (连续失败 %d 次)", self._failures)


# 全局实例
_auth_config: AuthConfig | None = None
_rate_limiter: RateLimiter | None = None
_circuit_breaker: CircuitBreaker | None = None
_singleton_lock = threading.Lock()


def get_auth_config() -> AuthConfig:
    """线程安全的认证配置单例。"""
    global _auth_config
    if _auth_config is not None:
        return _auth_config
    with _singleton_lock:
        if _auth_config is None:  # double-check
            _auth_config = AuthConfig.from_env()
            _auth_config.validate(strict=True)
    return _auth_config


def get_rate_limiter() -> RateLimiter:
    """线程安全的限流器单例。"""
    global _rate_limiter
    if _rate_limiter is not None:
        return _rate_limiter
    with _singleton_lock:
        if _rate_limiter is None:
            rate = float(os.environ.get("MCI_RATE_LIMIT", "10"))
            burst = int(os.environ.get("MCI_RATE_BURST", "20"))
            _rate_limiter = RateLimiter(rate=rate, burst=burst)
    return _rate_limiter


_trusted_proxies: TrustedProxies | None = None


def get_trusted_proxies() -> TrustedProxies:
    """线程安全的可信代理解析单例。"""
    global _trusted_proxies
    if _trusted_proxies is not None:
        return _trusted_proxies
    with _singleton_lock:
        if _trusted_proxies is None:
            _trusted_proxies = TrustedProxies.from_env()
    return _trusted_proxies


def get_circuit_breaker() -> CircuitBreaker:
    """线程安全的断路器单例。"""
    global _circuit_breaker
    if _circuit_breaker is not None:
        return _circuit_breaker
    with _singleton_lock:
        if _circuit_breaker is None:
            _circuit_breaker = CircuitBreaker()
    return _circuit_breaker
