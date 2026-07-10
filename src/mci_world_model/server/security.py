"""API 安全层 — 认证、限流、TLS 支持。

零第三方依赖，使用标准库实现。
生产环境建议前置 Nginx/Traefik 做 TLS 终止和 L7 限流，
本模块提供应用层的纵深防御。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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

    @classmethod
    def from_env(cls) -> AuthConfig:
        keys_str = os.environ.get("MCI_API_KEY", "")
        keys = {k.strip() for k in keys_str.split(",") if k.strip()}
        disabled = os.environ.get("MCI_AUTH_DISABLED", "").lower() == "true"
        return cls(api_keys=keys, disabled=disabled)


def verify_auth(headers: dict[str, str], config: AuthConfig) -> bool:
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
        logger.warning("MCI_API_KEY 未配置, 认证被跳过")
        return True

    # 规范化 header 为小写 (HTTP header 大小写不敏感)
    lower_headers = {k.lower(): v for k, v in headers.items()}

    # Bearer token
    auth_header = lower_headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if token and token in config.api_keys:
            return True

    # X-API-Key
    api_key = lower_headers.get("x-api-key", "")
    return bool(api_key and api_key in config.api_keys)


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


def get_circuit_breaker() -> CircuitBreaker:
    """线程安全的断路器单例。"""
    global _circuit_breaker
    if _circuit_breaker is not None:
        return _circuit_breaker
    with _singleton_lock:
        if _circuit_breaker is None:
            _circuit_breaker = CircuitBreaker()
    return _circuit_breaker
