"""状态外置化 — 证据和诊断结果持久化。

抽象存储接口, 默认 JSON 文件持久化 (零依赖),
可选 Redis (需安装 redis-py)。

环境变量:
    MCI_STORAGE_BACKEND=file|redis
    MCI_STORAGE_PATH=/path/to/data  (file 模式)
    MCI_REDIS_URL=redis://localhost:6379  (redis 模式)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from mci_world_model.server.security import Identity

logger = logging.getLogger(__name__)

DEFAULT_REDIS_SCAN_PAGE_SIZE = 100
DEFAULT_REDIS_MAX_SCAN_KEYS = 10_000
DEFAULT_REDIS_MAX_SCAN_SECONDS = 1.0


class StorageBackend(ABC):
    """存储后端抽象接口。"""

    @abstractmethod
    def save(self, key: str, value: dict[str, Any]) -> None:
        """保存一条记录。"""

    @abstractmethod
    def load(self, key: str) -> dict[str, Any] | None:
        """加载一条记录。"""

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]:
        """列出匹配前缀的 key。"""


class FileStorage(StorageBackend):
    """JSON 文件持久化 (零依赖, 默认)。"""

    # key 允许的字符: 字母、数字、冒号、下划线、短横线
    import re as _re

    _SAFE_KEY = _re.compile(r"[^a-zA-Z0-9:_-]")

    def __init__(self, base_path: str) -> None:
        self._base = Path(base_path).resolve()
        self._base.mkdir(parents=True, exist_ok=True)
        self._base.chmod(0o700)
        self._lock = threading.Lock()

    def _safe_path(self, key: str) -> Path:
        """H1 修复: 防.路径遍历 — key 净化 + 解析后必须在 base 内。"""
        if not key or self._SAFE_KEY.search(key):
            raise ValueError(f"非法存储 key: {key!r}")
        path = (self._base / f"{key}.json").absolute()
        # 二次校验: 解析后路径必须在 base 目录内
        try:
            path.relative_to(self._base)
        except ValueError as exc:
            raise ValueError(f"非法存储 key: {key!r}") from exc
        return path

    def save(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            path = self._safe_path(key)
            if path.is_symlink():
                raise ValueError(f"非法存储路径: {key!r}")
            # H8 修复: 原子写入 — 先写临时文件再 rename
            tmp = path.with_suffix(".tmp")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(tmp, flags, 0o600)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as file:
                    json.dump(value, file, ensure_ascii=False)
                os.replace(tmp, path)
            except Exception:
                tmp.unlink(missing_ok=True)
                raise

    def load(self, key: str) -> dict[str, Any] | None:
        path = self._safe_path(key)
        if path.is_symlink():
            raise ValueError(f"非法存储路径: {key!r}")
        if not path.exists():
            return None
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        with os.fdopen(fd, encoding="utf-8") as file:
            return json.load(file)

    def list_keys(self, prefix: str = "") -> list[str]:
        return [p.stem for p in self._base.glob("*.json") if p.stem.startswith(prefix)]


class RedisStorage(StorageBackend):
    """Redis 持久化 (可选, 需 redis-py)。"""

    def __init__(
        self,
        redis_url: str,
        scan_page_size: int = DEFAULT_REDIS_SCAN_PAGE_SIZE,
        max_scan_keys: int = DEFAULT_REDIS_MAX_SCAN_KEYS,
        max_scan_seconds: float = DEFAULT_REDIS_MAX_SCAN_SECONDS,
    ) -> None:
        if scan_page_size <= 0 or max_scan_keys <= 0 or max_scan_seconds <= 0:
            raise ValueError("Redis SCAN 预算必须为正数")
        try:
            import redis

            self._client = redis.from_url(redis_url, decode_responses=True)
            self._client.ping()
            self._scan_page_size = scan_page_size
            self._max_scan_keys = max_scan_keys
            self._max_scan_seconds = max_scan_seconds
            logger.info("Redis 存储连接成功: %s", redis_url)
        except ImportError:
            raise ImportError("redis-py 未安装: pip install redis") from None
        except Exception as e:
            raise ConnectionError(f"Redis 连接失败: {e}") from e

    def save(self, key: str, value: dict[str, Any], ttl_seconds: int | None = None) -> None:
        redis_key = f"mci:{key}"
        payload = json.dumps(value, ensure_ascii=False)
        if ttl_seconds is not None:
            self._client.setex(redis_key, ttl_seconds, payload)
        else:
            self._client.set(redis_key, payload)

    def load(self, key: str) -> dict[str, Any] | None:
        raw = self._client.get(f"mci:{key}")
        if raw is None:
            return None
        return json.loads(raw)

    def list_keys(self, prefix: str = "") -> list[str]:
        """用带预算的 SCAN 替代 KEYS；超过预算返回部分结果并告警。"""
        escaped_prefix = prefix.replace("\\", "\\\\").replace("*", "\\*").replace("?", "\\?").replace("[", "\\[")
        pattern = f"mci:{escaped_prefix}*"
        deadline = time.monotonic() + self._max_scan_seconds
        keys: list[str] = []
        partial_reason: str | None = None

        for key in self._client.scan_iter(match=pattern, count=self._scan_page_size):
            if len(keys) >= self._max_scan_keys:
                partial_reason = f"max_scan_keys={self._max_scan_keys}"
                break
            if time.monotonic() >= deadline:
                partial_reason = f"max_scan_seconds={self._max_scan_seconds}"
                break
            if key.startswith("mci:"):
                keys.append(key[len("mci:") :])

        if partial_reason is not None:
            logger.warning(
                "Redis SCAN 超过预算，返回部分结果: prefix=%s, returned=%s, %s",
                prefix,
                len(keys),
                partial_reason,
            )
        return keys


_storage: StorageBackend | None = None


_storage_lock = threading.Lock()


def get_storage() -> StorageBackend:
    """获取全局存储实例 (线程安全单例)。"""
    global _storage
    if _storage is not None:
        return _storage
    with _storage_lock:
        if _storage is not None:  # double-check
            return _storage
        backend = os.environ.get("MCI_STORAGE_BACKEND", "file")
    if backend == "redis":
        redis_url = os.environ.get("MCI_REDIS_URL", "redis://localhost:6379")
        _storage = RedisStorage(redis_url)
    else:
        path = os.environ.get("MCI_STORAGE_PATH", "").strip()
        if not path:
            raise ValueError("MCI_STORAGE_PATH 必须显式配置")
        if os.environ.get("MCI_ENV", "production") != "test" and _is_system_temp_path(path):
            raise ValueError("MCI_STORAGE_PATH 不能位于系统临时目录")
        _storage = FileStorage(path)
    return _storage


def reset_storage() -> None:
    """清空全局存储缓存，测试每用例可按当前环境重建。"""
    global _storage
    with _storage_lock:
        _storage = None


def _is_system_temp_path(path: str) -> bool:
    """判断显式存储路径是否落在系统临时目录内。"""
    candidate = Path(path).expanduser().resolve()
    roots = {
        Path(tempfile.gettempdir()).resolve(),
        Path("/tmp").resolve(),
        Path("/var/tmp").resolve(),
        Path("/private/tmp").resolve(),
        Path("/private/var/tmp").resolve(),
    }
    return any(candidate == root or root in candidate.parents for root in roots)


def save_diagnosis(patient_id: str, diagnosis: dict[str, Any], auth_context: Identity) -> str:
    """保存诊断结果, 返回记录 ID。"""
    import secrets

    if not patient_id or patient_id not in auth_context.patient_ids:
        raise PermissionError("patient_id 不在授权范围内")
    # H3 修复: 加随机后缀防碰撞 + patient_id 净化
    safe_pid = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(patient_id))
    record_id = f"diagnosis:{auth_context.tenant_id}:{safe_pid}:{int(time.time() * 1000)}:{secrets.token_hex(4)}"
    get_storage().save(
        record_id,
        {
            **diagnosis,
            "patient_id": patient_id,
            "tenant_id": auth_context.tenant_id,
            "owner_subject": auth_context.subject,
            "timestamp": time.time(),
        },
    )
    return record_id


def load_diagnosis(record_id: str, auth_context: Identity) -> dict[str, Any] | None:
    """加载诊断结果。"""
    record = get_storage().load(record_id)
    if record is None:
        return None
    if record.get("tenant_id") != auth_context.tenant_id:
        return None
    if record.get("patient_id") not in auth_context.patient_ids:
        return None
    return record


def list_diagnoses(patient_id: str, auth_context: Identity) -> list[str]:
    """列出诊断记录 ID。"""
    if not patient_id or patient_id not in auth_context.patient_ids:
        return []
    safe_pid = "".join(c if c.isalnum() or c in "-_" else "_" for c in patient_id)
    prefix = f"diagnosis:{auth_context.tenant_id}:{safe_pid}:"
    return get_storage().list_keys(prefix)
