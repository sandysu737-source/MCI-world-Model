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
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
        self._lock = threading.Lock()

    def _safe_path(self, key: str) -> Path:
        """H1 修复: 防.路径遍历 — key 净化 + 解析后必须在 base 内。"""
        # 净化 key: 只保留安全字符
        safe_key = self._SAFE_KEY.sub("_", key)
        path = (self._base / f"{safe_key}.json").resolve()
        # 二次校验: 解析后路径必须在 base 目录内
        if not str(path).startswith(str(self._base)):
            raise ValueError(f"非法存储 key: {key!r}")
        return path

    def save(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            path = self._safe_path(key)
            # H8 修复: 原子写入 — 先写临时文件再 rename
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(value, f, ensure_ascii=False)
            tmp.rename(path)

    def load(self, key: str) -> dict[str, Any] | None:
        path = self._safe_path(key)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def list_keys(self, prefix: str = "") -> list[str]:
        return [p.stem for p in self._base.glob("*.json") if p.stem.startswith(prefix)]


class RedisStorage(StorageBackend):
    """Redis 持久化 (可选, 需 redis-py)。"""

    def __init__(self, redis_url: str) -> None:
        try:
            import redis

            self._client = redis.from_url(redis_url, decode_responses=True)
            self._client.ping()
            logger.info("Redis 存储连接成功: %s", redis_url)
        except ImportError:
            raise ImportError("redis-py 未安装: pip install redis") from None
        except Exception as e:
            raise ConnectionError(f"Redis 连接失败: {e}") from e

    def save(self, key: str, value: dict[str, Any]) -> None:
        self._client.set(f"mci:{key}", json.dumps(value, ensure_ascii=False))

    def load(self, key: str) -> dict[str, Any] | None:
        raw = self._client.get(f"mci:{key}")
        if raw is None:
            return None
        return json.loads(raw)

    def list_keys(self, prefix: str = "") -> list[str]:
        keys = self._client.keys(f"mci:{prefix}*")
        return [k.replace("mci:", "") for k in keys]


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
        path = os.environ.get("MCI_STORAGE_PATH", "/tmp/mci-data")
        _storage = FileStorage(path)
    return _storage


def save_diagnosis(patient_id: str, diagnosis: dict[str, Any]) -> str:
    """保存诊断结果, 返回记录 ID。"""
    import secrets

    # H3 修复: 加随机后缀防碰撞 + patient_id 净化
    safe_pid = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(patient_id))
    record_id = f"diagnosis:{safe_pid}:{int(time.time() * 1000)}:{secrets.token_hex(4)}"
    get_storage().save(
        record_id,
        {
            **diagnosis,
            "patient_id": patient_id,
            "timestamp": time.time(),
        },
    )
    return record_id


def load_diagnosis(record_id: str) -> dict[str, Any] | None:
    """加载诊断结果。"""
    return get_storage().load(record_id)


def list_diagnoses(patient_id: str = "") -> list[str]:
    """列出诊断记录 ID。"""
    prefix = f"diagnosis:{patient_id}" if patient_id else "diagnosis:"
    return get_storage().list_keys(prefix)
