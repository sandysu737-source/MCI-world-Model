"""SuMemoryBridge — MCI World Model ↔ su-memory-sdk 集成桥接。

将 MCI World Model 的因果推理结果写入 su-memory-sdk 记忆引擎，
并从记忆引擎检索因果上下文以增强推理质量。

Usage::
    from adapters.su_memory_bridge import SuMemoryBridge

    bridge = SuMemoryBridge()
    bridge.store_experience(state_dict, causal_result)
    context = bridge.query_causal_context("天气 → 心情")
    print(bridge.health_check())
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Probe ──────────────────────────────────────────────────────────────
_SU_MEMORY_AVAILABLE = False
_SuMemory = None
try:
    from su_memory.client import SuMemory as _SuMemoryClass

    _SuMemory = _SuMemoryClass
    _SU_MEMORY_AVAILABLE = True
except ImportError:
    logger.debug("su_memory.client 不可用，_SU_MEMORY_AVAILABLE=False")


@dataclass
class BridgeRecord:
    """桥接记录 — 单次因果经验写入。"""

    query_key: str
    state_snapshot: dict[str, Any]
    result_summary: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class SuMemoryBridge:
    """MCI World Model → su-memory-sdk 桥接器。

    提供记忆引擎的读/写接口，将因果推理经验持久化到 su-memory-sdk。

    Attributes:
        memory_dir: su-memory 数据目录 (None 使用默认)。
        _client: SuMemory 客户端实例 (延迟初始化)。
        _records: 本地记录缓存。
        _bm25: BM25 检索器引用。
    """

    memory_dir: str | None = None
    _client: Any = field(default=None, repr=False, init=False)
    _records: list[BridgeRecord] = field(default_factory=list, repr=False)
    _bm25: Any = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        if not _SU_MEMORY_AVAILABLE:
            self._client = None

    # ── Lifecycle ───────────────────────────────────────────────────────

    def _ensure_client(self) -> Any:
        """延迟初始化 su-memory 客户端。"""
        if self._client is not None:
            return self._client
        if not _SU_MEMORY_AVAILABLE or _SuMemory is None:
            return None
        kwargs: dict[str, Any] = {}
        if self.memory_dir is not None:
            kwargs["memory_dir"] = self.memory_dir
        self._client = _SuMemory(**kwargs)
        return self._client

    @property
    def is_available(self) -> bool:
        """su-memory-sdk 是否可用。"""
        return _SU_MEMORY_AVAILABLE

    # ── Write ───────────────────────────────────────────────────────────

    def store_experience(
        self,
        state: dict[str, Any],
        causal_result: Any,
        *,
        query_key: str | None = None,
    ) -> bool:
        """将一次因果推理经验写入记忆引擎。

        Args:
            state: MCI 世界状态字典 (如 to_dict() 输出)。
            causal_result: 因果推理结果 (ATE / CounterfactualResult 等)。
            query_key: 检索关键词 (默认为因果图中变量连接)。

        Returns:
            True 如果写入成功。
        """
        # 生成检索关键词
        if query_key is None:
            keys: list[str] = []
            if isinstance(state, dict):
                keys.extend(str(k) for k in state if "causal" in str(k).lower())
            if hasattr(causal_result, "ate"):
                keys.append(f"ATE={getattr(causal_result, 'ate', 0):.3f}")
            query_key = " | ".join(keys) if keys else "causal_experience"

        # 序列化结果
        result_summary: dict[str, Any] = {"type": type(causal_result).__name__}
        if hasattr(causal_result, "ate"):
            result_summary["ate"] = float(getattr(causal_result, "ate", 0.0))
        if hasattr(causal_result, "method"):
            result_summary["method"] = str(getattr(causal_result, "method", ""))
        if hasattr(causal_result, "to_dict"):
            try:
                result_summary["detail"] = causal_result.to_dict()
            except Exception:
                logger.debug("causal_result.to_dict() 失败，detail 保持默认")

        # 存储到 su-memory
        client = self._ensure_client()
        if client is not None:
            try:
                # 使用 BM25 添加文档
                from su_memory.sdk._bm25 import BM25Searcher

                if self._bm25 is None:
                    self._bm25 = BM25Searcher()
                doc_text = json.dumps(
                    {"state": state, "result": result_summary},
                    ensure_ascii=False,
                    default=str,
                )
                self._bm25.add(query_key, doc_text)
            except Exception:
                logger.debug("bm25 添加失败，跳过该条记忆")

        # 本地缓存
        record = BridgeRecord(
            query_key=query_key,
            state_snapshot=state if isinstance(state, dict) else {"state": str(state)},
            result_summary=result_summary,
        )
        self._records.append(record)
        return True

    # ── Read ────────────────────────────────────────────────────────────

    def query_causal_context(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """从记忆引擎检索因果上下文。

        Args:
            query: 自然语言查询。
            top_k: 返回结果数。

        Returns:
            相关经验列表 (按相关性降序)。
        """
        results: list[dict[str, Any]] = []

        # su-memory BM25 检索
        if self._bm25 is not None:
            try:
                hits = self._bm25.search(query, top_k=top_k)
                for doc_id, score in hits:
                    results.append({"source": "bm25", "id": doc_id, "score": float(score)})
            except Exception:
                logger.debug("bm25 检索失败，回退到空结果")

        # 本地缓存回退
        if not results:
            for record in self._records[-top_k:]:
                results.append(
                    {
                        "source": "local_cache",
                        "query_key": record.query_key,
                        "result": record.result_summary,
                        "timestamp": record.timestamp,
                    }
                )

        return results

    # ── Health ──────────────────────────────────────────────────────────

    def health_check(self) -> dict[str, Any]:
        """健康检查 — 确认桥接器状态。"""
        client = self._ensure_client()
        status: dict[str, Any] = {
            "bridge": "su_memory",
            "available": self.is_available,
            "client_ok": client is not None,
            "bm25_ok": self._bm25 is not None,
            "records_cached": len(self._records),
        }
        if client is not None:
            try:
                status["embedding_dim"] = client.embedding_dim()
            except Exception as e:
                status["embedding_dim"] = str(e)
        return status

    # ── Maintenance ─────────────────────────────────────────────────────

    def clear_cache(self) -> None:
        """清空本地缓存 (不删除 su-memory 持久化数据)。"""
        self._records.clear()
        if self._bm25 is not None:
            self._bm25.clear()

    def __len__(self) -> int:
        return len(self._records)


__all__ = ["SuMemoryBridge", "BridgeRecord"]
