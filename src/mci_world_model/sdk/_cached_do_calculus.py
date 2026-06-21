"""MCI World Model v4.4.0 — CachedDoCalculus 干预推理缓存加速

对 DoCalculus 查询加 LRU 缓存层，组合键 = (graph_hash, X, Y, frozenset(Z))。

验收标准:
    - 缓存命中 < 5ms (仅 dict lookup)
    - LRU maxsize=1024, 逐出最旧条目
    - cache_info() 返回命中率统计
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from mci_world_model.sdk._do_calculus import DoCalculus

logger = logging.getLogger(__name__)


@dataclass
class CacheInfo:
    """缓存统计。"""

    hits: int = 0
    misses: int = 0
    size: int = 0
    maxsize: int = 1024

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CachedDoCalculus:
    """带 LRU 缓存的 DoCalculus 干预推理。

    缓存键: (graph_fingerprint, X, Y, Z_frozen)
    缓存值: dict (ATE/NDE/NIE 等结果)

    Example:
        >>> cdc = CachedDoCalculus(graph, data, maxsize=512)
        >>> result = cdc.query("X", "Y", Z_set=["Z1", "Z2"])
        >>> info = cdc.cache_info()
        >>> print(f"命中率: {info.hit_rate:.1%}")

    延迟目标: 缓存命中 < 5ms
    """

    def __init__(
        self,
        do_calculus: DoCalculus | None = None,
        maxsize: int = 1024,
    ):
        if maxsize <= 0:
            raise ValueError(f"maxsize 必须为正, 当前 {maxsize}")
        self._do = do_calculus or DoCalculus()
        self._maxsize = maxsize
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0

    # ── 公开 API ──

    def query(
        self,
        X: str,
        Y: str,
        Z_set: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """干预查询 — 先查缓存，未命中则计算并缓存。

        Args:
            X: 干预变量名
            Y: 目标变量名
            Z_set: 调整变量集 (可选, 不传则自动识别)

        Returns:
            {"ate": float, "method": str, "adjustment_set": [...], ...} 或 None
        """
        key = self._make_key(X, Y, Z_set or [])

        if key in self._cache:
            self._hits += 1
            # LRU: 移动到末尾
            self._cache.move_to_end(key)
            return self._cache[key]

        self._misses += 1
        result = self._compute(X, Y, Z_set)

        if result is not None:
            self._cache[key] = result
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)  # FIFO → LRU 逐出最旧

        return result

    def query_effect(
        self,
        X: str,
        Y: str,
        Z_set: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """query 的别名 (兼容 DoCalculus API)。"""
        return self.query(X, Y, Z_set)

    def cache_info(self) -> CacheInfo:
        """返回缓存统计信息。"""
        return CacheInfo(
            hits=self._hits,
            misses=self._misses,
            size=len(self._cache),
            maxsize=self._maxsize,
        )

    def clear_cache(self) -> None:
        """清空缓存，重置统计。"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def set_do_calculus(self, do_calculus: DoCalculus) -> None:
        """替换底层的 DoCalculus 实例 (同时清空缓存)。"""
        self._do = do_calculus
        self.clear_cache()

    # ── 属性 ──

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    # ── 内部方法 ──

    def _make_key(self, X: str, Y: str, Z_set: list[str]) -> str:
        """生成缓存键: graph_fp|X|Y|Z1,Z2,..."""
        graph_fp = self._graph_fingerprint()
        z_str = ",".join(sorted(Z_set))
        return f"{graph_fp}|{X}|{Y}|{z_str}"

    def _graph_fingerprint(self) -> str:
        """图的简短指纹 (基于 edges 的确定性哈希)。"""
        graph = self._do._graph
        if graph is None:
            return "no_graph"
        # 确定性排序 + MD5 指纹
        edges_str = ",".join(
            sorted(f"{a}->{b}" for a, b in graph.edges)
        ) if hasattr(graph, "adjacency") else str(id(graph))
        h = hashlib.md5(edges_str.encode()).hexdigest()[:8]
        return f"g{h}"

    def _compute(
        self, X: str, Y: str, Z_set: list[str] | None
    ) -> dict[str, Any] | None:
        """底层 DoCalculus 计算 (无缓存)。"""
        try:
            result = self._do.estimate_ate(X, Y)
            if result is None:
                return None
            return {
                "ate": result.ate if hasattr(result, "ate") else None,
                "method": result.method if hasattr(result, "method") else "auto",
                "adjustment_set": result.adjustment_set if hasattr(result, "adjustment_set") else [],
            }
        except Exception:
            logger.debug("DoCalculus compute failed for %s -> %s", X, Y, exc_info=True)
            return None
