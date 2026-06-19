"""CachedDoCalculus — 带 LRU 缓存的 Do-Calculus 干预推理。

P3 "赋魂" 核心模块 — 在 DoCalculus 基础上增加组合键缓存，
目标：缓存命中时延迟 <5ms。

Usage::
    from mci_world_model.sdk._cached_do_calculus import CachedDoCalculus
    from mci_world_model.sdk._do_calculus import DoCalculus
    from mci_world_model.sdk._causal_graph import CausalGraph

    cg = CausalGraph(nodes=["Z","X","Y"], edges=[("Z","X"),("Z","Y"),("X","Y")])
    dc = DoCalculus(cg)
    cached = CachedDoCalculus(dc, maxsize=1024)
    result = cached.estimate_ate("X", "Y")
    print(cached.cache_info())  # hits, misses, size
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


def _stable_hash(*args: Any) -> str:
    """生成稳定的组合哈希键。"""
    raw = str(args).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


@dataclass
class CacheEntry:
    """缓存条目。"""

    key: str
    value: Any
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class CachedDoCalculus:
    """带 LRU 缓存的 Do-Calculus 包装器。

    Attributes:
        do_calculus: 底层 DoCalculus 实例。
        maxsize: 最大缓存条目数 (默认 1024)。
        _cache: OrderedDict (LRU — 最近使用的排在最后)。
        hits: 缓存命中次数。
        misses: 缓存未命中次数。
    """

    do_calculus: Any  # DoCalculus (avoid circular import)
    maxsize: int = 1024
    _cache: OrderedDict[str, CacheEntry] = field(default_factory=OrderedDict)
    hits: int = 0
    misses: int = 0

    def _get_cache_key(
        self, X: str, Y: str, Z_set: frozenset[str] | None = None
    ) -> str:
        """生成缓存键：graph_hash + X + Y + Z_set_hash。"""
        # 从 DoCalculus 获取因果图节点和边
        graph_repr = ""
        if hasattr(self.do_calculus, "causal_graph"):
            cg = self.do_calculus.causal_graph
            graph_repr = str(sorted(cg.nodes)) + str(sorted(str(e) for e in cg.edges))
        elif hasattr(self.do_calculus, "_graph"):
            cg = self.do_calculus._graph
            graph_repr = str(sorted(cg.nodes)) + str(sorted(str(e) for e in cg.edges))
        Z_key = str(sorted(Z_set)) if Z_set else "no_Z"
        return _stable_hash(graph_repr, X, Y, Z_key)

    def _get(self, key: str) -> Any | None:
        """从 LRU 缓存获取条目 (命中时移到末尾)。"""
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key].value
        self.misses += 1
        return None

    def _put(self, key: str, value: Any) -> None:
        """存入 LRU 缓存 (超过 maxsize 时淘汰最旧条目)。"""
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.maxsize:
                self._cache.popitem(last=False)  # LRU eviction
            self._cache[key] = CacheEntry(key=key, value=value)

    def estimate_ate(
        self, X: str, Y: str, Z_set: frozenset[str] | None = None
    ) -> Any:
        """带缓存的 ATE 估计。

        Args:
            X: 原因变量。
            Y: 结果变量。
            Z_set: 调整集 (可选)。

        Returns:
            InterventionResult (同 DoCalculus.estimate_ate)。
        """
        cache_key = self._get_cache_key(X, Y, Z_set)
        cached = self._get(cache_key)
        if cached is not None:
            return cached

        start = time.perf_counter()
        if Z_set is not None:
            result = self.do_calculus.backdoor_adjustment(
                X, Y, Z_set=list(Z_set)
            )
        else:
            result = self.do_calculus.estimate_ate(X, Y)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # 记录延迟 (供测试)
        if not hasattr(self, "_last_compute_ms"):
            self._last_compute_ms: float = 0.0  # type: ignore[assignment]
        self._last_compute_ms = elapsed_ms

        self._put(cache_key, result)
        return result

    def backdoor_adjustment(
        self, X: str, Y: str, Z_set: list[str]
    ) -> Any:
        """带缓存的 backdoor 调整。

        Args:
            X: 原因变量。
            Y: 结果变量。
            Z_set: 调整集。

        Returns:
            InterventionResult。
        """
        cache_key = self._get_cache_key(X, Y, frozenset(Z_set))
        cached = self._get(cache_key)
        if cached is not None:
            return cached

        start = time.perf_counter()
        result = self.do_calculus.backdoor_adjustment(X, Y, Z_set=Z_set)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._last_compute_ms = elapsed_ms

        self._put(cache_key, result)
        return result

    def cache_info(self) -> dict[str, int | float]:
        """返回缓存统计信息。

        Returns:
            dict: hits, misses, size, maxsize, hit_rate。
        """
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self._cache),
            "maxsize": self.maxsize,
            "hit_rate": self.hits / total if total > 0 else 0.0,
        }

    def clear(self) -> None:
        """清空缓存和统计。"""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._cache)


__all__ = ["CachedDoCalculus", "CacheEntry"]
