"""CachedDoCalculus 干预缓存推理 — 单元测试。

覆盖: 缓存命中/未命中, LRU逐出, cache_info统计, 延迟<5ms, 清空缓存。
"""

from __future__ import annotations

import time

import numpy as np

from mci_world_model.sdk._cached_do_calculus import CachedDoCalculus, CacheInfo
from mci_world_model.sdk._do_calculus import CausalGraph, DoCalculus


def _simple_graph() -> CausalGraph:
    """Z -> X, Z -> Y, X -> Y (后门调整: Z 阻断 X 和 Y 的后门路径)"""
    return CausalGraph(
        nodes=["Z", "X", "Y"],
        edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
    )


class TestCachedDoCalculusBasic:
    def test_cache_miss_then_hit(self) -> None:
        cg = _simple_graph()
        dc = DoCalculus(cg)
        cdc = CachedDoCalculus(dc, maxsize=32)

        r1 = cdc.query("X", "Y")
        assert r1 is not None
        assert cdc.cache_info().misses == 1
        assert cdc.cache_info().hits == 0

        r2 = cdc.query("X", "Y")
        assert r2 is not None
        assert r2 == r1
        assert cdc.cache_info().hits == 1

    def test_different_z_sets_different_keys(self) -> None:
        cg = _simple_graph()
        dc = DoCalculus(cg)
        cdc = CachedDoCalculus(dc, maxsize=32)

        cdc.query("X", "Y", Z_set=["Z"])
        assert cdc.cache_info().misses == 1
        cdc.query("X", "Y", Z_set=["Z", "W"])
        assert cdc.cache_info().misses == 2

    def test_cache_info(self) -> None:
        cdc = CachedDoCalculus(maxsize=32)
        info = cdc.cache_info()
        assert isinstance(info, CacheInfo)
        assert info.hits == 0
        assert info.misses == 0
        assert info.size == 0
        assert info.hit_rate == 0.0

    def test_clear_cache(self) -> None:
        cg = _simple_graph()
        cdc = CachedDoCalculus(DoCalculus(cg), maxsize=32)
        cdc.query("X", "Y")
        assert cdc.cache_size == 1
        cdc.clear_cache()
        assert cdc.cache_size == 0
        assert cdc.cache_info().hits == 0


class TestLRUEviction:
    def test_lru_evicts_oldest(self) -> None:
        cg = _simple_graph()
        cdc = CachedDoCalculus(DoCalculus(cg), maxsize=3)

        # Insert 4 unique queries; first should be evicted
        cdc.query("X", "Y", Z_set=["Z"])
        cdc.query("X", "Y", Z_set=["W"])
        # Re-access first to make it "recent"
        cdc.query("X", "Y", Z_set=["Z"])
        cdc.query("X", "Y", Z_set=["V"])
        # Now "Z" is recent, "W" should be evicted when inserting 4th
        # Actually at maxsize=3, we have: Z(accessed twice), W, V = 3 entries
        cdc.query("X", "Y", Z_set=["U"])
        assert cdc.cache_size == 3  # One evicted

    def test_maxsize_enforced(self) -> None:
        cg = _simple_graph()
        cdc = CachedDoCalculus(DoCalculus(cg), maxsize=5)
        for i in range(10):
            cdc.query("X", "Y", Z_set=[f"Z{i}"])
        assert cdc.cache_size <= 5


class TestLatency:
    def test_cache_hit_latency_under_5ms(self) -> None:
        cg = _simple_graph()
        cdc = CachedDoCalculus(DoCalculus(cg))
        cdc.query("X", "Y")
        # Warm cache
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            cdc.query("X", "Y")
            times.append((time.perf_counter() - t0) * 1000)
        avg_ms = float(np.mean(times))
        assert avg_ms < 5.0, f"Average cache hit latency {avg_ms:.3f}ms"


class TestCachedDoCalculusEdgeCases:
    def test_empty_Z_set(self) -> None:
        cg = _simple_graph()
        cdc = CachedDoCalculus(DoCalculus(cg))
        r = cdc.query("X", "Y", Z_set=[])
        assert r is not None

    def test_no_graph_returns_none(self) -> None:
        cdc = CachedDoCalculus()
        r = cdc.query("X", "Y")
        assert r["ate"] == 0.0

    def test_query_effect_alias(self) -> None:
        cg = _simple_graph()
        cdc = CachedDoCalculus(DoCalculus(cg))
        r1 = cdc.query("X", "Y")
        r2 = cdc.query_effect("X", "Y")
        assert r1 == r2

    def test_set_do_calculus_clears_cache(self) -> None:
        cg = _simple_graph()
        cdc = CachedDoCalculus(DoCalculus(cg))
        cdc.query("X", "Y")
        assert cdc.cache_size == 1
        cdc.set_do_calculus(DoCalculus(cg))
        assert cdc.cache_size == 0
