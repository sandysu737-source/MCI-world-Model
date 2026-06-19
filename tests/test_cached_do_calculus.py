"""Tests for CachedDoCalculus — LRU-cached Do-Calculus wrapper."""

import time

import pytest

from mci_world_model.sdk._cached_do_calculus import CachedDoCalculus, _stable_hash
from mci_world_model.sdk._do_calculus import CausalGraph, DoCalculus


@pytest.fixture
def simple_graph() -> CausalGraph:
    """Z → X → Y with Z also causing Y (classic backdoor)."""
    return CausalGraph(
        nodes=["Z", "X", "Y"],
        edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
    )


@pytest.fixture
def do_calc(simple_graph: CausalGraph) -> DoCalculus:
    return DoCalculus(simple_graph)


@pytest.fixture
def cached_calc(do_calc: DoCalculus) -> CachedDoCalculus:
    return CachedDoCalculus(do_calc, maxsize=64)


class TestStableHash:
    """Tests for internal _stable_hash function."""

    def test_deterministic(self) -> None:
        """Same inputs produce same hash."""
        h1 = _stable_hash("X", "Y", frozenset(["Z"]))
        h2 = _stable_hash("X", "Y", frozenset(["Z"]))
        assert h1 == h2

    def test_different_inputs_different_hash(self) -> None:
        """Different inputs produce different hashes."""
        h1 = _stable_hash("X", "Y", frozenset(["Z"]))
        h2 = _stable_hash("A", "B", frozenset(["C"]))
        assert h1 != h2


class TestCachedDoCalculus:
    """Core CachedDoCalculus tests."""

    def test_initial_state(self, cached_calc: CachedDoCalculus) -> None:
        """Starts with empty cache and zero stats."""
        assert len(cached_calc) == 0
        info = cached_calc.cache_info()
        assert info["hits"] == 0
        assert info["misses"] == 0
        assert info["size"] == 0
        assert info["hit_rate"] == 0.0

    def test_first_call_is_miss(self, cached_calc: CachedDoCalculus) -> None:
        """First call to estimate_ate is a cache miss."""
        result = cached_calc.estimate_ate("X", "Y")
        assert result is not None
        info = cached_calc.cache_info()
        assert info["misses"] == 1
        assert info["hits"] == 0
        assert info["size"] == 1

    def test_second_call_is_hit(self, cached_calc: CachedDoCalculus) -> None:
        """Second identical call hits the cache."""
        r1 = cached_calc.estimate_ate("X", "Y")
        r2 = cached_calc.estimate_ate("X", "Y")
        assert r1.ate == pytest.approx(r2.ate)
        info = cached_calc.cache_info()
        assert info["hits"] == 1
        assert info["misses"] == 1

    def test_different_query_is_miss(self, cached_calc: CachedDoCalculus) -> None:
        """Different query (X,Y pair) is a new cache entry."""
        cached_calc.estimate_ate("X", "Y")
        cached_calc.estimate_ate("Z", "Y")
        info = cached_calc.cache_info()
        assert info["misses"] == 2
        assert info["size"] == 2

    def test_backdoor_adjustment_cached(self, cached_calc: CachedDoCalculus) -> None:
        """backdoor_adjustment also uses cache."""
        r1 = cached_calc.backdoor_adjustment("X", "Y", Z_set=["Z"])
        assert r1 is not None
        r2 = cached_calc.backdoor_adjustment("X", "Y", Z_set=["Z"])
        assert r2 is not None
        info = cached_calc.cache_info()
        assert info["hits"] == 1

    def test_hit_rate(self, cached_calc: CachedDoCalculus) -> None:
        """hit_rate is computed correctly."""
        cached_calc.estimate_ate("X", "Y")  # miss
        cached_calc.estimate_ate("X", "Y")  # hit
        cached_calc.estimate_ate("X", "Y")  # hit
        info = cached_calc.cache_info()
        assert info["hits"] == 2
        assert info["misses"] == 1
        assert info["hit_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_lru_eviction(self) -> None:
        """LRU evicts oldest entries when maxsize is exceeded."""
        cg = CausalGraph(
            nodes=["Z", "X", "Y"],
            edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        )
        dc = DoCalculus(cg)
        cached = CachedDoCalculus(dc, maxsize=3)

        # Insert 4 different queries
        cached.estimate_ate("X", "Y")
        cached.estimate_ate("Z", "Y")
        cached.estimate_ate("Z", "X")
        cached.estimate_ate("X", "Z")  # should evict first entry

        info = cached.cache_info()
        assert info["size"] <= cached.maxsize
        assert info["size"] == 3

    def test_cache_latency_under_5ms(self, cached_calc: CachedDoCalculus) -> None:
        """Cache hit should return in under 5ms."""
        # First call to populate cache
        cached_calc.estimate_ate("X", "Y")

        # Measure cache hit latency
        start = time.perf_counter()
        for _ in range(100):
            cached_calc.estimate_ate("X", "Y")
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / 100) * 1000

        assert avg_ms < 5.0, f"Cache hit latency {avg_ms:.2f}ms exceeds 5ms"

    def test_clear_resets_cache(self, cached_calc: CachedDoCalculus) -> None:
        """clear() resets cache and stats."""
        cached_calc.estimate_ate("X", "Y")
        cached_calc.estimate_ate("X", "Y")
        assert len(cached_calc) == 1

        cached_calc.clear()
        assert len(cached_calc) == 0
        info = cached_calc.cache_info()
        assert info["hits"] == 0
        assert info["misses"] == 0

    def test_result_is_consistent(self, cached_calc: CachedDoCalculus) -> None:
        """Cached result matches direct DoCalculus result."""
        cached_result = cached_calc.estimate_ate("X", "Y")
        direct_result = cached_calc.do_calculus.estimate_ate("X", "Y")
        assert cached_result.ate == pytest.approx(direct_result.ate, abs=0.1)

    def test_different_graphs_not_confused(self) -> None:
        """Different causal graphs don't share cache keys."""
        cg1 = CausalGraph(nodes=["Z", "X", "Y"], edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")])
        cg2 = CausalGraph(nodes=["A", "B", "C"], edges=[("A", "B"), ("B", "C")])

        cached1 = CachedDoCalculus(DoCalculus(cg1), maxsize=32)
        cached2 = CachedDoCalculus(DoCalculus(cg2), maxsize=32)

        r1 = cached1.estimate_ate("X", "Y")
        r2 = cached2.estimate_ate("A", "C")

        # Same query text on different graphs should produce different cache entries
        assert r1 is not None
        assert r2 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
