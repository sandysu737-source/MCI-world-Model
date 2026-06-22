"""tests/test_dense_retriever.py — zvec DenseRetriever tests"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("zvec", reason="zvec not installed")

from mci_world_model.sdk._dense_retriever import (
    DenseRetriever,
)
from mci_world_model.sdk._experience_memory import Experience, ExperienceType


class TestDenseRetriever:
    """zvec dense + hybrid search."""

    @pytest.fixture
    def retriever(self):
        dr = DenseRetriever(dim=64)
        exps = []
        np.random.seed(42)
        for i in range(30):
            tags = [f"topic_{i % 5}", f"type_{i % 3}", "causal", "inference"]
            if i < 10:
                tags.extend(["rare", f"code_{i:03d}"])
            exp = Experience(
                experience_id=f"exp_{i}", tags=tags, causal_edges=[],
                experience_type=ExperienceType.PREDICTION, importance=1.0,
                timestamp=time.time() - i * 3600,
            )
            exps.append(exp)
        dr.index(exps)
        return dr

    def test_index_count(self, retriever):
        assert retriever.count == 30

    def test_vector_search(self, retriever):
        r = retriever.search("topic_0 causal", top_k=5)
        assert len(r.results) == 5
        assert r.method == "vector"
        assert r.latency_ms >= 0

    def test_fts_search(self, retriever):
        r = retriever.fts_search("code_005", top_k=3)
        assert len(r.results) == 3
        assert r.method == "fts"
        # "code_005" should match exp_5
        assert "exp_5" in r.top_ids()

    def test_hybrid_search(self, retriever):
        r = retriever.hybrid_search("topic_0", keywords="code_003", top_k=5)
        assert len(r.results) == 5
        assert r.method == "hybrid"

    def test_empty_index(self):
        dr = DenseRetriever(dim=32)
        assert dr.count == 0
        n = dr.index([])
        assert n == 0

    def test_top_ids(self, retriever):
        r = retriever.search("topic_1", top_k=3)
        ids = r.top_ids()
        assert len(ids) == 3
        assert all(isinstance(x, str) for x in ids)

    def test_deterministic_vector(self, retriever):
        """Same text produces same vector."""
        r1 = retriever.search("test query", top_k=3)
        r2 = retriever.search("test query", top_k=3)
        assert r1.top_ids() == r2.top_ids()  # deterministic

    def test_latency_reasonable(self, retriever):
        r = retriever.search("causal inference", top_k=3)
        assert r.latency_ms < 50, f"Latency {r.latency_ms:.1f}ms > 50ms"
