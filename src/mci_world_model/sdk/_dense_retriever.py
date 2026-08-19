"""_dense_retriever.py — zvec-backed dense semantic retriever.

Provides vector + FTS + hybrid search over causal experience text.
Complements the TF-IDF semantic view in MultiViewRetriever.

Usage:
    >>> retriever = DenseRetriever(dim=128)
    >>> retriever.index(experiences)          # bulk index
    >>> results = retriever.hybrid_search(    # hybrid: vector + FTS
    ...     query_text="causal edge X->Y",
    ...     keywords=["X", "Y"],
    ...     top_k=5,
    ... )
"""

from __future__ import annotations

import atexit
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import zvec

    HAS_ZVEC = True
except ImportError:
    HAS_ZVEC = False
    logger.warning("zvec not installed; DenseRetriever disabled. Install: pip install zvec")


@dataclass
class DenseSearchResult:
    """Single dense search result."""

    doc_id: str
    score: float
    content: str = ""
    rank: int = 0


@dataclass
class DenseSearchResults:
    """Batch dense search results."""

    results: list[DenseSearchResult] = field(default_factory=list)
    query_text: str = ""
    method: str = "vector"  # vector, fts, hybrid
    latency_ms: float = 0.0

    def top_ids(self) -> list[str]:
        return [r.doc_id for r in self.results]


class DenseRetriever:
    """zvec-backed dense + sparse + hybrid semantic search.

    Features:
      - Vector search (cosine similarity on pre-computed embeddings)
      - FTS (BM25 full-text search on tag/description text)
      - Hybrid (vector + FTS with weighted re-ranking)

    Embedding strategy:
      - Tags → concatenated text → hash-based pseudo-embedding (128-d)
      - Replace with real embedding model (e.g., BGE-small) for production
    """

    def __init__(self, dim: int = 128, db_path: str | None = None):
        if not HAS_ZVEC:
            raise ImportError("zvec required: pip install zvec")
        self._dim = dim
        self._db_path = db_path or str(Path(tempfile.gettempdir()) / f"mci_dense_{id(self)}")
        self._col = self._create_collection()
        self._indexed_count = 0
        atexit.register(self._cleanup)

    # ── Indexing ──

    def index(self, experiences: list[Any]) -> int:
        """Index a batch of Experience objects.

        Each experience's tags are joined into text; a pseudo-embedding
        is derived from a deterministic hash of the text.
        """
        if not experiences:
            return 0

        docs = []
        rng = np.random.RandomState(42)
        for exp in experiences:
            text = " ".join(exp.tags) if hasattr(exp, "tags") else str(exp)
            vec = self._text_to_vector(text, rng)
            docs.append(
                zvec.Doc(
                    id=exp.experience_id if hasattr(exp, "experience_id") else str(id(exp)),
                    fields={"content": text},
                    vectors={"vec": vec.tolist()},
                )
            )

        self._col.insert(docs)
        self._col.flush()
        self._indexed_count += len(docs)
        return len(docs)

    # ── Search ──

    def search(self, query_text: str, top_k: int = 5) -> DenseSearchResults:
        """Vector similarity search."""
        import time

        t0 = time.perf_counter()
        rng = np.random.RandomState(hash(query_text) % (2**31))
        q_vec = self._text_to_vector(query_text, rng)
        q = zvec.Query(
            field_name="vec",
            vector=q_vec.tolist(),
            param=zvec.HnswQueryParam(ef=max(100, top_k * 20)),
        )
        results = self._col.query(q, topk=top_k, output_fields=["content"])
        elapsed = (time.perf_counter() - t0) * 1000
        return self._to_results(results, query_text, "vector", elapsed)

    def fts_search(self, query_text: str, top_k: int = 5) -> DenseSearchResults:
        """BM25 full-text search."""
        import time

        t0 = time.perf_counter()
        q = zvec.Query(field_name="content", fts=zvec.Fts(query_string=query_text))
        results = self._col.query(q, topk=top_k, output_fields=["content"])
        elapsed = (time.perf_counter() - t0) * 1000
        return self._to_results(results, query_text, "fts", elapsed)

    def hybrid_search(
        self,
        query_text: str,
        keywords: str | None = None,
        top_k: int = 5,
        vector_weight: float = 0.6,
    ) -> DenseSearchResults:
        """Hybrid vector + FTS with weighted re-ranking."""
        import time

        t0 = time.perf_counter()

        rng = np.random.RandomState(hash(query_text) % (2**31))
        q_vec_vec = self._text_to_vector(query_text, rng)

        q_vec = zvec.Query(
            field_name="vec",
            vector=q_vec_vec.tolist(),
            param=zvec.HnswQueryParam(ef=max(100, top_k * 20)),
        )
        fts_text = keywords if keywords else query_text
        q_fts = zvec.Query(field_name="content", fts=zvec.Fts(query_string=fts_text))

        fw = max(0.01, min(0.99, vector_weight))
        results = self._col.query(
            [q_vec, q_fts],
            topk=top_k,
            output_fields=["content"],
            reranker=zvec.WeightedReRanker(weights=[fw, 1.0 - fw]),
        )
        elapsed = (time.perf_counter() - t0) * 1000
        return self._to_results(results, query_text, "hybrid", elapsed)

    # ── Helpers ──

    def _text_to_vector(self, text: str, rng: np.random.RandomState) -> np.ndarray:
        """Deterministic pseudo-embedding from text hash."""
        # Use hash to seed the RNG for deterministic vectors
        seed = hash(text) % (2**31)
        local_rng = np.random.RandomState(seed)
        vec = local_rng.randn(self._dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-10)

    def _create_collection(self):
        schema = zvec.CollectionSchema(
            name="mci_dense",
            fields=[zvec.FieldSchema("content", zvec.DataType.STRING)],
            vectors=[zvec.VectorSchema("vec", zvec.DataType.VECTOR_FP32, dimension=self._dim)],
        )
        col = zvec.create_and_open(path=self._db_path, schema=schema)
        col.create_index("vec", zvec.HnswIndexParam(m=16, ef_construction=200))
        col.create_index("content", zvec.FtsIndexParam())
        col.flush()
        return col

    def _to_results(self, doclist, query_text: str, method: str, elapsed_ms: float) -> DenseSearchResults:
        results = []
        for rank, doc in enumerate(doclist, 1):
            content = ""
            if doc.fields and "content" in doc.fields:
                content = str(doc.fields["content"])
            results.append(
                DenseSearchResult(
                    doc_id=doc.id,
                    score=doc.score,
                    content=content,
                    rank=rank,
                )
            )
        return DenseSearchResults(
            results=results,
            query_text=query_text,
            method=method,
            latency_ms=elapsed_ms,
        )

    @property
    def count(self) -> int:
        return self._indexed_count

    def _cleanup(self):
        try:
            if hasattr(self, "_col") and self._col:
                self._col.destroy()
        except Exception:
            logger.warning("吞异常", exc_info=True)
