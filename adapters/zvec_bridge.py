"""ZVecBridge — MCI World Model ↔ zvec 进程内向量数据库。

zvec 0.5.0 (阿里开源) — 纯进程内、零服务依赖、仅 numpy 依赖的向量数据库。
完美契合 MCI World Model 的「CPU-first + 零外部依赖」原则。

集成场景:
    - ExperienceDB 后端替换: 持久化 + ANN 检索
    - JEPA 嵌入存储: 大规模嵌入离线查询
    - MultiViewRetriever: HNSW 近似检索替代 O(n) 线性扫描

Usage::
    from adapters.zvec_bridge import ZVecBridge

    bridge = ZVecBridge(path="/tmp/mci_vectors")
    bridge.insert(vectors, ids=["s0", "s1"])
    results = bridge.search(query_vector, top_k=10)
    print(bridge.health_check())
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ── Probe ──────────────────────────────────────────────────────────────
_ZVEC_AVAILABLE = False
_zvec = None
try:
    import zvec as _zvec_mod

    _zvec = _zvec_mod
    _ZVEC_AVAILABLE = True
except ImportError:
    pass


@dataclass
class SearchHit:
    """单条检索结果。"""

    id: str
    score: float
    rank: int


@dataclass
class ZVecBridge:
    """MCI World Model → zvec 桥接器。

    提供进程内向量数据库的读写/检索接口，zvec 不可用时无损降级为本地缓存。

    Attributes:
        path: zvec 数据目录路径。
        dim: 向量维度 (首次写入时自动推断)。
        metric: 距离度量 (cosine / l2 / ip)。
        _col: zvec Collection 实例。
    """

    path: str = "mci_zvec_data"
    dim: int = 128
    metric: str = "cosine"
    _col: Any = field(default=None, repr=False, init=False)
    _fallback_vectors: list[np.ndarray] = field(default_factory=list, repr=False)
    _fallback_ids: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not _ZVEC_AVAILABLE:
            self._col = None
            return
        # zvec handles its own directory creation

    # ── Lifecycle ───────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        return _ZVEC_AVAILABLE and _zvec is not None

    def _ensure_open(self, dim: int | None = None) -> bool:
        """延迟打开或创建 zvec Collection。"""
        assert _zvec is not None  # guarded by is_available
        if not self.is_available:
            return False
        if self._col is not None:
            return True
        assert _zvec is not None

        d = dim or self.dim
        metric_map = {"cosine": _zvec.MetricType.COSINE, "l2": _zvec.MetricType.L2, "ip": _zvec.MetricType.IP}
        metric_type = metric_map.get(self.metric, _zvec.MetricType.COSINE)

        try:
            # 尝试打开已有
            self._col = _zvec.open(self.path)
        except Exception:
            # 新建
            schema = _zvec.CollectionSchema(
                name="mci_vectors",
                vectors=[
                    _zvec.VectorSchema(
                        name="vec",
                        data_type=_zvec.DataType.VECTOR_FP32,
                        dimension=d,
                        index_param=_zvec.FlatIndexParam(metric_type=metric_type),
                    )
                ],
            )
            self._col = _zvec.create_and_open(self.path, schema)
            # 创建 HNSW 索引
            try:
                self._col.create_index(
                    "vec",
                    _zvec.HnswIndexParam(metric_type=metric_type, m=16, ef_construction=100),
                )
            except Exception:
                pass

        # 迁移 fallback 数据
        if self._fallback_vectors:
            self.insert(self._fallback_vectors, ids=self._fallback_ids)
            self._fallback_vectors.clear()
            self._fallback_ids.clear()

        return True

    # ── Write ───────────────────────────────────────────────────────────

    def insert(
        self,
        vectors: np.ndarray | list[np.ndarray],
        ids: list[str] | None = None,
    ) -> list[str]:
        """插入向量。

        Args:
            vectors: (n, dim) 数组或向量列表。
            ids: 主键列表 (自动生成 UUID 如果为空)。

        Returns:
            已插入的 id 列表。
        """
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        n = vecs.shape[0]

        if ids is None:
            ids = [uuid.uuid4().hex[:12] for _ in range(n)]

        # 降级: 本地缓存
        if not self.is_available or not self._ensure_open(dim=vecs.shape[1]):
            self._fallback_vectors.extend(list(vecs))
            self._fallback_ids.extend(ids)
            return ids

        # zvec 写入
        assert _zvec is not None
        docs = [
            _zvec.Doc(id=ids[i], vectors={"vec": vecs[i].tolist()})
            for i in range(n)
        ]
        self._col.insert(docs)
        self._col.flush()
        return ids

    def upsert(
        self,
        vectors: np.ndarray,
        ids: list[str],
    ) -> list[str]:
        """插入或更新向量 (id 存在则覆盖)。"""
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        n = vecs.shape[0]

        if not self.is_available or not self._ensure_open(dim=vecs.shape[1]):
            return self.insert(vecs, ids=ids)

        assert _zvec is not None
        docs = [
            _zvec.Doc(id=ids[i], vectors={"vec": vecs[i].tolist()})
            for i in range(n)
        ]
        self._col.upsert(docs)
        self._col.flush()
        return ids

    # ── Read ────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        ef: int = 20,
    ) -> list[SearchHit]:
        """ANN 向量检索。

        Args:
            query_vector: (dim,) 查询向量。
            top_k: 返回结果数。
            ef: HNSW ef_search 参数 (越大越精确)。

        Returns:
            SearchHit 列表 (按分数排序)。
        """
        q = np.asarray(query_vector, dtype=np.float32).ravel()

        # 降级: numpy 线性扫描
        if not self.is_available or self._col is None:
            return self._fallback_search(q, top_k)

        try:
            assert _zvec is not None
            results = self._col.query([
                _zvec.Query(
                    field_name="vec",
                    vector=q.tolist(),
                    param=_zvec.HnswQueryParam(ef=ef),
                )
            ])
            hits = []
            for i, doc in enumerate(results[:top_k]):
                hits.append(SearchHit(id=str(doc.id), score=float(doc.score), rank=i))
            return hits
        except Exception:
            return self._fallback_search(q, top_k)

    def _fallback_search(
        self, query: np.ndarray, top_k: int
    ) -> list[SearchHit]:
        """numpy 余弦相似度线性扫描 (降级方案)。"""
        if not self._fallback_vectors:
            return []
        mat = np.asarray(self._fallback_vectors)
        # L2 normalize
        q_norm = query / (np.linalg.norm(query) + 1e-12)
        mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        scores = mat_norm @ q_norm
        top_indices = np.argsort(-scores)[:top_k]
        return [
            SearchHit(
                id=self._fallback_ids[i],
                score=float(scores[i]),
                rank=rank,
            )
            for rank, i in enumerate(top_indices)
        ]

    # ── Delete ──────────────────────────────────────────────────────────

    def delete(self, ids: list[str]) -> int:
        """按 id 删除向量。

        Returns:
            删除数量。
        """
        if self.is_available and self._col is not None:
            try:
                self._col.delete(ids)
                return len(ids)
            except Exception:
                pass
        # Fallback
        removed = 0
        for vid in ids:
            if vid in self._fallback_ids:
                idx = self._fallback_ids.index(vid)
                self._fallback_ids.pop(idx)
                self._fallback_vectors.pop(idx)
                removed += 1
        return removed

    # ── Health ──────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        if self.is_available and self._col is not None:
            try:
                return self._col.stats.doc_count
            except Exception:
                pass
        return len(self._fallback_ids)

    def health_check(self) -> dict[str, Any]:
        return {
            "bridge": "zvec",
            "available": self.is_available,
            "path": self.path,
            "dim": self.dim,
            "metric": self.metric,
            "count": self.count,
            "fallback_active": not (self.is_available and self._col is not None),
        }

    def destroy(self) -> None:
        """销毁 zvec 数据。"""
        if self.is_available and self._col is not None:
            try:
                self._col.destroy()
            except Exception:
                pass
        self._col = None
        self._fallback_vectors.clear()
        self._fallback_ids.clear()


__all__ = ["ZVecBridge", "SearchHit"]
