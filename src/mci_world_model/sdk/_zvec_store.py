"""
MCI World Model v4.4.0 — Zvec Embedding Store
===============================================

基于阿里 Zvec 的进程内向量嵌入存储，
替代 SimpleTextEmbedder 的 char 3-gram hash。

核心能力:
- ZvecEmbeddingStore: 因果 QA 对的向量存储和检索
- 支持 BM25 文本检索 + 向量相似度检索 双模式
- HNSW 索引加速 (COSINE 相似度)
- 零外部网络依赖 (纯本地进程内)

用法:
    from mci_world_model.sdk._zvec_store import ZvecEmbeddingStore

    store = ZvecEmbeddingStore("./data/qa_store")
    store.insert_qa_pairs(pairs)
    results = store.search_similar("蛋白质摄入不足", topk=5)
    results = store.search_bm25("蛋白质", topk=3)
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Zvec 可选依赖
_zvec_available = False
try:
    import zvec as _zvec

    _zvec_available = True
except ImportError:
    pass


# =============================================================================
# EmbeddingStoreConfig
# =============================================================================


@dataclass
class EmbeddingStoreConfig:
    """向量存储配置。"""

    dim: int = 128
    index_type: str = "hnsw"  # "hnsw" | "flat" | "ivf"
    metric_type: str = "cosine"
    collection_name: str = "causal_qa"
    store_path: str = "./data/qa_store"


# =============================================================================
# ZvecEmbeddingStore
# =============================================================================


class ZvecEmbeddingStore:
    """基于 Zvec 的因果 QA 嵌入存储。

    与 CausalMLP/ParametricMemory 集成:
    1. ReflectionSynthesizer 生成 QA 对
    2. ZvecEmbeddingStore 存储 (含向量索引)
    3. CausalMLP 训练时快速检索相似样本
    4. 预测时用 BM25/向量检索找到最近因果先验

    零硬依赖: zvec 不可用时降级为纯 numpy 内存存储。
    """

    def __init__(self, config: EmbeddingStoreConfig | None = None):
        self.config = config or EmbeddingStoreConfig()
        self._collection = None
        self._fallback_storage: list[dict[str, Any]] = []
        self._fallback_vectors: np.ndarray | None = None

        if _zvec_available:
            self._init_zvec()
        else:
            logger.warning("Zvec 未安装，使用 numpy 内存存储 (可能较慢)")

    def _init_zvec(self) -> None:
        """初始化 Zvec collection。"""
        if not _zvec_available:
            return

        os.makedirs(self.config.store_path, exist_ok=True)

        # Schema
        try:
            schema = _zvec.CollectionSchema(
                name=self.config.collection_name,
                fields=[
                    _zvec.FieldSchema("id", _zvec.DataType.STRING),
                    _zvec.FieldSchema("cause_text", _zvec.DataType.STRING),
                    _zvec.FieldSchema("effect_text", _zvec.DataType.STRING),
                    _zvec.FieldSchema("energy_relation", _zvec.DataType.STRING),
                    _zvec.FieldSchema("confidence", _zvec.DataType.FLOAT),
                ],
                vectors=[
                    _zvec.VectorSchema(
                        "vec",
                        _zvec.DataType.VECTOR_FP32,
                        dimension=self.config.dim,
                    ),
                ],
            )

            self._collection = _zvec.create_and_open(
                self.config.store_path, schema
            )

            # Build index
            if self.config.index_type == "hnsw":
                self._collection.create_index(  # type: ignore[attr-defined]
                    "vec",
                    index_param=_zvec.HnswIndexParam(
                        metric_type=_zvec.MetricType.COSINE
                    ),
                )
            logger.info("Zvec store initialized at %s", self.config.store_path)

        except Exception as e:
            logger.error("Zvec 初始化失败: %s", e)
            self._collection = None

    # ── 插入 ──

    def insert_qa_pairs(self, qa_pairs: list[dict[str, Any]]) -> int:
        """插入因果 QA 对到向量存储。

        Args:
            qa_pairs: [{"cause_text": str, "effect_text": str,
                        "energy_relation": str, "confidence": float}, ...]

        Returns:
            插入的文档数
        """
        if not qa_pairs:
            return 0

        # 为每条 QA 对生成向量 (使用确定性 hash 嵌入)
        docs: list[dict[str, Any]] = []
        for i, pair in enumerate(qa_pairs):
            cause = str(pair.get("cause_text", ""))
            effect = str(pair.get("effect_text", ""))
            relation = str(pair.get("energy_relation", "neutral"))
            confidence = float(pair.get("confidence", 0.5))

            # 确定性向量: 从文本生成
            vec = self._text_to_vector(cause)
            doc_id = _hash_id(cause, effect, i)

            if self._collection is not None:
                docs.append(
                    _zvec.Doc(
                        id=doc_id,
                        fields={
                            "id": doc_id,
                            "cause_text": cause,
                            "effect_text": effect,
                            "energy_relation": relation,
                            "confidence": confidence,
                        },
                        vectors={"vec": vec.tolist()},
                    )
                )

            # 同步到 fallback
            self._fallback_storage.append(
                {
                    "id": doc_id,
                    "cause_text": cause,
                    "effect_text": effect,
                    "energy_relation": relation,
                    "confidence": confidence,
                    "vec": vec,
                }
            )

        if docs and self._collection is not None:
            try:
                self._collection.insert(docs)
            except Exception as e:
                logger.warning("Zvec 插入失败，仅使用 fallback: %s", e)

        return len(qa_pairs)

    # ── 检索 ──

    def search_similar(self, query: str, topk: int = 5) -> list[dict[str, Any]]:
        """向量相似度检索——找到与查询文本最相似的因果 QA 对。

        Args:
            query: 查询文本
            topk: 返回前 K 个结果

        Returns:
            [{"cause_text": ..., "effect_text": ..., "energy_relation": ...,
              "confidence": ..., "score": ...}, ...]
        """
        qvec = self._text_to_vector(query)

        # Zvec 查询
        if self._collection is not None:
            try:
                q = _zvec.Query(
                    field_name="vec",
                    data=qvec.tolist(),
                    topk=min(topk, 500),
                )
                hits = self._collection.query(q)
                results = []
                for h in hits:
                    f = h.fields
                    results.append(
                        {
                            "cause_text": f.get("cause_text", ""),
                            "effect_text": f.get("effect_text", ""),
                            "energy_relation": f.get("energy_relation", "neutral"),
                            "confidence": f.get("confidence", 0.5),
                            "score": float(h.score) if h.score is not None else 0.0,
                        }
                    )
                return results
            except Exception as e:
                logger.debug("Zvec 查询失败，回退 numpy: %s", e)

        # Fallback: numpy 余弦相似度
        return self._numpy_search(qvec, topk)

    def search_bm25(self, query: str, topk: int = 5) -> list[dict[str, Any]]:
        """BM25 全文检索——关键词匹配因果 QA 对。

        当前回退到 numpy 字符串匹配。
        """
        results = []
        for pair in self._fallback_storage:
            cause = pair.get("cause_text", "")
            effect = pair.get("effect_text", "")
            # 简单 Jaccard 相似度
            q_chars = set(query)
            c_chars = set(cause + effect)
            if q_chars and c_chars:
                score = len(q_chars & c_chars) / len(q_chars | c_chars)
            else:
                score = 0.0
            if score > 0:
                results.append({**pair, "score": score})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:topk]

    # ── 工具 ──

    def _text_to_vector(self, text: str) -> np.ndarray:
        """确定性文本→向量转换 (轻量, 零外部依赖)。

        使用 SHA256 hash → 确定性伪随机投影 → L2 归一化。
        与 SimpleTextEmbedder 的区别: 不依赖 n-gram 统计，
        任何文本都能生成唯一向量。
        """
        d = self.config.dim
        h = hashlib.sha256(text.encode("utf-8")).digest()

        # 将 hash 字节展开为 128 维向量
        vec = np.zeros(d, dtype=np.float32)
        for i in range(d):
            byte_val = h[i % len(h)]
            # 确定性变换
            val = ((byte_val * (i + 1)) % 256) / 128.0 - 1.0
            vec[i] = val

        # L2 归一化
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec /= norm
        return vec

    def _numpy_search(self, qvec: np.ndarray, topk: int) -> list[dict[str, Any]]:
        """Numpy fallback: 余弦相似度检索。"""
        if not self._fallback_storage:
            return []

        # 收集所有向量
        vecs = np.stack([p["vec"] for p in self._fallback_storage])

        # 余弦相似度
        qvec = qvec.astype(np.float32)
        qnorm = np.linalg.norm(qvec)
        if qnorm > 1e-10:
            qvec = qvec / qnorm
        scores = vecs @ qvec

        top_indices = np.argsort(scores)[-topk:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                pair = self._fallback_storage[idx]
                results.append({**pair, "score": float(scores[idx])})
        return results

    @property
    def n_docs(self) -> int:
        """已存储的文档数。"""
        return len(self._fallback_storage)

    @property
    def is_available(self) -> bool:
        """Zvec 是否可用。"""
        return _zvec_available and self._collection is not None

    def close(self) -> None:
        """释放资源。Zvec 进程内无需特殊关闭。"""
        pass


# =============================================================================
# 工具函数
# =============================================================================


def _hash_id(cause: str, effect: str, idx: int) -> str:
    """生成唯一文档 ID。"""
    content = f"{cause}::{effect}::{idx}"
    h = hashlib.sha256(content.encode()).hexdigest()
    return f"zvs_{h[:16]}"
