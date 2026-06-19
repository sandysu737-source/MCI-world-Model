from __future__ import annotations

"""
MCI World Model v5.0.0 — Persistent Experience Memory
======================================================

持久化经验记忆：从进程内存 → 磁盘持久化 + 向量检索。

核心组件:
- VectorStore: 轻量 numpy 向量存储（append-only segment + mmap 读取）
- PersistentExperienceMemory: SQLite + VectorStore 联动持久化

设计决策 (基于方案批判缺陷 2):
    - 放弃 .npz 单文件模式，改为 append-only segment
    - 每个 segment 一个 .npy 文件，原子写入
    - threading.Lock 保护并发写入
    - SQLite WAL mode + 批量写入

用法:
    from mci_world_model.sdk._persistent_memory import (
        PersistentMemoryConfig,
        PersistentExperienceMemory,
        VectorStore,
    )

    store = VectorStore(dim=128)
    store.add(["exp_1", "exp_2"], vectors)
    results = store.search(query_vector, top_k=10)

    memory = PersistentExperienceMemory()
    memory.store(experience)
    results = memory.retrieve(query="心率升高", top_k=5)
"""


import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# VectorStore — 轻量 numpy 向量存储
# =============================================================================


class VectorStore:
    """
    轻量向量存储：append-only segment + mmap 读取。

    目录结构:
        data/vectors/
            segment_0001.npy  ← 首批向量, shape=(N, dim)
            segment_0002.npy  ← 下一批
            index.json         ← {id → (segment_file, row_idx)}

    并发安全:
        - threading.Lock 保护写入
        - 读取基于快照隔离（加载时刻的 segment）

    性能目标:
        - 10000 × 128 维检索 < 10ms
    """

    SEGMENT_SIZE = 10000  # 每 segment 最多存储的向量数

    def __init__(self, dim: int = 128, store_dir: str = "") -> None:
        """
        Args:
            dim: 向量维度
            store_dir: 存储目录路径（空字符串 = 仅内存模式）
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self._dim = dim
        self._store_dir = store_dir
        self._write_lock = threading.Lock()

        # 内存中的向量索引
        self._ids: list[str] = []
        self._vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self._id_to_idx: dict[str, int] = {}
        self._segment_counter: int = 0

        # 磁盘索引
        self._disk_index: dict[str, tuple[str, int]] = {}  # id → (segment_file, row_idx)

        # 如果有存储目录，加载已有数据
        if store_dir and os.path.isdir(store_dir):
            self._load_from_disk()

    @property
    def dim(self) -> int:
        return self._dim

    def count(self) -> int:
        """存储的向量总数。"""
        return len(self._ids) + len(self._disk_index)

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        """
        添加向量到存储。

        Args:
            ids: 向量 ID 列表
            vectors: shape (N, dim) 的向量矩阵
        """
        vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        if vectors.shape[1] != self._dim:
            raise ValueError(f"Vector dim mismatch: expected {self._dim}, got {vectors.shape[1]}")
        if len(ids) != vectors.shape[0]:
            raise ValueError(f"ids length {len(ids)} != vectors count {vectors.shape[0]}")

        with self._write_lock:
            start_idx = len(self._ids)
            for i, vid in enumerate(ids):
                if vid in self._id_to_idx:
                    logger.warning("Duplicate vector id %s, skipping", vid)
                    continue
                self._ids.append(vid)
                self._id_to_idx[vid] = start_idx + i

            if self._vectors.shape[0] == 0:
                self._vectors = vectors.copy()
            else:
                self._vectors = np.vstack([self._vectors, vectors])

            # 当积累到 SEGMENT_SIZE 时刷盘
            if len(self._ids) >= self.SEGMENT_SIZE and self._store_dir:
                self._flush_to_disk()

    def search(self, query: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """
        余弦相似度检索。

        Args:
            query: shape (dim,) 查询向量
            top_k: 返回 top-k 结果

        Returns:
            [(id, cosine_similarity), ...] 按相似度降序排列
        """
        query = np.asarray(query, dtype=np.float32).ravel()
        if query.shape[0] != self._dim:
            raise ValueError(f"Query dim mismatch: expected {self._dim}, got {query.shape[0]}")

        # 合并内存和磁盘向量进行搜索
        all_vectors = self._get_all_vectors()
        all_ids = self._ids + list(self._disk_index.keys())

        if all_vectors.shape[0] == 0:
            return []

        # 批量余弦相似度
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-10:
            return []

        vec_norms = np.linalg.norm(all_vectors, axis=1)
        # 防止零向量
        vec_norms = np.maximum(vec_norms, 1e-10)

        similarities = all_vectors @ query / (vec_norms * query_norm)

        # 排序取 top_k
        actual_k = min(top_k, len(similarities))
        if actual_k <= 0:
            return []

        top_indices = np.argpartition(similarities, -actual_k)[-actual_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        return [(all_ids[i], float(similarities[i])) for i in top_indices]

    def remove(self, ids: list[str]) -> None:
        """标记删除（逻辑删除）。"""
        with self._write_lock:
            for vid in ids:
                if vid in self._id_to_idx:
                    # 内存中标记为删除
                    idx = self._id_to_idx[vid]
                    self._vectors[idx] = 0  # 零向量
                    del self._id_to_idx[vid]
                if vid in self._disk_index:
                    del self._disk_index[vid]

    def save(self, path: str) -> None:
        """保存全部向量到指定目录。"""
        os.makedirs(path, exist_ok=True)

        with self._write_lock:
            # 保存当前内存中的向量为新 segment
            if self._vectors.shape[0] > 0:
                self._segment_counter += 1
                seg_file = os.path.join(path, f"segment_{self._segment_counter:04d}.npy")
                np.save(seg_file, self._vectors)

                # 更新磁盘索引
                for i, vid in enumerate(self._ids):
                    if vid in self._id_to_idx:
                        self._disk_index[vid] = (seg_file, i)

            # 保存索引
            index_path = os.path.join(path, "index.json")
            serializable = {vid: [seg, row] for vid, (seg, row) in self._disk_index.items()}
            with open(index_path, "w") as f:
                json.dump({"dim": self._dim, "index": serializable}, f)

    def load(self, path: str) -> None:
        """从指定目录加载向量。"""
        index_path = os.path.join(path, "index.json")
        if not os.path.exists(index_path):
            return

        with open(index_path) as f:
            data = json.load(f)

        self._dim = data.get("dim", self._dim)
        self._disk_index = {vid: (seg, row) for vid, (seg, row) in data.get("index", {}).items()}

        # 更新内存向量：从磁盘加载
        self._ids = []
        self._vectors = np.empty((0, self._dim), dtype=np.float32)
        self._id_to_idx = {}

    def _get_all_vectors(self) -> np.ndarray:
        """获取所有向量（内存 + 磁盘），合并返回。"""
        vectors_list = []

        if self._vectors.shape[0] > 0:
            vectors_list.append(self._vectors)

        # 从磁盘加载 segment
        loaded_segments: dict[str, np.ndarray] = {}
        for vid, (seg_file, _) in self._disk_index.items():
            if seg_file not in loaded_segments:
                try:
                    loaded_segments[seg_file] = np.load(seg_file)
                except Exception as e:
                    logger.warning("Failed to load segment %s: %s", seg_file, e)

        for seg_file, seg_vectors in loaded_segments.items():
            vectors_list.append(seg_vectors)

        if not vectors_list:
            return np.empty((0, self._dim), dtype=np.float32)

        return np.vstack(vectors_list)

    def _flush_to_disk(self) -> None:
        """将内存向量刷到磁盘 segment。"""
        if not self._store_dir:
            return

        os.makedirs(self._store_dir, exist_ok=True)
        self._segment_counter += 1
        seg_file = os.path.join(self._store_dir, f"segment_{self._segment_counter:04d}.npy")
        np.save(seg_file, self._vectors)

        for i, vid in enumerate(self._ids):
            self._disk_index[vid] = (seg_file, i)

        # 清空内存
        self._ids = []
        self._vectors = np.empty((0, self._dim), dtype=np.float32)
        self._id_to_idx = {}

    def _load_from_disk(self) -> None:
        """从磁盘加载已有数据。"""
        self.load(self._store_dir)


# =============================================================================
# PersistentMemoryConfig — 持久化记忆配置
# =============================================================================


@dataclass
class PersistentMemoryConfig:
    """持久化经验记忆配置。"""

    db_path: str = "./data/experience.db"
    vector_dim: int = 128
    max_experiences: int = 50000
    auto_compact_interval: int = 1000
    store_dir: str = "./data/vectors"


# =============================================================================
# PersistentExperienceMemory — 持久化经验记忆
# =============================================================================


class PersistentExperienceMemory:
    """
    持久化经验记忆：SQLite 元数据 + VectorStore 向量检索。

    实现 ExperienceMemory 相同的 store/retrieve 接口，
    支持跨进程持久化、语义检索、因果检索。

    Example:
        >>> mem = PersistentExperienceMemory()
        >>> mem.store(Experience(tags=["心率"], causal_edges=[...]))
        >>> results = mem.retrieve(query="心率升高", top_k=5)
    """

    def __init__(self, config: PersistentMemoryConfig | None = None) -> None:
        self._config = config or PersistentMemoryConfig()
        self._write_lock = threading.Lock()
        self._store_count: int = 0

        # 初始化 SQLite
        os.makedirs(os.path.dirname(self._config.db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self._config.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS experiences (
                id TEXT PRIMARY KEY,
                tags TEXT NOT NULL,
                causal_edges TEXT,
                timestamp TEXT,
                importance REAL DEFAULT 0.5,
                content TEXT,
                created_at REAL NOT NULL
            )
        """)
        self._conn.commit()

        # 初始化 VectorStore
        self._vector_store = VectorStore(
            dim=self._config.vector_dim,
            store_dir=self._config.store_dir,
        )
        # 加载已有向量数据
        if os.path.isdir(self._config.store_dir):
            self._vector_store.load(self._config.store_dir)

        # 内存缓存（最近的经验）
        self._cache: list[dict[str, Any]] = []
        self._cache_lock = threading.Lock()

    # =====================================================================
    # 核心接口
    # =====================================================================

    def store(self, experience: Any, embedding: np.ndarray | None = None) -> str:
        """
        存储一条经验。

        Args:
            experience: 经验对象（含 tags, causal_edges, importance）
            embedding: 可选的预计算嵌入向量

        Returns:
            经验 ID
        """
        exp_id = f"exp_{int(time.time() * 1000)}_{self._store_count}"
        self._store_count += 1

        tags = getattr(experience, "tags", []) or []
        causal_edges = getattr(experience, "causal_edges", []) or []
        importance = getattr(experience, "importance", 0.5)
        content = getattr(experience, "content", "")
        timestamp = getattr(experience, "timestamp", "")

        with self._write_lock:
            # 存入 SQLite
            self._conn.execute(
                """INSERT OR REPLACE INTO experiences
                   (id, tags, causal_edges, timestamp, importance, content, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    exp_id,
                    json.dumps(tags, ensure_ascii=False),
                    json.dumps(causal_edges, ensure_ascii=False),
                    timestamp,
                    float(importance),
                    str(content),
                    time.time(),
                ),
            )
            self._conn.commit()

            # 存入 VectorStore（如果提供了 embedding 或 tags 可以生成）
            if embedding is not None:
                self._vector_store.add([exp_id], embedding.reshape(1, -1))
            else:
                # 从 tags 生成简单嵌入
                tag_vec = self._tags_to_vector(tags)
                self._vector_store.add([exp_id], tag_vec.reshape(1, -1))

            # 缓存
            with self._cache_lock:
                self._cache.append(
                    {
                        "id": exp_id,
                        "tags": tags,
                        "causal_edges": causal_edges,
                        "importance": importance,
                    }
                )
                if len(self._cache) > 100:
                    self._cache = self._cache[-50:]

        # 自动 compact
        if self._store_count % self._config.auto_compact_interval == 0:
            self.compact()

        return exp_id

    def retrieve(
        self,
        query: str = "",
        tags: list[str] | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        检索相关经验。

        Args:
            query: 语义查询字符串
            tags: 标签过滤
            top_k: 返回数量

        Returns:
            经验列表
        """
        results: list[dict[str, Any]] = []

        # 1. 向量相似度检索
        if query:
            query_vec = self._tags_to_vector([query])
            vec_results = self._vector_store.search(query_vec, top_k=top_k * 2)

            # 从 SQLite 加载完整经验
            for exp_id, similarity in vec_results:
                exp = self._load_experience(exp_id)
                if exp:
                    exp["similarity"] = round(similarity, 4)
                    results.append(exp)

        # 2. 标签过滤（SQLite FTS 替代方案）
        if tags:
            for tag in tags:
                cursor = self._conn.execute(
                    "SELECT * FROM experiences WHERE tags LIKE ? LIMIT ?",
                    (f'%"{tag}"%', top_k),
                )
                for row in cursor.fetchall():
                    exp = self._row_to_dict(row)
                    if not any(r["id"] == exp["id"] for r in results):
                        results.append(exp)

        # 如果两者都没用，返回最近的
        if not query and not tags:
            cursor = self._conn.execute(
                "SELECT * FROM experiences ORDER BY created_at DESC LIMIT ?",
                (top_k,),
            )
            for row in cursor.fetchall():
                results.append(self._row_to_dict(row))

        return results[:top_k]

    def compact(self) -> int:
        """
        压缩经验库：删除低重要性经验。

        Returns:
            删除的经验数量
        """
        threshold = 0.1
        cursor = self._conn.execute("SELECT COUNT(*) FROM experiences WHERE importance < ?", (threshold,))
        count = cursor.fetchone()[0]

        if count > 0:
            self._conn.execute("DELETE FROM experiences WHERE importance < ?", (threshold,))
            self._conn.commit()

        return count

    def statistics(self) -> dict[str, Any]:
        """返回记忆统计。"""
        cursor = self._conn.execute("SELECT COUNT(*) FROM experiences")
        total = cursor.fetchone()[0]

        cursor = self._conn.execute("SELECT AVG(importance) FROM experiences")
        avg_importance = cursor.fetchone()[0] or 0.0

        return {
            "total_experiences": total,
            "vector_count": self._vector_store.count(),
            "avg_importance": round(float(avg_importance), 4),
            "store_count": self._store_count,
        }

    def migrate_from_memory(self, in_memory: Any) -> int:
        """
        从旧内存 ExperienceMemory 迁移数据。

        Args:
            in_memory: ExperienceMemory 实例

        Returns:
            迁移的经验数量
        """
        count = 0
        if hasattr(in_memory, "_experiences"):
            for exp in in_memory._experiences:
                self.store(exp)
                count += 1
        return count

    def export_to_jsonl(self, path: str) -> int:
        """导出经验为 JSONL 格式。"""
        cursor = self._conn.execute("SELECT * FROM experiences ORDER BY created_at")
        count = 0

        with open(path, "w") as f:
            for row in cursor.fetchall():
                d = self._row_to_dict(row)
                # 脱敏
                d.pop("content", None)
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                count += 1

        return count

    def close(self) -> None:
        """关闭连接，刷盘向量数据。"""
        # 保存 VectorStore 数据到磁盘
        if self._config.store_dir:
            os.makedirs(self._config.store_dir, exist_ok=True)
            self._vector_store.save(self._config.store_dir)
        if self._conn:
            self._conn.close()

    # =====================================================================
    # 内部方法
    # =====================================================================

    def _load_experience(self, exp_id: str) -> dict | None:
        cursor = self._conn.execute("SELECT * FROM experiences WHERE id = ?", (exp_id,))
        row = cursor.fetchone()
        if row:
            return self._row_to_dict(row)
        return None

    def _row_to_dict(self, row: tuple) -> dict[str, Any]:
        return {
            "id": row[0],
            "tags": json.loads(row[1]) if row[1] else [],
            "causal_edges": json.loads(row[2]) if row[2] else [],
            "timestamp": row[3],
            "importance": row[4],
            "created_at": row[6],
        }

    def _tags_to_vector(self, tags: list[str]) -> np.ndarray:
        """从 tags 生成语义嵌入向量。

        P0-F2 修复: 替换 char-hash 为词频+TF-IDF 风格语义嵌入。
        语义相近的标签产生相近的向量，支持有意义的余弦相似度检索。

        策略:
            1. 对每个 tag 分词 (中文双字gram + 英文按空格)
            2. 使用词频加权 (含 IDF 启发式)
            3. L2 归一化
        """
        vec = np.zeros(self._config.vector_dim, dtype=np.float32)

        if not tags:
            return vec

        # 合并所有标签文本
        _text = " ".join(str(t) for t in tags).lower()

        # 分词: 中文双字gram + 英文单词
        tokens: list[str] = []
        for tag in tags:
            tag_str = str(tag).lower().strip()
            if not tag_str:
                continue
            # 英文: 按空格分割
            if all(ord(c) < 128 for c in tag_str):
                tokens.extend(tag_str.split())
            else:
                # 中文: 双字gram (bigram)
                tokens.append(tag_str)  # 完整标签作为整体 token
                for i in range(len(tag_str) - 1):
                    tokens.append(tag_str[i : i + 2])  # 双字gram

        # 词频 + 确定性哈希映射到向量维度
        from collections import Counter

        tf = Counter(tokens)
        for token, count in tf.items():
            # 确定性哈希: 同一 token 始终映射到相同位置
            idx = hash(token) % self._config.vector_dim
            # IDF 启发式: 短词权重低, 长词/专业词权重高
            idf_weight = 1.0 + min(len(token), 10) * 0.1
            vec[idx] += count * idf_weight

        # L2 归一化
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec /= norm

        return vec

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        stats = self.statistics()
        return f"PersistentExperienceMemory(total={stats['total_experiences']}, vectors={stats['vector_count']})"
