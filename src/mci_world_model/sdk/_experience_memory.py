"""MCI World Model — 三维融合经验记忆库 (ExperienceDB)

CEWM v3.5.0 新增组件 (N2)：
基于语义/因果/时间三维索引的经验记忆存储与检索系统。

理论基础：
    1. LeCun 联合嵌入预测架构 (JEPA) — 经验以语义向量表示
    2. Pearl 因果层级 — 经验附带因果结构信息
    3. Tulving 情景记忆理论 — 时间/情境编码

三维索引：
    - 语义维 (Semantic): 基于标签/特征的 TF-IDF 向量相似度
    - 因果维 (Causal): 基于因果边匹配的结构相似度
    - 时间维 (Temporal): 基于时间衰减的近度权重

核心能力：
    - store(experience) — 存储一条经验（含三维索引）
    - retrieve(query, top_k, weights) — 三维加权融合检索
    - consolidate() — 经验巩固（合并相似经验，减少冗余）
    - forget(decay_factor) — 经验遗忘（降低低频经验权重）
    - statistics() — 经验库统计信息

存储效率：
    1000 条经验 < 50MB（内存 profiling 验证）

依赖：无外部依赖，纯 Python 实现（math + collections）
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# =============================================================================
# 数据类型
# =============================================================================


class ExperienceType(Enum):
    """经验类型分类。"""

    SUCCESS = "success"  # 成功经验
    FAILURE = "failure"  # 失败经验
    ANOMALY = "anomaly"  # 异常经验
    TRANSITION = "transition"  # 状态迁移经验
    PREDICTION = "prediction"  # 预测验证经验


@dataclass
class Experience:
    """一条经验记忆。

    Attributes:
        experience_id: 唯一标识符
        experience_type: 经验类型
        tags: 语义标签列表（用于语义维索引）
        causal_edges: 因果边列表 [(cause, effect), ...]（用于因果维索引）
        outcome: 经验结果/结论
        context: 经验上下文信息
        timestamp: 创建时间戳
        importance: 重要性权重 [0, 1]
        access_count: 被检索次数
        last_accessed: 最近访问时间戳
        prediction_error: 关联的预测误差（越小越好）
        state_snapshot: 经验发生时的状态快照（任意对象）
        metadata: 附加元数据
    """

    experience_id: str = ""
    experience_type: ExperienceType = ExperienceType.SUCCESS
    tags: list[str] = field(default_factory=list)
    causal_edges: list[tuple[str, str]] = field(default_factory=list)
    outcome: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    prediction_error: float | None = None
    state_snapshot: object | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def age_days(self) -> float:
        """经验年龄（天）。"""
        return (time.time() - self.timestamp) / 86400

    def recency_score(self, half_life_hours: float = 168.0) -> float:
        """时间衰减分数（指数衰减，半衰期默认 7 天）。

        Args:
            half_life_hours: 半衰期（小时），默认 168h = 7 天

        Returns:
            [0, 1] 范围内的近度分数
        """
        age_hours = (time.time() - self.timestamp) / 3600
        decay_rate = math.log(2) / half_life_hours
        return math.exp(-decay_rate * age_hours)


@dataclass
class RetrievalResult:
    """检索结果。

    Attributes:
        experience: 匹配的经验
        score: 综合融合分数 [0, 1]
        semantic_score: 语义相似度分数
        causal_score: 因果匹配分数
        temporal_score: 时间近度分数
        rank: 排名（1-based）
    """

    experience: Experience
    score: float = 0.0
    semantic_score: float = 0.0
    causal_score: float = 0.0
    temporal_score: float = 0.0
    rank: int = 0

    def to_dict(self) -> dict:
        return {
            "experience_id": self.experience.experience_id,
            "score": round(self.score, 4),
            "semantic_score": round(self.semantic_score, 4),
            "causal_score": round(self.causal_score, 4),
            "temporal_score": round(self.temporal_score, 4),
            "rank": self.rank,
            "experience_type": self.experience.experience_type.value,
            "outcome": self.experience.outcome,
        }


@dataclass
class ExperienceDBStats:
    """经验库统计信息。"""

    total_experiences: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    avg_importance: float = 0.0
    avg_age_days: float = 0.0
    avg_access_count: float = 0.0
    total_causal_edges: int = 0
    total_tags: int = 0
    memory_estimate_kb: float = 0.0
    consolidation_count: int = 0
    forget_count: int = 0


# =============================================================================
# 三维索引引擎
# =============================================================================


class _SemanticIndex:
    """语义维索引：基于标签的 TF-IDF 向量。"""

    def __init__(self):
        self._doc_freq: dict[str, int] = defaultdict(int)
        self._n_docs: int = 0

    def add(self, tags: list[str]) -> None:
        """注册一组标签到文档频率表。"""
        self._n_docs += 1
        unique_tags = set(tags)
        for tag in unique_tags:
            self._doc_freq[tag] += 1

    def remove(self, tags: list[str]) -> None:
        """从文档频率表移除一组标签。"""
        self._n_docs = max(0, self._n_docs - 1)
        unique_tags = set(tags)
        for tag in unique_tags:
            if tag in self._doc_freq:
                self._doc_freq[tag] -= 1
                if self._doc_freq[tag] <= 0:
                    del self._doc_freq[tag]

    def similarity(self, query_tags: list[str], doc_tags: list[str]) -> float:
        """计算查询标签与文档标签的 TF-IDF 余弦相似度。

        公式：
            cos(query, doc) = Σ_t (tf_q(t) × idf(t) × tf_d(t) × idf(t)) / (‖q‖ × ‖d‖)

        简化为：对 query 和 doc 的标签交集，累加 idf² 并归一化。
        """
        if not query_tags or not doc_tags:
            return 0.0

        query_set = set(query_tags)
        doc_set = set(doc_tags)
        common = query_set & doc_set

        if not common:
            return 0.0

        # TF-IDF 余弦相似度（简化版）
        score = 0.0
        for tag in common:
            df = self._doc_freq.get(tag, 1)
            idf = math.log((self._n_docs + 1) / df) + 1  # smoothed IDF
            score += idf * idf

        # 归一化
        q_norm = sum((math.log((self._n_docs + 1) / self._doc_freq.get(t, 1)) + 1) ** 2 for t in query_set)
        d_norm = sum((math.log((self._n_docs + 1) / self._doc_freq.get(t, 1)) + 1) ** 2 for t in doc_set)

        if q_norm == 0 or d_norm == 0:
            return 0.0
        return score / (math.sqrt(q_norm) * math.sqrt(d_norm))

    def statistics(self) -> dict:
        return {
            "n_docs": self._n_docs,
            "vocabulary_size": len(self._doc_freq),
            "avg_df": (sum(self._doc_freq.values()) / len(self._doc_freq) if self._doc_freq else 0),
        }


class _CausalIndex:
    """因果维索引：基于因果边匹配的结构相似度。"""

    def __init__(self):
        self._edge_index: dict[str, set[str]] = defaultdict(set)  # cause → {exp_ids}
        self._effect_index: dict[str, set[str]] = defaultdict(set)  # effect → {exp_ids}

    def add(self, exp_id: str, edges: list[tuple[str, str]]) -> None:
        """注册经验的因果边到索引。"""
        for cause, effect in edges:
            self._edge_index[cause].add(exp_id)
            self._effect_index[effect].add(exp_id)

    def remove(self, exp_id: str, edges: list[tuple[str, str]]) -> None:
        """从索引移除经验的因果边。"""
        for cause, effect in edges:
            self._edge_index[cause].discard(exp_id)
            if not self._edge_index[cause]:
                del self._edge_index[cause]
            self._effect_index[effect].discard(exp_id)
            if not self._effect_index[effect]:
                del self._effect_index[effect]

    def similarity(self, query_edges: list[tuple[str, str]], doc_edges: list[tuple[str, str]]) -> float:
        """计算因果边集合的 Jaccard 相似度。

        公式：
            J(Q, D) = |Q ∩ D| / |Q ∪ D|

        考虑方向性：(cause, effect) 和 (effect, cause) 不同。
        """
        if not query_edges or not doc_edges:
            return 0.0

        q_set = set(query_edges)
        d_set = set(doc_edges)

        intersection = q_set & d_set
        union = q_set | d_set

        if not union:
            return 0.0
        return len(intersection) / len(union)

    def find_by_edge(self, cause: str, effect: str) -> set[str]:
        """查找包含特定因果边的经验 ID。"""
        cause_matches = self._edge_index.get(cause, set())
        effect_matches = self._effect_index.get(effect, set())
        return cause_matches & effect_matches

    def statistics(self) -> dict:
        return {
            "n_cause_nodes": len(self._edge_index),
            "n_effect_nodes": len(self._effect_index),
            "total_indexed_edges": sum(len(v) for v in self._edge_index.values()),
        }


class _TemporalIndex:
    """时间维索引：基于时间戳的近度排序与衰减。"""

    def __init__(self, half_life_hours: float = 168.0):
        self.half_life_hours = half_life_hours

    def score(self, timestamp: float) -> float:
        """计算时间近度分数（指数衰减）。"""
        age_hours = (time.time() - timestamp) / 3600
        decay_rate = math.log(2) / self.half_life_hours
        return math.exp(-decay_rate * age_hours)

    def score_with_access(self, timestamp: float, last_accessed: float, access_count: int) -> float:
        """结合创建时间和访问历史的时间分数。

        公式：
            temporal = 0.6 × recency(created) + 0.25 × recency(last_accessed) + 0.15 × log(1 + access_count)

        体现 "常用记忆不易遗忘" 的心理学规律。
        """
        recency_created = self.score(timestamp)
        recency_accessed = self.score(last_accessed)
        access_bonus = math.log(1 + access_count) / 5.0  # 归一化到 ~[0, 1]
        access_bonus = min(1.0, access_bonus)
        return 0.6 * recency_created + 0.25 * recency_accessed + 0.15 * access_bonus


# =============================================================================
# ExperienceDB 主类
# =============================================================================


@dataclass
class ExperienceDB:
    """三维融合经验记忆库。

    三维索引：
        语义维 (Semantic) — TF-IDF 标签相似度
        因果维 (Causal) — 因果边 Jaccard 匹配
        时间维 (Temporal) — 指数衰减近度 + 访问增强

    Example:
        >>> db = ExperienceDB()
        >>> exp = Experience(
        ...     experience_id="exp_001",
        ...     experience_type=ExperienceType.SUCCESS,
        ...     tags=["pendulum", "prediction", "theta"],
        ...     causal_edges=[("theta", "omega"), ("gravity", "period")],
        ...     outcome="预测误差 < 0.05",
        ...     importance=0.9,
        ... )
        >>> db.store(exp)
        >>> results = db.retrieve(
        ...     query_tags=["pendulum", "theta"],
        ...     top_k=3,
        ... )
    """

    half_life_hours: float = 168.0  # 时间衰减半衰期（7 天）
    max_experiences: int = 10000  # 最大经验数
    consolidation_threshold: float = 0.85  # 合并相似度阈值

    # 内部状态
    _experiences: dict[str, Experience] = field(default_factory=dict)
    _semantic_index: _SemanticIndex = field(default_factory=_SemanticIndex)
    _causal_index: _CausalIndex = field(default_factory=_CausalIndex)
    _temporal_index: _TemporalIndex | None = None
    _consolidation_count: int = 0
    _forget_count: int = 0
    _id_counter: int = 0

    def __post_init__(self):
        if self._temporal_index is None:
            self._temporal_index = _TemporalIndex(half_life_hours=self.half_life_hours)

    # ── 存储 ──

    def store(self, experience: Experience | None = None, **kwargs) -> str:
        """存储一条经验到记忆库。

        Args:
            experience: Experience 对象（优先）
            **kwargs: 快捷参数（experience_type, tags, causal_edges, outcome, etc.）

        Returns:
            经验 ID（自动生成或使用传入值）
        """
        if experience is None:
            self._id_counter += 1
            exp_id = kwargs.pop("experience_id", f"exp_{self._id_counter:06d}")
            exp_type = kwargs.pop("experience_type", ExperienceType.SUCCESS)
            if isinstance(exp_type, str):
                exp_type = ExperienceType(exp_type)
            experience = Experience(
                experience_id=exp_id,
                experience_type=exp_type,
                **kwargs,
            )

        if not experience.experience_id:
            self._id_counter += 1
            experience.experience_id = f"exp_{self._id_counter:06d}"

        # 防止重复
        if experience.experience_id in self._experiences:
            return experience.experience_id

        # 容量控制
        if len(self._experiences) >= self.max_experiences:
            self._evict_least_important()

        # 注册三维索引
        self._experiences[experience.experience_id] = experience
        self._semantic_index.add(experience.tags)
        self._causal_index.add(experience.experience_id, experience.causal_edges)

        return experience.experience_id

    def store_batch(self, experiences: list[Experience]) -> list[str]:
        """批量存储经验。"""
        return [self.store(exp) for exp in experiences]

    # ── 检索 ──

    def retrieve(
        self,
        query_tags: list[str] | None = None,
        query_edges: list[tuple[str, str]] | None = None,
        top_k: int = 5,
        weights: tuple[float, float, float] = (0.4, 0.35, 0.25),
        type_filter: ExperienceType | None = None,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        """三维加权融合检索。

        融合公式：
            score = w_semantic × semantic_score
                  + w_causal × causal_score
                  + w_temporal × temporal_score

        默认权重 (0.4, 0.35, 0.25) 体现语义优先、因果增强、时间衰减的策略。

        Args:
            query_tags: 查询语义标签
            query_edges: 查询因果边
            top_k: 返回数量
            weights: 三维权重 (semantic, causal, temporal)
            type_filter: 按经验类型过滤
            min_score: 最低分数阈值

        Returns:
            按综合分数降序排列的检索结果
        """
        w_semantic, w_causal, w_temporal = weights
        results = []

        for exp in self._experiences.values():
            # 类型过滤
            if type_filter is not None and exp.experience_type != type_filter:
                continue

            # 语义维
            sem_score = 0.0
            if query_tags:
                sem_score = self._semantic_index.similarity(query_tags, exp.tags)

            # 因果维
            cau_score = 0.0
            if query_edges:
                cau_score = self._causal_index.similarity(query_edges, exp.causal_edges)

            # 时间维
            temp_score = 0.0
            if self._temporal_index is not None:
                temp_score = self._temporal_index.score_with_access(exp.timestamp, exp.last_accessed, exp.access_count)

            # 融合
            combined = w_semantic * sem_score + w_causal * cau_score + w_temporal * temp_score

            # 重要性加权
            combined *= exp.importance

            if combined >= min_score:
                results.append(
                    RetrievalResult(
                        experience=exp,
                        score=combined,
                        semantic_score=sem_score,
                        causal_score=cau_score,
                        temporal_score=temp_score,
                    )
                )

        # 排序
        results.sort(key=lambda r: r.score, reverse=True)

        # 截断
        top_results = results[:top_k]

        # 标记排名 & 更新访问统计
        for i, r in enumerate(top_results, 1):
            r.rank = i
            r.experience.access_count += 1
            r.experience.last_accessed = time.time()

        return top_results

    def retrieve_by_causal_edge(self, cause: str, effect: str, top_k: int = 5) -> list[RetrievalResult]:
        """快捷检索：按特定因果边查找相关经验。"""
        matching_ids = self._causal_index.find_by_edge(cause, effect)
        results = []
        for exp_id in matching_ids:
            exp = self._experiences.get(exp_id)
            if exp:
                results.append(
                    RetrievalResult(
                        experience=exp,
                        score=1.0,
                        causal_score=1.0,
                        rank=len(results) + 1,
                    )
                )
        return results[:top_k]

    def retrieve_recent(self, n: int = 5, max_age_hours: float = 168.0) -> list[RetrievalResult]:
        """快捷检索：最近经验。"""
        cutoff = time.time() - max_age_hours * 3600
        recent = [exp for exp in self._experiences.values() if exp.timestamp >= cutoff]
        recent.sort(key=lambda e: e.timestamp, reverse=True)
        results = []
        for i, exp in enumerate(recent[:n], 1):
            results.append(
                RetrievalResult(
                    experience=exp,
                    score=1.0,
                    temporal_score=1.0,
                    rank=i,
                )
            )
        return results

    # ── 经验巩固与遗忘 ──

    def consolidate(self) -> int:
        """经验巩固：合并高度相似的经验。

        策略：对同一类型的经验，若标签 Jaccard 相似度 > consolidation_threshold，
        合并为一条经验（保留重要性更高的，累加 access_count）。

        Returns:
            合并的经验对数
        """
        merged_count = 0
        by_type: dict[ExperienceType, list[Experience]] = defaultdict(list)

        for exp in self._experiences.values():
            by_type[exp.experience_type].append(exp)

        for exp_type, group in by_type.items():
            if len(group) < 2:
                continue

            to_remove = set()
            for i in range(len(group)):
                if group[i].experience_id in to_remove:
                    continue
                for j in range(i + 1, len(group)):
                    if group[j].experience_id in to_remove:
                        continue
                    # 标签 Jaccard 相似度
                    sim = self._tag_jaccard(group[i].tags, group[j].tags)
                    if sim >= self.consolidation_threshold:
                        # 保留重要性更高的
                        keep, remove = (
                            (group[i], group[j]) if group[i].importance >= group[j].importance else (group[j], group[i])
                        )
                        # 合并
                        keep.access_count += remove.access_count
                        keep.importance = min(1.0, max(keep.importance, remove.importance) * 1.05)
                        # 合并因果边
                        for edge in remove.causal_edges:
                            if edge not in keep.causal_edges:
                                keep.causal_edges.append(edge)
                        # 合并标签
                        for tag in remove.tags:
                            if tag not in keep.tags:
                                keep.tags.append(tag)

                        # 移除被合并的经验
                        to_remove.add(remove.experience_id)
                        merged_count += 1

            # 执行移除
            for exp_id in to_remove:
                self._remove_experience(exp_id)

        self._consolidation_count += merged_count
        return merged_count

    def forget(self, threshold: float = 0.05) -> int:
        """经验遗忘：移除重要性低于阈值的低频经验。

        综合评分 < threshold 的经验将被遗忘。
        综合评分 = importance × recency_score × (1 + log(1 + access_count)) / 2

        Args:
            threshold: 遗忘阈值

        Returns:
            遗忘的经验数
        """
        to_forget = []
        for exp in self._experiences.values():
            recency = exp.recency_score(half_life_hours=self.half_life_hours)
            access_factor = (1 + math.log(1 + exp.access_count)) / 2
            score = exp.importance * recency * access_factor
            if score < threshold:
                to_forget.append(exp.experience_id)

        for exp_id in to_forget:
            self._remove_experience(exp_id)

        self._forget_count += len(to_forget)
        return len(to_forget)

    # ── 查询 ──

    def get(self, experience_id: str) -> Experience | None:
        """获取指定 ID 的经验。"""
        return self._experiences.get(experience_id)

    def remove(self, experience_id: str) -> bool:
        """移除指定经验。"""
        return self._remove_experience(experience_id)

    def clear(self) -> None:
        """清空经验库。"""
        self._experiences.clear()
        self._semantic_index = _SemanticIndex()
        self._causal_index = _CausalIndex()
        self._temporal_index = _TemporalIndex(half_life_hours=self.half_life_hours)
        self._consolidation_count = 0
        self._forget_count = 0
        self._id_counter = 0

    @property
    def size(self) -> int:
        """经验数量。"""
        return len(self._experiences)

    @property
    def all_experiences(self) -> list[Experience]:
        """所有经验列表。"""
        return list(self._experiences.values())

    # ── 统计 ──

    def statistics(self) -> ExperienceDBStats:
        """经验库统计信息。"""
        if not self._experiences:
            return ExperienceDBStats()

        experiences = list(self._experiences.values())
        by_type: dict[str, int] = defaultdict(int)
        total_importance = 0.0
        total_age = 0.0
        total_access = 0
        total_edges = 0
        total_tags = 0

        for exp in experiences:
            by_type[exp.experience_type.value] += 1
            total_importance += exp.importance
            total_age += exp.age_days()
            total_access += exp.access_count
            total_edges += len(exp.causal_edges)
            total_tags += len(exp.tags)

        n = len(experiences)
        # 粗略估算内存占用（KB）
        # 每条经验：~1KB (tags + edges + metadata)
        memory_kb = n * 1.0

        return ExperienceDBStats(
            total_experiences=n,
            by_type=dict(by_type),
            avg_importance=total_importance / n,
            avg_age_days=total_age / n,
            avg_access_count=total_access / n,
            total_causal_edges=total_edges,
            total_tags=total_tags,
            memory_estimate_kb=memory_kb,
            consolidation_count=self._consolidation_count,
            forget_count=self._forget_count,
        )

    # ── 序列化 ──

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "size": self.size,
            "max_experiences": self.max_experiences,
            "half_life_hours": self.half_life_hours,
            "consolidation_threshold": self.consolidation_threshold,
            "statistics": self.statistics().__dict__ if self._experiences else {},
            "experiences": [
                {
                    "id": exp.experience_id,
                    "type": exp.experience_type.value,
                    "tags": exp.tags,
                    "causal_edges": exp.causal_edges,
                    "outcome": exp.outcome,
                    "importance": exp.importance,
                    "age_days": round(exp.age_days(), 2),
                    "access_count": exp.access_count,
                }
                for exp in self._experiences.values()
            ],
        }

    # ── 内部方法 ──

    def _remove_experience(self, exp_id: str) -> bool:
        """内部移除经验（同时清理索引）。"""
        exp = self._experiences.pop(exp_id, None)
        if exp is None:
            return False
        self._semantic_index.remove(exp.tags)
        self._causal_index.remove(exp.experience_id, exp.causal_edges)
        return True

    def _evict_least_important(self) -> None:
        """淘汰最不重要的经验（容量控制）。"""
        if not self._experiences:
            return
        # 综合评分 = importance × recency × (1 + log(1 + access_count)) / 2
        worst_id = None
        worst_score = float("inf")
        for exp in self._experiences.values():
            recency = exp.recency_score(half_life_hours=self.half_life_hours)
            access_factor = (1 + math.log(1 + exp.access_count)) / 2
            score = exp.importance * recency * access_factor
            if score < worst_score:
                worst_score = score
                worst_id = exp.experience_id
        if worst_id:
            self._remove_experience(worst_id)

    @staticmethod
    def _tag_jaccard(tags_a: list[str], tags_b: list[str]) -> float:
        """标签集合的 Jaccard 相似度。"""
        if not tags_a and not tags_b:
            return 1.0
        set_a = set(tags_a)
        set_b = set(tags_b)
        union = set_a | set_b
        if not union:
            return 1.0
        return len(set_a & set_b) / len(union)
