from __future__ import annotations

"""MCI World Model — 五维融合检索器 (MultiViewRetriever)

CEWM v3.5.0 新增组件 (N3)：
基于语义/因果/时间/上下文/结构五维融合的多视角检索系统。

理论基础：
    1. Tulving 多视角记忆理论 — 记忆从多个维度编码与检索
    2. Pearl 因果层级 — L1(关联) → L2(干预) → L3(反事实)
    3. Kohonen 自组织映射 — 多维度特征空间的近邻搜索

五维检索视角：
    - 语义维 (Semantic): TF-IDF 标签/特征相似度
    - 因果维 (Causal): 因果边匹配 + 因果路径相似度
    - 时间维 (Temporal): 指数衰减近度 + 访问增强
    - 上下文维 (Contextual): 情境匹配（domain/scenario/environment 等）
    - 结构维 (Structural): 状态空间拓扑相似度

融合策略：
    加权线性融合 + 可选 Borda 排名融合 (rank fusion)

依赖：ExperienceDB（语义/因果/时间三维）
"""


import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from mci_world_model.sdk._experience_memory import (
    Experience,
    ExperienceDB,
    ExperienceType,
)

# =============================================================================
# 数据类型
# =============================================================================


class FusionStrategy(Enum):
    """融合策略。"""

    WEIGHTED = "weighted"  # 加权线性融合
    BORDA = "borda"  # Borda 排名融合
    HYBRID = "hybrid"  # 混合（加权 + Borda 加权平均）


class RetrievalView(Enum):
    """检索视角。"""

    SEMANTIC = "semantic"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    CONTEXTUAL = "contextual"
    STRUCTURAL = "structural"


@dataclass
class QuerySpec:
    """检索查询规格。

    Attributes:
        tags: 语义标签列表
        causal_edges: 因果边列表 [(cause, effect), ...]
        context: 上下文匹配条件 {key: value}
        state_features: 状态空间特征向量（用于结构维）
        type_filter: 按经验类型过滤
        min_score: 最低分数阈值
    """

    tags: list[str] = field(default_factory=list)
    causal_edges: list[tuple[str, str]] = field(default_factory=list)
    context: dict[str, str] = field(default_factory=dict)
    state_features: list[float] = field(default_factory=list)
    type_filter: ExperienceType | None = None
    min_score: float = 0.0


@dataclass
class MultiViewResult:
    """多视角检索结果。

    Attributes:
        experience: 匹配的经验
        score: 综合融合分数
        view_scores: 各视角分数 {view_name: score}
        rank: 排名（1-based）
        strategy: 使用的融合策略
    """

    experience: Experience
    score: float = 0.0
    view_scores: dict[str, float] = field(default_factory=dict)
    rank: int = 0
    strategy: str = "weighted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "experience_id": self.experience.experience_id,
            "score": round(self.score, 4),
            "view_scores": {k: round(v, 4) for k, v in self.view_scores.items()},
            "rank": self.rank,
            "strategy": self.strategy,
            "experience_type": self.experience.experience_type.value,
            "outcome": self.experience.outcome,
            "tags": self.experience.tags,
        }


@dataclass
class MultiViewStats:
    """检索器统计信息。"""

    total_queries: int = 0
    avg_result_count: float = 0.0
    avg_score: float = 0.0
    view_contributions: dict[str, float] = field(default_factory=dict)
    strategy_usage: dict[str, int] = field(default_factory=dict)
    avg_latency_ms: float = 0.0


# =============================================================================
# 上下文维索引
# =============================================================================


class _ContextIndex:
    """上下文维索引：基于情境特征匹配。"""

    def __init__(self) -> None:
        self._exp_contexts: dict[str, dict[str, str]] = {}

    def add(self, exp_id: str, context: dict[str, Any]) -> None:
        """注册经验上下文。"""
        self._exp_contexts[exp_id] = {str(k): str(v) for k, v in context.items() if k and v}

    def remove(self, exp_id: str) -> None:
        """移除经验上下文。"""
        self._exp_contexts.pop(exp_id, None)

    def similarity(self, query_context: dict[str, str], exp_id: str) -> float:
        """查询上下文与经验上下文的匹配度。

        策略：精确匹配 + 部分匹配加权。
        """
        if not query_context:
            return 0.5  # 无查询条件时返回中性分数

        exp_ctx = self._exp_contexts.get(exp_id, {})
        if not exp_ctx:
            return 0.0

        matches = 0.0
        total = len(query_context)

        for key, value in query_context.items():
            exp_value = exp_ctx.get(key, "")
            if not exp_value:
                continue
            if exp_value.lower() == value.lower():
                matches += 1.0  # 精确匹配
            elif value.lower() in exp_value.lower() or exp_value.lower() in value.lower():
                matches += 0.5  # 部分匹配

        return matches / total if total > 0 else 0.0

    def statistics(self) -> dict[str, Any]:
        return {
            "n_indexed": len(self._exp_contexts),
            "avg_keys": (
                sum(len(v) for v in self._exp_contexts.values()) / len(self._exp_contexts) if self._exp_contexts else 0
            ),
        }


# =============================================================================
# 结构维索引
# =============================================================================


class _StructuralIndex:
    """结构维索引：基于状态空间特征向量的余弦相似度。"""

    def __init__(self) -> None:
        self._features: dict[str, list[float]] = {}

    def add(self, exp_id: str, features: list[float]) -> None:
        """注册状态特征向量。"""
        if features:
            self._features[exp_id] = features

    def remove(self, exp_id: str) -> None:
        """移除特征向量。"""
        self._features.pop(exp_id, None)

    def similarity(self, query_features: list[float], exp_id: str) -> float:
        """查询特征与经验特征的余弦相似度。

        cos(q, d) = (q · d) / (‖q‖ × ‖d‖)
        """
        if not query_features:
            return 0.5  # 无查询特征时返回中性分数

        exp_features = self._features.get(exp_id)
        if not exp_features:
            return 0.0

        # 维度对齐（取最小维度）
        dim = min(len(query_features), len(exp_features))
        if dim == 0:
            return 0.0

        q = query_features[:dim]
        d = exp_features[:dim]

        dot = sum(a * b for a, b in zip(q, d))
        q_norm = math.sqrt(sum(a * a for a in q))
        d_norm = math.sqrt(sum(b * b for b in d))

        if q_norm == 0 or d_norm == 0:
            return 0.0
        return max(0.0, dot / (q_norm * d_norm))  # 确保非负

    def statistics(self) -> dict[str, Any]:
        return {
            "n_indexed": len(self._features),
            "avg_dims": (sum(len(v) for v in self._features.values()) / len(self._features) if self._features else 0),
        }


# =============================================================================
# MultiViewRetriever 主类
# =============================================================================


@dataclass
class MultiViewRetriever:
    """五维融合检索器。

    五维检索视角：
        1. 语义维 (Semantic) — TF-IDF 标签相似度
        2. 因果维 (Causal) — 因果边 Jaccard 匹配
        3. 时间维 (Temporal) — 指数衰减近度
        4. 上下文维 (Contextual) — 情境匹配
        5. 结构维 (Structural) — 特征向量余弦相似度

    Example:
        >>> from mci_world_model.sdk._experience_memory import ExperienceDB, Experience
        >>> db = ExperienceDB()
        >>> retriever = MultiViewRetriever(experience_db=db)
        >>> query = QuerySpec(
        ...     tags=["pendulum", "prediction"],
        ...     causal_edges=[("theta", "omega")],
        ...     context={"domain": "physics"},
        ... )
        >>> results = retriever.retrieve(query, top_k=5)
    """

    experience_db: ExperienceDB | None = None
    fusion_strategy: FusionStrategy = FusionStrategy.WEIGHTED
    view_weights: dict[str, float] = field(
        default_factory=lambda: {
            "semantic": 0.30,
            "causal": 0.25,
            "temporal": 0.20,
            "contextual": 0.15,
            "structural": 0.10,
        }
    )

    # 内部状态
    _context_index: _ContextIndex = field(default_factory=_ContextIndex)
    _structural_index: _StructuralIndex = field(default_factory=_StructuralIndex)
    _query_count: int = 0
    _total_results: int = 0
    _total_score: float = 0.0
    _view_score_sum: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _strategy_count: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _total_latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.experience_db is None:
            self.experience_db = ExperienceDB()

    # ── 索引管理 ──

    def register_context(self, exp_id: str, context: dict[str, Any]) -> None:
        """为已存储经验注册上下文信息。"""
        self._context_index.add(exp_id, context)

    def register_features(self, exp_id: str, features: list[float]) -> None:
        """为已存储经验注册状态特征向量。"""
        self._structural_index.add(exp_id, features)

    def unregister(self, exp_id: str) -> None:
        """移除经验的上下文和结构索引。"""
        self._context_index.remove(exp_id)
        self._structural_index.remove(exp_id)

    # ── 核心检索 ──

    def retrieve(  # type: ignore
        self,
        query: QuerySpec | None = None,
        top_k: int = 5,
        strategy: FusionStrategy | None = None,
        weights: dict[str, float] | None = None,
        **kwargs,
    ) -> list[MultiViewResult]:
        """五维融合检索。

        支持三种调用方式：
            1. retrieve(QuerySpec(...)) — 完整查询规格
            2. retrieve(query=None, tags=[...], ...) — 关键字参数快捷方式
            3. retrieve(query_dict) — dict 形式

        Args:
            query: 查询规格
            top_k: 返回数量
            strategy: 融合策略（覆盖默认值）
            weights: 视角权重（覆盖默认值）
            **kwargs: 快捷参数 (tags, causal_edges, context, state_features)

        Returns:
            按综合分数降序排列的检索结果
        """
        start_time = time.time()

        # 解析查询
        if query is None:
            query = QuerySpec(
                tags=kwargs.get("tags", []),
                causal_edges=kwargs.get("causal_edges", []),
                context=kwargs.get("context", {}),
                state_features=kwargs.get("state_features", []),
                type_filter=kwargs.get("type_filter"),
                min_score=kwargs.get("min_score", 0.0),
            )
        elif isinstance(query, dict):
            query = QuerySpec(**query)

        use_strategy = strategy or self.fusion_strategy
        use_weights = weights or self.view_weights

        # 执行各视角检索
        all_experiences = self.experience_db.all_experiences  # type: ignore

        # 类型过滤
        if query.type_filter is not None:
            all_experiences = [e for e in all_experiences if e.experience_type == query.type_filter]

        scored: list[tuple[Experience, dict[str, float]]] = []

        for exp in all_experiences:
            view_scores = {}

            # 1. 语义维
            view_scores["semantic"] = self._compute_semantic(query, exp)

            # 2. 因果维
            view_scores["causal"] = self._compute_causal(query, exp)

            # 3. 时间维
            view_scores["temporal"] = self._compute_temporal(exp)

            # 4. 上下文维
            view_scores["contextual"] = self._context_index.similarity(query.context, exp.experience_id)

            # 5. 结构维
            view_scores["structural"] = self._structural_index.similarity(query.state_features, exp.experience_id)

            scored.append((exp, view_scores))

        # 融合
        if use_strategy == FusionStrategy.WEIGHTED:
            results = self._weighted_fusion(scored, use_weights, query.min_score)
        elif use_strategy == FusionStrategy.BORDA:
            results = self._borda_fusion(scored, use_weights, query.min_score)
        else:  # HYBRID
            results = self._hybrid_fusion(scored, use_weights, query.min_score)

        # 排序 + 截断
        results.sort(key=lambda r: r.score, reverse=True)
        top_results = results[:top_k]

        # 标记排名 + 更新访问
        for i, r in enumerate(top_results, 1):
            r.rank = i
            r.strategy = use_strategy.value
            r.experience.access_count += 1
            r.experience.last_accessed = time.time()

        # 统计更新
        elapsed_ms = (time.time() - start_time) * 1000
        self._query_count += 1
        self._total_results += len(top_results)
        self._total_latency_ms += elapsed_ms
        self._strategy_count[use_strategy.value] += 1
        for r in top_results:
            self._total_score += r.score
            for view, vs in r.view_scores.items():
                self._view_score_sum[view] += vs

        return top_results

    # ── 快捷检索接口 ──

    def retrieve_by_tags(self, tags: list[str], top_k: int = 5) -> list[MultiViewResult]:
        """快捷：仅语义维检索。"""
        return self.retrieve(QuerySpec(tags=tags), top_k=top_k)

    def retrieve_by_causal(self, edges: list[tuple[str, str]], top_k: int = 5) -> list[MultiViewResult]:
        """快捷：仅因果维检索。"""
        return self.retrieve(QuerySpec(causal_edges=edges), top_k=top_k)

    def retrieve_by_context(self, context: dict[str, str], top_k: int = 5) -> list[MultiViewResult]:
        """快捷：仅上下文维检索。"""
        return self.retrieve(QuerySpec(context=context), top_k=top_k)

    # ── 融合策略 ──

    def _weighted_fusion(
        self,
        scored: list[tuple[Experience, dict[str, float]]],
        weights: dict[str, float],
        min_score: float,
    ) -> list[MultiViewResult]:
        """加权线性融合。

        score = Σ_view (w_view × s_view) × importance
        """
        results = []
        for exp, view_scores in scored:
            combined = sum(weights.get(view, 0.0) * score for view, score in view_scores.items())
            combined *= exp.importance

            if combined >= min_score:
                results.append(
                    MultiViewResult(
                        experience=exp,
                        score=combined,
                        view_scores=dict(view_scores),
                    )
                )
        return results

    def _borda_fusion(
        self,
        scored: list[tuple[Experience, dict[str, float]]],
        weights: dict[str, float],
        min_score: float,
    ) -> list[MultiViewResult]:
        """Borda 排名融合。

        对每个视角，按分数降序排名，排名转化为 Borda 分：
        borda_score = (N - rank) / N

        最终分数 = Σ_view (w_view × borda_view) × importance
        """
        n = len(scored)
        if n == 0:
            return []

        views = ["semantic", "causal", "temporal", "contextual", "structural"]

        # 按各视角排序，分配 Borda 分
        borda: dict[str, dict[str, float]] = {exp.experience_id: {} for exp, _ in scored}

        for view in views:
            # 按该视角分数降序
            ranked = sorted(scored, key=lambda x: x[1].get(view, 0.0), reverse=True)
            for rank, (exp, _) in enumerate(ranked):
                borda_score = (n - rank) / n if n > 0 else 0.0
                borda[exp.experience_id][view] = borda_score

        # 加权汇总
        results = []
        for exp, raw_scores in scored:
            combined = sum(weights.get(view, 0.0) * borda[exp.experience_id].get(view, 0.0) for view in views)
            combined *= exp.importance

            if combined >= min_score:
                results.append(
                    MultiViewResult(
                        experience=exp,
                        score=combined,
                        view_scores=dict(raw_scores),
                    )
                )
        return results

    def _hybrid_fusion(
        self,
        scored: list[tuple[Experience, dict[str, float]]],
        weights: dict[str, float],
        min_score: float,
    ) -> list[MultiViewResult]:
        """混合融合：50% weighted + 50% Borda。"""
        weighted = self._weighted_fusion(scored, weights, 0.0)
        borda = self._borda_fusion(scored, weights, 0.0)

        # 建索引
        w_map = {r.experience.experience_id: r for r in weighted}
        b_map = {r.experience.experience_id: r for r in borda}

        results = []
        for exp, view_scores in scored:
            eid = exp.experience_id
            w_score = w_map.get(eid, MultiViewResult(experience=exp)).score
            b_score = b_map.get(eid, MultiViewResult(experience=exp)).score
            combined = 0.5 * w_score + 0.5 * b_score

            if combined >= min_score:
                results.append(
                    MultiViewResult(
                        experience=exp,
                        score=combined,
                        view_scores=dict(view_scores),
                    )
                )
        return results

    # ── 视角分数计算 ──

    def _compute_semantic(self, query: QuerySpec, exp: Experience) -> float:
        """语义维分数：委托给 ExperienceDB 的 _SemanticIndex。"""
        if not query.tags:
            return 0.5  # 无查询标签 → 中性分数
        return self.experience_db._semantic_index.similarity(query.tags, exp.tags)  # type: ignore

    def _compute_causal(self, query: QuerySpec, exp: Experience) -> float:
        """因果维分数：委托给 ExperienceDB 的 _CausalIndex。"""
        if not query.causal_edges:
            return 0.5  # 无查询边 → 中性分数
        return self.experience_db._causal_index.similarity(query.causal_edges, exp.causal_edges)  # type: ignore

    def _compute_temporal(self, exp: Experience) -> float:
        """时间维分数：委托给 ExperienceDB 的 _TemporalIndex。"""
        if self.experience_db._temporal_index is not None:  # type: ignore
            return self.experience_db._temporal_index.score_with_access(  # type: ignore
                exp.timestamp, exp.last_accessed, exp.access_count
            )
        return exp.recency_score()

    # ── 统计 ──

    def statistics(self) -> MultiViewStats:
        """检索器统计信息。"""
        n_views = 5
        avg_contributions = {}
        total_queries_with_results = max(1, self._query_count)
        for view, total in self._view_score_sum.items():
            avg_contributions[view] = total / total_queries_with_results / n_views

        return MultiViewStats(
            total_queries=self._query_count,
            avg_result_count=self._total_results / max(1, self._query_count),
            avg_score=self._total_score / max(1, self._total_results),
            view_contributions=avg_contributions,
            strategy_usage=dict(self._strategy_count),
            avg_latency_ms=self._total_latency_ms / max(1, self._query_count),
        )

    def reset_stats(self) -> None:
        """重置统计计数器。"""
        self._query_count = 0
        self._total_results = 0
        self._total_score = 0.0
        self._view_score_sum = defaultdict(float)
        self._strategy_count = defaultdict(int)
        self._total_latency_ms = 0.0

    def clear(self) -> None:
        """清空检索器（同时清空关联的 ExperienceDB）。"""
        if self.experience_db:
            self.experience_db.clear()
        self._context_index = _ContextIndex()
        self._structural_index = _StructuralIndex()
        self.reset_stats()
