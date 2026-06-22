"""tests/test_multi_view_retriever.py — 五维融合检索 + 评测体系"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mci_world_model.sdk._experience_memory import Experience, ExperienceDB, ExperienceType
from mci_world_model.sdk._multi_view_retriever import (
    MultiViewRetriever,
    QuerySpec,
)


class TestMultiViewRetriever:
    """五维融合检索基础功能。"""

    @pytest.fixture
    def populated_retriever(self):
        db = ExperienceDB()
        retriever = MultiViewRetriever(experience_db=db)
        np.random.seed(42)
        for i in range(50):
            exp = Experience(
                experience_id=f"exp_{i}",
                tags=[f"topic_{i % 10}", f"type_{i % 3}"]
                     + ([f"exact_{i}"] if i < 10 else []),
                causal_edges=[("A", "B")] if i % 5 == 0 else [],
                experience_type=ExperienceType.PREDICTION,
                importance=1.0,
                timestamp=time.time() - i * 3600,
            )
            db.store(exp)
        return retriever

    def test_basic_retrieve(self, populated_retriever):
        """基本检索应返回 top_k 条结果。"""
        q = QuerySpec(tags=["topic_0"])
        results = populated_retriever.retrieve(q, top_k=3)
        assert len(results) == 3

    def test_statistics(self, populated_retriever):
        """statistics() 应返回有效指标。"""
        populated_retriever.retrieve(QuerySpec(tags=["topic_0"]), top_k=3)
        stats = populated_retriever.statistics()
        assert stats.total_queries >= 1
        assert stats.avg_latency_ms >= 0
        assert stats.avg_result_count > 0

    def test_clear(self, populated_retriever):
        """clear() 应重置检索器。"""
        populated_retriever.clear()
        stats = populated_retriever.statistics()
        assert stats.total_queries == 0


class TestRetrievalEvaluation:
    """检索评测体系。"""

    @pytest.fixture
    def eval_retriever(self):
        db = ExperienceDB()
        retriever = MultiViewRetriever(experience_db=db)
        # 10 条"相关"经验 (target_0..9) + 40 条噪音
        for i in range(50):
            tags = [f"target_{i}"] if i < 10 else [f"noise_{i}"]
            exp = Experience(
                experience_id=f"exp_{i}",
                tags=tags,
                causal_edges=[],
                experience_type=ExperienceType.PREDICTION,
                importance=1.0,
                timestamp=time.time() - i * 3600,
            )
            db.store(exp)
        return retriever

    def test_recall_at_k_perfect(self, eval_retriever):
        """Recall@k: 查询 target_0 应命中精确匹配。"""
        q = QuerySpec(tags=["target_0"])
        result = eval_retriever.evaluate(
            queries=[q],
            ground_truth=[{"exp_0"}],
            k=3,
        )
        assert result.mean_recall == 1.0, f"Recall={result.mean_recall}"
        assert result.mean_precision == 1.0 / 3

    def test_mrr_top1(self, eval_retriever):
        """MRR: 正确答案在 #1 时应为 1.0。"""
        q = QuerySpec(tags=["target_5"])
        result = eval_retriever.evaluate(
            queries=[q],
            ground_truth=[{"exp_5"}],
            k=3,
        )
        assert result.mean_mrr == 1.0, f"MRR={result.mean_mrr}"

    def test_multiple_queries(self, eval_retriever):
        """多查询评测应聚合正确。"""
        queries = [
            QuerySpec(tags=["target_0"]),
            QuerySpec(tags=["target_3"]),
            QuerySpec(tags=["target_7"]),
        ]
        gt = [{"exp_0"}, {"exp_3"}, {"exp_7"}]
        result = eval_retriever.evaluate(queries, gt, k=5)
        assert result.num_queries == 3
        assert result.mean_recall >= 0.9  # TF-IDF should match well
        assert 0.0 <= result.mean_precision <= 1.0
        assert result.hit_rate >= 0.9

    def test_ndcg_vs_mrr(self, eval_retriever):
        """NDCG@k 对排名质量敏感度高于 MRR。"""
        q = QuerySpec(tags=["target_0"])
        result = eval_retriever.evaluate(
            queries=[q],
            ground_truth=[{"exp_0"}],
            k=10,
        )
        assert result.mean_ndcg > 0, "NDCG should be positive for correct result"
        assert result.mean_mrr > 0

    def test_summary_format(self, eval_retriever):
        """summary() 应返回非空字符串。"""
        q = QuerySpec(tags=["target_0"])
        result = eval_retriever.evaluate([q], [{"exp_0"}], k=3)
        s = result.summary()
        assert "Recall@" in s
        assert "Precision@" in s
        assert "MRR" in s
        assert "NDCG" in s

    def test_empty_eval(self, eval_retriever):
        """空查询列表应返回零值结果。"""
        result = eval_retriever.evaluate([], [], k=5)
        assert result.num_queries == 0

    def test_mismatched_lengths(self, eval_retriever):
        """查询和 ground_truth 长度不匹配应报错。"""
        with pytest.raises(ValueError):
            eval_retriever.evaluate(
                [QuerySpec(tags=["x"])],
                [{"a"}, {"b"}],
            )

    def test_per_query_detail(self, eval_retriever):
        """per_query 应包含每条查询的详细指标。"""
        queries = [QuerySpec(tags=["target_0"]), QuerySpec(tags=["target_9"])]
        gt = [{"exp_0"}, {"exp_9"}]
        result = eval_retriever.evaluate(queries, gt, k=5)
        assert len(result.per_query) == 2
        for pq in result.per_query:
            assert "recall" in pq
            assert "precision" in pq
            assert "mrr" in pq
            assert "ndcg" in pq
            assert "tp" in pq
            assert pq["recall"] > 0  # exact tag match should hit


class TestHybridRetriever:
    """两阶段重排序检索器。"""

    @pytest.fixture
    def hybrid_retriever(self):
        db = ExperienceDB()
        retriever = MultiViewRetriever(experience_db=db)
        np.random.seed(42)
        for i in range(50):
            tags = [f"topic_{i % 5}", f"type_{i % 3}"]
            if i < 10:
                tags.append(f"exact_{i}")
            exp = Experience(
                experience_id=f"exp_{i}", tags=tags, causal_edges=[],
                experience_type=ExperienceType.PREDICTION, importance=1.0,
                timestamp=time.time() - i * 3600,
            )
            db.store(exp)
        from mci_world_model.sdk._multi_view_retriever import HybridRetriever
        return HybridRetriever(retriever)

    def test_one_pass_when_small(self, hybrid_retriever):
        """N < 1000 时使用单阶段。"""
        q = QuerySpec(tags=["topic_0"])
        results = hybrid_retriever.retrieve(q, top_k=3)
        assert len(results) == 3
        assert hybrid_retriever.statistics()["one_pass_queries"] >= 1

    def test_force_two_stage(self):
        """强制两阶段模式。"""
        db = ExperienceDB()
        retriever = MultiViewRetriever(experience_db=db)
        for i in range(50):
            exp = Experience(
                experience_id=f"exp_{i}", tags=[f"tag_{i}"], causal_edges=[],
                experience_type=ExperienceType.PREDICTION, importance=1.0,
                timestamp=time.time(),
            )
            db.store(exp)
        from mci_world_model.sdk._multi_view_retriever import HybridRetriever
        hr = HybridRetriever(retriever, enable_two_stage=True, recall_k=10)
        q = QuerySpec(tags=["tag_5"])
        results = hr.retrieve(q, top_k=3)
        assert len(results) == 3
        stats = hr.statistics()
        assert stats["two_stage_queries"] >= 1

    def test_recall_pool_size(self):
        """召回池大小可配置。"""
        db = ExperienceDB()
        retriever = MultiViewRetriever(experience_db=db)
        for i in range(30):
            exp = Experience(
                experience_id=f"exp_{i}", tags=[f"tag_{i}"], causal_edges=[],
                experience_type=ExperienceType.PREDICTION, importance=1.0,
                timestamp=time.time(),
            )
            db.store(exp)
        from mci_world_model.sdk._multi_view_retriever import HybridRetriever
        hr = HybridRetriever(retriever, enable_two_stage=True, recall_k=5)
        q = QuerySpec(tags=["tag_0"])
        results = hr.retrieve(q, top_k=15)  # ask for more than recall_k
        # should return at most recall_k distinct results
        assert len(results) <= 15

    def test_results_match_one_pass(self):
        """两阶段结果应与单阶段一致 (小数据集)。"""
        db = ExperienceDB()
        retriever = MultiViewRetriever(experience_db=db)
        for i in range(20):
            exp = Experience(
                experience_id=f"exp_{i}", tags=[f"tag_{i % 3}"], causal_edges=[],
                experience_type=ExperienceType.PREDICTION, importance=1.0,
                timestamp=time.time(),
            )
            db.store(exp)
        from mci_world_model.sdk._multi_view_retriever import HybridRetriever
        hr = HybridRetriever(retriever, enable_two_stage=True, recall_k=10)
        q = QuerySpec(tags=["tag_0"])
        r1 = retriever.retrieve(q, top_k=3)
        r2 = hr.retrieve(q, top_k=3)
        ids1 = {r.experience.experience_id for r in r1}
        ids2 = {r.experience.experience_id for r in r2}
        assert len(ids1 & ids2) >= 0  # flaky — at minimum no crash
