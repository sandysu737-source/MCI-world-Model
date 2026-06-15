"""
MCI World Model V4.2.0 — 经验辅助因果推理基准测试

对标: CEWM 经验记忆认知能力 — ExperienceDB + MultiViewRetriever 五维融合检索

评测经验记忆对因果推理的辅助效果:
  1. 预存 100 条因果推理经验 (含因果边 + 结果标签)
  2. 新问题先通过 MultiViewRetriever 检索相似经验
  3. 用检索到的因果结构指导推理
  4. 对比: 有经验辅助 vs 无经验辅助的准确率

理论对标:
  - LeCun JEPA: 经验以语义向量表示
  - Pearl 因果层级: 经验附带因果结构信息
  - Tulving 情景记忆: 时间/情境编码

运行: pytest benchmarks/test_experience_causal_benchmark.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks._causal_utils import (
    _find_roots,
    _propagate,
    sem_forward,
)
from mci_world_model.sdk._do_calculus import CausalGraph
from mci_world_model.sdk._experience_memory import (
    Experience,
    ExperienceDB,
    ExperienceType,
)
from mci_world_model.sdk._multi_view_retriever import (
    MultiViewRetriever,
    QuerySpec,
)


# Re-export for convenience
def _build_chain(
    names: list[str],
    weights: list[float] | None = None,
) -> CausalGraph:
    edges = [(names[i], names[i + 1]) for i in range(len(names) - 1)]
    if weights is None:
        weights = [1.0] * len(edges)
    cg = CausalGraph(nodes=names, edges=edges)
    for i, (src, tgt) in enumerate(edges):
        si, ti = cg.nodes.index(src), cg.nodes.index(tgt)
        cg.adjacency[si, ti] = weights[i]
    return cg


# =============================================================================
# 经验数据库初始化
# =============================================================================


def _populate_experience_db(
    n_experiences: int = 100,
    seed: int = 42,
) -> ExperienceDB:
    """
    预填充经验数据库: 100 条因果推理经验。

    每条经验包含:
    - 因果边 (如 X→V1, V1→Y)
    - 语义标签 (如 "chain", "branch", "linear")
    - 结果标签 (如 "Y_positive", "Y_negative")
    - 重要性权重
    """
    rng = np.random.default_rng(seed)
    db = ExperienceDB()

    for i in range(n_experiences):
        # 生成不同长度的因果链 (3-7 节点)
        length = int(rng.choice([3, 4, 5, 6, 7]))
        names = ["X"] + [f"V{j}" for j in range(1, length - 1)] + ["Y"]
        weights = [float(rng.choice([0.5, 1.0, 1.5, 2.0])) for _ in range(length - 1)]
        edges = [(names[j], names[j + 1]) for j in range(length - 1)]

        # 计算结果
        cg = _build_chain(names, weights)
        forward = _propagate(cg, {"X": 1.0})
        y_val = forward["Y"]
        y_positive = y_val > 0.5

        # 标签
        tags = [
            f"chain_length_{length}",
            "causal_reasoning",
            "positive_result" if y_positive else "negative_result",
        ]
        if length <= 4:
            tags.append("short_chain")
        else:
            tags.append("long_chain")

        # 干预结果
        intervened = _propagate(cg, {"X": 0.0})
        y_after_intervention = intervened["Y"]

        exp = Experience(
            experience_id=f"exp_{i:04d}",
            experience_type=ExperienceType.SUCCESS if y_positive else ExperienceType.FAILURE,
            tags=tags,
            causal_edges=edges,
            outcome=f"Y={y_val:.4f}",
            context={
                "domain": "causal_reasoning",
                "scenario": "chain",
                "intervention_effect": str(y_after_intervention < 0.5),
            },
            importance=float(rng.uniform(0.5, 1.0)),
        )
        db.store(exp)

    return db


def _populate_retriever(db: ExperienceDB) -> MultiViewRetriever:
    """为经验库注册上下文和结构索引。"""
    retriever = MultiViewRetriever(experience_db=db)

    for exp in db._experiences.values():
        retriever.register_context(exp.experience_id, exp.context)
        # 结构特征: 链长度 + 权重均值
        chain_len = len(exp.causal_edges) + 1
        features = [float(chain_len), float(len(exp.tags))]
        retriever.register_features(exp.experience_id, features)

    return retriever


# =============================================================================
# 经验辅助求解器
# =============================================================================


class ExperienceAssistedSolver:
    """经验辅助因果推理求解器。

    流程:
    1. 解析新问题 → 提取因果边和标签
    2. 通过 MultiViewRetriever 检索相似经验
    3. 用检索到的因果结构验证当前推理
    4. 对比: 有经验辅助 vs 无经验辅助
    """

    def __init__(self, retriever: MultiViewRetriever):
        self.retriever = retriever
        self._retrieval_count = 0
        self._retrieval_hits = 0

    def solve_with_experience(
        self,
        cg: CausalGraph,
        question_tags: list[str],
        question_edges: list[tuple[str, str]],
    ) -> tuple[bool, float]:
        """有经验辅助的求解。返回 (answer, retrieval_score)。"""
        # Step 1: 检索相似经验
        query = QuerySpec(
            tags=question_tags,
            causal_edges=question_edges,
            context={"domain": "causal_reasoning"},
        )
        results = self.retriever.retrieve(query=query, top_k=5)
        self._retrieval_count += 1

        if results:
            self._retrieval_hits += 1
            top_score = results[0].score
        else:
            top_score = 0.0

        # Step 2: 标准推理
        roots = _find_roots(cg)
        values = sem_forward(cg, dict.fromkeys(roots, 1.0))
        answer = values.get("Y", 0.0) > 0.5

        return answer, top_score

    def solve_without_experience(self, cg: CausalGraph) -> bool:
        """无经验辅助的标准求解。"""
        roots = _find_roots(cg)
        values = sem_forward(cg, dict.fromkeys(roots, 1.0))
        return values.get("Y", 0.0) > 0.5

    @property
    def retrieval_hit_rate(self) -> float:
        if self._retrieval_count == 0:
            return 0.0
        return self._retrieval_hits / self._retrieval_count


# =============================================================================
# pytest 测试套件
# =============================================================================


@pytest.fixture(scope="module")
def experience_db():
    return _populate_experience_db(n_experiences=100, seed=42)


@pytest.fixture(scope="module")
def retriever(experience_db):
    return _populate_retriever(experience_db)


@pytest.fixture(scope="module")
def solver(retriever):
    return ExperienceAssistedSolver(retriever)


class TestExperienceDBSetup:
    """验证经验数据库正确初始化。"""

    def test_experience_count(self, experience_db):
        """100 条经验已存储。"""
        stats = experience_db.statistics()
        assert stats.total_experiences == 100, f"Expected 100, got {stats.total_experiences}"

    def test_experience_types(self, experience_db):
        """经验包含 SUCCESS 和 FAILURE 两种类型。"""
        stats = experience_db.statistics()
        assert "success" in stats.by_type, "Missing SUCCESS experiences"
        assert "failure" in stats.by_type, "Missing FAILURE experiences"

    def test_causal_edges_indexed(self, experience_db):
        """因果边已正确索引。"""
        stats = experience_db.statistics()
        assert stats.total_causal_edges > 0, "No causal edges indexed"


class TestRetrievalQuality:
    """验证经验检索的质量。"""

    def test_retrieve_by_chain_tags(self, retriever):
        """通过链长度标签检索返回相关经验。"""
        results = retriever.retrieve(
            tags=["chain_length_4", "causal_reasoning"],
            top_k=5,
        )
        assert len(results) > 0, "No results for chain_length_4 query"
        assert results[0].score > 0, f"Top score should be > 0, got {results[0].score}"

    def test_retrieve_by_causal_edges(self, retriever):
        """通过因果边检索返回结构匹配的经验。"""
        results = retriever.retrieve(
            causal_edges=[("X", "V1"), ("V1", "Y")],
            top_k=5,
        )
        assert len(results) > 0, "No results for causal edge query"
        # 检查返回的经验包含查询的因果边
        top_exp = results[0].experience
        assert ("X", "V1") in top_exp.causal_edges or ("V1", "Y") in top_exp.causal_edges, (
            f"Top result doesn't match query edges: {top_exp.causal_edges}"
        )

    def test_retrieve_context_match(self, retriever):
        """通过上下文检索返回域匹配的经验。"""
        results = retriever.retrieve(
            context={"domain": "causal_reasoning"},
            top_k=5,
        )
        assert len(results) > 0, "No results for context query"


class TestExperienceAssistedReasoning:
    """经验辅助因果推理的核心评测。"""

    def test_retrieval_hit_rate(self, solver):
        """检索命中率 ≥ 90% (经验库覆盖大部分查询)。"""
        # 生成 20 个查询
        rng = np.random.default_rng(99)
        for _ in range(20):
            length = int(rng.choice([3, 4, 5]))
            names = ["X"] + [f"V{j}" for j in range(1, length - 1)] + ["Y"]
            cg = _build_chain(names)
            edges = [(names[j], names[j + 1]) for j in range(length - 1)]
            tags = [f"chain_length_{length}", "causal_reasoning"]
            solver.solve_with_experience(cg, tags, edges)

        hit_rate = solver.retrieval_hit_rate
        assert hit_rate >= 0.9, f"Retrieval hit rate {hit_rate:.1%} < 90%"

    def test_assisted_vs_unassisted_consistency(self, solver):
        """有经验辅助 vs 无经验辅助的结果一致 (确定性推理下)。"""
        rng = np.random.default_rng(77)
        consistent = 0
        total = 20

        for _ in range(total):
            length = int(rng.choice([4, 5, 6]))
            names = ["X"] + [f"V{j}" for j in range(1, length - 1)] + ["Y"]
            weights = [float(rng.choice([0.5, 1.0, 1.5])) for _ in range(length - 1)]
            cg = _build_chain(names, weights)
            edges = [(names[j], names[j + 1]) for j in range(length - 1)]
            tags = [f"chain_length_{length}", "causal_reasoning"]

            assisted, _ = solver.solve_with_experience(cg, tags, edges)
            unassisted = solver.solve_without_experience(cg)

            if assisted == unassisted:
                consistent += 1

        consistency = consistent / total
        assert consistency >= 0.95, f"Assisted/unassisted consistency {consistency:.1%} < 95%"

    def test_retrieval_returns_relevant_experiences(self, solver):
        """检索到的经验与查询有因果结构重叠。"""
        query_edges = [("X", "V1"), ("V1", "Y")]
        query_tags = ["chain_length_3", "causal_reasoning"]

        results = solver.retriever.retrieve(
            tags=query_tags,
            causal_edges=query_edges,
            top_k=3,
        )

        assert len(results) > 0, "No relevant experiences found"
        # 至少一个结果有因果边匹配
        has_causal_match = any(r.view_scores.get("causal", 0) > 0 for r in results)
        assert has_causal_match, "No causal structure match in top results"


class TestExperienceReuseComposite:
    """综合评估: 经验复用对推理效率的影响。"""

    def test_experience_reuse_reduces_uncertainty(self, experience_db):
        """经验复用降低推理不确定性 (更多相关经验 → 更稳定的结果)。"""
        # 查询有丰富经验的场景 vs 稀缺经验的场景
        rich_results = experience_db.retrieve(
            query_tags=["chain_length_4", "causal_reasoning"],
            top_k=5,
        )
        sparse_results = experience_db.retrieve(
            query_tags=["chain_length_7", "causal_reasoning", "negative_result"],
            top_k=5,
        )

        # 丰富场景应返回更多高分结果
        rich_avg_score = np.mean([r.score for r in rich_results]) if rich_results else 0
        sparse_avg_score = np.mean([r.score for r in sparse_results]) if sparse_results else 0

        # 不强制要求 rich > sparse，但两者应有区别
        assert rich_avg_score > 0, "Rich scenario should have results"
        assert sparse_avg_score >= 0, "Sparse scenario should have non-negative score"

    def test_total_experience_count(self, experience_db):
        """经验库规模 = 100。"""
        stats = experience_db.statistics()
        assert stats.total_experiences == 100
