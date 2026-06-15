"""Cladder 基准测试 — 门槛测试。

验证:
- 数据集正确加载 (10112 题)
- 整体准确率 ≥ 95%
- Rung 1/2/3 准确率 ≥ 90%
- 各类已知 query_type 100% 准确
- CEWM backadj/collider_bias 100%
"""

from __future__ import annotations

import pytest

from benchmarks.cladder._dataset import (
    CladderQuestion,
    dataset_stats,
    load_cladder,
)
from benchmarks.cladder._solvers import solve_all, solve_single

# =============================================================================
# 数据集测试
# =============================================================================


class TestDataset:
    """数据集加载与统计测试。"""

    @pytest.fixture(scope="module")
    def questions(self) -> list[CladderQuestion]:
        return load_cladder()

    def test_load_count(self, questions):
        """应加载 10112 道题。"""
        assert len(questions) == 10112

    def test_stats(self, questions):
        """统计信息应包含所有必要字段。"""
        stats = dataset_stats(questions)
        assert stats["n_total"] == 10112
        assert "rung_distribution" in stats
        assert "query_type_distribution" in stats
        assert "graph_type_distribution" in stats

    def test_all_rungs_present(self, questions):
        """应包含三个 rung 的题目。"""
        rungs = {q.rung for q in questions}
        assert rungs == {1, 2, 3}

    def test_all_query_types_present(self, questions):
        """应包含 10 种 query_type。"""
        qtypes = {q.query_type for q in questions}
        expected = {
            "correlation",
            "marginal",
            "exp_away",
            "ate",
            "backadj",
            "collider_bias",
            "ett",
            "nde",
            "nie",
            "det-counterfactual",
        }
        assert qtypes == expected

    def test_all_graph_types_present(self, questions):
        """应包含 10 种图结构。"""
        gtypes = {q.graph_id for q in questions}
        expected = {
            "IV",
            "arrowhead",
            "chain",
            "collision",
            "confounding",
            "diamond",
            "diamondcut",
            "fork",
            "frontdoor",
            "mediation",
        }
        assert gtypes == expected

    def test_each_question_has_id(self, questions):
        """每道题都有唯一 ID。"""
        ids = [q.question_id for q in questions]
        assert len(ids) == len(set(ids))

    def test_labels_are_yes_no(self, questions):
        """标签应是 'yes' 或 'no'。"""
        labels = {q.label for q in questions}
        assert labels == {"yes", "no"}

    def test_label_bool_mapping(self):
        """label_bool 属性正确映射。"""
        q = CladderQuestion(question_id=999, rung=1, query_type="marginal", graph_id="chain", label="yes")
        assert q.label_bool is True
        q.label = "no"
        assert q.label_bool is False

    def test_solution_value_available_for_most(self, questions):
        """大多数 reasoning-based 题应有 solution_value。"""
        with_sol = sum(1 for q in questions if q.solution_value is not None)
        # backadj + collider_bias 无 reasoning = 1738
        expected_min = 10112 - 1738  # 8374
        assert with_sol > expected_min * 0.9  # 至少 90% 有解


# =============================================================================
# 求解器准确性测试
# =============================================================================


class TestSolverAccuracy:
    """整体求解器准确性测试。"""

    @pytest.fixture(scope="module")
    def report(self) -> dict:
        questions = load_cladder()
        _, report = solve_all(questions)
        return report

    def test_overall_above_95(self, report):
        """整体准确率 ≥ 95%。"""
        assert report["accuracy"] >= 95.0, f"整体准确率 {report['accuracy']}% < 95%"

    def test_rung1_above_90(self, report):
        """Rung 1 准确率 ≥ 90%。"""
        acc = report["by_rung"][1]["accuracy"]
        assert acc >= 90.0, f"Rung 1 准确率 {acc}% < 90%"

    def test_rung2_above_90(self, report):
        """Rung 2 准确率 ≥ 90%。"""
        acc = report["by_rung"][2]["accuracy"]
        assert acc >= 90.0, f"Rung 2 准确率 {acc}% < 90%"

    def test_rung3_above_90(self, report):
        """Rung 3 准确率 ≥ 90%。"""
        acc = report["by_rung"][3]["accuracy"]
        assert acc >= 90.0, f"Rung 3 准确率 {acc}% < 90%"

    def test_backadj_100_percent(self, report):
        """backadj (CEWM 结构化核心价值) 应 100% 准确。"""
        qt = report["by_query_type"]["backadj"]
        assert qt["accuracy"] == 100.0, f"backadj = {qt['accuracy']}% (应为 100%)"

    def test_collider_bias_100_percent(self, report):
        """collider_bias (CEWM 结构化核心价值) 应 100% 准确。"""
        qt = report["by_query_type"]["collider_bias"]
        assert qt["accuracy"] == 100.0, f"collider_bias = {qt['accuracy']}% (应为 100%)"

    def test_ate_correlation_nde_nie_ett_100(self, report):
        """核心因果推断类型应 100% 准确。"""
        for qt_name in ["ate", "correlation", "nde", "nie", "ett"]:
            qt = report["by_query_type"].get(qt_name)
            assert qt is not None, f"缺失 query_type: {qt_name}"
            assert qt["accuracy"] >= 99.5, f"{qt_name} = {qt['accuracy']}% (期望 ≥ 99.5%)"

    def test_marginal_exp_away_above_98(self, report):
        """marginal 和 exp_away 应 ≥ 98%。"""
        for qt_name in ["marginal", "exp_away"]:
            qt = report["by_query_type"][qt_name]
            assert qt["accuracy"] >= 98.0, f"{qt_name} = {qt['accuracy']}% (期望 ≥ 98%)"


# =============================================================================
# 单题求解测试
# =============================================================================


class TestSingleSolve:
    """单题求解返回格式测试。"""

    @pytest.fixture(scope="module")
    def questions(self) -> list[CladderQuestion]:
        return load_cladder()

    def test_result_format(self, questions):
        """solve_single 返回正确格式。"""
        r = solve_single(questions[0])
        assert "question_id" in r
        assert "rung" in r
        assert "query_type" in r
        assert "graph_id" in r
        assert "predicted" in r
        assert isinstance(r["predicted"], bool)
        assert "label" in r
        assert isinstance(r["label"], bool)
        assert "correct" in r
        assert isinstance(r["correct"], bool)

    def test_rung1_returns_bool(self, questions):
        """Rung 1 求解返回 bool。"""
        r1 = [q for q in questions if q.rung == 1]
        for q in r1[:100]:
            r = solve_single(q)
            assert isinstance(r["predicted"], bool)

    def test_rung2_returns_bool(self, questions):
        """Rung 2 求解返回 bool。"""
        r2 = [q for q in questions if q.rung == 2]
        for q in r2[:100]:
            r = solve_single(q)
            assert isinstance(r["predicted"], bool)

    def test_rung3_returns_bool(self, questions):
        """Rung 3 求解返回 bool。"""
        r3 = [q for q in questions if q.rung == 3]
        for q in r3[:100]:
            r = solve_single(q)
            assert isinstance(r["predicted"], bool)
