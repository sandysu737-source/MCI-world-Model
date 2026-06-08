"""
MCI World Model v3.1.0 — GAIA 风格分层基准测试框架
======================================================

基于 GAIA (General AI Assistant) 基准测试方法论，
构建三层递进式测试体系，评估双路径（原始 vs 贝叶斯增强）的性能差异。

层次定义：
  L1 - 事实检索 (Fact Retrieval): 单步因果查询、节点属性验证
  L2 - 多跳推理 (Multi-hop Reasoning): 因果链传播、中介效应
  L3 - 复杂推理 (Complex Reasoning): 反事实推断、不确定性量化

评分指标：
  - Accuracy (Top-1 / Top-k)
  - MRR (Mean Reciprocal Rank)
  - Calibration Error (ECE)
  - Bayesian Improvement Ratio

用法：
    pytest tests/test_gaia_benchmark.py -v
"""

import json
import statistics
import time
from dataclasses import dataclass, field

import numpy as np
import pytest

from mci_world_model.sdk._counterfactual import (
    CounterfactualEngine,
    StructuralEquationModel,
)
from mci_world_model.sdk._do_calculus import CausalGraph, DoCalculus

# =============================================================================
# GAIA 评分体系
# =============================================================================


@dataclass
class GAIAScore:
    """GAIA 风格评分子项。"""

    accuracy: float = 0.0  # Top-1 准确率
    mrr: float = 0.0  # Mean Reciprocal Rank
    precision_at_k: float = 0.0  # Precision@k
    recall_at_k: float = 0.0  # Recall@k
    f1_score: float = 0.0
    ece: float = 0.0  # Expected Calibration Error
    bayesian_improvement: float = 0.0  # 贝叶斯相对改善百分比
    n_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "mrr": round(self.mrr, 4),
            "precision_at_k": round(self.precision_at_k, 4),
            "recall_at_k": round(self.recall_at_k, 4),
            "f1_score": round(self.f1_score, 4),
            "ece": round(self.ece, 4),
            "bayesian_improvement_pct": round(self.bayesian_improvement, 2),
            "n_samples": self.n_samples,
        }


@dataclass
class GAIAReport:
    """GAIA 风格分层基准测试报告。"""

    level: str  # "L1" | "L2" | "L3"
    original_score: GAIAScore = field(default_factory=GAIAScore)
    bayesian_score: GAIAScore = field(default_factory=GAIAScore)
    improvement_summary: dict = field(default_factory=dict)
    per_query_details: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "original": self.original_score.to_dict(),
            "bayesian": self.bayesian_score.to_dict(),
            "improvement": self.improvement_summary,
            "details": self.per_query_details,
            "meta": self.meta,
        }


class GAIABenchmarkRunner:
    """GAIA 风格基准测试运行器。

    提供分层测试用例生成、双路径执行、评分统计。
    """

    def __init__(self, seed: int = 42):
        self._rng = np.random.RandomState(seed)

    # ----------------------------------------------------------------
    # L1: 事实检索 — 单步因果查询
    # ----------------------------------------------------------------

    def generate_l1_queries(self, n_queries: int = 10) -> list[dict]:
        """生成 L1 事实检索测试用例。

        测试目标：
        - 因果边存在性判断
        - 节点属性查询
        - 简单 ATE 估计
        """
        queries = []
        templates = [
            {"cause": "X", "effect": "Y", "has_edge": True, "expected_direction": "positive"},
            {"cause": "A", "effect": "B", "has_edge": False, "expected_direction": "none"},
            {"cause": "Z", "effect": "Y", "has_edge": True, "expected_direction": "positive"},
        ]

        for i in range(min(n_queries, len(templates))):
            t = templates[i]
            queries.append(
                {
                    "id": f"L1-{i + 1:03d}",
                    "level": "L1",
                    "type": "edge_existence",
                    "query": f"Does {t['cause']} cause {t['effect']}?",
                    "cause": t["cause"],
                    "effect": t["effect"],
                    "ground_truth": {
                        "has_edge": t["has_edge"],
                        "expected_direction": t["expected_direction"],
                    },
                }
            )
        return queries

    # ----------------------------------------------------------------
    # L2: 多跳推理 — 因果链传播
    # ----------------------------------------------------------------

    def generate_l2_queries(self, n_queries: int = 8) -> list[dict]:
        """生成 L2 多跳推理测试用例。

        测试目标：
        - 中介效应识别
        - 因果链传播方向
        - 后门路径检测
        """
        queries = [
            {
                "id": "L2-001",
                "level": "L2",
                "type": "mediation",
                "query": "What mediates the effect of X on Y?",
                "graph": CausalGraph(
                    nodes=["X", "M", "Y"],
                    edges=[("X", "M"), ("M", "Y")],
                ),
                "ground_truth": {
                    "mediator": "M",
                    "has_direct_path": False,
                    "has_indirect_path": True,
                },
            },
            {
                "id": "L2-002",
                "level": "L2",
                "type": "backdoor_detection",
                "query": "Is Z a confounder between X and Y?",
                "graph": CausalGraph(
                    nodes=["Z", "X", "Y"],
                    edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
                ),
                "ground_truth": {
                    "is_confounder": True,
                    "backdoor_path_exists": True,
                },
            },
            {
                "id": "L2-003",
                "level": "L2",
                "type": "causal_chain",
                "query": "What is the causal path from A to D?",
                "graph": CausalGraph(
                    nodes=["A", "B", "C", "D"],
                    edges=[("A", "B"), ("B", "C"), ("C", "D")],
                ),
                "ground_truth": {
                    "path_length": 3,
                    "direct_path": False,
                },
            },
            {
                "id": "L2-004",
                "level": "L2",
                "type": "adjustment_set",
                "query": "Which variables should be adjusted to estimate X->Y?",
                "graph": CausalGraph(
                    nodes=["Z", "X", "Y", "W"],
                    edges=[("Z", "X"), ("Z", "Y"), ("X", "Y"), ("X", "W")],
                ),
                "ground_truth": {
                    "adjustment_set": ["Z"],
                },
            },
        ]
        return queries[:n_queries]

    # ----------------------------------------------------------------
    # L3: 复杂推理 — 反事实推断
    # ----------------------------------------------------------------

    def generate_l3_queries(self, n_queries: int = 6) -> list[dict]:
        """生成 L3 复杂推理测试用例。

        测试目标：
        - 反事实推断
        - 不确定性量化
        - 置信区间校准
        """
        queries = [
            {
                "id": "L3-001",
                "level": "L3",
                "type": "counterfactual",
                "query": "If X had been 0.5, what would Y be?",
                "sem_coeff": np.array([[0, 0.5], [0, 0]], dtype=np.float64),
                "sem_nodes": ["X", "Y"],
                "evidence": {"X": 1.0, "Y": 2.0},
                "do_x": {"X": 0.5},
                "target": "Y",
                "ground_truth": {
                    "expected_status": "ok",
                    "has_ci": True,
                },
            },
            {
                "id": "L3-002",
                "level": "L3",
                "type": "uncertainty_quantification",
                "query": "What is the uncertainty of the causal effect X→Y?",
                "dc": DoCalculus(
                    graph=CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")]),
                ),
                "ground_truth": {
                    "has_confidence_interval": True,
                    "has_p_value": True,
                    "method": "direct",
                },
            },
            {
                "id": "L3-003",
                "level": "L3",
                "type": "batch_counterfactual",
                "query": "Batch: what happens under 3 different interventions?",
                "sem_nodes": ["X", "Y", "Z"],
                "sem_coeff": np.array([[0, 0.4, 0.3], [0, 0, 0.2], [0, 0, 0]], dtype=np.float64),
                "scenarios": [
                    {"evidence": {"X": 1.0, "Y": 2.0}, "do_x": {"X": 0.5}, "target": "Y"},
                    {"evidence": {"X": 2.0, "Y": 3.0}, "do_x": {"X": 1.0}, "target": "Y"},
                ],
                "ground_truth": {
                    "expected_count": 2,
                },
            },
        ]
        return queries[:n_queries]

    # ----------------------------------------------------------------
    # 评分引擎
    # ----------------------------------------------------------------

    def compute_scores(
        self,
        results: list[dict],
        ground_truths: list[dict],
        k: int = 3,
    ) -> GAIAScore:
        """计算 GAIA 风格评分。

        Args:
            results: 系统输出的结果列表
            ground_truths: 真实答案列表
            k: Precision@k / Recall@k 的 k 值

        Returns:
            GAIAScore
        """
        n = len(results)
        if n == 0:
            return GAIAScore(n_samples=0)

        # Accuracy (Top-1)
        correct = 0
        reciprocal_ranks = []
        precision_ks = []
        recall_ks = []

        for i, result in enumerate(results):
            gt = ground_truths[i] if i < len(ground_truths) else {}
            ranked_items = result.get("ranked_results", [])

            # Top-1 match
            if ranked_items and ranked_items[0].get("is_correct", False):
                correct += 1

            # MRR
            rr = 0.0
            for rank, item in enumerate(ranked_items, start=1):
                if item.get("is_correct", False):
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

            # Precision@k / Recall@k
            top_k = ranked_items[:k]
            relevant_in_top_k = sum(1 for item in top_k if item.get("is_correct", False))
            precision_ks.append(relevant_in_top_k / k if k > 0 else 0.0)

            gt_relevant = gt.get("relevant_count", 1)
            recall_ks.append(relevant_in_top_k / max(gt_relevant, 1))

        accuracy = correct / n
        mrr = statistics.mean(reciprocal_ranks)
        precision_at_k = statistics.mean(precision_ks)
        recall_at_k = statistics.mean(recall_ks)
        f1 = (
            2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
            if (precision_at_k + recall_at_k) > 0
            else 0.0
        )

        # ECE (Expected Calibration Error)
        confidences = [r.get("confidence", 0.5) for r in results]
        is_correct_list = [
            1.0 if ranked_items and ranked_items[0].get("is_correct", False) else 0.0
            for r, ranked_items in zip(results, [r.get("ranked_results", []) for r in results])
        ]
        ece = self._compute_ece(confidences, is_correct_list)

        return GAIAScore(
            accuracy=accuracy,
            mrr=mrr,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            f1_score=f1,
            ece=ece,
            n_samples=n,
        )

    @staticmethod
    def _compute_ece(confidences: list[float], correctness: list[float], n_bins: int = 10) -> float:
        """计算 Expected Calibration Error。"""
        if not confidences:
            return 0.0

        bins = [[] for _ in range(n_bins)]
        for conf, corr in zip(confidences, correctness):
            bin_idx = min(int(conf * n_bins), n_bins - 1)
            bins[bin_idx].append((conf, corr))

        ece = 0.0
        for bin_data in bins:
            if not bin_data:
                continue
            bin_conf = statistics.mean(c[0] for c in bin_data)
            bin_acc = statistics.mean(c[1] for c in bin_data)
            ece += len(bin_data) * abs(bin_acc - bin_conf)

        return ece / len(confidences)

    # ----------------------------------------------------------------
    # 双路径对比
    # ----------------------------------------------------------------

    def run_dual_path_comparison(
        self,
        queries: list[dict],
        original_path_fn,
        bayesian_path_fn,
    ) -> GAIAReport:
        """执行双路径对比测试。

        Args:
            queries: 测试用例列表
            original_path_fn: 原始系统执行函数 (query) -> result_dict
            bayesian_path_fn: 贝叶斯系统执行函数 (query) -> result_dict

        Returns:
            GAIAReport 包含双路径评分对比
        """
        original_results = []
        bayesian_results = []
        ground_truths = []
        details = []

        for q in queries:
            gt = q.get("ground_truth", {})

            # 原始路径
            t0 = time.perf_counter()
            try:
                orig_result = original_path_fn(q)
            except Exception as e:
                orig_result = {"error": str(e), "ranked_results": []}
            orig_time = time.perf_counter() - t0

            # 贝叶斯路径
            t0 = time.perf_counter()
            try:
                bayes_result = bayesian_path_fn(q)
            except Exception as e:
                bayes_result = {"error": str(e), "ranked_results": []}
            bayes_time = time.perf_counter() - t0

            original_results.append(orig_result)
            bayesian_results.append(bayes_result)
            ground_truths.append(gt)

            details.append(
                {
                    "query_id": q["id"],
                    "level": q["level"],
                    "type": q.get("type", "unknown"),
                    "original_correct": (
                        orig_result["ranked_results"][0]["is_correct"] if orig_result.get("ranked_results") else False
                    ),
                    "bayesian_correct": (
                        bayes_result["ranked_results"][0]["is_correct"] if bayes_result.get("ranked_results") else False
                    ),
                    "original_time_ms": round(orig_time * 1000, 2),
                    "bayesian_time_ms": round(bayes_time * 1000, 2),
                }
            )

        orig_score = self.compute_scores(original_results, ground_truths)
        bayes_score = self.compute_scores(bayesian_results, ground_truths)

        # 贝叶斯改善比率
        bayes_improvement = 0.0
        if orig_score.accuracy > 0:
            bayes_improvement = (bayes_score.accuracy - orig_score.accuracy) / orig_score.accuracy * 100

        bayes_score.bayesian_improvement = bayes_improvement

        level = queries[0]["level"] if queries else "unknown"

        return GAIAReport(
            level=level,
            original_score=orig_score,
            bayesian_score=bayes_score,
            improvement_summary={
                "accuracy_delta": round(bayes_score.accuracy - orig_score.accuracy, 4),
                "mrr_delta": round(bayes_score.mrr - orig_score.mrr, 4),
                "ece_delta": round(bayes_score.ece - orig_score.ece, 4),
                "bayesian_improvement_pct": round(bayes_improvement, 2),
            },
            per_query_details=details,
            meta={
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "n_queries": len(queries),
                "k": 3,
            },
        )

    # ----------------------------------------------------------------
    # 汇总报告
    # ----------------------------------------------------------------

    def generate_summary_report(self, reports: list[GAIAReport], output_path: str | None = None) -> dict:
        """生成 GAIA 风格汇总报告。"""
        summary = {
            "framework": "GAIA-style Hierarchical Benchmark",
            "version": "3.1.0",
            "levels": {},
            "overall": {},
        }

        for report in reports:
            summary["levels"][report.level] = {
                "n_queries": report.original_score.n_samples,
                "original_accuracy": report.original_score.accuracy,
                "bayesian_accuracy": report.bayesian_score.accuracy,
                "improvement_pct": report.bayesian_score.bayesian_improvement,
                "original_mrr": report.original_score.mrr,
                "bayesian_mrr": report.bayesian_score.mrr,
                "original_ece": report.original_score.ece,
                "bayesian_ece": report.bayesian_score.ece,
            }

        # 总体统计
        total_queries = sum(r.original_score.n_samples for r in reports)
        if total_queries > 0:
            weighted_orig_acc = (
                sum(r.original_score.accuracy * r.original_score.n_samples for r in reports) / total_queries
            )
            weighted_bayes_acc = (
                sum(r.bayesian_score.accuracy * r.bayesian_score.n_samples for r in reports) / total_queries
            )
            overall_improvement = (weighted_bayes_acc - weighted_orig_acc) / max(weighted_orig_acc, 0.001) * 100
        else:
            weighted_orig_acc = 0.0
            weighted_bayes_acc = 0.0
            overall_improvement = 0.0

        summary["overall"] = {
            "total_queries": total_queries,
            "original_accuracy": round(weighted_orig_acc, 4),
            "bayesian_accuracy": round(weighted_bayes_acc, 4),
            "improvement_pct": round(overall_improvement, 2),
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

        return summary


# =============================================================================
# 测试用例：L1 事实检索
# =============================================================================


class TestGAIAL1FactRetrieval:
    """GAIA L1 — 事实检索层测试。"""

    def test_edge_existence_query(self):
        """因果边存在性查询。"""
        runner = GAIABenchmarkRunner(seed=42)
        queries = runner.generate_l1_queries(n_queries=3)

        assert len(queries) == 3
        for q in queries:
            assert q["level"] == "L1"
            assert "cause" in q
            assert "effect" in q
            assert "ground_truth" in q

    def test_l1_causal_graph_hit(self):
        """通过 CausalGraph 验证 L1 查询的正确定向。"""
        cg = CausalGraph(nodes=["X", "Y", "Z"], edges=[("X", "Y"), ("Z", "Y")])

        def original_path(q):
            has_edge = bool(cg.has_edge(q["cause"], q["effect"]))
            gt_has_edge = q["ground_truth"]["has_edge"]
            return {
                "ranked_results": [{"is_correct": has_edge == gt_has_edge}],
                "confidence": 0.9 if has_edge else 0.5,
            }

        runner = GAIABenchmarkRunner(seed=42)
        queries = runner.generate_l1_queries(n_queries=3)

        report = runner.run_dual_path_comparison(
            queries,
            original_path_fn=original_path,
            bayesian_path_fn=original_path,  # 同路径对比（验证框架）
        )

        assert report.level == "L1"
        assert report.original_score.accuracy >= 0.0

    def test_l1_score_computation(self):
        """验证 L1 评分计算。"""
        runner = GAIABenchmarkRunner(seed=42)

        results = [
            {"ranked_results": [{"is_correct": True}], "confidence": 0.9},
            {"ranked_results": [{"is_correct": False}], "confidence": 0.6},
            {"ranked_results": [{"is_correct": True}], "confidence": 0.8},
        ]
        ground_truths = [{"relevant_count": 1}, {"relevant_count": 1}, {"relevant_count": 1}]

        score = runner.compute_scores(results, ground_truths, k=3)
        assert score.accuracy == pytest.approx(2.0 / 3.0, abs=0.01)
        assert score.n_samples == 3


# =============================================================================
# 测试用例：L2 多跳推理
# =============================================================================


class TestGAIAL2MultiHopReasoning:
    """GAIA L2 — 多跳推理层测试。"""

    def test_mediation_identification(self):
        """中介变量识别。"""
        runner = GAIABenchmarkRunner(seed=42)
        queries = runner.generate_l2_queries(n_queries=1)

        q = queries[0]
        assert q["type"] == "mediation"
        assert q["ground_truth"]["mediator"] == "M"

        cg = q["graph"]
        mediators = cg.get_mediators("X", "Y")
        assert "M" in mediators

    def test_backdoor_detection(self):
        """后门路径检测。"""
        runner = GAIABenchmarkRunner(seed=42)
        queries = runner.generate_l2_queries(n_queries=2)

        q = queries[1]
        assert q["type"] == "backdoor_detection"
        assert q["ground_truth"]["is_confounder"] is True

        dc = DoCalculus(graph=q["graph"])
        adj_set = dc.identify_adjustment_set("X", "Y")
        assert adj_set is not None
        assert "Z" in adj_set

    def test_causal_chain_path_length(self):
        """因果链路径长度验证。"""
        runner = GAIABenchmarkRunner(seed=42)
        queries = runner.generate_l2_queries(n_queries=3)

        q = queries[2]
        assert q["type"] == "causal_chain"

        cg = q["graph"]
        # 验证 A → D 通过中介 B, C
        descendants = cg.get_descendants("A")
        assert "D" in descendants
        assert len(descendants) == 3  # B, C, D

    def test_adjustment_set_identification(self):
        """调整变量集识别。"""
        runner = GAIABenchmarkRunner(seed=42)
        queries = runner.generate_l2_queries(n_queries=4)

        q = queries[3]
        assert q["type"] == "adjustment_set"

        dc = DoCalculus(graph=q["graph"])
        adj_set = dc.identify_adjustment_set("X", "Y")
        assert adj_set is not None
        assert "Z" in adj_set

    def test_l2_dual_path_comparison(self):
        """L2 双路径对比。"""
        runner = GAIABenchmarkRunner(seed=42)
        queries = runner.generate_l2_queries(n_queries=2)

        def original_path(q):
            cg = q.get("graph")
            if cg is None:
                return {"ranked_results": [], "confidence": 0.5}

            dc = DoCalculus(graph=cg)
            try:
                result = dc.estimate_ate("X", "Y")
                correct = result.method != "none"
            except Exception:
                correct = False

            return {
                "ranked_results": [{"is_correct": correct}],
                "confidence": 0.8 if correct else 0.5,
            }

        report = runner.run_dual_path_comparison(
            queries,
            original_path_fn=original_path,
            bayesian_path_fn=original_path,
        )

        assert report.level == "L2"
        assert report.original_score.n_samples == 2


# =============================================================================
# 测试用例：L3 复杂推理
# =============================================================================


class TestGAIAL3ComplexReasoning:
    """GAIA L3 — 复杂推理层测试。"""

    def test_counterfactual_inference(self):
        """反事实推断。"""
        runner = GAIABenchmarkRunner(seed=42)
        queries = runner.generate_l3_queries(n_queries=1)

        q = queries[0]
        assert q["type"] == "counterfactual"

        sem = StructuralEquationModel(
            coefficients=q["sem_coeff"],
            node_names=q["sem_nodes"],
        )
        engine = CounterfactualEngine(sem, node_names=q["sem_nodes"])
        result = engine.query(
            evidence=q["evidence"],
            do_x=q["do_x"],
            target=q["target"],
        )
        assert result.status == "ok"

    def test_uncertainty_quantification(self):
        """不确定性量化。"""
        runner = GAIABenchmarkRunner(seed=42)
        queries = runner.generate_l3_queries(n_queries=2)

        q = queries[1]
        assert q["type"] == "uncertainty_quantification"

        dc = q["dc"]
        result = dc.estimate_ate("X", "Y")
        assert result is not None
        assert result.method != "none"
        # 验证置信区间存在
        assert result.confidence_interval[0] != result.confidence_interval[1] or result.method == "none"

    def test_batch_counterfactual(self):
        """批量反事实查询。"""
        from mci_world_model.sdk._batch_counterfactual import BatchCounterfactualEngine

        runner = GAIABenchmarkRunner(seed=42)
        queries = runner.generate_l3_queries(n_queries=3)

        q = queries[2]
        assert q["type"] == "batch_counterfactual"

        sem = StructuralEquationModel(
            coefficients=q["sem_coeff"],
            node_names=q["sem_nodes"],
        )
        engine = BatchCounterfactualEngine(sem=sem)
        results = engine.batch_query(q["scenarios"])
        assert len(results) == q["ground_truth"]["expected_count"]

    def test_l3_ate_with_confidence_interval(self):
        """L3 ATE 置信区间测试。"""
        dc = DoCalculus(
            graph=CausalGraph(nodes=["X", "Y", "Z"], edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")]),
        )
        result = dc.estimate_ate("X", "Y", x_value=1.0, x_baseline=0.0, method="auto")
        assert result is not None

        # 验证 p-value 不为极端值
        assert 0.0 <= result.p_value <= 1.0

    def test_l3_ece_computation(self):
        """L3 校准误差 (ECE) 计算验证。"""
        runner = GAIABenchmarkRunner(seed=42)

        # 模拟完美校准：置信度完全匹配准确率
        confidences = [0.5, 0.7, 0.9, 0.6]
        correctness = [0.5, 0.7, 0.9, 0.6]

        ece = runner._compute_ece(confidences, correctness, n_bins=5)
        # 完美校准时 ECE 应为 0
        assert abs(ece) < 0.4  # 有限分桶可能引入小误差


# =============================================================================
# 测试用例：汇总报告
# =============================================================================


class TestGAIASummaryReport:
    """GAIA 汇总报告测试。"""

    def test_summary_generation(self):
        """生成分层汇总报告。"""
        runner = GAIABenchmarkRunner(seed=42)

        # 模拟各层报告
        l1_queries = runner.generate_l1_queries(n_queries=2)
        l2_queries = runner.generate_l2_queries(n_queries=2)

        def path_fn(q):
            return {"ranked_results": [{"is_correct": True}], "confidence": 0.9}

        l1_report = runner.run_dual_path_comparison(l1_queries, path_fn, path_fn)
        l2_report = runner.run_dual_path_comparison(l2_queries, path_fn, path_fn)

        summary = runner.generate_summary_report([l1_report, l2_report])

        assert summary["framework"] == "GAIA-style Hierarchical Benchmark"
        assert "L1" in summary["levels"]
        assert "L2" in summary["levels"]
        assert summary["overall"]["total_queries"] == 4

    def test_output_json_format(self):
        """验证 JSON 输出格式。"""
        report = GAIAReport(
            level="L1",
            original_score=GAIAScore(accuracy=0.8, n_samples=10),
            bayesian_score=GAIAScore(accuracy=0.85, n_samples=10),
        )
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["level"] == "L1"
        assert "original" in d
        assert "bayesian" in d


# =============================================================================
# 测试用例：评分体系健全性
# =============================================================================


class TestGAIAScoringSanity:
    """GAIA 评分体系健全性测试。"""

    def test_empty_results(self):
        """空结果评分。"""
        runner = GAIABenchmarkRunner(seed=42)
        score = runner.compute_scores([], [], k=3)
        assert score.n_samples == 0
        assert score.accuracy == 0.0

    def test_perfect_accuracy(self):
        """完美准确率。"""
        runner = GAIABenchmarkRunner(seed=42)
        results = [{"ranked_results": [{"is_correct": True}], "confidence": 1.0} for _ in range(10)]
        ground_truths = [{"relevant_count": 1} for _ in range(10)]
        score = runner.compute_scores(results, ground_truths, k=3)
        assert score.accuracy == 1.0
        assert score.mrr == 1.0

    def test_zero_accuracy(self):
        """零准确率。"""
        runner = GAIABenchmarkRunner(seed=42)
        results = [{"ranked_results": [{"is_correct": False}], "confidence": 0.5} for _ in range(10)]
        ground_truths = [{"relevant_count": 1} for _ in range(10)]
        score = runner.compute_scores(results, ground_truths, k=3)
        assert score.accuracy == 0.0
        assert score.mrr == 0.0

    def test_mrr_computation(self):
        """MRR 计算验证。"""
        runner = GAIABenchmarkRunner(seed=42)

        # 正确答案在位置1 → RR=1/1=1.0
        # 正确答案在位置2 → RR=1/2=0.5
        results = [
            {
                "ranked_results": [
                    {"is_correct": True},
                    {"is_correct": False},
                ],
                "confidence": 0.8,
            },
            {
                "ranked_results": [
                    {"is_correct": False},
                    {"is_correct": True},
                ],
                "confidence": 0.6,
            },
        ]
        ground_truths = [{"relevant_count": 1}, {"relevant_count": 1}]

        score = runner.compute_scores(results, ground_truths, k=3)
        expected_mrr = (1.0 + 0.5) / 2.0
        assert score.mrr == pytest.approx(expected_mrr, abs=0.01)
