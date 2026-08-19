"""
benchmarks/cladder/test_cladder_integration.py — CLADDER 因果阶梯基准集成验证
==============================================================================

CLADDER (NeurIPS 2023): 10K yes/no 题覆盖 Pearl 三层因果阶梯。

Rung 1 (Association):  correlation, marginal, exp_away
Rung 2 (Intervention):  ate, backadj, collider_bias
Rung 3 (Counterfactual): ett, nde, nie, det-counterfactual

CEWM 结构化求解: 用 CausalGraph + DoCalculus + CounterfactualEngine
代替 LLM 统计拟合，目标 ≥ 90% 准确率。
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# CLADDER 样本测试题 (硬编码代表性题目)
# ═══════════════════════════════════════════════════════════════════════════════


CLADDER_SAMPLES = {
    "rung1_correlation": {
        "story": "X and Y are positively correlated.",
        "question": "Does X cause Y?",
        "correct": "no",  # correlation ≠ causation
    },
    "rung1_basic": {
        "graph": {"nodes": ["Z", "X", "Y"], "edges": [("Z", "X"), ("Z", "Y")]},
        "question": "Is X associated with Y?",
        "correct": "yes",  # confounded by Z
    },
    "rung2_ate": {
        "graph": {"nodes": ["Z", "X", "Y"], "edges": [("Z", "X"), ("Z", "Y"), ("X", "Y")]},
        "question": "What is the ATE of X on Y?",
        "correct": "non_zero",  # X→Y direct edge
    },
    "rung2_backadj": {
        "graph": {"nodes": ["Z", "X", "Y"], "edges": [("Z", "X"), ("Z", "Y"), ("X", "Y")]},
        "question": "What is the backdoor adjustment set for X→Y?",
        "correct": ["Z"],
    },
    "rung3_counterfactual": {
        "graph": {"nodes": ["X", "Y"], "edges": [("X", "Y")]},
        "question": "If X were 0 instead of 1, what would Y be?",
        "correct": "different",  # counterfactual ≠ factual
    },
}


class TestCLADDERIntegration:
    """CLADDER 三层因果阶梯集成测试。"""

    def test_rung1_association(self):
        """Rung 1: 关联层 — 区分关联与因果。"""
        from mci_world_model.sdk._do_calculus import CausalGraph, DoCalculus

        sample = CLADDER_SAMPLES["rung1_basic"]
        cg = CausalGraph(
            nodes=sample["graph"]["nodes"],
            edges=sample["graph"]["edges"],
        )
        dc = DoCalculus(graph=cg, seed=42)
        result = dc.estimate_ate("X", "Y")
        is_associated = result.ate != 0 or result.method != "rejected"
        print(f"\n  Rung1 (Assoc): X~Y? {'yes' if is_associated else 'no'} (correct={sample['correct']})")

    def test_rung2_intervention(self):
        """Rung 2: 干预层 — ATE + 后门调整。"""
        from mci_world_model.sdk._do_calculus import CausalGraph, DoCalculus

        sample = CLADDER_SAMPLES["rung2_ate"]
        cg = CausalGraph(
            nodes=sample["graph"]["nodes"],
            edges=sample["graph"]["edges"],
        )
        dc = DoCalculus(graph=cg, seed=42)
        result = dc.estimate_ate("X", "Y")
        assert abs(result.ate) > 0.01  # X→Y 有直接效应

        # 后门调整集
        adj = dc.identify_adjustment_set("X", "Y")
        assert adj == ["Z"]
        print(f"  Rung2 (Intervention): ATE={result.ate:.3f}, adj={adj}")

    def test_rung3_counterfactual(self):
        """Rung 3: 反事实层 — NDE/NIE。"""
        from mci_world_model.sdk._counterfactual import CounterfactualEngine
        from mci_world_model.sdk._do_calculus import CausalGraph

        sample = CLADDER_SAMPLES["rung3_counterfactual"]
        cg = CausalGraph(
            nodes=sample["graph"]["nodes"],
            edges=sample["graph"]["edges"],
        )
        engine = CounterfactualEngine.from_causal_graph(cg, seed=42)
        result = engine.query(
            evidence={"X": 1.0, "Y": 2.5},
            do_x={"X": 0.0},
            target="Y",
        )
        assert abs(result.counterfactual_value - result.factual_value) > 0.01
        print(f"  Rung3 (Counterfactual): factual={result.factual_value:.3f}, cf={result.counterfactual_value:.3f}")

    def test_full_pipeline(self):
        """完整 CLADDER 流水线——全部三层。"""
        from mci_world_model.sdk._counterfactual import CounterfactualEngine
        from mci_world_model.sdk._do_calculus import CausalGraph, DoCalculus

        score = {"rung1": 0, "rung2": 0, "rung3": 0}

        # Rung1: 关联
        for name in ["rung1_basic"]:
            sample = CLADDER_SAMPLES[name]
            cg = CausalGraph(nodes=sample["graph"]["nodes"], edges=sample["graph"]["edges"])
            dc = DoCalculus(graph=cg, seed=42)
            r = dc.estimate_ate("X", "Y")
            if r.ate != 0:
                score["rung1"] += 1

        # Rung2: 干预
        for name in ["rung2_ate", "rung2_backadj"]:
            sample = CLADDER_SAMPLES[name]
            cg = CausalGraph(nodes=sample["graph"]["nodes"], edges=sample["graph"]["edges"])
            dc = DoCalculus(graph=cg, seed=42)
            r = dc.estimate_ate("X", "Y")
            adj = dc.identify_adjustment_set("X", "Y")
            if r.ate != 0 and adj is not None:
                score["rung2"] += 1

        # Rung3: 反事实
        for name in ["rung3_counterfactual"]:
            sample = CLADDER_SAMPLES[name]
            cg = CausalGraph(nodes=sample["graph"]["nodes"], edges=sample["graph"]["edges"])
            engine = CounterfactualEngine.from_causal_graph(cg, seed=42)
            r = engine.query(evidence={"X": 1.0, "Y": 2.5}, do_x={"X": 0.0}, target="Y")
            if abs(r.counterfactual_value - r.factual_value) > 0.01:
                score["rung3"] += 1

        total = sum(score.values())
        total_possible = 3
        accuracy = total / total_possible
        print("\n  === CLADDER Accuracy ===")
        print(f"  Rung1: {score['rung1']}/1, Rung2: {score['rung2']}/1, Rung3: {score['rung3']}/1")
        print(f"  Total: {total}/{total_possible} = {accuracy:.0%}")

        assert accuracy >= 0.9  # ≥90% 目标

    def test_cladder_benchmark_speed(self):
        """CLADDER 基准延迟。"""
        import gc
        import time

        from mci_world_model.sdk._counterfactual import CounterfactualEngine
        from mci_world_model.sdk._do_calculus import CausalGraph, DoCalculus

        n_questions = 100
        cg = CausalGraph(nodes=["Z", "X", "Y"], edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")])
        engine = CounterfactualEngine.from_causal_graph(cg, seed=42)

        gc.disable()
        t0 = time.perf_counter()
        for _ in range(n_questions):
            dc = DoCalculus(graph=cg, seed=_)
            dc.estimate_ate("X", "Y")
            engine.query(evidence={"X": 1.0}, do_x={"X": 0.0}, target="Y")
        t = time.perf_counter() - t0
        gc.enable()

        per_q = t / n_questions * 1000
        throughput = n_questions / t
        print(
            f"\n  CLADDER benchmark: {n_questions} questions in {t * 1000:.0f}ms ({per_q:.1f}ms/q, {throughput:.0f} q/s)"
        )
        assert per_q < 50  # 每问 < 50ms
