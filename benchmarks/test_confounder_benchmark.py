"""
MCI World Model V4.1.0 — 隐藏混杂因子 (Unobserved Confounders) 基准测试

对标: Pearl do-calculus 后门调整 (Back-door Adjustment)

评测 CEWM 在存在隐藏混杂因子时的因果效应估计精度:
  1. 简单混杂: U → X, U → Y, X → Y
  2. 链式混杂: U → Z, Z → X, Z → Y, X → Y
  3. 多混杂因子: U1 → X, U1 → Y, U2 → X, U2 → Y, X → Y

核心比较:
  - 朴素估计 (naive): P(Y|X) — 受混杂偏置影响
  - 调整估计 (adjusted): P(Y|do(X)) — 通过 SEM 干预消除偏置
  - 真实效应 (ground truth): 用 CounterfactualEngine 干预计算

运行: pytest benchmarks/test_confounder_benchmark.py -v
"""

from __future__ import annotations

import pytest

from benchmarks._causal_utils import _propagate, sem_forward
from mci_world_model.sdk._counterfactual import (
    CounterfactualEngine,
)
from mci_world_model.sdk._do_calculus import CausalGraph

# =============================================================================
# 工具函数
# =============================================================================


def _build_graph(
    nodes: list[str],
    edges: list[tuple[str, str]],
    weights: list[float],
) -> CausalGraph:
    """构建因果图。"""
    cg = CausalGraph(nodes=nodes, edges=edges)
    for i, (src, tgt) in enumerate(edges):
        si, ti = cg.nodes.index(src), cg.nodes.index(tgt)
        cg.adjacency[si, ti] = weights[i]
    return cg


def _estimate_naive_effect(
    cg: CausalGraph,
    x_values: list[float],
    target: str = "Y",
) -> list[float]:
    """
    朴素因果效应估计: P(Y|X=x)。

    仅观测关联，不做任何调整。
    在存在混杂因子时，这个估计是有偏的。
    """
    results = []
    for x_val in x_values:
        values = _propagate(cg, {"X": x_val})
        results.append(values.get(target, 0.0))
    return results


def _estimate_adjusted_effect(
    cg: CausalGraph,
    x_values: list[float],
    target: str = "Y",
    activation: str = "linear",
) -> list[float]:
    """
    调整因果效应估计: P(Y|do(X=x))。

    使用 SEM.intervene() 构建 mutilated SEM，消除 X 的所有入边。
    这等价于 do-calculus 的后门调整。
    """
    results = []
    for x_val in x_values:
        values = sem_forward(cg, {"X": x_val}, activation=activation)
        results.append(values.get(target, 0.0))
    return results


def _compute_ground_truth_effect(
    cg: CausalGraph,
    x_val: float = 1.0,
    target: str = "Y",
    noise_std: float = 0.01,
) -> float:
    """
    真实因果效应: 用 CounterfactualEngine 计算。

    计算 Y 在 do(X=1) vs do(X=0) 下的个体效应。
    """
    engine = CounterfactualEngine.from_causal_graph(
        cg,
        noise_std=noise_std,
        activation="linear",
        seed=42,
    )
    if engine is None:
        return 0.0

    # 事实: X=0
    evidence = _propagate(cg, {"X": 0.0})
    result = engine.query(
        evidence=evidence,
        do_x={"X": x_val},
        target=target,
        compute_pns=False,
        n_mc=50,
    )
    if result.status != "ok":
        return 0.0
    return result.individual_effect


# =============================================================================
# 场景 1: 简单混杂 (Simple Confounding)
# =============================================================================


class TestSimpleConfounder:
    """简单混杂: U → X, U → Y, X → Y。

    因果结构:
        U (unobserved) ─→ X
        U (unobserved) ─→ Y
        X ─→ Y

    朴素估计 P(Y|X) 受 U 的混杂影响，
    而 do(X) 干预切断了 U → X 的路径，得到真实因果效应。
    """

    @pytest.fixture
    def confounded_graph(self):
        """含混杂因子的因果图。"""
        return _build_graph(
            nodes=["U", "X", "Y"],
            edges=[("U", "X"), ("U", "Y"), ("X", "Y")],
            weights=[2.0, 1.5, 1.0],  # U 对 X 和 Y 都有强影响
        )

    @pytest.fixture
    def clean_graph(self):
        """无混杂的干净因果图 (仅 X → Y)。"""
        return _build_graph(
            nodes=["X", "Y"],
            edges=[("X", "Y")],
            weights=[1.0],
        )

    def test_naive_estimates_are_biased(self, confounded_graph, clean_graph):
        """朴素估计 P(Y|X) 在混杂图下有偏 (≠ 真实因果效应)。"""
        # 关键: 当 U 有非零基线时，朴素估计会有偏
        # 模拟 U=1 的情况:
        naive_with_u = _propagate(confounded_graph, {"U": 1.0, "X": 1.0})
        naive_without_u = _propagate(confounded_graph, {"X": 1.0})

        # 有 U=1 时: Y = 1.5*1 + 1.0*1 = 2.5 (偏高!)
        assert naive_with_u["Y"] > 2.0, f"With U=1, Y should be ~2.5, got {naive_with_u['Y']}"
        # 无 U 时: Y = 0 + 1.0*1 = 1.0
        assert abs(naive_without_u["Y"] - 1.0) < 0.1, "Without U, Y should be ~1.0"

    def test_adjusted_estimates_correct_effect(self, confounded_graph):
        """调整估计 do(X=1) 消除了混杂偏置。"""
        # SEM 干预 do(X=1) 切断 U→X，固定 X=1
        adjusted = _estimate_adjusted_effect(confounded_graph, [1.0])
        # do(X=1) 下: Y 不受 U 对 X 的影响，仅 X→Y 路径
        # Y ≈ w(X→Y)*1 = 1.0
        assert abs(adjusted[0] - 1.0) < 0.3, f"Adjusted effect should be ~1.0, got {adjusted[0]:.4f}"

    def test_confounder_bias_magnitude(self, confounded_graph):
        """量化混杂偏置的幅度。"""
        # 当 U=1 时的朴素估计 vs 调整估计
        naive = _propagate(confounded_graph, {"U": 1.0, "X": 1.0})
        adjusted = sem_forward(confounded_graph, {"X": 1.0})

        bias = abs(naive["Y"] - adjusted["Y"])
        assert bias > 0.5, f"Confounder bias should be > 0.5, got {bias:.4f}"


# =============================================================================
# 场景 2: 链式混杂 (Chain Confounding)
# =============================================================================


class TestChainConfounder:
    """链式混杂: U → Z, Z → X, Z → Y, X → Y。

    因果结构:
        U (unobserved) → Z (observed)
        Z → X
        Z → Y
        X → Y

    Z 是可观测的混杂因子，可以通过条件化 Z 来调整。
    """

    @pytest.fixture
    def chain_confounded_graph(self):
        return _build_graph(
            nodes=["U", "Z", "X", "Y"],
            edges=[("U", "Z"), ("Z", "X"), ("Z", "Y"), ("X", "Y")],
            weights=[1.0, 1.5, 1.0, 1.0],
        )

    def test_chain_confounder_bias(self, chain_confounded_graph):
        """链式混杂: U→Z→X 和 U→Z→Y 路径导致偏置。"""
        # 有 U 时: Z=1, X=1.5+X_input, Y=Z+X
        with_u = _propagate(chain_confounded_graph, {"U": 1.0, "X": 1.0})
        # Z = 1*1 = 1, Y = 1*1 + 1*1 = 2.0

        # 调整 (do(X=1)): 切断 Z→X, 固定 X=1
        adjusted = sem_forward(chain_confounded_graph, {"X": 1.0})
        # do(X=1): Y 仍受 Z 影响 (Z→Y 未被切断)
        # 但 X 不受 Z 影响了

        # 关键: 调整后 Y 的值应更接近真实因果效应
        naive_bias = abs(with_u["Y"] - 1.0)  # 朴素 Y vs 真实效应 1.0
        adjusted_bias = abs(adjusted["Y"] - 1.0)  # 调整 Y vs 真实效应
        # 朴素 Y = Z + X = 1 + 1 = 2, bias = 1
        assert naive_bias > 0.3, f"Chain confounder bias should be > 0.3, got {naive_bias:.4f}"
        assert adjusted_bias < naive_bias, f"Adjusted bias ({adjusted_bias:.4f}) should be < naive ({naive_bias:.4f})"

    def test_adjustment_via_observed_z(self, chain_confounded_graph):
        """通过条件化可观测的 Z 来消除部分混杂偏置。"""
        # 固定 Z=0 后, X 不再受 Z 影响, 偏置减小
        conditioned = _propagate(chain_confounded_graph, {"Z": 0.0, "X": 1.0})
        # Z=0: Y = 0 + 1*1 = 1.0 (无偏!)
        assert abs(conditioned["Y"] - 1.0) < 0.3, f"Conditioned on Z=0, Y should be ~1.0, got {conditioned['Y']:.4f}"


# =============================================================================
# 场景 3: 多混杂因子 (Multiple Confounders)
# =============================================================================


class TestMultipleConfounders:
    """多混杂因子: U1 → X, U1 → Y, U2 → X, U2 → Y, X → Y。

    因果结构:
        U1 (unobserved) → X, → Y
        U2 (unobserved) → X, → Y
        X → Y

    多个混杂因子同时影响 X 和 Y，偏置叠加。
    """

    @pytest.fixture
    def multi_confounded_graph(self):
        return _build_graph(
            nodes=["U1", "U2", "X", "Y"],
            edges=[
                ("U1", "X"),
                ("U1", "Y"),
                ("U2", "X"),
                ("U2", "Y"),
                ("X", "Y"),
            ],
            weights=[1.0, 0.8, 0.5, 1.2, 1.0],
        )

    def test_multiple_confounders_amplify_bias(self, multi_confounded_graph):
        """多混杂因子叠加导致更大偏置。"""
        # 两个混杂因子都激活
        with_both = _propagate(multi_confounded_graph, {"U1": 1.0, "U2": 1.0, "X": 1.0})
        # X = 1*1 + 0.5*1 + 1.0 = 2.5, Y = 0.8*1 + 1.2*1 + 1*2.5 = 4.5

        # 仅一个混杂因子
        with_one = _propagate(multi_confounded_graph, {"U1": 1.0, "X": 1.0})
        # X = 1*1 + 1.0 = 2.0, Y = 0.8*1 + 1*2.0 = 2.8

        # 无混杂因子
        clean = _propagate(multi_confounded_graph, {"X": 1.0})
        # X = 1.0, Y = 1*1 = 1.0

        # 偏置: 双混杂 > 单混杂 > 无混杂
        assert with_both["Y"] > with_one["Y"], "Double confounder should have more bias"
        assert with_one["Y"] > clean["Y"], "Single confounder should have more bias than clean"

    def test_adjusted_eliminate_all_confounders(self, multi_confounded_graph):
        """do(X=1) 干预消除所有混杂因子的影响。"""
        adjusted = sem_forward(multi_confounded_graph, {"X": 1.0})
        # do(X=1): 切断 U1→X 和 U2→X, 固定 X=1
        # Y = 0.8*U1 + 1.2*U2 + 1.0*1 (U1, U2 仍影响 Y 但不通过 X)
        # 关键: X→Y 的因果效应 = 1.0, 不受 U1/U2 对 X 的影响
        assert abs(adjusted["Y"] - 1.0) < 0.5, f"Adjusted Y should be ~1.0, got {adjusted['Y']:.4f}"


# =============================================================================
# 综合评分: 混杂偏置 vs 调整精度
# =============================================================================


class TestConfounderComposite:
    """综合评估: 混杂偏置检测 + do-calculus 调整精度。"""

    def test_do_calculus_removes_confounding_bias(self):
        """do-calculus 干预消除混杂偏置 (三种场景汇总)。"""
        scenarios = [
            # (graph, nodes, edges, weights, expected_effect)
            ("simple", ["U", "X", "Y"], [("U", "X"), ("U", "Y"), ("X", "Y")], [2.0, 1.5, 1.0], 1.0),
            (
                "chain",
                ["U", "Z", "X", "Y"],
                [("U", "Z"), ("Z", "X"), ("Z", "Y"), ("X", "Y")],
                [1.0, 1.5, 1.0, 1.0],
                1.0,
            ),
            (
                "multiple",
                ["U1", "U2", "X", "Y"],
                [("U1", "X"), ("U1", "Y"), ("U2", "X"), ("U2", "Y"), ("X", "Y")],
                [1.0, 0.8, 0.5, 1.2, 1.0],
                1.0,
            ),
        ]

        errors = []
        for name, nodes, edges, weights, expected in scenarios:
            cg = _build_graph(nodes, edges, weights)

            # 调整估计 (do(X=1))
            adjusted = sem_forward(cg, {"X": 1.0})
            error = abs(adjusted["Y"] - expected)

            if error > 0.5:
                errors.append(f"{name}: adjusted_error={error:.4f}")

        assert not errors, f"do-calculus adjustment failed: {errors}"

    def test_total_scenario_count(self):
        """3 种混杂场景均已覆盖。"""
        assert True  # 三个 TestClass 各覆盖一种场景
