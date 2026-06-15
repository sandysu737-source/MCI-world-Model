"""
MCI World Model V4.3.0 — Peirce 链式推理基准测试

对标: CEWM Peirce 符号三元组 — 解释项→新符号→再推理

评测链式推理能力:
  1. 推理结果 R1 作为新一轮推理的输入符号
  2. 链式推理深度 ≥ 3 层 (R1 → R2 → R3)
  3. 每层推理附带置信度衰减: conf(R_{k+1}) = conf(R_k) × decay
  4. 终止条件: 置信度 < 阈值 或 达到最大深度

理论对标:
  - Peirce 符号三元组: 符号→对象→解释项 → 新符号
  - 元认知: 推理结果的递归应用
  - 置信度衰减: 链式推理的不确定性累积

运行: pytest benchmarks/test_chain_reasoning_benchmark.py -v
"""

from __future__ import annotations

import numpy as np

from benchmarks._causal_utils import (
    chain_reason,
)
from mci_world_model.sdk._do_calculus import CausalGraph

# =============================================================================
# 辅助函数
# =============================================================================


def _build_chain_graph(
    names: list[str],
    weights: list[float] | None = None,
) -> CausalGraph:
    """构建链式因果图: names[0] → names[1] → ... → names[-1]。"""
    edges = [(names[i], names[i + 1]) for i in range(len(names) - 1)]
    if weights is None:
        weights = [1.0] * len(edges)
    cg = CausalGraph(nodes=names, edges=edges)
    for i, (src, tgt) in enumerate(edges):
        si, ti = cg.nodes.index(src), cg.nodes.index(tgt)
        cg.adjacency[si, ti] = weights[i]
    return cg


def _make_graphs(n_layers: int, node_names: list[str], weight: float = 1.0):
    """创建 n_layers 个相同的链式因果图。"""
    cg = _build_chain_graph(node_names, [weight] * (len(node_names) - 1))
    return [cg for _ in range(n_layers)]


# =============================================================================
# 基本链式推理 (≥ 3 层)
# =============================================================================


class TestBasicChainReasoning:
    """验证链式推理基本功能: 深度 ≥ 3 层。"""

    def test_chain_depth_3(self):
        """3 层链式推理成功执行。"""
        graphs = _make_graphs(3, ["X", "A", "Y"], weight=0.5)
        result = chain_reason(
            causal_graphs=graphs,
            initial_interventions={"X": 2.0},
            target="Y",
            decay=0.9,
            min_confidence=0.1,
            max_depth=3,
        )

        assert result["depth"] == 3, f"Expected depth 3, got {result['depth']}"
        assert len(result["chain"]) == 3

    def test_chain_depth_5(self):
        """5 层链式推理成功执行。"""
        graphs = _make_graphs(5, ["X", "A", "Y"], weight=0.5)
        result = chain_reason(
            causal_graphs=graphs,
            initial_interventions={"X": 2.0},
            target="Y",
            decay=0.95,
            min_confidence=0.1,
            max_depth=5,
        )

        assert result["depth"] == 5, f"Expected depth 5, got {result['depth']}"

    def test_chain_produces_values(self):
        """每层推理产生有效数值。"""
        graphs = _make_graphs(4, ["X", "A", "Y"], weight=0.8)
        result = chain_reason(
            causal_graphs=graphs,
            initial_interventions={"X": 1.0},
            target="Y",
            decay=0.9,
            min_confidence=0.01,
            max_depth=4,
        )

        for depth, value, conf in result["chain"]:
            assert np.isfinite(value), f"Layer {depth}: value should be finite"
            assert 0 < conf <= 1.0, f"Layer {depth}: confidence should be in (0, 1]"


# =============================================================================
# 置信度衰减
# =============================================================================


class TestConfidenceDecay:
    """验证置信度随链式深度衰减。"""

    def test_confidence_decreases(self):
        """置信度逐层递减。"""
        graphs = _make_graphs(5, ["X", "A", "Y"], weight=0.5)
        result = chain_reason(
            causal_graphs=graphs,
            initial_interventions={"X": 1.0},
            target="Y",
            decay=0.8,
            min_confidence=0.01,
            max_depth=5,
        )

        confidences = [conf for _, _, conf in result["chain"]]
        for i in range(1, len(confidences)):
            assert confidences[i] < confidences[i - 1], (
                f"Confidence should decrease: layer {i}={confidences[i]:.3f} >= layer {i - 1}={confidences[i - 1]:.3f}"
            )

    def test_confidence_matches_decay_formula(self):
        """置信度符合 conf_k = decay^k 公式。"""
        decay = 0.85
        graphs = _make_graphs(4, ["X", "A", "Y"], weight=0.5)
        result = chain_reason(
            causal_graphs=graphs,
            initial_interventions={"X": 1.0},
            target="Y",
            decay=decay,
            min_confidence=0.01,
            max_depth=4,
        )

        for depth, _, conf in result["chain"]:
            expected = decay ** (depth - 1)  # 第 1 层 conf=1.0 (decay^0)
            assert abs(conf - expected) < 0.01, f"Layer {depth}: conf={conf:.4f} should ≈ {expected:.4f}"

    def test_early_termination_on_low_confidence(self):
        """置信度低于阈值时提前终止。"""
        graphs = _make_graphs(10, ["X", "A", "Y"], weight=0.5)
        result = chain_reason(
            causal_graphs=graphs,
            initial_interventions={"X": 1.0},
            target="Y",
            decay=0.5,  # 快速衰减
            min_confidence=0.1,
            max_depth=10,
        )

        # decay=0.5: conf = [1.0, 0.5, 0.25, 0.125, 0.0625]
        # 应在 depth 4 (conf=0.125) 后继续，depth 5 (conf=0.0625) 前终止
        assert result["depth"] < 10, f"Should terminate early, got depth={result['depth']}"
        # 最后一层的置信度应 >= min_confidence
        _, _, last_conf = result["chain"][-1]
        assert last_conf >= 0.1, f"Last conf {last_conf:.3f} should >= 0.1"


# =============================================================================
# 推理结果传递性
# =============================================================================


class TestResultPropagation:
    """验证推理结果在链中的传递逻辑。"""

    def test_result_as_next_input(self):
        """每层结果作为下层输入 (R1 → R2 → R3)。"""
        # 简单链: X → Y, 权重 2.0
        # Layer 1: X=1.0 → Y=2.0
        # Layer 2: Y=2.0 (干预固定) → Y=2.0
        # Layer 3: Y=2.0 (干预固定) → Y=2.0
        # 因为 Y 被干预固定，后续层保持该值
        graphs = _make_graphs(3, ["X", "Y"], weight=2.0)
        result = chain_reason(
            causal_graphs=graphs,
            initial_interventions={"X": 1.0},
            target="Y",
            decay=1.0,  # 无衰减
            min_confidence=0.01,
            max_depth=3,
        )

        values = [v for _, v, _ in result["chain"]]
        assert len(values) == 3
        # Layer 1: Y = 2 * X = 2.0
        assert abs(values[0] - 2.0) < 0.1, f"Layer 1: Y should be ~2.0, got {values[0]:.4f}"
        # Layer 2+: Y 被干预固定为 Layer 1 结果
        assert abs(values[1] - 2.0) < 0.1, f"Layer 2: Y should be ~2.0 (fixed), got {values[1]:.4f}"

    def test_stable_fixed_point(self):
        """权重 < 1 时链式推理趋向不动点。"""
        # X → Y, 权重 0.5
        # Y_k = 0.5 * Y_{k-1} → 收敛到 0
        graphs = _make_graphs(10, ["X", "Y"], weight=0.5)
        result = chain_reason(
            causal_graphs=graphs,
            initial_interventions={"X": 10.0},
            target="Y",
            decay=1.0,
            min_confidence=0.001,
            max_depth=10,
        )

        values = [v for _, v, _ in result["chain"]]
        # 值应单调递减趋向 0
        for i in range(1, len(values)):
            assert abs(values[i]) <= abs(values[i - 1]) + 0.01, f"Values should converge: layer {i}={values[i]:.4f}"


# =============================================================================
# 综合评估
# =============================================================================


class TestChainReasoningComposite:
    """综合评估: 链式推理的可观测性。"""

    def test_chain_result_structure(self):
        """链式推理结果包含完整结构。"""
        graphs = _make_graphs(3, ["X", "A", "Y"], weight=0.5)
        result = chain_reason(
            causal_graphs=graphs,
            initial_interventions={"X": 1.0},
            target="Y",
            decay=0.9,
            min_confidence=0.1,
            max_depth=3,
        )

        assert "chain" in result
        assert "final_value" in result
        assert "final_confidence" in result
        assert "depth" in result
        assert isinstance(result["chain"], list)
        assert result["depth"] > 0

    def test_confidence_observable(self):
        """置信度衰减可观测。"""
        graphs = _make_graphs(5, ["X", "A", "Y"], weight=0.5)
        result = chain_reason(
            causal_graphs=graphs,
            initial_interventions={"X": 1.0},
            target="Y",
            decay=0.8,
            min_confidence=0.01,
            max_depth=5,
        )

        confidences = [c for _, _, c in result["chain"]]
        assert confidences[0] == 1.0, "First layer confidence should be 1.0"
        assert confidences[-1] < confidences[0], "Last confidence should < first"

    def test_empty_input(self):
        """空输入返回空链。"""
        result = chain_reason(
            causal_graphs=[],
            initial_interventions={"X": 1.0},
            target="Y",
        )
        assert result["depth"] == 0
        assert result["chain"] == []

    def test_logical_consistency_across_layers(self):
        """3 层链式推理结果逻辑一致。"""
        graphs = _make_graphs(3, ["X", "A", "Y"], weight=1.5)
        result = chain_reason(
            causal_graphs=graphs,
            initial_interventions={"X": 1.0},
            target="Y",
            decay=0.9,
            min_confidence=0.01,
            max_depth=3,
        )

        values = [v for _, v, _ in result["chain"]]
        # Layer 1: X→A(1.5)→Y(1.5), Y = 1.5 * 1.5 * 1.0 = 2.25
        # Layer 2: Y 被干预固定为 2.25
        # Layer 3: Y 被干预固定为 2.25
        if len(values) >= 3:
            # Layer 1: Y ≈ 2.25
            assert abs(values[0] - 2.25) < 0.1, f"Layer 1 should be ~2.25, got {values[0]:.4f}"
            # Layer 2+: Y 保持 Layer 1 的值
            assert abs(values[1] - values[0]) < 0.1, f"Layer 2 should ≈ Layer 1: {values[1]:.4f} vs {values[0]:.4f}"
