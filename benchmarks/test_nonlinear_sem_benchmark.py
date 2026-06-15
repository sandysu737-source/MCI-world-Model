"""
MCI World Model V4.1.0 — 非线性 SEM 基准测试集

对标: Kant「质」范畴 — 因果关系的质量变化（非线性效应）

评测三种非线性激活函数下的因果推理:
  1. tanh   饱和效应: 大输入 → Y 趋近 ±1
  2. sigmoid 阈值效应: 输入低于阈值 → Y ≈ 0，高于阈值 → Y ≈ 1
  3. relu    单侧截断: 负输入 → Y = 0

理论对标:
  - 线性 SEM: Y = Σ(parent_i × w_i) — 无阈值/饱和/截断
  - 非线性 SEM: Y = σ(Σ(parent_i × w_i)) — σ ∈ {tanh, sigmoid, relu}

运行: pytest benchmarks/test_nonlinear_sem_benchmark.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks._causal_utils import sem_forward
from mci_world_model.sdk._do_calculus import CausalGraph

# =============================================================================
# 工具函数
# =============================================================================


def _build_chain(
    names: list[str],
    weights: list[float],
) -> CausalGraph:
    """构建因果链。"""
    edges = [(names[i], names[i + 1]) for i in range(len(names) - 1)]
    cg = CausalGraph(nodes=names, edges=edges)
    for i, (src, tgt) in enumerate(edges):
        si, ti = cg.nodes.index(src), cg.nodes.index(tgt)
        cg.adjacency[si, ti] = weights[i]
    return cg


# =============================================================================
# tanh 饱和效应测试
# =============================================================================


class TestTanhSaturation:
    """tanh 激活: 大输入值下 Y 趋近 ±1（饱和），小输入近线性。"""

    def test_tanh_saturation_large_input(self):
        """大输入: tanh(5.0) ≈ 0.9999, 而 linear(5.0) = 5.0。"""
        cg = _build_chain(["X", "Y"], [5.0])
        tanh_vals = sem_forward(cg, {"X": 1.0}, activation="tanh")
        linear_vals = sem_forward(cg, {"X": 1.0}, activation="linear")
        # tanh 应饱和到接近 1
        assert abs(tanh_vals["Y"]) < 1.05, f"tanh should saturate, got {tanh_vals['Y']}"
        assert abs(tanh_vals["Y"]) > 0.95, f"tanh should be near 1, got {tanh_vals['Y']}"
        # linear 应远大于 1
        assert linear_vals["Y"] > 4.0, f"linear should be ~5, got {linear_vals['Y']}"

    def test_tanh_symmetric_negative(self):
        """对称性: tanh(-x) = -tanh(x)。"""
        cg = _build_chain(["X", "Y"], [3.0])
        pos = sem_forward(cg, {"X": 1.0}, activation="tanh")
        neg = sem_forward(cg, {"X": -1.0}, activation="tanh")
        # tanh(3) ≈ 0.995, tanh(-3) ≈ -0.995
        assert abs(pos["Y"] + neg["Y"]) < 0.05, f"tanh symmetry broken: tanh(3)={pos['Y']:.4f}, tanh(-3)={neg['Y']:.4f}"

    def test_tanh_chain_propagation(self):
        """tanh 链式传播: X→V1→V2→Y, 每层都饱和。"""
        cg = _build_chain(["X", "V1", "V2", "Y"], [2.0, 2.0, 2.0])
        vals = sem_forward(cg, {"X": 1.0}, activation="tanh")
        # X=1 → V1=tanh(2)≈0.964 → V2=tanh(2*0.964)≈0.958 → Y=tanh(2*0.958)≈0.958
        assert 0.9 < vals["V1"] < 1.0, f"V1 should be ~0.96, got {vals['V1']:.4f}"
        assert 0.93 < vals["V2"] < 1.0, f"V2 should be ~0.96, got {vals['V2']:.4f}"
        assert 0.93 < vals["Y"] < 1.0, f"Y should be ~0.96, got {vals['Y']:.4f}"


# =============================================================================
# sigmoid 阈值效应测试
# =============================================================================


class TestSigmoidThreshold:
    """sigmoid 激活: 输入低于阈值 → Y ≈ 0，高于阈值 → Y ≈ 1。"""

    def test_sigmoid_low_input_near_zero(self):
        """小输入: sigmoid(-5) ≈ 0.007, 接近 0。"""
        cg = _build_chain(["X", "Y"], [1.0])
        vals = sem_forward(cg, {"X": -5.0}, activation="sigmoid")
        assert vals["Y"] < 0.05, f"sigmoid(-5) should be ~0, got {vals['Y']:.4f}"

    def test_sigmoid_high_input_near_one(self):
        """大输入: sigmoid(5) ≈ 0.993, 接近 1。"""
        cg = _build_chain(["X", "Y"], [1.0])
        vals = sem_forward(cg, {"X": 5.0}, activation="sigmoid")
        assert vals["Y"] > 0.95, f"sigmoid(5) should be ~1, got {vals['Y']:.4f}"

    def test_sigmoid_midpoint(self):
        """中点: sigmoid(0) = 0.5。"""
        cg = _build_chain(["X", "Y"], [1.0])
        vals = sem_forward(cg, {"X": 0.0}, activation="sigmoid")
        assert abs(vals["Y"] - 0.5) < 0.1, f"sigmoid(0) should be ~0.5, got {vals['Y']:.4f}"


# =============================================================================
# relu 单侧截断测试
# =============================================================================


class TestReLUTruncation:
    """relu 激活: 负输入 → Y = 0，正输入 → Y = input。"""

    def test_relu_negative_input_zero(self):
        """负输入: relu(-3) = 0。"""
        cg = _build_chain(["X", "Y"], [1.0])
        vals = sem_forward(cg, {"X": -3.0}, activation="relu")
        assert abs(vals["Y"]) < 0.05, f"relu(-3) should be ~0, got {vals['Y']:.4f}"

    def test_relu_positive_passthrough(self):
        """正输入: relu(3) = 3。"""
        cg = _build_chain(["X", "Y"], [1.0])
        vals = sem_forward(cg, {"X": 3.0}, activation="relu")
        assert abs(vals["Y"] - 3.0) < 0.2, f"relu(3) should be ~3, got {vals['Y']:.4f}"

    def test_relu_chain_blocking(self):
        """relu 链阻断: X→V1→Y, w1=-1 → V1=relu(-1)=0 → Y=0。"""
        cg = _build_chain(["X", "V1", "Y"], [-1.0, 2.0])
        vals = sem_forward(cg, {"X": 1.0}, activation="relu")
        # X=1 → V1=relu(-1)=0 → Y=relu(2*0)=0
        assert abs(vals["V1"]) < 0.05, f"V1 should be 0 (relu blocked), got {vals['V1']:.4f}"
        assert abs(vals["Y"]) < 0.05, f"Y should be 0 (blocked chain), got {vals['Y']:.4f}"


# =============================================================================
# 综合对比: 线性 vs 非线性
# =============================================================================


class TestLinearVsNonlinear:
    """量化线性与非线性 SEM 在相同图结构下的结果差异。"""

    @pytest.mark.parametrize("activation", ["tanh", "sigmoid", "relu"])
    def test_nonlinear_differs_from_linear(self, activation):
        """非线性 SEM 在大输入下与线性 SEM 结果不同。"""
        # 使用负权重确保 relu 也产生差异 (relu 截断负值)
        cg = _build_chain(["X", "V1", "Y"], [-3.0, -3.0])
        linear = sem_forward(cg, {"X": 1.0}, activation="linear")
        nonlinear = sem_forward(cg, {"X": 1.0}, activation=activation)
        # 线性: V1 = -3, Y = -3 * (-3) = 9
        # tanh: V1 = tanh(-3)≈-0.995, Y = tanh(-3*(-0.995)) = tanh(2.985)≈0.995
        # sigmoid: V1 = sig(-3)≈0.047, Y = sig(-3*0.047) = sig(-0.14)≈0.465
        # relu: V1 = relu(-3)=0, Y = relu(-3*0)=0
        assert abs(linear["Y"] - nonlinear["Y"]) > 0.5, (
            f"{activation} should differ from linear: linear={linear['Y']:.4f}, {activation}={nonlinear['Y']:.4f}"
        )

    def test_all_activations_produce_valid_output(self):
        """所有激活函数均产生有限数值输出。"""
        cg = _build_chain(["X", "V1", "V2", "Y"], [2.0, 2.0, 2.0])
        for act in ("linear", "tanh", "sigmoid", "relu"):
            vals = sem_forward(cg, {"X": 1.0}, activation=act)
            for node, v in vals.items():
                assert np.isfinite(v), f"{act} produced non-finite value for {node}: {v}"
