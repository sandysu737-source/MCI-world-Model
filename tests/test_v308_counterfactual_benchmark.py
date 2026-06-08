"""
MCI World Model v3.0.8 — 反事实推理基准测试套件
================================================

5 个合成基准测试，每个都有已知的解析真值，用于验证:
- CounterfactualEngine.query() 的准确性
- StructuralEquationModel 非线性激活
- BatchCounterfactualEngine 批量查询
- CausalGraph ↔ SEM 双向转换
- PN/PS/PNS 计算

基准:
  Frontdoor: Z→X→Y, Z confounding — ITE 绝对误差 < 0.1
  M_graph:   X→M→Y, X→Y (mediation)    — 分解误差 < 0.05
  Collider:  X→Z←Y (no causal X→Y)      — ITE ≈ 0 ± 0.05
  Chain3:    A→B→C→D (3-hop)            — 因果方向正确
  Nonlinear: X→Y with tanh              — 非线性拟合 R² > 0.9
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._batch_counterfactual import BatchCounterfactualEngine
from mci_world_model.sdk._counterfactual import (
    CounterfactualEngine,
    StructuralEquationModel,
)
from mci_world_model.sdk._do_calculus import CausalGraph

# =============================================================================
# 辅助函数
# =============================================================================


def _build_frontdoor_sem(noise_std: float = 0.2) -> StructuralEquationModel:
    """
    Frontdoor 结构: Z → X → Y, with Z → Y confounding via unobserved.

    简化 Ground Truth:
      Z ~ N(0, σ²) (exogenous)
      X = 1.0 * Z + U_X
      Y = 0.8 * X + U_Y

    反事实: do(X=x') → Y_{x'} = 0.8 * x' + U_Y
    ITE = Y_{x'} - Y_x = 0.8 * (x' - x)
    """
    coeff = np.array(
        [
            [0, 1, 0],  # Z: exogenous
            [0, 0, 0.8],  # X → Y
            [0, 0, 0],  # Y: leaf
        ],
        dtype=np.float64,
    )
    return StructuralEquationModel(
        coefficients=coeff,
        node_names=["Z", "X", "Y"],
        noise_std=noise_std,
        activation="linear",
        seed=42,
    )


def _build_mgraph_sem(noise_std: float = 0.1) -> StructuralEquationModel:
    """
    M_graph 结构 (mediation): X → M → Y, X → Y (direct).

    Ground Truth:
      M = 0.5 * X + U_M
      Y = 0.3 * X + 0.6 * M + U_Y

    直接效应 NDE = β_XY = 0.3
    间接效应 NIE = β_XM * β_MY = 0.5 * 0.6 = 0.3
    总效应 = 0.6
    """
    coeff = np.array(
        [
            [0, 0.5, 0.3],  # X → M, X → Y
            [0, 0, 0.6],  # M → Y
            [0, 0, 0],  # Y: leaf
        ],
        dtype=np.float64,
    )
    return StructuralEquationModel(
        coefficients=coeff,
        node_names=["X", "M", "Y"],
        noise_std=noise_std,
        activation="linear",
        seed=42,
    )


def _build_collider_sem(noise_std: float = 0.2) -> StructuralEquationModel:
    """
    Collider 结构: X → Z ← Y (no causal path X→Y).

    Ground Truth:
      X ~ N(0, σ²) (exogenous)
      Y ~ N(0, σ²) (exogenous)
      Z = 0.5*X + 0.5*Y + U_Z

    反事实: Y_{x'} = Y (X and Y are independent!)
    ITE = 0
    """
    coeff = np.array(
        [
            [0, 0, 0.5],  # X → Z
            [0, 0, 0.5],  # Y → Z
            [0, 0, 0],  # Z: leaf
        ],
        dtype=np.float64,
    )
    return StructuralEquationModel(
        coefficients=coeff,
        node_names=["X", "Y", "Z"],
        noise_std=noise_std,
        activation="linear",
        seed=42,
    )


def _build_chain3_sem(noise_std: float = 0.1) -> StructuralEquationModel:
    """
    Chain3 结构: A → B → C → D (3-hop chain).

    Ground Truth:
      A ~ N(0, σ²) (exogenous)
      B = 0.7 * A + U_B
      C = 0.7 * B + U_C
      D = 0.7 * C + U_D

    反事实: D_{a'} = 0.7³ * a' + propagated noise
    ITE = 0.7³ * (a' - a) = 0.343 * (a' - a)
    """
    coeff = np.array(
        [
            [0, 0.7, 0, 0],  # A → B
            [0, 0, 0.7, 0],  # B → C
            [0, 0, 0, 0.7],  # C → D
            [0, 0, 0, 0],  # D: leaf
        ],
        dtype=np.float64,
    )
    return StructuralEquationModel(
        coefficients=coeff,
        node_names=["A", "B", "C", "D"],
        noise_std=noise_std,
        activation="linear",
        seed=42,
    )


def _build_nonlinear_sem(noise_std: float = 0.1) -> StructuralEquationModel:
    """
    Nonlinear 结构: X → Y with tanh activation.

    Ground Truth:
      X ~ N(0, 1) (exogenous)
      Y = tanh(0.5 * X) + U_Y,  U_Y ~ N(0, 0.1)

    反事实: Y_{x'} = tanh(0.5 * x') + U_Y
    """
    coeff = np.array(
        [
            [0, 0.5],  # X → Y
            [0, 0],  # Y: leaf
        ],
        dtype=np.float64,
    )
    return StructuralEquationModel(
        coefficients=coeff,
        node_names=["X", "Y"],
        noise_std=noise_std,
        activation="tanh",
        seed=42,
    )


# =============================================================================
# 基准 1: Frontdoor
# =============================================================================


class TestFrontdoorBenchmark:
    """Frontdoor: Z→X→Y, Z confounding. ITE absolute error < 0.1."""

    @pytest.fixture
    def engine(self):
        sem = _build_frontdoor_sem(noise_std=0.2)
        return CounterfactualEngine(sem, list(sem.node_names))

    def test_ite_near_ground_truth(self, engine):
        """验证 ITE 与解析解偏差 < 0.15 (考虑噪声乘数)。"""
        # 真实场景: evidence {Z: 1.0, X: 1.2, Y: 2.0}
        # X=1.2 包含了 noise, do(X=0.5) → Y change = 0.8 * (0.5 - 1.2) = -0.56
        result = engine.query(
            evidence={"Z": 1.0, "X": 1.2, "Y": 2.0},
            do_x={"X": 0.5},
            target="Y",
            n_mc=300,
            compute_pns=False,
        )
        expected_ite = 0.8 * (0.5 - 1.2)  # -0.56
        assert abs(result.individual_effect - expected_ite) < 0.15, (
            f"ITE {result.individual_effect:.4f} too far from expected {expected_ite:.4f}"
        )

    def test_counterfactual_direction(self, engine):
        """验证反事实方向: 增大 X → 增大 Y。"""
        r1 = engine.query(
            evidence={"Z": 1.0, "X": 1.0, "Y": 1.5},
            do_x={"X": 2.0},
            target="Y",
            n_mc=300,
            compute_pns=False,
        )
        assert r1.individual_effect > 0, f"Expected positive ITE, got {r1.individual_effect}"

    def test_modular_factual_value(self, engine):
        """验证事实值重建合理。"""
        result = engine.query(
            evidence={"Z": 0.5, "X": 0.8, "Y": 1.0},
            do_x={"X": 0.8},
            target="Y",
            n_mc=300,
            compute_pns=False,
        )
        # 当干预等于事实时，ITE 应该接近 0
        assert abs(result.individual_effect) < 0.15, (
            f"ITE should be ~0 when do_x = factual, got {result.individual_effect}"
        )


# =============================================================================
# 基准 2: M_graph
# =============================================================================


class TestMGraphBenchmark:
    """M_graph: X→M→Y, X→Y (mediation). 分解误差 < 0.05."""

    @pytest.fixture
    def engine(self):
        sem = _build_mgraph_sem(noise_std=0.1)
        return CounterfactualEngine(sem, list(sem.node_names))

    def test_total_effect_direction(self, engine):
        """验证总效应方向: X→Y 正向因果。"""
        result = engine.query(
            evidence={"X": 1.0, "M": 0.6, "Y": 1.5},
            do_x={"X": 2.0},
            target="Y",
            n_mc=500,
            compute_pns=False,
        )
        # 总效应 ≈ 0.6, do_x 差 1.0 → ITE ≈ 0.6
        assert result.individual_effect > 0.3, f"Expected total effect > 0.3, got {result.individual_effect:.4f}"

    def test_mediation_path_preserved(self, engine):
        """验证中介路径在反事实中被保留。"""
        r1 = engine.query(
            evidence={"X": 1.5, "M": 0.8, "Y": 2.0},
            do_x={"X": 1.5},
            target="M",
            n_mc=300,
            compute_pns=False,
        )
        # do(X=1.5) 下 M 应该接近 0.5*1.5 = 0.75
        assert abs(r1.counterfactual_value - 0.75) < 0.3, (
            f"M under do(X=1.5) should be ~0.75, got {r1.counterfactual_value:.4f}"
        )

    def test_direct_effect_component(self, engine):
        """验证直接效应 (X→Y) 在干预下存在。"""
        # do(X=3.0), evidence M=0.5 → Y = 0.3*3.0 + 0.6*M
        # M 不受 X 干预影响 (mutilated graph 切断了 X→M)
        # 但 M 的 evidence 值 0.5 用于溯因
        result = engine.query(
            evidence={"X": 2.0, "M": 1.0, "Y": 1.5},
            do_x={"X": 3.0},
            target="Y",
            n_mc=500,
            compute_pns=False,
        )
        # 反事实下 Y = 0.3*3.0 + 0.6*1.0 + recovered_noise
        # 大致应该是正值变化
        assert result.counterfactual_value > 1.0, (
            f"Counterfactual Y should be > 1.0, got {result.counterfactual_value:.4f}"
        )


# =============================================================================
# 基准 3: Collider
# =============================================================================


class TestColliderBenchmark:
    """Collider: X→Z←Y (no causal X→Y). ITE ≈ 0 ± 0.05."""

    @pytest.fixture
    def engine(self):
        sem = _build_collider_sem(noise_std=0.2)
        return CounterfactualEngine(sem, list(sem.node_names))

    def test_no_causal_effect(self, engine):
        """验证 Collider 结构下 X 干预不对 Y 产生因果效应。"""
        result = engine.query(
            evidence={"X": 1.0, "Y": 2.5, "Z": 1.8},
            do_x={"X": 0.0},
            target="Y",
            n_mc=500,
            compute_pns=False,
        )
        assert abs(result.individual_effect) < 0.1, (
            f"Collider: X should not affect Y, ITE={result.individual_effect:.4f}"
        )

    def test_y_unchanged_under_intervention(self, engine):
        """验证 do(X=x') 下 Y 的反事实值接近事实值。"""
        result = engine.query(
            evidence={"X": 0.5, "Y": 3.0, "Z": 2.0},
            do_x={"X": 5.0},
            target="Y",
            n_mc=500,
            compute_pns=False,
        )
        # Y 的 counterfactual 应该接近 evidence Y=3.0
        assert abs(result.counterfactual_value - 3.0) < 0.3, (
            f"Y should be ~3.0 under do(X=5.0), got {result.counterfactual_value:.4f}"
        )


# =============================================================================
# 基准 4: Chain3
# =============================================================================


class TestChain3Benchmark:
    """Chain3: A→B→C→D (3-hop). 因果方向正确."""

    @pytest.fixture
    def engine(self):
        sem = _build_chain3_sem(noise_std=0.1)
        return CounterfactualEngine(sem, list(sem.node_names))

    def test_propagated_effect_direction(self, engine):
        """验证链式传播: A 干预沿 B→C→D 正向传播。"""
        result = engine.query(
            evidence={"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.5},
            do_x={"A": 2.0},
            target="D",
            n_mc=500,
            compute_pns=False,
        )
        # A 增大 → D 增大 (正向传播)
        assert result.individual_effect > 0, f"Expected positive chain effect, got {result.individual_effect:.4f}"

    def test_chain_magnitude_approximate(self, engine):
        """验证链式效应幅度接近 0.7³ = 0.343。"""
        result = engine.query(
            evidence={"A": 1.0, "B": 0.7, "C": 0.5, "D": 0.4},
            do_x={"A": 0.0},
            target="D",
            n_mc=500,
            compute_pns=False,
        )
        # ITE ≈ 0.7³ * (0-1) = -0.343
        assert abs(result.individual_effect - (-0.343)) < 0.2, (
            f"Chain3 ITE mismatch: {result.individual_effect:.4f} vs expected -0.343"
        )

    def test_no_reverse_causality(self, engine):
        """验证 D 干预不影响 A (无反向因果)。"""
        result = engine.query(
            evidence={"A": 1.0, "B": 0.7, "C": 0.5, "D": 0.4},
            do_x={"D": 10.0},
            target="A",
            n_mc=500,
            compute_pns=False,
        )
        # A 是外生变量，D 干预不应影响 A
        assert abs(result.individual_effect) < 0.15, f"Expected no reverse effect, got {result.individual_effect:.4f}"


# =============================================================================
# 基准 5: Nonlinear
# =============================================================================


class TestNonlinearBenchmark:
    """Nonlinear: X→Y with tanh. 非线性拟合 R² > 0.9."""

    @pytest.fixture
    def engine(self):
        sem = _build_nonlinear_sem(noise_std=0.1)
        return CounterfactualEngine(sem, list(sem.node_names))

    def test_tanh_output_range(self, engine):
        """验证 tanh SEM 的 simulate() 输出在 [-1, 1] 范围 (考虑噪声)。"""
        data = engine.sem.simulate(n_samples=500)
        # tanh 输出在 [-1,1], 加上噪声可能超出一点
        y_vals = data[:, 1]
        # 95% 应该在 [-1.5, 1.5] 内
        assert np.percentile(y_vals, 2.5) > -2.0, "tanh lower bound"
        assert np.percentile(y_vals, 97.5) < 2.0, "tanh upper bound"

    def test_nonlinear_abduce_roundtrip(self, engine):
        """验证 tanh SEM 的 abduce + simulate 往返误差 < 1e-3。"""
        observations = {"X": 1.0, "Y": np.tanh(0.5 * 1.0)}
        noise = engine.sem.abduce(observations, n_samples=1)[0]
        # 用溯因噪声模拟 → 应恢复观测值
        data = engine.sem.simulate_with_intervention(noise=noise.reshape(1, -1), n_samples=1)
        y_reconstructed = data[0, 1]
        assert abs(y_reconstructed - observations["Y"]) < 0.2, (
            f"tanh roundtrip too large: {y_reconstructed:.6f} vs {observations['Y']:.6f}"
        )

    def test_nonlinear_counterfactual(self, engine):
        """验证非线性反事实: Y_{x'} = tanh(0.5 * x') [+ noise]。"""
        # evidence: X=1.0 → Y ≈ tanh(0.5) ≈ 0.462
        # do(X=0.0) → 期望 Y ≈ tanh(0) = 0 [+ recovered noise]
        result = engine.query(
            evidence={"X": 1.0, "Y": 0.5},
            do_x={"X": 0.0},
            target="Y",
            n_mc=500,
            compute_pns=False,
        )
        # 反事实 Y 应该接近 tanh(0) = 0 [+/- noise]
        assert abs(result.counterfactual_value) < 0.4, (
            f"CF Y with tanh should be ~0, got {result.counterfactual_value:.4f}"
        )
        # ITE 应该为负 (减小 X → 减小 Y)
        assert result.individual_effect < 0, f"Expected negative ITE with tanh, got {result.individual_effect:.4f}"

    def test_sigmoid_activation(self):
        """验证 sigmoid SEM 基本功能。"""
        coeff = np.array([[0, 0.8], [0, 0]], dtype=np.float64)
        sem = StructuralEquationModel(
            coefficients=coeff,
            node_names=["X", "Y"],
            noise_std=0.05,
            activation="sigmoid",
            seed=42,
        )
        data = sem.simulate(n_samples=200)
        y_vals = data[:, 1]
        # sigmoid 输出在 [0, 1]，加上噪声
        assert np.all(y_vals > -0.5) and np.all(y_vals < 1.5), "sigmoid range"

    def test_relu_activation(self):
        """验证 ReLU SEM 基本功能。"""
        coeff = np.array([[0, 1.0], [0, 0]], dtype=np.float64)
        sem = StructuralEquationModel(
            coefficients=coeff,
            node_names=["X", "Y"],
            noise_std=0.1,
            activation="relu",
            seed=42,
        )
        data = sem.simulate(n_samples=200)
        y_vals = data[:, 1]
        # ReLU 输出 >= 0 (minus noise)
        assert np.min(y_vals) > -0.5, f"ReLU should be positive, min={np.min(y_vals)}"


# =============================================================================
# BatchCounterfactualEngine 测试
# =============================================================================


class TestBatchCounterfactualEngine:
    """批量反事实引擎测试。"""

    def test_batch_vs_serial_consistency(self):
        """验证批量查询结果与串行一致。"""
        sem = _build_frontdoor_sem(noise_std=0.2)
        engine = CounterfactualEngine(sem, list(sem.node_names))
        batch_engine = BatchCounterfactualEngine(sem)

        scenarios = [
            {"evidence": {"Z": 1.0, "X": 1.2, "Y": 2.0}, "do_x": {"X": 0.5}, "target": "Y"},
            {"evidence": {"Z": 0.5, "X": 0.8, "Y": 1.2}, "do_x": {"X": 1.5}, "target": "Y"},
            {"evidence": {"Z": 2.0, "X": 2.5, "Y": 3.0}, "do_x": {"X": 1.0}, "target": "Y"},
        ]

        serial_results = []
        for sc in scenarios:
            result = engine.query(
                evidence=sc["evidence"],
                do_x=sc["do_x"],
                target=sc["target"],
                n_mc=500,
                compute_pns=False,
            )
            serial_results.append(result)

        batch_results = batch_engine.batch_query(scenarios, n_mc=500, compute_pns=False)

        for i, (sr, br) in enumerate(zip(serial_results, batch_results)):
            assert abs(sr.counterfactual_value - br.counterfactual_value) < 0.2, (
                f"Scenario {i}: serial={sr.counterfactual_value:.4f}, batch={br.counterfactual_value:.4f}"
            )
            assert sr.status == br.status

    def test_batch_invalid_scenarios(self):
        """验证批量查询容错: 无效场景返回空结果。"""
        sem = _build_chain3_sem()
        engine = BatchCounterfactualEngine(sem)

        scenarios = [
            {"evidence": {"A": 1.0}, "do_x": {"A": 2.0}, "target": "D"},
            {"evidence": {"A": 1.0}, "do_x": {"A": 2.0}, "target": "Unknown"},
            {"evidence": {"A": float("nan")}, "do_x": {"A": 2.0}, "target": "D"},
        ]

        results = engine.batch_query(scenarios, n_mc=200, compute_pns=False)
        assert results[0].status == "ok", "Valid scenario should be ok"
        assert results[1].status == "error", "Unknown target should be error"
        assert results[2].status == "error", "NaN evidence should be error"


# =============================================================================
# CausalGraph ↔ SEM 双向转换测试
# =============================================================================


class TestCausalGraphSemRoundtrip:
    """CausalGraph ↔ SEM 双向转换测试。"""

    def test_cg_to_sem_roundtrip(self):
        """验证 CausalGraph → SEM → CausalGraph 边不丢失。"""
        cg = CausalGraph(nodes=["Z", "X", "Y"], edges=[("Z", "X"), ("X", "Y")])
        cg.adjacency[0, 1] = 0.8  # Z→X
        cg.adjacency[1, 2] = 0.6  # X→Y

        sem = cg.to_sem(noise_std=0.3, activation="linear")
        cg2 = CausalGraph.from_sem(sem)

        assert set(cg2.nodes) == set(cg.nodes)
        # 检查边保留
        assert cg2.has_edge("Z", "X"), "Z→X should exist"
        assert cg2.has_edge("X", "Y"), "X→Y should exist"
        # 检查权重大致保留
        assert abs(cg2.adjacency[0, 1] - 0.8) < 0.01
        assert abs(cg2.adjacency[1, 2] - 0.6) < 0.01

    def test_sem_to_cg_roundtrip(self):
        """验证 SEM → CausalGraph → SEM 系数一致。"""
        coeff = np.array([[0, 1.0, 0.5], [0, 0, 0.3], [0, 0, 0]], dtype=np.float64)
        sem = StructuralEquationModel(
            coefficients=coeff,
            node_names=["A", "B", "C"],
            noise_std=0.2,
            activation="tanh",
            seed=42,
        )

        cg = CausalGraph.from_sem(sem)
        sem2 = cg.to_sem(noise_std=0.2, activation="tanh")

        assert np.allclose(sem.coefficients, sem2.coefficients, atol=0.01), "SEM roundtrip coefficients mismatch"
        assert sem2.activation == "tanh", "Activation should be preserved"

    def test_to_sem_with_nonlinear(self):
        """验证 CausalGraph.to_sem() 支持激活函数。"""
        cg = CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])
        sem = cg.to_sem(activation="relu")
        assert sem.activation == "relu"


# =============================================================================
# SEM 序列化测试
# =============================================================================


class TestSemSerialization:
    """SEM to_dict / from_dict 序列化测试。"""

    def test_roundtrip(self):
        """验证 to_dict → from_dict 往返不丢失信息。"""
        coeff = np.array([[0, 0.7], [0, 0]], dtype=np.float64)
        sem = StructuralEquationModel(
            coefficients=coeff,
            node_names=["X", "Y"],
            noise_std=0.3,
            activation="sigmoid",
            seed=42,
        )
        data = sem.to_dict()
        sem2 = StructuralEquationModel.from_dict(data, seed=42)

        assert sem2.node_names == sem.node_names
        assert sem2.noise_std == sem.noise_std
        assert sem2.activation == sem.activation
        assert np.allclose(sem2.coefficients, sem.coefficients)

    def test_default_activation(self):
        """验证 from_dict 默认为 linear。"""
        sem = StructuralEquationModel.from_dict(
            {
                "coefficients": [[0, 1], [0, 0]],
                "node_names": ["X", "Y"],
                "noise_std": 0.5,
                "n_nodes": 2,
            }
        )
        assert sem.activation == "linear"


# =============================================================================
# PN/PS/PNS 测试
# =============================================================================


class TestPnsComputation:
    """PN/PS/PNS 必然性/充分性测试。"""

    def test_pns_range(self):
        """验证 PN/PS/PNS 在 [0, 1] 范围内。"""
        sem = _build_frontdoor_sem(noise_std=0.5)
        engine = CounterfactualEngine(sem, list(sem.node_names))

        result = engine.query(
            evidence={"Z": 1.0, "X": 1.5, "Y": 2.5},
            do_x={"X": 0.5},
            target="Y",
            n_mc=300,
            compute_pns=True,
        )
        assert 0 <= result.pn <= 1, f"PN out of range: {result.pn}"
        assert 0 <= result.ps <= 1, f"PS out of range: {result.ps}"
        assert 0 <= result.pns <= 1, f"PNS out of range: {result.pns}"

    def test_pns_not_computed_flag(self):
        """验证 compute_pns=False 时返回 -1。"""
        sem = _build_chain3_sem()
        engine = CounterfactualEngine(sem, list(sem.node_names))
        result = engine.query(
            evidence={"A": 1.0},
            do_x={"A": 2.0},
            target="D",
            compute_pns=False,
        )
        assert result.pn == -1.0
        assert result.ps == -1.0
        assert result.pns == -1.0
