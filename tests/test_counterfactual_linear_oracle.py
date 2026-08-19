"""L1 解析 oracle 测试 — 反事实引擎 (线性 SEM 闭式解)。

线性 SEM X = B·X + ε (B 下三角), 噪声固定时:
  - 事实值:   X_factual = (I - B)^{-1} · ε
  - 干预 do(X_k=c): 等价于清零 B 的第 k 行 + ε_k = c
  - 反事实:   X_cf = (I - B_mutilated)^{-1} · ε_modified

这些是确定性的解析解, 可作 oracle 验证反事实引擎的 Abduction+Action+Prediction。
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.oracle

import numpy as np
import pytest

from mci_world_model.sdk._counterfactual import CounterfactualEngine, StructuralEquationModel
from mci_world_model.sdk._do_calculus import CausalGraph


def _make_linear_chain_sem(seed: int = 42) -> StructuralEquationModel:
    """构造 X→Y→Z 链式线性 SEM。

    系数: X = ε_x, Y = 0.6·X + ε_y, Z = 0.8·Y + ε_z
    coefficients[parent, child] 约定: B[0,1]=0.6 (X→Y), B[1,2]=0.8 (Y→Z)
    """
    B = np.array(
        [
            [0.0, 0.6, 0.0],
            [0.0, 0.0, 0.8],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    return StructuralEquationModel(B, ["X", "Y", "Z"], noise_std=0.3, activation="linear", seed=seed)


class TestLinearSEMAnalyticCounterfactual:
    """线性 SEM 反事实的解析闭式解对照。"""

    def test_abduce_recovers_noise_linear(self):
        """Abduction: 线性 SEM 下, 从事实值反推的噪声应等于解析噪声。

        线性 SEM: ε_i = V_i - Σ_j B[j,i]·V_j  (逐节点求逆, 精确)
        """
        sem = _make_linear_chain_sem()
        factual = sem.simulate(n_samples=1)[0]  # 用 SEM 自身生成一致数据
        recovered = sem.abduce(
            {"X": factual[0], "Y": factual[1], "Z": factual[2]},
            n_samples=1,
        )
        # 手动计算解析噪声: ε = V - B^T·V (B[parent,child], 所以 parent_sum = B[:,i]·V)
        manual_noise = factual - sem.coefficients.T @ factual
        assert np.allclose(recovered[0], manual_noise, atol=1e-9), (
            f"abduction 失败: 解析{manual_noise}, abduce{recovered[0]}"
        )

    def test_counterfactual_closed_form(self):
        """反事实: do(Y=0) 后 Z 的值应有解析解。

        事实: X=ε_x, Y=0.6X+ε_y, Z=0.8Y+ε_z
        do(Y=0): Y被强制为0, Z = 0.8·0 + ε_z = ε_z (Z 的原始噪声)
        """
        sem = _make_linear_chain_sem()
        factual = sem.simulate(n_samples=1)[0]
        # Z 的原始噪声 = factual[Z] - 0.8·factual[Y]
        epsilon_z = factual[2] - 0.8 * factual[1]

        cg = CausalGraph(
            nodes=["X", "Y", "Z"],
            edges=[("X", "Y"), ("Y", "Z")],
            adjacency=sem.coefficients.astype(np.float32),
        )
        engine = CounterfactualEngine.from_causal_graph(cg)
        result = engine.query(
            evidence={"X": factual[0], "Y": factual[1], "Z": factual[2]},
            do_x={"Y": 0.0},
            target="Z",
            n_mc=200,
        )
        # 解析: Z_cf = ε_z
        assert abs(result.counterfactual_value - epsilon_z) < 0.15, (
            f"反事实 Z_cf: 期望≈{epsilon_z:.3f}, 得到{result.counterfactual_value:.3f}"
        )

    def test_counterfactual_preserves_evidence_when_no_intervention(self):
        """无干预时, 反事实值应等于事实值。"""
        sem = _make_linear_chain_sem()
        factual = sem.simulate(n_samples=1)[0]

        cg = CausalGraph(
            nodes=["X", "Y", "Z"],
            edges=[("X", "Y"), ("Y", "Z")],
            adjacency=sem.coefficients.astype(np.float32),
        )
        engine = CounterfactualEngine.from_causal_graph(cg)
        # do(X = factual[X]): 干预值为事实值, 应保持不变
        result = engine.query(
            evidence={"X": factual[0], "Y": factual[1], "Z": factual[2]},
            do_x={"X": factual[0]},
            target="Z",
            n_mc=100,
        )
        assert abs(result.counterfactual_value - factual[2]) < 0.15, (
            f"无效应干预: Z应≈事实{factual[2]:.3f}, 得到{result.counterfactual_value:.3f}"
        )


class TestPNPSBoundaries:
    """PN/PS/PNS 概率边界不变式 (L0)。"""

    def test_pn_ps_pns_in_unit_interval(self):
        """PN/PS/PNS 必须落在 [0,1]。"""
        sem = _make_linear_chain_sem()
        factual = sem.simulate(n_samples=1)[0]
        cg = CausalGraph(
            nodes=["X", "Y", "Z"],
            edges=[("X", "Y"), ("Y", "Z")],
            adjacency=sem.coefficients.astype(np.float32),
        )
        engine = CounterfactualEngine.from_causal_graph(cg)
        result = engine.query(
            evidence={"X": factual[0], "Y": factual[1]},
            do_x={"X": factual[0] + 1.0},
            target="Y",
            n_mc=100,
        )
        for name, val in [("PN", result.pn), ("PS", result.ps), ("PNS", result.pns)]:
            if val >= 0:  # -1 表示未计算
                assert 0.0 <= val <= 1.0, f"{name}={val} 越出[0,1]"


class TestUnobservedNodePosterior:
    """L1 oracle — 未观测节点的高斯后验推断。

    验证 abduce 对未观测节点用后验 P(U|E) 而非先验, 收紧不确定性。
    这是可证伪的改进: 证据应约束未观测共因。
    """

    def test_unobserved_posterior_tighter_than_prior(self):
        """未观测节点的后验 std 应小于先验 std (证据收紧不确定性)。"""
        # U(未观测) -> X, Y (观测): 共因结构
        B = np.array(
            [
                [0.0, 0.8, 0.6],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        sem = StructuralEquationModel(B, ["U", "X", "Y"], noise_std=0.5, activation="linear", seed=42)
        rng = np.random.RandomState(7)
        u_true, ex, ey = rng.randn(3)
        X = 0.8 * u_true + 0.5 * ex
        Y = 0.6 * u_true + 0.5 * ey

        # 只观测 X, Y (U 未观测)
        noise = sem.abduce({"X": X, "Y": Y}, n_samples=3000)
        post_U = noise[:, 0]
        assert post_U.std() < sem.noise_std * 0.9, (
            f"后验 std={post_U.std():.3f} 未显著小于先验 {sem.noise_std:.3f}, 证据未收紧未观测节点的不确定性"
        )

    def test_posterior_mean_informed_by_evidence(self):
        """后验均值应被证据"拉向"真实值 (非先验均值 0)。"""
        B = np.array(
            [
                [0.0, 0.8, 0.6],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        sem = StructuralEquationModel(B, ["U", "X", "Y"], noise_std=0.5, activation="linear", seed=42)
        # 强证据: X, Y 都大 => U 应该也大
        noise = sem.abduce({"X": 3.0, "Y": 2.5}, n_samples=3000)
        post_U = noise[:, 0]
        # 后验均值应明显 > 0 (证据指向 U 为正)
        assert post_U.mean() > 0.5, f"强正证据下 U 后验均值={post_U.mean():.3f}, 应被拉向正值"

    def test_nonlinear_falls_back_to_prior(self):
        """非线性 SEM 应回退到先验 (无闭式后验)。"""
        B = np.array([[0.0, 0.7], [0.0, 0.0]], dtype=np.float64)
        sem = StructuralEquationModel(B, ["X", "Y"], noise_std=0.4, activation="tanh", seed=42)
        # 不应崩溃, 回退先验采样
        noise = sem.abduce({"Y": 0.5}, n_samples=100)
        assert noise.shape == (100, 2)
        # X 未观测, 噪声 std 应接近先验 (非线性无后验收紧)
        assert abs(noise[:, 0].std() - sem.noise_std) < 0.15
