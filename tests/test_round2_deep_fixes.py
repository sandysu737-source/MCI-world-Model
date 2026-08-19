"""第二轮对抗性审查深层缺陷修复的回归测试。

D1: abduce() 噪声后验赋值 (节点值 vs 噪声)
D2: CausalDAG list-of-tuples 构造
D3: stationary_distribution 基于 Flow 而非 AffinityMatrix
D4: clip 后归一化保守恒
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.oracle


class TestD1AbduceNoisePosterior:
    """D1: abduce 未观测节点的噪声后验应减去 parent_sum。"""

    def test_unobserved_noise_is_correct(self):
        """A→B→C 链, 观测 A,C 未观测 B: U_B = V_B_post - 2*A。"""
        from mci_world_model.sdk._counterfactual import StructuralEquationModel

        coeff = np.array([[0, 2.0, 0], [0, 0, 3.0], [0, 0, 0]])
        sem = StructuralEquationModel(
            coefficients=coeff,
            node_names=["A", "B", "C"],
            noise_std=1.0,
            activation="linear",
        )
        noise = sem.abduce({"A": 1.0, "C": 5.0}, n_samples=3000)
        # U_A = 1 (根节点)
        assert abs(noise[:, 0].mean() - 1.0) < 0.05
        # V_B 后验 ≈ 1.70, U_B = 1.70 - 2*1 = -0.30 (不是 1.70!)
        assert noise[:, 1].mean() < 0.0, "U_B 应为负, 不是节点值后验"
        assert abs(noise[:, 1].mean() - (-0.3)) < 0.15


class TestD2CausalDAGConstruction:
    """D2: CausalDAG 支持 edges=[("A","B")] 便捷构造。"""

    def test_chain_d_separation(self):
        from mci_world_model.algebra.causal_graph import CausalDAG

        dag = CausalDAG(nodes=["A", "B", "C"], edges=[("A", "B"), ("B", "C")])
        assert dag.d_separated("A", "C", {"B"}) is True
        assert dag.d_separated("A", "C", set()) is False

    def test_collider_d_separation(self):
        from mci_world_model.algebra.causal_graph import CausalDAG

        dag = CausalDAG(nodes=["A", "B", "C"], edges=[("A", "C"), ("B", "C")])
        assert dag.d_separated("A", "B", set()) is True
        assert dag.d_separated("A", "B", {"C"}) is False

    def test_weighted_edge_construction(self):
        from mci_world_model.algebra.causal_graph import CausalDAG

        dag = CausalDAG(nodes=["X", "Y"], edges=[("X", "Y", 0.5)])
        assert dag.edge_weight("X", "Y") == pytest.approx(0.5)


class TestD3StationaryDistributionOnFlow:
    """D3: stationary_distribution 应基于 Flow 矩阵动力学。"""

    def test_returns_valid_distribution(self):
        from mci_world_model._sys._energy_core import EnergyCore

        ec = EnergyCore()
        pi = ec.stationary_distribution()
        assert len(pi) == 5
        assert pi.sum() == pytest.approx(1.0)
        assert np.all(pi >= 0)

    def test_symmetric_flow_gives_uniform(self):
        """对称 Flow (五行循环) 的平稳分布应为均匀分布。"""
        from mci_world_model._sys._energy_core import EnergyCore

        ec = EnergyCore()
        pi = ec.stationary_distribution()
        assert np.allclose(pi, 0.2, atol=0.05)


class TestD4ClipConservation:
    """D4: clip 后归一化恢复守恒。"""

    def test_normal_energy_conserved(self):
        from mci_world_model._sys._energy_core import EnergyCore

        ec = EnergyCore()
        energies = {"semantic": 10.0, "causal": 5.0, "spacetime": 3.0, "generative": 2.0, "trust": 1.0}
        total0 = sum(energies.values())
        for _ in range(10):
            energies = ec._calculate_flow_step(energies)
        assert abs(sum(energies.values()) - total0) < 1e-6


class TestD6SimulatedBackdoorLabel:
    """D6: 无数据时 backdoor 结果应标注为模拟。"""

    def test_no_data_labels_simulated(self):
        from mci_world_model.sdk._do_calculus import CausalGraph, DoCalculus

        cg = CausalGraph()
        cg.add_edge("X", "Y")
        dc = DoCalculus()
        dc.set_graph(cg)
        result = dc.backdoor_adjustment("X", "Y", Z_set=[])
        assert result.method == "backdoor_simulated", f"无数据时应标注 backdoor_simulated, 实际 {result.method}"


class TestD7EncodeDimensionValidation:
    """D7: encode 对错误维度应有有意义的错误。"""

    def test_wrong_dim_raises(self):
        from mci_world_model.sdk._true_jepa_encoder import TrueJEPAConfig, TrueJEPAEncoder

        enc = TrueJEPAEncoder(TrueJEPAConfig(obs_dim=8, latent_dim=16))
        with pytest.raises(ValueError, match="obs_dim"):
            enc.encode(np.random.randn(10))

    def test_correct_dim_works(self):
        from mci_world_model.sdk._true_jepa_encoder import TrueJEPAConfig, TrueJEPAEncoder

        enc = TrueJEPAEncoder(TrueJEPAConfig(obs_dim=8, latent_dim=16))
        z = enc.encode(np.random.randn(8))
        assert z.shape == (16,)


class TestD8NaNGuard:
    """D8: 环图/奇异矩阵的 abduce 不应产生 NaN。"""

    def test_cyclic_sem_no_nan(self):
        from mci_world_model.sdk._counterfactual import StructuralEquationModel

        coeff = np.array([[0, 1.0], [1.0, 0]])  # 环 A↔B
        sem = StructuralEquationModel(
            coefficients=coeff,
            node_names=["A", "B"],
            noise_std=1.0,
            activation="linear",
        )
        noise = sem.abduce({"A": 1.0}, n_samples=10)
        assert not np.any(np.isnan(noise)), "abduce 产生了 NaN"


class TestD9DSEPNodeValidation:
    """D9: d_separated 验证节点存在性。"""

    def test_nonexistent_node_raises(self):
        from mci_world_model.algebra.causal_graph import CausalDAG

        dag = CausalDAG(nodes=["A", "B"], edges=[("A", "B")])
        with pytest.raises(KeyError, match="不在图中"):
            dag.d_separated("A", "X", set())

    def test_nonexistent_cond_set_node_raises(self):
        from mci_world_model.algebra.causal_graph import CausalDAG

        dag = CausalDAG(nodes=["A", "B"], edges=[("A", "B")])
        with pytest.raises(KeyError):
            dag.d_separated("A", "B", {"X"})
