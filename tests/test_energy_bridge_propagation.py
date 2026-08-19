"""L1 oracle 测试 — EnergyCounterfactualBridge 矩阵传播反事实。

验证桥接器接入矩阵动力学后的传播效应是否符合生克关系的方向性。
这是"接入"的核心验证: 守恒系统中, 反事实效应 = Flow·δ (单步传播),
而非两个终态的差 (守恒系统终态都趋均匀, 无区分度)。
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.oracle

from mci_world_model._sys._energy_core import EnergyCore
from mci_world_model.sdk._energy_counterfactual_bridge import EnergyCounterfactualBridge


@pytest.fixture
def bridge():
    return EnergyCounterfactualBridge(EnergyCore(), sim_steps=30)


class TestPropagationMode:
    """矩阵传播模式 (默认, 有区分度)。"""

    def test_propagation_has_distinct_effects(self, bridge):
        """传播模式应对不同维度给出不同 delta (有区分度)。"""
        results = bridge.what_if("semantic", boost=2.0, mode="propagation")
        deltas = [r.delta for r in results]
        # 不应全部相同 (区别于稳态模式的全 0.04)
        spread = max(deltas) - min(deltas)
        assert spread > 0.01, f"传播效应无区分度, delta spread={spread:.4f}, deltas={deltas}"

    def test_steady_state_mode_uniform_delta(self, bridge):
        """对照: 稳态模式因守恒而 delta 趋同 (无区分度)。"""
        results = bridge.what_if("semantic", boost=2.0, mode="steady_state")
        deltas = [r.delta for r in results]
        spread = max(deltas) - min(deltas)
        assert spread < 0.01, f"稳态模式应有均匀 delta, spread={spread}"

    def test_generation_direction_positive(self, bridge):
        """相生: 增强 src 应使其所生目标 (tgt) 增加。

        ENERGY_ENHANCE: semantic→causal, 增强 semantic 应使 causal delta>0。
        """
        results = {r.target_energy: r.delta for r in bridge.what_if("semantic", boost=2.0, mode="propagation")}
        # semantic 生 causal
        assert results["causal"] > 0.001, f"semantic生causal, 但 causal delta={results['causal']:.4f} 非正"

    def test_boost_reversal(self, bridge):
        """减弱 (boost<1) 应反转传播方向。"""
        up = {r.target_energy: r.delta for r in bridge.what_if("semantic", boost=2.0, mode="propagation")}
        down = {r.target_energy: r.delta for r in bridge.what_if("semantic", boost=0.5, mode="propagation")}
        # causal: 增强时 +, 减弱时 -
        assert up["causal"] > 0 and down["causal"] < 0, (
            f"causal 未反转: up={up['causal']:.4f} down={down['causal']:.4f}"
        )

    def test_different_sources_different_effects(self, bridge):
        """不同干预源应产生不同的效应模式 (可区分性)。"""
        sem = {r.target_energy: r.delta for r in bridge.what_if("semantic", boost=2.0, mode="propagation")}
        cau = {r.target_energy: r.delta for r in bridge.what_if("causal", boost=2.0, mode="propagation")}
        # semantic 生 causal, causal 生 spacetime — 效应应不同
        assert sem != cau, "不同干预源产生相同效应, 无可区分性"

    def test_effect_consistent_with_flow_matrix(self, bridge):
        """传播效应应等于 Flow @ delta_input (矩阵定义)。"""
        import numpy as np

        core = EnergyCore()
        F = core.flow_matrix()
        cats = core.ENERGY_ORDER
        idx = {c: i for i, c in enumerate(cats)}

        results = bridge.what_if("semantic", boost=2.0, mode="propagation")
        # 手算: baseline 均匀 0.2, semantic×2 后归一化
        base = np.array([0.2] * 5)
        do = base.copy()
        do[0] *= 2
        do = do / do.sum() * base.sum()
        delta_in = do - base
        expected = F @ delta_in

        for r in results:
            i = idx[r.target_energy]
            assert abs(r.delta - expected[i]) < 1e-9, (
                f"{r.target_energy}: bridge={r.delta:.6f} vs F·δ={expected[i]:.6f}"
            )


class TestMechanismClassification:
    """机制分类 (相生/相克/间接) 正确性。"""

    def test_generated_target_marked_enhance(self, bridge):
        """被生目标应标注"相生"。"""
        results = bridge.what_if("semantic", boost=2.0, mode="propagation")
        # semantic 生 causal
        causal_result = next(r for r in results if r.target_energy == "causal")
        assert causal_result.mechanism == "相生"

    def test_unknown_energy_raises(self, bridge):
        """未知能量类型应报错。"""
        with pytest.raises(ValueError):
            bridge.what_if("nonexistent", boost=2.0)
