"""
tests/test_energy_counterfactual_bridge.py — EnergyCounterfactualBridge 测试
============================================================================

覆盖:
    - what_if 单变量干预
    - batch_what_if 批量干预
    - systemic_impact 全系统分析
    - counterfactual_energy 反事实分布
    - 边界情况 (无效能量类型)
"""

from __future__ import annotations

import pytest

from mci_world_model._sys._energy_core import EnergyCore
from mci_world_model.sdk._energy_counterfactual_bridge import (
    EnergyCounterfactualBridge,
    EnergyWhatIfResult,
)


@pytest.fixture
def ec():
    return EnergyCore()


@pytest.fixture
def bridge(ec):
    return EnergyCounterfactualBridge(ec, sim_steps=20, seed=42)


@pytest.fixture
def unbalanced_state():
    return {
        "semantic": 0.4,
        "causal": 0.1,
        "spacetime": 0.2,
        "generative": 0.2,
        "trust": 0.1,
    }


# =============================================================================
# 基本功能
# =============================================================================


class TestEnergyBridgeBasic:
    def test_categories(self, bridge):
        """categories 返回五维列表。"""
        cats = bridge.categories
        assert len(cats) == 5
        assert "semantic" in cats

    def test_what_if_returns_list(self, bridge, unbalanced_state):
        """what_if 返回 4 个结果 (排除自身)。"""
        results = bridge.what_if("semantic", boost=1.5, baseline_energies=unbalanced_state)
        assert len(results) == 4
        for r in results:
            assert isinstance(r, EnergyWhatIfResult)
            assert r.target_energy != "semantic"

    def test_what_if_delta_nonzero(self, bridge, unbalanced_state):
        """不平衡状态下，what_if 产生非零 Δ。"""
        results = bridge.what_if("semantic", boost=2.0, baseline_energies=unbalanced_state)
        deltas = [abs(r.delta) for r in results]
        assert max(deltas) > 0.001

    def test_what_if_boost_one_no_effect(self, bridge, unbalanced_state):
        """boost=1.0 应无影响。"""
        results = bridge.what_if("semantic", boost=1.0, baseline_energies=unbalanced_state)
        for r in results:
            assert abs(r.delta) < 0.02  # 仿真噪声容差


# =============================================================================
# 批量 + 系统
# =============================================================================


class TestEnergyBridgeBatch:
    def test_batch_what_if(self, bridge, unbalanced_state):
        """批量干预返回正确结构。"""
        interventions = [
            {"energy": "semantic", "boost": 1.5},
            {"energy": "causal", "boost": 0.5},
        ]
        results = bridge.batch_what_if(interventions, baseline_energies=unbalanced_state)
        assert len(results) == 2
        for group in results:
            assert len(group) == 4

    def test_systemic_impact(self, bridge, unbalanced_state):
        """systemic_impact 返回完整映射。"""
        impact = bridge.systemic_impact("semantic", boost=2.0, baseline_energies=unbalanced_state)
        assert "intervention" in impact
        assert "affected" in impact
        assert len(impact["affected"]) == 4


# =============================================================================
# 反事实
# =============================================================================


class TestEnergyBridgeCounterfactual:
    def test_counterfactual_returns_dict(self, bridge, unbalanced_state):
        """counterfactual_energy 返回完整分布。"""
        cf = bridge.counterfactual_energy(unbalanced_state, do={"semantic": 0.8})
        assert isinstance(cf, dict)
        assert len(cf) == 5

    def test_counterfactual_changes_distribution(self, bridge, unbalanced_state):
        """do 干预改变分布。"""
        cf = bridge.counterfactual_energy(unbalanced_state, do={"semantic": 0.8})
        # 至少有一个维度与基线不同
        diffs = [abs(cf.get(k, 0) - unbalanced_state.get(k, 0)) for k in cf]
        assert max(diffs) > 0.001


# =============================================================================
# 边界情况
# =============================================================================


class TestEnergyBridgeEdgeCases:
    def test_invalid_energy_type(self, bridge):
        """无效能量类型抛出 ValueError。"""
        with pytest.raises(ValueError):
            bridge.what_if("invalid", boost=1.5)

    def test_default_baseline(self, bridge):
        """None baseline 应使用默认均衡。"""
        results = bridge.what_if("semantic", boost=1.5)
        assert len(results) == 4

    def test_empty_batch(self, bridge):
        """空批量返回空列表。"""
        assert bridge.batch_what_if([]) == []

    def test_what_if_result_properties(self):
        """EnergyWhatIfResult 属性正确。"""
        r = EnergyWhatIfResult(
            intervention="do(x=2)",
            target_energy="y",
            baseline=0.2,
            counterfactual=0.3,
            delta=0.1,
            mechanism="相生",
        )
        assert r.direction == "增强"
        assert r.effect_ratio == 0.5

    def test_no_effect_result(self):
        """无影响结果。"""
        r = EnergyWhatIfResult(
            intervention="do(x=1)",
            target_energy="y",
            baseline=0.2,
            counterfactual=0.2,
            delta=0.0,
            mechanism="相克",
        )
        assert r.direction == "无影响"
        assert r.effect_ratio == 0.0

    def test_direction_negative(self):
        """减弱方向。"""
        r = EnergyWhatIfResult(
            intervention="do(x=0.5)",
            target_energy="y",
            baseline=0.2,
            counterfactual=0.15,
            delta=-0.05,
            mechanism="相克",
        )
        assert r.direction == "减弱"
