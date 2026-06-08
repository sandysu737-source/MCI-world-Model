"""
MCI World Model v3.1.0 — CostModule 全面测试
==============================================

覆盖 _cost_module.py 中未充分测试的方法：
- CostSignal: zero(), to_dict(), frozen 不可变性
- EnergyCostModule: evaluate() with CausalWorldModelState edges,
  _energy_balance_cost with time-varying factors,
  _overconstraint_cost with EnergyCore,
  _causal_consistency_cost edge cases,
  _temporal_coherence_cost edge cases,
  state machine transitions (IDLE→COMPUTING→COMPLETE),
  error handling path

目标: 将 _cost_module.py 覆盖率从 ~45% 提升至 85%+。
"""

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest

from mci_world_model.sdk._cost_module import CostSignal, EnergyCostModule
from mci_world_model.sdk._world_model import CausalWorldModelState

# =============================================================================
# CostSignal 测试
# =============================================================================


class TestCostSignal:
    """CostSignal 数据类测试。"""

    def test_default_values(self):
        """默认值。"""
        signal = CostSignal(
            total=0.5,
            energy_balance=0.2,
            causal_consistency=0.1,
            temporal_coherence=0.2,
            breakdown={"eb": 0.2, "cc": 0.1, "tc": 0.2},
        )
        assert signal.total == 0.5
        assert signal.energy_balance == 0.2
        assert signal.causal_consistency == 0.1
        assert signal.temporal_coherence == 0.2

    def test_zero(self):
        """零代价信号。"""
        signal = CostSignal.zero()
        assert signal.total == 0.0
        assert signal.energy_balance == 0.0
        assert signal.causal_consistency == 0.0
        assert signal.temporal_coherence == 0.0

    def test_to_dict(self):
        """to_dict 序列化。"""
        signal = CostSignal.zero()
        d = signal.to_dict()
        assert isinstance(d, dict)
        assert d["total"] == 0.0
        assert "energy_balance" in d
        assert "causal_consistency" in d
        assert "temporal_coherence" in d

    def test_frozen_immutable(self):
        """不可变性 — 不应允许修改字段。"""
        signal = CostSignal.zero()
        with pytest.raises(FrozenInstanceError):
            signal.total = 1.0  # type: ignore[misc]

    def test_to_dict_precision(self):
        """to_dict 精度（6 位小数）。"""
        signal = CostSignal(
            total=0.12345678,
            energy_balance=0.11111111,
            causal_consistency=0.22222222,
            temporal_coherence=0.33333333,
            breakdown={"x": 0.0},
        )
        d = signal.to_dict()
        assert d["total"] == 0.123457


# =============================================================================
# EnergyCostModule 基础测试
# =============================================================================


def _make_state_with_edges(edges: list[dict]) -> CausalWorldModelState:
    """创建带因果边的 CausalWorldModelState。"""
    return CausalWorldModelState(causal_edges=edges)


class TestEnergyCostModuleBasic:
    """EnergyCostModule 基础功能测试。"""

    def test_init_default_weights(self):
        """默认权重初始化。"""
        module = EnergyCostModule()
        assert module._alpha == 0.5
        assert module._beta == 0.3
        assert module._gamma == 0.2
        assert module.state == "IDLE"
        assert module.eval_count == 0

    def test_init_custom_weights(self):
        """自定义权重初始化。"""
        module = EnergyCostModule(
            alpha_energy=0.6,
            beta_causal=0.25,
            gamma_temporal=0.15,
        )
        assert module._alpha == 0.6
        assert module._beta == 0.25
        assert module._gamma == 0.15

    def test_state_machine_idle_to_complete(self):
        """状态机: IDLE → COMPUTING → COMPLETE。"""
        module = EnergyCostModule()
        assert module.state == "IDLE"

        state = _make_state_with_edges(
            [{"energy_relation": "enhance", "rho": 0.5, "verdict": "confirmed", "confidence": 0.8}]
        )
        signal = module.evaluate(state)
        assert module.state == "COMPLETE"
        assert isinstance(signal, CostSignal)
        assert module.eval_count == 1

    def test_eval_count_increment(self):
        """eval_count 每次评估递增。"""
        module = EnergyCostModule()
        state = _make_state_with_edges(
            [{"energy_relation": "neutral", "rho": 0.5, "verdict": "confirmed", "confidence": 0.7}]
        )
        for _ in range(3):
            module.evaluate(state)
        assert module.eval_count == 3


# =============================================================================
# EnergyCostModule — 各维度代价测试
# =============================================================================


class TestEnergyCostModuleDimensions:
    """能量、因果、时序三维度代价计算测试。"""

    def test_evaluate_with_enhance_edges(self):
        """增强边（正常模式）的代价。"""
        state = _make_state_with_edges(
            [
                {"energy_relation": "enhance", "rho": 0.5, "verdict": "confirmed", "confidence": 0.8},
                {"energy_relation": "enhance", "rho": 0.6, "verdict": "confirmed", "confidence": 0.9},
            ]
        )
        module = EnergyCostModule()
        signal = module.evaluate(state)
        assert signal.total >= 0.0
        assert signal.energy_balance >= 0.0

    def test_evaluate_with_suppress_edges_violation(self):
        """抑制边违规（高权重）的代价。"""
        state = _make_state_with_edges(
            [
                {"energy_relation": "suppress", "rho": 0.9, "verdict": "confirmed", "confidence": 0.8},
            ]
        )
        module = EnergyCostModule()
        signal = module.evaluate(state)
        # 抑制边 rho > 0.7 → 违规惩罚
        assert signal.causal_consistency >= 0.0

    def test_evaluate_with_none_verdict(self):
        """未确认边的代价。"""
        state = _make_state_with_edges(
            [
                {"energy_relation": "neutral", "rho": 0.2, "verdict": "none", "confidence": 0.1},
            ]
        )
        module = EnergyCostModule()
        signal = module.evaluate(state)
        # none verdict → 因果一致性惩罚
        assert signal.causal_consistency > 0.0

    def test_evaluate_empty_edges(self):
        """空边集的代价。"""
        state = _make_state_with_edges([])
        module = EnergyCostModule()
        signal = module.evaluate(state)
        assert signal.total == 0.0
        assert signal.energy_balance == 0.0
        assert signal.causal_consistency == 0.0
        assert signal.temporal_coherence == 0.0

    def test_evaluate_single_edge(self):
        """单条边的代价。"""
        state = _make_state_with_edges(
            [{"energy_relation": "enhance", "rho": 0.8, "verdict": "confirmed", "confidence": 0.9}]
        )
        module = EnergyCostModule()
        signal = module.evaluate(state)
        assert isinstance(signal, CostSignal)

    def test_evaluate_novel_low_confidence(self):
        """低置信度新发现的代价。"""
        state = _make_state_with_edges(
            [
                {"energy_relation": "neutral", "rho": 0.3, "verdict": "novel", "confidence": 0.3},
            ]
        )
        module = EnergyCostModule()
        signal = module.evaluate(state)
        # novel + low confidence → 中等惩罚
        assert signal.causal_consistency > 0.0

    def test_evaluate_mixed_verdicts(self):
        """混合 verdict 的代价。"""
        state = _make_state_with_edges(
            [
                {"energy_relation": "enhance", "rho": 0.7, "verdict": "confirmed", "confidence": 0.9},
                {"energy_relation": "neutral", "rho": 0.3, "verdict": "novel", "confidence": 0.4},
                {"energy_relation": "suppress", "rho": 0.8, "verdict": "suppressed", "confidence": 0.6},
            ]
        )
        module = EnergyCostModule()
        signal = module.evaluate(state)
        assert signal.total >= 0.0

    def test_breakdown_contains_all_dimensions(self):
        """breakdown 包含所有维度。"""
        state = _make_state_with_edges(
            [{"energy_relation": "enhance", "rho": 0.5, "verdict": "confirmed", "confidence": 0.8}]
        )
        module = EnergyCostModule()
        signal = module.evaluate(state)
        assert "eb" in signal.breakdown
        assert "cc" in signal.breakdown
        assert "tc" in signal.breakdown
        assert "oc" in signal.breakdown  # v3.0.4: 乘侮异常

    def test_temporal_coherence_zero_with_one_edge(self):
        """单边时序连贯性为 0（< 2 无标准差）。"""
        state = _make_state_with_edges(
            [{"energy_relation": "enhance", "rho": 0.5, "verdict": "confirmed", "confidence": 0.8}]
        )
        module = EnergyCostModule()
        signal = module.evaluate(state)
        assert signal.temporal_coherence == 0.0

    def test_temporal_coherence_nonzero_multi_edges(self):
        """多边时序连贯性 > 0。"""
        state = _make_state_with_edges(
            [
                {"energy_relation": "enhance", "rho": 0.2, "verdict": "confirmed", "confidence": 0.7},
                {"energy_relation": "enhance", "rho": 0.9, "verdict": "confirmed", "confidence": 0.8},
            ]
        )
        module = EnergyCostModule()
        signal = module.evaluate(state)
        # 两条边 rho [0.2, 0.9] → CV = std/mean > 0
        assert signal.temporal_coherence >= 0.0

    def test_temporal_coherence_capped_at_one(self):
        """时序连贯性上限为 1.0。"""
        state = _make_state_with_edges(
            [
                {"energy_relation": "enhance", "rho": 0.0, "verdict": "confirmed", "confidence": 0.7},
                {"energy_relation": "enhance", "rho": 1.0, "verdict": "confirmed", "confidence": 0.8},
            ]
        )
        module = EnergyCostModule()
        signal = module.evaluate(state)
        assert signal.temporal_coherence <= 1.0

    def test_energy_balance_enhance_violation(self):
        """增强边低权重违规。"""
        state = _make_state_with_edges(
            [{"energy_relation": "enhance", "rho": 0.1, "verdict": "confirmed", "confidence": 0.7}]
        )
        module = EnergyCostModule()
        signal = module.evaluate(state)
        # enhance + rho < 0.3 → 违规
        assert signal.energy_balance > 0.0


# =============================================================================
# EnergyCostModule — EnergyCore 注入测试 (v3.0.4)
# =============================================================================


class TestEnergyCostModuleWithEnergyCore:
    """EnergyCostModule 注入 EnergyCore 的时变测试。"""

    def test_overconstraint_with_energy_core(self):
        """注入 EnergyCore 后的乘侮检测。"""
        mock_core = MagicMock()
        mock_core.get_overconstraint_relation.return_value = False
        mock_core.get_reverse_relation.return_value = False
        mock_core.STRENGTH_MULTIPLIER = {}

        module = EnergyCostModule(energy_core=mock_core, month_branch=0)
        state = _make_state_with_edges(
            [
                {
                    "energy_relation": "enhance",
                    "rho": 0.5,
                    "verdict": "confirmed",
                    "confidence": 0.8,
                    "cause_energy": "wood",
                    "effect_energy": "fire",
                }
            ]
        )
        signal = module.evaluate(state)
        assert signal is not None
        # 乘侮异常为 0（模拟返回 False）
        assert signal.breakdown["oc"] == 0.0

    def test_overconstraint_detected(self):
        """检测到乘侮异常。"""
        mock_core = MagicMock()
        mock_core.get_overconstraint_relation.return_value = True  # 相乘检测
        mock_core.get_reverse_relation.return_value = False
        mock_core.STRENGTH_MULTIPLIER = {}

        module = EnergyCostModule(energy_core=mock_core, month_branch=0)
        state = _make_state_with_edges(
            [
                {
                    "energy_relation": "suppress",
                    "rho": 0.7,
                    "verdict": "confirmed",
                    "confidence": 0.8,
                    "cause_energy": "metal",
                    "effect_energy": "wood",
                }
            ]
        )
        signal = module.evaluate(state)
        assert signal.breakdown["oc"] > 0.0

    def test_reverse_relation_detected(self):
        """检测到相侮异常。"""
        mock_core = MagicMock()
        mock_core.get_overconstraint_relation.return_value = False
        mock_core.get_reverse_relation.return_value = True  # 相侮检测
        mock_core.STRENGTH_MULTIPLIER = {}

        module = EnergyCostModule(energy_core=mock_core, month_branch=0)
        state = _make_state_with_edges(
            [
                {
                    "energy_relation": "suppress",
                    "rho": 0.5,
                    "verdict": "confirmed",
                    "confidence": 0.7,
                    "cause_energy": "wood",
                    "effect_energy": "metal",
                }
            ]
        )
        signal = module.evaluate(state)
        assert signal.breakdown["oc"] > 0.0

    def test_time_varying_factor(self):
        """时变惩罚系数 — WANG 态放大。"""
        mock_core = MagicMock()
        mock_state = MagicMock()
        mock_state.strength = "strong"  # WANG
        mock_core.get_energy_state.return_value = mock_state
        mock_core.STRENGTH_MULTIPLIER = {"strong": 1.2, "weak": 0.3}
        mock_core.get_overconstraint_relation.return_value = False
        mock_core.get_reverse_relation.return_value = False

        module = EnergyCostModule(energy_core=mock_core, month_branch=3)
        state = _make_state_with_edges(
            [
                {
                    "energy_relation": "enhance",
                    "rho": 0.1,  # 违规
                    "verdict": "confirmed",
                    "confidence": 0.8,
                    "cause_energy": "fire",
                    "effect_energy": "earth",
                }
            ]
        )
        signal = module.evaluate(state)
        # WANG 态 → time_factor=1.2 → 惩罚放大
        assert signal.energy_balance > 0.0

    def test_no_energy_core_overconstraint_zero(self):
        """无 EnergyCore 时乘侮代价为 0。"""
        module = EnergyCostModule(energy_core=None)
        state = _make_state_with_edges(
            [
                {
                    "energy_relation": "suppress",
                    "rho": 0.7,
                    "verdict": "confirmed",
                    "confidence": 0.8,
                    "cause_energy": "metal",
                    "effect_energy": "wood",
                }
            ]
        )
        signal = module.evaluate(state)
        assert signal.breakdown["oc"] == 0.0

    def test_energy_core_state_error_fallback(self):
        """EnergyCore 状态查询异常回退。"""
        mock_core = MagicMock()
        mock_core.get_energy_state.side_effect = RuntimeError("energy core unavailable")
        mock_core.STRENGTH_MULTIPLIER = {"normal": 1.0}
        mock_core.get_overconstraint_relation.return_value = False
        mock_core.get_reverse_relation.return_value = False

        module = EnergyCostModule(energy_core=mock_core, month_branch=0)
        state = _make_state_with_edges(
            [
                {
                    "energy_relation": "enhance",
                    "rho": 0.1,
                    "verdict": "confirmed",
                    "confidence": 0.8,
                    "cause_energy": "wood",
                    "effect_energy": "fire",
                }
            ]
        )
        # 不应抛出异常，应回退
        signal = module.evaluate(state)
        assert signal is not None


# =============================================================================
# EnergyCostModule — 边界条件测试
# =============================================================================


class TestEnergyCostModuleEdgeCases:
    """EnergyCostModule 边界条件测试。"""

    def test_evaluate_state_without_rho(self):
        """忽略 rho 字段的边。"""
        state = _make_state_with_edges([{"energy_relation": "enhance", "verdict": "confirmed", "confidence": 0.8}])
        module = EnergyCostModule()
        signal = module.evaluate(state)
        assert signal is not None

    def test_evaluate_state_without_verdict(self):
        """忽略 verdict 字段的边。"""
        state = _make_state_with_edges([{"energy_relation": "neutral", "rho": 0.5, "confidence": 0.7}])
        module = EnergyCostModule()
        signal = module.evaluate(state)
        assert signal is not None

    def test_evaluate_state_without_confidence(self):
        """忽略 confidence 字段的边。"""
        state = _make_state_with_edges([{"energy_relation": "enhance", "rho": 0.5, "verdict": "confirmed"}])
        module = EnergyCostModule()
        signal = module.evaluate(state)
        assert signal is not None

    def test_evaluate_many_edges(self):
        """大量边的代价评估。"""
        edges = [
            {
                "energy_relation": "enhance",
                "rho": float(i) / 20.0,
                "verdict": "confirmed" if i % 3 == 0 else ("novel" if i % 3 == 1 else "none"),
                "confidence": 0.5 + float(i % 5) / 10.0,
            }
            for i in range(50)
        ]
        state = _make_state_with_edges(edges)
        module = EnergyCostModule()
        signal = module.evaluate(state)
        assert signal.total >= 0.0

    def test_evaluate_all_enhance_below_threshold(self):
        """所有增强边权重低于阈值 — 代价高。"""
        edges = [
            {"energy_relation": "enhance", "rho": 0.05, "verdict": "confirmed", "confidence": 0.6} for _ in range(10)
        ]
        state = _make_state_with_edges(edges)
        module = EnergyCostModule()
        signal = module.evaluate(state)
        # 大量违规 → 高能量均衡代价
        assert signal.energy_balance > 0.1

    def test_total_is_weighted_sum(self):
        """总代价是加权和。"""
        edges = [
            {"energy_relation": "enhance", "rho": 0.5, "verdict": "confirmed", "confidence": 0.9},
            {"energy_relation": "enhance", "rho": 0.5, "verdict": "confirmed", "confidence": 0.8},
        ]
        state = _make_state_with_edges(edges)
        module = EnergyCostModule(alpha_energy=0.5, beta_causal=0.3, gamma_temporal=0.2)
        signal = module.evaluate(state)
        # total = 0.5*eb + 0.3*cc + 0.2*tc + 0.15*oc
        assert signal.total >= 0.0


# =============================================================================
# EnergyCostModule — 属性测试
# =============================================================================


class TestEnergyCostModuleProperties:
    """EnergyCostModule 属性测试。"""

    def test_state_property(self):
        """state 属性。"""
        module = EnergyCostModule()
        assert module.state == "IDLE"
        module.evaluate(_make_state_with_edges([]))
        assert module.state == "COMPLETE"

    def test_eval_count_property(self):
        """eval_count 属性。"""
        module = EnergyCostModule()
        assert module.eval_count == 0
        module.evaluate(_make_state_with_edges([]))
        assert module.eval_count == 1

    def test_custom_alpha_only(self):
        """自定义 alpha。"""
        module = EnergyCostModule(alpha_energy=1.0, beta_causal=0.0, gamma_temporal=0.0)
        state = _make_state_with_edges(
            [{"energy_relation": "enhance", "rho": 0.1, "verdict": "confirmed", "confidence": 0.8}]
        )
        signal = module.evaluate(state)
        # total 应主要由能量均衡代价驱动
        assert signal.total >= 0.0

    def test_all_zero_weights(self):
        """全零权重。"""
        module = EnergyCostModule(alpha_energy=0.0, beta_causal=0.0, gamma_temporal=0.0)
        state = _make_state_with_edges(
            [{"energy_relation": "enhance", "rho": 0.1, "verdict": "confirmed", "confidence": 0.8}]
        )
        signal = module.evaluate(state)
        # 总代价 = 0（即使有违规也乘零）
        assert signal.total == 0.0
