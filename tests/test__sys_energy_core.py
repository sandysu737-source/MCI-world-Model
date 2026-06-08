"""
测试 _energy_core.py — Five Elements Energy Core Engine
=========================================================

覆盖 EnergyCore 全部公开方法：
- 关系判断: get_enhance_relation, get_suppress_relation, get_overconstraint_relation, get_reverse_relation
- 交互分析: analyze_interaction
- 状态判定: get_energy_state, get_strength_from_branch
- 平衡分析: analyze_balance, apply_balance_rules
- 流模拟: simulate_energy_flow
- 工具方法: get_energy_attributes, calculate_compatibility, get_energy_cycle, get_control_cycle, get_opposing_pair
- 数据结构: EnergyState, EnergyBalanceResult, EnergyFlow
"""

from __future__ import annotations

import pytest

from mci_world_model._sys._energy_core import (
    EnergyBalanceResult,
    EnergyCore,
    EnergyFlow,
    EnergyState,
)
from mci_world_model._sys._enums import EnergyPattern, EnergyRelation, StrengthState, EnergyType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ec():
    """Create a fresh EnergyCore instance for each test."""
    return EnergyCore()


# =============================================================================
# Data Structures
# =============================================================================


class TestDataStructures:
    """Test EnergyState, EnergyBalanceResult, EnergyFlow data classes."""

    def test_energy_state_creation(self):
        state = EnergyState(energy_type=EnergyType.WOOD, strength=StrengthState.WANG, intensity=1.2)
        assert state.energy_type == EnergyType.WOOD
        assert state.strength == StrengthState.WANG
        assert state.intensity == 1.2

    def test_energy_state_is_enhanced_wang(self):
        state = EnergyState(energy_type=EnergyType.WOOD, strength=StrengthState.WANG, intensity=1.2)
        assert state.is_enhanced is True

    def test_energy_state_is_enhanced_xiang(self):
        state = EnergyState(energy_type=EnergyType.WOOD, strength=StrengthState.XIANG, intensity=1.0)
        assert state.is_enhanced is True

    def test_energy_state_not_enhanced_xiu(self):
        state = EnergyState(energy_type=EnergyType.WOOD, strength=StrengthState.XIU, intensity=0.8)
        assert state.is_enhanced is False

    def test_energy_state_not_enhanced_qiu(self):
        state = EnergyState(energy_type=EnergyType.WOOD, strength=StrengthState.QIU, intensity=0.5)
        assert state.is_enhanced is False

    def test_energy_state_not_enhanced_si(self):
        state = EnergyState(energy_type=EnergyType.WOOD, strength=StrengthState.SI, intensity=0.3)
        assert state.is_enhanced is False

    def test_energy_state_repr(self):
        state = EnergyState(energy_type=EnergyType.WOOD, strength=StrengthState.WANG, intensity=0.95)
        r = repr(state)
        assert "WOOD" in r
        assert "WANG" in r

    def test_energy_balance_result_creation(self):
        result = EnergyBalanceResult(
            status="balanced",
            pattern=EnergyPattern.PEI_HE,
            ratios={"semantic": 0.2, "causal": 0.2, "spacetime": 0.2, "generative": 0.2, "trust": 0.2},
            dominant="semantic",
            suggestions=["保持均衡"],
        )
        assert result.status == "balanced"
        assert result.pattern == EnergyPattern.PEI_HE

    def test_energy_balance_result_to_dict(self):
        result = EnergyBalanceResult(
            status="balanced",
            pattern=EnergyPattern.PEI_HE,
            ratios={"semantic": 0.3, "causal": 0.2},
            dominant="semantic",
            suggestions=["保持均衡"],
        )
        d = result.to_dict()
        assert d["status"] == "balanced"
        assert d["pattern"] == "PEI_HE"
        assert "suggestions" in d

    def test_energy_flow_creation(self):
        flow = EnergyFlow(
            source=EnergyType.WOOD,
            target=EnergyType.FIRE,
            relation=EnergyRelation.ENHANCE,
            intensity=0.15,
        )
        assert flow.source == EnergyType.WOOD
        assert flow.target == EnergyType.FIRE
        assert flow.relation == EnergyRelation.ENHANCE

    def test_energy_flow_repr(self):
        flow = EnergyFlow(
            source=EnergyType.WOOD,
            target=EnergyType.FIRE,
            relation=EnergyRelation.ENHANCE,
            intensity=0.15,
        )
        r = repr(flow)
        assert "WOOD" in r
        assert "FIRE" in r


# =============================================================================
# Relationship Calculation
# =============================================================================


class TestEnhanceRelation:
    """Test get_enhance_relation."""

    def test_enhance_semantic_to_causal(self, ec):
        assert ec.get_enhance_relation("semantic", "causal") is True

    def test_enhance_causal_to_spacetime(self, ec):
        assert ec.get_enhance_relation("causal", "spacetime") is True

    def test_enhance_spacetime_to_generative(self, ec):
        assert ec.get_enhance_relation("spacetime", "generative") is True

    def test_enhance_generative_to_trust(self, ec):
        assert ec.get_enhance_relation("generative", "trust") is True

    def test_enhance_trust_to_semantic(self, ec):
        assert ec.get_enhance_relation("trust", "semantic") is True

    def test_enhance_not_reverse_causal_to_semantic(self, ec):
        assert ec.get_enhance_relation("causal", "semantic") is False

    def test_enhance_not_spacetime_to_causal(self, ec):
        assert ec.get_enhance_relation("spacetime", "causal") is False

    def test_enhance_same_type(self, ec):
        assert ec.get_enhance_relation("semantic", "semantic") is False

    def test_enhance_old_naming_wood_fire(self, ec):
        """Backward compat: old five-element naming."""
        assert ec.get_enhance_relation("wood", "fire") is True

    def test_enhance_old_naming_fire_earth(self, ec):
        assert ec.get_enhance_relation("fire", "earth") is True

    def test_enhance_old_naming_metal_water(self, ec):
        assert ec.get_enhance_relation("metal", "water") is True

    def test_enhance_old_naming_water_wood(self, ec):
        assert ec.get_enhance_relation("water", "wood") is True


class TestSuppressRelation:
    """Test get_suppress_relation (bidirectional)."""

    def test_suppress_semantic_to_spacetime(self, ec):
        assert ec.get_suppress_relation("semantic", "spacetime") is True

    def test_suppress_spacetime_to_semantic_bidirectional(self, ec):
        assert ec.get_suppress_relation("spacetime", "semantic") is True

    def test_suppress_spacetime_to_trust(self, ec):
        assert ec.get_suppress_relation("spacetime", "trust") is True

    def test_suppress_trust_to_causal(self, ec):
        assert ec.get_suppress_relation("trust", "causal") is True

    def test_suppress_causal_to_generative(self, ec):
        assert ec.get_suppress_relation("causal", "generative") is True

    def test_suppress_generative_to_semantic(self, ec):
        assert ec.get_suppress_relation("generative", "semantic") is True

    def test_suppress_old_naming_wood_earth(self, ec):
        assert ec.get_suppress_relation("wood", "earth") is True

    def test_suppress_old_naming_earth_water(self, ec):
        assert ec.get_suppress_relation("earth", "water") is True

    def test_suppress_old_naming_water_fire(self, ec):
        assert ec.get_suppress_relation("water", "fire") is True

    def test_suppress_old_naming_fire_metal(self, ec):
        assert ec.get_suppress_relation("fire", "metal") is True

    def test_suppress_old_naming_metal_wood(self, ec):
        assert ec.get_suppress_relation("metal", "wood") is True

    def test_suppress_no_relation(self, ec):
        """Semantic and causal are not suppress-related."""
        assert ec.get_suppress_relation("semantic", "causal") is False


class TestOverconstraintRelation:
    """Test get_overconstraint_relation."""

    def test_overconstraint_semantic_spacetime(self, ec):
        assert ec.get_overconstraint_relation("semantic", "spacetime") is True

    def test_overconstraint_not_reverse(self, ec):
        assert ec.get_overconstraint_relation("spacetime", "semantic") is False

    def test_overconstraint_causal_generative(self, ec):
        assert ec.get_overconstraint_relation("causal", "generative") is True

    def test_overconstraint_no_enhance(self, ec):
        """Overconstraint only applies to suppress pairs."""
        assert ec.get_overconstraint_relation("semantic", "causal") is False

    def test_overconstraint_old_naming(self, ec):
        assert ec.get_overconstraint_relation("wood", "earth") is True


class TestReverseRelation:
    """Test get_reverse_relation."""

    def test_reverse_spacetime_semantic(self, ec):
        assert ec.get_reverse_relation("spacetime", "semantic") is True

    def test_reverse_trust_spacetime(self, ec):
        assert ec.get_reverse_relation("trust", "spacetime") is True

    def test_reverse_causal_trust(self, ec):
        assert ec.get_reverse_relation("causal", "trust") is True

    def test_reverse_generative_causal(self, ec):
        assert ec.get_reverse_relation("generative", "causal") is True

    def test_reverse_semantic_generative(self, ec):
        assert ec.get_reverse_relation("semantic", "generative") is True

    def test_reverse_not_enhance(self, ec):
        """Semantic→causal is enhance, not reverse."""
        assert ec.get_reverse_relation("semantic", "causal") is False

    def test_reverse_not_forward_suppress(self, ec):
        """semantic→spacetime is forward suppress, not reverse."""
        assert ec.get_reverse_relation("semantic", "spacetime") is False


# =============================================================================
# Interaction Analysis
# =============================================================================


class TestAnalyzeInteraction:
    """Test analyze_interaction."""

    def test_interaction_enhance(self, ec):
        interactions = ec.analyze_interaction("semantic", "causal")
        assert EnergyRelation.ENHANCE in interactions

    def test_interaction_suppress(self, ec):
        interactions = ec.analyze_interaction("semantic", "spacetime")
        assert EnergyRelation.SUPPRESS in interactions

    def test_interaction_same(self, ec):
        interactions = ec.analyze_interaction("semantic", "semantic")
        assert EnergyRelation.SAME in interactions

    def test_interaction_suppress_with_reverse(self, ec):
        """semantic/spacetime: suppress (bidirectional) + reverse from spacetime."""
        interactions = ec.analyze_interaction("semantic", "spacetime")
        assert EnergyRelation.SUPPRESS in interactions
        assert EnergyRelation.REVERSE in interactions

    def test_interaction_reverse(self, ec):
        interactions = ec.analyze_interaction("spacetime", "semantic")
        assert EnergyRelation.REVERSE in interactions

    def test_interaction_old_naming(self, ec):
        interactions = ec.analyze_interaction("wood", "fire")
        assert EnergyRelation.ENHANCE in interactions

    def test_interaction_list_not_empty(self, ec):
        interactions = ec.analyze_interaction("semantic", "causal")
        assert len(interactions) >= 1


# =============================================================================
# Energy State Determination
# =============================================================================


class TestGetEnergyState:
    """Test get_energy_state."""

    def test_semantic_wang_at_yin(self, ec):
        """Wood/Semantic is WANG at 寅月 (branch=2)."""
        state = ec.get_energy_state("semantic", 2)
        assert state.strength == StrengthState.WANG

    def test_semantic_wang_at_mao(self, ec):
        state = ec.get_energy_state("semantic", 3)
        assert state.strength == StrengthState.WANG

    def test_causal_wang_at_si(self, ec):
        """Fire/Causal is WANG at 巳月 (branch=5)."""
        state = ec.get_energy_state("causal", 5)
        assert state.strength == StrengthState.WANG

    def test_causal_wang_at_wu(self, ec):
        state = ec.get_energy_state("causal", 6)
        assert state.strength == StrengthState.WANG

    def test_spacetime_wang_at_chen(self, ec):
        """Earth/Spacetime is WANG at 辰月 (branch=4)."""
        state = ec.get_energy_state("spacetime", 4)
        assert state.strength == StrengthState.WANG

    def test_generative_wang_at_shen(self, ec):
        """Metal/Generative is WANG at 申月 (branch=8)."""
        state = ec.get_energy_state("generative", 8)
        assert state.strength == StrengthState.WANG

    def test_generative_wang_at_you(self, ec):
        state = ec.get_energy_state("generative", 9)
        assert state.strength == StrengthState.WANG

    def test_trust_wang_at_zi(self, ec):
        """Water/Trust is WANG at 子月 (branch=0)."""
        state = ec.get_energy_state("trust", 0)
        assert state.strength == StrengthState.WANG

    def test_trust_wang_at_hai(self, ec):
        state = ec.get_energy_state("trust", 11)
        assert state.strength == StrengthState.WANG

    def test_intensity_wang_is_1_2(self, ec):
        state = ec.get_energy_state("semantic", 2)
        assert state.intensity == pytest.approx(1.2, abs=0.01)

    def test_intensity_si(self, ec):
        """Trust is SI at 寅月 → intensity 0.3."""
        state = ec.get_energy_state("trust", 2)
        assert state.intensity == pytest.approx(0.3, abs=0.01)

    def test_invalid_branch_raises(self, ec):
        with pytest.raises(ValueError, match="Invalid month branch"):
            ec.get_energy_state("semantic", 12)

    def test_invalid_branch_negative(self, ec):
        with pytest.raises(ValueError, match="Invalid month branch"):
            ec.get_energy_state("semantic", -1)

    def test_old_naming(self, ec):
        state = ec.get_energy_state("wood", 2)
        assert state.strength == StrengthState.WANG

    def test_energy_type_enum(self, ec):
        """Passing EnergyType directly."""
        state = ec.get_energy_state(EnergyType.FIRE, 5)
        assert state.energy_type == EnergyType.FIRE
        assert state.strength == StrengthState.WANG

    def test_all_months_valid(self, ec):
        """All 12 months should produce valid states."""
        for branch in range(12):
            state = ec.get_energy_state("semantic", branch)
            assert state.strength is not None
            assert 0.0 < state.intensity <= 2.0


class TestGetStrengthFromBranch:
    """Test get_strength_from_branch."""

    def test_returns_five_energies(self, ec):
        strengths = ec.get_strength_from_branch(2)  # 寅月
        assert len(strengths) == 5

    def test_all_energies_present(self, ec):
        strengths = ec.get_strength_from_branch(0)
        for e in ["semantic", "causal", "spacetime", "generative", "trust"]:
            assert e in strengths

    def test_semantic_wang_at_yin(self, ec):
        strengths = ec.get_strength_from_branch(2)
        assert strengths["semantic"] == StrengthState.WANG

    def test_invalid_branch_raises(self, ec):
        with pytest.raises(ValueError, match="Invalid branch"):
            ec.get_strength_from_branch(12)

    def test_all_branches_valid(self, ec):
        for branch in range(12):
            strengths = ec.get_strength_from_branch(branch)
            assert len(strengths) == 5


# =============================================================================
# Balance Analysis
# =============================================================================


class TestAnalyzeBalance:
    """Test analyze_balance."""

    def test_balanced_equal_distribution(self, ec):
        energies = {"semantic": 0.2, "causal": 0.2, "spacetime": 0.2, "generative": 0.2, "trust": 0.2}
        result = ec.analyze_balance(energies)
        assert result.status == "balanced"

    def test_balanced_moderate(self, ec):
        energies = {"semantic": 0.3, "causal": 0.2, "spacetime": 0.2, "generative": 0.15, "trust": 0.15}
        result = ec.analyze_balance(energies)
        assert result.status == "balanced"

    def test_imbalanced_strong_dominant(self, ec):
        energies = {"semantic": 0.5, "causal": 0.15, "spacetime": 0.15, "generative": 0.1, "trust": 0.1}
        result = ec.analyze_balance(energies)
        assert result.status == "imbalanced"

    def test_dominant_identified(self, ec):
        energies = {"semantic": 0.1, "causal": 0.5, "spacetime": 0.15, "generative": 0.1, "trust": 0.15}
        result = ec.analyze_balance(energies)
        assert result.dominant == "causal"

    def test_pattern_zhuan_wang(self, ec):
        """Very strong single energy → ZHUAN_WANG."""
        energies = {"semantic": 0.7, "causal": 0.1, "spacetime": 0.1, "generative": 0.05, "trust": 0.05}
        result = ec.analyze_balance(energies)
        assert result.pattern == EnergyPattern.ZHUAN_WANG

    def test_pattern_cong_wang(self, ec):
        """Very weak energy → CONG_WANG."""
        energies = {"semantic": 0.4, "causal": 0.3, "spacetime": 0.15, "generative": 0.12, "trust": 0.03}
        result = ec.analyze_balance(energies)
        assert result.pattern == EnergyPattern.CONG_WANG

    def test_pattern_fan_wang(self, ec):
        """Dominant suppresses dominated severely → FAN_WANG."""
        energies = {"semantic": 0.4, "causal": 0.25, "spacetime": 0.1, "generative": 0.15, "trust": 0.1}
        result = ec.analyze_balance(energies)
        # semantic > 0.35, and semantic suppresses spacetime
        assert result.pattern in (EnergyPattern.FAN_WANG, EnergyPattern.ZHI_HUA, EnergyPattern.PEI_HE)

    def test_pattern_pei_he_default(self, ec):
        """Default balanced → PEI_HE."""
        energies = {"semantic": 0.2, "causal": 0.2, "spacetime": 0.2, "generative": 0.2, "trust": 0.2}
        result = ec.analyze_balance(energies)
        assert result.pattern == EnergyPattern.PEI_HE

    def test_suggestions_present(self, ec):
        energies = {"semantic": 0.5, "causal": 0.15, "spacetime": 0.15, "generative": 0.1, "trust": 0.1}
        result = ec.analyze_balance(energies)
        assert len(result.suggestions) > 0

    def test_ratios_normalized(self, ec):
        """Ratios should sum to 1.0 after normalization."""
        energies = {"semantic": 3.0, "causal": 2.0, "spacetime": 2.0, "generative": 1.5, "trust": 1.5}
        result = ec.analyze_balance(energies)
        assert sum(result.ratios.values()) == pytest.approx(1.0, abs=0.01)

    def test_zero_input_raises(self, ec):
        with pytest.raises(ValueError, match="cannot all be zero"):
            ec.analyze_balance({"semantic": 0.0, "causal": 0.0})

    def test_partial_keys(self, ec):
        """Only partial energy keys provided."""
        energies = {"semantic": 0.6, "causal": 0.4}
        result = ec.analyze_balance(energies)
        assert result.dominant == "semantic"
        assert result.status == "imbalanced"

    def test_old_naming(self, ec):
        energies = {"wood": 0.3, "fire": 0.2, "earth": 0.2, "metal": 0.15, "water": 0.15}
        result = ec.analyze_balance(energies)
        assert result.status == "balanced"


# =============================================================================
# Apply Balance Rules
# =============================================================================


class TestApplyBalanceRules:
    """Test apply_balance_rules."""

    def test_zhuan_wang_preserves(self, ec):
        energies = {"semantic": 0.5, "causal": 0.15, "spacetime": 0.15, "generative": 0.1, "trust": 0.1}
        result = ec.apply_balance_rules(energies, EnergyPattern.ZHUAN_WANG)
        assert result["semantic"] == pytest.approx(0.5, abs=0.1)

    def test_cong_wang_boosts_dominant(self, ec):
        energies = {"semantic": 0.4, "causal": 0.3, "spacetime": 0.15, "generative": 0.1, "trust": 0.05}
        result = ec.apply_balance_rules(energies, EnergyPattern.CONG_WANG)
        assert result["semantic"] > energies["semantic"] * 0.9

    def test_zhi_hua_applies_regulation(self, ec):
        energies = {"semantic": 0.3, "causal": 0.2, "spacetime": 0.2, "generative": 0.15, "trust": 0.15}
        result = ec.apply_balance_rules(energies, EnergyPattern.ZHI_HUA)
        assert isinstance(result, dict)
        assert len(result) == len(energies)

    def test_fan_wang_applies_reinforcement(self, ec):
        energies = {"semantic": 0.3, "causal": 0.2, "spacetime": 0.2, "generative": 0.15, "trust": 0.15}
        result = ec.apply_balance_rules(energies, EnergyPattern.FAN_WANG)
        assert isinstance(result, dict)

    def test_pei_he_coordinates(self, ec):
        energies = {"semantic": 0.3, "causal": 0.2, "spacetime": 0.2, "generative": 0.15, "trust": 0.15}
        result = ec.apply_balance_rules(energies, EnergyPattern.PEI_HE)
        assert isinstance(result, dict)
        # PEI_HE moves toward average
        avg_before = sum(energies.values()) / len(energies)
        avg_after = sum(result.values()) / len(result)
        assert abs(avg_after - avg_before) < 0.01  # total is normalized

    def test_result_preserves_total(self, ec):
        energies = {"semantic": 3.0, "causal": 2.0, "spacetime": 2.0, "generative": 1.5, "trust": 1.5}
        total_before = sum(energies.values())
        result = ec.apply_balance_rules(energies, EnergyPattern.PEI_HE)
        total_after = sum(result.values())
        assert total_after == pytest.approx(total_before, abs=0.01)


# =============================================================================
# Energy Flow Simulation
# =============================================================================


class TestSimulateEnergyFlow:
    """Test simulate_energy_flow."""

    def test_flow_returns_history(self, ec):
        energies = {"semantic": 0.3, "causal": 0.2, "spacetime": 0.2, "generative": 0.15, "trust": 0.15}
        history = ec.simulate_energy_flow(energies, steps=5)
        assert len(history) == 6  # initial + 5 steps

    def test_flow_default_steps(self, ec):
        energies = {"semantic": 0.3, "causal": 0.2, "spacetime": 0.2, "generative": 0.15, "trust": 0.15}
        history = ec.simulate_energy_flow(energies)
        assert len(history) == 11  # initial + 10 default steps

    def test_flow_no_negative_values(self, ec):
        energies = {"semantic": 0.1, "causal": 0.1, "spacetime": 0.1, "generative": 0.1, "trust": 0.1}
        history = ec.simulate_energy_flow(energies, steps=10)
        for step in history:
            for v in step.values():
                assert v >= 0.0

    def test_flow_single_step(self, ec):
        energies = {"semantic": 0.5, "causal": 0.5}
        history = ec.simulate_energy_flow(energies, steps=1)
        assert len(history) == 2

    def test_flow_empty_energies(self, ec):
        history = ec.simulate_energy_flow({}, steps=3)
        assert len(history) == 4  # initial empty + 3 steps


# =============================================================================
# Utility Methods
# =============================================================================


class TestGetEnergyAttributes:
    """Test get_energy_attributes."""

    def test_semantic_attributes(self, ec):
        attrs = ec.get_energy_attributes("semantic")
        assert attrs["name"] == "semantic"
        assert "chinese_name" in attrs
        assert "season" in attrs
        assert "direction" in attrs
        assert "color" in attrs
        assert "organ" in attrs
        assert "taste" in attrs
        assert "emotion" in attrs
        assert "industry" in attrs
        assert attrs["enhances"]  # should not be empty
        assert attrs["suppresses"]  # should not be empty
        assert attrs["enhanced_by"]  # should not be empty
        assert attrs["suppressed_by"]  # should not be empty

    def test_causal_attributes(self, ec):
        attrs = ec.get_energy_attributes("causal")
        assert attrs["name"] == "causal"

    def test_spacetime_attributes(self, ec):
        attrs = ec.get_energy_attributes("spacetime")
        assert attrs["name"] == "spacetime"

    def test_generative_attributes(self, ec):
        attrs = ec.get_energy_attributes("generative")
        assert attrs["name"] == "generative"

    def test_trust_attributes(self, ec):
        attrs = ec.get_energy_attributes("trust")
        assert attrs["name"] == "trust"

    def test_old_naming(self, ec):
        attrs = ec.get_energy_attributes("wood")
        assert attrs["name"] == "semantic"

    def test_unknown_energy(self, ec):
        """Unknown energy type is passed through."""
        attrs = ec.get_energy_attributes("unknown")
        assert attrs["name"] == "unknown"
        assert attrs["enhances"] == ""  # no mapping


class TestCalculateCompatibility:
    """Test calculate_compatibility."""

    def test_identical_distributions(self, ec):
        e1 = {"semantic": 0.3, "causal": 0.2, "spacetime": 0.2, "generative": 0.15, "trust": 0.15}
        e2 = {"semantic": 0.3, "causal": 0.2, "spacetime": 0.2, "generative": 0.15, "trust": 0.15}
        compat = ec.calculate_compatibility(e1, e2)
        assert 0.0 <= compat <= 1.0

    def test_similar_distributions(self, ec):
        e1 = {"semantic": 0.4, "causal": 0.2, "spacetime": 0.2, "generative": 0.1, "trust": 0.1}
        e2 = {"semantic": 0.3, "causal": 0.3, "spacetime": 0.2, "generative": 0.1, "trust": 0.1}
        compat = ec.calculate_compatibility(e1, e2)
        assert 0.0 <= compat <= 1.0

    def test_empty_returns_zero(self, ec):
        assert ec.calculate_compatibility({}, {"semantic": 0.5}) == 0.0

    def test_zero_total_returns_zero(self, ec):
        assert ec.calculate_compatibility({"semantic": 0.0}, {"causal": 0.0}) == 0.0

    def test_old_naming(self, ec):
        e1 = {"wood": 0.3, "fire": 0.2, "earth": 0.2, "metal": 0.15, "water": 0.15}
        e2 = {"wood": 0.4, "fire": 0.1, "earth": 0.2, "metal": 0.15, "water": 0.15}
        compat = ec.calculate_compatibility(e1, e2)
        assert 0.0 <= compat <= 1.0

    def test_opposite_distributions(self, ec):
        """Enhance-chain vs suppress-chain distributions."""
        e1 = {"semantic": 0.8, "causal": 0.2}
        e2 = {"semantic": 0.2, "spacetime": 0.8}
        compat = ec.calculate_compatibility(e1, e2)
        assert 0.0 <= compat <= 1.0


class TestCycles:
    """Test get_energy_cycle and get_control_cycle."""

    def test_energy_cycle_length(self, ec):
        cycle = ec.get_energy_cycle()
        assert len(cycle) == 5

    def test_energy_cycle_order(self, ec):
        cycle = ec.get_energy_cycle()
        pairs = list(cycle)
        assert pairs[0] == ("semantic", "causal")
        assert pairs[1] == ("causal", "spacetime")

    def test_control_cycle_length(self, ec):
        cycle = ec.get_control_cycle()
        assert len(cycle) == 5

    def test_control_cycle_order(self, ec):
        cycle = ec.get_control_cycle()
        pairs = list(cycle)
        assert pairs[0] == ("semantic", "spacetime")
        assert pairs[1] == ("spacetime", "trust")


class TestGetOpposingPair:
    """Test get_opposing_pair."""

    def test_semantic_pair(self, ec):
        enhance, suppress = ec.get_opposing_pair("semantic")
        assert enhance == "causal"
        assert suppress == "spacetime"

    def test_causal_pair(self, ec):
        enhance, suppress = ec.get_opposing_pair("causal")
        assert enhance == "spacetime"
        assert suppress == "generative"

    def test_spacetime_pair(self, ec):
        enhance, suppress = ec.get_opposing_pair("spacetime")
        assert enhance == "generative"
        assert suppress == "trust"

    def test_generative_pair(self, ec):
        enhance, suppress = ec.get_opposing_pair("generative")
        assert enhance == "trust"
        assert suppress == "semantic"

    def test_trust_pair(self, ec):
        enhance, suppress = ec.get_opposing_pair("trust")
        assert enhance == "semantic"
        assert suppress == "causal"

    def test_old_naming(self, ec):
        enhance, suppress = ec.get_opposing_pair("wood")
        assert enhance == "causal"
        assert suppress == "spacetime"


# =============================================================================
# Normalization
# =============================================================================


class TestNormalizeEnergy:
    """Test _normalize_energy internal method."""

    def test_energy_type_enum(self, ec):
        result = ec._normalize_energy(EnergyType.FIRE)
        assert result == "fire"

    def test_old_naming_wood(self, ec):
        assert ec._normalize_energy("wood") == "semantic"

    def test_old_naming_fire(self, ec):
        assert ec._normalize_energy("fire") == "causal"

    def test_old_naming_earth(self, ec):
        assert ec._normalize_energy("earth") == "spacetime"

    def test_old_naming_metal(self, ec):
        assert ec._normalize_energy("metal") == "generative"

    def test_old_naming_water(self, ec):
        assert ec._normalize_energy("water") == "trust"

    def test_new_naming_passthrough(self, ec):
        assert ec._normalize_energy("semantic") == "semantic"

    def test_case_insensitive(self, ec):
        assert ec._normalize_energy("SEMANTIC") == "semantic"
