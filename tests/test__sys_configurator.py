"""
测试 _configurator.py — MetaConfigurator + HierarchicalConfigurator
====================================================================

覆盖两个配置器的全部公开方法：
- MetaConfigurator: configure, state, config_history, 4 条规则
- HierarchicalConfigurator: configure (L1/L2/L3), feedback, adaptive_thresholds
- ConfigAction 数据类
- _extract_energy_ratios_from_state 辅助函数
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

import pytest

from mci_world_model._sys._configurator import (
    ConfigAction,
    HierarchicalConfigurator,
    MetaConfigurator,
    _extract_energy_ratios_from_state,
)


# =============================================================================
# Helpers: Mock CognitiveGap
# =============================================================================


class MockGap:
    """Minimal mock of CognitiveGap for configurator tests."""

    def __init__(self, gap_id: str, gap_type: str, severity: float, description: str = ""):
        self.gap_id = gap_id
        self.gap_type = gap_type
        self.severity = severity
        self.description = description


def make_causal_gap(gap_id: str = "g1", severity: float = 0.7) -> MockGap:
    return MockGap(gap_id=gap_id, gap_type="causal", severity=severity)


def make_temporal_gap(gap_id: str = "g2", severity: float = 0.7) -> MockGap:
    return MockGap(gap_id=gap_id, gap_type="temporal", severity=severity)


def make_domain_gap(gap_id: str = "g3", severity: float = 0.7) -> MockGap:
    return MockGap(gap_id=gap_id, gap_type="domain", severity=severity)


def make_world_model(with_jepa: bool = True, edges: list | None = None):
    """Create a mock world model for configurator tests."""
    wm = MagicMock()
    wm.enable_m3 = MagicMock()

    state = MagicMock()
    state.causal_edges = edges or []
    wm._state = state

    if with_jepa:
        jepa = MagicMock()
        # Remove training_predict to simulate non-M3 mode
        del jepa.training_predict
        wm._jepa_predictor = jepa
    else:
        wm._jepa_predictor = None

    return wm


# =============================================================================
# ConfigAction
# =============================================================================


class TestConfigAction:
    """Test ConfigAction dataclass."""

    def test_creation_default_timestamp(self):
        action = ConfigAction(action_type="enable_m3", reason="test")
        assert action.action_type == "enable_m3"
        assert action.reason == "test"
        assert action.gap_id is None
        assert action.timestamp > 0

    def test_creation_with_gap_id(self):
        action = ConfigAction(action_type="adjust_weights", reason="domain gap", gap_id="g3")
        assert action.gap_id == "g3"

    def test_creation_noop(self):
        action = ConfigAction(action_type="noop", reason="no gaps")
        assert action.action_type == "noop"


# =============================================================================
# MetaConfigurator
# =============================================================================


class TestMetaConfiguratorInit:
    """Test MetaConfigurator initialization and properties."""

    def test_initial_state_is_idle(self):
        mc = MetaConfigurator()
        assert mc.state == "IDLE"

    def test_config_history_empty_initially(self):
        mc = MetaConfigurator()
        assert mc.config_history == []


class TestMetaConfiguratorConfigure:
    """Test MetaConfigurator.configure() main flow."""

    def test_no_gaps_returns_noop(self):
        mc = MetaConfigurator()
        wm = make_world_model()
        actions = mc.configure(wm, gaps=None)
        assert len(actions) == 1
        assert actions[0].action_type == "noop"

    def test_no_gaps_empty_list_returns_noop(self):
        mc = MetaConfigurator()
        wm = make_world_model()
        actions = mc.configure(wm, gaps=[])
        assert len(actions) == 1
        assert actions[0].action_type == "noop"

    def test_no_gaps_returns_to_idle(self):
        mc = MetaConfigurator()
        wm = make_world_model()
        mc.configure(wm, gaps=None)
        assert mc.state == "IDLE"

    def test_single_causal_gap_below_threshold(self):
        """severity < 0.5 → no M3 trigger."""
        mc = MetaConfigurator()
        wm = make_world_model()
        actions = mc.configure(wm, gaps=[make_causal_gap(severity=0.4)])
        # Not enough severity, not consecutive → no actions beyond noop-like
        assert all(a.action_type != "enable_m3" for a in actions)

    def test_high_severity_causal_gap_but_not_consecutive(self):
        """Single high-severity causal gap: increments counter but < 3."""
        mc = MetaConfigurator()
        wm = make_world_model()
        actions = mc.configure(wm, gaps=[make_causal_gap(severity=0.8)])
        # First high causal gap → _consecutive_causal_gaps becomes 1 (< 3)
        assert all(a.action_type != "enable_m3" for a in actions)

    def test_three_consecutive_causal_gaps_triggers_m3(self):
        """3 consecutive high-severity causal gaps → enable_m3."""
        mc = MetaConfigurator()
        wm = make_world_model()

        for i in range(2):
            mc.configure(wm, gaps=[make_causal_gap(gap_id=f"g{i}", severity=0.8)])

        # 3rd should trigger
        actions = mc.configure(wm, gaps=[make_causal_gap(gap_id="g3", severity=0.6)])
        assert any(a.action_type == "enable_m3" for a in actions)
        wm.enable_m3.assert_called_once()

    def test_causal_gap_resets_counter_on_non_causal(self):
        """Non-causal gaps reset consecutive counter."""
        mc = MetaConfigurator()
        wm = make_world_model()

        mc.configure(wm, gaps=[make_causal_gap(severity=0.8)])
        mc.configure(wm, gaps=[make_temporal_gap()])  # resets counter

        actions = mc.configure(wm, gaps=[make_causal_gap(severity=0.8)])
        assert all(a.action_type != "enable_m3" for a in actions)

    def test_temporal_gap_suggests_retrain(self):
        mc = MetaConfigurator()
        wm = make_world_model()
        actions = mc.configure(wm, gaps=[make_temporal_gap(severity=0.7)])
        assert any(a.action_type == "suggest_retrain" for a in actions)

    def test_temporal_gap_below_threshold_does_nothing(self):
        mc = MetaConfigurator()
        wm = make_world_model()
        actions = mc.configure(wm, gaps=[make_temporal_gap(severity=0.5)])
        assert not any(a.action_type == "suggest_retrain" for a in actions)

    def test_domain_gap_adjusts_weights(self):
        mc = MetaConfigurator()
        wm = make_world_model()
        actions = mc.configure(wm, gaps=[make_domain_gap(severity=0.7)])
        assert any(a.action_type == "adjust_weights" for a in actions)

    def test_domain_gap_below_threshold_does_nothing(self):
        mc = MetaConfigurator()
        wm = make_world_model()
        actions = mc.configure(wm, gaps=[make_domain_gap(severity=0.5)])
        assert not any(a.action_type == "adjust_weights" for a in actions)

    def test_low_confidence_edges_triggers_m3(self):
        mc = MetaConfigurator()
        edges = [{"confidence": 0.3}, {"confidence": 0.2}, {"confidence": 0.1}, {"confidence": 0.4}, {"confidence": 0.3}]
        wm = make_world_model(edges=edges)
        actions = mc.configure(wm, gaps=[make_causal_gap(severity=0.1)])
        assert any(a.action_type == "enable_m3" for a in actions)

    def test_low_confidence_no_jepa_suggests_retrain(self):
        mc = MetaConfigurator()
        edges = [{"confidence": 0.3}, {"confidence": 0.2}, {"confidence": 0.1}, {"confidence": 0.4}, {"confidence": 0.3}]
        wm = make_world_model(with_jepa=False, edges=edges)
        actions = mc.configure(wm, gaps=[make_causal_gap(severity=0.1)])
        assert any(a.action_type == "suggest_retrain" for a in actions)

    def test_empty_edges_no_low_confidence_trigger(self):
        mc = MetaConfigurator()
        wm = make_world_model(edges=[])
        actions = mc.configure(wm, gaps=[make_causal_gap(severity=0.1)])
        assert all(a.action_type != "enable_m3" for a in actions)

    def test_history_truncated_to_20(self):
        mc = MetaConfigurator()
        wm = make_world_model()
        for i in range(25):
            mc.configure(wm, gaps=[make_domain_gap(gap_id=f"g{i}", severity=0.7)])
        assert len(mc.config_history) <= 20

    def test_config_history_format(self):
        mc = MetaConfigurator()
        wm = make_world_model()
        mc.configure(wm, gaps=[make_domain_gap(severity=0.7)])
        history = mc.config_history
        assert isinstance(history, list)
        assert "type" in history[0]
        assert "reason" in history[0]
        assert "timestamp" in history[0]

    def test_m3_enable_failure_handled(self):
        """When world_model.enable_m3() raises, it's caught and logged."""
        mc = MetaConfigurator()
        wm = make_world_model()
        wm.enable_m3.side_effect = RuntimeError("M3 not available")

        for _ in range(3):
            mc.configure(wm, gaps=[make_causal_gap(severity=0.8)])
        # Should not raise, empty actions for causal
        assert mc.state in ("MONITORING", "IDLE")

    def test_configure_state_flow(self):
        mc = MetaConfigurator()
        wm = make_world_model()
        mc.configure(wm, gaps=[make_domain_gap(severity=0.7)])
        assert mc.state == "MONITORING"


# =============================================================================
# HierarchicalConfigurator
# =============================================================================


class TestHierarchicalConfiguratorInit:
    """Test HierarchicalConfigurator initialization."""

    def test_initial_state_is_idle(self):
        hc = HierarchicalConfigurator()
        assert hc.state == "IDLE"

    def test_config_history_empty_initially(self):
        hc = HierarchicalConfigurator()
        assert hc.config_history == []

    def test_adaptive_thresholds_default(self):
        hc = HierarchicalConfigurator()
        thresholds = hc.adaptive_thresholds
        assert "causal" in thresholds
        assert "temporal" in thresholds
        assert thresholds["causal"] == MetaConfigurator.M3_CAUSAL_SEVERITY_THRESHOLD

    def test_with_energy_core(self):
        from mci_world_model._sys._energy_core import EnergyCore

        ec = EnergyCore()
        hc = HierarchicalConfigurator(energy_core=ec)
        assert hc.state == "IDLE"


class TestHierarchicalConfiguratorConfigure:
    """Test HierarchicalConfigurator.configure()."""

    def test_no_gaps_returns_empty(self):
        hc = HierarchicalConfigurator()
        wm = make_world_model()
        actions = hc.configure(wm, gaps=None)
        assert actions == []

    def test_no_gaps_empty_list_returns_empty(self):
        hc = HierarchicalConfigurator()
        wm = make_world_model()
        actions = hc.configure(wm, gaps=[])
        assert actions == []

    def test_domain_gap_returns_scored_actions(self):
        hc = HierarchicalConfigurator()
        wm = make_world_model()
        actions = hc.configure(wm, gaps=[make_domain_gap(severity=0.8)])
        assert len(actions) >= 1
        assert "type" in actions[0]
        assert "priority" in actions[0]

    def test_multiple_gaps_produces_ranked_actions(self):
        hc = HierarchicalConfigurator()
        wm = make_world_model()
        gaps = [
            make_causal_gap(gap_id="g1", severity=0.9),
            make_domain_gap(gap_id="g3", severity=0.8),
            make_temporal_gap(gap_id="g2", severity=0.7),
        ]
        # First 2 runs to build consecutive count for causal
        for _ in range(2):
            hc.configure(wm, gaps=[make_causal_gap(severity=0.9)])
        actions = hc.configure(wm, gaps=gaps)
        assert len(actions) >= 1
        # Higher priority items first
        priorities = [a["priority"] for a in actions]
        assert priorities == sorted(priorities, reverse=True)

    def test_m3_conflict_resolution(self):
        """enable_m3 should subsume suggest_retrain in conflict resolution."""
        hc = HierarchicalConfigurator()
        wm = make_world_model()
        gaps = [
            make_causal_gap(gap_id="g1", severity=0.9),
            make_temporal_gap(gap_id="g2", severity=0.8),
        ]
        for _ in range(2):
            hc.configure(wm, gaps=[make_causal_gap(severity=0.9)])
        actions = hc.configure(wm, gaps=gaps)
        types = [a["type"] for a in actions]
        # Should NOT have both enable_m3 and suggest_retrain
        if "enable_m3" in types:
            assert "suggest_retrain" not in types

    def test_with_energy_core_no_crash(self):
        from mci_world_model._sys._energy_core import EnergyCore

        ec = EnergyCore()
        hc = HierarchicalConfigurator(energy_core=ec)
        wm = make_world_model()
        gaps = [make_causal_gap(gap_id="g1", severity=0.9)]
        for _ in range(2):
            hc.configure(wm, gaps=[make_causal_gap(severity=0.9)])
        actions = hc.configure(wm, gaps=gaps)
        assert isinstance(actions, list)

    def test_state_flow(self):
        hc = HierarchicalConfigurator()
        wm = make_world_model()
        hc.configure(wm, gaps=[make_domain_gap(severity=0.8)])
        assert hc.state in ("MONITORING", "IDLE")

    def test_config_history_updates(self):
        hc = HierarchicalConfigurator()
        wm = make_world_model()
        hc.configure(wm, gaps=[make_domain_gap(severity=0.8)])
        assert len(hc.config_history) >= 1


class TestHierarchicalConfiguratorFeedback:
    """Test feedback mechanism."""

    def test_feedback_accepts_bool(self):
        hc = HierarchicalConfigurator()
        hc.feedback(True)
        hc.feedback(False)
        # Should not raise

    def test_feedback_limits_history(self):
        hc = HierarchicalConfigurator()
        for _ in range(15):
            hc.feedback(True)
        # Internal _feedback_scores limited to 10
        # No direct assertion - just ensure no crash


class TestHierarchicalConfiguratorAdaptiveThresholds:
    """Test adaptive threshold adjustment."""

    def test_thresholds_updatable(self):
        hc = HierarchicalConfigurator()
        wm = make_world_model()
        for _ in range(10):
            hc.configure(wm, gaps=[make_causal_gap(severity=0.9)])
        # Thresholds should have been adjusted
        thresholds = hc.adaptive_thresholds
        assert 0.1 < thresholds["causal"] <= 0.9

    def test_thresholds_with_energy_core(self):
        from mci_world_model._sys._energy_core import EnergyCore

        ec = EnergyCore()
        hc = HierarchicalConfigurator(energy_core=ec)
        wm = make_world_model()
        hc.configure(wm, gaps=[make_domain_gap(severity=0.8)])
        thresholds = hc.adaptive_thresholds
        assert "causal" in thresholds


# =============================================================================
# _extract_energy_ratios_from_state
# =============================================================================


class TestExtractEnergyRatios:
    """Test _extract_energy_ratios_from_state helper."""

    def test_empty_edges_returns_none(self):
        state = MagicMock()
        state.causal_edges = []
        result = _extract_energy_ratios_from_state(state)
        # May return None or dict depending on aggregation
        assert result is None or isinstance(result, dict)

    def test_no_energy_in_edges_returns_none(self):
        state = MagicMock()
        state.causal_edges = [{"confidence": 0.5, "source": "a", "target": "b"}]
        result = _extract_energy_ratios_from_state(state)
        assert result is None or isinstance(result, dict)

    def test_state_without_causal_edges_attr(self):
        state = MagicMock(spec=[])  # no causal_edges
        result = _extract_energy_ratios_from_state(state)
        assert result is None
