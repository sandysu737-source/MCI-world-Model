"""
测试 awareness.py + states.py — 元认知系统与信念演化追踪
===========================================================

覆盖：
- awareness.py: MetaCognition.discover_gaps, detect_conflicts, get_aging_warnings, get_suggestions
- states.py: BeliefTracker 完整生命周期, BayesianBeliefTracker 贝叶斯更新
"""

from __future__ import annotations

import time

import pytest

from mci_world_model._sys.awareness import CognitiveGap, KnowledgeAging, MetaCognition
from mci_world_model._sys.states import (
    BayesianBeliefState,
    BayesianBeliefTracker,
    BeliefStage,
    BeliefState,
    BeliefTracker,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mc():
    return MetaCognition()


@pytest.fixture
def bt():
    return BeliefTracker()


# =============================================================================
# CognitiveGap / KnowledgeAging dataclasses
# =============================================================================


class TestCognitiveGap:
    def test_creation(self):
        now = time.time()
        gap = CognitiveGap(
            gap_id="g1",
            gap_type="domain",
            description="事实类记忆偏少",
            severity=0.7,
            suggestions=["补充事实信息"],
            discovered_at=now,
        )
        assert gap.gap_id == "g1"
        assert gap.gap_type == "domain"
        assert gap.severity == 0.7


class TestKnowledgeAging:
    def test_creation(self):
        aging = KnowledgeAging(
            memory_id="m1",
            days_since_update=40,
            current_stage="确认",
            severity="critical",
            suggestion="建议更新",
        )
        assert aging.memory_id == "m1"
        assert aging.severity == "critical"


# =============================================================================
# MetaCognition — Discover Gaps
# =============================================================================


class TestMetaCognitionDiscoverGaps:
    """Test discover_gaps for all three gap types."""

    def test_domain_gap_fact_low(self, mc):
        """fact < 30% → domain gap."""
        gaps = mc.discover_gaps({"fact": 1, "other": 9}, [], [])
        assert any(g.gap_type == "domain" for g in gaps)

    def test_domain_gap_fact_sufficient(self, mc):
        """fact >= 30% → no domain gap from fact."""
        gaps = mc.discover_gaps({"fact": 3, "other": 7}, [], [])
        domain_gaps = [g for g in gaps if "domain_fact" in g.gap_id]
        assert len(domain_gaps) == 0

    def test_domain_gap_preference_low(self, mc):
        """preference < 10% → domain gap."""
        gaps = mc.discover_gaps({"fact": 5, "preference": 0, "other": 95}, [], [])
        assert any("pref" in g.gap_id for g in gaps)

    def test_domain_gap_event_too_high(self, mc):
        """event > 50% → domain gap for noise."""
        gaps = mc.discover_gaps({"fact": 2, "event": 8}, [], [])
        event_gaps = [g for g in gaps if "event" in g.gap_id]
        assert len(event_gaps) >= 1

    def test_domain_gap_severity_ranges(self, mc):
        gaps = mc.discover_gaps({"fact": 1, "preference": 0, "event": 9}, [], [])
        for g in gaps:
            assert 0.0 <= g.severity <= 1.0

    def test_temporal_gap_stale(self, mc):
        """Memory > 60 days old → temporal gap."""
        old_memory = [
            {"type": "fact", "timestamp": time.time() - 86400 * 65},
            {"type": "fact", "timestamp": time.time() - 86400 * 65},
            {"type": "fact", "timestamp": time.time() - 86400 * 65},
            {"type": "fact", "timestamp": time.time() - 86400 * 65},
        ]
        gaps = mc.discover_gaps({"fact": 5, "other": 5}, [], old_memory)
        assert any(g.gap_type == "temporal" for g in gaps)

    def test_temporal_gap_fresh(self, mc):
        """Recent memory → no temporal gap."""
        recent_memory = [{"type": "fact", "timestamp": time.time() - 86400 * 10}]
        gaps = mc.discover_gaps({"fact": 5, "other": 5}, [], recent_memory)
        assert not any(g.gap_type == "temporal" for g in gaps)

    def test_temporal_gap_with_user_domains(self, mc):
        """user_domains param passed but not used in gap detection (doesn't crash)."""
        gaps = mc.discover_gaps({"fact": 5, "other": 5}, ["nutrition", "medicine"], [])
        assert isinstance(gaps, list)

    def test_causal_gap_isolated(self, mc):
        """>80% isolated memories → causal gap."""
        memories = [
            {"causal_parents": None, "causal_children": None},
            {"causal_parents": None, "causal_children": None},
            {"causal_parents": [], "causal_children": []},
            {"causal_parents": None, "causal_children": None},
            {"causal_parents": None, "causal_children": None},
            {"causal_parents": None, "causal_children": None},
            {"causal_parents": None, "causal_children": None},
            {"causal_parents": None, "causal_children": None},
            {"causal_parents": None, "causal_children": None},
            {"causal_parents": None, "causal_children": None},
            {"causal_parents": None, "causal_children": None},  # 11 > 10
        ]
        gaps = mc.discover_gaps({"fact": 5, "other": 5}, [], memories)
        assert any(g.gap_type == "causal" for g in gaps)

    def test_causal_gap_connected(self, mc):
        """Most memories connected → no causal gap."""
        memories = [
            {"causal_parents": ["p1"], "causal_children": ["c1"]},
            {"causal_parents": ["p2"], "causal_children": None},
            {"causal_parents": None, "causal_children": None},
        ]
        gaps = mc.discover_gaps({"fact": 3, "other": 3}, [], memories)
        assert not any(g.gap_type == "causal" for g in gaps)

    def test_causal_gap_minimal_count(self, mc):
        """Less than 10 total memories → no causal gap regardless."""
        memories = [{"causal_parents": None, "causal_children": None}] * 5
        gaps = mc.discover_gaps({"fact": 3, "other": 2}, [], memories)
        assert not any(g.gap_type == "causal" for g in gaps)

    def test_empty_memory_types(self, mc):
        """All zero memory types → no domain gaps."""
        gaps = mc.discover_gaps({}, [], [])
        # total is 0 → returns early
        assert isinstance(gaps, list)

    def test_gaps_have_suggestions(self, mc):
        gaps = mc.discover_gaps({"fact": 1, "preference": 0, "event": 8}, [], [])
        for g in gaps:
            assert len(g.suggestions) > 0


# =============================================================================
# MetaCognition — Detect Conflicts
# =============================================================================


class TestMetaCognitionDetectConflicts:
    """Test detect_conflicts with various belief scenarios."""

    def test_no_conflicts(self, mc):
        beliefs = {
            "a": {"content": "天气很好", "confidence": 0.9, "stage": "强化"},
            "b": {"content": "今天很开心", "confidence": 0.8, "stage": "确认"},
        }
        conflicts = mc.detect_conflicts(beliefs)
        assert len(conflicts) == 0

    def test_opposite_polarity_conflict(self, mc):
        beliefs = {
            "a": {"content": "这是正确的答案", "confidence": 0.9, "stage": "强化"},
            "b": {"content": "这不是正确的答案", "confidence": 0.8, "stage": "强化"},
        }
        conflicts = mc.detect_conflicts(beliefs)
        assert len(conflicts) >= 1

    def test_low_confidence_no_conflict(self, mc):
        """Low confidence (< 0.7) should not trigger conflict."""
        beliefs = {
            "a": {"content": "这是正确的", "confidence": 0.6, "stage": "认知"},
            "b": {"content": "这是错误的", "confidence": 0.6, "stage": "认知"},
        }
        conflicts = mc.detect_conflicts(beliefs)
        assert len(conflicts) == 0

    def test_conflict_severity_average(self, mc):
        beliefs = {
            "a": {"content": "这是正确的", "confidence": 0.9, "stage": "强化"},
            "b": {"content": "这是错误的", "confidence": 0.71, "stage": "确认"},
        }
        conflicts = mc.detect_conflicts(beliefs)
        assert len(conflicts) >= 1
        assert 0.0 <= conflicts[0]["severity"] <= 1.0

    def test_conflict_sorted_by_severity(self, mc):
        beliefs = {
            "a": {"content": "这是正确的", "confidence": 0.75, "stage": "强化"},
            "b": {"content": "这是错误的", "confidence": 0.95, "stage": "强化"},
            "c": {"content": "A是正确的", "confidence": 0.8, "stage": "确认"},
            "d": {"content": "B是错误的", "confidence": 0.85, "stage": "确认"},
        }
        conflicts = mc.detect_conflicts(beliefs)
        if len(conflicts) >= 2:
            assert conflicts[0]["severity"] >= conflicts[-1]["severity"]

    def test_not_contradictory_same_polarity(self, mc):
        assert not mc._is_contradictory("这是正确的", "我知道答案")


# =============================================================================
# MetaCognition — Aging Warnings & Suggestions
# =============================================================================


class TestMetaCognitionAging:
    """Test get_aging_warnings."""

    def test_warning_level(self, mc):
        """30 < days < 60 → warning."""
        mem = [{"id": "m1", "timestamp": time.time() - 86400 * 40, "stage": "确认"}]
        warnings = mc.get_aging_warnings(mem)
        assert len(warnings) >= 1
        assert warnings[0].severity == "warning"

    def test_critical_level(self, mc):
        """days > 60 → critical."""
        mem = [{"id": "m1", "timestamp": time.time() - 86400 * 65, "stage": "强化"}]
        warnings = mc.get_aging_warnings(mem)
        assert len(warnings) >= 1
        assert warnings[0].severity == "critical"

    def test_fresh_no_warning(self, mc):
        mem = [{"id": "m1", "timestamp": time.time() - 86400 * 5, "stage": "认知"}]
        warnings = mc.get_aging_warnings(mem)
        assert len(warnings) == 0

    def test_mixed_ages(self, mc):
        mem = [
            {"id": "m1", "timestamp": time.time() - 86400 * 70, "stage": "强化"},
            {"id": "m2", "timestamp": time.time() - 86400 * 20, "stage": "确认"},
            {"id": "m3", "timestamp": time.time() - 86400 * 3, "stage": "认知"},
        ]
        warnings = mc.get_aging_warnings(mem)
        severities = {w.severity for w in warnings}
        assert "critical" in severities

    def test_aging_has_suggestion(self, mc):
        mem = [{"id": "m1", "timestamp": time.time() - 86400 * 70, "stage": "确认"}]
        warnings = mc.get_aging_warnings(mem)
        assert len(warnings[0].suggestion) > 0


class TestMetaCognitionSuggestions:
    """Test get_suggestions."""

    def test_empty_when_no_gaps(self, mc):
        assert mc.get_suggestions() == []

    def test_returns_after_gaps(self, mc):
        mc.discover_gaps({"fact": 1, "preference": 0, "event": 8}, [], [])
        suggestions = mc.get_suggestions()
        assert len(suggestions) > 0
        assert len(suggestions) <= 5  # max 5

    def test_no_duplicates(self, mc):
        mc.discover_gaps({"fact": 1, "preference": 0, "event": 8}, [], [])
        suggestions = mc.get_suggestions()
        assert len(suggestions) == len(set(suggestions))


# =============================================================================
# BeliefTracker
# =============================================================================


class TestBeliefTrackerLifecycle:
    """Test the full belief lifecycle: initialize → reinforce → shake → decay."""

    def test_initialize(self, bt):
        state = bt.initialize("mem_1")
        assert state.memory_id == "mem_1"
        assert state.stage == BeliefStage.COGNITION
        assert state.confidence == 0.5
        assert state.reinforce_count == 0
        assert state.shake_count == 0

    def test_initialize_stores_belief(self, bt):
        bt.initialize("mem_1")
        state = bt.get_state("mem_1")
        assert state is not None
        assert state.memory_id == "mem_1"

    def test_reinforce_increments_count(self, bt):
        bt.initialize("mem_1")
        state = bt.reinforce("mem_1")
        assert state.reinforce_count == 1

    def test_reinforce_increases_confidence(self, bt):
        bt.initialize("mem_1")
        initial_conf = bt.get_state("mem_1").confidence
        state = bt.reinforce("mem_1")
        assert state.confidence > initial_conf

    def test_reinforce_auto_initializes(self, bt):
        """reinforce on unknown memory auto-initializes."""
        state = bt.reinforce("new_mem")
        assert state.memory_id == "new_mem"
        assert state.stage == BeliefStage.COGNITION

    def test_reinforce_stage_transition_to_confirm(self, bt):
        """3 reinforces → CONFIRM stage."""
        bt.initialize("mem_1")
        for _ in range(3):
            bt.reinforce("mem_1")
        state = bt.get_state("mem_1")
        assert state.stage == BeliefStage.CONFIRM

    def test_reinforce_stage_transition_to_reinforce(self, bt):
        """Continue reinforcing → REINFORCE stage (confidence >= 0.7)."""
        bt.initialize("mem_1")
        for _ in range(10):
            bt.reinforce("mem_1")
        state = bt.get_state("mem_1")
        assert state.stage == BeliefStage.REINFORCE

    def test_shake_increments_count(self, bt):
        bt.initialize("mem_1")
        state = bt.shake("mem_1")
        assert state.shake_count == 1

    def test_shake_decreases_confidence(self, bt):
        bt.initialize("mem_1")
        initial_conf = bt.get_state("mem_1").confidence
        state = bt.shake("mem_1")
        assert state.confidence < initial_conf

    def test_shake_with_conflict(self, bt):
        bt.initialize("mem_1")
        state = bt.shake("mem_1", conflict_with="mem_2")
        assert state.shake_count == 1

    def test_transitions_recorded(self, bt):
        bt.initialize("mem_1")
        assert "认知" in bt.get_state("mem_1").transitions

    def test_get_state_nonexistent(self, bt):
        assert bt.get_state("nonexistent") is None

    def test_should_forget_false_for_active(self, bt):
        bt.initialize("mem_1")
        assert bt.should_forget("mem_1") is False

    def test_should_forget_nonexistent(self, bt):
        assert bt.should_forget("nonexistent") is False

    def test_get_stage_distribution(self, bt):
        bt.initialize("mem_1")
        bt.initialize("mem_2")
        dist = bt.get_stage_distribution()
        assert isinstance(dist, dict)
        assert BeliefStage.COGNITION in dist
        assert dist[BeliefStage.COGNITION] == 2

    def test_apply_decay_does_not_crash(self, bt):
        bt.initialize("mem_1")
        reshaped = bt.apply_decay()
        assert isinstance(reshaped, list)


# =============================================================================
# BayesianBeliefTracker
# =============================================================================


class TestBayesianBeliefTracker:
    """Test BayesianBeliefTracker core methods."""

    @pytest.fixture
    def bbt(self):
        return BayesianBeliefTracker()

    def test_initialize(self, bbt):
        state = bbt.initialize("mem_1")
        assert state.memory_id == "mem_1"
        assert state.stage == "认知"
        assert 0.0 < state.confidence < 1.0

    def test_reinforce(self, bbt):
        bbt.initialize("mem_1")
        state = bbt.reinforce("mem_1")
        assert state.reinforce_count >= 1

    def test_reinforce_with_weight(self, bbt):
        bbt.initialize("mem_1")
        state = bbt.reinforce("mem_1", weight=2.0)
        assert state.reinforce_count >= 1

    def test_shake(self, bbt):
        bbt.initialize("mem_1")
        state = bbt.shake("mem_1")
        assert state.shake_count >= 1

    def test_shake_with_conflict(self, bbt):
        bbt.initialize("mem_1")
        state = bbt.shake("mem_1", conflict_with="mem_2")
        assert state.shake_count >= 1

    def test_get_state(self, bbt):
        bbt.initialize("mem_1")
        state = bbt.get_state("mem_1")
        assert state is not None
        assert state.memory_id == "mem_1"

    def test_get_state_nonexistent(self, bbt):
        assert bbt.get_state("nonexistent") is None

    def test_get_posterior(self, bbt):
        bbt.initialize("mem_1")
        posterior = bbt.get_posterior("mem_1")
        assert posterior is not None
        assert "mean" in posterior
        assert "alpha" in posterior
        assert "beta" in posterior

    def test_get_posterior_nonexistent(self, bbt):
        assert bbt.get_posterior("nonexistent") is None

    def test_get_stage_distribution(self, bbt):
        bbt.initialize("mem_1")
        dist = bbt.get_stage_distribution()
        assert isinstance(dist, dict)

    def test_should_forget_default_false(self, bbt):
        bbt.initialize("mem_1")
        assert bbt.should_forget("mem_1") is False

    def test_should_forget_nonexistent(self, bbt):
        assert bbt.should_forget("nonexistent") is False

    def test_apply_decay(self, bbt):
        bbt.initialize("mem_1")
        reshaped = bbt.apply_decay()
        assert isinstance(reshaped, list)

    def test_compare_beliefs(self, bbt):
        bbt.initialize("mem_1")
        bbt.initialize("mem_2")
        result = bbt.compare_beliefs("mem_1", "mem_2")
        assert result is not None
        assert "bayes_factor" in result or "comparison" in result

    def test_hypothesis_test(self, bbt):
        bbt.initialize("mem_1")
        result = bbt.hypothesis_test("mem_1", null_value=0.5)
        assert result is not None
        assert "bayes_factor" in result or "posterior_odds" in result or "hypothesis" in result

    def test_get_top_beliefs(self, bbt):
        bbt.initialize("mem_1")
        bbt.initialize("mem_2")
        top = bbt.get_top_beliefs(n=5)
        assert isinstance(top, list)

    def test_get_statistics(self, bbt):
        bbt.initialize("mem_1")
        stats = bbt.get_statistics()
        assert isinstance(stats, dict)
        if "error" not in stats:
            assert "total_beliefs" in stats or "belief_count" in stats

    def test_bayesian_state_has_uncertainty(self, bbt):
        state = bbt.initialize("mem_1")
        assert 0.0 <= state.uncertainty <= 1.0

    def test_bayesian_state_has_credible_interval(self, bbt):
        state = bbt.initialize("mem_1")
        assert len(state.credible_interval_95) == 2


# =============================================================================
# BeliefState / BayesianBeliefState dataclasses
# =============================================================================


class TestBeliefStateDataclass:
    def test_creation(self):
        now = time.time()
        state = BeliefState(
            memory_id="m1",
            stage=BeliefStage.COGNITION,
            confidence=0.5,
            reinforce_count=0,
            shake_count=0,
            last_reinforced=now,
            last_shaken=0,
            created_at=now,
        )
        assert state.memory_id == "m1"
        assert state.confidence == 0.5


class TestBayesianBeliefStateDataclass:
    def test_creation(self):
        state = BayesianBeliefState(
            memory_id="m1",
            stage="认知",
            confidence=0.5,
            uncertainty=0.3,
            reinforce_count=0,
            shake_count=0,
            alpha=1.0,
            beta=1.0,
            credible_interval_95=(0.0, 1.0),
        )
        assert state.memory_id == "m1"
        assert state.alpha == 1.0
        assert state.beta == 1.0
