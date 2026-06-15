"""
测试 evidence.py — 证据收集与似然函数计算机制
==============================================

覆盖 EvidenceCollector 全部公开方法：
- 来源管理: register_source, get_source_reliability
- 证据收集: collect, collect_batch
- 证据验证: verify_evidence
- 似然计算: compute_likelihood, compute_evidence_strength
- 冲突检测: detect_evidence_conflicts, detect_cross_belief_conflicts
- 自适应权重: calibrate_all_sources, get_source_rankings
- 查询: get_evidence_for_belief, get_recent_evidence, get_evidence_summary
- 统计与序列化: get_statistics, to_dict, from_dict, to_json, from_json, reset
- 数据结构: EvidenceRecord, SourceProfile
"""

from __future__ import annotations

import json
import time

import pytest

from mci_world_model._sys.evidence import (
    EvidenceCollector,
    EvidenceRecord,
    SourceProfile,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def collector():
    return EvidenceCollector()


@pytest.fixture
def populated_collector():
    """Collector with pre-registered sources and evidence."""
    col = EvidenceCollector()
    col.register_source("user_feedback", "user_feedback", 0.8)
    col.register_source("model_output", "model_output", 0.9)
    col.register_source("external_db", "external", 0.75)

    # Add some evidence
    for i in range(5):
        col.collect(
            belief_id="belief_1",
            is_positive=True,
            source="user_feedback",
            source_type="user_feedback",
            weight=1.0,
            context=f"test_{i}",
        )
    for i in range(3):
        col.collect(
            belief_id="belief_1",
            is_positive=False,
            source="model_output",
            source_type="model_output",
            weight=0.8,
            context=f"negative_{i}",
        )
    col.collect(belief_id="belief_2", is_positive=True, source="external_db", source_type="external", weight=1.0)
    col.collect(
        belief_id="belief_2", is_positive=False, source="user_feedback", source_type="user_feedback", weight=1.0
    )

    return col


# =============================================================================
# Data Structures
# =============================================================================


class TestEvidenceRecord:
    def test_creation(self):
        now = time.time()
        record = EvidenceRecord(
            evidence_id="ev_1",
            belief_id="b1",
            source="user",
            source_type="user_feedback",
            is_positive=True,
            raw_weight=1.0,
            calibrated_weight=0.8,
            timestamp=now,
            context="test evidence",
        )
        assert record.evidence_id == "ev_1"
        assert record.belief_id == "b1"
        assert record.is_positive is True
        assert record.source == "user"


class TestSourceProfile:
    def test_creation(self):
        profile = SourceProfile(
            source_id="user_feedback",
            source_type="user_feedback",
        )
        assert profile.source_id == "user_feedback"
        assert profile.total_evidence == 0

    def test_reliability_score_default(self):
        profile = SourceProfile(source_id="test", source_type="test")
        assert 0.0 <= profile.reliability_score <= 1.0

    def test_update_reliability_correct(self):
        profile = SourceProfile(source_id="test", source_type="test")
        profile.update_reliability(was_correct=True)
        assert profile.total_evidence == 1
        assert profile.verified_evidence == 1

    def test_update_reliability_incorrect(self):
        profile = SourceProfile(source_id="test", source_type="test")
        profile.update_reliability(was_correct=False)
        assert profile.total_evidence == 1
        assert profile.contradicted_evidence == 1


# =============================================================================
# Source Management
# =============================================================================


class TestSourceManagement:
    def test_register_new_source(self, collector):
        profile = collector.register_source("sensor_1", "sensor", 0.85)
        assert profile.source_id == "sensor_1"
        assert profile.source_type == "sensor"

    def test_register_existing_returns_same(self, collector):
        p1 = collector.register_source("sensor_1", "sensor")
        p2 = collector.register_source("sensor_1", "sensor")
        assert p1 is p2

    def test_register_default_reliability(self, collector):
        profile = collector.register_source("default_src")
        score = profile.reliability_score
        assert 0.0 <= score <= 1.0

    def test_get_source_reliability_known(self, collector):
        collector.register_source("reliable", "test", 0.9)
        assert collector.get_source_reliability("reliable") == pytest.approx(0.9, abs=0.15)

    def test_get_source_reliability_unknown(self, collector):
        assert collector.get_source_reliability("unknown") == 0.7  # default


# =============================================================================
# Evidence Collection
# =============================================================================


class TestEvidenceCollection:
    def test_collect_single(self, collector):
        record = collector.collect(
            belief_id="b1",
            is_positive=True,
            source="user",
            source_type="user_feedback",
            weight=1.0,
            context="test",
        )
        assert record.belief_id == "b1"
        assert record.source == "user"
        assert record.evidence_id.startswith("ev_")

    def test_collect_calibrates_weight(self, collector):
        """Weight should be calibrated by source reliability."""
        collector.register_source("trusted", "user_feedback", 1.0)
        collector.register_source("untrusted", "user_feedback", 0.5)

        r1 = collector.collect(belief_id="b1", is_positive=True, source="trusted", weight=1.0)
        r2 = collector.collect(belief_id="b1", is_positive=True, source="untrusted", weight=1.0)

        assert r1.calibrated_weight > r2.calibrated_weight

    def test_collect_negative_evidence(self, collector):
        record = collector.collect(
            belief_id="b1",
            is_positive=False,
            source="model",
            source_type="model_output",
            weight=0.5,
        )
        assert record.is_positive is False

    def test_collect_with_metadata(self, collector):
        record = collector.collect(
            belief_id="b1",
            is_positive=True,
            source="user",
            metadata={"key": "value", "nested": {"a": 1}},
        )
        assert record.metadata["key"] == "value"

    def test_collect_batch(self, collector):
        evidence_list = [
            {"belief_id": "b1", "is_positive": True, "source": "batch", "source_type": "batch", "weight": 1.0},
            {"belief_id": "b2", "is_positive": False, "source": "batch", "source_type": "batch", "weight": 0.5},
            {"belief_id": "b1", "is_positive": True, "source": "batch", "source_type": "batch", "weight": 1.0},
        ]
        records = collector.collect_batch(evidence_list)
        assert len(records) == 3
        assert all(isinstance(r, EvidenceRecord) for r in records)

    def test_collect_batch_with_metadata(self, collector):
        evidence_list = [
            {"belief_id": "b1", "is_positive": True, "source": "batch", "metadata": {"batch_id": "42"}},
        ]
        records = collector.collect_batch(evidence_list)
        assert records[0].metadata["batch_id"] == "42"

    def test_collect_increments_total(self, collector):
        collector.collect(belief_id="b1", is_positive=True, source="s1")
        collector.collect(belief_id="b1", is_positive=False, source="s1")
        stats = collector.get_statistics()
        assert stats["total_collected"] == 2


# =============================================================================
# Evidence Verification
# =============================================================================


class TestEvidenceVerification:
    def test_verify_correct_evidence(self, populated_collector):
        """Verifying evidence as correct updates source reliability."""
        records = populated_collector.get_evidence_for_belief("belief_1")
        if records:
            populated_collector.verify_evidence(records[0].evidence_id, was_correct=True)
            # Should not raise

    def test_verify_incorrect_evidence(self, populated_collector):
        records = populated_collector.get_evidence_for_belief("belief_1")
        if records:
            populated_collector.verify_evidence(records[0].evidence_id, was_correct=False)

    def test_verify_nonexistent_evidence(self, collector):
        """Verifying non-existent evidence should not crash."""
        collector.verify_evidence("nonexistent", was_correct=True)
        # Should not raise


# =============================================================================
# Likelihood & Evidence Strength
# =============================================================================


class TestLikelihoodComputation:
    def test_compute_likelihood(self, populated_collector):
        ll = populated_collector.compute_likelihood("belief_1")
        assert isinstance(ll, float)

    def test_compute_likelihood_nonexistent(self, collector):
        ll = collector.compute_likelihood("nonexistent")
        assert ll == 0.0

    def test_compute_likelihood_with_hypothesis(self, populated_collector):
        ll = populated_collector.compute_likelihood("belief_1", hypothesis_value=0.6)
        assert isinstance(ll, float)

    def test_compute_likelihood_time_window(self, populated_collector):
        ll = populated_collector.compute_likelihood("belief_1", time_window=3600)
        assert isinstance(ll, float)


class TestEvidenceStrength:
    def test_strength_has_keys(self, populated_collector):
        strength = populated_collector.compute_evidence_strength("belief_1")
        assert "positive_weight" in strength
        assert "negative_weight" in strength
        assert "total_weight" in strength
        assert "evidence_count" in strength

    def test_strength_empty(self, collector):
        strength = collector.compute_evidence_strength("nonexistent")
        assert strength["total_weight"] == 0.0
        assert strength["evidence_count"] == 0


# =============================================================================
# Conflict Detection
# =============================================================================


class TestConflictDetection:
    def test_detect_evidence_conflicts(self, populated_collector):
        conflicts = populated_collector.detect_evidence_conflicts("belief_1")
        assert isinstance(conflicts, list)

    def test_detect_evidence_conflicts_empty(self, collector):
        conflicts = collector.detect_evidence_conflicts("nonexistent")
        assert conflicts == []

    def test_detect_conflicts_with_threshold(self, populated_collector):
        conflicts = populated_collector.detect_evidence_conflicts("belief_1", threshold=0.1)
        assert isinstance(conflicts, list)

    def test_detect_cross_belief_conflicts(self, populated_collector):
        conflicts = populated_collector.detect_cross_belief_conflicts()
        assert isinstance(conflicts, list)


# =============================================================================
# Adaptive Weights
# =============================================================================


class TestAdaptiveWeights:
    def test_calibrate_all_sources(self, populated_collector):
        populated_collector.calibrate_all_sources()
        # Should not raise

    def test_calibrate_empty(self, collector):
        collector.calibrate_all_sources()
        # Should not raise

    def test_get_source_rankings(self, populated_collector):
        rankings = populated_collector.get_source_rankings()
        assert isinstance(rankings, list)
        if rankings:
            assert "source_id" in rankings[0]
            assert "reliability" in rankings[0]

    def test_get_source_rankings_empty(self, collector):
        rankings = collector.get_source_rankings()
        assert rankings == []


# =============================================================================
# Query Methods
# =============================================================================


class TestQueryMethods:
    def test_get_evidence_for_belief(self, populated_collector):
        records = populated_collector.get_evidence_for_belief("belief_1")
        assert len(records) > 0
        assert all(isinstance(r, EvidenceRecord) for r in records)

    def test_get_evidence_for_belief_empty(self, collector):
        records = collector.get_evidence_for_belief("nonexistent")
        assert records == []

    def test_get_evidence_with_time_window(self, populated_collector):
        records = populated_collector.get_evidence_for_belief("belief_1", time_window=60)
        assert isinstance(records, list)

    def test_get_recent_evidence(self, populated_collector):
        recent = populated_collector.get_recent_evidence(n=3)
        assert len(recent) <= 3

    def test_get_recent_evidence_empty(self, collector):
        recent = collector.get_recent_evidence()
        assert recent == []

    def test_get_evidence_summary(self, populated_collector):
        summary = populated_collector.get_evidence_summary("belief_1")
        assert summary["belief_id"] == "belief_1"
        assert "strength" in summary
        assert "sources" in summary
        assert "conflicts" in summary

    def test_get_evidence_summary_empty(self, collector):
        summary = collector.get_evidence_summary("nonexistent")
        assert summary["total_evidence"] == 0


# =============================================================================
# Statistics & Serialization
# =============================================================================


class TestStatistics:
    def test_get_statistics(self, populated_collector):
        stats = populated_collector.get_statistics()
        assert "total_collected" in stats
        assert "registered_sources" in stats
        assert stats["total_collected"] > 0

    def test_get_statistics_empty(self, collector):
        stats = collector.get_statistics()
        assert stats["total_collected"] == 0
        assert stats["registered_sources"] == 0


class TestSerialization:
    def test_to_dict(self, populated_collector):
        d = populated_collector.to_dict()
        assert "evidence_count" in d
        assert "sources" in d
        assert "engine" in d

    def test_from_dict(self, populated_collector):
        d = populated_collector.to_dict()
        restored = EvidenceCollector.from_dict(d)
        assert restored._total_collected == populated_collector._total_collected

    def test_to_json(self, populated_collector):
        j = populated_collector.to_json()
        assert isinstance(j, str)
        parsed = json.loads(j)
        assert "evidence_count" in parsed

    def test_from_json(self, populated_collector):
        j = populated_collector.to_json()
        restored = EvidenceCollector.from_json(j)
        assert restored._total_collected == populated_collector._total_collected

    def test_roundtrip(self, populated_collector):
        d = populated_collector.to_dict()
        restored = EvidenceCollector.from_dict(d)
        assert restored._total_collected == populated_collector._total_collected
        assert restored._conflicts_detected == populated_collector._conflicts_detected

    def test_reset(self, populated_collector):
        populated_collector.reset()
        assert populated_collector._total_collected == 0
        assert populated_collector._evidence_history == []
        assert populated_collector._source_profiles == {}
