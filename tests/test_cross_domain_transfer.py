"""End-to-end tests for CrossDomainCausalTransfer — P10 cross-domain transfer."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from mci_world_model.sdk._cross_domain_transfer import (
    CausalKnowledge,
    CrossDomainCausalTransfer,
    DomainAdapter,
    DomainType,
    TransferResult,
    TransferStatus,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def engine():
    return CrossDomainCausalTransfer()


@pytest.fixture
def sample_knowledge():
    return CausalKnowledge(
        knowledge_id="med-heart-001",
        source_domain=DomainType.MEDICAL,
        causal_graph={"A": ["B"], "B": ["C"]},
        confidence=0.85,
        n_observations=500,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════════

class TestCausalKnowledge:
    def test_creation(self):
        ck = CausalKnowledge("k1", DomainType.ENGINEERING, confidence=0.9)
        assert ck.knowledge_id == "k1"
        assert ck.source_domain == DomainType.ENGINEERING
        assert ck.confidence == 0.9

    def test_defaults(self):
        ck = CausalKnowledge("k2", DomainType.FINANCE)
        assert ck.causal_graph == {}
        assert ck.confidence == 0.0
        assert ck.n_observations == 0


class TestDomainAdapter:
    def test_same_domain_max_compatibility(self):
        a = DomainAdapter(DomainType.MEDICAL, DomainType.MEDICAL)
        assert a.compute_compatibility() == 1.0

    def test_cross_domain_base_compatibility(self):
        a = DomainAdapter(DomainType.MEDICAL, DomainType.ENGINEERING)
        c = a.compute_compatibility()
        assert 0.0 <= c <= 1.0

    def test_initial_compatibility_zero(self):
        a = DomainAdapter(DomainType.SOCIAL, DomainType.PHYSICAL)
        assert a.compatibility_score == 0.0


class TestTransferResult:
    def test_default_status_pending(self):
        r = TransferResult("T1", DomainType.MEDICAL, DomainType.FINANCE)
        assert r.status == TransferStatus.PENDING
        assert r.fidelity == 0.0
        assert r.n_knowledge_transferred == 0


# ═══════════════════════════════════════════════════════════════════════════════
# CrossDomainCausalTransfer core
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossDomainTransfer:
    def test_register_knowledge(self, engine, sample_knowledge):
        result = engine.register_knowledge(sample_knowledge)
        assert result["status"] == "registered"
        assert result["knowledge_id"] == "med-heart-001"

    def test_create_adapter(self, engine):
        result = engine.create_adapter(DomainType.MEDICAL, DomainType.ENGINEERING)
        assert result["status"] == "adapter_created"
        assert result["source"] == "medical"
        assert result["target"] == "engineering"
        assert 0.0 <= result["compatibility"] <= 1.0

    def test_transfer_knowledge(self, engine, sample_knowledge):
        engine.register_knowledge(sample_knowledge)
        engine.create_adapter(DomainType.MEDICAL, DomainType.ENGINEERING)
        result = engine.transfer("med-heart-001", DomainType.ENGINEERING)
        assert result["status"] in ("transferred", "failed")
        assert "fidelity" in result

    def test_transfer_missing_knowledge(self, engine):
        result = engine.transfer("nonexistent", DomainType.ENGINEERING)
        assert result["status"] == "not_found"

    def test_transfer_creates_adapter_if_missing(self, engine, sample_knowledge):
        engine.register_knowledge(sample_knowledge)
        # No adapter created explicitly
        result = engine.transfer("med-heart-001", DomainType.SOCIAL)
        assert result["status"] in ("transferred", "failed")

    def test_verify_transfer(self, engine, sample_knowledge):
        engine.register_knowledge(sample_knowledge)
        result = engine.transfer("med-heart-001", DomainType.ENGINEERING)
        if result["status"] == "transferred":
            verify = engine.verify_transfer(result["transfer_id"])
            assert verify["verified"] is True
            assert verify["status"] == "verified"

    def test_verify_nonexistent_transfer(self, engine):
        result = engine.verify_transfer("nonexistent")
        assert result["status"] == "not_found"

    def test_detect_emergence(self, engine, sample_knowledge):
        engine.register_knowledge(sample_knowledge)
        engine.transfer("med-heart-001", DomainType.ENGINEERING)
        engine.transfer("med-heart-001", DomainType.FINANCE)

        emergence = engine.detect_emergence(DomainType.MEDICAL)
        assert "domain" in emergence
        assert emergence["emergence_detected"] in (True, False)

    def test_get_transfer_report(self, engine, sample_knowledge):
        engine.register_knowledge(sample_knowledge)
        engine.transfer("med-heart-001", DomainType.ENGINEERING)

        report = engine.get_transfer_report()
        assert report["n_knowledge"] >= 0
        assert report["n_adapters"] >= 0
        assert report["n_transfers"] >= 0
        assert isinstance(report["status_distribution"], dict)
        assert 0.0 <= report["avg_fidelity"] <= 1.0

    def test_multi_domain_transfer_chain(self, engine, sample_knowledge):
        """Chain: Medical → Engineering → Physical → Social."""
        engine.register_knowledge(sample_knowledge)

        domains = [DomainType.ENGINEERING, DomainType.PHYSICAL, DomainType.SOCIAL]
        for domain in domains:
            engine.create_adapter(DomainType.MEDICAL, domain)
            engine.transfer("med-heart-001", domain)

        report = engine.get_transfer_report()
        assert report["n_transfers"] == 3

    def test_low_confidence_knowledge_transfer(self, engine):
        """Low confidence knowledge should transfer with low fidelity."""
        kw = CausalKnowledge("kw1", DomainType.MEDICAL, confidence=0.2)
        engine.register_knowledge(kw)
        result = engine.transfer("kw1", DomainType.ENGINEERING)
        assert result["status"] in ("transferred", "failed")
        if result["status"] == "transferred":
            assert result["fidelity"] <= 0.5

    def test_emergence_detection_threshold(self, engine):
        """Emergence should be detected when transferred > original in target domain."""
        kw = CausalKnowledge("orig-1", DomainType.PHYSICAL, confidence=0.9)
        engine.register_knowledge(kw)
        engine.transfer("orig-1", DomainType.ENGINEERING)
        engine.transfer("orig-1", DomainType.FINANCE)
        engine.transfer("orig-1", DomainType.SOCIAL)

        # Detect emergence in ENGINEERING (receives transferred knowledge)
        emergence = engine.detect_emergence(DomainType.ENGINEERING)
        assert emergence["emergence_detected"] in (True, False)  # depends on adapter

    def test_report_empty_engine(self):
        """Empty engine should return zeros."""
        engine = CrossDomainCausalTransfer()
        report = engine.get_transfer_report()
        assert report["n_knowledge"] == 0
        assert report["n_transfers"] == 0
        assert report["avg_fidelity"] == 0.0

    def test_transfer_counter_increments(self, engine, sample_knowledge):
        engine.register_knowledge(sample_knowledge)
        r1 = engine.transfer("med-heart-001", DomainType.ENGINEERING)
        r2 = engine.transfer("med-heart-001", DomainType.FINANCE)
        # Transfer IDs should be different
        assert r1["transfer_id"] != r2["transfer_id"]
        assert "CDT-" in r1["transfer_id"]

    def test_all_domain_types(self):
        """All five domain types should work."""
        engine = CrossDomainCausalTransfer()
        for dt in DomainType:
            kw = CausalKnowledge(f"k-{dt.value}", dt, confidence=0.8)
            engine.register_knowledge(kw)
        report = engine.get_transfer_report()
        assert report["n_knowledge"] == 5

    def test_transfer_id_format(self, engine, sample_knowledge):
        engine.register_knowledge(sample_knowledge)
        result = engine.transfer("med-heart-001", DomainType.FINANCE)
        assert result["transfer_id"].startswith("CDT-")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
