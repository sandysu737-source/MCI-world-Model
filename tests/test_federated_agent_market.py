"""End-to-end tests for FederatedAgentMarket — P12 multi-agent causal marketplace."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from mci_world_model.sdk._federated_agent_market import (
    AgentSpec,
    FederatedAgentMarket,
    TradeRecord,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def market():
    return FederatedAgentMarket(min_trust_for_trade=0.5)


@pytest.fixture
def sample_agent():
    return AgentSpec(
        name="CausalHeart",
        provider="node-001",
        domains=["medical", "cardiovascular"],
        capabilities=["causal_discovery", "counterfactual"],
        trust_score=0.85,
    )


@pytest.fixture
def low_trust_agent():
    return AgentSpec(
        name="UnreliableAgent",
        provider="node-002",
        domains=["finance"],
        capabilities=["correlation_only"],
        trust_score=0.3,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AgentSpec tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentSpec:
    def test_creation_with_id(self):
        a = AgentSpec(name="Test", provider="node-1")
        assert a.agent_id != ""
        assert len(a.agent_id) == 12

    def test_creation_with_provided_id(self):
        a = AgentSpec(agent_id="my-agent-id", name="Test", provider="node-1")
        assert a.agent_id == "my-agent-id"

    def test_domains_default_empty(self):
        a = AgentSpec(name="A", provider="p")
        assert a.domains == []

    def test_capabilities_list(self):
        a = AgentSpec(name="A", provider="p", capabilities=["c1", "c2"])
        assert a.capabilities == ["c1", "c2"]

    def test_default_trust_score(self):
        a = AgentSpec(name="A", provider="p")
        assert a.trust_score == 0.5


class TestTradeRecord:
    def test_creation_with_auto_id(self):
        t = TradeRecord(agent_id="a1", provider="p1", consumer="c1")
        assert t.trade_id != ""
        assert len(t.trade_id) == 12

    def test_creation_different_ids(self):
        t1 = TradeRecord(agent_id="a1", provider="p1", consumer="c1")
        t2 = TradeRecord(agent_id="a1", provider="p1", consumer="c2")
        assert t1.trade_id != t2.trade_id


# ═══════════════════════════════════════════════════════════════════════════════
# FederatedAgentMarket tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarketRegistration:
    def test_register_agent(self, market, sample_agent):
        result = market.register_agent(sample_agent)
        assert result["registered"] is True
        assert market.n_agents == 1

    def test_register_duplicate(self, market, sample_agent):
        market.register_agent(sample_agent)
        result = market.register_agent(sample_agent)
        assert result["registered"] is False
        assert result["reason"] == "agent_already_exists"

    def test_registration_initializes_ratings(self, market, sample_agent):
        market.register_agent(sample_agent)
        assert sample_agent.n_ratings == 0
        assert sample_agent.rating == 0.0


class TestMarketDiscovery:
    def test_discover_by_domain(self, market, sample_agent):
        market.register_agent(sample_agent)
        results = market.discover_agents({"domain": "medical"})
        assert len(results) == 1
        assert results[0].name == "CausalHeart"

    def test_discover_by_capability(self, market, sample_agent):
        market.register_agent(sample_agent)
        results = market.discover_agents({"capability": "counterfactual"})
        assert len(results) == 1

    def test_discover_no_match_domain(self, market, sample_agent):
        market.register_agent(sample_agent)
        results = market.discover_agents({"domain": "agriculture"})
        assert len(results) == 0

    def test_discover_min_trust_filters(self, market, sample_agent, low_trust_agent):
        market.register_agent(sample_agent)      # trust=0.85
        market.register_agent(low_trust_agent)   # trust=0.3
        results = market.discover_agents({"min_trust": 0.7})
        assert len(results) == 1
        assert results[0].name == "CausalHeart"

    def test_discover_empty_market(self, market):
        results = market.discover_agents({})
        assert results == []

    def test_discover_sorted_by_rating(self, market):
        a1 = AgentSpec(name="A", provider="p", rating=4.0, n_ratings=5)
        a2 = AgentSpec(name="B", provider="p", rating=4.5, n_ratings=3)
        market.register_agent(a1)
        market.register_agent(a2)
        results = market.discover_agents({})
        assert results[0].name == "B"  # higher rating first
        assert results[1].name == "A"


class TestMarketTrading:
    def test_trade_success(self, market, sample_agent):
        market.register_agent(sample_agent)
        result = market.trade_agent(sample_agent.agent_id, "consumer-1")
        assert result["traded"] is True
        assert market.n_trades == 1

    def test_trade_trust_below_threshold(self, market, low_trust_agent):
        market.register_agent(low_trust_agent)  # trust=0.3 < threshold=0.5
        result = market.trade_agent(low_trust_agent.agent_id, "consumer-1")
        assert result["traded"] is False
        assert result["reason"] == "trust_below_threshold"

    def test_trade_agent_not_found(self, market):
        result = market.trade_agent("nonexistent", "consumer-1")
        assert result["traded"] is False
        assert result["reason"] == "agent_not_found"


class TestMarketRating:
    def test_rate_success(self, market, sample_agent):
        market.register_agent(sample_agent)
        result = market.rate_agent(sample_agent.agent_id, 4.0)
        assert result["rated"] is True
        assert result["new_avg_rating"] == 4.0
        assert result["n_ratings"] == 1

    def test_rate_multiple_updates_average(self, market, sample_agent):
        market.register_agent(sample_agent)
        market.rate_agent(sample_agent.agent_id, 4.0)
        market.rate_agent(sample_agent.agent_id, 2.0)
        assert sample_agent.rating == pytest.approx(3.0)
        assert sample_agent.n_ratings == 2

    def test_rate_clamped_to_range(self, market, sample_agent):
        market.register_agent(sample_agent)
        r1 = market.rate_agent(sample_agent.agent_id, 10.0)  # clamped to 5
        assert r1["new_avg_rating"] == 5.0
        r2 = market.rate_agent(sample_agent.agent_id, -3.0)  # clamped to 0
        assert r2["new_avg_rating"] == pytest.approx(2.5)  # (5+0)/2

    def test_rate_nonexistent_agent(self, market):
        result = market.rate_agent("nonexistent", 4.0)
        assert result["rated"] is False
        assert result["reason"] == "agent_not_found"


class TestMarketStatistics:
    def test_statistics_empty(self, market):
        stats = market.market_statistics()
        assert stats["n_agents"] == 0
        assert stats["n_trades"] == 0
        assert stats["avg_rating"] == 0.0

    def test_statistics_with_data(self, market, sample_agent):
        market.register_agent(sample_agent)
        market.rate_agent(sample_agent.agent_id, 4.5)
        stats = market.market_statistics()
        assert stats["n_agents"] == 1
        assert stats["avg_rating"] == 4.5

    def test_top_domains(self, market):
        a1 = AgentSpec(name="A", provider="p", domains=["medical", "finance"])
        a2 = AgentSpec(name="B", provider="p", domains=["medical", "engineering"])
        a3 = AgentSpec(name="C", provider="p", domains=["medical"])
        market.register_agent(a1)
        market.register_agent(a2)
        market.register_agent(a3)
        stats = market.market_statistics()
        top = stats["top_domains"]
        assert top[0] == "medical"  # most common


class TestMarketIntegration:
    def test_register_discover_trade_rate_flow(self, market):
        """Full lifecycle: register → discover → trade → rate."""
        agent = AgentSpec(
            name="CausalFinance",
            provider="node-001",
            domains=["finance", "risk"],
            capabilities=["causal_discovery", "do_calculus"],
            trust_score=0.9,
        )

        # Register
        r1 = market.register_agent(agent)
        assert r1["registered"]
        assert market.n_agents == 1

        # Discover
        results = market.discover_agents({"domain": "finance", "min_trust": 0.8})
        assert len(results) == 1
        assert results[0].name == "CausalFinance"

        # Trade
        r2 = market.trade_agent(agent.agent_id, "bank-node-001")
        assert r2["traded"]
        assert market.n_trades == 1

        # Rate
        r3 = market.rate_agent(agent.agent_id, 4.8)
        assert r3["rated"]
        assert agent.rating == 4.8

        # Statistics
        stats = market.market_statistics()
        assert stats["n_agents"] == 1
        assert stats["n_trades"] == 1
        assert stats["avg_rating"] == 4.8

    def test_multi_agent_multi_trade(self, market):
        """Multiple agents, multiple trades, multiple ratings."""
        for i in range(5):
            agent = AgentSpec(
                name=f"Agent-{i}",
                provider=f"node-{i % 3}",
                domains=["medical"] if i % 2 == 0 else ["engineering"],
                capabilities=["causal"],
                trust_score=0.7,
            )
            market.register_agent(agent)
            market.trade_agent(agent.agent_id, "consumer-1")
            market.rate_agent(agent.agent_id, 4.0)

        assert market.n_agents == 5
        assert market.n_trades == 5
        stats = market.market_statistics()
        assert stats["n_agents"] == 5
        assert stats["avg_rating"] == 4.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
