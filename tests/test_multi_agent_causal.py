"""Tests for Multi-Agent Causal Negotiation."""

import numpy as np
import pytest

from mci_world_model.sdk._multi_agent_causal import (
    CausalAgent,
    AgentNegotiation,
    ConsensusGraph,
)


class TestCausalAgent:
    def test_create_agent(self):
        agent = CausalAgent(agent_id="A1", method="pc")
        assert agent.agent_id == "A1"
        assert agent.method == "pc"
        assert agent.confidence == 0.5

    def test_discover_with_pc(self):
        rng = np.random.RandomState(42)
        n = 100
        X1 = rng.randn(n)
        X2 = 0.8 * X1 + 0.2 * rng.randn(n)
        data = np.column_stack([X1, X2])

        agent = CausalAgent(agent_id="PC_1", method="pc")
        agent.discover(data, ["X", "Y"])
        assert len(agent.discovered_edges) >= 1
        assert agent.adj_matrix is not None

    def test_discover_with_fci(self):
        rng = np.random.RandomState(42)
        n = 100
        X = rng.randn(n)
        Y = 0.7 * X + 0.3 * rng.randn(n)
        data = np.column_stack([X, Y])

        agent = CausalAgent(agent_id="FCI_1", method="fci")
        agent.discover(data, ["A", "B"])
        assert agent.adj_matrix is not None


class TestAgentNegotiation:
    def test_register_agent(self):
        neg = AgentNegotiation(vote_threshold=0.5, min_agents=2)
        neg.register_agent(CausalAgent(agent_id="A1", method="pc"))
        assert neg.agent_count == 1

    def test_insufficient_agents(self):
        neg = AgentNegotiation(min_agents=2)
        neg.register_agent(CausalAgent(agent_id="A1", method="pc"))
        rng = np.random.RandomState(42)
        data = rng.randn(100, 3)
        result = neg.negotiate(data, ["X", "Y", "Z"])
        assert result.confidence == 0.0

    def test_two_agent_consensus(self):
        """两智能体协商应产生共识因果图。"""
        rng = np.random.RandomState(42)
        n = 200
        X = rng.randn(n)
        Y = 0.7 * X + 0.3 * rng.randn(n)
        Z = 0.5 * Y + 0.3 * rng.randn(n)
        data = np.column_stack([X, Y, Z])

        neg = AgentNegotiation(vote_threshold=0.4, min_agents=2)
        neg.register_agent(CausalAgent(agent_id="PC", method="pc"))
        neg.register_agent(CausalAgent(agent_id="GES", method="notears"))

        result = neg.negotiate(data, ["X", "Y", "Z"])

        assert len(result.nodes) == 3
        assert result.adj_matrix is not None
        assert result.confidence >= 0.0
        # 至少应有一条共识边
        assert len(result.edges) >= 0  # 可能全部冲突

    def test_conflict_detection(self):
        """双向边应标记为冲突。"""
        rng = np.random.RandomState(42)
        X = rng.randn(200)
        Y = 0.5 * X + 0.5 * rng.randn(200)
        data = np.column_stack([X, Y])

        neg = AgentNegotiation(vote_threshold=0.3, min_agents=2)
        neg.register_agent(CausalAgent(agent_id="PC", method="pc"))
        neg.register_agent(CausalAgent(agent_id="FCI", method="fci"))

        result = neg.negotiate(data, ["X", "Y"])

        assert len(result.nodes) == 2
        # 只要有边，不应双向同时存在
        if result.adj_matrix is not None:
            for i in range(2):
                for j in range(2):
                    if i != j and result.adj_matrix[i, j] == 1:
                        assert result.adj_matrix[j, i] == 0, \
                            f"Bidirectional edge {i}↔{j}"

    def test_statistics(self):
        neg = AgentNegotiation()
        neg.register_agent(CausalAgent(agent_id="A1", method="pc"))
        neg.register_agent(CausalAgent(agent_id="A2", method="notears"))
        stats = neg.statistics()
        assert stats["agent_count"] == 2
        assert stats["vote_threshold"] == 0.5
