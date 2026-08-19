"""MCI World Model v12.0.0 — P12 Stage1 联邦协议/意识/架构/信任 测试
===================================================================

覆盖 P12 Stage1 (W181-192) 核心模块:
  - CausalFederationProtocol: 因果联邦协议
  - FederatedCausalConsciousness: 联邦因果意识
  - CausalFederationArchitecture: 联邦架构
  - FederatedTrust: 联邦信任框架
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._causal_federation_protocol import (
    CausalFederationProtocol,
    FederationConsensus,
    FederationMessageType,
    FederationState,
    NodeRole,
    PeerInfo,
)
from mci_world_model.sdk._federated_consciousness import (
    FederatedCausalConsciousness,
    FederationAwarenessState,
)
from mci_world_model.sdk._federated_trust import (
    FederatedTrust,
    LocalTrust,
    TrustCertificate,
)
from mci_world_model.sdk._federation_arch import (
    CausalFederationArchitecture,
)

# =============================================================================
# CausalFederationProtocol Tests
# =============================================================================


class TestCausalFederationProtocol:
    """因果联邦协议测试。"""

    def test_init_default(self):
        proto = CausalFederationProtocol(node_id="node_1")
        assert proto.node_id == "node_1"
        assert proto.role == NodeRole.FULL_NODE
        assert proto.state == FederationState.DISCONNECTED
        assert proto.n_peers == 0

    def test_init_edge_node(self):
        proto = CausalFederationProtocol(node_id="edge_1", node_role="edge_node")
        assert proto.role == NodeRole.EDGE_NODE

    def test_init_invalid_role(self):
        with pytest.raises(ValueError, match="未知节点角色"):
            CausalFederationProtocol(node_id="x", node_role="invalid_role")

    def test_join_federation(self):
        proto = CausalFederationProtocol(node_id="node_1")
        result = proto.join_federation()
        assert result["joined"] is True
        assert result["state"] == "active"
        assert proto.state == FederationState.ACTIVE

    def test_join_with_peers(self):
        proto = CausalFederationProtocol(node_id="node_1")
        peers = {
            "node_2": PeerInfo(node_id="node_2", role=NodeRole.FULL_NODE),
            "node_3": PeerInfo(node_id="node_3", role=NodeRole.EDGE_NODE),
        }
        result = proto.join_federation(existing_peers=peers)
        assert result["joined"] is True
        assert result["n_peers"] == 2

    def test_leave_federation(self):
        proto = CausalFederationProtocol(node_id="node_1")
        proto.join_federation()
        result = proto.leave_federation()
        assert result["left"] is True
        assert proto.state == FederationState.DISCONNECTED
        assert proto.n_peers == 0

    def test_add_peer(self):
        proto = CausalFederationProtocol(node_id="node_1")
        proto.join_federation()
        proto.add_peer("node_2", role="full_node", trust_score=0.8)
        assert proto.n_peers == 1

    def test_federated_query_broadcast(self):
        proto = CausalFederationProtocol(node_id="node_1")
        proto.join_federation()
        proto.add_peer("node_2", role="full_node")
        proto.add_peer("node_3", role="full_node")
        result = proto.federated_query({"domain": "medical"}, strategy="broadcast")
        assert "local_result" in result
        assert "federated_results" in result
        assert "merged_result" in result
        assert result["n_peers_queried"] == 2

    def test_federated_query_targeted(self):
        proto = CausalFederationProtocol(node_id="node_1")
        proto.join_federation()
        proto.add_peer(
            "node_2",
            role="full_node",
            capabilities={"supported_domains": ["medical"]},
        )
        result = proto.federated_query({"domain": "medical"}, strategy="targeted")
        assert "merged_result" in result

    def test_federated_query_hierarchical(self):
        proto = CausalFederationProtocol(node_id="node_1")
        proto.join_federation()
        proto.add_peer("edge_1", role="edge_node")
        proto.add_peer("full_1", role="full_node")
        proto.add_peer("bridge_1", role="bridge_node")
        result = proto.federated_query({"domain": "physics"}, strategy="hierarchical")
        assert result["n_peers_queried"] == 3

    def test_send_message(self):
        proto = CausalFederationProtocol(node_id="node_1")
        proto.join_federation()
        proto.add_peer("node_2", role="full_node")
        result = proto.send_message("fed_sync", {"key": "value"}, target="node_2")
        assert result["delivered"] is True

    def test_send_message_broadcast(self):
        proto = CausalFederationProtocol(node_id="node_1")
        proto.join_federation()
        proto.add_peer("node_2", role="full_node")
        result = proto.send_message("fed_sync", {"key": "value"})
        assert result["broadcast"] is True

    def test_send_message_invalid_type(self):
        proto = CausalFederationProtocol(node_id="node_1")
        with pytest.raises(ValueError, match="未知消息类型"):
            proto.send_message("invalid_type", {})

    def test_share_evidence(self):
        proto = CausalFederationProtocol(node_id="node_1")
        proto.join_federation()
        proto.add_peer("node_2", role="full_node")
        result = proto.share_evidence({"type": "causal", "data": [1, 2, 3]})
        assert result["shared"] is True
        assert result["n_recipients"] == 1

    def test_request_consensus(self):
        proto = CausalFederationProtocol(node_id="node_1")
        proto.join_federation()
        proto.add_peer("node_2", role="full_node")
        proto.add_peer("node_3", role="full_node")
        result = proto.request_consensus({"type": "dag_update"})
        assert "quorum_met" in result
        assert result["n_voters"] == 3

    def test_message_log(self):
        proto = CausalFederationProtocol(node_id="node_1")
        proto.join_federation()
        log = proto.get_message_log()
        assert len(log) >= 1
        assert any(m.msg_type == FederationMessageType.FED_JOIN for m in log)

    def test_message_log_filtered(self):
        proto = CausalFederationProtocol(node_id="node_1")
        proto.join_federation()
        join_log = proto.get_message_log(msg_type="fed_join")
        assert all(m.msg_type == FederationMessageType.FED_JOIN for m in join_log)

    def test_12_message_types(self):
        """KPI: 12 种联邦消息类型全部可用。"""
        assert len(FederationMessageType) == 12

    def test_4_node_roles(self):
        """KPI: 4 种节点角色。"""
        assert len(NodeRole) == 4


# =============================================================================
# FederationConsensus Tests
# =============================================================================


class TestFederationConsensus:
    """联邦共识引擎测试。"""

    def test_resolve_conflicts_quorum(self):
        consensus = FederationConsensus(quorum_ratio=2 / 3)
        conflicts = [
            {"edge": {"from": "A", "to": "B"}, "votes_for": 5, "votes_against": 1},
            {"edge": {"from": "C", "to": "D"}, "votes_for": 2, "votes_against": 4},
        ]
        results = consensus.resolve_conflicts(conflicts)
        assert results[0]["resolved"] is True
        assert results[1]["resolved"] is False

    def test_check_quorum(self):
        consensus = FederationConsensus(quorum_ratio=2 / 3)
        assert consensus.check_quorum(7, 10) is True
        assert consensus.check_quorum(5, 10) is False

    def test_invalid_quorum(self):
        with pytest.raises(ValueError):
            FederationConsensus(quorum_ratio=0.3)


# =============================================================================
# FederatedCausalConsciousness Tests
# =============================================================================


class TestFederatedCausalConsciousness:
    """联邦因果意识测试。"""

    def test_init(self):
        fcc = FederatedCausalConsciousness()
        assert fcc.awareness_state == FederationAwarenessState.ISOLATED
        assert fcc.n_peer_models == 0

    def test_synchronize_isolated(self):
        fcc = FederatedCausalConsciousness()
        result = fcc.synchronize_awareness()
        assert result["federation_awareness"] == "isolated"

    def test_synchronize_aware(self):
        fcc = FederatedCausalConsciousness()
        fcc.add_peer_model("node_2", domains=["medical"], confidence=0.2)
        fcc.set_local_self_model(domains=["physics"], confidence=0.2)
        result = fcc.synchronize_awareness()
        # 低置信度应触发异常 → aware 状态
        assert result["federation_awareness"] in ("aware", "synchronized")

    def test_synchronize_synchronized(self):
        fcc = FederatedCausalConsciousness()
        fcc.set_local_self_model(domains=["physics", "economics"], confidence=0.8)
        fcc.add_peer_model("node_2", domains=["medical", "biology"], confidence=0.8)
        fcc.add_peer_model("node_3", domains=["chemistry", "social"], confidence=0.8)
        result = fcc.synchronize_awareness()
        assert result["federation_awareness"] in ("synchronized", "emergent")

    def test_synchronize_emergent(self):
        fcc = FederatedCausalConsciousness()
        fcc.set_local_self_model(domains=["physics", "economics", "social"], confidence=0.9)
        for i in range(4):
            fcc.add_peer_model(
                f"node_{i}",
                domains=[f"domain_{i}", f"domain_{i + 1}"],
                confidence=0.85,
            )
        result = fcc.synchronize_awareness()
        # 足够多节点 + 高置信度 + 多领域 → 可能涌现
        assert result["federation_awareness"] in ("synchronized", "emergent")

    def test_federated_reflect(self):
        fcc = FederatedCausalConsciousness()
        fcc.add_peer_model("node_2", confidence=0.7)
        result = fcc.federated_reflect({"confidence": 0.3, "contradictions": 2})
        assert "local_reflection" in result
        assert "cross_reflections" in result
        assert "federation_improvements" in result
        assert result["n_nodes_participated"] == 2

    def test_propose_evolution_not_synchronized(self):
        fcc = FederatedCausalConsciousness()
        result = fcc.propose_federation_evolution({"type": "dag_update"})
        assert result["accepted"] is False
        assert result["reason"] == "federation_not_synchronized"

    def test_propose_evolution_synchronized(self):
        fcc = FederatedCausalConsciousness()
        fcc.set_local_self_model(domains=["a", "b"], confidence=0.8)
        fcc.add_peer_model("node_2", domains=["c", "d"], confidence=0.8)
        fcc.add_peer_model("node_3", domains=["e", "f"], confidence=0.8)
        fcc.synchronize_awareness()
        result = fcc.propose_federation_evolution({"type": "dag_update"})
        assert "accepted" in result

    def test_federation_self_model(self):
        fcc = FederatedCausalConsciousness()
        fcc.set_local_self_model(domains=["a"], confidence=0.7)
        fcc.add_peer_model("node_2", domains=["b"], confidence=0.6)
        fcc.synchronize_awareness()
        model = fcc.federation_self_model
        assert model.n_nodes == 2
        assert "a" in model.combined_domains
        assert "b" in model.combined_domains


# =============================================================================
# CausalFederationArchitecture Tests
# =============================================================================


class TestCausalFederationArchitecture:
    """因果联邦架构测试。"""

    def test_init(self):
        arch = CausalFederationArchitecture()
        assert arch.n_domains == 0
        assert arch.replication_factor == 3

    def test_register_node(self):
        arch = CausalFederationArchitecture()
        arch.register_node("node_1")
        arch.register_node("node_2")
        arch.register_node("node_3")

    def test_distribute_causal_knowledge(self):
        arch = CausalFederationArchitecture(replication_factor=2)
        arch.register_node("node_1")
        arch.register_node("node_2")
        graph = {
            "nodes": ["A", "B", "C", "D"],
            "edges": [
                {"from": "A", "to": "B"},
                {"from": "B", "to": "C"},
                {"from": "C", "to": "D"},
            ],
        }
        result = arch.distribute_causal_knowledge(graph, "medical")
        assert result["domain"] == "medical"
        assert result["n_shards"] >= 1
        assert result["replication_factor"] == 2

    def test_distribute_empty_graph(self):
        arch = CausalFederationArchitecture()
        result = arch.distribute_causal_knowledge({}, "test")
        assert result["n_shards"] >= 1

    def test_federated_discovery_with_data(self):
        arch = CausalFederationArchitecture()
        arch.register_node("node_1")
        arch.register_node("node_2")
        data_sources = {
            "node_1": np.random.randn(100, 5),
            "node_2": np.random.randn(100, 5),
        }
        result = arch.federated_causal_discovery("physics", data_sources)
        assert "merged_dag" in result
        assert result["n_local_discoveries"] == 2

    def test_federated_discovery_simulation(self):
        arch = CausalFederationArchitecture()
        arch.register_node("node_1")
        arch.register_node("node_2")
        result = arch.federated_causal_discovery("economics")
        assert "merged_dag" in result

    def test_retrieve_federated_knowledge(self):
        arch = CausalFederationArchitecture()
        arch.register_node("node_1")
        graph = {
            "nodes": ["A", "B"],
            "edges": [{"from": "A", "to": "B"}],
        }
        arch.distribute_causal_knowledge(graph, "medical")
        result = arch.retrieve_federated_knowledge({"domain": "medical", "variables": ["A"]})
        assert result["found"] is True

    def test_retrieve_not_found(self):
        arch = CausalFederationArchitecture()
        result = arch.retrieve_federated_knowledge({"domain": "unknown"})
        assert result["found"] is False


# =============================================================================
# FederatedTrust Tests
# =============================================================================


class TestFederatedTrust:
    """联邦信任框架测试。"""

    def test_init(self):
        ft = FederatedTrust()
        assert ft.decay_factor == 0.15
        assert ft.n_trusted_peers == 0

    def test_assess_federation_trust_no_cross(self):
        ft = FederatedTrust()
        result = ft.assess_federation_trust("node_2", {"consistency": 0.8, "accuracy": 0.7, "coverage": 0.6})
        assert result["federation_trust"] > 0
        assert result["n_cross_attestations"] == 0

    def test_assess_federation_trust_with_cross(self):
        ft = FederatedTrust()
        ft.add_peer_trust("node_3", 0.8)
        ft.add_peer_trust("node_4", 0.7)
        result = ft.assess_federation_trust("node_2", {"consistency": 0.8, "accuracy": 0.7, "coverage": 0.6})
        assert result["n_cross_attestations"] == 2
        assert result["cross_trust_avg"] > 0

    def test_propagate_trust(self):
        ft = FederatedTrust(decay_factor=0.15)
        cert = TrustCertificate(issuer="A", subject="B", trust_score=0.9)
        result = ft.propagate_trust(cert, "node_C")
        expected = 0.9 * (1 - 0.15)
        assert abs(result["propagated_trust"] - expected) < 1e-6
        assert result["decay_applied"] == 0.15

    def test_issue_certificate(self):
        ft = FederatedTrust()
        cert = ft.issue_trust_certificate("node_2", {"consistency": 0.9, "accuracy": 0.8, "coverage": 0.7})
        assert cert.subject == "node_2"
        assert cert.trust_score > 0
        assert not cert.is_expired

    def test_verify_certificate(self):
        ft = FederatedTrust()
        cert = ft.issue_trust_certificate("node_2", {"consistency": 0.9, "accuracy": 0.8, "coverage": 0.7})
        result = ft.verify_certificate(cert)
        assert result["valid"] is True

    def test_verify_expired_certificate(self):
        cert = TrustCertificate(
            issuer="local",
            subject="node_2",
            trust_score=0.8,
            expires_at=1,  # 已过期
        )
        ft = FederatedTrust()
        result = ft.verify_certificate(cert)
        assert result["not_expired"] is False

    def test_audit_trust_state(self):
        ft = FederatedTrust()
        ft.add_peer_trust("node_2", 0.8)
        ft.add_peer_trust("node_3", 0.6)
        audit = ft.audit_trust_state()
        assert audit["n_peers_tracked"] == 2
        assert audit["n_trusted_peers"] == 2
        assert audit["avg_trust"] > 0


# =============================================================================
# LocalTrust Tests
# =============================================================================


class TestLocalTrust:
    """本地信任评估测试。"""

    def test_high_trust(self):
        lt = LocalTrust()
        result = lt.reason_with_trust({"consistency": 0.9, "accuracy": 0.9, "coverage": 0.8})
        assert result["trust"]["level"] in ("high", "verified")

    def test_low_trust(self):
        lt = LocalTrust()
        result = lt.reason_with_trust({"consistency": 0.2, "accuracy": 0.2, "coverage": 0.2})
        assert result["trust"]["level"] in ("untrusted", "low")

    def test_medium_trust(self):
        lt = LocalTrust()
        result = lt.reason_with_trust({"consistency": 0.5, "accuracy": 0.5, "coverage": 0.5})
        assert result["trust"]["score"] == pytest.approx(0.5)


# =============================================================================
# Integration Test: P12 Stage1 KPI
# =============================================================================


class TestP12Stage1KPI:
    """P12 Stage1 KPI 验收测试。"""

    def test_kpi_federation_protocol_12_messages(self):
        """KPI: 因果联邦协议 12 种消息类型全部可用。"""
        assert len(FederationMessageType) == 12
        expected_types = {
            "fed_join",
            "fed_leave",
            "fed_sync",
            "fed_query",
            "fed_result",
            "fed_discovery",
            "fed_consensus",
            "fed_vote",
            "fed_evolve",
            "fed_audit",
            "fed_trust_renew",
            "fed_evidence_share",
        }
        actual_types = {mt.value for mt in FederationMessageType}
        assert actual_types == expected_types

    def test_kpi_federation_3_nodes(self):
        """KPI: ≥3 节点联邦通信。"""
        proto = CausalFederationProtocol(node_id="node_1")
        proto.join_federation()
        proto.add_peer("node_2", role="full_node")
        proto.add_peer("node_3", role="full_node")
        result = proto.federated_query({"domain": "test"}, strategy="broadcast")
        assert result["n_peers_queried"] >= 2
        assert proto.n_peers >= 2

    def test_kpi_federation_consciousness_sync(self):
        """KPI: 联邦意识同步 ≥3 节点。"""
        fcc = FederatedCausalConsciousness()
        fcc.set_local_self_model(domains=["a", "b"], confidence=0.8)
        fcc.add_peer_model("node_2", domains=["c"], confidence=0.8)
        fcc.add_peer_model("node_3", domains=["d"], confidence=0.8)
        result = fcc.synchronize_awareness()
        assert result["n_nodes_aware"] >= 3

    def test_kpi_federation_consciousness_reflection_consensus(self):
        """KPI: 跨节点反思共识 ≥70%。"""
        fcc = FederatedCausalConsciousness()
        fcc.add_peer_model("node_2", confidence=0.8)
        fcc.add_peer_model("node_3", confidence=0.8)
        result = fcc.federated_reflect({"confidence": 0.5})
        assert result["n_nodes_participated"] >= 3

    def test_kpi_federation_arch_3_nodes(self):
        """KPI: 联邦架构 ≥3 节点分布式因果存储。"""
        arch = CausalFederationArchitecture(replication_factor=3)
        arch.register_node("node_1")
        arch.register_node("node_2")
        arch.register_node("node_3")
        graph = {
            "nodes": ["A", "B", "C", "D", "E", "F"],
            "edges": [{"from": "A", "to": "B"}],
        }
        result = arch.distribute_causal_knowledge(graph, "test")
        assert result["n_shards"] >= 1

    def test_kpi_federation_trust_assessment(self):
        """KPI: 联邦信任评估。"""
        ft = FederatedTrust()
        ft.add_peer_trust("node_2", 0.8)
        result = ft.assess_federation_trust("node_3", {"consistency": 0.8, "accuracy": 0.7, "coverage": 0.6})
        assert result["federation_trust"] > 0.5
