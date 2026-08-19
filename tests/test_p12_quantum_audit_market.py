"""MCI World Model v12.0.0 — P12 Stage2-3 量子推理/联邦审计/智能体市场 测试
===========================================================================

覆盖 P12 Stage2-3 (W193-216) 核心模块:
  - QuantumClassicalBridge: 量子经典桥接
  - QuantumCausalInference: 量子因果推理
  - FederationAudit: 联邦审计
  - FederatedAgentMarket: 联邦智能体市场
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._federated_agent_market import (
    AgentSpec,
    FederatedAgentMarket,
)
from mci_world_model.sdk._federation_audit import (
    AuditSeverity,
    AuditStatus,
    FederationAudit,
)
from mci_world_model.sdk._quantum_causal_inference import (
    CausalEffectResult,
    QuantumCausalInference,
)
from mci_world_model.sdk._quantum_classical_bridge import (
    QuantumBackend,
    QuantumCircuit,
    QuantumClassicalBridge,
    QuantumErrorMitigator,
    QuantumResult,
)

# =============================================================================
# QuantumClassicalBridge Tests
# =============================================================================


class TestQuantumClassicalBridge:
    """量子经典桥接层测试。"""

    def test_init(self):
        bridge = QuantumClassicalBridge()
        assert bridge.backend == QuantumBackend.SIMULATOR
        assert bridge.execution_count == 0

    def test_encode_amplitude(self):
        bridge = QuantumClassicalBridge()
        data = np.array([0.5, 0.3, 0.8, 0.1])
        circuit = bridge.encode_classical_to_quantum(data, method="amplitude")
        assert circuit.n_qubits >= 1
        assert circuit.depth > 0

    def test_encode_angle(self):
        bridge = QuantumClassicalBridge()
        data = np.array([0.1, 0.5, 0.9])
        circuit = bridge.encode_classical_to_quantum(data, method="angle")
        assert circuit.n_qubits >= 1

    def test_encode_basis(self):
        bridge = QuantumClassicalBridge()
        data = np.array([0.1, 0.9, 0.1, 0.9])
        circuit = bridge.encode_classical_to_quantum(data, method="basis")
        assert circuit.n_qubits >= 1

    def test_execute(self):
        bridge = QuantumClassicalBridge()
        circuit = QuantumCircuit(n_qubits=3)
        circuit.add_gate("H", [0])
        circuit.add_gate("H", [1])
        circuit.add_gate("H", [2])
        result = bridge.execute_on_quantum_hardware(circuit, n_shots=1024)
        assert result.n_shots == 1024
        assert len(result.counts) > 0
        assert bridge.execution_count == 1

    def test_decode(self):
        bridge = QuantumClassicalBridge()
        circuit = QuantumCircuit(n_qubits=2)
        circuit.add_gate("H", [0])
        result = bridge.execute_on_quantum_hardware(circuit, n_shots=2048)
        decoded = bridge.decode_quantum_to_classical(result)
        assert "probabilities" in decoded
        assert "dominant_state" in decoded
        assert decoded["dominant_probability"] > 0

    def test_quantum_advantage_estimate(self):
        bridge = QuantumClassicalBridge()
        result = bridge.quantum_advantage_estimate({"n_variables": 10, "complexity": "high"})
        assert result["advantage_ratio"] > 1
        assert result["recommended_backend"] in ("simulator", "ibm_quantum")

    def test_quantum_advantage_low_vars(self):
        bridge = QuantumClassicalBridge()
        result = bridge.quantum_advantage_estimate({"n_variables": 3, "complexity": "low"})
        assert result["recommended_backend"] == "simulator"


class TestQuantumErrorMitigator:
    """量子误差缓解测试。"""

    def test_zne(self):
        mitigator = QuantumErrorMitigator(method="zero_noise_extrapolation")
        result = QuantumResult(
            counts={"00": 600, "01": 200, "10": 150, "11": 74},
            n_shots=1024,
        )
        mitigated = mitigator.mitigate(result)
        assert mitigated.n_shots == 1024

    def test_readout(self):
        mitigator = QuantumErrorMitigator(method="readout_mitigation")
        result = QuantumResult(
            counts={"00": 900, "11": 100},
            n_shots=1000,
        )
        mitigated = mitigator.mitigate(result)
        assert mitigated.n_shots == 1000

    def test_none(self):
        mitigator = QuantumErrorMitigator(method="none")
        result = QuantumResult(counts={"0": 500, "1": 500}, n_shots=1000)
        mitigated = mitigator.mitigate(result)
        assert mitigated.counts == result.counts

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            QuantumErrorMitigator(method="invalid")


class TestQuantumCircuit:
    """量子电路测试。"""

    def test_init(self):
        qc = QuantumCircuit(n_qubits=4)
        assert qc.n_qubits == 4
        assert qc.depth == 0

    def test_add_gate(self):
        qc = QuantumCircuit(n_qubits=2)
        qc.add_gate("H", [0])
        qc.add_gate("CNOT", [0, 1])
        assert qc.depth == 2

    def test_default_shots(self):
        qc = QuantumCircuit(n_qubits=2)
        assert qc.n_shots == 1024


# =============================================================================
# QuantumCausalInference Tests
# =============================================================================


class TestQuantumCausalInference:
    """量子因果推理测试。"""

    def test_init(self):
        qci = QuantumCausalInference()
        assert qci is not None

    def test_causal_effect(self):
        qci = QuantumCausalInference()
        np.random.seed(42)
        data = np.random.randn(100, 2)
        data[:, 1] = 0.8 * data[:, 0] + 0.2 * np.random.randn(100)
        result = qci.quantum_causal_effect("X", "Y", data)
        assert isinstance(result, CausalEffectResult)
        assert result.cause == "X"
        assert result.effect == "Y"

    def test_causal_effect_1d(self):
        qci = QuantumCausalInference()
        data = np.random.randn(100)
        result = qci.quantum_causal_effect("A", "B", data)
        assert result.method == "quantum"

    def test_counterfactual(self):
        qci = QuantumCausalInference()
        factual = {"X": 0.7, "Y": 0.3}
        intervention = {"X": 0.9}
        result = qci.quantum_counterfactual(factual, intervention)
        assert "factual" in result
        assert "intervention" in result
        assert "counterfactual" in result

    def test_causal_discovery(self):
        qci = QuantumCausalInference()
        np.random.seed(42)
        data = np.random.randn(50, 3)
        data[:, 2] = 0.5 * data[:, 0] + 0.3 * data[:, 1] + 0.1 * np.random.randn(50)
        result = qci.quantum_causal_discovery(data, var_names=["A", "B", "C"])
        assert "nodes" in result
        assert "edges" in result
        assert result["n_nodes"] == 3


# =============================================================================
# FederationAudit Tests
# =============================================================================


class TestFederationAudit:
    """联邦审计测试。"""

    def test_init(self):
        audit = FederationAudit()
        assert audit.n_entries == 0

    def test_audit_operation_pass(self):
        audit = FederationAudit()
        entry = audit.audit_federation_operation({"type": "join", "node_id": "n1"})
        assert entry.status == AuditStatus.PASS

    def test_audit_operation_invalid_type(self):
        audit = FederationAudit()
        entry = audit.audit_federation_operation({"type": "hack", "node_id": "n1"})
        assert entry.status == AuditStatus.FAIL

    def test_audit_evolve_without_consensus(self):
        audit = FederationAudit()
        entry = audit.audit_federation_operation(
            {
                "type": "evolve",
                "node_id": "n1",
                "consensus_reached": False,
            }
        )
        assert entry.severity == AuditSeverity.CRITICAL
        assert entry.status == AuditStatus.FAIL

    def test_audit_evolve_with_consensus(self):
        audit = FederationAudit()
        entry = audit.audit_federation_operation(
            {
                "type": "evolve",
                "node_id": "n1",
                "consensus_reached": True,
            }
        )
        assert entry.status == AuditStatus.PASS

    def test_audit_trust_state(self):
        audit = FederationAudit()
        entry = audit.audit_trust_state({"peer_scores": {"n1": 0.8, "n2": 0.7}})
        assert entry.status == AuditStatus.PASS

    def test_audit_trust_low(self):
        audit = FederationAudit()
        entry = audit.audit_trust_state({"peer_scores": {"n1": 0.1}})
        assert entry.status == AuditStatus.FAIL

    def test_audit_consciousness(self):
        audit = FederationAudit()
        entry = audit.audit_consciousness_state(
            {
                "awareness_state": "synchronized",
                "n_nodes": 3,
            }
        )
        assert entry.status == AuditStatus.PASS

    def test_audit_consciousness_inconsistent(self):
        audit = FederationAudit()
        entry = audit.audit_consciousness_state(
            {
                "awareness_state": "synchronized",
                "n_nodes": 1,
            }
        )
        assert entry.status == AuditStatus.FAIL

    def test_generate_report(self):
        audit = FederationAudit()
        audit.audit_federation_operation({"type": "join", "node_id": "n1"})
        audit.audit_federation_operation({"type": "query", "node_id": "n2"})
        report = audit.generate_audit_report()
        assert "report_id" in report
        assert report["n_entries"] == 2
        assert "hash" in report

    def test_max_entries(self):
        audit = FederationAudit(max_entries=5)
        for i in range(10):
            audit.audit_federation_operation({"type": "join", "node_id": f"n{i}"})
        assert audit.n_entries == 5


# =============================================================================
# FederatedAgentMarket Tests
# =============================================================================


class TestFederatedAgentMarket:
    """联邦智能体市场测试。"""

    def test_init(self):
        market = FederatedAgentMarket()
        assert market.n_agents == 0

    def test_register_agent(self):
        market = FederatedAgentMarket()
        spec = AgentSpec(
            name="MedicalCausalAgent",
            provider="node_1",
            domains=["medical"],
            capabilities=["causal_effect", "counterfactual"],
        )
        result = market.register_agent(spec)
        assert result["registered"] is True
        assert market.n_agents == 1

    def test_register_duplicate(self):
        market = FederatedAgentMarket()
        spec = AgentSpec(agent_id="agent_1", name="Test", provider="n1")
        market.register_agent(spec)
        result = market.register_agent(spec)
        assert result["registered"] is False

    def test_discover_by_domain(self):
        market = FederatedAgentMarket()
        market.register_agent(
            AgentSpec(
                name="MedAgent",
                provider="n1",
                domains=["medical"],
            )
        )
        market.register_agent(
            AgentSpec(
                name="PhysAgent",
                provider="n2",
                domains=["physics"],
            )
        )
        results = market.discover_agents({"domain": "medical"})
        assert len(results) == 1
        assert results[0].name == "MedAgent"

    def test_discover_by_capability(self):
        market = FederatedAgentMarket()
        market.register_agent(
            AgentSpec(
                name="Agent1",
                provider="n1",
                capabilities=["causal_effect", "discovery"],
            )
        )
        market.register_agent(
            AgentSpec(
                name="Agent2",
                provider="n2",
                capabilities=["causal_effect"],
            )
        )
        results = market.discover_agents({"capability": "discovery"})
        assert len(results) == 1

    def test_discover_min_trust(self):
        market = FederatedAgentMarket()
        market.register_agent(
            AgentSpec(
                name="Low",
                provider="n1",
                trust_score=0.3,
            )
        )
        market.register_agent(
            AgentSpec(
                name="High",
                provider="n2",
                trust_score=0.9,
            )
        )
        results = market.discover_agents({"min_trust": 0.5})
        assert len(results) == 1
        assert results[0].name == "High"

    def test_trade_agent(self):
        market = FederatedAgentMarket(min_trust_for_trade=0.5)
        market.register_agent(
            AgentSpec(
                agent_id="a1",
                name="Agent",
                provider="n1",
                trust_score=0.8,
            )
        )
        result = market.trade_agent("a1", "consumer_1")
        assert result["traded"] is True
        assert market.n_trades == 1

    def test_trade_not_found(self):
        market = FederatedAgentMarket()
        result = market.trade_agent("nonexistent", "consumer_1")
        assert result["traded"] is False

    def test_trade_low_trust(self):
        market = FederatedAgentMarket(min_trust_for_trade=0.5)
        market.register_agent(
            AgentSpec(
                agent_id="a1",
                name="Agent",
                provider="n1",
                trust_score=0.3,
            )
        )
        result = market.trade_agent("a1", "consumer_1")
        assert result["traded"] is False
        assert result["reason"] == "trust_below_threshold"

    def test_rate_agent(self):
        market = FederatedAgentMarket()
        market.register_agent(AgentSpec(agent_id="a1", name="Agent", provider="n1"))
        result = market.rate_agent("a1", 4.5)
        assert result["rated"] is True
        assert result["new_avg_rating"] == 4.5

    def test_rate_not_found(self):
        market = FederatedAgentMarket()
        result = market.rate_agent("nonexistent", 5.0)
        assert result["rated"] is False

    def test_market_statistics(self):
        market = FederatedAgentMarket()
        result = market.register_agent(
            AgentSpec(
                name="Agent1",
                provider="n1",
                domains=["medical", "physics"],
                trust_score=0.8,
            )
        )
        agent_id = result["agent_id"]
        market.rate_agent(agent_id, 4.0)
        stats = market.market_statistics()
        assert stats["n_agents"] == 1


# =============================================================================
# P12 Stage2-3 KPI Tests
# =============================================================================


class TestP12Stage2KPI:
    """P12 Stage2 KPI 验收测试。"""

    def test_kpi_quantum_bridge_encoding(self):
        """KPI: 3 种量子编码方法全部可用。"""
        bridge = QuantumClassicalBridge()
        data = np.random.randn(10)
        for method in ("amplitude", "angle", "basis"):
            circuit = bridge.encode_classical_to_quantum(data, method=method)
            assert circuit.n_qubits >= 1

    def test_kpi_quantum_causal_effect(self):
        """KPI: 量子因果效应估计可执行。"""
        qci = QuantumCausalInference()
        data = np.random.randn(50, 2)
        result = qci.quantum_causal_effect("X", "Y", data)
        assert result.confidence >= 0

    def test_kpi_quantum_counterfactual(self):
        """KPI: 量子反事实推理可执行。"""
        qci = QuantumCausalInference()
        result = qci.quantum_counterfactual({"X": 0.5}, {"X": 0.9})
        assert "counterfactual" in result

    def test_kpi_federation_audit(self):
        """KPI: 联邦审计体系完整。"""
        audit = FederationAudit()
        audit.audit_federation_operation({"type": "join", "node_id": "n1"})
        audit.audit_trust_state({"peer_scores": {"n1": 0.8}})
        audit.audit_consciousness_state({"awareness_state": "synchronized", "n_nodes": 3})
        report = audit.generate_audit_report()
        assert report["n_entries"] == 3
        assert report["n_critical"] == 0

    def test_kpi_agent_market(self):
        """KPI: 智能体市场 ≥3 智能体注册。"""
        market = FederatedAgentMarket()
        for i in range(3):
            market.register_agent(
                AgentSpec(
                    name=f"Agent_{i}",
                    provider=f"node_{i}",
                    domains=[f"domain_{i}"],
                    trust_score=0.7,
                )
            )
        assert market.n_agents >= 3
        results = market.discover_agents({})
        assert len(results) >= 3
