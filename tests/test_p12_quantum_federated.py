"""P12 '传承' quantum波次测试: QuantumCausalInference + FederatedCausalConsciousness"""

from mci_world_model import sdk


class TestQuantumCausalInference:
    """测试量子因果推理"""

    def setup_method(self):
        self.qci = sdk.QuantumCausalInference()

    def test_instantiation(self):
        assert self.qci is not None

    def test_has_quantum_causal_effect(self):
        assert hasattr(self.qci, "quantum_causal_effect")

    def test_qci_quantum_circuit_alias(self):
        assert sdk.QCIQuantumCircuit is not None

    def test_qci_classical_bridge_alias(self):
        assert sdk.QCIClassicalBridge is not None

    def test_qci_quantum_result_alias(self):
        assert sdk.QCIQuantumResult is not None

    def test_causal_effect_result_creation(self):
        result = sdk.CausalEffectResult("X", "Y", 0.5, 0.95)
        assert result is not None

    def test_export_in_all(self):
        assert "QuantumCausalInference" in sdk.__all__
        assert "CausalEffectResult" in sdk.__all__
        assert "QCIQuantumCircuit" in sdk.__all__


class TestQuantumClassicalBridge:
    """测试量子-经典桥接"""

    def test_qcb_quantum_circuit_alias(self):
        assert sdk.QCBQuantumCircuit is not None

    def test_qcb_classical_bridge_alias(self):
        assert sdk.QCBClassicalBridge is not None

    def test_qcb_quantum_result_alias(self):
        assert sdk.QCBQuantumResult is not None

    def test_encoding_method_enum(self):
        assert len(list(sdk.EncodingMethod)) > 0

    def test_quantum_backend_enum(self):
        assert len(list(sdk.QuantumBackend)) > 0

    def test_quantum_error_mitigator(self):
        qem = sdk.QuantumErrorMitigator()
        assert qem is not None

    def test_export_in_all(self):
        assert "QCBQuantumCircuit" in sdk.__all__
        assert "QuantumErrorMitigator" in sdk.__all__


class TestFederatedCausalConsciousness:
    """测试联邦因果意识"""

    def setup_method(self):
        self.fcc = sdk.FederatedCausalConsciousness()

    def test_instantiation(self):
        assert self.fcc is not None

    def test_federation_awareness_state_enum(self):
        assert len(list(sdk.FederationAwarenessState)) > 0

    def test_federation_self_model_creation(self):
        sm = sdk.FederationSelfModel()
        assert sm is not None

    def test_reflection_result_creation(self):
        rr = sdk.ReflectionResult(
            source="test",
            issues=[],
            improvements=[],
            confidence=0.9,
        )
        assert rr is not None

    def test_export_in_all(self):
        assert "FederatedCausalConsciousness" in sdk.__all__
        assert "FederationAwarenessState" in sdk.__all__
