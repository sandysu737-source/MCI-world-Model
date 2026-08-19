"""P6-P8 波级集成测试 — CEWM闭环 + 认知诊断 + 神经符号融合
==========================================================

P6 "闭环": CEWM闭环 + 因果梯度 + 行动搜索
P7 "诊断": 元认知诊断 + 自主发现 + 负向启发
P8 "融合": 神经符号融合 + LLM桥接 + AGI协议
"""

from __future__ import annotations

from mci_world_model import sdk


class TestP6ClosedLoop:
    """P6 CEWM闭环波次集成测试。"""

    def test_cognitive_loop_bus_exported(self):
        assert "CognitiveLoopBus" in sdk.__all__

    def test_causal_gradient_exported(self):
        assert "CausalGradient" in sdk.__all__

    def test_causal_actor_exported(self):
        assert "CausalActor" in sdk.__all__

    def test_mcts_planner_exported(self):
        assert "MCTSPlanner" in sdk.__all__

    def test_cognitive_loop_bus_instantiable(self):
        cb = sdk.CognitiveLoopBus()
        assert cb is not None


class TestP7Diagnosis:
    """P7 认知诊断波次集成测试。"""

    def test_meta_diagnoser_exported(self):
        assert "MetaDiagnoser" in sdk.__all__

    def test_negative_heuristic_exported(self):
        assert "NegativeHeuristic" in sdk.__all__

    def test_self_repair_exported(self):
        assert "SelfRepairCognition" in sdk.__all__

    def test_meta_diagnoser_instantiable(self):
        md = sdk.MetaDiagnoser()
        assert md is not None


class TestP8Fusion:
    """P8 神经符号融合波次集成测试。"""

    def test_neural_symbolic_fusion_exported(self):
        assert "NeuralSymbolicFusionV2" in sdk.__all__

    def test_llm_bridge_exported(self):
        assert "LLMCEWMBridge" in sdk.__all__

    def test_agi_protocol_exported(self):
        assert "AGIIntegrationProtocol" in sdk.__all__


class TestP6P8KPIs:
    """P6-P8 波级 KPI 验证。"""

    def test_closed_loop_modules(self):
        p6_symbols = [
            s
            for s in sdk.__all__
            if s in ("CognitiveLoopBus", "CognitiveLayer", "CausalGradient", "CausalActor", "MCTSPlanner")
        ]
        assert len(p6_symbols) >= 3

    def test_diagnosis_modules(self):
        p7_symbols = [s for s in sdk.__all__ if s in ("MetaDiagnoser", "NegativeHeuristic", "SelfRepairCognition")]
        assert len(p7_symbols) >= 2

    def test_fusion_modules(self):
        p8_symbols = [
            s for s in sdk.__all__ if s in ("NeuralSymbolicFusionV2", "LLMCEWMBridge", "AGIIntegrationProtocol")
        ]
        assert len(p8_symbols) >= 2
