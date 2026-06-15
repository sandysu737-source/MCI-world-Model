"""MCI World Model v5.1.0 — P2 SDK 桥接集成测试

P2 SDK 桥接: MultiLLM 适配器 + Orchestrator 桥接。
"""

from __future__ import annotations

from mci_world_model.sdk._multillm_adapter import (
    MultiLLMAdapter,
    register_provider,
)
from mci_world_model.sdk._orchestrator_bridge import (
    AgentResult,
    OrchestratorBridge,
)

# ═══════════════════════════════════════════════════════════════════════════
# MultiLLM 适配器
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiLLMAdapter:
    """MultiLLM 适配器基础功能。"""

    def test_adapter_init(self):
        """适配器可初始化。"""
        adapter = MultiLLMAdapter(providers=["ollama"])
        assert adapter is not None

    def test_adapter_init_multiple_providers(self):
        """多 Provider 初始化。"""
        adapter = MultiLLMAdapter(providers=["ollama", "openai"])
        assert adapter is not None

    def test_adapter_init_empty(self):
        """空 Provider 列表可初始化 (降级模式)。"""
        adapter = MultiLLMAdapter(providers=[])
        assert adapter is not None

    def test_register_provider(self):
        """自定义 provider 可注册。"""

        def my_provider(prompt: str, **kwargs) -> str:
            return f"echo: {prompt}"

        register_provider("my_echo", my_provider)
        adapter = MultiLLMAdapter(providers=["my_echo"])
        result = adapter.generate("hello")
        assert "echo" in result or isinstance(result, str)

    def test_fallback_no_provider(self):
        """无可用 provider 时优雅降级。"""
        adapter = MultiLLMAdapter(providers=[])
        result = adapter.generate("test")
        # 不应崩溃，返回某种回退响应
        assert isinstance(result, str)

    def test_classify(self):
        """classify 方法可调用。"""
        adapter = MultiLLMAdapter(providers=[])
        result = adapter.classify("患者白蛋白 28g/L", ["低风险", "中风险", "高风险"])
        assert isinstance(result, dict)

    def test_embed(self):
        """embed 方法返回向量。"""
        import numpy as np

        adapter = MultiLLMAdapter(providers=[])
        vec = adapter.embed("测试文本")
        assert isinstance(vec, np.ndarray)

    def test_health_check(self):
        """health_check 返回状态字典。"""
        adapter = MultiLLMAdapter(providers=[])
        status = adapter.health_check()
        assert isinstance(status, dict)

    def test_importable_from_sdk(self):
        """可从 sdk 顶层导入。"""
        from mci_world_model.sdk import (
            MultiLLMAdapter,
            OllamaProvider,
            OpenAIProvider,
        )

        assert MultiLLMAdapter is not None
        assert OllamaProvider is not None
        assert OpenAIProvider is not None


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator 桥接
# ═══════════════════════════════════════════════════════════════════════════


class TestOrchestratorBridge:
    """Orchestrator 桥接基础功能。"""

    def test_bridge_init(self):
        """桥接可初始化。"""
        bridge = OrchestratorBridge()
        assert bridge is not None

    def test_bridge_with_world_model(self):
        """带世界模型的桥接。"""
        from mci_world_model.sdk import MCIWorldModel

        wm = MCIWorldModel()
        bridge = OrchestratorBridge(world_model=wm)
        assert bridge is not None

    def test_execute_intent_returns_agent_result(self):
        """execute_intent 返回 AgentResult。"""
        bridge = OrchestratorBridge()
        result = bridge.execute_intent("SCREENING", params={})
        assert isinstance(result, AgentResult)

    def test_execute_intent_unknown(self):
        """未知意图类型仍返回 AgentResult。"""
        bridge = OrchestratorBridge()
        result = bridge.execute_intent("UNKNOWN_INTENT", params={})
        assert isinstance(result, AgentResult)

    def test_agent_result_dataclass(self):
        """AgentResult 有正确字段。"""
        result = AgentResult(success=True, intent_type="SCREENING")
        assert result.success is True
        assert result.intent_type == "SCREENING"

    def test_importable_from_sdk(self):
        """可从 sdk 顶层导入。"""
        from mci_world_model.sdk import AgentResult, OrchestratorBridge

        assert OrchestratorBridge is not None
        assert AgentResult is not None


# ═══════════════════════════════════════════════════════════════════════════
# 集成: MultiLLM + Orchestrator
# ═══════════════════════════════════════════════════════════════════════════


class TestP2Integration:
    """P2 SDK 桥接集成测试。"""

    def test_multillm_with_bridge(self):
        """MultiLLM + Orchestrator 联合初始化。"""
        adapter = MultiLLMAdapter(providers=[])
        bridge = OrchestratorBridge(multillm=adapter)
        assert bridge is not None

    def test_bridge_intent_map_complete(self):
        """4 种核心意图均可执行。"""
        bridge = OrchestratorBridge()
        intents = ["SCREENING", "ASSESSMENT", "PLAN_GENERATION", "FOLLOWUP"]
        for intent in intents:
            result = bridge.execute_intent(intent, params={})
            assert isinstance(result, AgentResult)
