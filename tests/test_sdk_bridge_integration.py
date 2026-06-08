"""
MCI World Model P2 — SDK 桥接集成测试

覆盖:
- MultiLLMAdapter fallback 链路 (generate/classify/embed/health_check)
- OrchestratorBridge 全意图 (SCREENING/ASSESSMENT/PLAN_GENERATION/FOLLOWUP/MONITORING)
- EnhancedPerception 三层管道 (text→signal→state)
- 端到端: MultiLLM + Orchestrator + Perception 协同

运行: pytest tests/test_sdk_bridge_integration.py -v
"""

from __future__ import annotations

import numpy as np
import pytest


# =============================================================================
# 共享 fixtures
# =============================================================================


@pytest.fixture(scope="module")
def adapter():
    """MultiLLM 适配器 (fallback mode)。"""
    from mci_world_model.sdk._multillm_adapter import MultiLLMAdapter

    return MultiLLMAdapter(providers=["ollama"])


@pytest.fixture(scope="module")
def bridge(adapter):
    """Orchestrator 桥接 (无 MultiLLM 增强)。"""
    from mci_world_model.sdk._orchestrator_bridge import OrchestratorBridge

    return OrchestratorBridge(multillm=adapter)


@pytest.fixture(scope="module")
def perception():
    """增强感知管道。"""
    from mci_world_model.sdk._enhanced_perception import EnhancedPerception

    return EnhancedPerception()


@pytest.fixture(scope="module")
def patient_timeline():
    """30 天合成患者数据。"""
    rng = np.random.default_rng(42)
    timeline = []
    for d in range(1, 31):
        alb = max(25, min(45, 35 + rng.normal(0, 2)))
        timeline.append({
            "day": d,
            "albumin": round(alb, 1),
            "prealbumin": round(rng.uniform(150, 300), 1),
            "nrs2002_score": round(rng.uniform(2, 4), 1),
            "calorie_intake": round(rng.uniform(1200, 1800), 0),
            "body_weight": round(rng.uniform(60, 80), 1),
            "protein_intake": round(rng.uniform(50, 90), 1),
            "medication_dose": round(rng.uniform(100, 300), 1),
        })
    return timeline


# =============================================================================
# MultiLLMAdapter 测试
# =============================================================================


class TestMultiLLMAdapter:
    """MultiLLM 适配器核心功能。"""

    def test_health_check(self, adapter):
        """健康检查返回正确结构。"""
        status = adapter.health_check()
        assert "available" in status
        assert "active_provider" in status
        assert "providers" in status
        assert "mode" in status

    def test_generate_fallback(self, adapter):
        """降级 generate：所有 provider 不可用时返回预设响应。"""
        response = adapter.generate("测试")
        assert isinstance(response, str)
        assert len(response) > 10

    def test_classify_fallback(self, adapter):
        """降级 classify：规则匹配。"""
        result = adapter.classify("患者白蛋白 28，体重下降", ["低风险", "中风险", "高风险"])
        assert "label" in result
        assert "scores" in result
        assert "method" in result
        assert result["label"] in ["低风险", "中风险", "高风险"]

    def test_embed_fallback(self, adapter):
        """降级 embed：字符集嵌入。"""
        emb = adapter.embed("test text", dim=5)
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (5,)

    def test_embed_dimensions(self, adapter):
        """不同维度的嵌入。"""
        for dim in [3, 8, 16]:
            emb = adapter.embed("hello", dim=dim)
            assert emb.shape == (dim,)

    def test_custom_register(self):
        """自定义 provider 注册。"""
        from mci_world_model.sdk._multillm_adapter import register_provider, _LLM_REGISTRY

        def mock_provider(kwargs):
            return None

        register_provider("test_mock", mock_provider)
        assert "test_mock" in _LLM_REGISTRY

        # 清理
        del _LLM_REGISTRY["test_mock"]

    def test_charset_embed(self):
        """字符集嵌入独立性测试。"""
        from mci_world_model.sdk._multillm_adapter import _charset_embed

        emb1 = _charset_embed("hello world", dim=5)
        emb2 = _charset_embed("goodbye", dim=5)
        # 不同文本应产生不同嵌入
        assert not np.allclose(emb1, emb2)


# =============================================================================
# OrchestratorBridge 测试
# =============================================================================


class TestOrchestratorBridge:
    """Orchestrator 桥接核心功能。"""

    # ── SCREENING ──

    def test_screening_basic(self, bridge):
        """SCREENING: 基本贝叶斯初筛。"""
        result = bridge.execute_intent("SCREENING", {
            "evidence": {"albumin": 1.0},
            "nodes": ["albumin", "body_weight"],
        })
        assert result.success is True
        assert "posterior" in result.data
        assert "risk_nodes" in result.data

    def test_screening_no_nodes(self, bridge):
        """SCREENING: 无风险节点时应优雅处理。"""
        result = bridge.execute_intent("SCREENING", {
            "evidence": {},
            "nodes": [],
        })
        assert result.success is True

    def test_screening_with_causal_strength(self, bridge):
        """SCREENING: 包含因果强度。"""
        result = bridge.execute_intent("SCREENING", {
            "evidence": {"albumin": 1.0, "nrs2002_score": 1.0},
            "nodes": ["albumin", "nrs2002_score", "body_weight"],
        })
        assert "causal_strengths" in result.data

    # ── ASSESSMENT ──

    def test_assessment_counterfactual(self, bridge):
        """ASSESSMENT: 反事实推理。"""
        result = bridge.execute_intent("ASSESSMENT", {
            "evidence": {"calorie_intake": 1500.0, "albumin": 28.0},
            "do_x": {"calorie_intake": 2000.0},
            "target": "albumin",
            "compute_pns": False,
        })
        assert result.success is True
        assert "factual_value" in result.data
        assert "counterfactual_value" in result.data
        assert "individual_effect" in result.data

    def test_assessment_missing_target(self, bridge):
        """ASSESSMENT: 缺少 target 时返回错误。"""
        result = bridge.execute_intent("ASSESSMENT", {
            "evidence": {"albumin": 28.0},
            "do_x": {"calorie_intake": 2000.0},
        })
        assert result.success is False

    def test_assessment_missing_evidence(self, bridge):
        """ASSESSMENT: 缺少 evidence 时返回错误。"""
        result = bridge.execute_intent("ASSESSMENT", {
            "do_x": {"calorie_intake": 2000.0},
            "target": "albumin",
        })
        assert result.success is False

    # ── PLAN_GENERATION ──

    def test_plan_generation_with_signals(self, bridge, patient_timeline):
        """PLAN_GENERATION: 基于物理信号。"""
        result = bridge.execute_intent("PLAN_GENERATION", {
            "signals": patient_timeline,
            "patient_id": "P001",
        })
        assert result.success is True
        assert "causal_edges" in result.data

    def test_plan_generation_with_memories(self, bridge):
        """PLAN_GENERATION: 基于记忆 — memories 编码需 world_model。
        
        降级策略: 无 world_model 时返回错误而非崩溃。
        """
        memories = [
            {"content": "白蛋白 28", "timestamp": 1000000},
            {"content": "热量摄入不足", "timestamp": 1000100},
            {"content": "体重下降", "timestamp": 1000200},
        ]
        result = bridge.execute_intent("PLAN_GENERATION", {
            "memories": memories,
            "patient_id": "P002",
        })
        # memories 路径需要 world_model (不可用时 graceful 降级)
        if not result.success:
            assert "encode failed" in result.error or "world_model" in result.error
        else:
            assert "causal_edges" in result.data

    def test_plan_generation_empty(self, bridge):
        """PLAN_GENERATION: 无数据时返回错误。"""
        result = bridge.execute_intent("PLAN_GENERATION", {
            "patient_id": "P003",
        })
        assert result.success is False

    # ── FOLLOWUP ──

    def test_followup_with_timeline(self, bridge, patient_timeline):
        """FOLLOWUP: 基于时间线的随访规划。"""
        result = bridge.execute_intent("FOLLOWUP", {
            "timeline": patient_timeline,
            "patient_id": "P001",
        })
        assert result.success is True
        assert "trends" in result.data
        assert "followup_suggestions" in result.data

    def test_followup_empty(self, bridge):
        """FOLLOWUP: 无时间线数据时返回错误。"""
        result = bridge.execute_intent("FOLLOWUP", {
            "patient_id": "P001",
        })
        assert result.success is False

    # ── MONITORING ──

    def test_monitoring_alerts(self, bridge, patient_timeline):
        """MONITORING: 异常检测。"""
        result = bridge.execute_intent("MONITORING", {
            "timeline": patient_timeline,
        })
        assert result.success is True
        assert "alerts" in result.data
        assert "total_days" in result.data

    def test_monitoring_empty(self, bridge):
        """MONITORING: 无数据时返回错误。"""
        result = bridge.execute_intent("MONITORING", {})
        assert result.success is False

    # ── 工作流 ──

    def test_initial_screening_workflow(self, bridge):
        """initial_screening 工作流。"""
        wf = bridge.execute_workflow("initial_screening", "P001", {
            "evidence": {"albumin": 1.0},
            "risk_nodes": ["albumin", "nrs2002_score"],
        })
        assert wf["workflow"] == "initial_screening"
        assert wf["patient_id"] == "P001"
        assert len(wf["results"]) == 2
        assert "summary" in wf

    def test_followup_plan_workflow(self, bridge, patient_timeline):
        """followup_plan 工作流。"""
        wf = bridge.execute_workflow("followup_plan", "P001", {
            "timeline": patient_timeline,
        })
        assert wf["workflow"] == "followup_plan"
        assert len(wf["results"]) == 2

    def test_unknown_workflow(self, bridge):
        """未知工作流。"""
        wf = bridge.execute_workflow("unknown_wf", "P001", {})
        assert len(wf["results"]) == 1
        assert wf["results"][0]["success"] is False

    # ── 未知意图 ──

    def test_unknown_intent(self, bridge):
        """未知意图类型返回错误。"""
        result = bridge.execute_intent("UNKNOWN", {})
        assert result.success is False

    # ── AgentResult ──

    def test_agent_result_ok(self):
        from mci_world_model.sdk._orchestrator_bridge import AgentResult

        r = AgentResult.ok("TEST", {"key": "val"}, source="unit_test")
        assert r.success is True
        assert r.to_dict()["success"] is True

    def test_agent_result_fail(self):
        from mci_world_model.sdk._orchestrator_bridge import AgentResult

        r = AgentResult.fail("TEST", "something broke")
        assert r.success is False
        assert r.error == "something broke"

    def test_register_custom_intent(self, bridge):
        """自定义意图注册。"""
        from mci_world_model.sdk._orchestrator_bridge import register_intent, _INTENT_REGISTRY

        def custom_handler(inst, params):
            return {"custom": True}

        register_intent("CUSTOM_ECHO", custom_handler, description="Echo test")
        assert "CUSTOM_ECHO" in _INTENT_REGISTRY

        del _INTENT_REGISTRY["CUSTOM_ECHO"]


# =============================================================================
# EnhancedPerception 测试
# =============================================================================


class TestEnhancedPerception:
    """增强感知管道核心功能。"""

    def test_extract_signals_albumin(self, perception):
        """提取白蛋白。"""
        signals = perception.extract_signals("白蛋白 28 g/L")
        assert len(signals) >= 1
        assert any(s["name"] == "albumin" and s["value"] == 28.0 for s in signals)

    def test_extract_signals_body_weight(self, perception):
        """提取体重。"""
        signals = perception.extract_signals("体重 72 kg")
        assert any(s["name"] == "body_weight" and s["value"] == 72.0 for s in signals)

    def test_extract_signals_nrs2002(self, perception):
        """提取 NRS2002。"""
        signals = perception.extract_signals("NRS2002评分3分")
        assert any(s["name"] == "nrs2002_score" and s["value"] == 3.0 for s in signals)

    def test_extract_signals_multi(self, perception):
        """提取多个指标。"""
        signals = perception.extract_signals("白蛋白28 g/L，体重65kg，NRS2002=2")
        names = {s["name"] for s in signals}
        assert "albumin" in names
        assert "body_weight" in names
        assert "nrs2002_score" in names

    def test_extract_signals_falling_pattern(self, perception):
        """提取下降模式中的白蛋白。"""
        signals = perception.extract_signals("白蛋白从35降至28")
        assert any(s["name"] == "albumin" and s["value"] == 28.0 for s in signals)

    def test_extract_signals_empty(self, perception):
        """空文本返回空列表。"""
        signals = perception.extract_signals("无相关内容")
        assert signals == []

    def test_process_signals(self, perception):
        """信号处理 (dict→MultimodalSignal 转换)。"""
        raw = [
            {"signal_type": "numerical", "name": "albumin", "value": 28.0, "unit": "g/L"},
            {"signal_type": "numerical", "name": "body_weight", "value": 72.0, "unit": "kg"},
        ]
        features = perception.process_signals(raw)
        assert len(features) >= 2

    def test_process_signals_empty(self, perception):
        """空信号列表。"""
        features = perception.process_signals([])
        assert features == []

    def test_perceive_to_state(self, perception, patient_timeline):
        """端到端：信号→因果状态。"""
        state = perception.perceive_to_state(patient_timeline)
        assert state is not None
        assert isinstance(state.causal_edges, list)

    def test_text_to_state(self, perception):
        """端到端：文本→因果状态。"""
        state = perception.text_to_state("白蛋白28 g/L，体重72kg")
        assert state is not None

    def test_text_to_state_empty(self, perception):
        """空文本→空状态。"""
        state = perception.text_to_state("")
        assert len(state.causal_edges) == 0
        assert state.n_memories == 0


# =============================================================================
# 端到端协同测试
# =============================================================================


class TestE2ECoordination:
    """跨模块协同测试。"""

    def test_intent_through_perception(self, bridge, perception, patient_timeline):
        """Perception → Orchestrator 协同。"""
        # 提取信号
        signals = perception.extract_signals("白蛋白28，NRS2002=3，体重70kg")

        # SCREENING
        scr = bridge.execute_intent("SCREENING", {
            "evidence": {"albumin": 1.0, "nrs2002_score": 1.0},
            "nodes": ["albumin", "nrs2002_score", "body_weight"],
        })
        assert scr.success

        # PLAN_GENERATION (with timeline)
        plan = bridge.execute_intent("PLAN_GENERATION", {
            "signals": patient_timeline,
            "patient_id": "P001",
        })
        assert plan.success
        assert plan.data["n_novel"] >= 0

    def test_fallback_resilience(self, adapter, bridge):
        """降级韧性: MultiLLM 不可用时桥接仍可工作。"""
        # 检查 MultiLLM 状态
        status = adapter.health_check()
        assert "mode" in status

        # 桥接应不依赖 LLM 可用性
        result = bridge.execute_intent("SCREENING", {
            "evidence": {"albumin": 1.0},
            "nodes": ["albumin"],
        })
        assert result.success is True
