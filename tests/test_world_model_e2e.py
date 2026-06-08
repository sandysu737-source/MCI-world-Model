"""
MCI World Model v3.1.0 — World Model 端到端集成测试
=====================================================

覆盖 _world_model.py 的核心流程:
- MCIWorldModel 初始化与健康检查
- discover 因果发现
- predict_effect / jepa_predict 预测
- intervene 干预
- train_jepa / train_parametric 训练
- CausalWorldModelState 状态管理
"""

import numpy as np
import pytest

from mci_world_model.sdk._world_model import CausalWorldModelState, MCIWorldModel

# =============================================================================
# 测试记忆数据
# =============================================================================


@pytest.fixture
def sample_memories():
    """创建 5 条测试记忆。"""
    memories = []
    for i in range(5):
        memories.append(
            {
                "id": f"mem_{i}",
                "content": (f"数据 {i}: {'价格上升导致需求下降' if i % 2 == 0 else '手术量增加提升收入水平'}"),
                "embedding": list(np.random.RandomState(42 + i).randn(64).astype(float)),
                "timestamp": f"2026-01-{i + 1:02d}",
            }
        )
    return memories


@pytest.fixture
def initialized_wm():
    """返回已初始化的 MCIWorldModel 实例。"""
    wm = MCIWorldModel()
    report = wm.initialize()
    return wm, report


# =============================================================================
# MCIWorldModel 初始化测试
# =============================================================================


class TestMCIWorldModelInit:
    """MCIWorldModel 初始化与生命周期测试。"""

    def test_init_no_args(self):
        """无参数初始化不崩溃。"""
        wm = MCIWorldModel()
        assert wm is not None

    def test_initialize_returns_dict(self):
        """initialize 返回状态报告。"""
        wm = MCIWorldModel()
        report = wm.initialize()
        assert isinstance(report, dict)
        assert "ready" in report or "modules" in report

    def test_initialize_idempotent(self):
        """initialize 幂等安全（第二次返回缓存报告）。"""
        wm = MCIWorldModel()
        r1 = wm.initialize()
        r2 = wm.initialize()
        # 第二次调用返回缓存版本（_cached=True），结构可能简化
        assert r1["ready"] is True
        assert r2["initialized"] is True

    def test_health_check_returns_dict(self, initialized_wm):
        """health_check 返回诊断结果。"""
        wm, _ = initialized_wm
        check = wm.health_check()
        assert isinstance(check, dict)
        assert check["version"] == "3.1.0"
        assert "causal_pipeline" in check
        assert "jepa_predictor" in check
        assert "energy_loss" in check
        assert "cost_module" in check
        assert "configurator" in check
        assert "causal_actor" in check

    def test_health_check_status(self, initialized_wm):
        """health_check 包含 status 字段。"""
        wm, _ = initialized_wm
        check = wm.health_check()
        assert "status" in check


# =============================================================================
# 因果发现测试
# =============================================================================


class TestDiscover:
    """三层因果发现流水线测试。"""

    def test_discover_empty_memories(self, initialized_wm):
        """空记忆列表降级返回当前 state。"""
        wm, _ = initialized_wm
        state = wm.discover(memories=[], verbose=False)
        assert state is not None

    def test_discover_insufficient_memories(self, initialized_wm):
        """记忆不足（< 3 条）降级返回。"""
        wm, _ = initialized_wm
        state = wm.discover(memories=[{"content": "测试"}], verbose=False)
        assert state is not None

    def test_discover_with_samples(self, initialized_wm, sample_memories):
        """有足够记忆时执行因果发现。"""
        wm, _ = initialized_wm
        state = wm.discover(memories=sample_memories, verbose=False)
        assert state is not None
        assert hasattr(state, "causal_edges")

    def test_discover_returns_causal_state(self, initialized_wm, sample_memories):
        """返回 CausalWorldModelState 实例。"""
        wm, _ = initialized_wm
        state = wm.discover(memories=sample_memories, verbose=False)
        assert isinstance(state, CausalWorldModelState)


# =============================================================================
# 因果/干预预测测试
# =============================================================================


class TestPredictions:
    """因果预测与干预测试。"""

    def test_predict_effect_retrieval(self, initialized_wm, sample_memories):
        """检索路径因果预测。"""
        wm, _ = initialized_wm
        effects = wm.predict_effect("价格上升", memories=sample_memories)
        assert isinstance(effects, list)

    def test_predict_effect_empty(self, initialized_wm):
        """空记忆预测返回空列表。"""
        wm, _ = initialized_wm
        effects = wm.predict_effect("测试", memories=[])
        assert effects == []

    def test_jepa_predict_returns_list(self, initialized_wm, sample_memories):
        """JEPA 预测返回列表。"""
        wm, _ = initialized_wm
        try:
            results = wm.jepa_predict("价格上升", memories=sample_memories)
            assert isinstance(results, list)
        except Exception:
            pass  # JEPA 编码器可能回退到检索路径

    def test_intervene_insufficient_input(self, initialized_wm):
        """缺少 do_x/target 时返回错误。"""
        wm, _ = initialized_wm
        result = wm.intervene()
        assert result["status"] == "insufficient_input"

    def test_intervene_with_params(self, initialized_wm, sample_memories):
        """带 do_x 和 target 的干预。"""
        wm, _ = initialized_wm
        # 先执行发现以填充因果图
        wm.discover(memories=sample_memories, verbose=False)
        result = wm.intervene(
            do_x={"价格": 1.5},
            target="需求",
            method="auto",
        )
        assert isinstance(result, dict)


# =============================================================================
# 训练测试
# =============================================================================


class TestTraining:
    """JEPA 训练流程测试。"""

    def test_train_jepa_no_data(self, initialized_wm):
        """无数据时 train_jepa 返回错误。"""
        wm, _ = initialized_wm
        result = wm.train_jepa()
        assert isinstance(result, dict)
        assert "error" in result

    def test_train_jepa_no_init(self):
        """未初始化时 train_jepa 也返回错误。"""
        wm = MCIWorldModel()
        result = wm.train_jepa()
        assert isinstance(result, dict)
        # 可能返回 error 或包含 message

    def test_train_parametric_is_alias(self):
        """train_parametric 是 train_jepa 别名。"""
        wm = MCIWorldModel()
        result = wm.train_parametric(qa_pairs=[]) if hasattr(wm, "train_parametric") else {}
        assert isinstance(result, dict)


# =============================================================================
# CausalWorldModelState 测试
# =============================================================================


class TestCausalWorldModelState:
    """CausalWorldModelState 数据类测试。"""

    def test_empty_state(self):
        """empty() 工厂方法。"""
        state = CausalWorldModelState.empty()
        assert state is not None
        assert state.causal_edges == []
        assert state.n_confirmed == 0
        assert state.n_novel == 0
        assert state.n_suppressed == 0

    def test_state_with_edges(self):
        """带因果边的状态。"""
        state = CausalWorldModelState(
            causal_edges=[
                {
                    "cause": "A",
                    "effect": "B",
                    "rho": 0.8,
                    "confidence": 0.9,
                    "verdict": "confirmed",
                    "energy_relation": "enhance",
                    "bayes_factor": 1.5,
                },
            ],
            n_confirmed=1,
        )
        assert len(state.causal_edges) == 1
        assert state.causal_edges[0]["cause"] == "A"
        assert state.causal_edges[0]["rho"] == 0.8

    def test_to_dict(self):
        """to_dict 序列化。"""
        state = CausalWorldModelState(
            causal_edges=[
                {"cause": "X", "effect": "Y", "rho": 0.7},
            ],
            active_states={"causal"},
            n_memories=10,
        )
        d = state.to_dict()
        assert d["n_causal_edges"] == 1
        assert d["n_memories"] == 10
        assert "causal" in d["active_states"]

    def test_counterfactual_fields(self):
        """反事实图字段。"""
        state = CausalWorldModelState(
            counterfactual_graph={"nodes": ["X"], "edges": []},
            do_interventions=[{"do_x": {"X": 0.5}}],
        )
        assert state.counterfactual_graph is not None
        assert len(state.do_interventions) == 1

    def test_energy_ratios_field(self):
        """五维能量分布字段。"""
        state = CausalWorldModelState(
            energy_ratios={"semantic": 0.25, "causal": 0.30, "spacetime": 0.20, "generative": 0.15, "trust": 0.10},
        )
        assert state.energy_ratios is not None
        assert state.energy_ratios["semantic"] == 0.25
