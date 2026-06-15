"""Phase C 测试 — TASK-C1/C2/C3。

覆盖:
  C1: NeurosymbolicWorldModel 路由决策/融合推理/三元编码
  C2: VisualEncoder 图像编码/信号编码/多模态融合
  C3: AutonomousMemoryManager 决策/执行/策略更新/训练
"""

from __future__ import annotations

import numpy as np

# C3
from mci_world_model.sdk._autonomous_memory import (
    AMMConfig,
    AutonomousMemoryManager,
    MemoryAction,
    MemoryDecision,
    MemoryFeatures,
)

# 依赖
from mci_world_model.sdk._do_calculus import CausalGraph

# C1
from mci_world_model.sdk._neurosymbolic_world_model import (
    NeurosymbolicConfig,
    NeurosymbolicWorldModel,
    RouteType,
    TripleRepresentation,
)

# C2
from mci_world_model.sdk._visual_encoder import (
    MultimodalPair,
    VisualEncoder,
    VisualEncoderConfig,
)

# =============================================================================
# C1: NeurosymbolicWorldModel
# =============================================================================


class TestC1RouteDecision:
    def test_route_returns_decision(self):
        nswm = NeurosymbolicWorldModel()
        decision = nswm.route("增加多巴胺剂量后心率变化")
        assert decision.route_type in (RouteType.PHYSICAL, RouteType.CAUSAL, RouteType.SEMANTIC, RouteType.FUSED)
        assert len(decision.scores) == 3

    def test_route_deterministic_with_same_query(self):
        nswm = NeurosymbolicWorldModel(config=NeurosymbolicConfig(seed=42))
        d1 = nswm.route("物理预测查询")
        d2 = nswm.route("物理预测查询")
        assert d1.route_type == d2.route_type

    def test_route_with_causal_keywords(self):
        nswm = NeurosymbolicWorldModel(config=NeurosymbolicConfig(seed=42))
        # 因果相关查询应有偏向
        decision = nswm.route("X导致Y的因果效应是多少")
        assert decision.confidence > 0

    def test_route_type_enum(self):
        assert RouteType.PHYSICAL.value == "physical"
        assert RouteType.CAUSAL.value == "causal"
        assert RouteType.SEMANTIC.value == "semantic"
        assert RouteType.FUSED.value == "fused"


class TestC1Inference:
    def test_infer_no_backends(self):
        nswm = NeurosymbolicWorldModel()
        result = nswm.infer("测试查询")
        assert result.uncertainty == 1.0

    def test_infer_with_causal_graph(self):
        cg = CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])
        nswm = NeurosymbolicWorldModel(causal_graph=cg)
        result = nswm.infer("因果推断", context={"cause": "X", "effect": "Y"})
        assert result.output is not None

    def test_infer_latency_tracked(self):
        nswm = NeurosymbolicWorldModel()
        result = nswm.infer("查询")
        assert result.latency_ms >= 0

    def test_infer_with_all_backends(self):
        cg = CausalGraph(nodes=["A", "B"], edges=[("A", "B")])
        nswm = NeurosymbolicWorldModel(causal_graph=cg, config=NeurosymbolicConfig(seed=42))
        result = nswm.infer("综合查询")
        assert result.route_used in (RouteType.CAUSAL, RouteType.FUSED, RouteType.PHYSICAL, RouteType.SEMANTIC)


class TestC1TripleEncoding:
    def test_encode_triple_empty(self):
        nswm = NeurosymbolicWorldModel()
        triple = nswm.encode_triple(state=None)
        assert isinstance(triple, TripleRepresentation)

    def test_encode_triple_with_state(self):
        nswm = NeurosymbolicWorldModel()
        triple = nswm.encode_triple(state=np.array([0.5, -0.3]), query="测试")
        assert isinstance(triple, TripleRepresentation)

    def test_encode_triple_with_causal_graph(self):
        cg = CausalGraph(nodes=["A", "B"], edges=[("A", "B")])
        nswm = NeurosymbolicWorldModel(causal_graph=cg)
        triple = nswm.encode_triple(state=None)
        assert len(triple.causal_features) > 0


class TestC1Config:
    def test_default_config(self):
        cfg = NeurosymbolicConfig()
        assert cfg.z_dim == 16
        assert cfg.temperature == 1.0
        assert len(cfg.route_weights) == 3

    def test_custom_config(self):
        cfg = NeurosymbolicConfig(z_dim=8, temperature=0.5)
        nswm = NeurosymbolicWorldModel(config=cfg)
        assert nswm.config.z_dim == 8


# =============================================================================
# C2: VisualEncoder
# =============================================================================


class TestC2SignalEncoding:
    def test_encode_signals_shape(self):
        enc = VisualEncoder(VisualEncoderConfig(signal_dim=4, z_dim=8))
        z = enc.encode_signals(np.array([0.5, -0.3, 1.0, 0.2]))
        assert z.shape == (8,)

    def test_encode_signals_shorter_input(self):
        enc = VisualEncoder(VisualEncoderConfig(signal_dim=4, z_dim=8))
        z = enc.encode_signals(np.array([0.5]))  # 1 < signal_dim=4
        assert z.shape == (8,)
        assert not np.any(np.isnan(z))

    def test_encode_signals_longer_input(self):
        enc = VisualEncoder(VisualEncoderConfig(signal_dim=2, z_dim=8))
        z = enc.encode_signals(np.array([1.0, 2.0, 3.0, 4.0]))  # 4 > signal_dim=2
        assert z.shape == (8,)


class TestC2ImageEncoding:
    def test_encode_image_shape(self):
        enc = VisualEncoder(VisualEncoderConfig(image_height=8, image_width=8, z_dim=8))
        img = np.random.randn(1, 8, 8)
        z = enc.encode_image(img)
        assert z.shape == (8,)

    def test_encode_image_2d(self):
        enc = VisualEncoder(VisualEncoderConfig(image_height=8, image_width=8, z_dim=8))
        img = np.random.randn(8, 8)
        z = enc.encode_image(img)
        assert z.shape == (8,)

    def test_encode_image_no_nan(self):
        enc = VisualEncoder(VisualEncoderConfig(z_dim=8))
        img = np.random.randn(1, 32, 32) * 100
        z = enc.encode_image(img)
        assert not np.any(np.isnan(z))


class TestC2Multimodal:
    def test_encode_multimodal_shape(self):
        enc = VisualEncoder(VisualEncoderConfig(signal_dim=4, z_dim=8))
        img = np.random.randn(1, 32, 32)
        sig = np.array([0.5, -0.3, 1.0, 0.2])
        z = enc.encode_multimodal(img, sig)
        assert z.shape == (8,)

    def test_multimodal_different_from_signal_only(self):
        enc = VisualEncoder(VisualEncoderConfig(signal_dim=4, z_dim=8, seed=42))
        img = np.random.randn(1, 32, 32)
        sig = np.array([0.5, -0.3, 1.0, 0.2])
        z_mm = enc.encode_multimodal(img, sig)
        z_sig = enc.encode_signals(sig)
        # 融合结果应与纯信号不同
        assert not np.allclose(z_mm, z_sig)


class TestC2Training:
    def test_train_basic(self):
        enc = VisualEncoder(VisualEncoderConfig(z_dim=4, hidden_dim=8, n_epochs=3, seed=42))
        pairs = [
            MultimodalPair(
                image=np.random.randn(1, 8, 8),
                signals=np.array([1.0, 0.0]),
                target_latent=np.array([0.1, 0.2, 0.3, 0.4]),
            )
            for _ in range(5)
        ]
        result = enc.train(pairs)
        assert result["n_samples"] == 5
        assert enc.is_trained

    def test_train_empty(self):
        enc = VisualEncoder()
        result = enc.train([])
        assert result["n_epochs"] == 0


class TestC2FromSignalsAdapter:
    def test_from_signals_basic(self):
        class FakeSignal:
            def __init__(self, value):
                self.value = value
                self.modality = "test"

        signals = [FakeSignal(0.5), FakeSignal([1.0, 2.0])]
        z = VisualEncoder.from_signals_adapter(signals, z_dim=8)
        assert z.shape == (8,)

    def test_from_signals_empty(self):
        z = VisualEncoder.from_signals_adapter([], z_dim=8)
        assert z.shape == (8,)
        assert np.allclose(z, 0.0)


# =============================================================================
# C3: AutonomousMemoryManager
# =============================================================================


class TestC3MemoryFeatures:
    def test_to_vector(self):
        mf = MemoryFeatures(memory_id="m1", age=100, access_count=5, relevance_score=0.8, importance_score=0.9)
        vec = mf.to_vector()
        assert vec.shape == (5,)
        assert vec[0] == 0.1  # age/1000
        assert vec[1] == 0.05  # access/100
        assert vec[2] == 0.8  # relevance
        assert vec[3] == 0.9  # importance

    def test_default_values(self):
        mf = MemoryFeatures(memory_id="m1")
        assert mf.age == 0
        assert mf.access_count == 0


class TestC3Decision:
    def test_decide_consolidate(self):
        amm = AutonomousMemoryManager(AMMConfig(w_importance=0.5, w_access=0.3, w_relevance=0.3, w_age=-0.05))
        mem = MemoryFeatures(
            memory_id="m1", importance_score=0.99, access_count=100, relevance_score=0.99, decay_utility=1.0
        )
        decisions = amm.decide([mem])
        assert decisions[0].action == MemoryAction.CONSOLIDATE

    def test_decide_retain(self):
        amm = AutonomousMemoryManager(AMMConfig(w_importance=0.4, w_access=0.3, w_relevance=0.3, w_age=-0.05))
        mem = MemoryFeatures(memory_id="m1", importance_score=0.7, access_count=20, relevance_score=0.7)
        decisions = amm.decide([mem])
        assert decisions[0].action in (MemoryAction.RETAIN, MemoryAction.CONSOLIDATE)

    def test_decide_forget(self):
        amm = AutonomousMemoryManager()
        mem = MemoryFeatures(memory_id="m1", importance_score=0.1, age=500, access_count=0, relevance_score=0.1)
        decisions = amm.decide([mem])
        assert decisions[0].action in (MemoryAction.FORGET, MemoryAction.ARCHIVE)

    def test_decide_archive(self):
        amm = AutonomousMemoryManager()
        mem = MemoryFeatures(memory_id="m1", importance_score=0.3, age=200, access_count=2, relevance_score=0.3)
        decisions = amm.decide([mem])
        assert decisions[0].action in (MemoryAction.ARCHIVE, MemoryAction.FORGET, MemoryAction.RETAIN)

    def test_decide_multiple(self):
        amm = AutonomousMemoryManager()
        mems = [
            MemoryFeatures(memory_id=f"m{i}", importance_score=0.9 - i * 0.2, age=i * 100, access_count=10 - i)
            for i in range(5)
        ]
        decisions = amm.decide(mems)
        assert len(decisions) == 5


class TestC3Execute:
    def test_execute_forget(self):
        amm = AutonomousMemoryManager()
        store = {"m1": {"data": "test"}}
        decisions = [MemoryDecision(memory_id="m1", action=MemoryAction.FORGET)]
        counts = amm.execute(decisions, store)
        assert "m1" not in store
        assert counts["forget"] == 1

    def test_execute_consolidate(self):
        amm = AutonomousMemoryManager()
        store = {"m1": {"data": "test"}}
        decisions = [MemoryDecision(memory_id="m1", action=MemoryAction.CONSOLIDATE)]
        counts = amm.execute(decisions, store)
        assert store["m1"]["_priority"] == "high"
        assert counts["consolidate"] == 1

    def test_execute_archive(self):
        amm = AutonomousMemoryManager()
        store = {"m1": {"data": "test"}}
        decisions = [MemoryDecision(memory_id="m1", action=MemoryAction.ARCHIVE)]
        amm.execute(decisions, store)
        assert store["m1"].get("_archived") is True

    def test_execute_retain(self):
        amm = AutonomousMemoryManager()
        store = {"m1": {"data": "test"}}
        decisions = [MemoryDecision(memory_id="m1", action=MemoryAction.RETAIN)]
        counts = amm.execute(decisions, store)
        assert "m1" in store  # 仍存在
        assert counts["retain"] == 1


class TestC3PolicyUpdate:
    def test_update_policy_shifts_weights(self):
        amm = AutonomousMemoryManager(AMMConfig(learning_rate=0.1))
        old_w = amm.config.w_importance
        amm.update_policy({"retrieval_hit_rate": 0.9, "storage_usage": 0.3, "importance_preservation": 0.9})
        # 高奖励应提升权重
        assert amm.config.w_importance != old_w

    def test_reward_history(self):
        amm = AutonomousMemoryManager()
        amm.update_policy({"retrieval_hit_rate": 0.5})
        assert len(amm.reward_history) == 1


class TestC3Training:
    def test_train_basic(self):
        amm = AutonomousMemoryManager()
        episodes = [
            {
                "memories": [MemoryFeatures(memory_id="m1", importance_score=0.8)],
                "feedback": {"retrieval_hit_rate": 0.7, "storage_usage": 0.5, "importance_preservation": 0.8},
            }
            for _ in range(3)
        ]
        result = amm.train(episodes)
        assert result["n_episodes"] == 3
        assert result["avg_reward"] > 0

    def test_train_empty(self):
        amm = AutonomousMemoryManager()
        result = amm.train([])
        assert result["n_episodes"] == 0


class TestC3Config:
    def test_default_config(self):
        cfg = AMMConfig()
        assert cfg.w_importance > 0
        assert cfg.w_age < 0  # 越老越倾向遗忘
        assert cfg.theta_consolidate > cfg.theta_retain > cfg.theta_archive

    def test_custom_config(self):
        cfg = AMMConfig(w_importance=0.5, theta_retain=0.6)
        amm = AutonomousMemoryManager(cfg)
        assert amm.config.w_importance == 0.5
