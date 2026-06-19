"""Phase C+D+E: 清创系统集成测试 — 端到端闭环."""

import numpy as np
import pytest


class TestEndToEndPipeline:
    """全链路: 数据 → 训练 → 推理 → 安全。"""

    def test_full_pipeline_smoke(self):
        from mci_world_model.sdk._debridement_data import SyntheticDebridementGenerator
        from mci_world_model.sdk._debridement_world_model import DebridementWorldModel, DebridementConfig
        from mci_world_model.sdk._tissue_classifier import TissueClassifier
        from mci_world_model.sdk._force_tissue_dynamics import ForceTissueDynamics
        from mci_world_model.sdk._char_tokenizer import CharTokenizer, SimpleTextEmbedderV2

        # 1. Generate data
        gen = SyntheticDebridementGenerator(seed=42)
        samples = gen.generate_batch(60, balanced=True)
        assert len(samples) == 60

        # 2. Train world model
        config = DebridementConfig.tiny()
        dwm = DebridementWorldModel(config)
        stats = dwm.train(samples, n_epochs=5, lr=0.01, batch_size=8)
        assert stats["final_loss"] < 10.0

        # 3. Predict tissue
        sample = gen.generate_sample(0)
        probs = dwm.predict_tissue(sample)
        assert probs.shape == (4,)

        # 4. Safety check
        ftd = ForceTissueDynamics()
        pred = ftd.predict_removal(0, 2.0, 5.0)
        assert pred.is_safe

        # 5. Text embedding
        embedder = SimpleTextEmbedderV2()
        corpus = ["蛋白质摄入不足导致白蛋白下降", "高蛋白饮食有助于体重增加", "低蛋白饮食导致肌肉流失"]
        embedder.fit(corpus)
        v1 = embedder.embed("蛋白质摄入不足")
        v2 = embedder.embed("高蛋白饮食")
        assert v1.shape == (128,)
        assert not np.allclose(v1, v2, atol=0.1)

    def test_safety_gate_integration(self):
        from mci_world_model.sdk._safety import SafetyMonitor, TissueForceConstraint, ThermalSafetyConstraint, DepthLimitConstraint
        from mci_world_model.sdk._debridement_data import DebridementSample

        monitor = SafetyMonitor()
        monitor.register(TissueForceConstraint(tissue_label=0))
        monitor.register(ThermalSafetyConstraint(max_temp_c=42.0))
        monitor.register(DepthLimitConstraint(max_depth_mm=5.0))

        # Safe state
        s = DebridementSample(tool_force_n=2.0, wound_depth_mm=3.0)
        r = monitor.check_all(s)
        assert r.passed

        # Unsafe force
        s2 = DebridementSample(tool_force_n=5.0, wound_depth_mm=3.0)
        r2 = monitor.check_all(s2)
        assert not r2.passed

        # Unsafe thermal
        s3 = DebridementSample(thermal_image=np.full((224, 224), 45.0, dtype=np.float32))
        r3 = monitor.check_all(s3)
        assert not r3.passed


class TestTokenizerAndEmbedding:
    def test_char_tokenizer(self):
        from mci_world_model.sdk._char_tokenizer import CharTokenizer
        tokenizer = CharTokenizer()
        ids = tokenizer.encode("高蛋白饮食")
        assert len(ids) == 64  # max_len
        assert ids[0] >= 4  # Should be a CJK char
        decoded = tokenizer.decode(ids)
        assert isinstance(decoded, str)

    def test_embedder_semantic_similarity(self):
        from mci_world_model.sdk._char_tokenizer import SimpleTextEmbedderV2
        embedder = SimpleTextEmbedderV2()
        corpus = ["蛋白质摄入不足导致白蛋白下降", "高蛋白饮食有助于体重增加"]
        embedder.fit(corpus)
        v1 = embedder.embed("蛋白质摄入不足")
        v2 = embedder.embed("蛋白质缺乏")
        v3 = embedder.embed("运动训练")
        sim12 = np.dot(v1, v2)
        sim13 = np.dot(v1, v3)
        # Similar texts should have higher cosine similarity
        assert sim12 > sim13, f"Expected {sim12:.3f} > {sim13:.3f}"


class TestCEWMLoop:
    """CEWM 闭环接口验证。"""

    def test_cewm_debridement_integration(self):
        """确保 DebridementWorldModel 与现有 CEWM 接口兼容。"""
        from mci_world_model.sdk._debridement_world_model import DebridementWorldModel, DebridementConfig
        from mci_world_model.sdk._debridement_data import SyntheticDebridementGenerator

        gen = SyntheticDebridementGenerator(seed=42)
        samples = gen.generate_batch(30)

        dwm = DebridementWorldModel(DebridementConfig.tiny())
        dwm.train(samples, n_epochs=3, lr=0.01)

        sample = gen.generate_sample(0)
        out = dwm.forward(sample)

        # CEWM 兼容: forward 返回 dict 含 "dynamics" 和 "tissue_probs"
        assert "dynamics" in out
        assert "tissue_probs" in out
        assert len(out["dynamics"]) > 0
        assert out["tissue_probs"].sum() == pytest.approx(1.0)


class TestModelSizes:
    def test_config_sizes(self):
        from mci_world_model.sdk._debridement_world_model import DebridementWorldModel, DebridementConfig
        tiny = DebridementWorldModel(DebridementConfig.tiny())
        small = DebridementWorldModel(DebridementConfig.small())
        assert tiny.n_params > 1000
        assert small.n_params > tiny.n_params
