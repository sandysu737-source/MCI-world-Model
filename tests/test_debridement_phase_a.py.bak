"""Phase A: 清创数据管线 + 编码器 + 组织分类器 测试。

覆盖:
    - DebridementSample 构造与序列化
    - SyntheticDebridementGenerator 合成数据生成
    - DepthEncoder / ForceEncoder 编码
    - TissueClassifier 训练/分类/安全判断
    - ZvecEmbeddingStore 存储检索
"""

import numpy as np
import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def generator():
    from mci_world_model.sdk._debridement_data import SyntheticDebridementGenerator

    return SyntheticDebridementGenerator(seed=42)


@pytest.fixture
def depth_encoder():
    from mci_world_model.sdk._modality_encoders import DepthEncoder

    return DepthEncoder()


@pytest.fixture
def force_encoder():
    from mci_world_model.sdk._modality_encoders import ForceEncoder

    return ForceEncoder()


# =============================================================================
# DebridementSample
# =============================================================================


class TestDebridementSample:
    """DebridementSample 基础测试。"""

    def test_default_creation(self):
        from mci_world_model.sdk._debridement_data import DebridementSample

        s = DebridementSample()
        assert s.rgb_image.shape == (224, 224, 3)
        assert s.depth_image.shape == (224, 224)
        assert s.thermal_image.shape == (224, 224)
        assert len(s.force_torque) == 6
        assert len(s.joint_positions) == 7
        assert s.tissue_label == 0

    def test_tissue_names(self):
        from mci_world_model.sdk._debridement_data import DebridementSample

        s = DebridementSample(tissue_label=0)
        assert s.tissue_name == "坏死"
        s2 = DebridementSample(tissue_label=3)
        assert s2.tissue_name == "上皮"

    def test_phase_names(self):
        from mci_world_model.sdk._debridement_data import DebridementSample

        s = DebridementSample(surgical_phase=1)
        assert s.phase_name == "清创"

    def test_to_vector(self):
        from mci_world_model.sdk._debridement_data import DebridementSample

        s = DebridementSample()
        v = s.to_vector()
        assert v.ndim == 1
        assert v.dtype == np.float32
        assert s.n_features == len(v)

    def test_copy(self):
        from mci_world_model.sdk._debridement_data import DebridementSample

        s = DebridementSample(
            tissue_label=2,
            surgical_phase=1,
            wound_depth_mm=3.5,
            sample_id="test_001",
        )
        c = s.copy()
        assert c.tissue_label == s.tissue_label
        assert c.wound_depth_mm == s.wound_depth_mm
        assert c.sample_id == s.sample_id
        # 确保深拷贝
        c.rgb_image[0, 0, 0] = 255
        assert s.rgb_image[0, 0, 0] == 0


# =============================================================================
# SyntheticDebridementGenerator
# =============================================================================


class TestSyntheticGenerator:
    """合成数据生成器测试。"""

    def test_generate_single(self, generator):
        s = generator.generate_sample(tissue_label=0)
        assert s.rgb_image.shape == (224, 224, 3)
        assert s.tissue_label == 0
        assert s.sample_id.startswith("syn_")

    def test_generate_batch_balanced(self, generator):
        batch = generator.generate_batch(40, balanced=True)
        assert len(batch) == 40
        labels = [s.tissue_label for s in batch]
        for c in range(4):
            assert labels.count(c) == 10  # 均衡

    def test_generate_batch_unbalanced(self, generator):
        batch = generator.generate_batch(20, balanced=False)
        assert len(batch) == 20

    def test_all_tissue_types_generated(self, generator):
        batch = generator.generate_batch(100, balanced=True)
        labels = set(s.tissue_label for s in batch)
        assert labels == {0, 1, 2, 3}

    def test_all_phases_generated(self, generator):
        batch = generator.generate_batch(100, balanced=True)
        phases = set(s.surgical_phase for s in batch)
        assert 1 in phases  # 清创相最常见

    def test_tissue_specific_properties(self, generator):
        """不同组织类型的物理属性有区分度。"""
        nec = generator.generate_sample(0)
        epi = generator.generate_sample(3)
        # 坏死组织温度应低于上皮
        assert float(np.mean(nec.thermal_image)) < 36.0
        assert float(np.mean(epi.thermal_image)) >= 36.0


# =============================================================================
# DepthEncoder
# =============================================================================


class TestDepthEncoder:
    """深度编码器测试。"""

    def test_output_shape(self, depth_encoder):
        dm = np.random.randn(224, 224).astype(np.float32) * 5 + 10
        out = depth_encoder.encode(dm)
        assert out.shape == (32,)

    def test_different_inputs_different_outputs(self, depth_encoder):
        # 不同形态的深度图应产生不同输出（非纯常量）
        d1 = depth_encoder.encode(np.random.randn(224, 224).astype(np.float32) * 3 + 10)
        d2 = depth_encoder.encode(np.random.randn(224, 224).astype(np.float32) * 1 + 3)
        # 不同深度应产生不同输出
        assert not np.allclose(d1, d2, atol=0.01)

    def test_output_dim(self, depth_encoder):
        assert depth_encoder.output_dim == 32
        assert depth_encoder.feature_dim == 8


# =============================================================================
# ForceEncoder
# =============================================================================


class TestForceEncoder:
    """力编码器测试。"""

    def test_output_shape_single(self, force_encoder):
        ft = np.random.randn(6).astype(np.float32)
        out = force_encoder.encode(ft)
        assert out.shape == (32,)

    def test_output_shape_sequence(self, force_encoder):
        ft = np.random.randn(50, 6).astype(np.float32)
        out = force_encoder.encode(ft)
        assert out.shape == (32,)

    def test_different_forces_different_outputs(self, force_encoder):
        rng = np.random.RandomState(42)
        f1 = force_encoder.encode(rng.randn(6).astype(np.float32) + 2.0)
        f2 = force_encoder.encode(rng.randn(6).astype(np.float32) * 3 + 8.0)
        assert not np.allclose(f1, f2, atol=0.01)

    def test_output_dim(self, force_encoder):
        assert force_encoder.output_dim == 32
        assert force_encoder.feature_dim == 16


# =============================================================================
# TissueClassifier
# =============================================================================


class TestTissueClassifier:
    """组织分类器测试。"""

    @pytest.fixture
    def clf(self):
        from mci_world_model.sdk._tissue_classifier import TissueClassifier

        return TissueClassifier(input_dim=256)

    @pytest.fixture
    def trained_clf(self, clf):
        """训练好的分类器。"""
        np.random.seed(42)
        n = 200
        X = np.zeros((n, 256), dtype=np.float64)
        y = np.zeros(n, dtype=np.int64)
        for c in range(4):
            cen = np.random.randn(256) * 0.5
            cen[0] += c * 1.5
            X[c * 50 : (c + 1) * 50] = cen + np.random.randn(50, 256) * 0.3
            y[c * 50 : (c + 1) * 50] = c
        clf.train(X, y, n_epochs=20, lr=0.01)
        return clf, X, y

    def test_init(self, clf):
        assert clf.n_params > 0
        assert not clf.is_trained

    def test_forward_output_shape(self, clf):
        x = np.random.randn(10, 256).astype(np.float64)
        probs = clf._forward(x)
        assert probs.shape == (10, 4)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_train_converges(self, clf):
        np.random.seed(42)
        X = np.random.randn(200, 256)
        y = np.array(
            [0] * 50 + [1] * 50 + [2] * 50 + [3] * 50, dtype=np.int64
        )
        # 添加类别区分度
        for c in range(4):
            X[y == c, 0] += c * 2.0
        stats = clf.train(X, y, n_epochs=30, lr=0.01)
        assert stats["final_loss"] < 1.0  # 应该收敛

    def test_evaluate_returns_metrics(self, trained_clf):
        clf, X, y = trained_clf
        metrics = clf.evaluate(X, y)
        assert "accuracy" in metrics
        assert "balanced_accuracy" in metrics
        assert "confusion_matrix" in metrics
        assert metrics["accuracy"] >= 0.8

    def test_classify_smoke(self, trained_clf):
        clf, X, _ = trained_clf
        result = clf.classify(X[0])
        assert result.confidence > 0
        assert result.tissue_name in ["坏死", "腐肉", "肉芽", "上皮"]

    def test_safety_uncertain(self, clf):
        """低置信度应标记为不确定。"""
        # 不训练，随机特征
        x = np.random.randn(256).astype(np.float64)
        result = clf.classify(x)
        assert result.is_uncertain or result.confidence < 0.7

    def test_epithelial_safety(self, clf):
        """上皮组织应限制力。"""
        from mci_world_model.sdk._tissue_classifier import MAX_FORCE_BY_TISSUE

        x = np.zeros(256, dtype=np.float64)
        result = clf.classify(x)
        max_f = MAX_FORCE_BY_TISSUE.get(result.predicted_label, 0.5)
        assert result.max_force_n <= max_f + 0.01

    def test_save_load(self, trained_clf):
        clf, _, _ = trained_clf
        import os
        import shutil
        import tempfile

        d = tempfile.mkdtemp()
        try:
            path = os.path.join(d, "tissue_clf")
            assert clf.save(path)
            loaded = clf.load(path)
            assert loaded is not None
            assert loaded.n_params == clf.n_params
            assert loaded.is_trained
        finally:
            shutil.rmtree(d, ignore_errors=True)


# =============================================================================
# ZvecEmbeddingStore
# =============================================================================


class TestZvecEmbeddingStore:
    """Zvec 嵌入存储测试。"""

    @pytest.fixture
    def store(self):
        import shutil
        import tempfile

        from mci_world_model.sdk._zvec_store import (
            EmbeddingStoreConfig,
            ZvecEmbeddingStore,
        )

        d = tempfile.mkdtemp()
        config = EmbeddingStoreConfig(store_path=d, dim=128)
        s = ZvecEmbeddingStore(config)
        yield s
        shutil.rmtree(d, ignore_errors=True)

    def test_insert_and_search(self, store):
        pairs = [
            {
                "cause_text": "蛋白质摄入不足",
                "effect_text": "白蛋白下降",
                "energy_relation": "enhance",
                "confidence": 0.9,
            },
            {
                "cause_text": "高蛋白饮食",
                "effect_text": "体重增加",
                "energy_relation": "enhance",
                "confidence": 0.85,
            },
        ]
        n = store.insert_qa_pairs(pairs)
        assert n == 2
        assert store.n_docs == 2

        results = store.search_similar("蛋白质缺乏", topk=2)
        assert len(results) >= 1

    def test_bm25_search(self, store):
        pairs = [
            {"cause_text": "抗生素", "effect_text": "菌群紊乱", "energy_relation": "suppress", "confidence": 0.88},
            {"cause_text": "运动", "effect_text": "肌力增强", "energy_relation": "enhance", "confidence": 0.80},
        ]
        store.insert_qa_pairs(pairs)
        results = store.search_bm25("抗生素", topk=2)
        assert len(results) >= 1
        assert "抗生素" in str(results)

    def test_fallback_works_without_zvec(self, store):
        """Fallback numpy 检索应总是可用。"""
        pairs = [
            {"cause_text": "A", "effect_text": "B", "energy_relation": "enhance", "confidence": 0.5},
        ]
        store.insert_qa_pairs(pairs)
        results = store.search_similar("A", topk=1)
        assert len(results) >= 1
