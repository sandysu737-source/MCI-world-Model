"""
V3.0.7 QC 修复: CausalMLP + ParametricMemory 功能测试。

覆盖 1220 行核心代码的测试债务，验证:
- CausalMLP: 构造/前向/激活/多层/错误输入/因果标签映射
- ParametricMemory: 存取/遗忘/检索/空记忆/序列化

运行: pytest tests/test_v307_mlp_memory.py -v
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from mci_world_model.sdk._causal_mlp import (
    CATEGORY_TO_INDEX,
    ENERGY_CATEGORIES,
    INDEX_TO_CATEGORY,
    CausalMLP,
    SimpleTextEmbedder,
    energy_relation_to_category,
)
from mci_world_model.sdk._parametric_memory import (
    ParametricMemory,
    ParametricMemoryConfig,
)

# =============================================================================
# CausalMLP 测试 (7 tests)
# =============================================================================


class TestCausalMLPInit:
    """S2.1: CausalMLP 初始化与构造测试。"""

    def test_mlp_init_default(self):
        """默认参数构造验证。"""
        mlp = CausalMLP(input_dim=128, hidden_dims=(64, 32))
        assert mlp.input_dim == 128
        assert mlp.num_categories == 5
        assert not mlp.is_trained
        assert mlp.n_trainable_params > 0
        assert "CausalMLP" in repr(mlp)

    def test_mlp_init_custom_dims(self):
        """自定义架构参数构造。"""
        mlp = CausalMLP(input_dim=64, hidden_dims=(32,), num_categories=5, seed=99)
        assert mlp.input_dim == 64
        assert mlp.num_categories == 5
        assert mlp.n_trainable_params > 0


class TestCausalMLPForward:
    """S2.1: CausalMLP 前向传播测试。"""

    def test_mlp_forward_output_shape(self):
        """前向传播输出形状 (num_categories,) 且 sum=1。"""
        mlp = CausalMLP(input_dim=128, hidden_dims=(64, 32))
        x = np.random.randn(128).astype(np.float32)
        probs = mlp.forward(x)
        assert probs.shape == (5,)
        assert probs.dtype == np.float32
        assert abs(float(probs.sum()) - 1.0) < 1e-5

    def test_mlp_forward_all_nonnegative(self):
        """Softmax 输出全非负。"""
        mlp = CausalMLP(input_dim=128, hidden_dims=(64, 32))
        x = np.random.randn(128).astype(np.float32)
        probs = mlp.forward(x)
        assert np.all(probs >= 0.0)

    def test_mlp_predict_category_valid(self):
        """预测类别为有效五范畴之一。"""
        mlp = CausalMLP(input_dim=128, hidden_dims=(64, 32))
        x = np.random.randn(128).astype(np.float32)
        cat = mlp.predict_category(x)
        assert cat in ENERGY_CATEGORIES


class TestCausalMLPMultiLayer:
    """S2.1: 多层传播一致性测试。"""

    def test_mlp_multi_layer(self):
        """多层 (128→64→32→5) 传播不崩溃。"""
        mlp = CausalMLP(input_dim=128, hidden_dims=(64, 32, 16), num_categories=5)
        x = np.random.randn(128).astype(np.float32)
        probs = mlp.forward(x)
        assert probs.shape == (5,)
        assert abs(float(probs.sum()) - 1.0) < 1e-5

    def test_mlp_batch_forward(self):
        """批量前向传播输出形状 (B, num_categories)。"""
        mlp = CausalMLP(input_dim=128, hidden_dims=(64, 32))
        X = np.random.randn(10, 128).astype(np.float32)
        probs = mlp.batch_forward(X)
        assert probs.shape == (10, 5)


class TestCausalMLPCausalityTag:
    """S2.1: 因果标签映射测试。"""

    def test_mlp_causality_tag_mappings(self):
        """验证 CATEGORY_TO_INDEX / INDEX_TO_CATEGORY 双向一致。"""
        for cat in ENERGY_CATEGORIES:
            idx = CATEGORY_TO_INDEX[cat]
            assert INDEX_TO_CATEGORY[idx] == cat
        assert len(ENERGY_CATEGORIES) == 5
        assert set(ENERGY_CATEGORIES) == {"semantic", "causal", "spacetime", "generative", "trust"}

    def test_mlp_relation_to_category(self):
        """验证 energy_relation → category 映射。"""
        assert energy_relation_to_category("enhance") == CATEGORY_TO_INDEX["causal"]
        assert energy_relation_to_category("suppress") == CATEGORY_TO_INDEX["causal"]
        assert energy_relation_to_category("neutral") == CATEGORY_TO_INDEX["semantic"]
        assert energy_relation_to_category("unknown") == 0  # fallback


class TestCausalMLPInvalidInput:
    """S2.1: 错误输入容错测试。"""

    def test_mlp_wrong_dimension(self):
        """错误维度输入应抛出异常。"""
        mlp = CausalMLP(input_dim=128, hidden_dims=(64, 32))
        x_bad = np.random.randn(64).astype(np.float32)  # 应该是 128 维
        with pytest.raises(ValueError):
            mlp.forward(x_bad)


# =============================================================================
# SimpleTextEmbedder 测试
# =============================================================================


class TestSimpleTextEmbedder:
    """验证轻量文本嵌入器。"""

    def test_embed_output_shape(self):
        """嵌入输出维度正确。"""
        embedder = SimpleTextEmbedder(output_dim=128)
        vec = embedder.embed("物价上涨导致货币贬值")
        assert vec.shape == (128,)
        assert vec.dtype == np.float32

    def test_embed_normalized(self):
        """非空文本嵌入应归一化。"""
        embedder = SimpleTextEmbedder(output_dim=128)
        vec = embedder.embed("因果关系推理")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 0.01

    def test_embed_empty_text(self):
        """空文本返回零向量。"""
        embedder = SimpleTextEmbedder(output_dim=128)
        vec = embedder.embed("")
        assert np.all(vec == 0.0)


# =============================================================================
# ParametricMemory 测试 (5 tests)
# =============================================================================


class TestParametricMemoryStoreRetrieve:
    """S2.2: 存取闭环测试。"""

    def test_memory_store_retrieve(self):
        """prepare → train → predict 闭环。"""
        config = ParametricMemoryConfig(
            num_epochs=3,
            batch_size=4,
            learning_rate=0.05,
            min_training_pairs=5,
        )
        pm = ParametricMemory(config)

        # 构造 10 条训练数据
        qa_pairs = [
            {"cause_text": "摄入高热量", "effect_text": "体重增加", "energy_relation": "enhance", "confidence": 0.8},
            {"cause_text": "蛋白质摄入", "effect_text": "白蛋白升高", "energy_relation": "enhance", "confidence": 0.9},
            {
                "cause_text": "抗生素使用",
                "effect_text": "肠道菌群改变",
                "energy_relation": "suppress",
                "confidence": 0.7,
            },
            {"cause_text": "运动增加", "effect_text": "能量消耗增加", "energy_relation": "enhance", "confidence": 0.85},
            {"cause_text": "压力增加", "effect_text": "皮质醇升高", "energy_relation": "enhance", "confidence": 0.75},
            {"cause_text": "睡眠不足", "effect_text": "免疫功能下降", "energy_relation": "suppress", "confidence": 0.8},
            {"cause_text": "补钙", "effect_text": "骨密度增加", "energy_relation": "enhance", "confidence": 0.9},
            {"cause_text": "低蛋白饮食", "effect_text": "肌肉流失", "energy_relation": "enhance", "confidence": 0.8},
            {"cause_text": "高钠饮食", "effect_text": "血压升高", "energy_relation": "enhance", "confidence": 0.85},
            {"cause_text": "长期卧床", "effect_text": "肌肉萎缩", "energy_relation": "suppress", "confidence": 0.9},
        ]

        n, report = pm.prepare_training_data(qa_pairs)
        assert n == 10
        assert report["meets_minimum"]

        stats = pm.train()
        assert "final_loss" in stats
        assert stats["n_samples"] == 10

        result = pm.predict("高热量摄入有利于体重增加")
        assert isinstance(result, list)
        assert len(result) > 0
        assert "category" in result[0]


class TestParametricMemoryEmpty:
    """S2.2: 空记忆处理测试。"""

    def test_memory_empty_retrieval(self):
        """空记忆检索返回默认结果。"""
        pm = ParametricMemory()

        # 不准备训练数据直接预测
        result = pm.predict("测试文本")
        # 应该返回默认概率分布（未训练时向前传播的结果）
        assert isinstance(result, list)
        assert len(result) > 0


class TestParametricMemoryPredict:
    """S2.2: predict 方法测试。"""

    def test_memory_retrieve_topk(self):
        """验证 predict 返回概率和类别。"""
        config = ParametricMemoryConfig(
            num_epochs=2,
            batch_size=4,
            learning_rate=0.05,
            min_training_pairs=3,
        )
        pm = ParametricMemory(config)

        qa_pairs = [
            {"cause_text": "高热量", "effect_text": "增重", "energy_relation": "enhance", "confidence": 0.9},
            {"cause_text": "运动", "effect_text": "消耗热量", "energy_relation": "enhance", "confidence": 0.85},
            {"cause_text": "抗生素", "effect_text": "菌群紊乱", "energy_relation": "suppress", "confidence": 0.8},
        ]

        pm.prepare_training_data(qa_pairs)
        pm.train()

        result = pm.predict("高热量摄入")
        assert isinstance(result, list)
        assert len(result) >= 1
        # 验证有类别信息
        if len(result) > 0:
            assert "category" in result[0]
            assert "confidence" in result[0]


class TestParametricMemoryForget:
    """S2.2: 遗忘/重置测试。"""

    def test_memory_forget_on_new_instance(self):
        """新实例不应有旧训练记忆。"""
        pm1 = ParametricMemory()
        assert not pm1._is_trained
        assert len(pm1._training_data) == 0

        # 训练第一个实例
        qa_pairs = [{"cause_text": "A", "effect_text": "B", "energy_relation": "enhance", "confidence": 0.9}] * 10
        pm1.prepare_training_data(qa_pairs)
        pm1.train()
        assert pm1._is_trained

        # 新实例不受影响
        pm2 = ParametricMemory()
        assert not pm2._is_trained
        assert len(pm2._training_data) == 0


class TestParametricMemorySerialize:
    """S2.2: 序列化/反序列化测试。"""

    def test_memory_serialize(self):
        """save_adapter → load_adapter 闭环。"""
        config = ParametricMemoryConfig(
            num_epochs=2,
            batch_size=4,
            learning_rate=0.05,
            min_training_pairs=3,
        )
        pm = ParametricMemory(config)

        qa_pairs = [
            {"cause_text": "蛋白质", "effect_text": "白蛋白升高", "energy_relation": "enhance", "confidence": 0.9},
            {"cause_text": "低蛋白", "effect_text": "水肿", "energy_relation": "suppress", "confidence": 0.8},
            {"cause_text": "正常饮食", "effect_text": "体重稳定", "energy_relation": "same", "confidence": 0.7},
        ]
        pm.prepare_training_data(qa_pairs)
        pm.train()

        # 使用白名单内的路径（./checkpoints 在允许根目录中）
        adapter_path = os.path.join("./checkpoints", "test_adapter_serialize")
        try:
            saved = pm.save_adapter(adapter_path)
            assert saved

            # 验证文件存在
            assert os.path.exists(os.path.join(adapter_path, "causal_mlp_weights.npz"))
            assert os.path.exists(os.path.join(adapter_path, "adapter_config.json"))

            # 加载
            pm2 = ParametricMemory(config)
            loaded = pm2.load_adapter(adapter_path)
            assert loaded

            # 加载后应可预测
            result = pm2.predict("蛋白质摄入有助于白蛋白提升")
            assert isinstance(result, list)
            assert len(result) > 0
        finally:
            # 清理
            import shutil

            if os.path.exists(adapter_path):
                shutil.rmtree(adapter_path, ignore_errors=True)
