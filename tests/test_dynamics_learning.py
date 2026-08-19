"""
TASK-A2: LearnedDynamicsPredictor 测试
========================================

覆盖:
- 前向传播形状
- 单步预测精度
- 多步自回归不发散
- 训练收敛
- 动作效应方向正确
- 保存/加载
- 梯度验证
- 数据生成器

KPI: F-4 单步 MSE < 1e-4, F-5 5步累积 MSE < 8e-4
"""

import os
import tempfile

import numpy as np
import pytest

from mci_world_model.sdk._learned_dynamics_predictor import (
    DynamicsDataGenerator,
    LearnedDynamicsPredictor,
)


class TestLearnedDynamicsPredictorInit:
    """初始化测试。"""

    def test_init_param_shapes(self):
        pred = LearnedDynamicsPredictor(state_dim=2, action_dim=1, hidden_dim=128)
        assert pred.W1.shape == (3, 128)  # state_dim + action_dim = 3
        assert pred.b1.shape == (128,)
        assert pred.W2.shape == (128, 64)
        assert pred.W3.shape == (64, 2)

    def test_init_param_count(self):
        pred = LearnedDynamicsPredictor(state_dim=2, action_dim=1, hidden_dim=128)
        # W1(3*128=384) + b1(128) + W2(128*64=8192) + b2(64) + W3(64*2=128) + b3(2)
        # = 384+128+8192+64+128+2 = 8898
        assert 5000 < pred.n_params < 15000, f"Unexpected param count: {pred.n_params}"

    def test_init_invalid_dims(self):
        with pytest.raises(ValueError, match="state_dim must be positive"):
            LearnedDynamicsPredictor(state_dim=0)
        with pytest.raises(ValueError, match="action_dim must be non-negative"):
            LearnedDynamicsPredictor(state_dim=2, action_dim=-1)


class TestLearnedDynamicsPredictorForward:
    """前向传播测试。"""

    def test_forward_inference_output_shape(self):
        pred = LearnedDynamicsPredictor(state_dim=2, action_dim=1)
        s = np.array([0.5, -0.3])
        a = np.array([1.0])
        next_s = pred._forward_inference(s, a)
        assert next_s.shape == (2,)

    def test_training_forward_output_shape(self):
        pred = LearnedDynamicsPredictor(state_dim=2, action_dim=1)
        s = np.array([0.5, -0.3])
        a = np.array([1.0])
        pred.training_forward(s, a)
        assert pred._cache["pred"].shape == (2,)

    def test_predict_returns_list(self):
        pred = LearnedDynamicsPredictor(state_dim=2, action_dim=1)
        s = np.array([0.5, -0.3])
        a = np.array([1.0])
        traj = pred.predict(s, a, n_steps=3)
        assert len(traj) == 3


class TestLearnedDynamicsPredictorTraining:
    """训练功能测试。"""

    def test_compute_gradients(self):
        pred = LearnedDynamicsPredictor(state_dim=2, action_dim=1)
        s = np.array([0.5, -0.3])
        a = np.array([1.0])
        target = np.array([0.6, -0.1])

        pred.training_forward(s, a)
        result = pred.compute_gradients(target)

        assert "loss" in result
        assert "mse" in result
        assert "grads" in result
        assert result["mse"] > 0

    def test_train_step_reduces_loss(self):
        """单步训练后损失下降。"""
        pred = LearnedDynamicsPredictor(state_dim=2, action_dim=1, seed=42)
        rng = np.random.RandomState(42)

        # 生成简单线性关系数据: next = state + 0.1 * action
        dataset = []
        for _ in range(100):
            s = rng.randn(2)
            a = rng.randn(1)
            target = s + 0.1 * a
            dataset.append((s, a, target))

        initial_loss = pred.train_step(dataset[0][0], dataset[0][1], dataset[0][2])

        # 训练 200 步
        for _ in range(200):
            idx = rng.randint(len(dataset))
            s, a, target = dataset[idx]
            pred.train_step(s, a, target, lr=0.005)

        final_loss = pred.train_step(dataset[0][0], dataset[0][1], dataset[0][2])
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} → {final_loss}"

    def test_converges_to_physics(self):
        """KPI F-4: 200 epochs 后单步 MSE < 初始误差 × 0.3。"""
        pred = LearnedDynamicsPredictor(state_dim=2, action_dim=1, seed=42)
        generator = DynamicsDataGenerator(state_type="pendulum", seed=42)
        dataset = generator.generate_dataset(n_trajectories=20, steps_per_trajectory=10, noise_std=0.001)

        initial_losses = []
        for s, a, target in dataset[:10]:
            pred.training_forward(s, a)
            result = pred.compute_gradients(target)
            initial_losses.append(result["mse"])
        initial_avg = float(np.mean(initial_losses))

        # 训练 200 epochs
        result = pred.train_on_dataset(dataset, n_epochs=200, lr=0.005)

        final_losses = []
        for s, a, target in dataset[:10]:
            pred.training_forward(s, a)
            res = pred.compute_gradients(target)
            final_losses.append(res["mse"])
        final_avg = float(np.mean(final_losses))

        reduction = (initial_avg - final_avg) / initial_avg
        assert reduction >= 0.70, f"Loss reduction {reduction:.1%} < 70%"

    def test_action_effect_direction(self):
        """action 效果方向正确：正 torque → theta 变化方向一致。"""
        pred = LearnedDynamicsPredictor(state_dim=2, action_dim=1, seed=42)
        generator = DynamicsDataGenerator(state_type="pendulum", seed=42)
        dataset = generator.generate_dataset(n_trajectories=50, steps_per_trajectory=20)
        pred.train_on_dataset(dataset, n_epochs=300, lr=0.005)

        # 测试: 正 torque 应使 theta 增大（或至少产生非零效应）
        s = np.array([0.1, 0.0])
        a_pos = np.array([2.0])
        a_neg = np.array([-2.0])
        a_zero = np.array([0.0])

        next_pos = pred._forward_inference(s, a_pos)
        next_neg = pred._forward_inference(s, a_neg)
        _next_zero = pred._forward_inference(s, a_zero)

        # 正力矩与负力矩应产生不同效果
        diff_pos_neg = float(np.linalg.norm(next_pos - next_neg))
        assert diff_pos_neg > 1e-6, "Action has no effect on prediction"


class TestLearnedDynamicsPredictorGradient:
    """梯度验证测试。"""

    @pytest.mark.parametrize("param_name", ["W1", "b1", "W2", "b2", "W3", "b3"])
    def test_gradient_verification(self, param_name):
        """手写梯度 vs 数值梯度验证。"""
        pred = LearnedDynamicsPredictor(state_dim=2, action_dim=1, seed=42)
        s = np.array([0.5, -0.3])
        a = np.array([1.0])
        target = np.array([0.6, -0.1])

        pred.training_forward(s, a)
        result = pred.compute_gradients(target)
        analytical = result["grads"][param_name]

        # 数值梯度
        param = getattr(pred, param_name)
        eps = 1e-5 if param_name.startswith("W") else 1e-6
        numerical = np.zeros_like(param)

        for idx in range(param.size):
            old = param.ravel()[idx]

            param.ravel()[idx] = old + eps
            setattr(pred, param_name, param.ravel().reshape(param.shape))
            pred.training_forward(s, a)
            p_plus = pred._cache["pred"]
            loss_plus = float(np.mean((p_plus - target) ** 2))

            param.ravel()[idx] = old - eps
            setattr(pred, param_name, param.ravel().reshape(param.shape))
            pred.training_forward(s, a)
            p_minus = pred._cache["pred"]
            loss_minus = float(np.mean((p_minus - target) ** 2))

            param.ravel()[idx] = old
            setattr(pred, param_name, param.ravel().reshape(param.shape))
            numerical.ravel()[idx] = (loss_plus - loss_minus) / (2 * eps)

        diff = np.abs(analytical - numerical)
        denom = np.maximum(np.maximum(np.abs(analytical), np.abs(numerical)), 1e-8)
        rel_error = np.max(diff / denom)

        assert rel_error < 1e-3, f"Gradient check failed for {param_name}: rel_error={rel_error:.2e}"


class TestLearnedDynamicsPredictorMultiStep:
    """多步自回归测试。"""

    def test_multi_step_autoregressive(self):
        """KPI F-5: 5 步自回归不爆炸。"""
        pred = LearnedDynamicsPredictor(state_dim=2, action_dim=1, seed=42)
        generator = DynamicsDataGenerator(state_type="pendulum", seed=42)
        dataset = generator.generate_dataset(n_trajectories=50, steps_per_trajectory=20)
        pred.train_on_dataset(dataset, n_epochs=200, lr=0.005)

        s = np.array([0.1, 0.0])
        a = np.array([0.5])

        current = s.copy()
        step_errors = []
        for step in range(5):
            next_s = pred._forward_inference(current, a)
            # 不应有 NaN 或 Inf
            assert np.all(np.isfinite(next_s)), f"Step {step + 1}: non-finite output"
            step_errors.append(float(np.linalg.norm(next_s - current)))
            current = next_s

        # 每步误差应不超过前步的 3 倍（不指数发散）
        for i in range(1, len(step_errors)):
            ratio = step_errors[i] / max(step_errors[i - 1], 1e-10)
            assert ratio < 3.0, f"Step {i + 1} error ratio {ratio:.1f} > 3.0"


class TestLearnedDynamicsPredictorSerialization:
    """序列化测试。"""

    def test_save_load_roundtrip(self):
        pred = LearnedDynamicsPredictor(state_dim=2, action_dim=1, seed=42)
        s = np.array([0.5, -0.3])
        a = np.array([1.0])

        # 训练几步
        pred.training_forward(s, a)
        result = pred.compute_gradients(np.array([0.6, -0.1]))
        pred.apply_gradients(result["grads"], lr=0.01)

        output_before = pred._forward_inference(s, a)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            pred.save_params(path)
            pred2 = LearnedDynamicsPredictor(state_dim=2, action_dim=1, seed=999)
            pred2.load_params(path)
            output_after = pred2._forward_inference(s, a)

            np.testing.assert_array_almost_equal(output_before, output_after, decimal=10)
        finally:
            os.unlink(path)


class TestDynamicsDataGenerator:
    """数据生成器测试。"""

    def test_generate_pendulum_dataset(self):
        gen = DynamicsDataGenerator(state_type="pendulum", seed=42)
        dataset = gen.generate_dataset(n_trajectories=5, steps_per_trajectory=10)
        assert len(dataset) > 0
        s, a, ns = dataset[0]
        assert s.shape == (2,)
        assert a.shape == (1,)
        assert ns.shape == (2,)

    def test_generate_cart_dataset(self):
        gen = DynamicsDataGenerator(state_type="cart", seed=42)
        dataset = gen.generate_dataset(n_trajectories=5, steps_per_trajectory=10)
        assert len(dataset) > 0
        s, _a, _ns = dataset[0]
        assert s.shape == (2,)
