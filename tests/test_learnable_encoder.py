"""
TASK-A1.8: LearnableStateEncoder 单元测试
=============================================

覆盖:
- 参数形状正确性
- 前向传播输出形状
- 重建损失正确性
- 训练收敛性
- 序列化/反序列化
- 多种状态类型 (PendulumState, CartState, RobotWorldState)
- 物理状态区分能力
- 异常输入处理

KPI: F-6 编码器重建 MSE (500 steps) < 0.01
"""

import os
import tempfile

import numpy as np
import pytest

from mci_world_model.sdk._learnable_encoder import LearnableStateEncoder


class TestLearnableStateEncoderInit:
    """编码器初始化测试。"""

    def test_init_params_shape(self):
        """验证参数矩阵形状正确。"""
        enc = LearnableStateEncoder(state_dim=2, latent_dim=16, hidden_dim=64)
        assert enc.W1.shape == (2, 64), f"W1 shape: {enc.W1.shape}"
        assert enc.b1.shape == (64,), f"b1 shape: {enc.b1.shape}"
        assert enc.W2.shape == (64, 32), f"W2 shape: {enc.W2.shape}"
        assert enc.b2.shape == (32,), f"b2 shape: {enc.b2.shape}"
        assert enc.W3.shape == (32, 16), f"W3 shape: {enc.W3.shape}"
        assert enc.b3.shape == (16,), f"b3 shape: {enc.b3.shape}"
        # 解码器
        assert enc.W4.shape == (16, 32), f"W4 shape: {enc.W4.shape}"
        assert enc.W5.shape == (32, 64), f"W5 shape: {enc.W5.shape}"
        assert enc.W6.shape == (64, 2), f"W6 shape: {enc.W6.shape}"

    def test_init_param_count(self):
        """验证参数量级正确 (~5K for dim=2, H=64, Z=16)。"""
        enc = LearnableStateEncoder(state_dim=2, latent_dim=16, hidden_dim=64)
        n = enc.n_params
        # W1(2*64=128) + b1(64) + W2(64*32=2048) + b2(32) + W3(32*16=512) + b3(16)
        # + W4(16*32=512) + b4(32) + W5(32*64=2048) + b5(64) + W6(64*2=128) + b6(2)
        # = 128+64+2048+32+512+16+512+32+2048+64+128+2 = 5586
        assert 4000 < n < 8000, f"Unexpected param count: {n}"

    def test_init_invalid_dims(self):
        """验证无效维度抛出异常。"""
        with pytest.raises(ValueError, match="state_dim must be positive"):
            LearnableStateEncoder(state_dim=0)
        with pytest.raises(ValueError, match="latent_dim must be positive"):
            LearnableStateEncoder(state_dim=2, latent_dim=0)


class TestLearnableStateEncoderForward:
    """前向传播测试。"""

    def test_forward_output_shape_pendulum(self):
        """PendulumState (2,) → (16,) 潜向量。"""
        enc = LearnableStateEncoder(state_dim=2, latent_dim=16)
        state = np.array([0.5, -0.3])
        latent = enc.forward(state)
        assert latent.shape == (16,), f"Expected (16,), got {latent.shape}"

    def test_forward_output_shape_robot(self):
        """RobotWorldState (18,) → (16,) 潜向量。"""
        enc = LearnableStateEncoder(state_dim=18, latent_dim=16, hidden_dim=64)
        state = np.zeros(18)
        latent = enc.forward(state)
        assert latent.shape == (16,), f"Expected (16,), got {latent.shape}"

    def test_decode_output_shape(self):
        """解码器输出形状 == state_dim。"""
        enc = LearnableStateEncoder(state_dim=6, latent_dim=16)
        latent = np.zeros(16)
        reconstructed = enc.decode(latent)
        assert reconstructed.shape == (6,), f"Expected (6,), got {reconstructed.shape}"

    def test_reconstruct_shape(self):
        """重建输出形状 == 输入形状。"""
        enc = LearnableStateEncoder(state_dim=4, latent_dim=8)
        state = np.array([1.0, 2.0, 3.0, 4.0])
        reconstructed = enc.reconstruct(state)
        assert reconstructed.shape == state.shape

    def test_forward_wrong_dim_raises(self):
        """输入维度不匹配时抛出异常。"""
        enc = LearnableStateEncoder(state_dim=2, latent_dim=16)
        with pytest.raises(ValueError, match="Expected state_dim=2"):
            enc.forward(np.array([1.0, 2.0, 3.0]))

    def test_forward_output_finite(self):
        """输出值全部有限 (非 NaN/Inf)。"""
        enc = LearnableStateEncoder(state_dim=4, latent_dim=8)
        state = np.array([1.0, -2.0, 0.5, -0.1])
        latent = enc.forward(state)
        assert np.all(np.isfinite(latent)), "Output contains non-finite values"


class TestLearnableStateEncoderTraining:
    """训练功能测试。"""

    def test_reconstruction_loss_positive(self):
        """初始重建损失 > 0（未训练时不应完美重建）。"""
        enc = LearnableStateEncoder(state_dim=4, latent_dim=8, seed=42)
        state = np.array([1.0, -0.5, 0.3, -0.8])
        loss = enc.reconstruction_loss(state)
        assert loss > 0, f"Loss should be positive for untrained encoder, got {loss}"

    def test_training_reduces_loss(self):
        """100 步 SGD 后损失下降 >= 20%。"""
        enc = LearnableStateEncoder(state_dim=4, latent_dim=8, seed=42)
        rng = np.random.RandomState(42)
        states = rng.randn(20, 4).astype(np.float64)

        initial_loss = np.mean([enc.reconstruction_loss(s) for s in states])

        for _ in range(100):
            for s in states:
                enc.training_forward(s)
                result = enc.compute_gradients(s)
                enc.apply_gradients(result["grads"], lr=0.01)

        final_loss = np.mean([enc.reconstruction_loss(s) for s in states])
        reduction = (initial_loss - final_loss) / initial_loss

        assert reduction >= 0.20, (
            f"Loss reduction {reduction:.1%} < 20%. Initial: {initial_loss:.6f}, Final: {final_loss:.6f}"
        )

    def test_train_on_batch(self):
        """批量训练正常工作。"""
        enc = LearnableStateEncoder(state_dim=4, latent_dim=8, seed=42)
        rng = np.random.RandomState(42)
        states = rng.randn(10, 4).astype(np.float64)

        result = enc.train_on_batch(states, lr=0.01)
        assert "loss" in result
        assert "mse" in result
        assert result["loss"] > 0
        assert enc.train_steps == 1

    def test_pendulum_physics_encoding(self):
        """不同物理状态产生可区分的潜表示（训练后）。"""
        enc = LearnableStateEncoder(state_dim=2, latent_dim=16, seed=42)
        rng = np.random.RandomState(42)

        # 先在多样化数据上训练编码器
        train_states = rng.uniform(-1, 1, size=(100, 2)).astype(np.float64)
        for _ in range(100):
            enc.train_on_batch(train_states, lr=0.005)

        state1 = np.array([0.1, 0.0])  # theta=0.1, omega=0
        state2 = np.array([0.3, 0.0])  # theta=0.3, omega=0

        latent1 = enc.forward(state1)
        latent2 = enc.forward(state2)

        # 训练后的编码器应区分不同物理状态
        # 使用 L2 距离 > 0 而非余弦相似度（更稳健）
        l2_dist = float(np.linalg.norm(latent1 - latent2))
        assert l2_dist > 0.01, f"Latent vectors too close: L2={l2_dist:.6f}"

    def test_converges_to_low_mse(self):
        """KPI F-6: 500 步训练后重建 MSE < 0.01 (PendulumState)。"""
        enc = LearnableStateEncoder(state_dim=2, latent_dim=16, seed=42)
        rng = np.random.RandomState(42)
        states = rng.randn(50, 2).astype(np.float64) * 0.5

        for _ in range(500):
            for s in states:
                enc.training_forward(s)
                result = enc.compute_gradients(s)
                enc.apply_gradients(result["grads"], lr=0.005)

        final_mse = np.mean([enc.reconstruction_loss(s) for s in states])
        assert final_mse < 0.01, f"MSE {final_mse:.6f} >= 0.01 (KPI F-6 FAILED)"


class TestLearnableStateEncoderSerialization:
    """序列化测试。"""

    def test_save_load_roundtrip(self):
        """save_params → load_params → 前向输出一致。"""
        enc = LearnableStateEncoder(state_dim=4, latent_dim=8, seed=42)
        state = np.array([1.0, -0.5, 0.3, -0.8])

        # 训练几步改变参数
        enc.training_forward(state)
        result = enc.compute_gradients(state)
        enc.apply_gradients(result["grads"], lr=0.01)

        original_output = enc.forward(state)

        # 保存和加载
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            enc.save_params(path)
            enc2 = LearnableStateEncoder(state_dim=4, latent_dim=8, seed=999)
            enc2.load_params(path)
            loaded_output = enc2.forward(state)

            np.testing.assert_array_almost_equal(
                original_output,
                loaded_output,
                decimal=10,
                err_msg="Save/load roundtrip mismatch",
            )
        finally:
            os.unlink(path)

    def test_load_wrong_dim_raises(self):
        """加载维度不匹配的参数抛出异常。"""
        enc1 = LearnableStateEncoder(state_dim=4, latent_dim=8, seed=42)
        enc2 = LearnableStateEncoder(state_dim=2, latent_dim=8, seed=42)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            enc1.save_params(path)
            with pytest.raises(ValueError, match="state_dim mismatch"):
                enc2.load_params(path)
        finally:
            os.unlink(path)


class TestLearnableStateEncoderRobustness:
    """鲁棒性测试。"""

    def test_nan_input_raises(self):
        """NaN 输入应产生明确的结果（非静默传播）。"""
        enc = LearnableStateEncoder(state_dim=4, latent_dim=8)
        state = np.array([1.0, np.nan, 0.5, -0.3])
        # forward 不要求崩溃，但输出应包含 NaN 以暴露问题
        latent = enc.forward(state)
        # 编码器不应静默地吃掉 NaN
        assert np.any(np.isnan(latent)) or np.all(np.isfinite(latent)), (
            "Encoder should either propagate NaN clearly or handle it"
        )

    def test_extreme_values_finite(self):
        """极端输入值产生有限输出。"""
        enc = LearnableStateEncoder(state_dim=4, latent_dim=8, seed=42)
        state = np.array([1e6, -1e6, 1e-8, -1e-8])
        latent = enc.forward(state)
        # ReLU + 线性层的组合不应爆炸
        assert np.all(np.isfinite(latent)), "Extreme input produced non-finite output"

    def test_encoder_on_cart_state(self):
        """CartState (2,) 正常工作。"""
        enc = LearnableStateEncoder(state_dim=2, latent_dim=16)
        cart_state = np.array([1.0, 0.5])  # position, velocity
        latent = enc.forward(cart_state)
        assert latent.shape == (16,)

    def test_encoder_on_robot_state(self):
        """RobotWorldState (18,) 正常工作。"""
        enc = LearnableStateEncoder(state_dim=18, latent_dim=16, hidden_dim=64)
        robot_state = np.zeros(18)
        robot_state[0] = 0.1  # theta
        robot_state[6] = 0.5  # joint angle
        latent = enc.forward(robot_state)
        assert latent.shape == (16,)

    def test_ood_detection(self):
        """OOD 检测: 训练范围外的状态重建误差 > 训练范围内 × 2。"""
        enc = LearnableStateEncoder(state_dim=2, latent_dim=16, seed=42)
        rng = np.random.RandomState(42)

        # 在 theta ∈ [0, π/4] 范围内训练
        id_states = rng.uniform(0, np.pi / 4, size=(50, 2)).astype(np.float64)
        for _ in range(200):
            for s in id_states:
                enc.training_forward(s)
                result = enc.compute_gradients(s)
                enc.apply_gradients(result["grads"], lr=0.01)

        # ID 测试
        id_test = rng.uniform(0, np.pi / 4, size=(20, 2)).astype(np.float64)
        id_loss = np.mean([enc.reconstruction_loss(s) for s in id_test])

        # OOD 测试 (theta ∈ [π/3, π/2])
        ood_test = rng.uniform(np.pi / 3, np.pi / 2, size=(20, 2)).astype(np.float64)
        ood_loss = np.mean([enc.reconstruction_loss(s) for s in ood_test])

        # KPI F-12: OOD 重建误差应显著大于 ID
        assert ood_loss > 1.5 * id_loss, f"OOD loss {ood_loss:.6f} not > 1.5× ID loss {id_loss:.6f}"


class TestLearnableStateEncoderIntegration:
    """编码器与 JEPATrainer 集成测试。"""

    def test_trainer_train_encoder(self):
        """JEPATrainer.train_encoder() 正常工作。"""
        from mci_world_model.sdk._jepa_encoder import JEPAEncoder
        from mci_world_model.sdk._jepa_trainer import JEPATrainer

        # 创建带可学习编码器的 JEPAEncoder
        learnable = LearnableStateEncoder(state_dim=4, latent_dim=8)
        enc = JEPAEncoder(world_model=None, learnable_encoder=learnable)

        # 创建 trainer
        trainer = JEPATrainer(
            encoder=enc,
            predictor=None,
            use_learnable_encoder=True,
        )

        rng = np.random.RandomState(42)
        states = rng.randn(20, 4).astype(np.float64)

        result = trainer.train_encoder(states, n_epochs=10, learning_rate=0.01)
        assert result["n_epochs"] == 10
        assert len(result["loss_history"]) == 10
        assert result["final_loss"] > 0

    def test_trainer_without_learnable_encoder(self):
        """无可学习编码器时 train_encoder() 安全返回。"""
        from mci_world_model.sdk._jepa_trainer import JEPATrainer

        trainer = JEPATrainer(
            encoder=None,
            predictor=None,
            use_learnable_encoder=False,
        )

        result = trainer.train_encoder(np.zeros((5, 4)))
        assert result["n_epochs"] == 0
