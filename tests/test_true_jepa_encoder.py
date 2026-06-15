"""tests/test_true_jepa_encoder.py — TrueJEPA 编码器测试
==========================================================

F6 修复验证: encode() 返回 ndarray, 不是 CausalWorldModelState。
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._true_jepa_encoder import TrueJEPAConfig, TrueJEPAEncoder

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def encoder():
    return TrueJEPAEncoder(TrueJEPAConfig(obs_dim=8, latent_dim=32, hidden_dim=64, action_dim=2))


# =============================================================================
# F6 核心测试: encode() 返回 ndarray
# =============================================================================


class TestF6Fix:
    """F6: JEPA 名不副实修复验证。"""

    def test_encode_returns_ndarray(self, encoder):
        """encode() 返回 ndarray, 不是 CausalWorldModelState。"""
        obs = np.random.randn(8)
        z = encoder.encode(obs)
        assert isinstance(z, np.ndarray), f"encode() 应返回 ndarray, 实际返回 {type(z).__name__}"
        assert z.shape == (32,), f"潜向量维度应为 (32,), 实际 {z.shape}"

    def test_encode_not_causal_state(self, encoder):
        """encode() 结果不是 CausalWorldModelState。"""
        from mci_world_model.sdk._world_model import CausalWorldModelState

        obs = np.random.randn(8)
        z = encoder.encode(obs)
        assert not isinstance(z, CausalWorldModelState), "encode() 不应返回 CausalWorldModelState"

    def test_latent_dim_at_least_64(self):
        """F6 KPI: 潜向量维度 ≥ 64。"""
        enc = TrueJEPAEncoder(TrueJEPAConfig(obs_dim=16, latent_dim=64))
        obs = np.random.randn(16)
        z = enc.encode(obs)
        assert z.shape[0] >= 64


class TestEncoding:
    """编码基础功能测试。"""

    def test_deterministic(self, encoder):
        """相同输入 → 相同输出。"""
        obs = np.random.RandomState(42).randn(8)
        z1 = encoder.encode(obs)
        z2 = encoder.encode(obs)
        np.testing.assert_array_equal(z1, z2)

    def test_different_inputs_different_outputs(self, encoder):
        """不同输入 → 不同潜向量。"""
        obs1 = np.zeros(8)
        obs2 = np.ones(8)
        z1 = encoder.encode(obs1)
        z2 = encoder.encode(obs2)
        assert not np.allclose(z1, z2), "不同输入应产生不同潜向量"

    def test_target_encode(self, encoder):
        """目标编码器正常工作。"""
        obs = np.random.randn(8)
        z_target = encoder.encode_target(obs)
        assert isinstance(z_target, np.ndarray)
        assert z_target.shape == (32,)


class TestPrediction:
    """潜空间预测测试。"""

    def test_predict_next(self, encoder):
        """predict_next() 返回正确维度。"""
        z = np.random.randn(32)
        action = np.random.randn(2)
        z_next = encoder.predict_next(z, action)
        assert isinstance(z_next, np.ndarray)
        assert z_next.shape == (32,)

    def test_predict_next_no_action(self, encoder):
        """无动作时预测仍正常。"""
        z = np.random.randn(32)
        z_next = encoder.predict_next(z)
        assert z_next.shape == (32,)


class TestTraining:
    """训练功能测试。"""

    def test_train_step(self, encoder):
        """单步训练正常。"""
        obs_t = np.random.randn(8)
        obs_t1 = np.random.randn(8)
        action = np.random.randn(2)
        loss = encoder.train_step(obs_t, obs_t1, action)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_train_batch(self, encoder):
        """批量训练收敛。"""
        rng = np.random.RandomState(42)
        obs = rng.randn(20, 8)
        actions = rng.randn(20, 2)
        result = encoder.train(obs, actions, n_epochs=5)
        assert result["n_epochs"] == 5
        assert result["n_pairs"] == 19
        assert result["n_params"] > 0


class TestProperties:
    """属性测试。"""

    def test_n_params_positive(self, encoder):
        """参数量 > 0。"""
        assert encoder.n_params > 0

    def test_n_params_at_least_10k(self):
        """默认配置参数量 ≥ 10K。"""
        enc = TrueJEPAEncoder(TrueJEPAConfig(obs_dim=64, latent_dim=128))
        assert enc.n_params >= 10000, f"参数量 {enc.n_params} < 10000"

    def test_latent_dim_property(self, encoder):
        assert encoder.latent_dim == 32

    def test_repr(self, encoder):
        r = repr(encoder)
        assert "TrueJEPAEncoder" in r
        assert "latent_dim" in r
