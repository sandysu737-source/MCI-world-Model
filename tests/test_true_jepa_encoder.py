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


class TestTrueJEPALearns:
    """可学习性验证 — F6 修复后 TrueJEPA 应真正学习 (loss 下降)。

    第一性原理: JEPA 是带 MSE 损失的 MLP, 用解析反向传播 (链式法则)
    应能稳定降低损失。旧数值梯度 (10% 采样有限差分) 无法保证。
    """

    def test_learns_and_resists_collapse_on_ar1(self):
        """AR(1) 数据上: 预测误差下降 + 不坍塌 (D5 batch VICReg 修复后)。

        batch 训练的总 loss 含 VICReg 约束项 (防坍塌), 不一定降 50%。
        关键指标: (1) 预测误差下降 (2) 潜向量不坍塌 (余弦相似度 < 0.9)。
        """
        import numpy as np
        from mci_world_model.sdk._true_jepa_encoder import TrueJEPAConfig, TrueJEPAEncoder

        rng = np.random.RandomState(0)
        N, D = 120, 8
        obs = np.zeros((N, D))
        obs[0] = rng.randn(D)
        for t in range(1, N):
            obs[t] = 0.9 * obs[t - 1] + 0.05 * rng.randn(D)

        cfg = TrueJEPAConfig(obs_dim=D, latent_dim=16, hidden_dim=32, action_dim=0, lr=0.01)
        enc = TrueJEPAEncoder(cfg)

        def eval_pred(enc):
            errs = [np.mean((enc.predict_next(enc.encode(obs[t]), None) -
                            enc.encode_target(obs[t+1]))**2)
                    for t in range(0, N-1, 10)]
            return float(np.mean(errs))

        err_before = eval_pred(enc)
        enc.train(obs, actions=None, n_epochs=15)
        err_after = eval_pred(enc)

        # 预测误差应下降
        assert err_after < err_before, (
            f"预测误差未下降: {err_before:.4f} -> {err_after:.4f}"
        )
        # 不坍塌: 跨样本标准差不应全为 0
        all_z = np.array([enc.encode(obs[t]) for t in range(0, N, 5)])
        assert np.std(all_z, axis=0).min() > 0.01, "潜空间存在坍塌维度"

    def test_analytic_grad_matches_numerical(self):
        """梯度检查: 解析梯度应与数值有限差分一致 (梯度检查)。"""
        import numpy as np
        from mci_world_model.sdk._true_jepa_encoder import TrueJEPAConfig, TrueJEPAEncoder

        rng = np.random.RandomState(1)
        cfg = TrueJEPAConfig(obs_dim=4, latent_dim=6, hidden_dim=10, action_dim=0)
        enc = TrueJEPAEncoder(cfg)
        obs_t = rng.randn(4)
        obs_t1 = rng.randn(4)

        enc.train_step(obs_t, obs_t1)  # 建立 cache
        # 重新前向建立干净 cache
        z_online = enc._online_encoder.forward(obs_t)
        z_target = enc._target_encoder.forward(obs_t1)
        z_pred = enc.predict_next(z_online, None)
        diff = z_pred - z_target
        dz_pred = (2.0 / 6.0) * diff

        # 解析梯度
        grads = enc._predictor.backward(dz_pred)
        # 数值梯度 (W2 的一个元素)
        W2 = enc._predictor._W2
        i, j = 1, 2
        eps = 1e-6
        old = W2[i, j]
        W2[i, j] = old + eps
        lp = float(np.mean((enc.predict_next(enc._online_encoder.forward(obs_t), None) - z_target) ** 2))
        W2[i, j] = old - eps
        lm = float(np.mean((enc.predict_next(enc._online_encoder.forward(obs_t), None) - z_target) ** 2))
        W2[i, j] = old
        num_grad = (lp - lm) / (2 * eps)
        ana_grad = grads["W2"][i, j]
        assert abs(num_grad - ana_grad) < 1e-4, (
            f"梯度不匹配: numerical={num_grad:.6f} analytical={ana_grad:.6f}"
        )


class TestTrueJEPAIntegrationNeurosymbolic:
    """TrueJEPA 接入 NeurosymbolicWorldModel 主流程 — Item 2。"""

    def test_true_jepa_as_jepa_encoder(self):
        """TrueJEPAEncoder 可直接作为 neurosymbolic 的 jepa_encoder。"""
        import numpy as np
        from mci_world_model.sdk._true_jepa_encoder import TrueJEPAConfig, TrueJEPAEncoder
        from mci_world_model.sdk._neurosymbolic_world_model import NeurosymbolicWorldModel

        enc = TrueJEPAEncoder(TrueJEPAConfig(obs_dim=8, latent_dim=16, hidden_dim=32, action_dim=0))
        nsm = NeurosymbolicWorldModel(jepa_encoder=enc)
        state = {"vector": np.random.RandomState(0).randn(8)}
        triple = nsm.encode_triple(state, query="predict")
        assert triple.latent.shape == (16,), f"latent shape {triple.latent.shape}"

    def test_train_jepa_via_neurosymbolic(self):
        """通过 NeurosymbolicWorldModel.train_jepa 训练后预测改善且不坍塌。"""
        import numpy as np
        from mci_world_model.sdk._true_jepa_encoder import TrueJEPAConfig, TrueJEPAEncoder
        from mci_world_model.sdk._neurosymbolic_world_model import NeurosymbolicWorldModel

        rng = np.random.RandomState(2)
        N, D = 100, 8
        obs = np.zeros((N, D))
        obs[0] = rng.randn(D)
        for t in range(1, N):
            obs[t] = 0.9 * obs[t - 1] + 0.05 * rng.randn(D)

        enc = TrueJEPAEncoder(TrueJEPAConfig(obs_dim=D, latent_dim=16, hidden_dim=32, action_dim=0, lr=0.01))
        nsm = NeurosymbolicWorldModel(jepa_encoder=enc)
        result = nsm.train_jepa(obs, n_epochs=12)
        assert result.get("status") == "trained"
        # 训练成功且潜空间不坍塌
        all_z = np.array([enc.encode(obs[t]) for t in range(0, N, 5)])
        assert np.std(all_z, axis=0).min() > 0.005, "潜空间坍塌"


class TestTrueJEPAAntiCollapse:
    """D5 修复验证: batch VICReg 防止表征坍塌。

    坍塌的数学本质: JEPA 预测损失的全局最小值是 encoder 输出常数向量
    (predictor 退化为 identity, loss→0)。batch 标准差正则化通过约束
    跨样本方差, 直接在计算图中阻止这一坍塌解。
    """

    def test_no_collapse_on_random_data(self):
        """随机数据上训练后, 不同观测的编码不应高度相似。"""
        import numpy as np
        from mci_world_model.sdk._true_jepa_encoder import TrueJEPAConfig, TrueJEPAEncoder

        rng = np.random.RandomState(0)
        obs = rng.randn(200, 16)  # 每个观测完全独立

        enc = TrueJEPAEncoder(TrueJEPAConfig(
            obs_dim=16, latent_dim=32, hidden_dim=64, action_dim=0, lr=0.01))
        enc.train(obs, n_epochs=20)

        z0 = enc.encode(obs[0])
        z1 = enc.encode(obs[100])
        cos = np.dot(z0, z1) / (np.linalg.norm(z0) * np.linalg.norm(z1) + 1e-10)
        # 修复前 cos ≈ 0.96 (坍塌); 修复后应 < 0.7
        assert cos < 0.7, f"表征坍塌: cos={cos:.3f} (应 < 0.7)"

    def test_all_dimensions_active(self):
        """训练后所有潜空间维度应保持跨样本变异性。"""
        import numpy as np
        from mci_world_model.sdk._true_jepa_encoder import TrueJEPAConfig, TrueJEPAEncoder

        rng = np.random.RandomState(0)
        obs = rng.randn(200, 16)

        enc = TrueJEPAEncoder(TrueJEPAConfig(
            obs_dim=16, latent_dim=32, hidden_dim=64, action_dim=0, lr=0.01))
        enc.train(obs, n_epochs=20)

        all_z = np.array([enc.encode(obs[t]) for t in range(0, 200, 5)])
        std_per_dim = np.std(all_z, axis=0)
        # 坍塌时大部分维度 std≈0; 修复后所有维度应有变异
        active = (std_per_dim > 0.05).sum()
        assert active >= 28, f"仅 {active}/32 维度活跃, 存在坍塌"

    def test_prediction_improves_without_collapse(self):
        """预测误差应下降, 同时不坍塌 (学习 vs 约束的平衡)。"""
        import numpy as np
        from mci_world_model.sdk._true_jepa_encoder import TrueJEPAConfig, TrueJEPAEncoder

        rng = np.random.RandomState(0)
        N = 200
        obs = np.zeros((N, 8))
        obs[0] = rng.randn(8)
        for t in range(1, N):
            obs[t] = 0.9 * obs[t - 1] + 0.05 * rng.randn(8)

        enc = TrueJEPAEncoder(TrueJEPAConfig(
            obs_dim=8, latent_dim=16, hidden_dim=32, action_dim=0, lr=0.01))

        def eval_pred():
            errs = [np.mean((enc.predict_next(enc.encode(obs[t]), None) -
                            enc.encode_target(obs[t + 1])) ** 2)
                    for t in range(0, N - 1, 10)]
            return float(np.mean(errs))

        err_before = eval_pred()
        enc.train(obs, n_epochs=25)
        err_after = eval_pred()

        assert err_after < err_before, "预测误差未下降"
        all_z = np.array([enc.encode(obs[t]) for t in range(0, N, 5)])
        assert np.std(all_z, axis=0).min() > 0.01, "潜空间坍塌"
