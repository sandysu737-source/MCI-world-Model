"""
TASK-A1.5: 手写梯度数值验证测试（Finite Differences）
=========================================================

核心安全门：验证 LearnableStateEncoder 的手写梯度反向传播正确性。

验证方法:
    对每个参数 W，使用中心差分估计数值梯度:
    ∂L/∂W_numerical ≈ [L(W + ε·e_i) - L(W - ε·e_i)] / (2ε)

验证标准:
    ∀ param, max(|grad_analytical - grad_numerical|) / max(|grad_analytical|, |grad_numerical|, 1e-8) < 1e-4
    — 即相对误差不超过 0.01%

如果任一测试失败:
    1. 输出逐元素的误差矩阵
    2. 标记最大误差参数和位置
    3. 检查对应反向传播公式中矩阵转置和广播维度
    4. 修复后重新验证直到通过

KPI: PA-1 手写梯度 finite-diff 验证通过率 = 100%
"""

import numpy as np
import pytest

from mci_world_model.sdk._learnable_encoder import LearnableStateEncoder


def _compute_numerical_gradient(
    encoder: LearnableStateEncoder,
    param_name: str,
    original: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    对指定参数使用中心差分计算数值梯度。

    Args:
        encoder: 编码器实例
        param_name: 参数名 (如 "W1", "b1")
        original: 原始输入向量
        eps: 扰动大小

    Returns:
        与参数同 shape 的数值梯度
    """
    param = getattr(encoder, param_name)
    grad = np.zeros_like(param)
    flat_param = param.ravel()

    for idx in range(flat_param.size):
        # 保存原始值
        old_val = flat_param[idx]

        # +ε
        flat_param[idx] = old_val + eps
        setattr(encoder, param_name, flat_param.reshape(param.shape))
        x_hat_plus = encoder.reconstruct(original)
        loss_plus = float(np.mean((x_hat_plus - original) ** 2))

        # -ε
        flat_param[idx] = old_val - eps
        setattr(encoder, param_name, flat_param.reshape(param.shape))
        x_hat_minus = encoder.reconstruct(original)
        loss_minus = float(np.mean((x_hat_minus - original) ** 2))

        # 恢复原始值
        flat_param[idx] = old_val
        setattr(encoder, param_name, flat_param.reshape(param.shape))

        # 中心差分
        grad.ravel()[idx] = (loss_plus - loss_minus) / (2.0 * eps)

    return grad


def _compare_gradients(
    analytical: np.ndarray,
    numerical: np.ndarray,
    param_name: str,
    rel_tol: float = 1e-4,
) -> dict:
    """
    比较解析梯度和数值梯度。

    Returns:
        {"max_rel_error": float, "passed": bool, "worst_idx": int}
    """
    diff = np.abs(analytical - numerical)
    denom = np.maximum(np.maximum(np.abs(analytical), np.abs(numerical)), 1e-8)
    rel_error = diff / denom

    max_rel_err = float(np.max(rel_error))
    worst_idx = int(np.argmax(rel_error))

    return {
        "max_rel_error": max_rel_err,
        "passed": max_rel_err < rel_tol,
        "worst_idx": worst_idx,
        "analytical_at_worst": float(analytical.ravel()[worst_idx]),
        "numerical_at_worst": float(numerical.ravel()[worst_idx]),
    }


class TestEncoderGradientVerification:
    """LearnableStateEncoder 手写梯度 vs 数值梯度全面验证。"""

    PARAM_NAMES = ["W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4", "W5", "b5", "W6", "b6"]
    REL_TOL = 1e-4  # 相对误差容限 0.01%

    @pytest.fixture
    def encoder_and_input(self):
        """创建编码器和测试输入。"""
        enc = LearnableStateEncoder(state_dim=4, latent_dim=8, hidden_dim=32, seed=42)
        rng = np.random.RandomState(123)
        x = rng.randn(4).astype(np.float64)
        return enc, x

    @pytest.mark.parametrize("param_name", PARAM_NAMES)
    def test_gradient_verification(self, encoder_and_input, param_name):
        """对每个参数验证解析梯度 ≈ 数值梯度。"""
        enc, x = encoder_and_input

        # 1. 训练前向，缓存中间值
        enc.training_forward(x)

        # 2. 计算解析梯度
        result = enc.compute_gradients(x)
        analytical_grad = result["grads"][param_name]

        # 3. 计算数值梯度
        # 对不同参数类型选择不同 ε
        if param_name.startswith("W"):
            eps = 1e-5
        else:  # bias 参数通常更敏感
            eps = 1e-6

        numerical_grad = _compute_numerical_gradient(enc, param_name, x, eps=eps)

        # 4. 比较
        cmp = _compare_gradients(analytical_grad, numerical_grad, param_name, self.REL_TOL)

        assert cmp["passed"], (
            f"Gradient verification FAILED for {param_name}:\n"
            f"  max relative error = {cmp['max_rel_error']:.2e} (tolerance = {self.REL_TOL:.2e})\n"
            f"  worst index = {cmp['worst_idx']}\n"
            f"  analytical = {cmp['analytical_at_worst']:.8e}\n"
            f"  numerical  = {cmp['numerical_at_worst']:.8e}\n"
            f"  Check backward pass for {param_name}'s gradient formula."
        )

    def test_all_params_verified(self, encoder_and_input):
        """汇总验证所有参数的梯度。"""
        enc, x = encoder_and_input
        enc.training_forward(x)
        result = enc.compute_gradients(x)

        failures = []
        for param_name in self.PARAM_NAMES:
            analytical_grad = result["grads"][param_name]
            eps = 1e-5 if param_name.startswith("W") else 1e-6
            numerical_grad = _compute_numerical_gradient(enc, param_name, x, eps=eps)
            cmp = _compare_gradients(analytical_grad, numerical_grad, param_name, self.REL_TOL)
            if not cmp["passed"]:
                failures.append(f"{param_name}: rel_err={cmp['max_rel_error']:.2e}")

        assert not failures, (
            f"Gradient verification failed for {len(failures)}/{len(self.PARAM_NAMES)} params:\n" + "\n".join(failures)
        )


class TestEncoderGradientDifferentInputDims:
    """在不同 state_dim 下验证梯度正确性。"""

    @pytest.mark.parametrize("state_dim", [2, 6, 18])
    def test_gradient_with_varying_state_dim(self, state_dim):
        """不同输入维度下的梯度验证。"""
        enc = LearnableStateEncoder(state_dim=state_dim, latent_dim=8, hidden_dim=32, seed=42)
        rng = np.random.RandomState(456)
        x = rng.randn(state_dim).astype(np.float64)

        enc.training_forward(x)
        result = enc.compute_gradients(x)

        # 抽查关键参数: W1 (编码器第一层), W4 (解码器第一层), W6 (解码器最后一层)
        for param_name in ["W1", "W4", "W6"]:
            analytical_grad = result["grads"][param_name]
            numerical_grad = _compute_numerical_gradient(enc, param_name, x, eps=1e-5)
            cmp = _compare_gradients(analytical_grad, numerical_grad, param_name)
            assert cmp["passed"], (
                f"Gradient check failed for {param_name} with state_dim={state_dim}: rel_err={cmp['max_rel_error']:.2e}"
            )


class TestEncoderGradientEdgeCases:
    """梯度验证的边界情况。"""

    def test_gradient_with_zero_input(self):
        """零输入的梯度验证。"""
        enc = LearnableStateEncoder(state_dim=4, latent_dim=8, hidden_dim=32, seed=42)
        x = np.zeros(4, dtype=np.float64)
        enc.training_forward(x)
        result = enc.compute_gradients(x)

        # 零输入时梯度应该也是有效数值（非 NaN/Inf）
        for param_name in TestEncoderGradientVerification.PARAM_NAMES:
            grad = result["grads"][param_name]
            assert np.all(np.isfinite(grad)), f"Non-finite gradient for {param_name} with zero input"

    def test_gradient_with_large_input(self):
        """大值输入的梯度验证。"""
        enc = LearnableStateEncoder(state_dim=4, latent_dim=8, hidden_dim=32, seed=42)
        x = np.array([100.0, -50.0, 30.0, -80.0], dtype=np.float64)
        enc.training_forward(x)
        result = enc.compute_gradients(x)

        for param_name in ["W1", "W6"]:
            grad = result["grads"][param_name]
            assert np.all(np.isfinite(grad)), f"Non-finite gradient for {param_name} with large input"

    def test_gradient_l2_regularization(self):
        """L2 正则化对梯度的影响。"""
        enc_no_reg = LearnableStateEncoder(state_dim=4, latent_dim=8, hidden_dim=32, seed=42, l2_reg=0.0)
        enc_with_reg = LearnableStateEncoder(state_dim=4, latent_dim=8, hidden_dim=32, seed=42, l2_reg=0.01)

        x = np.array([1.0, -0.5, 0.3, -0.8], dtype=np.float64)

        enc_no_reg.training_forward(x)
        result_no_reg = enc_no_reg.compute_gradients(x)

        enc_with_reg.training_forward(x)
        result_with_reg = enc_with_reg.compute_gradients(x)

        # 正则化应增加 W 参数的梯度（但不影响 bias）
        for w_name in ["W1", "W2", "W3", "W4", "W5", "W6"]:
            grad_diff = result_with_reg["grads"][w_name] - result_no_reg["grads"][w_name]
            # L2 正则梯度 = 2 * l2_reg * W，应该 > 0 当 W > 0
            # 至少梯度应有差异
            assert np.any(grad_diff != 0), f"L2 reg had no effect on {w_name} gradient"

        # bias 不应有 L2 正则化影响
        for b_name in ["b1", "b2", "b3", "b4", "b5", "b6"]:
            np.testing.assert_array_equal(
                result_with_reg["grads"][b_name],
                result_no_reg["grads"][b_name],
                err_msg=f"L2 reg unexpectedly affected {b_name}",
            )
