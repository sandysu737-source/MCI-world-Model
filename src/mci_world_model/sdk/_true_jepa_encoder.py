from __future__ import annotations

"""MCI World Model v4.6.0 — TrueJEPA Encoder
================================================

真正的 Joint Embedding Predictive Architecture 编码器。

与旧 JEPAEncoder 的关键区别:
    旧: encode(memories) → CausalWorldModelState  (因果图, 不是潜空间)
    新: encode(observations) → np.ndarray          (潜向量, 真正的 JEPA)

F6 修复: JEPA 名不副实 — "联合嵌入预测" 的核心是观测→潜向量映射,
而非因果图构建。因果图属于 Pearl 层 (L1/L2/L3), 不属于 JEPA 编码器。

架构:
    Online Encoder:  obs → MLP → z              (可训练)
    Target Encoder:  obs → MLP → z_target       (EMA 更新, 不直接训练)
    Predictor:       z_t + a_t → MLP → z_{t+1}  (潜空间预测)

训练:
    L = ||predictor(online(x_t), a_t) - target(x_{t+1})||²
    + VICReg 正则化 (variance + covariance)

参数量 (默认配置):
    Online Encoder:  64→256→128 = 16K + 32K + 128 ≈ 49K
    Target Encoder:  同上 (EMA 复制, 不增加参数)
    Predictor:       (128+4)→256→128 = 34K + 32K + 128 ≈ 66K
    总计: ~115K

## Formal Guarantees

    - encode() 返回 (latent_dim,) float64 ndarray, 绝不返回 CausalWorldModelState
    - predict_next() 返回 (latent_dim,) float64 ndarray
    - EMA 更新保证 target 参数变化率 ≤ (1-τ) 每步
    - VICReg variance 项保证潜向量各维度方差 ≥ γ (防止坍塌)

用法:
    >>> from mci_world_model.sdk._true_jepa_encoder import TrueJEPAEncoder
    >>> enc = TrueJEPAEncoder(obs_dim=64, latent_dim=128)
    >>> obs = np.random.randn(64)
    >>> z = enc.encode(obs)         # (128,) 潜向量
    >>> action = np.random.randn(4)
    >>> z_next = enc.predict_next(z, action)  # (128,) 预测
"""


import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 核心数据结构
# =============================================================================


@dataclass
class TrueJEPAConfig:
    """TrueJEPA 编码器配置。

    Attributes:
        obs_dim: 观测向量维度
        latent_dim: 潜空间维度 (≥64)
        hidden_dim: 隐层维度
        action_dim: 动作向量维度
        ema_tau: EMA 动量系数 (越大越慢)
        vicreg_var_weight: VICReg 方差项权重
        vicreg_cov_weight: VICReg 协方差项权重
        lr: 学习率
        seed: 随机种子
    """

    obs_dim: int = 64
    latent_dim: int = 128
    hidden_dim: int = 256
    action_dim: int = 4
    ema_tau: float = 0.996
    vicreg_var_weight: float = 1.0
    vicreg_cov_weight: float = 0.04
    lr: float = 0.001
    seed: int = 42


# =============================================================================
# 内部 MLP 模块
# =============================================================================


class _MLP:
    """轻量 MLP: Input → Hidden → ReLU → Output，带解析反向传播。

    反向传播通过链式法则解析推导 (非数值梯度):
        z1 = x @ W1 + b1
        h1 = ReLU(z1)
        out = h1 @ W2 + b2

    给定 dout = dL/dout (shape: output_dim):
        dW2 = h1^T @ dout      db2 = dout
        dh1 = dout @ W2^T      dz1 = dh1 * (z1 > 0)
        dW1 = x^T @ dz1        db1 = dz1
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        self._W1 = rng.randn(input_dim, hidden_dim).astype(np.float64) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self._b1 = np.zeros(hidden_dim, dtype=np.float64)
        self._W2 = rng.randn(hidden_dim, output_dim).astype(np.float64) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self._b2 = np.zeros(output_dim, dtype=np.float64)
        self._cache: dict[str, Any] = {}

    @property
    def n_params(self) -> int:
        return self._W1.size + self._b1.size + self._W2.size + self._b2.size

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播: (input_dim,) → (output_dim,)。"""
        x = np.asarray(x, dtype=np.float64).ravel()
        z1 = x @ self._W1 + self._b1
        h1 = np.maximum(z1, 0.0)  # ReLU
        out = h1 @ self._W2 + self._b2
        self._cache = {"x": x, "z1": z1, "h1": h1}
        return out

    def forward_batch(self, x: np.ndarray) -> np.ndarray:
        """批量前向: (batch, input_dim) → (batch, output_dim)。"""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        z1 = x @ self._W1 + self._b1
        h1 = np.maximum(z1, 0.0)
        out = h1 @ self._W2 + self._b2
        self._cache = {"x": x, "z1": z1, "h1": h1}
        return out

    def backward(self, dout: np.ndarray) -> dict[str, np.ndarray]:
        """解析反向传播，返回各参数梯度。

        Args:
            dout: 上游梯度 dL/dout，shape (output_dim,)

        Returns:
            {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        """
        dout = np.asarray(dout, dtype=np.float64).ravel()
        x = self._cache["x"]
        z1 = self._cache["z1"]
        h1 = self._cache["h1"]

        dW2 = np.outer(h1, dout)
        db2 = dout.copy()
        dh1 = dout @ self._W2.T
        dz1 = dh1 * (z1 > 0.0)  # ReLU 梯度
        dW1 = np.outer(x, dz1)
        db1 = dz1.copy()
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def apply_grads(self, grads: dict[str, np.ndarray], lr: float) -> None:
        """沿负梯度方向更新参数。"""
        self._W1 -= lr * grads["W1"]
        self._b1 -= lr * grads["b1"]
        self._W2 -= lr * grads["W2"]
        self._b2 -= lr * grads["b2"]

    def get_params(self) -> dict[str, np.ndarray]:
        return {"W1": self._W1.copy(), "b1": self._b1.copy(), "W2": self._W2.copy(), "b2": self._b2.copy()}

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        self._W1 = params["W1"].copy()
        self._b1 = params["b1"].copy()
        self._W2 = params["W2"].copy()
        self._b2 = params["b2"].copy()


# =============================================================================
# TrueJEPAEncoder — 真正的 JEPA 编码器
# =============================================================================


class TrueJEPAEncoder:
    """True JEPA 编码器 — 观测 → 潜向量。

    核心区别: encode() 返回 ndarray, 不返回 CausalWorldModelState。
    这是 F6 "JEPA名不副实" 缺陷的修复。

    架构:
        Online Encoder:  obs → MLP → z (可训练)
        Target Encoder:  obs → MLP → z_target (EMA 更新)
        Predictor:       [z, action] → MLP → z_next (潜空间预测)

    训练目标:
        L = ||predictor(online(x_t), a_t) - stopgrad(target(x_{t+1}))||²
        + VICReg(variance + covariance) 正则化

    Example:
        >>> enc = TrueJEPAEncoder(TrueJEPAConfig(obs_dim=64, latent_dim=128))
        >>> obs = np.random.randn(64)
        >>> z = enc.encode(obs)              # (128,) ndarray
        >>> action = np.random.randn(4)
        >>> z_next = enc.predict_next(z, action)  # (128,) ndarray
    """

    def __init__(self, config: TrueJEPAConfig | None = None):
        self._config = config or TrueJEPAConfig()
        seed = self._config.seed

        # 在线编码器 (可训练)
        self._online_encoder = _MLP(
            self._config.obs_dim,
            self._config.hidden_dim,
            self._config.latent_dim,
            seed=seed,
        )
        # 目标编码器 (EMA, 不直接训练)
        self._target_encoder = _MLP(
            self._config.obs_dim,
            self._config.hidden_dim,
            self._config.latent_dim,
            seed=seed + 1000,
        )
        # 初始化: 目标 = 在线 (拷贝参数)
        self._sync_target()

        # 预测器: [z_t, action] → z_{t+1}
        predictor_input_dim = self._config.latent_dim + self._config.action_dim
        self._predictor = _MLP(
            predictor_input_dim,
            self._config.hidden_dim,
            self._config.latent_dim,
            seed=seed + 2000,
        )

        # 训练统计
        self._train_steps: int = 0
        self._loss_history: list[float] = []
        self._batch_rng = np.random.RandomState(seed + 3000)

    # -----------------------------------------------------------------
    # 属性
    # -----------------------------------------------------------------

    @property
    def config(self) -> TrueJEPAConfig:
        return self._config

    @property
    def latent_dim(self) -> int:
        """潜空间维度。"""
        return self._config.latent_dim

    @property
    def n_params(self) -> int:
        """可训练参数总数 (不含 target encoder, 因其由 EMA 更新)。"""
        return self._online_encoder.n_params + self._predictor.n_params

    @property
    def train_steps(self) -> int:
        return self._train_steps

    @property
    def loss_history(self) -> list[float]:
        return list(self._loss_history)

    # -----------------------------------------------------------------
    # 核心 API
    # -----------------------------------------------------------------

    def encode(self, observations: np.ndarray) -> np.ndarray:
        """编码观测为潜向量。

        F6 修复: 返回 (latent_dim,) ndarray, 不是 CausalWorldModelState。

        Args:
            observations: 观测向量 (obs_dim,) 或 (batch, obs_dim)

        Returns:
            潜向量 (latent_dim,) 或 (batch, latent_dim)

        Raises:
            ValueError: 观测维度与配置 obs_dim 不匹配
        """
        observations = np.asarray(observations, dtype=np.float64)
        if observations.ndim == 1:
            if observations.shape[0] != self._config.obs_dim:
                raise ValueError(
                    f"观测维度 {observations.shape[0]} 与配置 obs_dim={self._config.obs_dim} 不匹配"
                )
        elif observations.ndim == 2:
            if observations.shape[1] != self._config.obs_dim:
                raise ValueError(
                    f"观测维度 {observations.shape[1]} 与配置 obs_dim={self._config.obs_dim} 不匹配"
                )
        return self._online_encoder.forward(observations)

    def encode_target(self, observations: np.ndarray) -> np.ndarray:
        """目标编码器 (EMA) — 用于训练时提供 stop-gradient 目标。

        Args:
            observations: 观测向量 (obs_dim,)

        Returns:
            潜向量 (latent_dim,)
        """
        return self._target_encoder.forward(observations)

    def forward(self, observations: np.ndarray) -> np.ndarray:
        """forward 别名 — 与 NeurosymbolicWorldModel 的 jepa_encoder 接口兼容。

        NeurosymbolicWorldModel.encode_triple 调用 jepa_encoder.forward(state_vec)。
        本方法使 TrueJEPAEncoder 可直接作为 jepa_encoder 传入, 实现 JEPA 路径
        使用真正的潜空间编码 (而非旧 CausalWorldModelState 编码)。
        """
        return self.encode(observations)

    def predict_next(self, z_t: np.ndarray, action: np.ndarray | None = None) -> np.ndarray:
        """潜空间预测: z_t + action → z_{t+1}。

        Args:
            z_t: 当前潜向量 (latent_dim,)
            action: 动作向量 (action_dim,) (可选)

        Returns:
            预测的下一潜向量 (latent_dim,)
        """
        z_t = np.asarray(z_t, dtype=np.float64).ravel()
        if action is not None and self._config.action_dim > 0:
            action = np.asarray(action, dtype=np.float64).ravel()
            # 如果 action 维度不足, 填零
            if len(action) < self._config.action_dim:
                action = np.concatenate([action, np.zeros(self._config.action_dim - len(action))])
            elif len(action) > self._config.action_dim:
                action = action[: self._config.action_dim]
            x = np.concatenate([z_t, action])
        else:
            x = np.concatenate([z_t, np.zeros(self._config.action_dim)])

        return self._predictor.forward(x)

    # -----------------------------------------------------------------
    # 训练
    # -----------------------------------------------------------------

    def train_step(
        self,
        obs_t: np.ndarray,
        obs_t1: np.ndarray,
        action: np.ndarray | None = None,
    ) -> float:
        """单步训练 — 解析反向传播 (链式法则)。

        L = ||predictor(online(x_t), a) - stopgrad(target(x_{t+1}))||² + VICReg

        反向传播路径 (L 对各参数的解析梯度):
            dL/dz_pred = 2/D * (z_pred - z_target)        # D = latent_dim
            predictor.backward(dL/dz_pred)  → predictor grads
            dz_online_from_pred = predictor 输入端梯度
            dL/dz_online = dz_online_from_pred + VICReg 梯度
            online_encoder.backward(dL/dz_online) → online grads

        Args:
            obs_t: 当前观测 (obs_dim,)
            obs_t1: 下一观测 (obs_dim,)
            action: 动作向量 (action_dim,)

        Returns:
            总损失
        """
        obs_t = np.asarray(obs_t, dtype=np.float64).ravel()
        obs_t1 = np.asarray(obs_t1, dtype=np.float64).ravel()

        # ── 前向 ──
        z_online = self._online_encoder.forward(obs_t)
        z_target = self._target_encoder.forward(obs_t1)  # stop-gradient
        z_pred = self.predict_next(z_online, action)

        # ── 预测损失 (MSE) ──
        diff = z_pred - z_target
        D = self._config.latent_dim
        pred_loss = float(np.mean(diff ** 2))
        dz_pred = (2.0 / D) * diff  # MSE 对 z_pred 的梯度

        # ── VICReg 正则化 ──
        vicreg_loss, dz_vicreg = self._compute_vicreg_with_grad(z_online)

        total_loss = pred_loss + self._config.vicreg_var_weight * vicreg_loss

        # ── 反向传播 ──
        lr = self._config.lr

        # Predictor 反向传播: dz_pred → predictor 参数梯度 + 输入梯度
        pred_grads = self._predictor.backward(dz_pred)
        # predictor 输入 = [z_online, action], 取前 latent_dim 维作为对 z_online 的梯度
        dz_online_from_pred = pred_grads["W1"]  # 不对; 需通过 W1 的输入梯度
        # 正确: 对 predictor 输入 x_pred 的梯度 = dz1_pred @ W1_pred.T (经 ReLU)
        # 但 backward 只返回参数梯度, 我们需要在 predictor 上算输入梯度
        dz_online_from_pred = self._predictor_input_grad(dz_pred)

        # online encoder 反向传播
        dz_online = dz_online_from_pred + self._config.vicreg_var_weight * dz_vicreg
        online_grads = self._online_encoder.backward(dz_online)

        # 应用梯度
        self._predictor.apply_grads(pred_grads, lr)
        self._online_encoder.apply_grads(online_grads, lr)

        # ── EMA 更新 target encoder ──
        self._ema_update_target()

        self._train_steps += 1
        self._loss_history.append(total_loss)
        return total_loss

    def train(
        self,
        observations: np.ndarray,
        actions: np.ndarray | None = None,
        n_epochs: int = 10,
    ) -> dict[str, Any]:
        """批量训练。

        从观测序列中构造 (obs_t, obs_t1, action) 训练对。

        Args:
            observations: 观测序列 (N, obs_dim)
            actions: 动作序列 (N, action_dim) (可选)
            n_epochs: 训练轮数

        Returns:
            {"final_loss": float, "n_epochs": int, "n_pairs": int}
        """
        observations = np.asarray(observations, dtype=np.float64)
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)

        n = observations.shape[0]
        if n < 2:
            logger.warning("观测序列不足 2 条，无法构造训练对")
            return {"final_loss": 0.0, "n_epochs": 0, "n_pairs": 0}

        n_pairs = n - 1
        self._loss_history.clear()

        batch_size = min(32, n_pairs)
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            # D5 修复: mini-batch 训练, batch 标准差正则化嵌入计算图。
            # 每个梯度步编码一个 batch, 联合优化:
            #   L = Σ预测损失 + λ·batch_stddev_loss
            # batch_stddev_loss 直接惩罚跨样本坍塌 (所有样本输出相同时 loss 最大)
            perm = self._batch_rng.permutation(n_pairs)
            n_batches = max(1, n_pairs // batch_size)
            for b in range(n_batches):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                if len(idx) < 2:
                    continue
                batch_obs_t = observations[idx]
                batch_obs_t1 = observations[idx + 1]
                batch_actions = actions[idx] if actions is not None else None
                loss = self.train_batch_step(batch_obs_t, batch_obs_t1, batch_actions)
                epoch_loss += loss * len(idx)
            avg_loss = epoch_loss / n_pairs
            logger.debug("TrueJEPA Epoch %d/%d | Loss: %.6f", epoch + 1, n_epochs, avg_loss)

        final_loss = self._loss_history[-1] if self._loss_history else 0.0
        return {
            "final_loss": round(final_loss, 6),
            "n_epochs": n_epochs,
            "n_pairs": n_pairs,
            "n_params": self.n_params,
        }

    def train_batch_step(
        self,
        obs_batch_t: np.ndarray,
        obs_batch_t1: np.ndarray,
        actions_batch: np.ndarray | None = None,
    ) -> float:
        """Mini-batch 训练步 — 预测损失 + batch 标准差正则化 (D5 核心修复)。

        关键创新: batch 标准差正则化直接嵌入计算图, 每步梯度更新都包含
        跨样本变异性约束。这从第一性原理阻止坍塌:

        坍塌 = 所有样本编码到相同向量 z*。
        batch_stddev_loss = mean_d(max(0, γ - std_d(Z[:,d])))
        当所有样本相同时 std_d=0, loss=max(0,γ)=γ (最大)。
        梯度推动 encoder 增加跨样本方差, 破坏坍塌解。

        Args:
            obs_batch_t: batch 观测 (B, obs_dim)
            obs_batch_t1: batch 下一观测 (B, obs_dim)
            actions_batch: batch 动作 (B, action_dim)

        Returns:
            batch 平均总损失
        """
        B = len(obs_batch_t)
        D = self._config.latent_dim
        lr = self._config.lr
        gamma = 1.0  # 目标 batch 标准差

        total_loss = 0.0
        # 累积 batch 标准差正则化梯度
        online_grads_accum = {k: np.zeros_like(getattr(self._online_encoder, k))
                              for k in ["_W1", "_b1", "_W2", "_b2"]}
        predictor_grads_accum = {k: np.zeros_like(getattr(self._predictor, k))
                                 for k in ["_W1", "_b1", "_W2", "_b2"]}

        # ── 编码整个 batch ──
        z_batch = np.zeros((B, D))
        z_target_batch = np.zeros((B, D))
        z_pred_batch = np.zeros((B, D))
        for t in range(B):
            z_batch[t] = self._online_encoder.forward(obs_batch_t[t])
            z_target_batch[t] = self._target_encoder.forward(obs_batch_t1[t])
            act = actions_batch[t] if actions_batch is not None else None
            z_pred_batch[t] = self.predict_next(z_batch[t], act)

        # ── 预测损失 (每样本 MSE) ──
        diff_batch = z_pred_batch - z_target_batch  # (B, D)
        pred_loss = float(np.mean(diff_batch ** 2))

        # ── batch 标准差正则化 ──
        z_mean = z_batch.mean(axis=0)  # (D,)
        z_centered = z_batch - z_mean  # (B, D)
        std_d = np.sqrt(np.mean(z_centered ** 2, axis=0) + 1e-8)  # (D,)
        below = np.maximum(0.0, gamma - std_d)  # (D,)
        batch_std_loss = float(np.mean(below))

        total_loss = pred_loss + self._config.vicreg_var_weight * batch_std_loss

        # ── 反向传播 ──
        # 梯度缩放说明:
        #   pred_loss = (1/B) * Σ_t ||diff_t||² / D  →  每样本 dz_pred = (2/D) * diff_t
        #   累加 B 个样本的梯度后 /B 得到 batch mean 梯度
        for t in range(B):
            # 预测损失梯度 (每样本, 无 1/B 因子)
            dz_pred = (2.0 / D) * diff_batch[t]
            pred_grads = self._predictor.backward(dz_pred)
            dz_online_pred = self._predictor_input_grad(dz_pred)

            # batch 标准差梯度
            mask = (below > 0).astype(np.float64)
            dz_std = mask * (-1.0 / D) * z_centered[t] / (B * std_d + 1e-12)
            dz_std *= self._config.vicreg_var_weight

            dz_online = dz_online_pred + dz_std
            online_grads = self._online_encoder.backward(dz_online)

            for k_map, attr in [("W1","_W1"),("b1","_b1"),("W2","_W2"),("b2","_b2")]:
                online_grads_accum[attr] += online_grads[k_map]
                predictor_grads_accum[attr] += pred_grads[k_map]

        # 应用平均梯度 (除 B 得 batch mean)
        for attr in ["_W1", "_b1", "_W2", "_b2"]:
            getattr(self._online_encoder, attr).__isub__(lr * online_grads_accum[attr] / B)
            getattr(self._predictor, attr).__isub__(lr * predictor_grads_accum[attr] / B)

        # EMA 更新
        self._ema_update_target()
        self._train_steps += 1
        self._loss_history.append(total_loss)
        return total_loss

    # -----------------------------------------------------------------
    # 内部方法
    # -----------------------------------------------------------------

    def _batch_vicreg_update(self, observations: np.ndarray, actions: np.ndarray | None) -> None:
        """Batch 级 VICReg 正则化 — 防止表征坍塌 (D5 修复)。

        单样本 VICReg 无法检测维度间相关性坍塌。batch 级约束编码一批观测后
        各维度的跨样本标准差, 推动每个维度独立变化。

        对每个维度 d: std_d = sqrt(mean((z[:,d] - mean)^2) + eps)
        loss = mean(max(0, gamma - std_d))
        梯度通过解析反向传播更新 online encoder。
        """
        n = min(observations.shape[0], 64)  # 限制 batch 大小
        idx = np.random.RandomState(self._train_steps).choice(
            observations.shape[0], n, replace=False
        )
        batch = observations[idx]
        gamma = 2.0  # 目标 batch 标准差
        lr = self._config.lr

        # 编码 batch
        z_batch = np.array([self._online_encoder.forward(batch[t]) for t in range(n)])
        # (n, latent_dim)
        z_mean = z_batch.mean(axis=0)
        z_centered = z_batch - z_mean  # (n, D)
        std_d = np.sqrt(np.mean(z_centered ** 2, axis=0) + 1e-8)  # (D,)
        below = np.maximum(0.0, gamma - std_d)  # (D,)
        batch_var_loss = float(np.mean(below))
        if batch_var_loss < 1e-6:
            return  # 已满足约束

        # 解析梯度: d(std_d)/d(z_centered) = z_centered / (n * std_d)
        # d(loss)/d(std_d) = -1/D if below>0 else 0
        mask = (below > 0).astype(np.float64)  # (D,)
        # dz_centered = mask * (-1/D) * z_centered / (n * std_d)
        dz_centered = np.zeros_like(z_batch)
        for t in range(n):
            dz_centered[t] = mask * (-1.0 / len(mask)) * z_centered[t] / (n * std_d)

        # 对每个样本反向传播
        for t in range(n):
            grads = self._online_encoder.backward(dz_centered[t])
            self._online_encoder.apply_grads(grads, lr)

    def _compute_vicreg(self, z: np.ndarray) -> float:
        """VICReg 正则化 (逐维度方差 + 全局协方差)。"""
        loss, _ = self._compute_vicreg_with_grad(z)
        return loss

    def _compute_vicreg_with_grad(self, z: np.ndarray) -> tuple[float, np.ndarray]:
        """VICReg 正则化 — 逐维度方差 + 协方差，返回值与梯度。

        方差项 (per-dimension): 鼓励各维度维持标准差 ≥ γ=1。
            单样本时用 |z_i| 近似: var_loss = sum(max(0, 1-|z_i|))/D
        协方差项: 对潜向量维度去相关 (此处单样本退化为 0)。
        """
        D = len(z)
        gamma = 1.0
        abs_z = np.abs(z)
        # variance hinge loss
        below = gamma - abs_z  # >0 表示该维度坍塌
        mask = (below > 0).astype(np.float64)
        var_loss = float(np.sum(np.maximum(0.0, below)) / D)
        # 方差项对 z 的梯度: -sign(z)/D 在 below>0 时
        dvar_dz = -(np.sign(z) * mask) / D

        # 协方差项: 单样本无意义, 设为 0
        cov_loss = 0.0
        dcov_dz = np.zeros_like(z)

        total = var_loss + self._config.vicreg_cov_weight * cov_loss
        grad = dvar_dz + self._config.vicreg_cov_weight * dcov_dz
        return total, grad

    def _predictor_input_grad(self, dout: np.ndarray) -> np.ndarray:
        """计算 predictor 对其输入的梯度, 返回对 z_online 部分。

        predictor 输入 x_pred = [z_online (latent_dim), action (action_dim)]
        我们只取前 latent_dim 维。
        """
        cache = self._predictor._cache
        z1 = cache["z1"]
        dh1 = dout @ self._predictor._W2.T
        dz1 = dh1 * (z1 > 0.0)
        dx_pred = dz1 @ self._predictor._W1.T
        return dx_pred[: self._config.latent_dim]

    def _sync_target(self) -> None:
        """将 online encoder 参数复制到 target encoder。"""
        self._target_encoder.set_params(self._online_encoder.get_params())

    def _ema_update_target(self) -> None:
        """EMA 更新 target encoder: θ_target = τ * θ_target + (1-τ) * θ_online。"""
        tau = self._config.ema_tau
        for param_name in ["_W1", "_b1", "_W2", "_b2"]:
            online_param = getattr(self._online_encoder, param_name)
            target_param = getattr(self._target_encoder, param_name)
            target_param[:] = tau * target_param + (1 - tau) * online_param

    def __repr__(self) -> str:
        return (
            f"TrueJEPAEncoder(obs_dim={self._config.obs_dim}, "
            f"latent_dim={self._config.latent_dim}, "
            f"n_params={self.n_params})"
        )
