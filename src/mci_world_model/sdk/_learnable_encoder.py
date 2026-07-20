from __future__ import annotations

"""
MCI World Model v4.6.0 — Learnable State Encoder
====================================================

可学习状态编码器：将 WorldState 向量压缩为低维潜空间表示。

架构:
    Encoder: Input(state_dim) → Linear(hidden=64) → ReLU → Linear(hidden=32) → Linear(latent=16)
    Decoder: Input(latent=16) → Linear(hidden=32) → ReLU → Linear(hidden=64) → ReLU → Linear(state_dim)

训练目标:
    L = MSE(reconstructed, original) + l2_reg * ||W||²

参数量: ~5K (3 层编码器 + 3 层解码器)
训练: 手写梯度反向传播 + SGD（复用 GNNPredictor 的梯度范式）

用法:
    from mci_world_model.sdk._learnable_encoder import LearnableStateEncoder

    encoder = LearnableStateEncoder(state_dim=2, latent_dim=16)
    latent = encoder.forward(state_vector)           # → (16,) 推理
    latent = encoder.training_forward(state_vector)   # → (16,) 训练（含缓存）
    result = encoder.compute_gradients(original)      # → {loss, grads}
    encoder.apply_gradients(result["grads"], lr=0.01)
"""


import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class LearnableStateEncoder:
    """
    可学习状态编码器：自编码器架构，从 WorldState 向量学习压缩潜表示。

    核心设计:
    - 编码器将 state_dim 维输入压缩到 latent_dim 维潜空间
    - 解码器从潜空间重建原始输入
    - MSE 重建损失 + 手写梯度反向传播
    - float64 计算保证梯度数值稳定性
    - 线程安全：缓存使用 threading.Lock 保护

    Example:
        >>> enc = LearnableStateEncoder(state_dim=2, latent_dim=16)
        >>> state = np.array([0.5, -0.3])
        >>> latent = enc.forward(state)       # (16,) 推理
        >>> latent = enc.training_forward(state)  # (16,) 训练
        >>> result = enc.compute_gradients(state)
        >>> enc.apply_gradients(result["grads"], lr=0.01)
    """

    def __init__(
        self,
        state_dim: int = 2,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        seed: int = 42,
        l2_reg: float = 0.0,
    ):
        """
        Args:
            state_dim: 输入状态向量维度
            latent_dim: 潜空间维度
            hidden_dim: 隐层维度
            seed: 随机种子
            l2_reg: L2 正则化系数
        """
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self._state_dim = state_dim
        self._latent_dim = latent_dim
        self._hidden_dim = hidden_dim
        self._l2_reg = l2_reg
        self._rng = np.random.RandomState(seed)

        D = state_dim
        H = hidden_dim
        Z = latent_dim

        # ── 编码器参数 (Xavier/Glorot 初始化) ──
        self.W1: np.ndarray = self._rng.randn(D, H).astype(np.float64) * np.sqrt(2.0 / (D + H))
        self.b1: np.ndarray = np.zeros(H, dtype=np.float64)
        self.W2: np.ndarray = self._rng.randn(H, H // 2).astype(np.float64) * np.sqrt(2.0 / (H + H // 2))
        self.b2: np.ndarray = np.zeros(H // 2, dtype=np.float64)
        self.W3: np.ndarray = self._rng.randn(H // 2, Z).astype(np.float64) * np.sqrt(2.0 / (H // 2 + Z))
        self.b3: np.ndarray = np.zeros(Z, dtype=np.float64)

        # ── 解码器参数 ──
        self.W4: np.ndarray = self._rng.randn(Z, H // 2).astype(np.float64) * np.sqrt(2.0 / (Z + H // 2))
        self.b4: np.ndarray = np.zeros(H // 2, dtype=np.float64)
        self.W5: np.ndarray = self._rng.randn(H // 2, H).astype(np.float64) * np.sqrt(2.0 / (H // 2 + H))
        self.b5: np.ndarray = np.zeros(H, dtype=np.float64)
        self.W6: np.ndarray = self._rng.randn(H, D).astype(np.float64) * np.sqrt(2.0 / (H + D))
        self.b6: np.ndarray = np.zeros(D, dtype=np.float64)

        # ── 前向缓存（训练模式）──
        self._cache: dict[str, Any] = {}
        self._cache_lock = threading.Lock()

        # ── 训练统计 ──
        self._train_steps: int = 0

    # =====================================================================
    # 属性
    # =====================================================================

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def train_steps(self) -> int:
        return self._train_steps

    @property
    def n_params(self) -> int:
        """可训练参数总量。"""
        return sum(
            p.size
            for p in [
                self.W1,
                self.b1,
                self.W2,
                self.b2,
                self.W3,
                self.b3,
                self.W4,
                self.b4,
                self.W5,
                self.b5,
                self.W6,
                self.b6,
            ]
        )

    # =====================================================================
    # 参数访问
    # =====================================================================

    def get_params(self) -> dict[str, np.ndarray]:
        """返回可训练参数副本。"""
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
            "W3": self.W3.copy(),
            "b3": self.b3.copy(),
            "W4": self.W4.copy(),
            "b4": self.b4.copy(),
            "W5": self.W5.copy(),
            "b5": self.b5.copy(),
            "W6": self.W6.copy(),
            "b6": self.b6.copy(),
        }

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        """从字典加载参数。"""
        for name in ["W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4", "W5", "b5", "W6", "b6"]:
            if name in params:
                setattr(self, name, np.asarray(params[name], dtype=np.float64))

    # =====================================================================
    # 推理前向传播
    # =====================================================================

    def forward(self, state_vector: np.ndarray) -> np.ndarray:
        """
        推理模式前向传播（仅编码，不缓存中间值）。

        Args:
            state_vector: shape (state_dim,) 输入状态向量

        Returns:
            shape (latent_dim,) 潜向量
        """
        x = np.asarray(state_vector, dtype=np.float64).ravel()
        if x.shape[0] != self._state_dim:
            raise ValueError(f"Expected state_dim={self._state_dim}, got {x.shape[0]}")
        return self._encode(x)

    def encode(self, state_vector: np.ndarray) -> np.ndarray:
        """forward() 的别名。"""
        return self.forward(state_vector)

    def decode(self, latent_vector: np.ndarray) -> np.ndarray:
        """
        从潜向量重建状态向量（推理模式）。

        Args:
            latent_vector: shape (latent_dim,) 潜向量

        Returns:
            shape (state_dim,) 重建的状态向量
        """
        z = np.asarray(latent_vector, dtype=np.float64).ravel()
        return self._decode(z)

    def reconstruct(self, state_vector: np.ndarray) -> np.ndarray:
        """
        完整的自编码重建：编码 → 解码。

        Args:
            state_vector: shape (state_dim,)

        Returns:
            shape (state_dim,) 重建的状态向量
        """
        latent = self.forward(state_vector)
        return self.decode(latent)

    def reconstruction_loss(self, state_vector: np.ndarray) -> float:
        """
        计算单个样本的重建 MSE 损失。

        Args:
            state_vector: shape (state_dim,)

        Returns:
            MSE 标量值
        """
        x = np.asarray(state_vector, dtype=np.float64).ravel()
        x_recon = self.reconstruct(x)
        return float(np.mean((x_recon - x) ** 2))

    # =====================================================================
    # 内部前向传播
    # =====================================================================

    def _encode(self, x: np.ndarray) -> np.ndarray:
        """编码器前向传播。x: (D,) → z: (Z,)"""
        h1 = x @ self.W1 + self.b1  # (H,)
        a1 = np.maximum(h1, 0.0)  # ReLU
        h2 = a1 @ self.W2 + self.b2  # (H//2,)
        a2 = np.maximum(h2, 0.0)  # ReLU
        z = a2 @ self.W3 + self.b3  # (Z,) 线性输出
        return z

    def _decode(self, z: np.ndarray) -> np.ndarray:
        """解码器前向传播。z: (Z,) → x_hat: (D,)"""
        h4 = z @ self.W4 + self.b4  # (H//2,)
        a4 = np.maximum(h4, 0.0)  # ReLU
        h5 = a4 @ self.W5 + self.b5  # (H,)
        a5 = np.maximum(h5, 0.0)  # ReLU
        x_hat = a5 @ self.W6 + self.b6  # (D,) 线性输出
        return x_hat

    # =====================================================================
    # 训练接口
    # =====================================================================

    def training_forward(self, state_vector: np.ndarray) -> np.ndarray:
        """
        训练模式前向传播（缓存中间值用于反向传播）。

        Args:
            state_vector: shape (state_dim,)

        Returns:
            shape (latent_dim,) 潜向量
        """
        x = np.asarray(state_vector, dtype=np.float64).ravel()
        if x.shape[0] != self._state_dim:
            raise ValueError(f"Expected state_dim={self._state_dim}, got {x.shape[0]}")

        # ── 编码器前向 + 缓存 ──
        h1 = x @ self.W1 + self.b1
        a1 = np.maximum(h1, 0.0)
        h2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(h2, 0.0)
        z = a2 @ self.W3 + self.b3

        # ── 解码器前向 + 缓存 ──
        h4 = z @ self.W4 + self.b4
        a4 = np.maximum(h4, 0.0)
        h5 = a4 @ self.W5 + self.b5
        a5 = np.maximum(h5, 0.0)
        x_hat = a5 @ self.W6 + self.b6

        # ── 缓存中间值 ──
        with self._cache_lock:
            self._cache = {
                "x_input": x,
                "h1": h1,
                "a1": a1,
                "h2": h2,
                "a2": a2,
                "z": z,
                "h4": h4,
                "a4": a4,
                "h5": h5,
                "a5": a5,
                "x_hat": x_hat,
            }

        return z.copy()

    def compute_gradients(self, original_vector: np.ndarray) -> dict[str, Any]:
        """
        计算 MSE 重建损失 + 手写反向传播梯度。

        L = mean((x_hat - x)^2) + l2_reg * sum(||W_i||^2)

        反向传播链:
            dL/dx_hat → dL/da5 → dL/dh5 → dL/da4 → dL/dh4 → dL/dz
            → dL/da2 → dL/dh2 → dL/da1 → dL/dh1 → dL/dW/b

        Args:
            original_vector: 原始输入 state_vector, shape (state_dim,)

        Returns:
            {"loss": float, "mse": float, "l2": float, "grads": {...}}
        """
        with self._cache_lock:
            cache = self._cache.copy()

        if not cache:
            zero_grads = self._zero_grads()
            return {"loss": 0.0, "mse": 0.0, "l2": 0.0, "grads": zero_grads}

        x = cache["x_input"]
        x_hat = cache["x_hat"]

        original = np.asarray(original_vector, dtype=np.float64).ravel()
        if original.shape[0] != self._state_dim:
            raise ValueError(f"Expected state_dim={self._state_dim}, got {original.shape[0]}")

        # ── MSE 损失 ──
        diff = x_hat - original
        D_dim = self._state_dim
        mse = float(np.mean(diff**2))

        # ── L2 正则 ──
        l2 = 0.0
        if self._l2_reg > 0:
            for W in [self.W1, self.W2, self.W3, self.W4, self.W5, self.W6]:
                l2 += float(np.sum(W**2))
            l2 *= self._l2_reg

        loss = mse + l2

        # ── 反向传播 ──
        # dL/dx_hat = 2 * (x_hat - x) / D
        dx_hat = (2.0 / D_dim) * diff  # (D,)

        # ── 解码器反向 ──
        # Layer 6: x_hat = a5 @ W6 + b6
        # dL/dW6 = a5^T @ dx_hat (outer product)
        dW6 = np.outer(cache["a5"], dx_hat)
        db6 = dx_hat
        # dL/da5 = dx_hat @ W6^T
        da5 = dx_hat @ self.W6.T  # (H,)

        # ReLU backward: da5 * (h5 > 0)
        dh5 = da5 * (cache["h5"] > 0).astype(np.float64)

        # Layer 5: h5 = a4 @ W5 + b5
        dW5 = np.outer(cache["a4"], dh5)
        db5 = dh5
        da4 = dh5 @ self.W5.T  # (H//2,)

        # ReLU backward
        dh4 = da4 * (cache["h4"] > 0).astype(np.float64)

        # Layer 4: h4 = z @ W4 + b4
        dW4 = np.outer(cache["z"], dh4)
        db4 = dh4
        dz = dh4 @ self.W4.T  # (Z,)

        # ── 编码器反向 ──
        # Layer 3: z = a2 @ W3 + b3 (线性，无激活)
        dW3 = np.outer(cache["a2"], dz)
        db3 = dz
        da2 = dz @ self.W3.T  # (H//2,)

        # ReLU backward
        dh2 = da2 * (cache["h2"] > 0).astype(np.float64)

        # Layer 2: h2 = a1 @ W2 + b2
        dW2 = np.outer(cache["a1"], dh2)
        db2 = dh2
        da1 = dh2 @ self.W2.T  # (H,)

        # ReLU backward
        dh1 = da1 * (cache["h1"] > 0).astype(np.float64)

        # Layer 1: h1 = x @ W1 + b1
        dW1 = np.outer(x, dh1)
        db1 = dh1

        # ── L2 正则梯度 ──
        if self._l2_reg > 0:
            dW1 += 2.0 * self._l2_reg * self.W1
            dW2 += 2.0 * self._l2_reg * self.W2
            dW3 += 2.0 * self._l2_reg * self.W3
            dW4 += 2.0 * self._l2_reg * self.W4
            dW5 += 2.0 * self._l2_reg * self.W5
            dW6 += 2.0 * self._l2_reg * self.W6

        grads = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
            "W3": dW3,
            "b3": db3,
            "W4": dW4,
            "b4": db4,
            "W5": dW5,
            "b5": db5,
            "W6": dW6,
            "b6": db6,
        }

        return {"loss": loss, "mse": mse, "l2": l2, "grads": grads}

    def apply_gradients(self, grads: dict[str, np.ndarray], lr: float = 0.01) -> None:
        """
        应用梯度更新参数 (SGD)。

        Args:
            grads: compute_gradients() 返回的梯度字典
            lr: 学习率
        """
        for name in ["W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4", "W5", "b5", "W6", "b6"]:
            if name in grads:
                param = getattr(self, name)
                grad = np.asarray(grads[name], dtype=np.float64)
                if param.shape != grad.shape:
                    raise ValueError(f"Gradient shape mismatch for {name}: param {param.shape} vs grad {grad.shape}")
                setattr(self, name, param - lr * grad)

        self._train_steps += 1

    # =====================================================================
    # 训练循环
    # =====================================================================

    def train_on_batch(
        self,
        states: np.ndarray,
        lr: float = 0.01,
    ) -> dict[str, float]:
        """
        在一批样本上执行一步训练。

        Args:
            states: shape (B, state_dim) 或 (state_dim,) 的样本
            lr: 学习率

        Returns:
            {"loss": float, "mse": float} 平均损失
        """
        states = np.atleast_2d(np.asarray(states, dtype=np.float64))
        B = states.shape[0]

        total_loss = 0.0
        total_mse = 0.0

        # 累积梯度
        acc_grads: dict[str, np.ndarray] | None = None

        for i in range(B):
            self.training_forward(states[i])
            result = self.compute_gradients(states[i])

            total_loss += result["loss"]
            total_mse += result["mse"]

            if acc_grads is None:
                acc_grads = {k: v.copy() for k, v in result["grads"].items()}
            else:
                for k in acc_grads:
                    acc_grads[k] += result["grads"][k]

        # 平均梯度
        if acc_grads is not None:
            for k in acc_grads:
                acc_grads[k] /= B
            self.apply_gradients(acc_grads, lr=lr)

        return {
            "loss": total_loss / B,
            "mse": total_mse / B,
        }

    # =====================================================================
    # 序列化
    # =====================================================================

    def save_params(self, path: str) -> None:
        """
        保存参数到 .npz 文件。

        Args:
            path: 文件路径 (建议 .npz 后缀)
        """
        params = self.get_params()
        params["state_dim"] = np.array([self._state_dim])
        params["latent_dim"] = np.array([self._latent_dim])
        params["hidden_dim"] = np.array([self._hidden_dim])
        params["train_steps"] = np.array([self._train_steps])
        np.savez_compressed(path, **params)  # type: ignore
        logger.info("Saved encoder params to %s (%d params)", path, self.n_params)

    def load_params(self, path: str) -> None:
        """
        从 .npz 文件加载参数。

        Args:
            path: 文件路径

        Raises:
            ValueError: 如果维度不匹配
        """
        data = np.load(path)
        # 维度校验
        if "state_dim" in data and int(data["state_dim"][0]) != self._state_dim:
            raise ValueError(f"state_dim mismatch: model={self._state_dim}, file={int(data['state_dim'][0])}")
        if "latent_dim" in data and int(data["latent_dim"][0]) != self._latent_dim:
            raise ValueError(f"latent_dim mismatch: model={self._latent_dim}, file={int(data['latent_dim'][0])}")
        params = {k: v for k, v in data.items() if k not in ("state_dim", "latent_dim", "hidden_dim", "train_steps")}
        self.set_params(params)
        if "train_steps" in data:
            self._train_steps = int(data["train_steps"][0])
        logger.info("Loaded encoder params from %s", path)

    # =====================================================================
    # 辅助方法
    # =====================================================================

    def _zero_grads(self) -> dict[str, np.ndarray]:
        """返回与参数同 shape 的零梯度字典。"""
        return {
            "W1": np.zeros_like(self.W1),
            "b1": np.zeros_like(self.b1),
            "W2": np.zeros_like(self.W2),
            "b2": np.zeros_like(self.b2),
            "W3": np.zeros_like(self.W3),
            "b3": np.zeros_like(self.b3),
            "W4": np.zeros_like(self.W4),
            "b4": np.zeros_like(self.b4),
            "W5": np.zeros_like(self.W5),
            "b5": np.zeros_like(self.b5),
            "W6": np.zeros_like(self.W6),
            "b6": np.zeros_like(self.b6),
        }

    def __repr__(self) -> str:
        return (
            f"LearnableStateEncoder(state_dim={self._state_dim}, "
            f"latent_dim={self._latent_dim}, hidden_dim={self._hidden_dim}, "
            f"n_params={self.n_params}, train_steps={self._train_steps})"
        )
