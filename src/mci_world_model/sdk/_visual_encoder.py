from __future__ import annotations

"""多模态物理感知→潜向量编码器 — TASK-C2。

使用轻量级 CNN 架构从图像/信号编码物理状态潜向量。
替换 from_signals() 中的手写信号映射。

架构:
    图像路径: Image → Conv2D → ReLU → Flatten → Linear → latent (z_dim,)
    信号路径: Signals → Linear → ReLU → Linear → latent (z_dim,)
    融合路径: latent_img + latent_sig → Linear → fused_latent (z_dim,)

核心方法签名:
    VisualEncoder.__init__(config=VisualEncoderConfig())
    VisualEncoder.encode_image(image) → np.ndarray (z_dim,)
    VisualEncoder.encode_signals(signals) → np.ndarray (z_dim,)
    VisualEncoder.encode_multimodal(image, signals) → np.ndarray (z_dim,)
    VisualEncoder.train(pairs) → dict

训练伪代码:
    for (image, signals, target_state) in data:
        z_img = conv_encoder(image)
        z_sig = signal_encoder(signals)
        z_fused = fusion_layer(concat(z_img, z_sig))
        loss = MSE(z_fused, target_latent)
        grads = compute_gradients(loss)
        apply_gradients(grads, lr)

注意: 不依赖 MLX/ViT (避免平台限制), 使用纯 numpy 实现。
"""


import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 核心数据结构
# =============================================================================


@dataclass
class VisualEncoderConfig:
    """视觉编码器配置。

    Attributes:
        image_height: 输入图像高度
        image_width: 输入图像宽度
        image_channels: 输入图像通道数
        signal_dim: 信号向量维度
        z_dim: 潜向量维度
        hidden_dim: 隐层维度
        lr: 学习率
        n_epochs: 训练轮数
        seed: 随机种子
    """

    image_height: int = 32
    image_width: int = 32
    image_channels: int = 1
    signal_dim: int = 4
    z_dim: int = 16
    hidden_dim: int = 64
    lr: float = 0.01
    n_epochs: int = 50
    seed: int = 42


@dataclass
class MultimodalPair:
    """多模态训练对。

    Attributes:
        image: 图像数据 (C, H, W) 或 (H, W)
        signals: 信号向量 (signal_dim,)
        target_latent: 目标潜向量 (z_dim,)
    """

    image: np.ndarray = field(default_factory=lambda: np.array([]))
    signals: np.ndarray = field(default_factory=lambda: np.array([]))
    target_latent: np.ndarray = field(default_factory=lambda: np.array([]))


# =============================================================================
# VisualEncoder — 多模态物理感知编码器
# =============================================================================


class VisualEncoder:
    """多模态物理感知编码器。

    支持:
        - 纯图像编码 (CNN-like)
        - 纯信号编码 (MLP)
        - 多模态融合编码

    用法:
        >>> encoder = VisualEncoder(VisualEncoderConfig())
        >>> z = encoder.encode_signals(np.array([0.5, -0.3, 1.0, 0.2]))
        >>> z_img = encoder.encode_image(np.random.randn(1, 32, 32))
        >>> z_fused = encoder.encode_multimodal(image, signals)
    """

    def __init__(self, config: VisualEncoderConfig | None = None) -> None:
        self._config = config or VisualEncoderConfig()
        self._rng = np.random.RandomState(self._config.seed)
        self._trained = False

        # ── 信号编码器参数 ──
        S = self._config.signal_dim
        H = self._config.hidden_dim
        Z = self._config.z_dim

        self._sig_W1 = self._xavier(S, H)
        self._sig_b1 = np.zeros(H, dtype=np.float64)
        self._sig_W2 = self._xavier(H, Z)
        self._sig_b2 = np.zeros(Z, dtype=np.float64)

        # ── 图像编码器参数 (简化 CNN: flatten → MLP) ──
        flat_dim = self._config.image_channels * self._config.image_height * self._config.image_width
        self._img_W1 = self._xavier(flat_dim, H)
        self._img_b1 = np.zeros(H, dtype=np.float64)
        self._img_W2 = self._xavier(H, Z)
        self._img_b2 = np.zeros(Z, dtype=np.float64)

        # ── 融合层参数 ──
        self._fuse_W = self._xavier(2 * Z, Z)
        self._fuse_b = np.zeros(Z, dtype=np.float64)

    def _xavier(self, fan_in: int, fan_out: int) -> np.ndarray:
        return self._rng.randn(fan_in, fan_out).astype(np.float64) * np.sqrt(2.0 / (fan_in + fan_out))

    # -----------------------------------------------------------------
    # 公开 API
    # -----------------------------------------------------------------

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """从图像编码潜向量。

        简化 CNN 流程: flatten → Linear → ReLU → Linear → z

        Args:
            image: 输入图像 (C, H, W) 或 (H, W)

        Returns:
            潜向量 (z_dim,)
        """
        x = np.asarray(image, dtype=np.float64).ravel()

        # 维度对齐: 如果图像尺寸不匹配, 截断或填充
        expected = self._config.image_channels * self._config.image_height * self._config.image_width
        if len(x) < expected:
            x = np.pad(x, (0, expected - len(x)))
        elif len(x) > expected:
            x = x[:expected]

        h = x @ self._img_W1 + self._img_b1
        h = np.maximum(h, 0)  # ReLU
        z = h @ self._img_W2 + self._img_b2
        return z

    def encode_signals(self, signals: np.ndarray) -> np.ndarray:
        """从信号向量编码潜向量。

        MLP 流程: signals → Linear → ReLU → Linear → z

        Args:
            signals: 信号向量 (signal_dim,)

        Returns:
            潜向量 (z_dim,)
        """
        x = np.asarray(signals, dtype=np.float64).ravel()

        # 维度对齐
        expected = self._config.signal_dim
        if len(x) < expected:
            x = np.pad(x, (0, expected - len(x)))
        elif len(x) > expected:
            x = x[:expected]

        h = x @ self._sig_W1 + self._sig_b1
        h = np.maximum(h, 0)  # ReLU
        z = h @ self._sig_W2 + self._sig_b2
        return z

    def encode_multimodal(self, image: np.ndarray, signals: np.ndarray) -> np.ndarray:
        """多模态融合编码: 图像 + 信号 → 潜向量。

        融合流程:
            z_img = encode_image(image)
            z_sig = encode_signals(signals)
            z_fused = Linear(concat(z_img, z_sig))

        Args:
            image: 输入图像
            signals: 信号向量

        Returns:
            融合潜向量 (z_dim,)
        """
        z_img = self.encode_image(image)
        z_sig = self.encode_signals(signals)
        combined = np.concatenate([z_img, z_sig])
        z_fused = combined @ self._fuse_W + self._fuse_b
        return z_fused

    def train(self, pairs: list[MultimodalPair]) -> dict[str, Any]:
        """训练编码器。

        使用有限差分梯度 + SGD。

        Args:
            pairs: 训练数据对列表

        Returns:
            {"final_loss": float, "n_epochs": int, "n_samples": int}
        """
        if not pairs:
            return {"final_loss": 0.0, "n_epochs": 0, "n_samples": 0}

        lr = self._config.lr
        eps = 1e-4
        losses: list[float] = []

        for epoch in range(self._config.n_epochs):
            epoch_loss = 0.0
            for pair in pairs:
                # 前向
                z_img = self.encode_image(pair.image)
                z_sig = self.encode_signals(pair.signals)
                combined = np.concatenate([z_img, z_sig])
                z_fused = combined @ self._fuse_W + self._fuse_b

                loss = float(np.mean((z_fused - pair.target_latent) ** 2))
                epoch_loss += loss

                # 简化梯度: 对融合层参数做有限差分
                for param_name in ["_fuse_W", "_fuse_b"]:
                    param = getattr(self, param_name)
                    grad = np.zeros_like(param)
                    n_sample = max(1, param.size // 20)  # 采样 5% 参数
                    indices = list(np.ndindex(param.shape))
                    sampled = self._rng.choice(len(indices), size=min(n_sample, len(indices)), replace=False)
                    for si in sampled:
                        idx = indices[si]
                        old = param[idx]
                        param[idx] = old + eps
                        z_plus = (
                            np.concatenate([self.encode_image(pair.image), self.encode_signals(pair.signals)])
                            @ self._fuse_W
                            + self._fuse_b
                        )
                        loss_plus = np.mean((z_plus - pair.target_latent) ** 2)
                        param[idx] = old - eps
                        z_minus = (
                            np.concatenate([self.encode_image(pair.image), self.encode_signals(pair.signals)])
                            @ self._fuse_W
                            + self._fuse_b
                        )
                        loss_minus = np.mean((z_minus - pair.target_latent) ** 2)
                        param[idx] = old
                        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
                    param -= lr * grad

            losses.append(epoch_loss / len(pairs))

        self._trained = True
        return {
            "final_loss": losses[-1] if losses else 0.0,
            "n_epochs": self._config.n_epochs,
            "n_samples": len(pairs),
        }

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def config(self) -> VisualEncoderConfig:
        return self._config

    @staticmethod
    def from_signals_adapter(signals: list[Any], z_dim: int = 16) -> np.ndarray:
        """兼容旧 from_signals() 接口的适配方法。

        将 PhysicalSignal 列表转为潜向量，替代手写信号映射。

        Args:
            signals: PhysicalSignal 列表
            z_dim: 潜向量维度

        Returns:
            潜向量 (z_dim,)
        """
        # 提取信号值
        values = []
        for sig in signals:
            v = getattr(sig, "value", None)
            if v is not None:
                if isinstance(v, (int, float)):
                    values.append(float(v))
                elif isinstance(v, (list, tuple)):
                    values.extend([float(x) for x in v])
                elif hasattr(v, "__array__"):
                    values.extend(np.asarray(v).ravel().tolist())

        if not values:
            return np.zeros(z_dim, dtype=np.float64)

        # 简单投影到 z_dim
        vec = np.array(values, dtype=np.float64)
        if len(vec) >= z_dim:
            return vec[:z_dim]
        # 循环填充
        result = np.zeros(z_dim, dtype=np.float64)
        n = min(len(vec), z_dim)
        result[:n] = vec[:n]
        return result
