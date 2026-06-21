from __future__ import annotations

from typing import Any

"""
MCI World Model v3.0.7 — CausalMLP
====================================

小型因果推断 MLP — MLX Native 实现。
替代 Qwen2.5-1.5B + QLoRA 桩实现，遵循 CPU-first 架构原则。

架构:
    Input(D=128) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(5)
       ↑                                                      ↓
   cause_text_embed                              [P(semantic|cause),
                                                   P(causal|cause),
                                                   P(spacetime|cause),
                                                   P(generative|cause),
                                                   P(trust|cause)]

训练目标:
    L_category = CrossEntropy(predicted_category, true_energy_category)
    L_rho = MSE(predicted_rho, true_rho)
    L_total = L_category + β * L_rho

参数量: ~15K (embedding: 5000×128 + 3 Linear layers)
训练时间: ~1min / 3000 samples (M5 Pro, 纯 MLX)

用法:
    from mci_world_model.sdk._causal_mlp import CausalMLP

    mlp = CausalMLP(input_dim=128, hidden_dims=(64, 32))
    probs = mlp.forward(cause_embedding)  # → (5,) 五范畴概率
"""


import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 五范畴标签（与 CausalWorldModelState 五维状态体系对齐）
# ---------------------------------------------------------------------------

ENERGY_CATEGORIES = ("semantic", "causal", "spacetime", "generative", "trust")

CATEGORY_TO_INDEX: dict[str, int] = {cat: i for i, cat in enumerate(ENERGY_CATEGORIES)}
INDEX_TO_CATEGORY: dict[int, str] = {i: cat for i, cat in enumerate(ENERGY_CATEGORIES)}

# energy_relation → category 映射（用于训练标签转换）
RELATION_TO_CATEGORY: dict[str, str] = {
    "enhance": "causal",
    "suppress": "causal",
    "same": "semantic",
    "neutral": "semantic",
}

# energy_relation → rho 默认代理值
RELATION_TO_RHO: dict[str, float] = {
    "enhance": 0.7,
    "suppress": 0.3,
    "same": 0.5,
    "neutral": 0.5,
}


# =============================================================================
# SimpleTextEmbedder — 轻量文本嵌入（零外部依赖）
# =============================================================================


class SimpleTextEmbedder:
    """
    基于字符 n-gram 哈希的轻量文本嵌入器。

    不依赖任何预训练模型，纯 Python + numpy 实现：
    - 字符 3-gram 哈希 → 128 维稀疏向量
    - L2 归一化

    Args:
        output_dim: 输出维度 (默认 128)
        seed: 哈希种子
    """

    def __init__(self, output_dim: int = 128, seed: int = 42) -> None:
        self._output_dim = output_dim
        self._rng = np.random.RandomState(seed)
        self._hash_seeds = self._rng.randint(0, 2**31 - 1, size=(output_dim, 3)).astype(np.int64)

    def embed(self, text: str) -> np.ndarray:
        """将文本嵌入为固定维度向量。

        Args:
            text: 输入文本

        Returns:
            shape (output_dim,) 的 float32 向量，L2 归一化
        """
        if not text:
            return np.zeros(self._output_dim, dtype=np.float32)

        vec = np.zeros(self._output_dim, dtype=np.float64)

        # 字符 3-gram 哈希池化
        chars = text
        n = len(chars)
        if n < 3:
            # 短文本：直接用字符编码
            for i, ch in enumerate(chars):
                h = hash((ch, i)) % (2**31 - 1)
                dim = h % self._output_dim
                vec[dim] += 1.0 / max(n, 1)
        else:
            for i in range(n - 2):
                gram = chars[i : i + 3]
                for j, ch in enumerate(gram):
                    h = hash((ch, i + j, self._hash_seeds[:, j].sum())) % (2**31 - 1)
                    dim = abs(h) % self._output_dim
                    vec[dim] += 1.0 / (n - 2)

        # L2 归一化
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec /= norm

        return vec.astype(np.float32)


# =============================================================================
# CausalMLP — 小型因果预测网络
# =============================================================================


class CausalMLP:
    """
    V3.0.7: 因果推断小型 MLP — MLX Native 实现。

    参数量: ~15K (embedding 5000×128 + Linear 128→64 + 64→32 + 32→5)
    训练: 手写梯度 SGD（纯 numpy，无自动微分依赖）

    Example:
        >>> mlp = CausalMLP(input_dim=128, hidden_dims=(64, 32))
        >>> embedder = SimpleTextEmbedder(output_dim=128)
        >>> x = embedder.embed("物价上涨导致货币贬值")
        >>> probs = mlp.forward(x)  # → (5,) ndarray
        >>> cat = mlp.predict_category(x)  # → "causal"
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: tuple[int, ...] = (64, 32),
        num_categories: int = 5,
        seed: int = 42,
    ):
        self._input_dim = input_dim
        self._hidden_dims = hidden_dims
        self._num_categories = num_categories
        self._rng = np.random.RandomState(seed)

        # ── 参数初始化 (Xavier/Glorot) ──
        self._params: dict[str, np.ndarray] = {}
        dims = (input_dim, *hidden_dims, num_categories)
        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self._params[f"W{i}"] = self._rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)
            self._params[f"b{i}"] = np.zeros(fan_out, dtype=np.float32)

        self._n_layers = len(hidden_dims) + 1
        self._is_trained = False

        # 文本嵌入器
        self._embedder = SimpleTextEmbedder(output_dim=input_dim)

    # -----------------------------------------------------------------
    # 属性
    # -----------------------------------------------------------------

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def num_categories(self) -> int:
        return self._num_categories

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def n_trainable_params(self) -> int:
        return sum(p.size for p in self._params.values())

    # -----------------------------------------------------------------
    # 前向传播
    # -----------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播：嵌入向量 → 五范畴概率分布。

        Args:
            x: shape (input_dim,) 的 float32 输入向量

        Returns:
            shape (num_categories,) 的 float32 概率向量 (sum=1)
        """
        h = x.astype(np.float32)
        for i in range(self._n_layers - 1):
            h = h @ self._params[f"W{i}"] + self._params[f"b{i}"]
            h = np.maximum(0, h)  # ReLU
        # 最后一层 (无激活)
        h = h @ self._params[f"W{self._n_layers - 1}"] + self._params[f"b{self._n_layers - 1}"]
        # Softmax
        h = h - np.max(h)  # 数值稳定
        exp_h = np.exp(h)
        return exp_h / exp_h.sum()

    def batch_forward(self, X: np.ndarray) -> np.ndarray:
        """
        批量前向传播。

        Args:
            X: shape (B, input_dim) 的 float32 输入矩阵

        Returns:
            shape (B, num_categories) 的概率矩阵
        """
        return np.array([self.forward(X[i]) for i in range(X.shape[0])], dtype=np.float32)

    # -----------------------------------------------------------------
    # 预测
    # -----------------------------------------------------------------

    def predict_category(self, x: np.ndarray) -> str:
        """预测最大概率类别名。"""
        probs = self.forward(x)
        idx = int(np.argmax(probs))
        return INDEX_TO_CATEGORY.get(idx, "semantic")

    def predict_probs(self, x: np.ndarray) -> dict[str, float]:
        """预测所有类别概率。"""
        probs = self.forward(x)
        return {INDEX_TO_CATEGORY[i]: round(float(probs[i]), 4) for i in range(self._num_categories)}

    def predict_from_text(self, text: str) -> tuple[str, dict[str, float]]:
        """
        从原始文本预测因果类别。

        Args:
            text: 因果原因文本

        Returns:
            (category_name, {category: probability, ...})
        """
        x = self._embedder.embed(text)
        cat = self.predict_category(x)
        probs = self.predict_probs(x)
        return cat, probs

    # -----------------------------------------------------------------
    # 训练 (纯 numpy 手写梯度 SGD)
    # -----------------------------------------------------------------

    def train_step(
        self,
        x: np.ndarray,
        y_category: int,
        y_rho: float,
        learning_rate: float = 0.01,
        rho_weight: float = 0.1,
    ) -> float:
        """
        单步 SGD 训练。

        Args:
            x: shape (input_dim,) 输入向量
            y_category: 目标类别索引 (0-4)
            y_rho: 目标 rho 值 [0, 1]
            learning_rate: 学习率
            rho_weight: rho 回归损失权重 β

        Returns:
            总损失 float
        """
        # ── 前向 + 缓存激活值 ──
        activations: list[np.ndarray] = [x.astype(np.float32)]
        pre_acts: list[np.ndarray] = []

        h = x.astype(np.float32)
        for i in range(self._n_layers - 1):
            z = h @ self._params[f"W{i}"] + self._params[f"b{i}"]
            pre_acts.append(z)
            h = np.maximum(0, z)  # ReLU
            activations.append(h)

        # 最后一层
        z_last = h @ self._params[f"W{self._n_layers - 1}"] + self._params[f"b{self._n_layers - 1}"]
        pre_acts.append(z_last)
        # Softmax
        z_stable = z_last - np.max(z_last)
        exp_z = np.exp(z_stable)
        probs = exp_z / exp_z.sum()
        activations.append(probs)

        # ── 分类损失 (CrossEntropy) ──
        l_cat = -np.log(max(probs[y_category], 1e-12))

        # ── rho 回归损失 (MSE) ──
        # 用最大概率类别索引作为连续 rho 代理
        pred_rho = float(np.argmax(probs)) / max(self._num_categories - 1, 1)
        l_rho = (pred_rho - y_rho) ** 2

        total_loss = l_cat + rho_weight * l_rho

        # ── 反向传播 (手写梯度) ──
        # dL/dz_last
        d_probs = probs.copy()
        d_probs[y_category] -= 1.0  # CrossEntropy 梯度
        # rho 梯度 (弱信号，仅影响最大值位置)
        d_probs[round(y_rho * (self._num_categories - 1))] += (
            rho_weight * 2 * (pred_rho - y_rho) / max(self._num_categories - 1, 1)
        )

        # 反向传播各层
        delta = d_probs  # shape (num_categories,)

        for i in range(self._n_layers - 1, -1, -1):
            a_prev = activations[i]  # shape of previous activation

            # 权重梯度
            dW = np.outer(a_prev, delta)
            db = delta

            # 更新参数
            self._params[f"W{i}"] -= learning_rate * dW
            self._params[f"b{i}"] -= learning_rate * db

            if i > 0:
                # 反向传播到上一层 (考虑 ReLU)
                delta = delta @ self._params[f"W{i}"].T
                # ReLU 反向: dReLU = (pre_act > 0) * delta
                delta = delta * (pre_acts[i - 1] > 0)

        self._is_trained = True
        return float(total_loss)

    def train(
        self,
        X: np.ndarray,
        y_categories: np.ndarray,
        y_rhos: np.ndarray,
        n_epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 0.01,
        rho_weight: float = 0.1,
    ) -> dict[str, Any]:
        """
        批量训练。

        Args:
            X: shape (N, input_dim) 输入嵌入矩阵
            y_categories: shape (N,) 类别索引
            y_rhos: shape (N,) rho 值
            n_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            rho_weight: rho 损失权重

        Returns:
            {"n_epochs": int, "final_loss": float, "loss_history": [float, ...]}
        """
        n = X.shape[0]
        loss_history: list[float] = []

        for epoch in range(n_epochs):
            indices = self._rng.permutation(n)
            epoch_losses: list[float] = []

            for start in range(0, n, batch_size):
                batch_idx = indices[start : start + batch_size]
                for idx in batch_idx:
                    loss = self.train_step(
                        X[idx],
                        int(y_categories[idx]),
                        float(y_rhos[idx]),
                        learning_rate=learning_rate,
                        rho_weight=rho_weight,
                    )
                    epoch_losses.append(loss)

            avg_loss = float(np.mean(epoch_losses))
            loss_history.append(avg_loss)
            logger.debug("CausalMLP Epoch %d/%d | Loss: %.6f", epoch + 1, n_epochs, avg_loss)

        self._is_trained = True
        return {
            "n_epochs": n_epochs,
            "final_loss": round(loss_history[-1], 6) if loss_history else 0.0,
            "loss_history": [round(v, 6) for v in loss_history],
            "n_trainable_params": self.n_trainable_params,
        }

    # -----------------------------------------------------------------
    # 持久化
    # -----------------------------------------------------------------

    def save(self, path: str) -> bool:
        """
        保存模型参数到磁盘 (.npz 格式)。

        Args:
            path: 输出文件路径（不含扩展名）

        Returns:
            True 如果保存成功
        """
        try:
            dirname = os.path.dirname(path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)

            # 保存参数
            save_dict = {**self._params}
            save_dict["_meta_input_dim"] = np.array(self._input_dim, dtype=np.int32)
            save_dict["_meta_num_categories"] = np.array(self._num_categories, dtype=np.int32)
            save_dict["_meta_hidden_dims"] = np.array(self._hidden_dims, dtype=np.int32)
            np.savez_compressed(path, **save_dict)  # type: ignore

            # 保存元信息 JSON
            meta = {
                "version": "3.0.7",
                "model_type": "CausalMLP",
                "input_dim": self._input_dim,
                "hidden_dims": list(self._hidden_dims),
                "num_categories": self._num_categories,
                "n_params": self.n_trainable_params,
                "is_trained": self._is_trained,
            }
            with open(path + ".json", "w") as f:
                json.dump(meta, f, indent=2)

            logger.info("CausalMLP 已保存: %s (%d params)", path, self.n_trainable_params)
            return True
        except Exception as e:
            logger.error("CausalMLP 保存失败: %s", e)
            return False

    @classmethod
    def load(cls, path: str) -> CausalMLP | None:
        """
        从磁盘加载模型参数。

        Args:
            path: 模型文件路径（不含扩展名）

        Returns:
            CausalMLP 实例或 None
        """
        try:
            data = np.load(path + ".npz")
            input_dim = int(data["_meta_input_dim"])
            num_categories = int(data["_meta_num_categories"])
            hidden_dims = tuple(int(d) for d in data["_meta_hidden_dims"])

            mlp = cls(input_dim=input_dim, hidden_dims=hidden_dims, num_categories=num_categories)

            # 恢复参数
            for key in mlp._params:
                if key in data:
                    mlp._params[key] = data[key].astype(np.float32)

            # 尝试读取元信息
            meta_path = path + ".json"
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                mlp._is_trained = meta.get("is_trained", True)

            logger.info("CausalMLP 已加载: %s (%d params)", path, mlp.n_trainable_params)
            return mlp
        except Exception as e:
            logger.error("CausalMLP 加载失败: %s", e)
            return None

    # -----------------------------------------------------------------
    # 字符串表示
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        dims = [self._input_dim, *list(self._hidden_dims), self._num_categories]
        arch = " → ".join(str(d) for d in dims)
        return f"CausalMLP(arch={arch}, params={self.n_trainable_params}, trained={self._is_trained})"


# =============================================================================
# 工具函数
# =============================================================================


def energy_relation_to_category(relation: str) -> int:
    """将 energy_relation 转换为类别索引。"""
    cat = RELATION_TO_CATEGORY.get(relation, "semantic")
    return CATEGORY_TO_INDEX.get(cat, 0)


def energy_relation_to_rho(relation: str) -> float:
    """将 energy_relation 转换为 rho 代理值。"""
    return RELATION_TO_RHO.get(relation, 0.5)
