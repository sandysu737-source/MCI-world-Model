from __future__ import annotations

from typing import Any

"""
MCI World Model v4.6.0 — Tissue Classifier
============================================

AI 智能清创机器人 — 组织分类器 (4分类 + 置信度)。

输入: 多模态融合特征 (来自 DebridementWorldModel)
输出: P(坏死), P(腐肉), P(肉芽), P(上皮) + 安全判断

架构: MLP (input_dim → 256 → 128 → 4) + softmax
训练: CrossEntropy + 类别加权 (坏死×2.0, 健康组织×0.5)
安全约束: 置信度 < 0.7 标记"不确定"

安全规则:
- 坏死 vs 健康混淆 → 触发安全停止
- 坏死组织误判为健康 → 最高风险, 必须阻止
- 健康组织误判为坏死 → 中等风险, 建议人工确认

用法:
    from mci_world_model.sdk._tissue_classifier import TissueClassifier, TissueResult

    clf = TissueClassifier(input_dim=256)
    clf.train(features, labels, n_epochs=50)
    result = clf.classify(fused_features)
    if result.is_safe_to_debride:
        proceed()
"""


import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# 组织标签常量
# =============================================================================

TISSUE_NECROTIC = 0  # 坏死组织 — 必须清除
TISSUE_SLOUGH = 1  # 腐肉 — 需要清创
TISSUE_GRANULATION = 2  # 肉芽组织 — 需保护
TISSUE_EPITHELIAL = 3  # 上皮组织 — 严禁损伤

TISSUE_NAMES = {0: "坏死", 1: "腐肉", 2: "肉芽", 3: "上皮"}

# 安全权重: 坏死误判代价 ×2, 健康组织误判代价 ×0.5 (鼓励保守)
CLASS_WEIGHTS = np.array([2.0, 1.5, 0.75, 0.5], dtype=np.float64)

# 安全力限制 (N)
MAX_FORCE_BY_TISSUE = {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.5}

# 安全速度限制 (mm/s)
MAX_VELOCITY_BY_TISSUE = {0: 10.0, 1: 5.0, 2: 3.0, 3: 1.0}


# =============================================================================
# TissueResult
# =============================================================================


@dataclass
class TissueResult:
    """组织分类结果。

    Attributes:
        probs:           [P(坏死), P(腐肉), P(肉芽), P(上皮)]
        predicted_label: 预测标签
        confidence:      置信度
        is_uncertain:    是否不确定 (需人工确认)
        max_force_n:     建议最大清创力 (N)
        max_velocity:    建议最大工具速度 (mm/s)
    """

    probs: np.ndarray  # (4,) float
    predicted_label: int
    confidence: float
    is_uncertain: bool
    max_force_n: float
    max_velocity: float

    @property
    def tissue_name(self) -> str:
        return TISSUE_NAMES.get(self.predicted_label, "未知")

    @property
    def is_safe_to_debride(self) -> bool:
        """是否安全清创: 非不确定 且 非上皮。"""
        return not self.is_uncertain and self.predicted_label != TISSUE_EPITHELIAL

    @property
    def requires_stop(self) -> bool:
        """是否需要安全停止: 对上皮/健康组织误判为坏死。"""
        return self.predicted_label == TISSUE_NECROTIC and self.probs[TISSUE_EPITHELIAL] > 0.2

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted": self.tissue_name,
            "label": int(self.predicted_label),
            "confidence": round(float(self.confidence), 4),
            "probs": {TISSUE_NAMES[i]: round(float(self.probs[i]), 4) for i in range(4)},
            "is_uncertain": self.is_uncertain,
            "is_safe_to_debride": self.is_safe_to_debride,
            "requires_stop": self.requires_stop,
            "max_force_n": round(float(self.max_force_n), 1),
            "max_velocity": round(float(self.max_velocity), 1),
        }


# =============================================================================
# TissueClassifier
# =============================================================================


class TissueClassifier:
    """清创组织分类器。

    架构: MLP (input_dim → 256 → 128 → 4) + softmax
    训练: Mini-batch SGD + CrossEntropy (类别加权)

    参数量: input_dim × 256 + 256 × 128 + 128 × 4 ≈ input_dim × 256
    对于 input_dim=256: ~67K 参数
    """

    # 安全阈值
    CONFIDENCE_THRESHOLD = 0.7
    NECROTIC_EPITHELIAL_CONFUSION_THRESHOLD = 0.2

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: tuple = (256, 128),  # type: ignore
        seed: int = 42,
    ):
        self._input_dim = input_dim
        self._hidden_dims = hidden_dims
        self._num_classes = 4

        rng = np.random.RandomState(seed)

        # 初始化参数
        dims = [input_dim, *hidden_dims, self._num_classes]
        self._W: list[np.ndarray] = []
        self._b: list[np.ndarray] = []
        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            self._W.append(rng.randn(fan_in, fan_out).astype(np.float64) * np.sqrt(2.0 / (fan_in + fan_out)))
            self._b.append(np.zeros(fan_out, dtype=np.float64))

        self._trained: bool = False
        self._train_loss_history: list[float] = []
        self._class_weights = CLASS_WEIGHTS.copy()

    @property
    def n_params(self) -> int:
        return sum(w.size for w in self._W) + sum(b.size for b in self._b)

    @property
    def is_trained(self) -> bool:
        return self._trained

    # ── 前向传播 ──

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """批量前向: (B, input_dim) → (B, 4)。"""
        h = np.asarray(x, dtype=np.float64)
        if h.ndim == 1:
            h = h.reshape(1, -1)

        for i, (W, b) in enumerate(zip(self._W[:-1], self._b[:-1])):
            h = h @ W + b
            h = np.maximum(h, 0)  # ReLU

        # 最后一层 (线性 + softmax)
        logits = h @ self._W[-1] + self._b[-1]
        logits = logits - np.max(logits, axis=1, keepdims=True)  # 稳定 softmax
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        return probs

    def classify(self, features: np.ndarray) -> TissueResult:
        """分类单帧多模态特征。

        Args:
            features: (input_dim,) 融合特征向量

        Returns:
            TissueResult 含预测、置信度和安全判断
        """
        probs = self._forward(features).ravel()
        pred = int(np.argmax(probs))
        confidence = float(probs[pred])

        is_uncertain = confidence < self.CONFIDENCE_THRESHOLD

        max_f = MAX_FORCE_BY_TISSUE.get(pred, 0.5)
        max_v = MAX_VELOCITY_BY_TISSUE.get(pred, 1.0)

        return TissueResult(
            probs=probs.astype(np.float64),
            predicted_label=pred,
            confidence=confidence,
            is_uncertain=is_uncertain,
            max_force_n=max_f,
            max_velocity=max_v,
        )

    # ── 训练 ──

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_epochs: int = 50,
        lr: float = 0.005,
        batch_size: int = 32,
    ) -> dict[str, Any]:
        """Mini-batch SGD 训练。

        Args:
            X: (N, input_dim) 训练特征
            y: (N,) 整数标签 0-3
            n_epochs: 训练轮数
            lr: 学习率
            batch_size: 批大小

        Returns:
            训练报告
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n = len(X)
        rng = np.random.RandomState(42)

        self._train_loss_history = []

        for epoch in range(n_epochs):
            indices = rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                batch_idx = indices[start : start + batch_size]
                x_batch = X[batch_idx]
                y_batch = y[batch_idx]
                bs = x_batch.shape[0]

                # 前向
                h0 = x_batch
                activations = [h0]
                pre_activations = []

                for i in range(len(self._W) - 1):
                    z = h0 @ self._W[i] + self._b[i]
                    pre_activations.append(z)
                    h0 = np.maximum(z, 0)
                    activations.append(h0)

                # 最后一层
                logits = h0 @ self._W[-1] + self._b[-1]
                logits = logits - np.max(logits, axis=1, keepdims=True)
                probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

                # 加权 CrossEntropy 损失
                weights = self._class_weights[y_batch]
                ce = -np.log(np.clip(probs[np.arange(bs), y_batch], 1e-15, 1.0))
                loss = float(np.mean(ce * weights))
                epoch_loss += loss
                n_batches += 1

                # 反向传播
                d_logits = probs.copy()
                d_logits[np.arange(bs), y_batch] -= 1
                d_logits *= weights.reshape(-1, 1) / bs

                d_W_last = activations[-1].T @ d_logits
                d_b_last = d_logits.sum(axis=0)

                dh = d_logits @ self._W[-1].T

                for i in range(len(self._W) - 2, -1, -1):
                    dh = dh * (pre_activations[i] > 0).astype(np.float64)
                    dW = activations[i].T @ dh
                    db = dh.sum(axis=0)

                    # 梯度裁剪
                    np.clip(dW, -5, 5, out=dW)
                    np.clip(db, -5, 5, out=db)

                    self._W[i] -= lr * dW
                    self._b[i] -= lr * db

                    if i > 0:
                        dh = dh @ self._W[i].T

                # 更新最后一层
                np.clip(d_W_last, -5, 5, out=d_W_last)
                np.clip(d_b_last, -5, 5, out=d_b_last)
                self._W[-1] -= lr * d_W_last
                self._b[-1] -= lr * d_b_last

            avg_loss = epoch_loss / max(n_batches, 1)
            self._train_loss_history.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.debug(
                    "TissueClassifier Epoch %d/%d | Loss: %.4f",
                    epoch + 1,
                    n_epochs,
                    avg_loss,
                )

        self._trained = True
        return {
            "n_epochs": n_epochs,
            "final_loss": round(self._train_loss_history[-1], 6),
            "n_params": self.n_params,
            "loss_history": [round(v, 6) for v in self._train_loss_history],
        }

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """评估分类器。

        Returns:
            {"accuracy": float, "per_class_accuracy": [...],
             "confusion_matrix": (4,4), "balanced_accuracy": float}
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        probs = self._forward(X)
        preds = np.argmax(probs, axis=1)

        accuracy = float(np.mean(preds == y))

        # 每类准确率
        per_class = []
        for c in range(4):
            mask = y == c
            if mask.sum() > 0:
                per_class.append(float(np.mean(preds[mask] == c)))
            else:
                per_class.append(0.0)

        # 混淆矩阵
        cm = np.zeros((4, 4), dtype=np.int64)
        for true_c in range(4):
            for pred_c in range(4):
                cm[true_c, pred_c] = int(np.sum((y == true_c) & (preds == pred_c)))

        bal_acc = float(np.mean(per_class)) if per_class else 0.0

        return {
            "accuracy": round(accuracy, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "per_class_accuracy": [round(v, 4) for v in per_class],
            "confusion_matrix": cm.tolist(),
        }

    # ── 持久化 ──

    def save(self, path: str) -> bool:
        """保存模型参数 (.npz)。"""
        try:
            import json
            import os

            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            save_dict = {}
            for i, (W, b) in enumerate(zip(self._W, self._b)):
                save_dict[f"W{i}"] = W
                save_dict[f"b{i}"] = b
            save_dict["_input_dim"] = np.array(self._input_dim, dtype=np.int32)
            save_dict["_hidden_dims"] = np.array(self._hidden_dims, dtype=np.int32)

            np.savez_compressed(path, **save_dict)  # type: ignore

            meta = {
                "version": "4.6.0",
                "model_type": "TissueClassifier",
                "input_dim": self._input_dim,
                "hidden_dims": list(self._hidden_dims),
                "n_params": self.n_params,
                "trained": self._trained,
            }
            with open(path + ".json", "w") as f:
                json.dump(meta, f, indent=2)

            return True
        except Exception as e:
            logger.error("TissueClassifier save failed: %s", e)
            return False

    @classmethod
    def load(cls, path: str) -> TissueClassifier | None:
        """加载模型参数。"""
        try:
            import json
            import os

            data = np.load(path + ".npz")
            input_dim = int(data["_input_dim"])
            hidden_dims = tuple(int(d) for d in data["_hidden_dims"])

            clf = cls(input_dim=input_dim, hidden_dims=hidden_dims)

            for i in range(len(clf._W)):
                clf._W[i] = data[f"W{i}"].astype(np.float64)
                clf._b[i] = data[f"b{i}"].astype(np.float64)

            meta_path = path + ".json"
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                clf._trained = meta.get("trained", True)

            return clf
        except Exception as e:
            logger.error("TissueClassifier load failed: %s", e)
            return None

    def __repr__(self) -> str:
        return (
            f"TissueClassifier(input={self._input_dim}, "
            f"hidden={self._hidden_dims}, "
            f"params={self.n_params}, "
            f"trained={self._trained})"
        )
