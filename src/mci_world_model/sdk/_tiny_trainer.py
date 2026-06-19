from __future__ import annotations

"""
MCI World Model v4.4.0 — TinyTrainer
======================================

TinyTransformer 训练器，支持从 JSONL QA 数据训练。

用法:
    trainer = TinyTrainer(TinyTransformerConfig.nano())
    trainer.load_qa_data("data/debridement_qa_baseline.jsonl")
    history = trainer.train()
    trainer.model.save("checkpoints/tiny_v1")
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    from mci_world_model.sdk._tiny_transformer import TinyTransformer, TinyTransformerConfig
except ImportError:
    TinyTransformer = None  # type: ignore[assignment]
    TinyTransformerConfig = None  # type: ignore[assignment]


@dataclass
class TinyTrainConfig:
    """TinyTransformer 训练超参数。"""

    n_epochs: int = 30
    lr: float = 3e-4
    batch_size: int = 16
    val_split: float = 0.1
    early_stop_patience: int = 5
    early_stop_min_delta: float = 1e-4
    checkpoint_dir: str = "checkpoints"
    seed: int = 42
    augment_enabled: bool = True


@dataclass
class TinyTrainMetrics:
    """TinyTransformer 训练指标。"""
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    lr: float = 0.0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": round(self.train_loss, 6),
            "val_loss": round(self.val_loss, 6),
            "val_accuracy": round(self.val_accuracy, 4),
            "lr": self.lr,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }


class TinyTrainer:
    """TinyTransformer 训练器。

    训练范式: 对比学习 (问题-答案 cosine similarity 最大化)。
    """

    def __init__(
        self,
        model_config: TinyTransformerConfig | None = None,
        train_config: TinyTrainConfig | None = None,
    ) -> None:
        if TinyTransformer is None or TinyTransformerConfig is None:
            raise ImportError("TinyTransformer not available")

        self.model_config = model_config or TinyTransformerConfig.micro()
        self.train_cfg = train_config or TinyTrainConfig()
        self.model = TinyTransformer(self.model_config)
        self._history: list[TinyTrainMetrics] = []
        self._qa_pairs: list[tuple[str, str]] = []
        self._rng = np.random.RandomState(self.train_cfg.seed)

    # ── 数据加载 ──

    def load_qa_data(self, path: str) -> int:
        """从 JSONL 文件加载 QA 对。

        格式: {"question": "...", "answer": "..."}

        Returns:
            加载的 QA 对数量
        """
        self._qa_pairs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    q = item.get("question", "")
                    a = item.get("answer", "")
                    if q and a:
                        self._qa_pairs.append((q, a))
                except json.JSONDecodeError:
                    continue

        logger.info("Loaded %d QA pairs from %s", len(self._qa_pairs), path)
        return len(self._qa_pairs)

    def add_qa(self, question: str, answer: str) -> None:
        """手动添加 QA 对。"""
        self._qa_pairs.append((question, answer))

    # ── 训练 ──

    def train(self) -> list[TinyTrainMetrics]:
        """训练 TinyTransformer。

        Returns:
            训练历史指标列表
        """
        if not self._qa_pairs:
            logger.warning("No QA data loaded, skipping training")
            return []

        # Train/val split
        n_val = max(1, int(len(self._qa_pairs) * self.train_cfg.val_split))
        n_train = len(self._qa_pairs) - n_val
        indices = self._rng.permutation(len(self._qa_pairs))
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        best_val = float("inf")
        patience_counter = 0

        for epoch in range(self.train_cfg.n_epochs):
            epoch_start = time.time()

            # Training
            train_loss = self._train_epoch(train_indices)

            # Validation
            val_loss = self._validate(val_indices)

            # QA accuracy on validation
            val_acc = self._qa_accuracy(val_indices)

            self._history.append(TinyTrainMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy=val_acc,
                lr=self.train_cfg.lr,
                elapsed_seconds=time.time() - epoch_start,
            ))

            logger.info(
                "Epoch %d/%d: train_loss=%.4f val_loss=%.4f val_acc=%.2f%%",
                epoch + 1, self.train_cfg.n_epochs, train_loss, val_loss, val_acc * 100,
            )

            if val_loss < best_val - self.train_cfg.early_stop_min_delta:
                best_val = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.train_cfg.early_stop_patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        self.model._trained = True
        return self._history

    def _train_epoch(self, train_indices: np.ndarray) -> float:
        cfg = self.train_cfg
        total_loss = 0.0
        n_batches = 0

        perm = self._rng.permutation(len(train_indices))
        all_params = self.model._collect_params()

        for start in range(0, len(perm), cfg.batch_size):
            batch_idx = perm[start:start + cfg.batch_size]
            batch_loss = 0.0

            for idx in batch_idx:
                q, a = self._qa_pairs[train_indices[idx]]
                q_ids = self.model._tokenize(q)
                a_ids = self.model._tokenize(a)

                q_emb = self.model._forward_single(q_ids)
                a_emb = self.model._forward_single(a_ids)

                cos_sim = np.dot(q_emb, a_emb) / (
                    max(np.linalg.norm(q_emb), 1e-10) * max(np.linalg.norm(a_emb), 1e-10)
                )
                loss = -cos_sim + 1.0
                batch_loss += float(loss)

            bs_actual = len(batch_idx)
            total_loss += batch_loss / bs_actual
            n_batches += 1

            # SGD update
            for p in all_params:
                grad = self.model._finite_diff_gradient(p, 1e-6, self._rng)
                p -= cfg.lr * grad

        return total_loss / max(n_batches, 1)

    def _validate(self, val_indices: np.ndarray) -> float:
        total_loss = 0.0
        n = min(50, len(val_indices))
        for i in range(n):
            q, a = self._qa_pairs[val_indices[i]]
            q_ids = self.model._tokenize(q)
            a_ids = self.model._tokenize(a)
            q_emb = self.model._forward_single(q_ids)
            a_emb = self.model._forward_single(a_ids)
            cos_sim = np.dot(q_emb, a_emb) / (
                max(np.linalg.norm(q_emb), 1e-10) * max(np.linalg.norm(a_emb), 1e-10)
            )
            total_loss += float(-cos_sim + 1.0)
        return total_loss / n

    def _qa_accuracy(self, val_indices: np.ndarray) -> float:
        """计算 QA 准确率 (问题-答案检索)。"""
        n = min(50, len(val_indices))
        correct = 0
        # 预计算所有答案嵌入
        a_embs = {}
        for i in val_indices[:n]:
            _, a = self._qa_pairs[i]
            a_ids = self.model._tokenize(a)
            a_embs[i] = self.model._forward_single(a_ids)

        for i in range(n):
            q, _ = self._qa_pairs[val_indices[i]]
            q_ids = self.model._tokenize(q)
            q_emb = self.model._forward_single(q_ids)

            best_sim = -2.0
            best_j = -1
            for j in val_indices[:n]:
                if j not in a_embs:
                    continue
                sim = np.dot(q_emb, a_embs[j]) / (
                    max(np.linalg.norm(q_emb), 1e-10) * max(np.linalg.norm(a_embs[j]), 1e-10)
                )
                if sim > best_sim:
                    best_sim = sim
                    best_j = j

            if best_j == val_indices[i]:
                correct += 1

        return correct / n

    # ── 检查点 ──

    def save_checkpoint(self, path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self.model.save(path)
            state = {
                "version": "4.4.0",
                "model_config": {
                    "d_model": self.model_config.d_model,
                    "n_layers": self.model_config.n_layers,
                },
                "history": [m.to_dict() for m in self._history],
                "n_qa_pairs": len(self._qa_pairs),
            }
            with open(f"{path}_state.json", "w") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error("Checkpoint save failed: %s", e)
            return False

    def __repr__(self) -> str:
        return (
            f"TinyTrainer(d={self.model_config.d_model}, "
            f"L={self.model_config.n_layers}, "
            f"qa_pairs={len(self._qa_pairs)})"
        )
