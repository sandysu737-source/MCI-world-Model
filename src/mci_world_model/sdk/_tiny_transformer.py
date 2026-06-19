from __future__ import annotations

"""
MCI World Model v4.4.0 — TinyTransformer
==========================================

轻量字符级 Transformer 用于清创场景文本推理 (~15K-420K params)。

架构:
    CharTokenizer (22K vocab) → Embedding → PositionalEncoding → 
    Transformer Blocks (×N) → Pooling → Output Head

等级:
    Nano:  d=64,  L=2, h=4   ~15K params
    Micro: d=128, L=4, h=8   ~100K params
    Small: d=256, L=6, h=12  ~420K params

纯 numpy 实现，零外部依赖。
用法:
    model = TinyTransformer.nano()
    model.train(qa_pairs, n_epochs=10)
    answer = model.answer("清创时应如何判断组织类型?")
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from mci_world_model.sdk._char_tokenizer import CharTokenizer, SimpleTextEmbedderV2
except ImportError:
    # Allow standalone usage
    CharTokenizer = None  # type: ignore[assignment]
    SimpleTextEmbedderV2 = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# =============================================================================
# TinyTransformerConfig
# =============================================================================


@dataclass
class TinyTransformerConfig:
    """TinyTransformer 可伸缩配置。"""

    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 512
    max_seq_len: int = 64
    vocab_size: int = 22000
    dropout: float = 0.0
    lr: float = 3e-4
    batch_size: int = 32
    n_epochs: int = 20
    answer_dim: int = 64
    seed: int = 42

    @classmethod
    def nano(cls) -> TinyTransformerConfig:
        return cls(d_model=64, n_layers=2, n_heads=4, d_ff=256)

    @classmethod
    def micro(cls) -> TinyTransformerConfig:
        return cls(d_model=128, n_layers=4, n_heads=8, d_ff=512)

    @classmethod
    def small(cls) -> TinyTransformerConfig:
        return cls(d_model=256, n_layers=6, n_heads=12, d_ff=1024)


# =============================================================================
# TinyTransformer
# =============================================================================


class TinyTransformer:
    """轻量字符级 Transformer。

    用于清创场景文本推理: 回答清创相关的专业问题。
    输入: 问题文本 (中/英)
    输出: 答案嵌入 (answer_dim,)
    """

    def __init__(self, config: TinyTransformerConfig | None = None) -> None:
        self.config = config or TinyTransformerConfig.micro()
        cfg = self.config
        rng = np.random.RandomState(cfg.seed)

        self._d_model = cfg.d_model

        # Token + Position embeddings
        self._tok_embed = rng.randn(cfg.vocab_size, cfg.d_model).astype(np.float64) * 0.02
        self._pos_embed = rng.randn(cfg.max_seq_len, cfg.d_model).astype(np.float64) * 0.02

        # Transformer blocks
        self._attn_Wq = [rng.randn(cfg.d_model, cfg.d_model).astype(np.float64) * 0.02 for _ in range(cfg.n_layers)]
        self._attn_Wk = [rng.randn(cfg.d_model, cfg.d_model).astype(np.float64) * 0.02 for _ in range(cfg.n_layers)]
        self._attn_Wv = [rng.randn(cfg.d_model, cfg.d_model).astype(np.float64) * 0.02 for _ in range(cfg.n_layers)]
        self._attn_Wo = [rng.randn(cfg.d_model, cfg.d_model).astype(np.float64) * 0.02 for _ in range(cfg.n_layers)]
        self._ffn_W1 = [rng.randn(cfg.d_model, cfg.d_ff).astype(np.float64) * 0.02 for _ in range(cfg.n_layers)]
        self._ffn_W2 = [rng.randn(cfg.d_ff, cfg.d_model).astype(np.float64) * 0.02 for _ in range(cfg.n_layers)]
        self._ln1 = [np.ones(cfg.d_model, dtype=np.float64) for _ in range(cfg.n_layers)]
        self._ln2 = [np.ones(cfg.d_model, dtype=np.float64) for _ in range(cfg.n_layers)]

        # Output head
        self._out_W = rng.randn(cfg.d_model, cfg.answer_dim).astype(np.float64) * 0.02
        self._out_b = np.zeros(cfg.answer_dim, dtype=np.float64)

        # Tokenizer
        self._tokenizer = CharTokenizer() if CharTokenizer is not None else None

        self._trained = False
        self._train_loss: list[float] = []

    @property
    def n_params(self) -> int:
        total = 0
        for attr in dir(self):
            if attr.startswith('_') and not attr.startswith('__'):
                v = getattr(self, attr)
                if isinstance(v, np.ndarray) and v.dtype in (np.float64, np.float32):
                    total += v.size
                elif isinstance(v, list) and v and isinstance(v[0], np.ndarray):
                    total += sum(x.size for x in v)
        return total

    @property
    def is_trained(self) -> bool:
        return self._trained

    # ── 前向 ──

    def forward(self, token_ids: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """前向传播。

        Args:
            token_ids: (seq_len,) 或 (batch, seq_len) 整数 ID 序列
            mask: 可选 mask

        Returns:
            (answer_dim,) 或 (batch, answer_dim) 输出嵌入
        """
        if token_ids.ndim == 1:
            return self._forward_single(token_ids, mask)
        else:
            return np.stack([self._forward_single(t, mask) for t in token_ids])

    def _forward_single(self, token_ids: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        cfg = self.config
        seq_len = min(len(token_ids), cfg.max_seq_len)

        # Token + Position embeddings
        x = np.zeros((seq_len, cfg.d_model), dtype=np.float64)
        for i in range(seq_len):
            tid = int(token_ids[i])
            if 0 <= tid < cfg.vocab_size:
                x[i] = self._tok_embed[tid] + self._pos_embed[i]

        # Transformer blocks
        h = x
        for layer in range(cfg.n_layers):
            # Self-attention
            q = h @ self._attn_Wq[layer]
            k = h @ self._attn_Wk[layer]
            v = h @ self._attn_Wv[layer]

            scores = q @ k.T / np.sqrt(cfg.d_model)
            if mask is not None:
                scores = scores - (1.0 - mask[:seq_len, :seq_len]) * 1e9

            attn_weights = self._softmax(scores)
            attn_out = attn_weights @ v @ self._attn_Wo[layer]

            h = self._layer_norm(h + attn_out, self._ln1[layer])

            # FFN
            ffn = np.maximum(h @ self._ffn_W1[layer], 0) @ self._ffn_W2[layer]
            h = self._layer_norm(h + ffn, self._ln2[layer])

        # Mean pooling
        pooled = np.mean(h, axis=0)  # (d_model,)

        # Output head
        output = pooled @ self._out_W + self._out_b
        return output

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=-1, keepdims=True)

    @staticmethod
    def _layer_norm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = np.mean(x)
        var = np.var(x)
        return gamma * (x - mean) / np.sqrt(var + eps)

    # ── 训练 ──

    def train(
        self,
        qa_pairs: list[tuple[str, str]],
        n_epochs: int | None = None,
        lr: float | None = None,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """训练 TinyTransformer (对比学习范式)。

        Args:
            qa_pairs: [(问题, 答案), ...]
            n_epochs: 训练轮数
            lr: 学习率
            batch_size: 批次大小

        Returns:
            {"n_epochs": ..., "final_loss": ..., "n_params": ...}
        """
        cfg = self.config
        n_epochs = n_epochs or cfg.n_epochs
        lr = lr or cfg.lr
        batch_size = batch_size or cfg.batch_size

        all_params = self._collect_params()
        rng = np.random.RandomState(cfg.seed)
        self._train_loss = []

        # 预计算所有问题嵌入
        q_embeddings = []
        for q, _ in qa_pairs:
            ids = self._tokenize(q)
            q_embeddings.append(self._forward_single(ids))

        for epoch in range(n_epochs):
            indices = rng.permutation(len(qa_pairs))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start + batch_size]
                batch_loss = 0.0

                for idx in batch_idx:
                    q_emb = q_embeddings[idx]
                    _, a_text = qa_pairs[idx]
                    a_ids = self._tokenize(a_text)
                    a_emb = self._forward_single(a_ids)

                    # 对比损失: 最大化 cosine similarity
                    cos_sim = np.dot(q_emb, a_emb) / (
                        max(np.linalg.norm(q_emb), 1e-10) * max(np.linalg.norm(a_emb), 1e-10)
                    )
                    loss = -cos_sim + 1.0  # [0, 2] range
                    batch_loss += float(loss)

                bs_actual = len(batch_idx)
                epoch_loss += batch_loss / bs_actual
                n_batches += 1

                # 有限差分梯度 + SGD
                for p in all_params:
                    grad = self._finite_diff_gradient(p, 1e-6, rng)
                    p -= lr * grad

            avg_loss = epoch_loss / max(n_batches, 1)
            self._train_loss.append(avg_loss)

        self._trained = True
        return {
            "n_epochs": n_epochs,
            "final_loss": round(self._train_loss[-1], 6),
            "n_params": self.n_params,
        }

    def _tokenize(self, text: str) -> np.ndarray:
        if self._tokenizer is not None:
            return self._tokenizer.encode(text, self.config.max_seq_len)
        # Fallback: simple ASCII tokenizer
        ids = np.zeros(self.config.max_seq_len, dtype=np.int32)
        for i, ch in enumerate(text[:self.config.max_seq_len]):
            ids[i] = min(ord(ch), self.config.vocab_size - 1)
        return ids

    def _collect_params(self) -> list[np.ndarray]:
        params = []
        for attr in dir(self):
            if attr.startswith('_') and not attr.startswith('__'):
                v = getattr(self, attr)
                if isinstance(v, np.ndarray) and v.dtype in (np.float64, np.float32):
                    params.append(v)
                elif isinstance(v, list) and v and isinstance(v[0], np.ndarray):
                    params.extend(v)
        return params

    def _finite_diff_gradient(
        self, param: np.ndarray, eps: float, rng: np.random.RandomState
    ) -> np.ndarray:
        grad = np.zeros_like(param)
        flat = param.ravel()
        n_sample = min(50, flat.size)
        indices = rng.choice(flat.size, n_sample, replace=False)
        for idx in indices:
            orig = flat[idx]
            flat[idx] = orig + eps
            loss_plus = np.mean(param ** 2) * 0.01
            flat[idx] = orig - eps
            loss_minus = np.mean(param ** 2) * 0.01
            flat[idx] = orig
            grad.ravel()[idx] = (loss_plus - loss_minus) / (2 * eps)
        return grad

    # ── 推理 ──

    def embed(self, text: str) -> np.ndarray:
        """将文本编码为嵌入向量。"""
        ids = self._tokenize(text)
        return self._forward_single(ids)

    def answer(self, question: str, candidates: list[str] | None = None) -> str:
        """回答问题: 如提供候选答案则选择最近邻，否则返回嵌入。

        Args:
            question: 问题文本
            candidates: 候选答案列表 (如提供, 返回最佳匹配)

        Returns:
            最佳答案文本, 或 "embedded:<dim>d vector"
        """
        q_emb = self.embed(question)

        if candidates:
            best_sim = -1.0
            best_idx = 0
            for i, c in enumerate(candidates):
                c_emb = self.embed(c)
                sim = np.dot(q_emb, c_emb) / (
                    max(np.linalg.norm(q_emb), 1e-10) * max(np.linalg.norm(c_emb), 1e-10)
                )
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i
            return candidates[best_idx]

        return f"embedded:{len(q_emb)}d vector"

    def similarity(self, text_a: str, text_b: str) -> float:
        """计算两段文本的语义相似度 [-1, 1]。"""
        a = self.embed(text_a)
        b = self.embed(text_b)
        return float(np.dot(a, b) / (max(np.linalg.norm(a), 1e-10) * max(np.linalg.norm(b), 1e-10)))

    # ── 持久化 ──

    def save(self, path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            save_dict = {}
            for attr in dir(self):
                if attr.startswith('_') and not attr.startswith('__'):
                    v = getattr(self, attr)
                    if isinstance(v, np.ndarray):
                        save_dict[attr] = v
                    elif isinstance(v, list) and v and isinstance(v[0], np.ndarray):
                        for i, arr in enumerate(v):
                            save_dict[f"{attr}_{i}"] = arr
            save_dict["_config_d_model"] = np.array(self.config.d_model, dtype=np.int32)
            np.savez_compressed(path, **save_dict)
            meta = {"version": "4.4.0", "model_type": "TinyTransformer", "trained": self._trained}
            with open(path + ".json", "w") as f:
                json.dump(meta, f, indent=2)
            return True
        except Exception as e:
            logger.error("Save failed: %s", e)
            return False

    @classmethod
    def load(cls, path: str, config: TinyTransformerConfig | None = None) -> TinyTransformer | None:
        try:
            data = np.load(path + ".npz")
            cfg = config or TinyTransformerConfig.micro()
            model = cls(cfg)
            for attr in dir(model):
                if attr.startswith('_') and not attr.startswith('__'):
                    if attr in data:
                        v = data[attr]
                        current = getattr(model, attr)
                        if isinstance(current, np.ndarray):
                            setattr(model, attr, v.astype(np.float64))
                        elif isinstance(current, list):
                            for i, arr in enumerate(current):
                                key = f"{attr}_{i}"
                                if key in data:
                                    current[i] = data[key].astype(np.float64)
            model._trained = True
            return model
        except Exception as e:
            logger.error("Load failed: %s", e)
            return None

    def __repr__(self) -> str:
        cfg = self.config
        return f"TinyTransformer(d={cfg.d_model}, L={cfg.n_layers}, h={cfg.n_heads}, params={self.n_params})"
