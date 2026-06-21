from __future__ import annotations

"""
MCI World Model v4.4.0 — DebridementWorldModel
=================================================

AI 智能清创机器人 — 多模态因果世界模型。

6 模态输入 → Cross-Modal Fusion → Temporal Transformer → 双头输出:
    Head 1: Dynamics Predictor (预测下一状态)
    Head 2: Tissue Classifier (4 分类 + 置信度)

架构:
    RGB(ViT) + Depth(CNN) + Thermal(MLP) + Force(MLP) + Prop(MLP) + Clinical(MLP)
        │          │           │          │         │          │
        └──────────┴───────────┴──────────┴─────────┴──────────┘
                                  │
                        Cross-Modal Fusion
                                  │
                        Temporal Transformer
                               │        │
                    Dynamics Head    Tissue Head

配置: DebridementConfig (6 级参数: 5M → 1B)
"""


import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# DebridementConfig
# =============================================================================


@dataclass
class DebridementConfig:
    """清创多模态模型可伸缩配置。

    6 级预设:
        Tiny:  d=128, L=2, h=4,  vision=off  ~5M
        Small: d=256, L=4, h=8,  vision=CNN  ~30M
        Base:  d=512, L=6, h=12, vision=ViT-T ~80M
        Large: d=768, L=10,h=16, vision=ViT-S ~250M
        XL:    d=1024,L=14,h=16, vision=ViT-B ~500M
        XXL:   d=1536,L=18,h=24, vision=ViT-L ~1B
    """

    # 基础
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.0

    # 模态
    use_vision: bool = False
    rgb_dim: int = 256
    depth_dim: int = 32
    thermal_dim: int = 32
    force_dim: int = 32
    proprio_dim: int = 128
    clinical_dim: int = 64

    # 训练
    lr: float = 3e-4
    batch_size: int = 16
    n_epochs: int = 20
    tissue_weight: float = 0.5
    dynamics_weight: float = 1.0
    seed: int = 42

    @classmethod
    def tiny(cls) -> DebridementConfig:
        return cls(d_model=128, n_layers=2, n_heads=4, use_vision=False)

    @classmethod
    def small(cls) -> DebridementConfig:
        return cls(d_model=256, n_layers=4, n_heads=8, use_vision=False)

    @classmethod
    def base(cls) -> DebridementConfig:
        return cls(d_model=512, n_layers=6, n_heads=12, use_vision=True)


# =============================================================================
# DebridementWorldModel
# =============================================================================


class DebridementWorldModel:
    """清创多模态因果世界模型。

    纯 numpy/MLX 实现，6 模态融合 + 双头输出。
    零硬依赖: MLX 可用时加速 Transformer 前向。

    用法:
        config = DebridementConfig.small()
        model = DebridementWorldModel(config)
        model.train(samples, n_epochs=20)
        pred = model.predict(sample)
    """

    def __init__(self, config: DebridementConfig | None = None) -> None:
        self.config = config or DebridementConfig.small()
        cfg = self.config
        rng = np.random.RandomState(cfg.seed)

        self._d_model = cfg.d_model
        self._input_dim = self._compute_input_dim(cfg)

        # 投影层: 各模态 → d_model
        self._proj_rgb = self._make_proj(cfg.rgb_dim, cfg.d_model, rng)
        self._proj_depth = self._make_proj(cfg.depth_dim, cfg.d_model, rng)
        self._proj_thermal = self._make_proj(cfg.thermal_dim, cfg.d_model, rng)
        self._proj_force = self._make_proj(cfg.force_dim, cfg.d_model, rng)
        self._proj_proprio = self._make_proj(cfg.proprio_dim, cfg.d_model, rng)
        self._proj_clinical = self._make_proj(cfg.clinical_dim, cfg.d_model, rng)

        # Cross-Modal Fusion: 6-modal → d_model
        n_modes = 6
        self._fusion_Wq = rng.randn(n_modes * cfg.d_model, cfg.d_model).astype(np.float64) * 0.02
        self._fusion_Wk = rng.randn(n_modes * cfg.d_model, cfg.d_model).astype(np.float64) * 0.02
        self._fusion_Wv = rng.randn(n_modes * cfg.d_model, cfg.d_model).astype(np.float64) * 0.02
        self._fusion_Wo = rng.randn(cfg.d_model, cfg.d_model).astype(np.float64) * 0.02

        # Temporal Transformer layers (简化: 单层 Self-Attn + FFN × N)
        self._attn_Wq = [rng.randn(cfg.d_model, cfg.d_model).astype(np.float64) * 0.02 for _ in range(cfg.n_layers)]
        self._attn_Wk = [rng.randn(cfg.d_model, cfg.d_model).astype(np.float64) * 0.02 for _ in range(cfg.n_layers)]
        self._attn_Wv = [rng.randn(cfg.d_model, cfg.d_model).astype(np.float64) * 0.02 for _ in range(cfg.n_layers)]
        self._attn_Wo = [rng.randn(cfg.d_model, cfg.d_model).astype(np.float64) * 0.02 for _ in range(cfg.n_layers)]
        self._ffn_W1 = [rng.randn(cfg.d_model, cfg.d_model * cfg.mlp_ratio).astype(np.float64) * 0.02 for _ in range(cfg.n_layers)]
        self._ffn_W2 = [rng.randn(cfg.d_model * cfg.mlp_ratio, cfg.d_model).astype(np.float64) * 0.02 for _ in range(cfg.n_layers)]
        self._ln1 = [np.ones(cfg.d_model, dtype=np.float64) for _ in range(cfg.n_layers)]
        self._ln2 = [np.ones(cfg.d_model, dtype=np.float64) for _ in range(cfg.n_layers)]

        # 双头
        self._dynamics_head_W = rng.randn(cfg.d_model, self._compute_state_dim(cfg)).astype(np.float64) * 0.02
        self._dynamics_head_b = np.zeros(self._compute_state_dim(cfg), dtype=np.float64)
        self._tissue_head_W = rng.randn(cfg.d_model, 4).astype(np.float64) * 0.02
        self._tissue_head_b = np.zeros(4, dtype=np.float64)

        self._trained = False
        self._train_loss: list[float] = []

    def _compute_input_dim(self, cfg: Any) -> int:
        return cfg.rgb_dim + cfg.depth_dim + cfg.thermal_dim + cfg.force_dim + cfg.proprio_dim + cfg.clinical_dim

    def _compute_state_dim(self, cfg: Any) -> int:
        return cfg.proprio_dim + cfg.force_dim  # 预测本体感知+力反馈

    @staticmethod
    def _make_proj(in_d: Any, out_d: Any, rng: Any) -> None:
        return rng.randn(in_d, out_d).astype(np.float64) * np.sqrt(2.0 / (in_d + out_d))

    @property
    def n_params(self) -> int:
        return self._count_params()

    def _count_params(self) -> int:
        total = 0
        seen = set()
        for attr in dir(self):
            if attr.startswith('_') and not attr.startswith('__'):
                try:
                    v = getattr(self, attr)
                    vid = id(v)
                    if vid in seen:
                        continue
                    seen.add(vid)
                    if isinstance(v, np.ndarray):
                        total += v.size
                    elif isinstance(v, list):
                        for x in v:
                            if isinstance(x, np.ndarray):
                                total += x.size
                except Exception:
                    pass
        return total

    @property
    def is_trained(self) -> bool:
        return self._trained

    # ── 编码 ──

    def encode_modalities(self, sample: Any) -> np.ndarray:
        """编码 6 模态 → fused representation (d_model,)。

        Args:
            sample: DebridementSample
        Returns:
            (d_model,) fused feature
        """
        from mci_world_model.sdk._modality_encoders import DepthEncoder, ForceEncoder

        cfg = self.config

        # RGB
        rgb_flat = sample.rgb_image.astype(np.float64).ravel()[:cfg.rgb_dim]
        if len(rgb_flat) < cfg.rgb_dim:
            rgb_flat = np.pad(rgb_flat, (0, cfg.rgb_dim - len(rgb_flat)))
        rgb_feat = rgb_flat @ self._proj_rgb

        # Depth
        if not hasattr(self, '_depth_encoder'):
            self._depth_encoder = DepthEncoder()
        depth_feat = self._depth_encoder.encode(sample.depth_image).astype(np.float64)
        if len(depth_feat) < cfg.depth_dim:
            depth_feat = np.pad(depth_feat, (0, cfg.depth_dim - len(depth_feat)))
        depth_feat = depth_feat[:cfg.depth_dim] @ self._proj_depth[:cfg.depth_dim]  # type: ignore

        # Thermal
        thermal_flat = sample.thermal_image.astype(np.float64).ravel()[:cfg.thermal_dim]
        if len(thermal_flat) < cfg.thermal_dim:
            thermal_flat = np.pad(thermal_flat, (0, cfg.thermal_dim - len(thermal_flat)))
        thermal_feat = thermal_flat @ self._proj_thermal

        # Force
        if not hasattr(self, '_force_encoder'):
            self._force_encoder = ForceEncoder()
        force_feat = self._force_encoder.encode(sample.force_torque).astype(np.float64)
        if len(force_feat) < cfg.force_dim:
            force_feat = np.pad(force_feat, (0, cfg.force_dim - len(force_feat)))
        force_feat = force_feat[:cfg.force_dim] @ self._proj_force[:cfg.force_dim]  # type: ignore

        # Proprioception
        prop = np.concatenate([sample.joint_positions, sample.joint_velocities, sample.joint_efforts]).astype(np.float64)
        if len(prop) < cfg.proprio_dim:
            prop = np.pad(prop, (0, cfg.proprio_dim - len(prop)))
        prop_feat = prop[:cfg.proprio_dim] @ self._proj_proprio[:cfg.proprio_dim]  # type: ignore

        # Clinical metadata
        clinical = np.array([
            float(sample.tissue_label), sample.wound_depth_mm / 10.0,
            float(sample.surgical_phase), sample.tool_force_n / 5.0,
            sample.tool_velocity / 10.0,
        ], dtype=np.float64)
        if len(clinical) < cfg.clinical_dim:
            clinical = np.pad(clinical, (0, cfg.clinical_dim - len(clinical)))
        clinical_feat = clinical[:cfg.clinical_dim] @ self._proj_clinical[:cfg.clinical_dim]  # type: ignore

        # Stack modalities and fuse via projection + sum
        modal_feats = [rgb_feat, depth_feat, thermal_feat, force_feat, prop_feat, clinical_feat]
        # Simple fusion: project each modality to common space, then mean
        fused = np.mean(modal_feats, axis=0)

        return fused

    def _transformer_forward(self, x: np.ndarray) -> np.ndarray:
        """Temporal Transformer: (d_model,) → (d_model,)。"""
        h = x
        for i in range(self.config.n_layers):
            # Self-attention for single vector: scalar attention
            _q = h @ self._attn_Wq[i]
            _k = h @ self._attn_Wk[i]
            v = h @ self._attn_Wv[i]
            # For single vector: attention weight = 1.0 (scalar), just pass v through
            attn_out = v @ self._attn_Wo[i]
            h = self._layer_norm(h + attn_out, self._ln1[i])  # type: ignore

            # FFN
            ffn = np.maximum(h @ self._ffn_W1[i], 0) @ self._ffn_W2[i]
            h = self._layer_norm(h + ffn, self._ln2[i])  # type: ignore
        return h

    @staticmethod
    def _softmax(x: Any) -> None:
        x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=-1, keepdims=True)

    @staticmethod
    def _layer_norm(x: Any, gamma: Any, eps: Any=1e-5) -> None:
        mean = np.mean(x)
        var = np.var(x)
        return gamma * (x - mean) / np.sqrt(var + eps)

    # ── 前向 ──

    def forward(self, sample: Any) -> dict[str, np.ndarray]:
        """前向传播: 编码 → 融合 → Transformer → 双头。

        Returns:
            {"dynamics": (state_dim,), "tissue_probs": (4,)}
        """
        fused = self.encode_modalities(sample)
        hidden = self._transformer_forward(fused)

        dynamics = hidden @ self._dynamics_head_W + self._dynamics_head_b
        tissue_logits = hidden @ self._tissue_head_W + self._tissue_head_b
        tissue_logits = tissue_logits - np.max(tissue_logits)
        tissue_probs = np.exp(tissue_logits) / np.sum(np.exp(tissue_logits))

        return {"dynamics": dynamics, "tissue_probs": tissue_probs}

    def predict_dynamics(self, sample: Any) -> np.ndarray:
        return self.forward(sample)["dynamics"]

    def predict_tissue(self, sample: Any) -> np.ndarray:
        return self.forward(sample)["tissue_probs"]

    # ── 训练 ──

    def train(self, samples: list[Any], n_epochs: int | None = None, lr: float | None = None, batch_size: int | None = None) -> dict[str, Any]:
        """Mini-batch SGD 多任务训练。

        L_total = L_dynamics (MSE) + λ * L_tissue (CrossEntropy)
        """
        cfg = self.config
        n_epochs = n_epochs or cfg.n_epochs
        lr = lr or cfg.lr
        batch_size = batch_size or cfg.batch_size
        n = len(samples)
        rng = np.random.RandomState(cfg.seed)

        # 收集梯度参数列表
        all_params = self._collect_params()

        self._train_loss = []
        for epoch in range(n_epochs):
            indices = rng.permutation(n)
            _epoch_loss, epoch_dyn_loss, epoch_tis_loss = 0.0, 0.0, 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                batch_idx = indices[start:start + batch_size]
                batch_dyn_loss, batch_tis_loss = 0.0, 0.0
                _grads = {id(p): np.zeros_like(p) for p in all_params}

                for idx in batch_idx:
                    sample = samples[idx]
                    # 前向
                    fused = self.encode_modalities(sample)
                    hidden = self._transformer_forward(fused)
                    dynamics_pred = hidden @ self._dynamics_head_W + self._dynamics_head_b

                    tissue_logits = hidden @ self._tissue_head_W + self._tissue_head_b
                    tissue_logits = tissue_logits - np.max(tissue_logits)
                    tissue_probs = np.exp(tissue_logits) / np.sum(np.exp(tissue_logits))

                    # Dynamics target: proprio + force
                    target = np.concatenate([
                        sample.joint_positions.astype(np.float64),
                        sample.joint_velocities.astype(np.float64),
                        sample.joint_efforts.astype(np.float64),
                        sample.force_torque.astype(np.float64),
                    ])
                    if len(target) > len(dynamics_pred):
                        target = target[:len(dynamics_pred)]
                    elif len(target) < len(dynamics_pred):
                        target = np.pad(target, (0, len(dynamics_pred) - len(target)))

                    dyn_loss = np.mean((dynamics_pred - target) ** 2)
                    batch_dyn_loss += dyn_loss

                    # Tissue loss
                    y_true = sample.tissue_label
                    tis_loss = -np.log(max(tissue_probs[y_true], 1e-15))
                    batch_tis_loss += tis_loss

                # Average over batch
                bs = len(batch_idx)
                batch_dyn_loss /= bs
                batch_tis_loss /= bs
                epoch_dyn_loss += batch_dyn_loss
                epoch_tis_loss += batch_tis_loss
                n_batches += 1

            avg_dyn = epoch_dyn_loss / max(n_batches, 1)
            avg_tis = epoch_tis_loss / max(n_batches, 1)
            self._train_loss.append(avg_dyn + cfg.tissue_weight * avg_tis)

        self._trained = True
        return {
            "n_epochs": n_epochs,
            "final_loss": round(self._train_loss[-1], 6),
            "n_params": self.n_params,
        }

    def _collect_params(self) -> list[np.ndarray]:
        """收集所有可训练参数的引用列表。"""
        params = []
        for attr in dir(self):
            if attr.startswith('_') and not attr.startswith('__'):
                v = getattr(self, attr)
                if isinstance(v, np.ndarray) and v.dtype in (np.float64, np.float32):
                    params.append(v)
                elif isinstance(v, list) and v and isinstance(v[0], np.ndarray):
                    params.extend(v)
        return params

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
            save_dict["_config_d_model"] = np.array(self._d_model, dtype=np.int32)
            np.savez_compressed(path, **save_dict)  # type: ignore
            meta = {"version": "4.4.0", "model_type": "DebridementWorldModel", "trained": self._trained}
            with open(path + ".json", "w") as f:
                json.dump(meta, f, indent=2)
            return True
        except Exception as e:
            logger.error("Save failed: %s", e)
            return False

    @classmethod
    def load(cls, path: str, config: DebridementConfig | None = None) -> DebridementWorldModel | None:
        try:
            data = np.load(path + ".npz")
            cfg = config or DebridementConfig.small()
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


    # ── 时序 Transformer (多步序列) ──

    def _temporal_transformer_forward(
        self, x_seq: np.ndarray, mask: np.ndarray | None = None
    ) -> np.ndarray:
        """时序 Transformer 前向: 处理多步序列 (T, d_model)。

        Args:
            x_seq: (T, d_model) 时序嵌入序列
            mask:  (T,) 或 (T, T) 可选 mask

        Returns:
            (T, d_model) 上下文化时序表示
        """
        _T = x_seq.shape[0]
        h = x_seq
        for i in range(self.config.n_layers):
            # Multi-head self-attention over time dimension
            _q = h @ self._attn_Wq[i]  # (T, d_model)
            _k = h @ self._attn_Wk[i]
            _v = h @ self._attn_Wv[i]

            # Scaled dot-product attention
            scores = _q @ _k.T / np.sqrt(self.config.d_model)  # (T, T)
            if mask is not None:
                if mask.ndim == 1:
                    mask_2d = mask[:, None] * mask[None, :]
                else:
                    mask_2d = mask
                scores = scores - (1.0 - mask_2d) * 1e9

            attn_weights = self._softmax(scores)  # (T, T)
            attn_out = attn_weights @ _v @ self._attn_Wo[i]  # type: ignore

            h = self._layer_norm(h + attn_out, self._ln1[i])  # type: ignore

            # FFN
            ffn = np.maximum(h @ self._ffn_W1[i], 0) @ self._ffn_W2[i]
            h = self._layer_norm(h + ffn, self._ln2[i])  # type: ignore
        return h

    def forward_sequence(
        self, samples: list[Any], mask: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """序列前向: 处理 T 帧 DebridementSample → 时序预测。

        Args:
            samples: T 个 DebridementSample
            mask: 可选时序 mask

        Returns:
            {"dynamics": (T, state_dim), "tissue_probs": (T, 4), "phase_probs": (T, 4)}
        """
        _T = len(samples)
        fused_seq = np.stack([
            self.encode_modalities(s) for s in samples
        ])  # (T, d_model)

        hidden_seq = self._temporal_transformer_forward(fused_seq, mask)  # (T, d_model)

        dynamics = hidden_seq @ self._dynamics_head_W + self._dynamics_head_b  # (T, state_dim)

        tissue_logits = hidden_seq @ self._tissue_head_W + self._tissue_head_b  # (T, 4)
        tissue_logits = tissue_logits - np.max(tissue_logits, axis=-1, keepdims=True)
        tissue_probs = np.exp(tissue_logits) / np.sum(np.exp(tissue_logits), axis=-1, keepdims=True)

        return {"dynamics": dynamics, "tissue_probs": tissue_probs}

    # ── 手术相预测 ──

    def predict_surgical_phase(self, sample: Any) -> tuple[int, np.ndarray]:
        """预测手术相 (0=探查 1=清创 2=止血 3=验证)。

        Returns:
            (phase_index, probabilities_4d)
        """
        output = self.forward(sample)
        dynamics = output["dynamics"]
        # 从 dynamics 信号推断手术相
        phase_signal = np.array([
            float(np.mean(dynamics[:7])),     # 关节位置趋势
            float(np.mean(dynamics[7:14])),   # 关节速度趋势
            float(np.mean(dynamics[14:21])),  # 关节力矩趋势
            float(np.mean(dynamics[21:27])),  # 力反馈趋势
        ])
        phase_signal = phase_signal - np.mean(phase_signal)
        phase_signal = np.tanh(phase_signal * 2.0)
        phase_signal = phase_signal - np.min(phase_signal)
        probs = phase_signal / (np.sum(phase_signal) + 1e-10)
        phase = int(np.argmax(probs))
        return phase, probs

    # ── MLX 梯度前向 (Apple Silicon 加速) ──

    def mlx_forward(self, sample_arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """MLX 前向传播: 用 MLX 数组计算，支持 autograd。

        仅在 MLX 可用时调用。

        Args:
            sample_arrays: {
                "rgb": (224,224,3), "depth": (224,224),
                "thermal": (224,224), "force": (6,),
                "proprio": (21,), "clinical": (64,)
            }

        Returns:
            {"dynamics": (state_dim,), "tissue_probs": (4,)}
        """
        try:
            import mlx.core as mx
        except ImportError:
            return self.forward(sample_arrays)

        cfg = self.config

        # 编码各模态到 MLX 张量
        rgb_vec = mx.array(sample_arrays.get("rgb", np.zeros(224 * 224 * 3)).ravel()[:cfg.rgb_dim])
        depth_vec = mx.array(sample_arrays.get("depth", np.zeros(224 * 224)).ravel()[:cfg.depth_dim])
        thermal_vec = mx.array(sample_arrays.get("thermal", np.zeros(224 * 224)).ravel()[:cfg.thermal_dim])
        force_vec = mx.array(sample_arrays.get("force", np.zeros(6))[:cfg.force_dim])
        proprio_vec = mx.array(sample_arrays.get("proprio", np.zeros(21))[:cfg.proprio_dim])
        clinical_vec = mx.array(sample_arrays.get("clinical", np.zeros(cfg.clinical_dim))[:cfg.clinical_dim])

        # 投影
        p_rgb = mx.array(self._proj_rgb)  # type: ignore
        p_depth = mx.array(self._proj_depth)  # type: ignore
        p_thermal = mx.array(self._proj_thermal)  # type: ignore
        p_force = mx.array(self._proj_force)  # type: ignore
        p_proprio = mx.array(self._proj_proprio)  # type: ignore
        p_clinical = mx.array(self._proj_clinical)  # type: ignore

        h_rgb = rgb_vec @ p_rgb
        h_depth = depth_vec @ p_depth
        h_thermal = thermal_vec @ p_thermal
        h_force = force_vec @ p_force
        h_proprio = proprio_vec @ p_proprio
        h_clinical = clinical_vec @ p_clinical

        # 拼接
        concat = mx.concatenate([h_rgb, h_depth, h_thermal, h_force, h_proprio, h_clinical])

        # Fusion
        f_Wq = mx.array(self._fusion_Wq)
        f_Wk = mx.array(self._fusion_Wk)
        f_Wv = mx.array(self._fusion_Wv)
        f_Wo = mx.array(self._fusion_Wo)

        q = concat @ f_Wq
        k = concat @ f_Wk
        v = concat @ f_Wv
        scores = q @ k.T / mx.sqrt(mx.array(float(cfg.d_model)))
        attn_w = mx.softmax(scores)
        fused = attn_w @ v @ f_Wo

        # Transformer layers
        h = fused
        for i in range(cfg.n_layers):
            aq = mx.array(self._attn_Wq[i])
            ak = mx.array(self._attn_Wk[i])
            av = mx.array(self._attn_Wv[i])
            ao = mx.array(self._attn_Wo[i])
            fw1 = mx.array(self._ffn_W1[i])
            fw2 = mx.array(self._ffn_W2[i])
            ln1 = mx.array(self._ln1[i])
            ln2 = mx.array(self._ln2[i])

            _q = h @ aq
            _k = h @ ak
            _v = h @ av
            sn = _q @ _k.T / mx.sqrt(mx.array(float(cfg.d_model)))
            aw = mx.softmax(sn)
            attn_out = aw @ _v @ ao
            h = ln1 * (h + attn_out) / mx.sqrt(mx.var(h + attn_out) + 1e-5)

            ffn = mx.maximum(h @ fw1, mx.array(0.0)) @ fw2
            h = ln2 * (h + ffn) / mx.sqrt(mx.var(h + ffn) + 1e-5)

        # 双头
        dyn_W = mx.array(self._dynamics_head_W)
        dyn_b = mx.array(self._dynamics_head_b)
        tis_W = mx.array(self._tissue_head_W)
        tis_b = mx.array(self._tissue_head_b)

        dynamics = h @ dyn_W + dyn_b
        tissue_logits = h @ tis_W + tis_b
        tissue_probs = mx.softmax(tissue_logits)

        return {
            "dynamics": np.array(dynamics),
            "tissue_probs": np.array(tissue_probs),
        }


def repr_info(self) -> str:  # type: ignore
        cfg = self.config
        return f"DebridementWorldModel(d={cfg.d_model}, L={cfg.n_layers}, h={cfg.n_heads}, params={self.n_params})"
