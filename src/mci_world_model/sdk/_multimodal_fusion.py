from __future__ import annotations

"""
MCI World Model v3.3.0 — MultimodalFusion 跨模态融合层
========================================================

将来自不同模态编码器的特征向量融合为统一表示。

三种融合策略:
    attention — 跨模态注意力（Q=主模态, K/V=其他模态）
    weighted  — 基于 confidence 的加权平均（对齐到相同维度后）
    concat    — 简单拼接 + 线性投影到固定维度

设计原则:
    - 纯 numpy，零外部依赖
    - 输出 FusedRepresentation 包含融合向量 + 各模态贡献权重
    - 支持缺失模态（某些模态为 None 时跳过）
"""


import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# FusedRepresentation — 融合表示
# =============================================================================


@dataclass
class FusedRepresentation:
    """融合后的多模态表示。

    Attributes:
        fused_vector: 融合后的特征向量
        modality_contributions: 各模态贡献权重 {modality: weight}
        confidence: 融合置信度 [0, 1]
        strategy: 使用的融合策略
    """

    fused_vector: np.ndarray
    modality_contributions: dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    strategy: str = "attention"


# =============================================================================
# MultimodalFusion — 跨模态融合
# =============================================================================


class MultimodalFusion:
    """跨模态特征融合器。

    Example:
        >>> fusion = MultimodalFusion(strategy="weighted", output_dim=32)
        >>> features = {
        ...     "vision": np.random.rand(32),
        ...     "audio": np.random.rand(16),
        ... }
        >>> fused = fusion.fuse(features)
        >>> assert fused.fused_vector.shape == (32,)
    """

    def __init__(
        self,
        strategy: str = "attention",
        output_dim: int = 32,
        seed: int = 42,
    ):
        """
        Args:
            strategy: 融合策略 "attention" / "weighted" / "concat"
            output_dim: 输出向量维度
            seed: 随机种子（用于投影矩阵初始化）
        """
        if strategy not in ("attention", "weighted", "concat"):
            raise ValueError(f"未知融合策略: {strategy}")
        self._strategy = strategy
        self._output_dim = output_dim
        self._rng = np.random.RandomState(seed)
        # 投影矩阵缓存: {input_dim: projection_matrix}
        self._projections: dict[int, np.ndarray] = {}

    @property
    def strategy(self) -> str:
        return self._strategy

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _get_projection(self, input_dim: int) -> np.ndarray:
        """获取或创建投影矩阵 (input_dim → output_dim)。"""
        if input_dim not in self._projections:
            scale = np.sqrt(2.0 / (input_dim + self._output_dim))
            self._projections[input_dim] = self._rng.randn(input_dim, self._output_dim).astype(np.float64) * scale
        return self._projections[input_dim]

    def _project(self, vec: np.ndarray) -> np.ndarray:
        """将向量投影到 output_dim 维度。"""
        if len(vec) == self._output_dim:
            return vec.copy()
        W = self._get_projection(len(vec))
        return vec @ W

    # -----------------------------------------------------------------
    # fuse — 融合多模态特征
    # -----------------------------------------------------------------

    def fuse(
        self,
        modality_features: dict[str, np.ndarray],
        modality_confidences: dict[str, float] | None = None,
    ) -> FusedRepresentation:
        """融合多模态特征向量。

        Args:
            modality_features: {模态名: 特征向量}
            modality_confidences: {模态名: 置信度} (可选)

        Returns:
            FusedRepresentation
        """
        if not modality_features:
            return FusedRepresentation(
                fused_vector=np.zeros(self._output_dim, dtype=np.float64),
                modality_contributions={},
                confidence=0.0,
                strategy=self._strategy,
            )

        if self._strategy == "weighted":
            return self._fuse_weighted(modality_features, modality_confidences)
        elif self._strategy == "concat":
            return self._fuse_concat(modality_features)
        else:  # attention
            return self._fuse_attention(modality_features, modality_confidences)

    # -----------------------------------------------------------------
    # 策略实现
    # -----------------------------------------------------------------

    def _fuse_weighted(
        self,
        features: dict[str, np.ndarray],
        confidences: dict[str, float] | None,
    ) -> FusedRepresentation:
        """加权平均融合：对齐维度后按 confidence 加权。"""
        if confidences is None:
            confidences = dict.fromkeys(features, 1.0)

        projected = {}
        for name, vec in features.items():
            projected[name] = self._project(vec)

        total_weight = 0.0
        fused = np.zeros(self._output_dim, dtype=np.float64)
        contributions: dict[str, float] = {}

        for name, proj_vec in projected.items():
            w = confidences.get(name, 1.0)
            fused += w * proj_vec
            total_weight += w

        if total_weight > 0:
            fused /= total_weight

        for name in features:
            contributions[name] = confidences.get(name, 1.0) / max(total_weight, 1e-10)

        avg_conf = float(np.mean(list(confidences.values())))

        return FusedRepresentation(
            fused_vector=fused,
            modality_contributions=contributions,
            confidence=avg_conf,
            strategy="weighted",
        )

    def _fuse_concat(
        self,
        features: dict[str, np.ndarray],
    ) -> FusedRepresentation:
        """拼接 + 线性投影融合。"""
        # 按名称排序确保顺序一致
        sorted_names = sorted(features.keys())
        concat = np.concatenate([features[n] for n in sorted_names])
        projected = self._project(concat)

        # 各模态贡献按维度比例
        contributions: dict[str, float] = {}
        for name in sorted_names:
            contributions[name] = len(features[name]) / max(len(concat), 1)

        return FusedRepresentation(
            fused_vector=projected,
            modality_contributions=contributions,
            confidence=1.0,
            strategy="concat",
        )

    def _fuse_attention(
        self,
        features: dict[str, np.ndarray],
        confidences: dict[str, float] | None,
    ) -> FusedRepresentation:
        """跨模态注意力融合。

        Q = 主模态（维度最大的），K/V = 其他模态均值。
        attention_weight = softmax(Q · K^T / sqrt(d))
        """
        if confidences is None:
            confidences = dict.fromkeys(features, 1.0)

        sorted_names = sorted(features.keys())
        if len(sorted_names) == 1:
            name = sorted_names[0]
            proj = self._project(features[name])
            return FusedRepresentation(
                fused_vector=proj,
                modality_contributions={name: 1.0},
                confidence=confidences.get(name, 1.0),
                strategy="attention",
            )

        # 投影到统一维度
        projected = {n: self._project(features[n]) for n in sorted_names}

        # 主模态: 维度最大的
        primary = max(sorted_names, key=lambda n: len(features[n]))
        others = [n for n in sorted_names if n != primary]

        Q = projected[primary]  # (d,)
        K = np.mean([projected[n] for n in others], axis=0)  # (d,)

        # Attention score (scaled dot product)
        d = self._output_dim
        score = float(np.dot(Q, K) / np.sqrt(d))
        # 归一化到 [0,1] 使用 sigmoid
        attn_weight = 1.0 / (1.0 + np.exp(-score))

        # 融合: attn * Q + (1-attn) * K
        fused = attn_weight * Q + (1.0 - attn_weight) * K

        contributions: dict[str, float] = {}
        contributions[primary] = attn_weight
        other_weight = (1.0 - attn_weight) / max(len(others), 1)
        for n in others:
            contributions[n] = other_weight

        avg_conf = float(np.mean(list(confidences.values())))

        return FusedRepresentation(
            fused_vector=fused,
            modality_contributions=contributions,
            confidence=avg_conf,
            strategy="attention",
        )

    # -----------------------------------------------------------------
    # encode_to_state — 融合表示 → WorldState
    # -----------------------------------------------------------------

    def encode_to_state(
        self,
        fused: FusedRepresentation,
        state_class: type | None = None,
    ):
        """融合表示 → WorldState。

        Args:
            fused: FusedRepresentation
            state_class: WorldState 子类（可选，None 时使用 MultimodalWorldState）

        Returns:
            WorldState 实例
        """
        if state_class is None:
            from mci_world_model.sdk._world_state import MultimodalWorldState

            state_class = MultimodalWorldState

        if hasattr(state_class, "from_vector"):
            return state_class.from_vector(fused.fused_vector)
        raise NotImplementedError(f"{state_class.__name__} 不支持 from_vector()")

    # -----------------------------------------------------------------
    # 字符串表示
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        return f"MultimodalFusion(strategy={self._strategy!r}, output_dim={self._output_dim})"
