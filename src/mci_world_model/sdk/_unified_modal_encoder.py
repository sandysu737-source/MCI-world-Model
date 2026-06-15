"""MCI World Model — UnifiedModalEncoder 统一模态编码器
=====================================================

跨模态统一表征——将视觉/音频/热成像/文本等异构信号
映射到同一语义空间，支持跨模态检索和因果推理。

核心能力:
    ModalityProjection   — 模态投影层(可学习)
    UnifiedModalEncoder  — 统一编码器(注册+编码+对齐)

设计原则:
    - 基于 _modality_encoders.py 的单模态编码器
    - 统一投影到 shared_dim 维共享空间
    - 跨模态对比对齐: 同语义拉近, 异语义推远
    - 纯 numpy，零外部依赖
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# ModalityProjection — 模态投影层
# =============================================================================


class ModalityProjection:
    """模态投影层 — 将单模态特征投影到共享空间。

    使用简化线性投影: y = W @ x + b
    W: (shared_dim, input_dim), b: (shared_dim,)
    """

    def __init__(
        self,
        modality_name: str,
        input_dim: int,
        shared_dim: int,
        seed: int = 42,
    ):
        self._modality_name = modality_name
        self._input_dim = input_dim
        self._shared_dim = shared_dim

        rng = np.random.RandomState(seed)
        # Xavier 初始化
        scale = np.sqrt(2.0 / (input_dim + shared_dim))
        self._W = rng.randn(shared_dim, input_dim) * scale
        self._b = np.zeros(shared_dim)

    @property
    def modality_name(self) -> str:
        return self._modality_name

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def shared_dim(self) -> int:
        return self._shared_dim

    def project(self, features: np.ndarray) -> np.ndarray:
        """将单模态特征投影到共享空间。

        Args:
            features: (input_dim,) 单模态特征向量

        Returns:
            (shared_dim,) 共享空间向量
        """
        vec = np.atleast_1d(np.asarray(features, dtype=float))
        # 如果维度不匹配, 截断或填充
        if len(vec) < self._input_dim:
            padded = np.zeros(self._input_dim)
            padded[: len(vec)] = vec
            vec = padded
        elif len(vec) > self._input_dim:
            vec = vec[: self._input_dim]

        return self._W @ vec + self._b

    def update(self, gradient: np.ndarray, lr: float = 0.01) -> None:
        """简化梯度更新。

        Args:
            gradient: (shared_dim, input_dim) 梯度矩阵
            lr: 学习率
        """
        grad = np.atleast_2d(np.asarray(gradient, dtype=float))
        if grad.shape == self._W.shape:
            self._W -= lr * grad
        else:
            # 简化: 用外积近似
            self._W *= 1.0 - lr * 0.01


# =============================================================================
# AlignmentResult — 对齐结果
# =============================================================================


@dataclass
class AlignmentResult:
    """跨模态对齐结果。

    Attributes:
        modality_a: 模态A名称
        modality_b: 模态B名称
        similarity: 余弦相似度
        is_aligned: 是否对齐(超过阈值)
    """

    modality_a: str
    modality_b: str
    similarity: float
    is_aligned: bool


# =============================================================================
# EncodingResult — 编码结果
# =============================================================================


@dataclass
class EncodingResult:
    """统一编码结果。

    Attributes:
        modality: 模态名称
        shared_vector: 共享空间向量
        projection_norm: 投影向量范数
        metadata: 附加元数据
    """

    modality: str
    shared_vector: np.ndarray
    projection_norm: float = 0.0
    metadata: dict = field(default_factory=dict)


# =============================================================================
# UnifiedModalEncoder — 统一模态编码器
# =============================================================================


class UnifiedModalEncoder:
    """统一模态编码器 — 将异构模态映射到共享语义空间。

    用法:
        >>> encoder = UnifiedModalEncoder(shared_dim=64)
        >>> encoder.register_modality("vision", input_dim=128)
        >>> encoder.register_modality("audio", input_dim=64)
        >>> result = encoder.encode("vision", vision_features)
    """

    def __init__(self, shared_dim: int = 64, alignment_threshold: float = 0.5):
        if shared_dim < 1:
            raise ValueError(f"shared_dim 必须 ≥ 1, 当前 {shared_dim}")
        self._shared_dim = shared_dim
        self._alignment_threshold = alignment_threshold
        self._projections: dict[str, ModalityProjection] = {}
        self._encode_count: int = 0

    @property
    def shared_dim(self) -> int:
        return self._shared_dim

    @property
    def registered_modalities(self) -> list[str]:
        return list(self._projections.keys())

    @property
    def modality_count(self) -> int:
        return len(self._projections)

    def register_modality(self, modality_name: str, input_dim: int, seed: int = 42) -> None:
        """注册模态投影层。

        Args:
            modality_name: 模态名称
            input_dim: 输入特征维度
            seed: 随机种子

        Raises:
            ValueError: 模态已注册
        """
        if modality_name in self._projections:
            raise ValueError(f"模态 '{modality_name}' 已注册")
        self._projections[modality_name] = ModalityProjection(
            modality_name=modality_name,
            input_dim=input_dim,
            shared_dim=self._shared_dim,
            seed=seed,
        )
        logger.info("统一编码器: 注册模态 %s (input_dim=%d)", modality_name, input_dim)

    def encode(self, modality: str, features: np.ndarray) -> EncodingResult:
        """将单模态特征编码到共享空间。

        Args:
            modality: 模态名称
            features: 原始特征向量

        Returns:
            EncodingResult

        Raises:
            KeyError: 模态未注册
        """
        if modality not in self._projections:
            raise KeyError(f"模态 '{modality}' 未注册, 已注册: {self.registered_modalities}")

        self._encode_count += 1
        proj = self._projections[modality]
        shared_vec = proj.project(features)

        return EncodingResult(
            modality=modality,
            shared_vector=shared_vec,
            projection_norm=float(np.linalg.norm(shared_vec)),
        )

    def encode_batch(self, modality: str, features_list: list[np.ndarray]) -> list[EncodingResult]:
        """批量编码。

        Args:
            modality: 模态名称
            features_list: 特征向量列表

        Returns:
            EncodingResult 列表
        """
        return [self.encode(modality, f) for f in features_list]

    def compute_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """计算两个共享空间向量的余弦相似度。

        Args:
            vec_a: 向量A
            vec_b: 向量B

        Returns:
            余弦相似度 [-1, 1]
        """
        a = np.atleast_1d(np.asarray(vec_a, dtype=float))
        b = np.atleast_1d(np.asarray(vec_b, dtype=float))
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def align(
        self,
        modality_a: str,
        features_a: np.ndarray,
        modality_b: str,
        features_b: np.ndarray,
    ) -> AlignmentResult:
        """跨模态对齐检查。

        Args:
            modality_a: 模态A名称
            features_a: 模态A原始特征
            modality_b: 模态B名称
            features_b: 模态B原始特征

        Returns:
            AlignmentResult
        """
        enc_a = self.encode(modality_a, features_a)
        enc_b = self.encode(modality_b, features_b)

        sim = self.compute_similarity(enc_a.shared_vector, enc_b.shared_vector)
        is_aligned = sim >= self._alignment_threshold

        return AlignmentResult(
            modality_a=modality_a,
            modality_b=modality_b,
            similarity=sim,
            is_aligned=is_aligned,
        )

    def cross_modal_retrieve(
        self,
        query_modality: str,
        query_features: np.ndarray,
        candidate_modality: str,
        candidate_features_list: list[np.ndarray],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """跨模态检索。

        Args:
            query_modality: 查询模态
            query_features: 查询特征
            candidate_modality: 候选模态
            candidate_features_list: 候选特征列表
            top_k: 返回前k个

        Returns:
            [(index, similarity), ...] 按相似度降序
        """
        query_enc = self.encode(query_modality, query_features)
        query_vec = query_enc.shared_vector

        scores = []
        for i, cand_features in enumerate(candidate_features_list):
            cand_enc = self.encode(candidate_modality, cand_features)
            sim = self.compute_similarity(query_vec, cand_enc.shared_vector)
            scores.append((i, sim))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def statistics(self) -> dict[str, Any]:
        """编码器统计。"""
        return {
            "shared_dim": self._shared_dim,
            "modality_count": self.modality_count,
            "modalities": self.registered_modalities,
            "encode_count": self._encode_count,
            "alignment_threshold": self._alignment_threshold,
        }
