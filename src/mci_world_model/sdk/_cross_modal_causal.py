"""MCI World Model — CrossModalCausalReasoner 跨模态因果推理器
============================================================

在统一模态编码空间中进行跨模态因果推理——
利用 UnifiedModalEncoder 的共享表征，结合因果干预
识别不同模态之间的因果关系。

核心能力:
    CrossModalCausalLink — 跨模态因果链
    CrossModalCausalReasoner — 跨模态因果推理器

设计原则:
    - 依赖 UnifiedModalEncoder (T7) 提供共享空间
    - 依赖 DoCalculus 提供干预推理框架
    - 纯 numpy，零外部依赖
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CrossModalCausalLink — 跨模态因果链
# =============================================================================


@dataclass
class CrossModalCausalLink:
    """跨模态因果链。

    Attributes:
        source_modality: 源模态
        target_modality: 目标模态
        cause_concept: 源模态中的因果概念
        effect_concept: 目标模态中的结果概念
        strength: 因果链强度 [0, 1]
        confidence: 置信度
        evidence: 支撑证据
    """

    source_modality: str
    target_modality: str
    cause_concept: str
    effect_concept: str
    strength: float = 0.0
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)


# =============================================================================
# CrossModalCausalResult — 跨模态因果推理结果
# =============================================================================


@dataclass
class CrossModalCausalResult:
    """跨模态因果推理结果。

    Attributes:
        query: 原始查询
        links: 发现的跨模态因果链
        total_strength: 总因果强度
        is_reliable: 是否可靠
    """

    query: dict = field(default_factory=dict)
    links: list[CrossModalCausalLink] = field(default_factory=list)
    total_strength: float = 0.0
    is_reliable: bool = False


# =============================================================================
# CrossModalCausalReasoner — 跨模态因果推理器
# =============================================================================


class CrossModalCausalReasoner:
    """跨模态因果推理器 — 在统一空间中发现跨模态因果链。

    用法:
        >>> reasoner = CrossModalCausalReasoner(encoder)
        >>> reasoner.add_observation("vision", features, "red_light")
        >>> reasoner.add_observation("audio", features, "alarm_sound")
        >>> result = reasoner.reason(source="vision:red_light", target="audio")
    """

    def __init__(self, min_strength: float = 0.3, min_confidence: float = 0.5):
        if not 0.0 < min_strength < 1.0:
            raise ValueError("min_strength 必须在 (0,1)")
        if not 0.0 < min_confidence < 1.0:
            raise ValueError("min_confidence 必须在 (0,1)")
        self._min_strength = min_strength
        self._min_confidence = min_confidence
        self._observations: list[dict] = []
        self._discovered_links: list[CrossModalCausalLink] = []

    @property
    def observation_count(self) -> int:
        return len(self._observations)

    def add_observation(
        self,
        modality: str,
        features: np.ndarray,
        concept_label: str,
        timestamp: float = 0.0,
    ) -> None:
        """添加跨模态观测。

        Args:
            modality: 模态名称
            features: 原始特征向量
            concept_label: 概念标签
            timestamp: 时间戳
        """
        self._observations.append(
            {
                "modality": modality,
                "features": np.asarray(features, dtype=float),
                "concept": concept_label,
                "timestamp": timestamp,
            }
        )

    def reason(
        self,
        source: str,
        target: str,
    ) -> CrossModalCausalResult:
        """执行跨模态因果推理。

        Args:
            source: 源标识 "modality:concept"
            target: 目标模态名称

        Returns:
            CrossModalCausalResult
        """
        # 解析源
        if ":" in source:
            source_modality, source_concept = source.split(":", 1)
        else:
            source_modality = source
            source_concept = ""

        # 收集相关观测
        source_obs = [
            o for o in self._observations if o["modality"] == source_modality and o["concept"] == source_concept
        ]
        target_obs = [o for o in self._observations if o["modality"] == target]

        if not source_obs or not target_obs:
            return CrossModalCausalResult(
                query={"source": source, "target": target},
                is_reliable=False,
            )

        # 简化跨模态因果推理: 基于特征相关性
        links = []
        for s_obs in source_obs:
            for t_obs in target_obs:
                # 计算特征相似度作为因果强度代理
                s_feat = np.atleast_1d(s_obs["features"])
                t_feat = np.atleast_1d(t_obs["features"])

                # 确保维度一致
                min_dim = min(len(s_feat), len(t_feat))
                if min_dim == 0:
                    continue

                corr = self._compute_correlation(s_feat[:min_dim], t_feat[:min_dim])

                # 时间因果: 源先于目标 → 增强因果强度
                temporal_factor = 1.0
                if s_obs["timestamp"] < t_obs["timestamp"]:
                    temporal_factor = 1.2
                elif s_obs["timestamp"] > t_obs["timestamp"]:
                    temporal_factor = 0.8

                strength = min(abs(corr) * temporal_factor, 1.0)
                confidence = min(strength * 1.5, 1.0)

                if strength >= self._min_strength:
                    link = CrossModalCausalLink(
                        source_modality=source_modality,
                        target_modality=target,
                        cause_concept=source_concept,
                        effect_concept=t_obs["concept"],
                        strength=strength,
                        confidence=confidence,
                        evidence=[f"correlation={corr:.3f}", f"temporal_factor={temporal_factor:.2f}"],
                    )
                    links.append(link)
                    self._discovered_links.append(link)

        # 聚合
        total_strength = float(np.mean([link.strength for link in links])) if links else 0.0
        is_reliable = (
            len(links) > 0
            and total_strength >= self._min_strength
            and all(link.confidence >= self._min_confidence for link in links)
        )

        return CrossModalCausalResult(
            query={"source": source, "target": target},
            links=links,
            total_strength=total_strength,
            is_reliable=is_reliable,
        )

    @staticmethod
    def _compute_correlation(x: np.ndarray, y: np.ndarray) -> float:
        """计算 Pearson 相关系数。"""
        if len(x) < 2:
            return 0.0
        corr = np.corrcoef(x, y)
        if corr.shape == (2, 2):
            return float(corr[0, 1])
        return 0.0

    def get_discovered_links(self, modality: str | None = None) -> list[CrossModalCausalLink]:
        """获取已发现的跨模态因果链。"""
        if modality is None:
            return list(self._discovered_links)
        return [link for link in self._discovered_links if modality in (link.source_modality, link.target_modality)]

    def statistics(self) -> dict[str, Any]:
        return {
            "observation_count": self.observation_count,
            "discovered_links": len(self._discovered_links),
            "min_strength": self._min_strength,
            "min_confidence": self._min_confidence,
        }
