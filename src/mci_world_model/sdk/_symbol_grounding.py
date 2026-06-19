"""MCI World Model — SymbolGroundingLearning 符号接地学习
=========================================================

将抽象符号锚定到感知体验——使符号推理系统能够
理解符号的物理含义，实现"接地"的因果推理。

核心能力:
    GroundingEntry      — 接地条目
    SymbolGroundingLearning — 符号接地学习器

设计原则:
    - 依赖 UnifiedModalEncoder + NeuralSymbolicFusionV2
    - 纯 numpy，零外部依赖
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GroundingEntry:
    """符号接地条目。

    Attributes:
        symbol: 符号名称
        modality: 接地模态
        perceptual_vector: 感知向量
        grounding_strength: 接地强度 [0, 1]
        examples: 接地样例数
    """
    symbol: str
    modality: str = ""
    perceptual_vector: np.ndarray | None = None
    grounding_strength: float = 0.0
    examples: int = 0


class SymbolGroundingLearning:
    """符号接地学习器 — 将符号锚定到感知体验。

    用法:
        >>> sgl = SymbolGroundingLearning()
        >>> sgl.ground("red", "vision", red_features)
        >>> sgl.ground("hot", "thermal", hot_features)
        >>> strength = sgl.verify_grounding("red", test_features)
    """

    def __init__(self, similarity_threshold: float = 0.5):
        self._sim_threshold = similarity_threshold
        self._groundings: dict[str, GroundingEntry] = {}

    @property
    def grounded_symbol_count(self) -> int:
        return len(self._groundings)

    def ground(
        self, symbol: str, modality: str, perceptual_vector: np.ndarray
    ) -> GroundingEntry:
        """将符号接地到感知体验。

        Args:
            symbol: 符号名称
            modality: 模态
            perceptual_vector: 感知向量

        Returns:
            GroundingEntry
        """
        vec = np.atleast_1d(np.asarray(perceptual_vector, dtype=float))

        if symbol in self._groundings:
            entry = self._groundings[symbol]
            # 增量更新: 移动平均
            old_vec = entry.perceptual_vector
            if old_vec is not None and len(old_vec) == len(vec):
                alpha = 1.0 / (entry.examples + 1)
                entry.perceptual_vector = old_vec * (1 - alpha) + vec * alpha
            else:
                entry.perceptual_vector = vec
            entry.examples += 1
            entry.grounding_strength = min(1.0, entry.examples / 5.0)
        else:
            entry = GroundingEntry(
                symbol=symbol,
                modality=modality,
                perceptual_vector=vec,
                grounding_strength=0.2,
                examples=1,
            )
            self._groundings[symbol] = entry

        logger.info("符号接地: %s → %s (强度=%.2f, 样例=%d)",
                     symbol, modality, entry.grounding_strength, entry.examples)
        return entry

    def verify_grounding(
        self, symbol: str, test_vector: np.ndarray
    ) -> float:
        """验证符号接地——测试向量与接地向量的相似度。

        Args:
            symbol: 符号名称
            test_vector: 测试感知向量

        Returns:
            相似度 [0, 1]
        """
        if symbol not in self._groundings:
            return 0.0

        entry = self._groundings[symbol]
        if entry.perceptual_vector is None:
            return 0.0

        test = np.atleast_1d(np.asarray(test_vector, dtype=float))
        grounded = entry.perceptual_vector

        min_dim = min(len(test), len(grounded))
        if min_dim == 0:
            return 0.0

        cos_sim = self._cosine_similarity(test[:min_dim], grounded[:min_dim])
        return float(max(0.0, cos_sim))

    def is_grounded(self, symbol: str) -> bool:
        """检查符号是否已接地。"""
        if symbol not in self._groundings:
            return False
        return self._groundings[symbol].grounding_strength >= self._sim_threshold

    def get_ungrounded_symbols(self, symbols: list[str]) -> list[str]:
        """从符号列表中找出未接地的符号。"""
        return [s for s in symbols if not self.is_grounded(s)]

    def get_grounding(self, symbol: str) -> GroundingEntry | None:
        """获取符号的接地信息。"""
        return self._groundings.get(symbol)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def statistics(self) -> dict[str, Any]:
        return {
            "grounded_symbols": self.grounded_symbol_count,
            "symbols": list(self._groundings.keys()),
            "avg_grounding_strength": (
                float(np.mean([e.grounding_strength for e in self._groundings.values()]))
                if self._groundings else 0.0
            ),
        }
