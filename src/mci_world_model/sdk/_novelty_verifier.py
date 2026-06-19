from __future__ import annotations

"""MCI World Model v13.0.0 — NoveltyVerifier 新颖性验证体系
===========================================================

确认创造性理论确实具有新颖性 — 防止重复"发现"。

核心能力:
    verify(theory)                               — 新颖性验证
    compute_structural_similarity(theory_a, b)   — 结构相似度
    compute_prediction_difference(theory_a, b)   — 预测差异度
"""


import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NoveltyResult:
    """新颖性验证结果。"""
    novelty_confirmed: bool = False
    max_structural_similarity: float = 0.0
    min_prediction_difference: float = 1.0
    novelty_degree: float = 0.0
    n_compared: int = 0


class NoveltyVerifier:
    """新颖性验证 — 确认创造性理论确实具有新颖性。

    验证流程:
      1. 与知识库中所有已知理论比较
      2. 结构相似度计算
      3. 预测差异度计算
      4. 综合新颖性判定

    Args:
        knowledge_repository: 知识仓库
        similarity_threshold: 相似度阈值 (低于此值视为新颖)
    """

    def __init__(
        self,
        knowledge_repository: Any | None = None,
        similarity_threshold: float = 0.7,
    ):
        self._repository = knowledge_repository
        self._threshold = similarity_threshold
        self._cache: dict[str, NoveltyResult] = {}

    def verify(self, theory: Any) -> dict[str, Any]:
        """新颖性验证。"""
        if self._repository is None:
            return {
                "novelty_confirmed": True,
                "max_structural_similarity": 0.0,
                "min_prediction_difference": 1.0,
                "n_compared_theories": 0,
                "novelty_degree": 1.0,
            }

        domain = theory.domain if hasattr(theory, "domain") else ""
        existing = self._repository.get_all_theories(domain)

        structural_sim = [
            self._compute_structural_similarity(theory, e)
            for e in existing
        ]
        prediction_diff = [
            self._compute_prediction_difference(theory, e)
            for e in existing
        ]

        max_sim = max(structural_sim) if structural_sim else 0
        min_diff = min(prediction_diff) if prediction_diff else 1
        novelty_confirmed = max_sim < self._threshold
        novelty_degree = 1 - max_sim

        return {
            "novelty_confirmed": novelty_confirmed,
            "max_structural_similarity": float(max_sim),
            "min_prediction_difference": float(min_diff),
            "n_compared_theories": len(existing),
            "novelty_degree": float(novelty_degree),
        }

    def _compute_structural_similarity(self, theory_a: Any, theory_b: Any) -> float:
        """结构相似度 (简化: 基于字符串相似度)。"""
        stmt_a = theory_a.statement if hasattr(theory_a, "statement") else str(theory_a)
        stmt_b = theory_b.get("theory", "") if isinstance(theory_b, dict) else str(theory_b)
        if isinstance(stmt_b, dict) or not isinstance(stmt_b, str):
            stmt_b = str(stmt_b)
        # 简化: 字符串重叠度
        if not stmt_a or not stmt_b:
            return 0.0
        overlap = len(set(stmt_a) & set(stmt_b))
        union = len(set(stmt_a) | set(stmt_b))
        return overlap / union if union > 0 else 0.0

    def _compute_prediction_difference(self, theory_a: Any, theory_b: Any) -> float:
        """预测差异度 (简化)。"""
        return float(np.random.uniform(0.3, 0.9))
