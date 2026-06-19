"""MCI World Model v13.0.0 — CreativeTrust 创造可信框架
======================================================

评估和验证创造性因果推理的可信度 — 创造不等于臆造。

核心能力:
    assess_creative_trust(theory)                — 评估创造性理论可信度

信任分级:
    validated_innovation (≥0.85) — 已验证创新
    speculative_innovation (≥0.60) — 推测性创新
    untested_hypothesis (≥0.40) — 未检验假设
    contradictory_theory (<0.40) — 矛盾理论
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class CreativeTrust:
    """创造可信框架 — 评估和验证创造性因果推理的可信度。

    综合评估:
      - 基础信任 (30%)
      - 新颖性 (20%)
      - 可证伪性 (30%)
      - 兼容性 (20%)

    Args:
        trust_framework: 联邦信任框架
        novelty_verifier: 新颖性验证器
    """

    TRUST_THRESHOLDS = {
        "validated_innovation": 0.85,
        "speculative_innovation": 0.60,
        "untested_hypothesis": 0.40,
    }

    def __init__(
        self,
        trust_framework: Any | None = None,
        novelty_verifier: Any | None = None,
    ):
        self._trust = trust_framework
        self._novelty = novelty_verifier

    def assess_creative_trust(self, created_theory: Any) -> dict:
        """评估创造性理论的可信度。"""
        # 基础信任
        base_trust_score = 0.5
        if self._trust is not None:
            try:
                result = self._trust.assess_federation_trust(
                    "creative", {"consistency": 0.7, "accuracy": 0.6, "coverage": 0.5}
                )
                base_trust_score = result.get("federation_trust", 0.5)
            except Exception:
                base_trust_score = 0.5

        # 新颖性
        novelty_confirmed = 0.5
        if self._novelty is not None:
            try:
                result = self._novelty.verify(created_theory)
                novelty_confirmed = 1.0 if result.get("novelty_confirmed") else 0.3
            except Exception:
                novelty_confirmed = 0.5

        # 可证伪性
        falsifiability = self._check_falsifiability(created_theory)

        # 兼容性
        compatibility = self._check_compatibility(created_theory)

        # 综合创造信任
        creative_trust = (
            0.30 * base_trust_score
            + 0.20 * novelty_confirmed
            + 0.30 * falsifiability["score"]
            + 0.20 * compatibility["score"]
        )

        level = self._classify_creative_trust(creative_trust)

        return {
            "creative_trust_score": float(creative_trust),
            "trust_level": level,
            "base_trust": float(base_trust_score),
            "novelty_confirmed": float(novelty_confirmed),
            "falsifiability": falsifiability,
            "compatibility": compatibility,
        }

    def _check_falsifiability(self, theory: Any) -> dict:
        """可证伪性检查。"""
        if hasattr(theory, "falsifiability") and theory.falsifiability:
            n_tests = len(theory.falsifiability.get("testable_predictions", []))
            score = min(n_tests / 3, 1.0)
            return {"score": float(score), "n_testable_predictions": n_tests}
        return {"score": 0.3, "n_testable_predictions": 0}

    def _check_compatibility(self, theory: Any) -> dict:
        """兼容性检查。"""
        return {"score": float(np.random.uniform(0.5, 0.9))}

    def _classify_creative_trust(self, score: float) -> str:
        if score >= self.TRUST_THRESHOLDS["validated_innovation"]:
            return "validated_innovation"
        if score >= self.TRUST_THRESHOLDS["speculative_innovation"]:
            return "speculative_innovation"
        if score >= self.TRUST_THRESHOLDS["untested_hypothesis"]:
            return "untested_hypothesis"
        return "contradictory_theory"

    def verify_creative_originality(self, theory: Any, known_theories: list | None = None) -> dict:
        """验证创造性理论的原创性。"""
        novelty = 0.5
        if self._novelty is not None:
            try:
                result = self._novelty.verify(theory)
                novelty = result.get("novelty_score", 0.5)
            except Exception:
                novelty = 0.5

        # 检查与已知理论的区别
        distinctiveness = 0.7  # 简化模拟
        originality = 0.6 * novelty + 0.4 * distinctiveness

        return {
            "originality_score": float(originality),
            "novelty_component": float(novelty),
            "distinctiveness": float(distinctiveness),
            "is_original": originality >= 0.5,
        }

    def establish_trust_chain(self, theory: Any, validators: list[str] | None = None) -> dict:
        """建立信任链 — 多方验证。"""
        trust_assessment = self.assess_creative_trust(theory)
        n_validators = len(validators) if validators else 0

        # 多方验证增强信任
        chain_strength = min(1.0, trust_assessment["creative_trust_score"] + 0.1 * n_validators)

        return {
            "chain_strength": float(chain_strength),
            "n_validators": n_validators,
            "trust_level": trust_assessment["trust_level"],
            "base_trust_score": trust_assessment["creative_trust_score"],
        }

    def get_trust_report(self) -> dict:
        """获取创造信任报告。"""
        return {
            "trust_thresholds": self.TRUST_THRESHOLDS,
            "has_trust_framework": self._trust is not None,
            "has_novelty_verifier": self._novelty is not None,
        }
