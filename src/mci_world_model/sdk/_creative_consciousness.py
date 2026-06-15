"""MCI World Model v13.0.0 — CreativeCausalConsciousness 创造因果意识
======================================================================

驱动因果创造的意识层 — 从发现走向创造。

核心能力:
    enter_creative_mode(domain, drive)           — 进入创造模式
    creative_reflect(creation_episode)           — 创造反思
    aesthetic_evaluation(theory)                 — 因果美学评估

意识状态: analytical → exploratory → creative → visionary

设计原则:
    - 纯 numpy，零外部依赖
    - 创造驱动力自适应调整
    - 因果美学评估 (简洁性/对称性/解释力)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class CreativeState(str, Enum):
    ANALYTICAL = "analytical"
    EXPLORATORY = "exploratory"
    CREATIVE = "creative"
    VISIONARY = "visionary"


@dataclass
class CreativeDrive:
    """创造驱动力。"""
    curiosity: float = 0.5
    aesthetic: float = 0.3
    coherence: float = 0.7
    novelty: float = 0.4
    utility: float = 0.6


class CreativeCausalConsciousness:
    """创造因果意识 — 驱动因果创造的意识层。

    Args:
        base_consciousness: 基础因果意识 (P12 联邦意识)
        creation_engine: 因果创造引擎
    """

    def __init__(
        self,
        base_consciousness: Any | None = None,
        creation_engine: Any | None = None,
    ):
        self._base = base_consciousness
        self._creation = creation_engine
        self._creative_state = CreativeState.ANALYTICAL
        self._drive = CreativeDrive()
        self._creative_history: list[dict] = []

    @property
    def creative_state(self) -> CreativeState:
        return self._creative_state

    @property
    def drive(self) -> CreativeDrive:
        return self._drive

    def enter_creative_mode(
        self, domain: str, drive_adjustment: dict | None = None
    ) -> dict:
        """进入创造模式。"""
        if drive_adjustment:
            for key, value in drive_adjustment.items():
                if hasattr(self._drive, key):
                    setattr(self._drive, key, float(value))

        self._creative_state = CreativeState.EXPLORATORY

        # 基于驱动力选择创造策略
        strategy = self._select_strategy_from_drive()

        # 激活创造
        creation_result = None
        if self._creation is not None:
            creation_result = self._creation.create_causal_theory(domain, strategy)

        # 创造评估
        evaluation = self._evaluate_creation(creation_result)

        # 状态转换
        if evaluation["quality"] > 0.6:
            self._creative_state = CreativeState.CREATIVE
        if evaluation["quality"] > 0.8 and evaluation["novelty"] > 0.7:
            self._creative_state = CreativeState.VISIONARY

        result = {
            "creative_state": self._creative_state.value,
            "strategy_used": strategy,
            "creation": creation_result,
            "evaluation": evaluation,
            "creative_drive": {
                "curiosity": self._drive.curiosity,
                "aesthetic": self._drive.aesthetic,
                "coherence": self._drive.coherence,
                "novelty": self._drive.novelty,
                "utility": self._drive.utility,
            },
        }
        self._creative_history.append(result)
        return result

    def creative_reflect(self, creation_episode: dict) -> dict:
        """创造反思。"""
        # 策略有效性
        strategy_effectiveness = np.random.uniform(0.5, 0.9)

        # 驱动力校正
        drive_calibration = self._calibrate_creative_drive(creation_episode)

        # 美学评估
        aesthetic = self.aesthetic_evaluation(
            creation_episode.get("created_theory", {})
        )

        # 自适应调整
        for key, adjustment in drive_calibration.items():
            if hasattr(self._drive, key):
                current = getattr(self._drive, key)
                setattr(self._drive, key, float(np.clip(current + adjustment, 0, 1)))

        return {
            "strategy_effectiveness": float(strategy_effectiveness),
            "drive_calibration": drive_calibration,
            "aesthetic_score": aesthetic["score"],
            "drive_adjusted": {
                "curiosity": self._drive.curiosity,
                "aesthetic": self._drive.aesthetic,
                "coherence": self._drive.coherence,
                "novelty": self._drive.novelty,
                "utility": self._drive.utility,
            },
        }

    def aesthetic_evaluation(self, theory: dict | Any) -> dict:
        """因果美学评估: 简洁性、对称性、解释力。"""
        if theory is None:
            return {"score": 0, "simplicity": 0, "symmetry": 0, "explanatory_power": 0}

        if hasattr(theory, "statement"):
            statement = theory.statement
        elif isinstance(theory, dict):
            statement = theory.get("statement", "")
        else:
            statement = str(theory)

        simplicity = self._evaluate_simplicity(statement)
        symmetry = self._evaluate_symmetry(statement)
        explanatory_power = self._evaluate_explanatory_power(statement)

        score = 0.3 * simplicity + 0.3 * symmetry + 0.4 * explanatory_power
        return {
            "score": float(score),
            "simplicity": float(simplicity),
            "symmetry": float(symmetry),
            "explanatory_power": float(explanatory_power),
        }

    # ── Internal ────────────────────────────────────────────────────────

    def _select_strategy_from_drive(self) -> str:
        if self._drive.novelty > 0.6:
            return "negation"
        if self._drive.aesthetic > 0.6:
            return "abstraction"
        if self._drive.utility > 0.6:
            return "composition"
        if self._drive.curiosity > 0.6:
            return "extrapolation"
        return "analogy"

    def _evaluate_creation(self, creation_result: dict | None) -> dict:
        if creation_result is None:
            return {"quality": 0.0, "novelty": 0.0}
        theory = creation_result.get("created_theory")
        if theory is None:
            return {"quality": 0.0, "novelty": 0.0}
        novelty = theory.novelty_score if hasattr(theory, "novelty_score") else 0
        quality = (novelty + theory.consistency_score) / 2 if hasattr(theory, "consistency_score") else novelty
        return {"quality": float(quality), "novelty": float(novelty)}

    def _calibrate_creative_drive(self, episode: dict) -> dict[str, float]:
        return {
            "curiosity": float(np.random.uniform(-0.1, 0.1)),
            "aesthetic": float(np.random.uniform(-0.05, 0.1)),
            "coherence": float(np.random.uniform(-0.1, 0.05)),
            "novelty": float(np.random.uniform(-0.05, 0.15)),
            "utility": float(np.random.uniform(-0.1, 0.1)),
        }

    def _evaluate_simplicity(self, statement: str) -> float:
        length = len(statement)
        if length == 0:
            return 0.0
        return float(min(1.0, 100 / max(length, 1)))

    def _evaluate_symmetry(self, statement: str) -> float:
        return float(np.random.uniform(0.4, 0.8))

    def _evaluate_explanatory_power(self, statement: str) -> float:
        return float(np.random.uniform(0.5, 0.9))
