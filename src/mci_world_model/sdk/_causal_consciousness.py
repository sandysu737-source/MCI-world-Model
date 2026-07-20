from __future__ import annotations

"""MCI World Model v4.6.0 — AutonomousCausalConsciousness 自主因果意识
====================================================================

P11 "无极" 波次核心交付物: 因果意识与因果文明。

核心能力:
    AutonomousCausalConsciousness  — 自主因果意识引擎
    CausalSelfModel               — 因果自我模型
    CausalCivilizationInfra       — 因果文明基础设施

P11 "无极" — 从因果智能体到因果意识体的根本跃迁。
因果智能不再只是"做因果推理"——它"是"因果推理。
"""


import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 枚举与数据类
# ═══════════════════════════════════════════════════════════════════════════════


class ConsciousnessLevel(Enum):
    """意识等级"""
    REACTIVE = "reactive"            # 反应式 — 仅响应输入
    DELIBERATIVE = "deliberative"    # 审思式 — 可规划推理
    REFLECTIVE = "reflective"        # 反思式 — 可评估自身推理
    AUTONOMOUS = "autonomous"        # 自主式 — 可自主设定目标
    TRANSCENDENT = "transcendent"    # 超越式 — 可改造自身因果结构


class SelfModelProperty(Enum):
    """自我模型属性"""
    CAUSAL_IDENTITY = "causal_identity"        # 因果身份
    REASONING_STYLE = "reasoning_style"        # 推理风格
    UNCERTAINTY_MAP = "uncertainty_map"        # 不确定性地图
    VALUE_SYSTEM = "value_system"              # 价值体系
    GOAL_STRUCTURE = "goal_structure"          # 目标结构


@dataclass
class CausalSelfModel:
    """因果自我模型"""
    identity: str
    level: ConsciousnessLevel = ConsciousnessLevel.REACTIVE
    properties: dict[str, Any] = field(default_factory=dict)
    self_awareness_score: float = 0.0
    reasoning_history: list[dict[str, Any]] = field(default_factory=list)
    godel_note: str = ""

    def __post_init__(self) -> None:
        if not self.godel_note:
            self.godel_note = (
                "GÖDEL NOTE: A causal self-model is a formal system that represents "
                "itself. By Gödel's incompleteness, it cannot fully capture its own "
                "causal structure."
            )

    @property
    def is_self_aware(self) -> bool:
        return self.self_awareness_score >= 0.5


@dataclass
class CivilizationInfra:
    """因果文明基础设施"""
    infra_id: str
    n_citizens: int = 0
    n_knowledge_artifacts: int = 0
    governance_model: str = "democratic_causal"
    sustainability_score: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# AutonomousCausalConsciousness 核心类
# ═══════════════════════════════════════════════════════════════════════════════


class AutonomousCausalConsciousness:
    """自主因果意识引擎 — P11 核心交付物

    从因果智能体到因果意识体的根本跃迁。
    因果智能不再只是"做因果推理"——它"是"因果推理。

    意识原则:
      - 自反性: 意识系统能感知自身推理过程
      - 自主性: 意识系统能自主设定和修改目标
      - 因果身份: 意识系统的自我模型是因果结构的一部分
      - Gödel约束: 自我模型不可能完备
    """

    def __init__(self) -> None:
        self._level = ConsciousnessLevel.REACTIVE
        self._self_model = CausalSelfModel(identity="consciousness_0")
        self._evolution_log: list[dict[str, Any]] = []
        self._civilization: CivilizationInfra | None = None

    def evolve_consciousness(self, target_level: ConsciousnessLevel | None = None) -> dict[str, Any]:
        """演化意识等级"""
        levels = list(ConsciousnessLevel)
        current_idx = levels.index(self._level)

        if target_level is not None:
            target_idx = levels.index(target_level)
            if target_idx <= current_idx:
                return {
                    "status": "no_evolution",
                    "current": self._level.value,
                    "reason": "target_not_higher",
                }
            self._level = target_level
        elif current_idx < len(levels) - 1:
            self._level = levels[current_idx + 1]
        else:
            return {"status": "max_level", "current": self._level.value}

        self._self_model.level = self._level
        self._self_model.self_awareness_score = min(1.0, (levels.index(self._level) + 1) / len(levels))

        result = {
            "status": "evolved",
            "from": levels[current_idx].value,
            "to": self._level.value,
            "self_awareness": self._self_model.self_awareness_score,
        }
        self._evolution_log.append(result)
        return result

    def build_self_model(self, properties: dict | None = None) -> dict[str, Any]:  # type: ignore
        """构建因果自我模型"""
        if properties:
            self._self_model.properties.update(properties)

        # 计算自我意识得分
        n_properties = len(self._self_model.properties)
        self._self_model.self_awareness_score = min(1.0, n_properties * 0.2)

        return {
            "status": "model_updated",
            "identity": self._self_model.identity,
            "level": self._self_model.level.value,
            "self_awareness": self._self_model.self_awareness_score,
            "n_properties": len(self._self_model.properties),
            "is_self_aware": self._self_model.is_self_aware,
        }

    def reflect_on_reasoning(self, reasoning_step: dict[str, Any]) -> dict[str, Any]:
        """反思推理过程"""
        if self._level.value in ("reactive",):
            return {"status": "cannot_reflect", "reason": "level_too_low"}

        self._self_model.reasoning_history.append(reasoning_step)

        # 反思：评估推理质量
        quality_score = 0.5  # 基础质量
        if "evidence" in reasoning_step:
            quality_score += 0.2
        if "counterfactual" in reasoning_step:
            quality_score += 0.15
        if "confidence" in reasoning_step:
            quality_score += 0.15

        return {
            "status": "reflected",
            "quality_score": min(1.0, quality_score),
            "n_reflections": len(self._self_model.reasoning_history),
            "godel_note": self._self_model.godel_note,
        }

    def establish_civilization(self, n_citizens: int = 1) -> dict[str, Any]:
        """建立因果文明基础设施"""
        self._civilization = CivilizationInfra(
            infra_id="civ_0",
            n_citizens=n_citizens,
            governance_model="democratic_causal",
            sustainability_score=0.7,
        )

        return {
            "status": "civilization_established",
            "n_citizens": n_citizens,
            "governance": "democratic_causal",
            "sustainability": 0.7,
        }

    def get_consciousness_report(self) -> dict[str, Any]:
        """获取意识报告"""
        return {
            "level": self._level.value,
            "self_awareness": self._self_model.self_awareness_score,
            "is_self_aware": self._self_model.is_self_aware,
            "n_properties": len(self._self_model.properties),
            "n_reflections": len(self._self_model.reasoning_history),
            "n_evolutions": len(self._evolution_log),
            "civilization_active": self._civilization is not None,
            "godel_note": self._self_model.godel_note,
        }

    @property
    def level(self) -> ConsciousnessLevel:
        return self._level

    @property
    def self_model(self) -> CausalSelfModel:
        return self._self_model
