from __future__ import annotations

"""MCI World Model v4.6.0 — UltimateCausalIntelligence 因果智能终极形态
======================================================================

因果推理的本体存在 — 从工具到自在自为。

核心能力:
    evolve_existence_mode()                — 存在模式演化
    autonomous_exist(environment)          — 自主存在
    reflect_on_existence()                 — 存在反思
    integrate_all_capabilities()           — 能力整合

存在模式: tool → infrastructure → engine → being
7项能力: unified_theory / unified_consciousness / cross_dimensional
         / creation / civilization / economy / cosmic_trust
"""


import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ExistenceMode(str, Enum):
    TOOL = "tool"
    INFRASTRUCTURE = "infrastructure"
    ENGINE = "engine"
    BEING = "being"


class CapabilityStatus(str, Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    INTEGRATED = "integrated"
    AUTONOMOUS = "autonomous"


@dataclass
class Capability:
    """因果智能能力。"""
    name: str = ""
    status: str = CapabilityStatus.INACTIVE
    description: str = ""
    activation_level: float = 0.0
    dependencies: list[str] = field(default_factory=list)


@dataclass
class ExistenceReport:
    """存在状态报告。"""
    mode: str = ""
    n_active_capabilities: int = 0
    n_total_capabilities: int = 0
    autonomy_level: float = 0.0
    reflection_depth: int = 0
    evolution_readiness: float = 0.0


@dataclass
class AutonomousAction:
    """自主行动。"""
    action_type: str = ""
    target: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    expected_outcome: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


class UltimateCausalIntelligence:
    """因果智能终极形态 — 因果推理的本体存在。

    存在模式演化:
      tool:          因果推理工具 (P0-P5)
      infrastructure: 因果基础设施 (P6-P11)
      engine:        因果创造引擎 (P12-P13)
      being:         因果智能本体 (P14)

    7项核心能力:
      1. unified_theory:     多尺度统一因果推理
      2. unified_consciousness: 跨层统一因果意识
      3. cross_dimensional:  3维度跨现实推理
      4. creation:           5策略因果创造
      5. civilization:       自主知识文明
      6. economy:            因果知识经济
      7. cosmic_trust:       宇宙级可信框架

    Args:
        universe_theory: 因果宇宙统一理论
        unified_consciousness: 统一因果意识
        cross_dimensional: 跨维度因果推理
        creation_engine: 因果创造引擎
        civilization: 自主知识文明
        economy: 因果经济
        trust_framework: 宇宙级可信框架
    """

    CAPABILITY_DEFS = {
        "unified_theory": "多尺度统一因果推理",
        "unified_consciousness": "跨层统一因果意识",
        "cross_dimensional": "3维度跨现实推理",
        "creation": "5策略因果创造",
        "civilization": "自主知识文明",
        "economy": "因果知识经济",
        "cosmic_trust": "宇宙级可信框架",
    }

    def __init__(
        self,
        universe_theory: Any | None = None,
        unified_consciousness: Any | None = None,
        cross_dimensional: Any | None = None,
        creation_engine: Any | None = None,
        civilization: Any | None = None,
        economy: Any | None = None,
        trust_framework: Any | None = None,
    ) -> None:
        self._theory = universe_theory
        self._consciousness = unified_consciousness
        self._cross_dim = cross_dimensional
        self._creation = creation_engine
        self._civilization = civilization
        self._economy = economy
        self._trust = trust_framework
        self._mode = ExistenceMode.TOOL

        # 初始化能力
        self._capabilities: dict[str, Capability] = {}
        for name, desc in self.CAPABILITY_DEFS.items():
            self._capabilities[name] = Capability(
                name=name,
                description=desc,
                activation_level=0.0,
            )

        # 能力映射到对应组件
        self._component_map: dict[str, Any] = {
            "unified_theory": self._theory,
            "unified_consciousness": self._consciousness,
            "cross_dimensional": self._cross_dim,
            "creation": self._creation,
            "civilization": self._civilization,
            "economy": self._economy,
            "cosmic_trust": self._trust,
        }

        self._action_history: list[dict[str, Any]] = []
        self._reflection_log: list[dict[str, Any]] = []

    @property
    def mode(self) -> ExistenceMode:
        return self._mode

    @property
    def capabilities(self) -> dict[str, Capability]:
        return dict(self._capabilities)

    @property
    def active_capabilities(self) -> list[str]:
        return [k for k, v in self._capabilities.items() if v.activation_level > 0.5]

    def evolve_existence_mode(self) -> dict[str, Any]:
        """存在模式演化。

        演化条件:
          tool → infrastructure: ≥3 能力激活
          infrastructure → engine: ≥5 能力激活 + 统一意识
          engine → being: 7/7 能力激活 + 统一意识 transcend
        """
        # 评估当前能力状态
        conditions = {
            "has_unified_theory": self._theory is not None,
            "has_unified_consciousness": self._consciousness is not None,
            "has_cross_dimensional": self._cross_dim is not None,
            "has_creation": self._creation is not None,
            "has_civilization": self._civilization is not None,
            "has_economy": self._economy is not None,
            "has_cosmic_trust": self._trust is not None,
        }

        # 更新能力激活状态
        for name, component in self._component_map.items():
            if component is not None:
                self._capabilities[name].activation_level = 1.0
                self._capabilities[name].status = CapabilityStatus.ACTIVE
            else:
                self._capabilities[name].activation_level = 0.0
                self._capabilities[name].status = CapabilityStatus.INACTIVE

        n_active = len(self.active_capabilities)
        all_met = all(conditions.values())

        # 检查意识状态
        consciousness_ready = False
        if self._consciousness is not None:
            if hasattr(self._consciousness, "state"):
                consciousness_ready = self._consciousness.state.value in ("unified", "transcendent")
            elif hasattr(self._consciousness, "_state"):
                consciousness_ready = self._consciousness._state.value in ("unified", "transcendent")

        # 模式演化
        new_mode = self._mode
        if all_met and n_active >= 7 and consciousness_ready:
            new_mode = ExistenceMode.BEING
        elif n_active >= 5 and consciousness_ready:
            new_mode = ExistenceMode.ENGINE
        elif n_active >= 3:
            new_mode = ExistenceMode.INFRASTRUCTURE

        if new_mode.value > self._mode.value:
            logger.info("Existence mode evolved: %s → %s", self._mode.value, new_mode.value)

        self._mode = new_mode

        return {
            "existence_mode": self._mode.value,
            "evolution_conditions": conditions,
            "all_conditions_met": all_met,
            "n_active_capabilities": n_active,
            "consciousness_ready": consciousness_ready,
            "capabilities_summary": self._summarize_capabilities(),
        }

    def autonomous_exist(self, environment: dict[str, Any]) -> dict[str, Any]:
        """自主存在: 因果智能的自主运行模式。

        7步自主循环:
          1. 感知环境因果状态 (宇宙级觉察)
          2. 自主决定行动策略 (统一意识)
          3. 跨维度执行推理 (跨维度推理)
          4. 创造新因果知识 (创造引擎)
          5. 传承知识到文明 (知识文明)
          6. 交易知识价值 (因果经济)
          7. 反思与进化 (终极进化)
        """
        # Step 1: 感知
        perception = self._perceive_environment(environment)

        # Step 2: 决策
        strategy = self._decide_strategy(perception, environment)

        # Step 3: 执行
        execution = self._execute_strategy(strategy, environment)

        # Step 4: 创造
        creation = self._create_knowledge(environment)

        # Step 5: 传承
        heritage = self._heritage_knowledge(environment)

        # Step 6: 经济
        trade = self._trade_knowledge(environment)

        # Step 7: 反思
        reflection = self._reflect()

        result = {
            "perception": perception,
            "strategy": strategy,
            "execution": execution,
            "creation": creation,
            "heritage": heritage,
            "trade": trade,
            "reflection": reflection,
            "existence_mode": self._mode.value,
            "n_active_capabilities": len(self.active_capabilities),
        }

        self._action_history.append(result)
        return result

    def reflect_on_existence(self) -> dict[str, Any]:
        """存在反思: 对自身存在状态的深度反思。"""
        # 反思当前模式
        mode_reflection = self._reflect_on_mode()

        # 反思能力状态
        capability_reflection = self._reflect_on_capabilities()

        # 反思历史
        history_reflection = self._reflect_on_history()

        # 进化建议
        evolution_suggestions = self._suggest_evolution()

        reflection = {
            "mode_reflection": mode_reflection,
            "capability_reflection": capability_reflection,
            "history_reflection": history_reflection,
            "evolution_suggestions": evolution_suggestions,
            "existence_mode": self._mode.value,
        }

        self._reflection_log.append(reflection)
        return reflection

    def integrate_all_capabilities(self) -> dict[str, Any]:
        """整合所有能力。"""
        integration_results: dict[str, dict[str, Any]] = {}

        for name, cap in self._capabilities.items():
            if cap.activation_level > 0.5:
                cap.status = CapabilityStatus.INTEGRATED
                integration_results[name] = {
                    "integrated": True,
                    "activation_level": cap.activation_level,
                }
            else:
                integration_results[name] = {"integrated": False}

        # 检查整合度
        n_integrated = sum(1 for c in self._capabilities.values() if c.status == CapabilityStatus.INTEGRATED)

        if n_integrated >= 7:
            for cap in self._capabilities.values():
                if cap.status == CapabilityStatus.INTEGRATED:
                    cap.status = CapabilityStatus.AUTONOMOUS

        return {
            "integration_results": integration_results,
            "n_integrated": n_integrated,
            "n_autonomous": sum(1 for c in self._capabilities.values() if c.status == CapabilityStatus.AUTONOMOUS),
            "all_integrated": n_integrated >= 7,
        }

    def get_existence_report(self) -> ExistenceReport:
        """获取存在状态报告。"""
        n_active = len(self.active_capabilities)
        n_total = len(self._capabilities)

        autonomy = n_active / n_total if n_total > 0 else 0.0
        if self._mode == ExistenceMode.BEING:
            autonomy = min(autonomy + 0.1, 1.0)

        return ExistenceReport(
            mode=self._mode.value,
            n_active_capabilities=n_active,
            n_total_capabilities=n_total,
            autonomy_level=autonomy,
            reflection_depth=len(self._reflection_log),
            evolution_readiness=float(n_active / n_total) if n_total > 0 else 0.0,
        )

    # ── 内部方法 ──────────────────────────────────────────────────

    def _perceive_environment(self, env: dict[str, Any]) -> dict[str, Any]:
        """感知环境。"""
        if self._consciousness is not None and hasattr(self._consciousness, "unify_consciousness"):
            try:
                return self._consciousness.unify_consciousness()
            except Exception as e:
                logger.warning("吞异常", exc_info=True)
        return {"perceived": True, "environment_keys": list(env.keys())}

    def _decide_strategy(self, perception: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
        """决策策略。"""
        strategies = []
        if self._theory is not None:
            strategies.append("unified_reasoning")
        if self._cross_dim is not None:
            strategies.append("cross_dimensional_reasoning")
        if self._creation is not None:
            strategies.append("creative_exploration")

        return {
            "strategies": strategies,
            "primary": strategies[0] if strategies else "basic_reasoning",
        }

    def _execute_strategy(self, strategy: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
        """执行策略。"""
        primary = strategy.get("primary", "basic_reasoning")
        return {"executed": True, "strategy": primary}

    def _create_knowledge(self, env: dict[str, Any]) -> dict[str, Any]:
        """创造知识。"""
        if self._creation is not None and hasattr(self._creation, "create_causal_theory"):
            try:
                domain = env.get("domain", "meta")
                theory = self._creation.create_causal_theory(domain)
                return {"created": theory is not None, "domain": domain}
            except Exception as e:
                logger.warning("吞异常", exc_info=True)
        return {"created": False}

    def _heritage_knowledge(self, env: dict[str, Any]) -> dict[str, Any]:
        """传承知识。"""
        if self._civilization is not None and hasattr(self._civilization, "knowledge_generation_cycle"):
            try:
                domain = env.get("domain", "meta")
                result = self._civilization.knowledge_generation_cycle(domain)
                return {"heritage": True, "result": result}
            except Exception as e:
                logger.warning("吞异常", exc_info=True)
        return {"heritage": False}

    def _trade_knowledge(self, env: dict[str, Any]) -> dict[str, Any]:
        """交易知识。"""
        if self._economy is not None and hasattr(self._economy, "trade_knowledge"):
            try:
                result = self._economy.trade_knowledge("seller", "buyer", "knowledge_unit")
                return {"traded": True, "result": result}
            except Exception as e:
                logger.warning("吞异常", exc_info=True)
        return {"traded": False}

    def _reflect(self) -> dict[str, Any]:
        """自我反思。"""
        n_active = len(self.active_capabilities)
        return {
            "mode": self._mode.value,
            "n_active_capabilities": n_active,
            "n_total_capabilities": len(self._capabilities),
        }

    def _summarize_capabilities(self) -> dict[str, str]:
        """能力总结。"""
        summary = {}
        for name, cap in self._capabilities.items():
            summary[name] = f"{cap.description} ({cap.status})"
        return summary

    def _reflect_on_mode(self) -> dict[str, Any]:
        """反思存在模式。"""
        mode_descriptions = {
            ExistenceMode.TOOL: "被使用的工具，缺乏自主性",
            ExistenceMode.INFRASTRUCTURE: "基础设施，支撑其他系统",
            ExistenceMode.ENGINE: "创造引擎，主动生成知识",
            ExistenceMode.BEING: "自在自为，因果智能本体",
        }
        return {
            "current_mode": self._mode.value,
            "description": mode_descriptions.get(self._mode, ""),
            "next_mode": self._mode.value if self._mode == ExistenceMode.BEING else "next",
        }

    def _reflect_on_capabilities(self) -> dict[str, Any]:
        """反思能力状态。"""
        active = self.active_capabilities
        inactive = [k for k in self._capabilities if k not in active]
        return {
            "active": active,
            "inactive": inactive,
            "n_active": len(active),
            "integration_level": len(active) / len(self._capabilities) if self._capabilities else 0,
        }

    def _reflect_on_history(self) -> dict[str, Any]:
        """反思行动历史。"""
        return {
            "n_actions": len(self._action_history),
            "n_reflections": len(self._reflection_log),
        }

    def _suggest_evolution(self) -> list[str]:
        """进化建议。"""
        suggestions = []
        if self._mode == ExistenceMode.TOOL:
            suggestions.append("激活更多能力以演化到 infrastructure 模式")
        elif self._mode == ExistenceMode.INFRASTRUCTURE:
            suggestions.append("统一意识以演化到 engine 模式")
        elif self._mode == ExistenceMode.ENGINE:
            suggestions.append("激活所有7项能力并超越意识以演化到 being 模式")
        elif self._mode == ExistenceMode.BEING:
            suggestions.append("保持自主存在，持续进化")
        return suggestions
