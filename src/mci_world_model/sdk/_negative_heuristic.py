"""MCI World Model — Lakatos 负面启发法 (NegativeHeuristic)

CEWM v3.7.0 新增组件 (N8)：
Lakatos 科学研究纲领方法论 — 硬核保护 + 保护带修正约束。

理论基础：
    1. Imre Lakatos "科学研究纲领方法论" (MSRP)
       - 硬核 (Hard Core): 不可放弃的基本假设
       - 保护带 (Protective Belt): 可修正的辅助假设
       - 负面启发法 (Negative Heuristic): 禁止将反驳指向硬核
       - 正面启发法 (Positive Heuristic): 指导如何修正保护带

    2. 在 CEWM 中的映射：
       硬核 = 系统不可违反的结构性质（因果性、记忆性、闭环性等）
       保护带 = 可调参数、策略选择、阈值设定
       负面启发法 = 检测提议变更是否违反硬核

核心能力：
    - violations(proposed_change) — 返回违反的硬核规则列表
    - is_admissible(proposed_change) — 判断变更是否可接受
    - protective_belt_suggestions(diagnosis) — 基于诊断建议保护带修正
    - hard_core_status() — 查看当前硬核规则状态

硬核规则 (≥5 条)：
    HC-1: 因果性不可放弃 (Causality Inviolability)
    HC-2: 记忆性不可丧失 (Memory Persistence)
    HC-3: 闭环性不可断裂 (Closed-Loop Integrity)
    HC-4: 能量守恒不可违反 (Energy Conservation)
    HC-5: 认知多样性不可坍缩 (Cognitive Diversity)
    HC-6: 时间不可逆性 (Temporal Irreversibility)
    HC-7: 可解释性不可退化 (Explainability Preservation)

Example:
    >>> nh = NegativeHeuristic()
    >>> change = ProposedChange(
    ...     description="移除因果图模块",
    ...     affected_components=["causal_graph"],
    ...     change_type="remove",
    ... )
    >>> violations = nh.violations(change)
    >>> print(len(violations))
    1
    >>> print(violations[0].rule_id)
    'HC-1'
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# =============================================================================
# 数据类型
# =============================================================================


class ChangeType(Enum):
    """变更类型枚举。"""

    ADD = "add"
    MODIFY = "modify"
    REMOVE = "remove"
    REPLACE = "replace"
    DISABLE = "disable"
    PARAMETER_TUNE = "parameter_tune"


class RuleSeverity(Enum):
    """规则严重度。"""

    ABSOLUTE = "absolute"  # 绝对不可违反
    STRONG = "strong"  # 强烈不建议
    ADVISORY = "advisory"  # 建议性


@dataclass
class ProposedChange:
    """提议的系统变更。

    Attributes:
        description: 变更描述
        affected_components: 受影响的组件列表
        change_type: 变更类型
        parameters: 变更参数
        justification: 变更理由
        source: 变更来源 (diagnosis/configurator/manual)
    """

    description: str
    affected_components: list[str] = field(default_factory=list)
    change_type: ChangeType = ChangeType.MODIFY
    parameters: dict[str, Any] = field(default_factory=dict)
    justification: str = ""
    source: str = "manual"


@dataclass
class HardCoreViolation:
    """硬核规则违反记录。

    Attributes:
        rule_id: 硬核规则 ID (HC-1 ~ HC-7)
        rule_name: 规则名称
        severity: 严重度
        description: 违反描述
        affected_component: 受影响组件
        recommendation: 修正建议
    """

    rule_id: str
    rule_name: str
    severity: RuleSeverity
    description: str
    affected_component: str = ""
    recommendation: str = ""


@dataclass
class ProtectiveBeltSuggestion:
    """保护带修正建议。

    Attributes:
        target: 修正目标组件
        action: 修正动作
        priority: 优先级 [0, 1]
        rationale: 修正理由
    """

    target: str
    action: str
    priority: float = 0.5
    rationale: str = ""


@dataclass
class NegativeHeuristicStats:
    """NegativeHeuristic 运行统计。

    Attributes:
        total_checks: 总检查次数
        total_violations: 总违反次数
        rejected_changes: 被拒绝的变更数
        accepted_changes: 被接受的变更数
        violation_history: 违反历史
    """

    total_checks: int = 0
    total_violations: int = 0
    rejected_changes: int = 0
    accepted_changes: int = 0
    violation_history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_checks": self.total_checks,
            "total_violations": self.total_violations,
            "rejected_changes": self.rejected_changes,
            "accepted_changes": self.accepted_changes,
            "violation_history_count": len(self.violation_history),
        }


# =============================================================================
# 硬核规则定义
# =============================================================================

# 因果相关组件
_CAUSAL_COMPONENTS = {
    "causal_graph",
    "causal_updater",
    "causal_edges",
    "do_calculus",
    "causal_engine",
    "category_causal",
    "causal_actor",
    "causal_mlp",
    "bayesian_network",
    "fourier_causal",
    "gaussian_dag",
}

# 记忆相关组件
_MEMORY_COMPONENTS = {
    "experience_db",
    "working_memory",
    "parametric_memory",
    "multi_view_retriever",
    "knowledge_aging",
    "memory_nodes",
    "episodic_memory",
    "semantic_memory",
}

# 闭环相关组件
_LOOP_COMPONENTS = {
    "feedback_loop",
    "prediction_error",
    "surprise_signal",
    "cognitive_loop_bus",
    "error_backprop",
    "perception_pipeline",
    "energy_bus",
    "configurator",
}

# 能量相关组件
_ENERGY_COMPONENTS = {
    "energy_core",
    "energy_bus",
    "energy_cost",
    "energy_memory",
    "energy_relations",
    "five_states",
}

# 多样性相关组件
_DIVERSITY_COMPONENTS = {
    "multi_view_retriever",
    "experience_db",
    "pattern_inference",
    "meta_cognition",
    "meta_diagnoser",
}

# 时间相关组件
_TEMPORAL_COMPONENTS = {
    "temporal_core",
    "temporal_encoding",
    "time_series",
    "trajectory",
    "working_memory",
}

# 可解释性相关组件
_EXPLAIN_COMPONENTS = {
    "causal_graph",
    "root_cause",
    "explanation",
    "meta_diagnoser",
    "diagnosis",
    "causal_updater",
}


# =============================================================================
# NegativeHeuristic 主类
# =============================================================================


class NegativeHeuristic:
    """Lakatos 负面启发法引擎。

    保护系统硬核不被修改，仅允许保护带（参数、策略、阈值）调整。

    Usage:
        >>> nh = NegativeHeuristic()
        >>> change = ProposedChange(description="调整学习率", change_type=ChangeType.PARAMETER_TUNE)
        >>> nh.is_admissible(change)
        True
    """

    def __init__(self, custom_rules: list[dict] | None = None):
        """初始化。

        Args:
            custom_rules: 自定义硬核规则列表，每项含
                rule_id, rule_name, components, severity, check_fn_name
        """
        self._stats = NegativeHeuristicStats()
        self._rules: list[dict] = self._build_default_rules()
        if custom_rules:
            self._rules.extend(custom_rules)

    @property
    def stats(self) -> NegativeHeuristicStats:
        return self._stats

    @property
    def rules(self) -> list[dict]:
        """返回当前硬核规则列表。"""
        return [dict(r) for r in self._rules]

    # -----------------------------------------------------------------
    # 核心方法
    # -----------------------------------------------------------------

    def violations(self, proposed_change: ProposedChange) -> list[HardCoreViolation]:
        """检测提议变更违反了哪些硬核规则。

        Args:
            proposed_change: 提议的系统变更

        Returns:
            违反的硬核规则列表（空 = 无违反）
        """
        self._stats.total_checks += 1
        result: list[HardCoreViolation] = []

        components = set(proposed_change.affected_components)

        for rule in self._rules:
            rule_components = set(rule["components"])
            overlap = components & rule_components

            if not overlap:
                continue

            violation = self._check_rule(rule, proposed_change, overlap)
            if violation is not None:
                result.append(violation)

        if result:
            self._stats.total_violations += len(result)
            self._stats.rejected_changes += 1
            self._stats.violation_history.append(
                {
                    "timestamp": time.time(),
                    "change": proposed_change.description,
                    "violations": [v.rule_id for v in result],
                }
            )
        else:
            self._stats.accepted_changes += 1

        return result

    def is_admissible(self, proposed_change: ProposedChange) -> bool:
        """判断变更是否可接受（无 ABSOLUTE 级别违反）。

        Args:
            proposed_change: 提议的系统变更

        Returns:
            True = 可接受, False = 被硬核规则拒绝
        """
        viols = self.violations(proposed_change)
        if not viols:
            return True
        # 只有 ABSOLUTE 级别才绝对拒绝
        return not any(v.severity == RuleSeverity.ABSOLUTE for v in viols)

    def protective_belt_suggestions(
        self,
        diagnosis: dict | None = None,
        **kwargs,
    ) -> list[ProtectiveBeltSuggestion]:
        """基于诊断结果生成保护带修正建议。

        正面启发法：指导如何在不违反硬核的前提下修正保护带。

        Args:
            diagnosis: MetaDiagnoser 诊断结果字典

        Returns:
            保护带修正建议列表
        """
        suggestions: list[ProtectiveBeltSuggestion] = []

        if diagnosis is None:
            return suggestions

        pattern = diagnosis.get("pattern", "")
        severity = diagnosis.get("severity", "low")
        diagnosis.get("root_cause_chain", [])

        # 基于失败模式生成保护带修正
        if pattern == "PERCEPTION_DRIFT":
            suggestions.append(
                ProtectiveBeltSuggestion(
                    target="perception_pipeline",
                    action="增加感知通道权重调整频率",
                    priority=0.8,
                    rationale="感知漂移需要更频繁的校准，而非替换感知模块",
                )
            )
            suggestions.append(
                ProtectiveBeltSuggestion(
                    target="attention_policy",
                    action="提升失败通道的采样权重",
                    priority=0.7,
                    rationale="通过注意力策略补偿漂移通道",
                )
            )

        elif pattern == "PREDICTION_BIAS":
            suggestions.append(
                ProtectiveBeltSuggestion(
                    target="prediction_weights",
                    action="调整预测模型权重衰减率",
                    priority=0.85,
                    rationale="预测偏差可通过参数正则化修正",
                )
            )
            suggestions.append(
                ProtectiveBeltSuggestion(
                    target="experience_db",
                    action="增加多样化训练经验",
                    priority=0.75,
                    rationale="扩展经验库覆盖度以减少偏差",
                )
            )

        elif pattern == "CAUSAL_COLLAPSE":
            suggestions.append(
                ProtectiveBeltSuggestion(
                    target="causal_updater",
                    action="降低因果边删除阈值",
                    priority=0.9,
                    rationale="保护因果图连通性，宁可保留弱边也不轻易删除",
                )
            )
            suggestions.append(
                ProtectiveBeltSuggestion(
                    target="causal_updater",
                    action="提高新证据的初始置信度",
                    priority=0.7,
                    rationale="鼓励因果探索，减少过早坍缩",
                )
            )

        elif pattern == "MEMORY_DECAY":
            suggestions.append(
                ProtectiveBeltSuggestion(
                    target="experience_db",
                    action="降低遗忘阈值",
                    priority=0.85,
                    rationale="保护记忆持久性，减缓经验淘汰速率",
                )
            )
            suggestions.append(
                ProtectiveBeltSuggestion(
                    target="consolidation",
                    action="提高合并相似度阈值",
                    priority=0.7,
                    rationale="减少经验合并以保持多样性",
                )
            )

        elif pattern == "ACTION_OSCILLATION":
            suggestions.append(
                ProtectiveBeltSuggestion(
                    target="action_policy",
                    action="增加行动惯性系数",
                    priority=0.8,
                    rationale="引入行动平滑约束以减少振荡",
                )
            )
            suggestions.append(
                ProtectiveBeltSuggestion(
                    target="configurator",
                    action="延长配置效果评估窗口",
                    priority=0.7,
                    rationale="更长的评估窗口可避免短期波动导致的频繁切换",
                )
            )

        elif pattern == "ENERGY_IMBALANCE":
            suggestions.append(
                ProtectiveBeltSuggestion(
                    target="energy_core",
                    action="调整能量分配权重",
                    priority=0.85,
                    rationale="通过参数调整而非结构变更恢复能量平衡",
                )
            )

        elif pattern == "COGNITIVE_OVERLOAD":
            suggestions.append(
                ProtectiveBeltSuggestion(
                    target="perception_pipeline",
                    action="降低感知通道采样率",
                    priority=0.75,
                    rationale="减少信息输入速率以缓解认知过载",
                )
            )
            suggestions.append(
                ProtectiveBeltSuggestion(
                    target="working_memory",
                    action="缩短工作记忆容量",
                    priority=0.6,
                    rationale="限制同时处理的信息量",
                )
            )

        elif pattern == "FEEDBACK_LOOP_BROKEN":
            suggestions.append(
                ProtectiveBeltSuggestion(
                    target="cognitive_loop_bus",
                    action="增加反馈通道的冗余路径",
                    priority=0.9,
                    rationale="修复反馈环而非移除，保护闭环完整性",
                )
            )

        elif pattern in ("MODEL_DRIFT", "CONFOUNDER_INTRUSION"):
            suggestions.append(
                ProtectiveBeltSuggestion(
                    target="model_parameters",
                    action="触发增量再训练",
                    priority=0.8,
                    rationale="参数层面的适应而非架构变更",
                )
            )

        # 通用建议：严重度高时优先保护带修正
        if severity in ("high", "critical") and not suggestions:
            suggestions.append(
                ProtectiveBeltSuggestion(
                    target="general",
                    action="优先调整参数和阈值，避免结构性变更",
                    priority=0.9,
                    rationale=f"高严重度诊断 ({pattern}) 需要保守修正策略",
                )
            )

        return suggestions

    def hard_core_status(self) -> dict[str, Any]:
        """查看当前硬核规则状态。

        Returns:
            规则 ID → 规则信息映射
        """
        return {
            r["rule_id"]: {
                "name": r["rule_name"],
                "severity": r["severity"].value,
                "protected_components": list(r["components"]),
            }
            for r in self._rules
        }

    # -----------------------------------------------------------------
    # 内部方法
    # -----------------------------------------------------------------

    def _build_default_rules(self) -> list[dict]:
        """构建默认硬核规则集 (7 条)。"""
        return [
            {
                "rule_id": "HC-1",
                "rule_name": "因果性不可放弃 (Causality Inviolability)",
                "components": _CAUSAL_COMPONENTS,
                "severity": RuleSeverity.ABSOLUTE,
                "forbidden_types": {ChangeType.REMOVE, ChangeType.DISABLE},
                "description": "禁止移除或禁用因果推理相关组件",
                "recommendation": "因果性是系统认知能力的基础，只能通过参数调整优化",
            },
            {
                "rule_id": "HC-2",
                "rule_name": "记忆性不可丧失 (Memory Persistence)",
                "components": _MEMORY_COMPONENTS,
                "severity": RuleSeverity.ABSOLUTE,
                "forbidden_types": {ChangeType.REMOVE, ChangeType.DISABLE},
                "description": "禁止移除或禁用记忆系统相关组件",
                "recommendation": "记忆是认知连续性的保证，可调整容量/阈值但不可移除",
            },
            {
                "rule_id": "HC-3",
                "rule_name": "闭环性不可断裂 (Closed-Loop Integrity)",
                "components": _LOOP_COMPONENTS,
                "severity": RuleSeverity.ABSOLUTE,
                "forbidden_types": {ChangeType.REMOVE, ChangeType.DISABLE},
                "description": "禁止移除或禁用反馈闭环相关组件",
                "recommendation": "闭环是自适应系统的基础架构，不可断裂",
            },
            {
                "rule_id": "HC-4",
                "rule_name": "能量守恒不可违反 (Energy Conservation)",
                "components": _ENERGY_COMPONENTS,
                "severity": RuleSeverity.STRONG,
                "forbidden_types": {ChangeType.REMOVE},
                "description": "禁止移除能量系统组件（可能导致守恒破坏）",
                "recommendation": "能量系统可通过参数调整优化，但不可移除核心组件",
            },
            {
                "rule_id": "HC-5",
                "rule_name": "认知多样性不可坍缩 (Cognitive Diversity)",
                "components": _DIVERSITY_COMPONENTS,
                "severity": RuleSeverity.STRONG,
                "forbidden_types": {ChangeType.REMOVE},
                "description": "禁止移除影响认知多样性的组件",
                "recommendation": "多样性是 Ashby 必要多样性定律的要求",
            },
            {
                "rule_id": "HC-6",
                "rule_name": "时间不可逆性 (Temporal Irreversibility)",
                "components": _TEMPORAL_COMPONENTS,
                "severity": RuleSeverity.STRONG,
                "forbidden_types": {ChangeType.REMOVE},
                "description": "禁止移除时间编码和时序相关组件",
                "recommendation": "时间维度是世界模型的基本结构",
            },
            {
                "rule_id": "HC-7",
                "rule_name": "可解释性不可退化 (Explainability Preservation)",
                "components": _EXPLAIN_COMPONENTS,
                "severity": RuleSeverity.ADVISORY,
                "forbidden_types": {ChangeType.REMOVE, ChangeType.DISABLE},
                "description": "不建议移除影响可解释性的组件",
                "recommendation": "可解释性是系统可信度的基础，应优先保留",
            },
        ]

    def _check_rule(
        self,
        rule: dict,
        change: ProposedChange,
        overlap: set[str],
    ) -> HardCoreViolation | None:
        """检查单条规则是否被违反。"""
        forbidden = rule.get("forbidden_types", set())

        if change.change_type in forbidden:
            return HardCoreViolation(
                rule_id=rule["rule_id"],
                rule_name=rule["rule_name"],
                severity=rule["severity"],
                description=f"{rule['description']}: 变更 '{change.description}' "
                f"试图 {change.change_type.value} 受保护组件 {sorted(overlap)}",
                affected_component=sorted(overlap)[0],
                recommendation=rule.get("recommendation", ""),
            )

        return None

    def reset_stats(self) -> None:
        """重置统计数据。"""
        self._stats = NegativeHeuristicStats()
