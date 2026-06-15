"""MCI World Model — EngineeringSafetySDK 工程安全因果 SDK
=======================================================

面向工程安全关键场景的因果推理 SDK，
确保因果推理结论满足工程安全标准和冗余要求。

核心能力:
    SafetyParameter      — 安全参数数据类
    EngineeringCausalResult — 工程因果分析结果
    EngineeringSafetySDK — 工程安全推理入口

设计原则:
    - 安全裕度优先: 20%最低裕度强制
    - FMEA 集成: 因果结论须关联 FMEA
    - 冗余检查: 安全关键路径须有备份
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# SafetyParameter — 安全参数
# =============================================================================


@dataclass
class SafetyParameter:
    """安全参数数据类。

    Attributes:
        name: 参数名称
        design_value: 设计值
        limit_value: 极限值
        unit: 单位
        safety_margin: 安全裕度 (自动计算)
    """

    name: str
    design_value: float
    limit_value: float
    unit: str = ""
    safety_margin: float = 0.0

    def __post_init__(self):
        if self.limit_value != 0.0:
            self.safety_margin = abs(self.limit_value - self.design_value) / abs(self.limit_value)
        else:
            self.safety_margin = 0.0


# =============================================================================
# FMEAItem — FMEA 条目
# =============================================================================


@dataclass
class FMEAItem:
    """失效模式与影响分析条目。

    Attributes:
        failure_mode: 失效模式
        effect: 影响
        severity: 严重度 (1-10)
        occurrence: 发生度 (1-10)
        detection: 检出度 (1-10)
        rpn: 风险优先数 (自动计算)
        mitigated: 是否已缓解
    """

    failure_mode: str
    effect: str = ""
    severity: int = 5
    occurrence: int = 5
    detection: int = 5
    rpn: int = 0
    mitigated: bool = False

    def __post_init__(self):
        self.rpn = self.severity * self.occurrence * self.detection


# =============================================================================
# EngineeringCausalResult — 工程因果分析结果
# =============================================================================


@dataclass
class EngineeringCausalResult:
    """工程因果分析结果。

    Attributes:
        cause: 因果变量
        effect: 结果变量
        causal_confidence: 因果置信度
        safety_assessment: 安全评估
        margin_sufficient: 裕度是否充足
        fmea_rpn_max: 最高RPN
        redundancy_ok: 冗余检查结果
        audit_trail: 审计轨迹
    """

    cause: str
    effect: str
    causal_confidence: float = 0.0
    safety_assessment: str = "unknown"
    margin_sufficient: bool = False
    fmea_rpn_max: int = 0
    redundancy_ok: bool = False
    audit_trail: list[dict] = field(default_factory=list)


# =============================================================================
# EngineeringSafetySDK — 工程安全因果 SDK
# =============================================================================


class EngineeringSafetySDK:
    """工程安全因果 SDK — 满足工程安全标准的因果推理。

    用法:
        >>> sdk = EngineeringSafetySDK()
        >>> sdk.add_parameter(SafetyParameter("temp", 80.0, 120.0))
        >>> sdk.add_fmea(FMEAItem("valve_stuck", severity=8, occurrence=3, detection=2))
        >>> result = sdk.analyze(cause="high_temp", effect="system_failure")
    """

    MIN_SAFETY_MARGIN = 0.20  # 最低20%安全裕度
    CRITICAL_RPN_THRESHOLD = 200

    def __init__(self, system_name: str = "", redundancy_required: bool = True):
        self._system_name = system_name
        self._redundancy_required = redundancy_required
        self._parameters: dict[str, SafetyParameter] = {}
        self._fmea_items: list[FMEAItem] = []
        self._redundancy_status: dict[str, bool] = {}
        self._results: list[EngineeringCausalResult] = []

    @property
    def system_name(self) -> str:
        return self._system_name

    @property
    def parameter_count(self) -> int:
        return len(self._parameters)

    def add_parameter(self, param: SafetyParameter) -> None:
        """添加安全参数。"""
        self._parameters[param.name] = param
        logger.info("工程SDK: 添加参数 %s (裕度=%.1f%%)", param.name, param.safety_margin * 100)

    def add_fmea(self, item: FMEAItem) -> None:
        """添加 FMEA 条目。"""
        self._fmea_items.append(item)
        logger.info("工程SDK: 添加FMEA %s (RPN=%d)", item.failure_mode, item.rpn)

    def set_redundancy(self, path_name: str, has_redundancy: bool) -> None:
        """设置冗余状态。"""
        self._redundancy_status[path_name] = has_redundancy

    def analyze(
        self,
        cause: str,
        effect: str,
        causal_evidence_strength: float = 0.5,
    ) -> EngineeringCausalResult:
        """工程安全因果分析。

        Args:
            cause: 因果变量
            effect: 结果变量
            causal_evidence_strength: 因果证据强度

        Returns:
            EngineeringCausalResult
        """
        audit_trail = []

        # Step 1: 安全裕度检查
        margin_violations = [name for name, p in self._parameters.items() if p.safety_margin < self.MIN_SAFETY_MARGIN]
        margin_sufficient = len(margin_violations) == 0
        audit_trail.append(
            {
                "step": "margin_check",
                "violations": margin_violations,
                "sufficient": margin_sufficient,
            }
        )

        # Step 2: FMEA 检查
        rpn_max = max((item.rpn for item in self._fmea_items), default=0)
        unmitigated_high_rpn = [
            item for item in self._fmea_items if item.rpn > self.CRITICAL_RPN_THRESHOLD and not item.mitigated
        ]
        audit_trail.append(
            {
                "step": "fmea_check",
                "rpn_max": rpn_max,
                "unmitigated_critical": len(unmitigated_high_rpn),
            }
        )

        # Step 3: 冗余检查
        redundancy_ok = True
        if self._redundancy_required:
            redundancy_ok = all(self._redundancy_status.values()) if self._redundancy_status else False
        audit_trail.append(
            {
                "step": "redundancy_check",
                "redundancy_ok": redundancy_ok,
            }
        )

        # Step 4: 综合安全评估
        if not margin_sufficient:
            safety_assessment = "unsafe"
        elif unmitigated_high_rpn or (not redundancy_ok and self._redundancy_required):
            safety_assessment = "conditional"
        else:
            safety_assessment = "safe"

        # Step 5: 因果置信度调整 (安全因素影响)
        confidence_modifier = 1.0
        if safety_assessment == "unsafe":
            confidence_modifier = 0.3
        elif safety_assessment == "conditional":
            confidence_modifier = 0.7

        causal_confidence = causal_evidence_strength * confidence_modifier

        result = EngineeringCausalResult(
            cause=cause,
            effect=effect,
            causal_confidence=causal_confidence,
            safety_assessment=safety_assessment,
            margin_sufficient=margin_sufficient,
            fmea_rpn_max=rpn_max,
            redundancy_ok=redundancy_ok,
            audit_trail=audit_trail,
        )

        self._results.append(result)
        logger.info(
            "工程SDK: 分析 %s→%s, 安全=%s, 置信度=%.2f",
            cause,
            effect,
            safety_assessment,
            causal_confidence,
        )

        return result

    def statistics(self) -> dict[str, Any]:
        """SDK 统计。"""
        return {
            "system_name": self._system_name,
            "parameter_count": self.parameter_count,
            "fmea_items": len(self._fmea_items),
            "redundancy_paths": len(self._redundancy_status),
            "analysis_count": len(self._results),
            "safe_count": sum(1 for r in self._results if r.safety_assessment == "safe"),
        }
