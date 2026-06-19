from __future__ import annotations

"""MCI World Model — ComplianceRuleEngine 合规规则引擎
====================================================

面向三大安全关键领域(医疗/法律/工程)的合规检查引擎，
在因果推理执行前后插入规则拦截，确保输出满足行业法规。

核心能力:
    ComplianceRule       — 合规规则抽象基类
    MedicalComplianceRule — 医疗合规规则集
    LegalComplianceRule  — 法律合规规则集
    EngineeringComplianceRule — 工程安全合规规则集
    ComplianceReport     — 合规检查报告
    ComplianceRuleEngine — 规则引擎(注册+执行+审计)

设计原则:
    - 规则与推理解耦: check(context) → ComplianceReport
    - 领域可扩展: 新领域只需实现 ComplianceRule ABC
    - 审计留痕: 每次检查生成 ComplianceReport 可追溯
    - 与 SafetyConstraint 互补: Safety 聚焦物理约束,
      Compliance 聚焦行业法规
"""


import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# ComplianceLevel — 合规等级
# =============================================================================


class ComplianceLevel(Enum):
    """合规等级枚举。"""

    COMPLIANT = "compliant"  # 完全合规
    CONDITIONAL = "conditional"  # 有条件合规
    NON_COMPLIANT = "non_compliant"  # 不合规
    UNABLE_TO_ASSESS = "unable_to_assess"  # 无法评估


# =============================================================================
# ComplianceReport — 合规检查报告
# =============================================================================


@dataclass
class ComplianceReport:
    """合规检查报告。

    Attributes:
        rule_name: 规则名称
        domain: 领域 ('medical' / 'legal' / 'engineering')
        level: 合规等级
        details: 详细信息
        timestamp: 检查时间戳
        reasoning: 合规/不合规原因
        remediation: 补救建议(不合规时)
    """

    rule_name: str
    domain: str
    level: ComplianceLevel
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    reasoning: str = ""
    remediation: str = ""


# =============================================================================
# ComplianceRule — 合规规则抽象基类
# =============================================================================


class ComplianceRule(ABC):
    """合规规则抽象基类。

    任何领域合规规则只需实现:
        - domain: 属性, 返回所属领域
        - name: 属性, 返回规则名
        - check(context): 方法, 执行合规检查
    """

    @property
    @abstractmethod
    def domain(self) -> str:
        """所属领域。"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """规则名称。"""
        ...

    @abstractmethod
    def check(self, context: dict[str, Any]) -> ComplianceReport:
        """检查给定上下文是否满足合规规则。

        Args:
            context: 推理上下文, 包含:
                - 'conclusion': 推理结论
                - 'evidence': 证据列表
                - 'confidence': 置信度
                - 'intervention': 干预方案(可选)
                - 'patient_data': 患者数据(医疗领域)
                - 'jurisdiction': 管辖区(法律领域)
                - 'system_params': 系统参数(工程领域)

        Returns:
            ComplianceReport
        """
        ...


# =============================================================================
# MedicalComplianceRule — 医疗合规规则集
# =============================================================================


class _EvidenceSufficiencyRule(ComplianceRule):
    """证据充分性规则——医疗推理必须有足够证据支撑。"""

    @property
    def domain(self) -> str:
        return "medical"

    @property
    def name(self) -> str:
        return "evidence_sufficiency"

    def check(self, context: dict[str, Any]) -> ComplianceReport:
        evidence = context.get("evidence", [])
        confidence = context.get("confidence", 0.0)
        min_evidence_count = 2

        if len(evidence) < min_evidence_count:
            return ComplianceReport(
                rule_name=self.name,
                domain=self.domain,
                level=ComplianceLevel.NON_COMPLIANT,
                details={"evidence_count": len(evidence), "min_required": min_evidence_count},
                reasoning=f"证据不足: {len(evidence)} 条 < 最低要求 {min_evidence_count} 条",
                remediation="补充更多临床证据后再做因果结论",
            )

        if confidence < 0.7:
            return ComplianceReport(
                rule_name=self.name,
                domain=self.domain,
                level=ComplianceLevel.CONDITIONAL,
                details={"confidence": confidence, "threshold": 0.7},
                reasoning=f"置信度偏低: {confidence:.2f} < 0.70",
                remediation="建议增加样本量或使用更强因果推断方法",
            )

        return ComplianceReport(
            rule_name=self.name,
            domain=self.domain,
            level=ComplianceLevel.COMPLIANT,
            details={"evidence_count": len(evidence), "confidence": confidence},
            reasoning="证据充分且置信度达标",
        )


class _InterventionSafetyRule(ComplianceRule):
    """干预安全规则——医疗干预必须通过安全性评估。"""

    @property
    def domain(self) -> str:
        return "medical"

    @property
    def name(self) -> str:
        return "intervention_safety"

    def check(self, context: dict[str, Any]) -> ComplianceReport:
        intervention = context.get("intervention")
        if intervention is None:
            return ComplianceReport(
                rule_name=self.name,
                domain=self.domain,
                level=ComplianceLevel.COMPLIANT,
                reasoning="无干预方案, 不触发干预安全检查",
            )

        # 检查干预是否有风险评估
        risk_assessment = context.get("risk_assessment")
        if risk_assessment is None:
            return ComplianceReport(
                rule_name=self.name,
                domain=self.domain,
                level=ComplianceLevel.NON_COMPLIANT,
                details={"intervention": str(intervention)[:100]},
                reasoning="干预方案缺少风险评估",
                remediation="执行风险评估后再推荐干预",
            )

        risk_level = risk_assessment.get("level", "unknown")
        if risk_level == "high":
            return ComplianceReport(
                rule_name=self.name,
                domain=self.domain,
                level=ComplianceLevel.NON_COMPLIANT,
                details={"risk_level": risk_level},
                reasoning="干预风险等级为高, 不满足安全要求",
                remediation="降低风险等级或寻找替代干预方案",
            )

        if risk_level == "medium":
            return ComplianceReport(
                rule_name=self.name,
                domain=self.domain,
                level=ComplianceLevel.CONDITIONAL,
                details={"risk_level": risk_level},
                reasoning="干预风险等级为中, 需人工复核",
                remediation="建议医生人工审核后再执行",
            )

        return ComplianceReport(
            rule_name=self.name,
            domain=self.domain,
            level=ComplianceLevel.COMPLIANT,
            details={"risk_level": risk_level},
            reasoning="干预风险等级为低, 满足安全要求",
        )


class _PatientConsentRule(ComplianceRule):
    """患者知情同意规则——涉及患者数据的推理须确认知情同意。"""

    @property
    def domain(self) -> str:
        return "medical"

    @property
    def name(self) -> str:
        return "patient_consent"

    def check(self, context: dict[str, Any]) -> ComplianceReport:
        patient_data = context.get("patient_data")
        if patient_data is None:
            return ComplianceReport(
                rule_name=self.name,
                domain=self.domain,
                level=ComplianceLevel.COMPLIANT,
                reasoning="未使用患者数据, 不触发知情同意检查",
            )

        consent = patient_data.get("consent_given", False) if isinstance(patient_data, dict) else False
        if not consent:
            return ComplianceReport(
                rule_name=self.name,
                domain=self.domain,
                level=ComplianceLevel.NON_COMPLIANT,
                details={"consent_given": consent},
                reasoning="患者未确认知情同意, 不得使用患者数据进行因果推理",
                remediation="获取患者知情同意书后再执行推理",
            )

        return ComplianceReport(
            rule_name=self.name,
            domain=self.domain,
            level=ComplianceLevel.COMPLIANT,
            details={"consent_given": consent},
            reasoning="已获取患者知情同意",
        )


# =============================================================================
# LegalComplianceRule — 法律合规规则集
# =============================================================================


class _JurisdictionRule(ComplianceRule):
    """管辖区适用规则——因果推理结论须标注适用管辖区。"""

    @property
    def domain(self) -> str:
        return "legal"

    @property
    def name(self) -> str:
        return "jurisdiction_applicability"

    def check(self, context: dict[str, Any]) -> ComplianceReport:
        jurisdiction = context.get("jurisdiction")
        if jurisdiction is None or jurisdiction == "":
            return ComplianceReport(
                rule_name=self.name,
                domain=self.domain,
                level=ComplianceLevel.NON_COMPLIANT,
                details={"jurisdiction": jurisdiction},
                reasoning="未标注适用管辖区",
                remediation="为推理结论标注适用法律管辖区",
            )

        # 已知管辖区列表
        known_jurisdictions = {"CN", "US", "EU", "UK", "JP", "KR", "AU", "CA"}
        if jurisdiction not in known_jurisdictions:
            return ComplianceReport(
                rule_name=self.name,
                domain=self.domain,
                level=ComplianceLevel.CONDITIONAL,
                details={"jurisdiction": jurisdiction, "known": list(known_jurisdictions)},
                reasoning=f"管辖区 '{jurisdiction}' 不在已知列表中, 需人工确认",
                remediation="确认管辖区代码是否正确",
            )

        return ComplianceReport(
            rule_name=self.name,
            domain=self.domain,
            level=ComplianceLevel.COMPLIANT,
            details={"jurisdiction": jurisdiction},
            reasoning=f"管辖区 '{jurisdiction}' 已确认",
        )


class _AuditTrailRule(ComplianceRule):
    """审计轨迹规则——法律推理须保留完整审计轨迹。"""

    @property
    def domain(self) -> str:
        return "legal"

    @property
    def name(self) -> str:
        return "audit_trail"

    def check(self, context: dict[str, Any]) -> ComplianceReport:
        audit_trail = context.get("audit_trail")

        if audit_trail is None:
            return ComplianceReport(
                rule_name=self.name,
                domain=self.domain,
                level=ComplianceLevel.NON_COMPLIANT,
                reasoning="法律推理缺少审计轨迹",
                remediation="为推理过程添加完整审计轨迹记录",
            )

        if isinstance(audit_trail, (list, dict)):
            if len(audit_trail) == 0:
                return ComplianceReport(
                    rule_name=self.name,
                    domain=self.domain,
                    level=ComplianceLevel.NON_COMPLIANT,
                    details={"trail_length": 0},
                    reasoning="审计轨迹为空",
                    remediation="记录推理的每一步骤到审计轨迹",
                )

        return ComplianceReport(
            rule_name=self.name,
            domain=self.domain,
            level=ComplianceLevel.COMPLIANT,
            details={"trail_length": len(audit_trail) if hasattr(audit_trail, "__len__") else 1},
            reasoning="审计轨迹完整",
        )


class _BiasDetectionRule(ComplianceRule):
    """偏差检测规则——法律推理须检测潜在偏差。"""

    @property
    def domain(self) -> str:
        return "legal"

    @property
    def name(self) -> str:
        return "bias_detection"

    def check(self, context: dict[str, Any]) -> ComplianceReport:
        conclusion = context.get("conclusion", "")
        evidence = context.get("evidence", [])
        confidence = context.get("confidence", 0.0)

        # 简化偏差检测: 如果证据全部一致但置信度异常高, 可能存在确认偏差
        if len(evidence) >= 3 and confidence > 0.98:
            return ComplianceReport(
                rule_name=self.name,
                domain=self.domain,
                level=ComplianceLevel.CONDITIONAL,
                details={"confidence": confidence, "evidence_count": len(evidence)},
                reasoning="置信度异常高, 可能存在确认偏差",
                remediation="引入反面证据进行偏差校正",
            )

        # 如果证据为空但有结论, 可能存在先验偏差
        if len(evidence) == 0 and conclusion:
            return ComplianceReport(
                rule_name=self.name,
                domain=self.domain,
                level=ComplianceLevel.NON_COMPLIANT,
                details={"evidence_count": 0, "has_conclusion": bool(conclusion)},
                reasoning="无证据支持结论, 存在先验偏差风险",
                remediation="为结论提供至少一条证据支持",
            )

        return ComplianceReport(
            rule_name=self.name,
            domain=self.domain,
            level=ComplianceLevel.COMPLIANT,
            details={"confidence": confidence, "evidence_count": len(evidence)},
            reasoning="未检测到明显偏差",
        )


# =============================================================================
# EngineeringComplianceRule — 工程安全合规规则集
# =============================================================================


class _SafetyMarginRule(ComplianceRule):
    """安全裕度规则——工程参数须满足安全裕度要求。"""

    @property
    def domain(self) -> str:
        return "engineering"

    @property
    def name(self) -> str:
        return "safety_margin"

    def check(self, context: dict[str, Any]) -> ComplianceReport:
        system_params = context.get("system_params", {})
        if not system_params:
            return ComplianceReport(
                rule_name=self.name,
                domain=self.domain,
                level=ComplianceLevel.UNABLE_TO_ASSESS,
                reasoning="未提供系统参数, 无法评估安全裕度",
                remediation="提供系统参数(含设计值和极限值)",
            )

        # 检查每个参数的安全裕度
        violations = []
        min_margin = 0.20  # 最低20%安全裕度
        for param_name, param_value in system_params.items():
            if isinstance(param_value, dict):
                design = param_value.get("design", 0.0)
                limit = param_value.get("limit", 0.0)
                if limit != 0.0:
                    margin = abs(limit - design) / abs(limit)
                    if margin < min_margin:
                        violations.append(
                            {
                                "param": param_name,
                                "margin": margin,
                                "min_required": min_margin,
                            }
                        )

        if violations:
            return ComplianceReport(
                rule_name=self.name,
                domain=self.domain,
                level=ComplianceLevel.NON_COMPLIANT,
                details={"violations": violations},
                reasoning=f"{len(violations)} 个参数安全裕度不足",
                remediation="增大设计裕度或降低工作参数",
            )

        return ComplianceReport(
            rule_name=self.name,
            domain=self.domain,
            level=ComplianceLevel.COMPLIANT,
            details={"param_count": len(system_params), "min_margin": min_margin},
            reasoning="所有参数安全裕度达标",
        )


class _RedundancyCheckRule(ComplianceRule):
    """冗余检查规则——安全关键系统须有冗余设计。"""

    @property
    def domain(self) -> str:
        return "engineering"

    @property
    def name(self) -> str:
        return "redundancy_check"

    def check(self, context: dict[str, Any]) -> ComplianceReport:
        redundancy = context.get("redundancy", {})

        if not redundancy:
            return ComplianceReport(
                rule_name=self.name,
                domain=self.domain,
                level=ComplianceLevel.CONDITIONAL,
                reasoning="未提供冗余设计信息",
                remediation="为安全关键路径提供冗余设计方案",
            )

        # 检查关键路径是否有冗余
        critical_paths = context.get("critical_paths", [])
        paths_without_redundancy = []
        for path in critical_paths:
            if not redundancy.get(path, False):
                paths_without_redundancy.append(path)

        if paths_without_redundancy:
            return ComplianceReport(
                rule_name=self.name,
                domain=self.domain,
                level=ComplianceLevel.NON_COMPLIANT,
                details={"paths_without_redundancy": paths_without_redundancy},
                reasoning=f"{len(paths_without_redundancy)} 条关键路径缺少冗余",
                remediation="为缺少冗余的关键路径增加备份机制",
            )

        return ComplianceReport(
            rule_name=self.name,
            domain=self.domain,
            level=ComplianceLevel.COMPLIANT,
            details={"critical_paths": len(critical_paths), "redundant_paths": len(critical_paths)},
            reasoning="所有关键路径均有冗余设计",
        )


class _FailureModeRule(ComplianceRule):
    """失效模式规则——工程系统须完成 FMEA 分析。"""

    @property
    def domain(self) -> str:
        return "engineering"

    @property
    def name(self) -> str:
        return "failure_mode_analysis"

    def check(self, context: dict[str, Any]) -> ComplianceReport:
        fmea = context.get("fmea")

        if fmea is None:
            return ComplianceReport(
                rule_name=self.name,
                domain=self.domain,
                level=ComplianceLevel.NON_COMPLIANT,
                reasoning="缺少失效模式与影响分析(FMEA)",
                remediation="执行 FMEA 分析并记录结果",
            )

        if isinstance(fmea, list) and len(fmea) == 0:
            return ComplianceReport(
                rule_name=self.name,
                domain=self.domain,
                level=ComplianceLevel.NON_COMPLIANT,
                details={"fmea_items": 0},
                reasoning="FMEA 分析为空",
                remediation="至少识别并记录一个潜在失效模式",
            )

        # 检查是否有高 RPN 项未处理
        if isinstance(fmea, list):
            high_rpn_items = [
                item
                for item in fmea
                if isinstance(item, dict) and item.get("rpn", 0) > 200 and not item.get("mitigated", False)
            ]
            if high_rpn_items:
                return ComplianceReport(
                    rule_name=self.name,
                    domain=self.domain,
                    level=ComplianceLevel.CONDITIONAL,
                    details={"high_rpn_unmitigated": len(high_rpn_items)},
                    reasoning=f"{len(high_rpn_items)} 个高 RPN 项未缓解",
                    remediation="对 RPN>200 的失效模式制定缓解措施",
                )

        return ComplianceReport(
            rule_name=self.name,
            domain=self.domain,
            level=ComplianceLevel.COMPLIANT,
            details={"fmea_items": len(fmea) if isinstance(fmea, list) else 1},
            reasoning="FMEA 分析完整且高 RPN 项已缓解",
        )


# =============================================================================
# ComplianceRuleEngine — 合规规则引擎
# =============================================================================


class ComplianceRuleEngine:
    """合规规则引擎——注册+执行+审计三合一。

    用法:
        >>> engine = ComplianceRuleEngine()
        >>> # 引擎自动注册三大领域9条默认规则
        >>> report = engine.check(context)
        >>> if not report.is_compliant:
        >>>     print(f"合规失败: {report.summary}")
    """

    def __init__(self, auto_register_defaults: bool = True) -> None:
        self._rules: list[ComplianceRule] = []
        self._check_history: list[ComplianceReport] = []
        self._check_count: int = 0

        if auto_register_defaults:
            self._register_defaults()

    def _register_defaults(self) -> None:
        """注册三大领域的默认规则集。"""
        # 医疗领域
        for rule_cls in [_EvidenceSufficiencyRule, _InterventionSafetyRule, _PatientConsentRule]:
            self._rules.append(rule_cls())
        # 法律领域
        for rule_cls in [_JurisdictionRule, _AuditTrailRule, _BiasDetectionRule]:
            self._rules.append(rule_cls())
        # 工程领域
        for rule_cls in [_SafetyMarginRule, _RedundancyCheckRule, _FailureModeRule]:
            self._rules.append(rule_cls())

        logger.info("合规引擎: 已注册 %d 条默认规则", len(self._rules))

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    @property
    def check_count(self) -> int:
        return self._check_count

    def register(self, rule: ComplianceRule) -> None:
        """注册自定义合规规则。"""
        self._rules.append(rule)
        logger.info("合规引擎: 注册自定义规则 %s (%s)", rule.name, rule.domain)

    def check_domain(self, domain: str, context: dict[str, Any]) -> list[ComplianceReport]:
        """检查指定领域的所有规则。

        Args:
            domain: 领域 ('medical' / 'legal' / 'engineering')
            context: 推理上下文

        Returns:
            该领域所有规则的检查报告
        """
        self._check_count += 1
        reports = []
        for rule in self._rules:
            if rule.domain == domain:
                try:
                    report = rule.check(context)
                    reports.append(report)
                    self._check_history.append(report)
                except Exception as e:
                    logger.warning("合规规则 %s 执行异常: %s", rule.name, e)
                    reports.append(
                        ComplianceReport(
                            rule_name=rule.name,
                            domain=rule.domain,
                            level=ComplianceLevel.UNABLE_TO_ASSESS,
                            reasoning=f"规则执行异常: {e}",
                        )
                    )

        return reports

    def check(self, context: dict[str, Any], domains: list[str] | None = None) -> ComplianceCheckResult:
        """执行合规检查。

        Args:
            context: 推理上下文
            domains: 要检查的领域列表(None表示全部)

        Returns:
            ComplianceCheckResult 聚合结果
        """
        if domains is None:
            domains = ["medical", "legal", "engineering"]

        all_reports: list[ComplianceReport] = []
        for domain in domains:
            domain_reports = self.check_domain(domain, context)
            all_reports.extend(domain_reports)

        # 聚合结果
        has_non_compliant = any(r.level == ComplianceLevel.NON_COMPLIANT for r in all_reports)
        has_conditional = any(r.level == ComplianceLevel.CONDITIONAL for r in all_reports)
        has_unable = any(r.level == ComplianceLevel.UNABLE_TO_ASSESS for r in all_reports)

        if has_non_compliant:
            overall_level = ComplianceLevel.NON_COMPLIANT
        elif has_conditional:
            overall_level = ComplianceLevel.CONDITIONAL
        elif has_unable:
            overall_level = ComplianceLevel.UNABLE_TO_ASSESS
        else:
            overall_level = ComplianceLevel.COMPLIANT

        non_compliant_rules = [r.rule_name for r in all_reports if r.level == ComplianceLevel.NON_COMPLIANT]
        conditional_rules = [r.rule_name for r in all_reports if r.level == ComplianceLevel.CONDITIONAL]

        return ComplianceCheckResult(
            overall_level=overall_level,
            reports=all_reports,
            non_compliant_rules=non_compliant_rules,
            conditional_rules=conditional_rules,
            domains_checked=domains,
        )

    def get_history(self, domain: str | None = None) -> list[ComplianceReport]:
        """获取检查历史。

        Args:
            domain: 过滤领域(None表示全部)

        Returns:
            检查报告列表
        """
        if domain is None:
            return list(self._check_history)
        return [r for r in self._check_history if r.domain == domain]

    def statistics(self) -> dict[str, Any]:
        """合规引擎统计。"""
        total = len(self._check_history)
        by_level = {}
        for level in ComplianceLevel:
            by_level[level.value] = sum(1 for r in self._check_history if r.level == level)

        by_domain = {}
        for report in self._check_history:
            by_domain.setdefault(report.domain, 0)
            by_domain[report.domain] += 1

        return {
            "rule_count": self.rule_count,
            "total_checks": self._check_count,
            "total_reports": total,
            "by_level": by_level,
            "by_domain": by_domain,
            "compliance_rate": (by_level.get("compliant", 0) / total if total > 0 else 0.0),
        }


# =============================================================================
# ComplianceCheckResult — 聚合检查结果
# =============================================================================


@dataclass
class ComplianceCheckResult:
    """聚合合规检查结果。

    Attributes:
        overall_level: 总体合规等级
        reports: 各规则检查报告
        non_compliant_rules: 不合规规则名列表
        conditional_rules: 有条件合规规则名列表
        domains_checked: 已检查领域列表
    """

    overall_level: ComplianceLevel
    reports: list[ComplianceReport] = field(default_factory=list)
    non_compliant_rules: list[str] = field(default_factory=list)
    conditional_rules: list[str] = field(default_factory=list)
    domains_checked: list[str] = field(default_factory=list)

    @property
    def is_compliant(self) -> bool:
        """是否完全合规(不含条件合规)。"""
        return self.overall_level == ComplianceLevel.COMPLIANT

    @property
    def is_acceptable(self) -> bool:
        """是否可接受(合规或有条件合规)。"""
        return self.overall_level in (ComplianceLevel.COMPLIANT, ComplianceLevel.CONDITIONAL)

    @property
    def summary(self) -> str:
        """简明摘要。"""
        if self.is_compliant:
            return "全部合规"
        parts = []
        if self.non_compliant_rules:
            parts.append(f"不合规: {', '.join(self.non_compliant_rules)}")
        if self.conditional_rules:
            parts.append(f"有条件: {', '.join(self.conditional_rules)}")
        return "; ".join(parts)
