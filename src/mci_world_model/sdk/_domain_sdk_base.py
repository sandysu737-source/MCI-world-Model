from __future__ import annotations

"""MCI World Model — MCIDomainSDK 领域SDK基座
=============================================

三大领域SDK (医疗/法律/工程) 的统一入口——
提供跨领域的因果推理和合规检查编排能力。

核心能力:
    DomainType          — 领域类型枚举
    DomainResult        — 领域推理结果
    MCIDomainSDK        — 统一领域SDK

设计原则:
    - 编排不重实现: 委托给3个专业SDK
    - 统一接口: 一个入口覆盖三大领域
"""


import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# DomainType — 领域类型
# =============================================================================


class DomainType(Enum):
    """领域类型枚举。"""

    MEDICAL = "medical"
    LEGAL = "legal"
    ENGINEERING = "engineering"


# =============================================================================
# DomainResult — 领域推理结果
# =============================================================================


@dataclass
class DomainResult:
    """领域推理结果。

    Attributes:
        domain: 领域类型
        cause: 因果变量
        effect: 结果变量
        confidence: 置信度
        is_compliant: 是否合规
        details: 详细信息
    """

    domain: DomainType
    cause: str = ""
    effect: str = ""
    confidence: float = 0.0
    is_compliant: bool = False
    details: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# MCIDomainSDK — 统一领域SDK
# =============================================================================


class MCIDomainSDK:
    """统一领域SDK — 医疗/法律/工程三大领域的因果推理入口。

    用法:
        >>> sdk = MCIDomainSDK()
        >>> sdk.add_domain_evidence("medical", evidence_dict)
        >>> result = sdk.reason(domain="medical", cause="X", effect="Y")
    """

    def __init__(self) -> None:
        self._domain_evidence: dict[str, list[dict[str, Any]]] = {
            "medical": [],
            "legal": [],
            "engineering": [],
        }
        self._reason_count: int = 0
        self._results: list[DomainResult] = []

    def add_domain_evidence(self, domain: str, evidence: dict[str, Any]) -> None:
        """添加领域证据。"""
        if domain not in self._domain_evidence:
            self._domain_evidence[domain] = []
        self._domain_evidence[domain].append(evidence)

    def reason(
        self,
        domain: str,
        cause: str,
        effect: str,
        prior_strength: float = 0.5,
    ) -> DomainResult:
        """领域因果推理。

        Args:
            domain: 领域名称
            cause: 因果变量
            effect: 结果变量
            prior_strength: 先验强度

        Returns:
            DomainResult
        """
        self._reason_count += 1
        evidence = self._domain_evidence.get(domain, [])

        # 简化领域推理
        if domain == "medical":
            return self._reason_medical(cause, effect, evidence, prior_strength)
        elif domain == "legal":
            return self._reason_legal(cause, effect, evidence, prior_strength)
        elif domain == "engineering":
            return self._reason_engineering(cause, effect, evidence, prior_strength)
        else:
            return DomainResult(
                domain=DomainType.MEDICAL,
                cause=cause,
                effect=effect,
                details={"error": f"未知领域: {domain}"},
            )

    def _reason_medical(self, cause: str, effect: str, evidence: list[dict[str, Any]], prior: float) -> DomainResult:
        evidence_count = len(evidence)
        avg_confidence = (
            float(sum(e.get("confidence", 0.5) for e in evidence)) / evidence_count if evidence_count > 0 else 0.0
        )
        is_compliant = evidence_count >= 2 and avg_confidence >= 0.7
        confidence = prior * 0.3 + avg_confidence * 0.5 + min(evidence_count / 5.0, 1.0) * 0.2

        result = DomainResult(
            domain=DomainType.MEDICAL,
            cause=cause,
            effect=effect,
            confidence=min(confidence, 1.0),
            is_compliant=is_compliant,
            details={"evidence_count": evidence_count, "avg_confidence": avg_confidence},
        )
        self._results.append(result)
        return result

    def _reason_legal(self, cause: str, effect: str, evidence: list[dict[str, Any]], prior: float) -> DomainResult:
        evidence_count = len(evidence)
        has_jurisdiction = any(e.get("jurisdiction") for e in evidence)
        has_audit = any(e.get("audit_trail") for e in evidence)
        avg_reliability = (
            float(sum(e.get("reliability", 0.5) for e in evidence)) / evidence_count if evidence_count > 0 else 0.0
        )
        is_compliant = has_jurisdiction and has_audit and avg_reliability >= 0.51

        result = DomainResult(
            domain=DomainType.LEGAL,
            cause=cause,
            effect=effect,
            confidence=avg_reliability,
            is_compliant=is_compliant,
            details={"has_jurisdiction": has_jurisdiction, "has_audit": has_audit},
        )
        self._results.append(result)
        return result

    def _reason_engineering(
        self, cause: str, effect: str, evidence: list[dict[str, Any]], prior: float
    ) -> DomainResult:
        evidence_count = len(evidence)
        has_margin = any(e.get("safety_margin", 0) >= 0.2 for e in evidence)
        is_compliant = has_margin and evidence_count > 0
        confidence = prior * 0.5 + (0.5 if has_margin else 0.0)

        result = DomainResult(
            domain=DomainType.ENGINEERING,
            cause=cause,
            effect=effect,
            confidence=min(confidence, 1.0),
            is_compliant=is_compliant,
            details={"evidence_count": evidence_count, "has_margin": has_margin},
        )
        self._results.append(result)
        return result

    def get_results(self, domain: str | None = None) -> list[DomainResult]:
        """获取推理结果。"""
        if domain is None:
            return list(self._results)
        return [r for r in self._results if r.domain.value == domain]

    def statistics(self) -> dict[str, Any]:
        return {
            "reason_count": self._reason_count,
            "results_count": len(self._results),
            "evidence_counts": {d: len(e) for d, e in self._domain_evidence.items()},
        }
