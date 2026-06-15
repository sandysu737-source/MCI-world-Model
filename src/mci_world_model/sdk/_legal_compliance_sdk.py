"""MCI World Model — LegalComplianceSDK 法律合规因果 SDK
=====================================================

面向法律合规场景的因果推理 SDK，
确保因果结论满足法律证据标准和审计要求。

核心能力:
    LegalEvidence        — 法律证据数据类
    LegalCausalConclusion — 法律因果结论
    LegalComplianceSDK   — 法律合规推理入口

设计原则:
    - 审计轨迹100%留存
    - 管辖区标注强制
    - 偏差检测集成
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# LegalEvidence — 法律证据
# =============================================================================


@dataclass
class LegalEvidence:
    """法律证据数据类。

    Attributes:
        evidence_id: 证据唯一ID
        evidence_type: 证据类型 ('document' / 'testimony' / 'data' / 'expert')
        description: 证据描述
        reliability: 可靠性 [0, 1]
        jurisdiction: 适用管辖区
        timestamp: 时间戳
    """

    evidence_id: str
    evidence_type: str = "document"
    description: str = ""
    reliability: float = 0.5
    jurisdiction: str = ""
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# LegalCausalConclusion — 法律因果结论
# =============================================================================


@dataclass
class LegalCausalConclusion:
    """法律因果结论。

    Attributes:
        cause: 因果变量
        effect: 结果变量
        causal_link_strength: 因果链强度 [0, 1]
        legal_standard_met: 是否满足法律证明标准
        standard_type: 证明标准类型
        jurisdiction: 适用管辖区
        evidence_ids: 支撑证据ID
        audit_trail: 审计轨迹
        bias_flags: 偏差标记
    """

    cause: str
    effect: str
    causal_link_strength: float = 0.0
    legal_standard_met: bool = False
    standard_type: str = "preponderance"
    jurisdiction: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    audit_trail: list[dict] = field(default_factory=list)
    bias_flags: list[str] = field(default_factory=list)


# =============================================================================
# LegalComplianceSDK — 法律合规因果 SDK
# =============================================================================


class LegalComplianceSDK:
    """法律合规因果 SDK — 满足法律证据标准的因果推理。

    用法:
        >>> sdk = LegalComplianceSDK(jurisdiction="CN")
        >>> sdk.add_evidence(LegalEvidence(evidence_id="doc1", reliability=0.8))
        >>> conclusion = sdk.reason(cause="action_A", effect="harm_B")
    """

    # 法律证明标准阈值
    LEGAL_STANDARDS = {
        "preponderance": 0.51,  # 优势证据 (民事)
        "clear_convincing": 0.75,  # 明确且令人信服
        "beyond_reasonable_doubt": 0.95,  # 排除合理怀疑 (刑事)
    }

    def __init__(
        self,
        jurisdiction: str = "CN",
        standard: str = "preponderance",
    ):
        if standard not in self.LEGAL_STANDARDS:
            raise ValueError(f"未知证明标准: {standard}, 可选: {list(self.LEGAL_STANDARDS.keys())}")
        self._jurisdiction = jurisdiction
        self._standard = standard
        self._threshold = self.LEGAL_STANDARDS[standard]
        self._evidence: list[LegalEvidence] = []
        self._conclusions: list[LegalCausalConclusion] = []
        self._audit_trail: list[dict] = []

    @property
    def jurisdiction(self) -> str:
        return self._jurisdiction

    @property
    def standard(self) -> str:
        return self._standard

    @property
    def threshold(self) -> float:
        return self._threshold

    def add_evidence(self, evidence: LegalEvidence) -> None:
        """添加法律证据。"""
        self._evidence.append(evidence)
        self._audit_trail.append(
            {
                "action": "add_evidence",
                "evidence_id": evidence.evidence_id,
                "reliability": evidence.reliability,
                "timestamp": time.time(),
            }
        )

    def reason(
        self,
        cause: str,
        effect: str,
        prior_strength: float = 0.5,
    ) -> LegalCausalConclusion:
        """法律合规因果推理。

        Args:
            cause: 假设原因
            effect: 观测结果
            prior_strength: 先验因果强度

        Returns:
            LegalCausalConclusion
        """
        audit_steps = []
        bias_flags = []

        # Step 1: 证据可靠性加权
        if not self._evidence:
            audit_steps.append({"step": "evidence_check", "result": "no_evidence"})
            return LegalCausalConclusion(
                cause=cause,
                effect=effect,
                jurisdiction=self._jurisdiction,
                standard_type=self._standard,
                audit_trail=audit_steps,
                bias_flags=["no_evidence"],
            )

        avg_reliability = float(np.mean([e.reliability for e in self._evidence]))
        audit_steps.append(
            {
                "step": "reliability_assessment",
                "avg_reliability": avg_reliability,
                "evidence_count": len(self._evidence),
            }
        )

        # Step 2: 偏差检测
        reliabilities = [e.reliability for e in self._evidence]
        if len(reliabilities) >= 2:
            std_reliability = float(np.std(reliabilities))
            if std_reliability < 0.05:
                bias_flags.append("uniform_reliability_suspicion")
            if avg_reliability > 0.98:
                bias_flags.append("excessive_confidence")

        audit_steps.append(
            {
                "step": "bias_detection",
                "bias_flags": bias_flags,
            }
        )

        # Step 3: 因果链强度
        evidence_weight = min(len(self._evidence) / 3.0, 1.0)
        causal_link_strength = prior_strength * 0.3 + avg_reliability * 0.5 + evidence_weight * 0.2

        # Step 4: 法律证明标准检查
        legal_standard_met = causal_link_strength >= self._threshold and len(bias_flags) == 0

        audit_steps.append(
            {
                "step": "standard_check",
                "causal_link_strength": causal_link_strength,
                "threshold": self._threshold,
                "standard_met": legal_standard_met,
            }
        )

        conclusion = LegalCausalConclusion(
            cause=cause,
            effect=effect,
            causal_link_strength=causal_link_strength,
            legal_standard_met=legal_standard_met,
            standard_type=self._standard,
            jurisdiction=self._jurisdiction,
            evidence_ids=[e.evidence_id for e in self._evidence],
            audit_trail=audit_steps,
            bias_flags=bias_flags,
        )

        self._conclusions.append(conclusion)
        self._audit_trail.append(
            {
                "action": "reason",
                "cause": cause,
                "effect": effect,
                "strength": causal_link_strength,
                "standard_met": legal_standard_met,
                "timestamp": time.time(),
            }
        )

        return conclusion

    def get_audit_trail(self) -> list[dict]:
        """获取完整审计轨迹。"""
        return list(self._audit_trail)

    def statistics(self) -> dict[str, Any]:
        """SDK 统计。"""
        met = sum(1 for c in self._conclusions if c.legal_standard_met)
        return {
            "jurisdiction": self._jurisdiction,
            "standard": self._standard,
            "threshold": self._threshold,
            "evidence_count": len(self._evidence),
            "conclusion_count": len(self._conclusions),
            "standards_met": met,
            "standards_met_rate": met / max(len(self._conclusions), 1),
            "audit_entries": len(self._audit_trail),
        }
