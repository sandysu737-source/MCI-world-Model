from __future__ import annotations

"""MCI World Model v4.6.0 — FederatedTrust 联邦信任框架
============================================================

跨节点信任传递与联邦信任评估 — 可信因果联邦的基础。

核心能力:
    assess_federation_trust(node_id, evidence)   — 联邦信任评估
    propagate_trust(cert, target_node)           — 信任传播
    issue_trust_certificate(node_id, evidence)   — 颁发信任证书
    verify_certificate(cert)                     — 验证信任证书

设计原则:
    - 纯 numpy，零外部依赖
    - 信任衰减模型 (跨节点传播)
    - 本地权重 > 跨节点佐证
"""


import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class TrustLevel(str, Enum):
    """信任等级。"""

    UNTRUSTED = "untrusted"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED = "verified"


# =============================================================================
# TrustCertificate — 信任证书
# =============================================================================


@dataclass
class TrustCertificate:
    """跨节点信任证书。

    Attributes:
        cert_id: 证书 ID
        issuer: 颁发者节点 ID
        subject: 被认证节点 ID
        trust_score: 信任分数
        evidence_hash: 证据哈希
        issued_at: 颁发时间
        expires_at: 过期时间
    """

    cert_id: str = ""
    issuer: str = ""
    subject: str = ""
    trust_score: float = 0.0
    evidence_hash: str = ""
    issued_at: float = field(default_factory=time.time)
    expires_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.cert_id:
            raw = f"{self.issuer}:{self.subject}:{self.issued_at}"
            self.cert_id = hashlib.md5(raw.encode()).hexdigest()[:12]
        if self.expires_at == 0:
            self.expires_at = self.issued_at + 86400 * 30  # 30 天有效期

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at


# =============================================================================
# LocalTrust — 本地信任评估
# =============================================================================


class LocalTrust:
    """本地信任评估 — 基于证据的信任计算。"""

    def reason_with_trust(
        self, evidence: dict[str, Any], context: dict | None = None  # type: ignore
    ) -> dict[str, Any]:
        """基于证据的信任评估。

        Args:
            evidence: 证据数据
            context: 上下文

        Returns:
            信任评估结果
        """
        # 基于证据质量的简化信任计算
        consistency = evidence.get("consistency", 0.5)
        accuracy = evidence.get("accuracy", 0.5)
        coverage = evidence.get("coverage", 0.5)

        score = 0.35 * consistency + 0.40 * accuracy + 0.25 * coverage
        level = self._score_to_level(score)

        return {
            "trust": {"score": float(score), "level": level.value},
            "evidence_quality": {
                "consistency": consistency,
                "accuracy": accuracy,
                "coverage": coverage,
            },
        }

    @staticmethod
    def _score_to_level(score: float) -> TrustLevel:
        if score >= 0.85:
            return TrustLevel.VERIFIED
        if score >= 0.7:
            return TrustLevel.HIGH
        if score >= 0.5:
            return TrustLevel.MEDIUM
        if score >= 0.3:
            return TrustLevel.LOW
        return TrustLevel.UNTRUSTED


# =============================================================================
# FederatedTrust — 联邦信任框架
# =============================================================================


class FederatedTrust:
    """联邦信任框架 — 跨节点信任传递与联邦信任评估。

    信任模型:
      - 本地信任: 基于直接证据
      - 跨节点信任: 基于其他节点的证明
      - 联邦信任: 本地(60%) + 跨节点(40%) 加权

    信任传播衰减: 每跳衰减 decay_factor

    Args:
        local_trust: 本地信任评估器
        federation_protocol: 联邦协议实例
        decay_factor: 信任传播衰减因子
    """

    def __init__(
        self,
        local_trust: LocalTrust | None = None,
        federation_protocol: Any | None = None,
        decay_factor: float = 0.15,
    ):
        self._local_trust = local_trust or LocalTrust()
        self._protocol = federation_protocol
        self._decay_factor = decay_factor
        self._peer_trust_scores: dict[str, float] = {}
        self._certificates: dict[str, TrustCertificate] = {}
        self._trust_history: list[dict[str, Any]] = []

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def decay_factor(self) -> float:
        return self._decay_factor

    @property
    def n_trusted_peers(self) -> int:
        return sum(1 for s in self._peer_trust_scores.values() if s >= 0.5)

    # ── Trust Assessment ────────────────────────────────────────────────

    def assess_federation_trust(
        self, node_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
        """联邦信任评估。

        流程:
          1. 本地信任评估
          2. 收集跨节点信任证明
          3. 综合评估联邦信任

        Args:
            node_id: 目标节点 ID
            evidence: 证据数据

        Returns:
            联邦信任评估结果
        """
        # 本地评估
        local_assessment = self._local_trust.reason_with_trust(
            evidence, context={"source_node": node_id}
        )

        # 跨节点证明 (仿真)
        cross_attestations = self._collect_cross_attestations(node_id)
        cross_scores = [a.get("trust_score", 0.5) for a in cross_attestations]

        # 综合: 本地 60% + 跨节点 40%
        local_score = local_assessment["trust"]["score"]
        if cross_scores:
            cross_avg = float(np.mean(cross_scores))
            federation_trust = 0.6 * local_score + 0.4 * cross_avg
        else:
            federation_trust = local_score

        # 记录
        self._peer_trust_scores[node_id] = federation_trust
        result = {
            "node_id": node_id,
            "local_trust": local_score,
            "cross_trust_avg": float(np.mean(cross_scores)) if cross_scores else 0,
            "federation_trust": float(federation_trust),
            "n_cross_attestations": len(cross_attestations),
            "trust_level": LocalTrust._score_to_level(federation_trust).value,
        }
        self._trust_history.append(result)
        return result

    # ── Trust Propagation ───────────────────────────────────────────────

    def propagate_trust(
        self, source_cert: TrustCertificate, target_node: str
    ) -> dict[str, Any]:
        """联邦信任传播: 跨节点信任证书传递。

        每跳衰减 decay_factor。

        Args:
            source_cert: 源信任证书
            target_node: 目标节点

        Returns:
            传播结果
        """
        decay = 1 - self._decay_factor
        propagated_score = source_cert.trust_score * decay

        return {
            "source_cert": source_cert.cert_id,
            "target_node": target_node,
            "propagated_trust": float(propagated_score),
            "decay_applied": self._decay_factor,
            "original_trust": source_cert.trust_score,
        }

    # ── Certificate Management ──────────────────────────────────────────

    def issue_trust_certificate(
        self, node_id: str, evidence: dict[str, Any]) -> TrustCertificate:
        """颁发信任证书。

        Args:
            node_id: 被认证节点 ID
            evidence: 证据

        Returns:
            信任证书
        """
        assessment = self.assess_federation_trust(node_id, evidence)
        cert = TrustCertificate(
            issuer="local",
            subject=node_id,
            trust_score=assessment["federation_trust"],
            evidence_hash=hashlib.md5(str(evidence).encode()).hexdigest()[:8],
        )
        self._certificates[cert.cert_id] = cert
        return cert

    def verify_certificate(self, cert: TrustCertificate) -> dict[str, Any]:
        """验证信任证书。

        Args:
            cert: 信任证书

        Returns:
            验证结果
        """
        valid = not cert.is_expired and cert.trust_score > 0

        # 检查颁发者是否在已知证书中
        issuer_known = cert.issuer in self._peer_trust_scores or cert.issuer == "local"

        return {
            "valid": valid and issuer_known,
            "cert_id": cert.cert_id,
            "issuer_known": issuer_known,
            "not_expired": not cert.is_expired,
            "trust_score": cert.trust_score,
        }

    # ── Audit ───────────────────────────────────────────────────────────

    def audit_trust_state(self) -> dict[str, Any]:
        """审计当前信任状态。"""
        scores = list(self._peer_trust_scores.values())
        return {
            "n_peers_tracked": len(self._peer_trust_scores),
            "n_certificates": len(self._certificates),
            "n_trusted_peers": sum(1 for s in scores if s >= 0.5),
            "avg_trust": float(np.mean(scores)) if scores else 0,
            "min_trust": float(min(scores)) if scores else 0,
            "max_trust": float(max(scores)) if scores else 0,
        }

    # ── Internal Methods ────────────────────────────────────────────────

    def _collect_cross_attestations(self, node_id: str) -> list[dict[str, Any]]:
        """收集跨节点信任证明 (仿真模式)。"""
        attestations = []
        for peer_id, score in self._peer_trust_scores.items():
            if peer_id != node_id:
                attestations.append(
                    {"attester": peer_id, "trust_score": score}
                )
        return attestations

    def add_peer_trust(self, peer_id: str, trust_score: float) -> None:
        """手动设置对等节点信任分数 (仿真模式)。"""
        self._peer_trust_scores[peer_id] = trust_score
