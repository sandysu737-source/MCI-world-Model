from __future__ import annotations

"""MCI World Model v4.6.0 — FederationAudit 联邦审计体系
==========================================================

联邦因果推理的审计与治理 — 确保联邦操作的合规性与可追溯性。

核心能力:
    audit_federation_operation(operation)     — 审计联邦操作
    audit_trust_state()                       — 审计信任状态
    audit_consciousness_state()               — 审计意识状态
    generate_audit_report()                   — 生成审计报告

设计原则:
    - 纯 numpy，零外部依赖
    - 所有联邦操作可追溯
    - 审计结果不可篡改
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


class AuditSeverity(str, Enum):
    """审计严重程度。"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AuditStatus(str, Enum):
    """审计状态。"""

    PASS = "pass"
    FAIL = "fail"
    PENDING = "pending"


# =============================================================================
# AuditEntry — 审计条目
# =============================================================================


@dataclass
class AuditEntry:
    """审计条目。

    Attributes:
        entry_id: 条目 ID
        operation: 操作类型
        severity: 严重程度
        status: 审计状态
        details: 详细信息
        timestamp: 时间戳
    """

    entry_id: str = ""
    operation: str = ""
    severity: AuditSeverity = AuditSeverity.INFO
    status: AuditStatus = AuditStatus.PASS
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.entry_id:
            raw = f"{self.operation}:{self.timestamp}"
            self.entry_id = hashlib.md5(raw.encode()).hexdigest()[:12]


# =============================================================================
# FederationAudit — 联邦审计体系
# =============================================================================


class FederationAudit:
    """联邦审计体系 — 审计联邦因果推理的操作。

    职责:
      - 操作审计: 检查联邦操作的合规性
      - 信任审计: 检查信任状态的健康性
      - 意识审计: 检查联邦意识的一致性
      - 报告生成: 生成不可篡改的审计报告

    Args:
        max_entries: 最大审计条目数
    """

    def __init__(self, max_entries: int = 10000) -> None:
        self._entries: list[AuditEntry] = []
        self._max_entries = max_entries
        self._report_hashes: list[str] = []

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def n_entries(self) -> int:
        return len(self._entries)

    @property
    def n_critical(self) -> int:
        return sum(1 for e in self._entries if e.severity == AuditSeverity.CRITICAL)

    @property
    def n_failures(self) -> int:
        return sum(1 for e in self._entries if e.status == AuditStatus.FAIL)

    # ── Operation Audit ─────────────────────────────────────────────────

    def audit_federation_operation(self, operation: dict[str, Any]) -> AuditEntry:
        """审计联邦操作。

        Args:
            operation: 操作描述 {type, node_id, details, ...}

        Returns:
            审计条目
        """
        op_type = operation.get("type", "unknown")
        node_id = operation.get("node_id", "unknown")
        severity = AuditSeverity.INFO
        status = AuditStatus.PASS

        # 检查1: 操作类型是否合法
        valid_types = {
            "join", "leave", "query", "discovery",
            "consensus", "evolve", "evidence_share", "sync",
        }
        if op_type not in valid_types:
            severity = AuditSeverity.WARNING
            status = AuditStatus.FAIL

        # 检查2: 进化操作需要共识
        if op_type == "evolve":
            consensus = operation.get("consensus_reached", False)
            if not consensus:
                severity = AuditSeverity.CRITICAL
                status = AuditStatus.FAIL

        entry = AuditEntry(
            operation=op_type,
            severity=severity,
            status=status,
            details={
                "node_id": node_id,
                "operation": operation,
            },
        )

        self._add_entry(entry)
        return entry

    # ── Trust Audit ─────────────────────────────────────────────────────

    def audit_trust_state(self, trust_data: dict[str, Any]) -> AuditEntry:
        """审计信任状态。

        Args:
            trust_data: 信任数据 {peer_scores, certificates, ...}

        Returns:
            审计条目
        """
        severity = AuditSeverity.INFO
        status = AuditStatus.PASS
        details: dict[str, Any] = {}

        peer_scores = trust_data.get("peer_scores", {})
        if peer_scores:
            scores = list(peer_scores.values())
            avg_trust = float(np.mean(scores))
            min_trust = float(min(scores))
            details = {
                "avg_trust": avg_trust,
                "min_trust": min_trust,
                "n_peers": len(scores),
            }

            # 检查: 最低信任过低
            if min_trust < 0.3:
                severity = AuditSeverity.WARNING
                status = AuditStatus.FAIL
                details["issue"] = f"Low trust detected: {min_trust:.2f}"

        entry = AuditEntry(
            operation="trust_audit",
            severity=severity,
            status=status,
            details=details,
        )
        self._add_entry(entry)
        return entry

    # ── Consciousness Audit ─────────────────────────────────────────────

    def audit_consciousness_state(self, consciousness_data: dict[str, Any]) -> AuditEntry:
        """审计联邦意识状态。

        Args:
            consciousness_data: 意识数据 {awareness_state, n_nodes, ...}

        Returns:
            审计条目
        """
        severity = AuditSeverity.INFO
        status = AuditStatus.PASS
        details: dict[str, Any] = consciousness_data.copy()

        state = consciousness_data.get("awareness_state", "unknown")
        n_nodes = consciousness_data.get("n_nodes", 0)

        # 检查: 联邦应该至少有2个节点才有效
        if n_nodes < 2 and state in ("synchronized", "emergent"):
            severity = AuditSeverity.WARNING
            status = AuditStatus.FAIL
            details["issue"] = "Federation claims synchronized but has <2 nodes"

        entry = AuditEntry(
            operation="consciousness_audit",
            severity=severity,
            status=status,
            details=details,
        )
        self._add_entry(entry)
        return entry

    # ── Report Generation ───────────────────────────────────────────────

    def generate_audit_report(self) -> dict[str, Any]:
        """生成不可篡改的审计报告。"""
        report = {
            "report_id": hashlib.md5(
                f"audit:{time.time()}:{len(self._entries)}".encode()
            ).hexdigest()[:12],
            "timestamp": time.time(),
            "n_entries": len(self._entries),
            "n_critical": self.n_critical,
            "n_failures": self.n_failures,
            "pass_rate": (
                1 - self.n_failures / max(len(self._entries), 1)
            ),
            "entries_summary": [
                {
                    "id": e.entry_id,
                    "op": e.operation,
                    "severity": e.severity.value,
                    "status": e.status.value,
                }
                for e in self._entries[-100:]  # 最近 100 条
            ],
        }

        # 报告哈希 (不可篡改)
        report_hash = hashlib.md5(str(report).encode()).hexdigest()
        self._report_hashes.append(report_hash)
        report["hash"] = report_hash

        return report

    # ── Internal Methods ────────────────────────────────────────────────

    def _add_entry(self, entry: AuditEntry) -> None:
        """添加审计条目 (FIFO)。"""
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]
