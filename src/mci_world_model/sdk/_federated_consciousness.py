from __future__ import annotations

"""MCI World Model v4.6.0 — FederatedCausalConsciousness 联邦因果意识
======================================================================

跨节点的共享因果意识 — 从单系统意识扩展为联邦意识。

核心能力:
    synchronize_awareness()                    — 联邦意识同步
    federated_reflect(reasoning_episode)       — 联邦反思
    detect_federation_anomaly(fed_model)       — 联邦异常检测
    propose_federation_evolution(proposal)     — 联邦进化提案

意识状态:
    isolated → aware → synchronized → emergent

设计原则:
    - 纯 numpy，零外部依赖
    - 与 CausalFederationProtocol 正交互操作
    - 联邦共识投票 (2/3 多数)
"""


import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class FederationAwarenessState(str, Enum):
    """联邦意识状态。"""

    ISOLATED = "isolated"
    AWARE = "aware"
    SYNCHRONIZED = "synchronized"
    EMERGENT = "emergent"


# =============================================================================
# SelfModel — 自我模型
# =============================================================================


@dataclass
class SelfModel:
    """节点自我模型 — 描述自身的推理能力和局限。

    Attributes:
        node_id: 节点 ID
        domains: 覆盖领域
        confidence: 整体置信度
        limitations: 已知局限
    """

    node_id: str
    domains: list[str] = field(default_factory=list)
    confidence: float = 0.5
    limitations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"SelfModel({self.node_id}): "
            f"domains={self.domains}, "
            f"confidence={self.confidence:.2f}, "
            f"limitations={len(self.limitations)}"
        )


# =============================================================================
# FederationSelfModel — 联邦自我模型
# =============================================================================


@dataclass
class FederationSelfModel:
    """联邦自我模型 — 聚合所有节点的自我模型。

    Attributes:
        n_nodes: 节点数量
        combined_domains: 联邦覆盖的所有领域
        avg_confidence: 平均置信度
        combined_limitations: 所有节点的局限合集
    """

    n_nodes: int = 0
    combined_domains: list[str] = field(default_factory=list)
    avg_confidence: float = 0.0
    combined_limitations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"FederationSelfModel: {self.n_nodes} nodes, "
            f"covering {len(self.combined_domains)} domains, "
            f"avg_confidence={self.avg_confidence:.2f}"
        )


# =============================================================================
# ReflectionResult — 反思结果
# =============================================================================


@dataclass
class ReflectionResult:
    """反思结果。

    Attributes:
        source: 反思来源节点
        issues: 发现的问题列表
        improvements: 改进建议
        confidence: 反思置信度
    """

    source: str
    issues: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    confidence: float = 0.5


# =============================================================================
# FederatedCausalConsciousness — 联邦因果意识
# =============================================================================


class FederatedCausalConsciousness:
    """联邦因果意识 — 跨节点的共享因果意识。

    将 P11 的单系统因果意识扩展为联邦层面的共享意识。
    从 isolated (孤立) → aware (觉察) → synchronized (同步) → emergent (涌现)。

    Args:
        local_consciousness: 本地因果意识 (P11 AutonomousCausalConsciousness 兼容)
        federation_protocol: 因果联邦协议实例
    """

    def __init__(
        self,
        local_consciousness: Any | None = None,
        federation_protocol: Any | None = None,
    ):
        self._local = local_consciousness
        self._protocol = federation_protocol
        self._awareness_state = FederationAwarenessState.ISOLATED
        self._federation_self_model = FederationSelfModel()
        self._peer_models: dict[str, SelfModel] = {}
        self._reflection_history: list[dict[str, Any]] = []
        self._evolution_proposals: list[dict[str, Any]] = []
        self._local_self_model = SelfModel(node_id="local")

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def awareness_state(self) -> FederationAwarenessState:
        return self._awareness_state

    @property
    def federation_self_model(self) -> FederationSelfModel:
        return self._federation_self_model

    @property
    def n_peer_models(self) -> int:
        return len(self._peer_models)

    # ── Synchronization ─────────────────────────────────────────────────

    def synchronize_awareness(self) -> dict[str, Any]:
        """联邦意识同步。

        流程:
          1. 各节点共享自我模型摘要
          2. 识别联邦层面的推理模式异常
          3. 建立联邦自我模型
          4. 进入 synchronized 状态

        Returns:
            同步结果 {federation_awareness, n_nodes_aware, ...}
        """
        # 收集对等节点自我模型 (仿真模式)
        peer_models = self._collect_peer_models()

        # 联邦自我模型构建
        fed_model = self._build_federation_self_model(self._local_self_model, peer_models)
        self._federation_self_model = fed_model

        # 联邦异常检测
        anomaly = self._detect_federation_anomaly(fed_model)

        # 状态转换
        if len(peer_models) > 0 and anomaly["detected"] is False:
            self._awareness_state = FederationAwarenessState.SYNCHRONIZED
        elif len(peer_models) > 0:
            self._awareness_state = FederationAwarenessState.AWARE
        else:
            self._awareness_state = FederationAwarenessState.ISOLATED

        # 检查涌现
        if self._awareness_state == FederationAwarenessState.SYNCHRONIZED:
            emergence = self._check_emergence(fed_model)
            if emergence["detected"]:
                self._awareness_state = FederationAwarenessState.EMERGENT

        return {
            "federation_awareness": self._awareness_state.value,
            "n_nodes_aware": len(peer_models) + 1,
            "federation_anomaly": anomaly,
            "fed_self_model_summary": fed_model.summary(),
        }

    # ── Reflection ──────────────────────────────────────────────────────

    def federated_reflect(self, reasoning_episode: dict[str, Any]) -> dict[str, Any]:
        """联邦反思: 多节点协同审视推理过程。

        流程:
          1. 本地反思
          2. 请求跨节点反思
          3. 合并反思结论
          4. 形成联邦改进方案

        Args:
            reasoning_episode: 推理过程记录

        Returns:
            联邦反思结果
        """
        # 本地反思
        local_reflection = self._local_reflect(reasoning_episode)

        # 跨节点反思 (仿真)
        cross_reflections = self._collect_cross_reflections(reasoning_episode)

        # 合并
        merged = self._merge_reflections(local_reflection, cross_reflections)

        # 识别共识问题
        consensus_issues = self._identify_consensus_issues(local_reflection, cross_reflections)

        result = {
            "local_reflection": local_reflection,
            "cross_reflections": cross_reflections,
            "federation_improvements": merged,
            "consensus_on_issues": consensus_issues,
            "n_nodes_participated": len(cross_reflections) + 1,
        }
        self._reflection_history.append(result)
        return result

    # ── Evolution ───────────────────────────────────────────────────────

    def propose_federation_evolution(self, proposal: dict[str, Any]) -> dict[str, Any]:
        """联邦进化提案。

        Args:
            proposal: 进化提案 {type, target, description}

        Returns:
            提案结果
        """
        if self._awareness_state not in (
            FederationAwarenessState.SYNCHRONIZED,
            FederationAwarenessState.EMERGENT,
        ):
            return {
                "accepted": False,
                "reason": "federation_not_synchronized",
                "current_state": self._awareness_state.value,
            }

        # 安全约束: 联邦进化需 2/3 多数投票通过
        proposal_record = {
            "proposal_id": f"evo_{len(self._evolution_proposals)}",
            "proposal": proposal,
            "proposer": "local",
            "timestamp": len(self._evolution_proposals),
        }
        self._evolution_proposals.append(proposal_record)

        # 仿真投票
        n_nodes = self.n_peer_models + 1
        votes_for = int(n_nodes * 0.75)  # 模拟 75% 赞成
        quorum_ratio = 2 / 3
        accepted = votes_for / n_nodes >= quorum_ratio

        return {
            "accepted": accepted,
            "proposal_id": proposal_record["proposal_id"],
            "votes_for": votes_for,
            "votes_against": n_nodes - votes_for,
            "quorum_required": f"{quorum_ratio:.0%}",
        }

    # ── Anomaly Detection ───────────────────────────────────────────────

    def _detect_federation_anomaly(self, fed_model: FederationSelfModel) -> dict[str, Any]:
        """联邦异常检测。"""
        anomalies = []

        # 检测1: 置信度异常低
        if fed_model.avg_confidence < 0.3:
            anomalies.append(f"Low federation confidence: {fed_model.avg_confidence:.2f}")

        # 检测2: 领域覆盖空洞
        if len(fed_model.combined_domains) < 2:
            anomalies.append("Insufficient domain coverage")

        # 检测3: 局限性过多
        if len(fed_model.combined_limitations) > fed_model.n_nodes * 5:
            anomalies.append("Excessive combined limitations")

        return {
            "detected": len(anomalies) > 0,
            "anomalies": anomalies,
            "n_anomalies": len(anomalies),
        }

    # ── Internal Methods ────────────────────────────────────────────────

    def _collect_peer_models(self) -> list[SelfModel]:
        """收集对等节点自我模型 (仿真模式)。"""
        return list(self._peer_models.values())

    def _build_federation_self_model(self, local: SelfModel, peers: list[SelfModel]) -> FederationSelfModel:
        """构建联邦自我模型。"""
        all_models = [local, *peers]
        all_domains = list({d for m in all_models for d in m.domains})
        avg_conf = np.mean([m.confidence for m in all_models]) if all_models else 0
        all_limits = list({lim for m in all_models for lim in m.limitations})

        return FederationSelfModel(
            n_nodes=len(all_models),
            combined_domains=all_domains,
            avg_confidence=float(avg_conf),
            combined_limitations=all_limits,
        )

    def _check_emergence(self, fed_model: FederationSelfModel) -> dict[str, Any]:
        """检查联邦涌现。"""
        # 涌现条件: 足够多的节点 + 足够高的置信度 + 多领域覆盖
        detected = fed_model.n_nodes >= 3 and fed_model.avg_confidence >= 0.6 and len(fed_model.combined_domains) >= 3
        return {
            "detected": detected,
            "emergence_indicators": {
                "n_nodes": fed_model.n_nodes,
                "avg_confidence": fed_model.avg_confidence,
                "n_domains": len(fed_model.combined_domains),
            },
        }

    def _local_reflect(self, episode: dict[str, Any]) -> ReflectionResult:
        """本地反思。"""
        issues = []
        if episode.get("confidence", 1.0) < 0.5:
            issues.append("Low confidence in reasoning")
        if episode.get("contradictions", 0) > 0:
            issues.append(f"Found {episode['contradictions']} contradictions")

        return ReflectionResult(
            source="local",
            issues=issues,
            improvements=[f"Address: {i}" for i in issues],
            confidence=episode.get("confidence", 0.7),
        )

    def _collect_cross_reflections(self, episode: dict[str, Any]) -> list[ReflectionResult]:
        """收集跨节点反思 (仿真)。"""
        results = []
        for peer_id, peer_model in self._peer_models.items():
            # 模拟对等节点反思
            n_issues = np.random.randint(0, 3)
            results.append(
                ReflectionResult(
                    source=peer_id,
                    issues=[f"Peer concern {i}" for i in range(n_issues)],
                    improvements=[f"Peer improvement {i}" for i in range(n_issues)],
                    confidence=np.random.uniform(0.5, 0.9),
                )
            )
        return results

    def _merge_reflections(
        self,
        local: ReflectionResult,
        cross: list[ReflectionResult],
    ) -> list[str]:
        """合并反思结论。"""
        all_improvements = list(local.improvements)
        for r in cross:
            all_improvements.extend(r.improvements)
        return list(set(all_improvements))

    def _identify_consensus_issues(
        self,
        local: ReflectionResult,
        cross: list[ReflectionResult],
    ) -> list[str]:
        """识别共识问题。"""
        local_issues = set(local.issues)
        consensus = []  # type: ignore
        for r in cross:
            overlap = local_issues.intersection(set(r.issues))
            consensus.extend(overlap)
        return list(set(consensus))

    # ── Simulation Helpers ──────────────────────────────────────────────

    def add_peer_model(
        self,
        peer_id: str,
        domains: list[str] | None = None,
        confidence: float = 0.5,
        limitations: list[str] | None = None,
    ) -> None:
        """添加对等节点自我模型 (仿真模式)。"""
        self._peer_models[peer_id] = SelfModel(
            node_id=peer_id,
            domains=domains or ["general"],
            confidence=confidence,
            limitations=limitations or [],
        )

    def set_local_self_model(
        self,
        domains: list[str] | None = None,
        confidence: float = 0.5,
        limitations: list[str] | None = None,
    ) -> None:
        """设置本地自我模型。"""
        self._local_self_model = SelfModel(
            node_id="local",
            domains=domains or ["general"],
            confidence=confidence,
            limitations=limitations or [],
        )
