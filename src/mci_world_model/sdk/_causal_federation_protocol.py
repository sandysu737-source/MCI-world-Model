from __future__ import annotations

"""MCI World Model v4.6.0 — CausalFederationProtocol 因果联邦协议
================================================================

多系统因果联邦通信标准 — 让多个独立的因果推理系统形成联邦。

核心能力:
    join_federation(endpoint, credentials)  — 加入因果联邦
    federated_query(query, strategy)        — 联邦因果查询
    send_message(msg_type, payload, target) — 发送联邦消息
    broadcast_query(query)                  — 广播查询到所有对等节点

设计原则:
    - 纯 numpy，零外部依赖
    - 12 种联邦消息类型覆盖完整通信场景
    - 4 种节点角色 (full/edge/witness/bridge)
    - 与 P10/P11 模块正交互操作
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
# Enums & Constants
# =============================================================================


class NodeRole(str, Enum):
    """联邦节点角色。"""

    FULL_NODE = "full_node"
    EDGE_NODE = "edge_node"
    WITNESS_NODE = "witness_node"
    BRIDGE_NODE = "bridge_node"


class FederationState(str, Enum):
    """联邦节点状态。"""

    DISCONNECTED = "disconnected"
    JOINING = "joining"
    ACTIVE = "active"
    SUSPENDED = "suspended"


class FederationMessageType(str, Enum):
    """联邦消息类型 — 12 种覆盖完整通信场景。"""

    FED_JOIN = "fed_join"
    FED_LEAVE = "fed_leave"
    FED_SYNC = "fed_sync"
    FED_QUERY = "fed_query"
    FED_RESULT = "fed_result"
    FED_DISCOVERY = "fed_discovery"
    FED_CONSENSUS = "fed_consensus"
    FED_VOTE = "fed_vote"
    FED_EVOLVE = "fed_evolve"
    FED_AUDIT = "fed_audit"
    FED_TRUST_RENEW = "fed_trust_renew"
    FED_EVIDENCE_SHARE = "fed_evidence_share"


# =============================================================================
# FederationMessage — 联邦消息
# =============================================================================


@dataclass
class FederationMessage:
    """联邦消息数据结构。

    Attributes:
        msg_type: 消息类型
        sender: 发送者节点 ID
        payload: 消息负载
        target: 目标节点 ID (None = 广播)
        timestamp: 时间戳
        msg_id: 消息唯一 ID
    """

    msg_type: FederationMessageType
    sender: str
    payload: dict[str, Any] = field(default_factory=dict)
    target: str | None = None
    timestamp: float = field(default_factory=time.time)
    msg_id: str = ""

    def __post_init__(self) -> None:
        if not self.msg_id:
            raw = f"{self.msg_type.value}:{self.sender}:{self.timestamp}"
            self.msg_id = hashlib.md5(raw.encode()).hexdigest()[:12]


# =============================================================================
# PeerInfo — 对等节点信息
# =============================================================================


@dataclass
class PeerInfo:
    """对等节点信息。

    Attributes:
        node_id: 节点 ID
        role: 节点角色
        capabilities: 能力声明
        trust_score: 信任分数
        last_seen: 最后在线时间
    """

    node_id: str
    role: NodeRole = NodeRole.FULL_NODE
    capabilities: dict[str, Any] = field(default_factory=dict)
    trust_score: float = 0.5
    last_seen: float = field(default_factory=time.time)


# =============================================================================
# FederationConsensus — 联邦共识引擎
# =============================================================================


class FederationConsensus:
    """联邦共识引擎 — 简化的 2/3 多数投票共识。"""

    def __init__(self, quorum_ratio: float = 2 / 3) -> None:
        if not 0.5 < quorum_ratio <= 1.0:
            raise ValueError(f"quorum_ratio 必须在 (0.5, 1.0], 当前 {quorum_ratio}")
        self._quorum_ratio = quorum_ratio

    def resolve_conflicts(self, conflicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """解决联邦因果发现中的冲突边。

        Args:
            conflicts: 冲突列表, 每项含 {edge, votes_for, votes_against}

        Returns:
            解决方案列表
        """
        resolutions = []
        for conflict in conflicts:
            votes_for = conflict.get("votes_for", 0)
            votes_against = conflict.get("votes_against", 0)
            total = votes_for + votes_against

            if total == 0:
                resolved = False
            else:
                # 2/3 多数投票
                resolved = votes_for / total >= self._quorum_ratio

            resolutions.append(
                {
                    "edge": conflict.get("edge"),
                    "resolved": resolved,
                    "votes_for": votes_for,
                    "votes_against": votes_against,
                    "consensus_ratio": votes_for / total if total > 0 else 0,
                }
            )
        return resolutions

    def check_quorum(self, n_agreeing: int, n_total: int) -> bool:
        """检查是否达到法定人数。"""
        if n_total == 0:
            return False
        return n_agreeing / n_total >= self._quorum_ratio


# =============================================================================
# CausalFederationProtocol — 因果联邦协议
# =============================================================================


class CausalFederationProtocol:
    """因果联邦协议 — 多个独立因果推理系统的联邦化通信标准。

    联邦节点角色:
        full_node:   完整节点 — 拥有完整因果图 + 推理能力
        edge_node:   边缘节点 — 轻量推理 + 联邦协作
        witness_node: 见证节点 — 仅验证 + 审计
        bridge_node: 桥接节点 — 跨联邦网关

    联邦消息类型 (12种):
        fed_join / fed_leave / fed_sync / fed_query / fed_result
        fed_discovery / fed_consensus / fed_vote / fed_evolve
        fed_audit / fed_trust_renew / fed_evidence_share
    """

    FEDERATION_VERSION = "1.0.0"

    def __init__(
        self,
        node_id: str,
        node_role: str = "full_node",
        federation_id: str = "default",
    ):
        if node_role not in [r.value for r in NodeRole]:
            raise ValueError(f"未知节点角色: {node_role}")
        self._node_id = node_id
        self._role = NodeRole(node_role)
        self._fed_id = federation_id
        self._peers: dict[str, PeerInfo] = {}
        self._state = FederationState.DISCONNECTED
        self._causal_graph_version = 0
        self._message_log: list[FederationMessage] = []
        self._consensus = FederationConsensus()
        self._capabilities: dict[str, Any] = {
            "supported_domains": [],
            "reasoning_depth": 0,
            "causal_graph_size": 0,
        }

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def role(self) -> NodeRole:
        return self._role

    @property
    def state(self) -> FederationState:
        return self._state

    @property
    def n_peers(self) -> int:
        return len(self._peers)

    @property
    def federation_id(self) -> str:
        return self._fed_id

    # ── Join / Leave ────────────────────────────────────────────────────

    def join_federation(
        self,
        federation_endpoint: str | None = None,
        credentials: dict | None = None,  # type: ignore
        existing_peers: dict[str, PeerInfo] | None = None,
    ) -> dict[str, Any]:
        """加入因果联邦。

        Args:
            federation_endpoint: 联邦端点 (仿真模式下可忽略)
            credentials: 信任证书
            existing_peers: 已有对等节点 (仿真模式下直接传入)

        Returns:
            加入结果 {joined, federation_id, n_peers, state}
        """
        self._state = FederationState.JOINING
        credentials = credentials or {}

        # 能力声明
        cap_decl = self._declare_capabilities()
        graph_hash = self._compute_graph_hash()

        # 在仿真模式下, 直接接受加入
        if existing_peers:
            self._peers.update(existing_peers)

        self._state = FederationState.ACTIVE

        join_msg = FederationMessage(
            msg_type=FederationMessageType.FED_JOIN,
            sender=self._node_id,
            payload={
                "protocol_version": self.FEDERATION_VERSION,
                "node_role": self._role.value,
                "capabilities": cap_decl,
                "causal_graph_hash": graph_hash,
                "trust_cert": credentials.get("trust_cert"),
            },
        )
        self._message_log.append(join_msg)

        return {
            "joined": True,
            "federation_id": self._fed_id,
            "n_peers": len(self._peers),
            "state": self._state.value,
            "capabilities_declared": cap_decl,
        }

    def leave_federation(self, reason: str = "voluntary") -> dict[str, Any]:
        """退出因果联邦。"""
        leave_msg = FederationMessage(
            msg_type=FederationMessageType.FED_LEAVE,
            sender=self._node_id,
            payload={"reason": reason},
        )
        self._message_log.append(leave_msg)
        self._state = FederationState.DISCONNECTED
        self._peers.clear()
        return {"left": True, "reason": reason}

    # ── Query ───────────────────────────────────────────────────────────

    def federated_query(
        self, query: dict[str, Any], strategy: str = "broadcast"
    ) -> dict[str, Any]:
        """联邦因果查询。

        Args:
            query: 查询内容
            strategy: 查询策略 (broadcast/targeted/hierarchical)

        Returns:
            联邦查询结果
        """
        local_result = self._local_reason(query)

        if strategy == "broadcast":
            fed_results = self.broadcast_query(query)
        elif strategy == "targeted":
            fed_results = self._targeted_query(query)
        elif strategy == "hierarchical":
            fed_results = self._hierarchical_query(query)
        else:
            fed_results = []

        merged = self._merge_federated_results(local_result, fed_results)

        return {
            "query": query,
            "local_result": local_result,
            "federated_results": fed_results,
            "merged_result": merged,
            "n_peers_queried": len(fed_results),
            "consensus_level": merged.get("consensus_level", 0),
        }

    def broadcast_query(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        """广播查询到所有对等节点 (仿真模式)。"""
        results = []
        for peer_id, peer in self._peers.items():
            if peer.role in (NodeRole.FULL_NODE, NodeRole.BRIDGE_NODE):
                # 仿真: 基于对等节点能力生成模拟响应
                result = self._simulate_peer_response(peer_id, query)
                results.append(result)

        msg = FederationMessage(
            msg_type=FederationMessageType.FED_QUERY,
            sender=self._node_id,
            payload=query,
        )
        self._message_log.append(msg)
        return results

    def send_message(
        self,
        msg_type: str,
        payload: dict[str, Any],
        target: str | None = None,
    ) -> dict | None:  # type: ignore
        """发送联邦消息。

        Args:
            msg_type: 消息类型
            payload: 消息负载
            target: 目标节点 (None = 广播)

        Returns:
            消息发送确认
        """
        try:
            mt = FederationMessageType(msg_type)
        except ValueError:
            raise ValueError(f"未知消息类型: {msg_type}") from None

        msg = FederationMessage(
            msg_type=mt,
            sender=self._node_id,
            payload=payload,
            target=target,
        )
        self._message_log.append(msg)

        if target and target in self._peers:
            peer = self._peers[target]
            return {
                "delivered": True,
                "target": target,
                "peer_role": peer.role.value,
            }
        elif target is None:
            return {"delivered": True, "broadcast": True, "n_recipients": len(self._peers)}
        return {"delivered": False, "reason": "target_not_found"}

    # ── Evidence Sharing ────────────────────────────────────────────────

    def share_evidence(
        self, evidence: dict[str, Any], target_nodes: list[str] | None = None
    ) -> dict[str, Any]:
        """联邦证据共享。

        Args:
            evidence: 因果证据数据
            target_nodes: 目标节点列表 (None = 广播)

        Returns:
            共享结果
        """
        msg = FederationMessage(
            msg_type=FederationMessageType.FED_EVIDENCE_SHARE,
            sender=self._node_id,
            payload=evidence,
        )
        self._message_log.append(msg)

        if target_nodes is None:
            # 广播
            recipients = [
                pid
                for pid, p in self._peers.items()
                if p.role in (NodeRole.FULL_NODE, NodeRole.BRIDGE_NODE)
            ]
        else:
            recipients = [n for n in target_nodes if n in self._peers]

        return {
            "shared": True,
            "evidence_id": hashlib.md5(str(evidence).encode()).hexdigest()[:8],
            "n_recipients": len(recipients),
            "recipients": recipients,
        }

    # ── Consensus ───────────────────────────────────────────────────────

    def request_consensus(self, proposal: dict[str, Any]) -> dict[str, Any]:
        """请求联邦共识投票。

        Args:
            proposal: 共识提案

        Returns:
            共识结果
        """
        msg = FederationMessage(
            msg_type=FederationMessageType.FED_CONSENSUS,
            sender=self._node_id,
            payload=proposal,
        )
        self._message_log.append(msg)

        # 仿真: 模拟投票
        n_voters = len(self._peers) + 1
        simulated_for = int(n_voters * 0.75)  # 模拟 75% 赞成
        quorum_met = self._consensus.check_quorum(simulated_for, n_voters)

        return {
            "proposal_id": hashlib.md5(str(proposal).encode()).hexdigest()[:8],
            "n_voters": n_voters,
            "votes_for": simulated_for,
            "votes_against": n_voters - simulated_for,
            "quorum_met": quorum_met,
        }

    # ── Internal Methods ────────────────────────────────────────────────

    def _declare_capabilities(self) -> dict[str, Any]:
        """声明节点能力。"""
        return {
            "causal_graph_size": self._causal_graph_version,
            "supported_domains": self._capabilities.get("supported_domains", ["general"]),
            "reasoning_depth": self._capabilities.get("reasoning_depth", 5),
            "trust_score": 0.85,
            "node_role": self._role.value,
        }

    def _compute_graph_hash(self) -> str:
        """计算因果图哈希。"""
        raw = f"{self._node_id}:{self._causal_graph_version}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _local_reason(self, query: dict[str, Any]) -> dict[str, Any]:
        """本地推理 (简化版)。"""
        return {
            "conclusion": {"local_confidence": 0.7},
            "trust_score": 0.85,
            "source": self._node_id,
            "agrees_with_merge": True,
        }

    def _targeted_query(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        """定向查询: 路由到最相关领域节点。"""
        domain = query.get("domain", "general")
        results = []
        for peer_id, peer in self._peers.items():
            if domain in peer.capabilities.get("supported_domains", []):
                results.append(self._simulate_peer_response(peer_id, query))
        # 如果没有匹配, 退回广播
        if not results:
            results = self.broadcast_query(query)
        return results

    def _hierarchical_query(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        """层级查询: 边缘 → 完整 → 桥接。"""
        results = []
        # 第一层: 边缘节点
        for pid, peer in self._peers.items():
            if peer.role == NodeRole.EDGE_NODE:
                results.append(self._simulate_peer_response(pid, query))
        # 第二层: 完整节点
        for pid, peer in self._peers.items():
            if peer.role == NodeRole.FULL_NODE:
                results.append(self._simulate_peer_response(pid, query))
        # 第三层: 桥接节点
        for pid, peer in self._peers.items():
            if peer.role == NodeRole.BRIDGE_NODE:
                results.append(self._simulate_peer_response(pid, query))
        return results

    def _simulate_peer_response(self, peer_id: str, query: dict[str, Any]) -> dict[str, Any]:
        """仿真模式下模拟对等节点响应。"""
        peer = self._peers[peer_id]
        return {
            "conclusion": {
                "federated_confidence": np.random.uniform(0.5, 0.95),
                "domain": query.get("domain", "general"),
            },
            "trust_score": peer.trust_score,
            "source": peer_id,
            "agrees_with_merge": True,
        }

    def _merge_federated_results(
        self, local: dict[str, Any], federated: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """联邦结果合并: 加权投票 + 冲突检测。"""
        all_results = [local, *federated]
        weights = [r.get("trust_score", 0.5) for r in all_results]
        total_weight = sum(weights)
        if total_weight == 0:
            total_weight = 1.0

        merged_conclusion: dict[str, float] = {}
        for r, w in zip(all_results, weights):
            for key, val in r.get("conclusion", {}).items():
                if isinstance(val, (int, float)):
                    merged_conclusion[key] = merged_conclusion.get(key, 0) + val * w
        for key in merged_conclusion:
            merged_conclusion[key] /= total_weight

        consensus_level = self._compute_consensus(all_results)

        return {
            "conclusion": merged_conclusion,
            "consensus_level": consensus_level,
            "n_agreeing": sum(1 for r in all_results if r.get("agrees_with_merge")),
            "n_total": len(all_results),
        }

    def _compute_consensus(self, results: list[dict[str, Any]]) -> float:
        """计算共识水平。"""
        if not results:
            return 0.0
        agreeing = sum(1 for r in results if r.get("agrees_with_merge"))
        return agreeing / len(results)

    def get_message_log(self, msg_type: str | None = None) -> list[FederationMessage]:
        """获取消息日志。"""
        if msg_type:
            mt = FederationMessageType(msg_type)
            return [m for m in self._message_log if m.msg_type == mt]
        return list(self._message_log)

    def add_peer(self, peer_id: str, role: str = "full_node", **kwargs: Any) -> None:
        """手动添加对等节点 (仿真模式)。"""
        self._peers[peer_id] = PeerInfo(
            node_id=peer_id,
            role=NodeRole(role),
            capabilities=kwargs.get("capabilities", {}),
            trust_score=kwargs.get("trust_score", 0.5),
        )

    def update_capabilities(self, capabilities: dict[str, Any]) -> None:
        """更新节点能力声明。"""
        self._capabilities.update(capabilities)
