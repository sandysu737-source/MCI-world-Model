from __future__ import annotations

"""MCI World Model v4.6.0 — CausalFederationArchitecture 因果联邦架构
======================================================================

多节点因果推理的分布式架构 — 因果知识分片、副本复制、联邦因果发现。

核心能力:
    distribute_causal_knowledge(graph, domain)              — 分布式因果知识存储
    federated_causal_discovery(domain, data_sources)        — 联邦因果发现
    retrieve_federated_knowledge(query)                     — 联邦知识检索

设计原则:
    - 纯 numpy，零外部依赖
    - 因果图分片 + 副本复制 (容错)
    - 最终一致性模型
"""


import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CausalShard — 因果图分片
# =============================================================================


@dataclass
class CausalShard:
    """因果图分片。

    Attributes:
        shard_id: 分片 ID
        domain: 所属领域
        variables: 变量列表
        edges: 因果边列表
        assigned_nodes: 分配的存储节点
    """

    shard_id: str
    domain: str
    variables: list[str] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    assigned_nodes: list[str] = field(default_factory=list)


# =============================================================================
# FederationConsensus — 联邦共识 (简化版)
# =============================================================================


class FederationConsensus:
    """联邦共识引擎 — 简化的多数投票共识。"""

    def __init__(self, quorum_ratio: float = 2 / 3) -> None:
        if not 0.5 < quorum_ratio <= 1.0:
            raise ValueError(f"quorum_ratio 必须在 (0.5, 1.0], 当前 {quorum_ratio}")
        self._quorum_ratio = quorum_ratio

    def resolve_conflicts(self, conflicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """解决联邦因果发现中的冲突边。"""
        resolutions = []
        for conflict in conflicts:
            votes_for = conflict.get("votes_for", 0)
            votes_against = conflict.get("votes_against", 0)
            total = votes_for + votes_against
            resolved = (votes_for / total >= self._quorum_ratio) if total > 0 else False

            resolutions.append(
                {
                    "edge": conflict.get("edge"),
                    "resolved": resolved,
                    "votes_for": votes_for,
                    "votes_against": votes_against,
                }
            )
        return resolutions


# =============================================================================
# CausalFederationArchitecture — 因果联邦架构
# =============================================================================


class CausalFederationArchitecture:
    """因果联邦架构 — 多节点因果推理的分布式架构。

    职责:
      - 因果图分片 (按领域/变量组)
      - 副本复制 (容错)
      - 一致性保证 (最终一致性)
      - 联邦因果发现 (多节点协同)

    Args:
        protocol: CausalFederationProtocol 实例
        consensus_engine: 共识引擎 (默认 FederationConsensus)
        replication_factor: 副本因子
    """

    def __init__(
        self,
        protocol: Any | None = None,
        consensus_engine: FederationConsensus | None = None,
        replication_factor: int = 3,
    ):
        self._protocol = protocol
        self._consensus = consensus_engine or FederationConsensus()
        self._replication_factor = max(1, replication_factor)
        self._shard_map: dict[str, list[CausalShard]] = {}  # domain → shards
        self._node_shard_assignment: dict[str, list[str]] = {}  # node_id → shard_ids
        self._federation_nodes: list[str] = []

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def n_domains(self) -> int:
        return len(self._shard_map)

    @property
    def n_shards(self) -> int:
        return sum(len(shards) for shards in self._shard_map.values())

    @property
    def replication_factor(self) -> int:
        return self._replication_factor

    # ── Knowledge Distribution ──────────────────────────────────────────

    def distribute_causal_knowledge(
        self, causal_graph: dict[str, Any], domain: str
    ) -> dict[str, Any]:
        """分布式因果知识存储。

        流程:
          1. 因果图分片 (按变量组)
          2. 副本复制 (容错)
          3. 分配到节点

        Args:
            causal_graph: 因果图 {nodes: [...], edges: [...]}
            domain: 领域名称

        Returns:
            分配结果 {domain, n_shards, assignments, ...}
        """
        # 分片
        shards = self._shard_causal_graph(causal_graph, domain)

        # 分配分片到节点
        assignments: dict[str, list[str]] = {}
        for shard in shards:
            target_nodes = self._select_shard_nodes(shard.shard_id)
            shard.assigned_nodes = target_nodes
            assignments[shard.shard_id] = target_nodes

            for node_id in target_nodes:
                if node_id not in self._node_shard_assignment:
                    self._node_shard_assignment[node_id] = []
                self._node_shard_assignment[node_id].append(shard.shard_id)

        self._shard_map[domain] = shards

        return {
            "domain": domain,
            "n_shards": len(shards),
            "assignments": assignments,
            "replication_factor": self._replication_factor,
        }

    # ── Federated Discovery ─────────────────────────────────────────────

    def federated_causal_discovery(
        self,
        domain: str,
        data_sources: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """联邦因果发现: 多节点协同发现因果结构。

        流程:
          1. 各节点本地因果发现
          2. 因果结构合并
          3. 冲突边消解 (联邦共识)
          4. 联邦因果图更新

        Args:
            domain: 目标领域
            data_sources: 各节点的数据 (node_id → data)

        Returns:
            发现结果 {domain, merged_dag, n_conflicts, ...}
        """
        if data_sources is None:
            data_sources = {}

        # 各节点本地发现 (仿真)
        local_discoveries: dict[str, dict[str, Any]] = {}
        for node_id, data in data_sources.items():
            local_discoveries[node_id] = self._local_discovery(data, domain)

        # 如果没有数据源, 使用已注册节点模拟
        if not data_sources:
            for node_id in self._federation_nodes:
                n_vars = np.random.randint(3, 8)
                local_discoveries[node_id] = {
                    "dag": {
                        "nodes": [f"var_{i}" for i in range(n_vars)],
                        "edges": [
                            {"from": f"var_{i}", "to": f"var_{i+1}"}
                            for i in range(n_vars - 1)
                        ],
                    },
                    "confidence": np.random.uniform(0.5, 0.9),
                }

        # 合并因果结构
        merged_dag = self._merge_causal_structures(local_discoveries)

        # 检测冲突
        conflicts = self._detect_dag_conflicts(local_discoveries)

        # 解决冲突
        resolved = []
        if conflicts:
            resolved = self._consensus.resolve_conflicts(conflicts)
            merged_dag = self._apply_resolutions(merged_dag, resolved)

        return {
            "domain": domain,
            "merged_dag": merged_dag,
            "n_local_discoveries": len(local_discoveries),
            "n_conflicts": len(conflicts),
            "consensus_reached": all(r.get("resolved", False) for r in resolved)
            if resolved
            else True,
        }

    # ── Knowledge Retrieval ─────────────────────────────────────────────

    def retrieve_federated_knowledge(self, query: dict[str, Any]) -> dict[str, Any]:
        """联邦知识检索。

        Args:
            query: 查询 {domain, variables, ...}

        Returns:
            检索结果
        """
        domain = query.get("domain", "")
        if domain not in self._shard_map:
            return {"found": False, "reason": "domain_not_found"}

        relevant_shards = []
        variables = set(query.get("variables", []))
        for shard in self._shard_map[domain]:
            if not variables or variables.intersection(set(shard.variables)):
                relevant_shards.append(shard)

        return {
            "found": len(relevant_shards) > 0,
            "domain": domain,
            "n_shards_matched": len(relevant_shards),
            "shard_ids": [s.shard_id for s in relevant_shards],
        }

    # ── Node Management ─────────────────────────────────────────────────

    def register_node(self, node_id: str) -> None:
        """注册联邦节点。"""
        if node_id not in self._federation_nodes:
            self._federation_nodes.append(node_id)

    def unregister_node(self, node_id: str) -> None:
        """注销联邦节点。"""
        if node_id in self._federation_nodes:
            self._federation_nodes.remove(node_id)
        self._node_shard_assignment.pop(node_id, None)

    # ── Internal Methods ────────────────────────────────────────────────

    def _shard_causal_graph(
        self, graph: dict[str, Any], domain: str
    ) -> list[CausalShard]:
        """因果图分片: 按变量组切分。"""
        variables = list(graph.get("nodes", []))
        edges = graph.get("edges", [])

        if not variables:
            return [
                CausalShard(
                    shard_id=f"{domain}_shard_0",
                    domain=domain,
                )
            ]

        shard_size = max(len(variables) // self._replication_factor, 1)
        shards = []
        for i in range(0, len(variables), shard_size):
            shard_vars = variables[i : i + shard_size]
            shard_id = f"{domain}_shard_{i // shard_size}"
            shard_edges = [
                e
                for e in edges
                if e.get("from") in shard_vars or e.get("to") in shard_vars
            ]
            shards.append(
                CausalShard(
                    shard_id=shard_id,
                    domain=domain,
                    variables=shard_vars,
                    edges=shard_edges,
                )
            )
        return shards

    def _select_shard_nodes(self, shard_id: str) -> list[str]:
        """选择分片存储节点 (轮询分配 + 副本)。"""
        if not self._federation_nodes:
            return ["local"]

        n_nodes = len(self._federation_nodes)
        shard_hash = int(hashlib.md5(shard_id.encode()).hexdigest(), 16)
        start_idx = shard_hash % n_nodes
        selected = []
        for i in range(min(self._replication_factor, n_nodes)):
            selected.append(self._federation_nodes[(start_idx + i) % n_nodes])
        return selected

    def _local_discovery(self, data: np.ndarray, domain: str) -> dict[str, Any]:
        """本地因果发现 (简化版: 基于相关性)。"""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n_vars = data.shape[1] if data.ndim > 1 else 1
        nodes = [f"{domain}_var_{i}" for i in range(n_vars)]
        edges = []
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if n_vars > 1 and data.shape[1] > max(i, j):
                    corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
                    if abs(corr) > 0.3:
                        edges.append({"from": nodes[i], "to": nodes[j]})
        return {
            "dag": {"nodes": nodes, "edges": edges},
            "confidence": 0.7,
        }

    def _merge_causal_structures(
        self, discoveries: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """合并因果结构。"""
        all_nodes = set()
        all_edges = []
        edge_votes: dict[str, int] = {}

        for node_id, disc in discoveries.items():
            dag = disc.get("dag", {})
            all_nodes.update(dag.get("nodes", []))
            for edge in dag.get("edges", []):
                edge_key = f"{edge.get('from')}->{edge.get('to')}"
                all_edges.append(edge)
                edge_votes[edge_key] = edge_votes.get(edge_key, 0) + 1

        # 只保留多数投票通过的边
        n_discoveries = max(len(discoveries), 1)
        threshold = max(n_discoveries // 2, 1)
        merged_edges = []  # type: ignore
        for edge in all_edges:
            edge_key = f"{edge.get('from')}->{edge.get('to')}"
            if edge_votes[edge_key] >= threshold:
                if not any(
                    e.get("from") == edge.get("from")
                    and e.get("to") == edge.get("to")
                    for e in merged_edges
                ):
                    merged_edges.append(edge)

        return {"nodes": list(all_nodes), "edges": merged_edges}

    def _detect_dag_conflicts(self, discoveries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        """检测 DAG 冲突 (方向相反的边)。"""
        forward: dict[str, int] = {}
        backward: dict[str, int] = {}

        for disc in discoveries.values():
            for edge in disc.get("dag", {}).get("edges", []):
                f, t = edge.get("from"), edge.get("to")
                forward[f"{f}->{t}"] = forward.get(f"{f}->{t}", 0) + 1
                backward[f"{t}->{f}"] = backward.get(f"{t}->{f}", 0) + 1

        conflicts = []
        all_pairs = set(forward.keys()) | set(backward.keys())
        for pair in all_pairs:
            f_count = forward.get(pair, 0)
            b_count = backward.get(pair, 0)
            if f_count > 0 and b_count > 0:
                parts = pair.split("->")
                conflicts.append(
                    {
                        "edge": {"from": parts[0], "to": parts[1]},
                        "votes_for": f_count,
                        "votes_against": b_count,
                    }
                )
        return conflicts

    def _apply_resolutions(
        self, merged_dag: dict[str, Any], resolutions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """应用冲突解决方案到合并 DAG。"""
        resolved_edges = []
        removed_edges = set()

        for r in resolutions:
            edge = r.get("edge", {})
            edge_key = f"{edge.get('from')}->{edge.get('to')}"
            if not r.get("resolved"):
                removed_edges.add(edge_key)
                # 移除反向边
                rev_key = f"{edge.get('to')}->{edge.get('from')}"
                removed_edges.add(rev_key)

        for edge in merged_dag.get("edges", []):
            edge_key = f"{edge.get('from')}->{edge.get('to')}"
            if edge_key not in removed_edges:
                resolved_edges.append(edge)

        return {"nodes": merged_dag["nodes"], "edges": resolved_edges}
