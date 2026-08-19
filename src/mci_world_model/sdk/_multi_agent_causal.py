"""MCI World Model — Multi-Agent Causal Negotiation SDK
=========================================================

多智能体因果协商：多个因果推理智能体独立发现后
通过置信度加权投票、冲突检测、博弈论协商达成一致因果图。

核心能力:
    CausalAgent        — 单个因果推理智能体
    AgentNegotiation   — 多智能体协商协议
    ConsensusGraph     — 协商后的一致因果图

设计原则:
    - 独立推理: 每个智能体独立运行因果发现
    - 投票聚合: 置信度加权 Borda 计数
    - 冲突检测: 边方向冲突标记
    - 纯 numpy，零外部依赖
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# ConsensusGraph
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class ConsensusGraph:
    """多智能体协商后的一致因果图。

    Attributes:
        nodes: 变量名列表
        edges: 有向边列表 [(src, dst), ...]
        adj_matrix: 邻接矩阵
        confidence: 共识置信度 [0, 1]
        conflict_edges: 有冲突的边列表
        agent_votes: 每条边的投票详情
    """

    nodes: list[str] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    adj_matrix: np.ndarray | None = None
    confidence: float = 0.0
    conflict_edges: list[tuple[str, str]] = field(default_factory=list)
    agent_votes: dict[str, dict[str, float]] = field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════════
# CausalAgent
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class CausalAgent:
    """因果推理智能体。

    Attributes:
        agent_id: 智能体唯一标识
        method: 因果发现方法 ('pc' / 'ges' / 'notears' / 'fci' / 'cam')
        discovered_edges: 发现的边列表
        adj_matrix: 发现的邻接矩阵
        confidence: 发现置信度
    """

    agent_id: str
    method: str = "pc"
    discovered_edges: list[tuple[str, str]] = field(default_factory=list)
    adj_matrix: np.ndarray | None = None
    confidence: float = 0.5

    def discover(self, data: np.ndarray, var_names: list[str]) -> None:
        """运行因果发现（简化版：直接设置结果）。

        Args:
            data: 数据矩阵 (n_samples, n_vars)
            var_names: 变量名列表
        """
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import (
            CAMDiscoverer,
            FCIDiscoverer,
            NOTEARSDiscoverer,
            PCSkeletonDiscoverer,
        )

        methods = {
            "pc": PCSkeletonDiscoverer,
            "notears": NOTEARSDiscoverer,
            "fci": FCIDiscoverer,
            "cam": CAMDiscoverer,
        }
        cls = methods.get(self.method, PCSkeletonDiscoverer)

        discoverer = cls(alpha=0.05)
        skel = discoverer.discover(data, var_names)

        self.discovered_edges = skel.edges
        self.adj_matrix = skel.adj_matrix.copy()
        self.confidence = skel.confidence


# ═════════════════════════════════════════════════════════════════════════════
# AgentNegotiation
# ═════════════════════════════════════════════════════════════════════════════


class AgentNegotiation:
    """多智能体因果协商协议。

    流程:
      1. 每个智能体独立发现因果图
      2. 边投票: 每条候选边按置信度加权投票
      3. 冲突检测: 双向边标记为冲突
      4. 共识图输出: Borda 计数 + 阈值筛选
    """

    def __init__(self, vote_threshold: float = 0.5, min_agents: int = 2):
        if min_agents < 2:
            raise ValueError(f"至少需要2个智能体, 当前 {min_agents}")
        self._vote_threshold = vote_threshold
        self._min_agents = min_agents
        self._agents: list[CausalAgent] = []
        self._negotiation_history: list[dict[str, Any]] = []

    @property
    def agent_count(self) -> int:
        return len(self._agents)

    def register_agent(self, agent: CausalAgent) -> None:
        """注册因果推理智能体。"""
        self._agents.append(agent)
        logger.info("注册智能体: %s (method=%s)", agent.agent_id, agent.method)

    def negotiate(self, data: np.ndarray, var_names: list[str]) -> ConsensusGraph:
        """执行多智能体协商。

        Args:
            data: 数据矩阵
            var_names: 变量名列表

        Returns:
            ConsensusGraph 共识因果图
        """
        if len(self._agents) < self._min_agents:
            return ConsensusGraph(
                nodes=list(var_names),
                confidence=0.0,
                conflict_edges=[],
            )

        n_vars = len(var_names)
        name_to_idx = {n: i for i, n in enumerate(var_names)}

        # Step 1: 所有智能体独立发现
        for agent in self._agents:
            try:
                agent.discover(data, var_names)
            except Exception as e:
                logger.warning("智能体 %s 发现失败: %s", agent.agent_id, e)

        # Step 2: 边投票 (置信度加权)
        edge_votes: dict[tuple[int, int], float] = {}
        agent_votes: dict[str, dict[str, float]] = {}

        for agent in self._agents:
            if agent.adj_matrix is None:
                continue
            weight = max(agent.confidence, 0.01)
            agent_edge_set = set()
            for src, dst in agent.discovered_edges:
                if src in name_to_idx and dst in name_to_idx:
                    i, j = name_to_idx[src], name_to_idx[dst]
                    edge_votes[(i, j)] = edge_votes.get((i, j), 0.0) + weight
                    agent_edge_set.add((i, j))

            agent_votes[agent.agent_id] = {
                "edges": len(agent_edge_set),
                "confidence": agent.confidence,
            }

        # Step 3: 共识筛选 (Borda 加权阈值)
        max_votes = max(len(self._agents), 1)
        candidates = {
            (i, j): score / max_votes
            for (i, j), score in edge_votes.items()
            if score / max_votes >= self._vote_threshold
        }

        # Step 4: 冲突检测 (双向边)
        conflict_edges: list[tuple[str, str]] = []
        edges: list[tuple[str, str]] = []
        adj = np.zeros((n_vars, n_vars), dtype=int)

        for (i, j), score in sorted(candidates.items(), key=lambda x: -x[1]):
            if adj[j, i] == 1:
                # 双向冲突: 得分高者保留
                reverse_score = candidates.get((j, i), 0.0)
                if score < reverse_score:
                    # 反向得分更高，跳过当前方向
                    continue
                elif score > reverse_score:
                    # 当前方向得分更高，移除反向
                    adj[j, i] = 0
                    edges = [(s, d) for (s, d) in edges if not (s == var_names[j] and d == var_names[i])]
                    conflict_edges.append((var_names[i], var_names[j]))
                else:
                    # 平局 = 冲突
                    conflict_edges.append((var_names[i], var_names[j]))
                    continue

            adj[i, j] = 1
            edges.append((var_names[i], var_names[j]))

        # Step 5: 共识置信度
        consensus_conf = float(np.mean(list(candidates.values()))) if candidates else 0.0

        result = ConsensusGraph(
            nodes=list(var_names),
            edges=edges,
            adj_matrix=adj,
            confidence=consensus_conf,
            conflict_edges=conflict_edges,
            agent_votes=agent_votes,
        )

        self._negotiation_history.append(
            {
                "n_agents": len(self._agents),
                "n_edges": len(edges),
                "n_conflicts": len(conflict_edges),
                "consensus_confidence": consensus_conf,
            }
        )

        return result

    def statistics(self) -> dict[str, Any]:
        return {
            "agent_count": self.agent_count,
            "negotiation_count": len(self._negotiation_history),
            "vote_threshold": self._vote_threshold,
            "min_agents": self._min_agents,
        }
