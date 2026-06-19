"""MCI World Model v12.0.0 — FederatedAgentMarket 联邦因果智能体市场
====================================================================

联邦因果智能体的交易与发现平台 — 因果智能体的"应用商店"。

核心能力:
    register_agent(agent_spec)         — 注册因果智能体
    discover_agents(query)             — 发现智能体
    trade_agent(agent_id, consumer)    — 交易智能体
    rate_agent(agent_id, rating)       — 评价智能体

设计原则:
    - 纯 numpy，零外部依赖
    - 去中心化智能体注册
    - 基于联邦信任的交易验证
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# AgentSpec — 智能体规格
# =============================================================================


@dataclass
class AgentSpec:
    """因果智能体规格。

    Attributes:
        agent_id: 智能体 ID
        name: 名称
        description: 描述
        provider: 提供者节点 ID
        domains: 覆盖领域
        capabilities: 能力列表
        version: 版本号
        trust_score: 信任分数
        rating: 平均评分
        n_ratings: 评分数量
    """

    agent_id: str = ""
    name: str = ""
    description: str = ""
    provider: str = ""
    domains: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    version: str = "1.0.0"
    trust_score: float = 0.5
    rating: float = 0.0
    n_ratings: int = 0

    def __post_init__(self):
        if not self.agent_id:
            self.agent_id = hashlib.md5(
                f"{self.name}:{self.provider}:{time.time()}".encode()
            ).hexdigest()[:12]


# =============================================================================
# TradeRecord — 交易记录
# =============================================================================


@dataclass
class TradeRecord:
    """智能体交易记录。

    Attributes:
        trade_id: 交易 ID
        agent_id: 智能体 ID
        provider: 提供者
        consumer: 消费者
        timestamp: 交易时间
    """

    trade_id: str = ""
    agent_id: str = ""
    provider: str = ""
    consumer: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.trade_id:
            self.trade_id = hashlib.md5(
                f"{self.agent_id}:{self.consumer}:{self.timestamp}".encode()
            ).hexdigest()[:12]


# =============================================================================
# FederatedAgentMarket — 联邦因果智能体市场
# =============================================================================


class FederatedAgentMarket:
    """联邦因果智能体市场 — 因果智能体的交易与发现平台。

    职责:
      - 智能体注册与发现
      - 基于联邦信任的交易验证
      - 智能体评价体系
      - 交易记录追踪

    Args:
        min_trust_for_trade: 交易最低信任分数
    """

    def __init__(self, min_trust_for_trade: float = 0.5):
        self._agents: dict[str, AgentSpec] = {}
        self._trades: list[TradeRecord] = []
        self._ratings: dict[str, list[float]] = {}
        self._min_trust = min_trust_for_trade

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def n_agents(self) -> int:
        return len(self._agents)

    @property
    def n_trades(self) -> int:
        return len(self._trades)

    # ── Registration ────────────────────────────────────────────────────

    def register_agent(self, agent_spec: AgentSpec) -> dict:
        """注册因果智能体。

        Args:
            agent_spec: 智能体规格

        Returns:
            注册结果
        """
        if agent_spec.agent_id in self._agents:
            return {
                "registered": False,
                "reason": "agent_already_exists",
            }

        self._agents[agent_spec.agent_id] = agent_spec
        self._ratings[agent_spec.agent_id] = []

        return {
            "registered": True,
            "agent_id": agent_spec.agent_id,
        }

    # ── Discovery ───────────────────────────────────────────────────────

    def discover_agents(self, query: dict) -> list[AgentSpec]:
        """发现智能体。

        Args:
            query: 查询 {domain, capability, min_trust, ...}

        Returns:
            匹配的智能体列表
        """
        domain = query.get("domain")
        capability = query.get("capability")
        min_trust = query.get("min_trust", 0)

        results = []
        for agent in self._agents.values():
            # 领域过滤
            if domain and domain not in agent.domains:
                continue
            # 能力过滤
            if capability and capability not in agent.capabilities:
                continue
            # 信任过滤
            if agent.trust_score < min_trust:
                continue
            results.append(agent)

        # 按评分排序
        results.sort(key=lambda a: a.rating, reverse=True)
        return results

    # ── Trading ─────────────────────────────────────────────────────────

    def trade_agent(self, agent_id: str, consumer: str) -> dict:
        """交易智能体。

        Args:
            agent_id: 智能体 ID
            consumer: 消费者节点 ID

        Returns:
            交易结果
        """
        if agent_id not in self._agents:
            return {"traded": False, "reason": "agent_not_found"}

        agent = self._agents[agent_id]

        # 信任检查
        if agent.trust_score < self._min_trust:
            return {"traded": False, "reason": "trust_below_threshold"}

        # 记录交易
        trade = TradeRecord(
            agent_id=agent_id,
            provider=agent.provider,
            consumer=consumer,
        )
        self._trades.append(trade)

        return {
            "traded": True,
            "trade_id": trade.trade_id,
            "agent_id": agent_id,
        }

    # ── Rating ──────────────────────────────────────────────────────────

    def rate_agent(self, agent_id: str, rating: float) -> dict:
        """评价智能体。

        Args:
            agent_id: 智能体 ID
            rating: 评分 (0-5)

        Returns:
            评价结果
        """
        if agent_id not in self._agents:
            return {"rated": False, "reason": "agent_not_found"}

        rating = max(0, min(5, rating))
        self._ratings[agent_id].append(rating)

        # 更新平均评分
        agent = self._agents[agent_id]
        all_ratings = self._ratings[agent_id]
        agent.rating = float(np.mean(all_ratings))
        agent.n_ratings = len(all_ratings)

        return {
            "rated": True,
            "agent_id": agent_id,
            "new_avg_rating": agent.rating,
            "n_ratings": agent.n_ratings,
        }

    # ── Statistics ──────────────────────────────────────────────────────

    def market_statistics(self) -> dict:
        """市场统计。"""
        ratings = [a.rating for a in self._agents.values() if a.n_ratings > 0]
        return {
            "n_agents": len(self._agents),
            "n_trades": len(self._trades),
            "avg_rating": float(np.mean(ratings)) if ratings else 0,
            "top_domains": self._top_domains(),
        }

    def _top_domains(self, n: int = 5) -> list[str]:
        """热门领域。"""
        domain_counts: dict[str, int] = {}
        for agent in self._agents.values():
            for d in agent.domains:
                domain_counts[d] = domain_counts.get(d, 0) + 1
        sorted_domains = sorted(domain_counts, key=domain_counts.get, reverse=True)
        return sorted_domains[:n]
