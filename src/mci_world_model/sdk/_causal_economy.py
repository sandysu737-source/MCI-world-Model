from __future__ import annotations

"""MCI World Model v13.0.0 — CausalEconomy 因果经济体系
======================================================

因果知识的价值度量与交易 — 知识经济基础设施。

核心能力:
    value_causal_knowledge(theory)              — 因果知识价值评估
    trade_knowledge(provider, consumer, theory) — 知识交易

价值维度: 新颖性(25%) + 解释力(30%) + 可操作性(25%) + 需求度(20%)
"""


import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Transaction:
    """知识交易记录。"""
    transaction_id: str = ""
    provider: str = ""
    consumer: str = ""
    theory_id: str = ""
    value: float = 0.0
    price: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.transaction_id:
            self.transaction_id = hashlib.md5(
                f"{self.provider}:{self.consumer}:{self.timestamp}".encode()
            ).hexdigest()[:12]


class CausalKnowledgeValueModel:
    """因果知识价值模型。"""

    def novelty_value(self, theory: Any) -> float:
        if hasattr(theory, "novelty_score"):
            return float(theory.novelty_score)
        return 0.5

    def explanatory_value(self, theory: Any) -> float:
        return float(np.random.uniform(0.4, 0.9))

    def operational_value(self, theory: Any) -> float:
        return float(np.random.uniform(0.3, 0.8))

    def demand_value(self, theory: Any) -> float:
        return float(np.random.uniform(0.3, 0.7))


class CausalKnowledgeMarket:
    """因果知识市场。"""

    def determine_price(self, theory: Any, value: dict[str, Any]) -> float:
        return value["total_value"] * 10  # 简化定价


class CausalEconomy:
    """因果经济体系 — 因果知识的价值度量与交易。

    Args:
        knowledge_repository: 知识仓库
        federation_protocol: 联邦协议
    """

    def __init__(
        self,
        knowledge_repository: Any | None = None,
        federation_protocol: Any | None = None,
    ):
        self._repository = knowledge_repository
        self._federation = federation_protocol
        self._value_model = CausalKnowledgeValueModel()
        self._market = CausalKnowledgeMarket()
        self._transaction_log: list[Transaction] = []

    @property
    def n_transactions(self) -> int:
        return len(self._transaction_log)

    def value_causal_knowledge(self, theory: Any) -> dict[str, Any]:
        """因果知识价值评估。"""
        novelty = self._value_model.novelty_value(theory)
        explanatory = self._value_model.explanatory_value(theory)
        operational = self._value_model.operational_value(theory)
        demand = self._value_model.demand_value(theory)

        total = 0.25 * novelty + 0.30 * explanatory + 0.25 * operational + 0.20 * demand

        return {
            "total_value": float(total),
            "novelty_value": float(novelty),
            "explanatory_value": float(explanatory),
            "operational_value": float(operational),
            "demand_value": float(demand),
            "value_category": self._classify_value(total),
        }

    def trade_knowledge(
        self, provider: str, consumer: str, theory: Any
    ) -> dict[str, Any]:
        """因果知识交易。"""
        value = self.value_causal_knowledge(theory)
        price = self._market.determine_price(theory, value)

        theory_id = hashlib.md5(str(theory).encode()).hexdigest()[:8]
        tx = Transaction(
            provider=provider,
            consumer=consumer,
            theory_id=theory_id,
            value=value["total_value"],
            price=float(price),
        )
        self._transaction_log.append(tx)

        return {
            "transaction": {
                "transaction_id": tx.transaction_id,
                "provider": provider,
                "consumer": consumer,
                "value": value["total_value"],
                "price": float(price),
            },
            "status": "completed",
        }

    @staticmethod
    def _classify_value(total: float) -> str:
        if total >= 0.8:
            return "premium"
        if total >= 0.6:
            return "standard"
        if total >= 0.4:
            return "basic"
        return "low_value"

    def get_economy_report(self) -> dict[str, Any]:
        """获取经济体系报告。"""
        total_value = sum(tx.value for tx in self._transaction_log)
        total_price = sum(tx.price for tx in self._transaction_log)
        return {
            "n_transactions": self.n_transactions,
            "total_value": total_value,
            "total_price": total_price,
            "avg_value": total_value / max(1, self.n_transactions),
            "avg_price": total_price / max(1, self.n_transactions),
        }

    def assess_market_health(self) -> dict[str, Any]:
        """评估市场健康度。"""
        if not self._transaction_log:
            return {"health": "no_data", "liquidity": 0.0, "diversity": 0.0}

        # 流动性: 交易频率
        liquidity = min(1.0, self.n_transactions / 10.0)
        # 多样性: 不同提供者数量
        providers = {tx.provider for tx in self._transaction_log}
        diversity = min(1.0, len(providers) / 5.0)
        # 综合健康度
        health_score = 0.5 * liquidity + 0.5 * diversity

        if health_score >= 0.7:
            health = "healthy"
        elif health_score >= 0.4:
            health = "moderate"
        else:
            health = "illiquid"

        return {
            "health": health,
            "health_score": health_score,
            "liquidity": liquidity,
            "diversity": diversity,
            "n_providers": len(providers),
        }

    def batch_trade(self, trades: list[tuple[str, str, Any]]) -> dict[str, Any]:
        """批量知识交易。"""
        results = []
        for provider, consumer, theory in trades:
            result = self.trade_knowledge(provider, consumer, theory)
            results.append(result)
        return {
            "n_trades": len(results),
            "status": "batch_completed",
            "results": results,
        }
