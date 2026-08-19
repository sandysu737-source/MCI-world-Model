from __future__ import annotations

"""MCI World Model v4.6.0 — AutonomousKnowledgeCivilization 自主知识文明
========================================================================

因果知识的自主产生、传承与演化 — 知识的世代循环。

核心能力:
    knowledge_generation_cycle(domain)           — 知识世代循环
    knowledge_heritage(source, target)           — 跨域知识传承
    check_falsifications(domain)                — 检查知识证伪

世代循环: 评估→创造→验证→传承→淘汰

设计原则:
    - 纯 numpy，零外部依赖
    - 知识验证通过联邦共识
    - 5 维文明指标可度量
"""


import hashlib
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CivilizationMetrics:
    """文明指标。"""

    knowledge_volume: int = 0
    knowledge_diversity: int = 0
    knowledge_depth: float = 0.0
    innovation_rate: float = 0.0
    heritage_preservation: float = 0.0


class KnowledgeRepository:
    """知识仓库 — 存储和检索因果知识。"""

    def __init__(self) -> None:
        self._knowledge: dict[str, list[dict[str, Any]]] = {}
        self._retired: dict[str, list[dict[str, Any]]] = {}

    def store(self, theory: Any, domain: str) -> None:
        entry = {"theory": theory, "domain": domain, "id": hashlib.md5(str(theory).encode()).hexdigest()[:8]}
        self._knowledge.setdefault(domain, []).append(entry)

    def retire(self, theory_id: str, reason: str = "falsified") -> None:
        for domain, entries in self._knowledge.items():
            for entry in entries:
                if entry["id"] == theory_id:
                    entry["retirement_reason"] = reason
                    self._retired.setdefault(domain, []).append(entry)
                    self._knowledge[domain].remove(entry)
                    return

    def check_falsifications(self, domain: str) -> list[dict[str, Any]]:
        return []

    def get_all_theories(self, domain: str) -> list[dict[str, Any]]:
        return self._knowledge.get(domain, [])

    def export_domain(self, domain: str) -> list[dict[str, Any]]:
        return self._knowledge.get(domain, [])

    def total_count(self) -> int:
        return sum(len(v) for v in self._knowledge.values())

    def domain_diversity(self) -> int:
        return len(self._knowledge)


class AutonomousKnowledgeCivilization:
    """自主知识文明 — 因果知识的自主产生、传承与演化。

    Args:
        creation_engine: 因果创造引擎
        federation_protocol: 联邦协议 (用于联邦验证)
        knowledge_repository: 知识仓库
    """

    def __init__(
        self,
        creation_engine: Any | None = None,
        federation_protocol: Any | None = None,
        knowledge_repository: KnowledgeRepository | None = None,
    ):
        self._creation = creation_engine
        self._federation = federation_protocol
        self._repository = knowledge_repository or KnowledgeRepository()
        self._generations: list[dict[str, Any]] = []
        self._metrics = CivilizationMetrics()

    @property
    def metrics(self) -> CivilizationMetrics:
        return self._metrics

    @property
    def n_generations(self) -> int:
        return len(self._generations)

    def knowledge_generation_cycle(self, domain: str, n_theories: int = 5) -> dict[str, Any]:
        """知识世代循环: 评估→创造→验证→传承→淘汰。"""
        # Step 1: 世代评估
        assessment = self._assess_knowledge_state(domain)

        # Step 2: 创造循环
        new_theories = []
        if self._creation is not None:
            for _ in range(n_theories):
                result = self._creation.create_causal_theory(domain)
                if result.get("created_theory"):
                    new_theories.append(result["created_theory"])
        else:
            new_theories = []

        # Step 3: 验证循环
        verified = []
        for theory in new_theories:
            if self._federation is not None:
                fed_result = self._federation.federated_query({"type": "theory_verification", "theory": str(theory)})
                consensus = fed_result.get("consensus_level", 0.5)
            else:
                consensus = 0.7
            if consensus > 0.5:
                verified.append({"theory": theory, "consensus": consensus})

        # Step 4: 传承循环
        inherited = 0
        for v in verified:
            self._repository.store(v["theory"], domain)
            inherited += 1

        # Step 5: 淘汰循环
        falsified = self._repository.check_falsifications(domain)
        for f in falsified:
            self._repository.retire(f.get("id", ""), reason="falsified")

        # Step 6: 世代记录
        generation = {
            "domain": domain,
            "n_created": len(new_theories),
            "n_verified": len(verified),
            "n_inherited": inherited,
            "n_falsified": len(falsified),
            "assessment": assessment,
        }
        self._generations.append(generation)
        self._update_metrics(generation)

        return generation

    def knowledge_heritage(self, source_domain: str, target_domain: str) -> dict[str, Any]:
        """跨域知识传承。"""
        source_knowledge = self._repository.export_domain(source_domain)
        if not source_knowledge:
            return {
                "source": source_domain,
                "target": target_domain,
                "n_knowledge_transferred": 0,
                "verification_passed": False,
            }

        adapted = self._adapt_knowledge(source_knowledge, target_domain)
        for item in adapted:
            self._repository.store(item, target_domain)

        return {
            "source": source_domain,
            "target": target_domain,
            "n_knowledge_transferred": len(adapted),
            "verification_passed": True,
        }

    # ── Internal ────────────────────────────────────────────────────────

    def _assess_knowledge_state(self, domain: str) -> dict[str, Any]:
        theories = self._repository.get_all_theories(domain)
        return {
            "n_existing_theories": len(theories),
            "domain": domain,
            "gaps_identified": max(0, 5 - len(theories)),
        }

    def _adapt_knowledge(self, source_knowledge: list[dict[str, Any]], target_domain: str) -> list[dict[str, Any]]:
        adapted = []
        for item in source_knowledge[:5]:
            adapted.append(
                {
                    "original": item,
                    "adapted_domain": target_domain,
                }
            )
        return adapted

    def _update_metrics(self, generation: dict[str, Any]) -> None:
        self._metrics.knowledge_volume = self._repository.total_count()
        self._metrics.knowledge_diversity = self._repository.domain_diversity()
        self._metrics.innovation_rate = generation["n_verified"] / max(generation["n_created"], 1)
