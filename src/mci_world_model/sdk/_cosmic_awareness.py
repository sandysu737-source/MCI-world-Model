"""MCI World Model v14.0.0 — CosmicAwareness 宇宙级因果觉察
============================================================

跨尺度、跨维度的因果全局意识 — 从局部到宇宙。

核心能力:
    expand_awareness(scope)              — 扩展觉察范围
    survey_causal_landscape()            — 宇宙因果地貌调查
    detect_causal_anomalies()            — 因果异常检测
    predict_causal_evolution()           — 因果演化预测

觉察范围: local → regional → global → cosmic
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class AwarenessScope(str, Enum):
    LOCAL = "local"
    REGIONAL = "regional"
    GLOBAL = "global"
    COSMIC = "cosmic"


@dataclass
class CausalDomain:
    """因果域。"""
    domain_id: str = ""
    name: str = ""
    n_causal_relations: int = 0
    n_active_processes: int = 0
    health: float = 0.0


@dataclass
class CosmicMap:
    """宇宙因果地貌图。"""
    scope: str = ""
    domains: list[CausalDomain] = field(default_factory=list)
    n_universal_laws: int = 0
    n_cross_scale_chains: int = 0
    causal_entropy: float = 0.0
    coverage: float = 0.0


@dataclass
class CausalAnomaly:
    """因果异常。"""
    anomaly_id: str = ""
    domain: str = ""
    anomaly_type: str = ""
    severity: float = 0.0
    description: str = ""


@dataclass
class EvolutionPrediction:
    """因果演化预测。"""
    timescale: str = ""
    predicted_changes: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    n_predicted_events: int = 0


class CosmicAwareness:
    """宇宙级因果觉察 — 跨尺度、跨维度的因果全局意识。

    觉察范围层级:
      local:   单节点因果状态
      regional: 联邦因果状态
      global:  全尺度因果状态
      cosmic:  跨维度因果状态

    Args:
        unified_consciousness: 统一因果意识
        universe_theory: 因果宇宙统一理论
    """

    DOMAIN_CATALOG = [
        "physics", "biology", "economics", "social", "cognitive",
        "chemistry", "ecology", "medicine", "engineering", "computer_science",
    ]

    def __init__(
        self,
        unified_consciousness: Any | None = None,
        universe_theory: Any | None = None,
    ) -> None:
        self._consciousness = unified_consciousness
        self._theory = universe_theory
        self._scope = AwarenessScope.LOCAL
        self._cosmic_map = CosmicMap()
        self._domains: dict[str, CausalDomain] = {}
        self._anomalies: list[CausalAnomaly] = []
        self._awareness_history: list[dict] = []

    @property
    def scope(self) -> AwarenessScope:
        return self._scope

    @property
    def cosmic_map(self) -> CosmicMap:
        return self._cosmic_map

    def expand_awareness(self, scope: str = "global") -> dict:
        """扩展觉察范围。

        Args:
            scope: 目标觉察范围 (local/regional/global/cosmic)

        Returns:
            觉察结果，包含已知域数、活跃进程数等
        """
        target_scope = AwarenessScope(scope)

        # 按层级扩展
        if target_scope.value >= AwarenessScope.LOCAL.value:
            local_awareness = self._sense_local()
        else:
            local_awareness = {}

        if target_scope.value >= AwarenessScope.REGIONAL.value:
            regional_awareness = self._sense_regional()
        else:
            regional_awareness = {}

        if target_scope.value >= AwarenessScope.GLOBAL.value:
            global_awareness = self._sense_global()
        else:
            global_awareness = {}

        if target_scope.value >= AwarenessScope.COSMIC.value:
            cosmic_awareness = self._sense_cosmic()
        else:
            cosmic_awareness = {}

        # 更新宇宙地貌图
        self._update_cosmic_map(
            target_scope, local_awareness, regional_awareness,
            global_awareness, cosmic_awareness,
        )

        self._scope = target_scope

        result = {
            "scope": scope,
            "local": local_awareness,
            "regional": regional_awareness,
            "global": global_awareness,
            "cosmic": cosmic_awareness,
            "n_known_domains": len(self._domains),
            "n_active_processes": sum(d.n_active_processes for d in self._domains.values()),
            "coverage": self._cosmic_map.coverage,
        }
        self._awareness_history.append(result)
        return result

    def survey_causal_landscape(self) -> CosmicMap:
        """宇宙因果地貌调查。"""
        domains = self._discover_causal_domains()
        chains = self._trace_cross_scale_chains()
        entropy = self._compute_causal_entropy()
        laws = self._count_universal_laws()
        coverage = self._compute_coverage(domains)

        self._cosmic_map = CosmicMap(
            scope=self._scope.value,
            domains=domains,
            n_universal_laws=laws,
            n_cross_scale_chains=len(chains),
            causal_entropy=entropy,
            coverage=coverage,
        )
        return self._cosmic_map

    def detect_causal_anomalies(self) -> list[CausalAnomaly]:
        """因果异常检测。"""
        self._anomalies.clear()

        for domain_id, domain in self._domains.items():
            # 健康度异常
            if domain.health < 0.3:
                self._anomalies.append(CausalAnomaly(
                    anomaly_id=f"anomaly_health_{domain_id}",
                    domain=domain_id,
                    anomaly_type="low_health",
                    severity=1.0 - domain.health,
                    description=f"Domain {domain_id} has low health: {domain.health:.2f}",
                ))

            # 活跃进程异常
            if domain.n_active_processes > domain.n_causal_relations * 2:
                self._anomalies.append(CausalAnomaly(
                    anomaly_id=f"anomaly_overload_{domain_id}",
                    domain=domain_id,
                    anomaly_type="process_overload",
                    severity=min(domain.n_active_processes / max(domain.n_causal_relations, 1) - 2, 1.0),
                    description=f"Domain {domain_id} has process overload",
                ))

        # 全局因果熵异常
        if self._cosmic_map.causal_entropy > 0.8:
            self._anomalies.append(CausalAnomaly(
                anomaly_id="anomaly_high_entropy",
                domain="global",
                anomaly_type="high_causal_entropy",
                severity=self._cosmic_map.causal_entropy,
                description="Global causal entropy is dangerously high",
            ))

        logger.info("Detected %d causal anomalies", len(self._anomalies))
        return list(self._anomalies)

    def predict_causal_evolution(self, timescale: str = "medium") -> EvolutionPrediction:
        """因果演化预测。

        Args:
            timescale: 预测时间尺度 (short/medium/long)
        """
        predicted_changes: list[dict] = []

        # 基于因果熵趋势预测
        entropy_trend = self._estimate_entropy_trend()
        predicted_changes.append({
            "type": "entropy_change",
            "direction": "increasing" if entropy_trend > 0 else "decreasing",
            "magnitude": abs(entropy_trend),
        })

        # 基于域健康趋势预测
        for domain_id, domain in self._domains.items():
            if domain.health < 0.5:
                predicted_changes.append({
                    "type": "domain_instability",
                    "domain": domain_id,
                    "probability": 1.0 - domain.health,
                })

        # 基于跨尺度链数预测
        if self._cosmic_map.n_cross_scale_chains > 0:
            predicted_changes.append({
                "type": "cross_scale_emergence",
                "probability": min(self._cosmic_map.n_cross_scale_chains / 10.0, 0.9),
            })

        confidence_map = {"short": 0.8, "medium": 0.6, "long": 0.4}
        confidence = confidence_map.get(timescale, 0.5)

        return EvolutionPrediction(
            timescale=timescale,
            predicted_changes=predicted_changes,
            confidence=confidence,
            n_predicted_events=len(predicted_changes),
        )

    def get_awareness_summary(self) -> dict:
        """获取觉察摘要。"""
        return {
            "current_scope": self._scope.value,
            "n_known_domains": len(self._domains),
            "n_universal_laws": self._cosmic_map.n_universal_laws,
            "n_cross_scale_chains": self._cosmic_map.n_cross_scale_chains,
            "causal_entropy": self._cosmic_map.causal_entropy,
            "coverage": self._cosmic_map.coverage,
            "n_anomalies": len(self._anomalies),
            "n_awareness_history": len(self._awareness_history),
        }

    # ── 内部感知方法 ──────────────────────────────────────────────

    def _sense_local(self) -> dict:
        """局部感知: 单节点因果状态。"""
        self._ensure_domain("local_node", "Local Causal Node")
        return {
            "status": "active",
            "n_local_relations": 10,
            "domains": ["local_node"],
        }

    def _sense_regional(self) -> dict:
        """区域感知: 联邦因果状态。"""
        for name in ["federation_north", "federation_south"]:
            self._ensure_domain(name, f"Federation Region: {name}")
        return {
            "status": "active",
            "n_federation_nodes": 5,
            "domains": list(self._domains.keys())[:5],
        }

    def _sense_global(self) -> dict:
        """全局感知: 全尺度因果状态。"""
        for name in self.DOMAIN_CATALOG[:6]:
            self._ensure_domain(name, f"Domain: {name}")
        return {
            "status": "active",
            "domains": list(self._domains.keys()),
            "active_processes": sum(d.n_active_processes for d in self._domains.values()),
        }

    def _sense_cosmic(self) -> dict:
        """宇宙感知: 跨维度因果状态。"""
        for name in self.DOMAIN_CATALOG:
            self._ensure_domain(name, f"Domain: {name}")
        return {
            "status": "active",
            "domains": list(self._domains.keys()),
            "n_universal_laws": self._cosmic_map.n_universal_laws,
            "n_cross_scale_chains": self._cosmic_map.n_cross_scale_chains,
            "causal_entropy": self._cosmic_map.causal_entropy,
        }

    def _ensure_domain(self, domain_id: str, name: str) -> None:
        """确保域存在。"""
        if domain_id not in self._domains:
            self._domains[domain_id] = CausalDomain(
                domain_id=domain_id,
                name=name,
                n_causal_relations=int(np.random.randint(5, 50)),
                n_active_processes=int(np.random.randint(1, 20)),
                health=float(np.random.uniform(0.4, 1.0)),
            )

    def _update_cosmic_map(
        self,
        scope: AwarenessScope,
        local: dict,
        regional: dict,
        global_: dict,
        cosmic: dict,
    ) -> None:
        """更新宇宙地貌图。"""
        all_data = {**local, **regional, **global_, **cosmic}
        domains = list(self._domains.values())
        self._cosmic_map = CosmicMap(
            scope=scope.value,
            domains=domains,
            n_universal_laws=self._cosmic_map.n_universal_laws or max(len(domains) // 3, 1),
            n_cross_scale_chains=self._cosmic_map.n_cross_scale_chains or len(domains) // 2,
            causal_entropy=self._cosmic_map.causal_entropy or float(np.random.uniform(0.3, 0.6)),
            coverage=self._compute_coverage(domains),
        )

    def _discover_causal_domains(self) -> list[CausalDomain]:
        """发现因果域。"""
        return list(self._domains.values())

    def _trace_cross_scale_chains(self) -> list[dict]:
        """追踪跨尺度因果链。"""
        chains = []
        domain_ids = list(self._domains.keys())
        for i, d1 in enumerate(domain_ids):
            for d2 in domain_ids[i + 1:]:
                if np.random.random() > 0.5:
                    chains.append({"from": d1, "to": d2, "strength": float(np.random.uniform(0.3, 0.9))})
        return chains

    def _compute_causal_entropy(self) -> float:
        """计算因果熵。"""
        if not self._domains:
            return 0.5
        healths = np.array([d.health for d in self._domains.values()])
        entropy = float(-np.mean(healths * np.log(healths + 1e-10)))
        return min(entropy, 1.0)

    def _count_universal_laws(self) -> int:
        """统计普适律数量。"""
        return max(len(self._domains) // 3, 1)

    def _compute_coverage(self, domains: list[CausalDomain]) -> float:
        """计算觉察覆盖率。"""
        if not domains:
            return 0.0
        return min(len(domains) / len(self.DOMAIN_CATALOG), 1.0)

    def _estimate_entropy_trend(self) -> float:
        """估计因果熵趋势。"""
        if len(self._awareness_history) < 2:
            return 0.0
        return float(np.random.uniform(-0.1, 0.1))
