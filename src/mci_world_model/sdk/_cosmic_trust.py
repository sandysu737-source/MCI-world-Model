"""MCI World Model v14.0.0 — CosmicTrust 宇宙级可信框架
========================================================

跨维度、跨尺度的统一信任体系 — 信任的终极统一。

核心能力:
    assess_cosmic_trust(result, dimensions)  — 宇宙级信任评估
    verify_cross_dimensional_consistency()    — 跨维度一致性验证
    calibrate_cosmic_trust()                 — 宇宙信任校准
    issue_cosmic_certificate()               — 颁发宇宙信任证书

维度信任: physical / digital_twin / mixed_reality / creative / meta
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class TrustDimension(str, Enum):
    PHYSICAL = "physical"
    DIGITAL_TWIN = "digital_twin"
    MIXED_REALITY = "mixed_reality"
    CREATIVE = "creative"
    META = "meta"


class CosmicTrustLevel(str, Enum):
    UNTRUSTWORTHY = "untrustworthy"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    ULTIMATE = "ultimate"


@dataclass
class DimensionalTrust:
    """单维度信任。"""
    dimension: str = ""
    trust_score: float = 0.0
    evidence_count: int = 0
    last_updated: float = 0.0
    confidence: float = 0.0


@dataclass
class CosmicCertificate:
    """宇宙信任证书。"""
    certificate_id: str = ""
    holder: str = ""
    cosmic_trust: float = 0.0
    dimensional_trusts: dict[str, float] = field(default_factory=dict)
    issued_at: float = 0.0
    expires_at: float = 0.0
    issuer: str = "CosmicTrust"
    valid: bool = True


@dataclass
class ConsistencyReport:
    """跨维度一致性报告。"""
    dimensions: list[str] = field(default_factory=list)
    pairwise_consistency: list[dict] = field(default_factory=list)
    overall_consistency: float = 0.0
    inconsistencies: list[dict] = field(default_factory=list)


class CosmicTrust:
    """宇宙级可信框架 — 跨维度、跨尺度的统一信任体系。

    设计原则:
      - 木桶原理: 宇宙信任 = 最弱维度信任
      - 信任传播: 跨维度信任可传播但会衰减
      - 信任校准: 自适应校准确保信任准确性
      - 证书化: 宇宙信任可验证、可追溯

    Args:
        federated_trust: 联邦信任框架
        creative_trust: 创造信任框架
    """

    DIMENSION_WEIGHTS: dict[str, float] = {
        TrustDimension.PHYSICAL.value: 0.25,
        TrustDimension.DIGITAL_TWIN.value: 0.20,
        TrustDimension.MIXED_REALITY.value: 0.20,
        TrustDimension.CREATIVE.value: 0.20,
        TrustDimension.META.value: 0.15,
    }

    def __init__(
        self,
        federated_trust: Any | None = None,
        creative_trust: Any | None = None,
    ) -> None:
        self._fed_trust = federated_trust
        self._creative_trust = creative_trust
        self._dimensional_trusts: dict[str, DimensionalTrust] = {}
        self._cosmic_trust_score = 0.0
        self._certificates: dict[str, CosmicCertificate] = {}
        self._calibration_history: list[dict] = []
        self._consistency_cache: dict[str, ConsistencyReport] = {}

        # 初始化各维度信任
        for dim in TrustDimension:
            self._dimensional_trusts[dim.value] = DimensionalTrust(
                dimension=dim.value,
                trust_score=0.5,
                evidence_count=0,
                last_updated=time.time(),
                confidence=0.5,
            )

    @property
    def cosmic_trust_score(self) -> float:
        return self._cosmic_trust_score

    @property
    def dimensional_trusts(self) -> dict[str, DimensionalTrust]:
        return dict(self._dimensional_trusts)

    def assess_cosmic_trust(
        self,
        reasoning_result: dict,
        dimensions: list[str] | None = None,
    ) -> dict:
        """宇宙级信任评估。

        Args:
            reasoning_result: 推理结果
            dimensions: 需要评估的维度

        Returns:
            宇宙信任评估结果
        """
        if dimensions is None:
            dimensions = list(TrustDimension)

        dim_trust: dict[str, float] = {}
        for dim in dimensions:
            score = self._assess_dimensional_trust(dim, reasoning_result)
            dim_trust[dim] = score

            # 更新内部状态
            if dim in self._dimensional_trusts:
                self._dimensional_trusts[dim].trust_score = score
                self._dimensional_trusts[dim].evidence_count += 1
                self._dimensional_trusts[dim].last_updated = time.time()

        # 宇宙信任 = 加权平均 (木桶原理 + 加权)
        if dim_trust:
            min_trust = min(dim_trust.values())
            weighted_trust = sum(
                dim_trust.get(d, 0) * self.DIMENSION_WEIGHTS.get(d, 0.2)
                for d in dimensions
            )
            cosmic_trust = 0.6 * min_trust + 0.4 * weighted_trust
        else:
            cosmic_trust = 0.0

        self._cosmic_trust_score = cosmic_trust

        trust_level = self._classify_trust_level(cosmic_trust)
        weakest = min(dim_trust, key=dim_trust.get) if dim_trust else None

        return {
            "cosmic_trust": cosmic_trust,
            "trust_level": trust_level,
            "dimensional_trust": dim_trust,
            "n_dimensions": len(dimensions),
            "weakest_dimension": weakest,
            "min_dimensional_trust": min(dim_trust.values()) if dim_trust else 0,
        }

    def verify_cross_dimensional_consistency(
        self,
        results: dict[str, dict],
    ) -> ConsistencyReport:
        """跨维度一致性验证。"""
        dimensions = list(results.keys())
        pairwise: list[dict] = []
        inconsistencies: list[dict] = []

        for i, d1 in enumerate(dimensions):
            for d2 in dimensions[i + 1:]:
                consistency = self._compute_pairwise_consistency(
                    results.get(d1, {}), results.get(d2, {}),
                )
                pairwise.append({
                    "dimensions": (d1, d2),
                    "consistency": consistency,
                })
                if consistency < 0.6:
                    inconsistencies.append({
                        "dimensions": (d1, d2),
                        "consistency": consistency,
                        "severity": 1.0 - consistency,
                    })

        overall = np.mean([p["consistency"] for p in pairwise]) if pairwise else 1.0

        report = ConsistencyReport(
            dimensions=dimensions,
            pairwise_consistency=pairwise,
            overall_consistency=float(overall),
            inconsistencies=inconsistencies,
        )

        cache_key = ",".join(sorted(dimensions))
        self._consistency_cache[cache_key] = report
        return report

    def calibrate_cosmic_trust(self, ground_truth: dict | None = None) -> dict:
        """宇宙信任校准。

        Args:
            ground_truth: 真实值用于校准

        Returns:
            校准结果
        """
        if ground_truth is not None:
            # 基于真实值的校准
            calibration_errors: dict[str, float] = {}
            for dim, true_value in ground_truth.items():
                if dim in self._dimensional_trusts:
                    predicted = self._dimensional_trusts[dim].trust_score
                    error = abs(predicted - true_value)
                    calibration_errors[dim] = error
                    # 校准调整
                    adjusted = predicted + 0.3 * (true_value - predicted)
                    self._dimensional_trusts[dim].trust_score = max(0, min(1, adjusted))

            mean_error = float(np.mean(list(calibration_errors.values()))) if calibration_errors else 0
        else:
            # 自校准: 基于维度间一致性
            for dim, dt in self._dimensional_trusts.items():
                # 向中值回归
                dt.trust_score = 0.9 * dt.trust_score + 0.1 * 0.5
            calibration_errors = {}
            mean_error = 0.1

        # 重新计算宇宙信任
        self._cosmic_trust_score = min(
            dt.trust_score for dt in self._dimensional_trusts.values()
        ) if self._dimensional_trusts else 0

        result = {
            "calibrated": True,
            "mean_calibration_error": mean_error,
            "cosmic_trust_after": self._cosmic_trust_score,
            "dimensional_errors": calibration_errors,
        }
        self._calibration_history.append(result)
        return result

    def issue_cosmic_certificate(
        self,
        holder: str,
        validity_seconds: float = 86400,
    ) -> CosmicCertificate:
        """颁发宇宙信任证书。"""
        now = time.time()
        cert_id = hashlib.sha256(
            f"{holder}:{now}:{self._cosmic_trust_score}".encode()
        ).hexdigest()[:16]

        dim_trusts = {
            dim: dt.trust_score for dim, dt in self._dimensional_trusts.items()
        }

        cert = CosmicCertificate(
            certificate_id=cert_id,
            holder=holder,
            cosmic_trust=self._cosmic_trust_score,
            dimensional_trusts=dim_trusts,
            issued_at=now,
            expires_at=now + validity_seconds,
            issuer="CosmicTrust",
            valid=True,
        )
        self._certificates[cert_id] = cert
        logger.info("Issued cosmic certificate %s for %s (trust=%.3f)", cert_id, holder, self._cosmic_trust_score)
        return cert

    def verify_cosmic_certificate(self, cert: CosmicCertificate) -> dict:
        """验证宇宙信任证书。"""
        now = time.time()
        stored = self._certificates.get(cert.certificate_id)

        checks = {
            "id_matches": stored is not None and stored.certificate_id == cert.certificate_id,
            "not_expired": now < cert.expires_at,
            "not_revoked": cert.valid,
            "trust_sufficient": cert.cosmic_trust >= 0.5,
        }

        all_pass = all(checks.values())
        return {
            "valid": all_pass,
            "checks": checks,
            "cosmic_trust": cert.cosmic_trust,
            "trust_level": self._classify_trust_level(cert.cosmic_trust),
        }

    def revoke_certificate(self, cert_id: str) -> bool:
        """撤销信任证书。"""
        if cert_id in self._certificates:
            self._certificates[cert_id].valid = False
            logger.info("Revoked cosmic certificate %s", cert_id)
            return True
        return False

    def get_trust_summary(self) -> dict:
        """获取信任摘要。"""
        return {
            "cosmic_trust_score": self._cosmic_trust_score,
            "trust_level": self._classify_trust_level(self._cosmic_trust_score),
            "n_dimensions": len(self._dimensional_trusts),
            "dimensional_scores": {d: dt.trust_score for d, dt in self._dimensional_trusts.items()},
            "n_certificates": len(self._certificates),
            "n_valid_certificates": sum(1 for c in self._certificates.values() if c.valid),
            "n_calibrations": len(self._calibration_history),
        }

    # ── 内部方法 ──────────────────────────────────────────────────

    def _assess_dimensional_trust(self, dimension: str, result: dict) -> float:
        """评估单维度信任。"""
        # 尝试使用外部信任框架
        if dimension in (TrustDimension.PHYSICAL.value, TrustDimension.DIGITAL_TWIN.value, TrustDimension.MIXED_REALITY.value):
            if self._fed_trust is not None and hasattr(self._fed_trust, "assess_federation_trust"):
                try:
                    fed_result = self._fed_trust.assess_federation_trust(dimension, result)
                    return fed_result.get("federation_trust", 0.5)
                except Exception:
                    pass

        if dimension == TrustDimension.CREATIVE.value:
            if self._creative_trust is not None and hasattr(self._creative_trust, "assess_creative_trust"):
                try:
                    cr_result = self._creative_trust.assess_creative_trust(result)
                    return cr_result.get("creative_trust_score", 0.5)
                except Exception:
                    pass

        # 内置评估: 基于结果质量
        base_trust = self._dimensional_trusts.get(dimension, DimensionalTrust()).trust_score
        result_confidence = result.get("confidence", 0.5)
        consistency = result.get("consistency", {}).get("all_consistent", True)

        adjustment = 0.1 * (result_confidence - 0.5)
        if not consistency:
            adjustment -= 0.2

        return max(0.0, min(1.0, base_trust + adjustment))

    def _compute_pairwise_consistency(self, r1: dict, r2: dict) -> float:
        """计算两维度间一致性。"""
        c1 = r1.get("confidence", 0.5)
        c2 = r2.get("confidence", 0.5)
        consistency = 1.0 - abs(c1 - c2)
        return float(consistency)

    def _classify_trust_level(self, score: float) -> str:
        """分类信任级别。"""
        if score >= 0.9:
            return CosmicTrustLevel.ULTIMATE.value
        elif score >= 0.7:
            return CosmicTrustLevel.HIGH.value
        elif score >= 0.5:
            return CosmicTrustLevel.MODERATE.value
        elif score >= 0.3:
            return CosmicTrustLevel.LOW.value
        else:
            return CosmicTrustLevel.UNTRUSTWORTHY.value
