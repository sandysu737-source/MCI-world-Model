from __future__ import annotations

"""MCI World Model — MedicalCausalSDK 医疗因果推理 SDK
====================================================

面向医疗安全关键场景的因果推理 SDK，
在临床决策支持中提供可验证因果推理能力。

核心能力:
    ClinicalEvidence     — 临床证据数据类
    CausalDiagnosis      — 因果诊断结果
    MedicalCausalSDK     — 医疗因果推理入口

设计原则:
    - 证据驱动: 至少2条证据才出因果结论
    - 可审计: 每步推理留下审计轨迹
    - 安全第一: 置信度 < 0.7 时拒绝给出确定性结论
"""


import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mci_world_model.sdk._confidence_calibrator import ConfidenceCalibrator

logger = logging.getLogger(__name__)


# =============================================================================
# ClinicalEvidence — 临床证据
# =============================================================================


@dataclass
class ClinicalEvidence:
    """临床证据数据类。

    Attributes:
        evidence_id: 证据唯一ID
        evidence_type: 证据类型 ('observation' / 'lab_result' / 'imaging' / 'vital_sign')
        description: 证据描述
        confidence: 证据置信度 [0, 1]
        source: 来源
        timestamp: 时间戳
    """

    evidence_id: str
    evidence_type: str = "observation"
    description: str = ""
    confidence: float = 0.5
    source: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        # D10 修复: confidence 必须在 [0, 1], 医疗安全关键校验
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence 必须在 [0, 1], 当前 {self.confidence}")


# =============================================================================
# CausalDiagnosis — 因果诊断结果
# =============================================================================


@dataclass
class CausalDiagnosis:
    """因果诊断结果。

    Attributes:
        cause: 因果变量
        effect: 结果变量
        causal_strength: 因果强度 [0, 1]
        confidence: 诊断置信度
        evidence_ids: 支撑证据ID列表
        is_conclusive: 是否确定性结论
        audit_trail: 审计轨迹
        warnings: 警告信息
    """

    cause: str
    effect: str
    causal_strength: float = 0.0
    confidence: float = 0.0
    evidence_ids: list[str] = field(default_factory=list)
    is_conclusive: bool = False
    audit_trail: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# =============================================================================
# MedicalCausalSDK — 医疗因果推理 SDK
# =============================================================================


class MedicalCausalSDK:
    """医疗因果推理 SDK — 安全关键的因果推理入口。

    用法:
        >>> sdk = MedicalCausalSDK()
        >>> sdk.add_evidence(ClinicalEvidence(evidence_id="e1", ...))
        >>> diagnosis = sdk.diagnose(cause="drug_X", effect="symptom_Y")
    """

    # 安全约束
    MIN_EVIDENCE_COUNT = 2
    MIN_CONFIDENCE_FOR_CONCLUSIVE = 0.7
    MIN_CAUSAL_STRENGTH = 0.3

    def __init__(self, patient_id: str = "", strict_mode: bool = True) -> None:
        self._patient_id = patient_id
        self._strict_mode = strict_mode
        self._lock = threading.Lock()
        self._evidence: list[ClinicalEvidence] = []
        self._diagnoses: list[CausalDiagnosis] = []
        self._audit_log: list[dict[str, Any]] = []
        self._calibrator: ConfidenceCalibrator | None = None

    @property
    def patient_id(self) -> str:
        return self._patient_id

    @property
    def evidence_count(self) -> int:
        return len(self._evidence)

    @property
    def diagnosis_count(self) -> int:
        return len(self._diagnoses)

    MAX_EVIDENCE_COUNT = 1000

    def set_calibrator(self, calibrator: ConfidenceCalibrator) -> None:
        """设置置信度校准器 (可选)。

        Adapt-EPA 借鉴: 轻量校准可显著提升诊断准确率。
        设置后, diagnose() 的 confidence 输出会被校准映射。

        Args:
            calibrator: ConfidenceCalibrator 实例
        """
        self._calibrator = calibrator

    def record_outcome(self, diagnosis_index: int, is_correct: bool) -> None:
        """记录诊断结果是否正确, 用于校准器在线学习。

        生产环境由医生标注后调用此方法。

        Args:
            diagnosis_index: diagnose() 返回的诊断在内部列表中的索引
            is_correct: 诊断是否正确
        """
        if 0 <= diagnosis_index < len(self._diagnoses):
            diag = self._diagnoses[diagnosis_index]
            if self._calibrator is not None:
                self._calibrator.update(diag.confidence, is_correct)

    def add_evidence(self, evidence: ClinicalEvidence) -> None:
        """添加临床证据。

        Args:
            evidence: 临床证据

        Raises:
            ValueError: 超过证据数量上限 (D14 DoS 防护)
        """
        with self._lock:
            if len(self._evidence) >= self.MAX_EVIDENCE_COUNT:
                raise ValueError(f"证据数量超过上限 {self.MAX_EVIDENCE_COUNT}")
            self._evidence.append(evidence)
        self._audit_log.append(
            {
                "action": "add_evidence",
                "evidence_id": evidence.evidence_id,
                "timestamp": time.time(),
            }
        )
        logger.debug(
            "医疗SDK: 添加证据 %s (类型=%s, 置信度=%.2f)",
            evidence.evidence_id,
            evidence.evidence_type,
            evidence.confidence,
        )

    def diagnose(
        self,
        cause: str,
        effect: str,
        prior_strength: float = 0.5,
    ) -> CausalDiagnosis:
        """执行因果诊断。

        Args:
            cause: 假设原因
            effect: 观测结果
            prior_strength: 先验因果强度 [0, 1]

        Returns:
            CausalDiagnosis 因果诊断结果

        Raises:
            ValueError: prior_strength 不在 [0, 1]
        """
        # D11 修复: prior_strength 范围校验
        if not 0.0 <= prior_strength <= 1.0:
            raise ValueError(f"prior_strength 必须在 [0, 1], 当前 {prior_strength}")
        warnings = []
        audit_trail = []

        # Step 1: 证据充分性检查
        with self._lock:
            evidence_snapshot = list(self._evidence)
        if len(evidence_snapshot) < self.MIN_EVIDENCE_COUNT:
            warnings.append(f"证据不足: {len(evidence_snapshot)} < {self.MIN_EVIDENCE_COUNT}")
            if self._strict_mode:
                audit_trail.append({"step": "evidence_check", "passed": False})
                return CausalDiagnosis(
                    cause=cause,
                    effect=effect,
                    confidence=0.0,
                    is_conclusive=False,
                    evidence_ids=[e.evidence_id for e in evidence_snapshot],
                    audit_trail=audit_trail,
                    warnings=warnings,
                )

        audit_trail.append({"step": "evidence_check", "passed": True})

        # Step 2: 综合证据置信度
        relevant_evidence = [
            e
            for e in evidence_snapshot
            if cause in e.description or effect in e.description or e.evidence_type == "observation"
        ]
        if not relevant_evidence:
            relevant_evidence = evidence_snapshot  # 降级使用全部证据

        evidence_confidence = float(np.mean([e.confidence for e in relevant_evidence]))

        # Step 3: 因果强度计算 (证据置信度主导, 先验为锚)
        evidence_weight = min(len(relevant_evidence) / 5.0, 1.0)
        # evidence-dominated: prior 仅作 0.1 锚定, evidence 主导 (0.9)。
        # Sachs 真实数据验证: 旧 0.3/0.7 给虚假边 0.15 prior 地板,
        # 导致 [0.3-0.4] 桶 confidence 高估 (0.314 vs 0.05 accuracy)。
        # 0.1/0.9 使虚假边 ev_conf=0.2 → cs=0.23 (更低, 更准确),
        # 真实边 ev_conf=0.85 → cs=0.82 (不被 evidence_weight 过度压低)。
        causal_strength = prior_strength * 0.1 + evidence_confidence * 0.9
        causal_strength *= evidence_weight

        # Step 4: 综合置信度
        # evidence_confidence 已在 Step 3 以 0.7 权重参与, 不再重复相乘。
        confidence = causal_strength

        # Step 4.5: 置信度校准 (Adapt-EPA 借鉴, 可选)
        # 轻量后验映射, 不改变模型本体。保守原则: 校准只能降低 confidence。
        if self._calibrator is not None:
            confidence = self._calibrator.calibrate(confidence, cause, effect)

        # Step 5: 确定性判定
        is_conclusive = confidence >= self.MIN_CONFIDENCE_FOR_CONCLUSIVE and causal_strength >= self.MIN_CAUSAL_STRENGTH

        audit_trail.append(
            {
                "step": "diagnosis",
                "evidence_count": len(relevant_evidence),
                "evidence_confidence": evidence_confidence,
                "causal_strength": causal_strength,
            }
        )

        if not is_conclusive:
            warnings.append(f"置信度不足: {confidence:.2f} < {self.MIN_CONFIDENCE_FOR_CONCLUSIVE}")

        diagnosis = CausalDiagnosis(
            cause=cause,
            effect=effect,
            causal_strength=causal_strength,
            confidence=confidence,
            evidence_ids=[e.evidence_id for e in relevant_evidence],
            is_conclusive=is_conclusive,
            audit_trail=audit_trail,
            warnings=warnings,
        )

        self._diagnoses.append(diagnosis)
        self._audit_log.append(
            {
                "action": "diagnose",
                "cause": cause,
                "effect": effect,
                "confidence": confidence,
                "is_conclusive": is_conclusive,
                "timestamp": time.time(),
            }
        )

        logger.info(
            "医疗SDK: 诊断 %s→%s, 强度=%.2f, 置信度=%.2f, 确定=%s",
            cause,
            effect,
            causal_strength,
            confidence,
            is_conclusive,
        )

        return diagnosis

    def batch_diagnose(
        self,
        queries: list[dict[str, Any]],
    ) -> list[CausalDiagnosis]:
        """批量因果诊断 — 一次调用处理多个诊断查询。

        生产场景: 单次网络请求处理多个诊断, 减少往返开销。

        Args:
            queries: [{"cause": str, "effect": str, "prior_strength": float,
                       "evidence": [{"id":..., "type":..., "description":...,
                                     "confidence": float}, ...]}, ...]

        Returns:
            CausalDiagnosis 列表, 与 queries 一一对应
        """
        results: list[CausalDiagnosis] = []
        for q in queries:
            sdk = MedicalCausalSDK(
                patient_id=q.get("patient_id", self._patient_id),
                strict_mode=q.get("strict_mode", self._strict_mode),
            )
            for ev in q.get("evidence", []):
                sdk.add_evidence(
                    ClinicalEvidence(
                        evidence_id=ev.get("id", ""),
                        evidence_type=ev.get("type", "observation"),
                        description=ev.get("description", ""),
                        confidence=ev.get("confidence", 0.5),
                    )
                )
            diag = sdk.diagnose(
                q.get("cause", ""),
                q.get("effect", ""),
                q.get("prior_strength", 0.5),
            )
            results.append(diag)
        return results

    def get_audit_log(self) -> list[dict[str, Any]]:
        """获取审计日志。"""
        return list(self._audit_log)

    def clear_evidence(self) -> None:
        """清除当前证据(新患者)。"""
        self._evidence.clear()

    def statistics(self) -> dict[str, Any]:
        """SDK 统计。"""
        conclusive = sum(1 for d in self._diagnoses if d.is_conclusive)
        return {
            "patient_id": self._patient_id,
            "evidence_count": self.evidence_count,
            "diagnosis_count": self.diagnosis_count,
            "conclusive_count": conclusive,
            "conclusive_rate": conclusive / max(self.diagnosis_count, 1),
            "audit_entries": len(self._audit_log),
            "strict_mode": self._strict_mode,
        }
