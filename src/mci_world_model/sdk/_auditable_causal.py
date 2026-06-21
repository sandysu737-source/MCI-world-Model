from __future__ import annotations

"""MCI World Model — AuditableCausalReasoning 可审计因果推理
=========================================================

因果推理全链路审计——从假设到结论的每一步都留存审计轨迹，
满足法律/医疗/工程等安全关键领域的可追溯性要求。

核心能力:
    AuditStep            — 审计步骤记录
    AuditTrail           — 完整审计轨迹
    AuditableCausalReasoning — 可审计因果推理引擎

设计原则:
    - 100%审计轨迹留存
    - 与 DoCalculus 正交组合
    - 纯 numpy，零外部依赖
"""


import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# AuditStep — 审计步骤
# =============================================================================


@dataclass
class AuditStep:
    """审计步骤记录。

    Attributes:
        step_id: 步骤唯一ID
        step_type: 步骤类型 ('hypothesis' / 'evidence' / 'inference' / 'validation' / 'conclusion')
        description: 步骤描述
        inputs: 输入数据摘要
        outputs: 输出数据摘要
        confidence: 步骤置信度
        timestamp: 时间戳
    """

    step_id: str
    step_type: str
    description: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# AuditTrail — 完整审计轨迹
# =============================================================================


@dataclass
class AuditTrail:
    """完整审计轨迹。

    Attributes:
        trail_id: 轨迹唯一ID
        reasoner: 推理者标识
        steps: 审计步骤列表
        start_time: 开始时间
        end_time: 结束时间
        conclusion: 最终结论
        is_complete: 是否完整
    """

    trail_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    reasoner: str = ""
    steps: list[AuditStep] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    conclusion: str = ""
    is_complete: bool = False

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def duration(self) -> float:
        end = self.end_time if self.end_time > 0 else time.time()
        return end - self.start_time

    def add_step(self, step: AuditStep) -> None:
        self.steps.append(step)

    def complete(self, conclusion: str) -> None:
        self.conclusion = conclusion
        self.end_time = time.time()
        self.is_complete = True


# =============================================================================
# AuditableCausalReasoning — 可审计因果推理
# =============================================================================


class AuditableCausalReasoning:
    """可审计因果推理 — 100%审计轨迹留存。

    用法:
        >>> acr = AuditableCausalReasoning()
        >>> trail = acr.begin("hypothesis: X causes Y")
        >>> acr.add_evidence_step(trail, "observed correlation r=0.85")
        >>> acr.add_inference_step(trail, "backdoor adjustment", confidence=0.8)
        >>> acr.conclude(trail, "X causes Y with strength 0.8")
    """

    def __init__(self, reasoner_id: str = "mci_acr") -> None:
        self._reasoner_id = reasoner_id
        self._trails: list[AuditTrail] = []
        self._step_counter: int = 0

    @property
    def trail_count(self) -> int:
        return len(self._trails)

    def begin(self, hypothesis: str) -> AuditTrail:
        """开始可审计推理。

        Args:
            hypothesis: 初始假设

        Returns:
            AuditTrail 新审计轨迹
        """
        trail = AuditTrail(reasoner=self._reasoner_id)

        step = AuditStep(
            step_id=self._next_step_id(),
            step_type="hypothesis",
            description=hypothesis,
            confidence=0.0,
        )
        trail.add_step(step)
        self._trails.append(trail)

        logger.info("可审计推理: 开始轨迹 %s, 假设=%s", trail.trail_id, hypothesis)
        return trail

    def add_evidence_step(
        self,
        trail: AuditTrail,
        description: str,
        evidence_data: dict | None = None,  # type: ignore
        confidence: float = 0.0,
    ) -> AuditStep:
        """添加证据步骤。

        Args:
            trail: 审计轨迹
            description: 证据描述
            evidence_data: 证据数据
            confidence: 证据置信度

        Returns:
            AuditStep
        """
        step = AuditStep(
            step_id=self._next_step_id(),
            step_type="evidence",
            description=description,
            inputs=evidence_data or {},
            confidence=confidence,
        )
        trail.add_step(step)
        return step

    def add_inference_step(
        self,
        trail: AuditTrail,
        method: str,
        confidence: float = 0.0,
        inputs: dict | None = None,  # type: ignore
        outputs: dict | None = None,  # type: ignore
    ) -> AuditStep:
        """添加推理步骤。

        Args:
            trail: 审计轨迹
            method: 推理方法
            confidence: 推理置信度
            inputs: 输入摘要
            outputs: 输出摘要

        Returns:
            AuditStep
        """
        step = AuditStep(
            step_id=self._next_step_id(),
            step_type="inference",
            description=f"推理方法: {method}",
            inputs=inputs or {},
            outputs=outputs or {},
            confidence=confidence,
        )
        trail.add_step(step)
        return step

    def add_validation_step(
        self,
        trail: AuditTrail,
        validation_type: str,
        passed: bool,
        details: dict | None = None,  # type: ignore
    ) -> AuditStep:
        """添加验证步骤。

        Args:
            trail: 审计轨迹
            validation_type: 验证类型
            passed: 是否通过
            details: 详细信息

        Returns:
            AuditStep
        """
        step = AuditStep(
            step_id=self._next_step_id(),
            step_type="validation",
            description=f"验证: {validation_type}, {'通过' if passed else '未通过'}",
            outputs={"passed": passed, **(details or {})},
        )
        trail.add_step(step)
        return step

    def conclude(self, trail: AuditTrail, conclusion: str) -> AuditTrail:
        """结束推理并记录结论。

        Args:
            trail: 审计轨迹
            conclusion: 最终结论

        Returns:
            完成的 AuditTrail
        """
        step = AuditStep(
            step_id=self._next_step_id(),
            step_type="conclusion",
            description=conclusion,
        )
        trail.add_step(step)
        trail.complete(conclusion)

        logger.info(
            "可审计推理: 完成轨迹 %s, 步骤=%d, 结论=%s",
            trail.trail_id,
            trail.step_count,
            conclusion[:50],
        )
        return trail

    def verify_trail(self, trail: AuditTrail) -> dict[str, Any]:
        """验证审计轨迹完整性。

        Args:
            trail: 审计轨迹

        Returns:
            验证报告
        """
        issues = []

        # 必须有假设步骤
        has_hypothesis = any(s.step_type == "hypothesis" for s in trail.steps)
        if not has_hypothesis:
            issues.append("missing_hypothesis")

        # 必须有证据步骤
        has_evidence = any(s.step_type == "evidence" for s in trail.steps)
        if not has_evidence:
            issues.append("missing_evidence")

        # 必须有推理步骤
        has_inference = any(s.step_type == "inference" for s in trail.steps)
        if not has_inference:
            issues.append("missing_inference")

        # 必须有结论步骤
        has_conclusion = any(s.step_type == "conclusion" for s in trail.steps)
        if not has_conclusion:
            issues.append("missing_conclusion")

        # 置信度单调性检查 (后续步骤不应比前一步骤置信度低太多)
        confidences = [s.confidence for s in trail.steps if s.confidence > 0]
        if len(confidences) >= 2:
            for i in range(1, len(confidences)):
                if confidences[i] < confidences[i - 1] * 0.5:
                    issues.append(f"confidence_drop_at_step_{i}")
                    break

        is_valid = len(issues) == 0

        return {
            "is_valid": is_valid,
            "issues": issues,
            "step_count": trail.step_count,
            "trail_id": trail.trail_id,
            "duration": trail.duration,
        }

    def get_trail(self, trail_id: str) -> AuditTrail | None:
        """按 ID 查找审计轨迹。"""
        for t in self._trails:
            if t.trail_id == trail_id:
                return t
        return None

    def _next_step_id(self) -> str:
        self._step_counter += 1
        return f"step_{self._step_counter:04d}"

    def statistics(self) -> dict[str, Any]:
        """统计信息。"""
        total_steps = sum(t.step_count for t in self._trails)
        completed = sum(1 for t in self._trails if t.is_complete)
        return {
            "reasoner_id": self._reasoner_id,
            "trail_count": self.trail_count,
            "completed_trails": completed,
            "total_steps": total_steps,
            "avg_steps_per_trail": total_steps / max(self.trail_count, 1),
        }
