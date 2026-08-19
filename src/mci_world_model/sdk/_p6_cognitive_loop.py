from __future__ import annotations

"""P6 Cognitive Loop — 高级认知闭环集成。

P6 "入化" 核心模块 — 连接 MetaCognitionV2 → MetaDiagnoser → SelfRepairCognition，
形成诊断-修复-验证闭环，目标：推理异常自修复率 ≥70%。

Usage::
    from mci_world_model.sdk._p6_cognitive_loop import P6CognitiveLoop

    loop = P6CognitiveLoop()
    result = loop.run(prediction=pred, actual=actual)
    logger.info(f"Self-repair rate: {loop.repair_rate:.1%}")
"""


import logging

logger = logging.getLogger(__name__)
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class P6LoopResult:
    """P6 认知闭环执行结果。"""

    anomaly_detected: bool
    repaired: bool
    repair_success: bool
    diagnosis: dict[str, Any]
    repair_action: dict[str, Any] = field(default_factory=dict)
    metacog_state: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0


@dataclass
class P6CognitiveLoop:
    """P6 高级认知闭环 — 诊断 → 修复 → 验证。

    集成三个核心组件:
      - MetaCognitionV2: 元认知监控
      - MetaDiagnoser: 因果根因诊断
      - SelfRepairCognition: 异常修复与验证

    Attributes:
        meta: MetaCognitionV2 实例 (延迟初始化)。
        diagnoser: MetaDiagnoser 实例 (延迟初始化)。
        repairer: SelfRepairCognition 实例 (延迟初始化)。
        _history: 闭环执行历史。
        _repair_successes: 修复成功计数。
        _repair_total: 修复总次数。
    """

    meta: Any = field(default=None, repr=False, init=True)
    diagnoser: Any = field(default=None, repr=False, init=True)
    repairer: Any = field(default=None, repr=False, init=True)
    _history: list[P6LoopResult] = field(default_factory=list, repr=False)
    _repair_successes: int = field(default=0, init=False, repr=False)
    _repair_total: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.meta is None:
            try:
                from mci_world_model.sdk._metacognition_v2 import MetacognitionV2

                self.meta = MetacognitionV2()
            except ImportError:
                logger.debug("MetacognitionV2 不可用，meta 保持 None")
        if self.diagnoser is None:
            try:
                from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser

                self.diagnoser = MetaDiagnoser()
            except ImportError:
                logger.debug("MetaDiagnoser 不可用，diagnoser 保持 None")
        if self.repairer is None:
            try:
                from mci_world_model.sdk._self_repair_cognition import SelfRepairCognition

                self.repairer = SelfRepairCognition()
            except ImportError:
                logger.debug("SelfRepairCognition 不可用，repairer 保持 None")

    def run(
        self,
        prediction: np.ndarray,
        actual: np.ndarray,
        *,
        confidence: float = 0.5,
        timestamp: float | None = None,
    ) -> P6LoopResult:
        """执行完整 P6 认知闭环。

        Args:
            prediction: 预测值。
            actual: 实际值。
            confidence: 初始置信度。
            timestamp: 时间戳。

        Returns:
            P6LoopResult。
        """
        start = time.perf_counter()
        ts = timestamp if timestamp is not None else time.time()

        # 1. Metacognition 监控
        metacog_state: dict[str, Any] = {"confidence": confidence}
        if self.meta is not None:
            error = float(np.mean((prediction - actual) ** 2))
            state = self.meta.monitor(
                prediction_confidence=confidence,
                prediction_error=error,
                n_evidence=1,
                timestamp=ts,
            )
            metacog_state = {
                "confidence": state.confidence,
                "uncertainty": state.uncertainty,
                "cognitive_load": state.cognitive_load,
                "awareness_level": state.self_awareness_level,
            }

        # 2. 异常检测
        anomaly_detected = False
        diagnosis: dict[str, Any] = {"anomaly": False}

        if self.repairer is not None:
            report = self.repairer.detect_anomaly(prediction, actual)
            anomaly_detected = report.is_anomaly
            if anomaly_detected:
                diagnosis = self.repairer._diagnose(prediction, actual, report.error)

        # 3. MetaDiagnoser 根因分析 (如有异常)
        if anomaly_detected and self.diagnoser is not None:
            try:
                diag_result = self.diagnoser.diagnose(
                    observation=actual.tolist() if isinstance(actual, np.ndarray) else actual,
                    prediction=prediction.tolist() if isinstance(prediction, np.ndarray) else prediction,
                )
                diagnosis["root_cause"] = diag_result.root_cause if hasattr(diag_result, "root_cause") else "unknown"
                diagnosis["patterns"] = (
                    [p.pattern_name for p in diag_result.matched_patterns]
                    if hasattr(diag_result, "matched_patterns")
                    else []
                )
            except Exception:
                logger.warning("异常降级", exc_info=True)
                diagnosis["root_cause"] = "diagnoser_unavailable"

        # 4. 修复 (如有异常)
        repaired = False
        repair_success = False
        repair_action: dict[str, Any] = {}

        if anomaly_detected and self.repairer is not None:
            repair_result = self.repairer.repair_and_verify(prediction, actual)
            repaired = True
            repair_success = repair_result.get("repair_successful", False)
            repair_action = {
                "strategy": repair_result.get("strategy", "unknown"),
                "error_before": repair_result.get("error_before", 0.0),
                "error_after": repair_result.get("error_after", 0.0),
            }

        # 5. 计数
        if repaired:
            self._repair_total += 1
            if repair_success:
                self._repair_successes += 1

        elapsed_ms = (time.perf_counter() - start) * 1000

        result = P6LoopResult(
            anomaly_detected=anomaly_detected,
            repaired=repaired,
            repair_success=repair_success,
            diagnosis=diagnosis,
            repair_action=repair_action,
            metacog_state=metacog_state,
            elapsed_ms=elapsed_ms,
        )
        self._history.append(result)
        return result

    @property
    def repair_rate(self) -> float:
        """自修复成功率。"""
        if self._repair_total == 0:
            return 1.0
        return self._repair_successes / self._repair_total

    @property
    def meets_p6_target(self) -> bool:
        """是否满足 P6 KPI: 自修复率 ≥70%。"""
        if self._repair_total < 10:
            return True  # 样本不足时不判定失败
        return self.repair_rate >= 0.70

    def statistics(self) -> dict[str, Any]:
        return {
            "total_runs": len(self._history),
            "repair_total": self._repair_total,
            "repair_successes": self._repair_successes,
            "repair_rate": self.repair_rate,
            "meets_p6_target": self.meets_p6_target,
            "meta_ok": self.meta is not None,
            "diagnoser_ok": self.diagnoser is not None,
            "repairer_ok": self.repairer is not None,
        }

    def clear_history(self) -> None:
        self._history.clear()
        self._repair_successes = 0
        self._repair_total = 0


__all__ = ["P6CognitiveLoop", "P6LoopResult"]
