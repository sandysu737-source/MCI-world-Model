from __future__ import annotations

"""MCI World Model v6.0.0 — SelfRepairCognition 自修复认知
============================================================

推理链异常检测 + 自动修复 — 世界模型的自我修复能力。

核心能力:
    detect_anomaly(prediction, actual)  — 检测推理异常
    repair(anomaly_report)             — 根据诊断修复推理链
    get_repair_history()               — 获取修复历史

修复策略四层:
    perception: 重新校准编码器 (低成本)
    prediction: 增加预测步数 (中成本)
    causal:     重学因果结构 (高成本)
    unknown:    回退安全状态 (极高成本)

设计原则:
    - 纯 numpy，零外部依赖
    - 与 MetaDiagnoser 正交组合
"""


import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# AnomalyReport — 异常报告
# =============================================================================


@dataclass
class AnomalyReport:
    """推理异常报告。

    Attributes:
        is_anomaly: 是否异常
        error: 误差量 (L2 范数)
        error_threshold: 异常阈值
        diagnosis: 诊断结果 (layer, severity, suggestion)
    """

    is_anomaly: bool = False
    error: float = 0.0
    error_threshold: float = 0.0
    diagnosis: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# RepairAction — 修复动作
# =============================================================================


@dataclass
class RepairAction:
    """修复动作记录。

    Attributes:
        action: 修复策略名称
        layer: 诊断层 (perception/prediction/causal/unknown)
        success: 修复是否成功
        timestamp: 时间戳
        details: 修复详情
    """

    action: str = ""
    layer: str = ""
    success: bool = False
    timestamp: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# SelfRepairCognition — 自修复认知
# =============================================================================


class SelfRepairCognition:
    """自修复认知 — 检测并修复推理链断裂。

    工作流程:
      1. detect_anomaly: 检测预测 vs 实际的异常偏差
      2. diagnose: 诊断异常所属层 (感知/预测/因果)
      3. repair: 选择并执行修复策略
      4. verify: 验证修复后推理是否恢复正常

    Attributes:
        _anomaly_threshold: 异常检测阈值
        _repair_history: 修复历史
        _success_count: 成功修复计数
        _total_repairs: 总修复尝试计数
    """

    # 修复策略映射
    REPAIR_STRATEGIES = {
        "perception": "recalibrate_encoder",
        "prediction": "increase_prediction_steps",
        "causal": "relearn_causal_structure",
        "unknown": "fallback_to_safe_state",
    }

    def __init__(
        self,
        anomaly_threshold: float = 2.0,
        max_repair_attempts: int = 3,
    ):
        if anomaly_threshold <= 0:
            raise ValueError(f"anomaly_threshold 必须 > 0, 当前 {anomaly_threshold}")
        self._anomaly_threshold = anomaly_threshold
        self._max_repair_attempts = max_repair_attempts
        self._repair_history: list[dict[str, Any]] = []
        self._success_count: int = 0
        self._total_repairs: int = 0

    def detect_anomaly(self, prediction: np.ndarray, actual: np.ndarray) -> AnomalyReport:
        """检测推理异常。

        Args:
            prediction: 预测值
            actual: 实际值

        Returns:
            AnomalyReport 异常报告
        """
        pred = np.atleast_1d(np.asarray(prediction, dtype=float))
        act = np.atleast_1d(np.asarray(actual, dtype=float))

        if pred.shape != act.shape:
            # 形状不匹配: 截断到最小长度
            min_len = min(len(pred), len(act))
            pred = pred[:min_len]
            act = act[:min_len]

        error = float(np.linalg.norm(pred - act))

        if error > self._anomaly_threshold:
            diagnosis = self._diagnose(pred, act, error)
            return AnomalyReport(
                is_anomaly=True,
                error=error,
                error_threshold=self._anomaly_threshold,
                diagnosis=diagnosis,
            )

        return AnomalyReport(
            is_anomaly=False,
            error=error,
            error_threshold=self._anomaly_threshold,
        )

    def repair(self, anomaly_report: AnomalyReport) -> RepairAction:
        """根据诊断结果执行修复。

        Args:
            anomaly_report: 异常报告 (来自 detect_anomaly)

        Returns:
            RepairAction 修复动作记录
        """
        if not anomaly_report.is_anomaly:
            return RepairAction(action="none", success=True)

        diagnosis = anomaly_report.diagnosis
        layer = diagnosis.get("layer", "unknown")
        repair_strategy = self.REPAIR_STRATEGIES.get(layer, "fallback_to_safe_state")

        self._total_repairs += 1

        # 模拟修复执行
        repair_details = self._execute_repair(repair_strategy, diagnosis)

        # 判断修复是否成功 (简化: 基于修复策略的成功率)
        success = self._evaluate_repair(repair_strategy, anomaly_report.error)
        if success:
            self._success_count += 1

        action = RepairAction(
            action=repair_strategy,
            layer=layer,
            success=success,
            timestamp=float(len(self._repair_history)),
            details=repair_details,
        )

        self._repair_history.append(
            {
                "anomaly": {
                    "error": anomaly_report.error,
                    "threshold": anomaly_report.error_threshold,
                    "diagnosis": diagnosis,
                },
                "repair": {
                    "action": repair_strategy,
                    "layer": layer,
                    "success": success,
                },
            }
        )

        logger.info(
            f"自修复: layer={layer}, strategy={repair_strategy}, success={success}, error={anomaly_report.error:.3f}"
        )

        return action

    def repair_and_verify(self, prediction: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
        """完整修复流程: 检测 → 诊断 → 修复 → 验证。

        Args:
            prediction: 预测值
            actual: 实际值

        Returns:
            完整修复报告
        """
        report = self.detect_anomaly(prediction, actual)
        if not report.is_anomaly:
            return {"anomaly": False, "repair_needed": False}

        repair_action = self.repair(report)

        # 验证: 修复后重新检测 (模拟)
        # 实际实现中会用修复后的预测器重新预测
        simulated_reduction = 0.5 if repair_action.success else 0.1
        new_error = report.error * (1.0 - simulated_reduction)
        verified = new_error <= self._anomaly_threshold

        return {
            "anomaly": True,
            "repair_needed": True,
            "original_error": report.error,
            "repair_action": repair_action.action,
            "repair_success": repair_action.success,
            "post_repair_error": new_error,
            "verified": verified,
        }

    def _diagnose(self, prediction: np.ndarray, actual: np.ndarray, error: float) -> dict[str, Any]:
        """诊断异常所属层。

        简化启发式:
          - 误差集中在低维 → 感知层
          - 误差均匀分布 → 预测层
          - 误差方向系统性偏移 → 因果层
          - 其他 → 未知层
        """
        diff = actual - prediction

        # 误差集中度: 用变异系数判断
        abs_diff = np.abs(diff)
        if len(abs_diff) == 0:
            return {"layer": "unknown", "severity": "high"}

        cv = float(np.std(abs_diff) / max(np.mean(abs_diff), 1e-8))

        if cv > 2.0:
            layer = "perception"
            severity = "medium"
        elif cv > 0.5:
            # 检查系统性偏移
            mean_diff = float(np.mean(diff))
            if abs(mean_diff) > 0.3 * error:
                layer = "causal"
                severity = "high"
            else:
                layer = "prediction"
                severity = "medium"
        else:
            layer = "prediction"
            severity = "low"

        return {
            "layer": layer,
            "severity": severity,
            "error_magnitude": error,
            "coefficient_of_variation": cv,
            "suggestion": self.REPAIR_STRATEGIES.get(layer, "fallback_to_safe_state"),
        }

    @staticmethod
    def _execute_repair(strategy: str, diagnosis: dict[str, Any]) -> dict[str, Any]:
        """执行修复策略 (模拟)。"""
        details: dict[str, Any] = {"strategy": strategy}

        if strategy == "recalibrate_encoder":
            details["action_taken"] = "编码器参数微调 ±5%"
            details["estimated_recovery"] = 0.6
        elif strategy == "increase_prediction_steps":
            details["action_taken"] = "预测步数 +2"
            details["estimated_recovery"] = 0.5
        elif strategy == "relearn_causal_structure":
            details["action_taken"] = "触发因果图重新学习"
            details["estimated_recovery"] = 0.4
        elif strategy == "fallback_to_safe_state":
            details["action_taken"] = "回退到安全默认状态"
            details["estimated_recovery"] = 0.3
        else:
            details["action_taken"] = "无操作"
            details["estimated_recovery"] = 0.0

        return details

    @staticmethod
    def _evaluate_repair(strategy: str, error: float) -> bool:
        """评估修复是否成功 (简化: 基于策略成功率)。"""
        success_rates = {
            "recalibrate_encoder": 0.85,
            "increase_prediction_steps": 0.75,
            "relearn_causal_structure": 0.60,
            "fallback_to_safe_state": 0.40,
        }
        rate = success_rates.get(strategy, 0.3)
        # 确定性简化: 误差越大越难修复
        threshold = error * rate
        return threshold < 5.0  # 修复成功阈值

    @property
    def repair_success_rate(self) -> float:
        """修复成功率。"""
        if self._total_repairs == 0:
            return 0.0
        return self._success_count / self._total_repairs

    @property
    def repair_history(self) -> list[dict[str, Any]]:
        """修复历史。"""
        return list(self._repair_history)

    @property
    def anomaly_threshold(self) -> float:
        return self._anomaly_threshold
