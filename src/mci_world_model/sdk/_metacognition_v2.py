from __future__ import annotations

"""MCI World Model — MetacognitionV2 元认知 2.0
===============================================

从 V1 单层监控升级到多层元认知:
    V1: 预测 → 监控 → 异常报警
    V2: 预测 → 监控 → 自诊断 → 自修复 → 自评估 → 能力边界标注

核心能力:
    MetacognitionState   — 元认知状态快照
    MetacognitionV2      — 元认知控制器

设计原则:
    - 与 SelfRepairCognition 正交组合
    - 纯 numpy，零外部依赖
"""


import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# MetacognitionState — 元认知状态快照
# =============================================================================


@dataclass
class MetacognitionState:
    """元认知状态快照。

    Attributes:
        confidence: 当前推理置信度 [0, 1]
        uncertainty: 不确定性水平 [0, 1]
        cognitive_load: 认知负荷 [0, 1]
        self_awareness_level: 自我意识等级 (0-6 WMMM)
        capability_boundary: 能力边界标注
        repair_count: 累计修复次数
        timestamp: 时间戳
    """

    confidence: float = 0.5
    uncertainty: float = 0.5
    cognitive_load: float = 0.0
    self_awareness_level: int = 0
    capability_boundary: list[str] = field(default_factory=list)
    repair_count: int = 0
    timestamp: float = 0.0


# =============================================================================
# MetacognitionV2 — 元认知控制器
# =============================================================================


class MetacognitionV2:
    """元认知 2.0 — 多层自监控 + 自诊断 + 自评估。

    工作流程:
      1. monitor: 监控推理过程质量
      2. diagnose: 诊断推理瓶颈
      3. assess_capability: 评估能力边界
      4. adjust: 调整推理策略

    用法:
        >>> meta = MetacognitionV2()
        >>> state = meta.monitor(prediction_confidence=0.8, prediction_error=0.3)
        >>> if state.uncertainty > 0.7:
        ...     meta.adjust("increase_evidence")
    """

    # 认知负荷阈值
    COGNITIVE_LOAD_LEVELS = {
        "low": 0.3,
        "medium": 0.6,
        "high": 0.9,
    }

    def __init__(
        self,
        uncertainty_threshold: float = 0.7,
        confidence_floor: float = 0.3,
        max_self_awareness: int = 6,
    ):
        if not 0.0 < uncertainty_threshold < 1.0:
            raise ValueError("uncertainty_threshold 必须在 (0,1)")
        if not 0.0 < confidence_floor < 1.0:
            raise ValueError("confidence_floor 必须在 (0,1)")
        self._uncertainty_threshold = uncertainty_threshold
        self._confidence_floor = confidence_floor
        self._max_awareness = max_self_awareness
        self._state_history: list[MetacognitionState] = []
        self._adjustment_count: int = 0
        self._boundary_labels: list[str] = []

    @property
    def current_state(self) -> MetacognitionState | None:
        """最新元认知状态。"""
        return self._state_history[-1] if self._state_history else None

    @property
    def state_history_count(self) -> int:
        return len(self._state_history)

    def monitor(
        self,
        prediction_confidence: float = 0.5,
        prediction_error: float = 0.0,
        n_evidence: int = 0,
        timestamp: float = 0.0,
    ) -> MetacognitionState:
        """监控推理过程质量。

        Args:
            prediction_confidence: 预测置信度 [0, 1]
            prediction_error: 预测误差
            n_evidence: 证据数量
            timestamp: 时间戳

        Returns:
            MetacognitionState 当前状态
        """
        # 不确定性: 基于置信度反向 + 误差贡献
        base_uncertainty = 1.0 - prediction_confidence
        error_contribution = min(prediction_error * 0.1, 0.3)
        uncertainty = min(base_uncertainty + error_contribution, 1.0)

        # 认知负荷: 基于证据数量和不确定性
        evidence_factor = max(0.0, 1.0 - n_evidence / 10.0)
        cognitive_load = min(uncertainty * 0.6 + evidence_factor * 0.4, 1.0)

        # 自我意识等级: 基于监控深度
        awareness = self._compute_awareness_level(prediction_confidence, uncertainty, cognitive_load)

        state = MetacognitionState(
            confidence=prediction_confidence,
            uncertainty=uncertainty,
            cognitive_load=cognitive_load,
            self_awareness_level=awareness,
            repair_count=sum(1 for s in self._state_history if s.repair_count > 0),
            timestamp=timestamp,
        )

        self._state_history.append(state)

        # 能力边界标注
        if prediction_confidence < self._confidence_floor:
            self._boundary_labels.append("low_confidence")
        if uncertainty > self._uncertainty_threshold:
            self._boundary_labels.append("high_uncertainty")

        logger.info(
            "元认知监控: conf=%.2f, uncert=%.2f, load=%.2f, aware=%d",
            state.confidence,
            state.uncertainty,
            state.cognitive_load,
            state.self_awareness_level,
        )

        return state

    def diagnose(self) -> dict[str, Any]:
        """诊断推理瓶颈。

        Returns:
            诊断报告
        """
        if not self._state_history:
            return {"diagnosis": "no_data", "bottleneck": "unknown"}

        latest = self._state_history[-1]

        # 瓶颈判定
        bottleneck = "none"
        if latest.cognitive_load > self.COGNITIVE_LOAD_LEVELS["high"]:
            bottleneck = "cognitive_overload"
        elif latest.uncertainty > self._uncertainty_threshold:
            bottleneck = "high_uncertainty"
        elif latest.confidence < self._confidence_floor:
            bottleneck = "low_confidence"

        # 趋势分析
        trend = "stable"
        if len(self._state_history) >= 3:
            recent = [s.uncertainty for s in self._state_history[-3:]]
            if recent[-1] > recent[0]:
                trend = "deteriorating"
            elif recent[-1] < recent[0]:
                trend = "improving"

        return {
            "bottleneck": bottleneck,
            "trend": trend,
            "confidence": latest.confidence,
            "uncertainty": latest.uncertainty,
            "cognitive_load": latest.cognitive_load,
            "self_awareness": latest.self_awareness_level,
            "recommendation": self._recommend(bottleneck),
        }

    def assess_capability(self) -> dict[str, Any]:
        """评估能力边界。

        Returns:
            能力评估报告
        """
        boundaries = list(set(self._boundary_labels))

        # 基于历史状态统计
        if not self._state_history:
            return {
                "capability_level": "unknown",
                "boundaries": [],
                "safe_to_proceed": False,
            }

        avg_conf = np.mean([s.confidence for s in self._state_history])
        avg_uncert = np.mean([s.uncertainty for s in self._state_history])

        if avg_conf > 0.8 and avg_uncert < 0.3:
            level = "high"
        elif avg_conf > 0.5 and avg_uncert < 0.6:
            level = "moderate"
        else:
            level = "low"

        safe = bool(avg_conf >= self._confidence_floor and avg_uncert <= self._uncertainty_threshold)

        return {
            "capability_level": level,
            "boundaries": boundaries,
            "safe_to_proceed": safe,
            "avg_confidence": float(avg_conf),
            "avg_uncertainty": float(avg_uncert),
        }

    def adjust(self, strategy: str) -> dict[str, Any]:
        """调整推理策略。

        Args:
            strategy: 调整策略名称

        Returns:
            调整结果
        """
        self._adjustment_count += 1

        valid_strategies = {
            "increase_evidence": "增加证据收集",
            "reduce_complexity": "降低推理复杂度",
            "switch_method": "切换推理方法",
            "request_help": "请求外部协助",
            "fallback_safe": "回退安全模式",
        }

        if strategy not in valid_strategies:
            return {
                "adjusted": False,
                "reason": f"未知策略: {strategy}",
                "valid_strategies": list(valid_strategies.keys()),
            }

        return {
            "adjusted": True,
            "strategy": strategy,
            "description": valid_strategies[strategy],
            "adjustment_count": self._adjustment_count,
        }

    def _compute_awareness_level(self, confidence: float, uncertainty: float, cognitive_load: float) -> int:
        """计算 WMMM 自我意识等级 (0-6)。"""
        if uncertainty < 0.1:
            return 6  # 协作级
        if uncertainty < 0.3:
            return 5  # 创造级
        if confidence > 0.7 and cognitive_load < 0.3:
            return 4  # 因果级
        if confidence > 0.5:
            return 3  # 预测级
        if confidence > 0.3:
            return 2  # 模式级
        if confidence > 0.1:
            return 1  # 反射级
        return 0  # 反应级

    @staticmethod
    def _recommend(bottleneck: str) -> str:
        """基于瓶颈推荐策略。"""
        recommendations = {
            "cognitive_overload": "reduce_complexity",
            "high_uncertainty": "increase_evidence",
            "low_confidence": "switch_method",
            "none": "continue",
        }
        return recommendations.get(bottleneck, "fallback_safe")

    def statistics(self) -> dict[str, Any]:
        """元认知统计。"""
        return {
            "state_history_count": self.state_history_count,
            "adjustment_count": self._adjustment_count,
            "boundary_labels": list(set(self._boundary_labels)),
            "uncertainty_threshold": self._uncertainty_threshold,
            "confidence_floor": self._confidence_floor,
        }
