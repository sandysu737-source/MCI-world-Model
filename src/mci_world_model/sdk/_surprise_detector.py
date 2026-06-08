"""
MCI World Model v3.3.0 — SurpriseDetector 惊奇误差检测器
==========================================================

预测 vs 观测的偏差量化——世界模型的"惊奇信号"。

当预测状态与实际观测状态偏差超过阈值时，触发惊奇信号，
驱动 PlanAgent 重规划或 Configurator 调整模型参数。

核心能力:
    compute_surprise(predicted, actual) — 单次惊奇度计算
    detect_anomalies(history)           — 批量异常检测
    running_statistics()                — 滚动统计
    adapt_threshold(n_std)              — 自适应阈值调整

惊奇度量化三维度:
    1. 状态距离 — WorldState.distance()
    2. 向量偏差 — to_vector() 的 L2 距离
    3. 方向偏差 — 向量余弦相似度反转

设计原则:
    - 纯 numpy，零外部依赖
    - 与 WorldState ABC 正交组合
    - 阈值自适应：基于历史惊奇度的均值 + n_std * 标准差
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mci_world_model.sdk._world_state import WorldState

logger = logging.getLogger(__name__)


# =============================================================================
# SurpriseSignal — 惊奇信号
# =============================================================================


@dataclass
class SurpriseSignal:
    """惊奇信号 — 预测 vs 观测的偏差量化结果。

    Attributes:
        score: 惊奇度 [0, 1]，0 = 完美预测，1 = 极度偏离
        predicted: 预测状态
        actual: 实际状态
        breakdown: 各维度惊奇分解
            - "state_distance": WorldState.distance() 归一化
            - "vector_deviation": L2 向量偏差归一化
            - "direction_error": 余弦相似度偏差
        is_anomaly: 是否超过阈值
        threshold: 当前阈值
    """

    score: float
    predicted: WorldState
    actual: WorldState
    breakdown: dict[str, float] = field(default_factory=dict)
    is_anomaly: bool = False
    threshold: float = 0.5


# =============================================================================
# SurpriseDetector — 惊奇误差检测器
# =============================================================================


class SurpriseDetector:
    """惊奇误差检测器 — 预测 vs 观测的偏差量化。

    Example:
        >>> from mci_world_model.sdk import PendulumState
        >>> detector = SurpriseDetector(threshold=0.3)
        >>> predicted = PendulumState(theta=0.5, omega=1.0)
        >>> actual = PendulumState(theta=0.6, omega=0.8)
        >>> signal = detector.compute_surprise(predicted, actual)
        >>> print(signal.score, signal.is_anomaly)
    """

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: 惊奇度阈值，超过此值判定为异常
        """
        self._threshold = threshold
        self._history: list[float] = []

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = max(0.0, min(1.0, value))

    @property
    def history(self) -> list[float]:
        return list(self._history)

    @property
    def n_observations(self) -> int:
        return len(self._history)

    # -----------------------------------------------------------------
    # compute_surprise — 单次惊奇度
    # -----------------------------------------------------------------

    def compute_surprise(
        self,
        predicted: WorldState,
        actual: WorldState,
    ) -> SurpriseSignal:
        """计算单次惊奇度。

        三维度融合:
            1. state_distance: WorldState.distance() 归一化到 [0,1]
            2. vector_deviation: to_vector() L2 距离归一化到 [0,1]
            3. direction_error: 1 - cosine_similarity 归一化到 [0,1]

        最终 score = 0.4 * state_distance + 0.3 * vector_deviation + 0.3 * direction_error

        Args:
            predicted: 预测状态
            actual: 实际状态

        Returns:
            SurpriseSignal 惊奇信号
        """
        # 维度1: 状态距离 (通过 WorldState.distance())
        raw_dist = predicted.distance(actual)
        # 归一化: 使用 sigmoid 映射到 [0,1]
        state_distance = 1.0 - np.exp(-raw_dist)

        # 维度2: 向量 L2 偏差
        pred_vec = predicted.to_vector()
        actual_vec = actual.to_vector()
        l2_dist = float(np.linalg.norm(pred_vec - actual_vec))
        # 归一化
        vector_deviation = 1.0 - np.exp(-l2_dist)

        # 维度3: 方向偏差 (余弦相似度)
        pred_norm = float(np.linalg.norm(pred_vec))
        actual_norm = float(np.linalg.norm(actual_vec))
        if pred_norm > 1e-10 and actual_norm > 1e-10:
            cosine_sim = float(np.dot(pred_vec, actual_vec) / (pred_norm * actual_norm))
            cosine_sim = max(-1.0, min(1.0, cosine_sim))
        else:
            cosine_sim = 1.0  # 零向量视为完全一致
        direction_error = (1.0 - cosine_sim) / 2.0  # [0, 1]

        # 加权融合
        score = 0.4 * state_distance + 0.3 * vector_deviation + 0.3 * direction_error
        score = float(np.clip(score, 0.0, 1.0))

        is_anomaly = score >= self._threshold
        self._history.append(score)

        breakdown = {
            "state_distance": round(float(state_distance), 6),
            "vector_deviation": round(float(vector_deviation), 6),
            "direction_error": round(float(direction_error), 6),
        }

        return SurpriseSignal(
            score=round(score, 6),
            predicted=predicted,
            actual=actual,
            breakdown=breakdown,
            is_anomaly=is_anomaly,
            threshold=self._threshold,
        )

    # -----------------------------------------------------------------
    # detect_anomalies — 批量异常检测
    # -----------------------------------------------------------------

    def detect_anomalies(
        self,
        history: list[tuple[WorldState, WorldState]],
    ) -> list[SurpriseSignal]:
        """批量检测 (predicted, actual) 对中的异常。

        Args:
            history: [(predicted_state, actual_state), ...] 预测-实际对列表

        Returns:
            SurpriseSignal 列表（只返回超过阈值的异常信号）
        """
        anomalies: list[SurpriseSignal] = []
        for predicted, actual in history:
            signal = self.compute_surprise(predicted, actual)
            if signal.is_anomaly:
                anomalies.append(signal)
        return anomalies

    # -----------------------------------------------------------------
    # running_statistics — 滚动统计
    # -----------------------------------------------------------------

    def running_statistics(self) -> dict:
        """返回惊奇度历史滚动统计。

        Returns:
            {
                "n": 观测总数,
                "mean": 均值,
                "std": 标准差,
                "max": 最大惊奇度,
                "min": 最小惊奇度,
                "anomaly_rate": 异常率 (is_anomaly 的比例),
            }
        """
        if not self._history:
            return {
                "n": 0,
                "mean": 0.0,
                "std": 0.0,
                "max": 0.0,
                "min": 0.0,
                "anomaly_rate": 0.0,
            }

        arr = np.array(self._history, dtype=np.float64)
        n_anomalies = int(np.sum(arr >= self._threshold))

        return {
            "n": len(self._history),
            "mean": round(float(np.mean(arr)), 6),
            "std": round(float(np.std(arr)), 6),
            "max": round(float(np.max(arr)), 6),
            "min": round(float(np.min(arr)), 6),
            "anomaly_rate": round(n_anomalies / len(self._history), 6),
        }

    # -----------------------------------------------------------------
    # adapt_threshold — 自适应阈值
    # -----------------------------------------------------------------

    def adapt_threshold(self, n_std: float = 2.0) -> float:
        """基于历史数据自适应调整阈值。

        新阈值 = mean + n_std * std

        Args:
            n_std: 标准差倍数（默认 2.0，约 95% 置信）

        Returns:
            新的阈值（已自动应用到 self._threshold）
        """
        if len(self._history) < 2:
            return self._threshold

        arr = np.array(self._history, dtype=np.float64)
        new_threshold = float(np.mean(arr) + n_std * np.std(arr))
        new_threshold = max(0.01, min(0.99, new_threshold))

        self._threshold = new_threshold
        logger.info(
            "SurpriseDetector threshold adapted: %.4f (n=%d, mean=%.4f, std=%.4f)",
            new_threshold, len(self._history), float(np.mean(arr)), float(np.std(arr)),
        )
        return new_threshold

    # -----------------------------------------------------------------
    # reset — 重置
    # -----------------------------------------------------------------

    def reset(self) -> None:
        """清空历史记录，不改变阈值。"""
        self._history.clear()

    # -----------------------------------------------------------------
    # 字符串表示
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SurpriseDetector(threshold={self._threshold:.3f}, "
            f"n_observations={self.n_observations})"
        )
