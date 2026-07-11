"""轻量置信度校准器 — Adapt-EPA 借鉴。

不改变模型本体，仅对 diagnose() 输出的 raw confidence 做后验映射，
使其更贴近真实正确概率。支持 Platt Scaling 和 Isotonic Regression。

医疗保守原则：校准只能降低过高的 confidence，不能人为抬高。
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    """数值稳定的 sigmoid。"""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class _CalibrationRecord:
    """单条校准记录。"""

    raw_confidence: float
    actual_outcome: bool  # True = 诊断正确


class ConfidenceCalibrator:
    """轻量置信度校准器。

    通过历史 (raw_confidence, actual_correctness) 数据拟合校准映射，
    使输出 confidence 更接近真实正确概率。

    策略:
        platt:    σ(a·raw + b)，两参数 logistic 校准
        isotonic: 非参数单调映射
        none:     不校准，返回原始值

    医疗保守原则: calibrate(raw) <= raw (不抬高 confidence)。
    """

    def __init__(
        self,
        method: str = "platt",
        max_history: int = 10000,
        n_bins: int = 10,
    ) -> None:
        self._method = method
        self._max_history = max_history
        self._n_bins = n_bins
        self._lock = threading.Lock()

        # Platt 参数
        self._platt_a: float = 1.0
        self._platt_b: float = 0.0

        # Isotonic 映射: sorted (raw, calibrated) 对
        self._iso_map: list[tuple[float, float]] = []

        # 历史数据 (环形缓冲)
        self._history: deque[_CalibrationRecord] = deque(maxlen=max_history)

        # 是否已拟合
        self._fitted: bool = False

    @property
    def method(self) -> str:
        return self._method

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def sample_count(self) -> int:
        return len(self._history)

    def calibrate(self, raw_confidence: float, cause: str = "", effect: str = "") -> float:
        """将原始 confidence 映射为校准后的概率。

        医疗保守原则: 校准后值 ≤ raw_confidence。
        校准器未拟合或出错时降级返回原始值。

        Args:
            raw_confidence: 原始置信度 [0, 1]
            cause: 因果原因 (保留接口, 当前不影响校准)
            effect: 因果结果

        Returns:
            校准后置信度 [0, raw_confidence]
        """
        if not self._fitted or self._method == "none":
            return raw_confidence

        try:
            if self._method == "platt":
                calibrated = _sigmoid(self._platt_a * raw_confidence + self._platt_b)
            elif self._method == "isotonic":
                calibrated = self._isotonic_lookup(raw_confidence)
            else:
                return raw_confidence

            # 数值安全: nan/inf 降级
            if not math.isfinite(calibrated):
                logger.warning("校准结果非有限值 (%.4f), 降级返回原始值", calibrated)
                return raw_confidence
            # 保守原则: 校准只能降低, 不能抬高
            return min(calibrated, raw_confidence)
        except Exception:
            logger.warning("校准计算失败, 降级返回原始值", exc_info=True)
            return raw_confidence

    def update(self, raw_confidence: float, actual_outcome: bool) -> None:
        """在线增量记录一条校准数据 (单样本)。

        注意: 仅记录数据, 不立即重新拟合。需调用 refit() 触发参数更新。

        Args:
            raw_confidence: 诊断时的原始 confidence
            actual_outcome: 诊断是否正确 (True=正确)
        """
        with self._lock:
            self._history.append(
                _CalibrationRecord(
                    raw_confidence=raw_confidence,
                    actual_outcome=actual_outcome,
                )
            )

    def fit(self, history: list[tuple[float, bool]]) -> None:
        """批量拟合校准参数。

        Args:
            history: [(raw_confidence, actual_outcome), ...]
        """
        with self._lock:
            self._history.clear()
            for raw, outcome in history:
                self._history.append(_CalibrationRecord(raw_confidence=raw, actual_outcome=outcome))
            self._do_fit()

    def refit(self) -> None:
        """用已积累的历史数据重新拟合参数。"""
        with self._lock:
            self._do_fit()

    def _do_fit(self) -> None:
        """实际拟合逻辑 (调用者持锁)。"""
        if len(self._history) < 10:
            logger.debug("校准数据不足 (%d < 10), 跳过拟合", len(self._history))
            return

        raws = np.array([r.raw_confidence for r in self._history])
        outcomes = np.array([1.0 if r.actual_outcome else 0.0 for r in self._history])

        if self._method == "platt":
            self._fit_platt(raws, outcomes)
        elif self._method == "isotonic":
            self._fit_isotonic(raws, outcomes)

        self._fitted = True
        logger.info(
            "校准器拟合完成: method=%s, samples=%d, ECE=%.4f",
            self._method,
            len(self._history),
            self.expected_calibration_error(),
        )

    def _fit_platt(self, raws: np.ndarray, outcomes: np.ndarray) -> None:
        """Platt Scaling: 拟合 σ(a·raw + b)。

        用梯度下降最小化 NLL (Negative Log-Likelihood)。
        """
        a, b = 1.0, 0.0
        lr = 0.01
        n_iterations = 200

        for _ in range(n_iterations):
            z = a * raws + b
            # 数值稳定 sigmoid
            p = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

            # NLL 对 a, b 的梯度
            grad_factor = p - outcomes
            grad_a = np.mean(grad_factor * raws)
            grad_b = np.mean(grad_factor)

            a -= lr * grad_a
            b -= lr * grad_b

        self._platt_a = float(a)
        self._platt_b = float(b)

    def _fit_isotonic(self, raws: np.ndarray, outcomes: np.ndarray) -> None:
        """Isotonic Regression: 非参数单调映射 (PAV 算法)。"""
        # 按 raw 排序
        order = np.argsort(raws)
        sorted_raws = raws[order]
        sorted_outcomes = outcomes[order]

        # Pool Adjacent Violators
        weights = np.ones(len(sorted_raws))
        values = sorted_outcomes.copy()
        boundaries = sorted_raws.copy()

        n = len(values)
        i = 0
        while i < n - 1:
            if values[i] > values[i + 1]:
                # 合并 i 和 i+1
                w_total = weights[i] + weights[i + 1]
                values[i] = (values[i] * weights[i] + values[i + 1] * weights[i + 1]) / w_total
                weights[i] = w_total
                # 删除 i+1
                values = np.delete(values, i + 1)
                weights = np.delete(weights, i + 1)
                boundaries = np.delete(boundaries, i + 1)
                n -= 1
                if i > 0:
                    i -= 1  # 回溯检查
            else:
                i += 1

        self._iso_map = list(zip(boundaries.tolist(), values.tolist()))

    def _isotonic_lookup(self, raw: float) -> float:
        """Isotonic 映射查表: 找到 raw 对应的校准值。"""
        if not self._iso_map:
            return raw

        # 二分查找第一个 > raw 的边界
        lo, hi = 0, len(self._iso_map)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._iso_map[mid][0] <= raw:
                lo = mid + 1
            else:
                hi = mid

        if lo == 0:
            return self._iso_map[0][1]
        return self._iso_map[lo - 1][1]

    def expected_calibration_error(self) -> float:
        """计算 ECE (Expected Calibration Error)。

        将 [0,1] 分成 n_bins 个桶, 对每个桶计算 |平均置信度 - 实际正确率|,
        按桶大小加权平均。

        Returns:
            ECE 值 [0, 1], 越低越好
        """
        if len(self._history) == 0:
            return 0.0

        raws = np.array([r.raw_confidence for r in self._history])
        outcomes = np.array([1.0 if r.actual_outcome else 0.0 for r in self._history])

        bin_edges = np.linspace(0, 1, self._n_bins + 1)
        ece = 0.0
        n_total = len(raws)

        for i in range(self._n_bins):
            mask = (raws >= bin_edges[i]) & (raws < bin_edges[i + 1])
            if i == self._n_bins - 1:
                mask = (raws >= bin_edges[i]) & (raws <= bin_edges[i + 1])
            n_bin = np.sum(mask)
            if n_bin == 0:
                continue
            avg_conf = np.mean(raws[mask])
            accuracy = np.mean(outcomes[mask])
            ece += (n_bin / n_total) * abs(avg_conf - accuracy)

        return float(ece)
