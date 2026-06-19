from __future__ import annotations

"""MCI World Model v7.0.0 — AutoScaler 推理服务自动伸缩
========================================================

QPS 驱动的推理服务自动伸缩 — 根据负载动态调整服务副本数。

核心能力:
    compute_desired_replicas(current_qps, avg_latency_ms) — 计算目标副本数
    record_metrics(qps, latency, replicas)                  — 记录指标历史

设计原则:
    - 纯 numpy，零外部依赖
    - 基于延迟反馈的伸缩策略
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# ScaleDecision — 伸缩决策
# =============================================================================


@dataclass
class ScaleDecision:
    """伸缩决策结果。

    Attributes:
        desired_replicas: 目标副本数
        current_replicas: 当前副本数
        action: 伸缩动作 ('scale_up' / 'scale_down' / 'hold')
        reason: 决策原因
        qps: 当前 QPS
        latency_ms: 当前平均延迟
    """

    desired_replicas: int
    current_replicas: int
    action: str
    reason: str
    qps: float = 0.0
    latency_ms: float = 0.0


# =============================================================================
# AutoScaler — 推理服务自动伸缩
# =============================================================================


class AutoScaler:
    """推理服务自动伸缩 — QPS + 延迟驱动的副本数调整。

    伸缩策略:
      - scale_up: 平均延迟 > target * 1.5 时，按 QPS/100 + 1 扩容
      - scale_down: 平均延迟 < target * 0.5 时，按 QPS/150 缩容
      - hold: 延迟在正常范围内，维持 QPS/100

    Attributes:
        _min_replicas: 最小副本数
        _max_replicas: 最大副本数
        _target_latency_ms: 目标延迟 (ms)
        _metrics_history: 指标历史
    """

    def __init__(
        self,
        min_replicas: int = 1,
        max_replicas: int = 10,
        target_latency_ms: float = 50.0,
        qps_per_replica: float = 100.0,
        cooldown_seconds: float = 30.0,
    ):
        if min_replicas < 1:
            raise ValueError(f"min_replicas 必须 ≥ 1, 当前 {min_replicas}")
        if max_replicas < min_replicas:
            raise ValueError(f"max_replicas ({max_replicas}) < min_replicas ({min_replicas})")
        if target_latency_ms <= 0:
            raise ValueError(f"target_latency_ms 必须 > 0, 当前 {target_latency_ms}")
        self._min = min_replicas
        self._max = max_replicas
        self._target = target_latency_ms
        self._qps_per_replica = qps_per_replica
        self._cooldown = cooldown_seconds
        self._metrics_history: list[dict[str, Any]] = []
        self._current_replicas = min_replicas

    def compute_desired_replicas(self, current_qps: float, avg_latency_ms: float) -> ScaleDecision:
        """根据 QPS 和延迟计算目标副本数。

        Args:
            current_qps: 当前每秒请求数
            avg_latency_ms: 当前平均延迟 (ms)

        Returns:
            ScaleDecision 伸缩决策
        """
        if current_qps < 0:
            raise ValueError(f"current_qps 必须 ≥ 0, 当前 {current_qps}")
        if avg_latency_ms < 0:
            raise ValueError(f"avg_latency_ms 必须 ≥ 0, 当前 {avg_latency_ms}")

        current = self._current_replicas

        if avg_latency_ms > self._target * 1.5:
            # 延迟过高 → 扩容
            desired = int(current_qps / self._qps_per_replica) + 1
            desired = min(desired, self._max)
            action = "scale_up"
            reason = f"延迟 {avg_latency_ms:.0f}ms > 阈值 {self._target * 1.5:.0f}ms, QPS={current_qps:.0f}"
        elif avg_latency_ms < self._target * 0.5:
            # 延迟过低 → 缩容
            desired = max(int(current_qps / (self._qps_per_replica * 1.5)), self._min)
            action = "scale_down"
            reason = f"延迟 {avg_latency_ms:.0f}ms < 阈值 {self._target * 0.5:.0f}ms, 可释放资源"
        else:
            # 延迟正常 → 维持
            desired = max(int(current_qps / self._qps_per_replica), self._min)
            action = "hold"
            reason = f"延迟 {avg_latency_ms:.0f}ms 在正常范围"

        desired = np.clip(desired, self._min, self._max)
        self._current_replicas = int(desired)

        decision = ScaleDecision(
            desired_replicas=int(desired),
            current_replicas=current,
            action=action,
            reason=reason,
            qps=current_qps,
            latency_ms=avg_latency_ms,
        )

        self._metrics_history.append(
            {
                "qps": current_qps,
                "latency_ms": avg_latency_ms,
                "replicas_before": current,
                "replicas_after": int(desired),
                "action": action,
            }
        )

        return decision

    def record_metrics(self, qps: float, latency_ms: float, replicas: int) -> None:
        """记录指标历史。

        Args:
            qps: 当前 QPS
            latency_ms: 当前延迟
            replicas: 当前副本数
        """
        self._metrics_history.append(
            {
                "qps": qps,
                "latency_ms": latency_ms,
                "replicas": replicas,
                "action": "record",
            }
        )

    @property
    def current_replicas(self) -> int:
        return self._current_replicas

    @property
    def metrics_history(self) -> list[dict[str, Any]]:
        return list(self._metrics_history)

    @property
    def min_replicas(self) -> int:
        return self._min

    @property
    def max_replicas(self) -> int:
        return self._max
