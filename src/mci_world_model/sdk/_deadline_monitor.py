from __future__ import annotations

"""MCI World Model v4.6.0 — DeadlineMonitor 超时检测 + 降级策略
================================================================

硬实时保证的监控组件——检测 CEWM 步骤是否在 deadline 内完成，
超时后自动降级到简化预测器，确保安全关键系统不因计算延迟而失控。

核心能力:
    DeadlineMonitor — 超时检测 + 降级策略
    - configure(deadline_ms) — 设置 deadline
    - start() / stop() — 开始/结束计时
    - check() — 检查是否超时
    - should_degrade() — 判断是否应降级
    - statistics() — p50/p95/p99 耗时统计

设计原则:
    - 最小开销: 计时使用 time.monotonic()，纳秒精度
    - 统计窗口: 滑动窗口统计延迟分布
    - 降级策略: 超时比例超过阈值时触发降级
    - 诚实定位: Python GIL 限制了真正的 WCET 保证

重要声明:
    Phase 3 交付的是架构验证原型。真正的硬实时保证需要:
    RTOS 内核、专用硬件定时器、C++/Rust 热路径。
"""


import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# DeadlineConfig — Deadline 配置
# =============================================================================


@dataclass
class DeadlineConfig:
    """Deadline 监控配置。

    Attributes:
        deadline_ms: 目标 deadline（毫秒），0 表示不限制
        degrade_threshold: 超时比例超过此值时触发降级 [0, 1]
        stats_window: 统计窗口大小（最近 N 次操作）
        degrade_strategy: 降级策略 ('skip_prediction' / 'simplified' / 'abort')
    """

    deadline_ms: int = 0
    degrade_threshold: float = 0.3
    stats_window: int = 1000
    degrade_strategy: str = "simplified"


# =============================================================================
# DeadlineStats — Deadline 统计
# =============================================================================


@dataclass
class DeadlineStats:
    """Deadline 监控统计。

    Attributes:
        total_steps: 总步骤数
        deadline_violations: 超时次数
        violation_rate: 超时率
        p50_ms: p50 延迟
        p95_ms: p95 延迟
        p99_ms: p99 延迟
        max_ms: 最大延迟
        is_degraded: 当前是否处于降级模式
    """

    total_steps: int = 0
    deadline_violations: int = 0
    violation_rate: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0
    is_degraded: bool = False


# =============================================================================
# DeadlineMonitor — 超时检测 + 降级
# =============================================================================


class DeadlineMonitor:
    """超时检测 + 降级监控器。

    用法:
        >>> monitor = DeadlineMonitor(DeadlineConfig(deadline_ms=10))
        >>> monitor.start()
        >>> # ... 执行 cewm_step() ...
        >>> monitor.stop()
        >>> if monitor.should_degrade():
        ...     # 使用简化预测器
        ...     pass
        >>> stats = monitor.statistics()
        >>> print(f"p99: {stats.p99_ms:.2f} ms")
    """

    def __init__(self, config: DeadlineConfig | None = None):
        """
        Args:
            config: Deadline 配置（默认: 无 deadline）
        """
        self._config = config or DeadlineConfig()
        self._start_time: float = 0.0
        self._latencies: deque[float] = deque(maxlen=self._config.stats_window)
        self._total_steps: int = 0
        self._deadline_violations: int = 0
        self._is_degraded: bool = False
        self._lock = threading.Lock()

    @property
    def is_degraded(self) -> bool:
        """当前是否处于降级模式。"""
        return self._is_degraded

    @property
    def config(self) -> DeadlineConfig:
        """当前配置。"""
        return self._config

    def configure(self, config: DeadlineConfig) -> None:
        """更新配置。

        Args:
            config: 新的 Deadline 配置
        """
        with self._lock:
            self._config = config
            self._latencies = deque(maxlen=config.stats_window)

    def start(self) -> None:
        """开始计时（在 cewm_step 开始时调用）。"""
        self._start_time = time.monotonic()

    def stop(self) -> float:
        """停止计时并记录延迟（在 cewm_step 结束时调用）。

        Returns:
            本次步骤的延迟（毫秒）
        """
        elapsed_ms = (time.monotonic() - self._start_time) * 1000.0

        with self._lock:
            self._total_steps += 1
            self._latencies.append(elapsed_ms)

            # 检查 deadline 违规
            if self._config.deadline_ms > 0 and elapsed_ms > self._config.deadline_ms:
                self._deadline_violations += 1
                logger.warning(
                    "Deadline 违规: %.2f ms > %d ms",
                    elapsed_ms,
                    self._config.deadline_ms,
                )

        return elapsed_ms

    def check(self) -> bool:
        """检查当前步骤是否已超时（在步骤执行中调用）。

        Returns:
            True 如果已超时
        """
        if self._config.deadline_ms <= 0:
            return False

        elapsed_ms = (time.monotonic() - self._start_time) * 1000.0
        return elapsed_ms > self._config.deadline_ms

    def should_degrade(self) -> bool:
        """判断是否应该降级。

        基于最近的超时率是否超过阈值。

        Returns:
            True 如果应该降级
        """
        if self._config.deadline_ms <= 0:
            return False

        with self._lock:
            if self._total_steps == 0:
                return False

            rate = self._deadline_violations / self._total_steps
            self._is_degraded = rate > self._config.degrade_threshold
            return self._is_degraded

    def reset_degrade(self) -> None:
        """重置降级状态。"""
        with self._lock:
            self._is_degraded = False
            self._deadline_violations = 0
            self._total_steps = 0
            self._latencies.clear()

    def statistics(self) -> DeadlineStats:
        """计算延迟统计。"""
        with self._lock:
            latencies = sorted(self._latencies)
            n = len(latencies)

            if n == 0:
                return DeadlineStats(
                    total_steps=self._total_steps,
                    deadline_violations=self._deadline_violations,
                    violation_rate=0.0,
                    is_degraded=self._is_degraded,
                )

            violation_rate = self._deadline_violations / self._total_steps if self._total_steps > 0 else 0.0

            return DeadlineStats(
                total_steps=self._total_steps,
                deadline_violations=self._deadline_violations,
                violation_rate=violation_rate,
                p50_ms=latencies[int(n * 0.50)] if n > 0 else 0.0,
                p95_ms=latencies[min(int(n * 0.95), n - 1)] if n > 0 else 0.0,
                p99_ms=latencies[min(int(n * 0.99), n - 1)] if n > 0 else 0.0,
                max_ms=latencies[-1] if n > 0 else 0.0,
                is_degraded=self._is_degraded,
            )

    def percentile(self, p: float) -> float:
        """计算指定百分位的延迟。

        Args:
            p: 百分位 (0.0 - 1.0)

        Returns:
            延迟（毫秒）
        """
        with self._lock:
            latencies = sorted(self._latencies)
            n = len(latencies)
            if n == 0:
                return 0.0
            idx = min(int(n * p), n - 1)
            return latencies[idx]

    def __repr__(self) -> str:
        stats = self.statistics()
        return (
            f"DeadlineMonitor(deadline={self._config.deadline_ms}ms, "
            f"steps={stats.total_steps}, "
            f"violations={stats.deadline_violations}, "
            f"p99={stats.p99_ms:.2f}ms, "
            f"degraded={stats.is_degraded})"
        )
