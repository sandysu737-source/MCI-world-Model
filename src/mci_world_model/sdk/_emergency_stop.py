"""MCI World Model v4.5.0 — EmergencyStop 紧急停止
====================================================

安全关键系统的紧急停止机制——独立线程监听停止信号，
在 50ms 内设置全局停止标志，使 cewm_step() 返回紧急停止状态。

核心能力:
    EmergencyStop — 紧急停止控制器
    - 信号注入: trigger() / signal_handler() (SIGUSR1)
    - 状态查询: is_stopped / is_armed
    - 超时保护: arm(timeout_ms) / disarm()
    - 恢复: reset()

设计原则:
    - 独立于 CEWM 主循环，不依赖任何 CEWM 组件
    - stop 标志使用 threading.Event，确保线程安全
    - 信号处理器仅设置标志，不做任何复杂操作
    - 诚实定位: Python GIL 限制了真正的硬实时保证

重要声明:
    Phase 3 交付的是架构验证原型，不是生产级紧急停止系统。
    生产部署需要: RTOS 内核驱动、硬件看门狗、双通道冗余。
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# EmergencyStopState — 紧急停止状态
# =============================================================================


@dataclass
class EmergencyStopState:
    """紧急停止状态快照。

    Attributes:
        is_stopped: 是否已触发紧急停止
        is_armed: 是否已武装（激活超时监控）
        trigger_count: 触发次数
        last_trigger_time: 最后触发时间戳 (monotonic)
        last_trigger_reason: 最后触发原因
        arm_timeout_ms: 武装超时 (ms)
    """

    is_stopped: bool = False
    is_armed: bool = False
    trigger_count: int = 0
    last_trigger_time: float = 0.0
    last_trigger_reason: str = ""
    arm_timeout_ms: int = 0


# =============================================================================
# EmergencyStop — 紧急停止控制器
# =============================================================================


class EmergencyStop:
    """紧急停止控制器——独立线程安全地监听和响应停止信号。

    用法:
        >>> estop = EmergencyStop()
        >>> estop.arm(timeout_ms=5000)  # 武装 5 秒
        >>> # ... CEWM 循环中 ...
        >>> if estop.is_stopped:
        ...     # 立即停止
        >>> estop.disarm()  # 解除武装

    信号触发:
        >>> # 在另一个进程中
        >>> os.kill(pid, signal.SIGUSR1)
    """

    # 类级别单例引用（信号处理器需要）
    _instance: EmergencyStop | None = None
    _instance_lock: threading.Lock = threading.Lock()  # v4.5.0: 多实例安全锁

    def __init__(
        self,
        auto_install_signal: bool = False,
        on_stop: Callable[[str], None] | None = None,
    ):
        """
        Args:
            auto_install_signal: 是否自动安装 SIGUSR1 信号处理器
            on_stop: 紧急停止回调函数（接收触发原因）
        """
        self._stop_event = threading.Event()
        self._armed_event = threading.Event()
        self._lock = threading.Lock()
        self._trigger_count: int = 0
        self._last_trigger_time: float = 0.0
        self._last_trigger_reason: str = ""
        self._arm_timeout_ms: int = 0
        self._arm_start_time: float = 0.0
        self._on_stop = on_stop
        self._monitor_thread: threading.Thread | None = None
        self._monitor_stop = threading.Event()

        # 注册为类级别实例（线程安全）
        with EmergencyStop._instance_lock:
            EmergencyStop._instance = self

        if auto_install_signal:
            self.install_signal_handler()

    @property
    def is_stopped(self) -> bool:
        """是否已触发紧急停止。"""
        return self._stop_event.is_set()

    @property
    def is_armed(self) -> bool:
        """是否已武装。"""
        return self._armed_event.is_set()

    @property
    def trigger_count(self) -> int:
        """触发次数。"""
        with self._lock:
            return self._trigger_count

    def trigger(self, reason: str = "manual") -> None:
        """手动触发紧急停止。

        Args:
            reason: 触发原因
        """
        with self._lock:
            self._trigger_count += 1
            self._last_trigger_time = time.monotonic()
            self._last_trigger_reason = reason

        self._stop_event.set()
        self._armed_event.clear()

        if self._on_stop is not None:
            try:
                self._on_stop(reason)
            except Exception as e:
                logger.error("紧急停止回调异常: %s", e)

        logger.critical("紧急停止触发: %s (第 %d 次)", reason, self._trigger_count)

    def arm(self, timeout_ms: int = 0) -> None:
        """武装紧急停止——激活超时监控。

        Args:
            timeout_ms: 超时时间（毫秒），0 表示不设超时
        """
        with self._lock:
            self._arm_timeout_ms = timeout_ms
            self._arm_start_time = time.monotonic()

        self._armed_event.set()

        if timeout_ms > 0:
            # 启动超时监控线程
            self._start_monitor(timeout_ms)

        logger.info("紧急停止已武装 (超时: %d ms)", timeout_ms)

    def disarm(self) -> None:
        """解除武装——停止超时监控。"""
        self._armed_event.clear()
        self._stop_monitor()
        logger.info("紧急停止已解除武装")

    def reset(self) -> None:
        """重置紧急停止状态——清除停止标志。

        注意: 仅在确认安全后才应调用此方法。
        """
        self._stop_event.clear()
        self._armed_event.clear()
        self._stop_monitor()
        logger.info("紧急停止已重置")

    def check(self) -> EmergencyStopState:
        """获取当前状态快照。"""
        with self._lock:
            return EmergencyStopState(
                is_stopped=self._stop_event.is_set(),
                is_armed=self._armed_event.is_set(),
                trigger_count=self._trigger_count,
                last_trigger_time=self._last_trigger_time,
                last_trigger_reason=self._last_trigger_reason,
                arm_timeout_ms=self._arm_timeout_ms,
            )

    def wait_for_stop(self, timeout: float = -1.0) -> bool:
        """等待紧急停止信号。

        Args:
            timeout: 等待超时（秒），-1 表示无限等待

        Returns:
            True 如果收到停止信号，False 如果超时
        """
        if timeout < 0:
            self._stop_event.wait()
            return True
        return self._stop_event.wait(timeout=timeout)

    # ── 信号处理器 ──

    def install_signal_handler(self) -> None:
        """安装 SIGUSR1 信号处理器。"""
        try:
            signal.signal(signal.SIGUSR1, self._signal_handler)
            logger.info("SIGUSR1 信号处理器已安装 (PID: %d)", os.getpid())
        except (OSError, ValueError) as e:
            logger.warning("无法安装信号处理器: %s", e)

    def uninstall_signal_handler(self) -> None:
        """卸载信号处理器。"""
        try:
            signal.signal(signal.SIGUSR1, signal.SIG_DFL)
            logger.info("SIGUSR1 信号处理器已卸载")
        except (OSError, ValueError):
            pass

    @classmethod
    def _signal_handler(cls, signum: int, frame: Any) -> None:
        """SIGUSR1 信号处理器——设置停止标志。"""
        if cls._instance is not None:
            cls._instance.trigger(reason=f"SIGUSR1 (signal {signum})")
        else:
            logger.warning("收到 SIGUSR1 但无 EmergencyStop 实例")

    # ── 超时监控线程 ──

    def _start_monitor(self, timeout_ms: int) -> None:
        """启动超时监控线程。"""
        self._stop_monitor()
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(timeout_ms,),
            daemon=True,
            name="EmergencyStop-Monitor",
        )
        self._monitor_thread.start()

    def _stop_monitor(self) -> None:
        """停止超时监控线程。"""
        self._monitor_stop.set()
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=0.1)
        self._monitor_thread = None

    def _monitor_loop(self, timeout_ms: int) -> None:
        """超时监控循环。"""
        timeout_s = timeout_ms / 1000.0
        deadline = self._arm_start_time + timeout_s

        while not self._monitor_stop.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.trigger(reason=f"arm_timeout ({timeout_ms} ms)")
                return
            # 每 10ms 检查一次
            self._monitor_stop.wait(timeout=min(0.01, remaining))

    def __repr__(self) -> str:
        state = self.check()
        return f"EmergencyStop(stopped={state.is_stopped}, armed={state.is_armed}, triggers={state.trigger_count})"
