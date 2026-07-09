"""MCI World Model 指标收集器 — 零依赖轻量实现。

提供 Prometheus 兼容的指标格式 (text exposition)，无需第三方库。
支持: Counter, Histogram (分位数), Gauge。

用法:
    from mci_world_model.server.metrics import metrics
    metrics.inc_request("diagnose")
    metrics.observe_latency("diagnose", 0.012)
    print(metrics.expose())  # Prometheus text format
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class _Histogram:
    """简易直方图: 固定桶 + 分位数近似。"""
    buckets: list[float] = field(default_factory=lambda: [
        0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0
    ])
    counts: list[int] = field(default_factory=lambda: [0] * 9)
    total: int = 0
    sum_value: float = 0.0

    def observe(self, value: float) -> None:
        self.total += 1
        self.sum_value += value
        for i, b in enumerate(self.buckets):
            if value <= b:
                self.counts[i] += 1
                break
        else:
            # 超过最大桶
            pass

    def quantile(self, q: float) -> float:
        if self.total == 0:
            return 0.0
        target = q * self.total
        cumulative = 0
        for i, c in enumerate(self.counts):
            cumulative += c
            if cumulative >= target:
                return self.buckets[i]
        return self.buckets[-1]


class MetricsCollector:
    """线程安全的指标收集器。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, _Histogram] = defaultdict(_Histogram)
        self._gauges: dict[str, float] = {}
        self._start_time = time.time()

    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._histograms[name].observe(value)

    def inc_request(self, endpoint: str) -> None:
        self.inc(f"requests_total{{endpoint=\"{endpoint}\"}}")

    def inc_error(self, endpoint: str) -> None:
        self.inc(f"errors_total{{endpoint=\"{endpoint}\"}}")

    def observe_latency(self, endpoint: str, seconds: float) -> None:
        self.observe(f"request_duration{{endpoint=\"{endpoint}\"}}", seconds)

    def uptime(self) -> float:
        return time.time() - self._start_time

    def expose(self) -> str:
        """Prometheus text exposition format。"""
        with self._lock:
            lines = []
            # Counters
            for name, value in sorted(self._counters.items()):
                lines.append(f"# TYPE {name.split('{')[0]} counter")
                lines.append(f"{name} {value}")
            # Gauges
            for name, value in sorted(self._gauges.items()):
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value}")
            # Histograms
            for name, hist in sorted(self._histograms.items()):
                parts = name.split('{')
                base = parts[0]
                lines.append(f"# TYPE {base} histogram")
                # tag 是完整的标签串如 endpoint="diagnose"
                tag = parts[1].rstrip('}') if len(parts) > 1 else ''
                cumulative = 0
                for i, b in enumerate(hist.buckets):
                    cumulative += hist.counts[i]
                    label = f'{tag},le="{b}"' if tag else f'le="{b}"'
                    lines.append(f'{base}_bucket{{{label}}} {cumulative}')
                inf_label = f'{tag},le="+Inf"' if tag else 'le="+Inf"'
                lines.append(f'{base}_bucket{{{inf_label}}} {hist.total}')
                lines.append(f'{base}_count{{{tag}}} {hist.total}')
                lines.append(f'{base}_sum{{{tag}}} {hist.sum_value:.6f}')
            # Uptime
            lines.append(f"# TYPE mci_uptime gauge")
            lines.append(f"mci_uptime {self.uptime():.2f}")
            return '\n'.join(lines) + '\n'


# 全局单例
metrics = MetricsCollector()
