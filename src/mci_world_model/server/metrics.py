"""MCI World Model 指标收集器 — 零依赖轻量实现。

提供 Prometheus 兼容的指标格式 (text exposition)，无需第三方库。
支持: Counter, Histogram (分位数), Gauge。

用法:
    from mci_world_model.server.metrics import metrics
    metrics.inc_request("diagnose")
    metrics.observe_latency("diagnose", 0.012)
    logger.info(metrics.expose())  # Prometheus text format)
"""

from __future__ import annotations

import logging
from collections import OrderedDict, defaultdict

logger = logging.getLogger(__name__)
import threading
import time
from dataclasses import dataclass, field

KNOWN_ENDPOINTS: frozenset[str] = frozenset(
    {
        "/health",
        "/ready",
        "/metrics",
        "/api/v1/diagnose",
        "/api/v1/diagnose/batch",
        "/api/v1/diagnosis/",
        "/api/v1/backdoor",
        "/api/v1/energy/what_if",
    }
)
MAX_LABEL_COMBINATIONS = 64
MAX_EXPOSURE_BYTES = 64 * 1024


@dataclass
class _Histogram:
    """简易直方图: 固定桶 + 分位数近似。"""

    buckets: list[float] = field(default_factory=lambda: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0])
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
        self._counters: dict[str, float] = {}
        self._histograms: dict[str, _Histogram] = defaultdict(_Histogram)
        self._gauges: dict[str, float] = {}
        self._label_rings: dict[str, OrderedDict[str, None]] = {}
        self._max_label_combinations = MAX_LABEL_COMBINATIONS
        self._start_time = time.time()

    @staticmethod
    def normalize_endpoint(endpoint: str) -> str:
        """把路由收敛到固定标签；query string 和动态 ID 不进入指标。"""
        path = endpoint.split("?", 1)[0]
        if path in KNOWN_ENDPOINTS:
            return path
        if path.startswith("/api/v1/diagnosis/"):
            return "/api/v1/diagnosis/"
        return "unmatched"

    @staticmethod
    def _split_metric_key(key: str) -> tuple[str, str]:
        base, separator, label = key.partition("{")
        if not separator:
            return base, ""
        return base, label[:-1] if label.endswith("}") else label

    def _admit_label(self, metric_key: str) -> bool:
        base, label = self._split_metric_key(metric_key)
        if not label:
            return True
        ring = self._label_rings.setdefault(base, OrderedDict())
        if label in ring:
            ring.move_to_end(label)
            return True
        if len(ring) >= self._max_label_combinations:
            ring.popitem(last=False)
            self._counters["metrics_cardinality_dropped_total"] = (
                self._counters.get("metrics_cardinality_dropped_total", 0.0) + 1.0
            )
        ring[label] = None
        return True

    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            if not self._admit_label(name):
                return
            self._counters[name] = self._counters.get(name, 0.0) + value

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            if not self._admit_label(name):
                return
            self._histograms[name].observe(value)

    def inc_request(self, endpoint: str) -> None:
        safe_endpoint = self.normalize_endpoint(endpoint)
        self.inc(f'requests_total{{endpoint="{safe_endpoint}"}}')

    def inc_error(self, endpoint: str) -> None:
        safe_endpoint = self.normalize_endpoint(endpoint) if endpoint.startswith("/") else endpoint
        self.inc(f'errors_total{{endpoint="{safe_endpoint}"}}')

    def observe_latency(self, endpoint: str, seconds: float) -> None:
        safe_endpoint = self.normalize_endpoint(endpoint)
        self.observe(f'request_duration{{endpoint="{safe_endpoint}"}}', seconds)

    def uptime(self) -> float:
        return time.time() - self._start_time

    def expose(self) -> str:
        """渲染有长度上限的 Prometheus text exposition format。"""
        started_at = time.perf_counter()
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
                parts = name.split("{")
                base = parts[0]
                lines.append(f"# TYPE {base} histogram")
                # tag 是完整的标签串如 endpoint="diagnose"
                tag = parts[1].rstrip("}") if len(parts) > 1 else ""
                cumulative = 0
                for i, b in enumerate(hist.buckets):
                    cumulative += hist.counts[i]
                    label = f'{tag},le="{b}"' if tag else f'le="{b}"'
                    lines.append(f"{base}_bucket{{{label}}} {cumulative}")
                inf_label = f'{tag},le="+Inf"' if tag else 'le="+Inf"'
                lines.append(f"{base}_bucket{{{inf_label}}} {hist.total}")
                lines.append(f"{base}_count{{{tag}}} {hist.total}")
                lines.append(f"{base}_sum{{{tag}}} {hist.sum_value:.6f}")
            # Uptime
            lines.append("# TYPE mci_uptime gauge")
            lines.append(f"mci_uptime {self.uptime():.2f}")

        rendered: list[str] = []
        rendered_bytes = 0
        truncated = 0
        for line in lines:
            line_bytes = len(line.encode("utf-8")) + 1
            if rendered_bytes + line_bytes > MAX_EXPOSURE_BYTES:
                truncated += 1
                continue
            rendered.append(line)
            rendered_bytes += line_bytes

        elapsed = time.perf_counter() - started_at
        if truncated:
            self.inc("metrics_exposure_truncated_total", float(truncated))
        self.observe("metrics_exposure_duration_seconds", elapsed)
        return "\n".join(rendered) + "\n"


# 全局单例
metrics = MetricsCollector()
