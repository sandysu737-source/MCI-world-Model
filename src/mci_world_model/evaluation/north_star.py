"""North Star 基准的统一报告协议。"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = "1.0"
SUITES = frozenset({"smoke", "full"})
SuiteName = Literal["smoke", "full"]


@dataclass(frozen=True)
class MetricResult:
    """单个基准指标的可追溯结果。"""

    benchmark: str
    metric: str
    value: float
    threshold: float
    comparison: Literal[">=", "<="]
    samples: int
    source: str
    seed: int
    config: Mapping[str, Any]

    @property
    def passed(self) -> bool:
        if not math.isfinite(self.value) or not math.isfinite(self.threshold):
            return False
        if self.comparison == ">=":
            return self.value >= self.threshold
        return self.value <= self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "passed": self.passed,
            "samples": self.samples,
            "source": self.source,
            "seed": self.seed,
            "config": dict(self.config),
        }


@dataclass(frozen=True)
class MetricSpec:
    """延迟执行的基准规格，保证只有声明的指标会运行。"""

    key: str
    runner: Callable[[], MetricResult]


@dataclass(frozen=True)
class BenchmarkReport:
    """统一 North Star 报告。"""

    schema_version: str
    suite: SuiteName
    commit: str
    metrics: tuple[MetricResult, ...]

    @property
    def status(self) -> Literal["pass", "fail"]:
        return "pass" if all(metric.passed for metric in self.metrics) else "fail"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite": self.suite,
            "commit": self.commit,
            "status": self.status,
            "metrics": [metric.to_dict() for metric in self.metrics],
        }


def collect_report(
    suite: SuiteName,
    specs: Sequence[MetricSpec],
    *,
    commit: str,
) -> BenchmarkReport:
    """执行声明式指标并聚合报告。"""
    if suite not in SUITES:
        raise ValueError(f"未知 suite: {suite}")

    keys = [spec.key for spec in specs]
    if len(keys) != len(set(keys)):
        raise ValueError("metric key 存在重复")
    if not specs:
        raise ValueError("至少需要一个 metric")

    results = tuple(spec.runner() for spec in specs)
    return BenchmarkReport(
        schema_version=SCHEMA_VERSION,
        suite=suite,
        commit=commit,
        metrics=results,
    )


def write_report(report: BenchmarkReport, output: Path) -> Path:
    """写出 UTF-8 JSON 报告。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
