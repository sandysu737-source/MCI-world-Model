"""North Star 基准协议测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mci_world_model.evaluation import (
    BenchmarkReport,
    MetricResult,
    MetricSpec,
    collect_report,
    write_report,
)


def _metric(
    value: float,
    *,
    benchmark: str = "unit",
    metric: str = "accuracy",
    threshold: float = 0.5,
) -> MetricResult:
    return MetricResult(
        benchmark=benchmark,
        metric=metric,
        value=value,
        threshold=threshold,
        comparison=">=",
        samples=10,
        source="test",
        seed=42,
        config={"suite": "test"},
    )


def test_collect_report_passes_and_serializes() -> None:
    report = collect_report(
        "smoke",
        [MetricSpec("unit.accuracy", lambda: _metric(0.8))],
        commit="abc123",
    )

    assert isinstance(report, BenchmarkReport)
    assert report.status == "pass"
    payload = report.to_dict()
    assert payload["schema_version"] == "1.0"
    assert payload["suite"] == "smoke"
    assert payload["commit"] == "abc123"
    assert payload["status"] == "pass"
    assert payload["metrics"][0]["passed"] is True


def test_collect_report_fails_below_threshold() -> None:
    report = collect_report(
        "full",
        [MetricSpec("unit.accuracy", lambda: _metric(0.4))],
        commit="abc123",
    )

    assert report.status == "fail"


def test_collect_report_rejects_invalid_inputs() -> None:
    spec = MetricSpec("unit.accuracy", lambda: _metric(0.8))

    with pytest.raises(ValueError, match="未知 suite"):
        collect_report("ci", [spec], commit="abc123")
    with pytest.raises(ValueError, match="至少需要一个 metric"):
        collect_report("smoke", [], commit="abc123")
    with pytest.raises(ValueError, match="重复"):
        collect_report("smoke", [spec, spec], commit="abc123")


def test_metric_rejects_nonfinite_value(tmp_path: Path) -> None:
    report = collect_report(
        "smoke",
        [MetricSpec("unit.accuracy", lambda: _metric(float("nan")))],
        commit="abc123",
    )

    assert report.status == "fail"
    output = tmp_path / "report.json"
    write_report(report, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
