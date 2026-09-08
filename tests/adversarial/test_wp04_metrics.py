"""WP-04 Metrics 基数与响应边界对抗测试。"""

from __future__ import annotations

import pytest

from mci_world_model.server.metrics import MAX_EXPOSURE_BYTES, MetricsCollector

pytestmark = pytest.mark.contract


def test_unknown_paths_and_query_strings_use_unmatched_label() -> None:
    collector = MetricsCollector()

    assert collector.normalize_endpoint("/health?verbose=1") == "/health"
    assert collector.normalize_endpoint("/api/v1/diagnosis/record-1?scope=a") == "/api/v1/diagnosis/"
    assert collector.normalize_endpoint('/unknown/..%2f?x="\\n') == "unmatched"

    collector.inc_request('/unknown/..%2f?x="\\n')
    rendered = collector.expose()

    assert 'requests_total{endpoint="unmatched"} 1' in rendered
    assert "\\n" not in rendered
    assert '"' not in rendered.split('endpoint="', 1)[1].split('"', 2)[1]


def test_repeated_paths_aggregate_under_fixed_label() -> None:
    collector = MetricsCollector()

    for _ in range(3):
        collector.inc_request("/api/v1/diagnose?candidate=A")

    assert 'requests_total{endpoint="/api/v1/diagnose"} 3' in collector.expose()


def test_two_hundred_random_paths_do_not_expand_metric_schema() -> None:
    collector = MetricsCollector()

    for index in range(200):
        path = f"/random/{index}?token={index}"
        collector.inc_request(path)
        collector.inc_error(path)
        collector.observe_latency(path, 0.001)

    rendered = collector.expose()

    assert len(collector._label_rings["requests_total"]) == 1
    assert len(collector._label_rings["errors_total"]) == 1
    assert len(collector._label_rings["request_duration"]) == 1
    assert collector._counters.get("metrics_cardinality_dropped_total", 0) == 0
    assert rendered.count('endpoint="unmatched"') == 14
    assert rendered.count("\n") <= 512
    assert 'endpoint="/random/' not in rendered
    assert "token=" not in rendered


def test_label_ring_cap_discards_oldest_combination() -> None:
    collector = MetricsCollector()

    for index in range(200):
        collector.inc(f'requests_total{{endpoint="/generic/{index}"}}')

    assert len(collector._label_rings["requests_total"]) == 64
    assert collector._counters["metrics_cardinality_dropped_total"] == 136
    assert collector._counters['requests_total{endpoint="/generic/199"}'] == 1


def test_exposure_has_hard_byte_limit() -> None:
    collector = MetricsCollector()
    for index in range(5000):
        collector.set_gauge(f"custom_gauge_{index:05d}", float(index))

    collector.expose()
    rendered = collector.expose()

    assert len(rendered.encode("utf-8")) <= MAX_EXPOSURE_BYTES
    assert "metrics_exposure_truncated_total" in rendered
