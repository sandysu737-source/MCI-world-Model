#!/usr/bin/env python3
"""运行 North Star 基准并输出统一 JSON 报告。"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.real_world import tuebingen_pairs
from mci_world_model.evaluation import (
    BenchmarkReport,
    MetricResult,
    MetricSpec,
    collect_report,
    write_report,
)

SEED = 42


def _import(module_name: str, attribute: str) -> Any:
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def _current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _run_tuebingen_like(n_pairs: int) -> MetricResult:
    """离线运行固定 seed 的 Tübingen-like 方向基准。"""
    generate_pairs = _import("benchmarks.real_world.tuebingen_pairs", "_generate_synthetic_pairs")
    load_pairs = _import("benchmarks.real_world.tuebingen_pairs", "load_tuebingen_pairs")
    evaluate_direction = _import("benchmarks.real_world.tuebingen_pairs", "evaluate_direction")

    with tempfile.TemporaryDirectory(prefix="mci-north-star-") as temp_dir:
        original_dir = tuebingen_pairs.TUEBINGEN_DIR
        tuebingen_pairs.TUEBINGEN_DIR = Path(temp_dir) / "tuebingen_data"
        try:
            generate_pairs(n_pairs)
            result = evaluate_direction(load_pairs())
        finally:
            tuebingen_pairs.TUEBINGEN_DIR = original_dir

    return MetricResult(
        benchmark="tuebingen_like",
        metric="direction_accuracy",
        value=float(result["accuracy"]),
        threshold=0.40,
        comparison=">=",
        samples=int(result["total"]),
        source="synthetic-offline",
        seed=SEED,
        config={"n_pairs": n_pairs},
    )


def _run_bnlearn_f1(dag_names: list[str]) -> MetricResult:
    """运行 PC 结构发现并返回多个 BNLearn DAG 的平均 F1。"""
    generate_dag_data = _import("benchmarks.bnlearn.test_bnlearn_benchmark", "_generate_dag_data")
    precision_recall_f1 = _import("benchmarks.bnlearn.test_bnlearn_benchmark", "_precision_recall_f1")
    discoverer_type = _import(
        "mci_world_model.sdk._autonomous_law_discoverer_v2",
        "PCSkeletonDiscoverer",
    )

    values: list[float] = []
    for dag_name in dag_names:
        data, nodes, ground_truth, _ = generate_dag_data(dag_name)
        discoverer = discoverer_type(alpha=0.50, min_corr=0.05, nonlinear=True)
        skeleton = discoverer.discover(data, nodes)
        _precision, _recall, f1 = precision_recall_f1(skeleton.adj_matrix, ground_truth)
        values.append(float(f1))

    return MetricResult(
        benchmark="bnlearn",
        metric="mean_structure_f1",
        value=sum(values) / len(values),
        threshold=0.35,
        comparison=">=",
        samples=len(values),
        source="synthetic-standard-dag",
        seed=SEED,
        config={"dags": dag_names, "method": "PC"},
    )


def _run_mimic_like_f1(n_patients: int) -> MetricResult:
    """运行固定 seed 的合成 ICU 因果边发现基准。"""
    benchmark_type = _import("benchmarks.real_world.mimic_causal_benchmark", "MIMICCausalBenchmark")
    generate_patients = _import(
        "benchmarks.real_world.mimic_causal_benchmark",
        "generate_synthetic_icu_patients",
    )
    benchmark = benchmark_type()
    patients = generate_patients(n_patients=n_patients, seed=SEED)
    result = benchmark.run_cewm_benchmark(patients)
    return MetricResult(
        benchmark="mimic_like",
        metric="edge_f1",
        value=float(result.metrics.f1),
        threshold=0.40,
        comparison=">=",
        samples=len(patients),
        source="synthetic-icu",
        seed=SEED,
        config={"n_patients": n_patients},
    )


def _metric_specs(suite: str) -> list[MetricSpec]:
    if suite == "smoke":
        return [
            MetricSpec("tuebingen_like.direction_accuracy", lambda: _run_tuebingen_like(50)),
            MetricSpec("bnlearn.mean_structure_f1", lambda: _run_bnlearn_f1(["asia"])),
            MetricSpec("mimic_like.edge_f1", lambda: _run_mimic_like_f1(20)),
        ]
    return [
        MetricSpec("tuebingen_like.direction_accuracy", lambda: _run_tuebingen_like(108)),
        MetricSpec("bnlearn.mean_structure_f1", lambda: _run_bnlearn_f1(["asia", "sachs", "child"])),
        MetricSpec("mimic_like.edge_f1", lambda: _run_mimic_like_f1(50)),
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", type=Path, default=Path("docs/north-star-result.json"))
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    suite = args.suite
    specs: list[MetricSpec] = _metric_specs(suite)
    report: BenchmarkReport = collect_report(suite, specs, commit=_current_commit())
    output = write_report(report, args.output)
    print(f"status={report.status} metrics={len(report.metrics)} report={output}")
    return 0 if report.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
