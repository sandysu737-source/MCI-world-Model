"""CEWM v4.5.0 CounterfactualOracle 延迟基准

测量 CounterfactualOracle 在不同场景数下的推演延迟。
KPI 门禁: N=10 场景 < 500ms

用法:
    python -m benchmarks.cf_oracle_latency
    python -m benchmarks.cf_oracle_latency --n-scenarios 20
"""

from __future__ import annotations

import argparse
import json
import time

from mci_world_model.sdk import CFScenario, CounterfactualOracle


def _measure_query(oracle: CounterfactualOracle, n_scenarios: int, n_runs: int = 10) -> dict:
    """测量 CF query 延迟。"""
    scenarios = [
        CFScenario(
            name=f"方案{chr(65 + i)}",
            description=f"假设方案 {chr(65 + i)}",
            intervention={"treatment": chr(65 + i)},
            target="nutrition_index",
        )
        for i in range(n_scenarios)
    ]

    latencies: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        oracle.query(
            hypotheses=[{"name": s.name, "intervention": s.intervention, "target": s.target} for s in scenarios],
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies.sort()
    return {
        "n_scenarios": n_scenarios,
        "n_runs": n_runs,
        "p50_ms": latencies[int(len(latencies) * 0.50)] if latencies else 0.0,
        "p95_ms": latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)] if latencies else 0.0,
        "p99_ms": latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)] if latencies else 0.0,
        "max_ms": latencies[-1] if latencies else 0.0,
        "mean_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "gate": "PASS" if (latencies[-1] if latencies else 0) < 500.0 else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CounterfactualOracle 延迟基准")
    parser.add_argument("--n-scenarios", type=int, default=10, help="场景数量 (默认: 10)")
    parser.add_argument("--n-runs", type=int, default=10, help="运行次数 (默认: 10)")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="输出格式")
    args = parser.parse_args()

    oracle = CounterfactualOracle(world_model=None)

    results = []
    for n in [3, 5, 10, args.n_scenarios]:
        print(f"  测量 N={n} 场景...", flush=True)
        result = _measure_query(oracle, n_scenarios=n, n_runs=args.n_runs)
        results.append(result)

    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        print("\n" + "=" * 60)
        print("  CounterfactualOracle 延迟基准")
        print("=" * 60)
        for r in results:
            print(
                f"  N={r['n_scenarios']:>2d}: "
                f"p50={r['p50_ms']:.2f}ms  p95={r['p95_ms']:.2f}ms  "
                f"p99={r['p99_ms']:.2f}ms  max={r['max_ms']:.2f}ms  "
                f"gate={r['gate']}"
            )
        overall = "PASS" if all(r["gate"] == "PASS" for r in results) else "FAIL"
        print(f"\n  总体门禁: {overall}")
        print("=" * 60)

    if overall == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
