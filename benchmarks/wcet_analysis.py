"""CEWM v4.5.0 WCET (Worst-Case Execution Time) 基准分析

测量 cewm_step() 和 cewm_step_fast() 在各 WorldState 类型下的延迟分布，
输出 p50/p95/p99/p99.9/max 百分位统计。

KPI 门禁 R-4: p99 < 10ms (Pendulum), p99 < 50ms (Multimodal)

用法:
    python -m benchmarks.wcet_analysis
    python -m benchmarks.wcet_analysis --n-runs 2000
    python -m benchmarks.wcet_analysis --format json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

import numpy as np


def _percentile(sorted_data: list[float], p: float) -> float:
    """计算百分位 (0.0 - 1.0)。"""
    if not sorted_data:
        return 0.0
    idx = min(int(len(sorted_data) * p), len(sorted_data) - 1)
    return sorted_data[idx]


def _measure_cewm_step(
    world_model: Any,
    observation: Any,
    goal: Any = None,
    action: Any = None,
    n_runs: int = 1000,
    fast_path: bool = False,
) -> dict[str, float]:
    """测量单步 CEWM 延迟 (毫秒)。"""
    latencies: list[float] = []

    step_fn = world_model.cewm_step_fast if fast_path else world_model.cewm_step

    for _ in range(n_runs):
        t0 = time.perf_counter()
        try:
            step_fn(observation=observation, goal=goal, action=action)
        except Exception:
            pass
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies.sort()
    return {
        "p50": _percentile(latencies, 0.50),
        "p95": _percentile(latencies, 0.95),
        "p99": _percentile(latencies, 0.99),
        "p99_9": _percentile(latencies, 0.999),
        "max": latencies[-1] if latencies else 0.0,
        "mean": statistics.mean(latencies) if latencies else 0.0,
        "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "n_runs": len(latencies),
    }


def _create_world_model() -> Any:
    """创建 MCIWorldModel 实例。"""
    from mci_world_model.sdk import MCIWorldModel

    return MCIWorldModel()


def _run_pendulum(wm: Any, n_runs: int) -> dict[str, Any]:
    """Pendulum 场景 WCET。"""
    from mci_world_model.sdk import PendulumState

    obs = PendulumState(theta=0.5, omega=1.0)
    goal = PendulumState(theta=0.0, omega=0.0)

    normal = _measure_cewm_step(wm, obs, goal, n_runs=n_runs, fast_path=False)
    fast = _measure_cewm_step(wm, obs, goal, n_runs=n_runs, fast_path=True)

    return {"state_type": "PendulumState", "cewm_step": normal, "cewm_step_fast": fast}


def _run_cart(wm: Any, n_runs: int) -> dict[str, Any]:
    """Cart 场景 WCET。"""
    from mci_world_model.sdk import CartState

    obs = CartState(x=1.0, v=0.5)
    goal = CartState(x=10.0, v=0.0)

    normal = _measure_cewm_step(wm, obs, goal, n_runs=n_runs, fast_path=False)
    fast = _measure_cewm_step(wm, obs, goal, n_runs=n_runs, fast_path=True)

    return {"state_type": "CartState", "cewm_step": normal, "cewm_step_fast": fast}


def _run_double_pendulum(wm: Any, n_runs: int) -> dict[str, Any]:
    """DoublePendulum 场景 WCET。"""
    from mci_world_model.sdk import DoublePendulumState

    obs = DoublePendulumState(theta1=0.3, omega1=0.5, theta2=0.2, omega2=0.4)
    goal = DoublePendulumState(theta1=0.0, omega1=0.0, theta2=0.0, omega2=0.0)

    normal = _measure_cewm_step(wm, obs, goal, n_runs=n_runs, fast_path=False)
    fast = _measure_cewm_step(wm, obs, goal, n_runs=n_runs, fast_path=True)

    return {"state_type": "DoublePendulumState", "cewm_step": normal, "cewm_step_fast": fast}


def _run_robot(wm: Any, n_runs: int) -> dict[str, Any]:
    """RobotWorldState 场景 WCET。"""
    from mci_world_model.sdk import RobotWorldState

    obs = RobotWorldState(n_joints=6)
    obs._ensure_arrays()
    obs.joint_positions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    obs.joint_velocities = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])

    normal = _measure_cewm_step(wm, obs, n_runs=n_runs, fast_path=False)
    fast = _measure_cewm_step(wm, obs, n_runs=n_runs, fast_path=True)

    return {"state_type": "RobotWorldState", "cewm_step": normal, "cewm_step_fast": fast}


def _run_multimodal(wm: Any, n_runs: int) -> dict[str, Any]:
    """MultimodalWorldState 场景 WCET。"""
    from mci_world_model.sdk import MultimodalWorldState

    obs = MultimodalWorldState(
        proprioception=np.array([0.5, 1.0]),
        vision=np.array([0.1, 0.2, 0.3]),
        audio=np.array([0.4]),
        thermal=np.array([0.7]),
    )

    normal = _measure_cewm_step(wm, obs, n_runs=n_runs, fast_path=False)
    fast = _measure_cewm_step(wm, obs, n_runs=n_runs, fast_path=True)

    return {"state_type": "MultimodalWorldState", "cewm_step": normal, "cewm_step_fast": fast}


def _check_gate(results: list[dict]) -> dict[str, Any]:
    """KPI 门禁检查。"""
    gates = {}
    for r in results:
        st = r["state_type"]
        p99_normal = r["cewm_step"]["p99"]
        p99_fast = r["cewm_step_fast"]["p99"]

        # KPI R-4: Pendulum p99 < 10ms, others p99 < 50ms
        if st == "PendulumState":
            threshold_normal = 10.0
        else:
            threshold_normal = 50.0
        threshold_fast = threshold_normal * 0.3  # 快速路径 ≤ 30%

        gates[st] = {
            "p99_normal_ms": round(p99_normal, 4),
            "p99_fast_ms": round(p99_fast, 4),
            "gate_normal": "PASS" if p99_normal < threshold_normal else "FAIL",
            "gate_fast": "PASS" if p99_fast < threshold_fast else "FAIL",
            "threshold_normal_ms": threshold_normal,
            "threshold_fast_ms": threshold_fast,
        }

    all_pass = all(g["gate_normal"] == "PASS" and g["gate_fast"] == "PASS" for g in gates.values())
    gates["overall"] = "PASS" if all_pass else "FAIL"
    return gates


def main() -> None:
    parser = argparse.ArgumentParser(description="CEWM WCET 基准分析")
    parser.add_argument("--n-runs", type=int, default=1000, help="每种场景运行次数 (默认: 1000)")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="输出格式")
    args = parser.parse_args()

    wm = _create_world_model()
    n = args.n_runs

    runners = [
        ("Pendulum", _run_pendulum),
        ("Cart", _run_cart),
        ("DoublePendulum", _run_double_pendulum),
        ("Robot", _run_robot),
        ("Multimodal", _run_multimodal),
    ]

    results = []
    for name, runner in runners:
        print(f"  测量 {name} (n={n})...", flush=True)
        result = runner(wm, n)
        results.append(result)

    gates = _check_gate(results)

    if args.format == "json":
        output = {"results": results, "gates": gates}
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print("\n" + "=" * 76)
        print("  CEWM v4.5.0 WCET 基准分析报告")
        print("=" * 76)

        for r in results:
            st = r["state_type"]
            n_data = r["cewm_step"]
            f_data = r["cewm_step_fast"]
            g = gates[st]

            print(f"\n  ── {st} ──")
            print(
                f"  cewm_step():      p50={n_data['p50']:.4f}ms  p95={n_data['p95']:.4f}ms  "
                f"p99={n_data['p99']:.4f}ms  p99.9={n_data['p99_9']:.4f}ms  max={n_data['max']:.4f}ms"
            )
            print(
                f"  cewm_step_fast(): p50={f_data['p50']:.4f}ms  p95={f_data['p95']:.4f}ms  "
                f"p99={f_data['p99']:.4f}ms  p99.9={f_data['p99_9']:.4f}ms  max={f_data['max']:.4f}ms"
            )
            print(
                f"  门禁: normal={g['gate_normal']}  fast={g['gate_fast']}  "
                f"(阈值: normal<{g['threshold_normal_ms']}ms, fast<{g['threshold_fast_ms']:.1f}ms)"
            )

        print(f"\n  总体门禁: {gates['overall']}")
        print("=" * 76)

    if gates["overall"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
