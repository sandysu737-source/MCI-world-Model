"""Cladder 基准测试入口。

用法:
    python -m benchmarks.cladder              # 全量评测
    python -m benchmarks.cladder --verbose     # 详细输出
    python -m benchmarks.cladder --format json # JSON 输出
    python -m benchmarks.cladder --predictor-backend generalized  # 消融: 切换预测器后端
    python -m benchmarks.cladder --safety off                # 消融: 关闭安全约束
    python -m benchmarks.cladder --fusion-strategy attention  # 消融: 切换融合策略
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from benchmarks.cladder._dataset import dataset_stats, load_cladder
from benchmarks.cladder._solvers import solve_all


def _print_report(report: dict) -> None:
    """打印格式化的评测报告。"""
    print("=" * 68)
    print("  Cladder 因果推理基准测试 — CEWM 结构化求解器")
    print("=" * 68)
    print(f"  总题数:    {report['total']:>6d}")
    print(f"  正确数:    {report['correct']:>6d}")
    print(f"  准确率:    {report['accuracy']:>6.2f}%")
    print()

    # Rung 维度
    print("  ── 按 Pearl 因果阶梯 ──")
    for rung in [1, 2, 3]:
        r = report["by_rung"].get(rung, {})
        name = {1: "关联层 (Association)", 2: "干预层 (Intervention)", 3: "反事实层 (Counterfactual)"}.get(
            rung, f"Rung {rung}"
        )
        print(f"    {name:30s}: {r.get('accuracy', 0):6.2f}% ({r.get('correct', 0)}/{r.get('total', 0)})")

    print()
    print("  ── 按问题类型 ──")
    for qt in sorted(report["by_query_type"]):
        r = report["by_query_type"][qt]
        print(f"    {qt:28s}: {r['accuracy']:6.2f}% ({r['correct']:4d}/{r['total']:4d})")

    print()
    print("  ── 按图结构类型 ──")
    for gid in sorted(report["by_graph_type"]):
        r = report["by_graph_type"][gid]
        print(f"    {gid:15s}: {r['accuracy']:6.2f}% ({r['correct']:4d}/{r['total']:4d})")
    print("=" * 68)


def _print_verbose(results: list[dict], questions: list) -> None:
    """打印详细错误信息。"""
    errors = [(r, q) for r, q in zip(results, questions) if not r["correct"]]
    if not errors:
        print("\n  🎉 全部正确！没有错误。")
        return

    print(f"\n  ❌ 错误详情 ({len(errors)} 题):")
    print(f"  {'ID':<8} {'类型':<22} {'图':<15} {'标签':<6} {'预测':<6}")
    print(f"  {'─' * 8} {'─' * 22} {'─' * 15} {'─' * 6} {'─' * 6}")
    for r, q in errors[:30]:
        print(f"  {q.question_id:<8} {q.query_type:<22} {q.graph_id:<15} {r['label']!s:<6} {r['predicted']!s:<6}")
    if len(errors) > 30:
        print(f"  ... 还有 {len(errors) - 30} 个错误")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cladder 因果推理基准测试 — CEWM 结构化求解器")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细错误信息")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="输出格式 (默认: text)")
    parser.add_argument("--stats", action="store_true", help="显示数据集统计信息")

    # v4.5.0: 消融追踪 flag (KPI I-4)
    parser.add_argument(
        "--predictor-backend",
        choices=["pendulum", "cart", "generalized", "double_pendulum"],
        default="pendulum",
        help="P0-1/P1-1: 预测器后端 (默认: pendulum)",
    )
    parser.add_argument("--safety", choices=["on", "off"], default="off", help="P2-2: 安全约束层开关 (默认: off)")
    parser.add_argument(
        "--fusion-strategy",
        choices=["attention", "weighted", "concat", "off"],
        default="off",
        help="P1-3: 多模态融合策略 (默认: off)",
    )
    parser.add_argument("--enable-cf-oracle", action="store_true", help="P2-1: 启用 CounterfactualOracle")
    parser.add_argument("--deadline-ms", type=int, default=0, help="P3-3: 设置 WCET deadline (0=不限制)")
    args = parser.parse_args()

    # 加载数据
    t0 = time.perf_counter()
    questions = load_cladder()
    load_time = time.perf_counter() - t0

    if args.stats:
        stats = dataset_stats(questions)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return

    # 运行评测
    t0 = time.perf_counter()
    results, report = solve_all(questions)
    solve_time = time.perf_counter() - t0

    report["ablation"] = {
        "predictor_backend": args.predictor_backend,
        "safety": args.safety,
        "fusion_strategy": args.fusion_strategy,
        "cf_oracle": args.enable_cf_oracle,
        "deadline_ms": args.deadline_ms,
    }
    report["timing"] = {
        "load_sec": round(load_time, 3),
        "solve_sec": round(solve_time, 3),
        "total_sec": round(load_time + solve_time, 3),
    }

    if args.format == "json":
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_report(report)
        print(f"  加载耗时: {load_time:.2f}s  求解耗时: {solve_time:.2f}s")
        if args.verbose:
            _print_verbose(results, questions)

    # 门槛检查: ≥95% 准确率
    if report["accuracy"] < 95.0:
        print(f"\n⚠️  未达到 95% 准确率门槛！当前: {report['accuracy']:.2f}%")
        sys.exit(1)


if __name__ == "__main__":
    main()
