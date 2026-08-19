#!/usr/bin/env python3
"""
QUAL-03 (S-5): Pendulum 硬编码引用 CI 追踪脚本
================================================

自动化扫描 mci_world_model 源码中的 Pendulum 特定硬编码引用，
统计数量并在 CI 中生成趋势报告。

KPI 对标: E-5 Pendulum 硬编码引用数 ≤ 2（仅保留物理引擎和测试）

用法:
    python scripts/track_pendulum_hardcode.py            # 打印报告
    python scripts/track_pendulum_hardcode.py --ci        # CI 模式（超阈值返回 1）
    python scripts/track_pendulum_hardcode.py --json      # JSON 输出

Exit codes:
    0 — 所有文件在阈值以内
    1 — 至少一个文件超过阈值（CI 应阻断）
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src" / "mci_world_model"

# Pendulum 特定硬编码模式
# 每个模式对应一种类型的硬编码引用
PATTERNS: dict[str, str] = {
    "hasattr_theta_omega": r'hasattr\([^)]*["\']theta["\']\).*hasattr\([^)]*["\']omega["\']\)',
    "direct_theta_attr": r"\.theta\b",
    "direct_omega_attr": r"\.omega\b",
    "pendulumstate_import": r"import\s+PendulumState|from.*import.*PendulumState",
    "pendulumstate_instantiation": r"PendulumState\s*\(",
    "pendulum_literal_string": r'["\']pendulum["\']',
}

# 允许出现 Pendulum 引用的白名单文件（物理引擎、状态定义、解析器）
ALLOWLIST: set[str] = {
    "_generalized_physics.py",  # 物理动力学引擎（合法）
    "_world_state.py",  # PendulumState 定义（合法）
    "_protocols.py",  # PendulumStateParser（合法）
}

# 每个文件的硬编码引用阈值（超出则 CI 失败）
# 白名单文件不限，非白名单文件阈值 ≤ 2
HARDCODE_THRESHOLD = 2


# ── 数据结构 ──────────────────────────────────────────


@dataclass
class FileReport:
    """单个文件的扫描报告。"""

    file: str
    total_refs: int = 0
    by_pattern: dict[str, int] = field(default_factory=dict)
    is_allowlisted: bool = False

    @property
    def exceeds_threshold(self) -> bool:
        """是否超过阈值（白名单文件豁免）。"""
        if self.is_allowlisted:
            return False
        return self.total_refs > HARDCODE_THRESHOLD


# ── 核心逻辑 ──────────────────────────────────────────


def scan_file(filepath: Path) -> FileReport:
    """扫描单个 Python 文件中的 Pendulum 硬编码引用。"""
    report = FileReport(
        file=str(filepath.relative_to(PROJECT_ROOT)),
        is_allowlisted=filepath.name in ALLOWLIST,
    )

    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception:
        return report

    for pattern_name, pattern in PATTERNS.items():
        matches = re.findall(pattern, content)
        count = len(matches)
        if count > 0:
            report.by_pattern[pattern_name] = count
            report.total_refs += count

    return report


def scan_directory(directory: Path) -> list[FileReport]:
    """递归扫描目录下所有 Python 文件。"""
    reports: list[FileReport] = []
    for py_file in sorted(directory.rglob("*.py")):
        # 跳过 __pycache__ 和测试文件
        if "__pycache__" in str(py_file) or "/tests/" in str(py_file):
            continue
        report = scan_file(py_file)
        if report.total_refs > 0:
            reports.append(report)
    return reports


def format_report(reports: list[FileReport]) -> str:
    """格式化为人类可读的报告。"""
    lines = [
        "=" * 72,
        "QUAL-03 (S-5): Pendulum 硬编码引用追踪报告",
        "=" * 72,
        f"扫描目录: {SRC_DIR.relative_to(PROJECT_ROOT)}",
        f"阈值: 非白名单文件 ≤ {HARDCODE_THRESHOLD} 处引用",
        "",
    ]

    total_all = sum(r.total_refs for r in reports)
    total_non_allowlisted = sum(r.total_refs for r in reports if not r.is_allowlisted)
    violations = [r for r in reports if r.exceeds_threshold]

    for report in reports:
        status = "✓" if not report.exceeds_threshold else "✗"
        allow = " (白名单)" if report.is_allowlisted else ""
        lines.append(f"  {status} {report.file}: {report.total_refs} 处引用{allow}")
        for pattern, count in sorted(report.by_pattern.items(), key=lambda x: -x[1]):
            lines.append(f"      - {pattern}: {count}")

    lines.extend(
        [
            "",
            "-" * 72,
            f"总计: {total_all} 处引用 ({total_non_allowlisted} 处非白名单)",
            f"违规文件: {len(violations)}",
        ]
    )

    if violations:
        lines.append("\n⚠️  超过阈值的文件:")
        for v in violations:
            lines.append(f"   ✗ {v.file}: {v.total_refs} > {HARDCODE_THRESHOLD}")
    else:
        lines.append("\n✅ 所有非白名单文件在阈值以内")

    lines.append("=" * 72)
    return "\n".join(lines)


def format_json(reports: list[FileReport]) -> str:
    """格式化为 JSON 输出。"""
    return json.dumps(
        {
            "total_refs": sum(r.total_refs for r in reports),
            "total_non_allowlisted": sum(r.total_refs for r in reports if not r.is_allowlisted),
            "violations": [r.file for r in reports if r.exceeds_threshold],
            "files": [
                {
                    "file": r.file,
                    "total_refs": r.total_refs,
                    "by_pattern": r.by_pattern,
                    "is_allowlisted": r.is_allowlisted,
                    "exceeds_threshold": r.exceeds_threshold,
                }
                for r in reports
            ],
        },
        indent=2,
        ensure_ascii=False,
    )


# ── 入口 ─────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Pendulum 硬编码引用追踪")
    parser.add_argument("--ci", action="store_true", help="CI 模式（超阈值返回 exit 1）")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    args = parser.parse_args()

    reports = scan_directory(SRC_DIR)

    if args.json:
        print(format_json(reports))
    else:
        print(format_report(reports))

    if args.ci:
        violations = [r for r in reports if r.exceeds_threshold]
        if violations:
            print(f"\n❌ CI 失败: {len(violations)} 个文件超过阈值 {HARDCODE_THRESHOLD}", file=sys.stderr)
            return 1
        print("\n✅ CI 通过: 所有文件在阈值以内", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
