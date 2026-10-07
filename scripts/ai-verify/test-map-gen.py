#!/usr/bin/env python3
"""从覆盖率上下文生成变异门禁测试映射（F-25 治理 L）。

输入: ``coverage json --show-contexts`` 的 JSON，由
      ``pytest -m "not slow" --cov=mci_world_model --cov=benchmarks --cov=adapters \
      --cov-context=test`` 产生（context 为测试节点 id）。
      注: 多写 ``--cov=benchmarks --cov=adapters`` 只为让研究/适配层文件进入映射，
      与 CI 的覆盖率阈值口径（source=mci_world_model）无关，不得据此判断覆盖率。
输出: ``.ai-governance/test-map.tsv``，每行 ``<源文件>\t<测试目标>\t<覆盖行数>``。

口径:
    只取"覆盖该文件执行行数最多"的 top-1 测试文件（控制变异门禁耗时）；
    context 形如 ``tests/test_x.py::test_y``，归一到测试文件路径；
    session 级 context（空串、``pytest``）不计入；只收录 src/adapters/benchmarks 下的源文件。
手工条目: 输出文件中 ``MANUAL_MARKER`` 之后的行由人工维护（生成器原样保留），
    用于覆盖率不测量的路径（如 ``scripts/ai-verify/*.py``），其测试目标必须真实存在。

用法: test-map-gen.py <coverage.json> [输出路径] [--root <仓库根>] [--sha <提交>]
退出码: 0=成功 2=输入非法
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import pathlib
import sys

SOURCE_PREFIXES = ("src/", "adapters/", "benchmarks/")
# 手工条目区起始标记：生成器不覆盖其后的行（用于 scripts/** 等覆盖率不测量的治理工具）
MANUAL_MARKER = "# --- 以下为手工条目（生成器保留，勿删）---"
IGNORED_CONTEXTS = {"", "pytest"}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成变异门禁测试映射")
    parser.add_argument("coverage_json", help="coverage json --show-contexts 输出文件")
    parser.add_argument("output", nargs="?", default=".ai-governance/test-map.tsv")
    parser.add_argument("--root", default=".")
    parser.add_argument("--sha", default="")
    parser.add_argument("--now", default="")
    return parser.parse_args(argv)


def normalize_context(context: str) -> str:
    """``tests/test_x.py::test_y[param]`` → ``tests/test_x.py``"""
    return context.partition("::")[0].strip()


def build_map(files: dict) -> dict[str, tuple[str, int]]:
    result: dict[str, tuple[str, int]] = {}
    for path, data in files.items():
        if not path.endswith(".py"):
            continue
        if not path.startswith(SOURCE_PREFIXES):
            continue
        if pathlib.PurePosixPath(path).name == "__init__.py":
            continue
        contexts = data.get("contexts") or {}
        counted: collections.Counter[str] = collections.Counter()
        for line_no in data.get("executed_lines", []):
            for raw in contexts.get(str(line_no), []) or []:
                ctx = normalize_context(str(raw))
                if ctx in IGNORED_CONTEXTS or not ctx.endswith(".py"):
                    continue
                if "tests/" not in ctx and not ctx.startswith("tests"):
                    continue
                counted[ctx] += 1
        if not counted:
            continue
        test_path, hits = counted.most_common(1)[0]
        result[path] = (test_path, hits)
    return result


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    src = pathlib.Path(args.coverage_json)
    if not src.is_file():
        print(f"ERROR: 找不到覆盖率 JSON: {src}", file=sys.stderr)
        return 2
    try:
        payload = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: JSON 解析失败: {exc}", file=sys.stderr)
        return 2
    files = payload.get("files")
    if not isinstance(files, dict):
        print("ERROR: 缺 files 字段；请用 coverage json --show-contexts 生成", file=sys.stderr)
        return 2

    mapping = build_map(files)
    root = pathlib.Path(args.root).resolve()
    now = args.now or dt.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "# 由 scripts/ai-verify/test-map-gen.py 生成，请勿手工编辑",
        '# 源: pytest -m "not slow" --cov=mci_world_model --cov=benchmarks --cov=adapters --cov-context=test',
        "#     + coverage json --show-contexts",
        f"# 生成时间: {now}" + (f"  提交: {args.sha}" if args.sha else ""),
        "# 格式: <repo 相对源文件>\\t<测试目标>\\t<覆盖该文件的行数>",
    ]
    for rel in sorted(mapping):
        test_path, hits = mapping[rel]
        lines.append(f"{rel}\t{test_path}\t{hits}")
    out = pathlib.Path(args.output)
    if not out.is_absolute():
        out = root / out
    manual: list[str] = []
    if out.is_file():
        prev = out.read_text(encoding="utf-8").splitlines()
        if MANUAL_MARKER in prev:
            manual = prev[prev.index(MANUAL_MARKER) + 1 :]
    lines.append(MANUAL_MARKER)
    lines.extend(manual)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入 {out}（{len(mapping)} 条映射 / {len(files)} 个覆盖文件 / 手工 {len(manual)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
