"""治理工具单元测试：``scripts/ai-verify/test-map-gen.py``（F-25 治理 L）。

被测文件不在包内（文件名含连字符，无法按模块名 import），故用 importlib 按路径加载。
测试覆盖：context 归一化、映射筛选口径（top-1 / 前缀 / __init__ / 非测试 context）、
CLI 错误路径（缺文件 / 坏 JSON / 缺 files 字段）、手工条目区的保留行为。
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
from typing import Any

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GEN_PATH = ROOT / "scripts" / "ai-verify" / "test-map-gen.py"


def _load_generator() -> Any:
    spec = importlib.util.spec_from_file_location("ai_verify_test_map_gen", GEN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gen = _load_generator()


def _write_coverage(tmp_path: pathlib.Path, files: dict[str, Any]) -> pathlib.Path:
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps({"files": files}), encoding="utf-8")
    return path


def _single_file(lines: list[int], contexts: dict[str, list[str]]) -> dict[str, Any]:
    return {"executed_lines": lines, "contexts": contexts}


def test_normalize_context_strips_node_id() -> None:
    assert gen.normalize_context("tests/test_x.py::test_y[a]") == "tests/test_x.py"
    assert gen.normalize_context("tests/test_x.py") == "tests/test_x.py"


def test_build_map_picks_top_test_and_filters() -> None:
    files = {
        "src/pkg/mod.py": _single_file(
            [1, 2, 3],
            {
                "1": ["tests/test_a.py::t1"],
                "2": ["tests/test_a.py::t2"],
                "3": ["tests/test_b.py::t3"],
            },
        ),
        "src/pkg/__init__.py": _single_file([1], {"1": ["tests/test_a.py::t1"]}),
        "other/mod.py": _single_file([1], {"1": ["tests/test_a.py::t1"]}),
        "src/pkg/no_ctx.py": _single_file([1], {"1": ["pytest"]}),
        "src/pkg/not_py.txt": _single_file([1], {"1": ["tests/test_a.py::t1"]}),
    }
    mapping = gen.build_map(files)
    assert mapping == {"src/pkg/mod.py": ("tests/test_a.py", 2)}


def test_build_map_ignores_non_test_contexts() -> None:
    files = {
        "benchmarks/m.py": _single_file(
            [1, 2, 3],
            {"1": [""], "2": ["pytest"], "3": ["scripts/tool.sh"]},
        )
    }
    assert gen.build_map(files) == {}


def test_main_rejects_missing_or_invalid_input(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert gen.main([str(tmp_path / "absent.json")]) == 2
    assert "找不到覆盖率 JSON" in capsys.readouterr().err

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert gen.main([str(bad)]) == 2
    assert "JSON 解析失败" in capsys.readouterr().err

    nofiles = tmp_path / "nofiles.json"
    nofiles.write_text(json.dumps({"meta": {}}), encoding="utf-8")
    assert gen.main([str(nofiles)]) == 2
    assert "缺 files 字段" in capsys.readouterr().err


def test_main_writes_sorted_rows_with_header(tmp_path: pathlib.Path) -> None:
    cov = _write_coverage(
        tmp_path,
        {
            "src/b.py": _single_file([1], {"1": ["tests/test_b.py::t"]}),
            "src/a.py": _single_file([1], {"1": ["tests/test_a.py::t"]}),
        },
    )
    out = gen.main([str(cov), "map.tsv", "--root", str(tmp_path), "--sha", "abc1234", "--now", "固定时间"])
    assert out == 0
    lines = (tmp_path / "map.tsv").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# 由 scripts/ai-verify/test-map-gen.py 生成")
    assert any("abc1234" in line and "固定时间" in line for line in lines)
    assert lines[-1] == gen.MANUAL_MARKER
    rows = [line for line in lines if line and not line.startswith("#")]
    assert rows == ["src/a.py\ttests/test_a.py\t1", "src/b.py\ttests/test_b.py\t1"]


def test_main_preserves_manual_section(tmp_path: pathlib.Path) -> None:
    cov = _write_coverage(tmp_path, {"src/a.py": _single_file([1], {"1": ["tests/test_a.py::t"]})})
    target = tmp_path / "map.tsv"
    target.write_text(
        "# 旧内容\n" + gen.MANUAL_MARKER + "\nscripts/tool.py\ttests/test_tool.py\t0\n",
        encoding="utf-8",
    )
    assert gen.main([str(cov), "map.tsv", "--root", str(tmp_path)]) == 0
    text = target.read_text(encoding="utf-8")
    assert text.count(gen.MANUAL_MARKER) == 1
    assert text.rstrip().endswith("scripts/tool.py\ttests/test_tool.py\t0")
    assert "# 旧内容" not in text


def test_main_creates_missing_output_directories(tmp_path: pathlib.Path) -> None:
    cov = _write_coverage(tmp_path, {"src/a.py": _single_file([1], {"1": ["tests/test_a.py::t"]})})
    assert gen.main([str(cov), "nested/dir/map.tsv", "--root", str(tmp_path)]) == 0
    assert (tmp_path / "nested" / "dir" / "map.tsv").is_file()


def test_cli_end_to_end_as_script(tmp_path: pathlib.Path) -> None:
    """按脚本方式执行（``__main__`` 分支），覆盖 CLI 入口与打印。"""
    cov = _write_coverage(tmp_path, {"src/a.py": _single_file([1], {"1": ["tests/test_a.py::t"]})})
    proc = subprocess.run(
        [sys.executable, str(GEN_PATH), str(cov), "cli.tsv", "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "cli.tsv").is_file()
    assert "已写入" in proc.stdout


def test_parse_args_default_output() -> None:
    args = gen.parse_args(["cov.json"])
    assert args.output == ".ai-governance/test-map.tsv"
    assert args.root == "."
