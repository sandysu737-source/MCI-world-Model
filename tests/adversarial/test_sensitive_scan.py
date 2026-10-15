from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ai-verify/sensitive_scan.py"
WORDLIST = ROOT / "scripts/ai-verify/sensitive-terms.toml"


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_sensitive_scan_fails_and_does_not_leak_terms(tmp_path: Path) -> None:
    target = tmp_path / "release.md"
    target.write_text("safe\n", encoding="utf-8")
    wordlist = tmp_path / "terms.toml"
    wordlist.write_text(
        "\n".join(
            [
                "schema_version = 1",
                'wordlist_version = "test"',
                'single_cjk_match = "concept"',
                'source = "test"',
                'source_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"',
                "[[categories]]",
                'name = "category"',
                'terms = ["secret-term"]',
                'safe_aliases = ["safe"]',
            ]
        ),
        encoding="utf-8",
    )
    target.write_text("prefix secret-term suffix\n", encoding="utf-8")
    report = tmp_path / "report.json"

    result = _run(tmp_path, "release.md", "--wordlist", str(wordlist), "--report", str(report))

    assert result.returncode == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert payload["files_scanned"] == 1
    assert payload["findings_count"] == 1
    assert "secret-term" not in report.read_text(encoding="utf-8")


def test_sensitive_scan_passes_when_clean(tmp_path: Path) -> None:
    target = tmp_path / "release.md"
    target.write_text("safe\n", encoding="utf-8")
    wordlist = tmp_path / "terms.toml"
    wordlist.write_text(
        "\n".join(
            [
                "schema_version = 1",
                'wordlist_version = "test"',
                'single_cjk_match = "concept"',
                'source = "test"',
                'source_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"',
                "[[categories]]",
                'name = "category"',
                'terms = ["secret-term"]',
                'safe_aliases = ["safe"]',
            ]
        ),
        encoding="utf-8",
    )

    result = _run(tmp_path, "release.md", "--wordlist", str(wordlist))

    assert result.returncode == 0
    assert result.stdout.startswith("status=pass files=1 findings=0")


def test_sensitive_scan_fails_closed_on_missing_wordlist(tmp_path: Path) -> None:
    target = tmp_path / "release.md"
    target.write_text("safe\n", encoding="utf-8")

    result = _run(tmp_path, "release.md", "--wordlist", str(tmp_path / "missing.toml"))

    assert result.returncode == 2
    assert "fail-closed" in result.stderr


def test_single_cjk_concept_match_avoids_substrings(tmp_path: Path) -> None:
    target = tmp_path / "release.md"
    target.write_text("水分\n水\n", encoding="utf-8")
    wordlist = tmp_path / "terms.toml"
    wordlist.write_text(
        "\n".join(
            [
                "schema_version = 1",
                'wordlist_version = "test"',
                'single_cjk_match = "concept"',
                'source = "test"',
                'source_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"',
                "[[categories]]",
                'name = "category"',
                'terms = ["水"]',
                'safe_aliases = ["safe"]',
            ]
        ),
        encoding="utf-8",
    )

    result = _run(tmp_path, "release.md", "--wordlist", str(wordlist))

    assert result.returncode == 1
    assert result.stdout.startswith("status=fail files=1 findings=1")


def _wordlist(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                'wordlist_version = "test"',
                'single_cjk_match = "concept"',
                'source = "test"',
                'source_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"',
                "[[categories]]",
                'name = "category"',
                'terms = ["secret-term"]',
                'safe_aliases = ["safe"]',
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_wordlist_self_reference_is_excluded_and_recorded(tmp_path: Path) -> None:
    """词表自身必然命中自己：不得计入 findings，且豁免必须在报告里可见（不静默）。

    实证（OODA-20261015-PUBSURFACE-016）：全仓口径下 `sensitive-terms.toml` 单文件
    贡献 22/74 条命中——若把它算作泄漏，扫描器永远无法对自己所在目录给出结论，
    扩口径这件事从第一步就不可行。
    """
    _wordlist(tmp_path / "terms.toml")
    (tmp_path / "release.md").write_text("secret-term\n", encoding="utf-8")
    report = tmp_path / "report.json"

    result = _run(tmp_path, ".", "--wordlist", str(tmp_path / "terms.toml"), "--report", str(report))

    assert result.returncode == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["self_excluded"] == ["terms.toml"]
    assert payload["files_scanned"] == 1
    assert payload["findings_count"] == 1
    assert payload["findings"][0]["path"] == "release.md"


def test_tracked_scope_scans_versioned_files_only(tmp_path: Path) -> None:
    """公开仓口径：发布面 = 受版本控制文件；未入库文件不计。

    口径来源：公开仓里「入库即发布」——只扫交付路径会给出假绿。
    """
    if shutil.which("git") is None:  # pragma: no cover - 环境兜底
        pytest.skip("git 不可用")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.md").write_text("secret-term\n", encoding="utf-8")
    (tmp_path / "terms.toml").write_bytes(
        (tmp_path / "terms.toml").read_bytes() if (tmp_path / "terms.toml").exists() else b""
    )
    # 词表放仓外，避免影响 tracked 集合
    wordlist = _wordlist(tmp_path.parent / f"wl-{tmp_path.name}.toml")
    subprocess.run(["git", "add", "tracked.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    (tmp_path / "untracked.md").write_text("secret-term\n", encoding="utf-8")
    report = tmp_path / "report.json"

    result = _run(tmp_path, "--tracked", "--wordlist", str(wordlist), "--report", str(report))

    assert result.returncode == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["scope"] == "tracked"
    assert payload["files_scanned"] == 1
    assert [f["path"] for f in payload["findings"]] == ["tracked.md"]


def test_requires_roots_or_tracked(tmp_path: Path) -> None:
    result = _run(tmp_path, "--wordlist", str(_wordlist(tmp_path / "terms.toml")))

    assert result.returncode == 2
    assert "需要 roots" in result.stderr


def test_wordlist_outside_scanned_roots_is_not_recorded_as_self_excluded(tmp_path: Path) -> None:
    """词表不在扫描面内时，`self_excluded` 必须为空——「没排除过」不得留痕。

    判据动机：自排除逻辑的两种写法（`any(==)` 与 `any(!=)`）在「词表恰在扫描面内」时
    结果相同，只有在「词表不在扫描面内」时才分叉——后者会把一个**从未发生的排除**
    写进报告，等于把「没扫到词表」伪造成「主动豁免了词表」，属静默失真。
    """
    wordlist = _wordlist(tmp_path.parent / f"outside-{tmp_path.name}.toml")
    (tmp_path / "release.md").write_text("safe\n", encoding="utf-8")
    report = tmp_path / "report.json"

    result = _run(tmp_path, ".", "--wordlist", str(wordlist), "--report", str(report))

    assert result.returncode == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["self_excluded"] == []
    assert payload["files_scanned"] == 1


def _load_module():
    import importlib.util

    # dataclasses 依赖 `sys.modules[cls.__module__]`，故须先注册再 exec_module
    name = "sensitive_scan_under_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def test_scan_defaults_to_roots_scope_not_tracked(tmp_path: Path) -> None:
    """`scan()` 省略 `use_tracked` 时必须是 roots 口径——默认值即对外契约。

    判据动机：CLI 恒显式传 `use_tracked=args.tracked`，故默认值只对**直接调用方**
    可见；一旦默认翻转为 True，任何省略该参数的调用会静默从「按 roots 扫」变成
    「只扫入库文件」，未入库内容被无声跳过（公开仓口径下正好是相反的漏检方向）。
    """
    if shutil.which("git") is None:  # pragma: no cover - 环境兜底
        pytest.skip("git 不可用")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    wordlist = _wordlist(tmp_path.parent / f"default-{tmp_path.name}.toml")
    (tmp_path / "untracked.md").write_text("secret-term\n", encoding="utf-8")

    module = _load_module()
    payload = module.scan(root=tmp_path, roots=[tmp_path], wordlist_path=wordlist, excludes=[])

    assert payload["scope"] == "roots"
    assert [f["path"] for f in payload["findings"]] == ["untracked.md"]
