from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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
