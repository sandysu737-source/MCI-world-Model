#!/usr/bin/env python3
"""按受控词表扫描发布路径，输出可复现 JSON 报告。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

DEFAULT_EXCLUDES = (
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
)


@dataclass(frozen=True)
class Wordlist:
    version: str
    source: str
    source_sha256: str
    terms: tuple[str, ...]


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    column: int
    category: str
    term_digest: str


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_wordlist(path: Path) -> tuple[Wordlist, str]:
    raw = path.read_bytes()
    data = tomllib.loads(raw.decode("utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("schema_version 必须为 1")
    if data.get("single_cjk_match") != "concept":
        raise ValueError("single_cjk_match 必须为 concept")
    version = data.get("wordlist_version")
    source = data.get("source")
    source_sha256 = data.get("source_sha256")
    if not isinstance(version, str) or not version:
        raise ValueError("wordlist_version 缺失或为空")
    if not isinstance(source, str) or not source:
        raise ValueError("source 缺失或为空")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise ValueError("source_sha256 缺失或无效")
    categories = data.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError("categories 缺失或为空")

    terms: list[str] = []
    for category in categories:
        if not isinstance(category, dict):
            raise ValueError("category 必须是表")
        category_name = category.get("name")
        category_terms = category.get("terms")
        category_aliases = category.get("safe_aliases")
        if not isinstance(category_name, str) or not category_name:
            raise ValueError("category name 缺失或为空")
        if not isinstance(category_terms, list) or not category_terms:
            raise ValueError(f"category {category_name} terms 缺失或为空")
        if not isinstance(category_aliases, list) or not category_aliases:
            raise ValueError(f"category {category_name} safe_aliases 缺失或为空")
        if not all(isinstance(term, str) and term for term in category_terms):
            raise ValueError(f"category {category_name} 存在空 term")
        if not all(isinstance(alias, str) and alias for alias in category_aliases):
            raise ValueError(f"category {category_name} 存在空 safe_alias")
        terms.extend(category_terms)
    if len(terms) != len(set(terms)):
        raise ValueError("词表存在重复 term")
    return Wordlist(version, source, source_sha256, tuple(terms)), _sha256_bytes(raw)


def _is_excluded(path: Path, root: Path, excludes: Iterable[str]) -> bool:
    relative_parts = path.relative_to(root).parts
    return any(part in set(excludes) for part in relative_parts)


def iter_files(root: Path, roots: Iterable[Path], excludes: Iterable[str]) -> list[Path]:
    files: set[Path] = set()
    for scan_root in roots:
        if scan_root.is_file():
            files.add(scan_root)
        elif scan_root.is_dir():
            files.update(path for path in scan_root.rglob("*") if path.is_file())
    return sorted(path for path in files if not _is_excluded(path, root, excludes))


def scan_file(
    path: Path,
    root: Path,
    wordlist: Wordlist,
    term_categories: dict[str, str],
) -> list[Finding]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return []
    findings: list[Finding] = []
    patterns = _compiled_patterns(wordlist.terms)
    for line_number, line in enumerate(lines, start=1):
        for term, pattern in patterns.items():
            for match in pattern.finditer(line):
                digest = hashlib.sha256(term.encode("utf-8")).hexdigest()[:12]
                findings.append(
                    Finding(
                        path=path.relative_to(root).as_posix(),
                        line=line_number,
                        column=match.start() + 1,
                        category=term_categories[term],
                        term_digest=digest,
                    )
                )
    return findings


def _compiled_patterns(terms: tuple[str, ...]) -> dict[str, re.Pattern[str]]:
    patterns: dict[str, re.Pattern[str]] = {}
    for term in terms:
        if len(term) == 1 and "\u4e00" <= term <= "\u9fff":
            pattern = re.compile(rf"(?<![一-鿿]){re.escape(term)}(?:属性|行|生|克)?(?![一-鿿])")
        else:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
        patterns[term] = pattern
    return patterns


def scan(
    root: Path,
    roots: Iterable[Path],
    wordlist_path: Path,
    excludes: Iterable[str],
) -> dict[str, object]:
    wordlist, wordlist_sha256 = load_wordlist(wordlist_path)
    term_categories = {term: category["name"] for category in _categories(wordlist_path) for term in category["terms"]}
    findings: list[Finding] = []
    files = iter_files(root, roots, excludes)
    for path in files:
        findings.extend(scan_file(path, root, wordlist, term_categories))
    return {
        "wordlist_version": wordlist.version,
        "source": wordlist.source,
        "source_sha256": wordlist.source_sha256,
        "wordlist_sha256": wordlist_sha256,
        "roots": [
            path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix() for path in roots
        ],
        "files_scanned": len(files),
        "findings_count": len(findings),
        "status": "pass" if not findings else "fail",
        "findings": [finding.__dict__ for finding in findings],
    }


def _categories(wordlist_path: Path):
    data = tomllib.loads(wordlist_path.read_text(encoding="utf-8"))
    return data["categories"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="要扫描的文件或目录")
    parser.add_argument(
        "--wordlist",
        type=Path,
        default=Path(__file__).with_name("sensitive-terms.toml"),
    )
    parser.add_argument("--report", type=Path, help="JSON 报告输出路径")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="按路径段排除，可重复",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    resolved_roots = [path.resolve() for path in args.roots]
    root = Path(os.path.commonpath([str(Path.cwd().resolve()), *(str(path) for path in resolved_roots)]))
    excludes = (*DEFAULT_EXCLUDES, *args.exclude)
    try:
        result = scan(root, resolved_roots, args.wordlist, excludes)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"fail-closed: {error}", file=sys.stderr)
        return 2
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"status={result['status']} files={result['files_scanned']} findings={result['findings_count']}")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
