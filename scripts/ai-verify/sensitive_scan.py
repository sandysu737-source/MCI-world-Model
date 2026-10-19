#!/usr/bin/env python3
"""按受控词表扫描发布路径，输出可复现 JSON 报告。

两种口径（2026-10-15，台账 OODA-20261015-PUBSURFACE-016）：
- 显式 roots（默认）："发布面 = 交付路径"，适用于**私有仓**（只有被交付的路径会离开本机）。
- `--tracked`："发布面 = 全部受版本控制文件"，适用于**公开仓**——公开仓里入库即发布，
  只扫交付路径会给出**假绿**（实测 MCI-world-Model 公开仓：4 路径口径 pass，全仓口径 fail）。

自排除：词表文件自身必然命中自己的词条，一律不计入命中并在报告 `self_excluded` 留痕
（豁免必须可见，不得静默）。
"""

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


def iter_files(
    root: Path,
    roots: Iterable[Path],
    excludes: Iterable[str],
    skip: Iterable[Path] = (),
) -> list[Path]:
    files: set[Path] = set()
    for scan_root in roots:
        if scan_root.is_file():
            files.add(scan_root)
        elif scan_root.is_dir():
            files.update(path for path in scan_root.rglob("*") if path.is_file())
    skip_set = {path.resolve() for path in skip}
    return sorted(path for path in files if path.resolve() not in skip_set and not _is_excluded(path, root, excludes))


def tracked_files(root: Path) -> list[Path]:
    """公开仓口径：受版本控制文件即发布面（git ls-files）。"""
    import subprocess

    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout
    names = [name for name in out.decode("utf-8").split("\0") if name]
    return [root / name for name in names]


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
    scope: str = "roots",
    use_tracked: bool = False,
) -> dict[str, object]:
    wordlist, wordlist_sha256 = load_wordlist(wordlist_path)
    term_categories = {term: category["name"] for category in _categories(wordlist_path) for term in category["terms"]}
    findings: list[Finding] = []
    files = tracked_files(root) if use_tracked else iter_files(root, roots, excludes)
    self_excluded = []
    wordlist_resolved = wordlist_path.resolve()
    if any(path.resolve() == wordlist_resolved for path in files):
        files = [path for path in files if path.resolve() != wordlist_resolved]
        self_excluded.append(
            str(Path(wordlist_resolved).relative_to(root.resolve()).as_posix())
            if wordlist_resolved.is_relative_to(root.resolve())
            else str(wordlist_resolved)
        )
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
        "scope": scope,
        "self_excluded": self_excluded,
        "files_scanned": len(files),
        "findings_count": len(findings),
        "status": "pass" if not findings else "fail",
        "findings": [finding.__dict__ for finding in findings],
    }


def _categories(wordlist_path: Path):
    data = tomllib.loads(wordlist_path.read_text(encoding="utf-8"))
    return data["categories"]


def read_baseline(path: Path) -> int:
    """基线件只允许「注释行 + 恰一个整数」；缺失、多值或非整数一律 fail-closed。

    动机（A-2 棘轮）：非阻断的审计没有基线就无法区分「存量债」与「新增泄漏」，
    也就永远不会下降。基线只记一个数——数字本身不是敏感面，故可入库。
    """
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if len(values) != 1:
        raise ValueError(f"基线件须恰有一个整数（当前 {len(values)} 个非注释行）")
    try:
        number = int(values[0])
    except ValueError as exc:
        raise ValueError(f"基线值非整数: {values[0]!r}") from exc
    if number < 0:
        raise ValueError(f"基线值不可为负: {number}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path, help="要扫描的文件或目录")
    parser.add_argument(
        "--tracked",
        action="store_true",
        help="公开仓口径：扫描全部受版本控制文件（git ls-files），忽略 roots",
    )
    parser.add_argument(
        "--wordlist",
        type=Path,
        default=Path(__file__).with_name("sensitive-terms.toml"),
    )
    parser.add_argument("--report", type=Path, help="JSON 报告输出路径")
    parser.add_argument(
        "--baseline",
        type=Path,
        help="棘轮基线件（只允许「注释 + 一个整数」，现 52）。"
        "findings > 基线 ⇒ rc=1（新增泄漏，阻断）；findings < 基线 ⇒ rc=0 且提示下调；相等 ⇒ rc=0。",
    )
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
    if not args.roots and not args.tracked:
        parser.error("需要 roots，或使用 --tracked 采用全仓口径")
    resolved_roots = [path.resolve() for path in args.roots]
    if args.tracked:
        root = Path.cwd().resolve()
    else:
        root = Path(os.path.commonpath([str(Path.cwd().resolve()), *(str(path) for path in resolved_roots)]))
    excludes = (*DEFAULT_EXCLUDES, *args.exclude)
    try:
        result = scan(
            root,
            resolved_roots,
            args.wordlist,
            excludes,
            scope="tracked" if args.tracked else "roots",
            use_tracked=args.tracked,
        )
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"fail-closed: {error}", file=sys.stderr)
        return 2
    baseline: int | None = None
    if args.baseline is not None:
        try:
            baseline = read_baseline(args.baseline)
        except (OSError, ValueError) as error:
            print(f"fail-closed: 基线不可读: {error}", file=sys.stderr)
            return 2
        result["baseline"] = baseline
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    summary = f"status={result['status']} files={result['files_scanned']} findings={result['findings_count']}"
    if baseline is None:
        print(summary)
        return 0 if result["status"] == "pass" else 1
    # 棘轮（A-2）：存量债不阻断，新增泄漏阻断，下降必须人工下调基线（不自动改文件）
    count = int(result["findings_count"])
    if count > baseline:
        print(f"{summary} baseline={baseline} ratchet=increase")
        print(
            f"新增敏感面命中 {count - baseline} 条（{count} > 基线 {baseline}）："
            "请脱敏，或在白名单件中显式登记（含 reason/owner/过期日）",
            file=sys.stderr,
        )
        return 1
    if count < baseline:
        print(f"{summary} baseline={baseline} ratchet=decrease")
        print(
            f"存量已降低（{count} < 基线 {baseline}）：请下调基线件并在提交信息留痕",
            file=sys.stderr,
        )
        return 0
    print(f"{summary} baseline={baseline} ratchet=hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
