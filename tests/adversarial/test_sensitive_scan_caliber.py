"""口径灵敏度 canary（台账 OODA-20261016-AIGUARD-R2-CALIBER-027）。

为什么存在：R2 交付物宣称敏感面「52 -> 0」，但这个 0 **只在门禁口径内成立**。
门禁口径见 ``scripts/ai-verify/sensitive_scan.py::_compiled_patterns``：
单字术语必须「前后非汉字」（concept 形状）才计入；双字及以上按裸字面匹配。
而公开仓真实文本里，单字术语最常出现在**连续汉字上下文**中（映射表整行、中文注释、
中文文档），门禁口径对这些形态**不可见** —— 口径内 0 不等于公开面 0。

为什么样本用「真实形态」而非 concept 形状：用检测器唯一认得的形状去测检测器，
必然全绿、不携带信息（自证）。真实形态 = 术语嵌在连续汉字里。

三条断言就是口径决策的验收条件：
  1. 门禁口径对真实形态命中 0 ⇒ 缺口存在，且是**显式**登记的（不是静默）；
  2. 同一批样本裸口径命中 > 0 ⇒ 样本有效，不是空样本；
  3. 门禁口径对 concept 形状仍能命中 ⇒ 缺口是**口径问题**，不是扫描器故障；
  4. 口径一旦放宽（或缺口被修复）⇒ 本测试失败，**强制重登口径决策**。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests._gov_terms import ENERGY_TYPES

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ai-verify/sensitive_scan.py"


def _scan(root: Path, target: str) -> dict:
    report = root / "report.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--report", str(report), target],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert report.exists(), result.stderr
    return json.loads(report.read_text(encoding="utf-8"))


def _real_shapes() -> str:
    """真实形态：单字术语嵌在**连续汉字**里（映射表整行／中文注释的典型形状）。"""
    return "\n".join(f"甲乙{term}丙丁" for term in ENERGY_TYPES) + "\n"


def test_canary_precondition_terms_are_single_cjk() -> None:
    """本 canary 只对单字术语有意义：双字及以上走裸字面匹配，不存在口径缺口。"""
    assert ENERGY_TYPES
    assert all(len(term) == 1 and "\u4e00" <= term <= "\u9fff" for term in ENERGY_TYPES)


def test_gate_caliber_is_blind_to_real_shapes(tmp_path: Path) -> None:
    """缺口断言（口径决策的验收条件）：真实形态在门禁口径下不可见。"""
    raw = _real_shapes()
    (tmp_path / "sample.md").write_text(raw, encoding="utf-8")

    payload = _scan(tmp_path, "sample.md")

    assert payload["findings_count"] == 0, (
        "门禁口径对「连续汉字上下文中的单字术语」不再不可见 ⇒ 口径已变化，"
        "必须重登口径决策（本 canary 是决策的验收条件，不得静默放宽）"
    )
    assert sum(raw.count(term) for term in ENERGY_TYPES) == len(ENERGY_TYPES), "样本构造失效（裸口径应为 0）"


def test_gate_caliber_sees_concept_shape(tmp_path: Path) -> None:
    """对照组：门禁口径对 concept 形状**能**命中 ⇒ 上面的 0 是口径问题，不是扫描器坏了。"""
    (tmp_path / "sample.md").write_text("\n".join(f"({term})\n" for term in ENERGY_TYPES), encoding="utf-8")

    payload = _scan(tmp_path, "sample.md")

    assert payload["findings_count"] == len(ENERGY_TYPES)
