"""`tests/_gov_terms.py` 的不变式测试（台账 OODA-20261016-AIGUARD-R2-CLOSE-023）。

为什么这个测试存在：`_gov_terms.py` 是公开面脱敏（R2）的**承重件**——对抗扫描用例的
测试数据由它构造。它一旦错配（取错位次／取到空串），下游用例会「看起来仍通过但断言力
被悄悄削弱」。因此这里逐位对照受控词表独立复读，并把「词表形状变化必须显式失败」钉死。
"""

from __future__ import annotations

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 兼容（依赖已声明 tomli）
    import tomli as tomllib

from tests import _gov_terms as gt

EXPORTED = (
    gt.WOOD,
    gt.FIRE,
    gt.EARTH,
    gt.METAL,
    gt.WATER,
    gt.ENHANCE,
    gt.SUPPRESS,
    gt.TRIGRAM,
    gt.TIME_STEMS,
    gt.TIME_BRANCHES,
    gt.TIME_PILLARS,
    gt.DUALITY,
)


def _registry() -> list[list[str]]:
    data = tomllib.loads(gt.WORDLIST.read_text(encoding="utf-8"))
    return [list(cat["terms"]) for cat in data["categories"]]


def _cats(**over: list[str]) -> list[list[str]]:
    """构造形状合法的最小词表；`over` 用于把某位次替换为畸形内容。"""
    base = [
        ["cat0"],
        ["e1", "e2", "e3", "e4", "e5"],
        ["c1", "c2"],
        ["d1"],
        ["f1", "f2", "f3"],
        ["g1"],
    ]
    for key, value in over.items():
        base[int(key.removeprefix("idx"))] = value
    return base


def test_wordlist_is_the_documented_single_source() -> None:
    assert gt.WORDLIST.exists()
    assert gt.WORDLIST.name == "sensitive-terms.toml"
    assert gt.WORDLIST.parts[-3:] == ("scripts", "ai-verify", "sensitive-terms.toml")


def test_constants_equal_registry_positions() -> None:
    cats = _registry()
    assert list(gt.ENERGY_TYPES) == cats[1]
    assert (cats[2][0], cats[2][1]) == (gt.ENHANCE, gt.SUPPRESS)
    assert cats[3][0] == gt.TRIGRAM
    assert (cats[4][0], cats[4][1], cats[4][2]) == (gt.TIME_STEMS, gt.TIME_BRANCHES, gt.TIME_PILLARS)
    assert cats[5][0] == gt.DUALITY


def test_energy_unpacking_follows_registry_order() -> None:
    assert (gt.WOOD, gt.FIRE, gt.EARTH, gt.METAL, gt.WATER) == gt.ENERGY_TYPES
    assert len(gt.ENERGY_TYPES) == 5


def test_exports_are_nonempty_and_distinct() -> None:
    assert all(isinstance(v, str) and v for v in EXPORTED)
    assert len(set(EXPORTED)) == len(EXPORTED)


def test_load_reads_terms_from_given_path(tmp_path) -> None:
    synth = tmp_path / "w.toml"
    synth.write_text(
        '[[categories]]\nname = "a"\nterms = ["x", "y"]\n',
        encoding="utf-8",
    )
    assert gt._load(synth) == [["x", "y"]]


def test_check_shape_accepts_minimal_valid_wordlist() -> None:
    gt.check_shape(_cats())


@pytest.mark.parametrize("idx,need", sorted(gt.SHAPE.items()))
def test_check_shape_rejects_short_category(idx: int, need: int) -> None:
    with pytest.raises(ValueError, match=f"位次 {idx}"):
        gt.check_shape(_cats(**{f"idx{idx}": ["only"] * (need - 1)}))


def test_check_shape_rejects_missing_category() -> None:
    with pytest.raises(ValueError, match="类目数不足"):
        gt.check_shape(_cats()[:3])


def test_check_shape_rejects_category_at_exact_boundary_length() -> None:
    """位次 5 只需 1 条：恰好 1 条合法、0 条非法（防 `<=` / `<` 互串）。"""
    gt.check_shape(_cats(idx5=["g1"]))
    with pytest.raises(ValueError, match="位次 5"):
        gt.check_shape(_cats(idx5=[]))
