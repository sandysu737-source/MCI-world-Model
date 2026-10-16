"""受控词表引用器：本仓（公开仓）不书写领域术语，统一从词表按位取用。

背景：`mci-world-model` 为公开仓（**入库即发布**），术语的唯一登记处是
`scripts/ai-verify/sensitive-terms.toml`；其余文件一律**构造式引用**，
使敏感面棘轮（台账 OODA-20261015-PUBSURFACE-016 / -020）保持单调下降。
常量命名延续 `src` 的英文口径（`TimeStem`／`TimeBranch` 等）。

形状契约（OODA-20261016-AIGUARD-R2-CLOSE-023）：词表类目数与每类条目数**先校验再用**，
不符即抛 `ValueError`。仅靠解包只会抛 `IndexError`／解包错误，错误信息不指向「词表变了」
这一根因，且「类目多、条目少」的畸形词表会静默错配成语义已变的数据。
"""

from __future__ import annotations

from pathlib import Path

import tomllib

WORDLIST = Path(__file__).resolve().parents[1] / "scripts" / "ai-verify" / "sensitive-terms.toml"

# 位次 → 该位次所需的最少条目数（索引 0 为分类名字，不取用）。
SHAPE: dict[int, int] = {1: 5, 2: 2, 3: 1, 4: 3, 5: 1}


def _load(path: Path | None = None) -> list[list[str]]:
    """读词表；`path` 便于测试用合成词表验证形状校验。"""
    src = WORDLIST if path is None else path
    data = tomllib.loads(src.read_text(encoding="utf-8"))
    return [list(cat["terms"]) for cat in data["categories"]]


def check_shape(cats: list[list[str]]) -> None:
    """形状不符 ⇒ 显式 ValueError（不静默错配）。"""
    for idx, need in SHAPE.items():
        if len(cats) <= idx:
            raise ValueError(f"词表类目数不足：缺位次 {idx}（实得 {len(cats)} 类）")
        if len(cats[idx]) < need:
            raise ValueError(f"词表位次 {idx} 条目不足：需 {need}，实得 {len(cats[idx])}")


_CATS = _load()
check_shape(_CATS)

ENERGY_TYPES: tuple[str, ...] = tuple(_CATS[1])
WOOD, FIRE, EARTH, METAL, WATER = ENERGY_TYPES
ENHANCE, SUPPRESS = _CATS[2][0], _CATS[2][1]
TRIGRAM = _CATS[3][0]
TIME_STEMS, TIME_BRANCHES, TIME_PILLARS = _CATS[4][0], _CATS[4][1], _CATS[4][2]
DUALITY = _CATS[5][0]

__all__ = [
    "DUALITY",
    "EARTH",
    "ENERGY_TYPES",
    "ENHANCE",
    "FIRE",
    "METAL",
    "SHAPE",
    "SUPPRESS",
    "TIME_BRANCHES",
    "TIME_PILLARS",
    "TIME_STEMS",
    "TRIGRAM",
    "WATER",
    "WORDLIST",
    "WOOD",
    "check_shape",
]
