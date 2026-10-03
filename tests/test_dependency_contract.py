"""可选依赖版本契约回归。"""

from __future__ import annotations

from pathlib import Path

import pytest

from mci_world_model.sdk._world_model_state import _monthly_cycle_index

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 兼容
    import tomli as tomllib


def test_su_memory_dependency_uses_closed_compatibility_range() -> None:
    """su-memory 依赖必须与上游 V4.4.1 兼容证据口径一致。"""
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)

    memory_dependencies = data["project"]["optional-dependencies"]["memory"]

    assert memory_dependencies == ["su-memory>=4.4.1,<4.5"]


def test_monthly_cycle_index_always_satisfies_parity() -> None:
    """任意年月的 60 月周期都必须生成同奇偶干支组合。"""
    for year in range(1980, 2101):
        for month in range(1, 13):
            cycle_index = _monthly_cycle_index(year, month)
            assert (cycle_index % 10) % 2 == (cycle_index % 12) % 2


def test_monthly_cycle_index_rejects_invalid_month() -> None:
    """非法月份必须 fail-closed。"""
    with pytest.raises(ValueError, match="month"):
        _monthly_cycle_index(2026, 13)
