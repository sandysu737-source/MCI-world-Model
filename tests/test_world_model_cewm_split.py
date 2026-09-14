"""P0-2c CEWM 拆分的结构与入口兼容测试。"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from mci_world_model.sdk._world_model import MCIWorldModel

CEWM_MODULE = import_module("mci_world_model.sdk._cewm_engine")


def test_cewm_methods_are_owned_by_mixin() -> None:
    method_names = (
        "_init_cewm_result",
        "_cewm_perceive",
        "_cewm_safety_check",
        "_cewm_cognize",
        "_cewm_evaluate_action",
        "_cewm_predict",
        "_cewm_feedback",
        "cewm_step",
        "_store_replay_experience",
        "_replay_train",
        "cewm_step_fast",
        "_cewm_parse_state",
        "_cewm_state_change",
    )
    assert any(base is CEWM_MODULE.CEWMEngineMixin for base in MCIWorldModel.__mro__)
    for name in method_names:
        assert name not in MCIWorldModel.__dict__
        assert getattr(MCIWorldModel, name).__qualname__.startswith("CEWMEngineMixin.")


def test_legacy_cewm_entry_points_exist() -> None:
    assert callable(MCIWorldModel.cewm_step)
    assert callable(MCIWorldModel.cewm_step_fast)
    assert callable(MCIWorldModel._init_cewm_result)
    assert callable(MCIWorldModel._cewm_parse_state)
    assert callable(MCIWorldModel._cewm_state_change)


def test_cewm_module_does_not_import_world_model() -> None:
    source = Path(CEWM_MODULE.__file__).read_text(encoding="utf-8")
    assert "from mci_world_model.sdk._world_model" not in source
    assert "import mci_world_model.sdk._world_model" not in source
