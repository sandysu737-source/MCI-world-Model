"""P0-2e 认知组件拆分的结构与入口兼容测试。"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from mci_world_model.sdk._world_model import MCIWorldModel

COGNITIVE_MODULE = import_module("mci_world_model.sdk._cognitive_components")


def test_cognitive_methods_are_owned_by_mixin() -> None:
    method_names = (
        "run_cognitive_loop",
        "diagnose_failure",
        "retrieve_experiences",
        "detect_surprise",
        "plan_action",
        "synthesize_training_data",
        "assess_diversity",
        "check_admissibility",
    )
    assert any(base is COGNITIVE_MODULE.CognitiveComponentsMixin for base in MCIWorldModel.__mro__)
    for name in method_names:
        assert name not in MCIWorldModel.__dict__
        assert getattr(MCIWorldModel, name).__qualname__.startswith("CognitiveComponentsMixin.")


def test_legacy_cognitive_entry_points_exist() -> None:
    assert callable(MCIWorldModel.run_cognitive_loop)
    assert callable(MCIWorldModel.diagnose_failure)
    assert callable(MCIWorldModel.retrieve_experiences)
    assert callable(MCIWorldModel.detect_surprise)
    assert callable(MCIWorldModel.plan_action)
    assert callable(MCIWorldModel.synthesize_training_data)
    assert callable(MCIWorldModel.assess_diversity)
    assert callable(MCIWorldModel.check_admissibility)


def test_cognitive_module_does_not_import_world_model() -> None:
    source = Path(COGNITIVE_MODULE.__file__).read_text(encoding="utf-8")
    assert "from mci_world_model.sdk._world_model import" not in source
    assert "import mci_world_model.sdk._world_model\n" not in source
