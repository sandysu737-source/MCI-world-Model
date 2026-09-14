"""P0-2d 因果推理拆分的结构与入口兼容测试。"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from mci_world_model.sdk._world_model import MCIWorldModel

CAUSAL_MODULE = import_module("mci_world_model.sdk._causal_inference")


def test_causal_methods_are_owned_by_mixin() -> None:
    method_names = (
        "_build_causal_graph_from_state",
        "intervene",
        "decompose_effect",
        "query_counterfactual",
        "explain",
        "_trace_causal_chains",
        "_generate_explanation_summary",
    )
    assert any(base is CAUSAL_MODULE.CausalInferenceMixin for base in MCIWorldModel.__mro__)
    for name in method_names:
        assert name not in MCIWorldModel.__dict__
        assert getattr(MCIWorldModel, name).__qualname__.startswith("CausalInferenceMixin.")


def test_legacy_causal_entry_points_exist() -> None:
    assert callable(MCIWorldModel.intervene)
    assert callable(MCIWorldModel.decompose_effect)
    assert callable(MCIWorldModel.query_counterfactual)
    assert callable(MCIWorldModel.explain)
    assert callable(MCIWorldModel._build_causal_graph_from_state)


def test_causal_module_does_not_import_world_model() -> None:
    source = Path(CAUSAL_MODULE.__file__).read_text(encoding="utf-8")
    assert "from mci_world_model.sdk._world_model import" not in source
    assert "import mci_world_model.sdk._world_model\n" not in source
