"""P0-2f 预测域拆分的结构与入口兼容测试。"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from mci_world_model.sdk._world_model import MCIWorldModel

PREDICTION_MODULE = import_module("mci_world_model.sdk._prediction_components")


def test_prediction_methods_are_owned_by_mixin() -> None:
    method_names = (
        "attach_true_jepa",
        "predict_effect",
        "jepa_predict",
        "parametric_predict",
        "predict_from_memories_m3",
        "fused_predict",
        "train_jepa",
        "_is_gnn_predictor",
        "_is_e2e_mode",
        "enable_m3",
    )
    assert any(base is PREDICTION_MODULE.PredictionComponentsMixin for base in MCIWorldModel.__mro__)
    for name in method_names:
        assert name not in MCIWorldModel.__dict__
        assert getattr(MCIWorldModel, name).__qualname__.startswith("PredictionComponentsMixin.")


def test_legacy_prediction_entry_points_exist() -> None:
    assert callable(MCIWorldModel.attach_true_jepa)
    assert callable(MCIWorldModel.predict_effect)
    assert callable(MCIWorldModel.jepa_predict)
    assert callable(MCIWorldModel.parametric_predict)
    assert callable(MCIWorldModel.predict_from_memories_m3)
    assert callable(MCIWorldModel.fused_predict)
    assert callable(MCIWorldModel.train_jepa)
    assert callable(MCIWorldModel._is_gnn_predictor)
    assert callable(MCIWorldModel._is_e2e_mode)
    assert callable(MCIWorldModel.enable_m3)


def test_prediction_module_does_not_import_world_model() -> None:
    source = Path(PREDICTION_MODULE.__file__).read_text(encoding="utf-8")
    assert "from mci_world_model.sdk._world_model import" not in source
    assert "import mci_world_model.sdk._world_model\n" not in source


def test_prediction_mode_helpers_keep_host_state_semantics() -> None:
    world_model = MCIWorldModel()
    world_model._jepa_predictor = object()
    assert world_model._is_gnn_predictor() is False

    class GNNPredictor:
        def training_predict(self) -> None: ...

    world_model._jepa_predictor = GNNPredictor()
    assert world_model._is_gnn_predictor() is True
    assert world_model._is_e2e_mode() is False
