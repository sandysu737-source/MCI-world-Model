"""P0-2b 能量流拆分的结构与行为兼容测试。"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

from mci_world_model.sdk._world_model import MCIWorldModel

ENERGY_MODULE = import_module("mci_world_model.sdk._energy_flow")
WORLD_MODEL_MODULE = import_module("mci_world_model.sdk._world_model")


def _bare_model() -> MCIWorldModel:
    return object.__new__(MCIWorldModel)


def test_private_energy_function_is_reexported_without_duplicate() -> None:
    assert WORLD_MODEL_MODULE._aggregate_energy_ratios is ENERGY_MODULE._aggregate_energy_ratios


def test_energy_methods_are_owned_by_mixin() -> None:
    method_names = (
        "_extract_energy_ratios",
        "_compute_energy_coverage",
        "predict_energy_flow",
    )
    assert any(base is ENERGY_MODULE.EnergyFlowMixin for base in MCIWorldModel.__mro__)
    for name in method_names:
        assert name not in MCIWorldModel.__dict__
        assert getattr(MCIWorldModel, name).__qualname__.startswith("EnergyFlowMixin.")
    for adapter_name in ("_build_energy_bus", "_propagate_energy"):
        assert adapter_name in MCIWorldModel.__dict__


def test_energy_module_does_not_import_world_model() -> None:
    source = Path(ENERGY_MODULE.__file__).read_text(encoding="utf-8")
    assert "from mci_world_model.sdk._world_model" not in source
    assert "import mci_world_model.sdk._world_model" not in source
    assert "from su_memory" not in source
    assert "import su_memory" not in source


def test_aggregate_energy_ratios_keeps_empty_and_unlabeled_behavior() -> None:
    aggregate = ENERGY_MODULE._aggregate_energy_ratios
    assert aggregate([]) is None
    assert aggregate([{"rho": 0.5}]) is None
    result = aggregate(
        [
            {"cause_energy": "semantic", "effect_energy": "causal"},
            {"cause_energy": "semantic", "effect_energy": "trust"},
        ]
    )
    assert result == {
        "semantic": 0.5,
        "causal": 0.25,
        "spacetime": 0.0,
        "generative": 0.0,
        "trust": 0.25,
    }


def test_extract_and_coverage_preserve_empty_state_behavior() -> None:
    model = _bare_model()
    model._state = SimpleNamespace(causal_edges=[])
    assert model._extract_energy_ratios(model._state) is None
    assert model._extract_energy_ratios(object()) is None
    assert model._compute_energy_coverage() == {
        "ratios": {},
        "coverage_score": 0.0,
        "warning": "能量维度覆盖不足，建议丰富数据源",
    }


def test_predict_energy_flow_reuses_existing_predictor() -> None:
    model = _bare_model()
    model._state = SimpleNamespace(causal_edges=[])
    model._energy_core = object()
    expected_ratios = {
        "semantic": 0.2,
        "causal": 0.2,
        "spacetime": 0.2,
        "generative": 0.2,
        "trust": 0.2,
    }

    class FakePredictor:
        def __init__(self) -> None:
            self.predict_calls = 0

        def predict(self, ratios: dict[str, float], steps: int) -> list[dict[str, float]]:
            self.predict_calls += 1
            assert steps == 3
            assert ratios == expected_ratios
            return [expected_ratios]

        def detect_anomaly(self, flow: list[dict[str, float]]) -> bool:
            assert flow == [expected_ratios]
            return True

    predictor = FakePredictor()
    model._energy_flow_predictor = predictor
    result = model.predict_energy_flow(steps=3)
    assert result == {
        "steps": 3,
        "flow": [expected_ratios],
        "anomaly_detected": True,
        "current_ratios": expected_ratios,
    }
    assert predictor.predict_calls == 1
    assert model._energy_flow_predictor is predictor
