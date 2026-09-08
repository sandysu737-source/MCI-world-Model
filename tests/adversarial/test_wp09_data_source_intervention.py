"""WP-09 数据来源与多变量干预契约测试。"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._do_calculus import CausalGraph, DoCalculus, ObservationDataset
from mci_world_model.sdk._world_model import MCIWorldModel

pytestmark = pytest.mark.contract


def _direct_graph() -> CausalGraph:
    return CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])


def test_estimate_ate_without_data_returns_no_data() -> None:
    result = DoCalculus(graph=_direct_graph()).estimate_ate("X", "Y")

    assert result.method == "no_data"
    assert result.mode == "no_data"
    assert result.is_conclusive is False
    assert result.sample_size == 0
    assert result.note == "no_data"


def test_explicit_simulation_marks_mode_and_disallows_conclusion() -> None:
    calculus = DoCalculus(graph=_direct_graph(), seed=123)
    simulated = calculus.simulate(n_samples=200, seed=123)

    assert simulated.source == "simulated"
    assert simulated.seed == 123
    result = calculus.estimate_ate("X", "Y")

    assert result.mode == "simulated"
    assert result.is_conclusive is False
    assert result.sample_size == 200
    assert result.dataset_hash == simulated.dataset_hash
    assert result.seed == 123


def test_simulated_data_is_reproducible() -> None:
    graph = _direct_graph()
    first = DoCalculus(graph=graph, seed=7).simulate(n_samples=100, seed=42, bind=False)
    second = DoCalculus(graph=graph, seed=8).simulate(n_samples=100, seed=42, bind=False)

    assert first.dataset_hash == second.dataset_hash
    assert all(np.array_equal(first.values[name], second.values[name]) for name in first.values)


def test_observed_dataset_writes_evidence_fields() -> None:
    dataset = ObservationDataset(
        values={"X": np.array([0.0, 1.0, 0.0, 1.0]), "Y": np.array([0.0, 1.1, 0.1, 1.0])},
        dataset_id="dataset-001",
    )
    result = DoCalculus(graph=_direct_graph(), dataset=dataset).estimate_ate("X", "Y")

    assert result.mode == "observed"
    assert result.is_conclusive is True
    assert result.dataset_hash == dataset.dataset_hash
    assert result.estimator == "direct"
    assert result.do_x == {"X": 1.0}
    assert result.x_baseline == 0.0
    assert result.confidence_interval[0] <= result.confidence_interval[1]


def test_multivariate_do_is_unsupported_independently_of_key_order() -> None:
    calculus = DoCalculus(graph=CausalGraph(nodes=["A", "B", "Y"], edges=[]))

    first = calculus.estimate_intervention({"A": 1.0, "B": 2.0}, target="Y")
    second = calculus.estimate_intervention({"B": 2.0, "A": 1.0}, target="Y")

    assert first.method == "unsupported"
    assert first.note == "multivariate_do_not_implemented"
    assert first.to_dict() == second.to_dict()


def test_world_model_rejects_multivariate_do_with_422() -> None:
    world_model = MCIWorldModel()

    first = world_model.intervene(do_x={"A": 1.0, "B": 2.0}, target="Y")
    second = world_model.intervene(do_x={"B": 2.0, "A": 1.0}, target="Y")

    assert first["status"] == "unsupported"
    assert first["code"] == 422
    assert first["reason"] == "multivariate_do_not_implemented"
    assert first == second


def test_world_model_rejects_non_finite_do_values() -> None:
    world_model = MCIWorldModel()

    result = world_model.intervene(do_x={"A": float("nan")}, target="Y")

    assert result["status"] == "rejected"
    assert result["code"] == 422
    assert result["reason"] == "non_finite_do_x"


def test_observation_dataset_rejects_invalid_values() -> None:
    ragged = ObservationDataset(values={"X": np.array([1.0]), "Y": np.array([1.0, 2.0])})
    assert ragged.values["Y"].size == 2
    with pytest.raises(ValueError, match="非有限"):
        ObservationDataset(values={"X": np.array([np.inf])})
