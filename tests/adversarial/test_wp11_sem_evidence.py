"""WP-11 反事实 SEM 数据拟合与溯源对抗测试。"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._counterfactual import CounterfactualEngine
from mci_world_model.sdk._do_calculus import CausalGraph, ObservationDataset

pytestmark = pytest.mark.contract


def _dataset(x: np.ndarray, y: np.ndarray) -> ObservationDataset:
    return ObservationDataset(
        values={"X": x, "Y": y},
        dataset_id="linear-sem-audit",
        source="observed",
        seed=42,
    )


def test_query_without_dataset_is_fail_closed() -> None:
    graph = CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])
    engine = CounterfactualEngine.from_causal_graph(graph)

    result = engine.query(evidence={"X": 1.0, "Y": 2.0}, do_x={"X": 0.5}, target="Y")

    assert result.status == "error"
    assert result.mode == "no_data"
    assert result.sem_fit_id is None
    assert result.is_conclusive is False


def test_linear_coefficients_are_fitted_from_data() -> None:
    rng = np.random.RandomState(42)
    x = rng.randn(1000)
    y = 2.0 * x + 0.2 * rng.randn(1000)
    graph = CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])
    engine = CounterfactualEngine.from_causal_graph(graph, dataset=_dataset(x, y), seed=42)
    sem_fit = engine.sem_fit

    assert sem_fit is not None
    assert abs(sem_fit.coefficients[0, 1] - 2.0) < 0.05
    assert sem_fit.identifiability == "identified"
    assert sem_fit.fit_metrics["r2"] > 0.98


def test_result_is_traceable_and_conclusive() -> None:
    rng = np.random.RandomState(42)
    x = rng.randn(1000)
    y = 2.0 * x + 0.2 * rng.randn(1000)
    graph = CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])
    engine = CounterfactualEngine.from_causal_graph(graph, dataset=_dataset(x, y), seed=42)
    result = engine.query(evidence={"X": 1.0, "Y": 2.0}, do_x={"X": 0.5}, target="Y")

    assert result.status == "ok"
    assert result.sem_fit_id == engine.sem_fit.fit_id
    assert result.mode == "fitted_sem_observed"
    assert result.identifiability == "identified"
    assert result.is_conclusive is True
    assert result.fit_error >= 0.0
    assert result.to_dict()["sem_fit_id"] == result.sem_fit_id
