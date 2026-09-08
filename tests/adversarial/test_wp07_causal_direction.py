"""WP-07 因果方向与确定性标识对抗测试。"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from mci_world_model.sdk._counterfactual import CounterfactualEngine
from mci_world_model.sdk._do_calculus import CausalGraph, DoCalculus
from mci_world_model.sdk._spectral_causal import GaussianDAG

pytestmark = pytest.mark.contract


def _correlation_dag(y_values: list[float]) -> GaussianDAG:
    """构造矩阵化 DAG，避免文本词频噪声影响方向断言。"""
    dag = GaussianDAG([{"id": "0", "content": "x"}, {"id": "1", "content": "y"}, {"id": "2", "content": "z"}])
    dag._tfidf_matrix = np.vstack(
        [
            np.arange(1.0, 7.0),
            np.asarray(y_values, dtype=float),
            np.asarray([6.0, 5.0, 3.0, 4.0, 1.0, 2.0]),
        ]
    )
    return dag


@pytest.mark.parametrize(
    ("y_values", "expected_rho_sign"),
    [
        ([1.1, 2.1, 2.9, 4.2, 4.8, 6.1], 1),
        ([-1.1, -2.1, -2.9, -4.2, -4.8, -6.1], -1),
    ],
)
def test_positive_and_negative_correlation_do_not_orient(y_values: list[float], expected_rho_sign: int) -> None:
    edge = next(
        edge
        for edge in _correlation_dag(y_values).discover_hidden_edges(min_correlation=0.1, p_threshold=0.2)
        if {edge["source_idx"], edge["target_idx"]} == {0, 1}
    )

    assert np.sign(edge["rho"]) == expected_rho_sign
    assert edge["orientation"] == "undirected"
    assert edge["edge_mode"] == "correlation"
    assert edge["source_idx"] == 0
    assert edge["target_idx"] == 1


def test_zero_correlation_does_not_create_oriented_edge() -> None:
    edges = _correlation_dag([1.0, -1.0, 1.0, -1.0, 1.0, -1.0]).discover_hidden_edges(
        min_correlation=0.1,
        p_threshold=0.2,
    )

    assert not any({edge["source_idx"], edge["target_idx"]} == {0, 1} for edge in edges)


@pytest.mark.parametrize("hash_seed", ["0", "1", "42"])
def test_energy_type_rule_is_stable_across_hash_seeds(hash_seed: str) -> None:
    code = (
        "import json;"
        "from mci_world_model.sdk._spectral_causal import GaussianDAG;"
        "dag=GaussianDAG([]);"
        "value, rule=dag.infer_energy_type_with_rule({'content':'高温激活'}) ;"
        "print(json.dumps([value, rule.rule_id, rule.version]))"
    )
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = hash_seed
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout.strip()) == ["fire", "activation", "energy-rules-v1"]


def test_explicit_energy_type_has_priority_over_rule_table() -> None:
    dag = GaussianDAG([])
    inferred, rule = dag.infer_energy_type_with_rule({"energy_type": "water", "content": "高温激活"})

    assert inferred == "water"
    assert rule is None


def test_energy_rule_version_change_is_reproducible() -> None:
    dag = GaussianDAG([])
    old_value, old_rule = dag.infer_energy_type_with_rule({"content": "血压体液循环"})
    dag.energy_type_rules = tuple(
        dataclasses.replace(rule, version="energy-rules-v2") for rule in dag.energy_type_rules
    )
    new_value, new_rule = dag.infer_energy_type_with_rule({"content": "血压体液循环"})

    assert old_value == new_value == "water"
    assert old_rule is not None and new_rule is not None
    assert old_rule.version == "energy-rules-v1"
    assert new_rule.version == "energy-rules-v2"


def test_association_edge_cannot_enter_do_calculus() -> None:
    graph = CausalGraph.build_from_gaussian_dag(
        [{"cause_idx": 0, "effect_idx": 1, "confidence": 0.9, "edge_mode": "correlation"}],
        n_nodes=2,
        node_names=["X", "Y"],
    )
    result = DoCalculus(graph=graph).estimate_ate("X", "Y")

    assert graph.edge_mode == "correlation"
    assert graph.is_causal_graph is False
    assert result.method == "rejected"
    assert result.note == "association_graph_not_causal"
    assert result.is_conclusive is False


def test_missing_orientation_evidence_defaults_to_correlation() -> None:
    graph = CausalGraph.build_from_gaussian_dag(
        [{"cause_idx": 0, "effect_idx": 1, "confidence": 0.9}],
        n_nodes=2,
        node_names=["X", "Y"],
    )

    assert graph.edge_mode == "correlation"
    assert DoCalculus(graph=graph).identify_adjustment_set("X", "Y") is None
    assert DoCalculus(graph=graph).identify_frontdoor_mediators("X", "Y") is None


def test_explicit_orientation_builds_causal_graph() -> None:
    graph = CausalGraph.build_from_gaussian_dag(
        [
            {
                "cause_idx": 0,
                "effect_idx": 1,
                "confidence": 0.9,
                "edge_mode": "orientation_by_intervention",
            }
        ],
        n_nodes=2,
        node_names=["X", "Y"],
    )

    assert graph.edge_mode == "causal"
    assert graph.is_causal_graph is True
    result = DoCalculus(graph=graph).estimate_ate("X", "Y")
    assert result.method == "no_data"


def test_unknown_edge_mode_fails_closed() -> None:
    with pytest.raises(ValueError, match="不支持的 edge_mode"):
        CausalGraph.build_from_gaussian_dag(
            [{"cause_idx": 0, "effect_idx": 1, "confidence": 0.9, "edge_mode": "sign"}],
            n_nodes=2,
        )


def test_counterfactual_rejects_association_graph() -> None:
    graph = CausalGraph.build_from_gaussian_dag(
        [{"cause_idx": 0, "effect_idx": 1, "confidence": 0.9}],
        n_nodes=2,
    )

    with pytest.raises(ValueError, match="关联图禁止构建反事实引擎"):
        CounterfactualEngine.from_causal_graph(graph)


def test_graph_serialization_preserves_association_contract() -> None:
    graph = CausalGraph.build_from_gaussian_dag(
        [{"cause_idx": 0, "effect_idx": 1, "confidence": 0.9}],
        n_nodes=2,
        dataset_id="dataset-007",
    )

    restored = CausalGraph.from_dict(graph.to_dict())
    assert restored.edge_mode == "correlation"
    assert restored.is_causal_graph is False
    assert restored.dataset_id == "dataset-007"
