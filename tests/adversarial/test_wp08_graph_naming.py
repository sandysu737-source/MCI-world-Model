"""WP-08 图命名、业务查询与标识映射测试。"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._do_calculus import CausalGraph, DoCalculus, UnknownNodeError

pytestmark = pytest.mark.contract


def _named_graph() -> CausalGraph:
    return CausalGraph(
        nodes=["混杂因素", "干预", "结果"],
        edges=[("混杂因素", "干预"), ("混杂因素", "结果"), ("干预", "结果")],
        node_aliases={"干预": ["treatment", "V1"], "结果": ["outcome", "V2"]},
    )


def test_build_from_gaussian_dag_uses_business_names() -> None:
    edges = [{"cause_idx": 0, "effect_idx": 1, "confidence": 0.9}]
    graph = CausalGraph.build_from_gaussian_dag(
        edges,
        n_nodes=2,
        node_names=["低白蛋白", "营养不良"],
        dataset_id="dataset-001",
    )

    assert graph.nodes == ["低白蛋白", "营养不良"]
    assert graph.edges == [("低白蛋白", "营养不良")]
    assert graph.node_aliases == {"低白蛋白": ["V0"], "营养不良": ["V1"]}
    assert graph.dataset_id == "dataset-001"
    assert graph.edge_mode == "correlation"


def test_build_from_gaussian_dag_uses_memory_names() -> None:
    memories = [
        {"id": "低白蛋白", "content": "患者检测"},
        {"id": "营养不良", "content": "临床判断"},
    ]
    edges = [{"cause_idx": 0, "effect_idx": 1, "confidence": 0.8}]
    graph = CausalGraph.build_from_gaussian_dag(edges, n_nodes=2, memories=memories)

    assert graph.resolve_node("低白蛋白") == "低白蛋白"
    assert graph.resolve_node("V1") == "营养不良"


def test_build_from_gaussian_dag_rejects_name_contract_violations() -> None:
    edges: list[dict[str, object]] = []
    with pytest.raises(ValueError, match="数量必须等于"):
        CausalGraph.build_from_gaussian_dag(edges, n_nodes=2, node_names=["低白蛋白"])
    with pytest.raises(ValueError, match="不能重复"):
        CausalGraph.build_from_gaussian_dag(edges, n_nodes=2, node_names=["同名", "同名"])


def test_resolve_node_supports_business_name_alias_and_index() -> None:
    graph = _named_graph()

    assert graph.resolve_node("干预") == "干预"
    assert graph.resolve_node("treatment") == "干预"
    assert graph.resolve_node(1) == "干预"
    assert graph.idx_to_name[2] == "结果"
    assert graph.node_index("outcome") == graph.node_index("V2") == 2


def test_unknown_node_fails_structured() -> None:
    graph = _named_graph()

    with pytest.raises(UnknownNodeError, match="unknown_node") as exc_info:
        graph.resolve_node("未知变量")
    assert exc_info.value.query == "未知变量"
    assert exc_info.value.kind == "unknown_node"


def test_do_calculus_accepts_business_name_after_discovery() -> None:
    graph = CausalGraph.build_from_gaussian_dag(
        [{"cause_idx": 0, "effect_idx": 1, "confidence": 0.9}],
        n_nodes=2,
        node_names=["低白蛋白", "营养不良"],
    )
    calculus = DoCalculus(graph=graph)
    result = calculus.estimate_ate("低白蛋白", "营养不良")

    assert result.target == "营养不良"
    assert result.method == "rejected"
    assert result.note == "association_graph_not_causal"


def test_do_calculus_rejects_unknown_node() -> None:
    graph = _named_graph()
    calculus = DoCalculus(graph=graph)

    result = calculus.estimate_ate("干预", "未知结果")

    assert result.method == "rejected"
    assert result.note.startswith("unknown_node")
    assert "未知结果" in result.note


def test_graph_serialization_preserves_name_mapping() -> None:
    graph = _named_graph()
    graph.adjacency[1, 2] = 0.7

    restored = CausalGraph.from_dict(graph.to_dict())

    assert restored.nodes == graph.nodes
    assert restored.node_aliases == graph.node_aliases
    assert restored.edge_mode == graph.edge_mode
    assert restored.dataset_id == graph.dataset_id
    assert np.array_equal(restored.adjacency, graph.adjacency)
    assert restored.resolve_node("V1") == "干预"
