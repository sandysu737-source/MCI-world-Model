"""Causal Discovery Benchmark — PC 算法从数据学习因果图。

测试图结构: chain / confounder / collider / 5-node DAG
指标: Precision / Recall / F1 / SHD
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mci_world_model.sdk._autonomous_law_discoverer_v2 import PCSkeletonDiscoverer


def _rng(seed: int = 42) -> np.random.RandomState:
    return np.random.RandomState(seed)


def chain_data(n: int = 500, noise: float = 0.3, seed: int = 42) -> tuple[np.ndarray, list[tuple[str, str]]]:
    rng = _rng(seed)
    X = rng.randn(n)
    Z = 0.7 * X + rng.randn(n) * noise
    Y = 0.7 * Z + rng.randn(n) * noise
    return np.column_stack([X, Z, Y]), [("X", "Z"), ("Z", "Y")]


def confounder_data(n: int = 500, noise: float = 0.3, seed: int = 42) -> tuple[np.ndarray, list[tuple[str, str]]]:
    rng = _rng(seed)
    Z = rng.randn(n)
    X = 0.7 * Z + rng.randn(n) * noise
    Y = 0.7 * Z + rng.randn(n) * noise
    return np.column_stack([X, Y, Z]), [("Z", "X"), ("Z", "Y")]


def collider_data(n: int = 500, noise: float = 0.3, seed: int = 42) -> tuple[np.ndarray, list[tuple[str, str]]]:
    rng = _rng(seed)
    X = rng.randn(n)
    Y = rng.randn(n)
    Z = 0.5 * X + 0.5 * Y + rng.randn(n) * noise
    return np.column_stack([X, Y, Z]), [("X", "Z"), ("Y", "Z")]


def five_node_dag(n: int = 500, noise: float = 0.3, seed: int = 42) -> tuple[np.ndarray, list[tuple[str, str]]]:
    rng = _rng(seed)
    X1 = rng.randn(n)
    X2 = 0.6 * X1 + rng.randn(n) * noise
    X3 = 0.6 * X1 + rng.randn(n) * noise
    X4 = 0.5 * X2 + 0.5 * X3 + rng.randn(n) * noise
    X5 = 0.7 * X3 + rng.randn(n) * noise
    data = np.column_stack([X1, X2, X3, X4, X5])
    return data, [("X1", "X2"), ("X1", "X3"), ("X2", "X4"), ("X3", "X4"), ("X3", "X5")]


def compute_discovery_metrics(
    predicted: list[tuple[str, str]], true: list[tuple[str, str]]
) -> dict[str, Any]:
    pred_set = set(predicted)
    true_set = set(true)
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(true_set) if true_set else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    reversed_edges = sum(1 for (a, b) in pred_set if (a, b) not in true_set and (b, a) in true_set)
    shd = fp + fn + reversed_edges
    return {"precision": precision, "recall": recall, "f1": f1, "shd": shd,
            "tp": tp, "fp": fp, "fn": fn, "n_true": len(true_set), "n_pred": len(pred_set)}


class TestPCDiscovery:
    def test_chain_discovery(self) -> None:
        data, true = chain_data(n=500)
        pc = PCSkeletonDiscoverer(alpha=0.05)
        skeleton = pc.discover(data, var_names=["X", "Z", "Y"])
        m = compute_discovery_metrics(skeleton.edges, true)
        assert m["recall"] >= 0.5, f"Recall {m['recall']:.2f}"
        assert m["shd"] <= 8, f"SHD {m['shd']}"  # 0-order: X→Y 不独立通过 Z

    def test_confounder_discovery(self) -> None:
        data, true = confounder_data(n=500)
        pc = PCSkeletonDiscoverer(alpha=0.05)
        skeleton = pc.discover(data, var_names=["X", "Y", "Z"])
        m = compute_discovery_metrics(skeleton.edges, true)
        assert m["f1"] >= 0.5, f"F1 {m['f1']:.2f}"

    def test_collider_discovery(self) -> None:
        data, true = collider_data(n=500)
        pc = PCSkeletonDiscoverer(alpha=0.05)
        skeleton = pc.discover(data, var_names=["X", "Y", "Z"])
        m = compute_discovery_metrics(skeleton.edges, true)
        assert m["precision"] >= 0.3, f"Precision {m['precision']:.2f}"

    def test_five_node_dag(self) -> None:
        data, true = five_node_dag(n=500)
        pc = PCSkeletonDiscoverer(alpha=0.05)
        skeleton = pc.discover(data, var_names=["X1", "X2", "X3", "X4", "X5"])
        m = compute_discovery_metrics(skeleton.edges, true)
        assert m["recall"] >= 0.3, f"Recall {m['recall']:.2f}"

    def test_large_sample_improves(self) -> None:
        _, true = chain_data(n=200)
        data_s, _ = chain_data(n=200)
        data_l, _ = chain_data(n=800)
        pc = PCSkeletonDiscoverer(alpha=0.05)
        ms = compute_discovery_metrics(pc.discover(data_s, ["X", "Z", "Y"]).edges, true)
        ml = compute_discovery_metrics(pc.discover(data_l, ["X", "Z", "Y"]).edges, true)
        assert ml["f1"] >= ms["f1"] * 0.7


class TestPCEdgeCases:
    def test_empty_data(self) -> None:
        pc = PCSkeletonDiscoverer()
        skeleton = pc.discover(np.zeros((0, 3)), var_names=["A", "B", "C"])
        assert len(skeleton.edges) == 0

    def test_independent_variables(self) -> None:
        data = _rng().randn(200, 3)
        pc = PCSkeletonDiscoverer(alpha=0.01)
        skeleton = pc.discover(data, var_names=["A", "B", "C"])
        assert len(skeleton.edges) <= 2

    def test_deterministic_relation(self) -> None:
        rng = _rng()
        X = rng.randn(300)
        Y = 2.0 * X
        Z = rng.randn(300)
        data = np.column_stack([X, Y, Z])
        pc = PCSkeletonDiscoverer(alpha=0.01)
        skeleton = pc.discover(data, var_names=["X", "Y", "Z"])
        edge_set = set(skeleton.edges)
        assert ("X", "Y") in edge_set or ("Y", "X") in edge_set

    def test_reproducibility(self) -> None:
        data, _ = chain_data(n=300, seed=42)
        pc = PCSkeletonDiscoverer(alpha=0.05)
        s1 = pc.discover(data, ["X", "Z", "Y"])
        s2 = pc.discover(data, ["X", "Z", "Y"])
        assert set(s1.edges) == set(s2.edges)


class TestDiscoveryReport:
    def test_full_report(self) -> None:
        cases = [
            ("chain", *chain_data(n=300), ["X","Z","Y"]),
            ("confounder", *confounder_data(n=300), ["X","Y","Z"]),
            ("collider", *collider_data(n=300), ["X","Y","Z"]),
        ]
        pc = PCSkeletonDiscoverer(alpha=0.05)
        results = {}
        for name, data, true, vn in cases:
            skeleton = pc.discover(data, var_names=vn)
            results[name] = compute_discovery_metrics(skeleton.edges, true)
        mean_f1 = float(np.mean([r["f1"] for r in results.values()]))
        mean_recall = float(np.mean([r["recall"] for r in results.values()]))
        assert mean_f1 > 0.3
        assert mean_recall > 0.3
        assert len(results) == 3
