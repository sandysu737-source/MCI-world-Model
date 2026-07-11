"""Sachs 真实数据校准器验证 — 用真实蛋白质网络数据验证 ConfidenceCalibrator 的价值。

数据: Sachs et al. (2005) 离散版 10000×11 (bnlearn)
Ground truth: 17 条有向因果边

方法:
    1. 从真实数据中计算变量对的统计显著性 (-log10 p-value) → 转化为 evidence confidence
    2. 对 ground truth 边 (正样本) 和随机非边 (负样本) 用 SDK 诊断
    3. 记录 raw confidence → 拟合校准器 → 对比校准前后 ECE/准确率

运行: pytest benchmarks/real_world/test_sachs_calibration.py -v -s
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.oracle, pytest.mark.realdata]

SACHS_DISCRETE = Path(__file__).parent / "sachs_data" / "sachs_discrete.txt"

SACHS_NODES = [f"X{i}" for i in range(1, 12)]
SACHS_REAL_EDGES = [
    ("X2", "X1"),
    ("X4", "X2"),
    ("X7", "X6"),
    ("X8", "X1"),
    ("X8", "X2"),
    ("X8", "X3"),
    ("X8", "X4"),
    ("X8", "X5"),
    ("X8", "X11"),
    ("X9", "X3"),
    ("X9", "X4"),
    ("X9", "X5"),
    ("X9", "X8"),
    ("X9", "X11"),
    ("X10", "X6"),
    ("X10", "X7"),
    ("X11", "X4"),
]


def _load_sachs() -> np.ndarray:
    if not SACHS_DISCRETE.exists():
        pytest.skip(f"Sachs 数据未找到: {SACHS_DISCRETE}")
    return np.loadtxt(SACHS_DISCRETE, skiprows=1)


def _compute_significance_matrix(data: np.ndarray) -> np.ndarray:
    """计算 n×n 的统计显著性分数 (-log10(p-value), capped at 300)。"""
    from scipy.stats import spearmanr

    n_vars = data.shape[1]
    sig = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            _, p = spearmanr(data[:, i], data[:, j])
            score = min(300.0, -np.log10(max(p, 1e-300)))
            sig[i, j] = score
            sig[j, i] = score
    return sig


def _significance_to_confidence(sig_score: float) -> float:
    """将 -log10(p-value) 映射为 evidence confidence [0.1, 0.95]。

    sig_score > 50 (p < 1e-50) -> confidence > 0.85 (强证据)
    sig_score 10-50 (p < 1e-10) -> confidence 0.65-0.85 (中等证据)
    sig_score < 5 (p > 1e-5) -> confidence < 0.4 (弱证据)
    """
    conf = 0.1 + 0.85 / (1.0 + np.exp(-(sig_score - 15) / 8))
    return float(min(0.95, max(0.1, conf)))


def _generate_diagnosis_cases(
    data: np.ndarray,
    nodes: list[str],
    true_edges: list[tuple[str, str]],
    n_negative: int = 50,
    seed: int = 42,
) -> list[dict]:
    """从真实数据生成诊断用例。

    正样本: ground truth 边 (17 条)
    负样本: 随机非边 (50 条)

    evidence confidence = 统计显著性 (-log10 p-value) 映射
    """
    rng = np.random.RandomState(seed)
    node_idx = {name: i for i, name in enumerate(nodes)}
    sig = _compute_significance_matrix(data)

    true_edge_set = set(true_edges)
    all_pairs = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(len(nodes)) if i != j]
    non_edges = [p for p in all_pairs if p not in true_edge_set and (p[1], p[0]) not in true_edge_set]

    neg_indices = rng.choice(len(non_edges), size=min(n_negative, len(non_edges)), replace=False)
    negative_sample = [non_edges[int(i)] for i in neg_indices]

    cases = []

    for cause, effect in true_edges:
        i, j = node_idx[cause], node_idx[effect]
        base_conf = _significance_to_confidence(sig[i, j])

        n_evidence = rng.randint(5, 10)
        evidence = []
        for k in range(n_evidence):
            sample_idx = rng.randint(0, data.shape[0])
            val_i = data[sample_idx, i]
            val_j = data[sample_idx, j]
            evidence.append(
                {
                    "id": f"ev_{cause}_{effect}_{k}",
                    "type": "observation",
                    "description": f"{cause}={val_i} {effect}={val_j}",
                    "confidence": max(0.0, min(1.0, base_conf + rng.normal(0, 0.05))),
                }
            )
        cases.append(
            {
                "cause": cause,
                "effect": effect,
                "evidence": evidence,
                "prior_strength": 0.5,
                "is_correct": True,
            }
        )

    for cause, effect in negative_sample:
        i, j = node_idx[cause], node_idx[effect]
        base_conf = _significance_to_confidence(sig[i, j])

        n_evidence = rng.randint(5, 10)
        evidence = []
        for k in range(n_evidence):
            sample_idx = rng.randint(0, data.shape[0])
            val_i = data[sample_idx, i]
            val_j = data[sample_idx, j]
            evidence.append(
                {
                    "id": f"ev_{cause}_{effect}_{k}",
                    "type": "observation",
                    "description": f"{cause}={val_i} {effect}={val_j}",
                    "confidence": max(0.0, min(1.0, base_conf + rng.normal(0, 0.05))),
                }
            )
        cases.append(
            {
                "cause": cause,
                "effect": effect,
                "evidence": evidence,
                "prior_strength": 0.5,
                "is_correct": False,
            }
        )

    return cases


def _run_diagnosis(cases: list[dict], calibrator=None) -> list[dict]:
    """执行诊断, 返回 [{confidence, is_conclusive, ground_truth}]。"""
    from mci_world_model.sdk._medical_causal_sdk import (
        ClinicalEvidence,
        MedicalCausalSDK,
    )

    results = []
    for case in cases:
        sdk = MedicalCausalSDK()
        if calibrator is not None:
            sdk.set_calibrator(calibrator)
        for ev in case["evidence"]:
            sdk.add_evidence(
                ClinicalEvidence(
                    evidence_id=ev["id"],
                    evidence_type=ev["type"],
                    description=ev["description"],
                    confidence=max(0.0, min(1.0, ev["confidence"])),
                )
            )
        diag = sdk.diagnose(case["cause"], case["effect"], case["prior_strength"])
        results.append(
            {
                "confidence": diag.confidence,
                "is_conclusive": diag.is_conclusive,
                "ground_truth": case["is_correct"],
            }
        )
    return results


def _compute_metrics(results: list[dict]) -> dict:
    """计算准确率和 ECE。"""
    n = len(results)
    if n == 0:
        return {"accuracy": 0, "ece": 0, "n": 0}

    correct = sum(1 for r in results if r["is_conclusive"] == r["ground_truth"])
    accuracy = correct / n

    confs = np.array([r["confidence"] for r in results])
    outcomes = np.array([1.0 if r["ground_truth"] else 0.0 for r in results])
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        mask = (confs >= bins[i]) & (confs < bins[i + 1] if i < 9 else confs <= bins[i + 1])
        n_bin = np.sum(mask)
        if n_bin > 0:
            ece += (n_bin / n) * abs(np.mean(confs[mask]) - np.mean(outcomes[mask]))

    return {"accuracy": round(accuracy, 4), "ece": round(float(ece), 4), "n": n}


class TestSachsCalibration:
    """用 Sachs 真实数据验证校准器价值。"""

    def test_sachs_raw_vs_calibrated(self):
        """Sachs 真实数据: 校准前后 ECE/准确率对比。"""
        data = _load_sachs()
        cases = _generate_diagnosis_cases(data, SACHS_NODES, SACHS_REAL_EDGES)

        # 分割: 前 50% 训练校准器, 后 50% 测试
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(cases))
        train_idx = indices[: len(cases) // 2]
        test_idx = indices[len(cases) // 2 :]

        train_cases = [cases[i] for i in train_idx]
        test_cases = [cases[i] for i in test_idx]

        # 1. 无校准基线
        results_raw = _run_diagnosis(test_cases, calibrator=None)
        metrics_raw = _compute_metrics(results_raw)

        # 2. 用训练集拟合校准器
        from mci_world_model.sdk._confidence_calibrator import ConfidenceCalibrator

        train_results = _run_diagnosis(train_cases, calibrator=None)
        cal_history = [(r["confidence"], r["ground_truth"]) for r in train_results]

        # 3. threshold_aware 校准
        cal_ta = ConfidenceCalibrator(method="threshold_aware")
        cal_ta.fit(cal_history)

        results_ta = _run_diagnosis(test_cases, calibrator=cal_ta)
        metrics_ta = _compute_metrics(results_ta)

        # 4. Platt 校准 (对照)
        cal_platt = ConfidenceCalibrator(method="platt")
        cal_platt.fit(cal_history)

        results_platt = _run_diagnosis(test_cases, calibrator=cal_platt)
        metrics_platt = _compute_metrics(results_platt)

        print("\n" + "=" * 65)
        print("  Sachs 真实数据校准器验证 (11 蛋白, 10000 观测)")
        print("=" * 65)
        print(f"{'方法':<22} {'准确率':<10} {'ECE':<10} {'测试数':<8}")
        print("-" * 65)
        print(f"{'A: 无校准 (raw)':<22} {metrics_raw['accuracy']:<10} {metrics_raw['ece']:<10} {metrics_raw['n']:<8}")
        print(f"{'B: Platt':<22} {metrics_platt['accuracy']:<10} {metrics_platt['ece']:<10} {metrics_platt['n']:<8}")
        print(f"{'C: threshold_aware':<22} {metrics_ta['accuracy']:<10} {metrics_ta['ece']:<10} {metrics_ta['n']:<8}")
        print("=" * 65)

        # 分析 confidence 分布
        train_confs_true = [r["confidence"] for r in train_results if r["ground_truth"]]
        train_confs_false = [r["confidence"] for r in train_results if not r["ground_truth"]]
        print(f"\n训练集 (n={len(train_results)}):")
        if train_confs_true:
            print(
                f"  真实边 confidence: mean={np.mean(train_confs_true):.3f}, "
                f"range=[{min(train_confs_true):.3f}, {max(train_confs_true):.3f}]"
            )
        if train_confs_false:
            print(
                f"  虚假边 confidence: mean={np.mean(train_confs_false):.3f}, "
                f"range=[{min(train_confs_false):.3f}, {max(train_confs_false):.3f}]"
            )

        # 验证: threshold_aware 不恶化
        assert metrics_ta["ece"] <= metrics_raw["ece"] + 0.03
        assert metrics_ta["accuracy"] >= metrics_raw["accuracy"] - 0.05

    def test_sachs_edge_level_analysis(self):
        """逐边分析: 真实边 vs 虚假边的 raw confidence 差异。"""
        data = _load_sachs()
        cases = _generate_diagnosis_cases(data, SACHS_NODES, SACHS_REAL_EDGES, n_negative=30)

        results = _run_diagnosis(cases, calibrator=None)

        true_confs = [r["confidence"] for r in results if r["ground_truth"]]
        false_confs = [r["confidence"] for r in results if not r["ground_truth"]]

        print(f"\n真实边 (n={len(true_confs)}): mean confidence = {np.mean(true_confs):.3f}")
        print(f"虚假边 (n={len(false_confs)}): mean confidence = {np.mean(false_confs):.3f}")
        print(f"差异: {np.mean(true_confs) - np.mean(false_confs):.3f}")

        # 真实边的平均 confidence 应高于虚假边
        assert np.mean(true_confs) > np.mean(false_confs), "真实边 confidence 未高于虚假边, 数据映射可能无效"

    def test_sachs_significance_separation(self):
        """验证: 统计显著性能有效区分真实边和虚假边。"""
        data = _load_sachs()
        sig = _compute_significance_matrix(data)
        node_idx = {n: i for i, n in enumerate(SACHS_NODES)}

        true_sigs = [sig[node_idx[c], node_idx[e]] for c, e in SACHS_REAL_EDGES]

        # 随机非边的显著性
        rng = np.random.RandomState(42)
        all_pairs = [(SACHS_NODES[i], SACHS_NODES[j]) for i in range(11) for j in range(11) if i != j]
        true_set = set(SACHS_REAL_EDGES)
        non_edges = [p for p in all_pairs if p not in true_set and (p[1], p[0]) not in true_set]
        neg_sample = [non_edges[int(i)] for i in rng.choice(len(non_edges), size=30, replace=False)]
        false_sigs = [sig[node_idx[c], node_idx[e]] for c, e in neg_sample]

        print(f"\n真实边 -log10(p): mean={np.mean(true_sigs):.1f}, median={np.median(true_sigs):.1f}")
        print(f"虚假边 -log10(p): mean={np.mean(false_sigs):.1f}, median={np.median(false_sigs):.1f}")
        print(f"区分度: {np.mean(true_sigs) - np.mean(false_sigs):.1f}")

        # 真实边的显著性应显著高于虚假边
        assert np.mean(true_sigs) > np.mean(false_sigs)
