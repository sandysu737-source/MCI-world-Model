"""P0-3b 对抗性探针之二：样本量敏感性、聚类修复可行性与接口可得性。

> 2026-10-06 P0-3b-fix 环：评估器已强制 window→patient 映射（缺映射 fail-closed），
> 本探针的 `mk_split` 已补齐映射；探针8 用 per_patient=4 对齐"10 患者 × 4 窗口"。
"""

from __future__ import annotations

import numpy as np

from benchmarks.real_world.p03_metrics import compare_models
from benchmarks.real_world.p03_timeseries import VITAL_KEYS, P03PipelineStats, P03WindowSplit

N_V = len(VITAL_KEYS)
BASE = np.array([80.0, 120.0, 80.0, 98.0, 16.0, 36.8, 15.0])
H = 3


class ErrPred:
    def __init__(self, errors: np.ndarray) -> None:
        self.errors = errors

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return BASE - self.errors


def mk_split(n_windows: int, per_patient: int = 1) -> P03WindowSplit:
    """默认每窗口一个患者（窗口独立口径）；探针8 传 per_patient=4。"""
    inputs = np.tile(BASE, (n_windows, 4, 1))
    targets = np.tile(BASE, (n_windows, H, 1))
    subjects = tuple(f"p{index // per_patient}" for index in range(n_windows))
    n_patients = len(set(subjects))
    stats = P03PipelineStats(
        n_rows=10 * n_windows,
        n_subjects=n_patients,
        n_windows=n_windows,
        missing_ratio=0.0,
        imputed_ratio=0.0,
        out_of_range_count=0,
        duplicate_count=0,
        out_of_order_count=0,
        dropped_short_subjects=0,
        n_train_subjects=1,
        n_test_subjects=n_patients,
    )
    return P03WindowSplit(inputs, targets, inputs, targets, ("tr",), ("te",), stats, subjects, subjects)


def clustered_errors(rng: np.random.Generator, n_patients: int, per_patient: int):
    n = n_patients * per_patient
    diff = np.zeros((n, H, N_V))
    for p in range(n_patients):
        effect = rng.normal(0.0, 0.8, (H, N_V))
        for w in range(per_patient):
            diff[p * per_patient + w] = effect + rng.normal(0.0, 0.2, (H, N_V))
    return 1.0 + diff / 2.0, 1.0 - diff / 2.0, diff


def block_bootstrap_significant(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    diff: np.ndarray,
    n_patients: int,
    per_patient: int,
    rng: np.random.Generator,
    resamples: int = 600,
    alpha: float = 0.05,
) -> tuple[bool, float, float]:
    """患者级 block bootstrap（假设存在 window→patient 映射时本应采用的算法）。"""
    d = np.abs(errors_b).mean(axis=(1, 2)) - np.abs(errors_a).mean(axis=(1, 2))
    d = d.reshape(n_patients, per_patient)
    idx = rng.integers(0, n_patients, size=(resamples, n_patients))
    reps = d[idx].mean(axis=(1, 2))
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return bool(lo > 0 or hi < 0), float(lo), float(hi)


print("== 探针7：CI 覆盖率/FPR 随窗口数变化（窗口独立，600 次/档，resamples=600）==")
for n_windows in (5, 10, 20, 50):
    fpr = cover = 0
    trials = 600
    for t in range(trials):
        rng = np.random.default_rng(50_000 + t)
        a = np.abs(rng.normal(1.0, 0.6, (n_windows, H, N_V)))
        b = np.abs(rng.normal(1.0, 0.6, (n_windows, H, N_V)))
        rep = compare_models(ErrPred(a), ErrPred(b), mk_split(n_windows), seed=7, n_resamples=600, source="mimic")
        fpr += int(rep.statistical_test.significant)
        cover += int(rep.statistical_test.ci95_low <= 0.0 <= rep.statistical_test.ci95_high)
    print(f"  N={n_windows:3d}: significant={fpr / trials:.2%}（期望5%）CI覆盖0={cover / trials:.2%}（期望95%）")

print("== 探针8：聚类场景下 现有窗口级 vs 患者级 block bootstrap（10 患者×4 窗口，400 次）==")
trials = 400
fpr_window = fpr_block = cover_window = cover_block = 0
for t in range(trials):
    rng = np.random.default_rng(90_000 + t)
    a, b, diff = clustered_errors(rng, 10, 4)
    rep = compare_models(ErrPred(a), ErrPred(b), mk_split(40, per_patient=4), seed=7, n_resamples=600, source="mimic")
    fpr_window += int(rep.statistical_test.significant)
    cover_window += int(rep.statistical_test.ci95_low <= 0.0 <= rep.statistical_test.ci95_high)
    sig, lo, hi = block_bootstrap_significant(a, b, diff, 10, 4, np.random.default_rng(7))
    fpr_block += int(sig)
    cover_block += int(lo <= 0.0 <= hi)
print(f"  评估器（P0-A 后，患者级）: significant={fpr_window / trials:.2%} CI覆盖0={cover_window / trials:.2%}")
print(f"  患者级 block  : significant={fpr_block / trials:.2%} CI覆盖0={cover_block / trials:.2%}")

print("== 探针9：P03WindowSplit 是否暴露 window→patient 映射 ==")
print(f"  字段={list(P03WindowSplit.__dataclass_fields__)}")
