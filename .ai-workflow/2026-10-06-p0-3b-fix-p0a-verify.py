"""P0-A 验收探针：聚类场景下评估器是否已改按患者整簇重采样。

口径与审计探针 8 一致（10 患者 × 4 窗口，H0：两模型误差差值的患者级效应均值为 0），
但 split 现在携带 window→patient 映射（P0-A 的新契约）。
期望：significant ≈ 患者级 block 参考（≤10%），CI 覆盖 0 ≈ ≥90%；
修复前窗口级口径的 36–41% 假阳性不再出现。
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


def mk_split(n_patients: int, per_patient: int) -> P03WindowSplit:
    n_windows = n_patients * per_patient
    inputs = np.tile(BASE, (n_windows, 4, 1))
    targets = np.tile(BASE, (n_windows, H, 1))
    stats = P03PipelineStats(
        n_rows=10 * n_windows,
        n_subjects=12,
        n_windows=n_windows,
        missing_ratio=0.0,
        imputed_ratio=0.0,
        out_of_range_count=0,
        duplicate_count=0,
        out_of_order_count=0,
        dropped_short_subjects=0,
        n_train_subjects=6,
        n_test_subjects=n_patients,
    )
    subjects = tuple(f"p{index // per_patient}" for index in range(n_windows))
    return P03WindowSplit(
        inputs,
        targets,
        inputs,
        targets,
        ("tr",),
        tuple(dict.fromkeys(subjects)),
        stats,
        subjects,
        subjects,
    )


def clustered_errors(rng: np.random.Generator, n_patients: int, per_patient: int):
    diff = np.zeros((n_patients * per_patient, H, N_V))
    for p in range(n_patients):
        effect = rng.normal(0.0, 0.8, (H, N_V))
        for w in range(per_patient):
            diff[p * per_patient + w] = effect + rng.normal(0.0, 0.2, (H, N_V))
    return 1.0 + diff / 2.0, 1.0 - diff / 2.0, diff


def window_level_significant(errors_a, errors_b, rng, resamples=600, alpha=0.05):
    """对照：旧的窗口级重采样（按窗口独立抽样）。"""
    d = np.abs(errors_b).mean(axis=(1, 2)) - np.abs(errors_a).mean(axis=(1, 2))
    n = d.shape[0]
    idx = rng.integers(0, n, size=(resamples, n))
    reps = d[idx].mean(axis=1)
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return bool(lo > 0 or hi < 0), float(lo), float(hi)


def block_level_significant(errors_a, errors_b, n_patients, per_patient, rng, resamples=600, alpha=0.05):
    """参考：患者级 block bootstrap（审计探针 8 的参考实现）。"""
    d = np.abs(errors_b).mean(axis=(1, 2)) - np.abs(errors_a).mean(axis=(1, 2))
    d = d.reshape(n_patients, per_patient)
    idx = rng.integers(0, n_patients, size=(resamples, n_patients))
    reps = d[idx].mean(axis=(1, 2))
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return bool(lo > 0 or hi < 0), float(lo), float(hi)


TRIALS = 200
N_PATIENTS, PER_PATIENT = 10, 4
RESAMPLES = 600
fpr_new = cover_new = 0
fpr_window = cover_window = 0
fpr_block = cover_block = 0
for t in range(TRIALS):
    rng = np.random.default_rng(90_000 + t)
    a, b, _diff = clustered_errors(rng, N_PATIENTS, PER_PATIENT)
    split = mk_split(N_PATIENTS, PER_PATIENT)
    rep = compare_models(ErrPred(a), ErrPred(b), split, seed=7, n_resamples=RESAMPLES, source="mimic")
    fpr_new += int(rep.statistical_test.significant)
    cover_new += int(rep.statistical_test.ci95_low <= 0.0 <= rep.statistical_test.ci95_high)
    sig_w, lo_w, hi_w = window_level_significant(a, b, np.random.default_rng(7), RESAMPLES)
    fpr_window += int(sig_w)
    cover_window += int(lo_w <= 0.0 <= hi_w)
    sig_b, lo_b, hi_b = block_level_significant(a, b, N_PATIENTS, PER_PATIENT, np.random.default_rng(7), RESAMPLES)
    fpr_block += int(sig_b)
    cover_block += int(lo_b <= 0.0 <= hi_b)

print(f"trials={TRIALS}  10 患者 × 4 窗口  H0（真实无差异）")
print(f"  评估器（P0-A 后，患者级）: significant={fpr_new / TRIALS:.2%}  CI覆盖0={cover_new / TRIALS:.2%}")
print(f"  对照·窗口级（修复前口径）: significant={fpr_window / TRIALS:.2%}  CI覆盖0={cover_window / TRIALS:.2%}")
print(f"  参考·患者级 block        : significant={fpr_block / TRIALS:.2%}  CI覆盖0={cover_block / TRIALS:.2%}")
