"""P2-D 验收探针：N=100k 窗口、2000 次重采样下 bootstrap 的时延与峰值内存。

旧口径（分块只约束索引矩阵）实测 8.2–8.9s / 峰值 360MB（审计探针 6）；
新口径按 gather 元素量分块后要求 < 5s 且峰值 < 120MB。
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np

from benchmarks.real_world.p03_metrics import evaluate_model
from benchmarks.real_world.p03_timeseries import VITAL_KEYS, P03PipelineStats, P03WindowSplit

N_V = len(VITAL_KEYS)
BASE = np.array([80.0, 120.0, 80.0, 98.0, 16.0, 36.8, 15.0])
N_WINDOWS = 100_000
N_PATIENTS = 5_000
HISTORY = 4
HORIZON = 3
RESAMPLES = 2000


class OffsetPredictor:
    def __init__(self, offset: float) -> None:
        self.offset = offset

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return np.full((inputs.shape[0], HORIZON, N_V), self.offset) + BASE


def build_split() -> P03WindowSplit:
    rng = np.random.default_rng(2026)
    inputs = np.tile(BASE, (N_WINDOWS, HISTORY, 1))
    targets = BASE + rng.normal(0.0, 0.5, size=(N_WINDOWS, HORIZON, N_V))
    subjects = tuple(f"p{index % N_PATIENTS}" for index in range(N_WINDOWS))
    stats = P03PipelineStats(
        n_rows=10 * N_WINDOWS,
        n_subjects=N_PATIENTS,
        n_windows=N_WINDOWS,
        missing_ratio=0.0,
        imputed_ratio=0.0,
        out_of_range_count=0,
        duplicate_count=0,
        out_of_order_count=0,
        dropped_short_subjects=0,
        n_train_subjects=1,
        n_test_subjects=N_PATIENTS,
    )
    return P03WindowSplit(inputs, targets, inputs, targets, ("tr",), ("te",), stats, subjects, subjects)


split = build_split()
tracemalloc.start()
start = time.perf_counter()
metrics = evaluate_model(OffsetPredictor(0.25), split, seed=42, n_resamples=RESAMPLES, name="p2d")
elapsed = time.perf_counter() - start
_current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"N={N_WINDOWS} 患者={N_PATIENTS} resamples={RESAMPLES}")
print(f"  用时      : {elapsed:.2f}s  (门槛 < 5s)")
print(f"  峰值内存  : {peak / 1024 / 1024:.1f}MB  (门槛 < 120MB)")
print(f"  指标健全性: status={metrics.status} n_windows={metrics.n_windows} mae_hr={metrics.overall_mae[0]:.4f}")
assert elapsed < 5.0, f"P2-D 时延超预算: {elapsed:.2f}s"
assert peak < 120 * 1024 * 1024, f"P2-D 内存超预算: {peak / 1024 / 1024:.1f}MB"
print("P2-D 验收通过")
