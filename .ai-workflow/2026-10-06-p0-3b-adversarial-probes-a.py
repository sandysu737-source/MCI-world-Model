"""P0-3b 评估器对抗性探针：试图证伪指标口径、统计口径与防伪声明。

只读脚本，不修改仓库；所有结论以本文件输出为准。

> 2026-10-06 P0-3b-fix 环：P0-A 之后评估器强制要求 window→patient 映射（缺映射 fail-closed），
> 本探针的 `mk_split` 已补齐映射（默认 per_patient=1，保持"窗口独立"的原始口径）。
"""

from __future__ import annotations

import json
import time
import tracemalloc

import numpy as np

from benchmarks.real_world.p03_metrics import (
    compare_models,
    evaluate_model,
)
from benchmarks.real_world.p03_timeseries import VITAL_KEYS, P03PipelineStats, P03WindowSplit

N_V = len(VITAL_KEYS)
BASE = np.array([80.0, 120.0, 80.0, 98.0, 16.0, 36.8, 15.0])
H = 3


def mk_split(n_windows: int, horizon: int = H, per_patient: int = 1) -> P03WindowSplit:
    """默认每窗口一个患者（per_patient=1）：患者级重采样与窗口级等价，保持原口径。"""
    inputs = np.tile(BASE, (n_windows, 4, 1))
    targets = np.tile(BASE, (n_windows, horizon, 1))
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
    return P03WindowSplit(inputs, targets, inputs, targets, ("tr-a",), ("te-a",), stats, subjects, subjects)


class ErrPred:
    """预测 = BASE - errors（保证落在生理范围内且误差恰为 errors）。"""

    def __init__(self, errors: np.ndarray) -> None:
        self.errors = errors

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return BASE - self.errors


def probe_h0_and_coverage(trials: int = 300, n_windows: int = 20, clustered: bool = False) -> tuple[int, int]:
    fpr = cover = 0
    for t in range(trials):
        rng = np.random.default_rng(10_000 + t)
        if not clustered:
            a = np.abs(rng.normal(1.0, 0.6, (n_windows, H, N_V)))
            b = np.abs(rng.normal(1.0, 0.6, (n_windows, H, N_V)))
        else:
            n_patients, per_patient = 10, n_windows // 10
            diff = np.zeros((n_windows, H, N_V))
            for p in range(n_patients):
                patient_effect = rng.normal(0.0, 0.8, (H, N_V))
                for w in range(per_patient):
                    diff[p * per_patient + w] = patient_effect + rng.normal(0.0, 0.2, (H, N_V))
            a = 1.0 + diff / 2.0
            b = 1.0 - diff / 2.0
        per_patient = n_windows // 10 if clustered else 1
        report = compare_models(
            ErrPred(a),
            ErrPred(b),
            mk_split(n_windows, per_patient=per_patient),
            seed=7,
            n_resamples=600,
            source="mimic",
        )
        test = report.statistical_test
        fpr += int(test.significant)
        cover += int(test.ci95_low <= 0.0 <= test.ci95_high)
    return fpr, cover


print("== 探针1：H0 误报率与 95% CI 覆盖率（窗口独立，N=20，300 次）==")
fpr, cover = probe_h0_and_coverage()
print(f"  significant 比例 = {fpr / 300:.3%}（期望≈5%）；CI 覆盖 0 的比例 = {cover / 300:.2%}（期望≈95%）")

print("== 探针2：窗口按患者聚类（10 患者 × 4 窗口 + 正确映射，患者级差异效应）==")
fpr_c, cover_c = probe_h0_and_coverage(trials=200, n_windows=40, clustered=True)
print(f"  significant 比例 = {fpr_c / 200:.3%}（期望≈5%）；CI 覆盖 0 的比例 = {cover_c / 200:.2%}（期望≈95%）")

print("== 探针3：整体 MAE 显著更优 ↔ 多数体征相对劣化 是否可同时成立 ==")
rng = np.random.default_rng(3)
base_err = np.array([3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0])
jepa_err = np.array([0.5, 0.5, 1.3, 1.3, 1.3, 1.3, 1.3])


def noise() -> np.ndarray:
    return 1.0 + 0.05 * rng.normal(size=(12, H, N_V))


base = base_err * noise()
jepa = jepa_err * noise()
report = compare_models(ErrPred(jepa), ErrPred(base), mk_split(12), seed=11, n_resamples=800, source="mimic")
payload = report.to_dict()
rel_base = payload["models"]["clinical_dynamics_baseline"]["relative_mae"]
rel_jepa = payload["models"]["jepa_clinical_bridge"]["relative_mae"]
worse = [k for k in VITAL_KEYS if rel_jepa[k] > rel_base[k]]
print(
    f"  status={report.status} interpretation={report.statistical_test.interpretation} "
    f"mean_diff={report.statistical_test.mean_difference:.4f} CI={payload['statistical_test']['ci95']}"
)
print(f"  JEPA 相对 MAE 更差的体征（{len(worse)}/{N_V}）：{worse}")
print(
    f"  raw overall MAE: baseline={np.mean([payload['models']['clinical_dynamics_baseline']['mae'][k] for k in VITAL_KEYS]):.4f} "
    f"jepa={np.mean([payload['models']['jepa_clinical_bridge']['mae'][k] for k in VITAL_KEYS]):.4f}"
)

print("== 探针4：source 只是字符串，合成数据 + source='mimic' 即可 passed ==")
syn = np.abs(np.random.default_rng(5).normal(1.0, 0.3, (16, H, N_V)))
good = np.abs(np.random.default_rng(6).normal(0.2, 0.1, (16, H, N_V)))
for src in ("mimic", "MIMIC", "degraded_synthetic", None):
    rep = compare_models(ErrPred(good), ErrPred(syn), mk_split(16), seed=5, n_resamples=400, source=src)
    print(f"  source={src!r:22} → status={rep.status}")

print("== 探针5：边界与鲁棒性 ==")
edge = []
try:
    m = evaluate_model(ErrPred(np.ones((5, 1, N_V))), mk_split(5, horizon=1), name="h1")
    edge.append(("horizon=1", f"ok status={m.status} mae_hr={m.to_dict()['mae']['hr']:.3f}"))
except Exception as exc:
    edge.append(("horizon=1", f"raised {type(exc).__name__}: {exc}"))
try:
    rep = compare_models(ErrPred(np.ones((1, H, N_V))), ErrPred(np.ones((1, H, N_V)) * 2), mk_split(1), seed=5)
    edge.append(
        (
            "N=1 配对",
            f"status={rep.status} significant={rep.statistical_test.significant} p={rep.to_dict()['statistical_test']['p_value']}",
        )
    )
except Exception as exc:
    edge.append(("N=1 配对", f"raised {type(exc).__name__}: {exc}"))
try:
    m = evaluate_model(ErrPred(np.ones((8, H, N_V))), mk_split(8), n_resamples=np.int64(100), name="npint")
    edge.append(("n_resamples=np.int64", f"ok status={m.status}"))
except Exception as exc:
    edge.append(("n_resamples=np.int64", f"raised {type(exc).__name__}"))
try:
    m = evaluate_model(ErrPred(np.ones((8, H, N_V))), mk_split(8), n_resamples=2, alpha=0.5, name="min")
    edge.append(("n_resamples=2, alpha=0.5", f"ok ci={m.to_dict()['ci95']['overall']['hr']}"))
except Exception as exc:
    edge.append(("n_resamples=2, alpha=0.5", f"raised {type(exc).__name__}"))
ro_inputs = np.tile(BASE, (8, 4, 1))
ro_inputs.flags.writeable = False
sp = mk_split(8)
sp_ro = P03WindowSplit(
    ro_inputs,
    sp.train_targets,
    ro_inputs,
    sp.test_targets,
    sp.train_subject_ids,
    sp.test_subject_ids,
    sp.stats,
    sp.train_window_subjects,
    sp.test_window_subjects,
)
try:
    before = ro_inputs.copy()
    payload = evaluate_model(ErrPred(np.ones((8, H, N_V))), sp_ro, name="ro").to_dict()
    edge.append(("只读输入 + 是否篡改输入", f"ok 未改动={np.array_equal(before, ro_inputs)}"))
except Exception as exc:
    edge.append(("只读输入", f"raised {type(exc).__name__}"))
try:
    m = evaluate_model(ErrPred(np.ones((8, H, N_V), dtype=np.float32)), mk_split(8), name="f32")
    edge.append(("float32 输入/预测", f"ok status={m.status}"))
except Exception as exc:
    edge.append(("float32 输入/预测", f"raised {type(exc).__name__}: {exc}"))
try:
    json.dumps(payload, allow_nan=False)
    edge.append(("报告 JSON 严格可序列化", "ok（无 NaN/Inf 字面量）"))
except Exception as exc:
    edge.append(("报告 JSON 严格可序列化", f"raised {type(exc).__name__}"))
tiny = mk_split(8)
tiny_targets = np.full((8, H, N_V), 1e-8)
tiny_split = P03WindowSplit(
    tiny.train_inputs,
    tiny.train_targets,
    tiny.test_inputs,
    tiny_targets,
    tiny.train_subject_ids,
    tiny.test_subject_ids,
    tiny.stats,
    tiny.train_window_subjects,
    tiny.test_window_subjects,
)
m = evaluate_model(ErrPred(np.ones((8, H, N_V))), tiny_split, name="tiny")
edge.append(("目标量级 1e-8 的 relative_mae", f"hr={m.to_dict()['relative_mae']['hr']}"))
for name, result in edge:
    print(f"  {name}: {result}")

print("== 探针6：大窗口量内存/耗时（N=100000, n_resamples=2000）==")
# 大窗口量探针：每患者 200 窗口（500 患者），避免 10 万"患者"把簇重采样变成压力测试
big = mk_split(100_000, per_patient=200)
errors = np.ones((100_000, H, N_V))
tracemalloc.start()
start = time.perf_counter()
metric = evaluate_model(ErrPred(errors), big, seed=1, n_resamples=2000, name="big")
elapsed = time.perf_counter() - start
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"  status={metric.status} 用时={elapsed:.2f}s tracemalloc 峰值={peak / 1e6:.1f}MB")
