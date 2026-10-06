"""P0-3b 评估器回归：指标口径、fail-closed、bootstrap 与安全约束。"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from benchmarks.real_world.p03_metrics import (
    DEFAULT_N_RESAMPLES,
    DEFAULT_SEED,
    INTERPRETATION_BASELINE_BETTER,
    INTERPRETATION_JEPA_BETTER,
    INTERPRETATION_NO_DIFFERENCE,
    MODEL_STATUS_EVALUATED,
    MODEL_STATUS_FAILED,
    MODEL_STATUS_NOT_RUN,
    RELATIVE_MAE_MIN_SCALE_FRACTION,
    REPORT_STATUS_DEGRADED,
    REPORT_STATUS_FAILED,
    REPORT_STATUS_NOT_RUN,
    REPORT_STATUS_PASSED,
    VITAL_ERROR_SCALES,
    P03DatasetManifest,
    P03MetricsError,
    _relative_mae,
    compare_models,
    evaluate_model,
)
from benchmarks.real_world.p03_timeseries import (
    N_VITAL_KEYS,
    VITAL_KEYS,
    P03PipelineStats,
    P03WindowSplit,
)

BASE = np.array([80.0, 120.0, 80.0, 98.0, 16.0, 36.8, 15.0])
HISTORY = 4
HORIZON = 3


def _stats(**overrides: int) -> P03PipelineStats:
    payload = {
        "n_rows": 400,
        "n_subjects": 6,
        "n_windows": 16,
        "missing_ratio": 0.01,
        "imputed_ratio": 0.01,
        "out_of_range_count": 0,
        "duplicate_count": 0,
        "out_of_order_count": 0,
        "dropped_short_subjects": 0,
        "n_train_subjects": 4,
        "n_test_subjects": 2,
    }
    payload.update(overrides)
    return P03PipelineStats(**payload)


def _split(n_windows: int = 16, *, seed: int = 7) -> P03WindowSplit:
    """构造测试窗口：末步观测为 BASE，目标在 BASE 附近随机波动。

    P0-A：窗口按患者成块划分（与管道产出口径一致），并附上 window→patient 映射；
    评估器按患者整簇重采样，缺映射会 fail-closed。
    """
    rng = np.random.default_rng(seed)
    inputs = np.tile(BASE, (n_windows, HISTORY, 1))
    targets = BASE + rng.normal(0.0, 0.5, size=(n_windows, HORIZON, N_VITAL_KEYS))
    n_train = n_windows // 2
    train_half = max(1, n_train // 2)
    test_half = max(1, n_windows // 2)
    return P03WindowSplit(
        train_inputs=inputs[:n_train],
        train_targets=targets[:n_train],
        test_inputs=inputs,
        test_targets=targets,
        train_subject_ids=("train-a", "train-b"),
        test_subject_ids=("test-a", "test-b"),
        stats=_stats(n_windows=n_windows),
        train_window_subjects=tuple("train-a" if i < train_half else "train-b" for i in range(n_train)),
        test_window_subjects=tuple("test-a" if i < test_half else "test-b" for i in range(n_windows)),
    )


def _manifest(**overrides: object) -> P03DatasetManifest:
    """构造合法 manifest（P0-B）；overrides 用于单字段破坏。"""
    payload: dict[str, object] = {
        "version": "mimic-iv-demo-v1.0",
        "sha256": "a" * 64,
        "authorization": "IRB-WAIVER-2026-001",
        "n_subjects": 100,
        "source": "mimic",
    }
    payload.update(overrides)
    return P03DatasetManifest(**payload)  # type: ignore[arg-type]


def _passing_split() -> P03WindowSplit:
    """满足 P2-E 门槛的切分：6 患者 × 4 窗口 = 24 测试窗口。"""
    return _split_with_patient_mapping(6, 4)


def _empty_split() -> P03WindowSplit:
    empty = np.zeros((0, HISTORY, N_VITAL_KEYS))
    return P03WindowSplit(empty, empty.copy(), empty.copy(), empty.copy(), (), (), _stats(n_windows=0))


class _OffsetPredictor:
    """返回 ``BASE + offset``，误差随窗口目标波动而变化。"""

    def __init__(self, offset: float) -> None:
        self.offset = offset

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return np.full((inputs.shape[0], HORIZON, N_VITAL_KEYS), self.offset) + BASE


class _TargetPredictor:
    """完美预测器：直接返回给定目标。"""

    def __init__(self, targets: np.ndarray) -> None:
        self.targets = targets

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.targets


class _BrokenPredictor:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        if self.mode == "nan":
            return np.full((inputs.shape[0], HORIZON, N_VITAL_KEYS), np.nan)
        return np.zeros((inputs.shape[0], HORIZON - 1, N_VITAL_KEYS))


class _NoPredictor:
    pass


def test_perfect_predictor_scores_zero_error() -> None:
    split = _split()
    metrics = evaluate_model(_TargetPredictor(split.test_targets), split, name="perfect")
    payload = metrics.to_dict()

    assert metrics.status == MODEL_STATUS_EVALUATED
    assert metrics.overall_mae == pytest.approx(np.zeros(N_VITAL_KEYS))
    assert payload["mae"]["hr"] == pytest.approx(0.0)
    assert payload["relative_mae"]["gcs"] == pytest.approx(0.0)
    assert payload["direction_accuracy"]["spo2"] == pytest.approx(1.0)
    assert payload["constraint_violation_rate"]["rr"] == pytest.approx(0.0)
    assert payload["ci95"]["overall"]["hr"] == pytest.approx([0.0, 0.0])


def test_constant_offset_matches_analytic_mae() -> None:
    split = _split()
    metrics = evaluate_model(_OffsetPredictor(2.0), split, name="offset")
    analytic_error = np.abs(split.test_targets - (BASE + 2.0))
    expected_overall = analytic_error.mean(axis=(0, 1))
    expected_relative = expected_overall / np.abs(split.test_targets).mean(axis=(0, 1))
    expected_per_step = analytic_error.mean(axis=0)

    assert metrics.overall_mae == pytest.approx(expected_overall)
    assert metrics.overall_relative_mae == pytest.approx(expected_relative)
    assert metrics.mae_per_step == pytest.approx(expected_per_step)
    assert np.isfinite(metrics.overall_mae).all()


def test_direction_accuracy_excludes_flat_targets() -> None:
    split = _split()
    targets = split.test_targets.copy()
    targets[:, :, 0] = BASE[0]
    flat = P03WindowSplit(
        split.train_inputs,
        split.train_targets,
        split.test_inputs,
        targets,
        split.train_subject_ids,
        split.test_subject_ids,
        split.stats,
        split.train_window_subjects,
        split.test_window_subjects,
    )
    payload = evaluate_model(_TargetPredictor(targets), flat, name="flat").to_dict()

    assert payload["direction_accuracy"]["hr"] is None
    assert payload["direction_accuracy_per_step"]["hr"] == [None, None, None]
    assert payload["direction_accuracy"]["sbp"] == pytest.approx(1.0)


def test_constraint_violation_rate_flags_out_of_range_predictions() -> None:
    split = _split()
    predictions = split.test_targets.copy()
    predictions[:8, :, 0] = 300.0

    payload = evaluate_model(_TargetPredictor(predictions), split, name="oos").to_dict()

    assert payload["constraint_violation_rate"]["hr"] == pytest.approx(0.5)
    assert payload["constraint_violation_rate_per_step"]["hr"] == pytest.approx([0.5, 0.5, 0.5])
    assert payload["constraint_violation_rate"]["sbp"] == pytest.approx(0.0)


def test_relative_mae_is_null_when_target_is_flat_zero() -> None:
    split = _split()
    zeros = np.zeros((split.test_targets.shape[0], HORIZON, N_VITAL_KEYS))
    payload = evaluate_model(_OffsetPredictor(1.0), _with_targets(split, zeros), name="zero").to_dict()

    assert payload["relative_mae"]["hr"] is None
    assert payload["relative_mae_per_step"]["hr"] == [None, None, None]
    assert payload["mae"]["hr"] == pytest.approx(float(BASE[0] + 1.0))


def _with_targets(split: P03WindowSplit, targets: np.ndarray) -> P03WindowSplit:
    return P03WindowSplit(
        split.train_inputs,
        split.train_targets,
        split.test_inputs,
        targets,
        split.train_subject_ids,
        split.test_subject_ids,
        split.stats,
        split.train_window_subjects,
        split.test_window_subjects,
    )


def test_non_finite_prediction_fails_closed() -> None:
    payload = evaluate_model(_BrokenPredictor("nan"), _split(), name="nan").to_dict()

    assert payload["status"] == MODEL_STATUS_FAILED
    assert payload["failure_reason"]
    assert payload["mae"]["hr"] is None
    assert payload["direction_accuracy"]["hr"] is None


@pytest.mark.parametrize("model", [_BrokenPredictor("shape"), _NoPredictor()])
def test_contract_violations_raise(model: object) -> None:
    with pytest.raises(P03MetricsError):
        evaluate_model(model, _split(), name="broken")


@pytest.mark.parametrize("kwargs", [{"n_resamples": 1}, {"alpha": 1.5}, {"name": " "}])
def test_option_validation_raises(kwargs: dict[str, object]) -> None:
    with pytest.raises(P03MetricsError):
        evaluate_model(_OffsetPredictor(0.5), _split(), **kwargs)  # type: ignore[arg-type]


def test_empty_split_reports_not_run() -> None:
    payload = evaluate_model(_OffsetPredictor(0.5), _empty_split(), name="empty").to_dict()

    assert payload["status"] == MODEL_STATUS_NOT_RUN
    assert payload["n_windows"] == 0
    assert payload["mae"]["hr"] is None
    assert any("测试窗口为 0" in item for item in payload["limitations"])


def test_bootstrap_is_reproducible_and_seed_sensitive() -> None:
    # P0-A：重采样单位是患者，患者数过少（如 2 个）时百分位 CI 退化、对种子不敏感，
    # 故这里用 6 个测试患者；"结果对种子敏感"的前提是簇数足够。
    split = _split_with_patient_mapping(6, 4)
    first = evaluate_model(_OffsetPredictor(0.4), split, seed=11, n_resamples=200, name="a")
    second = evaluate_model(_OffsetPredictor(0.4), split, seed=11, n_resamples=200, name="a")
    other = evaluate_model(_OffsetPredictor(0.4), split, seed=12, n_resamples=200, name="a")

    assert np.array_equal(first.mae_ci95_low, second.mae_ci95_low)
    assert np.array_equal(first.mae_ci95_high, second.mae_ci95_high)
    assert not np.array_equal(first.mae_ci95_low, other.mae_ci95_low)
    assert np.all(first.mae_ci95_low < first.overall_mae) or np.all(first.mae_ci95_high > first.overall_mae)


def test_compare_models_detects_better_model() -> None:
    split = _passing_split()
    report = compare_models(
        _OffsetPredictor(0.1),
        _OffsetPredictor(2.0),
        split,
        seed=5,
        n_resamples=500,
        source="mimic",
        dataset_manifest=_manifest(),
    )
    payload = report.to_dict()

    assert report.status == REPORT_STATUS_PASSED
    assert report.statistical_test.interpretation == INTERPRETATION_JEPA_BETTER
    assert report.statistical_test.significant is True
    assert report.statistical_test.mean_difference > 0
    assert payload["statistical_test"]["ci95"][0] > 0
    assert set(payload["statistical_test"]["per_vital"]) == set(VITAL_KEYS)
    assert payload["split"] == {
        "strategy": "patient_holdout",
        "n_train_subjects": 2,
        "n_test_subjects": 6,
        "n_train_windows": 12,
        "n_test_windows": 24,
        "n_test_patients": 6,
    }


def test_compare_models_reversed_favours_baseline() -> None:
    report = compare_models(_OffsetPredictor(2.0), _OffsetPredictor(0.1), _split(), seed=5, n_resamples=500)

    assert report.status == REPORT_STATUS_FAILED
    assert report.statistical_test.interpretation == INTERPRETATION_BASELINE_BETTER
    assert report.statistical_test.mean_difference < 0


def test_compare_models_identical_predictors_are_not_significant() -> None:
    split = _split()
    report = compare_models(
        _TargetPredictor(split.test_targets),
        _TargetPredictor(split.test_targets),
        split,
        seed=5,
        n_resamples=200,
    )

    assert report.status == REPORT_STATUS_FAILED
    assert report.statistical_test.interpretation == INTERPRETATION_NO_DIFFERENCE
    assert report.statistical_test.significant is False
    assert report.statistical_test.mean_difference == pytest.approx(0.0)


def test_unknown_source_cannot_pass() -> None:
    report = compare_models(_OffsetPredictor(0.1), _OffsetPredictor(2.0), _split(), seed=5, n_resamples=300)

    assert report.statistical_test.interpretation == INTERPRETATION_JEPA_BETTER
    assert report.status == REPORT_STATUS_FAILED
    assert any("source 不是 mimic" in item for item in report.limitations)


def test_degraded_source_never_reports_passed() -> None:
    report = compare_models(
        _OffsetPredictor(0.1),
        _OffsetPredictor(2.0),
        _split(),
        seed=5,
        n_resamples=300,
        source="degraded_synthetic",
    )

    assert report.status == REPORT_STATUS_DEGRADED
    assert report.status != REPORT_STATUS_PASSED


def test_report_without_windows_is_not_run() -> None:
    report = compare_models(_OffsetPredictor(0.1), _OffsetPredictor(2.0), _empty_split(), seed=5)

    assert report.status == REPORT_STATUS_NOT_RUN
    assert report.statistical_test.significant is False
    assert report.to_dict()["statistical_test"]["p_value"] is None


def test_report_leaks_no_patient_identifiers() -> None:
    split = _split()
    payload = compare_models(_OffsetPredictor(0.1), _OffsetPredictor(2.0), split, seed=5, n_resamples=200).to_dict()
    serialized = json.dumps(payload, ensure_ascii=False)

    assert set(payload) == {
        "status",
        "metric",
        "n_deteriorated_vitals",
        "source",
        "dataset_manifest",
        "split",
        "models",
        "statistical_test",
        "seed",
        "n_resamples",
        "alpha",
        "limitations",
    }
    for identifier in (*split.test_subject_ids, *split.train_subject_ids):
        assert identifier not in serialized
    assert "charttime" not in serialized
    assert set(payload["models"]) == {"jepa_clinical_bridge", "clinical_dynamics_baseline"}


# ---- P0-A（OODA-20261006-043）：结论必须绑定独立单元（患者），窗口级重采样不得再用 ----


class _WindowOffsetPredictor:
    """按窗口返回固定预测值，用于构造"簇内完全相关"的误差结构。"""

    def __init__(self, window_values: np.ndarray) -> None:
        self.window_values = window_values

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        base = np.zeros((inputs.shape[0], HORIZON, N_VITAL_KEYS), dtype=np.float64)
        return base + self.window_values[:, None, None]


def _split_with_patient_mapping(
    n_patients: int, per_patient: int, *, targets: np.ndarray | None = None
) -> P03WindowSplit:
    """按"每患者 per_patient 个窗口"重排测试映射（可替换目标以构造特定误差结构）。"""
    n_windows = n_patients * per_patient
    split = _split(n_windows=n_windows)
    subjects = tuple(f"p{index // per_patient}" for index in range(n_windows))
    return P03WindowSplit(
        split.train_inputs,
        split.train_targets,
        split.test_inputs,
        split.test_targets if targets is None else targets,
        split.train_subject_ids,
        tuple(dict.fromkeys(subjects)),
        _stats(n_windows=n_windows, n_subjects=n_patients + 2, n_train_subjects=2, n_test_subjects=n_patients),
        split.train_window_subjects,
        subjects,
    )


def _clustered_split(n_patients: int, per_patient: int) -> P03WindowSplit:
    """患者内误差完全相关：p0/p1 误差 0，p2/p3 误差 10。"""
    return _split_with_patient_mapping(
        n_patients, per_patient, targets=np.zeros((n_patients * per_patient, HORIZON, N_VITAL_KEYS))
    )


def test_bootstrap_resamples_patients_not_windows() -> None:
    """患者内误差完全相关时 CI 必须反映簇间方差（窗口级 bootstrap 会窄约 2 倍）。"""
    offsets = np.repeat(np.array([0.0, 0.0, 10.0, 10.0]), 5)
    split = _clustered_split(n_patients=4, per_patient=5)

    metrics = evaluate_model(_WindowOffsetPredictor(offsets), split, name="clustered", seed=11, n_resamples=2000)

    width = float(metrics.mae_ci95_high.mean() - metrics.mae_ci95_low.mean())
    # 簇间 SD=5、簇数=4 → 患者级宽度 ≈ 2*1.96*5/2 ≈ 9.8；按 20 个"独立"窗口只有 ≈ 4.4
    assert width > 7.0


def test_split_without_window_patient_mapping_fails_closed() -> None:
    split = _split()
    legacy = P03WindowSplit(
        split.train_inputs,
        split.train_targets,
        split.test_inputs,
        split.test_targets,
        split.train_subject_ids,
        split.test_subject_ids,
        split.stats,
    )

    with pytest.raises(P03MetricsError, match="test_window_subjects"):
        evaluate_model(_OffsetPredictor(0.5), legacy, name="legacy")


def test_split_with_mismatched_mapping_fails_closed() -> None:
    split = _split()
    bad = P03WindowSplit(
        split.train_inputs,
        split.train_targets,
        split.test_inputs,
        split.test_targets,
        split.train_subject_ids,
        split.test_subject_ids,
        split.stats,
        split.train_window_subjects,
        split.test_window_subjects[:-1],
    )

    with pytest.raises(P03MetricsError, match="不一致"):
        evaluate_model(_OffsetPredictor(0.5), bad, name="bad-mapping")


# ---- P0-B / P1-C / P2-E / P2-F / P3-G：P0-3b-fix 环（对抗性审计整改） ----


def test_manifest_binding_is_required_for_passed() -> None:
    """P0-B：JEPA 显著更优但没有 manifest 绑定时 passed 不可达。"""
    report = compare_models(
        _OffsetPredictor(0.1),
        _OffsetPredictor(2.0),
        _passing_split(),
        seed=5,
        n_resamples=500,
        source="mimic",
    )

    assert report.statistical_test.interpretation == INTERPRETATION_JEPA_BETTER
    assert report.status == REPORT_STATUS_NOT_RUN
    assert report.dataset_manifest is None
    assert report.to_dict()["dataset_manifest"] is None
    assert any("dataset_manifest" in item for item in report.limitations)


def test_valid_manifest_binding_allows_passed() -> None:
    report = compare_models(
        _OffsetPredictor(0.1),
        _OffsetPredictor(2.0),
        _passing_split(),
        seed=5,
        n_resamples=500,
        source="mimic",
        dataset_manifest=_manifest(),
    )
    payload = report.to_dict()

    assert report.status == REPORT_STATUS_PASSED
    assert report.n_deteriorated_vitals == 0
    assert payload["metric"] == "scale_normalized_mae"
    assert payload["dataset_manifest"] == {
        "version": "mimic-iv-demo-v1.0",
        "sha256": "a" * 64,
        "authorization": "IRB-WAIVER-2026-001",
        "n_subjects": 100,
        "source": "mimic",
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"sha256": "not-a-hash"},
        {"authorization": " "},
        {"n_subjects": 2},
        {"source": "degraded_synthetic"},
    ],
)
def test_invalid_manifest_blocks_passed(overrides: dict[str, object]) -> None:
    """P0-B：manifest 任一字段不合法/与切分或 source 不一致 → passed 不可达。"""
    report = compare_models(
        _OffsetPredictor(0.1),
        _OffsetPredictor(2.0),
        _passing_split(),
        seed=5,
        n_resamples=300,
        source="mimic",
        dataset_manifest=_manifest(**overrides),
    )

    assert report.status == REPORT_STATUS_NOT_RUN
    assert report.status != REPORT_STATUS_PASSED
    assert any("dataset_manifest 校验失败" in item for item in report.limitations)


@pytest.mark.parametrize("n_patients,per_patient", [(3, 4), (2, 15)])
def test_small_sample_cannot_pass(n_patients: int, per_patient: int) -> None:
    """P2-E：窗口 <20 或患者 <5 时，即使显著更优也不得 passed。"""
    report = compare_models(
        _OffsetPredictor(0.1),
        _OffsetPredictor(2.0),
        _split_with_patient_mapping(n_patients, per_patient),
        seed=5,
        n_resamples=300,
        source="mimic",
        dataset_manifest=_manifest(),
    )

    assert report.status == REPORT_STATUS_NOT_RUN
    assert any("低于 passed 门槛" in item for item in report.limitations)


class _FixedErrorPredictor:
    """返回 ``BASE + 固定误差``（逐体征），用于构造"多数体征劣化"反例。"""

    def __init__(self, errors: np.ndarray) -> None:
        self.errors = np.asarray(errors, dtype=np.float64)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        base = np.zeros((inputs.shape[0], HORIZON, N_VITAL_KEYS), dtype=np.float64)
        return base + BASE + self.errors


def test_majority_vital_deterioration_blocks_passed() -> None:
    """P1-C 验收：JEPA 在 hr/sbp 大幅更优、其余 5/7 体征劣化 → 守卫拦下 passed。"""
    targets = np.tile(BASE, (24, HORIZON, 1))
    split = _split_with_patient_mapping(6, 4, targets=targets)
    jepa = _FixedErrorPredictor(np.array([10.0, 10.0, 1.5, 1.5, 1.5, 0.15, 0.15]))
    baseline = _FixedErrorPredictor(np.array([200.0, 200.0, 1.0, 1.0, 1.0, 0.1, 0.1]))

    report = compare_models(
        jepa,
        baseline,
        split,
        seed=5,
        n_resamples=300,
        source="mimic",
        dataset_manifest=_manifest(),
    )
    payload = report.to_dict()

    assert report.statistical_test.interpretation == INTERPRETATION_JEPA_BETTER
    assert report.statistical_test.significant is True
    assert report.statistical_test.n_deteriorated_vitals == 5
    assert set(report.statistical_test.deteriorated_vitals) == {"dbp", "spo2", "rr", "temp", "gcs"}
    assert payload["statistical_test"]["per_vital"]["dbp"]["worse"] is True
    assert payload["statistical_test"]["per_vital"]["hr"]["worse"] is False
    assert report.status == REPORT_STATUS_NOT_RUN
    assert any("多数体征相对劣化" in item for item in report.limitations)


def test_relative_mae_uses_clinical_scale_floor() -> None:
    """P2-F：目标量级远低于临床尺度（1e-8）时相对 MAE 必须为 null。"""
    split = _split()
    tiny = np.full((split.test_targets.shape[0], HORIZON, N_VITAL_KEYS), 1e-8)
    payload = evaluate_model(_OffsetPredictor(1.0), _with_targets(split, tiny), name="tiny").to_dict()

    assert payload["mae"]["hr"] == pytest.approx(float(BASE[0] + 1.0))
    assert all(value is None for value in payload["relative_mae"].values())
    assert payload["relative_mae_per_step"]["hr"] == [None, None, None]


def test_relative_mae_uses_denominator_at_scale_floor_boundary() -> None:
    """P2-F：分母恰为体征临床尺度下限时仍按分母计算（`>=` 边界，不得变 `null`）。"""
    floors = np.asarray(VITAL_ERROR_SCALES, dtype=np.float64) * RELATIVE_MAE_MIN_SCALE_FRACTION
    finite = _relative_mae(np.ones(N_VITAL_KEYS), floors)
    below = _relative_mae(np.ones(N_VITAL_KEYS), floors * 0.5)

    assert np.isfinite(finite).all()
    assert np.isnan(below).all()


def test_manifest_is_immutable() -> None:
    """P0-B：P03DatasetManifest 必须冻结——授权凭据不允许被下游就地改写。"""
    with pytest.raises(dataclasses.FrozenInstanceError):
        _manifest().version = "tampered"  # type: ignore[misc]


def test_n_resamples_accepts_numpy_integer() -> None:
    """P3-G：np.int64 与 int 行为一致；bool 仍被拒绝。"""
    split = _split()
    via_int = evaluate_model(_OffsetPredictor(0.5), split, n_resamples=100, name="a")
    via_numpy = evaluate_model(_OffsetPredictor(0.5), split, n_resamples=np.int64(100), name="a")

    assert via_numpy.n_resamples == 100
    assert np.array_equal(via_int.mae_ci95_low, via_numpy.mae_ci95_low)
    with pytest.raises(P03MetricsError):
        evaluate_model(_OffsetPredictor(0.5), split, n_resamples=True)  # type: ignore[arg-type]


# ---- J1（OODA-20261006-047）：变异门禁暴露的边界/契约盲区 ----
# 其中"`>=` 边界"一项：P2-F 已把旧的统一 epsilon 常量换成按体征临床尺度确定的
# 分母下限，故该断言改由 test_relative_mae_uses_denominator_at_scale_floor_boundary
# 覆盖（同一变异点改指新常量，不是删测试）。


def test_zero_difference_bootstrap_p_value_is_one() -> None:
    """全零配对差值 → 双侧百分位 p 值为 1（零方差边界，不得退化为 0）。"""
    split = _split()
    report = compare_models(
        _TargetPredictor(split.test_targets),
        _TargetPredictor(split.test_targets),
        split,
        seed=5,
        n_resamples=200,
    )

    assert report.statistical_test.mean_difference == pytest.approx(0.0)
    assert report.statistical_test.p_value == pytest.approx(1.0)
    assert report.statistical_test.significant is False


def test_model_metrics_is_immutable() -> None:
    """`P03ModelMetrics` 必须冻结：报告对象不允许被下游就地改写。"""
    metrics = evaluate_model(_OffsetPredictor(0.5), _split(), name="frozen")

    with pytest.raises(dataclasses.FrozenInstanceError):
        metrics.status = "tampered"  # type: ignore[misc]


def test_default_constants_match_documented_contract() -> None:
    """默认 seed / 重采样次数是对外契约（结构化需求 §参数），不得静默变更。"""
    assert DEFAULT_SEED == 42
    assert DEFAULT_N_RESAMPLES == 2000
