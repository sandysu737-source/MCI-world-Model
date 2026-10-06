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
    RELATIVE_MAE_EPSILON,
    REPORT_STATUS_DEGRADED,
    REPORT_STATUS_FAILED,
    REPORT_STATUS_NOT_RUN,
    REPORT_STATUS_PASSED,
    P03MetricsError,
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
    """构造测试窗口：末步观测为 BASE，目标在 BASE 附近随机波动。"""
    rng = np.random.default_rng(seed)
    inputs = np.tile(BASE, (n_windows, HISTORY, 1))
    targets = BASE + rng.normal(0.0, 0.5, size=(n_windows, HORIZON, N_VITAL_KEYS))
    return P03WindowSplit(
        train_inputs=inputs[: n_windows // 2],
        train_targets=targets[: n_windows // 2],
        test_inputs=inputs,
        test_targets=targets,
        train_subject_ids=("train-a", "train-b"),
        test_subject_ids=("test-a", "test-b"),
        stats=_stats(n_windows=n_windows),
    )


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
    split = _split()
    first = evaluate_model(_OffsetPredictor(0.4), split, seed=11, n_resamples=200, name="a")
    second = evaluate_model(_OffsetPredictor(0.4), split, seed=11, n_resamples=200, name="a")
    other = evaluate_model(_OffsetPredictor(0.4), split, seed=12, n_resamples=200, name="a")

    assert np.array_equal(first.mae_ci95_low, second.mae_ci95_low)
    assert np.array_equal(first.mae_ci95_high, second.mae_ci95_high)
    assert not np.array_equal(first.mae_ci95_low, other.mae_ci95_low)
    assert np.all(first.mae_ci95_low < first.overall_mae) or np.all(first.mae_ci95_high > first.overall_mae)


def test_compare_models_detects_better_model() -> None:
    split = _split()
    report = compare_models(
        _OffsetPredictor(0.1), _OffsetPredictor(2.0), split, seed=5, n_resamples=500, source="mimic"
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
        "n_train_subjects": 4,
        "n_test_subjects": 2,
        "n_train_windows": 8,
        "n_test_windows": 16,
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
        "source",
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


# ---- J1（OODA-20261006-047）：变异门禁暴露的 4 处边界/契约盲区 ----
# 这些断言按结构化需求 §参数/§指标口径 的既有契约写成，用于让变异门禁的
# 红灯指向"真实盲区"而非脚本口径问题（不可杀变异不应被计入分母）。


def test_relative_mae_uses_denominator_at_epsilon_boundary() -> None:
    """目标幅度恰为 `RELATIVE_MAE_EPSILON` 时仍按分母计算（`>=` 边界，不是 `null`）。"""
    tiny = np.full((1, HORIZON, N_VITAL_KEYS), RELATIVE_MAE_EPSILON)
    flat_zero = np.zeros((1, HORIZON, N_VITAL_KEYS))
    split = _with_targets(_split(n_windows=1), tiny)

    payload = evaluate_model(_TargetPredictor(flat_zero), split, name="eps-boundary").to_dict()

    assert payload["relative_mae"]["hr"] == pytest.approx(1.0)
    assert payload["relative_mae_per_step"]["hr"] == pytest.approx([1.0] * HORIZON)


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
