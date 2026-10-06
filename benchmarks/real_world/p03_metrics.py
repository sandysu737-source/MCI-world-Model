"""P0-3b 预测质量指标与配对 bootstrap 显著性检验。

指标口径见 ``.ai-workflow/2026-10-06-p0-3b-metrics-structured-requirement.md``：
MAE / relative MAE / direction accuracy / constraint violation rate 按（步长, 体征）计算，
95% CI 与显著性用按窗口配对的 bootstrap 百分位法。

安全约束:
    本模块只输出聚合结果，禁止携带 ``subject_id``、时间戳原文或原始观测；
    不读文件、不访问网络、不直连数据库；预测值不做 clip（违反率按原始预测统计）。

预测器契约（鸭子类型）:
    模型对象必须提供 ``predict(inputs) -> ndarray``，输入 ``(N, history_window, 7)``，
    输出 ``(N, forecast_horizon, 7)``；SDK 预测器由编排层适配成该形态。
"""

from __future__ import annotations

import numbers
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from benchmarks.real_world.p03_timeseries import (
    N_VITAL_KEYS,
    VARIABLE_KEY_TO_VITAL,
    VITAL_KEYS,
    P03WindowSplit,
)
from mci_world_model.sdk._clinical_world_state import VITAL_FEASIBLE_RANGES

DEFAULT_SEED = 42
DEFAULT_N_RESAMPLES = 2000
DEFAULT_ALPHA = 0.05

DIRECTION_EPSILON = 1e-9

# P2-F: relative MAE 的分母下限不再用统一常量，而是按体征临床尺度（可行范围跨度的 1%）
# 确定。旧口径（统一 1e-9）在目标量级远低于临床尺度时仍会算出 1e9 量级的伪精度。
RELATIVE_MAE_MIN_SCALE_FRACTION = 0.01
VITAL_ERROR_SCALES: tuple[float, ...] = tuple(
    float(VITAL_FEASIBLE_RANGES[VARIABLE_KEY_TO_VITAL[key]][1] - VITAL_FEASIBLE_RANGES[VARIABLE_KEY_TO_VITAL[key]][0])
    for key in VITAL_KEYS
)
_SCALE_ARRAY = np.asarray(VITAL_ERROR_SCALES, dtype=np.float64)
_RELATIVE_MAE_MIN_DENOMINATORS = _SCALE_ARRAY * RELATIVE_MAE_MIN_SCALE_FRACTION

# P1-C: 主判据改为尺度归一化 MAE，并加"多数体征劣化"守卫（7 个体征中 >= 4 个劣化）。
SCALE_NORMALIZED_MAE = "scale_normalized_mae"
MIN_DETERIORATED_VITALS_FOR_GUARD = 4

# P2-E: 只有达到最小样本门槛（测试窗口 >= 20、测试患者 >= 5）才允许产出 passed。
MIN_PASSED_TEST_WINDOWS = 20
MIN_PASSED_TEST_PATIENTS = 5

# P0-B: 数据集清单绑定（version/sha256/authorization）——结论必须可追溯到授权数据。
_SHA256_HEX = re.compile(r"^[0-9a-fA-F]{64}$")
# bootstrap 索引矩阵分块上限，避免 (n_resamples, n_windows) 一次性物化
# P2-D: 分块预算按"gather 元素量"计，而不是只数索引个数。旧口径只限制索引矩阵大小，
# 但 `values[indices]` 会实拷贝 (chunk, N, V) 的浮点数组，N=1e5 时峰值内存可达数百 MB。
# 现在每个分块处理的浮点元素总量有上界（2e6 ≈ 16MB float64）。
_BOOTSTRAP_GATHER_ELEMENTS = 2_000_000

SPLIT_STRATEGY = "patient_holdout"
MIMIC_SOURCE = "mimic"
DEGRADED_SOURCE = "degraded_synthetic"

MODEL_STATUS_EVALUATED = "evaluated"
MODEL_STATUS_FAILED = "failed"
MODEL_STATUS_NOT_RUN = "not_run"

REPORT_STATUS_PASSED = "passed"
REPORT_STATUS_FAILED = "failed"
REPORT_STATUS_NOT_RUN = "not_run"
REPORT_STATUS_DEGRADED = "degraded"

INTERPRETATION_JEPA_BETTER = "jepa_better"
INTERPRETATION_BASELINE_BETTER = "baseline_better"
INTERPRETATION_NO_DIFFERENCE = "no_significant_difference"

PAIRED_BOOTSTRAP = "paired_bootstrap"
# 差值方向：baseline_mae - jepa_mae，>0 表示 JEPA 更优
DIFFERENCE_DIRECTION = "baseline_minus_jepa"

LIMITATION_DIRECTION = "direction_accuracy 以输入窗口末步为参照，目标无变化的样本被剔除"
LIMITATION_RAW_PREDICTION = "constraint_violation_rate 基于 clip 之前的原始预测"
LIMITATION_BOOTSTRAP_UNIT = "bootstrap 以患者为整簇重采样单位；患者数过少时 CI 仍不稳定"
LIMITATION_P_VALUE = "p 值为百分位 bootstrap 近似，未对 7 个体征做多重比较校正"
LIMITATION_EMPTY_WINDOWS = "测试窗口为 0：未产生任何指标，禁止据此判定验证通过"
LIMITATION_TOO_FEW_WINDOWS = "测试窗口少于 2：bootstrap CI 与显著性未定义"
LIMITATION_SOURCE_UNKNOWN = "source 不是 mimic：即使 JEPA 显著更优也不判定 passed"
LIMITATION_MANIFEST_MISSING = "缺少 dataset_manifest 绑定：结论无法追溯到授权数据，passed 不可达"
LIMITATION_MANIFEST_INVALID = "dataset_manifest 校验失败（{reason}）：passed 不可达"
LIMITATION_INSUFFICIENT_SAMPLE = (
    "测试样本低于 passed 门槛（需测试窗口 >= {min_windows} 且测试患者 >= {min_patients}，"
    "当前 {n_windows} 窗口 / {n_patients} 患者）：不判定 passed"
)
LIMITATION_MAJORITY_VITAL_DETERIORATION = (
    "多数体征相对劣化（{n_deteriorated} / 7，>= {threshold} 即触发守卫）：主判据不足以判定 passed"
)


class P03MetricsError(ValueError):
    """评估输入违反数据契约（编程错误，直接抛出）。"""


@dataclass(frozen=True)
class P03DatasetManifest:
    """授权数据集清单：``passed`` 结论只能绑定到带版本/哈希/授权凭据的数据集。

    P0-B：``source`` 字符串是调用方自述，不构成证据；只有显式传入且校验通过的
    manifest 才允许产出 ``passed``，否则报告状态降为 ``not_run``。
    """

    version: str
    sha256: str
    authorization: str
    n_subjects: int
    source: str = MIMIC_SOURCE

    def to_dict(self) -> dict[str, Any]:
        """序列化为聚合字典（不含任何患者级内容）。"""
        return {
            "version": self.version,
            "sha256": self.sha256,
            "authorization": self.authorization,
            "n_subjects": int(self.n_subjects),
            "source": self.source,
        }


def _manifest_error(manifest: P03DatasetManifest | None, *, source: str | None, n_test_patients: int) -> str | None:
    """校验 manifest 绑定；返回错误原因（None 表示校验通过）。"""
    if manifest is None:
        return "manifest 缺失"
    if not isinstance(manifest.version, str) or not manifest.version.strip():
        return "version 为空"
    if not isinstance(manifest.sha256, str) or _SHA256_HEX.fullmatch(manifest.sha256) is None:
        return "sha256 不是 64 位十六进制"
    if not isinstance(manifest.authorization, str) or not manifest.authorization.strip():
        return "authorization 为空"
    n_subjects = manifest.n_subjects
    if isinstance(n_subjects, bool) or not isinstance(n_subjects, numbers.Integral) or int(n_subjects) < 1:
        return "n_subjects 必须是 >= 1 的整数"
    if int(n_subjects) < n_test_patients:
        return f"n_subjects({int(n_subjects)}) 小于测试患者数({n_test_patients})"
    if source is None or manifest.source != source:
        return f"manifest.source({manifest.source!r}) 与报告 source({source!r}) 不一致"
    return None


def _json_scalar(value: float) -> float | None:
    """把 NaN/Inf 序列化为 ``null``，其余转为 float。"""
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def _per_vital(values: np.ndarray) -> dict[str, float | None]:
    return {key: _json_scalar(values[index]) for index, key in enumerate(VITAL_KEYS)}


def _per_vital_series(values: np.ndarray) -> dict[str, list[float | None]]:
    return {key: [_json_scalar(step_value) for step_value in values[:, index]] for index, key in enumerate(VITAL_KEYS)}


def _validate_options(name: str, n_resamples: int, alpha: float) -> None:
    if not isinstance(name, str) or not name.strip():
        raise P03MetricsError("模型名称不能为空")
    # P3-G: 接受 numbers.Integral（含 np.int64），但 bool 仍非法；失败信息不变。
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, numbers.Integral) or int(n_resamples) < 2:
        raise P03MetricsError(f"n_resamples 必须是 >= 2 的整数，收到 {n_resamples!r}")
    if not isinstance(alpha, float) or not 0.0 < alpha < 1.0:
        raise P03MetricsError(f"alpha 必须位于 (0, 1) 区间，收到 {alpha!r}")


def _validate_split(split: P03WindowSplit) -> None:
    inputs = np.asarray(split.test_inputs)
    targets = np.asarray(split.test_targets)
    if inputs.ndim != 3 or targets.ndim != 3:
        raise P03MetricsError(f"测试窗口必须是三阶数组，收到 {inputs.ndim} 与 {targets.ndim} 阶")
    if inputs.shape[2] != N_VITAL_KEYS or targets.shape[2] != N_VITAL_KEYS:
        raise P03MetricsError(f"体征维度必须是 {N_VITAL_KEYS}，收到 {inputs.shape[2]} 与 {targets.shape[2]}")
    if inputs.shape[0] != targets.shape[0]:
        raise P03MetricsError("测试输入与目标窗口数不一致")
    if inputs.shape[1] < 1 or targets.shape[1] < 1:
        raise P03MetricsError("历史窗口与预测步长都必须 >= 1")
    if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
        raise P03MetricsError("测试输入或目标包含 NaN/Inf，请先修复上游管道")
    # P0-A: 结论必须绑定独立单元。没有 window→patient 映射就无法按患者重采样，
    # 只能退化成窗口级 bootstrap（会低估 CI、放大假阳性）→ 直接 fail-closed。
    if inputs.shape[0] > 0:
        window_subjects = split.test_window_subjects
        if not window_subjects:
            raise P03MetricsError(
                "缺少 window→patient 映射（test_window_subjects），无法按患者整簇重采样；"
                "请使用 P0-3a 管道产出的 P03WindowSplit"
            )
        if len(window_subjects) != inputs.shape[0]:
            raise P03MetricsError(
                f"window→patient 映射长度 {len(window_subjects)} 与测试窗口数 {inputs.shape[0]} 不一致"
            )


def _call_predictor(model: Any, inputs: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise P03MetricsError(f"模型 {type(model).__name__} 未提供可调用的 predict()")
    try:
        raw = predict(inputs)
    except Exception as exc:  # 预测器异常需带上评估上下文重新抛出
        raise P03MetricsError(f"模型 predict() 执行失败: {exc}") from exc
    try:
        output = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise P03MetricsError("模型输出无法转换为 float64 数组") from exc
    if output.shape != expected_shape:
        raise P03MetricsError(f"模型输出形状 {output.shape} 与目标形状 {expected_shape} 不一致")
    return output


def _relative_mae(error: np.ndarray, mean_abs_target: np.ndarray) -> np.ndarray:
    """相对 MAE；分母低于该体征临床尺度下限（可行范围跨度的 1%）时视为未定义 → ``null``。

    P2-F：阈值按体征尺度确定（如 temp 0.09、hr 2.0），目标量级 1e-8 这类
    "远低于临床尺度"的输入不再产出 1e9 量级的伪精度。
    """
    denominator = np.where(mean_abs_target >= _RELATIVE_MAE_MIN_DENOMINATORS, mean_abs_target, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return error / denominator


def _direction_hits(
    predictions: np.ndarray, targets: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """返回（命中数, 有效样本数），形状均为 ``(forecast_horizon, 7)``。"""
    target_delta = targets - reference[:, None, :]
    prediction_delta = predictions - reference[:, None, :]
    valid = np.abs(target_delta) > DIRECTION_EPSILON
    hits = (np.sign(target_delta) == np.sign(prediction_delta)) & valid
    # 按窗口求和，得到 (forecast_horizon, 7) 的命中数与有效样本数
    return hits.sum(axis=0, dtype=np.int64), valid.sum(axis=0, dtype=np.int64)


def _ratio(hits: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.where(counts > 0, hits / np.maximum(counts, 1), np.nan)


def _constraint_violation_rate(predictions: np.ndarray) -> np.ndarray:
    rates = np.zeros((predictions.shape[1], N_VITAL_KEYS), dtype=np.float64)
    for index, vital_key in enumerate(VITAL_KEYS):
        lower, upper = VITAL_FEASIBLE_RANGES[VARIABLE_KEY_TO_VITAL[vital_key]]
        column = predictions[:, :, index]
        violated = ~np.isfinite(column) | (column < lower) | (column > upper)
        rates[:, index] = violated.mean(axis=0)
    return rates


def _bootstrap_cluster_replicates(
    values: np.ndarray,
    window_subjects: tuple[str, ...],
    rng: np.random.Generator,
    n_resamples: int,
) -> np.ndarray:
    """按患者整簇重采样（block bootstrap）求均值，``(n_resamples, *values.shape[1:])``。

    P0-A：统计单元是窗口，但独立单元是患者——同一患者的窗口高度相关，按窗口独立重采样会
    低估 CI 宽度、放大假阳性。这里以患者为簇整块重采样。

    实现上先按簇聚合「求和 / 计数」，再对簇做有放回抽样后合并（``sum(簇和) / sum(簇窗口数)``），
    与"把被抽中患者的窗口 gather 起来求均值"完全等价，但内存只与簇数相关，不会随窗口数爆掉。
    """
    values = np.asarray(values, dtype=np.float64)
    n_windows = int(values.shape[0])
    if n_windows == 0:
        raise P03MetricsError("无法对空窗口做 bootstrap")
    if len(window_subjects) != n_windows:
        raise P03MetricsError(f"window→patient 映射长度 {len(window_subjects)} 与窗口数 {n_windows} 不一致")
    rest = values.shape[1:]
    flat = values.reshape(n_windows, -1)
    _labels, inverse = np.unique(np.asarray(window_subjects, dtype=object), return_inverse=True)
    inverse = np.asarray(inverse, dtype=np.int64)
    n_clusters = int(inverse.max()) + 1 if n_windows else 0
    cluster_sums = np.zeros((n_clusters, flat.shape[1]), dtype=np.float64)
    cluster_counts = np.zeros(n_clusters, dtype=np.float64)
    np.add.at(cluster_sums, inverse, flat)
    np.add.at(cluster_counts, inverse, 1.0)
    per_replicate = max(1, n_clusters * flat.shape[1])
    chunk = max(1, _BOOTSTRAP_GATHER_ELEMENTS // per_replicate)
    replicates: list[np.ndarray] = []
    for start in range(0, n_resamples, chunk):
        size = min(chunk, n_resamples - start)
        picks = rng.integers(0, n_clusters, size=(size, n_clusters))
        totals = cluster_sums[picks].sum(axis=1)
        weights = cluster_counts[picks].sum(axis=1)
        replicates.append((totals / np.maximum(weights, 1.0)[:, None]).reshape((size, *rest)))
    return np.concatenate(replicates, axis=0)


def _ci_bounds(replicates: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(np.percentile(replicates, 100.0 * alpha / 2.0, axis=0), dtype=np.float64)
    upper = np.asarray(np.percentile(replicates, 100.0 * (1.0 - alpha / 2.0), axis=0), dtype=np.float64)
    return lower, upper


def _bootstrap_p_values(replicates: np.ndarray) -> np.ndarray:
    """百分位近似双侧 p 值：取偏离 0 的两侧尾部较小者乘 2。"""
    left = np.mean(replicates <= 0.0, axis=0)
    right = np.mean(replicates >= 0.0, axis=0)
    return np.minimum(1.0, 2.0 * np.minimum(left, right))


def _nan_step(horizon: int) -> np.ndarray:
    return np.full((horizon, N_VITAL_KEYS), np.nan, dtype=np.float64)


def _nan_overall() -> np.ndarray:
    return np.full(N_VITAL_KEYS, np.nan, dtype=np.float64)


def _blank_model_metrics(
    *,
    name: str,
    status: str,
    n_windows: int,
    history_window: int,
    forecast_horizon: int,
    seed: int,
    n_resamples: int,
    alpha: float,
    failure_reason: str | None,
    limitations: tuple[str, ...],
) -> P03ModelMetrics:
    return P03ModelMetrics(
        model_name=name,
        status=status,
        n_windows=n_windows,
        history_window=history_window,
        forecast_horizon=forecast_horizon,
        seed=seed,
        n_resamples=n_resamples,
        alpha=alpha,
        overall_mae=_nan_overall(),
        mae_per_step=_nan_step(forecast_horizon),
        overall_relative_mae=_nan_overall(),
        relative_mae_per_step=_nan_step(forecast_horizon),
        overall_direction_accuracy=_nan_overall(),
        direction_accuracy_per_step=_nan_step(forecast_horizon),
        overall_constraint_violation_rate=_nan_overall(),
        constraint_violation_rate_per_step=_nan_step(forecast_horizon),
        mae_ci95_low=_nan_overall(),
        mae_ci95_high=_nan_overall(),
        failure_reason=failure_reason,
        limitations=limitations,
    )


@dataclass(frozen=True)
class P03ModelMetrics:
    """单模型聚合指标；只含聚合量，可安全写入报告。"""

    model_name: str
    status: str
    n_windows: int
    history_window: int
    forecast_horizon: int
    seed: int
    n_resamples: int
    alpha: float
    overall_mae: np.ndarray
    mae_per_step: np.ndarray
    overall_relative_mae: np.ndarray
    relative_mae_per_step: np.ndarray
    overall_direction_accuracy: np.ndarray
    direction_accuracy_per_step: np.ndarray
    overall_constraint_violation_rate: np.ndarray
    constraint_violation_rate_per_step: np.ndarray
    mae_ci95_low: np.ndarray
    mae_ci95_high: np.ndarray
    failure_reason: str | None = None
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """序列化为只含聚合结果的字典。"""
        payload: dict[str, Any] = {
            "model_name": self.model_name,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "n_windows": self.n_windows,
            "history_window": self.history_window,
            "forecast_horizon": self.forecast_horizon,
            "vital_keys": list(VITAL_KEYS),
        }
        for metric_name, per_step, overall in (
            ("mae", self.mae_per_step, self.overall_mae),
            ("relative_mae", self.relative_mae_per_step, self.overall_relative_mae),
            ("direction_accuracy", self.direction_accuracy_per_step, self.overall_direction_accuracy),
            (
                "constraint_violation_rate",
                self.constraint_violation_rate_per_step,
                self.overall_constraint_violation_rate,
            ),
        ):
            payload[metric_name] = _per_vital(overall)
            payload[f"{metric_name}_per_step"] = _per_vital_series(per_step)
        payload["ci95"] = {
            "metric": "mae",
            "level": _json_scalar(1.0 - self.alpha),
            "n_resamples": self.n_resamples,
            "seed": self.seed,
            "overall": {
                key: [
                    _json_scalar(self.mae_ci95_low[index]),
                    _json_scalar(self.mae_ci95_high[index]),
                ]
                for index, key in enumerate(VITAL_KEYS)
            },
        }
        payload["limitations"] = list(self.limitations)
        return payload


@dataclass(frozen=True)
class P03StatisticalTest:
    """JEPA 与基线的配对 bootstrap 显著性检验结果。"""

    n_resamples: int
    seed: int
    alpha: float
    mean_difference: float
    ci95_low: float
    ci95_high: float
    p_value: float
    significant: bool
    interpretation: str
    per_vital: dict[str, dict[str, Any]]
    method: str = PAIRED_BOOTSTRAP
    metric: str = SCALE_NORMALIZED_MAE
    direction: str = DIFFERENCE_DIRECTION
    note: str | None = None
    n_deteriorated_vitals: int = 0
    deteriorated_vitals: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """序列化为聚合字典。"""
        return {
            "method": self.method,
            "metric": self.metric,
            "direction": self.direction,
            "n_resamples": self.n_resamples,
            "seed": self.seed,
            "alpha": _json_scalar(self.alpha),
            "mean_difference": _json_scalar(self.mean_difference),
            "ci95": [_json_scalar(self.ci95_low), _json_scalar(self.ci95_high)],
            "p_value": _json_scalar(self.p_value),
            "significant": self.significant,
            "interpretation": self.interpretation,
            "n_deteriorated_vitals": self.n_deteriorated_vitals,
            "deteriorated_vitals": list(self.deteriorated_vitals),
            "per_vital": self.per_vital,
            "note": self.note,
        }


@dataclass(frozen=True)
class P03Report:
    """P0-3 双模型对比报告；``to_dict()`` 可直接落盘为聚合结论。"""

    status: str
    source: str | None
    seed: int
    n_resamples: int
    alpha: float
    split_summary: dict[str, Any]
    models: dict[str, P03ModelMetrics]
    statistical_test: P03StatisticalTest
    limitations: tuple[str, ...]
    dataset_manifest: P03DatasetManifest | None = None
    metric: str = SCALE_NORMALIZED_MAE

    @property
    def n_deteriorated_vitals(self) -> int:
        """相对劣化的体征数（P1-C 守卫的可读入口）。"""
        return self.statistical_test.n_deteriorated_vitals

    def to_dict(self) -> dict[str, Any]:
        """序列化为只含聚合结果的字典。"""
        return {
            "status": self.status,
            "metric": self.metric,
            "n_deteriorated_vitals": self.n_deteriorated_vitals,
            "source": self.source,
            "dataset_manifest": None if self.dataset_manifest is None else self.dataset_manifest.to_dict(),
            "split": dict(self.split_summary),
            "models": {name: metrics.to_dict() for name, metrics in self.models.items()},
            "statistical_test": self.statistical_test.to_dict(),
            "seed": self.seed,
            "n_resamples": self.n_resamples,
            "alpha": _json_scalar(self.alpha),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class _EvalOutcome:
    """内部使用：公开指标 + 逐窗口误差（禁止外泄患者级内容）。

    ``*_normalized`` 系列是 P1-C 的尺度归一化误差（误差 / 体征可行范围跨度），
    仅用于主判据与劣化守卫，不进入对外报告。
    """

    metrics: P03ModelMetrics
    window_overall_error: np.ndarray | None
    window_vital_error: np.ndarray | None
    window_normalized_error: np.ndarray | None = None
    window_vital_normalized: np.ndarray | None = None
    vital_normalized_error: np.ndarray | None = None


def _evaluate(
    model: Any,
    split: P03WindowSplit,
    *,
    name: str,
    seed: int,
    n_resamples: int,
    alpha: float,
) -> _EvalOutcome:
    _validate_options(name, n_resamples, alpha)
    n_resamples = int(n_resamples)
    _validate_split(split)
    targets = np.asarray(split.test_targets, dtype=np.float64)
    inputs = np.asarray(split.test_inputs, dtype=np.float64)
    n_windows = int(targets.shape[0])
    horizon = int(targets.shape[1])
    history = int(inputs.shape[1])
    base_limitations = (LIMITATION_DIRECTION, LIMITATION_RAW_PREDICTION)

    if n_windows == 0:
        metrics = _blank_model_metrics(
            name=name,
            status=MODEL_STATUS_NOT_RUN,
            n_windows=0,
            history_window=history,
            forecast_horizon=horizon,
            seed=seed,
            n_resamples=n_resamples,
            alpha=alpha,
            failure_reason=None,
            limitations=(*base_limitations, LIMITATION_EMPTY_WINDOWS),
        )
        return _EvalOutcome(metrics, None, None)

    predictions = _call_predictor(model, inputs, tuple(targets.shape))
    if not np.isfinite(predictions).all():
        metrics = _blank_model_metrics(
            name=name,
            status=MODEL_STATUS_FAILED,
            n_windows=n_windows,
            history_window=history,
            forecast_horizon=horizon,
            seed=seed,
            n_resamples=n_resamples,
            alpha=alpha,
            failure_reason="预测结果包含 NaN 或 Inf，按 fail-closed 返回 failed",
            limitations=base_limitations,
        )
        return _EvalOutcome(metrics, None, None)

    error = np.abs(predictions - targets)
    mae_per_step = error.mean(axis=0)
    overall_mae = error.mean(axis=(0, 1))
    relative_mae_per_step = _relative_mae(mae_per_step, np.abs(targets).mean(axis=0))
    overall_relative_mae = _relative_mae(overall_mae, np.abs(targets).mean(axis=(0, 1)))

    reference = inputs[:, -1, :]
    hits, counts = _direction_hits(predictions, targets, reference)
    direction_per_step = _ratio(hits, counts)
    overall_direction = _ratio(hits.sum(axis=0), counts.sum(axis=0))

    violation_per_step = _constraint_violation_rate(predictions)
    overall_violation = violation_per_step.mean(axis=0)

    window_overall_error = error.mean(axis=(1, 2))
    window_vital_error = error.mean(axis=1)
    # P1-C: 主判据改用尺度归一化误差——原始单位 MAE 被 hr/sbp 等大数值体征主导，
    # 会掩盖 gcs/temp 等小尺度但临床关键体征的劣化。
    window_vital_normalized = window_vital_error / _SCALE_ARRAY
    window_normalized_error = window_vital_normalized.mean(axis=1)
    vital_normalized_error = window_vital_normalized.mean(axis=0)
    rng = np.random.default_rng(seed)
    ci_low, ci_high = _ci_bounds(
        _bootstrap_cluster_replicates(window_vital_error, split.test_window_subjects, rng, n_resamples),
        alpha,
    )

    metrics = P03ModelMetrics(
        model_name=name,
        status=MODEL_STATUS_EVALUATED,
        n_windows=n_windows,
        history_window=history,
        forecast_horizon=horizon,
        seed=seed,
        n_resamples=n_resamples,
        alpha=alpha,
        overall_mae=overall_mae,
        mae_per_step=mae_per_step,
        overall_relative_mae=overall_relative_mae,
        relative_mae_per_step=relative_mae_per_step,
        overall_direction_accuracy=overall_direction,
        direction_accuracy_per_step=direction_per_step,
        overall_constraint_violation_rate=overall_violation,
        constraint_violation_rate_per_step=violation_per_step,
        mae_ci95_low=ci_low,
        mae_ci95_high=ci_high,
        failure_reason=None,
        limitations=base_limitations,
    )
    return _EvalOutcome(
        metrics,
        window_overall_error,
        window_vital_error,
        window_normalized_error,
        window_vital_normalized,
        vital_normalized_error,
    )


def evaluate_model(
    model: Any,
    split: P03WindowSplit,
    *,
    name: str = "model",
    seed: int = DEFAULT_SEED,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
) -> P03ModelMetrics:
    """计算单模型聚合指标。

    Args:
        model: 提供 ``predict(inputs) -> (N, horizon, 7)`` 的数组级预测器。
        split: P0-3a 产出的患者级窗口切分。
        name: 报告中的模型名称。
        seed: bootstrap 随机种子（应与 manifest.seed 对齐）。
        n_resamples: bootstrap 重采样次数，必须 >= 2。
        alpha: 显著性水平，必须位于 (0, 1)。

    Returns:
        只含聚合量的 ``P03ModelMetrics``。

    Raises:
        P03MetricsError: split 或选项违反契约、模型缺少 ``predict()``、输出形状不符。
    """
    return _evaluate(model, split, name=name, seed=seed, n_resamples=n_resamples, alpha=alpha).metrics


def _split_summary(split: P03WindowSplit) -> dict[str, Any]:
    return {
        "strategy": SPLIT_STRATEGY,
        "n_train_subjects": int(split.stats.n_train_subjects),
        "n_test_subjects": int(split.stats.n_test_subjects),
        "n_train_windows": int(np.asarray(split.train_inputs).shape[0]),
        "n_test_windows": int(np.asarray(split.test_inputs).shape[0]),
        "n_test_patients": len(set(split.test_window_subjects)),
    }


def _blank_statistical_test(
    *,
    seed: int,
    n_resamples: int,
    alpha: float,
    note: str,
) -> P03StatisticalTest:
    nan = float("nan")
    return P03StatisticalTest(
        n_resamples=n_resamples,
        seed=seed,
        alpha=alpha,
        mean_difference=nan,
        ci95_low=nan,
        ci95_high=nan,
        p_value=nan,
        significant=False,
        interpretation=INTERPRETATION_NO_DIFFERENCE,
        per_vital={},
        note=note,
    )


def _paired_bootstrap_test(
    jepa_outcome: _EvalOutcome,
    baseline_outcome: _EvalOutcome,
    window_subjects: tuple[str, ...],
    *,
    seed: int,
    n_resamples: int,
    alpha: float,
) -> P03StatisticalTest:
    jepa_window = jepa_outcome.window_overall_error
    baseline_window = baseline_outcome.window_overall_error
    jepa_normalized = jepa_outcome.window_normalized_error
    baseline_normalized = baseline_outcome.window_normalized_error
    jepa_vital_normalized = jepa_outcome.window_vital_normalized
    baseline_vital_normalized = baseline_outcome.window_vital_normalized
    jepa_vital_mean = jepa_outcome.vital_normalized_error
    baseline_vital_mean = baseline_outcome.vital_normalized_error
    if (
        jepa_window is None
        or baseline_window is None
        or jepa_normalized is None
        or baseline_normalized is None
        or jepa_vital_normalized is None
        or baseline_vital_normalized is None
        or jepa_vital_mean is None
        or baseline_vital_mean is None
    ):
        raise P03MetricsError("两个模型缺少逐窗口误差，无法配对")
    if jepa_window.shape != baseline_window.shape:
        raise P03MetricsError("两个模型的测试窗口数不一致，无法配对")
    n_windows = int(jepa_window.shape[0])
    if n_windows < 2:
        return _blank_statistical_test(seed=seed, n_resamples=n_resamples, alpha=alpha, note=LIMITATION_TOO_FEW_WINDOWS)

    # P1-C：主判据是"尺度归一化 MAE"的配对差（baseline - jepa，>0 表示 JEPA 更优），
    # 并统计相对劣化的体征（守卫用：>= 4/7 劣化时 passed 不可达）。
    difference = baseline_normalized - jepa_normalized
    difference_vital = baseline_vital_normalized - jepa_vital_normalized
    deteriorated_vitals = tuple(
        key for index, key in enumerate(VITAL_KEYS) if not jepa_vital_mean[index] < baseline_vital_mean[index]
    )
    rng = np.random.default_rng(seed)
    replicates = _bootstrap_cluster_replicates(difference, window_subjects, rng, n_resamples)
    vital_replicates = _bootstrap_cluster_replicates(difference_vital, window_subjects, rng, n_resamples)

    mean_difference = float(difference.mean())
    ci_low_array, ci_high_array = _ci_bounds(replicates, alpha)
    ci_low = float(ci_low_array)
    ci_high = float(ci_high_array)
    p_value = float(_bootstrap_p_values(replicates))
    significant = bool(ci_low > 0.0 or ci_high < 0.0)
    if not significant:
        interpretation = INTERPRETATION_NO_DIFFERENCE
    elif mean_difference > 0.0:
        interpretation = INTERPRETATION_JEPA_BETTER
    else:
        interpretation = INTERPRETATION_BASELINE_BETTER

    vital_low, vital_high = _ci_bounds(vital_replicates, alpha)
    vital_p = _bootstrap_p_values(vital_replicates)
    vital_mean = difference_vital.mean(axis=0)
    per_vital: dict[str, dict[str, Any]] = {}
    for index, key in enumerate(VITAL_KEYS):
        vital_significant = bool(vital_low[index] > 0.0 or vital_high[index] < 0.0)
        per_vital[key] = {
            "mean_difference": _json_scalar(vital_mean[index]),
            "ci95": [_json_scalar(vital_low[index]), _json_scalar(vital_high[index])],
            "p_value": _json_scalar(vital_p[index]),
            "significant": vital_significant,
            "normalized_mae_jepa": _json_scalar(jepa_vital_mean[index]),
            "normalized_mae_baseline": _json_scalar(baseline_vital_mean[index]),
            "worse": key in deteriorated_vitals,
        }

    return P03StatisticalTest(
        n_resamples=n_resamples,
        seed=seed,
        alpha=alpha,
        mean_difference=mean_difference,
        ci95_low=ci_low,
        ci95_high=ci_high,
        p_value=p_value,
        significant=significant,
        interpretation=interpretation,
        per_vital=per_vital,
        n_deteriorated_vitals=len(deteriorated_vitals),
        deteriorated_vitals=deteriorated_vitals,
    )


def compare_models(
    jepa: Any,
    baseline: Any,
    split: P03WindowSplit,
    *,
    seed: int = DEFAULT_SEED,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    source: str | None = None,
    dataset_manifest: P03DatasetManifest | None = None,
    jepa_name: str = "jepa_clinical_bridge",
    baseline_name: str = "clinical_dynamics_baseline",
) -> P03Report:
    """在同一 split 上配对比较 JEPA 与基线，输出聚合报告。

    主判据是**尺度归一化 MAE**（误差 / 体征可行范围跨度）的配对差
    ``baseline_normalized - jepa_normalized``，>0 表示 JEPA 更优。报告状态映射：
    ``failed``（任一模型失败）/ ``not_run``（无测试窗口，或未通过 pass 守卫）/
    ``degraded``（降级数据源）/ ``passed``（同时满足：JEPA 显著更优、``source="mimic"``、
    manifest 绑定校验通过、样本达门槛、无多数体征劣化）/ 其余为 ``failed``。

    P0-B：``source`` 只是调用方自述，``passed`` 必须额外绑定 ``dataset_manifest``
    （version / sha256 / authorization / n_subjects），缺失或不匹配时 ``passed`` 不可达。
    P2-E：``passed`` 还要求测试窗口 >= 20 且测试患者 >= 5。
    P1-C：7 个体征中 >= 4 个相对劣化时，``passed`` 不可达。

    Args:
        jepa: JEPA 侧数组级预测器。
        baseline: 基线侧数组级预测器。
        split: 患者级窗口切分，两侧必须使用同一 split 才能配对。
        seed: bootstrap 种子。
        n_resamples: bootstrap 重采样次数。
        alpha: 显著性水平。
        source: 数据源标识；``degraded_synthetic`` 时报告降级，不写 ``passed``。
        dataset_manifest: 授权数据集清单；缺失/不匹配时 ``passed`` 不可达（P0-B）。
        jepa_name: JEPA 模型报告键。
        baseline_name: 基线模型报告键。

    Returns:
        ``P03Report``（``to_dict()`` 只含聚合结果）。
    """
    jepa_outcome = _evaluate(jepa, split, name=jepa_name, seed=seed, n_resamples=n_resamples, alpha=alpha)
    baseline_outcome = _evaluate(baseline, split, name=baseline_name, seed=seed, n_resamples=n_resamples, alpha=alpha)
    n_resamples = int(n_resamples)
    models = {jepa_name: jepa_outcome.metrics, baseline_name: baseline_outcome.metrics}
    limitations: list[str] = [LIMITATION_BOOTSTRAP_UNIT, LIMITATION_P_VALUE]
    for outcome in (jepa_outcome, baseline_outcome):
        limitations.extend(outcome.metrics.limitations)
    ordered_limitations = tuple(dict.fromkeys(limitations))

    jepa_failed = jepa_outcome.metrics.status == MODEL_STATUS_FAILED
    baseline_failed = baseline_outcome.metrics.status == MODEL_STATUS_FAILED
    no_windows = jepa_outcome.window_overall_error is None or baseline_outcome.window_overall_error is None

    if jepa_failed or baseline_failed:
        statistical_test = _blank_statistical_test(
            seed=seed,
            n_resamples=n_resamples,
            alpha=alpha,
            note="存在 failed 模型，未执行显著性检验",
        )
        status = REPORT_STATUS_FAILED
    elif no_windows:
        statistical_test = _blank_statistical_test(
            seed=seed,
            n_resamples=n_resamples,
            alpha=alpha,
            note=LIMITATION_EMPTY_WINDOWS,
        )
        status = REPORT_STATUS_NOT_RUN
    else:
        statistical_test = _paired_bootstrap_test(
            jepa_outcome,
            baseline_outcome,
            split.test_window_subjects,
            seed=seed,
            n_resamples=n_resamples,
            alpha=alpha,
        )
        n_windows = int(np.asarray(split.test_inputs).shape[0])
        n_test_patients = len(set(split.test_window_subjects))
        # P0-B / P1-C / P2-E：passed 必须同时通过"清单绑定 + 样本门槛 + 劣化守卫"。
        blockers: list[str] = []
        manifest_error = _manifest_error(dataset_manifest, source=source, n_test_patients=n_test_patients)
        if dataset_manifest is None:
            blockers.append(LIMITATION_MANIFEST_MISSING)
        elif manifest_error is not None:
            blockers.append(LIMITATION_MANIFEST_INVALID.format(reason=manifest_error))
        if n_windows < MIN_PASSED_TEST_WINDOWS or n_test_patients < MIN_PASSED_TEST_PATIENTS:
            blockers.append(
                LIMITATION_INSUFFICIENT_SAMPLE.format(
                    min_windows=MIN_PASSED_TEST_WINDOWS,
                    min_patients=MIN_PASSED_TEST_PATIENTS,
                    n_windows=n_windows,
                    n_patients=n_test_patients,
                )
            )
        if statistical_test.n_deteriorated_vitals >= MIN_DETERIORATED_VITALS_FOR_GUARD:
            blockers.append(
                LIMITATION_MAJORITY_VITAL_DETERIORATION.format(
                    n_deteriorated=statistical_test.n_deteriorated_vitals,
                    threshold=MIN_DETERIORATED_VITALS_FOR_GUARD,
                )
            )
        if source == DEGRADED_SOURCE:
            status = REPORT_STATUS_DEGRADED
        elif statistical_test.interpretation == INTERPRETATION_JEPA_BETTER and source == MIMIC_SOURCE:
            if blockers:
                status = REPORT_STATUS_NOT_RUN
                ordered_limitations = (*ordered_limitations, *blockers)
            else:
                status = REPORT_STATUS_PASSED
        else:
            status = REPORT_STATUS_FAILED
            if statistical_test.interpretation == INTERPRETATION_JEPA_BETTER:
                ordered_limitations = (*ordered_limitations, LIMITATION_SOURCE_UNKNOWN)

    if statistical_test.note is not None:
        ordered_limitations = (*ordered_limitations, statistical_test.note)

    return P03Report(
        status=status,
        source=source,
        seed=seed,
        n_resamples=n_resamples,
        alpha=alpha,
        split_summary=_split_summary(split),
        models=models,
        statistical_test=statistical_test,
        limitations=ordered_limitations,
        dataset_manifest=dataset_manifest,
    )
