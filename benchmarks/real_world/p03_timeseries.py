"""P0-3 生理时序清洗、窗口化与患者级切分管道。"""

from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from benchmarks.real_world.p03_manifest import P03Manifest
from mci_world_model.sdk._clinical_world_state import VITAL_FEASIBLE_RANGES

VARIABLE_KEY_TO_VITAL: dict[str, str] = {
    "hr": "heart_rate",
    "sbp": "systolic_bp",
    "dbp": "diastolic_bp",
    "spo2": "oxygen_saturation",
    "rr": "respiratory_rate",
    "temp": "temperature",
    "gcs": "gcs",
}
VITAL_KEYS: tuple[str, ...] = tuple(VARIABLE_KEY_TO_VITAL)
N_VITAL_KEYS = len(VITAL_KEYS)


class P03PipelineError(ValueError):
    """生理时序管道违反数据契约。"""


@dataclass(frozen=True)
class P03PipelineStats:
    """只包含计数和比例的管道统计，禁止携带患者级内容。"""

    n_rows: int
    n_subjects: int
    n_windows: int
    missing_ratio: float
    imputed_ratio: float
    out_of_range_count: int
    duplicate_count: int
    out_of_order_count: int
    dropped_short_subjects: int
    n_train_subjects: int
    n_test_subjects: int

    def to_dict(self) -> dict[str, int | float]:
        """转换为可写入聚合报告的字典。"""
        return {
            "n_rows": self.n_rows,
            "n_subjects": self.n_subjects,
            "n_windows": self.n_windows,
            "missing_ratio": self.missing_ratio,
            "imputed_ratio": self.imputed_ratio,
            "out_of_range_count": self.out_of_range_count,
            "duplicate_count": self.duplicate_count,
            "out_of_order_count": self.out_of_order_count,
            "dropped_short_subjects": self.dropped_short_subjects,
            "n_train_subjects": self.n_train_subjects,
            "n_test_subjects": self.n_test_subjects,
        }


@dataclass(frozen=True)
class P03WindowSplit:
    """患者级留出后的输入与目标窗口。

    Attributes:
        train_inputs: 训练输入 ``(N, history_window, 7)``。
        train_targets: 训练目标 ``(N, forecast_horizon, 7)``。
        test_inputs: 测试输入，形状同训练输入。
        test_targets: 测试目标，形状同训练目标。
        train_subject_ids: 训练患者标识，仅用于管道内泄漏检查。
        test_subject_ids: 测试患者标识，仅用于管道内泄漏检查。
        stats: 聚合统计。
    """

    train_inputs: np.ndarray
    train_targets: np.ndarray
    test_inputs: np.ndarray
    test_targets: np.ndarray
    train_subject_ids: tuple[str, ...]
    test_subject_ids: tuple[str, ...]
    stats: P03PipelineStats


@dataclass
class _SubjectAccumulator:
    """单个患者的原始观测缓冲区。"""

    observations: list[tuple[datetime, np.ndarray]] = field(default_factory=list)
    missing_count: int = 0
    imputed_count: int = 0
    out_of_range_count: int = 0
    duplicate_count: int = 0
    out_of_order_count: int = 0


def _parse_float(value: str, field_name: str) -> float | None:
    stripped = value.strip()
    if not stripped or stripped.lower() == "nan":
        return None
    try:
        return float(stripped)
    except ValueError as exc:
        raise P03PipelineError(f"{field_name} 不是合法数值") from exc


def _parse_timestamp(value: str) -> datetime:
    stripped = value.strip()
    if not stripped:
        raise P03PipelineError("charttime 不能为空")
    try:
        return datetime.fromisoformat(stripped.replace("Z", "+00:00"))
    except ValueError as exc:
        raise P03PipelineError("charttime 不是合法 ISO-8601 时间") from exc


def _required_columns(manifest: P03Manifest) -> tuple[str, ...]:
    return ("subject_id", "charttime", *(manifest.variable_mapping[key] for key in VITAL_KEYS))


def _read_csv_rows(manifest: P03Manifest) -> tuple[list[str], dict[str, _SubjectAccumulator]]:
    path = manifest.data_path
    if path.suffix.lower() == ".parquet":
        raise P03PipelineError("P0-3a 当前只实现 CSV/CSV.GZ；parquet 管道将在后续增量交付")
    columns = _required_columns(manifest)
    vital_columns = tuple(manifest.variable_mapping[key] for key in VITAL_KEYS)
    subjects: dict[str, _SubjectAccumulator] = defaultdict(_SubjectAccumulator)
    n_rows = 0
    n_missing_cells = 0
    with ExitStack() as stack:
        if path.suffix.lower() == ".gz":
            handle = stack.enter_context(gzip.open(path, mode="rt", encoding="utf-8-sig", newline=""))
        else:
            handle = stack.enter_context(path.open(mode="r", encoding="utf-8-sig", newline=""))
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise P03PipelineError("CSV 文件没有表头")
        missing_columns = [column for column in columns if column not in reader.fieldnames]
        if missing_columns:
            raise P03PipelineError(f"CSV 缺少必填列: {', '.join(missing_columns)}")
        for row in reader:
            subject_id = (row["subject_id"] or "").strip()
            if not subject_id:
                raise P03PipelineError("subject_id 不能为空")
            timestamp = _parse_timestamp(row["charttime"])
            if not manifest.time_start <= timestamp <= manifest.time_end:
                raise P03PipelineError("数据时间戳超出 manifest 声明范围")
            values = np.empty(N_VITAL_KEYS, dtype=np.float64)
            for index, vital_key in enumerate(VITAL_KEYS):
                raw_value = row[vital_columns[index]] or ""
                parsed = _parse_float(raw_value, vital_columns[index])
                if parsed is None:
                    values[index] = np.nan
                    n_missing_cells += 1
                    subjects[subject_id].missing_count += 1
                    continue
                lower, upper = VITAL_FEASIBLE_RANGES[VARIABLE_KEY_TO_VITAL[vital_key]]
                if parsed < lower or parsed > upper:
                    subjects[subject_id].out_of_range_count += 1
                    parsed = float(np.clip(parsed, lower, upper))
                values[index] = parsed
            subjects[subject_id].observations.append((timestamp, values))
            n_rows += 1
    if n_rows == 0:
        raise P03PipelineError("CSV 没有有效数据行")
    if manifest.n_subjects is not None and len(subjects) != manifest.n_subjects:
        raise P03PipelineError(f"实际患者数 {len(subjects)} 与 manifest 声明 {manifest.n_subjects} 不一致")
    total_cells = n_rows * N_VITAL_KEYS
    if total_cells == 0 or n_missing_cells / total_cells > 0.5:
        raise P03PipelineError("原始缺失值比例超过 50%，数据质量不足")
    return list(subjects), subjects


def _deduplicate_and_impute(subjects: dict[str, _SubjectAccumulator]) -> dict[str, np.ndarray]:
    """按患者排序、去重和填补，返回长度足够的生理矩阵。"""
    cleaned: dict[str, np.ndarray] = {}
    for subject_id, accumulator in subjects.items():
        raw_observations = accumulator.observations
        accumulator.out_of_order_count = sum(
            1
            for index in range(1, len(raw_observations))
            if raw_observations[index][0] < raw_observations[index - 1][0]
        )
        ordered = sorted(accumulator.observations, key=lambda item: item[0])
        grouped: dict[datetime, list[np.ndarray]] = defaultdict(list)
        for timestamp, values in ordered:
            grouped[timestamp].append(values)
        deduplicated: list[tuple[datetime, np.ndarray]] = []
        for timestamp, repeated_values in sorted(grouped.items()):
            if len(repeated_values) > 1:
                accumulator.duplicate_count += len(repeated_values) - 1
            deduplicated.append((timestamp, np.mean(np.stack(repeated_values), axis=0)))

        series = np.stack([values for _, values in deduplicated]).astype(np.float64)
        for column in range(series.shape[1]):
            valid = ~np.isnan(series[:, column])
            if not np.any(valid):
                raise P03PipelineError("患者序列存在全缺失生理变量")
            for index in range(len(series)):
                if np.isnan(series[index, column]):
                    accumulator.imputed_count += 1
            series[:, column] = _fill_missing(series[:, column], valid)
        cleaned[subject_id] = series
    return cleaned


def _fill_missing(series: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """执行前向填补；开头缺失使用首个有效值后向填补。"""
    filled = series.copy()
    last_valid = np.nan
    for index in range(len(filled)):
        if valid[index]:
            last_valid = filled[index]
        elif not np.isnan(last_valid):
            filled[index] = last_valid
    if np.isnan(filled[0]):
        first_valid = filled[valid][0]
        for index in range(len(filled)):
            if np.isnan(filled[index]):
                filled[index] = first_valid
            else:
                break
    return filled


def _window_subject(series: np.ndarray, manifest: P03Manifest) -> tuple[np.ndarray, np.ndarray]:
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    required_length = manifest.history_window + manifest.forecast_horizon
    for start in range(series.shape[0] - required_length + 1):
        inputs.append(series[start : start + manifest.history_window])
        targets.append(
            series[start + manifest.history_window : start + manifest.history_window + manifest.forecast_horizon]
        )
    return np.stack(inputs), np.stack(targets)


def load_windows(manifest: P03Manifest) -> P03WindowSplit:
    """加载宽表并生成患者级留出窗口。

    Args:
        manifest: 已通过数据准入的 manifest。

    Returns:
        输入、目标、患者切分和聚合统计。

    Raises:
        P03PipelineError: schema、数据质量、窗口数量或切分样本不足。
    """
    _subject_ids, accumulators = _read_csv_rows(manifest)
    cleaned = _deduplicate_and_impute(accumulators)
    usable_subjects = sorted(
        subject_id
        for subject_id, series in cleaned.items()
        if series.shape[0] >= manifest.history_window + manifest.forecast_horizon
    )
    dropped_short_subjects = len(cleaned) - len(usable_subjects)
    if len(usable_subjects) < 2:
        raise P03PipelineError("可用患者数不足，无法执行患者级留出")

    rng = np.random.default_rng(manifest.seed)
    shuffled_indices = rng.permutation(len(usable_subjects))
    n_test = max(1, round(len(usable_subjects) * manifest.test_fraction))
    if n_test >= len(usable_subjects):
        raise P03PipelineError("测试患者数占用全部可用患者，训练集为空")
    test_indices = set(shuffled_indices[:n_test].tolist())
    train_subjects = [usable_subjects[i] for i in range(len(usable_subjects)) if i not in test_indices]
    test_subjects = [usable_subjects[i] for i in sorted(test_indices)]

    train_inputs: list[np.ndarray] = []
    train_targets: list[np.ndarray] = []
    test_inputs: list[np.ndarray] = []
    test_targets: list[np.ndarray] = []
    for subject_id in train_subjects:
        inputs, targets = _window_subject(cleaned[subject_id], manifest)
        train_inputs.append(inputs)
        train_targets.append(targets)
    for subject_id in test_subjects:
        inputs, targets = _window_subject(cleaned[subject_id], manifest)
        test_inputs.append(inputs)
        test_targets.append(targets)

    train_inputs_arr = np.concatenate(train_inputs, axis=0)
    train_targets_arr = np.concatenate(train_targets, axis=0)
    test_inputs_arr = np.concatenate(test_inputs, axis=0)
    test_targets_arr = np.concatenate(test_targets, axis=0)
    if not np.isfinite(train_inputs_arr).all() or not np.isfinite(train_targets_arr).all():
        raise P03PipelineError("训练窗口存在 NaN 或 Inf")
    if not np.isfinite(test_inputs_arr).all() or not np.isfinite(test_targets_arr).all():
        raise P03PipelineError("测试窗口存在 NaN 或 Inf")
    if set(train_subjects).intersection(test_subjects):
        raise P03PipelineError("患者级切分发生泄漏")

    total_missing = sum(accumulator.missing_count for accumulator in accumulators.values())
    total_imputed = sum(accumulator.imputed_count for accumulator in accumulators.values())
    raw_rows = sum(len(accumulator.observations) for accumulator in accumulators.values())
    total_cells = raw_rows * N_VITAL_KEYS
    cleaned_cells = sum(len(series) * N_VITAL_KEYS for series in cleaned.values())
    stats = P03PipelineStats(
        n_rows=sum(len(accumulator.observations) for accumulator in accumulators.values()),
        n_subjects=len(usable_subjects),
        n_windows=len(train_inputs_arr) + len(test_inputs_arr),
        missing_ratio=round(total_missing / total_cells, 6),
        imputed_ratio=round(total_imputed / cleaned_cells, 6),
        out_of_range_count=sum(accumulator.out_of_range_count for accumulator in accumulators.values()),
        duplicate_count=sum(accumulator.duplicate_count for accumulator in accumulators.values()),
        out_of_order_count=sum(accumulator.out_of_order_count for accumulator in accumulators.values()),
        dropped_short_subjects=dropped_short_subjects,
        n_train_subjects=len(train_subjects),
        n_test_subjects=len(test_subjects),
    )
    return P03WindowSplit(
        train_inputs=train_inputs_arr,
        train_targets=train_targets_arr,
        test_inputs=test_inputs_arr,
        test_targets=test_targets_arr,
        train_subject_ids=tuple(train_subjects),
        test_subject_ids=tuple(test_subjects),
        stats=stats,
    )
