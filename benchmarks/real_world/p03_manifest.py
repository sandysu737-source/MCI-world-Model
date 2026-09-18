"""P0-3 真实生理时序验证的数据清单契约。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REQUIRED_VITAL_KEYS: tuple[str, ...] = ("hr", "sbp", "dbp", "spo2", "rr", "temp", "gcs")
ALLOWED_SOURCES: frozenset[str] = frozenset({"mimic", "degraded_synthetic"})
REQUIRED_SPLIT_STRATEGY = "patient_holdout"


class ManifestError(ValueError):
    """manifest 不满足数据准入契约。"""


@dataclass(frozen=True)
class P03Manifest:
    """P0-3 输入清单。

    Attributes:
        source: 数据源，MIMIC 或明确标注的降级仿真数据。
        version: 数据集版本。
        authorization: 授权状态，当前仅接受 ``authorized``。
        data_path: 本地宽表路径，仓库外数据由调用方持有。
        sha256: 数据文件 SHA-256，用于锁定实验输入。
        variable_mapping: 业务变量到宽表列名的映射。
        time_start: 数据起始时间。
        time_end: 数据结束时间。
        history_window: 输入窗口长度。
        forecast_horizon: 预测步长。
        split_strategy: 切分策略，当前只允许患者级留出。
        test_fraction: 患者留出比例。
        seed: 全流程随机种子。
        n_subjects: 数据提供方声明的患者数；为空时由管道实际统计。
    """

    source: str
    version: str
    authorization: str
    data_path: Path
    sha256: str
    variable_mapping: dict[str, str]
    time_start: datetime
    time_end: datetime
    history_window: int
    forecast_horizon: int
    split_strategy: str
    test_fraction: float
    seed: int
    n_subjects: int | None = None


def _parse_datetime(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field_name} 必须是 ISO-8601 字符串")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManifestError(f"{field_name} 不是合法 ISO-8601 时间: {value}") from exc


def _parse_positive_int(value: Any, field_name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ManifestError(f"{field_name} 必须是不小于 {minimum} 的整数")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> P03Manifest:
    """加载并校验 P0-3 manifest。

    Args:
        path: manifest JSON 路径。

    Returns:
        通过全部准入校验的 manifest。

    Raises:
        ManifestError: 授权、来源、哈希、schema 或窗口参数不合规。
    """
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"manifest 不存在: {path}")
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"manifest 无法读取或解析: {path}") from exc
    if not isinstance(raw, dict):
        raise ManifestError("manifest 顶层必须是 JSON object")

    required_fields = {
        "source",
        "version",
        "authorization",
        "data_path",
        "sha256",
        "variable_mapping",
        "time_start",
        "time_end",
        "history_window",
        "forecast_horizon",
        "split_strategy",
        "test_fraction",
        "seed",
    }
    missing_fields = sorted(required_fields - set(raw))
    if missing_fields:
        raise ManifestError(f"manifest 缺少字段: {', '.join(missing_fields)}")

    source = raw["source"]
    if source not in ALLOWED_SOURCES:
        raise ManifestError(f"source 只允许 {sorted(ALLOWED_SOURCES)}，当前为 {source!r}")
    authorization = raw["authorization"]
    if authorization != "authorized":
        raise ManifestError("数据授权未确认，禁止运行 P0-3")

    version = raw["version"]
    if not isinstance(version, str) or not version.strip():
        raise ManifestError("version 必须是非空字符串")

    raw_data_path = raw["data_path"]
    if not isinstance(raw_data_path, str) or not raw_data_path.strip():
        raise ManifestError("data_path 必须是非空本地路径")
    if not raw_data_path.endswith((".csv", ".csv.gz", ".parquet")):
        raise ManifestError("data_path 只支持 .csv、.csv.gz 或 .parquet")
    data_path = (path.parent / raw_data_path).resolve()
    if not data_path.is_file():
        raise ManifestError(f"数据文件不存在: {data_path}")

    expected_sha256 = raw["sha256"]
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ManifestError("sha256 必须是 64 位十六进制字符串")
    try:
        actual_sha256 = _sha256_file(data_path)
    except OSError as exc:
        raise ManifestError(f"数据文件无法读取: {data_path}") from exc
    if actual_sha256 != expected_sha256.lower():
        raise ManifestError(f"数据 SHA-256 不匹配: expected={expected_sha256.lower()} actual={actual_sha256}")

    variable_mapping_raw = raw["variable_mapping"]
    if not isinstance(variable_mapping_raw, dict):
        raise ManifestError("variable_mapping 必须是 string -> string 映射")
    variable_mapping: dict[str, str] = {}
    for key, value in variable_mapping_raw.items():
        if not isinstance(key, str) or not isinstance(value, str) or not key or not value:
            raise ManifestError("variable_mapping 的键值都必须是非空字符串")
        variable_mapping[key] = value
    missing_variables = [key for key in REQUIRED_VITAL_KEYS if key not in variable_mapping]
    if missing_variables:
        raise ManifestError(f"variable_mapping 缺少生理变量: {', '.join(missing_variables)}")
    if len(set(variable_mapping.values())) != len(variable_mapping):
        raise ManifestError("variable_mapping 不允许把不同变量映射到同一列")

    time_start = _parse_datetime(raw["time_start"], "time_start")
    time_end = _parse_datetime(raw["time_end"], "time_end")
    if time_start >= time_end:
        raise ManifestError("time_start 必须早于 time_end")

    split_strategy = raw["split_strategy"]
    if split_strategy != REQUIRED_SPLIT_STRATEGY:
        raise ManifestError(f"split_strategy 只允许 {REQUIRED_SPLIT_STRATEGY}")
    test_fraction = raw["test_fraction"]
    if isinstance(test_fraction, bool) or not isinstance(test_fraction, (int, float)):
        raise ManifestError("test_fraction 必须是 (0, 1) 内的数值")
    if not 0.0 < float(test_fraction) < 1.0:
        raise ManifestError("test_fraction 必须在 (0, 1) 内")
    seed = _parse_positive_int(raw["seed"], "seed", 0)

    n_subjects_raw = raw.get("n_subjects")
    n_subjects: int | None = None
    if n_subjects_raw is not None:
        n_subjects = _parse_positive_int(n_subjects_raw, "n_subjects", 1)

    return P03Manifest(
        source=source,
        version=version,
        authorization=authorization,
        data_path=data_path,
        sha256=actual_sha256,
        variable_mapping=variable_mapping,
        time_start=time_start,
        time_end=time_end,
        history_window=_parse_positive_int(raw["history_window"], "history_window", 2),
        forecast_horizon=_parse_positive_int(raw["forecast_horizon"], "forecast_horizon", 1),
        split_strategy=split_strategy,
        test_fraction=float(test_fraction),
        seed=seed,
        n_subjects=n_subjects,
    )
