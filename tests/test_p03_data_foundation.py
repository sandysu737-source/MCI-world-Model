"""P0-3a 数据清单与患者级窗口管道回归。"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.real_world.p03_manifest import ManifestError, load_manifest
from benchmarks.real_world.p03_timeseries import P03PipelineError, load_windows

COLUMNS = (
    "subject_id",
    "charttime",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "oxygen_saturation",
    "respiratory_rate",
    "temperature",
    "gcs",
)
MAPPING = {
    "hr": "heart_rate",
    "sbp": "systolic_bp",
    "dbp": "diastolic_bp",
    "spo2": "oxygen_saturation",
    "rr": "respiratory_rate",
    "temp": "temperature",
    "gcs": "gcs",
}
BASE_TIME = "2020-01-01T00:{minute:02d}:00"


def _normal_row(subject_id: int, minute: int) -> str:
    return f"{subject_id},{BASE_TIME.format(minute=minute)},80,120,80,98,16,36.8,15"


def _write_csv(path: Path, rows: list[str], compress: bool = False) -> Path:
    content = "\n".join([",".join(COLUMNS), *rows]) + "\n"
    if compress:
        target = path.with_suffix(".csv.gz")
        with gzip.open(target, "wt", encoding="utf-8", newline="") as handle:
            handle.write(content)
        return target
    path.write_text(content, encoding="utf-8")
    return path


def _write_manifest(data_path: Path, tmp_path: Path, n_subjects: int | None = 4) -> Path:
    manifest = {
        "source": "mimic",
        "version": "unit-test-v1",
        "authorization": "authorized",
        "data_path": data_path.name,
        "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "variable_mapping": MAPPING,
        "time_start": "2020-01-01T00:00:00",
        "time_end": "2020-01-01T00:30:00",
        "history_window": 2,
        "forecast_horizon": 1,
        "split_strategy": "patient_holdout",
        "test_fraction": 0.5,
        "seed": 42,
    }
    if n_subjects is not None:
        manifest["n_subjects"] = n_subjects
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _unit_rows() -> list[str]:
    rows = [_normal_row(1, minute) for minute in (0, 5, 10, 15)]
    rows.extend(
        [
            f"2,{BASE_TIME.format(minute=0)},,120,80,98,16,36.8,15",
            f"2,{BASE_TIME.format(minute=5)},82,121,81,98,16,36.9,15",
            f"2,{BASE_TIME.format(minute=5)},83,122,82,98,16,36.9,15",
            f"2,{BASE_TIME.format(minute=10)},84,123,83,98,16,37.0,15",
            f"2,{BASE_TIME.format(minute=15)},85,124,84,98,16,37.0,15",
            f"3,{BASE_TIME.format(minute=0)},80,120,80,98,16,36.8,15",
            f"3,{BASE_TIME.format(minute=10)},300,125,85,98,16,37.1,15",
            f"3,{BASE_TIME.format(minute=5)},82,122,82,98,16,36.9,15",
            f"3,{BASE_TIME.format(minute=15)},84,124,84,98,16,37.0,15",
            _normal_row(4, 0),
            _normal_row(4, 5),
        ]
    )
    return rows


def test_load_windows_creates_patient_level_split(tmp_path: Path) -> None:
    data_path = _write_csv(tmp_path / "wide.csv", _unit_rows())
    manifest_path = _write_manifest(data_path, tmp_path)
    manifest = load_manifest(manifest_path)

    split = load_windows(manifest)

    assert set(split.train_subject_ids).isdisjoint(split.test_subject_ids)
    assert split.train_inputs.shape[1:] == (2, 7)
    assert split.train_targets.shape[1:] == (1, 7)
    assert split.test_inputs.shape[1:] == (2, 7)
    assert split.test_targets.shape[1:] == (1, 7)
    assert split.stats.n_subjects == 3
    assert split.stats.dropped_short_subjects == 1
    assert split.stats.duplicate_count == 1
    assert split.stats.out_of_range_count == 1
    assert split.stats.out_of_order_count == 1
    assert np.isfinite(split.train_inputs).all()
    assert np.isfinite(split.test_targets).all()


def test_load_windows_is_reproducible(tmp_path: Path) -> None:
    data_path = _write_csv(tmp_path / "wide.csv", _unit_rows())
    manifest = load_manifest(_write_manifest(data_path, tmp_path))

    first = load_windows(manifest)
    second = load_windows(manifest)

    assert first.train_subject_ids == second.train_subject_ids
    assert first.test_subject_ids == second.test_subject_ids
    np.testing.assert_array_equal(first.train_inputs, second.train_inputs)
    np.testing.assert_array_equal(first.test_targets, second.test_targets)


def test_load_windows_supports_gzip(tmp_path: Path) -> None:
    data_path = _write_csv(tmp_path / "wide.csv", _unit_rows(), compress=True)
    manifest = load_manifest(_write_manifest(data_path, tmp_path))

    split = load_windows(manifest)

    assert split.stats.n_subjects == 3


def test_manifest_rejects_wrong_sha256(tmp_path: Path) -> None:
    data_path = _write_csv(tmp_path / "wide.csv", _unit_rows())
    manifest_path = _write_manifest(data_path, tmp_path)
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_data["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(ManifestError, match="SHA-256 不匹配"):
        load_manifest(manifest_path)


def test_manifest_rejects_unauthorized_data(tmp_path: Path) -> None:
    data_path = _write_csv(tmp_path / "wide.csv", _unit_rows())
    manifest_path = _write_manifest(data_path, tmp_path)
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_data["authorization"] = "unknown"
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(ManifestError, match="授权未确认"):
        load_manifest(manifest_path)


def test_pipeline_rejects_missing_required_columns(tmp_path: Path) -> None:
    data_path = _write_csv(tmp_path / "wide.csv", _unit_rows())
    manifest_path = _write_manifest(data_path, tmp_path)
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_data["variable_mapping"] = {**MAPPING, "gcs": "missing_column"}
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    manifest = load_manifest(manifest_path)

    with pytest.raises(P03PipelineError, match="缺少必填列"):
        load_windows(manifest)
