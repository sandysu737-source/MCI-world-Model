"""WP-02 医疗授权与存储最小权限反向测试。"""

from __future__ import annotations

import os
import stat

import pytest

from mci_world_model.server.security import Identity
from mci_world_model.server.storage import (
    FileStorage,
    list_diagnoses,
    load_diagnosis,
    save_diagnosis,
)


@pytest.fixture
def storage(tmp_path, monkeypatch: pytest.MonkeyPatch) -> FileStorage:
    """隔离全局存储单例，确保反向测试只作用于临时目录。"""
    import mci_world_model.server.storage as storage_mod

    monkeypatch.setenv("MCI_ENV", "test")
    monkeypatch.setenv("MCI_STORAGE_PATH", str(tmp_path))
    storage = FileStorage(str(tmp_path))
    monkeypatch.setattr(storage_mod, "_storage", storage)
    return storage


def _identity(tenant_id: str, patient_id: str) -> Identity:
    return Identity(subject="subject", tenant_id=tenant_id, patient_ids=frozenset({patient_id}))


def test_patient_isolation(storage: FileStorage) -> None:
    """同租户下患者 A 的记录不能被患者 B 列出或读取。"""
    owner = _identity("tenant", "P001")
    other = _identity("tenant", "P002")
    record_id = save_diagnosis("P001", {"confidence": 0.9}, owner)

    assert load_diagnosis(record_id, other) is None
    assert list_diagnoses("P002", other) == []
    assert list_diagnoses("P001", owner) == [record_id]
    assert storage.load(record_id)["patient_id"] == "P001"


def test_empty_patient_does_not_expand_access() -> None:
    """空 patient_id 不能写入，也不能展开全部记录。"""
    identity = _identity("tenant", "P001")

    with pytest.raises(PermissionError, match="授权范围"):
        save_diagnosis("", {"confidence": 0.9}, identity)
    assert list_diagnoses("", identity) == []


def test_unauthorized_patient_cannot_write() -> None:
    """请求体患者 ID 不在身份映射中时，存储层必须拒绝写入。"""
    identity = _identity("tenant", "P001")

    with pytest.raises(PermissionError, match="授权范围"):
        save_diagnosis("P002", {"confidence": 0.9}, identity)


def test_storage_permissions_are_minimal(storage: FileStorage, tmp_path) -> None:
    """存储目录 0700，记录文件 0600。"""
    record_id = save_diagnosis("P001", {"confidence": 0.9}, _identity("tenant", "P001"))
    path = storage._safe_path(record_id)

    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_path_traversal_key_is_rejected(storage: FileStorage) -> None:
    """路径遍历 key 不得净化后落盘，必须直接失败。"""
    with pytest.raises(ValueError, match="非法存储 key"):
        storage.save("../outside", {"confidence": 0.9})


def test_symlink_load_is_rejected(storage: FileStorage) -> None:
    """符号链接不得作为读取入口。"""
    record_id = save_diagnosis("P001", {"confidence": 0.9}, _identity("tenant", "P001"))
    os.symlink(storage._safe_path(record_id), storage._safe_path("link"))

    with pytest.raises(ValueError, match="非法存储路径"):
        storage.load("link")


def test_production_rejects_system_temp_storage(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """生产环境显式配置系统临时目录时启动失败。"""
    import mci_world_model.server.storage as storage_mod

    monkeypatch.setenv("MCI_ENV", "production")
    monkeypatch.setenv("MCI_STORAGE_PATH", str(tmp_path))
    monkeypatch.setattr(storage_mod, "_storage", None)

    with pytest.raises(ValueError, match="系统临时目录"):
        storage_mod.get_storage()
