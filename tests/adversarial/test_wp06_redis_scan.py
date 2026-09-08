"""WP-06 Redis SCAN 预算与可用性对抗测试。"""

from __future__ import annotations

import sys
import time
import types
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest

from mci_world_model.server.storage import RedisStorage

pytestmark = pytest.mark.contract


class FakeRedisClient:
    def __init__(self, keys: list[str] | None = None) -> None:
        self.keys = keys
        self.set_calls: list[tuple[str, str]] = []
        self.setex_calls: list[tuple[str, int, str]] = []
        self.scan_calls: list[tuple[str, int]] = []

    def ping(self) -> bool:
        return True

    def set(self, key: str, value: str) -> None:
        self.set_calls.append((key, value))

    def setex(self, key: str, ttl_seconds: int, value: str) -> None:
        self.setex_calls.append((key, ttl_seconds, value))

    def scan_iter(self, match: str, count: int) -> Iterator[str]:
        self.scan_calls.append((match, count))
        if self.keys is None:
            for index in range(100_000):
                yield f"mci:diagnosis:tenant-a:patient-a:{index:06d}"
        else:
            import fnmatch

            yield from (key for key in self.keys if fnmatch.fnmatchcase(key, match))


@pytest.fixture
def redis_storage(monkeypatch: pytest.MonkeyPatch) -> RedisStorage:
    fake_module = types.ModuleType("redis")
    fake_module.from_url = lambda *args, **kwargs: FakeRedisClient()
    monkeypatch.setitem(sys.modules, "redis", fake_module)
    return RedisStorage(
        "redis://localhost:6379",
        scan_page_size=25,
        max_scan_keys=100,
        max_scan_seconds=1.0,
    )


def test_scan_uses_page_budget_and_never_calls_keys(
    redis_storage: RedisStorage,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        keys = redis_storage.list_keys("diagnosis:tenant-a:patient-a:")

    assert len(keys) == 100
    assert redis_storage._client.scan_calls == [("mci:diagnosis:tenant-a:patient-a:*", 25)]
    assert keys[0].endswith("000000")
    assert keys[-1].endswith("000099")
    assert "max_scan_keys=100" in caplog.text


def test_scan_time_budget_returns_partial_result(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class SlowRedisClient(FakeRedisClient):
        def scan_iter(self, match: str, count: int) -> Iterator[str]:
            time.sleep(0.01)
            for index in range(1000):
                yield f"mci:diagnosis:{index:04d}"

    fake_module = types.ModuleType("redis")
    fake_module.from_url = lambda *args, **kwargs: SlowRedisClient()
    monkeypatch.setitem(sys.modules, "redis", fake_module)
    storage = RedisStorage(
        "redis://localhost:6379",
        scan_page_size=10,
        max_scan_keys=1000,
        max_scan_seconds=0.001,
    )

    with caplog.at_level("WARNING"):
        keys = storage.list_keys("diagnosis:")

    assert len(keys) < 1000
    assert "max_scan_seconds=0.001" in caplog.text


def test_scan_prefix_isolation_and_glob_safety(monkeypatch: pytest.MonkeyPatch) -> None:
    keys = [
        "mci:diagnosis:tenant-a:patient-a:1",
        "mci:diagnosis:tenant-a:patient-b:1",
        "mci:other:tenant-a:patient-a:1",
    ]
    fake_module = types.ModuleType("redis")
    fake_module.from_url = lambda *args, **kwargs: FakeRedisClient(keys=keys)
    monkeypatch.setitem(sys.modules, "redis", fake_module)
    storage = RedisStorage("redis://localhost:6379")

    assert storage.list_keys("diagnosis:tenant-a:patient-a:") == ["diagnosis:tenant-a:patient-a:1"]
    assert storage.list_keys("diagnosis:tenant-a:patient-*") == []


def test_concurrent_scans_return_consistent_partial_results(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType("redis")
    fake_module.from_url = lambda *args, **kwargs: FakeRedisClient()
    monkeypatch.setitem(sys.modules, "redis", fake_module)
    storage = RedisStorage(
        "redis://localhost:6379",
        scan_page_size=50,
        max_scan_keys=200,
        max_scan_seconds=1.0,
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: storage.list_keys("diagnosis:tenant-a:patient-a:"), range(8)))

    assert all(len(result) == 200 for result in results)
    assert all(result == results[0] for result in results)


def test_save_supports_ttl_and_isolated_prefix(redis_storage: RedisStorage) -> None:
    value = {"confidence": 0.9}

    redis_storage.save("diagnosis:tenant-a:patient-a:record", value)
    redis_storage.save("diagnosis:tenant-a:patient-a:ttl-record", value, ttl_seconds=60)

    assert redis_storage._client.set_calls == [("mci:diagnosis:tenant-a:patient-a:record", '{"confidence": 0.9}')]
    assert redis_storage._client.setex_calls == [
        ("mci:diagnosis:tenant-a:patient-a:ttl-record", 60, '{"confidence": 0.9}')
    ]


def test_scan_budget_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType("redis")
    fake_module.from_url = lambda *args, **kwargs: FakeRedisClient()
    monkeypatch.setitem(sys.modules, "redis", fake_module)

    with pytest.raises(ValueError, match="正数"):
        RedisStorage("redis://localhost:6379", max_scan_keys=0)
