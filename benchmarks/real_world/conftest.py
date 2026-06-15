"""
benchmarks/real_world/conftest.py — pytest fixtures 与 CI 自动跳过逻辑

CI 环境规则:
- 无 MIMIC 数据 → 自动使用合成数据 (不阻塞)
- 无 LLM API key → 跳过 LLM 基线测试
- Ollama 不可达 → 跳过 Ollama 测试
"""

from __future__ import annotations

import os

import pytest


def _is_ci() -> bool:
    """检测是否在 CI 环境中。"""
    return bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))


def _has_mimic_data() -> bool:
    """检查 MIMIC 数据是否已下载。"""
    mimic_path = os.environ.get("MIMIC_DATA_PATH", "")
    return bool(mimic_path and os.path.exists(mimic_path))


def _has_api_key() -> bool:
    """检查是否有 LLM API key。"""
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("API2D_KEY"))


def _ollama_available() -> bool:
    """检查 Ollama 是否可达。"""
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


# ── Fixtures ──


@pytest.fixture(scope="session")
def mimic_data_path():
    """MIMIC 数据路径 (可能不存在)。"""
    return os.environ.get("MIMIC_DATA_PATH", "")


@pytest.fixture(scope="session")
def use_synthetic():
    """是否使用合成数据 (CI 或无 MIMIC 数据时为 True)。"""
    return _is_ci() or not _has_mimic_data()


@pytest.fixture(scope="session")
def benchmark_patients(use_synthetic):
    """加载患者数据集 (合成 or MIMIC)。"""
    from benchmarks.real_world.mimic_causal_benchmark import MIMICCausalBenchmark

    bench = MIMICCausalBenchmark()
    if use_synthetic:
        return bench.load_synthetic_dataset(n_patients=20, n_timesteps=48, seed=42)
    else:
        path = os.environ.get("MIMIC_DATA_PATH", "")
        return bench.load_mimic_dataset(path)


@pytest.fixture(scope="session")
def causal_benchmark():
    """MIMICCausalBenchmark 实例。"""
    from benchmarks.real_world.mimic_causal_benchmark import MIMICCausalBenchmark

    return MIMICCausalBenchmark()


# ── 跳过标记 ──

skip_without_api_key = pytest.mark.skipif(
    not _has_api_key(),
    reason="No LLM API key set (OPENAI_API_KEY / ANTHROPIC_API_KEY / API2D_KEY)",
)

skip_without_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not available at localhost:11434",
)

skip_in_ci = pytest.mark.skipif(
    _is_ci(),
    reason="Skipped in CI environment",
)
