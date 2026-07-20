"""ClinicalCausalDiscovery 单元测试 — D4 数据驱动因果发现验证。

验证从患者体征时序数据发现因果结构的核心契约：
    1. 算法正确性：相关变量相连，独立变量不连
    2. 数值健壮：小样本/常数列/NaN 不崩溃
    3. 决策引擎集成：discover_causal_structure 接口可用
    4. 边界合规：不依赖持久化（su-memory 隔离）
"""

from __future__ import annotations

import numpy as np

from mci_world_model.sdk._clinical_causal_discovery import (
    CausalStructure,
    ClinicalCausalDiscovery,
)

SEED = 42


def make_correlated_data(n=100, seed=SEED):
    """构造有真实因果链的合成数据: HR→SBP→DBP。"""
    rng = np.random.default_rng(seed)
    hr = rng.normal(80, 10, n)
    sbp = 120 + 0.6 * hr + rng.normal(0, 3, n)
    dbp = 80 + 0.4 * sbp + rng.normal(0, 2, n)
    spo2 = rng.normal(98, 1, n)
    rr = rng.normal(16, 2, n)
    temp = rng.normal(36.8, 0.3, n)
    gcs = np.full(n, 15.0)
    return np.stack([hr, sbp, dbp, spo2, rr, temp, gcs], axis=1)


# =============================================================================
# 1. 算法正确性
# =============================================================================


class TestCausalStructureLearning:
    """验证因果结构学习正确性。"""

    def test_correlated_variables_connected(self):
        """相关变量在骨架中相连。"""
        data = make_correlated_data(n=100)
        discovery = ClinicalCausalDiscovery(significance=0.05, min_samples=10)
        struct = discovery.discover(data, max_conditioning_size=1)
        # HR 和 SBP 强相关，应在 links 中（某方向）
        connected = {(link.cause, link.effect) for link in struct.links}
        related_pair = ("heart_rate", "systolic_bp") in connected or (
            "systolic_bp",
            "heart_rate",
        ) in connected
        assert related_pair, "HR-SBP 应因果相连"

    def test_independent_variables_disconnected(self):
        """独立变量在骨架中不相连。"""
        data = make_correlated_data(n=100)
        discovery = ClinicalCausalDiscovery(significance=0.05, min_samples=10)
        struct = discovery.discover(data, max_conditioning_size=1)
        # SPO2 与 HR/SBP/DBP 独立，不应相连
        for link in struct.links:
            pair = {link.cause, link.effect}
            assert "spo2" not in pair or pair == {"spo2"}, f"SPO2 不应与 {pair} 相连"

    def test_links_sorted_by_strength(self):
        """links 按 strength 降序排列。"""
        data = make_correlated_data(n=100)
        struct = ClinicalCausalDiscovery().discover(data, max_conditioning_size=1)
        strengths = [link.strength for link in struct.links]
        assert strengths == sorted(strengths, reverse=True)

    def test_strength_in_valid_range(self):
        """strength ∈ [0, 1]。"""
        data = make_correlated_data(n=100)
        struct = ClinicalCausalDiscovery().discover(data, max_conditioning_size=1)
        for link in struct.links:
            assert 0.0 <= link.strength <= 1.0
            assert -1.0 <= link.direction <= 1.0
            assert 0.0 <= link.p_value <= 1.0


# =============================================================================
# 2. 数值健壮性
# =============================================================================


class TestNumericRobustness:
    """验证数值健壮性。"""

    def test_insufficient_samples_returns_empty(self):
        """样本不足返回空结构（避免过拟合）。"""
        data = make_correlated_data(n=5)  # < min_samples=10
        struct = ClinicalCausalDiscovery(min_samples=10).discover(data)
        assert len(struct.links) == 0
        assert struct.method == "insufficient_data"

    def test_constant_column_handled(self):
        """常数列（如 GCS=15）不崩溃。"""
        data = make_correlated_data(n=100)
        struct = ClinicalCausalDiscovery().discover(data, max_conditioning_size=1)
        # 不应崩溃，且 GCS（常数）不应出现在 links 中
        for link in struct.links:
            assert "gcs" not in (link.cause, link.effect), "常数列 GCS 不应产生因果边"

    def test_nan_input_handled(self):
        """含 NaN 的输入不崩溃（被相关系数计算过滤）。"""
        data = make_correlated_data(n=50)
        data[0, 0] = np.nan
        struct = ClinicalCausalDiscovery().discover(data, max_conditioning_size=1)
        assert isinstance(struct, CausalStructure)

    def test_single_sample_handled(self):
        """单样本不崩溃。"""
        data = np.array([[80.0, 120, 80, 98, 16, 36.8, 15]])
        struct = ClinicalCausalDiscovery().discover(data)
        assert struct.n_samples == 1
        assert len(struct.links) == 0

    def test_degenerate_data_returns_empty(self):
        """全常数数据返回空结构。"""
        data = np.full((20, 7), 80.0)
        struct = ClinicalCausalDiscovery().discover(data)
        assert struct.method == "degenerate_data" or len(struct.links) == 0


# =============================================================================
# 3. 序列化
# =============================================================================


class TestSerialization:
    """验证 to_dict 序列化。"""

    def test_to_dict_structure(self):
        """to_dict 返回完整结构。"""
        data = make_correlated_data(n=100)
        struct = ClinicalCausalDiscovery().discover(data, max_conditioning_size=1)
        d = struct.to_dict()
        assert "links" in d
        assert "n_samples" in d
        assert "method" in d
        assert "n_strong_links" in d
        assert d["n_samples"] == 100
        assert isinstance(d["links"], list)


# =============================================================================
# 4. 决策引擎集成
# =============================================================================


class TestDecisionEngineIntegration:
    """验证决策引擎的 discover_causal_structure 接口。"""

    def test_decision_engine_has_discover_method(self):
        """ClinicalDecisionEngine 有 discover_causal_structure 方法。"""
        from mci_world_model.sdk import ClinicalDecisionEngine

        engine = ClinicalDecisionEngine()
        assert hasattr(engine, "discover_causal_structure")
        assert callable(engine.discover_causal_structure)

    def test_discover_causal_structure_returns_dict(self):
        """决策引擎的因果发现返回字典。"""
        from mci_world_model.sdk import ClinicalDecisionEngine

        engine = ClinicalDecisionEngine()
        data = make_correlated_data(n=100)
        result = engine.discover_causal_structure(data, max_conditioning_size=1)
        assert isinstance(result, dict)
        assert "links" in result
        assert "n_samples" in result

    def test_decision_engine_no_persistence(self):
        """验证因果发现不引入持久化（su-memory 边界）。

        确认 ClinicalCausalDiscovery 不持有跨调用状态（无文件/DB/缓存属性）。
        """
        discovery = ClinicalCausalDiscovery()
        # 不应有持久化相关属性
        persist_attrs = [
            a for a in dir(discovery) if any(kw in a.lower() for kw in ["cache", "store", "db", "file", "persist"])
        ]
        assert persist_attrs == [], f"发现持久化属性: {persist_attrs}"
