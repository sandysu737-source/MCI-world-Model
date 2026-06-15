"""
MCI World Model v3.3.1 — Spectral Causal Engine 测试

覆盖 GaussianDAG, FourierCausal, GaussianDistribution, BayesianCausal。
目标: _spectral_causal.py 覆盖率从 45% → 65%+。
"""

import math

import numpy as np
import pytest

from mci_world_model.sdk._spectral_causal import (
    BayesianCausal,
    FourierCausal,
    GaussianDAG,
    GaussianDistribution,
)

# =============================================================================
# Fixtures
# =============================================================================


def _make_memories(n: int = 20) -> list[dict]:
    """构造含关键词的测试记忆集。"""
    templates = [
        "高温导致金属疲劳",
        "压力引发系统崩溃",
        "暴雨促使交通瘫痪",
        "政策调整影响市场预期",
        "温度变化影响化学反应",
        "供需关系决定价格走势",
        "算法优化提升系统性能",
        "数据质量决定模型精度",
        "能量守恒约束物理过程",
        "因果关系需要统计验证",
    ]
    memories = []
    for i in range(n):
        content = templates[i % len(templates)]
        memories.append({"id": f"mem_{i}", "content": content})
    return memories


@pytest.fixture
def dag_basic():
    """基本 GaussianDAG 实例。"""
    memories = _make_memories(15)
    return GaussianDAG(memories)


@pytest.fixture
def dag_with_energy():
    """带 energy_bus=None 的 GaussianDAG。"""
    memories = _make_memories(10)
    return GaussianDAG(memories, tfidf_index=None, energy_bus=None)


@pytest.fixture
def fourier():
    """FourierCausal 实例。"""
    return FourierCausal(energy_bus=None)


@pytest.fixture
def bayesian():
    """BayesianCausal 实例。"""
    return BayesianCausal(energy_bus=None)


# =============================================================================
# M1: GaussianDAG
# =============================================================================


class TestGaussianDAGInit:
    """GaussianDAG 初始化测试。"""

    def test_empty_memories(self):
        dag = GaussianDAG([])
        assert dag._n_effective == 0
        assert len(dag._vocab) == 0

    def test_vocab_from_content(self):
        memories = [{"id": "1", "content": "高温导致金属疲劳"}]
        dag = GaussianDAG(memories)
        assert len(dag._vocab) > 0
        # 中文字符应被提取
        assert any("\u4e00" <= ch <= "\u9fff" for ch in dag._vocab)

    def test_vocab_from_tfidf_index(self):
        index = {"词a": {1}, "词b": {2}}
        memories = [{"id": "1", "content": "test"}]
        dag = GaussianDAG(memories, tfidf_index=index)
        assert "词a" in dag._vocab

    def test_energy_stats_init(self):
        dag = _make_memories(3)
        g = GaussianDAG(dag)
        assert isinstance(g._energy_stats, dict)


class TestGaussianDAGBuildTFIDF:
    """TF-IDF 矩阵构建测试。"""

    def test_build_shape(self, dag_basic):
        mat = dag_basic.build_tfidf_matrix()
        assert mat.shape[0] == 15  # n_memories
        assert mat.shape[1] == len(dag_basic._vocab)

    def test_build_empty(self):
        dag = GaussianDAG([])
        mat = dag.build_tfidf_matrix()
        assert mat.shape == (0, 1)  # max(d, 1)

    def test_l2_normalized(self, dag_basic):
        mat = dag_basic.build_tfidf_matrix()
        norms = np.linalg.norm(mat, axis=1)
        # 非零行应接近 1.0
        nonzero = norms[norms > 0]
        np.testing.assert_allclose(nonzero, 1.0, atol=0.01)

    def test_get_vector(self, dag_basic):
        vec = dag_basic.get_vector(0)
        assert vec.shape[0] == len(dag_basic._vocab)

    def test_ensure_matrix_lazy(self):
        dag = GaussianDAG(_make_memories(5))
        assert dag._tfidf_matrix is None
        dag._ensure_matrix()
        assert dag._tfidf_matrix is not None


class TestGaussianDAGPartialCorrelation:
    """偏相关系数计算测试。"""

    def test_perfect_correlation(self, dag_basic):
        rng = np.random.RandomState(42)
        x = rng.randn(50)
        y = x + 0.01 * rng.randn(50)  # 近似完美相关
        z = rng.randn(50)
        rho, _p = dag_basic.partial_correlation(x, y, z)
        assert abs(rho) > 0.9

    def test_no_correlation(self, dag_basic):
        rng = np.random.RandomState(123)
        x = rng.randn(100)
        y = rng.randn(100)
        z = rng.randn(100)
        rho, _p = dag_basic.partial_correlation(x, y, z)
        assert abs(rho) < 0.3

    def test_degenerate_denom(self, dag_basic):
        # 使分母趋近 0: r_xz = 1.0
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 1.0, 3.0, 2.0, 4.0])
        z = x  # r_xz = 1.0
        rho, p = dag_basic.partial_correlation(x, y, z)
        assert rho == 0.0 and p == 1.0

    def test_perfect_partial_corr(self, dag_basic):
        # x 和 y 高度相关，z 独立
        rng = np.random.RandomState(99)
        x = rng.randn(50)
        y = x + 0.01 * rng.randn(50)
        z = rng.randn(50)
        rho, _p = dag_basic.partial_correlation(x, y, z)
        assert abs(rho) > 0.9


class TestGaussianDAGEnergyPrior:
    """能量先验相关测试。"""

    def test_energy_prior_boost_no_bus(self, dag_basic):
        conf, verdict = dag_basic.energy_prior_boost({"id": "1"}, {"id": "2"}, 0.5)
        assert conf == 0.5
        assert verdict == "none"

    def test_energy_prior_boost_no_etype(self, dag_with_energy):
        _conf, verdict = dag_with_energy.energy_prior_boost({"id": "1", "content": ""}, {"id": "2", "content": ""}, 0.6)
        assert verdict == "none"

    def test_infer_energy_type_direct(self, dag_basic):
        mem = {"id": "1", "energy_type": "fire"}
        assert dag_basic._infer_energy_type(mem) == "fire"

    def test_infer_energy_type_content_hash(self, dag_basic):
        mem = {"id": "1", "content": "高温导致金属疲劳"}
        etype = dag_basic._infer_energy_type(mem)
        assert etype in GaussianDAG.FIVE_ELEMENTS

    def test_infer_energy_type_empty(self, dag_basic):
        mem = {"id": "1", "content": ""}
        assert dag_basic._infer_energy_type(mem) is None

    def test_with_reflection_prior(self, dag_basic):
        prior = np.eye(15, dtype=np.float32)
        dag_basic.with_reflection_prior(prior)
        assert dag_basic._reflection_prior is not None
        assert dag_basic._reflection_prior.shape == (15, 15)

    def test_with_parametric_prior_ndarray(self, dag_basic):
        prior = np.random.rand(5, 5)
        dag_basic.with_parametric_prior(prior)
        assert dag_basic._parametric_prior is not None

    def test_with_parametric_prior_has_method(self, dag_basic):
        class FakeTopo:
            def to_flat_vector(self):
                return np.arange(25, dtype=np.float32)

        dag_basic.with_parametric_prior(FakeTopo())
        assert dag_basic._parametric_prior.shape == (5, 5)


class TestGaussianDAGDiscover:
    """隐藏因果边发现测试。"""

    def test_discover_too_few(self):
        dag = GaussianDAG([{"id": "1", "content": "only one"}])
        assert dag.discover_hidden_edges() == []

    def test_discover_returns_list(self, dag_basic):
        edges = dag_basic.discover_hidden_edges(min_correlation=0.1, max_scan=15)
        assert isinstance(edges, list)

    def test_discover_edge_structure(self, dag_basic):
        edges = dag_basic.discover_hidden_edges(min_correlation=0.05, p_threshold=0.5, max_scan=15)
        if edges:
            e = edges[0]
            assert "cause_idx" in e
            assert "effect_idx" in e
            assert "rho" in e
            assert "confidence" in e
            assert "verdict" in e

    def test_discover_with_reflection(self, dag_basic):
        n = len(dag_basic.memories)
        prior = np.random.RandomState(42).rand(n, n).astype(np.float32)
        dag_basic.with_reflection_prior(prior)
        edges = dag_basic.discover_hidden_edges(min_correlation=0.05, p_threshold=0.5, max_scan=15)
        # Should not crash
        assert isinstance(edges, list)

    def test_discover_with_parametric(self, dag_basic):
        prior = np.random.RandomState(42).rand(5, 5)
        dag_basic.with_parametric_prior(prior)
        edges = dag_basic.discover_hidden_edges(min_correlation=0.05, p_threshold=0.5, max_scan=15)
        assert isinstance(edges, list)


class TestGaussianDAGConfounder:
    """混淆因子检测测试。"""

    def test_confounder_detection(self, dag_basic):
        dag_basic._ensure_matrix()
        n = dag_basic._tfidf_matrix.shape[0]
        if n >= 3:
            result = dag_basic.detect_confounder(0, 1, 2)
            assert "is_confounder" in result
            assert "confounder_score" in result
            assert 0 <= result["confounder_score"] <= 1.0


class TestGaussianDAGStatistics:
    """统计摘要测试。"""

    def test_get_statistics(self, dag_basic):
        stats = dag_basic.get_statistics()
        assert stats["n_memories"] == 15
        assert stats["vocab_size"] == len(dag_basic._vocab)
        assert stats["energy_bus_available"] is False


# =============================================================================
# M2: FourierCausal
# =============================================================================


class TestFourierCausalInit:
    """FourierCausal 初始化测试。"""

    def test_init_no_bus(self):
        fc = FourierCausal()
        assert fc._bus is None
        assert fc._snapshot_count == 0


class TestFourierCausalSnapshot:
    """快照采集测试。"""

    def test_manual_intensities(self, fourier):
        count = fourier.record_snapshot({"wood": 0.5, "fire": 0.8})
        assert count == 1
        assert len(fourier.get_series("wood")) == 1
        assert fourier.get_series("wood")[0] == 0.5

    def test_multiple_snapshots(self, fourier):
        for i in range(10):
            fourier.record_snapshot({"wood": float(i), "fire": float(i * 2)})
        assert fourier._snapshot_count == 10
        assert len(fourier.get_series("wood")) == 10

    def test_get_series_empty(self, fourier):
        series = fourier.get_series("nonexistent")
        assert len(series) == 0


class TestFourierCausalFFT:
    """FFT 分解测试。"""

    def test_fft_insufficient_samples(self, fourier):
        fourier.record_snapshot({"wood": 0.5})
        result = fourier.fft_decompose("wood")
        assert result.get("error") == "insufficient_samples"

    def test_fft_periodic_signal(self, fourier):
        # 构造周期信号
        for i in range(20):
            fourier.record_snapshot({"wood": 0.5 + 0.3 * math.sin(2 * math.pi * i / 4)})
        result = fourier.fft_decompose("wood")
        assert result["n_samples"] == 20
        assert "dc_ratio" in result
        assert "anomaly_score" in result
        assert 0 <= result["anomaly_score"] <= 1

    def test_fft_stable_signal(self, fourier):
        # 稳定信号 → 低异常得分
        for i in range(15):
            fourier.record_snapshot({"wood": 0.5})
        result = fourier.fft_decompose("wood")
        assert result["anomaly_score"] < 0.5


class TestFourierCausalEvents:
    """因果事件检测测试。"""

    def test_detect_events_empty(self, fourier):
        events = fourier.detect_causal_events()
        assert events == []

    def test_detect_events_with_data(self, fourier):
        rng = np.random.RandomState(42)
        for i in range(20):
            intensities = {e: float(rng.rand()) for e in FourierCausal.FIVE_ELEMENTS}
            fourier.record_snapshot(intensities)
        events = fourier.detect_causal_events(threshold=0.01)
        assert isinstance(events, list)


class TestFourierCausalCoherence:
    """双元素频域相干性测试。"""

    def test_insufficient_samples(self, fourier):
        result = fourier.cross_spectral_coherence("wood", "fire")
        assert result["error"] == "insufficient_samples"

    def test_synchronized_signal(self, fourier):
        for i in range(20):
            val = 0.5 + 0.3 * math.sin(2 * math.pi * i / 5)
            fourier.record_snapshot({"wood": val, "fire": val})
        result = fourier.cross_spectral_coherence("wood", "fire")
        assert "coherence" in result
        assert "sync_band" in result


class TestFourierCausalFiltering:
    """周期过滤测试。"""

    def test_filter_periodic_noise(self, fourier):
        for i in range(20):
            val = 0.5 + 0.3 * math.sin(2 * math.pi * i / 4)
            fourier.record_snapshot(dict.fromkeys(FourierCausal.FIVE_ELEMENTS, val))
        corr = np.eye(5)
        filtered = fourier.filter_periodic_noise(corr, cutoff=0.5)
        assert filtered.shape == (5, 5)

    def test_filter_periodic_edges(self, fourier):
        dag = GaussianDAG(_make_memories(10))
        edges = [{"cause_idx": 0, "effect_idx": 1, "confidence": 0.8, "verdict": "confirmed"}]
        result = fourier.filter_periodic_edges(edges, dag)
        assert isinstance(result, list)


class TestFourierCausalBalance:
    """频谱平衡报告测试。"""

    def test_balance_report_empty(self, fourier):
        report = fourier.spectral_balance_report()
        assert report["health_status"] == "healthy"
        assert report["global_anomaly_score"] == 0.0

    def test_balance_report_with_data(self, fourier):
        rng = np.random.RandomState(42)
        for i in range(20):
            intensities = {e: float(rng.rand()) for e in FourierCausal.FIVE_ELEMENTS}
            fourier.record_snapshot(intensities)
        report = fourier.spectral_balance_report()
        assert report["health_status"] in ("healthy", "warning", "critical")
        assert "per_element" in report
        assert len(report["per_element"]) == 5


# =============================================================================
# M3: GaussianDistribution
# =============================================================================


class TestGaussianDistribution:
    """高斯分布测试。"""

    def test_default(self):
        g = GaussianDistribution()
        assert g.mu == 0.0
        assert g.sigma == 1.0
        assert g.n_observations == 0

    def test_properties(self):
        g = GaussianDistribution(mu=2.0, sigma=0.5)
        assert g.mean == 2.0
        assert g.variance == 0.25
        assert abs(g.precision - 4.0) < 0.01

    def test_pdf_at_mean(self):
        g = GaussianDistribution(mu=0.0, sigma=1.0)
        assert abs(g.pdf(0.0) - 0.3989) < 0.01

    def test_cdf_at_mean(self):
        g = GaussianDistribution(mu=0.0, sigma=1.0)
        assert abs(g.cdf(0.0) - 0.5) < 0.01

    def test_credible_interval(self):
        g = GaussianDistribution(mu=0.0, sigma=1.0)
        lo, hi = g.credible_interval(0.95)
        assert lo < 0.0 < hi
        assert abs(lo - (-1.96)) < 0.1

    def test_update(self):
        g = GaussianDistribution(mu=0.0, sigma=10.0)  # 弱先验
        g.update(sample_mean=5.0, sample_std=1.0, n=100)
        # 强数据 → 后验应接近样本均值
        assert abs(g.mu - 5.0) < 0.5
        assert g.n_observations == 100

    def test_update_n_zero(self):
        g = GaussianDistribution(mu=1.0, sigma=2.0)
        result = g.update(0.0, 0.0, 0)
        assert result.mu == 1.0  # 不应改变

    def test_serialization(self):
        g = GaussianDistribution(mu=3.0, sigma=0.5, n_observations=50)
        d = g.to_dict()
        g2 = GaussianDistribution.from_dict(d)
        assert g2.mu == g.mu
        assert g2.sigma == g.sigma
        assert g2.n_observations == g.n_observations


# =============================================================================
# M3: BayesianCausal
# =============================================================================


class TestBayesianCausalInit:
    """BayesianCausal 初始化测试。"""

    def test_init(self, bayesian):
        assert bayesian._bus is None
        assert len(bayesian._posteriors) == 0
        assert len(bayesian._test_history) == 0


class TestBayesianCausalHypothesisTest:
    """贝叶斯假设检验测试。"""

    def test_strong_effect(self, bayesian):
        result = bayesian.causal_hypothesis_test(edge_id="0_1", rho=0.8, n_samples=100, energy_relation="enhance")
        assert "posterior_mean" in result
        assert "bayes_factor" in result
        assert result["energy_prior_used"] == "enhance"

    def test_weak_effect(self, bayesian):
        result = bayesian.causal_hypothesis_test(edge_id="0_1", rho=0.01, n_samples=10, energy_relation=None)
        assert result["conclusion"] in (
            "inconclusive",
            "evidence_for_no_causal",
        )

    def test_perfect_rho(self, bayesian):
        result = bayesian.causal_hypothesis_test(edge_id="0_1", rho=0.999999, n_samples=100)
        assert result["bayes_factor"] == float("inf") or result["bayes_factor"] > 100

    def test_weak_prior(self, bayesian):
        result = bayesian.causal_hypothesis_test(edge_id="0_1", rho=0.5, n_samples=50, prior_strength="weak")
        assert "credible_interval_95" in result

    def test_suppress_prior(self, bayesian):
        result = bayesian.causal_hypothesis_test(edge_id="0_1", rho=0.3, n_samples=50, energy_relation="suppress")
        assert result["energy_prior_used"] == "suppress"


class TestBayesianCausalBatch:
    """批量更新测试。"""

    def test_batch_update(self, bayesian):
        edges = [
            {"cause_idx": 0, "effect_idx": 1, "rho": 0.6, "energy_relation": "enhance"},
            {"cause_idx": 2, "effect_idx": 3, "rho": 0.1, "energy_relation": None},
        ]
        result = bayesian.batch_update(edges)
        assert len(result) == 2
        assert "posterior_mean" in result[0]


class TestBayesianCausalSummary:
    """摘要测试。"""

    def test_summary_empty(self, bayesian):
        summary = bayesian.get_summary()
        assert summary["n_edges_tested"] == 0

    def test_summary_after_test(self, bayesian):
        bayesian.causal_hypothesis_test("0_1", 0.7, 100, "enhance")
        bayesian.causal_hypothesis_test("2_3", 0.05, 10)
        summary = bayesian.get_summary()
        assert summary["n_edges_tested"] == 2
        assert len(summary["edges"]) == 2


class TestBayesianCausalCompare:
    """假设比较测试。"""

    def test_compare_missing_posteriors(self, bayesian):
        result = bayesian.compare_hypotheses("a", "b")
        assert result["favored"] == "unknown"

    def test_compare_two_edges(self, bayesian):
        bayesian.causal_hypothesis_test("edge_a", 0.8, 100)
        bayesian.causal_hypothesis_test("edge_b", 0.1, 100)
        result = bayesian.compare_hypotheses("edge_a", "edge_b")
        assert result["favored"] == "edge_a"
        assert result["confidence"] in ("strong", "moderate")
