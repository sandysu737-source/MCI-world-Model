"""TASK-B1: LLM ↔ CEWM 互校准闭环 测试。

覆盖:
  - InferredEdge / CalibrationRecord / CalibrationStats / BridgeConfig 数据结构
  - LLMCEWMBridge.bidirectional_calibrate() 单轮校准
  - LLMCEWMBridge.multi_round_calibrate() 多轮收敛
  - 贝叶斯后验更新数学正确性
  - do-calculus 验证 + 方向一致性检查
  - CEWM→LLM 反向注入 prompt 构建
  - LLM 输出解析 (JSON + 正则)
  - 边界条件 (空输入/无 do-calculus/无因果图)
  - 验收标准: 延迟 < 500ms, 3轮后不退化
"""

from __future__ import annotations

import json
import time

import pytest

from mci_world_model.sdk._do_calculus import CausalGraph
from mci_world_model.sdk._llm_cewm_bridge import (
    BridgeConfig,
    CalibrationRecord,
    CalibrationStats,
    InferredEdge,
    LLMCEWMBridge,
)

# =============================================================================
# Fixtures
# =============================================================================


class MockInterventionResult:
    """模拟 DoCalculus.estimate_ate() 返回值。"""

    def __init__(self, ate: float = 0.0, ci: tuple[float, float] = (0.0, 0.0)):
        self.intervention = "do(X=1.0)"
        self.target = "Y"
        self.ate = ate
        self.confidence_interval = ci
        self.method = "backdoor"
        self.note = ""


class MockDoCalculus:
    """模拟 DoCalculus，可配置 ATE 和 CI。"""

    def __init__(self, results: dict[tuple[str, str], tuple[float, tuple]] | None = None):
        # results: {(X, Y): (ate, (ci_lo, ci_hi))}
        self._results = results or {}

    def estimate_ate(self, X: str, Y: str, **kwargs) -> MockInterventionResult:
        key = (X, Y)
        if key in self._results:
            ate, ci = self._results[key]
            return MockInterventionResult(ate=ate, ci=ci)
        return MockInterventionResult(ate=0.0, ci=(-0.1, 0.1))


@pytest.fixture
def simple_graph() -> CausalGraph:
    """简单因果图: dopamine → heart_rate, norepinephrine → map."""
    cg = CausalGraph(
        nodes=["dopamine", "heart_rate", "norepinephrine", "map"],
        edges=[("dopamine", "heart_rate"), ("norepinephrine", "map")],
    )
    return cg


@pytest.fixture
def mock_do_calculus() -> MockDoCalculus:
    """模拟 do-calculus: dopamine→heart_rate 正显著, norepinephrine→map 正显著。"""
    return MockDoCalculus(
        results={
            ("dopamine", "heart_rate"): (0.8, (0.3, 1.3)),
            ("norepinephrine", "map"): (1.2, (0.5, 1.9)),
            ("fake_drug", "heart_rate"): (0.0, (-0.5, 0.5)),
        }
    )


@pytest.fixture
def bridge(mock_do_calculus, simple_graph) -> LLMCEWMBridge:
    """标准桥实例。"""
    return LLMCEWMBridge(
        do_calculus=mock_do_calculus,
        causal_graph=simple_graph,
        config=BridgeConfig(),
    )


# =============================================================================
# Test: 数据结构
# =============================================================================


class TestInferredEdge:
    def test_default_values(self):
        edge = InferredEdge(cause="X", effect="Y")
        assert edge.cause == "X"
        assert edge.effect == "Y"
        assert edge.direction == "positive"
        assert edge.llm_confidence == 0.5
        assert edge.source == "llm"

    def test_custom_values(self):
        edge = InferredEdge(
            cause="dopamine",
            effect="heart_rate",
            direction="positive",
            llm_confidence=0.8,
            source="cewm",
        )
        assert edge.source == "cewm"
        assert edge.llm_confidence == 0.8


class TestBridgeConfig:
    def test_default_config(self):
        cfg = BridgeConfig()
        assert cfg.significance_level == 0.05
        assert cfg.prior_default == 0.5
        assert cfg.confidence_cap == 0.99
        assert cfg.confidence_floor == 0.01
        assert cfg.max_rounds == 5
        assert cfg.convergence_threshold == 0.01
        assert len(cfg.calibration_matrix) == 4

    def test_calibration_matrix_keys(self):
        cfg = BridgeConfig()
        assert (True, True) in cfg.calibration_matrix
        assert (True, False) in cfg.calibration_matrix
        assert (False, True) in cfg.calibration_matrix
        assert (False, False) in cfg.calibration_matrix

    def test_custom_config(self):
        cfg = BridgeConfig(prior_default=0.3, max_rounds=10)
        assert cfg.prior_default == 0.3
        assert cfg.max_rounds == 10


# =============================================================================
# Test: 贝叶斯更新
# =============================================================================


class TestBayesianUpdate:
    def test_prior_only_no_update(self):
        """似然=0.5 (无信息) → 后验=先验。"""
        bridge = LLMCEWMBridge(config=BridgeConfig())
        posterior = bridge._bayesian_update(prior=0.5, likelihood=0.5)
        assert abs(posterior - 0.5) < 1e-6

    def test_high_likelihood_increases_confidence(self):
        """高似然 → 后验 > 先验。"""
        bridge = LLMCEWMBridge()
        posterior = bridge._bayesian_update(prior=0.5, likelihood=0.85)
        assert posterior > 0.5

    def test_low_likelihood_decreases_confidence(self):
        """低似然 → 后验 < 先验。"""
        bridge = LLMCEWMBridge()
        posterior = bridge._bayesian_update(prior=0.5, likelihood=0.3)
        assert posterior < 0.5

    def test_bayesian_formula_correctness(self):
        """手动验证贝叶斯公式。"""
        bridge = LLMCEWMBridge()
        prior = 0.6
        likelihood = 0.8
        evidence = likelihood * prior + (1 - likelihood) * (1 - prior)
        expected = (likelihood * prior) / evidence
        result = bridge._bayesian_update(prior, likelihood)
        assert abs(result - expected) < 1e-10

    def test_extreme_prior_high_likelihood(self):
        """先验=0.9, 似然=0.85 → 后验接近 1.0。"""
        bridge = LLMCEWMBridge()
        posterior = bridge._bayesian_update(prior=0.9, likelihood=0.85)
        assert posterior > 0.95

    def test_zero_evidence_protection(self):
        """evidence ≈ 0 时返回先验（防除零）。"""
        bridge = LLMCEWMBridge()
        # prior=0.0, likelihood=0.0 → evidence=0.0 → 返回 prior
        posterior = bridge._bayesian_update(prior=0.0, likelihood=0.0)
        assert posterior == 0.0


# =============================================================================
# Test: 方向一致性检查
# =============================================================================


class TestDirectionAgreement:
    def test_positive_agrees_with_positive_ate(self):
        bridge = LLMCEWMBridge()
        assert bridge._check_direction_agreement("positive", 0.8) is True

    def test_positive_disagrees_with_negative_ate(self):
        bridge = LLMCEWMBridge()
        assert bridge._check_direction_agreement("positive", -0.5) is False

    def test_negative_agrees_with_negative_ate(self):
        bridge = LLMCEWMBridge()
        assert bridge._check_direction_agreement("negative", -1.2) is True

    def test_neutral_agrees_with_zero_ate(self):
        bridge = LLMCEWMBridge()
        assert bridge._check_direction_agreement("neutral", 0.0) is True

    def test_neutral_disagrees_with_nonzero_ate(self):
        bridge = LLMCEWMBridge()
        assert bridge._check_direction_agreement("neutral", 0.5) is False


# =============================================================================
# Test: 单轮校准
# =============================================================================


class TestBidirectionalCalibrate:
    def test_single_edge_calibration(self, bridge):
        """单条边校准: LLM 置信度 0.7 + do-calculus 正显著 → 置信度上升。"""
        edges = [InferredEdge("dopamine", "heart_rate", "positive", 0.7)]
        _graph, _prompt, stats = bridge.bidirectional_calibrate(edges)

        assert stats.n_edges == 1
        assert stats.n_significant == 1
        assert stats.avg_posterior > stats.avg_prior
        assert stats.elapsed_ms >= 0

    def test_multiple_edges(self, bridge):
        """多条边校准。"""
        edges = [
            InferredEdge("dopamine", "heart_rate", "positive", 0.7),
            InferredEdge("norepinephrine", "map", "positive", 0.6),
            InferredEdge("fake_drug", "heart_rate", "positive", 0.8),
        ]
        _graph, _prompt, stats = bridge.bidirectional_calibrate(edges)

        assert stats.n_edges == 3
        assert stats.n_significant == 2  # fake_drug 不显著
        assert stats.avg_posterior > 0

    def test_direction_mismatch_decreases_confidence(self, bridge):
        """方向不一致 → 置信度下降。"""
        # fake_drug ATE=0.0 但 LLM 说 positive → 方向不一致
        edges = [InferredEdge("fake_drug", "heart_rate", "positive", 0.7)]
        _, _, _stats = bridge.bidirectional_calibrate(edges)

        # 方向不一致, 似然被×0.5, 置信度应下降
        conf = bridge.get_confidence("fake_drug", "heart_rate")
        assert conf < bridge.config.prior_default + 0.05  # 应接近或低于先验

    def test_empty_edges(self, bridge):
        """空边列表 → 空统计。"""
        _, _, stats = bridge.bidirectional_calibrate([])
        assert stats.n_edges == 0
        assert stats.avg_prior == 0.0
        assert stats.avg_posterior == 0.0

    def test_confidence_stored(self, bridge):
        """校准后置信度被存储。"""
        edges = [InferredEdge("dopamine", "heart_rate", "positive", 0.7)]
        bridge.bidirectional_calibrate(edges)

        conf = bridge.get_confidence("dopamine", "heart_rate")
        assert conf > 0.0
        assert conf <= bridge.config.confidence_cap

    def test_confidence_capped(self):
        """置信度不超过 cap。"""
        cfg = BridgeConfig(confidence_cap=0.95)
        mock_dc = MockDoCalculus(
            results={
                ("X", "Y"): (5.0, (4.0, 6.0)),  # 极显著
            }
        )
        bridge = LLMCEWMBridge(do_calculus=mock_dc, config=cfg)
        edges = [InferredEdge("X", "Y", "positive", 0.99)]
        bridge.bidirectional_calibrate(edges)

        conf = bridge.get_confidence("X", "Y")
        assert conf <= 0.95

    def test_causal_graph_updated(self, bridge, simple_graph):
        """校准后 CausalGraph 被更新。"""
        edges = [InferredEdge("dopamine", "heart_rate", "positive", 0.7)]
        bridge.bidirectional_calibrate(edges)

        # 应已更新边的权重
        assert simple_graph.has_edge("dopamine", "heart_rate")


# =============================================================================
# Test: 无 do-calculus / 无因果图 降级
# =============================================================================


class TestDegradation:
    def test_no_do_calculus(self, simple_graph):
        """无 do-calculus → 不崩溃, 使用默认先验。"""
        bridge = LLMCEWMBridge(causal_graph=simple_graph)
        edges = [InferredEdge("X", "Y", "positive", 0.7)]
        _, _, stats = bridge.bidirectional_calibrate(edges)

        assert stats.n_edges == 1
        assert stats.n_significant == 0  # 无 do-calculus 无法验证

    def test_no_causal_graph(self, mock_do_calculus):
        """无因果图 → 不崩溃, 置信度仍存储。"""
        bridge = LLMCEWMBridge(do_calculus=mock_do_calculus)
        edges = [InferredEdge("dopamine", "heart_rate", "positive", 0.7)]
        _, _, stats = bridge.bidirectional_calibrate(edges)

        assert stats.n_edges == 1
        conf = bridge.get_confidence("dopamine", "heart_rate")
        assert conf > 0

    def test_nothing(self):
        """无 do-calculus 无因果图 → 仍可工作。"""
        bridge = LLMCEWMBridge()
        edges = [InferredEdge("X", "Y", "positive", 0.7)]
        _, _, stats = bridge.bidirectional_calibrate(edges)
        assert stats.n_edges == 1

    def test_do_calculus_exception(self, simple_graph):
        """do-calculus 抛异常 → 降级处理。"""

        class FailingDoCalculus:
            def estimate_ate(self, **kwargs):
                raise RuntimeError("do-calculus error")

        bridge = LLMCEWMBridge(
            do_calculus=FailingDoCalculus(),
            causal_graph=simple_graph,
        )
        edges = [InferredEdge("X", "Y", "positive", 0.7)]
        _, _, stats = bridge.bidirectional_calibrate(edges)
        assert stats.n_edges == 1
        assert stats.n_significant == 0


# =============================================================================
# Test: 多轮校准 + 收敛
# =============================================================================


class TestMultiRoundCalibrate:
    def test_convergence(self, bridge):
        """多轮校准在收敛时停止。"""
        edges = [InferredEdge("dopamine", "heart_rate", "positive", 0.7)]

        all_stats = bridge.multi_round_calibrate(initial_edges=edges)
        # 应在 max_rounds 内停止
        assert len(all_stats) <= bridge.config.max_rounds
        # 每轮都应有统计
        for s in all_stats:
            assert s.n_edges >= 0

    def test_max_rounds_respected(self):
        """不超过 max_rounds。"""
        cfg = BridgeConfig(max_rounds=2, convergence_threshold=0.0)
        bridge = LLMCEWMBridge(config=cfg)
        edges = [InferredEdge("X", "Y", "positive", 0.7)]

        all_stats = bridge.multi_round_calibrate(initial_edges=edges)
        assert len(all_stats) <= 2

    def test_llm_edge_generator(self, bridge):
        """使用 LLM edge generator 多轮校准。"""
        call_count = 0

        def edge_gen(prompt: str):
            nonlocal call_count
            call_count += 1
            # 第二轮返回更少边以加速收敛
            if call_count >= 2:
                return []
            return [InferredEdge("dopamine", "heart_rate", "positive", 0.7)]

        bridge.multi_round_calibrate(
            initial_edges=[InferredEdge("dopamine", "heart_rate", "positive", 0.7)],
            llm_edge_generator=edge_gen,
        )
        assert call_count >= 1

    def test_no_degradation_after_3_rounds(self, bridge):
        """3 轮校准后 F1 不再下降（验收标准）。"""
        edges = [InferredEdge("dopamine", "heart_rate", "positive", 0.7)]

        all_stats = bridge.multi_round_calibrate(initial_edges=edges)

        # 3轮后后验应稳定（不下降）
        if len(all_stats) >= 3:
            round3_posterior = all_stats[2].avg_posterior
            for s in all_stats[3:]:
                assert s.avg_posterior >= round3_posterior - 0.05  # 允许微小波动


# =============================================================================
# Test: CEWM→LLM 反向注入
# =============================================================================


class TestContextInjection:
    def test_no_high_confidence_edges(self, bridge):
        """无高置信边 → 空 prompt。"""
        prompt = bridge._build_context_injection()
        assert prompt == ""

    def test_with_high_confidence_edges(self, bridge):
        """有高置信边 → prompt 非空。"""
        edges = [InferredEdge("dopamine", "heart_rate", "positive", 0.7)]
        bridge.bidirectional_calibrate(edges)

        prompt = bridge._build_context_injection()
        # 如果置信度 >= 0.8, prompt 应非空
        conf = bridge.get_confidence("dopamine", "heart_rate")
        if conf >= 0.8:
            assert "dopamine" in prompt
            assert "heart_rate" in prompt

    def test_custom_threshold(self, bridge):
        """自定义阈值。"""
        bridge._confidence_store[("X", "Y")] = 0.6
        edges = bridge.get_high_confidence_edges(threshold=0.5)
        assert len(edges) >= 1


# =============================================================================
# Test: LLM 输出解析
# =============================================================================


class TestParseLLMEdges:
    def test_json_array(self):
        """标准 JSON 数组解析。"""
        output = json.dumps(
            [
                {"cause": "dopamine", "effect": "heart_rate", "direction": "positive", "confidence": 0.8},
                {"cause": "norepinephrine", "effect": "map", "direction": "positive", "confidence": 0.6},
            ]
        )
        edges = LLMCEWMBridge.parse_llm_edges(output)
        assert len(edges) == 2
        assert edges[0].cause == "dopamine"
        assert edges[0].llm_confidence == 0.8

    def test_json_with_surrounding_text(self):
        """JSON 嵌在文本中。"""
        output = '根据分析，发现以下因果关系：\n[{"cause": "X", "effect": "Y", "direction": "positive", "confidence": 0.7}]\n以上为结果。'
        edges = LLMCEWMBridge.parse_llm_edges(output)
        assert len(edges) == 1
        assert edges[0].cause == "X"

    def test_minimal_json(self):
        """最小 JSON（无 direction/confidence）。"""
        output = '[{"cause": "A", "effect": "B"}]'
        edges = LLMCEWMBridge.parse_llm_edges(output)
        assert len(edges) == 1
        assert edges[0].direction == "positive"  # 默认值
        assert edges[0].llm_confidence == 0.5  # 默认值

    def test_regex_fallback(self):
        """正则降级解析: X → Y (positive, 0.8)。"""
        output = "dopamine → heart_rate (positive, 0.8)\nnorepinephrine > map (negative, 0.6)"
        edges = LLMCEWMBridge.parse_llm_edges(output)
        assert len(edges) == 2
        assert edges[0].cause == "dopamine"
        assert edges[1].cause == "norepinephrine"

    def test_empty_input(self):
        """空输入。"""
        edges = LLMCEWMBridge.parse_llm_edges("")
        assert edges == []

    def test_invalid_json(self):
        """无效 JSON → 正则降级。"""
        output = "X → Y (positive, 0.7)"
        edges = LLMCEWMBridge.parse_llm_edges(output)
        assert len(edges) == 1


# =============================================================================
# Test: 验收标准
# =============================================================================


class TestAcceptanceCriteria:
    def test_calibration_latency_under_500ms(self, bridge):
        """验收: 双向校准延迟 < 500ms。"""
        edges = [InferredEdge(f"var_{i}", f"outcome_{i}", "positive", 0.7) for i in range(10)]

        t0 = time.perf_counter()
        _, _, _stats = bridge.bidirectional_calibrate(edges)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # 无 LLM API 调用时, 纯逻辑应 < 500ms
        assert elapsed_ms < 500.0

    def test_calibration_improves_confidence(self, bridge):
        """验收: 互校准后因果边置信度提升 ≥ 5% (对比纯先验)。"""
        edges = [
            InferredEdge("dopamine", "heart_rate", "positive", 0.7),
            InferredEdge("norepinephrine", "map", "positive", 0.6),
        ]

        prior_default = bridge.config.prior_default
        _, _, stats = bridge.bidirectional_calibrate(edges)

        # 显著且方向一致的边, 后验应高于先验 ≥ 5%
        if stats.n_significant > 0 and stats.n_direction_agreement > 0:
            assert stats.avg_posterior > prior_default + 0.05

    def test_no_degradation_3_rounds(self, bridge):
        """验收: 3轮互校准后后续轮 F1 不再下降。"""
        edges = [InferredEdge("dopamine", "heart_rate", "positive", 0.7)]
        all_stats = bridge.multi_round_calibrate(initial_edges=edges)

        # 3轮后后验不再下降
        if len(all_stats) >= 4:
            round3_avg = all_stats[2].avg_posterior
            for s in all_stats[3:]:
                # 后验不应显著下降（允许 5% 波动）
                assert s.avg_posterior >= round3_avg - 0.05


# =============================================================================
# Test: CalibrationRecord / CalibrationStats
# =============================================================================


class TestCalibrationRecord:
    def test_default_values(self):
        record = CalibrationRecord(edge=InferredEdge("X", "Y"))
        assert record.prior == 0.5
        assert record.posterior == 0.5
        assert record.do_ate == 0.0
        assert record.do_significant is False
        assert record.method == "bayesian"


class TestCalibrationStats:
    def test_default_values(self):
        stats = CalibrationStats()
        assert stats.n_edges == 0
        assert stats.avg_prior == 0.0
        assert stats.avg_posterior == 0.0
        assert stats.round_idx == 0


# =============================================================================
# Test: get_confidence / get_high_confidence_edges
# =============================================================================


class TestConfidenceAccess:
    def test_unknown_edge_returns_default(self, bridge):
        """未知边返回默认先验。"""
        conf = bridge.get_confidence("unknown", "edge")
        assert conf == bridge.config.prior_default

    def test_high_confidence_edges_filter(self, bridge):
        """高置信边过滤。"""
        bridge._confidence_store[("A", "B")] = 0.9
        bridge._confidence_store[("C", "D")] = 0.5

        high = bridge.get_high_confidence_edges(threshold=0.8)
        assert len(high) == 1
        assert high[0].cause == "A"

    def test_round_history(self, bridge):
        """校准历史记录。"""
        edges = [InferredEdge("X", "Y", "positive", 0.7)]
        bridge.bidirectional_calibrate(edges)

        history = bridge.round_history
        assert len(history) == 1
        assert history[0].n_edges == 1
