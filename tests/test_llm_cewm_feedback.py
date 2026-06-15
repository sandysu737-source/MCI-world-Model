"""Phase 2 LLM↔CEWM 反馈闭环测试 — CounterfactualOracle + CF_QUERY.

覆盖:
    - CFScenario / CFRanking 数据类
    - CounterfactualOracle.batch_what_if() — 批量反事实推演
    - CounterfactualOracle.rank_scenarios() — 排序
    - CounterfactualOracle.query() — 完整查询流程
    - OrchestratorBridge CF_QUERY intent
    - 降级路径（无引擎、无 world_model）
"""

from mci_world_model.sdk._counterfactual_oracle import (
    CFRanking,
    CFScenario,
    CounterfactualOracle,
)
from mci_world_model.sdk._orchestrator_bridge import OrchestratorBridge

# =============================================================================
# CFScenario / CFRanking
# =============================================================================


class TestCFScenario:
    """CFScenario 数据类测试。"""

    def test_creation(self):
        s = CFScenario(name="方案A", intervention={"treatment": "A"}, target="outcome")
        assert s.name == "方案A"
        assert s.intervention == {"treatment": "A"}
        assert s.target == "outcome"

    def test_default_values(self):
        s = CFScenario(name="test")
        assert s.description == ""
        assert s.intervention == {}
        assert s.target == ""

    def test_description(self):
        s = CFScenario(name="B", description="方案B描述")
        assert s.description == "方案B描述"


class TestCFRanking:
    """CFRanking 数据类测试。"""

    def test_creation(self):
        s = CFScenario(name="A")
        r = CFRanking(scenario=s, effect=0.5, rank=0, confidence=0.9)
        assert r.scenario.name == "A"
        assert r.effect == 0.5
        assert r.rank == 0
        assert r.confidence == 0.9
        assert r.is_uncertain is False

    def test_default_values(self):
        s = CFScenario(name="test")
        r = CFRanking(scenario=s)
        assert r.counterfactual_value is None
        assert r.factual_value is None
        assert r.effect == 0.0
        assert r.rank == -1
        assert r.confidence == 1.0
        assert r.is_uncertain is False


# =============================================================================
# CounterfactualOracle — 降级路径
# =============================================================================


class TestCounterfactualOracleDegraded:
    """无引擎、无 world_model 时的降级行为。"""

    def test_oracle_no_backend(self):
        """无后端 — 标记为 uncertain。"""
        oracle = CounterfactualOracle()
        scenarios = [CFScenario(name="A", intervention={"x": 1}, target="y")]
        results = oracle.batch_what_if(scenarios)
        assert len(results) == 1
        assert results[0].is_uncertain is True
        assert results[0].confidence == 0.0

    def test_rank_scenarios_uncertain(self):
        """uncertain 结果也能排序。"""
        oracle = CounterfactualOracle()
        scenarios = [
            CFScenario(name="A"),
            CFScenario(name="B"),
        ]
        rankings = oracle.rank_scenarios(scenarios)
        assert len(rankings) == 2
        assert rankings[0].rank == 0
        assert rankings[1].rank == 1

    def test_query_no_backend(self):
        """无后端 query — 返回不确定推荐。"""
        oracle = CounterfactualOracle()
        hypotheses = [
            {"name": "A", "intervention": {"x": 1}, "target": "y"},
            {"name": "B", "intervention": {"x": 2}, "target": "y"},
        ]
        result = oracle.query(hypotheses)
        assert result["best_scenario"] is not None
        assert result["n_scenarios"] == 2
        assert "不确定" in result["recommendation"]

    def test_query_count(self):
        """查询计数正确。"""
        oracle = CounterfactualOracle()
        assert oracle.query_count == 0

        scenarios = [CFScenario(name="A"), CFScenario(name="B")]
        oracle.batch_what_if(scenarios)
        assert oracle.query_count == 2


# =============================================================================
# CounterfactualOracle — 带模拟引擎
# =============================================================================


class TestCounterfactualOracleWithMockEngine:
    """带模拟反事实引擎的行为。"""

    def _make_mock_engine(self):
        """创建一个模拟的 CounterfactualEngine。"""

        class MockResult:
            def __init__(self, cf_value, f_value, effect):
                self.counterfactual_value = cf_value
                self.factual_value = f_value
                self.individual_effect = effect

        class MockEngine:
            def query(self, do_x, target, **kwargs):
                # 简单模拟: intervention 值越大，效应越大
                val = sum(abs(v) for v in do_x.values()) if isinstance(do_x, dict) else 1.0
                return MockResult(cf_value=val, f_value=0.0, effect=val)

        return MockEngine()

    def test_batch_what_if_with_engine(self):
        """有引擎 — 正常推演。"""
        engine = self._make_mock_engine()
        oracle = CounterfactualOracle(counterfactual_engine=engine)

        scenarios = [
            CFScenario(name="A", intervention={"x": 1}, target="y"),
            CFScenario(name="B", intervention={"x": 2}, target="y"),
        ]
        results = oracle.batch_what_if(scenarios)
        assert len(results) == 2
        assert results[0].is_uncertain is False
        assert results[0].confidence == 0.9
        assert results[0].effect > 0

    def test_rank_scenarios_with_engine(self):
        """有引擎 — 排序正确。"""
        engine = self._make_mock_engine()
        oracle = CounterfactualOracle(counterfactual_engine=engine)

        scenarios = [
            CFScenario(name="A", intervention={"x": 1}, target="y"),
            CFScenario(name="B", intervention={"x": 3}, target="y"),
        ]
        rankings = oracle.rank_scenarios(scenarios, target_direction="higher_is_better")
        assert rankings[0].rank == 0
        # B 的效应更大 (3 > 1)，应该排第一
        assert rankings[0].scenario.name == "B"

    def test_rank_lower_is_better(self):
        """lower_is_better 排序 — 效应小的排前面。"""
        engine = self._make_mock_engine()
        oracle = CounterfactualOracle(counterfactual_engine=engine)

        scenarios = [
            CFScenario(name="A", intervention={"x": 1}, target="y"),
            CFScenario(name="B", intervention={"x": 3}, target="y"),
        ]
        rankings = oracle.rank_scenarios(scenarios, target_direction="lower_is_better")
        assert rankings[0].scenario.name == "A"  # 效应更小排前面

    def test_query_with_engine(self):
        """有引擎 — 完整 query 流程。"""
        engine = self._make_mock_engine()
        oracle = CounterfactualOracle(counterfactual_engine=engine)

        hypotheses = [
            {"name": "方案A", "intervention": {"x": 1}, "target": "y"},
            {"name": "方案B", "intervention": {"x": 3}, "target": "y"},
        ]
        result = oracle.query(hypotheses, target_direction="higher_is_better")
        assert result["best_scenario"] == "方案B"
        assert result["n_scenarios"] == 2
        assert "推荐" in result["recommendation"]

    def test_engine_exception_fallback(self):
        """引擎异常 — 降级为 uncertain。"""

        class FailingEngine:
            def query(self, **kwargs):
                raise RuntimeError("engine error")

        oracle = CounterfactualOracle(counterfactual_engine=FailingEngine())
        scenarios = [CFScenario(name="A", intervention={"x": 1})]
        results = oracle.batch_what_if(scenarios)
        # 降级到 world_model -> 没有 world_model -> uncertain
        assert results[0].is_uncertain is True


# =============================================================================
# OrchestratorBridge CF_QUERY
# =============================================================================


class TestOrchestratorBridgeCFQuery:
    """CF_QUERY intent 测试。"""

    def test_cf_query_no_hypotheses(self):
        """缺少 hypotheses — 失败。"""
        bridge = OrchestratorBridge()
        result = bridge.execute_intent("CF_QUERY", {})
        assert result.success is False
        assert "hypotheses" in result.error

    def test_cf_query_with_hypotheses(self):
        """有 hypotheses — 返回排序结果。"""
        bridge = OrchestratorBridge()
        result = bridge.execute_intent(
            "CF_QUERY",
            {
                "hypotheses": [
                    {"name": "A", "intervention": {"x": 1}, "target": "y"},
                    {"name": "B", "intervention": {"x": 2}, "target": "y"},
                ],
            },
        )
        assert result.success is True
        assert "rankings" in result.data
        assert result.data["n_scenarios"] == 2

    def test_cf_query_in_default_map(self):
        """CF_QUERY 在默认映射表中。"""
        bridge = OrchestratorBridge()
        assert "CF_QUERY" in bridge._intent_map

    def test_cf_query_ranking_order(self):
        """CF_QUERY 返回正确排序。"""
        _engine_mock = None  # 无引擎 → uncertain

        # 用带引擎的 oracle
        class MockResult:
            def __init__(self, cf_value, f_value, effect):
                self.counterfactual_value = cf_value
                self.factual_value = f_value
                self.individual_effect = effect

        class MockEngine:
            def query(self, do_x, target, **kwargs):
                val = sum(abs(v) for v in do_x.values()) if isinstance(do_x, dict) else 1.0
                return MockResult(val, 0.0, val)

        from mci_world_model.sdk._counterfactual_oracle import CounterfactualOracle

        oracle = CounterfactualOracle(counterfactual_engine=MockEngine())

        # 创建 world_model 并注入 oracle
        from mci_world_model.sdk._world_model import MCIWorldModel

        wm = MCIWorldModel()
        wm._cf_oracle = oracle

        bridge = OrchestratorBridge(world_model=wm)
        result = bridge.execute_intent(
            "CF_QUERY",
            {
                "hypotheses": [
                    {"name": "方案A", "intervention": {"x": 1}, "target": "y"},
                    {"name": "方案B", "intervention": {"x": 5}, "target": "y"},
                ],
                "target_direction": "higher_is_better",
            },
        )
        assert result.success is True
        assert result.data["best_scenario"] == "方案B"
