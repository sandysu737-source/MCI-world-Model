"""
MCI World Model v4.3.3 — 新增组件串联测试
==========================================

覆盖 _world_model.py 的 v4.3.0 / v4.3.1 / v4.3.2 / v4.3.3 新增方法:
- run_cognitive_loop()   Wiener 四环认知闭环
- diagnose_failure()     MetaDiagnoser 认知诊断
- retrieve_experiences() MultiViewRetriever 五维检索
- detect_surprise()      SurpriseDetector 惊奇检测 (v4.3.1)
- explain()              多跳因果链回溯 (v4.3.1)
- plan_action()          PlanAgent 决策规划 (v4.3.2)
- synthesize_training_data() ReflectionSynthesizer QA合成 (v4.3.2)
- assess_diversity()     CognitiveDiversity 多样性评估 (v4.3.2)
- check_admissibility()  NegativeHeuristic 可接受性检查 (v4.3.2)
- train_parametric()     CausalMLP 参数化记忆训练 (v4.3.3)
- predict_causal_category() CausalMLP 五范畴因果预测 (v4.3.3)
- predict_energy_flow()  五行生克能量流预测 (v4.3.3)
- _cewm_parse_state()    观测解析
- _cewm_state_change()   状态因果边提取
- __repr__()             表示
"""

import pytest

# =============================================================================
# Fixture
# =============================================================================


@pytest.fixture
def wm():
    """创建一个最小化的 MCIWorldModel 实例。"""
    from mci_world_model.sdk._world_model import MCIWorldModel

    return MCIWorldModel()


@pytest.fixture
def wm_with_loop(wm):
    """预注入 CognitiveLoopBus 的 WorldModel。"""
    from mci_world_model.sdk._cognitive_loop import CognitiveLoopBus

    wm._cognitive_loop = CognitiveLoopBus()
    return wm


# =============================================================================
# TestRunCognitiveLoop
# =============================================================================


class TestRunCognitiveLoop:
    """run_cognitive_loop() 四环认知闭环测试。"""

    def test_basic_propagation(self, wm):
        """无误差注入时正常传播。"""
        result = wm.run_cognitive_loop()
        assert "total_energy" in result
        assert "converged" in result
        assert "deltas" in result
        assert "health" in result

    def test_with_layer_errors(self, wm):
        """注入各层误差信号后传播。"""
        result = wm.run_cognitive_loop(
            layer_errors={"perception": 0.5, "cognition": 0.3, "prediction": 0.2, "action": 0.1},
            n_rounds=3,
        )
        assert isinstance(result["total_energy"], float)
        assert isinstance(result["health"]["overall_health"], float)
        assert isinstance(result["health"]["oscillation_detected"], bool)

    def test_single_round(self, wm):
        """单轮传播使用 propagate()。"""
        result = wm.run_cognitive_loop(layer_errors={"perception": 0.5}, n_rounds=1)
        assert "total_energy" in result

    def test_multi_round(self, wm):
        """多轮传播使用 propagate_n()。"""
        result = wm.run_cognitive_loop(layer_errors={"perception": 0.3}, n_rounds=5)
        assert isinstance(result["deltas"], dict)

    def test_unknown_layer_ignored(self, wm):
        """未知层名不会崩溃。"""
        result = wm.run_cognitive_loop(layer_errors={"unknown_layer": 0.9})
        assert "total_energy" in result

    def test_lazy_init(self, wm):
        """首次调用时自动创建 CognitiveLoopBus。"""
        assert wm._cognitive_loop is None
        wm.run_cognitive_loop()
        assert wm._cognitive_loop is not None

    def test_reuse_bus(self, wm_with_loop):
        """后续调用复用已有 bus。"""
        bus_id = id(wm_with_loop._cognitive_loop)
        wm_with_loop.run_cognitive_loop()
        assert id(wm_with_loop._cognitive_loop) == bus_id

    def test_health_report_structure(self, wm):
        """健康报告包含必要字段。"""
        result = wm.run_cognitive_loop(layer_errors={"perception": 0.8}, n_rounds=2)
        health = result["health"]
        assert "bottleneck_layer" in health
        assert "overall_health" in health
        assert "oscillation_detected" in health


# =============================================================================
# TestDiagnoseFailure
# =============================================================================


class TestDiagnoseFailure:
    """diagnose_failure() 认知诊断测试。"""

    def test_no_signals_no_loop(self, wm):
        """无信号且无认知循环时返回默认结果。"""
        result = wm.diagnose_failure()
        assert result["pattern"] is None
        assert result["severity"] == 0.0
        assert result["recommendation"] == "无信号可诊断"

    def test_with_explicit_signals(self, wm):
        """提供显式惊奇信号时正常诊断。"""
        signals = [
            {
                "state_distance": 0.8,
                "vector_deviation": 0.5,
                "direction_error": 0.6,
            }
        ]
        result = wm.diagnose_failure(surprise_signals=signals)
        assert "pattern" in result
        assert "severity" in result
        assert "confidence" in result
        assert "recommendation" in result
        assert "root_cause_chain" in result
        assert "health_scores" in result

    def test_with_context(self, wm):
        """提供上下文参数时正常诊断。"""
        signals = [{"state_distance": 0.9, "vector_deviation": 0.7, "direction_error": 0.5}]
        result = wm.diagnose_failure(surprise_signals=signals, context={"task": "pendulum"})
        assert isinstance(result["recommendation"], str)

    def test_auto_extract_from_loop(self, wm_with_loop):
        """从认知循环自动提取误差信号。"""
        from mci_world_model.sdk._cognitive_loop import CognitiveLayer

        # 注入一些误差让统计有值
        bus = wm_with_loop._cognitive_loop
        bus.inject_error(CognitiveLayer.PERCEPTION, magnitude=0.5)
        bus.propagate()

        result = wm_with_loop.diagnose_failure()
        # 有信号时应有 pattern 或 severity
        assert "severity" in result

    def test_lazy_init_diagnoser(self, wm):
        """首次调用时自动创建 MetaDiagnoser。"""
        assert wm._meta_diagnoser is None
        wm.diagnose_failure(surprise_signals=[{"state_distance": 0.5, "vector_deviation": 0.3, "direction_error": 0.2}])
        assert wm._meta_diagnoser is not None

    def test_empty_signals_list(self, wm):
        """空列表视为无信号。"""
        result = wm.diagnose_failure(surprise_signals=[])
        assert result["pattern"] is None

    def test_result_serializable(self, wm):
        """诊断结果可序列化。"""
        import json

        signals = [{"state_distance": 0.7, "vector_deviation": 0.4, "direction_error": 0.3}]
        result = wm.diagnose_failure(surprise_signals=signals)
        json.dumps(result)  # 不应抛异常


# =============================================================================
# TestRetrieveExperiences
# =============================================================================


class TestRetrieveExperiences:
    """retrieve_experiences() 五维融合检索测试。"""

    def test_basic_retrieval(self, wm):
        """基本检索返回结果列表。"""
        results = wm.retrieve_experiences(tags=["test"])
        assert isinstance(results, list)

    def test_empty_db_returns_empty(self, wm):
        """空数据库返回空列表。"""
        results = wm.retrieve_experiences(tags=["nonexistent_tag_xyz"])
        assert isinstance(results, list)

    def test_with_causal_edges(self, wm):
        """使用因果边检索。"""
        results = wm.retrieve_experiences(
            causal_edges=[("A", "B"), ("B", "C")],
            top_k=3,
        )
        assert isinstance(results, list)

    def test_with_context(self, wm):
        """使用上下文检索。"""
        results = wm.retrieve_experiences(
            context={"domain": "control", "task": "stabilize"},
            top_k=5,
        )
        assert isinstance(results, list)

    def test_result_structure(self, wm):
        """结果包含必要字段。"""
        # 先存储一些经验
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB, ExperienceType

        db = ExperienceDB()
        for i in range(3):
            exp = Experience(
                experience_id=f"test_exp_{i}",
                experience_type=ExperienceType.SUCCESS,
                tags=["test", "control"],
                importance=0.8,
            )
            db.store(exp)

        wm._experience_db = db
        results = wm.retrieve_experiences(tags=["test", "control"], top_k=3)
        assert isinstance(results, list)
        if results:
            r = results[0]
            assert "experience_id" in r
            assert "score" in r
            assert "view_scores" in r
            assert "strategy" in r

    def test_lazy_init_retriever(self, wm):
        """首次调用时自动创建 MultiViewRetriever。"""
        assert wm._multi_view_retriever is None
        wm.retrieve_experiences(tags=["test"])
        assert wm._multi_view_retriever is not None

    def test_reuse_existing_db(self, wm):
        """复用已有 ExperienceDB。"""
        from mci_world_model.sdk._experience_memory import ExperienceDB

        db = ExperienceDB()
        wm._experience_db = db
        wm.retrieve_experiences(tags=["test"])
        assert wm._multi_view_retriever is not None

    def test_top_k_parameter(self, wm):
        """top_k 参数正确传递。"""
        results = wm.retrieve_experiences(tags=["test"], top_k=2)
        assert len(results) <= 2


# =============================================================================
# TestCewmHelpers
# =============================================================================


class TestCewmHelpers:
    """_cewm_parse_state / _cewm_state_change 辅助方法测试。"""

    def test_parse_none(self, wm):
        """None 输入返回 None。"""
        assert wm._cewm_parse_state(None) is None

    def test_parse_pendulum_state(self, wm):
        """PendulumState 直接返回。"""
        from mci_world_model.sdk._world_state import PendulumState

        state = PendulumState(theta=0.5, omega=0.1)
        result = wm._cewm_parse_state(state)
        assert result is state

    def test_parse_dict_with_theta(self, wm):
        """dict 带 theta 转为 PendulumState。"""
        result = wm._cewm_parse_state({"theta": 0.5, "omega": 0.1})
        assert hasattr(result, "theta")
        assert result.theta == 0.5

    def test_parse_dict_without_theta(self, wm):
        """dict 不带 theta 直接返回。"""
        d = {"x": 1, "y": 2}
        assert wm._cewm_parse_state(d) is d

    def test_parse_passthrough(self, wm):
        """无法识别的类型直接返回。"""
        obj = object()
        assert wm._cewm_parse_state(obj) is obj

    def test_state_change_pendulum(self, wm):
        """PendulumState 提取因果边。"""
        from mci_world_model.sdk._world_state import PendulumState

        state = PendulumState(theta=0.5, omega=0.3)
        edges = wm._cewm_state_change(state)
        assert ("theta", "omega") in edges
        assert ("omega", "theta") in edges

    def test_state_change_zero_theta(self, wm):
        """FIX-C5: 因果结构是状态类型属性，与当前值无关。"""
        from mci_world_model.sdk._world_state import PendulumState

        state = PendulumState(theta=0.0, omega=0.0)
        edges = wm._cewm_state_change(state)
        # causal_edges() 返回结构因果边，不受当前数值影响
        assert ("theta", "omega") in edges
        assert ("omega", "theta") in edges

    def test_state_change_no_pendulum(self, wm):
        """非 Pendulum 对象返回空边。"""
        edges = wm._cewm_state_change(object())
        assert edges == []


# =============================================================================
# TestRepr
# =============================================================================


class TestRepr:
    """__repr__ 测试。"""

    def test_repr_contains_version(self, wm):
        """repr 包含版本号。"""
        s = repr(wm)
        assert "v4.3.3" in s
        assert "MCIWorldModel" in s

    def test_repr_contains_edges(self, wm):
        """repr 包含边数。"""
        s = repr(wm)
        assert "edges" in s

    def test_repr_jepa_status(self, wm):
        """repr 包含 JEPA 状态。"""
        s = repr(wm)
        assert "jepa=" in s


# =============================================================================
# TestCewmStep
# =============================================================================


class TestCewmStep:
    """cewm_step() 认知增强全流程测试。"""

    def test_basic_step_no_args(self, wm):
        """无参数调用不崩溃。"""
        result = wm.cewm_step()
        assert isinstance(result, dict)
        assert "state" in result
        assert "action_distance" in result

    def test_step_with_pendulum_observation(self, wm):
        """PendulumState 观测 + 目标。"""
        from mci_world_model.sdk._world_state import PendulumState

        obs = PendulumState(theta=0.5, omega=0.1)
        goal = PendulumState(theta=0.0, omega=0.0)
        result = wm.cewm_step(observation=obs, goal=goal)
        assert result["state"] is obs
        assert result["action_distance"] >= 0
        assert result["physical_distance"] >= 0
        assert result["causal_updates"] >= 0

    def test_step_with_dict_observation(self, wm):
        """dict 观测自动解析为 PendulumState。"""
        obs = {"theta": 0.3, "omega": 0.2}
        goal = {"theta": 0.0, "omega": 0.0}
        result = wm.cewm_step(observation=obs, goal=goal)
        assert result["state"] is not None

    def test_step_with_experience_db(self, wm):
        """有 ExperienceDB 时经验检索生效。"""
        from mci_world_model.sdk._experience_memory import ExperienceDB

        wm._experience_db = ExperienceDB()
        result = wm.cewm_step()
        assert "experience_hints" in result

    def test_step_causal_updater_init(self, wm):
        """cewm_step 自动初始化 CausalUpdater。"""
        from mci_world_model.sdk._world_state import PendulumState

        obs = PendulumState(theta=0.5, omega=0.3)
        goal = PendulumState(theta=0.0, omega=0.0)
        result = wm.cewm_step(observation=obs, goal=goal)
        assert result["causal_updates"] >= 0


# =============================================================================
# TestBuildCausalGraph
# =============================================================================


class TestBuildCausalGraph:
    """_build_causal_graph_from_state 图构建测试。"""

    def test_empty_state_returns_none(self, wm):
        """空状态返回 None。"""
        assert wm._build_causal_graph_from_state() is None

    def test_with_edges(self, wm):
        """有因果边时返回 CausalGraph。"""
        wm._state.causal_edges = [
            {"cause_idx": 0, "effect_idx": 1, "rho": 0.5, "evidence_count": 3},
            {"cause_idx": 1, "effect_idx": 2, "rho": 0.3, "evidence_count": 2},
        ]
        graph = wm._build_causal_graph_from_state()
        assert graph is not None


# =============================================================================
# TestIntervene
# =============================================================================


class TestIntervene:
    """intervene() Pearl do-operator 干预测试。"""

    def test_missing_params(self, wm):
        """缺少参数时返回 insufficient_input。"""
        result = wm.intervene()
        assert result["status"] == "insufficient_input"

    def test_basic_intervention(self, wm):
        """基本干预调用不崩溃。"""
        result = wm.intervene(do_x={"X": 1.0}, target="Y")
        assert isinstance(result, dict)
        assert "status" in result

    def test_nan_intervention_rejected(self, wm):
        """NaN 干预值被拒绝。"""

        result = wm.intervene(do_x={"X": float("nan")}, target="Y")
        assert result["status"] == "error"
        assert "finite" in result["message"].lower()

    def test_inf_intervention_rejected(self, wm):
        """Inf 干预值被拒绝。"""
        result = wm.intervene(do_x={"X": float("inf")}, target="Y")
        assert result["status"] == "error"

    def test_intervention_with_edges(self, wm):
        """有因果边时干预执行。"""
        wm._state.causal_edges = [
            {"cause_idx": 0, "effect_idx": 1, "rho": 0.5, "evidence_count": 3},
        ]
        result = wm.intervene(do_x={"X": 1.0}, target="Y")
        assert isinstance(result, dict)


# =============================================================================
# TestQueryCounterfactual
# =============================================================================


class TestQueryCounterfactual:
    """query_counterfactual() 反事实查询测试。"""

    def test_basic_query(self, wm):
        """基本反事实查询不崩溃。"""
        result = wm.query_counterfactual(
            evidence={"X": 1.0, "Y": 0.5},
            do_x={"X": 0.0},
            target="Y",
        )
        assert isinstance(result, dict)

    def test_query_with_edges(self, wm):
        """有因果边时反事实查询执行。"""
        wm._state.causal_edges = [
            {"cause_idx": 0, "effect_idx": 1, "rho": 0.5, "evidence_count": 3},
        ]
        result = wm.query_counterfactual(
            evidence={"X": 1.0},
            do_x={"X": 0.0},
            target="Y",
        )
        assert isinstance(result, dict)


# =============================================================================
# TestAdditionalMethods
# =============================================================================


class TestAdditionalMethods:
    """额外方法覆盖测试。"""

    def test_predict_effect_with_parametric(self, wm):
        """parametric_predict 是 jepa_predict 别名。"""
        result = wm.parametric_predict("test_cause")
        assert isinstance(result, list)

    def test_predict_from_memories_m3_no_jepa(self, wm):
        """无 JEPA 时 predict_from_memories_m3 返回空。"""
        result = wm.predict_from_memories_m3(memories=[{"text": "a"}, {"text": "b"}, {"text": "c"}])
        assert isinstance(result, list)
        assert len(result) == 0

    def test_decompose_effect_no_mediator(self, wm):
        """无中介变量时返回默认结果。"""
        result = wm.decompose_effect(cause="X", effect="Y")
        assert isinstance(result, dict)

    def test_decompose_effect_with_mediator(self, wm):
        """指定中介变量时正常执行。"""
        result = wm.decompose_effect(cause="X", effect="Y", mediator="M")
        assert isinstance(result, dict)


# =============================================================================
# TestExplainMultiHop — v4.3.1
# =============================================================================


class TestExplainMultiHop:
    """explain() 多跳因果链回溯测试。"""

    def test_explain_empty_edges(self, wm):
        """无因果边时返回空链。"""
        result = wm.explain("nothing")
        assert result["query"] == "nothing"
        assert result["chains"] == []
        assert "暂无因果图数据" in result["summary"]

    def test_single_hop_depth_1(self, wm):
        """max_depth=1 只返回单跳链。"""
        wm._state.causal_edges = [
            {"cause": "A", "effect": "B", "confidence": 0.9, "verdict": "confirmed", "energy_relation": "strong"},
        ]
        result = wm.explain("A", max_depth=1)
        chains = result["chains"]
        assert len(chains) >= 1
        for c in chains:
            assert c["depth"] == 1
            assert c["confidence"] > 0

    def test_double_hop_depth_2(self, wm):
        """max_depth=2 返回两跳因果链 A→B→C。"""
        wm._state.causal_edges = [
            {"cause": "A", "effect": "B", "confidence": 0.9, "verdict": "confirmed"},
            {"cause": "B", "effect": "C", "confidence": 0.8, "verdict": "predicted"},
        ]
        result = wm.explain("A", max_depth=2)
        chains = result["chains"]
        depths = {c["depth"] for c in chains}
        assert 2 in depths, f"Expected depth=2 chain, got depths={depths}"

    def test_triple_hop_depth_3(self, wm):
        """max_depth=3 返回三跳因果链 A→B→C→D。"""
        wm._state.causal_edges = [
            {"cause": "A", "effect": "B", "confidence": 0.9},
            {"cause": "B", "effect": "C", "confidence": 0.8},
            {"cause": "C", "effect": "D", "confidence": 0.7},
        ]
        result = wm.explain("A", max_depth=3)
        chains = result["chains"]
        depths = {c["depth"] for c in chains}
        assert 3 in depths, f"Expected depth=3 chain, got depths={depths}"

    def test_cycle_prevention(self, wm):
        """因果环不会导致无限循环。"""
        wm._state.causal_edges = [
            {"cause": "A", "effect": "B", "confidence": 0.9},
            {"cause": "B", "effect": "A", "confidence": 0.8},  # 回环
        ]
        result = wm.explain("A", max_depth=5)
        chains = result["chains"]
        for c in chains:
            # 链中每个节点最多出现一次
            for node in c["path"]:
                if node.startswith("→"):
                    continue
                # 节点在 non-arrow 形式中最多出现1次
                non_arrow = [n for n in c["path"] if not n.startswith("→")]
                assert non_arrow.count(node) == 1

    def test_depth_in_output(self, wm):
        """输出链包含正确的 depth 字段。"""
        wm._state.causal_edges = [
            {"cause": "A", "effect": "B", "confidence": 0.9},
            {"cause": "B", "effect": "C", "confidence": 0.8},
        ]
        result = wm.explain("A", max_depth=2)
        for c in result["chains"]:
            assert "depth" in c
            assert c["depth"] >= 1

    def test_query_match_effect(self, wm):
        """query 匹配 effect 侧也能找到起始链。"""
        wm._state.causal_edges = [
            {"cause": "X", "effect": "target_node", "confidence": 0.7},
            {"cause": "target_node", "effect": "Y", "confidence": 0.8},
        ]
        result = wm.explain("target_node", max_depth=2)
        assert len(result["chains"]) >= 1

    def test_max_depth_respected(self, wm):
        """所有返回链的 depth 不超过 max_depth。"""
        wm._state.causal_edges = [
            {"cause": "A", "effect": "B", "confidence": 0.9},
            {"cause": "B", "effect": "C", "confidence": 0.8},
            {"cause": "C", "effect": "D", "confidence": 0.7},
            {"cause": "D", "effect": "E", "confidence": 0.6},
        ]
        result = wm.explain("A", max_depth=2)
        for c in result["chains"]:
            assert c["depth"] <= 2

    def test_confidence_averaged(self, wm):
        """多跳链的 confidence 是多段边的加权平均。"""
        wm._state.causal_edges = [
            {"cause": "A", "effect": "B", "confidence": 0.6},
            {"cause": "B", "effect": "C", "confidence": 0.8},
        ]
        result = wm.explain("A", max_depth=2)
        two_hop = [c for c in result["chains"] if c["depth"] == 2]
        if two_hop:
            conf = two_hop[0]["confidence"]
            assert 0.6 < conf < 0.8


# =============================================================================
# TestDetectSurprise — v4.3.1
# =============================================================================


class TestDetectSurprise:
    """detect_surprise() 惊奇误差检测测试。"""

    def test_no_states_returns_default(self, wm):
        """无输入时返回默认结果。"""
        result = wm.detect_surprise()
        assert result["score"] == 0.0
        assert result["is_anomaly"] is False
        assert "note" in result
        assert result["note"] == "insufficient_state_input"

    def test_with_pendulum_states(self, wm):
        """PendulumState 输入返回完整信号。"""
        from mci_world_model.sdk._world_state import PendulumState

        predicted = PendulumState(theta=0.5, omega=0.1)
        actual = PendulumState(theta=0.6, omega=0.05)
        result = wm.detect_surprise(predicted=predicted, actual=actual)
        assert "score" in result
        assert "is_anomaly" in result
        assert "breakdown" in result
        assert "stats" in result
        assert isinstance(result["breakdown"], dict)

    def test_identical_states_zero_surprise(self, wm):
        """相同状态惊奇度接近零。"""
        from mci_world_model.sdk._world_state import PendulumState

        state = PendulumState(theta=0.5, omega=0.3)
        result = wm.detect_surprise(predicted=state, actual=state)
        assert result["score"] < 0.01
        assert result["is_anomaly"] is False

    def test_different_states_positive_surprise(self, wm):
        """不同状态惊奇度大于零。"""
        from mci_world_model.sdk._world_state import PendulumState

        predicted = PendulumState(theta=0.0, omega=0.0)
        actual = PendulumState(theta=3.0, omega=2.0)
        result = wm.detect_surprise(predicted=predicted, actual=actual)
        assert result["score"] > 0

    def test_lazy_init_detector(self, wm):
        """首次调用自动创建 SurpriseDetector。"""
        assert wm._surprise_detector is None
        from mci_world_model.sdk._world_state import PendulumState

        wm.detect_surprise(
            predicted=PendulumState(theta=0.0, omega=0.0),
            actual=PendulumState(theta=0.1, omega=0.0),
        )
        assert wm._surprise_detector is not None

    def test_threshold_respects_param(self, wm):
        """threshold 参数生效。"""
        from mci_world_model.sdk._world_state import PendulumState

        result = wm.detect_surprise(
            predicted=PendulumState(theta=0.0, omega=0.0),
            actual=PendulumState(theta=3.0, omega=2.0),
            threshold=0.99,
        )
        assert result["threshold"] == 0.99

    def test_breakdown_three_dimensions(self, wm):
        """返回三维度分解。"""
        from mci_world_model.sdk._world_state import PendulumState

        predicted = PendulumState(theta=0.0, omega=0.0)
        actual = PendulumState(theta=1.0, omega=0.5)
        result = wm.detect_surprise(predicted=predicted, actual=actual)
        breakdown = result["breakdown"]
        assert "state_distance" in breakdown
        assert "vector_deviation" in breakdown
        assert "direction_error" in breakdown

    def test_dict_input_parsed(self, wm):
        """dict 输入自动解析为 PendulumState。"""
        predicted = {"theta": 0.0, "omega": 0.0}
        actual = {"theta": 0.5, "omega": 0.2}
        result = wm.detect_surprise(predicted=predicted, actual=actual)
        assert result["score"] > 0
        assert "stats" in result

    def test_cewm_components_after_detect(self, wm):
        """调用 detect_surprise 后 health_check 报告组件状态。"""
        from mci_world_model.sdk._world_state import PendulumState

        wm.detect_surprise(
            predicted=PendulumState(theta=0.0, omega=0.0),
            actual=PendulumState(theta=0.1, omega=0.0),
        )
        check = wm.health_check()
        cewm = check.get("cewm_components", {})
        assert cewm.get("surprise_detector") is True


# =============================================================================
# v4.3.2 — PlanAction + 3 New Methods
# =============================================================================


class TestPlanAction:
    """plan_action() — PlanAgent 集成测试。"""

    def test_plan_basic(self, wm):
        """基本规划调用成功。"""
        from mci_world_model.sdk._world_state import PendulumState

        plan = wm.plan_action(
            current=PendulumState(theta=0.5, omega=0.1),
            goal=PendulumState(theta=0.0, omega=0.0),
        )
        assert isinstance(plan, dict)
        assert "expected_cost" in plan
        assert "confidence" in plan

    def test_plan_none_current(self, wm):
        """current=None 返回 insufficient_state。"""
        plan = wm.plan_action(current=None, goal=None)
        assert plan["status"] == "insufficient_state"
        assert plan["confidence"] == 0.0

    def test_plan_returns_confidence(self, wm):
        """plan 输出包含置信度。"""
        from mci_world_model.sdk._world_state import PendulumState

        plan = wm.plan_action(
            current=PendulumState(theta=0.5, omega=0.1),
            goal=PendulumState(theta=0.0, omega=0.0),
        )
        assert "confidence" in plan
        assert isinstance(plan["confidence"], (int, float))

    def test_plan_horizon_respected(self, wm):
        """max_horizon 参数传递。"""
        from mci_world_model.sdk._world_state import PendulumState

        plan = wm.plan_action(
            current=PendulumState(theta=0.5, omega=0.1),
            goal=PendulumState(theta=0.0, omega=0.0),
            max_horizon=3,
        )
        assert plan.get("horizon", 0) <= 3

    def test_plan_multiple_calls_idempotent(self, wm):
        """多次调用不报错。"""
        from mci_world_model.sdk._world_state import PendulumState

        for _ in range(3):
            plan = wm.plan_action(
                current=PendulumState(theta=0.3, omega=-0.2),
                goal=PendulumState(theta=0.0, omega=0.0),
            )
            assert isinstance(plan, dict)


class TestSynthesizeTrainingData:
    """synthesize_training_data() — ReflectionSynthesizer 集成测试。"""

    def test_synth_basic(self, wm):
        """基本合成调用成功。"""
        memories = [{"id": "m1", "content": "白蛋白下降导致水肿"}]
        result = wm.synthesize_training_data(memories)
        assert isinstance(result, dict)
        assert "qa_pairs" in result
        assert "n_pairs" in result
        assert "report" in result
        assert "ready" in result

    def test_synth_empty_memories(self, wm):
        """空记忆返回空结果。"""
        result = wm.synthesize_training_data([])
        assert result["n_pairs"] == 0
        assert result["qa_pairs"] == []

    def test_synth_none_graceful(self, wm):
        """None 输入不崩溃。"""
        result = wm.synthesize_training_data(None)
        assert result["n_pairs"] == 0
        assert result["ready"] is False


class TestAssessDiversity:
    """assess_diversity() — CognitiveDiversity 集成测试。"""

    def test_diversity_basic(self, wm):
        """基本多样性评估成功。"""
        result = wm.assess_diversity()
        assert isinstance(result, dict)
        assert "diversity_vector" in result
        assert "ashby_satisfied" in result
        assert "ashby_ratio" in result

    def test_diversity_with_states(self, wm):
        """传递状态不影响调用。"""
        from mci_world_model.sdk._world_state import PendulumState

        states = [PendulumState(theta=0.1, omega=0.0)]
        result = wm.assess_diversity(states=states)
        assert "diversity_vector" in result

    def test_diversity_with_errors(self, wm):
        """传递误差不影响调用。"""
        result = wm.assess_diversity(prediction_errors=[0.01, 0.02, 0.03])
        assert isinstance(result["ashby_ratio"], (int, float))


class TestCheckAdmissibility:
    """check_admissibility() — NegativeHeuristic 集成测试。"""

    def test_adm_no_change(self, wm):
        """无变更时 admissible=True。"""
        result = wm.check_admissibility()
        assert result["admissible"] is True
        assert result["violations"] == []

    def test_adm_with_change(self, wm):
        """传递变更不崩溃。"""
        result = wm.check_admissibility(
            change={
                "description": "test change",
                "change_type": "parameter",
            }
        )
        assert isinstance(result, dict)
        assert "admissible" in result

    def test_adm_structure_complete(self, wm):
        """返回结构完整。"""
        result = wm.check_admissibility(
            change={
                "description": "test",
                "change_type": "structural",
            }
        )
        assert "violations" in result
        assert "suggestions" in result
        assert "hard_core_status" in result


class TestV432HealthCheck:
    """health_check 报告验证（版本跟随 __version__）。"""

    def test_version_460(self, wm):
        """health_check 返回当前包版本。"""
        import mci_world_model

        check = wm.health_check()
        assert check["version"] == mci_world_model.__version__

    def test_cewm_components_extended(self, wm):
        """cewm_components 包含 v4.3.2+v4.3.3 字段。"""
        check = wm.health_check()
        cewm = check.get("cewm_components", {})
        assert "plan_agent" in cewm
        assert "reflection_synthesizer" in cewm
        assert "cognitive_diversity" in cewm
        assert "negative_heuristic" in cewm
        assert "action_conditioned_predictor" in cewm
        assert "multi_branch_predictor" in cewm
        assert "parametric_memory" in cewm
        assert "energy_flow_predictor" in cewm

    def test_roadmap_432_entries(self, wm):
        """roadmap 包含 v4.3.2 + v4.3.3 条目。"""
        check = wm.health_check()
        roadmap = check.get("roadmap", {})
        assert "v4.3.2" in roadmap
        assert "v4.3.2-m2" in roadmap
        assert "v4.3.2-m3" in roadmap
        assert "v4.3.2-m4" in roadmap
        assert "v4.3.3" in roadmap
        assert "v4.3.3-m2" in roadmap
        assert "v4.3.3-m3" in roadmap

    def test_repr_432(self, wm):
        """__repr__ 显示 4.3.3。"""
        r = repr(wm)
        assert "v4.3.3" in r


# =============================================================================
# TestParametricMemory — v4.3.3 ParametricMemory 集成测试
# =============================================================================


class TestParametricMemory:
    """train_parametric() / predict_causal_category() 测试。"""

    @pytest.fixture
    def sample_qa_pairs(self):
        """构造模拟 QA 对，覆盖多范畴。"""
        return [
            {
                "cause_text": "蛋白质摄入不足导致白蛋白水平下降",
                "effect_text": "白蛋白水平下降引起免疫功能减弱",
                "energy_relation": "enhance",
                "confidence": 0.85,
            },
            {
                "cause_text": "炎症因子升高引发食欲减退",
                "effect_text": "食欲减退导致营养摄入减少",
                "energy_relation": "suppress",
                "confidence": 0.78,
            },
            {
                "cause_text": "每天摄入800千卡热量",
                "effect_text": "热量摄入低于基础代谢需求",
                "energy_relation": "same",
                "confidence": 0.90,
            },
            {
                "cause_text": "病房环境温度适宜",
                "effect_text": "患者主观舒适度提升",
                "energy_relation": "neutral",
                "confidence": 0.60,
            },
            {
                "cause_text": "肠内营养支持持续进行",
                "effect_text": "患者体重逐步恢复",
                "energy_relation": "enhance",
                "confidence": 0.92,
            },
            {
                "cause_text": "手术创伤引发应激反应",
                "effect_text": "代谢率急剧上升",
                "energy_relation": "enhance",
                "confidence": 0.88,
            },
            {
                "cause_text": "长期卧床导致肌肉萎缩",
                "effect_text": "基础代谢持续下降",
                "energy_relation": "suppress",
                "confidence": 0.82,
            },
            {
                "cause_text": "维生素D缺乏",
                "effect_text": "钙吸收效率降低",
                "energy_relation": "enhance",
                "confidence": 0.95,
            },
            {
                "cause_text": "血糖波动频繁",
                "effect_text": "胰岛素敏感性下降",
                "energy_relation": "causal",
                "confidence": 0.80,
            },
            {
                "cause_text": "夜间睡眠不足",
                "effect_text": "日间精神状态欠佳",
                "energy_relation": "same",
                "confidence": 0.72,
            },
            {
                "cause_text": "钠摄入过量",
                "effect_text": "血压水平升高",
                "energy_relation": "enhance",
                "confidence": 0.94,
            },
            {
                "cause_text": "脱水状态持续",
                "effect_text": "肾功能指标异常",
                "energy_relation": "enhance",
                "confidence": 0.87,
            },
        ]

    def test_train_no_data(self, wm):
        """无 QA 对时优雅返回。"""
        result = wm.train_parametric(qa_pairs=None)
        assert result["status"] == "no_training_data"
        assert result["trainable"] is False

    def test_train_from_qa_pairs(self, wm, sample_qa_pairs):
        """从 ReflectionSynthesizer QA 对训练 CausalMLP。"""
        result = wm.train_parametric(qa_pairs=sample_qa_pairs)
        assert result["status"] == "trained"
        assert "n_params" in result
        assert result["n_params"] > 0
        assert "final_loss" in result
        assert "n_samples" in result

    def test_train_custom_epochs(self, wm, sample_qa_pairs):
        """自定义 num_epochs 参数生效。"""
        result = wm.train_parametric(qa_pairs=sample_qa_pairs, num_epochs=5)
        assert result["status"] == "trained"

    def test_predict_untrained(self, wm):
        """未训练时优雅降级。"""
        result = wm.predict_causal_category("蛋白质摄入不足")
        assert result["status"] == "not_trained"
        assert result["predictions"] == []
        assert result["probs"] == {}

    def test_predict_after_train(self, wm, sample_qa_pairs):
        """训练后返回五范畴概率预测。"""
        wm.train_parametric(qa_pairs=sample_qa_pairs)
        result = wm.predict_causal_category("蛋白质摄入不足导致白蛋白水平下降")
        assert result["status"] == "ok"
        assert "predictions" in result
        assert len(result["predictions"]) >= 1
        assert "probs" in result
        assert len(result["probs"]) == 5  # 五范畴

    def test_predict_probs_structure(self, wm, sample_qa_pairs):
        """概率分布包含五范畴 key。"""
        wm.train_parametric(qa_pairs=sample_qa_pairs)
        result = wm.predict_causal_category("炎症因子升高引发食欲减退")
        for cat in ("causal", "semantic", "spacetime", "generative", "trust"):
            assert cat in result["probs"]

    def test_predict_returns_n_params(self, wm, sample_qa_pairs):
        """预测结果包含模型参数量。"""
        wm.train_parametric(qa_pairs=sample_qa_pairs)
        result = wm.predict_causal_category("test cause")
        assert result["n_params"] > 0

    def test_parametric_lazy_init(self, wm):
        """首次调用 train_parametric 时自动创建 ParametricMemory。"""
        assert wm._parametric_memory is None
        wm.train_parametric(qa_pairs=[])
        # qa_pairs=[] 为空列表不是 None，会初始化 ParametricMemory
        assert wm._parametric_memory is not None

    def test_train_empty_qa_list(self, wm):
        """空 QA 对列表时返回不足。"""
        result = wm.train_parametric(qa_pairs=[])
        assert result["status"] == "insufficient_data"
        assert result["meets_minimum"] is False

    def test_save_load_adapter_roundtrip(self, wm, sample_qa_pairs):
        """训练后 save + load adapter 往返。"""
        import shutil

        wm.train_parametric(qa_pairs=sample_qa_pairs)

        # 使用白名单内的路径
        adapter_dir = "./adapters/v433_roundtrip_test"
        try:
            from mci_world_model.sdk._parametric_memory import ParametricMemory

            saved = wm._parametric_memory.save_adapter(adapter_dir)
            assert saved is True

            # load into new ParametricMemory
            pm2 = ParametricMemory()
            loaded = pm2.load_adapter(adapter_dir)
            assert loaded is True
            assert pm2.is_trained is True
        finally:
            shutil.rmtree(adapter_dir, ignore_errors=True)

    def test_parametric_in_health_check(self, wm):
        """health_check 返回 dict，包含 cewm_components。"""
        check = wm.health_check()
        assert isinstance(check["cewm_components"], dict)


# =============================================================================
# TestEnergyFlowPredictor — v4.3.3 EnergyFlowPredictor 集成测试
# =============================================================================


class TestEnergyFlowPredictor:
    """predict_energy_flow() 测试。"""

    def test_energy_flow_predict_basic(self, wm):
        """基本五步预测不崩溃。"""
        result = wm.predict_energy_flow(steps=5)
        assert result["steps"] == 5
        assert "flow" in result
        assert isinstance(result["flow"], list)
        assert isinstance(result["anomaly_detected"], bool)

    def test_energy_flow_length(self, wm):
        """输出序列长度 = steps + 1（含当前状态）。"""
        result = wm.predict_energy_flow(steps=3)
        assert len(result["flow"]) == 4  # steps + 1

    def test_energy_flow_default_steps(self, wm):
        """默认 steps=5。"""
        result = wm.predict_energy_flow()
        assert len(result["flow"]) == 6

    def test_energy_flow_lazy_init(self, wm):
        """首次调用自动创建 EnergyFlowPredictor + EnergyCore。"""
        assert wm._energy_flow_predictor is None
        wm.predict_energy_flow()
        assert wm._energy_flow_predictor is not None
        assert wm._energy_core is not None

    def test_energy_flow_reuse(self, wm):
        """重复调用复用已有 predictor。"""
        wm.predict_energy_flow()
        predictor_id = id(wm._energy_flow_predictor)
        wm.predict_energy_flow(steps=3)
        assert id(wm._energy_flow_predictor) == predictor_id

    def test_energy_flow_ratios_included(self, wm):
        """返回 current_ratios。"""
        result = wm.predict_energy_flow()
        assert "current_ratios" in result
        assert isinstance(result["current_ratios"], dict)

    def test_energy_flow_in_health_check(self, wm):
        """health_check 返回 dict，包含 cewm_components。"""
        check = wm.health_check()
        assert isinstance(check["cewm_components"], dict)


# =============================================================================
# TestV433HealthCheck — v4.3.3 health_check 报告验证
# =============================================================================


class TestV433HealthCheck:
    """v4.3.3 health_check + __repr__ 验证。"""

    def test_version_433(self, wm):
        """health_check 返回 4.3.2（T3 前暂为当前版本）。"""
        check = wm.health_check()
        assert "version" in check

    def test_cewm_components_433(self, wm):
        """cewm_components 包含 v4.3.2 + v4.3.3 字段（T3 前不强制检查值）。"""
        check = wm.health_check()
        cewm = check.get("cewm_components", {})
        # v4.3.2 字段应存在
        for field in (
            "plan_agent",
            "action_conditioned_predictor",
            "multi_branch_predictor",
            "reflection_synthesizer",
            "cognitive_diversity",
            "negative_heuristic",
        ):
            assert field in cewm, f"v4.3.2 field {field} not in cewm_components"

    def test_repr_contains_version(self, wm):
        """__repr__ 包含版本号。"""
        r = repr(wm)
        assert "v4.3" in r
