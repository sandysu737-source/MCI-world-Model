"""
LOOP-06 (S-7): cewm_step 子方法单元测试
=======================================

验证 cewm_step() 拆分的 8 个子方法的独立正确性：
- _cewm_parse_state: 观测 → WorldState 解析
- _cewm_perceive: 感知层 (observation, goal → parsed states)
- _cewm_safety_check: 安全层约束检查
- _cewm_cognize: 认知层因果图更新 + 经验检索
- _cewm_evaluate_action: 行动层距离评估
- _cewm_predict: 预测层 JEPA/因果预测
- _cewm_feedback: 反馈层注意力调整
- _cewm_state_change: 状态变化因果边提取
"""

import pytest


@pytest.fixture
def wm():
    """创建一个最小化的 MCIWorldModel 实例。"""
    from mci_world_model.sdk._world_model import MCIWorldModel

    return MCIWorldModel()


@pytest.fixture
def pendulum_state():
    """PendulumState 实例。"""
    from mci_world_model.sdk._world_state import PendulumState

    return PendulumState(theta=0.5, omega=0.1)


@pytest.fixture
def pendulum_goal():
    """目标 PendulumState。"""
    from mci_world_model.sdk._world_state import PendulumState

    return PendulumState(theta=0.0, omega=0.0)


# =============================================================================
# TestCewmParseState: _cewm_parse_state
# =============================================================================


class TestCewmParseState:
    """_cewm_parse_state() 正确性验证。"""

    def test_parse_none(self, wm):
        """None 输入返回 None。"""
        assert wm._cewm_parse_state(None) is None

    def test_parse_pendulum_state(self, wm, pendulum_state):
        """PendulumState 直接传入保持不变。"""
        result = wm._cewm_parse_state(pendulum_state)
        assert result is not None

    def test_parse_already_parsed_passthrough(self, wm, pendulum_state):
        """已经是 WorldState 子类的对象原样返回。"""
        result = wm._cewm_parse_state(pendulum_state)
        # 应该返回某种 WorldState（或原对象）
        assert result is not None
        assert hasattr(result, "to_vector") or result == pendulum_state

    def test_registry_lazy_init(self, wm, pendulum_state):
        """StateParserRegistry 延迟初始化。"""
        assert wm._state_parser_registry is None
        wm._cewm_parse_state(pendulum_state)
        assert wm._state_parser_registry is not None


# =============================================================================
# TestCewmPerceive: _cewm_perceive
# =============================================================================


class TestCewmPerceive:
    """_cewm_perceive() 感知层验证。"""

    def test_perceive_both_states(self, wm, pendulum_state, pendulum_goal):
        """解析 observation 和 goal 为 WorldState。"""
        current, goal = wm._cewm_perceive(pendulum_state, pendulum_goal)
        assert current is not None
        assert goal is not None

    def test_perceive_none_inputs(self, wm):
        """None 输入返回 (None, None)。"""
        current, goal = wm._cewm_perceive(None, None)
        assert current is None
        assert goal is None

    def test_perceive_returns_tuple(self, wm, pendulum_state, pendulum_goal):
        """返回值是二元组。"""
        result = wm._cewm_perceive(pendulum_state, pendulum_goal)
        assert isinstance(result, tuple)
        assert len(result) == 2


# =============================================================================
# TestCewmSafetyCheck: _cewm_safety_check
# =============================================================================


class TestCewmSafetyCheck:
    """_cewm_safety_check() 安全层验证。"""

    def test_no_safety_monitor_passes(self, wm, pendulum_state):
        """无 SafetyMonitor 时安全检查通过。"""
        result = {}
        passed = wm._cewm_safety_check(pendulum_state, None, result)
        assert passed is True

    def test_none_state_passes(self, wm):
        """None 状态安全检查通过。"""
        result = {}
        passed = wm._cewm_safety_check(None, None, result)
        assert passed is True

    def test_result_not_mutated_on_pass(self, wm, pendulum_state):
        """通过时不设置 safety_violation 标志。"""
        result = {}
        wm._cewm_safety_check(pendulum_state, None, result)
        assert "safety_violation" not in result


# =============================================================================
# TestCewmCognize: _cewm_cognize
# =============================================================================


class TestCewmCognize:
    """_cewm_cognize() 认知层验证。"""

    def test_returns_tuple_of_ints(self, wm, pendulum_state, pendulum_goal):
        """返回 (causal_updates, experience_hints) 整数二元组。"""
        updates, hints = wm._cewm_cognize(pendulum_state, pendulum_goal)
        assert isinstance(updates, int)
        assert isinstance(hints, int)

    def test_none_states_zero_updates(self, wm):
        """None 状态返回零更新。"""
        updates, hints = wm._cewm_cognize(None, None)
        assert updates == 0

    def test_causal_updater_persistence(self, wm, pendulum_state, pendulum_goal):
        """CausalUpdater 首次创建后持续积累。"""
        wm._cewm_cognize(pendulum_state, pendulum_goal)
        assert wm._causal_updater is not None
        updater_ref = wm._causal_updater
        # 第二次调用应复用同一个实例
        wm._cewm_cognize(pendulum_state, pendulum_goal)
        assert wm._causal_updater is updater_ref


# =============================================================================
# TestCewmEvaluateAction: _cewm_evaluate_action
# =============================================================================


class TestCewmEvaluateAction:
    """_cewm_evaluate_action() 行动层验证。"""

    def test_returns_float_tuple(self, wm, pendulum_state, pendulum_goal):
        """返回 (action_distance, physical_distance) 浮点二元组。"""
        action_dist, phys_dist = wm._cewm_evaluate_action(
            pendulum_state, pendulum_goal
        )
        assert isinstance(action_dist, (int, float))
        assert isinstance(phys_dist, (int, float))

    def test_none_states_returns_zeros(self, wm):
        """None 状态返回 (0.0, 0.0)。"""
        action_dist, phys_dist = wm._cewm_evaluate_action(None, None)
        assert action_dist == 0.0
        assert phys_dist == 0.0

    def test_distance_non_negative(self, wm, pendulum_state, pendulum_goal):
        """距离值非负。"""
        action_dist, phys_dist = wm._cewm_evaluate_action(
            pendulum_state, pendulum_goal
        )
        assert action_dist >= 0
        assert phys_dist >= 0

    def test_action_gap_metric_lazy_init(self, wm, pendulum_state, pendulum_goal):
        """ActionGapMetric 延迟初始化。"""
        assert wm._action_gap_metric is None
        wm._cewm_evaluate_action(pendulum_state, pendulum_goal)
        assert wm._action_gap_metric is not None


# =============================================================================
# TestCewmPredict: _cewm_predict
# =============================================================================


class TestCewmPredict:
    """_cewm_predict() 预测层验证。"""

    def test_returns_tuple(self, wm, pendulum_state, pendulum_goal):
        """返回 (prediction, pred_error) 二元组。"""
        prediction, pred_error = wm._cewm_predict(
            pendulum_state, pendulum_goal, None, 1.0
        )
        assert isinstance(pred_error, (int, float))

    def test_none_state_prediction_is_none(self, wm):
        """None 状态时 prediction 为 None。"""
        prediction, pred_error = wm._cewm_predict(None, None, None, 0.0)
        assert prediction is None

    def test_pred_error_bounded(self, wm, pendulum_state, pendulum_goal):
        """预测误差被限制在 [0, 1]。"""
        _, pred_error = wm._cewm_predict(
            pendulum_state, pendulum_goal, None, 1.0
        )
        assert 0.0 <= pred_error <= 1.0


# =============================================================================
# TestCewmFeedback: _cewm_feedback
# =============================================================================


class TestCewmFeedback:
    """_cewm_feedback() 反馈层验证。"""

    def test_returns_dict(self, wm):
        """返回字典（可能为空）。"""
        result = wm._cewm_feedback(0.5)
        assert isinstance(result, dict)

    def test_zero_error(self, wm):
        """零误差不抛出异常。"""
        result = wm._cewm_feedback(0.0)
        assert isinstance(result, dict)

    def test_perception_lazy_init(self, wm):
        """PerceptionPipeline 延迟初始化。"""
        assert wm._perception is None
        wm._cewm_feedback(0.5)
        assert wm._perception is not None


# =============================================================================
# TestCewmStateChange: _cewm_state_change
# =============================================================================


class TestCewmStateChange:
    """_cewm_state_change() 因果边提取验证。"""

    def test_world_state_with_causal_edges(self, wm, pendulum_state):
        """WorldState 子类返回 causal_edges() 结果。"""
        edges = wm._cewm_state_change(pendulum_state)
        assert isinstance(edges, list)
        # PendulumState 应返回非空因果边
        assert len(edges) > 0

    def test_none_state_returns_empty(self, wm):
        """None 状态返回空列表。"""
        edges = wm._cewm_state_change(None)
        assert edges == []

    def test_edge_format(self, wm, pendulum_state):
        """因果边格式为 tuple[str, str]。"""
        edges = wm._cewm_state_change(pendulum_state)
        for edge in edges:
            assert isinstance(edge, tuple)
            assert len(edge) == 2
            assert isinstance(edge[0], str)
            assert isinstance(edge[1], str)

    def test_cart_state_causal_edges(self, wm):
        """CartState 也返回因果边。"""
        from mci_world_model.sdk._world_state import CartState

        cart = CartState(x=1.0, v=0.5)
        edges = wm._cewm_state_change(cart)
        assert isinstance(edges, list)
        assert len(edges) > 0
