"""CEWM v4.4.0 Phase 0 — CartState 闭环测试
===============================================

验证 CartState 作为第二种 WorldState 可以完整参与 CEWM 五层闭环：
1. 感知层: 观测 → CartState 解析
2. 认知层: 因果图更新
3. 预测层: CartPhysicsPredictor 预测
4. 行动层: CartAction + PlanAgent 决策
5. 反馈层: 预测误差 → 注意力调整

这些测试证明 PredictorProtocol + StateParserProtocol 抽象是正确的。
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._action_conditioned_predictor import (
    CartPhysicsPredictor,
    PendulumPhysicsPredictor,
)
from mci_world_model.sdk._action_gap import ActionGapMetric
from mci_world_model.sdk._multi_branch_predictor import MultiBranchPredictor
from mci_world_model.sdk._plan_agent import PlanAgent
from mci_world_model.sdk._protocols import (
    CartStateParser,
    GenericStateParser,
    PendulumStateParser,
    PredictorProtocol,
    StateParserProtocol,
    StateParserRegistry,
)
from mci_world_model.sdk._world_state import (
    CartAction,
    CartState,
    PendulumState,
    WorldState,
)

# =============================================================================
# CRT-01: CartState WorldState 契约验证
# =============================================================================


class TestCartStateContract:
    """CartState 实现了 WorldState ABC 的四个核心方法。"""

    def test_to_vector(self):
        state = CartState(x=1.5, v=-0.5)
        vec = state.to_vector()
        assert vec.shape == (2,)
        assert float(vec[0]) == pytest.approx(1.5)
        assert float(vec[1]) == pytest.approx(-0.5)

    def test_from_vector(self):
        vec = np.array([2.0, 3.0])
        state = CartState.from_vector(vec)
        assert isinstance(state, CartState)
        assert state.x == pytest.approx(2.0)
        assert state.v == pytest.approx(3.0)

    def test_from_vector_roundtrip(self):
        """to_vector → from_vector 往返保真。"""
        original = CartState(x=1.23, v=-4.56)
        reconstructed = CartState.from_vector(original.to_vector())
        assert reconstructed.x == pytest.approx(original.x)
        assert reconstructed.v == pytest.approx(original.v)

    def test_distance_same_type(self):
        s1 = CartState(x=0.0, v=0.0)
        s2 = CartState(x=3.0, v=4.0)
        assert s1.distance(s2) == pytest.approx(5.0)

    def test_distance_self_is_zero(self):
        s = CartState(x=5.0, v=-3.0)
        assert s.distance(s) == pytest.approx(0.0)

    def test_distance_cross_type(self):
        """跨类型距离使用向量 L2。"""
        cart = CartState(x=1.0, v=2.0)
        pend = PendulumState(theta=1.0, omega=2.0)
        dist = cart.distance(pend)
        assert dist == pytest.approx(0.0)  # 相同向量

    def test_copy_independence(self):
        original = CartState(x=1.0, v=2.0)
        copied = original.copy()
        copied.x = 999.0
        assert original.x == pytest.approx(1.0)

    def test_step_physics_free(self):
        """无外力自然演化: v 不变, x += v*dt。"""
        state = CartState(x=0.0, v=2.0, dt=0.01)
        next_state = state.step_physics()
        assert next_state.x == pytest.approx(0.02)
        assert next_state.v == pytest.approx(2.0)

    def test_isinstance_worldstate(self):
        state = CartState(x=0.0, v=0.0)
        assert isinstance(state, WorldState)


# =============================================================================
# CRT-02: CartAction 契约验证
# =============================================================================


class TestCartActionContract:
    """CartAction 实现了 Action ABC。"""

    def test_apply_force(self):
        state = CartState(x=0.0, v=0.0)
        action = CartAction(force=10.0, dt=0.01)
        next_state = action.apply(state)
        assert next_state.v == pytest.approx(0.1)  # v += force*dt = 0.1
        assert next_state.x == pytest.approx(0.0)  # x += v*dt = 0 (v was 0)

    def test_apply_negative_force(self):
        state = CartState(x=5.0, v=2.0)
        action = CartAction(force=-3.0, dt=0.01)
        next_state = action.apply(state)
        assert next_state.v == pytest.approx(2.0 - 0.03)

    def test_apply_wrong_state_type(self):
        state = PendulumState(theta=0.5, omega=0.0)
        action = CartAction(force=1.0)
        with pytest.raises(TypeError, match="CartAction"):
            action.apply(state)

    def test_isinstance_action(self):
        from mci_world_model.sdk._world_state import Action

        action = CartAction(force=5.0)
        assert isinstance(action, Action)


# =============================================================================
# CRT-03: CartPhysicsPredictor 验证
# =============================================================================


class TestCartPhysicsPredictor:
    """CartPhysicsPredictor 物理正确性验证。"""

    def test_predict_one_step(self):
        pred = CartPhysicsPredictor()
        state = CartState(x=0.0, v=1.0)
        action = CartAction(force=2.0, dt=0.01)
        traj = pred.predict(state, action, n_steps=1)
        assert len(traj) == 1
        # v += force*dt = 1.0 + 2.0*0.01 = 1.02
        assert traj[0].v == pytest.approx(1.02)
        # x += v*dt = 0.0 + 1.0*0.01 = 0.01
        assert traj[0].x == pytest.approx(0.01)

    def test_predict_multi_step(self):
        pred = CartPhysicsPredictor()
        state = CartState(x=0.0, v=0.0)
        action = CartAction(force=10.0, dt=0.01)
        traj = pred.predict(state, action, n_steps=10)
        assert len(traj) == 10
        # 逐步累积
        for i, s in enumerate(traj):
            assert s.v == pytest.approx(10.0 * 0.01 * (i + 1), abs=1e-10)

    def test_predict_no_action(self):
        pred = CartPhysicsPredictor()
        state = CartState(x=1.0, v=2.0, dt=0.01)
        traj = pred.predict(state, None, n_steps=1)
        assert traj[0].x == pytest.approx(1.02)  # x + v*dt
        assert traj[0].v == pytest.approx(2.0)  # v 不变

    def test_predict_wrong_state_type(self):
        pred = CartPhysicsPredictor()
        state = PendulumState(theta=0.5, omega=0.0)
        with pytest.raises(TypeError, match="CartState"):
            pred.predict(state, None, n_steps=1)

    def test_isinstance_predictor_protocol(self):
        pred = CartPhysicsPredictor()
        assert isinstance(pred, PredictorProtocol)

    def test_name(self):
        pred = CartPhysicsPredictor()
        assert pred.name == "cart_physics"

    def test_evaluate(self):
        pred = CartPhysicsPredictor()
        s0 = CartState(x=0.0, v=0.0)
        a = CartAction(force=10.0, dt=0.01)
        s1 = a.apply(s0)
        dataset = [(s0, a, s1)]
        result = pred.evaluate(dataset)
        assert result["avg_distance"] == pytest.approx(0.0, abs=1e-10)
        assert result["n"] == 1

    def test_hand_calculated_ground_truth(self):
        """手算验证: x=0, v=0, force=5N, dt=0.01, 5步。"""
        pred = CartPhysicsPredictor()
        state = CartState(x=0.0, v=0.0, dt=0.01)
        action = CartAction(force=5.0, dt=0.01)
        traj = pred.predict(state, action, n_steps=5)

        # Step 1: v=0.05, x=0.0
        assert traj[0].v == pytest.approx(0.05)
        assert traj[0].x == pytest.approx(0.0)

        # Step 2: v=0.10, x=0.05*0.01=0.0005
        assert traj[1].v == pytest.approx(0.10)
        assert traj[1].x == pytest.approx(0.0005)

        # Step 5: v=0.25
        assert traj[4].v == pytest.approx(0.25)


# =============================================================================
# CRT-04: PredictorProtocol 验证
# =============================================================================


class TestPredictorProtocol:
    """验证现有预测器满足 PredictorProtocol。"""

    def test_pendulum_physics_satisfies_protocol(self):
        pred = PendulumPhysicsPredictor()
        assert isinstance(pred, PredictorProtocol)

    def test_cart_physics_satisfies_protocol(self):
        pred = CartPhysicsPredictor()
        assert isinstance(pred, PredictorProtocol)

    def test_protocol_has_required_methods(self):
        """验证协议定义了 name/predict/evaluate。"""
        # Protocol 的结构化子类型检查通过 isinstance
        pred = CartPhysicsPredictor()
        assert hasattr(pred, "name")
        assert hasattr(pred, "predict")
        assert hasattr(pred, "evaluate")
        assert callable(pred.predict)
        assert callable(pred.evaluate)


# =============================================================================
# CRT-05: StateParserProtocol 验证
# =============================================================================


class TestStateParserProtocol:
    """验证状态解析器契约。"""

    def test_pendulum_parser_satisfies_protocol(self):
        parser = PendulumStateParser()
        assert isinstance(parser, StateParserProtocol)

    def test_cart_parser_satisfies_protocol(self):
        parser = CartStateParser()
        assert isinstance(parser, StateParserProtocol)

    def test_generic_parser_satisfies_protocol(self):
        parser = GenericStateParser()
        assert isinstance(parser, StateParserProtocol)

    def test_registry_default(self):
        registry = StateParserRegistry.default()
        # PendulumState 应被解析
        state = registry.parse(PendulumState(theta=0.5, omega=0.0))
        assert isinstance(state, PendulumState)

        # CartState 应被解析
        state = registry.parse(CartState(x=1.0, v=2.0))
        assert isinstance(state, CartState)

        # dict 含 theta 应解析为 PendulumState
        state = registry.parse({"theta": 0.3, "omega": 0.1})
        assert isinstance(state, PendulumState)

        # dict 含 x+v 应解析为 CartState
        state = registry.parse({"x": 1.0, "v": 2.0})
        assert isinstance(state, CartState)

    def test_registry_unknown_returns_none(self):
        registry = StateParserRegistry.default()
        result = registry.parse(42)
        assert result is None


# =============================================================================
# CRT-05: CartState 闭环测试 — Perception → Predict → Act → Feedback
# =============================================================================


class TestCartClosedLoop:
    """CartState 端到端闭环测试。"""

    def test_perception_to_state(self):
        """感知层: dict → CartState。"""
        registry = StateParserRegistry.default()
        obs = {"x": 1.0, "v": 2.0}
        state = registry.parse(obs)
        assert isinstance(state, CartState)
        assert state.x == pytest.approx(1.0)
        assert state.v == pytest.approx(2.0)

    def test_prediction_trajectory(self):
        """预测层: CartPhysicsPredictor 多步预测。"""
        pred = CartPhysicsPredictor()
        state = CartState(x=0.0, v=1.0)
        traj = pred.predict(state, None, n_steps=10)
        assert len(traj) == 10
        # 每步 x += v*dt, v 不变
        for i, s in enumerate(traj):
            assert s.x == pytest.approx(1.0 * 0.01 * (i + 1), abs=1e-10)

    def test_action_execution(self):
        """行动层: CartAction 改变状态。"""
        state = CartState(x=0.0, v=0.0)
        action = CartAction(force=5.0)
        next_state = action.apply(state)
        assert next_state.v > 0
        assert next_state.v == pytest.approx(5.0 * 0.01)

    def test_feedback_prediction_error(self):
        """反馈层: 预测误差计算。"""
        pred = CartPhysicsPredictor()
        current = CartState(x=0.0, v=1.0)
        predicted = pred.predict(current, None, n_steps=1)[0]
        actual = CartState(x=0.01, v=1.0)
        error = predicted.distance(actual)
        assert error == pytest.approx(0.0, abs=1e-10)  # 预测精确

    def test_plan_agent_with_cart(self):
        """PlanAgent 使用 CartPhysicsPredictor 规划路径。"""
        pred = CartPhysicsPredictor()
        agent = PlanAgent(predictor=pred)

        current = CartState(x=0.0, v=0.0)
        goal = CartState(x=10.0, v=0.0)

        plan = agent.plan(current, goal, max_horizon=3)
        assert plan.horizon > 0
        assert len(plan.actions) > 0

    def test_multi_branch_with_cart(self):
        """MultiBranchPredictor 使用 CartPhysicsPredictor 多分支推演。"""
        pred = CartPhysicsPredictor()
        mbp = MultiBranchPredictor(pred)

        state = CartState(x=0.0, v=0.0)
        goal = CartState(x=5.0, v=0.0)

        action_seqs = [
            [CartAction(force=5.0), CartAction(force=5.0)],
            [CartAction(force=10.0), CartAction(force=0.0)],
        ]

        branches = mbp.predict_branches(state, action_seqs)
        assert len(branches) == 2

        evals = mbp.compare_branches(branches, goal)
        assert len(evals) == 2
        assert evals[0].rank == 0  # 最优

    def test_action_gap_with_cart(self):
        """ActionGapMetric 对 CartState 计算行动距离。"""
        metric = ActionGapMetric()
        state = CartState(x=0.0, v=1.0)
        goal = CartState(x=10.0, v=0.0)

        result = metric.distance(state, goal)
        assert result.physical_distance > 0
        # 行动距离 = 加权距离（不含重力势垒时可能小于物理距离）
        assert result.action_distance > 0


# =============================================================================
# CRT-06: CartState 插入 cewm_step() E2E
# =============================================================================


class TestCartCewmStep:
    """CartState 通过 cewm_step() 完成闭环。"""

    @pytest.fixture
    def wm(self):
        from mci_world_model.sdk._world_model import MCIWorldModel

        return MCIWorldModel()

    def test_cewm_step_with_cart_state(self, wm):
        """CartState 观测可以进入 cewm_step。"""
        result = wm.cewm_step(
            observation=CartState(x=0.0, v=1.0),
            goal=CartState(x=10.0, v=0.0),
        )
        assert result["state"] is not None
        assert isinstance(result["state"], CartState)

    def test_cewm_step_with_cart_dict(self, wm):
        """dict 含 x/v 也可以进入 cewm_step。"""
        result = wm.cewm_step(
            observation={"x": 0.0, "v": 1.0},
            goal={"x": 10.0, "v": 0.0},
        )
        assert result["state"] is not None
        assert isinstance(result["state"], CartState)

    def test_plan_action_with_cart(self, wm):
        """plan_action() 使用 CartState 自动选择 CartPhysicsPredictor。"""
        result = wm.plan_action(
            current=CartState(x=0.0, v=0.0),
            goal=CartState(x=5.0, v=0.0),
        )
        # 成功时没有 'status' 键或有 'horizon' 键
        assert result.get("status") != "insufficient_state"
        assert result.get("horizon", 0) > 0 or result.get("expected_cost", float("inf")) < float("inf")
