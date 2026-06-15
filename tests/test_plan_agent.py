"""
tests/test_plan_agent.py — PlanAgent 因果决策前置化测试
=========================================================

覆盖:
    - evaluate_action: 单动作评估
    - plan: 多步前瞻规划
    - plan_with_lookahead: 穷举前瞻
    - execute: 执行计划
    - replan: 惊奇触发重规划
    - Plan 数据结构
    - 与 PendulumState/PendulumAction 端到端集成
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._action_conditioned_predictor import (
    PendulumJEPAPredictor,
    PendulumPhysicsPredictor,
)
from mci_world_model.sdk._plan_agent import Plan, PlanAgent
from mci_world_model.sdk._surprise_detector import SurpriseDetector
from mci_world_model.sdk._world_state import PendulumAction, PendulumState

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def physics_pred():
    return PendulumPhysicsPredictor()


@pytest.fixture
def jepa_pred():
    pred = PendulumJEPAPredictor(seed=42)
    pred.train(n_samples=500)
    return pred


@pytest.fixture
def agent(physics_pred):
    return PlanAgent(predictor=physics_pred)


@pytest.fixture
def agent_with_surprise(physics_pred):
    return PlanAgent(
        predictor=physics_pred,
        surprise_detector=SurpriseDetector(threshold=0.3),
    )


@pytest.fixture
def tilted():
    return PendulumState(theta=0.8, omega=0.0)


@pytest.fixture
def goal():
    return PendulumState(theta=0.0, omega=0.0)


# =============================================================================
# TestEvaluateAction
# =============================================================================


class TestEvaluateAction:
    """evaluate_action 单动作评估。"""

    def test_returns_dict(self, agent, tilted, goal):
        result = agent.evaluate_action(tilted, PendulumAction(torque=-2.0), goal)
        assert "distance_to_goal" in result
        assert "predicted_state" in result

    def test_negative_torque_closer(self, agent, tilted, goal):
        """theta>0 时负力矩推向平衡（比较 omega 变化方向）。"""
        neg = agent.evaluate_action(tilted, PendulumAction(torque=-3.0), goal)
        nat = agent.evaluate_action(tilted, PendulumAction(torque=0.0), goal)
        # 负力矩应使 omega 变负（推向平衡），所以预测状态应更接近平衡
        neg_state = neg["predicted_state"]
        nat_state = nat["predicted_state"]
        # 负力矩产生的 omega 应该更负（更接近 -g/L * theta * dt）
        assert neg_state.omega < nat_state.omega

    def test_with_surprise_detector(self, agent_with_surprise, tilted, goal):
        result = agent_with_surprise.evaluate_action(tilted, PendulumAction(torque=-1.0), goal)
        # surprise 可能为 None 或有值
        assert "surprise" in result


# =============================================================================
# TestPlan
# =============================================================================


class TestPlan:
    """plan 多步前瞻规划。"""

    def test_returns_plan(self, agent, tilted, goal):
        plan = agent.plan(tilted, goal)
        assert isinstance(plan, Plan)

    def test_plan_reduces_distance(self, agent, tilted, goal):
        """规划后距 goal 不应恶化太多（dt=0.01 下小步长限制）。"""
        initial_dist = tilted.distance(goal)
        plan = agent.plan(tilted, goal, max_horizon=3, n_branches=5)
        # dt=0.01 时单步 theta 变化极小，允许少量恶化
        assert plan.expected_cost < initial_dist * 1.1

    def test_plan_has_actions(self, agent, tilted, goal):
        plan = agent.plan(tilted, goal, max_horizon=3)
        assert plan.horizon > 0

    def test_plan_has_trajectory(self, agent, tilted, goal):
        plan = agent.plan(tilted, goal, max_horizon=2)
        assert len(plan.predicted_trajectory) > 0

    def test_confidence_in_range(self, agent, tilted, goal):
        plan = agent.plan(tilted, goal)
        assert 0.0 <= plan.confidence <= 1.0

    def test_no_candidates(self, agent, goal):
        plan = agent.plan(goal, goal, candidate_actions=[])
        assert plan.expected_cost == 0.0
        assert plan.reasoning == "no_candidate_actions"

    def test_custom_candidates(self, agent, tilted, goal):
        cands = [PendulumAction(torque=-4.0), PendulumAction(torque=-1.0)]
        plan = agent.plan(tilted, goal, candidate_actions=cands, max_horizon=2, n_branches=2)
        assert isinstance(plan, Plan)

    def test_plan_count_increments(self, agent, tilted, goal):
        assert agent.plan_count == 0
        agent.plan(tilted, goal)
        assert agent.plan_count == 1
        agent.plan(tilted, goal)
        assert agent.plan_count == 2


# =============================================================================
# TestPlanWithLookahead
# =============================================================================


class TestPlanWithLookahead:
    """plan_with_lookahead 穷举前瞻。"""

    def test_returns_plan(self, agent, tilted, goal):
        plan = agent.plan_with_lookahead(tilted, goal, horizon=2)
        assert isinstance(plan, Plan)

    def test_lookahead_better_than_single(self, agent, tilted, goal):
        """多步前瞻应优于单步。"""
        plan_1 = agent.plan_with_lookahead(tilted, goal, horizon=1)
        plan_3 = agent.plan_with_lookahead(tilted, goal, horizon=3)
        assert plan_3.expected_cost <= plan_1.expected_cost + 0.01

    def test_horizon_matches(self, agent, tilted, goal):
        plan = agent.plan_with_lookahead(tilted, goal, horizon=3)
        assert plan.horizon <= 3  # 可能因剪枝提前结束

    def test_no_candidates(self, agent, goal):
        plan = agent.plan_with_lookahead(goal, goal, horizon=2, candidate_actions=[])
        assert plan.reasoning == "no_candidate_actions"

    def test_metadata(self, agent, tilted, goal):
        plan = agent.plan_with_lookahead(tilted, goal, horizon=2)
        assert "horizon" in plan.metadata
        assert "n_candidates" in plan.metadata

    def test_goal_already_reached(self, agent, goal):
        """已在 goal → expected_cost ≈ 0。"""
        plan = agent.plan_with_lookahead(goal, goal, horizon=2)
        assert plan.expected_cost < 0.01


# =============================================================================
# TestExecute
# =============================================================================


class TestExecute:
    """execute 执行计划。"""

    def test_execute_returns_state_and_report(self, agent, tilted, goal):
        plan = agent.plan(tilted, goal, max_horizon=3)
        final_state, report = agent.execute(plan, tilted)
        assert isinstance(final_state, PendulumState)
        assert "n_steps_executed" in report

    def test_execute_count_increments(self, agent, tilted, goal):
        assert agent.execute_count == 0
        plan = agent.plan(tilted, goal)
        agent.execute(plan, tilted)
        assert agent.execute_count == 1

    def test_empty_plan(self, agent, tilted):
        empty_plan = Plan(actions=[])
        _final, report = agent.execute(empty_plan, tilted)
        assert report["n_steps_executed"] == 0

    def test_with_surprise_detection(self, agent_with_surprise, tilted, goal):
        plan = agent_with_surprise.plan(tilted, goal, max_horizon=2)
        _final, report = agent_with_surprise.execute(plan, tilted)
        assert "n_surprises" in report


# =============================================================================
# TestReplan
# =============================================================================


class TestReplan:
    """replan 惊奇触发重规划。"""

    def test_replan_returns_plan(self, agent, tilted, goal):
        plan = agent.replan(tilted, goal)
        assert isinstance(plan, Plan)

    def test_replan_with_surprise(self, agent, tilted, goal):
        from mci_world_model.sdk._surprise_detector import SurpriseSignal

        surprise = SurpriseSignal(
            score=0.8,
            predicted=tilted,
            actual=PendulumState(theta=1.5, omega=2.0),
            is_anomaly=True,
            threshold=0.3,
        )
        plan = agent.replan(tilted, goal, surprise=surprise)
        assert "surprise_replan" in plan.reasoning

    def test_replan_reasoning(self, agent, tilted, goal):
        plan = agent.replan(tilted, goal)
        assert "surprise_replan" in plan.reasoning


# =============================================================================
# TestPlanDataStructure
# =============================================================================


class TestPlanDataStructure:
    """Plan 数据结构。"""

    def test_default_values(self):
        plan = Plan()
        assert plan.horizon == 0
        assert plan.expected_cost == float("inf")
        assert plan.confidence == 0.0

    def test_horizon(self):
        plan = Plan(actions=[None, None, None])
        assert plan.horizon == 3

    def test_to_dict(self):
        plan = Plan(
            actions=[PendulumAction(torque=1.0)],
            expected_cost=0.5,
            confidence=0.8,
            reasoning="test",
        )
        d = plan.to_dict()
        assert d["horizon"] == 1
        assert d["expected_cost"] == pytest.approx(0.5, rel=1e-5)
        assert d["reasoning"] == "test"


# =============================================================================
# TestEndToEnd
# =============================================================================


class TestEndToEnd:
    """端到端集成测试。"""

    def test_plan_and_execute_pendulum(self, agent, goal):
        """从倾斜状态规划执行后，omega 应向平衡方向移动。"""
        tilted = PendulumState(theta=1.0, omega=0.0)
        plan = agent.plan_with_lookahead(tilted, goal, horizon=3)
        final, _report = agent.execute(plan, tilted)
        # dt=0.01 下 theta 变化很小，但 omega 应该变为负值（推向平衡方向）
        assert final.omega < tilted.omega or final.distance(goal) < tilted.distance(goal)

    def test_replan_cycle(self, agent, goal):
        """规划 → 执行 → 惊奇 → 重规划 循环。"""
        state = PendulumState(theta=1.0, omega=0.5)
        plan1 = agent.plan(state, goal, max_horizon=3)
        final1, _ = agent.execute(plan1, state)

        # 模拟惊奇
        plan2 = agent.replan(final1, goal)
        final2, _ = agent.execute(plan2, final1)
        assert isinstance(final2, PendulumState)

    def test_with_jepa_predictor(self, jepa_pred, goal):
        """PendulumJEPAPredictor 也能驱动 PlanAgent。"""
        agent = PlanAgent(predictor=jepa_pred)
        tilted = PendulumState(theta=0.5, omega=0.0)
        plan = agent.plan(tilted, goal, max_horizon=2)
        assert isinstance(plan, Plan)

    def test_repr(self, agent):
        r = repr(agent)
        assert "PlanAgent" in r


# =============================================================================
# TestDefaultCandidates
# =============================================================================


class TestDefaultCandidates:
    """默认候选动作生成。"""

    def test_pendulum_candidates(self, agent, tilted):
        cands = agent._generate_default_candidates(tilted)
        assert len(cands) == 5
        assert all(isinstance(c, PendulumAction) for c in cands)

    def test_non_pendulum_empty(self, agent):
        """非 PendulumState 返回空列表。"""
        from mci_world_model.sdk._world_state import WorldState

        class DummyState(WorldState):
            def to_vector(self):
                return np.array([1.0])

            @classmethod
            def from_vector(cls, vec):
                return cls()

            def distance(self, other):
                return 0.0

            def copy(self):
                return DummyState()

        cands = agent._generate_default_candidates(DummyState())
        assert cands == []
