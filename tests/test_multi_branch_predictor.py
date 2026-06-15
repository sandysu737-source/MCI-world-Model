"""
tests/test_multi_branch_predictor.py — MultiBranchPredictor 测试
=================================================================

覆盖:
    - predict_branches: 单/多/空分支推演
    - compare_branches: 对比排序 + goal 距离
    - best_branch: 最优分支选择
    - what_if: 反事实分析 + 降级回退
    - 与 PendulumPhysicsPredictor / PendulumJEPAPredictor 集成
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._action_conditioned_predictor import (
    PendulumJEPAPredictor,
    PendulumPhysicsPredictor,
)
from mci_world_model.sdk._multi_branch_predictor import (
    BranchEvaluation,
    MultiBranchPredictor,
)
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
def mbp(physics_pred):
    return MultiBranchPredictor(physics_pred)


@pytest.fixture
def rest_state():
    return PendulumState(theta=0.0, omega=0.0)


@pytest.fixture
def tilted_state():
    return PendulumState(theta=0.5, omega=0.0)


@pytest.fixture
def goal():
    return PendulumState(theta=0.0, omega=0.0)


# =============================================================================
# TestPredictBranches
# =============================================================================


class TestPredictBranches:
    """predict_branches 基础推演测试。"""

    def test_single_branch(self, mbp, tilted_state):
        actions = [[PendulumAction(torque=1.0), PendulumAction(torque=2.0)]]
        branches = mbp.predict_branches(tilted_state, actions)
        assert len(branches) == 1
        assert len(branches[0]) == 2

    def test_multiple_branches(self, mbp, tilted_state):
        actions = [
            [PendulumAction(torque=1.0)],
            [PendulumAction(torque=-1.0)],
            [PendulumAction(torque=0.0)],
        ]
        branches = mbp.predict_branches(tilted_state, actions)
        assert len(branches) == 3
        for b in branches:
            assert len(b) == 1

    def test_empty_action_sequence(self, mbp, tilted_state):
        """空动作序列 → 自然演化 1 步。"""
        branches = mbp.predict_branches(tilted_state, [[]])
        assert len(branches) == 1
        assert len(branches[0]) == 1

    def test_empty_list(self, mbp, tilted_state):
        branches = mbp.predict_branches(tilted_state, [])
        assert len(branches) == 0

    def test_branch_independence(self, mbp, tilted_state):
        """不同分支互不影响（状态隔离）。"""
        actions = [
            [PendulumAction(torque=5.0)],
            [PendulumAction(torque=-5.0)],
        ]
        branches = mbp.predict_branches(tilted_state, actions)
        # theta 在 dt=0.01 时变化极小，比较 omega（力矩直接影响角速度）
        assert branches[0][0].omega != branches[1][0].omega

    def test_trajectory_length_matches_actions(self, mbp, tilted_state):
        actions = [[PendulumAction(torque=1.0)] * 5]
        branches = mbp.predict_branches(tilted_state, actions)
        assert len(branches[0]) == 5


# =============================================================================
# TestCompareBranches
# =============================================================================


class TestCompareBranches:
    """compare_branches 对比排序测试。"""

    def test_ranking_by_distance(self, mbp, tilted_state, goal):
        """距 goal 最近的分支 rank=0。"""
        actions = [
            [PendulumAction(torque=-2.0)],  # 推向平衡
            [PendulumAction(torque=2.0)],  # 推离平衡
        ]
        branches = mbp.predict_branches(tilted_state, actions)
        evals = mbp.compare_branches(branches, goal)
        assert evals[0].rank == 0
        assert evals[0].final_distance <= evals[1].final_distance

    def test_no_goal(self, mbp, tilted_state):
        """无 goal 时使用向量 L2 范数。"""
        actions = [[PendulumAction(torque=0.0)]]
        branches = mbp.predict_branches(tilted_state, actions)
        evals = mbp.compare_branches(branches, goal=None)
        assert len(evals) == 1
        assert evals[0].final_distance >= 0

    def test_empty_trajectory(self, mbp):
        evals = mbp.compare_branches([[]], goal=PendulumState(theta=0.0, omega=0.0))
        assert evals[0].final_distance == float("inf")
        assert evals[0].trajectory_length == 0

    def test_total_distance_computed(self, mbp, tilted_state):
        actions = [[PendulumAction(torque=1.0)] * 3]
        branches = mbp.predict_branches(tilted_state, actions)
        evals = mbp.compare_branches(branches)
        assert evals[0].total_distance > 0

    def test_avg_step_distance(self, mbp, tilted_state):
        actions = [[PendulumAction(torque=1.0)] * 4]
        branches = mbp.predict_branches(tilted_state, actions)
        evals = mbp.compare_branches(branches)
        ev = evals[0]
        assert ev.avg_step_distance == pytest.approx(ev.total_distance / 4, rel=1e-5)


# =============================================================================
# TestBestBranch
# =============================================================================


class TestBestBranch:
    """best_branch 最优分支选择。"""

    def test_selects_closest_to_goal(self, mbp, tilted_state, goal):
        """选距 goal 最近的分支。"""
        actions = [
            [PendulumAction(torque=-3.0)],
            [PendulumAction(torque=3.0)],
        ]
        idx, trajectory = mbp.best_branch(tilted_state, actions, goal)
        # theta=0.5 时负力矩推向平衡
        final_dist = trajectory[-1].distance(goal)
        other_branches = mbp.predict_branches(tilted_state, actions)
        for i, b in enumerate(other_branches):
            if i != idx:
                assert final_dist <= b[-1].distance(goal) + 1e-10

    def test_returns_index_and_trajectory(self, mbp, tilted_state):
        actions = [
            [PendulumAction(torque=0.0)],
            [PendulumAction(torque=1.0)],
        ]
        idx, traj = mbp.best_branch(tilted_state, actions)
        assert idx in (0, 1)
        assert isinstance(traj, list)


# =============================================================================
# TestWhatIf
# =============================================================================


class TestWhatIf:
    """what_if 反事实分析。"""

    def test_empty_interventions(self, mbp, tilted_state):
        result = mbp.what_if(tilted_state, interventions=None)
        assert result == []
        result = mbp.what_if(tilted_state, interventions=[])
        assert result == []

    def test_rollout_fallback(self, mbp, tilted_state):
        """无引擎时降级为 rollout_fallback。"""
        interventions = [
            {"do_x": {"X": 1.0}, "target": "Y"},
        ]
        results = mbp.what_if(tilted_state, interventions=interventions)
        assert len(results) == 1
        assert results[0]["method"] == "rollout_fallback"

    def test_with_counterfactual_engine(self, tilted_state):
        """有反事实引擎时使用引擎。"""
        from mci_world_model.sdk._counterfactual import StructuralEquationModel

        sem = StructuralEquationModel(
            coefficients=np.array([[0, 0.8], [0, 0]], dtype=np.float64),
            node_names=["X", "Y"],
        )
        from mci_world_model.sdk._batch_counterfactual import BatchCounterfactualEngine

        engine = BatchCounterfactualEngine(sem)
        mbp = MultiBranchPredictor(PendulumPhysicsPredictor())

        interventions = [
            {"evidence": {"X": 1.0}, "do_x": {"X": 0.5}, "target": "Y"},
        ]
        results = mbp.what_if(tilted_state, counterfactual_engine=engine, interventions=interventions)
        assert len(results) == 1
        assert results[0]["method"] == "counterfactual_engine"
        assert results[0]["individual_effect"] is not None


# =============================================================================
# TestBranchEvaluation
# =============================================================================


class TestBranchEvaluation:
    """BranchEvaluation 数据类。"""

    def test_fields(self):
        ev = BranchEvaluation(
            branch_index=0,
            final_distance=0.5,
            trajectory_length=3,
        )
        assert ev.rank == -1
        assert ev.total_distance == 0.0

    def test_rank_assignment(self):
        evals = [
            BranchEvaluation(branch_index=0, final_distance=1.0, trajectory_length=2),
            BranchEvaluation(branch_index=1, final_distance=0.5, trajectory_length=2),
        ]
        evals.sort(key=lambda e: e.final_distance)
        for i, e in enumerate(evals):
            e.rank = i
        assert evals[0].rank == 0
        assert evals[0].branch_index == 1


# =============================================================================
# TestJEPAPredictorIntegration
# =============================================================================


class TestJEPAPredictorIntegration:
    """与 PendulumJEPAPredictor 集成。"""

    def test_jepa_multi_branch(self, jepa_pred, tilted_state, goal):
        mbp = MultiBranchPredictor(jepa_pred)
        actions = [
            [PendulumAction(torque=-2.0)],
            [PendulumAction(torque=2.0)],
        ]
        _idx, traj = mbp.best_branch(tilted_state, actions, goal)
        assert len(traj) == 1
        assert isinstance(traj[0], PendulumState)

    def test_repr(self, mbp):
        r = repr(mbp)
        assert "MultiBranchPredictor" in r
