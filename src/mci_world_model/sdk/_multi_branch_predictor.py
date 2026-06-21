from __future__ import annotations

"""
MCI World Model v3.3.0 — MultiBranchPredictor 多分支未来推演引擎
==================================================================

从单个当前状态出发，并行推演多条候选动作轨迹，
对比评估各分支的"未来结局"，选出最优分支。

核心能力:
    predict_branches(state, action_sequences)  — 并行推演 N 条轨迹
    compare_branches(branches, goal)           — 分支对比排序
    best_branch(state, action_sequences, goal)  — 选最优分支
    what_if(state, engine, interventions)       — 反事实 what-if 分析

复用:
    ActionConditionedPredictor.rollout() — 单分支推演内核
    BatchCounterfactualEngine.batch_query() — 反事实评估

设计原则:
    - 纯 numpy，零外部依赖
    - 与 ActionConditionedPredictor 正交组合（不继承）
    - 向后兼容：不修改任何现有接口
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mci_world_model.sdk._action_conditioned_predictor import ActionConditionedPredictor
    from mci_world_model.sdk._world_state import Action, WorldState

logger = logging.getLogger(__name__)


# =============================================================================
# BranchEvaluation — 分支评估结果
# =============================================================================


@dataclass
class BranchEvaluation:
    """分支评估结果。

    Attributes:
        branch_index: 分支编号（0-based）
        final_distance: 终点状态与 goal 的距离
        trajectory_length: 轨迹长度
        total_distance: 轨迹累计距离（各步距离之和）
        avg_step_distance: 平均每步移动距离
        rank: 排名（0 = 最优）
    """

    branch_index: int
    final_distance: float
    trajectory_length: int
    total_distance: float = 0.0
    avg_step_distance: float = 0.0
    rank: int = -1


# =============================================================================
# MultiBranchPredictor — 多分支未来推演引擎
# =============================================================================


class MultiBranchPredictor:
    """多分支未来推演引擎 — 从单状态出发，并行推演多条动作轨迹。

    Example:
        >>> from mci_world_model.sdk import (
        ...     PendulumPhysicsPredictor, PendulumState, PendulumAction,
        ... )
        >>> pred = PendulumPhysicsPredictor()
        >>> mbp = MultiBranchPredictor(pred)
        >>> state = PendulumState(theta=0.5, omega=0.0)
        >>> actions_list = [
        ...     [PendulumAction(torque=2.0), PendulumAction(torque=1.0)],
        ...     [PendulumAction(torque=-1.0), PendulumAction(torque=-2.0)],
        ... ]
        >>> branches = mbp.predict_branches(state, actions_list)
        >>> evals = mbp.compare_branches(branches, goal=PendulumState(theta=0.0, omega=0.0))
        >>> print(evals[0].rank, evals[0].final_distance)
    """

    def __init__(self, predictor: ActionConditionedPredictor) -> None:
        """
        Args:
            predictor: 动作条件化预测器（用于单分支推演）
        """
        self._predictor = predictor

    @property
    def predictor(self) -> ActionConditionedPredictor:
        return self._predictor

    # -----------------------------------------------------------------
    # predict_branches — 并行推演 N 条轨迹
    # -----------------------------------------------------------------

    def predict_branches(
        self,
        state: WorldState,
        action_sequences: list[list[Action]],
    ) -> list[list[WorldState]]:
        """从当前状态出发，对每条动作序列执行 rollout 推演。

        Args:
            state: 初始世界状态
            action_sequences: N 条动作序列，每条为 [a_0, a_1, ...]

        Returns:
            N 条状态轨迹，每条为 [s_1, s_2, ...]
        """
        branches: list[list[WorldState]] = []
        for actions in action_sequences:
            if not actions:
                # 空动作序列：自然演化 1 步
                trajectory = self._predictor.predict(state.copy(), None, n_steps=1)
            else:
                trajectory = self._predictor.rollout(state.copy(), actions)
            branches.append(trajectory)
        return branches

    # -----------------------------------------------------------------
    # compare_branches — 分支对比排序
    # -----------------------------------------------------------------

    def compare_branches(
        self,
        branches: list[list[WorldState]],
        goal: WorldState | None = None,
    ) -> list[BranchEvaluation]:
        """对比各分支，按终点距离 goal 排序。

        Args:
            branches: predict_branches() 返回的 N 条轨迹
            goal: 目标状态。None 时使用各分支终点与原点的距离

        Returns:
            BranchEvaluation 列表，按 final_distance 升序排列
        """
        evals: list[BranchEvaluation] = []

        for i, trajectory in enumerate(branches):
            if not trajectory:
                evals.append(
                    BranchEvaluation(
                        branch_index=i,
                        final_distance=float("inf"),
                        trajectory_length=0,
                    )
                )
                continue

            final_state = trajectory[-1]

            if goal is not None:
                final_dist = final_state.distance(goal)
            else:
                # 无 goal 时，用终点向量的 L2 范数
                final_dist = float(np.linalg.norm(final_state.to_vector()))

            # 计算轨迹累计距离
            total_dist = 0.0
            prev = None
            for s in trajectory:
                if prev is not None:
                    total_dist += prev.distance(s)
                prev = s

            n_steps = len(trajectory)
            avg_step = total_dist / n_steps if n_steps > 0 else 0.0

            evals.append(
                BranchEvaluation(
                    branch_index=i,
                    final_distance=final_dist,
                    trajectory_length=n_steps,
                    total_distance=total_dist,
                    avg_step_distance=avg_step,
                )
            )

        # 按 final_distance 升序排列，赋 rank
        evals.sort(key=lambda e: e.final_distance)
        for rank, ev in enumerate(evals):
            ev.rank = rank

        return evals

    # -----------------------------------------------------------------
    # best_branch — 选最优分支
    # -----------------------------------------------------------------

    def best_branch(
        self,
        state: WorldState,
        action_sequences: list[list[Action]],
        goal: WorldState | None = None,
    ) -> tuple[int, list[WorldState]]:
        """推演所有分支并选出最优（距 goal 最近）。

        Args:
            state: 初始状态
            action_sequences: N 条候选动作序列
            goal: 目标状态

        Returns:
            (最优分支索引, 最优分支轨迹)
        """
        branches = self.predict_branches(state, action_sequences)
        evals = self.compare_branches(branches, goal)
        best = evals[0]  # rank=0 的最优
        return best.branch_index, branches[best.branch_index]

    # -----------------------------------------------------------------
    # what_if — 反事实分析
    # -----------------------------------------------------------------

    def what_if(
        self,
        state: WorldState,
        counterfactual_engine: object | None = None,
        interventions: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """反事实 what-if 分析。

        如果有 counterfactual_engine（BatchCounterfactualEngine 或
        CounterfactualEngine），使用它评估干预效果。
        否则降级为纯推演式 what-if（对每个干预做 rollout）。

        Args:
            state: 当前状态
            counterfactual_engine: 反事实引擎（可选）
            interventions: 干预列表 [{"do_x": {...}, "target": "..."}, ...]

        Returns:
            每个干预的分析结果字典
        """
        if not interventions:
            return []

        results: list[dict[str, Any]] = []

        if counterfactual_engine is not None:
            # 使用反事实引擎
            try:
                cf_results = counterfactual_engine.batch_query(interventions)  # type: ignore
                for i, cf in enumerate(cf_results):
                    results.append(
                        {
                            "intervention": interventions[i],
                            "factual_value": getattr(cf, "factual_value", None),
                            "counterfactual_value": getattr(cf, "counterfactual_value", None),
                            "individual_effect": getattr(cf, "individual_effect", None),
                            "status": getattr(cf, "status", "ok"),
                            "method": "counterfactual_engine",
                        }
                    )
            except Exception as e:
                logger.warning("反事实引擎异常，降级为推演式: %s", e)
                return self._what_if_rollout(state, interventions)
        else:
            # 降级：纯推演式
            return self._what_if_rollout(state, interventions)

        return results

    def _what_if_rollout(
        self,
        state: WorldState,
        interventions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """降级 what-if：对每个干预做简单推演。"""
        results: list[dict[str, Any]] = []
        for intervention in interventions:
            do_x = intervention.get("do_x", {})
            target = intervention.get("target", "")

            # 无动作推演（自然演化）
            natural = self._predictor.predict(state.copy(), None, n_steps=1)
            natural_vec = natural[0].to_vector() if natural else state.to_vector()

            results.append(
                {
                    "intervention": intervention,
                    "natural_outcome": natural_vec.tolist() if len(natural_vec) < 100 else None,
                    "n_do_variables": len(do_x),
                    "target": target,
                    "method": "rollout_fallback",
                }
            )
        return results

    # -----------------------------------------------------------------
    # 字符串表示
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        return f"MultiBranchPredictor(predictor={self._predictor!r})"
