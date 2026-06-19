from __future__ import annotations

"""
MCI World Model v3.3.0 — PlanAgent 因果决策前置化
====================================================

"先模拟后执行"的因果决策 Agent。在真正执行动作前，
先用 ActionConditionedPredictor 模拟未来轨迹，
用 MultiBranchPredictor 评估多分支，用 SurpriseDetector 检测异常，
最终选出最优计划再执行。

核心能力:
    plan(current, goal, ...)         — 多步前瞻规划
    plan_with_lookahead(...)         — 穷举前瞻
    evaluate_action(...)             — 单动作评估
    execute(plan, state)             — 执行计划
    replan(current, goal, surprise)  — 惊奇触发重规划

复用:
    ActionConditionedPredictor — 状态模拟器
    MultiBranchPredictor       — 多分支评估
    SurpriseDetector           — 异常触发器
    CausalActor                — 候选动作生成（可选）

设计原则:
    - 纯 numpy，零外部依赖
    - 不修改 CausalActor 内部（包裹而非继承）
    - 支持 Plan 数据结构的序列化与回放
"""


import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mci_world_model.sdk._action_conditioned_predictor import ActionConditionedPredictor
    from mci_world_model.sdk._cost_module import EnergyCostModule
    from mci_world_model.sdk._multi_branch_predictor import MultiBranchPredictor
    from mci_world_model.sdk._surprise_detector import SurpriseDetector, SurpriseSignal
    from mci_world_model.sdk._world_state import Action, WorldState

logger = logging.getLogger(__name__)


# =============================================================================
# Plan — 决策计划
# =============================================================================


@dataclass
class Plan:
    """决策计划 — 包含动作序列、预测轨迹和评估信息。

    Attributes:
        actions: 推荐的动作序列
        predicted_trajectory: 预测的状态轨迹 [s_1, s_2, ...]
        expected_cost: 预期总代价（越小越好）
        confidence: 置信度 [0, 1]
        reasoning: 推理说明
        metadata: 额外信息
    """

    actions: list[Any] = field(default_factory=list)
    predicted_trajectory: list[Any] = field(default_factory=list)
    expected_cost: float = float("inf")
    confidence: float = 0.0
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        """规划步数。"""
        return len(self.actions)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "horizon": self.horizon,
            "expected_cost": round(self.expected_cost, 6),
            "confidence": round(self.confidence, 6),
            "reasoning": self.reasoning,
            "metadata": self.metadata,
        }


# =============================================================================
# PlanAgent — 因果决策前置化 Agent
# =============================================================================


class PlanAgent:
    """因果决策前置化 Agent — 先模拟后执行。

    Example:
        >>> from mci_world_model.sdk import (
        ...     PendulumPhysicsPredictor, PendulumState, PendulumAction,
        ... )
        >>> pred = PendulumPhysicsPredictor()
        >>> agent = PlanAgent(predictor=pred)
        >>> current = PendulumState(theta=1.0, omega=0.0)
        >>> goal = PendulumState(theta=0.0, omega=0.0)
        >>> plan = agent.plan_with_lookahead(current, goal, horizon=3)
        >>> print(plan.expected_cost, plan.horizon)
    """

    def __init__(
        self,
        predictor: ActionConditionedPredictor,
        cost_module: EnergyCostModule | None = None,
        multi_branch: MultiBranchPredictor | None = None,
        surprise_detector: SurpriseDetector | None = None,
    ):
        """
        Args:
            predictor: 动作条件化预测器（必须）
            cost_module: 代价评估模块（可选）
            multi_branch: 多分支推演引擎（可选，不提供时自动创建）
            surprise_detector: 惊奇检测器（可选）
        """
        self._predictor = predictor
        self._cost_module = cost_module
        self._surprise_detector = surprise_detector

        if multi_branch is not None:
            self._multi_branch = multi_branch
        else:
            from mci_world_model.sdk._multi_branch_predictor import MultiBranchPredictor

            self._multi_branch = MultiBranchPredictor(predictor)

        self._plan_count: int = 0
        self._execute_count: int = 0

    @property
    def predictor(self) -> ActionConditionedPredictor:
        return self._predictor

    @property
    def plan_count(self) -> int:
        return self._plan_count

    @property
    def execute_count(self) -> int:
        return self._execute_count

    # -----------------------------------------------------------------
    # evaluate_action — 单动作评估
    # -----------------------------------------------------------------

    def evaluate_action(
        self,
        current: WorldState,
        action: Action,
        goal: WorldState,
    ) -> dict[str, Any]:
        """评估单个动作的效果。

        Args:
            current: 当前状态
            action: 候选动作
            goal: 目标状态

        Returns:
            {"distance_to_goal": float, "predicted_state": WorldState,
             "surprise": SurpriseSignal | None}
        """
        preds = self._predictor.predict(current.copy(), action, n_steps=1)
        predicted = preds[0]
        dist = predicted.distance(goal)

        surprise = None
        if self._surprise_detector is not None:
            # 用自然演化做 baseline 来检测惊奇
            natural = self._predictor.predict(current.copy(), None, n_steps=1)
            surprise = self._surprise_detector.compute_surprise(natural[0], predicted)

        return {
            "distance_to_goal": dist,
            "predicted_state": predicted,
            "surprise": surprise,
        }

    # -----------------------------------------------------------------
    # plan — 多步前瞻规划
    # -----------------------------------------------------------------

    def plan(
        self,
        current: WorldState,
        goal: WorldState,
        candidate_actions: list | None = None,
        max_horizon: int = 5,
        n_branches: int = 3,
    ) -> Plan:
        """多步前瞻规划：候选动作组合 → 多分支模拟 → 选最优。

        Args:
            current: 当前状态
            goal: 目标状态
            candidate_actions: 候选动作列表。None 时生成默认候选
            max_horizon: 最大规划步数
            n_branches: 评估分支数

        Returns:
            最优 Plan
        """
        self._plan_count += 1

        if candidate_actions is None:
            candidate_actions = self._generate_default_candidates(current)

        if not candidate_actions:
            return Plan(
                actions=[],
                predicted_trajectory=[],
                expected_cost=current.distance(goal),
                confidence=0.0,
                reasoning="no_candidate_actions",
            )

        # 构建多分支动作序列
        action_sequences: list[list[Any]] = []
        for action in candidate_actions[:n_branches]:
            seq = [action] * max_horizon  # 零阶保持
            action_sequences.append(seq)

        # 多分支推演
        branches = self._multi_branch.predict_branches(current, action_sequences)
        evals = self._multi_branch.compare_branches(branches, goal)

        if not evals:
            return Plan(
                actions=[],
                predicted_trajectory=[],
                expected_cost=current.distance(goal),
                confidence=0.0,
                reasoning="no_valid_branches",
            )

        # 选最优
        best_eval = evals[0]
        best_idx = best_eval.branch_index
        best_trajectory = branches[best_idx]

        # 计算置信度
        initial_dist = current.distance(goal)
        improvement = 1.0 - (best_eval.final_distance / max(initial_dist, 1e-10))
        confidence = max(0.0, min(1.0, improvement))

        return Plan(
            actions=action_sequences[best_idx],
            predicted_trajectory=best_trajectory,
            expected_cost=best_eval.final_distance,
            confidence=round(confidence, 4),
            reasoning=f"multi_branch_selection: branch={best_idx}, improvement={improvement:.3f}",
            metadata={
                "n_branches": len(action_sequences),
                "horizon": max_horizon,
                "all_evals": [
                    {"branch": e.branch_index, "rank": e.rank, "distance": round(e.final_distance, 6)} for e in evals
                ],
            },
        )

    # -----------------------------------------------------------------
    # plan_with_lookahead — 穷举前瞻
    # -----------------------------------------------------------------

    def plan_with_lookahead(
        self,
        current: WorldState,
        goal: WorldState,
        horizon: int = 3,
        candidate_actions: list | None = None,
    ) -> Plan:
        """穷举前瞻：在有限候选动作中搜索最优动作序列。

        对 horizon=3 和 K 个候选动作，搜索 K^3 种组合（限制 K <= 5）。

        Args:
            current: 当前状态
            goal: 目标状态
            horizon: 前瞻步数
            candidate_actions: 候选动作（限制 <= 5 个）

        Returns:
            最优 Plan
        """
        self._plan_count += 1

        if candidate_actions is None:
            candidate_actions = self._generate_default_candidates(current)

        # 限制候选数以控制组合爆炸
        candidate_actions = candidate_actions[:5]
        n_candidates = len(candidate_actions)

        if n_candidates == 0:
            return Plan(
                actions=[],
                predicted_trajectory=[],
                expected_cost=current.distance(goal),
                confidence=0.0,
                reasoning="no_candidate_actions",
            )

        best_cost = float("inf")
        best_actions: list[Any] = []
        best_trajectory: list[Any] = []

        # 递归穷举（最多 5^3 = 125 种组合）
        def _search(
            state: WorldState,
            depth: int,
            actions_so_far: list[Any],
            trajectory_so_far: list[Any],
        ) -> None:
            nonlocal best_cost, best_actions, best_trajectory

            if depth >= horizon:
                cost = state.distance(goal)
                if cost < best_cost:
                    best_cost = cost
                    best_actions = list(actions_so_far)
                    best_trajectory = list(trajectory_so_far)
                return

            for action in candidate_actions:
                preds = self._predictor.predict(state.copy(), action, n_steps=1)
                next_state = preds[0]

                # 剪枝：如果已经比 best_cost 差且还在中间步，跳过
                if depth < horizon - 1:
                    current_dist = next_state.distance(goal)
                    if current_dist > best_cost * 2:
                        continue

                actions_so_far.append(action)
                trajectory_so_far.append(next_state)
                _search(next_state, depth + 1, actions_so_far, trajectory_so_far)
                actions_so_far.pop()
                trajectory_so_far.pop()

        _search(current, 0, [], [])

        initial_dist = current.distance(goal)
        improvement = 1.0 - (best_cost / max(initial_dist, 1e-10))
        confidence = max(0.0, min(1.0, improvement))

        return Plan(
            actions=best_actions,
            predicted_trajectory=best_trajectory,
            expected_cost=best_cost,
            confidence=round(confidence, 4),
            reasoning=f"lookahead_search: horizon={horizon}, "
            f"candidates={n_candidates}, "
            f"combinations={n_candidates**horizon}",
            metadata={
                "horizon": horizon,
                "n_candidates": n_candidates,
                "total_combinations": n_candidates**horizon,
            },
        )

    # -----------------------------------------------------------------
    # execute — 执行计划
    # -----------------------------------------------------------------

    def execute(
        self,
        plan: Plan,
        state: WorldState,
    ) -> tuple[Any, ...]:
        """执行计划（模拟执行），逐步推进状态。

        Args:
            plan: 要执行的计划
            state: 当前实际状态

        Returns:
            (final_state, report) — 最终状态 + 执行报告
        """
        self._execute_count += 1

        current = state.copy()
        executed_states: list[Any] = [current]
        surprises: list[Any] = []

        for i, action in enumerate(plan.actions):
            preds = self._predictor.predict(current, action, n_steps=1)
            current = preds[0]
            executed_states.append(current)

            # 惊奇检测（如果有预测轨迹）
            if self._surprise_detector and i < len(plan.predicted_trajectory):
                expected = plan.predicted_trajectory[i]
                sig = self._surprise_detector.compute_surprise(expected, current)
                if sig.is_anomaly:
                    surprises.append(
                        {
                            "step": i,
                            "surprise_score": sig.score,
                            "expected_vs_actual": sig.breakdown,
                        }
                    )

        report = {
            "n_steps_executed": len(plan.actions),
            "initial_distance": round(state.distance(current), 6),
            "final_state": current,
            "n_surprises": len(surprises),
            "surprises": surprises,
        }

        return current, report

    # -----------------------------------------------------------------
    # replan — 惊奇触发重规划
    # -----------------------------------------------------------------

    def replan(
        self,
        current: WorldState,
        goal: WorldState,
        surprise: SurpriseSignal | None = None,
        candidate_actions: list | None = None,
    ) -> Plan:
        """惊奇触发重规划：当实际偏离预期时重新规划。

        Args:
            current: 当前实际状态
            goal: 目标状态
            surprise: 触发重规划的惊奇信号（可选，用于记录原因）
            candidate_actions: 新候选动作（可选）

        Returns:
            新的 Plan
        """
        reason = "surprise_replan"
        if surprise is not None:
            reason = f"surprise_replan: score={surprise.score:.3f}, threshold={surprise.threshold:.3f}"

        new_plan = self.plan(
            current=current,
            goal=goal,
            candidate_actions=candidate_actions,
        )
        new_plan.reasoning = reason + " | " + new_plan.reasoning
        return new_plan

    # -----------------------------------------------------------------
    # _generate_default_candidates — 默认候选动作生成
    # -----------------------------------------------------------------

    def _generate_default_candidates(self, state: WorldState) -> list[Any]:
        """根据 WorldState 类型生成默认候选动作。

        v4.4.0: 泛化为支持 PendulumState / CartState / 任意 WorldState。
        """
        from mci_world_model.sdk._world_state import (
            CartAction,
            CartState,
            PendulumAction,
            PendulumState,
        )

        if isinstance(state, PendulumState):
            return [
                PendulumAction(torque=-5.0),
                PendulumAction(torque=-2.0),
                PendulumAction(torque=0.0),
                PendulumAction(torque=2.0),
                PendulumAction(torque=5.0),
            ]

        if isinstance(state, CartState):
            return [
                CartAction(force=-5.0),
                CartAction(force=-2.0),
                CartAction(force=0.0),
                CartAction(force=2.0),
                CartAction(force=5.0),
            ]

        # 通用 WorldState: 无法生成候选动作
        return []

    # -----------------------------------------------------------------
    # 字符串表示
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        return f"PlanAgent(predictor={self._predictor!r}, plans={self._plan_count}, executes={self._execute_count})"
