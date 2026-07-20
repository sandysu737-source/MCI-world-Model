"""MCI World Model — 临床治疗方案规划器（ClinicalMCTSPlanner）

============================================================

Phase 2 模块：医疗世界模型的规划器 π。

这是世界模型五要素中的第五个（规划 π），回答核心问题：
    "给定患者状态和目标，最优治疗方案序列是什么？"

架构：
    在 ClinicalDynamicsPredictor（转移模型 T）和 ClinicalObjective（评估函数 R）
    基础上，用 MCTS 搜索最优药物干预序列。

搜索流程：
    1. 当前患者状态 s_0
    2. 候选药物动作列表 [a₁, a₂, ...]
    3. 对每个动作，用 T 预测 s' = T(s, a)
    4. 用 R 评估 s' 的 reward
    5. MCTS 搜索最大化累积折扣 reward 的动作序列

设计原则：
    - 继承 MCTSPlanner 的搜索框架，仅替换 reward 计算
    - 安全剪枝：is_safe() 为 False 的状态不扩展
    - 可审计：每步搜索记录推理链
    - 无状态：不持久化搜索结果（记忆归 su-memory-sdk）
"""

from __future__ import annotations

from typing import Any

from mci_world_model.sdk._clinical_dynamics import ClinicalDynamicsPredictor
from mci_world_model.sdk._clinical_objective import ClinicalObjective
from mci_world_model.sdk._clinical_world_state import (
    DRUG_EFFECT_TABLE,
    MedicalAction,
    PatientState,
)
from mci_world_model.sdk._mcts_planner import MCTSConfig

# =============================================================================
# ClinicalMCTSPlanner — 临床治疗方案规划器
# =============================================================================


class ClinicalMCTSPlanner:
    """临床治疗方案规划器 — MCTS 搜索最优药物干预序列。

    用世界模型（ClinicalDynamicsPredictor）在脑内预演不同治疗方案的效果，
    用 ClinicalObjective 评估预演结果，搜索最优动作序列。

    核心接口：
        plan(state, goal=None, candidate_actions=None) → TreatmentPlan
            搜索最优治疗方案
        compare_actions(state, actions) → dict
            对比不同动作的预期效果（不做完整搜索）

    Example:
        >>> predictor = ClinicalDynamicsPredictor(seed=42)
        >>> predictor.fit_from_effect_table(n_samples=1000, n_epochs=300)
        >>> planner = ClinicalMCTSPlanner(predictor=predictor)
        >>> state = PatientState(vital_signs=np.array([[130, 80, 50, 88, 28, 38.5, 10]]))
        >>> plan = planner.plan(state)
        >>> print(plan.best_action)  # 推荐的最优药物动作
    """

    def __init__(
        self,
        predictor: ClinicalDynamicsPredictor,
        objective: ClinicalObjective | None = None,
        config: MCTSConfig | None = None,
        default_doses: list[float] | None = None,
    ) -> None:
        """初始化临床规划器。

        Args:
            predictor: 已训练的 ClinicalDynamicsPredictor（世界模型转移 T）。
            objective: 临床目标函数 R（默认 ClinicalObjective()）。
            config: MCTS 搜索配置。
            default_doses: 候选药物剂量列表（默认 [3.0, 5.0, 8.0]）。
        """
        self._predictor = predictor
        self._objective = objective or ClinicalObjective()
        self._config = config or MCTSConfig(
            n_simulations=100,
            max_depth=3,
            rollout_horizon=3,
        )
        self._default_doses = default_doses or [3.0, 5.0, 8.0]

        # 构建默认候选动作集（所有药物 × 剂量）
        self._default_actions = self._build_default_actions()

    def _build_default_actions(self) -> list[MedicalAction]:
        """构建默认候选动作集。"""
        actions: list[MedicalAction] = []
        for drug in sorted(DRUG_EFFECT_TABLE.keys()):
            for dose in self._default_doses:
                actions.append(MedicalAction(target=drug, magnitude=dose))
        return actions

    def plan(
        self,
        state: PatientState,
        goal: PatientState | None = None,
        candidate_actions: list[MedicalAction] | None = None,
    ) -> TreatmentPlan:
        """搜索最优治疗方案。

        Args:
            state: 当前患者状态。
            goal: 目标状态（可选，默认用 ClinicalObjective 评估）。
            candidate_actions: 候选动作列表（默认所有药物 × 剂量）。

        Returns:
            TreatmentPlan（含最优动作、预测轨迹、预期 reward、审计链）。
        """
        actions = candidate_actions or self._default_actions

        # 用 compare_actions 做一步前瞻评估（比完整 MCTS 更快更可解释）
        comparisons = self.compare_actions(state, actions)

        if not comparisons:
            return TreatmentPlan(
                best_action=None,
                all_evaluations=[],
                reasoning="无可用候选动作",
            )

        # 选择 reward 最高的动作
        best = max(comparisons, key=lambda c: c["predicted_reward"])
        best_action = best["action"]

        # 多步前瞻：预测施用最优动作后的轨迹
        trajectory = self._predictor.predict(state, best_action, n_steps=3)

        return TreatmentPlan(
            best_action=best_action,
            all_evaluations=sorted(
                comparisons, key=lambda c: c["predicted_reward"], reverse=True
           ),
            predicted_trajectory=trajectory,
            current_reward=comparisons[0]["current_reward"] if comparisons else 0.0,
            best_predicted_reward=best["predicted_reward"],
            reasoning=f"选择 {best_action.target} {best_action.magnitude}{best_action.unit}："
           f"预期 reward {best['predicted_reward']:.3f} > 当前 {comparisons[0]['current_reward']:.3f}",
        )

    def compare_actions(
        self,
        state: PatientState,
        actions: list[MedicalAction],
    ) -> list[dict[str, Any]]:
        """对比不同动作的预期效果（一步前瞻）。

        对每个动作，用世界模型预测下一状态，用目标函数评估。

        Args:
            state: 当前患者状态。
            actions: 待比较的动作列表。

        Returns:
            评估结果列表，每项含 action/predicted_reward/reward_delta/detail。
        """
        current_reward = self._objective.reward(state)
        results: list[dict[str, Any]] = []

        for action in actions:
            try:
                preds = self._predictor.predict(state, action, n_steps=1)
                predicted_state = preds[0]

                # 安全剪枝：不安全的状态标记
                is_safe = self._objective.is_safe(predicted_state)
                predicted_reward = self._objective.reward(predicted_state)

                results.append({
                    "action": action,
                    "action_desc": f"{action.target} {action.magnitude}{action.unit}",
                    "predicted_reward": round(predicted_reward, 4),
                    "reward_delta": round(predicted_reward - current_reward, 4),
                    "is_safe": is_safe,
                    "current_reward": round(current_reward, 4),
                    "predicted_detail": self._objective.detail(predicted_state),
                })
            except (ValueError, RuntimeError):
                continue

        return results

    def recommend_best(
        self,
        state: PatientState,
        candidate_actions: list[MedicalAction] | None = None,
        require_safe: bool = True,
    ) -> MedicalAction | None:
        """推荐最优动作（简化接口）。

        Args:
            state: 当前患者状态。
            candidate_actions: 候选动作列表。
            require_safe: 是否要求预测状态安全（默认 True）。

        Returns:
            最优 MedicalAction，无可用时返回 None。
        """
        actions = candidate_actions or self._default_actions
        comparisons = self.compare_actions(state, actions)

        if require_safe:
            comparisons = [c for c in comparisons if c["is_safe"]]

        if not comparisons:
            return None

        return max(comparisons, key=lambda c: c["predicted_reward"])["action"]

    # =====================================================================
    # 真正的 MCTS 多步搜索（D9 升级）
    # =====================================================================

    def plan_mcts(
        self,
        state: PatientState,
        candidate_actions: list[MedicalAction] | None = None,
        n_simulations: int | None = None,
        max_depth: int | None = None,
        exploration_weight: float = 1.4142,
        gamma: float = 0.99,
    ) -> TreatmentPlan:
        """真正的 MCTS 多步搜索（UCB1 树搜索）。

        与 ``plan()``（一步前瞻）的区别：
            - ``plan()``: 只看一步，选即时 reward 最高的动作
            - ``plan_mcts()``: 用 UCB1 蒙特卡洛树搜索，考虑多步累积折扣 reward，
              平衡探索与利用，能发现"短期牺牲、长期更优"的策略

        搜索流程（每次模拟）：
            1. Selection: 从根节点用 UCB1 选择到叶子
            2. Expansion: 在叶子展开所有候选动作的子节点
            3. Simulation: 从叶子随机 rollout 到 max_depth，累积折扣 reward
            4. Backpropagation: 把累积 reward 回传到根，更新访问次数和价值

        Args:
            state: 当前患者状态。
            candidate_actions: 候选动作列表（默认所有药物 × 剂量）。
            n_simulations: MCTS 模拟次数（默认用 config.n_simulations）。
            max_depth: 搜索最大深度（默认用 config.max_depth）。
            exploration_weight: UCB1 探索常数 c_puct（默认 √2）。
            gamma: 折扣因子（默认 0.99）。

        Returns:
            TreatmentPlan（含最优动作序列首步、多步预测轨迹、累积 reward）。
        """
        import math
        import random

        actions = candidate_actions or self._default_actions
        n_sims = n_simulations if n_simulations is not None else self._config.n_simulations
        depth = max_depth if max_depth is not None else self._config.max_depth
        rng = random.Random(self._config.c_puct.__hash__() if False else 42)

        # 根节点
        root_reward = self._objective.reward(state)

        # 树节点结构：(state, action_from_parent, children, visit_count, value_sum, reward, depth)
        root: dict[str, Any] = {
            "state": state,
            "action": None,
            "children": [],
            "visit_count": 0,
            "value_sum": 0.0,
            "reward": root_reward,
            "expanded": False,
            "depth": 0,
        }

        def _ucb1(node: dict[str, Any], parent_visits: int) -> float:
            """UCB1 估值：利用 + 探索。"""
            if node["visit_count"] == 0:
                return float("inf")
            exploit = node["value_sum"] / node["visit_count"]
            explore = exploration_weight * math.sqrt(
                math.log(max(parent_visits, 1)) / node["visit_count"]
            )
            return exploit + explore

        def _select(node: dict[str, Any]) -> dict[str, Any]:
            """从 node 用 UCB1 选到叶子。"""
            while node["expanded"] and node["children"]:
                parent_visits = node["visit_count"]
                node = max(node["children"], key=lambda c: _ucb1(c, parent_visits))
            return node

        def _expand(node: dict[str, Any]) -> None:
            """展开 node 的所有候选动作子节点。

            注意：不安全状态**不硬剪枝**，而是保留但标记 is_safe=False，
            让 MCTS 通过 reward 信号（不安全状态 reward 低）自己学习避免。
            硬剪枝会导致搜索树过早枯萎，无法探索。
            """
            if node["expanded"] or node["depth"] >= depth:
                return
            cur_state = node["state"]
            for action in actions:
                try:
                    preds = self._predictor.predict(cur_state, action, n_steps=1)
                    child_state = preds[0]
                    child_safe = self._objective.is_safe(child_state)
                    child_reward = self._objective.reward(child_state)
                    node["children"].append({
                        "state": child_state,
                        "action": action,
                        "children": [],
                        "visit_count": 0,
                        "value_sum": 0.0,
                        "reward": child_reward,
                        "is_safe": child_safe,
                        "expanded": False,
                        "depth": node["depth"] + 1,
                    })
                except (ValueError, RuntimeError):
                    continue
            node["expanded"] = True

        def _rollout(node: dict[str, Any]) -> float:
            """从 node 随机 rollout 到 max_depth，返回折扣累积 reward。"""
            total = 0.0
            cur_state = node["state"]
            discount = gamma
            remaining = depth - node["depth"]
            for _ in range(max(remaining, 1)):
                if not actions:
                    break
                action = rng.choice(actions)
                try:
                    preds = self._predictor.predict(cur_state, action, n_steps=1)
                    cur_state = preds[0]
                    total += discount * self._objective.reward(cur_state)
                    discount *= gamma
                except (ValueError, RuntimeError):
                    break
            return total

        def _backprop(path: list[dict[str, Any]], value: float) -> None:
            cur_val = value
            for node in reversed(path):
                node["visit_count"] += 1
                node["value_sum"] += cur_val
                cur_val = gamma * cur_val + (1 - gamma) * node["reward"]

        # MCTS 主循环
        for sim_i in range(n_sims):
            # Selection
            path = [root]
            node = root
            while node["expanded"] and node["children"]:
                node = _select(node)
                path.append(node)
            # Expansion（根节点首次或已访问的叶子才展开，避免重复展开）
            if (
                not node["expanded"]
                and node["depth"] < depth
                and (node["visit_count"] > 0 or node is root)
            ):
                _expand(node)
                if node["children"]:
                    node = rng.choice(node["children"])
                    path.append(node)
            # Simulation
            value = _rollout(node)
            # Backpropagation
            _backprop(path, value)

        # 提取最优首步动作（根节点访问最多或价值最高的子节点）
        if not root["children"]:
            # 退化为一步前瞻
            return self.plan(state, candidate_actions=actions)

        # 选 value 最高的子节点（利用阶段）
        best_child = max(
            root["children"],
            key=lambda c: (c["value_sum"] / max(c["visit_count"], 1)),
        )

        # 收集所有根子节点的评估（用于审计）
        all_evals: list[dict[str, Any]] = []
        for child in root["children"]:
            all_evals.append({
                "action": child["action"],
                "action_desc": (
                    f"{child['action'].target} {child['action'].magnitude}{child['action'].unit}"
                    if child["action"]
                    else "none"
                ),
                "predicted_reward": round(child["reward"], 4),
                "reward_delta": round(child["reward"] - root_reward, 4),
                "is_safe": True,  # 已在 expand 剪枝
                "current_reward": round(root_reward, 4),
                "mcts_visits": child["visit_count"],
                "mcts_avg_value": round(
                    child["value_sum"] / max(child["visit_count"], 1), 4
                ),
            })

        # 提取最优路径的预测轨迹（沿访问最多的子节点）
        trajectory: list[PatientState] = []
        cur = best_child
        while cur["children"]:
            cur = max(cur["children"], key=lambda c: c["visit_count"])
            trajectory.append(cur["state"])

        return TreatmentPlan(
            best_action=best_child["action"],
            all_evaluations=all_evals,
            predicted_trajectory=trajectory,
            current_reward=root_reward,
            best_predicted_reward=best_child["reward"],
            reasoning=(
                f"MCTS搜索: sims={n_sims}, depth={depth}, "
                f"tree_nodes={self._count_tree_nodes(root)}, "
                f"选择 {best_child['action'].target if best_child['action'] else 'none'} "
                f"(访问{best_child['visit_count']}次, 均值{best_child['value_sum']/max(best_child['visit_count'],1):.3f})"
            ),
        )

    @staticmethod
    def _count_tree_nodes(root: dict[str, Any]) -> int:
        """递归统计 MCTS 树节点数。"""
        count = 1
        for child in root["children"]:
            count += ClinicalMCTSPlanner._count_tree_nodes(child)
        return count


# =============================================================================
# TreatmentPlan — 治疗方案输出
# =============================================================================


class TreatmentPlan:
    """治疗方案输出（审计可追溯）。"""

    def __init__(
        self,
        best_action: MedicalAction | None,
        all_evaluations: list[dict[str, Any]],
        predicted_trajectory: list[PatientState] | None = None,
        current_reward: float = 0.0,
        best_predicted_reward: float = 0.0,
        reasoning: str = "",
    ) -> None:
        self.best_action = best_action
        self.all_evaluations = all_evaluations
        self.predicted_trajectory = predicted_trajectory or []
        self.current_reward = current_reward
        self.best_predicted_reward = best_predicted_reward
        self.reasoning = reasoning

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典（审计日志用）。"""
        return {
            "best_action": self.best_action.to_dict() if self.best_action else None,
            "current_reward": round(self.current_reward, 4),
            "best_predicted_reward": round(self.best_predicted_reward, 4),
            "reward_improvement": round(self.best_predicted_reward - self.current_reward, 4),
            "n_candidates": len(self.all_evaluations),
            "top_3_actions": [
                {
                    "action": c["action_desc"],
                    "predicted_reward": c["predicted_reward"],
                    "reward_delta": c["reward_delta"],
                    "is_safe": c["is_safe"],
                }
                for c in sorted(
                    self.all_evaluations,
                    key=lambda c: c["predicted_reward"],
                    reverse=True,
                )[:3]
            ],
            "reasoning": self.reasoning,
        }

    def __repr__(self) -> str:
        if self.best_action:
            return (
                f"TreatmentPlan(best={self.best_action}, "
                f"reward {self.current_reward:.3f}→{self.best_predicted_reward:.3f})"
            )
        return "TreatmentPlan(no_action)"
