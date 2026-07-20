from __future__ import annotations

"""MCI World Model v4.6.0 — SocialCognition 多智能体社会认知
=================================================================

多智能体博弈论 + 心智理论 — 世界模型的社会认知能力。

核心能力:
    observe_interaction(agent_id, action, outcome) — 观察其他智能体行为
    predict_others(context)                         — 心智理论: 预测他人行为
    nash_equilibrium(payoff_matrix)                 — 2x2 博弈纳什均衡
    negotiate(my_preferences, others_predicted)     — 社会协商

设计原则:
    - 纯 numpy，零外部依赖
    - 2x2 矩阵博弈 (简化版纳什均衡)
    - 心智理论深度 ≤ 2 层
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# AgentModel — 其他智能体的内部模型
# =============================================================================


@dataclass
class AgentAction:
    """智能体动作记录。

    Attributes:
        action: 动作名称
        outcome: 动作结果
        timestamp: 时间戳 (逻辑时钟)
    """

    action: str
    outcome: dict[str, Any] = field(default_factory=dict)
    timestamp: int = 0


class AgentModel:
    """其他智能体的内部模型 — 心智理论的基础。

    维护对某个智能体的偏好模型和行为预测。

    Attributes:
        agent_id: 智能体 ID
        action_history: 行为历史
        preference_model: 偏好模型 {action: count}
    """

    def __init__(self, agent_id: int) -> None:
        if agent_id < 0:
            raise ValueError(f"agent_id 必须 ≥ 0, 当前 {agent_id}")
        self._agent_id = agent_id
        self._action_history: list[AgentAction] = []
        self._preference_model: dict[str, int] = {}

    @property
    def agent_id(self) -> int:
        return self._agent_id

    def update(self, action: str, outcome: dict[str, Any], timestamp: int = 0) -> None:
        """更新智能体行为观测。

        Args:
            action: 观察到的动作
            outcome: 动作结果
            timestamp: 逻辑时钟
        """
        self._action_history.append(AgentAction(action=action, outcome=outcome, timestamp=timestamp))
        self._preference_model[action] = self._preference_model.get(action, 0) + 1

    def predict_action(self, context: dict[str, Any]) -> str:
        """基于历史行为预测下一步。

        简化策略: 选历史频率最高的动作。

        Args:
            context: 当前上下文 (预留, 当前未使用)

        Returns:
            预测的动作名称
        """
        if not self._preference_model:
            return "unknown"
        return max(self._preference_model, key=self._preference_model.get)  # type: ignore

    def get_action_probability(self) -> dict[str, float]:
        """获取动作概率分布。"""
        total = sum(self._preference_model.values())
        if total == 0:
            return {}
        return {a: c / total for a, c in self._preference_model.items()}

    @property
    def action_history(self) -> list[AgentAction]:
        return list(self._action_history)


# =============================================================================
# NashEquilibriumResult — 纳什均衡结果
# =============================================================================


@dataclass
class NashEquilibriumResult:
    """2x2 博弈纳什均衡结果。

    Attributes:
        player1_strategy: 玩家1混合策略 [p_cooperate, p_defect]
        player2_strategy: 玩家2混合策略
        player1_payoff: 玩家1期望收益
        player2_payoff: 玩家2期望收益
        is_pure: 是否纯策略均衡
        nash_type: 均衡类型 ('pure_dominant' / 'mixed' / 'pure_nash')
    """

    player1_strategy: np.ndarray
    player2_strategy: np.ndarray
    player1_payoff: float
    player2_payoff: float
    is_pure: bool
    nash_type: str = "mixed"


# =============================================================================
# SocialCognition — 多智能体社会认知
# =============================================================================


class SocialCognition:
    """多智能体社会认知 — 博弈论 + 心智理论。

    核心能力:
      - 观察其他智能体行为并建立内部模型
      - 心智理论: 预测其他智能体行为
      - 2x2 博弈纳什均衡计算
      - 社会协商: 基于 Pareto 最优

    Attributes:
        _n_agents: 智能体数
        _tom_depth: 心智理论递归深度
        _agent_models: 其他智能体的内部模型
    """

    def __init__(self, n_agents: int = 3, theory_of_mind_depth: int = 2) -> None:
        if n_agents < 2:
            raise ValueError(f"n_agents 必须 ≥ 2, 当前 {n_agents}")
        if theory_of_mind_depth < 1 or theory_of_mind_depth > 3:
            raise ValueError(f"theory_of_mind_depth 必须在 [1,3], 当前 {theory_of_mind_depth}")
        self._n_agents = n_agents
        self._tom_depth = theory_of_mind_depth
        self._agent_models: dict[int, AgentModel] = {}
        self._interaction_log: list[dict[str, Any]] = []

    def observe_interaction(self, agent_id: int, action: str, outcome: dict[str, Any], timestamp: int = 0) -> None:
        """观察其他智能体的行为。

        Args:
            agent_id: 被观察的智能体 ID
            action: 观察到的动作
            outcome: 动作结果
            timestamp: 逻辑时钟
        """
        if agent_id not in self._agent_models:
            self._agent_models[agent_id] = AgentModel(agent_id)
        self._agent_models[agent_id].update(action, outcome, timestamp)
        self._interaction_log.append(
            {
                "agent_id": agent_id,
                "action": action,
                "outcome": outcome,
                "timestamp": timestamp,
            }
        )

    def predict_others(self, context: dict[str, Any]) -> dict[int, str]:
        """心智理论: 预测其他智能体的行为。

        Args:
            context: 当前上下文信息

        Returns:
            {agent_id: predicted_action} 映射
        """
        predictions: dict[int, str] = {}
        for aid, model in self._agent_models.items():
            predictions[aid] = model.predict_action(context)
        return predictions

    def get_agent_probabilities(self) -> dict[int, dict[str, float]]:
        """获取所有已知智能体的动作概率分布。"""
        return {aid: model.get_action_probability() for aid, model in self._agent_models.items()}

    def nash_equilibrium(self, payoff_matrix: np.ndarray) -> NashEquilibriumResult:
        """2x2 博弈纳什均衡计算。

        Args:
            payoff_matrix: 收益矩阵 (2, 2, 2)
                payoff_matrix[i, j, 0] = 玩家1收益 (玩家1选i, 玩家2选j)
                payoff_matrix[i, j, 1] = 玩家2收益

        Returns:
            NashEquilibriumResult 纳什均衡结果
        """
        if payoff_matrix.shape != (2, 2, 2):
            raise ValueError(f"payoff_matrix 形状必须为 (2,2,2), 当前 {payoff_matrix.shape}")

        # 检查纯策略纳什均衡
        pure_result = self._find_pure_nash(payoff_matrix)
        if pure_result is not None:
            return pure_result

        # 混合策略纳什均衡
        return self._find_mixed_nash(payoff_matrix)

    def negotiate(self, my_preferences: dict[str, Any], others_predicted: dict[str, Any]) -> dict[str, Any]:
        """社会协商: 基于 Pareto 最优。

        简化实现: 寻找最大化联合偏好的动作。

        Args:
            my_preferences: 我的偏好 {action: utility}
            others_predicted: 其他智能体预测偏好 {agent_id: {action: utility}}

        Returns:
            协商结果 {"agreed_action": str, "my_utility": float, "joint_utility": float}
        """
        all_actions = set(my_preferences.keys())
        for prefs in others_predicted.values():
            all_actions.update(prefs.keys())

        best_action = ""
        best_joint = -np.inf
        best_my = 0.0

        for action in all_actions:
            my_util = my_preferences.get(action, 0.0)
            others_util = sum(prefs.get(action, 0.0) for prefs in others_predicted.values())
            joint = my_util + others_util
            if joint > best_joint:
                best_joint = joint
                best_action = action
                best_my = my_util

        return {
            "agreed_action": best_action,
            "my_utility": best_my,
            "joint_utility": best_joint,
        }

    def _find_pure_nash(self, payoff: np.ndarray) -> NashEquilibriumResult | None:
        """查找纯策略纳什均衡。"""
        for i in range(2):
            for j in range(2):
                # 玩家1不偏离
                p1_best = payoff[i, j, 0] >= payoff[1 - i, j, 0]
                # 玩家2不偏离
                p2_best = payoff[i, j, 1] >= payoff[i, 1 - j, 1]
                if p1_best and p2_best:
                    s1 = np.array([1.0, 0.0]) if i == 0 else np.array([0.0, 1.0])
                    s2 = np.array([1.0, 0.0]) if j == 0 else np.array([0.0, 1.0])
                    return NashEquilibriumResult(
                        player1_strategy=s1,
                        player2_strategy=s2,
                        player1_payoff=float(payoff[i, j, 0]),
                        player2_payoff=float(payoff[i, j, 1]),
                        is_pure=True,
                        nash_type="pure_nash",
                    )
        return None

    @staticmethod
    def _find_mixed_nash(payoff: np.ndarray) -> NashEquilibriumResult:
        """2x2 混合策略纳什均衡 (解析解)。

        p = (c - a) / (b + c - a - d)  其中 payoff = [[(a,e), (b,f)], [(c,g), (d,h)]]
        """
        a, b = payoff[0, 0, 0], payoff[0, 1, 0]
        c, d = payoff[1, 0, 0], payoff[1, 1, 0]
        e, f = payoff[0, 0, 1], payoff[0, 1, 1]
        g, h = payoff[1, 0, 1], payoff[1, 1, 1]

        # 玩家1混合策略
        denom1 = b + c - a - d
        if abs(denom1) < 1e-12:
            p1 = np.array([0.5, 0.5])
        else:
            p = np.clip((c - a) / denom1, 0.0, 1.0)
            p1 = np.array([p, 1 - p])

        # 玩家2混合策略
        denom2 = f + g - e - h
        if abs(denom2) < 1e-12:
            p2 = np.array([0.5, 0.5])
        else:
            q = np.clip((f - e) / denom2, 0.0, 1.0)
            p2 = np.array([q, 1 - q])

        # 期望收益
        exp1 = float(p1[0] * p2[0] * a + p1[0] * p2[1] * b + p1[1] * p2[0] * c + p1[1] * p2[1] * d)
        exp2 = float(p1[0] * p2[0] * e + p1[0] * p2[1] * f + p1[1] * p2[0] * g + p1[1] * p2[1] * h)

        return NashEquilibriumResult(
            player1_strategy=p1,
            player2_strategy=p2,
            player1_payoff=exp1,
            player2_payoff=exp2,
            is_pure=False,
            nash_type="mixed",
        )

    @property
    def n_agents(self) -> int:
        return self._n_agents

    @property
    def known_agents(self) -> list[int]:
        return list(self._agent_models.keys())

    @property
    def interaction_count(self) -> int:
        return len(self._interaction_log)
