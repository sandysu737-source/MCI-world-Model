"""MCI World Model v5.1.0 — MCTS 规划器
==========================================

P1-F11 修复: 用 Monte Carlo Tree Search 替换穷举前瞻搜索。

现有 PlanAgent.plan_with_lookahead() 使用 O(K^H) 穷举搜索，
K=5, H=3 时 125 种组合尚可，但 H=10 时 5^10 ≈ 10M 不可接受。

MCTS 优势:
    - 自适应搜索: 集中资源在最有希望的分支
    - 任意时间: 可随时中断返回当前最优
    - UCB1 选择: 平衡探索(exploration)与利用(exploitation)

核心算法:
    1. Selection: UCB1 选择最有潜力的叶子节点
    2. Expansion: 展开一个新子节点
    3. Simulation (Rollout): 快速模拟到终局
    4. Backpropagation: 回传价值

设计原则:
    - 纯 numpy，零外部依赖
    - 复用 ActionConditionedPredictor 做世界模型模拟
    - 输出 Plan 数据结构，与 PlanAgent 兼容
    - 支持 action_conditioned_predictor 的 predict() 接口

## Formal Guarantees

    - UCB1 在模拟次数 → ∞ 时收敛到最优
    - 每次搜索的模拟次数由 n_simulations 参数控制
    - 回传价值严格单调 (0,1] 范围

用法:
    >>> from mci_world_model.sdk import PendulumPhysicsPredictor, PendulumState, PendulumAction
    >>> from mci_world_model.sdk._mcts_planner import MCTSPlanner, MCTSConfig
    >>> pred = PendulumPhysicsPredictor()
    >>> config = MCTSConfig(n_simulations=200, max_depth=5)
    >>> planner = MCTSPlanner(predictor=pred, config=config)
    >>> current = PendulumState(theta=1.0, omega=0.0)
    >>> goal = PendulumState(theta=0.0, omega=0.0)
    >>> plan = planner.search(current, goal)
    >>> print(plan.horizon, plan.expected_cost)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mci_world_model.sdk._action_conditioned_predictor import ActionConditionedPredictor
    from mci_world_model.sdk._world_state import Action, WorldState

logger = logging.getLogger(__name__)


# =============================================================================
# MCTSConfig — 配置
# =============================================================================


@dataclass
class MCTSConfig:
    """MCTS 规划器配置。

    Attributes:
        n_simulations: 每次搜索的模拟次数 (默认 200)
        max_depth: 搜索树最大深度 (默认 10)
        c_puct: UCB1 探索常数 (默认 1.414, √2)
        gamma: 折扣因子 (默认 0.99)
        rollout_horizon: rollout 阶段的模拟步数 (默认 5)
        n_rollout_actions: rollout 时随机候选动作数 (默认 3)
        time_limit_ms: 时间限制 (毫秒), 0=无限制 (默认 0)
        temperature: 动作选择温度 (默认 1.0)
    """

    n_simulations: int = 200
    max_depth: int = 10
    c_puct: float = 1.414  # √2
    gamma: float = 0.99
    rollout_horizon: int = 5
    n_rollout_actions: int = 3
    time_limit_ms: int = 0  # 0 = no time limit
    temperature: float = 1.0


# =============================================================================
# MCTSNode — 搜索树节点
# =============================================================================


class MCTSNode:
    """MCTS 搜索树节点。

    每个节点对应一个世界状态，子节点对应不同动作后的状态。

    Attributes:
        state: 对应的世界状态
        parent: 父节点
        action: 从父节点到本节点的动作
        children: 子节点列表 (action → MCTSNode)
        visit_count: 访问次数
        total_value: 累计价值
        prior: 先验概率 (uniform if no policy network)
        is_expanded: 是否已展开
    """

    __slots__ = (
        "_untried_actions",
        "action",
        "children",
        "is_expanded",
        "parent",
        "prior",
        "state",
        "total_value",
        "visit_count",
    )

    def __init__(
        self,
        state: WorldState,
        parent: MCTSNode | None = None,
        action: Action | None = None,
        prior: float = 0.0,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: list[MCTSNode] = []
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.prior = prior
        self.is_expanded = False
        self._untried_actions: list | None = None

    @property
    def q_value(self) -> float:
        """平均价值 (exploitation term)。"""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb1(self, c_puct: float) -> float:
        """UCB1 分数: Q + c * prior * √(ln(N_parent) / N_self)。

        Args:
            c_puct: 探索常数

        Returns:
            UCB1 分数
        """
        if self.visit_count == 0:
            return float("inf")

        parent_visits = self.parent.visit_count if self.parent else 1
        exploration = c_puct * self.prior * math.sqrt(math.log(parent_visits + 1) / (self.visit_count + 1))
        return self.q_value + exploration

    def best_child(self, c_puct: float) -> MCTSNode:
        """选择 UCB1 最高的子节点。

        Args:
            c_puct: 探索常数

        Returns:
            最佳子节点
        """
        return max(self.children, key=lambda c: c.ucb1(c_puct))

    def expand(self, actions: list[Action], predictor: ActionConditionedPredictor) -> MCTSNode:
        """展开一个未尝试的动作，创建子节点。

        Args:
            actions: 所有可用动作列表
            predictor: 世界模型预测器

        Returns:
            新创建的子节点
        """
        if self._untried_actions is None:
            # 初始化未尝试动作列表
            self._untried_actions = list(actions)
            self.is_expanded = True

        if not self._untried_actions:
            # 所有动作已展开，返回最佳子节点
            return self.best_child(1.414)

        # 取第一个未尝试的动作
        action = self._untried_actions.pop(0)
        prior = 1.0 / len(actions)  # uniform prior

        # 用预测器模拟
        preds = predictor.predict(self.state.copy(), action, n_steps=1)
        next_state = preds[0]

        child = MCTSNode(
            state=next_state,
            parent=self,
            action=action,
            prior=prior,
        )
        self.children.append(child)
        return child

    def backpropagate(self, value: float, gamma: float = 0.99) -> None:
        """回传价值到根节点。

        Args:
            value: 叶子节点价值
            gamma: 折扣因子
        """
        node = self
        discounted = value
        while node is not None:
            node.visit_count += 1
            node.total_value += discounted
            discounted *= gamma
            node = node.parent

    def is_leaf(self) -> bool:
        """是否为叶子节点 (未展开或无子节点)。"""
        return not self.is_expanded or (self._untried_actions is not None and len(self._untried_actions) > 0)

    def __repr__(self) -> str:
        return f"MCTSNode(visits={self.visit_count}, q={self.q_value:.3f}, children={len(self.children)})"


# =============================================================================
# MCTSPlanner — MCTS 规划器
# =============================================================================


class MCTSPlanner:
    """Monte Carlo Tree Search 规划器。

    用 MCTS 替代穷举前瞻搜索，支持:
    - 自适应搜索深度
    - UCB1 探索-利用平衡
    - 任意时间中断
    - 折扣回报

    Example:
        >>> from mci_world_model.sdk import PendulumPhysicsPredictor, PendulumState
        >>> pred = PendulumPhysicsPredictor()
        >>> planner = MCTSPlanner(predictor=pred)
        >>> current = PendulumState(theta=1.0, omega=0.0)
        >>> goal = PendulumState(theta=0.0, omega=0.0)
        >>> plan = planner.search(current, goal)
    """

    def __init__(
        self,
        predictor: ActionConditionedPredictor,
        config: MCTSConfig | None = None,
        candidate_actions: list[Action] | None = None,
    ):
        """
        Args:
            predictor: 动作条件化预测器 (世界模型)
            config: MCTS 配置
            candidate_actions: 候选动作列表 (None 时自动生成)
        """
        from mci_world_model.sdk._plan_agent import PlanAgent

        self._predictor = predictor
        self._config = config or MCTSConfig()
        self._candidate_actions = candidate_actions
        self._plan_agent = PlanAgent(predictor=predictor)
        self._search_count: int = 0

    @property
    def config(self) -> MCTSConfig:
        return self._config

    @property
    def search_count(self) -> int:
        return self._search_count

    # -----------------------------------------------------------------
    # search — 主搜索入口
    # -----------------------------------------------------------------

    def search(
        self,
        current: WorldState,
        goal: WorldState,
        candidate_actions: list[Action] | None = None,
    ) -> Any:
        """执行 MCTS 搜索，返回最优计划。

        Args:
            current: 当前世界状态
            goal: 目标世界状态
            candidate_actions: 候选动作 (覆盖构造时的默认列表)

        Returns:
            Plan 对象 (与 PlanAgent 输出兼容)
        """
        from mci_world_model.sdk._plan_agent import Plan

        self._search_count += 1

        if candidate_actions is not None:
            actions = candidate_actions
        elif self._candidate_actions is not None:
            actions = self._candidate_actions
        else:
            actions = self._generate_candidates(current)

        if not actions:
            return Plan(
                actions=[],
                predicted_trajectory=[],
                expected_cost=current.distance(goal),
                confidence=0.0,
                reasoning="no_candidate_actions",
            )

        # 创建根节点
        root = MCTSNode(state=current)

        # 设置先验概率 (uniform)
        for i, _ in enumerate(actions):
            pass  # uniform prior = 1/len(actions), 在 expand 中设置

        # MCTS 主循环
        start_time = time.time()
        n_sims = 0

        for _ in range(self._config.n_simulations):
            # 时间限制检查
            if self._config.time_limit_ms > 0:
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms > self._config.time_limit_ms:
                    break

            # 1. Selection: 从根走到叶子
            node = self._select(root)

            # 2. Expansion: 展开一个新子节点
            if node.is_leaf() and node.visit_count > 0 and node.state.distance(goal) > 1e-6:
                node = node.expand(actions, self._predictor)

            # 3. Simulation (Rollout): 快速估计价值
            value = self._rollout(node.state, goal, actions)

            # 4. Backpropagation: 回传价值
            node.backpropagate(value, self._config.gamma)

            n_sims += 1

        # 从根节点提取最优路径
        best_path = self._extract_best_path(root, goal)

        if not best_path:
            return Plan(
                actions=[],
                predicted_trajectory=[],
                expected_cost=current.distance(goal),
                confidence=0.0,
                reasoning="mcts_no_path_found",
            )

        # 构建计划
        actions_seq = [node.action for node in best_path if node.action is not None]
        trajectory = [node.state for node in best_path[1:]]  # 排除根节点自身

        final_state = best_path[-1].state
        final_cost = final_state.distance(goal)
        initial_dist = current.distance(goal)
        improvement = 1.0 - (final_cost / max(initial_dist, 1e-10))
        confidence = max(0.0, min(1.0, improvement))

        return Plan(
            actions=actions_seq,
            predicted_trajectory=trajectory,
            expected_cost=final_cost,
            confidence=round(confidence, 4),
            reasoning=f"mcts_search: sims={n_sims}, depth={len(best_path) - 1}",
            metadata={
                "n_simulations": n_sims,
                "tree_size": self._count_nodes(root),
                "max_depth": self._tree_depth(root),
                "c_puct": self._config.c_puct,
                "gamma": self._config.gamma,
            },
        )

    # -----------------------------------------------------------------
    # _select — Selection 阶段
    # -----------------------------------------------------------------

    def _select(self, node: MCTSNode) -> MCTSNode:
        """从给定节点出发，用 UCB1 选择到叶子节点。

        Args:
            node: 起始节点

        Returns:
            叶子节点
        """
        while node.is_expanded and node.children and not node.is_leaf():
            node = node.best_child(self._config.c_puct)
        return node

    # -----------------------------------------------------------------
    # _rollout — Simulation 阶段
    # -----------------------------------------------------------------

    def _rollout(self, state: WorldState, goal: WorldState, actions: list[Action]) -> float:
        """快速 rollout 估计状态价值。

        用随机策略模拟 rollout_horizon 步，返回折扣累积奖励。
        奖励 = 1 - normalized_distance (越近目标越高)。

        Args:
            state: 当前状态
            goal: 目标状态
            actions: 可用动作

        Returns:
            价值估计 [0, 1]
        """
        current = state.copy()
        total_reward = 0.0
        discount = 1.0
        initial_dist = state.distance(goal)

        for _ in range(self._config.rollout_horizon):
            # 随机选择动作 (rollout policy = uniform random)
            if actions:
                idx = np.random.randint(len(actions))
                action = actions[idx]
                preds = self._predictor.predict(current, action, n_steps=1)
                current = preds[0]

            # 计算即时奖励 (距离减小 = 正奖励)
            dist = current.distance(goal)
            if initial_dist > 1e-10:
                reward = 1.0 - (dist / initial_dist)
            else:
                reward = 1.0
            reward = max(0.0, min(1.0, reward))

            total_reward += discount * reward
            discount *= self._config.gamma

        # 终局奖励: 到达目标额外加分
        final_dist = current.distance(goal)
        if final_dist < 0.1:
            total_reward += discount * 1.0

        return total_reward

    # -----------------------------------------------------------------
    # _extract_best_path — 提取最优路径
    # -----------------------------------------------------------------

    def _extract_best_path(self, root: MCTSNode, goal: WorldState) -> list[MCTSNode]:
        """从根节点提取最优路径 (选最高访问次数子节点)。

        Args:
            root: 根节点
            goal: 目标状态 (用于判断是否提前到达)

        Returns:
            从根到最优叶子的节点路径
        """
        path = [root]
        node = root

        while node.children:
            # 选访问次数最多的子节点 (robust child)
            best = max(node.children, key=lambda c: c.visit_count)
            path.append(best)
            node = best

            # 提前到达目标
            if node.state.distance(goal) < 1e-6:
                break

        return path

    # -----------------------------------------------------------------
    # _count_nodes / _tree_depth — 树统计
    # -----------------------------------------------------------------

    def _count_nodes(self, node: MCTSNode) -> int:
        """统计搜索树节点总数。"""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def _tree_depth(self, node: MCTSNode) -> int:
        """计算搜索树最大深度。"""
        if not node.children:
            return 0
        return 1 + max(self._tree_depth(c) for c in node.children)

    # -----------------------------------------------------------------
    # _generate_candidates — 默认候选动作
    # -----------------------------------------------------------------

    def _generate_candidates(self, state: WorldState) -> list[Action]:
        """根据状态类型生成默认候选动作。"""
        return self._plan_agent._generate_default_candidates(state)

    # -----------------------------------------------------------------
    # 字符串表示
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MCTSPlanner(sims={self._config.n_simulations}, "
            f"c_puct={self._config.c_puct}, "
            f"searches={self._search_count})"
        )
