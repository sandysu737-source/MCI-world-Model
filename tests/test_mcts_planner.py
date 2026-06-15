"""MCI World Model v5.1.0 — MCTS 规划器 测试

P1-F11 修复验证: 用 MCTS 替换穷举前瞻搜索。
"""

from __future__ import annotations

import pytest

from mci_world_model.sdk._mcts_planner import MCTSConfig, MCTSNode, MCTSPlanner
from mci_world_model.sdk._plan_agent import Plan
from mci_world_model.sdk._world_state import PendulumAction, PendulumState

# ═══════════════════════════════════════════════════════════════════════════
# Test F11: MCTS 替换穷举前瞻
# ═══════════════════════════════════════════════════════════════════════════


class TestF11Fix:
    """P1-F11 修复: MCTS 规划器存在且可搜索。"""

    def test_mcts_planner_exists(self):
        """MCTSPlanner 可实例化。"""
        from mci_world_model.sdk import PendulumPhysicsPredictor

        pred = PendulumPhysicsPredictor()
        planner = MCTSPlanner(predictor=pred)
        assert planner is not None

    def test_mcts_config_exists(self):
        """MCTSConfig 可实例化且有合理默认值。"""
        config = MCTSConfig()
        assert config.n_simulations == 200
        assert config.c_puct > 0
        assert 0 < config.gamma <= 1.0

    def test_mcts_importable_from_sdk(self):
        """MCTS 类可从 sdk 顶层导入。"""
        from mci_world_model.sdk import MCTSConfig, MCTSNode, MCTSPlanner

        assert MCTSPlanner is not None
        assert MCTSNode is not None
        assert MCTSConfig is not None

    def test_search_returns_plan(self):
        """search() 返回 Plan 对象。"""
        from mci_world_model.sdk import PendulumPhysicsPredictor

        pred = PendulumPhysicsPredictor()
        config = MCTSConfig(n_simulations=20, max_depth=3)
        planner = MCTSPlanner(predictor=pred, config=config)

        current = PendulumState(theta=1.0, omega=0.0)
        goal = PendulumState(theta=0.0, omega=0.0)

        plan = planner.search(current, goal)
        assert isinstance(plan, Plan)


# ═══════════════════════════════════════════════════════════════════════════
# MCTSNode 测试
# ═══════════════════════════════════════════════════════════════════════════


class TestMCTSNode:
    """MCTS 树节点单元测试。"""

    def test_node_creation(self):
        """节点创建属性正确。"""
        state = PendulumState(theta=0.5, omega=0.1)
        node = MCTSNode(state=state)
        assert node.visit_count == 0
        assert node.total_value == 0.0
        assert node.q_value == 0.0
        assert not node.is_expanded

    def test_ucb1_unvisited_is_inf(self):
        """未访问节点的 UCB1 为 ∞。"""
        state = PendulumState(theta=0.5, omega=0.1)
        parent = MCTSNode(state=PendulumState(theta=0.3, omega=0.0))
        parent.visit_count = 10
        child = MCTSNode(state=state, parent=parent, prior=0.5)
        assert child.ucb1(1.414) == float("inf")

    def test_ucb1_visited_finite(self):
        """已访问节点的 UCB1 为有限值。"""
        state = PendulumState(theta=0.5, omega=0.1)
        parent = MCTSNode(state=PendulumState(theta=0.3, omega=0.0))
        parent.visit_count = 10
        child = MCTSNode(state=state, parent=parent, prior=0.5)
        child.visit_count = 5
        child.total_value = 2.0
        ucb = child.ucb1(1.414)
        assert 0 < ucb < float("inf")

    def test_backpropagate(self):
        """回传价值更新访问次数和总价值。"""
        state = PendulumState(theta=0.5, omega=0.1)
        node = MCTSNode(state=state)
        node.backpropagate(1.0, gamma=0.99)
        assert node.visit_count == 1
        assert node.total_value == 1.0

        node.backpropagate(0.5, gamma=0.99)
        assert node.visit_count == 2
        assert node.total_value == 1.5

    def test_backpropagate_with_parent(self):
        """回传价值传递到父节点。"""
        parent = MCTSNode(state=PendulumState(theta=0.3, omega=0.0))
        child = MCTSNode(state=PendulumState(theta=0.5, omega=0.1), parent=parent)

        child.backpropagate(1.0, gamma=0.99)
        assert parent.visit_count == 1
        assert abs(parent.total_value - 0.99) < 1e-6  # discounted

    def test_best_child(self):
        """best_child 选择 UCB1 最高的子节点。"""
        root = MCTSNode(state=PendulumState(theta=0.3, omega=0.0))
        root.visit_count = 100
        root.is_expanded = True

        # 高价值子节点
        c1 = MCTSNode(state=PendulumState(theta=0.1, omega=0.0), parent=root, prior=0.5)
        c1.visit_count = 50
        c1.total_value = 40.0

        # 低价值子节点
        c2 = MCTSNode(state=PendulumState(theta=0.9, omega=0.0), parent=root, prior=0.5)
        c2.visit_count = 50
        c2.total_value = 10.0

        root.children = [c1, c2]
        best = root.best_child(1.414)
        assert best is c1  # c1 Q=0.8 > c2 Q=0.2

    def test_is_leaf_unexpanded(self):
        """未展开的节点是叶子。"""
        node = MCTSNode(state=PendulumState(theta=0.5, omega=0.1))
        assert node.is_leaf()


# ═══════════════════════════════════════════════════════════════════════════
# MCTSPlanner 搜索测试
# ═══════════════════════════════════════════════════════════════════════════


class TestMCTSPlannerSearch:
    """MCTS 搜索集成测试。"""

    @pytest.fixture
    def planner(self):
        from mci_world_model.sdk import PendulumPhysicsPredictor

        pred = PendulumPhysicsPredictor()
        config = MCTSConfig(n_simulations=50, max_depth=5)
        return MCTSPlanner(predictor=pred, config=config)

    def test_search_finds_plan(self, planner):
        """搜索返回有效计划。"""
        current = PendulumState(theta=1.0, omega=0.0)
        goal = PendulumState(theta=0.0, omega=0.0)

        plan = planner.search(current, goal)
        assert isinstance(plan, Plan)
        assert plan.horizon >= 0
        assert plan.expected_cost >= 0

    def test_search_reduces_cost(self, planner):
        """MCTS 搜索应找到某条路径 (代价合理)。"""
        current = PendulumState(theta=1.0, omega=0.0)
        goal = PendulumState(theta=0.0, omega=0.0)

        plan = planner.search(current, goal)
        # MCTS 在有限模拟次数下可能不比初始好，但代价应有限
        assert plan.expected_cost < 10.0

    def test_search_with_custom_actions(self, planner):
        """自定义候选动作可传入搜索。"""
        current = PendulumState(theta=1.0, omega=0.0)
        goal = PendulumState(theta=0.0, omega=0.0)
        actions = [
            PendulumAction(torque=-2.0),
            PendulumAction(torque=0.0),
            PendulumAction(torque=2.0),
        ]

        plan = planner.search(current, goal, candidate_actions=actions)
        assert isinstance(plan, Plan)

    def test_search_empty_actions(self, planner):
        """空候选动作返回空计划。"""
        current = PendulumState(theta=1.0, omega=0.0)
        goal = PendulumState(theta=0.0, omega=0.0)

        plan = planner.search(current, goal, candidate_actions=[])
        assert plan.horizon == 0
        assert plan.reasoning == "no_candidate_actions"

    def test_search_at_goal(self, planner):
        """已在目标状态时搜索立即返回。"""
        current = PendulumState(theta=0.0, omega=0.0)
        goal = PendulumState(theta=0.0, omega=0.0)

        plan = planner.search(current, goal)
        assert isinstance(plan, Plan)
        # 已在目标，代价应很小
        assert plan.expected_cost < 0.1

    def test_metadata_contains_tree_info(self, planner):
        """计划元数据包含树信息。"""
        current = PendulumState(theta=1.0, omega=0.0)
        goal = PendulumState(theta=0.0, omega=0.0)

        plan = planner.search(current, goal)
        assert "n_simulations" in plan.metadata
        assert "tree_size" in plan.metadata
        assert "max_depth" in plan.metadata
        assert plan.metadata["n_simulations"] > 0


# ═══════════════════════════════════════════════════════════════════════════
# MCTSConfig 测试
# ═══════════════════════════════════════════════════════════════════════════


class TestMCTSConfig:
    """MCTS 配置测试。"""

    def test_default_values(self):
        """默认值合理。"""
        config = MCTSConfig()
        assert config.n_simulations > 0
        assert config.max_depth > 0
        assert config.c_puct > 0
        assert 0 < config.gamma <= 1.0
        assert config.rollout_horizon > 0

    def test_custom_values(self):
        """自定义值正确设置。"""
        config = MCTSConfig(
            n_simulations=500,
            max_depth=20,
            c_puct=2.0,
            gamma=0.95,
        )
        assert config.n_simulations == 500
        assert config.max_depth == 20
        assert config.c_puct == 2.0
        assert config.gamma == 0.95

    def test_time_limit(self):
        """时间限制配置。"""
        config = MCTSConfig(time_limit_ms=100)
        assert config.time_limit_ms == 100


# ═══════════════════════════════════════════════════════════════════════════
# MCTS vs 穷举前瞻对比
# ═══════════════════════════════════════════════════════════════════════════


class TestMCTSVsLookahead:
    """MCTS 与穷举前瞻对比测试。"""

    def test_mcts_similar_quality_to_lookahead(self):
        """MCTS 搜索质量与穷举前瞻相当。"""
        from mci_world_model.sdk import PendulumPhysicsPredictor
        from mci_world_model.sdk._plan_agent import PlanAgent

        pred = PendulumPhysicsPredictor()
        current = PendulumState(theta=1.0, omega=0.0)
        goal = PendulumState(theta=0.0, omega=0.0)

        # 穷举前瞻
        agent = PlanAgent(predictor=pred)
        lookahead_plan = agent.plan_with_lookahead(current, goal, horizon=3)

        # MCTS
        config = MCTSConfig(n_simulations=100, max_depth=3)
        planner = MCTSPlanner(predictor=pred, config=config)
        mcts_plan = planner.search(current, goal)

        # MCTS 代价不应显著差于穷举 (允许 3x 差距)
        assert mcts_plan.expected_cost <= lookahead_plan.expected_cost * 3 + 0.1

    def test_mcts_scales_better(self):
        """MCTS 在更深搜索时仍可运行 (穷举 O(K^H) 不可行)。"""
        from mci_world_model.sdk import PendulumPhysicsPredictor

        pred = PendulumPhysicsPredictor()
        current = PendulumState(theta=1.0, omega=0.0)
        goal = PendulumState(theta=0.0, omega=0.0)

        # depth=8 的 MCTS 仍可在合理时间内完成
        config = MCTSConfig(n_simulations=30, max_depth=8)
        planner = MCTSPlanner(predictor=pred, config=config)
        plan = planner.search(current, goal)
        assert isinstance(plan, Plan)


# ═══════════════════════════════════════════════════════════════════════════
# 属性与字符串表示
# ═══════════════════════════════════════════════════════════════════════════


class TestMCTSProperties:
    """MCTS 属性与 repr 测试。"""

    def test_search_count_increments(self):
        """搜索计数递增。"""
        from mci_world_model.sdk import PendulumPhysicsPredictor

        pred = PendulumPhysicsPredictor()
        planner = MCTSPlanner(predictor=pred, config=MCTSConfig(n_simulations=10, max_depth=2))

        assert planner.search_count == 0
        planner.search(PendulumState(theta=1.0, omega=0.0), PendulumState(theta=0.0, omega=0.0))
        assert planner.search_count == 1

    def test_repr(self):
        """repr 包含关键信息。"""
        from mci_world_model.sdk import PendulumPhysicsPredictor

        pred = PendulumPhysicsPredictor()
        planner = MCTSPlanner(predictor=pred)
        r = repr(planner)
        assert "MCTSPlanner" in r
        assert "sims=" in r

    def test_config_property(self):
        """config 属性可访问。"""
        config = MCTSConfig(n_simulations=100)
        planner = MCTSPlanner(predictor=None, config=config)  # type: ignore
        assert planner.config.n_simulations == 100
