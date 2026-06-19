"""
LOOP-07: cewm_step E2E 集成测试
================================

验证 cewm_step() 五层闭环全流程:
1. 感知层 → 2. 安全层 → 3. 认知层 → 4. 行动层 → 5. 预测层 → 6. 反馈层
以及 cewm_step_fast() 快速路径。
"""

import pytest


@pytest.fixture
def wm():
    """创建一个最小化的 MCIWorldModel 实例。"""
    from mci_world_model.sdk._world_model import MCIWorldModel

    return MCIWorldModel()


@pytest.fixture
def pendulum_factory():
    """PendulumState 工厂。"""
    from mci_world_model.sdk._world_state import PendulumState

    def _make(theta=0.0, omega=0.0):
        return PendulumState(theta=theta, omega=omega)

    return _make


# =============================================================================
# TestCEWMClosedLoop: cewm_step() 完整闭环
# =============================================================================


class TestCEWMClosedLoop:
    """验证 cewm_step() 五层闭环全流程。"""

    def test_perception_layer(self, wm, pendulum_factory):
        """感知层: PendulumState → 解析为 WorldState。"""
        obs = pendulum_factory(theta=0.3, omega=0.05)
        goal = pendulum_factory(theta=0.0, omega=0.0)
        result = wm.cewm_step(observation=obs, goal=goal)
        assert result["state"] is not None

    def test_safety_layer_no_violation(self, wm, pendulum_factory):
        """安全层: 无 SafetyMonitor 时不违规。"""
        obs = pendulum_factory(theta=0.1, omega=0.0)
        result = wm.cewm_step(observation=obs, goal=pendulum_factory())
        assert result["safety_violation"] is False

    def test_cognition_layer_accumulation(self, wm, pendulum_factory):
        """认知层: 多步因果图积累。"""
        goal = pendulum_factory(theta=0.0, omega=0.0)
        results = []
        for i in range(5):
            result = wm.cewm_step(
                observation=pendulum_factory(theta=0.1 * i, omega=0.01),
                goal=goal,
            )
            results.append(result)
        # 至少有一个步骤产生了因果更新
        total_updates = sum(r["causal_updates"] for r in results)
        assert total_updates > 0

    def test_action_layer_distance(self, wm, pendulum_factory):
        """行动层: 行动距离计算正确。"""
        obs = pendulum_factory(theta=0.5, omega=0.2)
        goal = pendulum_factory(theta=0.0, omega=0.0)
        result = wm.cewm_step(observation=obs, goal=goal)
        assert result["action_distance"] >= 0
        assert result["physical_distance"] >= 0

    def test_prediction_layer_embedding(self, wm, pendulum_factory):
        """预测层: JEPA 嵌入查询（无 predictor 时 prediction 可能为 None）。"""
        obs = pendulum_factory(theta=0.3, omega=0.0)
        result = wm.cewm_step(observation=obs, goal=pendulum_factory())
        assert "prediction" in result
        assert "prediction_error" in result
        assert 0.0 <= result["prediction_error"] <= 1.0

    def test_feedback_layer_attention(self, wm, pendulum_factory):
        """反馈层: 注意力权重基于预测误差调整。"""
        obs = pendulum_factory(theta=0.3, omega=0.0)
        result = wm.cewm_step(observation=obs, goal=pendulum_factory())
        assert "attention_weights" in result
        assert isinstance(result["attention_weights"], dict)

    def test_full_closed_loop_5_steps(self, wm, pendulum_factory):
        """完整闭环: 连续 5 步 cewm_step()。"""
        goal = pendulum_factory(theta=0.0, omega=0.0)
        for i in range(5):
            result = wm.cewm_step(
                observation=pendulum_factory(theta=0.1 * i, omega=0.01),
                goal=goal,
            )
            assert "state" in result
            assert "action_distance" in result
            assert "prediction_error" in result

    def test_result_keys_completeness(self, wm, pendulum_factory):
        """结果字典包含所有预期键。"""
        obs = pendulum_factory(theta=0.1, omega=0.0)
        result = wm.cewm_step(observation=obs, goal=pendulum_factory())
        expected_keys = {
            "state",
            "action_distance",
            "physical_distance",
            "prediction",
            "prediction_error",
            "causal_updates",
            "attention_weights",
            "experience_hints",
            "safety_violation",
            "safety_reason",
        }
        assert expected_keys.issubset(set(result.keys()))


# =============================================================================
# TestCEWMFastPath: cewm_step_fast() 快速路径
# =============================================================================


class TestCEWMFastPath:
    """验证 cewm_step_fast() 快速路径。"""

    def test_fast_path_returns_result(self, wm, pendulum_factory):
        """快速路径返回结果字典。"""
        obs = pendulum_factory(theta=0.3, omega=0.05)
        result = wm.cewm_step_fast(observation=obs, goal=pendulum_factory())
        assert isinstance(result, dict)

    def test_fast_path_flag(self, wm, pendulum_factory):
        """结果包含 fast_path=True 标志。"""
        obs = pendulum_factory(theta=0.2, omega=0.0)
        result = wm.cewm_step_fast(observation=obs, goal=pendulum_factory())
        assert result.get("fast_path") is True

    def test_fast_path_latency_recorded(self, wm, pendulum_factory):
        """快速路径记录延迟。"""
        obs = pendulum_factory(theta=0.2, omega=0.0)
        result = wm.cewm_step_fast(observation=obs, goal=pendulum_factory())
        assert "latency_ms" in result
        assert result["latency_ms"] >= 0

    def test_fast_path_skips_cognition(self, wm, pendulum_factory):
        """快速路径跳过认知层（无 causal_updates 键）。"""
        obs = pendulum_factory(theta=0.2, omega=0.0)
        result = wm.cewm_step_fast(observation=obs, goal=pendulum_factory())
        assert "causal_updates" not in result

    def test_fast_path_no_violation(self, wm, pendulum_factory):
        """快速路径安全检查通过。"""
        obs = pendulum_factory(theta=0.2, omega=0.0)
        result = wm.cewm_step_fast(observation=obs, goal=pendulum_factory())
        assert result["safety_violation"] is False


# =============================================================================
# TestCEWMCrossState: 跨状态类型验证
# =============================================================================


class TestCEWMCrossState:
    """验证不同状态类型的 cewm_step 正确工作。"""

    def test_cart_state_cewm_step(self, wm):
        """CartState cewm_step 正常运行。"""
        from mci_world_model.sdk._world_state import CartState

        obs = CartState(x=1.0, v=0.5)
        goal = CartState(x=0.0, v=0.0)
        result = wm.cewm_step(observation=obs, goal=goal)
        assert result["state"] is not None
        assert result["safety_violation"] is False

    def test_double_pendulum_cewm_step(self, wm):
        """DoublePendulumState cewm_step 正常运行。"""
        from mci_world_model.sdk._world_state import DoublePendulumState

        obs = DoublePendulumState(
            theta1=0.3, omega1=0.0, theta2=0.1, omega2=0.0
        )
        goal = DoublePendulumState(
            theta1=0.0, omega1=0.0, theta2=0.0, omega2=0.0
        )
        result = wm.cewm_step(observation=obs, goal=goal)
        assert result["state"] is not None

    def test_none_inputs_handled(self, wm):
        """None 输入不抛出异常。"""
        result = wm.cewm_step()
        assert result["state"] is None
        assert result["safety_violation"] is False


# =============================================================================
# TestCEWMCausalAccumulation: 因果图持久积累
# =============================================================================


class TestCEWMCausalAccumulation:
    """验证 FIX-C1: CausalUpdater 持久化积累。"""

    def test_updater_singleton_across_steps(self, wm, pendulum_factory):
        """CausalUpdater 单例：多步使用同一实例。"""
        goal = pendulum_factory()
        wm.cewm_step(observation=pendulum_factory(theta=0.1), goal=goal)
        first_ref = wm._causal_updater
        wm.cewm_step(observation=pendulum_factory(theta=0.2), goal=goal)
        assert wm._causal_updater is first_ref

    def test_causal_edges_accumulate(self, wm, pendulum_factory):
        """因果边逐步积累：后步 > 前步。"""
        goal = pendulum_factory()
        wm.cewm_step(observation=pendulum_factory(theta=0.1), goal=goal)
        edges_after_1 = len(wm._causal_updater._edges)
        for i in range(2, 5):
            wm.cewm_step(
                observation=pendulum_factory(theta=0.1 * i, omega=0.01),
                goal=goal,
            )
        edges_after_4 = len(wm._causal_updater._edges)
        assert edges_after_4 >= edges_after_1
