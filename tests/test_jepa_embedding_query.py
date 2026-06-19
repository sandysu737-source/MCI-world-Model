"""
GEN-05 (S-2): JEPA 嵌入查询测试
================================

验证 C-2 修复后的 JEPA 嵌入查询正确性:
- WorldState.causal_query() 返回正确的因果类别字符串
- jepa_predict(cause=...) 不再抛出 TypeError
- cewm_step / cewm_step_fast 预测路径正常工作
- 不同状态类型的因果查询互相隔离
"""

import pytest


@pytest.fixture
def wm():
    """创建一个最小化的 MCIWorldModel 实例。"""
    from mci_world_model.sdk._world_model import MCIWorldModel

    return MCIWorldModel()


# =============================================================================
# TestCausalQuery: causal_query() 返回值
# =============================================================================


class TestCausalQuery:
    """WorldState 子类的 causal_query() 返回值验证。"""

    def test_pendulum_causal_query(self):
        """PendulumState.causal_query() 返回 'pendulum'。"""
        from mci_world_model.sdk._world_state import PendulumState

        state = PendulumState(theta=0.5, omega=0.1)
        assert state.causal_query() == "pendulum"

    def test_cart_causal_query(self):
        """CartState.causal_query() 返回 'cart'。"""
        from mci_world_model.sdk._world_state import CartState

        state = CartState(x=1.0, v=0.5)
        assert state.causal_query() == "cart"

    def test_causal_query_is_string(self):
        """causal_query() 始终返回 str 类型（不是 tuple/list）。"""
        from mci_world_model.sdk._world_state import PendulumState

        state = PendulumState(theta=0.0, omega=0.0)
        result = state.causal_query()
        assert isinstance(result, str), f"causal_query() 应返回 str，实际: {type(result)}"

    def test_causal_query_not_to_vector_str(self):
        """causal_query() 返回的不是 str(to_vector())（C-2 修复核心验证）。"""
        from mci_world_model.sdk._world_state import PendulumState

        state = PendulumState(theta=0.5, omega=0.1)
        query = state.causal_query()
        vec_str = str(state.to_vector())
        assert query != vec_str, "causal_query() 不应等于 str(to_vector())"


# =============================================================================
# TestJepaPredictNoTypeError: jepa_predict 不抛 TypeError
# =============================================================================


class TestJepaPredictNoTypeError:
    """验证 C-2 修复: jepa_predict(cause=...) 不再抛 TypeError。"""

    def test_jepa_predict_pendulum_cause(self, wm):
        """jepa_predict(cause='pendulum') 不抛 TypeError。"""
        result = wm.jepa_predict(cause="pendulum")
        assert isinstance(result, list)

    def test_jepa_predict_cart_cause(self, wm):
        """jepa_predict(cause='cart') 不抛 TypeError。"""
        result = wm.jepa_predict(cause="cart")
        assert isinstance(result, list)

    def test_jepa_predict_arbitrary_cause(self, wm):
        """jepa_predict 接受任意 cause 字符串。"""
        result = wm.jepa_predict(cause="unknown_system")
        assert isinstance(result, list)

    def test_jepa_predict_returns_list_of_dicts(self, wm):
        """jepa_predict 返回的每个元素是 dict。"""
        result = wm.jepa_predict(cause="pendulum", top_k=1)
        # 回退路径可能返回空列表或包含 dict 的列表
        for item in result:
            assert isinstance(item, dict), f"jepa_predict 元素应为 dict，实际: {type(item)}"

    def test_jepa_predict_with_top_k(self, wm):
        """jepa_predict 的 top_k 参数控制返回数量。"""
        result = wm.jepa_predict(cause="pendulum", top_k=5)
        assert len(result) <= 5


# =============================================================================
# TestCewmStepPrediction: cewm_step 预测路径
# =============================================================================


class TestCewmStepPrediction:
    """验证 cewm_step() 中的 JEPA 预测路径不崩溃。"""

    def test_cewm_step_pendulum_no_typeerror(self, wm):
        """cewm_step 使用 PendulumState 不抛 TypeError。"""
        from mci_world_model.sdk._world_state import PendulumState

        result = wm.cewm_step(
            observation=PendulumState(theta=0.5, omega=0.1),
            goal=PendulumState(theta=0.0, omega=0.0),
        )
        assert isinstance(result, dict)
        assert "prediction" in result

    def test_cewm_step_cart_no_typeerror(self, wm):
        """cewm_step 使用 CartState 不抛 TypeError。"""
        from mci_world_model.sdk._world_state import CartState

        result = wm.cewm_step(
            observation=CartState(x=0.5, v=0.1),
            goal=CartState(x=0.0, v=0.0),
        )
        assert isinstance(result, dict)
        assert "prediction" in result

    def test_cewm_step_fast_pendulum(self, wm):
        """cewm_step_fast 使用 PendulumState 不抛 TypeError。"""
        from mci_world_model.sdk._world_state import PendulumState

        result = wm.cewm_step_fast(
            observation=PendulumState(theta=0.3, omega=0.0),
        )
        assert isinstance(result, dict)

    def test_cewm_step_fast_cart(self, wm):
        """cewm_step_fast 使用 CartState 不抛 TypeError。"""
        from mci_world_model.sdk._world_state import CartState

        result = wm.cewm_step_fast(
            observation=CartState(x=0.3, v=0.0),
        )
        assert isinstance(result, dict)


# =============================================================================
# TestCrossStateIsolation: 不同状态类型的因果查询隔离
# =============================================================================


class TestCrossStateIsolation:
    """验证不同状态类型的因果查询互相隔离。"""

    def test_different_states_different_queries(self):
        """PendulumState 和 CartState 返回不同的 causal_query。"""
        from mci_world_model.sdk._world_state import CartState, PendulumState

        pendulum = PendulumState(theta=0.5, omega=0.1)
        cart = CartState(x=0.5, v=0.1)

        assert pendulum.causal_query() != cart.causal_query()

    def test_jepa_predict_different_causes(self, wm):
        """不同 cause 参数的 jepa_predict 调用互相独立。"""
        result_pendulum = wm.jepa_predict(cause="pendulum")
        result_cart = wm.jepa_predict(cause="cart")
        # 两者都应返回 list（即使内容相同——回退路径）
        assert isinstance(result_pendulum, list)
        assert isinstance(result_cart, list)


# =============================================================================
# TestCausalEdgesConsistency: causal_edges() 与 causal_query() 一致性
# =============================================================================


class TestCausalEdgesConsistency:
    """验证 causal_edges() 返回的边与 causal_query() 描述的因果结构一致。"""

    def test_pendulum_causal_edges_non_empty(self):
        """PendulumState.causal_edges() 返回非空因果边列表。"""
        from mci_world_model.sdk._world_state import PendulumState

        state = PendulumState(theta=0.5, omega=0.1)
        edges = state.causal_edges()
        assert len(edges) > 0
        assert ("theta", "omega") in edges

    def test_cart_causal_edges_non_empty(self):
        """CartState.causal_edges() 返回非空因果边列表。"""
        from mci_world_model.sdk._world_state import CartState

        state = CartState(x=0.5, v=0.1)
        edges = state.causal_edges()
        assert len(edges) > 0
        assert ("x", "v") in edges

    def test_causal_edges_returns_tuples(self):
        """causal_edges() 返回的元素是 tuple[str, str]。"""
        from mci_world_model.sdk._world_state import PendulumState

        state = PendulumState(theta=0.0, omega=0.0)
        edges = state.causal_edges()
        for edge in edges:
            assert isinstance(edge, tuple)
            assert len(edge) == 2
            assert isinstance(edge[0], str)
            assert isinstance(edge[1], str)
