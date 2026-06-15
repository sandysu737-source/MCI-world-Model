"""CEWM v4.4.1 Phase 1 — GeneralizedPhysicsPredictor 测试
============================================================

验证通用物理预测器：
1. Euler 步进数值精度
2. 与 PendulumPhysicsPredictor 等价性
3. 与 CartPhysicsPredictor 等价性
4. 自定义动力学函数注册
5. DoublePendulumState 泛化验证
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._action_conditioned_predictor import (
    CartPhysicsPredictor,
    PendulumPhysicsPredictor,
)
from mci_world_model.sdk._generalized_physics import (
    GeneralizedPhysicsPredictor,
    double_pendulum_dynamics,
    euler_step,
)
from mci_world_model.sdk._world_state import (
    CartAction,
    CartState,
    PendulumAction,
    PendulumState,
)

# =============================================================================
# GPH-02: Euler 步进数值精度
# =============================================================================


class TestEulerStep:
    """Euler 积分单步数值精度验证。"""

    def test_linear_dynamics_exact(self):
        """线性动力学: Euler 积分为精确解。"""

        # dx/dt = 1, dv/dt = 0 → x = x0 + t, v = v0
        def linear_dynamics(state_vec, action_vec):
            return np.array([state_vec[1], 0.0])

        state = np.array([0.0, 1.0])
        action = np.array([0.0])
        dt = 0.01

        result = euler_step(state, action, linear_dynamics, dt)
        assert result[0] == pytest.approx(0.01)  # x += v*dt
        assert result[1] == pytest.approx(1.0)  # v 不变

    def test_zero_action_zero_dynamics(self):
        """零动作+零动力学=状态不变。"""

        def zero_dynamics(state_vec, action_vec):
            return np.zeros_like(state_vec)

        state = np.array([5.0, 3.0])
        action = np.array([0.0])
        result = euler_step(state, action, zero_dynamics, dt=0.01)
        np.testing.assert_array_almost_equal(result, state)


# =============================================================================
# GPH-04: 与 CartPhysicsPredictor 等价性
# =============================================================================


class TestCartEquivalence:
    """GeneralizedPhysicsPredictor(cart) 与 CartPhysicsPredictor 等价。"""

    def test_single_step_equivalence(self):
        gen_pred = GeneralizedPhysicsPredictor(default_backend="cart")
        cart_pred = CartPhysicsPredictor()

        state = CartState(x=0.0, v=1.0)
        action = CartAction(force=5.0)

        gen_traj = gen_pred.predict(state, action, n_steps=1)
        cart_traj = cart_pred.predict(state, action, n_steps=1)

        assert gen_traj[0].x == pytest.approx(cart_traj[0].x)
        assert gen_traj[0].v == pytest.approx(cart_traj[0].v)

    def test_multi_step_equivalence(self):
        gen_pred = GeneralizedPhysicsPredictor(default_backend="cart")
        cart_pred = CartPhysicsPredictor()

        state = CartState(x=0.0, v=2.0)
        action = CartAction(force=3.0)

        gen_traj = gen_pred.predict(state, action, n_steps=50)
        cart_traj = cart_pred.predict(state, action, n_steps=50)

        for i in range(50):
            assert gen_traj[i].x == pytest.approx(cart_traj[i].x, abs=1e-8)
            assert gen_traj[i].v == pytest.approx(cart_traj[i].v, abs=1e-8)

    def test_no_action_equivalence(self):
        gen_pred = GeneralizedPhysicsPredictor(default_backend="cart")
        cart_pred = CartPhysicsPredictor()

        state = CartState(x=1.0, v=3.0)

        gen_traj = gen_pred.predict(state, None, n_steps=10)
        cart_traj = cart_pred.predict(state, None, n_steps=10)

        for i in range(10):
            assert gen_traj[i].x == pytest.approx(cart_traj[i].x, abs=1e-8)
            assert gen_traj[i].v == pytest.approx(cart_traj[i].v, abs=1e-8)


# =============================================================================
# GPH-05: 与 PendulumPhysicsPredictor 等价性
# =============================================================================


class TestPendulumEquivalence:
    """GeneralizedPhysicsPredictor(pendulum) 与 PendulumPhysicsPredictor 等价。"""

    def test_single_step_equivalence(self):
        gen_pred = GeneralizedPhysicsPredictor(default_backend="pendulum")
        phys_pred = PendulumPhysicsPredictor()

        state = PendulumState(theta=0.5, omega=0.0)
        action = PendulumAction(torque=3.0)

        gen_traj = gen_pred.predict(state, action, n_steps=1)
        phys_traj = phys_pred.predict(state, action, n_steps=1)

        assert gen_traj[0].theta == pytest.approx(phys_traj[0].theta)
        assert gen_traj[0].omega == pytest.approx(phys_traj[0].omega)

    def test_multi_step_equivalence_small_angle(self):
        gen_pred = GeneralizedPhysicsPredictor(default_backend="pendulum")
        phys_pred = PendulumPhysicsPredictor()

        state = PendulumState(theta=0.1, omega=0.0)
        action = PendulumAction(torque=0.0)

        gen_traj = gen_pred.predict(state, action, n_steps=10)
        phys_traj = phys_pred.predict(state, action, n_steps=10)

        for i in range(10):
            assert gen_traj[i].theta == pytest.approx(phys_traj[i].theta, abs=1e-10)
            assert gen_traj[i].omega == pytest.approx(phys_traj[i].omega, abs=1e-10)


# =============================================================================
# GPH-03: 自定义动力学函数注册
# =============================================================================


class TestCustomDynamicsRegistration:
    """自定义动力学函数注册验证。"""

    def test_register_and_use(self):
        pred = GeneralizedPhysicsPredictor()

        def harmonic_dynamics(state_vec, action_vec):
            """简谐振子: d2x/dt2 = -k*x, k=1"""
            x, v = float(state_vec[0]), float(state_vec[1])
            return np.array([v, -x])

        pred.register_dynamics("harmonic", harmonic_dynamics, state_dim=2, action_dim=0)
        assert "harmonic" in pred.available_backends

        pred.set_backend("harmonic")
        # 使用 CartState 作为载体
        state = CartState(x=1.0, v=0.0)
        traj = pred.predict(state, None, n_steps=1)
        assert len(traj) == 1

    def test_unknown_backend_raises(self):
        pred = GeneralizedPhysicsPredictor()
        with pytest.raises(ValueError, match="未知后端"):
            pred.set_backend("nonexistent")


# =============================================================================
# GPH-07: DoublePendulumState 泛化验证
# =============================================================================


class TestDoublePendulum:
    """双摆系统泛化验证——4D 状态 + 2D 动作。"""

    def test_double_pendulum_dynamics_shape(self):
        state_vec = np.array([0.5, 0.0, 0.3, 0.0])
        action_vec = np.array([1.0, -0.5])
        d_state = double_pendulum_dynamics(state_vec, action_vec)
        assert d_state.shape == (4,)

    def test_generalized_predictor_double_pendulum(self):
        """GeneralizedPhysicsPredictor 支持 double_pendulum 后端。"""
        pred = GeneralizedPhysicsPredictor(default_backend="double_pendulum")
        assert "double_pendulum" in pred.available_backends

    def test_four_dimensional_prediction(self):
        """4D 状态预测验证。"""
        pred = GeneralizedPhysicsPredictor(default_backend="double_pendulum")

        # 使用 MultimodalWorldState 作为 4D 载体
        from mci_world_model.sdk._world_state import MultimodalWorldState

        state = MultimodalWorldState(proprioception=np.array([0.5, 0.0, 0.3, 0.0]))
        # 无动作预测
        traj = pred.predict(state, None, n_steps=5)
        assert len(traj) == 5
        # 每个状态应该是 4D+ 的向量
        for s in traj:
            assert len(s.to_vector()) >= 4
