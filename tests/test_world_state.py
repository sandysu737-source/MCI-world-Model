"""
MCI World Model v3.2.0 — WorldState 与 PendulumState 单元测试
===============================================================

验证目标:
1. WorldState 抽象基类契约完备性
2. PendulumState 物理定律正确性
3. 角度距离工具函数
4. from_signals() 感知→认知桥接
5. CausalWorldModelState.world_state 桥接字段
"""

import numpy as np
import pytest

from mci_world_model.sdk._world_model import CausalWorldModelState
from mci_world_model.sdk._world_state import (
    PendulumAction,
    PendulumState,
    WorldState,
    _angular_distance,
)

# =============================================================================
# 测试数据
# =============================================================================


@pytest.fixture
def pendulum_at_rest():
    """垂直静止的单摆。"""
    return PendulumState(theta=0.0, omega=0.0)


@pytest.fixture
def pendulum_displaced():
    """偏离平衡点的单摆。"""
    return PendulumState(theta=0.5, omega=0.0, dt=0.001)


# =============================================================================
# WorldState 抽象基类契约
# =============================================================================


class TestWorldStateInterface:
    """验证 WorldState 抽象基类定义正确。"""

    def test_pendulum_is_worldstate(self):
        """PendulumState 是 WorldState 实例。"""
        p = PendulumState(theta=0.1, omega=0.0)
        assert isinstance(p, WorldState)

    def test_worldstate_has_four_abstract_methods(self):
        """WorldState 定义了四个核心抽象方法。"""
        assert hasattr(WorldState, "to_vector")
        assert hasattr(WorldState, "from_vector")
        assert hasattr(WorldState, "distance")
        assert hasattr(WorldState, "copy")

    def test_cannot_instantiate_worldstate_directly(self):
        """不能直接实例化 WorldState（抽象类）。"""
        with pytest.raises(TypeError):
            WorldState()  # type: ignore


# =============================================================================
# PendulumState 核心操作
# =============================================================================


class TestPendulumVectorRoundtrip:
    """to_vector / from_vector 往返测试。"""

    def test_identity(self):
        """零状态往返不变。"""
        p = PendulumState(theta=0.0, omega=0.0)
        vec = p.to_vector()
        assert vec.shape == (2,)
        assert vec.dtype == np.float64

        p2 = PendulumState.from_vector(vec)
        assert p.distance(p2) < 1e-15

    def test_positive_theta(self):
        """正角度往返。"""
        p = PendulumState(theta=0.5, omega=-0.3)
        p2 = PendulumState.from_vector(p.to_vector())
        assert p.distance(p2) < 1e-15

    def test_large_values(self):
        """大值往返（含周期性）。"""
        p = PendulumState(theta=10.0, omega=50.0)
        p2 = PendulumState.from_vector(p.to_vector())
        assert p.distance(p2) < 1e-15


class TestPendulumDistance:
    """distance() 度量测试。"""

    def test_same_state_zero_distance(self):
        """相同状态距离为 0。"""
        p = PendulumState(theta=0.5, omega=-0.3)
        assert p.distance(p) == 0.0

    def test_theta_omega_independent(self):
        """theta 和 omega 独立贡献距离。"""
        p1 = PendulumState(theta=0.0, omega=0.0)
        p2 = PendulumState(theta=0.3, omega=0.4)
        d = p1.distance(p2)
        expected = np.sqrt(0.3**2 + 0.4**2)
        assert abs(d - expected) < 1e-10

    def test_non_pendulum_returns_inf(self):
        """非 PendulumState 返回无穷大。"""

        class FakeState(WorldState):
            def to_vector(self):
                return np.array([0.0])

            @classmethod
            def from_vector(cls, vec):
                return cls()

            def distance(self, other):
                return 0.0

            def copy(self):
                return self

        p = PendulumState(theta=0.0, omega=0.0)
        fake = FakeState()
        assert p.distance(fake) == float("inf")

    def test_periodic_theta_distance(self):
        """角度周期性: -π 和 π 是同一位置。"""
        p1 = PendulumState(theta=3.14, omega=0.0)
        p2 = PendulumState(theta=-3.14, omega=0.0)
        d = p1.distance(p2)
        # 距离应接近于 0（同一位置）
        assert d < 0.1


class TestPendulumCopy:
    """copy() 深拷贝测试。"""

    def test_copy_is_independent(self):
        """修改拷⻉不影响原状态。"""
        p = PendulumState(theta=0.5, omega=0.0)
        p2 = p.copy()
        assert p.distance(p2) == 0.0

        # 修改拷贝
        p2 = PendulumState(theta=1.0, omega=2.0, g=p2.g, L=p2.L, dt=p2.dt)
        assert p.theta == 0.5  # 原状态不变
        assert p.omega == 0.0


# =============================================================================
# PendulumState 物理定律验证（核心验收标准）
# =============================================================================


class TestPendulumPhysics:
    """单摆物理定律——这是 MCI 世界模型架构验证的金标准。

    这些测试确保世界模型"知道"它正在建模的物理世界的 ground truth。
    """

    def test_equilibrium_stays(self, pendulum_at_rest):
        """垂直静止的摆保持静止。"""
        p = pendulum_at_rest
        for _ in range(10):
            p = p.step_physics()
        assert abs(p.theta) < 1e-15
        assert abs(p.omega) < 1e-15

    def test_gravity_pulls_toward_equilibrium(self, pendulum_displaced):
        """正角度 → 负角速度（重力拉回平衡点）。"""
        p = pendulum_displaced
        p_after = p.step_physics()
        # dt=0.001 很短，theta 几乎不变，但 omega 应为负
        assert p_after.omega < 0, f"重力应该产生负角速度，实际 omega={p_after.omega}"

    def test_negative_theta_gravity_pulls_positive(self):
        """负角度 → 正角速度。"""
        p = PendulumState(theta=-0.5, omega=0.0, dt=0.001)
        p_after = p.step_physics()
        assert p_after.omega > 0, f"重力应该产生正角速度，实际 omega={p_after.omega}"

    def test_energy_not_increasing(self, pendulum_displaced):
        """自由振荡中机械能不增加（热力学第一定律）。"""
        p = pendulum_displaced
        E_initial = _pendulum_energy(p)
        for _ in range(100):
            p = p.step_physics()
        E_final = _pendulum_energy(p)
        # 允许 1% 数值误差
        assert E_final <= E_initial * 1.01, f"能量从 {E_initial} 增加到 {E_final}"

    def test_small_angle_period_approximation(self):
        """小角度振荡周期 ≈ 2π√(L/g)。

        策略: 模拟振荡，记录相邻正→负穿越点的时间差，
        取平均作为半周期×2的验证。
        """
        p = PendulumState(theta=0.1, omega=0.0, dt=0.001, g=9.81, L=1.0)
        expected_period = 2 * np.pi * np.sqrt(1.0 / 9.81)  # ≈ 2.006 s

        crossing_times: list[float] = []
        prev_theta = p.theta
        t = 0.0
        MAX_CROSSINGS = 6  # 记录 6 次穿越 = 5 个间隔
        for _ in range(100000):
            p = p.step_physics()
            t += p.dt
            if prev_theta > 0 and p.theta <= 0:
                crossing_times.append(t)
                if len(crossing_times) >= MAX_CROSSINGS:
                    break
            prev_theta = p.theta

        # 相邻正→负穿越的间隔 = 1 个完整周期
        intervals = [crossing_times[i] - crossing_times[i - 1] for i in range(1, len(crossing_times))]
        measured_period = float(np.mean(intervals))
        rel_error = abs(measured_period - expected_period) / expected_period
        assert rel_error < 0.01, f"周期 {measured_period:.3f}s 偏离预期 {expected_period:.3f}s (误差 {rel_error:.2%})"


def _pendulum_energy(p: PendulumState) -> float:
    """计算单摆的总机械能 E = K + U。

    K = ½ m L² ω² (动能)
    U = m g L (1 − cos θ) ≈ m g L θ²/2 (势能，取最低点为零点)
    令 m=1 简化。
    """
    kinetic = 0.5 * p.L**2 * p.omega**2
    potential = p.g * p.L * (1.0 - np.cos(p.theta))
    return kinetic + potential


# =============================================================================
# _angular_distance 工具函数
# =============================================================================


class TestAngularDistance:
    def test_same_angle(self):
        assert _angular_distance(0.5, 0.5) == 0.0

    def test_small_difference(self):
        assert abs(_angular_distance(0.0, 0.1) - 0.1) < 1e-10

    def test_cross_zero(self):
        assert abs(_angular_distance(0.1, -0.1) - 0.2) < 1e-10

    def test_pi_boundary(self):
        """π 和 -π 是同一位置，距离为 0。"""
        d = _angular_distance(np.pi, -np.pi)
        assert d < 1e-10

    def test_near_2pi(self):
        """3.1 和 -3.18 很近（都接近 π 的对面）。"""
        d = _angular_distance(3.1, -3.18)
        assert d < 0.1


# =============================================================================
# PendulumState.from_signals() 感知→认知桥接
# =============================================================================


class TestPendulumFromSignals:
    """PhysicalSignal → PendulumState 的桥接测试。"""

    def test_empty_signals_returns_default(self):
        """无信号时返回零状态。"""
        state = PendulumState.from_signals([])
        assert state.theta == 0.0
        assert state.omega == 0.0

    def test_encoder_signal_only(self):
        """仅编码器位置信号。"""

        class FakeSignal:
            modality = "proprioception"
            sub_type = "encoder_position"
            value = 0.5

        state = PendulumState.from_signals([FakeSignal()])
        assert abs(state.theta - 0.5) < 1e-10
        assert state.omega == 0.0

    def test_imu_signal_only(self):
        """仅陀螺仪信号。"""

        class FakeSignal:
            modality = "proprioception"
            sub_type = "imu_9axis"
            value = (0.0, 0.0, -0.3)  # gyro_z = -0.3

        state = PendulumState.from_signals([FakeSignal()])
        assert state.theta == 0.0
        assert abs(state.omega + 0.3) < 1e-10

    def test_both_signals(self):
        """编码器 + IMU 同时提供。"""

        class FakeEncoder:
            modality = "proprioception"
            sub_type = "encoder_position"
            value = 0.5

        class FakeIMU:
            modality = "proprioception"
            sub_type = "imu_9axis"
            value = (0.0, 0.0, -0.3)

        state = PendulumState.from_signals([FakeEncoder(), FakeIMU()])
        assert abs(state.theta - 0.5) < 1e-10
        assert abs(state.omega + 0.3) < 1e-10

    def test_non_proprioception_ignored(self):
        """非本体感觉信号被忽略。"""

        class FakeVision:
            modality = "vision"
            sub_type = "rgb_frame"
            value = "frame_data"

        state = PendulumState.from_signals([FakeVision()])
        assert state.theta == 0.0
        assert state.omega == 0.0


# =============================================================================
# CausalWorldModelState.world_state 桥接
# =============================================================================


class TestCausalWorldModelStateBridge:
    """CausalWorldModelState.world_state 桥接字段测试。"""

    def test_empty_state_has_no_world_state(self):
        """empty() 创建的 state 的 world_state 为 None。"""
        cs = CausalWorldModelState.empty()
        assert cs.world_state is None

    def test_has_world_state_in_to_dict(self):
        """to_dict() 包含 has_world_state 字段。"""
        cs = CausalWorldModelState.empty()
        d = cs.to_dict()
        assert "has_world_state" in d
        assert d["has_world_state"] is False

    def test_set_world_state_reflected_in_dict(self):
        """设置 world_state 后 to_dict 反映变化。"""
        p = PendulumState(theta=0.5, omega=-0.3)
        cs = CausalWorldModelState(world_state=p)
        d = cs.to_dict()
        assert d["has_world_state"] is True
        assert "world_state" in d
        assert d["world_state"]["type"] == "PendulumState"

    def test_backward_compatible_no_world_state(self):
        """不设置 world_state 时，所有现有行为不变。"""
        cs = CausalWorldModelState(
            causal_edges=[{"cause": "A", "effect": "B", "rho": 0.8}],
            n_confirmed=1,
        )
        # world_state 默认为 None
        assert cs.world_state is None
        # 因果边字段不受影响
        assert len(cs.causal_edges) == 1
        assert cs.n_confirmed == 1


# =============================================================================
# PendulumState.to_dict() 序列化
# =============================================================================


class TestPendulumToDict:
    def test_contains_all_fields(self):
        p = PendulumState(theta=0.5, omega=-0.3)
        d = p.to_dict()
        assert d["type"] == "PendulumState"
        assert "theta" in d
        assert "omega" in d
        assert "g" in d
        assert "L" in d
        assert "dt" in d


# =============================================================================
# Action 与 PendulumAction 测试（v3.2.0 Task 2）
# =============================================================================


class TestActionInterface:
    """Action 抽象基类契约测试。"""

    def test_pendulum_action_is_action(self):
        """PendulumAction 是 Action 实例。"""
        from mci_world_model.sdk._world_state import Action, PendulumAction

        a = PendulumAction(torque=1.0)
        assert isinstance(a, Action)

    def test_cannot_instantiate_action_directly(self):
        """不能直接实例化 Action（抽象类）。"""
        from mci_world_model.sdk._world_state import Action

        with pytest.raises(TypeError):
            Action()  # type: ignore


class TestPendulumAction:
    """PendulumAction 物理正确性测试。"""

    def test_zero_torque_equals_free_physics(self):
        """零力矩 = 自由演化（无外力自然摆动）。"""
        p = PendulumState(theta=0.5, omega=0.0)
        action = PendulumAction(torque=0.0)
        p_actioned = action.apply(p)
        p_free = p.step_physics()
        assert p_actioned.distance(p_free) < 1e-10

    def test_positive_torque_increases_omega(self):
        """正力矩应增加角速度（顺时针推）。"""
        p = PendulumState(theta=0.0, omega=0.0)
        action = PendulumAction(torque=10.0)
        p_actioned = action.apply(p)
        # sin(0)=0，所以 ω' = 0 + torque/L² * dt
        assert p_actioned.omega > 0, f"推后角速度应为正，实际 {p_actioned.omega}"

    def test_negative_torque_decreases_omega(self):
        """负力矩应减少角速度（逆时针推）。"""
        p = PendulumState(theta=0.0, omega=0.0)
        action = PendulumAction(torque=-5.0)
        p_actioned = action.apply(p)
        assert p_actioned.omega < 0, f"推后角速度应为负，实际 {p_actioned.omega}"

    def test_torque_prediction_close_to_physics(self):
        """推力的数值效果符合物理公式。

        对于 PendulumState(theta=0, omega=0), sin(0)=0, α = torque / L²:
        ω' = torque / L² * dt
        θ' = 0 + 0 * dt = 0
        """
        p = PendulumState(theta=0.0, omega=0.0, dt=0.01, L=1.0)
        torque = 3.0
        expected_omega = torque / (1.0**2) * 0.01  # = 0.03
        action = PendulumAction(torque=torque, dt=0.01)
        p_actioned = action.apply(p)
        assert abs(p_actioned.omega - expected_omega) < 1e-10
        assert abs(p_actioned.theta) < 1e-10  # theta 不变（ω 从零开始）

    def test_torque_fights_gravity(self):
        """推力可以对抗重力。

        摆偏离平衡点时重力拉回，但足够大的正向力矩可以抵消甚至反转。
        """
        p = PendulumState(theta=0.5, omega=0.0, dt=0.001)
        # 不加外力时 omega 应变负
        no_push = p.step_physics()
        assert no_push.omega < 0

        # 加足够大的正力矩
        push = PendulumAction(torque=15.0, dt=0.001)
        pushed = push.apply(p)
        # 力矩应足够强以反转 omega
        assert pushed.omega > 0, f"推力应克服重力，实际 omega={pushed.omega}"

    def test_apply_does_not_modify_original(self):
        """apply 不修改原状态（函数式语义）。"""
        p = PendulumState(theta=0.3, omega=0.1)
        original_theta = p.theta
        original_omega = p.omega

        action = PendulumAction(torque=5.0)
        _ = action.apply(p)

        assert p.theta == original_theta
        assert p.omega == original_omega

    def test_wrong_state_type_raises(self):
        """非 PendulumState 的参数抛出 TypeError。"""
        from mci_world_model.sdk._world_state import WorldState

        class FakeState(WorldState):
            def to_vector(self):
                return np.array([0.0])

            @classmethod
            def from_vector(cls, vec):
                return cls()

            def distance(self, other):
                return 0.0

            def copy(self):
                return self

        action = PendulumAction(torque=1.0)
        with pytest.raises(TypeError, match="PendulumAction"):
            action.apply(FakeState())

    def test_to_dict(self):
        """to_dict 序列化。"""
        action = PendulumAction(torque=3.5, dt=0.02)
        d = action.to_dict()
        assert d["type"] == "PendulumAction"
        assert d["torque"] == 3.5
        assert d["dt"] == 0.02
