"""P2-3: RobotWorldState 边界条件测试 (E-4)
==========================================

覆盖 n_joints=0, NaN 输入, 长度不为 3 的倍数, 数组长度不匹配,
超大关节, 负 n_joints, inf 输入, 空数组等边界场景。
"""

import numpy as np
import pytest

from mci_world_model.sdk._robot_state import RobotAction, RobotWorldState
from mci_world_model.sdk._world_state import WorldState

# =============================================================================
# RobotWorldState 边界条件
# =============================================================================


class TestRobotWorldStateZeroJoints:
    """n_joints=0 边界。"""

    def test_zero_joints_to_vector_empty(self):
        """n_joints=0 → to_vector 返回空数组。"""
        state = RobotWorldState(n_joints=0)
        state._ensure_arrays()
        vec = state.to_vector()
        assert vec.shape == (0,)

    def test_zero_joints_from_vector_empty(self):
        """空向量 → from_vector 创建 n_joints=0。"""
        vec = np.array([], dtype=np.float64)
        state = RobotWorldState.from_vector(vec)
        assert state.n_joints == 0

    def test_zero_joints_copy(self):
        """n_joints=0 拷贝不变。"""
        state = RobotWorldState(n_joints=0)
        cp = state.copy()
        assert cp.n_joints == 0

    def test_zero_joints_distance_self(self):
        """n_joints=0 自身距离为 0。"""
        s1 = RobotWorldState(n_joints=0)
        s2 = RobotWorldState(n_joints=0)
        assert s1.distance(s2) == pytest.approx(0.0)

    def test_zero_joints_step_physics(self):
        """n_joints=0 step_physics 不报错。"""
        state = RobotWorldState(n_joints=0)
        next_state = state.step_physics()
        assert next_state.n_joints == 0

    def test_zero_joints_to_dict(self):
        """n_joints=0 to_dict 正常。"""
        state = RobotWorldState(n_joints=0)
        d = state.to_dict()
        assert d["type"] == "RobotWorldState"
        assert d["n_joints"] == 0


class TestRobotWorldStateFromVectorEdgeCases:
    """from_vector 边界。"""

    def test_length_not_divisible_by_3(self):
        """长度非 3 的倍数 → 整数除法截断。"""
        vec = np.arange(5, dtype=np.float64)  # 5 // 3 = 1
        state = RobotWorldState.from_vector(vec)
        assert state.n_joints == 1
        assert len(state.joint_positions) == 1
        # 尾部 2 个元素被丢弃 (vec[3:6] 越界)

    def test_single_joint(self):
        """3 元素 → 单关节。"""
        vec = np.array([1.0, 2.0, 3.0])
        state = RobotWorldState.from_vector(vec)
        assert state.n_joints == 1
        assert float(state.joint_positions[0]) == pytest.approx(1.0)
        assert float(state.joint_velocities[0]) == pytest.approx(2.0)
        assert float(state.joint_efforts[0]) == pytest.approx(3.0)

    def test_two_joints(self):
        """6 元素 → 双关节。"""
        vec = np.arange(6, dtype=np.float64)
        state = RobotWorldState.from_vector(vec)
        assert state.n_joints == 2

    def test_very_long_vector(self):
        """100 关节 → 300 元素。"""
        n = 100
        vec = np.random.randn(n * 3)
        state = RobotWorldState.from_vector(vec)
        assert state.n_joints == n
        assert state.to_vector().shape == (n * 3,)

    def test_roundtrip_preserves_values(self):
        """to_vector → from_vector 往返精度。"""
        original = RobotWorldState(
            joint_positions=np.array([0.1, -0.2, 0.3, 1e-8]),
            joint_velocities=np.array([0.5, -0.5, 1.0, -1.0]),
            joint_efforts=np.array([0.0, 0.0, 0.0, 0.0]),
            n_joints=4,
        )
        reconstructed = RobotWorldState.from_vector(original.to_vector())
        assert original.distance(reconstructed) < 1e-12


class TestRobotWorldStateNaNInf:
    """NaN / Inf 输入。"""

    def test_nan_positions(self):
        """NaN 位置 → to_vector 含 NaN。"""
        state = RobotWorldState(
            joint_positions=np.array([np.nan, 0.0]),
            joint_velocities=np.zeros(2),
            joint_efforts=np.zeros(2),
            n_joints=2,
        )
        vec = state.to_vector()
        assert np.isnan(vec[0])

    def test_nan_velocity_step_physics(self):
        """NaN 速度 → step_physics 传播 NaN。"""
        state = RobotWorldState(
            joint_positions=np.zeros(2),
            joint_velocities=np.array([np.nan, 1.0]),
            joint_efforts=np.zeros(2),
            n_joints=2,
            dt=0.01,
        )
        next_state = state.step_physics()
        assert np.isnan(next_state.joint_positions[0])
        assert not np.isnan(next_state.joint_positions[1])

    def test_inf_distance(self):
        """含 inf 的状态距离为 inf。"""
        s1 = RobotWorldState(n_joints=2)
        s1._ensure_arrays()
        s2 = RobotWorldState(
            joint_positions=np.array([np.inf, 0.0]),
            joint_velocities=np.zeros(2),
            joint_efforts=np.zeros(2),
            n_joints=2,
        )
        d = s1.distance(s2)
        assert np.isinf(d)

    def test_nan_distance_is_nan(self):
        """含 NaN 的状态距离为 NaN。"""
        s1 = RobotWorldState(n_joints=2)
        s1._ensure_arrays()
        s2 = RobotWorldState(
            joint_positions=np.array([np.nan, 0.0]),
            joint_velocities=np.zeros(2),
            joint_efforts=np.zeros(2),
            n_joints=2,
        )
        d = s1.distance(s2)
        assert np.isnan(d)


class TestRobotWorldStateMismatchedArrays:
    """数组长度不匹配。"""

    def test_mismatched_positions_velocities(self):
        """positions=6, velocities=3 → to_vector 拼接为 9（不崩溃）。"""
        state = RobotWorldState(
            joint_positions=np.zeros(6),
            joint_velocities=np.zeros(3),
            joint_efforts=np.zeros(6),
            n_joints=6,
        )
        # _ensure_arrays 不会覆盖已设置的非 None 数组
        vec = state.to_vector()
        # 拼接: 6 + 3 + 6 = 15, 不是 18
        assert len(vec) == 15

    def test_mismatched_n_joints_vs_arrays(self):
        """n_joints=4 但 positions=6 → _ensure_arrays 不覆盖。"""
        state = RobotWorldState(
            joint_positions=np.ones(6),
            joint_velocities=np.ones(6),
            joint_efforts=np.ones(6),
            n_joints=4,
        )
        # _ensure_arrays 只在 None 时填充，不影响已有数组
        state._ensure_arrays()
        assert len(state.joint_positions) == 6


class TestRobotWorldStateNegativeNJoints:
    """负 n_joints 行为。"""

    def test_negative_n_joints_ensure_arrays(self):
        """n_joints=-1 → _ensure_arrays 创建空/负大小数组（应视为 undefined behavior）。"""
        state = RobotWorldState(n_joints=-1)
        # 这是个边界条件，不应崩溃（但行为未定义）
        with pytest.raises(ValueError):
            state._ensure_arrays()


class TestRobotWorldStateExtremeValues:
    """极值输入。"""

    def test_very_large_values(self):
        """极大关节角度。"""
        state = RobotWorldState(
            joint_positions=np.full(3, 1e10),
            joint_velocities=np.zeros(3),
            joint_efforts=np.zeros(3),
            n_joints=3,
        )
        vec = state.to_vector()
        assert float(vec[0]) == pytest.approx(1e10)

    def test_very_small_values(self):
        """极小关节角度（接近机器精度）。"""
        state = RobotWorldState(
            joint_positions=np.full(3, 1e-15),
            joint_velocities=np.zeros(3),
            joint_efforts=np.zeros(3),
            n_joints=3,
        )
        vec = state.to_vector()
        assert float(vec[0]) == pytest.approx(1e-15)

    def test_mixed_signs(self):
        """正负混合值。"""
        state = RobotWorldState(
            joint_positions=np.array([-1.0, 0.0, 1.0]),
            joint_velocities=np.array([0.5, -0.5, 0.0]),
            joint_efforts=np.array([10.0, -10.0, 0.0]),
            n_joints=3,
        )
        vec = state.to_vector()
        assert float(vec[0]) == pytest.approx(-1.0)


class TestRobotWorldStateCopyEdgeCases:
    """copy() 边界。"""

    def test_copy_with_none_arrays(self):
        """None 数组的 copy 不崩溃。"""
        state = RobotWorldState(n_joints=3)
        cp = state.copy()
        # copy 时 joint_positions 为 None
        assert cp.joint_positions is None
        assert cp.n_joints == 3

    def test_copy_is_deep(self):
        """copy 后修改不影响原始。"""
        state = RobotWorldState(
            joint_positions=np.array([1.0, 2.0]),
            joint_velocities=np.zeros(2),
            joint_efforts=np.zeros(2),
            n_joints=2,
        )
        cp = state.copy()
        cp.joint_positions[0] = 999.0
        assert float(state.joint_positions[0]) == pytest.approx(1.0)


class TestRobotWorldStateDistanceEdgeCases:
    """distance() 边界。"""

    def test_distance_to_non_robot_fallback(self):
        """与不同类型 WorldState 比较走 to_vector 回退路径。"""

        class FakeState(WorldState):
            def to_vector(self):
                return np.zeros(6)  # 相同长度可计算

            @classmethod
            def from_vector(cls, vec):
                return cls()

            def distance(self, other):
                return 0.0

            def copy(self):
                return self

        state = RobotWorldState(n_joints=2)
        state._ensure_arrays()
        fake = FakeState()
        # RobotWorldState.distance 对非同类走 to_vector 回退
        d = state.distance(fake)
        assert isinstance(d, float)
        assert d == pytest.approx(0.0)  # 两个零向量距离为 0

    def test_distance_different_n_joints(self):
        """不同 n_joints 的 RobotWorldState 距离（向量长度不同 → 广播失败）。"""
        s1 = RobotWorldState(
            joint_positions=np.zeros(2),
            joint_velocities=np.zeros(2),
            joint_efforts=np.zeros(2),
            n_joints=2,
        )
        s2 = RobotWorldState(
            joint_positions=np.zeros(3),
            joint_velocities=np.zeros(3),
            joint_efforts=np.zeros(3),
            n_joints=3,
        )
        # to_vector 长度不同, numpy 会 raise
        with pytest.raises(ValueError):
            s1.distance(s2)


# =============================================================================
# RobotAction 边界条件
# =============================================================================


class TestRobotActionEdgeCases:
    """RobotAction 边界。"""

    def test_no_targets(self):
        """无目标 → apply 返回原始状态的拷贝。"""
        state = RobotWorldState(
            joint_positions=np.ones(3),
            joint_velocities=np.zeros(3),
            joint_efforts=np.zeros(3),
            n_joints=3,
        )
        action = RobotAction()
        result = action.apply(state)
        np.testing.assert_array_equal(result.joint_positions, state.joint_positions)

    def test_wrong_state_type(self):
        """非 RobotWorldState → TypeError。"""
        action = RobotAction(target_positions=np.zeros(3))

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

        with pytest.raises(TypeError, match="RobotAction"):
            action.apply(FakeState())

    def test_from_vector_odd_length(self):
        """from_vector 奇数长度 → positions 取前半，efforts 取后半。"""
        vec = np.arange(5, dtype=np.float64)  # n=5//2=2
        action = RobotAction.from_vector(vec)
        assert action.target_positions is not None
        assert len(action.target_positions) == 2
        assert action.target_efforts is not None
        assert len(action.target_efforts) == 3  # vec[2:]

    def test_from_vector_empty(self):
        """空向量 from_vector。"""
        vec = np.array([], dtype=np.float64)
        action = RobotAction.from_vector(vec)
        assert action.target_positions is not None
        assert len(action.target_positions) == 0

    def test_to_vector_no_targets(self):
        """无目标 → to_vector 返回 [0.0]。"""
        action = RobotAction()
        vec = action.to_vector()
        assert vec.shape == (1,)
        assert vec[0] == 0.0

    def test_only_positions(self):
        """仅有 target_positions。"""
        action = RobotAction(target_positions=np.array([1.0, 2.0]))
        vec = action.to_vector()
        np.testing.assert_array_equal(vec, [1.0, 2.0])

    def test_apply_computes_velocity(self):
        """apply 计算 delta_pos / dt 作为速度。"""
        state = RobotWorldState(
            joint_positions=np.zeros(3),
            joint_velocities=np.zeros(3),
            joint_efforts=np.zeros(3),
            n_joints=3,
            dt=0.01,
        )
        action = RobotAction(target_positions=np.array([0.1, 0.2, 0.3]), dt=0.01)
        result = action.apply(state)
        expected_vel = np.array([10.0, 20.0, 30.0])
        np.testing.assert_allclose(result.joint_velocities, expected_vel)


class TestRobotWorldStateToDictEdgeCases:
    """to_dict 边界。"""

    def test_to_dict_with_nan(self):
        """NaN 值 → to_dict 不崩溃（tolist 处理 NaN）。"""
        state = RobotWorldState(
            joint_positions=np.array([np.nan, 0.0]),
            joint_velocities=np.zeros(2),
            joint_efforts=np.zeros(2),
            n_joints=2,
        )
        d = state.to_dict()
        assert d["type"] == "RobotWorldState"
        # nan .tolist() → float('nan')
        assert d["joint_positions"][0] != d["joint_positions"][0]  # NaN != NaN

    def test_to_dict_preserves_n_joints(self):
        """to_dict 正确保存 n_joints。"""
        state = RobotWorldState(n_joints=7)
        d = state.to_dict()
        assert d["n_joints"] == 7
