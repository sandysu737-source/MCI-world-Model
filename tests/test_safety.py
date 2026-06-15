"""Phase 2 安全约束层测试 — SafetyConstraint + SafetyMonitor + cewm_step 集成.

覆盖:
    - ForceLimitConstraint: 力矩/力超限检测
    - PositionBoundConstraint: 位置/角度越界检测
    - VelocityLimitConstraint: 速度/角速度超限检测
    - SafetyMonitor: 链式注册 + 短路求值
    - cewm_step() 安全集成: 约束违反时返回 safety_violation
"""

import pytest

from mci_world_model.sdk._safety import (
    ForceLimitConstraint,
    PositionBoundConstraint,
    SafetyCheckResult,
    SafetyConstraint,
    SafetyMonitor,
    VelocityLimitConstraint,
)
from mci_world_model.sdk._world_state import (
    CartAction,
    CartState,
    PendulumAction,
    PendulumState,
)

# =============================================================================
# SafetyConstraint ABC
# =============================================================================


class TestSafetyConstraintABC:
    """SafetyConstraint 抽象基类测试。"""

    def test_cannot_instantiate_directly(self):
        """SafetyConstraint 是 ABC，不能直接实例化。"""
        with pytest.raises(TypeError):
            SafetyConstraint()

    def test_subclass_must_implement_check(self):
        """子类必须实现 check() 方法。"""

        class IncompleteConstraint(SafetyConstraint):
            @property
            def name(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError):
            IncompleteConstraint()

    def test_subclass_with_all_methods(self):
        """完整实现可以实例化。"""

        class DummyConstraint(SafetyConstraint):
            @property
            def name(self) -> str:
                return "dummy"

            def check(self, state, action):
                return SafetyCheckResult(passed=True, constraint_name=self.name)

        c = DummyConstraint()
        assert c.name == "dummy"


# =============================================================================
# ForceLimitConstraint
# =============================================================================


class TestForceLimitConstraint:
    """力矩/力限制约束测试。"""

    def test_pendulum_torque_within_limit(self):
        """力矩在限制内 — 通过。"""
        c = ForceLimitConstraint(max_torque=10.0)
        state = PendulumState(theta=0.1, omega=0.0)
        action = PendulumAction(torque=5.0)
        result = c.check(state, action)
        assert result.passed is True

    def test_pendulum_torque_exceeds_limit(self):
        """力矩超限 — 不通过。"""
        c = ForceLimitConstraint(max_torque=10.0)
        state = PendulumState(theta=0.1, omega=0.0)
        action = PendulumAction(torque=15.0)
        result = c.check(state, action)
        assert result.passed is False
        assert "力矩超限" in result.reason
        assert result.details["torque"] == 15.0

    def test_cart_force_within_limit(self):
        """力在限制内 — 通过。"""
        c = ForceLimitConstraint(max_force=20.0)
        state = CartState(x=0.0, v=0.0)
        action = CartAction(force=10.0)
        result = c.check(state, action)
        assert result.passed is True

    def test_cart_force_exceeds_limit(self):
        """力超限 — 不通过。"""
        c = ForceLimitConstraint(max_force=5.0)
        state = CartState(x=0.0, v=0.0)
        action = CartAction(force=8.0)
        result = c.check(state, action)
        assert result.passed is False
        assert "力超限" in result.reason

    def test_no_action_passes(self):
        """无动作时通过。"""
        c = ForceLimitConstraint()
        state = PendulumState(theta=0.1, omega=0.0)
        result = c.check(state, None)
        assert result.passed is True

    def test_name(self):
        c = ForceLimitConstraint()
        assert c.name == "force_limit"


# =============================================================================
# PositionBoundConstraint
# =============================================================================


class TestPositionBoundConstraint:
    """位置/角度边界约束测试。"""

    def test_pendulum_theta_within_bound(self):
        """角度在边界内 — 通过。"""
        c = PositionBoundConstraint(max_theta=3.14159)
        state = PendulumState(theta=1.0, omega=0.0)
        result = c.check(state, None)
        assert result.passed is True

    def test_pendulum_theta_exceeds_bound(self):
        """角度越界 — 不通过。"""
        c = PositionBoundConstraint(max_theta=3.0)
        state = PendulumState(theta=3.5, omega=0.0)
        result = c.check(state, None)
        assert result.passed is False
        assert "角度越界" in result.reason

    def test_cart_x_within_bound(self):
        """位置在边界内 — 通过。"""
        c = PositionBoundConstraint(max_x=100.0)
        state = CartState(x=50.0, v=0.0)
        result = c.check(state, None)
        assert result.passed is True

    def test_cart_x_exceeds_bound(self):
        """位置越界 — 不通过。"""
        c = PositionBoundConstraint(max_x=10.0)
        state = CartState(x=15.0, v=0.0)
        result = c.check(state, None)
        assert result.passed is False
        assert "位置越界" in result.reason

    def test_name(self):
        c = PositionBoundConstraint()
        assert c.name == "position_bound"


# =============================================================================
# VelocityLimitConstraint
# =============================================================================


class TestVelocityLimitConstraint:
    """速度/角速度限制约束测试。"""

    def test_pendulum_omega_within_limit(self):
        """角速度在限制内 — 通过。"""
        c = VelocityLimitConstraint(max_omega=8.0)
        state = PendulumState(theta=0.1, omega=5.0)
        result = c.check(state, None)
        assert result.passed is True

    def test_pendulum_omega_exceeds_limit(self):
        """角速度超限 — 不通过。"""
        c = VelocityLimitConstraint(max_omega=5.0)
        state = PendulumState(theta=0.1, omega=7.0)
        result = c.check(state, None)
        assert result.passed is False
        assert "角速度超限" in result.reason

    def test_cart_v_within_limit(self):
        """速度在限制内 — 通过。"""
        c = VelocityLimitConstraint(max_velocity=50.0)
        state = CartState(x=0.0, v=30.0)
        result = c.check(state, None)
        assert result.passed is True

    def test_cart_v_exceeds_limit(self):
        """速度超限 — 不通过。"""
        c = VelocityLimitConstraint(max_velocity=10.0)
        state = CartState(x=0.0, v=15.0)
        result = c.check(state, None)
        assert result.passed is False
        assert "速度超限" in result.reason

    def test_name(self):
        c = VelocityLimitConstraint()
        assert c.name == "velocity_limit"


# =============================================================================
# SafetyMonitor
# =============================================================================


class TestSafetyMonitor:
    """安全监控器测试。"""

    def test_empty_monitor_passes(self):
        """空监控器 — 全部通过。"""
        monitor = SafetyMonitor()
        state = PendulumState(theta=0.1, omega=5.0)
        result = monitor.check_all(state)
        assert result.passed is True
        assert result.constraint_name == "all_constraints"

    def test_single_constraint_passes(self):
        """单个约束通过。"""
        monitor = SafetyMonitor()
        monitor.register(ForceLimitConstraint(max_torque=10.0))
        state = PendulumState(theta=0.1, omega=0.0)
        action = PendulumAction(torque=5.0)
        result = monitor.check_all(state, action)
        assert result.passed is True

    def test_single_constraint_violates(self):
        """单个约束不通过。"""
        monitor = SafetyMonitor()
        monitor.register(ForceLimitConstraint(max_torque=10.0))
        state = PendulumState(theta=0.1, omega=0.0)
        action = PendulumAction(torque=15.0)
        result = monitor.check_all(state, action)
        assert result.passed is False

    def test_multiple_constraints_all_pass(self):
        """多个约束全部通过。"""
        monitor = SafetyMonitor()
        monitor.register(ForceLimitConstraint(max_torque=10.0))
        monitor.register(PositionBoundConstraint(max_theta=3.14))
        monitor.register(VelocityLimitConstraint(max_omega=8.0))

        state = PendulumState(theta=1.0, omega=5.0)
        action = PendulumAction(torque=5.0)
        result = monitor.check_all(state, action)
        assert result.passed is True

    def test_multiple_constraints_short_circuit(self):
        """短路求值：第一个不通过就返回。"""
        monitor = SafetyMonitor()
        monitor.register(ForceLimitConstraint(max_torque=10.0))
        monitor.register(PositionBoundConstraint(max_theta=3.14))

        state = PendulumState(theta=0.1, omega=0.0)
        action = PendulumAction(torque=15.0)
        result = monitor.check_all(state, action)
        assert result.passed is False
        assert result.constraint_name == "force_limit"

    def test_check_individual(self):
        """逐个检查所有约束。"""
        monitor = SafetyMonitor()
        monitor.register(ForceLimitConstraint(max_torque=10.0))
        monitor.register(PositionBoundConstraint(max_theta=3.0))

        state = PendulumState(theta=3.5, omega=0.0)
        action = PendulumAction(torque=15.0)
        results = monitor.check_individual(state, action)
        assert len(results) == 2
        assert results[0].passed is False  # force exceeded
        assert results[1].passed is False  # position exceeded

    def test_statistics(self):
        """统计信息正确。"""
        monitor = SafetyMonitor()
        monitor.register(ForceLimitConstraint())
        monitor.register(PositionBoundConstraint())

        state = PendulumState(theta=0.1, omega=0.0)
        action = PendulumAction(torque=5.0)

        monitor.check_all(state, action)
        stats = monitor.statistics()
        assert stats["constraint_count"] == 2
        assert stats["total_checks"] == 1
        assert stats["total_violations"] == 0

    def test_constraint_count(self):
        monitor = SafetyMonitor()
        assert monitor.constraint_count == 0
        monitor.register(ForceLimitConstraint())
        assert monitor.constraint_count == 1

    def test_violation_count(self):
        monitor = SafetyMonitor()
        monitor.register(ForceLimitConstraint(max_torque=1.0))

        state = PendulumState(theta=0.1, omega=0.0)
        action = PendulumAction(torque=5.0)

        monitor.check_all(state, action)
        assert monitor.violation_count == 1


# =============================================================================
# cewm_step() 安全集成
# =============================================================================


class TestCewmStepSafetyIntegration:
    """cewm_step() 安全检查集成测试。"""

    def test_cewm_step_no_safety_monitor(self):
        """无 SafetyMonitor — 正常运行。"""
        from mci_world_model.sdk._world_model import MCIWorldModel

        wm = MCIWorldModel()
        result = wm.cewm_step(
            observation=PendulumState(theta=0.1, omega=0.0),
            goal=PendulumState(theta=0.0, omega=0.0),
        )
        assert result["safety_violation"] is False
        assert result["safety_reason"] == ""

    def test_cewm_step_with_safety_monitor_passes(self):
        """有 SafetyMonitor，约束通过 — 正常运行。"""
        from mci_world_model.sdk._world_model import MCIWorldModel

        wm = MCIWorldModel()
        wm._safety_monitor = SafetyMonitor()
        wm._safety_monitor.register(ForceLimitConstraint(max_torque=10.0))

        result = wm.cewm_step(
            observation=PendulumState(theta=0.1, omega=0.0),
            goal=PendulumState(theta=0.0, omega=0.0),
            action=PendulumAction(torque=5.0),
        )
        assert result["safety_violation"] is False

    def test_cewm_step_safety_violation(self):
        """有 SafetyMonitor，约束违反 — 返回 safety_violation。"""
        from mci_world_model.sdk._world_model import MCIWorldModel

        wm = MCIWorldModel()
        wm._safety_monitor = SafetyMonitor()
        wm._safety_monitor.register(ForceLimitConstraint(max_torque=2.0))

        result = wm.cewm_step(
            observation=PendulumState(theta=0.1, omega=0.0),
            goal=PendulumState(theta=0.0, omega=0.0),
            action=PendulumAction(torque=10.0),
        )
        assert result["safety_violation"] is True
        assert "力矩超限" in result["safety_reason"]

    def test_cewm_step_position_violation(self):
        """位置越界时安全检查拦截。"""
        from mci_world_model.sdk._world_model import MCIWorldModel

        wm = MCIWorldModel()
        wm._safety_monitor = SafetyMonitor()
        wm._safety_monitor.register(PositionBoundConstraint(max_theta=3.0))

        result = wm.cewm_step(
            observation=PendulumState(theta=3.5, omega=0.0),
            goal=PendulumState(theta=0.0, omega=0.0),
        )
        assert result["safety_violation"] is True
        assert "角度越界" in result["safety_reason"]
