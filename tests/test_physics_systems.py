"""MCI World Model v5.1.0 — 新增物理系统动力学 测试

v5.1.0 新增: spring_mass, projectile, fluid_flow 动力学函数。
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._generalized_physics import (
    GeneralizedPhysicsPredictor,
    euler_step,
    fluid_flow_dynamics,
    projectile_dynamics,
    spring_mass_dynamics,
)

# ═══════════════════════════════════════════════════════════════════════════
# 弹簧-质量阻尼系统
# ═══════════════════════════════════════════════════════════════════════════


class TestSpringMass:
    """弹簧-质量阻尼系统动力学测试。"""

    def test_spring_mass_exists(self):
        """spring_mass_dynamics 可调用。"""
        state = np.array([0.0, 0.0])
        action = np.array([0.0])
        result = spring_mass_dynamics(state, action)
        assert result.shape == (2,)

    def test_spring_restoring_force(self):
        """偏离平衡位置时产生恢复力。"""
        # x=1.0, v=0: dx/dt=0, dv/dt=-10*1 - 0.5*0 = -10.0
        state = np.array([1.0, 0.0])
        action = np.array([0.0])
        result = spring_mass_dynamics(state, action)
        assert result[0] == 0.0  # dx/dt = v = 0
        assert result[1] == pytest.approx(-10.0)  # dv/dt = -k/m * x = -10

    def test_spring_damping(self):
        """阻尼减缓运动。"""
        # x=0, v=1: dv/dt = -0.5*1 = -0.5
        state = np.array([0.0, 1.0])
        action = np.array([0.0])
        result = spring_mass_dynamics(state, action)
        assert result[0] == 1.0  # dx/dt = v = 1
        assert result[1] == pytest.approx(-0.5)  # dv/dt = -c/m * v = -0.5

    def test_spring_external_force(self):
        """外力加速质量块。"""
        # x=0, v=0, F=5.0: dv/dt = 5.0
        state = np.array([0.0, 0.0])
        action = np.array([5.0])
        result = spring_mass_dynamics(state, action)
        assert result[1] == pytest.approx(5.0)

    def test_spring_equilibrium(self):
        """平衡位置无加速度。"""
        state = np.array([0.0, 0.0])
        action = np.array([0.0])
        result = spring_mass_dynamics(state, action)
        assert result[0] == 0.0
        assert result[1] == 0.0

    def test_spring_euler_step_stable(self):
        """Euler 步进 100 步不爆炸 (小 dt)。"""
        state = np.array([1.0, 0.0])
        action = np.array([0.0])
        dt = 0.001
        for _ in range(100):
            state = euler_step(state, action, spring_mass_dynamics, dt)
        assert np.all(np.isfinite(state))
        # 能量应衰减 (有阻尼)
        assert abs(state[0]) < 1.1  # 位移应减小

    def test_spring_registered_in_predictor(self):
        """spring_mass 已注册到 GeneralizedPhysicsPredictor。"""
        pred = GeneralizedPhysicsPredictor()
        assert "spring_mass" in pred.available_backends


# ═══════════════════════════════════════════════════════════════════════════
# 抛体运动
# ═══════════════════════════════════════════════════════════════════════════


class TestProjectile:
    """抛体运动动力学测试。"""

    def test_projectile_exists(self):
        """projectile_dynamics 可调用。"""
        state = np.array([0.0, 0.0, 10.0, 10.0])
        action = np.array([0.0, 0.0])
        result = projectile_dynamics(state, action)
        assert result.shape == (4,)

    def test_freefall_gravity(self):
        """自由落体: dv_y/dt = -g。"""
        # x=0, y=0, vx=0, vy=0 (静止释放)
        state = np.array([0.0, 0.0, 0.0, 0.0])
        action = np.array([0.0, 0.0])
        result = projectile_dynamics(state, action)
        assert result[0] == 0.0  # dx/dt = vx = 0
        assert result[1] == 0.0  # dy/dt = vy = 0
        assert result[2] == 0.0  # dvx/dt = -drag*0 = 0
        assert result[3] == pytest.approx(-9.81)  # dvy/dt = -g

    def test_horizontal_throw(self):
        """水平抛出: vx 保持, vy 受重力。"""
        # vx=10, vy=0
        state = np.array([0.0, 0.0, 10.0, 0.0])
        action = np.array([0.0, 0.0])
        result = projectile_dynamics(state, action)
        assert result[0] == 10.0  # dx/dt = vx = 10
        assert result[3] == pytest.approx(-9.81)  # dvy/dt = -g

    def test_projectile_with_thrust(self):
        """有推力时加速度增加。"""
        state = np.array([0.0, 0.0, 0.0, 0.0])
        action = np.array([5.0, 10.0])  # F_x=5, F_y=10
        result = projectile_dynamics(state, action)
        assert result[2] == pytest.approx(5.0)  # dvx/dt = F_x/m = 5
        assert result[3] == pytest.approx(-9.81 + 10.0)  # dvy/dt = -g + F_y/m

    def test_projectile_trajectory_parabolic(self):
        """抛体轨迹呈抛物线 (y 先升后降)。"""
        state = np.array([0.0, 0.0, 5.0, 10.0])  # 斜抛
        action = np.array([0.0, 0.0])
        dt = 0.01

        y_values = [state[1]]
        for _ in range(200):  # 2 秒
            state = euler_step(state, action, projectile_dynamics, dt)
            y_values.append(state[1])

        # y 应先增后减
        max_y = max(y_values)
        assert max_y > 0.0
        assert y_values[-1] < max_y  # 最终下落

    def test_projectile_registered_in_predictor(self):
        """projectile 已注册到 GeneralizedPhysicsPredictor。"""
        pred = GeneralizedPhysicsPredictor()
        assert "projectile" in pred.available_backends


# ═══════════════════════════════════════════════════════════════════════════
# 流体管道系统
# ═══════════════════════════════════════════════════════════════════════════


class TestFluidFlow:
    """流体管道系统动力学测试。"""

    def test_fluid_exists(self):
        """fluid_flow_dynamics 可调用。"""
        state = np.array([0.0, 0.0, 293.15])
        action = np.array([0.0, 0.0])
        result = fluid_flow_dynamics(state, action)
        assert result.shape == (3,)

    def test_pressure_decays(self):
        """无入口压力时压力衰减。"""
        # P=10, Q=0, T=293.15, P_in=0
        state = np.array([10.0, 0.0, 293.15])
        action = np.array([0.0, 0.0])
        result = fluid_flow_dynamics(state, action)
        # dP/dt = -k_p*P + R*Q + P_in = -1*10 + 0 + 0 = -10
        assert result[0] == pytest.approx(-10.0)

    def test_inlet_pressure_increases(self):
        """入口压力增加管道压力。"""
        state = np.array([0.0, 0.0, 293.15])
        action = np.array([5.0, 0.0])  # P_in = 5
        result = fluid_flow_dynamics(state, action)
        # dP/dt = -0 + 0 + 5 = 5
        assert result[0] == pytest.approx(5.0)

    def test_temperature_equilibrium(self):
        """环境温度时 dT/dt = 0。"""
        state = np.array([0.0, 0.0, 293.15])
        action = np.array([0.0, 0.0])
        result = fluid_flow_dynamics(state, action)
        # dT/dt = -k_t*(T - T_amb) + Q*T_heat = -0.1*0 + 0 = 0
        assert result[2] == pytest.approx(0.0)

    def test_temperature_cooling(self):
        """高温流体自然冷却。"""
        state = np.array([0.0, 0.0, 373.15])  # 100°C
        action = np.array([0.0, 0.0])
        result = fluid_flow_dynamics(state, action)
        # dT/dt = -0.1*(373.15 - 293.15) = -8.0
        assert result[2] == pytest.approx(-8.0)

    def test_flow_rate_pressure_driven(self):
        """压力差驱动流量。"""
        # P=5, Q=0: dQ/dt = (5 - 0) / 1 = 5
        state = np.array([5.0, 0.0, 293.15])
        action = np.array([0.0, 0.0])
        result = fluid_flow_dynamics(state, action)
        assert result[1] == pytest.approx(5.0)

    def test_fluid_registered_in_predictor(self):
        """fluid_flow 已注册到 GeneralizedPhysicsPredictor。"""
        pred = GeneralizedPhysicsPredictor()
        assert "fluid_flow" in pred.available_backends


# ═══════════════════════════════════════════════════════════════════════════
# GeneralizedPhysicsPredictor 新增后端
# ═══════════════════════════════════════════════════════════════════════════


class TestGeneralizedPhysicsNewBackends:
    """GeneralizedPhysicsPredictor 支持 6 个后端。"""

    def test_six_backends_registered(self):
        """6 个后端已注册。"""
        pred = GeneralizedPhysicsPredictor()
        expected = {"pendulum", "cart", "double_pendulum", "spring_mass", "projectile", "fluid_flow"}
        assert expected.issubset(set(pred.available_backends))

    def test_spring_mass_backend(self):
        """spring_mass 后端可切换。"""
        pred = GeneralizedPhysicsPredictor()
        pred.set_backend("spring_mass")
        assert pred.backend == "spring_mass"

    def test_projectile_backend(self):
        """projectile 后端可切换。"""
        pred = GeneralizedPhysicsPredictor()
        pred.set_backend("projectile")
        assert pred.backend == "projectile"

    def test_fluid_flow_backend(self):
        """fluid_flow 后端可切换。"""
        pred = GeneralizedPhysicsPredictor()
        pred.set_backend("fluid_flow")
        assert pred.backend == "fluid_flow"
