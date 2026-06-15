"""CEWM v4.5.0 实时性集成测试 — KPI R-4

验证 cewm_step() / cewm_step_fast() 在配置 deadline 下 95% 运行达标。
"""

from __future__ import annotations

import time

from mci_world_model.sdk import (
    CartState,
    DeadlineConfig,
    DeadlineMonitor,
    EmergencyStop,
    ForceLimitConstraint,
    MCIWorldModel,
    PendulumState,
    PositionBoundConstraint,
    RobotWorldState,
    SafetyMonitor,
)

# ── 辅助 ──


def _percentile(data: list[float], p: float) -> float:
    s = sorted(data)
    idx = min(int(len(s) * p), len(s) - 1)
    return s[idx]


# ── DeadlineMonitor 基础测试 ──


class TestDeadlineMonitorBasic:
    """DeadlineMonitor 基本功能验证。"""

    def test_configure_and_start_stop(self):
        cfg = DeadlineConfig(deadline_ms=10)
        dm = DeadlineMonitor(cfg)
        dm.start()
        time.sleep(0.001)
        elapsed = dm.stop()
        assert elapsed > 0
        assert dm.statistics().total_steps == 1

    def test_deadline_violation_detected(self):
        cfg = DeadlineConfig(deadline_ms=1)
        dm = DeadlineMonitor(cfg)
        dm.start()
        time.sleep(0.005)  # 5ms > 1ms deadline
        dm.stop()
        assert dm.statistics().deadline_violations >= 1

    def test_no_deadline_violation_when_fast(self):
        cfg = DeadlineConfig(deadline_ms=1000)
        dm = DeadlineMonitor(cfg)
        dm.start()
        dm.stop()
        assert dm.statistics().deadline_violations == 0

    def test_should_degrade(self):
        cfg = DeadlineConfig(deadline_ms=1, degrade_threshold=0.5)
        dm = DeadlineMonitor(cfg)
        for _ in range(10):
            dm.start()
            time.sleep(0.005)  # 5ms > 1ms → 每次都违规
            dm.stop()
        assert dm.should_degrade()

    def test_percentile_statistics(self):
        cfg = DeadlineConfig(deadline_ms=0)
        dm = DeadlineMonitor(cfg)
        for _ in range(100):
            dm.start()
            time.sleep(0.0001)
            dm.stop()
        stats = dm.statistics()
        assert stats.p50_ms > 0
        assert stats.p95_ms >= stats.p50_ms
        assert stats.p99_ms >= stats.p95_ms

    def test_reset_degrade(self):
        cfg = DeadlineConfig(deadline_ms=1, degrade_threshold=0.5)
        dm = DeadlineMonitor(cfg)
        for _ in range(10):
            dm.start()
            time.sleep(0.005)
            dm.stop()
        assert dm.should_degrade()
        dm.reset_degrade()
        assert not dm.is_degraded


# ── cewm_step 延迟测量 ──


class TestCewmStepLatency:
    """cewm_step() / cewm_step_fast() 延迟分布。"""

    N_RUNS = 100

    def test_pendulum_step_latency(self):
        wm = MCIWorldModel()
        obs = PendulumState(theta=0.5, omega=1.0)
        goal = PendulumState(theta=0.0, omega=0.0)
        latencies = []
        for _ in range(self.N_RUNS):
            t0 = time.perf_counter()
            wm.cewm_step(observation=obs, goal=goal)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        _p95 = _percentile(latencies, 0.95)
        p99 = _percentile(latencies, 0.99)
        # KPI: Pendulum p99 < 100ms (宽松门禁)
        assert p99 < 100.0, f"Pendulum p99={p99:.2f}ms > 100ms"

    def test_pendulum_step_fast_latency(self):
        wm = MCIWorldModel()
        obs = PendulumState(theta=0.5, omega=1.0)
        goal = PendulumState(theta=0.0, omega=0.0)
        latencies = []
        for _ in range(self.N_RUNS):
            t0 = time.perf_counter()
            wm.cewm_step_fast(observation=obs, goal=goal)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        p99 = _percentile(latencies, 0.99)
        # KPI: 快速路径 p99 < 10ms
        assert p99 < 10.0, f"Pendulum fast p99={p99:.2f}ms > 10ms"

    def test_cart_step_latency(self):
        wm = MCIWorldModel()
        obs = CartState(x=1.0, v=0.5)
        goal = CartState(x=10.0, v=0.0)
        latencies = []
        for _ in range(self.N_RUNS):
            t0 = time.perf_counter()
            wm.cewm_step(observation=obs, goal=goal)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        p99 = _percentile(latencies, 0.99)
        assert p99 < 100.0, f"Cart p99={p99:.2f}ms > 100ms"

    def test_robot_step_fast_latency(self):
        wm = MCIWorldModel()
        obs = RobotWorldState(n_joints=6)
        obs._ensure_arrays()
        latencies = []
        for _ in range(self.N_RUNS):
            t0 = time.perf_counter()
            wm.cewm_step_fast(observation=obs)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        p99 = _percentile(latencies, 0.99)
        assert p99 < 50.0, f"Robot fast p99={p99:.2f}ms > 50ms"


# ── Deadline 集成闭环测试 ──


class TestDeadlineIntegration:
    """DeadlineMonitor + cewm_step 闭环。"""

    def test_deadline_monitor_wraps_cewm_step(self):
        wm = MCIWorldModel()
        cfg = DeadlineConfig(deadline_ms=100)
        dm = DeadlineMonitor(cfg)
        obs = PendulumState(theta=0.5, omega=1.0)
        goal = PendulumState(theta=0.0, omega=0.0)

        for _ in range(20):
            dm.start()
            wm.cewm_step(observation=obs, goal=goal)
            dm.stop()

        stats = dm.statistics()
        assert stats.total_steps == 20
        # 100ms deadline 下应全通过
        assert stats.deadline_violations == 0

    def test_95_percent_deadline_achievement(self):
        """KPI R-4: 100 次运行 95% 在 deadline 内完成。"""
        wm = MCIWorldModel()
        cfg = DeadlineConfig(deadline_ms=50)
        dm = DeadlineMonitor(cfg)
        obs = PendulumState(theta=0.5, omega=1.0)

        for _ in range(100):
            dm.start()
            wm.cewm_step(observation=obs)
            dm.stop()

        stats = dm.statistics()
        achievement_rate = 1.0 - stats.violation_rate
        assert achievement_rate >= 0.95, f"仅 {achievement_rate:.1%} 在 deadline 内，需 ≥ 95%"

    def test_fast_path_deadline_achievement(self):
        """快速路径 100 次运行 95% 在 deadline 内。"""
        wm = MCIWorldModel()
        cfg = DeadlineConfig(deadline_ms=10)
        dm = DeadlineMonitor(cfg)
        obs = PendulumState(theta=0.5, omega=1.0)

        for _ in range(100):
            dm.start()
            wm.cewm_step_fast(observation=obs)
            dm.stop()

        stats = dm.statistics()
        achievement_rate = 1.0 - stats.violation_rate
        assert achievement_rate >= 0.95, f"快速路径仅 {achievement_rate:.1%} 在 deadline 内，需 ≥ 95%"


# ── SafetyMonitor + DeadlineMonitor 组合测试 ──


class TestSafetyDeadlineCombo:
    """安全约束 + DeadlineMonitor 组合。"""

    def test_safety_check_does_not_exceed_deadline(self):
        wm = MCIWorldModel()
        monitor = SafetyMonitor()
        monitor.register(ForceLimitConstraint(max_torque=10.0))
        monitor.register(PositionBoundConstraint(max_theta=3.14))
        wm._safety_monitor = monitor

        cfg = DeadlineConfig(deadline_ms=50)
        dm = DeadlineMonitor(cfg)

        obs = PendulumState(theta=0.5, omega=1.0)
        for _ in range(50):
            dm.start()
            wm.cewm_step(observation=obs)
            dm.stop()

        stats = dm.statistics()
        assert stats.violation_rate < 0.1, "安全检查不应导致频繁 deadline 违规"


# ── EmergencyStop 延迟测试 ──


class TestEmergencyStopLatency:
    """EmergencyStop 响应延迟。"""

    def test_trigger_sets_stop_flag_under_50ms(self):
        estop = EmergencyStop()
        estop.arm(timeout_ms=5000)
        t0 = time.perf_counter()
        estop.trigger(reason="test")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 50.0, f"trigger 延迟 {elapsed_ms:.2f}ms > 50ms"
        assert estop.is_stopped
        estop.disarm()

    def test_is_stopped_check_under_1ms(self):
        estop = EmergencyStop()
        estop.trigger(reason="test")
        latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            _ = estop.is_stopped
            latencies.append((time.perf_counter() - t0) * 1000.0)
        p99 = _percentile(latencies, 0.99)
        assert p99 < 1.0, f"is_stopped 检查 p99={p99:.4f}ms > 1ms"
        estop.reset()
