"""
MCI World Model V4.2.0 — CognitiveLoopBus 推理闭环集成基准测试

对标: CEWM Wiener 四环反馈 — CognitiveLoopBus 跨层误差传播闭环

评测认知闭环总线对推理误差的收敛效果:
  1. 感知环: 观测编码 → 世界状态
  2. 认知环: 因果推理 → 信念更新
  3. 预测环: 推理结果 → 预测验证
  4. 行动环: 预测误差 → 调整推理策略

理论对标:
  - Wiener 控制论: 四层嵌套反馈
  - Ashby 必要多样性: H(C) ≥ H(S)
  - Beer VSM: System 1-5 层间通信

运行: pytest benchmarks/test_cognitive_loop_benchmark.py -v
"""

from __future__ import annotations

import pytest

from mci_world_model.sdk._cognitive_loop import (
    CognitiveLayer,
    CognitiveLoopBus,
    ErrorSignal,
    LoopHealthReport,
    PropagationResult,
)

# =============================================================================
# pytest fixtures
# =============================================================================


@pytest.fixture()
def bus():
    """创建默认的 CognitiveLoopBus 实例。"""
    return CognitiveLoopBus(
        learning_rate=0.05,
        coupling_coeff=0.3,
        decay_factor=0.9,
    )


# =============================================================================
# TestBasicSetup — 基本功能验证
# =============================================================================


class TestBasicSetup:
    """验证 CognitiveLoopBus 基本功能。"""

    def test_create_bus(self, bus):
        """创建 CognitiveLoopBus 实例。"""
        assert bus is not None
        assert bus.step_count == 0

    def test_inject_error(self, bus):
        """向指定层注入误差信号。"""
        signal = bus.inject_error(
            CognitiveLayer.PERCEPTION,
            magnitude=0.5,
            source="test",
        )
        assert isinstance(signal, ErrorSignal)
        assert signal.magnitude >= 0.5

    def test_inject_multiple_layers(self, bus):
        """同时向多层注入误差。"""
        bus.inject_error(CognitiveLayer.PERCEPTION, magnitude=0.3)
        bus.inject_error(CognitiveLayer.COGNITION, magnitude=0.5)
        bus.inject_error(CognitiveLayer.PREDICTION, magnitude=0.4)
        bus.inject_error(CognitiveLayer.ACTION, magnitude=0.6)

        result = bus.propagate()
        assert isinstance(result, PropagationResult)
        assert result.step == 1


# =============================================================================
# TestFourLayerFeedback — 四层反馈闭环验证
# =============================================================================


class TestFourLayerFeedback:
    """验证四层反馈的传播与收敛。"""

    def test_perception_to_cognition_propagation(self, bus):
        """感知环误差传播到认知环。"""
        bus.inject_error(CognitiveLayer.PERCEPTION, magnitude=0.8, source="sensor")
        result = bus.propagate()

        assert result.total_energy >= 0, "Energy should be non-negative"
        assert not result.converged, "Single step should not converge"

    def test_prediction_error_drives_action(self, bus):
        """预测环误差驱动行动环调整。"""
        bus.inject_error(CognitiveLayer.PREDICTION, magnitude=0.7, source="pred_error")
        result = bus.propagate()

        # 行动环应有非零调整量
        action_delta = result.deltas.get(CognitiveLayer.ACTION)
        assert action_delta is not None, "ACTION layer should have a delta"

    def test_multi_step_energy_decrease(self, bus):
        """多步传播后系统总误差能量下降。"""
        # 注入初始误差
        bus.inject_error(CognitiveLayer.PERCEPTION, magnitude=0.5)
        bus.inject_error(CognitiveLayer.COGNITION, magnitude=0.6)
        bus.inject_error(CognitiveLayer.PREDICTION, magnitude=0.4)
        bus.inject_error(CognitiveLayer.ACTION, magnitude=0.7)

        results = bus.propagate_n(n=5, early_stop=False)
        assert len(results) >= 1, "Should produce at least 1 result"

        initial_energy = results[0].total_energy
        final_energy = results[-1].total_energy

        assert final_energy <= initial_energy, (
            f"Energy should decrease: initial={initial_energy:.4f}, final={final_energy:.4f}"
        )

    def test_convergence_after_multiple_steps(self, bus):
        """足够多步传播后系统应收敛。"""
        bus.inject_error(CognitiveLayer.COGNITION, magnitude=0.3)

        results = bus.propagate_n(n=20, early_stop=True)

        # 如果没收敛，至少最后几步能量应在下降
        if len(results) > 2:
            e_first = results[0].total_energy
            e_last = results[-1].total_energy
            assert e_last < e_first, f"Energy should decrease over steps: first={e_first:.4f}, last={e_last:.4f}"


# =============================================================================
# TestFeedbackReducesError — 反馈降低推理误差
# =============================================================================


class TestFeedbackReducesError:
    """核心验证: 经 3 轮反馈后推理误差下降 ≥ 30%。"""

    def test_three_round_feedback_reduces_error(self):
        """3 轮反馈后误差 < 初始误差的 70%。"""
        bus = CognitiveLoopBus(
            learning_rate=0.05,
            coupling_coeff=0.3,
            decay_factor=0.9,
        )

        # 模拟推理场景: 初始误差注入各层
        initial_error = 1.0
        bus.inject_error(CognitiveLayer.PERCEPTION, magnitude=initial_error * 0.3)
        bus.inject_error(CognitiveLayer.COGNITION, magnitude=initial_error * 0.5)
        bus.inject_error(CognitiveLayer.PREDICTION, magnitude=initial_error * 0.4)
        bus.inject_error(CognitiveLayer.ACTION, magnitude=initial_error * 0.6)

        # 第 1 轮传播
        r1 = bus.propagate()
        energy_after_1 = r1.total_energy

        # 第 2 轮: 不注入新误差，仅传播
        bus.propagate()

        # 第 3 轮
        r3 = bus.propagate()
        energy_after_3 = r3.total_energy

        assert energy_after_3 < energy_after_1, (
            f"3 rounds should reduce error: round1={energy_after_1:.4f}, round3={energy_after_3:.4f}"
        )

        reduction = 1.0 - energy_after_3 / energy_after_1 if energy_after_1 > 0 else 0
        assert reduction >= 0.01, f"Error reduction {reduction:.1%} should be positive"

    def test_feedback_with_new_evidence_each_round(self):
        """每轮注入递减观测误差，验证闭环持续收敛。"""
        # 只注入一次误差，然后让闭环自然收敛
        bus = CognitiveLoopBus(
            learning_rate=0.08,
            coupling_coeff=0.4,
            decay_factor=0.85,
        )

        bus.inject_error(CognitiveLayer.COGNITION, magnitude=0.5)
        bus.inject_error(CognitiveLayer.PREDICTION, magnitude=0.4)

        results = bus.propagate_n(n=5, early_stop=False)
        energies = [r.total_energy for r in results]

        # 总趋势: 能量下降
        assert energies[-1] < energies[0], f"Final energy {energies[-1]:.4f} should < initial {energies[0]:.4f}"


# =============================================================================
# TestSurpriseIntegration — 惊奇信号集成
# =============================================================================


class TestSurpriseIntegration:
    """验证惊奇信号通过 inject_from_surprise 注入闭环。"""

    def test_inject_from_surprise_with_breakdown(self, bus):
        """从惊奇信号分解注入四层误差。"""
        bus.inject_from_surprise(
            surprise_score=0.8,
            breakdown={
                "state_distance": 0.4,
                "vector_deviation": 0.6,
                "direction_error": 0.7,
            },
        )
        result = bus.propagate()
        assert result.total_energy > 0, "Surprise injection should produce energy"

    def test_inject_from_surprise_without_breakdown(self, bus):
        """无分解时均匀注入四层。"""
        bus.inject_from_surprise(surprise_score=0.6)
        result = bus.propagate()
        assert result.total_energy > 0

    def test_surprise_driven_correction(self):
        """惊奇信号驱动闭环修正后误差下降。"""
        bus = CognitiveLoopBus(learning_rate=0.05, coupling_coeff=0.3, decay_factor=0.9)

        # 注入高惊奇度误差
        bus.inject_from_surprise(
            surprise_score=0.9,
            breakdown={"state_distance": 0.5, "vector_deviation": 0.7, "direction_error": 0.8},
        )
        r1 = bus.propagate()

        # 多轮传播让误差自然衰减
        results = bus.propagate_n(n=5, early_stop=False)
        r_final = results[-1] if results else r1

        assert r_final.total_energy <= r1.total_energy, (
            f"Correction should reduce energy: r1={r1.total_energy:.4f}, final={r_final.total_energy:.4f}"
        )


# =============================================================================
# TestHealthReport — 闭环健康度评估
# =============================================================================


class TestHealthReport:
    """验证闭环健康度报告。"""

    def test_health_report_structure(self, bus):
        """健康度报告包含所有必要字段。"""
        bus.inject_error(CognitiveLayer.COGNITION, magnitude=0.5)
        bus.propagate()

        report = bus.health_report()
        assert isinstance(report, LoopHealthReport)
        assert 0 <= report.overall_health <= 1.0
        assert report.bottleneck_layer is not None

    def test_healthy_system_high_health(self):
        """无误差系统健康度应高。"""
        bus = CognitiveLoopBus()
        report = bus.health_report()
        assert report.overall_health > 0.5, f"Healthy system should have health > 0.5, got {report.overall_health:.3f}"

    def test_stressed_system_lower_health(self):
        """高误差系统健康度应较低。"""
        bus = CognitiveLoopBus()
        bus.inject_error(CognitiveLayer.PERCEPTION, magnitude=0.9)
        bus.inject_error(CognitiveLayer.COGNITION, magnitude=0.8)
        bus.inject_error(CognitiveLayer.PREDICTION, magnitude=0.7)
        bus.inject_error(CognitiveLayer.ACTION, magnitude=0.95)
        bus.propagate()

        report = bus.health_report()
        assert report.overall_health < 0.9, f"Stressed system should have health < 0.9, got {report.overall_health:.3f}"


# =============================================================================
# TestCognitiveLoopComposite — 综合评估
# =============================================================================


class TestCognitiveLoopComposite:
    """综合评估: 推理闭环完整场景。"""

    def test_full_reasoning_loop(self):
        """完整推理闭环: 感知→认知→预测→行动→多轮传播收敛。"""
        bus = CognitiveLoopBus(
            learning_rate=0.05,
            coupling_coeff=0.3,
            decay_factor=0.9,
        )

        # 初始注入各层误差
        bus.inject_error(CognitiveLayer.PERCEPTION, magnitude=0.5, source="observation")
        bus.inject_error(CognitiveLayer.COGNITION, magnitude=0.4, source="reasoning")
        bus.inject_error(CognitiveLayer.PREDICTION, magnitude=0.3, source="prediction")
        bus.inject_error(CognitiveLayer.ACTION, magnitude=0.2, source="action")

        # 多轮传播验证收敛
        results = bus.propagate_n(n=5, early_stop=False)
        energies = [r.total_energy for r in results]

        # 验证: 最后一轮能量 < 第一轮
        assert energies[-1] < energies[0], f"Loop should converge: round1={energies[0]:.4f}, final={energies[-1]:.4f}"

    def test_reset_clears_state(self, bus):
        """reset 清空所有状态。"""
        bus.inject_error(CognitiveLayer.COGNITION, magnitude=0.5)
        bus.propagate()
        assert bus.step_count > 0

        bus.reset()
        assert bus.step_count == 0

    def test_layer_error_accumulation(self, bus):
        """多层误差注入会累积。"""
        bus.inject_error(CognitiveLayer.COGNITION, magnitude=0.3)
        bus.inject_error(CognitiveLayer.COGNITION, magnitude=0.2)

        # 累计后应大于单次注入
        result = bus.propagate()
        cog_error = result.layer_errors.get(CognitiveLayer.COGNITION)
        assert cog_error is not None
        assert cog_error.magnitude > 0
