"""
MCI World Model v3.2.0 — ActionConditionedPredictor 单元测试
=============================================================

验证目标:
1. ActionConditionedPredictor 抽象基类契约
2. PendulumPhysicsPredictor 物理公式预测精度（金标准）
3. PendulumJEPAPredictor 学习型预测器收敛能力
"""

import numpy as np
import pytest

from mci_world_model.sdk._action_conditioned_predictor import (
    ActionConditionedPredictor,
    PendulumJEPAPredictor,
    PendulumPhysicsPredictor,
)
from mci_world_model.sdk._world_state import PendulumAction, PendulumState

# =============================================================================
# ActionConditionedPredictor 抽象基类
# =============================================================================


class TestActionConditionedPredictorInterface:
    """验证 ActionConditionedPredictor 抽象基类定义正确。"""

    def test_cannot_instantiate_directly(self):
        """不能直接实例化抽象类。"""
        with pytest.raises(TypeError):
            ActionConditionedPredictor()  # type: ignore

    def test_physics_is_subclass(self):
        """PendulumPhysicsPredictor 是 ActionConditionedPredictor 子类。"""
        p = PendulumPhysicsPredictor()
        assert isinstance(p, ActionConditionedPredictor)

    def test_jepa_is_subclass(self):
        """PendulumJEPAPredictor 是 ActionConditionedPredictor 子类。"""
        p = PendulumJEPAPredictor()
        assert isinstance(p, ActionConditionedPredictor)


# =============================================================================
# PendulumPhysicsPredictor — 金标准测试
# =============================================================================


class TestPendulumPhysicsPredictor:
    """物理公式预测器——MCI 世界模型预测环的金标准。

    所有测试必须零误差通过。任何学习型预测器必须逼近此精度。
    """

    def test_predict_one_step_no_action(self):
        """单步预测（无动作）等于自由演化。"""
        p = PendulumState(theta=0.5, omega=0.0)
        pred = PendulumPhysicsPredictor()
        trajectory = pred.predict(p, action=None, n_steps=1)

        free = p.step_physics()
        assert trajectory[0].distance(free) < 1e-10

    def test_predict_one_step_with_push(self):
        """单步预测（有推力）等于施加动作。"""
        p = PendulumState(theta=0.3, omega=0.1)
        push = PendulumAction(torque=3.0)
        pred = PendulumPhysicsPredictor()

        trajectory = pred.predict(p, action=push, n_steps=1)
        gt = push.apply(p)
        assert trajectory[0].distance(gt) < 1e-10

    def test_predict_multi_step_trajectory(self):
        """多步预测轨迹长度正确。"""
        p = PendulumState(theta=0.5, omega=0.0, dt=0.01)
        push = PendulumAction(torque=2.0)
        pred = PendulumPhysicsPredictor()

        trajectory = pred.predict(p, action=push, n_steps=10)
        assert len(trajectory) == 10

    def test_predict_multi_step_accuracy(self):
        """多步预测每一步都与逐步物理公式一致。"""
        p = PendulumState(theta=0.5, omega=0.0, dt=0.01)
        push = PendulumAction(torque=2.0)
        pred = PendulumPhysicsPredictor()

        trajectory = pred.predict(p, action=push, n_steps=10)

        # 逐步验证
        current = p.copy()
        for i, pred_state in enumerate(trajectory):
            current = push.apply(current) if push else current.step_physics()
            assert pred_state.distance(current) < 1e-10, f"步 {i} 误差过大"

    def test_predict_zero_steps(self):
        """零步预测返回空列表。"""
        p = PendulumState(theta=0.5, omega=0.0)
        pred = PendulumPhysicsPredictor()
        trajectory = pred.predict(p, action=None, n_steps=0)
        assert trajectory == []

    def test_rollout_action_sequence(self):
        """动作序列推演。"""
        p = PendulumState(theta=0.0, omega=0.0, dt=0.01)
        pred = PendulumPhysicsPredictor()

        actions = [
            PendulumAction(torque=5.0, dt=0.01),
            PendulumAction(torque=0.0, dt=0.01),
            PendulumAction(torque=-3.0, dt=0.01),
        ]
        trajectory = pred.rollout(p, actions)
        assert len(trajectory) == 3

    def test_evaluate_perfect_accuracy(self):
        """物理公式预测器评估自身应完美。"""
        p1 = PendulumState(theta=0.5, omega=0.0)
        push = PendulumAction(torque=2.0)
        gt = push.apply(p1)

        pred = PendulumPhysicsPredictor()
        result = pred.evaluate([(p1, push, gt)])
        assert result["avg_distance"] < 1e-10
        assert result["n"] == 1

    def test_non_pendulum_raises_error(self):
        """非 PendulumState 抛出 TypeError。"""
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

        pred = PendulumPhysicsPredictor()
        with pytest.raises(TypeError, match="PendulumPhysics"):
            pred.predict(FakeState(), action=None)


# =============================================================================
# PendulumJEPAPredictor — 学习型预测器测试
# =============================================================================


class TestPendulumJEPAPredictorInit:
    """JEPA 预测器初始化测试。"""

    def test_default_init(self):
        """默认参数初始化不崩溃。"""
        jepa = PendulumJEPAPredictor()
        assert jepa.name == "pendulum_jepa"
        assert jepa.is_trained is False

    def test_untrained_predict_does_not_crash(self):
        """未训练的预测器调用 predict 不崩溃（返回随机预测）。"""
        jepa = PendulumJEPAPredictor()
        p = PendulumState(theta=0.5, omega=0.0)
        push = PendulumAction(torque=2.0)
        trajectory = jepa.predict(p, action=push, n_steps=3)
        assert len(trajectory) == 3
        for state in trajectory:
            assert isinstance(state, PendulumState)


class TestPendulumJEPAPredictorTrain:
    """JEPA 学习型预测器训练测试。"""

    def test_train_converges(self):
        """最小二乘训练应收敛（final_loss < 0.01）。

        线性模型 + 无噪声数据 = 几乎零误差收敛。
        """
        jepa = PendulumJEPAPredictor(seed=42)
        report = jepa.train(n_samples=2000, noise_std=0.0)
        assert isinstance(report, dict)
        assert "final_loss" in report
        assert report["final_loss"] < 0.01, f"loss={report['final_loss']} too high"
        assert report["converged"] is True
        assert jepa.is_trained is True

    def test_training_improves_accuracy(self):
        """训练后预测精度显著优于初始随机预测。"""
        jepa = PendulumJEPAPredictor(seed=42)
        physics = PendulumPhysicsPredictor()

        # 测试数据
        test_states = [
            PendulumState(theta=0.3, omega=0.1),
            PendulumState(theta=-0.5, omega=0.2),
            PendulumState(theta=0.0, omega=0.0),
        ]
        test_actions = [
            PendulumAction(torque=2.0),
            PendulumAction(torque=-1.5),
            PendulumAction(torque=0.0),
        ]

        # 训练前误差
        distances_before = []
        for s, a in zip(test_states, test_actions):
            gt = physics.predict(s, a, n_steps=1)[0]
            pred = jepa.predict(s, a, n_steps=1)[0]
            distances_before.append(pred.distance(gt))

        avg_before = np.mean(distances_before)

        # 训练
        jepa.train(n_samples=2000)

        # 训练后误差
        distances_after = []
        for s, a in zip(test_states, test_actions):
            gt = physics.predict(s, a, n_steps=1)[0]
            pred = jepa.predict(s, a, n_steps=1)[0]
            distances_after.append(pred.distance(gt))

        avg_after = np.mean(distances_after)

        # 训练后误差应显著小于训练前
        assert avg_after < avg_before, f"训练后误差 {avg_after:.4f} 未改善 (前 {avg_before:.4f})"
        # 线性模型对无噪声数据应收敛到低误差（sin非线性引入少量残差）
        assert avg_after < 0.02, f"训练后误差 {avg_after:.4f} 过高"

    def test_trained_predict_multi_step(self):
        """训练后多步预测不崩溃。"""
        jepa = PendulumJEPAPredictor(seed=42)
        jepa.train(n_samples=500)

        p = PendulumState(theta=0.5, omega=0.0)
        push = PendulumAction(torque=2.0)
        trajectory = jepa.predict(p, action=push, n_steps=5)
        assert len(trajectory) == 5


# =============================================================================
# 两种预测器对比测试
# =============================================================================


class TestPredictorComparison:
    """PhysicsPredictor vs JEPAPredictor 对比。"""

    def test_physics_is_gold_standard(self):
        """物理公式预测器应与 ground truth 零误差。"""
        physics = PendulumPhysicsPredictor()
        p = PendulumState(theta=0.5, omega=0.0)
        push = PendulumAction(torque=3.0)

        # PhysicsPredictor 的预测
        trajectory = physics.predict(p, action=push, n_steps=1)

        # 手动 ground truth
        gt = push.apply(p)

        assert trajectory[0].distance(gt) < 1e-10

    def test_jepa_name_and_repr(self):
        """名称和 repr 正确。"""
        jepa = PendulumJEPAPredictor()
        assert jepa.name == "pendulum_jepa"
        assert "PendulumJEPAPredictor" in repr(jepa)

    def test_physics_name_and_repr(self):
        """名称和 repr 正确。"""
        physics = PendulumPhysicsPredictor()
        assert physics.name == "pendulum_physics"
        assert "PendulumPhysicsPredictor" in repr(physics)
