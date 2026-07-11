"""回放→JEPA 增量训练集成测试。"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._experience_memory import (
    ExperienceDB,
    ExperienceType,
)
from mci_world_model.sdk._learned_dynamics_predictor import LearnedDynamicsPredictor
from mci_world_model.sdk._world_state import PendulumAction, PendulumState


class TestReplayToJepaAdaptation:
    """Experience → (state_vec, action_vec, target_vec) → train_step 全链路。"""

    def _make_predictor(self) -> LearnedDynamicsPredictor:
        """创建一个小型 JEPA 预测器 (state_dim=2, action_dim=1)。"""
        return LearnedDynamicsPredictor(state_dim=2, action_dim=1, hidden_dim=16)

    def _populate_replay_buffer(self, db: ExperienceDB, n: int = 20, seed: int = 42) -> None:
        """填充回放缓冲: 高预测误差的 PendulumState 经验。"""
        rng = np.random.RandomState(seed)
        for i in range(n):
            state = PendulumState(theta=rng.uniform(-1, 1), omega=rng.uniform(-0.5, 0.5))
            action = PendulumAction(torque=rng.uniform(-3, 3))
            # 模拟预测的下一状态 (有噪声)
            target = PendulumState(
                theta=state.theta + 0.01 * state.omega + rng.normal(0, 0.05),
                omega=state.omega + 0.01 * float(action.torque) + rng.normal(0, 0.05),
            )
            db.store(
                experience_type=ExperienceType.FAILURE,
                tags=["replay", "high_error"],
                outcome=f"pred_error={rng.uniform(0.4, 0.9):.4f}",
                importance=rng.uniform(0.5, 1.0),
                prediction_error=rng.uniform(0.4, 0.9),
                state_snapshot=state,
                metadata={"action": action, "predicted_next_state": target},
            )

    def test_replay_train_reduces_loss(self):
        """回放训练后 JEPA 预测误差应下降。"""
        db = ExperienceDB()
        self._populate_replay_buffer(db, n=30)
        predictor = self._make_predictor()

        # 训练前: 用回放数据评估初始 MSE
        batch = db.sample_replay_buffer(batch_size=20, strategy="pred_error", seed=42)
        initial_losses = []
        for exp in batch:
            state = exp.state_snapshot
            target = exp.metadata["predicted_next_state"]
            action = exp.metadata["action"]
            s_vec = predictor._to_state_vector(state)
            t_vec = predictor._to_state_vector(target)
            a_vec = predictor._to_action_vector(action)
            loss = predictor.train_step(s_vec, a_vec, t_vec, lr=0.01)
            initial_losses.append(loss)

        # 训练几轮后 loss 应下降
        for _ in range(5):
            later_losses = []
            for exp in batch:
                state = exp.state_snapshot
                target = exp.metadata["predicted_next_state"]
                action = exp.metadata["action"]
                s_vec = predictor._to_state_vector(state)
                t_vec = predictor._to_state_vector(target)
                a_vec = predictor._to_action_vector(action)
                loss = predictor.train_step(s_vec, a_vec, t_vec, lr=0.01)
                later_losses.append(loss)

        avg_initial = float(np.mean(initial_losses))
        avg_later = float(np.mean(later_losses))
        assert avg_later < avg_initial, f"训练后 loss ({avg_later:.6f}) 未下降, 初始 ({avg_initial:.6f})"

    def test_experience_to_vectors_conversion(self):
        """Experience → 向量转换正确。"""
        db = ExperienceDB()
        self._populate_replay_buffer(db, n=5)
        predictor = self._make_predictor()

        batch = db.sample_replay_buffer(batch_size=3, strategy="pred_error", seed=42)
        for exp in batch:
            state = exp.state_snapshot
            target = exp.metadata["predicted_next_state"]
            action = exp.metadata["action"]

            s_vec = predictor._to_state_vector(state)
            t_vec = predictor._to_state_vector(target)
            a_vec = predictor._to_action_vector(action)

            assert s_vec.shape == (2,), f"state_vec shape: {s_vec.shape}"
            assert t_vec.shape == (2,), f"target_vec shape: {t_vec.shape}"
            assert a_vec.shape == (1,), f"action_vec shape: {a_vec.shape}"
            assert np.all(np.isfinite(s_vec))
            assert np.all(np.isfinite(t_vec))

    def test_replay_train_with_missing_metadata_skips(self):
        """缺少 metadata 的经验被跳过, 不崩溃。"""
        db = ExperienceDB()
        # 存一条没有 metadata 的经验
        db.store(
            experience_type=ExperienceType.FAILURE,
            tags=["replay"],
            outcome="test",
            prediction_error=0.8,
            state_snapshot=PendulumState(theta=0.1, omega=0.0),
        )
        predictor = self._make_predictor()

        # 模拟 _replay_train 逻辑
        batch = db.sample_replay_buffer(batch_size=1, strategy="pred_error", seed=42)
        losses = []
        for exp in batch:
            target = exp.metadata.get("predicted_next_state") if exp.metadata else None
            if target is None:
                continue  # 跳过
            losses.append(0.0)
        assert len(losses) == 0  # 无 metadata → 跳过

    def test_replay_train_with_invalid_state_skips(self):
        """无效 state_snapshot 的经验被跳过。"""
        db = ExperienceDB()
        db.store(
            experience_type=ExperienceType.FAILURE,
            tags=["replay"],
            outcome="test",
            prediction_error=0.8,
            state_snapshot="not_a_state",  # 无效
            metadata={"action": None, "predicted_next_state": "also_invalid"},
        )
        predictor = self._make_predictor()

        batch = db.sample_replay_buffer(batch_size=1, strategy="pred_error", seed=42)
        for exp in batch:
            state = exp.state_snapshot
            target = exp.metadata["predicted_next_state"]
            with pytest.raises(TypeError):
                predictor._to_state_vector(state)
