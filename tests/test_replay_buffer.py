"""P2: 回放采样测试 — Adapt-EPA 借鉴。"""

from __future__ import annotations

import random

from mci_world_model.sdk._experience_memory import ExperienceDB, ExperienceType


class TestReplayBufferSampling:
    """sample_replay_buffer 策略测试。"""

    def _populate_db(self, n: int = 50, seed: int = 42) -> ExperienceDB:
        """填充测试数据: 高/低预测误差经验混合。"""
        rng = random.Random(seed)
        db = ExperienceDB()
        for i in range(n):
            # 30% 高误差, 70% 低误差
            pred_error = rng.uniform(0.5, 0.9) if i % 3 == 0 else rng.uniform(0.01, 0.1)
            db.store(
                experience_id=f"exp_{i:03d}",
                experience_type=ExperienceType.SUCCESS,
                tags=[f"tag_{i}"],
                causal_edges=[("A", "B")],
                outcome=f"case_{i}",
                importance=0.5,
                prediction_error=pred_error,
            )
        return db

    def test_pred_error_prioritizes_high_error(self):
        """pred_error 策略: 采样中高误差经验占比 > 均匀采样。"""
        db = self._populate_db(60)
        batch = db.sample_replay_buffer(batch_size=20, strategy="pred_error", seed=42)
        assert len(batch) == 20

        # 高误差经验 (pred_error > 0.4) 占比
        high_error_count = sum(1 for e in batch if e.prediction_error and e.prediction_error > 0.4)
        high_error_ratio = high_error_count / len(batch)

        # 均匀采样作为对照
        uniform_batch = db.sample_replay_buffer(batch_size=20, strategy="uniform", seed=42)
        uniform_high = sum(1 for e in uniform_batch if e.prediction_error and e.prediction_error > 0.4)
        uniform_ratio = uniform_high / len(uniform_batch)

        assert high_error_ratio > uniform_ratio, (
            f"pred_error 策略高误差占比 {high_error_ratio:.2%} 应 > 均匀 {uniform_ratio:.2%}"
        )

    def test_uniform_sampling(self):
        """uniform 策略: 随机采样。"""
        db = self._populate_db(30)
        batch = db.sample_replay_buffer(batch_size=10, strategy="uniform", seed=123)
        assert len(batch) == 10
        # 两次不同 seed 采样结果不同
        batch2 = db.sample_replay_buffer(batch_size=10, strategy="uniform", seed=999)
        ids1 = {e.experience_id for e in batch}
        ids2 = {e.experience_id for e in batch2}
        assert ids1 != ids2  # 不同 seed → 不同结果

    def test_recent_sampling(self):
        """recent 策略: 最近优先。"""
        db = ExperienceDB()
        for i in range(10):
            db.store(
                experience_id=f"r_{i}",
                tags=[],
                causal_edges=[],
                outcome="",
                importance=0.5,
                prediction_error=0.1,
            )
        batch = db.sample_replay_buffer(batch_size=3, strategy="recent")
        assert len(batch) == 3
        # 应该是最新的 3 条 (存储顺序的逆序)
        timestamps = [e.timestamp for e in batch]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_empty_db_returns_empty(self):
        """空库返回空列表。"""
        db = ExperienceDB()
        assert db.sample_replay_buffer(batch_size=10) == []

    def test_batch_size_exceeds_db(self):
        """batch_size > 库大小时返回全部。"""
        db = ExperienceDB()
        db.store(tags=[], causal_edges=[], outcome="", prediction_error=0.5)
        batch = db.sample_replay_buffer(batch_size=100, strategy="pred_error", seed=42)
        assert len(batch) == 1

    def test_no_prediction_error_falls_back(self):
        """无 prediction_error 的经验用基线权重。"""
        db = ExperienceDB()
        for i in range(20):
            db.store(
                experience_id=f"ne_{i}",
                tags=[],
                causal_edges=[],
                outcome="",
                importance=0.5,
                prediction_error=None,  # 无误差
            )
        batch = db.sample_replay_buffer(batch_size=5, strategy="pred_error", seed=42)
        assert len(batch) == 5  # 不崩溃, 均匀回退

    def test_reproducible_with_seed(self):
        """相同 seed 产生相同结果。"""
        db = self._populate_db(50)
        b1 = db.sample_replay_buffer(batch_size=10, strategy="pred_error", seed=42)
        b2 = db.sample_replay_buffer(batch_size=10, strategy="pred_error", seed=42)
        assert [e.experience_id for e in b1] == [e.experience_id for e in b2]


class TestCewmReplayIntegration:
    """CEWM 回放集成测试。"""

    def test_replay_disabled_by_default(self):
        """默认关闭回放。"""
        from mci_world_model.sdk._world_model import MCIWorldModel

        wm = MCIWorldModel()
        assert wm._replay_enabled is False
        assert wm._step_count == 0

    def test_replay_enabled_via_config(self):
        """通过 config 开启回放。"""
        from mci_world_model.sdk._world_model import MCIWorldModel

        wm = MCIWorldModel(
            config={
                "replay_enabled": True,
                "replay_threshold": 0.2,
                "replay_interval": 10,
            }
        )
        assert wm._replay_enabled is True
        assert wm._replay_threshold == 0.2
        assert wm._replay_interval == 10

    def test_store_replay_no_db_no_crash(self):
        """无 ExperienceDB 时 _store_replay_experience 不崩溃。"""
        from mci_world_model.sdk._world_model import MCIWorldModel

        wm = MCIWorldModel(config={"replay_enabled": True})
        wm._store_replay_experience(None, 0.5)  # 不应抛异常

    def test_replay_train_no_jepa_no_crash(self):
        """无 JEPA 预测器时 _replay_train 不崩溃。"""
        from mci_world_model.sdk._world_model import MCIWorldModel

        wm = MCIWorldModel(config={"replay_enabled": True})
        wm._replay_train()  # 不应抛异常
