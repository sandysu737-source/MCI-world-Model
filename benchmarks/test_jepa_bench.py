"""
MCI World Model — JEPA 训练/预测性能基准测试

覆盖:
- T1: JEPATrainer 训练循环 (Identity/BeliefPropagation 基线)
- T2: JEPAPredictor 预测延迟对比 (3 种基线)
- T3: JEPAEncoder 编码延迟 (不同数据规模)
- T4: JEPA evaluate 评估延迟

运行: pytest benchmarks/test_jepa_bench.py -m benchmark --benchmark-only
"""

from __future__ import annotations

import numpy as np
import pytest


# =============================================================================
# 共享 fixtures
# =============================================================================


@pytest.fixture(scope="module")
def jepa_encoder():
    """JEPAEncoder (非可微模式)。"""
    from mci_world_model.sdk._jepa_encoder import JEPAEncoder

    return JEPAEncoder(world_model=None)


@pytest.fixture(scope="module")
def encoded_state_30d(jepa_encoder, patient_timeline_30d):
    """30 天编码后的 CausalWorldModelState。"""
    return jepa_encoder.encode(signals=patient_timeline_30d)


@pytest.fixture(scope="module")
def training_dataset(jepa_encoder, patient_timeline_30d):
    """构造 JEPA 训练对 (s_t, s_{t+1})。"""
    from mci_world_model.sdk._jepa_dataset import JEPADataset

    ds = JEPADataset()
    # 滑动窗口构造训练对
    for i in range(len(patient_timeline_30d) - 5):
        window_t = patient_timeline_30d[i : i + 5]
        window_next = patient_timeline_30d[i + 1 : i + 6]
        s_t = jepa_encoder.encode(signals=window_t)
        s_next = jepa_encoder.encode(signals=window_next)
        ds.add_pair(s_t, s_next)
    return ds


# =============================================================================
# T1: JEPATrainer 训练循环
# =============================================================================


@pytest.mark.benchmark(min_rounds=3, max_time=10.0)
class TestJEPATrainingLatency:
    """JEPATrainer.train() 训练循环性能基线。"""

    def test_train_identity_5_epochs(self, benchmark, jepa_encoder, training_dataset):
        """IdentityPredictor 5 轮训练（基线下界）。"""
        from mci_world_model.sdk._jepa_predictor import IdentityPredictor
        from mci_world_model.sdk._jepa_trainer import JEPATrainer

        trainer = JEPATrainer(jepa_encoder, IdentityPredictor(), training_dataset)

        stats = benchmark(trainer.train, n_epochs=5)
        assert stats.n_epochs == 5

    def test_train_belief_prop_5_epochs(self, benchmark, jepa_encoder, training_dataset):
        """BeliefPropagationPredictor 5 轮训练。"""
        from mci_world_model.sdk._jepa_predictor import BeliefPropagationPredictor
        from mci_world_model.sdk._jepa_trainer import JEPATrainer

        trainer = JEPATrainer(jepa_encoder, BeliefPropagationPredictor(), training_dataset)

        stats = benchmark(trainer.train, n_epochs=5)
        assert stats.n_epochs == 5

    def test_train_energy_prop_5_epochs(self, benchmark, jepa_encoder, training_dataset):
        """EnergyPropagationPredictor 5 轮训练。"""
        from mci_world_model.sdk._jepa_predictor import EnergyPropagationPredictor
        from mci_world_model.sdk._jepa_trainer import JEPATrainer

        trainer = JEPATrainer(jepa_encoder, EnergyPropagationPredictor(), training_dataset)

        stats = benchmark(trainer.train, n_epochs=5)
        assert stats.n_epochs == 5


# =============================================================================
# T2: JEPAPredictor 预测延迟对比
# =============================================================================


@pytest.mark.benchmark(min_rounds=10, max_time=2.0)
class TestJEPAPredictorLatency:
    """三种基线预测器 predict() 延迟对比。"""

    def test_identity_predict(self, benchmark, encoded_state_30d):
        """IdentityPredictor 单次预测。"""
        from mci_world_model.sdk._jepa_predictor import IdentityPredictor

        predictor = IdentityPredictor()
        result = benchmark(predictor.predict, encoded_state_30d)
        assert result is not None

    def test_energy_propagation_predict(self, benchmark, encoded_state_30d):
        """EnergyPropagationPredictor 单次预测。"""
        from mci_world_model.sdk._jepa_predictor import EnergyPropagationPredictor

        predictor = EnergyPropagationPredictor()
        result = benchmark(predictor.predict, encoded_state_30d)
        assert result is not None

    def test_belief_propagation_predict(self, benchmark, encoded_state_30d):
        """BeliefPropagationPredictor 单次预测。"""
        from mci_world_model.sdk._jepa_predictor import BeliefPropagationPredictor

        predictor = BeliefPropagationPredictor()
        result = benchmark(predictor.predict, encoded_state_30d)
        assert result is not None


# =============================================================================
# T3: JEPAEncoder 编码延迟 (不同规模)
# =============================================================================


@pytest.mark.benchmark(min_rounds=5, max_time=2.0)
class TestJEPAEncoderScaling:
    """JEPAEncoder.encode() 不同数据规模编码延迟。"""

    def test_encode_7d_window(self, benchmark, jepa_encoder, patient_timeline_30d):
        """7 天窗口编码。"""
        window = patient_timeline_30d[:7]
        result = benchmark(jepa_encoder.encode, signals=window)
        assert result is not None

    def test_encode_14d_window(self, benchmark, jepa_encoder, patient_timeline_30d):
        """14 天窗口编码。"""
        window = patient_timeline_30d[:14]
        result = benchmark(jepa_encoder.encode, signals=window)
        assert result is not None

    def test_encode_30d_full(self, benchmark, jepa_encoder, patient_timeline_30d):
        """30 天全量编码。"""
        result = benchmark(jepa_encoder.encode, signals=patient_timeline_30d)
        assert result is not None
        assert len(result.causal_edges) > 0


# =============================================================================
# T4: JEPA evaluate 评估延迟
# =============================================================================


@pytest.mark.benchmark(min_rounds=3, max_time=5.0)
class TestJEPAEvaluateLatency:
    """JEPAPredictor.evaluate() 评估性能。"""

    @pytest.fixture(scope="class")
    def eval_pairs(self, jepa_encoder, patient_timeline_30d):
        """构造评估对。"""
        pairs = []
        for i in range(0, len(patient_timeline_30d) - 5, 2):
            s_t = jepa_encoder.encode(signals=patient_timeline_30d[i : i + 5])
            s_next = jepa_encoder.encode(signals=patient_timeline_30d[i + 1 : i + 6])
            pairs.append((s_t, s_next))
        return pairs

    def test_identity_evaluate(self, benchmark, eval_pairs):
        """IdentityPredictor.evaluate() 评估延迟。"""
        from mci_world_model.sdk._jepa_predictor import IdentityPredictor

        predictor = IdentityPredictor()
        result = benchmark(predictor.evaluate, eval_pairs)
        assert isinstance(result, dict)

    def test_belief_prop_evaluate(self, benchmark, eval_pairs):
        """BeliefPropagationPredictor.evaluate() 评估延迟。"""
        from mci_world_model.sdk._jepa_predictor import BeliefPropagationPredictor

        predictor = BeliefPropagationPredictor()
        result = benchmark(predictor.evaluate, eval_pairs)
        assert isinstance(result, dict)
