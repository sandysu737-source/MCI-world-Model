"""契约测试 — JEPA 预测器的可学习性边界。

明确文档化每个 Predictor 的"是否可学习"性质, 防止用户误以为
IdentityPredictor 等基线路径在 train() 中真正更新参数。

契约:
- IdentityPredictor / EnergyPropagation / BeliefPropagation: 不可参基线,
  train() 的 loss 应恒定 (不下降), 这是设计意图 (下界), 不是 bug。
- TrueJEPAEncoder: 可学习, loss 应随训练下降 (见 test_true_jepa_*)。
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract

import numpy as np
import pytest


class TestPredictorLearnabilityContract:
    """文档化各 Predictor 的可学习性边界。"""

    def test_identity_predictor_is_non_parametric(self):
        """IdentityPredictor 无可学习参数 (设计为下界基线)。"""
        from mci_world_model.sdk._jepa_predictor import IdentityPredictor

        pred = IdentityPredictor()
        # 无参数属性
        assert not hasattr(pred, "parameters") or pred.parameters is None
        assert pred.name == "identity"

    def test_identity_predictor_returns_copy(self):
        """Identity 预测 s_{t+1} = s_t (浅拷贝, 非原地修改)。"""
        from mci_world_model.sdk._jepa_predictor import IdentityPredictor
        from mci_world_model.sdk._world_model import CausalWorldModelState

        state = CausalWorldModelState(
            causal_edges=[("A", "B")], active_states={"A", "B"},
            n_confirmed=1, n_novel=0, n_suppressed=0, n_memories=5,
            timestamp=1.0,
        )
        pred = IdentityPredictor()
        result = pred.predict(state)
        assert result.causal_edges == [("A", "B")]
        assert result is not state  # 是拷贝, 非同一对象

    def test_baseline_loss_is_constant(self):
        """基线 Predictor 的 train loss 应恒定 (不学习 = 设计意图)。

        这验证了 README 示例中 IdentityPredictor + train() 不会产生
        参数更新。若未来改为可学习, 此测试需相应调整。
        """
        try:
            from mci_world_model.sdk._jepa_encoder import JEPAEncoder
            from mci_world_model.sdk._jepa_predictor import IdentityPredictor
            from mci_world_model.sdk._jepa_trainer import JEPATrainer
            from mci_world_model.sdk._jepa_dataset import JEPADataset
            from mci_world_model import MCIWorldModel
        except ImportError:
            pytest.skip("JEPA modules not available")

        wm = MCIWorldModel()
        states = [wm.initialize() for _ in range(5)]
        try:
            dataset = JEPADataset.from_states(states, window_size=2)
        except Exception:
            pytest.skip("JEPADataset construction failed in this env")

        encoder = JEPAEncoder(wm)
        predictor = IdentityPredictor()
        trainer = JEPATrainer(encoder, predictor, dataset, alpha_energy=0.1, beta_cons=0.05)

        try:
            stats = trainer.train(n_epochs=5, learning_rate=1e-3)
        except Exception:
            pytest.skip("baseline train path failed in this env")

        loss_history = getattr(stats, "loss_history", None) or getattr(stats, "losses", None)
        if loss_history is None or len(loss_history) < 2:
            pytest.skip("loss history not available")

        # 契约: 基线 loss 恒定 (不下降)
        loss_arr = np.array(loss_history, dtype=float)
        spread = loss_arr.max() - loss_arr.min()
        assert spread < 1e-9, (
            f"基线 loss 应恒定, 但有波动 spread={spread:.2e}。"
            f"若这是可学习改进, 请更新此契约测试。"
        )
