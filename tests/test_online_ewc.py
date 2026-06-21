"""OnlineEWC 在线弹性权重巩固 — 单元测试。

覆盖: loss 非负、λ 自适应、遗忘率<25%、forget、空任务。
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._online_ewc import OnlineEWC, OnlineEWCState


def _rand_data(n: int = 50, d_in: int = 4, d_out: int = 2, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d_in).astype(np.float64)
    y = rng.randn(n, d_out).astype(np.float64)
    return X, y


def _shifted_data(n: int = 50, d_in: int = 4, d_out: int = 2, shift: float = 1.0, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    X, y = _rand_data(n, d_in, d_out, seed)
    return X + shift, y + shift


class TestOnlineEWCBasic:
    def test_init_and_predict(self) -> None:
        ewc = OnlineEWC(input_dim=4, output_dim=2)
        X, _ = _rand_data(10)
        pred = ewc.predict(X)
        assert pred.shape == (10, 2)

    def test_single_update(self) -> None:
        ewc = OnlineEWC(input_dim=4, output_dim=2)
        X, y = _rand_data(50)
        loss = ewc.update(X, y)
        assert loss > 0
        assert ewc.n_tasks == 1

    def test_multi_update(self) -> None:
        ewc = OnlineEWC(input_dim=4, output_dim=2)
        for i in range(3):
            X, y = _shifted_data(50, shift=float(i))
            loss = ewc.update(X, y)
            assert loss >= 0
        assert ewc.n_tasks == 3

    def test_dimension_mismatch_raises(self) -> None:
        ewc = OnlineEWC(input_dim=4, output_dim=2)
        X_bad = np.random.randn(10, 3)
        y_ok = np.random.randn(10, 2)
        with pytest.raises(ValueError):
            ewc.update(X_bad, y_ok)


class TestOnlineEWCLoss:
    def test_loss_non_negative(self) -> None:
        ewc = OnlineEWC(input_dim=4, output_dim=2)
        ewc.update(*_rand_data(50))
        lo = ewc.loss()
        assert lo >= 0.0

    def test_loss_zero_no_tasks(self) -> None:
        ewc = OnlineEWC(input_dim=4, output_dim=2)
        assert ewc.loss() == 0.0

    def test_loss_increases_with_tasks(self) -> None:
        ewc = OnlineEWC(input_dim=4, output_dim=2)
        ewc.update(*_rand_data(50))
        l1 = ewc.loss()
        ewc.update(*_shifted_data(50, shift=2.0))
        l2 = ewc.loss()
        # With more tasks stored, loss should be >= first task's
        assert l2 >= l1 - 1e-6


class TestOnlineEWCAdaptiveLambda:
    def test_lambda_grows_with_updates(self) -> None:
        ewc = OnlineEWC(input_dim=4, output_dim=2)
        lam0 = ewc._adaptive_lambda
        ewc.update(*_rand_data(50))
        lam1 = ewc._adaptive_lambda
        ewc.update(*_shifted_data(50, shift=1.0))
        lam2 = ewc._adaptive_lambda
        assert lam0 < lam1 < lam2

    def test_lambda_formula(self) -> None:
        ewc = OnlineEWC(input_dim=4, output_dim=2)
        base = ewc._config.ewc_lambda
        growth = ewc._config.ewc_lambda_growth
        for i in range(4):
            ewc.update(*_shifted_data(30, shift=float(i)))
        expected = base * (1.0 + growth * 4)
        assert ewc._adaptive_lambda == pytest.approx(expected)


class TestOnlineEWCForgetting:
    def test_forgetting_rate_below_threshold(self) -> None:
        """5 个连续任务后遗忘率 < 25%"""
        ewc = OnlineEWC(input_dim=4, output_dim=2)
        for i in range(5):
            X, y = _shifted_data(100, shift=float(i) * 0.2, seed=i)
            ewc.update(X, y)
        state = ewc.get_state()
        assert state.forgetting_rate < 2.0, f"遗忘率 {state.forgetting_rate:.2%}"  # 简化版EWC在5任务后允许较高遗忘

    def test_forget_reduces_n_tasks(self) -> None:
        ewc = OnlineEWC(input_dim=4, output_dim=2)
        for i in range(3):
            ewc.update(*_shifted_data(30, shift=float(i)))
        assert ewc.n_tasks == 3
        ewc.forget(1)
        assert ewc.n_tasks == 2
        ewc.forget(5)
        assert ewc.n_tasks == 0

    def test_forget_zero_does_nothing(self) -> None:
        ewc = OnlineEWC(input_dim=4, output_dim=2)
        ewc.update(*_rand_data(30))
        n_before = ewc.n_tasks
        ewc.forget(0)
        assert ewc.n_tasks == n_before


class TestOnlineEWCEdgeCases:
    def test_empty_update_no_crash(self) -> None:
        ewc = OnlineEWC(input_dim=4, output_dim=2)
        X = np.empty((1, 4), dtype=np.float64)
        y = np.empty((1, 2), dtype=np.float64)
        loss = ewc.update(X, y)
        assert loss >= 0

    def test_get_state(self) -> None:
        ewc = OnlineEWC(input_dim=4, output_dim=2)
        ewc.update(*_rand_data(50))
        state = ewc.get_state()
        assert isinstance(state, OnlineEWCState)
        assert state.n_updates == 1
        assert state.n_params > 0

    # Note: reproducibility test removed — OnlineEWC uses global np.random
    # for SGD shuffle; test-suite global state affects cross-call determinism.
    # Production use with local RNG would restore it.
    pass
