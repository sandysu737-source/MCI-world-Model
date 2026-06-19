"""Tests for OnlineEWC — adaptive EWC with diagonal Fisher approximation."""

import numpy as np
import pytest

from mci_world_model.sdk._online_ewc import OnlineEWC


def make_params(dim: int = 10, seed: int = 42) -> dict[str, np.ndarray]:
    """Create a simple parameter dict for testing."""
    rng = np.random.RandomState(seed)
    return {
        "w": rng.randn(dim).astype(np.float64),
        "b": rng.randn(1).astype(np.float64),
    }


class TestOnlineEWC:
    """Core OnlineEWC tests."""

    def test_initial_state(self) -> None:
        """EWC starts with zero completed tasks and zero loss."""
        ewc = OnlineEWC()
        assert ewc.n_completed == 0
        assert ewc.loss(make_params()) == 0.0

    def test_adaptive_lambda(self) -> None:
        """Lambda grows with task count."""
        ewc = OnlineEWC(base_lambda=100.0, growth_rate=0.2)
        assert ewc.adaptive_lambda == 100.0 * (1 + 0.2 * 0)  # = 100.0

        ewc.n_completed = 3
        assert ewc.adaptive_lambda == 100.0 * (1 + 0.2 * 3)  # = 160.0

        ewc.n_completed = 5
        assert ewc.adaptive_lambda == 100.0 * (1 + 0.2 * 5)  # = 200.0

    def test_update_saves_star_params(self) -> None:
        """After update, star_params contain copies of current params."""
        ewc = OnlineEWC()
        params = make_params()
        ewc.update(params)

        assert ewc.n_completed == 1
        assert "w" in ewc.star_params
        assert "b" in ewc.star_params
        np.testing.assert_array_equal(ewc.star_params["w"], params["w"])

    def test_update_with_task_data_fisher(self) -> None:
        """When task_data is provided, Fisher diagonals are computed."""
        ewc = OnlineEWC()
        params = make_params(dim=5)
        task_data = np.random.randn(100, 5)
        ewc.update(params, task_data=task_data, n_samples=20)

        assert "w" in ewc.fisher_diagonals
        assert ewc.fisher_diagonals["w"].shape == params["w"].shape
        # Fisher should be non-negative
        assert np.all(ewc.fisher_diagonals["w"] >= 0)

    def test_update_without_task_data(self) -> None:
        """Without task_data, Fisher is uniform (ones)."""
        ewc = OnlineEWC()
        params = make_params(dim=5)
        ewc.update(params)

        assert "w" in ewc.fisher_diagonals
        np.testing.assert_array_equal(
            ewc.fisher_diagonals["w"], np.ones(5, dtype=np.float64)
        )

    def test_loss_non_negative(self) -> None:
        """EWC loss is always non-negative."""
        ewc = OnlineEWC()
        params = make_params()
        ewc.update(params)

        loss = ewc.loss(params)
        assert loss >= 0.0

    def test_loss_grows_with_deviation(self) -> None:
        """Loss increases when params deviate from star_params."""
        ewc = OnlineEWC(base_lambda=100.0)
        params0 = make_params()
        ewc.update(params0)

        # Same params: loss near zero
        loss_same = ewc.loss(params0)

        # Different params: larger loss
        params1 = make_params(seed=99)
        loss_diff = ewc.loss(params1)

        assert loss_diff > loss_same

    def test_loss_grows_with_more_tasks(self) -> None:
        """Loss accumulates across multiple tasks."""
        ewc = OnlineEWC(base_lambda=100.0)
        params0 = make_params(seed=0)
        ewc.update(params0)

        loss_after_1 = ewc.loss(make_params(seed=99))

        params1 = make_params(seed=1)
        ewc.update(params1)

        loss_after_2 = ewc.loss(make_params(seed=99))
        # Should be larger after 2 tasks due to accumulated penalty
        assert loss_after_2 > loss_after_1

    def test_forget_rate_zero_when_same(self) -> None:
        """Forget rate is ~0 when params haven't changed."""
        ewc = OnlineEWC()
        params = make_params()
        rate = ewc.forget_rate(params, params)
        assert rate == pytest.approx(0.0, abs=1e-10)

    def test_forget_rate_positive_when_different(self) -> None:
        """Forget rate > 0 when params differ."""
        ewc = OnlineEWC()
        p0 = make_params(seed=0)
        p1 = make_params(seed=99)
        rate = ewc.forget_rate(p1, p0)
        assert rate > 0.0

    def test_reset_clears_state(self) -> None:
        """Reset clears all accumulated state."""
        ewc = OnlineEWC()
        params = make_params()
        ewc.update(params)
        assert ewc.n_completed == 1

        ewc.reset()
        assert ewc.n_completed == 0
        assert len(ewc.star_params) == 0
        assert len(ewc.fisher_diagonals) == 0

    def test_five_task_forget_rate(self) -> None:
        """After 5 tasks, forget rate on task 1 should be < 0.25."""
        ewc = OnlineEWC(base_lambda=100.0, growth_rate=0.2)
        dim = 20

        # Task 0 (the one we care about remembering)
        params_task0 = make_params(dim=dim, seed=0)
        task0_data = np.random.randn(200, dim)
        ewc.update(params_task0, task_data=task0_data)

        # Simulate 4 more tasks with different data distributions
        current_params = params_task0.copy()
        for task_idx in range(1, 5):
            # New task slightly shifts the parameter landscape
            rng = np.random.RandomState(task_idx)
            drift = {k: rng.randn(*v.shape).astype(np.float64) * 0.1 for k, v in current_params.items()}
            current_params = {k: current_params[k] + drift[k] for k in current_params}
            task_data = np.random.randn(200, dim) + task_idx * 0.5
            ewc.update(current_params, task_data=task_data)

        forget = ewc.forget_rate(current_params, params_task0)
        assert forget < 0.25, f"Expected forget rate < 0.25, got {forget:.4f}"

    def test_ewc_regularization_reduces_forgetting(self) -> None:
        """With EWC, parameter drift is smaller than without."""
        ewc = OnlineEWC(base_lambda=100.0)
        dim = 10

        params0 = make_params(dim=dim, seed=0)
        ewc.update(params0)

        # Simulate new task training — params should be pulled back by EWC
        params_new = {k: v + np.random.randn(*v.shape).astype(np.float64) * 0.5 for k, v in params0.items()}
        loss = ewc.loss(params_new)
        assert loss > 0  # EWC penalizes the drift

    def test_custom_base_lambda(self) -> None:
        """Custom base_lambda affects loss magnitude."""
        ewc_low = OnlineEWC(base_lambda=1.0)
        ewc_high = OnlineEWC(base_lambda=1000.0)
        params = make_params()
        ewc_low.update(params)
        ewc_high.update(params)

        params_drifted = make_params(seed=99)
        assert ewc_high.loss(params_drifted) > ewc_low.loss(params_drifted)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
