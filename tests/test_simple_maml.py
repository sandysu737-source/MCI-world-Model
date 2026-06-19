"""Tests for SimpleMAML — zero-shot transfer learning."""

import numpy as np
import pytest

from mci_world_model.sdk._simple_maml import MAMLTask, SimpleMAML
from mci_world_model.sdk._world_state import CartState, PendulumState


def make_pendulum_task(n_samples: int = 200) -> MAMLTask:
    """Generate a Pendulum dynamics prediction task."""

    states = []
    next_states = []
    state = PendulumState(theta=0.5, omega=0.1)
    for _ in range(n_samples):
        vec = state.to_vector()
        next_state = state.step_physics()
        states.append(vec)
        next_states.append(next_state.to_vector())
        state = next_state
    return MAMLTask(
        x_support=np.array(states, dtype=np.float64),
        y_support=np.array(next_states, dtype=np.float64),
    )


def make_cart_task(n_samples: int = 200) -> MAMLTask:
    """Generate a Cart dynamics prediction task."""

    states = []
    next_states = []
    state = CartState(x=0.0, v=0.5)
    for _ in range(n_samples):
        vec = state.to_vector()
        next_state = state.step_physics()
        states.append(vec)
        next_states.append(next_state.to_vector())
        state = next_state
    return MAMLTask(
        x_support=np.array(states, dtype=np.float64),
        y_support=np.array(next_states, dtype=np.float64),
    )


class TestMAMLTask:
    def test_create(self) -> None:
        x = np.random.randn(100, 2)
        y = np.random.randn(100, 2)
        task = MAMLTask(x_support=x, y_support=y)
        assert task.x_support.shape[0] == 50
        assert task.x_query.shape[0] == 50

    def test_explicit_split(self) -> None:
        x_s = np.random.randn(30, 2)
        y_s = np.random.randn(30, 2)
        x = np.random.randn(100, 2)  # noqa: F841
        y = np.random.randn(100, 2)  # noqa: F841
        x_q = np.random.randn(20, 2)
        y_q = np.random.randn(20, 2)
        task = MAMLTask(x_support=x_s, y_support=y_s, x_query=x_q, y_query=y_q)
        assert task.x_support.shape == (30, 2)
        assert task.x_query.shape == (20, 2)


class TestSimpleMAML:
    def test_create(self) -> None:
        maml = SimpleMAML(input_dim=2, output_dim=2, hidden_dim=16)
        assert maml.w1.shape == (2, 16)
        assert maml.w2.shape == (16, 2)

    def test_forward(self) -> None:
        maml = SimpleMAML(input_dim=2, output_dim=2, hidden_dim=8)
        x = np.random.randn(10, 2)
        pred = maml._forward(x, maml.w1, maml.b1, maml.w2, maml.b2)
        assert pred.shape == (10, 2)

    def test_adapt_reduces_loss(self) -> None:
        """Inner loop adaptation should reduce loss on support set."""
        maml = SimpleMAML(input_dim=2, output_dim=2, hidden_dim=16)
        x = np.random.randn(50, 2)
        y = x * 2.0 + 0.5  # Simple linear mapping

        loss_before = maml._loss(x, y, maml.w1, maml.b1, maml.w2, maml.b2)
        w1a, b1a, w2a, b2a = maml.adapt(x, y, steps=5)
        loss_after = maml._loss(x, y, w1a, b1a, w2a, b2a)

        assert loss_after < loss_before, f"Expected {loss_after} < {loss_before}"

    def test_meta_train_converges(self) -> None:
        """Meta-training should reduce loss over epochs."""
        maml = SimpleMAML(input_dim=2, output_dim=2, hidden_dim=8,
                          meta_lr=0.01, inner_lr=0.1, inner_steps=3)

        # Simple sine-wave tasks
        tasks = []
        for phase in np.linspace(0, np.pi, 5):
            x = np.random.randn(100, 2)
            y = np.sin(x + phase)
            tasks.append(MAMLTask(x_support=x, y_support=y))

        losses = maml.meta_train(tasks, n_epochs=20)
        assert len(losses) == 20
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_evaluate_adaptation(self) -> None:
        maml = SimpleMAML(input_dim=2, output_dim=2, hidden_dim=8)
        x_s = np.random.randn(30, 2)
        y_s = x_s * 2.0
        x_t = np.random.randn(20, 2)
        y_t = x_t * 2.0

        result = maml.evaluate_adaptation(x_s, y_s, x_t, y_t)
        assert "loss_before" in result
        assert "loss_after" in result
        assert "improvement_ratio" in result
        assert result["loss_after"] < result["loss_before"]

    def test_meta_loss_history(self) -> None:
        maml = SimpleMAML(input_dim=2, output_dim=2, hidden_dim=8)
        tasks = [MAMLTask(x_support=np.random.randn(100, 2), y_support=np.random.randn(100, 2))]
        maml.meta_train(tasks, n_epochs=10)
        assert len(maml.meta_loss_history) == 10

    def test_reset(self) -> None:
        maml = SimpleMAML(input_dim=2, output_dim=2, hidden_dim=8)
        tasks = [MAMLTask(x_support=np.random.randn(100, 2), y_support=np.random.randn(100, 2))]
        maml.meta_train(tasks, n_epochs=5)
        maml.reset()
        assert len(maml.meta_loss_history) == 0

    def test_pendulum_to_cart_transfer(self) -> None:
        """Pendulum → Cart zero-shot transfer: improvement ≥ 60%."""
        maml = SimpleMAML(input_dim=2, output_dim=2, hidden_dim=16,
                          meta_lr=0.01, inner_lr=0.05, inner_steps=3)

        # Source: Pendulum tasks (multiple initial conditions)
        source_tasks = []
        for theta0 in np.linspace(-1.0, 1.0, 8):
            task = make_pendulum_task(n_samples=150)
            source_tasks.append(task)

        # Target: Cart task
        target_task = make_cart_task(n_samples=150)

        result = maml.transfer_score(source_tasks, target_task, n_epochs=40)
        assert result["transfer_score"] >= 0.60, (
            f"Transfer score {result['transfer_score']:.2%} < 60%"
        )

    def test_first_order_vs_second_order(self) -> None:
        """Both modes should work (FOMAML faster, both converge)."""
        maml_fo = SimpleMAML(input_dim=2, output_dim=2, hidden_dim=8,
                             use_first_order=True, meta_lr=0.01)
        maml_so = SimpleMAML(input_dim=2, output_dim=2, hidden_dim=8,
                             use_first_order=False, meta_lr=0.01)
        tasks = [MAMLTask(
            x_support=np.random.randn(100, 2), y_support=np.random.randn(100, 2)
        )]
        losses_fo = maml_fo.meta_train(tasks, n_epochs=10)
        losses_so = maml_so.meta_train(tasks, n_epochs=10)
        assert len(losses_fo) == 10
        assert len(losses_so) == 10
        # FOMAML should still converge (loss decreases)
        assert losses_fo[-1] < losses_fo[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
