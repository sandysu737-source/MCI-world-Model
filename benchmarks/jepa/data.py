"""JEPA benchmark data generators.

Generates synthetic state sequences for JEPA training and evaluation:
    - Linear dynamics: x_{t+1} = A·x_t + noise
    - Pendulum physics: θ̈ = -g/L·sin(θ) + u
    - Multi-variable coupled: multiple interacting variables
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class StateSequence:
    """Time series of world states for JEPA training.

    Attributes:
        states: (n_steps, state_dim) state sequence
        timestamps: (n_steps,) time points
        name: dataset identifier
    """

    states: np.ndarray
    timestamps: np.ndarray
    name: str = "unknown"

    @property
    def n_steps(self) -> int:
        return len(self.states)

    @property
    def state_dim(self) -> int:
        return self.states.shape[1]

    def make_pairs(self, window_size: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Generate (x_t, x_{t+window}) prediction pairs."""
        x = self.states[:-window_size]
        y = self.states[window_size:]
        return x, y


def _rng(seed: int = 42) -> np.random.RandomState:
    return np.random.RandomState(seed)


# ── Linear Dynamics ──────────────────────────────────────────────────


def linear_dynamics(
    n_steps: int = 1000,
    state_dim: int = 8,
    noise_std: float = 0.1,
    seed: int = 42,
) -> StateSequence:
    """Linear state transition: x_{t+1} = A·x_t + ε.

    A is a stable matrix with eigenvalues inside the unit circle.
    """
    rng = _rng(seed)

    # Stable transition matrix
    A = rng.randn(state_dim, state_dim) * 0.3 / np.sqrt(state_dim)
    # Ensure stability: scale down eigenvalues
    eigvals = np.linalg.eigvals(A)
    max_eig = np.max(np.abs(eigvals))
    if max_eig > 0.95:
        A *= 0.9 / max_eig

    states = np.zeros((n_steps, state_dim))
    states[0] = rng.randn(state_dim)

    for t in range(n_steps - 1):
        noise = rng.randn(state_dim) * noise_std
        states[t + 1] = states[t] @ A.T + noise

    return StateSequence(
        states=states,
        timestamps=np.arange(n_steps, dtype=np.float64),
        name=f"linear_dynamics_n{n_steps}_d{state_dim}",
    )


# ── Pendulum Physics ─────────────────────────────────────────────────


def pendulum_physics(
    n_steps: int = 500,
    dt: float = 0.05,
    seed: int = 42,
) -> StateSequence:
    """Damped pendulum simulation.

    State: [θ, ω] (angle and angular velocity)
    Dynamics: θ̈ = -(g/L)·sin(θ) - b·ω + u
    """
    rng = _rng(seed)
    g, L, b = 9.81, 1.0, 0.3

    states = np.zeros((n_steps, 2))
    theta, omega = 0.5 * np.pi, 0.0  # Start at 90 degrees

    for t in range(n_steps):
        states[t] = [theta, omega]
        u = rng.randn() * 0.1  # Small random torque
        alpha = -(g / L) * np.sin(theta) - b * omega + u
        omega += alpha * dt
        theta += omega * dt

    return StateSequence(
        states=states,
        timestamps=np.arange(n_steps, dtype=np.float64) * dt,
        name=f"pendulum_n{n_steps}",
    )


# ── Coupled Variables ────────────────────────────────────────────────


def coupled_variables(
    n_steps: int = 800,
    n_vars: int = 4,
    coupling_strength: float = 0.5,
    seed: int = 42,
) -> StateSequence:
    """Multi-variable coupled dynamics.

    Each variable x_i follows:
        x_i(t+1) = 0.7·x_i(t) + 0.3·tanh(x_{i-1}(t)) + ε
    """
    rng = _rng(seed)
    states = np.zeros((n_steps, n_vars))
    states[0] = rng.randn(n_vars)

    for t in range(n_steps - 1):
        noise = rng.randn(n_vars) * 0.1
        states[t + 1] = 0.7 * states[t]
        for i in range(n_vars):
            j = (i - 1) % n_vars  # Couple to previous variable
            states[t + 1, i] += coupling_strength * np.tanh(states[t, j])
        states[t + 1] += noise

    return StateSequence(
        states=states,
        timestamps=np.arange(n_steps, dtype=np.float64),
        name=f"coupled_n{n_steps}_v{n_vars}",
    )


# ── Dataset registry ──────────────────────────────────────────────────

DATASET_REGISTRY: dict[str, Any] = {
    "linear": linear_dynamics,
    "pendulum": pendulum_physics,
    "coupled": coupled_variables,
}
