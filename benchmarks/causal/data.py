"""Causal benchmark data generators.

Generates standard causal inference datasets for benchmarking:
    - LinearGaussian: Y = τ·T + β·X + noise
    - NonlinearSCM: Y = f(T, X) + noise
    - BackdoorGraph: Chain/confounder/collider DAGs
    - HeterogeneousTreatment: τ(x) varies with covariates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CausalDataset:
    """Standard causal inference dataset.

    Attributes:
        treatment: (n,) binary or continuous treatment
        outcome: (n,) observed outcome
        covariates: (n, d) covariates/confounders
        true_ate: ground-truth ATE
        true_ite: (n,) individual treatment effects (if available)
        dag_edges: list of (parent, child) edges (if available)
        name: dataset identifier
    """

    treatment: np.ndarray
    outcome: np.ndarray
    covariates: np.ndarray
    true_ate: float
    true_ite: np.ndarray | None = None
    dag_edges: list[tuple[str, str]] = field(default_factory=list)
    name: str = "unknown"

    @property
    def n_samples(self) -> int:
        return len(self.treatment)

    @property
    def n_covariates(self) -> int:
        return self.covariates.shape[1]


def _rng(seed: int = 42) -> np.random.RandomState:
    return np.random.RandomState(seed)


# ── Linear Gaussian SCM ───────────────────────────────────────────────


def linear_gaussian(
    n: int = 1000,
    d: int = 10,
    ate: float = 2.0,
    seed: int = 42,
) -> CausalDataset:
    """Linear Gaussian SCM: Y = τ·T + β·X + ε.

    T ← Bernoulli(0.5)  or  T = X₁ + noise  (confounded)
    Y ← τ·T + β·X + N(0, 1)

    Parameters:
        n: number of samples
        d: covariate dimension
        ate: true average treatment effect
        seed: random seed
    """
    rng = _rng(seed)
    X = rng.randn(n, d)
    beta = rng.randn(d) * 0.5

    # Treatment assignment with confounding
    propensity = 1.0 / (1.0 + np.exp(-X[:, 0]))
    X = (rng.rand(n) < propensity).astype(np.float64)

    # Outcome
    noise = rng.randn(n) * 0.5
    treatment_col = (rng.rand(n) < propensity).astype(np.float64)
    Y = ate * treatment_col + X @ beta + noise

    return CausalDataset(
        treatment=X,
        outcome=Y,
        covariates=X,
        true_ate=ate,
        true_ite=np.full(n, ate),
        dag_edges=[("X0", "T"), ("X0", "Y"), ("T", "Y")],
        name=f"linear_gaussian_n{n}_d{d}",
    )


# ── Nonlinear SCM ─────────────────────────────────────────────────────


def nonlinear_scm(
    n: int = 1000,
    d: int = 5,
    seed: int = 42,
) -> CausalDataset:
    """Nonlinear SCM with heterogeneous treatment effects.

    T ← Bernoulli(σ(X₀ + X₁))
    τ(x) = 1 + x₀·x₁  (individual effect varies)
    Y ← τ(x)·T + sin(X₂) + exp(0.3·X₃) + noise

    ATE is computed by Monte Carlo over the population.
    """
    rng = _rng(seed)
    X = rng.randn(n, d)

    # Treatment assignment
    propensity = 1.0 / (1.0 + np.exp(-(X[:, 0] + X[:, 1])))
    X = (rng.rand(n) < propensity).astype(np.float64)

    # Heterogeneous treatment effect
    tau = 1.0 + 0.5 * X[:, 0] * X[:, 1]
    true_ate = float(np.mean(tau))

    # Outcome
    noise = rng.randn(n) * 0.3
    Y = tau * X + np.sin(X[:, 2]) + np.exp(0.3 * X[:, 3]) + noise

    return CausalDataset(
        treatment=X,
        outcome=Y,
        covariates=X,
        true_ate=true_ate,
        true_ite=tau,
        dag_edges=[("X0", "T"), ("X1", "T"), ("X0", "Y"), ("X1", "Y"),
                    ("X2", "Y"), ("X3", "Y"), ("T", "Y")],
        name=f"nonlinear_scm_n{n}_d{d}",
    )


# ── Backdoor Graph ────────────────────────────────────────────────────


def backdoor_graph(
    n: int = 1000,
    seed: int = 42,
) -> CausalDataset:
    """Classic backdoor graph: Z → T, Z → Y, T → Y.

    Z ~ N(0, 1)
    T ~ Bernoulli(σ(Z))
    Y ← T + 0.5·Z + N(0, 0.3)

    ATE of T on Y = 1.0
    Valid adjustment set: {Z}
    """
    rng = _rng(seed)
    Z = rng.randn(n, 1)
    # Binary treatment with confounding
    propensity = 1.0 / (1.0 + np.exp(-Z[:, 0]))
    X = (rng.rand(n) < propensity).astype(np.float64)
    Y = X + 0.5 * Z[:, 0] + rng.randn(n) * 0.3

    return CausalDataset(
        treatment=X,
        outcome=Y,
        covariates=Z,
        true_ate=1.0,
        true_ite=np.ones(n),
        dag_edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        name=f"backdoor_n{n}",
    )


# ── M-Bias Graph ──────────────────────────────────────────────────────


def m_bias_graph(
    n: int = 1000,
    seed: int = 42,
) -> CausalDataset:
    """M-bias graph: U1→X, U1→Z, U2→Z, U2→Y, X→Y.

    Classic example where conditioning on Z opens a backdoor path.
    Correct strategy: DO NOT adjust for Z.

    X ← U1 + noise
    Z ← U1 + U2 + noise
    Y ← X + U2 + noise
    """
    rng = _rng(seed)
    U1 = rng.randn(n)
    U2 = rng.randn(n)

    X = U1 + rng.randn(n) * 0.3
    Z = U1 + U2 + rng.randn(n) * 0.3
    Y = X + U2 + rng.randn(n) * 0.3

    return CausalDataset(
        treatment=X,
        outcome=Y,
        covariates=np.column_stack([Z]),
        true_ate=1.0,
        true_ite=np.ones(n),
        dag_edges=[("U1", "X"), ("U1", "Z"), ("U2", "Z"), ("U2", "Y"), ("X", "Y")],
        name=f"m_bias_n{n}",
    )


# ── High-dimensional confounder ───────────────────────────────────────


def high_dim_confounder(
    n: int = 500,
    d: int = 50,
    seed: int = 42,
) -> CausalDataset:
    """High-dimensional confounders: only first 5 X's are true confounders.

    T ← Bernoulli(σ(∑_{i=0}^4 X_i))
    Y ← T + ∑_{i=0}^4 X_i + N(0, 0.5)

    ATE = 1.0
    """
    rng = _rng(seed)
    X = rng.randn(n, d)

    # Only first 5 covariates affect treatment and outcome
    score = X[:, :5].sum(axis=1)
    propensity = 1.0 / (1.0 + np.exp(-score))
    X = (rng.rand(n) < propensity).astype(np.float64)

    Y = X + X[:, :5].sum(axis=1) + rng.randn(n) * 0.5

    return CausalDataset(
        treatment=X,
        outcome=Y,
        covariates=X,
        true_ate=1.0,
        true_ite=np.ones(n),
        name=f"high_dim_n{n}_d{d}",
    )


# ── Dataset registry ──────────────────────────────────────────────────

DATASET_REGISTRY: dict[str, Any] = {
    "linear_gaussian": linear_gaussian,
    "nonlinear_scm": nonlinear_scm,
    "backdoor": backdoor_graph,
    "m_bias": m_bias_graph,
    "high_dim": high_dim_confounder,
}
