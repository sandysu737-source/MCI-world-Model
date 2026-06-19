"""Causal benchmark evaluation metrics.

Standard metrics for causal inference evaluation:
    - ε_ATE: |ATÊ - ATE|  absolute ATE error
    - ε_PEHE: √E[(τ(x) - τ̂(x))²]  precision in heterogeneous effect estimation
    - Policy Risk: E[Y*(π̂(x))] - E[Y*(π*(x))]  regret
    - RMSE: root mean squared error on counterfactuals
    - F1-Discovery: causal graph edge detection F1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ATEBenchmarkResult:
    """ATE estimation benchmark result."""

    dataset: str
    n_samples: int
    true_ate: float
    estimated_ate: float
    abs_error: float
    method: str = "do_calculus"
    adjustment_set: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.adjustment_set is None:
            self.adjustment_set = []

    @property
    def relative_error(self) -> float:
        if abs(self.true_ate) < 1e-8:
            return abs(self.estimated_ate)
        return abs(self.abs_error / self.true_ate)


@dataclass
class CounterfactualBenchmarkResult:
    """Counterfactual prediction benchmark result."""

    dataset: str
    n_samples: int
    rmse: float
    mae: float
    r2: float
    method: str = "counterfactual_engine"


@dataclass
class DiscoveryBenchmarkResult:
    """Causal discovery benchmark result."""

    dataset: str
    n_true_edges: int
    n_predicted_edges: int
    true_positives: int
    false_positives: int
    false_negatives: int
    method: str = "causal_graph"

    @property
    def precision(self) -> float:
        if self.n_predicted_edges == 0:
            return 0.0
        return self.true_positives / self.n_predicted_edges

    @property
    def recall(self) -> float:
        if self.n_true_edges == 0:
            return 1.0
        return self.true_positives / self.n_true_edges

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r < 1e-8:
            return 0.0
        return 2 * p * r / (p + r)


@dataclass
class CausalBenchmarkReport:
    """Aggregated causal benchmark report."""

    ate_results: list[ATEBenchmarkResult] = None  # type: ignore[assignment]
    cf_results: list[CounterfactualBenchmarkResult] = None  # type: ignore[assignment]
    disc_results: list[DiscoveryBenchmarkResult] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.ate_results is None:
            self.ate_results = []
        if self.cf_results is None:
            self.cf_results = []
        if self.disc_results is None:
            self.disc_results = []

    @property
    def mean_ate_error(self) -> float:
        if not self.ate_results:
            return float("nan")
        return float(np.mean([r.abs_error for r in self.ate_results]))

    @property
    def mean_cf_rmse(self) -> float:
        if not self.cf_results:
            return float("nan")
        return float(np.mean([r.rmse for r in self.cf_results]))

    @property
    def mean_discovery_f1(self) -> float:
        if not self.disc_results:
            return float("nan")
        return float(np.mean([r.f1 for r in self.disc_results]))

    def summary(self) -> dict[str, Any]:
        return {
            "n_ate_benchmarks": len(self.ate_results),
            "n_cf_benchmarks": len(self.cf_results),
            "n_discovery_benchmarks": len(self.disc_results),
            "mean_ate_error": self.mean_ate_error,
            "mean_cf_rmse": self.mean_cf_rmse,
            "mean_discovery_f1": self.mean_discovery_f1,
        }


def compute_ate_error(true_ate: float, estimated_ate: float) -> float:
    """Absolute ATE estimation error."""
    return abs(true_ate - estimated_ate)


def compute_pehe(true_ite: np.ndarray, predicted_ite: np.ndarray) -> float:
    """Precision in Estimation of Heterogeneous Effect.

    PEHE = √(E[(τ(x) - τ̂(x))²])
    """
    return float(np.sqrt(np.mean((true_ite - predicted_ite) ** 2)))


def compute_counterfactual_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute RMSE, MAE, R² for counterfactual predictions."""
    residuals = y_true - y_pred
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-8 else 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2}
