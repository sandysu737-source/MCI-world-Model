from __future__ import annotations

"""MCI World Model — Temporal Causal Inference

Pure-numpy Granger causality testing and lagged correlation analysis
for time-series causal discovery.

Reference: Granger (1969) "Investigating Causal Relations by Econometric
Models and Cross-spectral Methods"
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import f as f_dist


@dataclass
class TemporalCausalReport:
    """Unified output for temporal causal analysis."""

    method: str = ""
    causal: bool = False
    p_value: float = 1.0
    f_statistic: float = 0.0
    best_lag: int = 0
    peak_correlation: float = 0.0
    details: dict = field(default_factory=dict)


class GrangerCausality:
    """Granger causality test for bivariate time series.

    Tests whether past values of X help predict Y beyond Y's own history.

    H0: X does NOT Granger-cause Y (all gamma_i = 0)
    H1: X does Granger-cause Y (at least one gamma_i != 0)

    Reject H0 if p_value < alpha.
    """

    def __init__(self, max_lag: int = 5, alpha: float = 0.05):
        if max_lag < 1:
            raise ValueError(f"max_lag must be >= 1, got {max_lag}")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        self._max_lag = max_lag
        self._alpha = alpha

    def test(self, x: np.ndarray, y: np.ndarray) -> TemporalCausalReport:
        """Test if x Granger-causes y.

        Args:
            x: (n_samples,) time series of potential cause
            y: (n_samples,) time series of potential effect

        Returns:
            TemporalCausalReport with f_statistic, p_value, causal flag
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()

        if len(x) != len(y):
            raise ValueError(
                f"x and y must have same length, got {len(x)} and {len(y)}"
            )

        n = len(x)
        max_lag = self._max_lag

        # Build lagged matrices
        # Y_t depends on Y_{t-1}...Y_{t-max_lag} AND X_{t-1}...X_{t-max_lag}
        # Need at least max_lag + 1 samples
        T = n - max_lag
        if 2 * max_lag >= T:
            return TemporalCausalReport(
                method="granger",
                causal=False,
                p_value=1.0,
                f_statistic=0.0,
                details={"error": f"Too few samples: {n} with max_lag={max_lag}"},
            )

        # Build design matrices
        Y_lagged = np.column_stack([y[max_lag - i - 1 : n - i - 1] for i in range(max_lag)])
        X_lagged = np.column_stack([x[max_lag - i - 1 : n - i - 1] for i in range(max_lag)])
        Y_target = y[max_lag:]

        # Full model: Y ~ Y_lagged + X_lagged
        X_full = np.column_stack([np.ones(T), Y_lagged, X_lagged])
        theta_full, rss_full, _rank_full, _ = np.linalg.lstsq(X_full, Y_target, rcond=None)
        rss_full = float(rss_full[0]) if rss_full.size > 0 else float(np.sum((Y_target - X_full @ theta_full) ** 2))

        # Reduced model: Y ~ Y_lagged only
        X_reduced = np.column_stack([np.ones(T), Y_lagged])
        theta_reduced, rss_reduced, _, _ = np.linalg.lstsq(X_reduced, Y_target, rcond=None)
        rss_reduced = float(rss_reduced[0]) if rss_reduced.size > 0 else float(np.sum((Y_target - X_reduced @ theta_reduced) ** 2))

        # Degrees of freedom
        df1 = max_lag  # Number of restrictions (number of X lags)
        df2 = T - 2 * max_lag - 1  # Residual df in full model

        if df2 <= 0 or rss_full < 1e-15:
            return TemporalCausalReport(
                method="granger",
                causal=False,
                p_value=1.0,
                f_statistic=0.0,
                details={"df1": df1, "df2": df2, "warning": "Degrees of freedom exhausted"},
            )

        # F-statistic
        f_stat = ((rss_reduced - rss_full) / df1) / (rss_full / df2)
        if f_stat < 0:
            f_stat = 0.0

        p_value = float(1.0 - f_dist.cdf(f_stat, df1, df2))
        causal = p_value < self._alpha

        return TemporalCausalReport(
            method="granger",
            causal=causal,
            p_value=float(p_value),
            f_statistic=float(f_stat),
            best_lag=max_lag,
            details={
                "max_lag": max_lag,
                "df1": df1,
                "df2": df2,
                "rss_full": float(rss_full),
                "rss_reduced": float(rss_reduced),
            },
        )


class LaggedCorrelationScanner:
    """Scan cross-correlation across lags to find peak lag and direction.

    Computes correlation(X_t, Y_{t+lag}) for lag in [-max_lag, max_lag].
    Peak correlation suggests potential causal direction:
      - Positive peak lag: X leads Y (X may cause Y)
      - Negative peak lag: Y leads X (Y may cause X)
    """

    def __init__(self, max_lag: int = 10):
        if max_lag < 1:
            raise ValueError(f"max_lag must be >= 1, got {max_lag}")
        self._max_lag = max_lag

    def scan(self, x: np.ndarray, y: np.ndarray) -> TemporalCausalReport:
        """Scan cross-correlation across lags.

        Args:
            x: (n_samples,) first time series ("cause" candidate)
            y: (n_samples,) second time series ("effect" candidate)

        Returns:
            TemporalCausalReport with best_lag, peak_correlation, causal flag
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()

        if len(x) != len(y):
            raise ValueError(
                f"x and y must have same length, got {len(x)} and {len(y)}"
            )

        n = len(x)
        max_lag = self._max_lag

        best_lag = 0
        best_corr = 0.0
        all_corrs: dict[int, float] = {}

        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                # Y_{t+lag} ~ X_t
                if n - lag < 10:
                    continue
                corr = np.corrcoef(x[: n - lag], y[lag:])[0, 1]
            else:
                # X_{t+|lag|} ~ Y_t
                abs_lag = -lag
                if n - abs_lag < 10:
                    continue
                corr = np.corrcoef(y[: n - abs_lag], x[abs_lag:])[0, 1]

            if np.isfinite(corr):
                all_corrs[lag] = float(corr)
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag

        # Causal hint: if peak lag > 0 and correlation is significant
        causal = best_lag > 0 and abs(best_corr) > 0.1

        return TemporalCausalReport(
            method="lagged_correlation",
            causal=causal,
            p_value=1.0,  # Not a formal test
            f_statistic=0.0,
            best_lag=best_lag,
            peak_correlation=float(best_corr),
            details={"all_correlations": all_corrs, "max_lag_scanned": max_lag},
        )
