from __future__ import annotations

"""MCI World Model — Meta-learners for Heterogeneous Treatment Effects

Pure-numpy T-learner and S-learner for estimating CATE/ITE without
external ML dependencies.

Reference: Kunzel et al. (2019) "Metalearners for estimating heterogeneous
treatment effects using machine learning"
"""


import numpy as np


class TLearner:
    """T-learner: two separate outcome models per treatment group.

    For binary treatment T in {0,1}:
      mu_0(x) = E[Y | X=x, T=0]
      mu_1(x) = E[Y | X=x, T=1]
      CATE(x) = mu_1(x) - mu_0(x)
      ATE = mean(CATE(x_i))
    """

    def __init__(self) -> None:
        self._coef_0: np.ndarray | None = None
        self._coef_1: np.ndarray | None = None
        self._intercept_0: float = 0.0
        self._intercept_1: float = 0.0
        self._fitted: bool = False

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> TLearner:
        """Fit separate linear models for T=0 and T=1 groups.

        Args:
            X: (n_samples, n_features) covariate matrix
            T: (n_samples,) binary treatment indicator {0, 1}
            Y: (n_samples,) observed outcome

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        T = np.asarray(T, dtype=np.int64)
        Y = np.asarray(Y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        mask_0 = T == 0
        mask_1 = T == 1

        n_0, n_1 = int(np.sum(mask_0)), int(np.sum(mask_1))
        if n_0 < 2 or n_1 < 2:
            raise ValueError(
                f"Each treatment group needs >=2 samples, got {n_0} (T=0) and {n_1} (T=1)"
            )

        X_aug_0 = np.column_stack([np.ones(n_0), X[mask_0]])
        X_aug_1 = np.column_stack([np.ones(n_1), X[mask_1]])

        theta_0, _, _, _ = np.linalg.lstsq(X_aug_0, Y[mask_0], rcond=None)
        theta_1, _, _, _ = np.linalg.lstsq(X_aug_1, Y[mask_1], rcond=None)

        self._intercept_0 = float(theta_0[0])
        self._coef_0 = theta_0[1:].copy()
        self._intercept_1 = float(theta_1[0])
        self._coef_1 = theta_1[1:].copy()
        self._fitted = True
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """Predict CATE for each sample.

        Args:
            X: (n_samples, n_features) covariate matrix

        Returns:
            cate: (n_samples,) CATE predictions
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        mu_0 = self._intercept_0 + X @ self._coef_0
        mu_1 = self._intercept_1 + X @ self._coef_1
        return mu_1 - mu_0

    def estimate_ate(self, X: np.ndarray) -> float:
        """Estimate ATE as mean of CATE predictions."""
        return float(np.mean(self.predict_cate(X)))

    def heterogeneous_effect_summary(self, X: np.ndarray) -> dict[str, float]:
        """Return CATE distribution summary statistics.

        Returns:
            dict with keys: ate, cate_mean, cate_std, cate_q10, cate_q90
        """
        cate = self.predict_cate(X)
        return {
            "ate": float(np.mean(cate)),
            "cate_mean": float(np.mean(cate)),
            "cate_std": float(np.std(cate)),
            "cate_q10": float(np.percentile(cate, 10)),
            "cate_q90": float(np.percentile(cate, 90)),
        }

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("TLearner not fitted. Call fit() first.")


class SLearner:
    """S-learner: single outcome model with treatment as a covariate.

    For binary treatment T in {0,1}:
      mu(x, t) = E[Y | X=x, T=t]
      CATE(x) = mu(x, 1) - mu(x, 0)
      ATE = mean(CATE(x_i))

    Advantage: shares information across treatment arms.
    Disadvantage: may regularize away treatment effect if T is weak.
    """

    def __init__(self) -> None:
        self._coef: np.ndarray | None = None
        self._intercept: float = 0.0
        self._n_features: int = 0
        self._fitted: bool = False

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> SLearner:
        """Fit a single linear model with treatment as an extra feature.

        Args:
            X: (n_samples, n_features) covariate matrix
            T: (n_samples,) binary treatment indicator {0, 1}
            Y: (n_samples,) observed outcome

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._n_features = X.shape[1]
        n = X.shape[0]
        X_aug = np.column_stack([np.ones(n), X, T])

        theta, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)
        self._intercept = float(theta[0])
        self._coef = theta[1:].copy()
        self._fitted = True
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """Predict CATE as mu(x, 1) - mu(x, 0).

        Since the model is linear in T, CATE(x) = coef_T (last coefficient).
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # Treatment coefficient is the last element
        treatment_coef = float(self._coef[-1])
        return np.full(X.shape[0], treatment_coef, dtype=np.float64)

    def predict_outcome(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Predict outcome Y given covariates X and treatment T."""
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = X.shape[0]
        X_aug = np.column_stack([np.ones(n), X, T])
        return X_aug @ np.concatenate([[self._intercept], self._coef])

    def estimate_ate(self, X: np.ndarray) -> float:
        """Estimate ATE as mean of CATE (constant for linear S-learner)."""
        return float(np.mean(self.predict_cate(X)))

    def heterogeneous_effect_summary(self, X: np.ndarray) -> dict[str, float]:
        """Return CATE distribution summary.

        For linear S-learner, CATE is constant across X.
        """
        cate_val = float(self._coef[-1])
        return {
            "ate": cate_val,
            "cate_mean": cate_val,
            "cate_std": 0.0,
            "cate_q10": cate_val,
            "cate_q90": cate_val,
        }

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("SLearner not fitted. Call fit() first.")
