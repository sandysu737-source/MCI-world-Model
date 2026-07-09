"""Empirical Causal Inference — 数据驱动因果效应估计。

纯 numpy 实现，零外部依赖。

支持:
    - 线性回归 ATE (协变量调整)
    - 倾向得分加权 (IPW)
    -  doubly robust 估计
    - 反事实预测
    - 方法对比报告

用法:
    from mci_world_model.sdk._empirical_causal import EmpiricalCausal

    ec = EmpiricalCausal()
    result = ec.estimate_ate(X=covariates, T=treatment, Y=outcome, Z=confounders)
    logger.info(f"ATE = {result.ate:.4f} ± {result.se:.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import logging

logger = logging.getLogger(__name__)
import numpy as np


@dataclass
class ATEEstimate:
    """ATE 估计结果。

    Attributes:
        ate: 平均处理效应
        se: 标准误
        ci_lower: 95% CI 下界
        ci_upper: 95% CI 上界
        method: 估计方法
    """

    ate: float
    se: float
    ci_lower: float
    ci_upper: float
    method: str = "linear_regression"

    @property
    def significant(self) -> bool:
        """95% CI 是否不跨零。"""
        return self.ci_lower > 0 or self.ci_upper < 0


@dataclass
class CausalEstimateReport:
    """多方法因果估计对比报告。"""

    estimates: list[ATEEstimate] = None  # type: ignore[assignment]
    naive_ate: float = 0.0  # 未调整的 ATE

    def __post_init__(self) -> None:
        if self.estimates is None:
            self.estimates = []

    def best(self) -> ATEEstimate | None:
        if not self.estimates:
            return None
        # 选 SE 最小的
        return min(self.estimates, key=lambda e: e.se)

    def summary(self) -> dict[str, Any]:
        return {
            "n_estimates": len(self.estimates),
            "naive_ate": self.naive_ate,
            "best_method": self.best().method if self.best() else "none",  # type: ignore
            "best_ate": self.best().ate if self.best() else float("nan"),  # type: ignore
            "methods": [e.method for e in self.estimates],
        }


class EmpiricalCausal:
    """经验因果推断 — 从数据中估计因果效应。

    纯 numpy, 线性模型为基础。
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.RandomState(seed)

    # ── Naive ATE (无调整) ─────────────────────────────────────────

    def naive_ate(self, T: np.ndarray, Y: np.ndarray) -> ATEEstimate:
        """朴素 ATE: E[Y|T=1] - E[Y|T=0].

        不控制任何混杂, 仅用于基线对比。
        """
        T = np.asarray(T, dtype=np.float64).ravel()
        Y = np.asarray(Y, dtype=np.float64).ravel()
        mask1 = T > 0.5
        mask0 = ~mask1
        y1, y0 = Y[mask1], Y[mask0]
        ate = float(np.mean(y1) - np.mean(y0))
        # Welch's t-test SE
        se = float(np.sqrt(np.var(y1) / len(y1) + np.var(y0) / len(y0)))
        return ATEEstimate(ate=ate, se=se, ci_lower=ate - 1.96 * se,
                           ci_upper=ate + 1.96 * se, method="naive")

    # ── 线性回归 ATE ──────────────────────────────────────────────

    def estimate_ate_linear(
        self, X: np.ndarray, T: np.ndarray, Y: np.ndarray
    ) -> ATEEstimate:
        """线性回归 ATE: Y = α + τ·T + β·X + ε。

        通过 OLS 回归控制协变量 X, τ 即为 ATE。
        """
        X = np.asarray(X, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64).reshape(-1, 1)
        Y = np.asarray(Y, dtype=np.float64).reshape(-1, 1)
        n = len(Y)

        # 构建设计矩阵 [1, T, X]
        design = np.column_stack([np.ones((n, 1)), T, X])

        # OLS: β = (D'D)⁻¹ D'Y
        try:
            DtD = design.T @ design
            DtY = design.T @ Y
            beta = np.linalg.solve(DtD, DtY)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(design, Y, rcond=None)[0]

        ate = float(beta[1, 0])  # τ

        # SE 估计
        residuals = Y - design @ beta
        sigma2 = float(np.sum(residuals**2) / (n - design.shape[1]))
        try:
            cov = sigma2 * np.linalg.inv(DtD)
            se = float(np.sqrt(cov[1, 1]))
        except np.linalg.LinAlgError:
            se = float(np.std(residuals) / np.sqrt(n))

        return ATEEstimate(ate=ate, se=se, ci_lower=ate - 1.96 * se,
                           ci_upper=ate + 1.96 * se, method="linear_regression")

    # ── 倾向得分加权 (IPW) ──────────────────────────────────────────

    def estimate_ate_ipw(
        self, X: np.ndarray, T: np.ndarray, Y: np.ndarray
    ) -> ATEEstimate:
        """IPW ATE: 用倾向得分的倒数加权。

        e(x) = P(T=1|X=x) 通过 logistic 回归估计
        ATE = E[TY/e(X) - (1-T)Y/(1-e(X))]
        """
        X = np.asarray(X, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64).ravel()
        Y = np.asarray(Y, dtype=np.float64).ravel()
        _n = len(Y)

        # 倾向得分: logistic regression via gradient descent
        propensity = self._estimate_propensity(X, T)

        # 稳定化权重
        eps = 1e-8
        w1 = T / np.clip(propensity, eps, 1 - eps)
        w0 = (1 - T) / np.clip(1 - propensity, eps, 1 - eps)

        # 标准化权重
        w1 /= np.mean(w1)
        w0 /= np.mean(w0)

        ate = float(np.mean(w1 * Y) - np.mean(w0 * Y))

        # Bootstrap SE
        se = self._bootstrap_se(X, T, Y, n_bootstrap=200)

        return ATEEstimate(ate=ate, se=se, ci_lower=ate - 1.96 * se,
                           ci_upper=ate + 1.96 * se, method="ipw")

    # ── Doubly Robust ───────────────────────────────────────────────

    def estimate_ate_doubly_robust(
        self, X: np.ndarray, T: np.ndarray, Y: np.ndarray
    ) -> ATEEstimate:
        """Doubly Robust ATE: 结合 outcome regression 和 IPW。

        只要 outcome model 或 propensity model 之一正确即可得到一致估计。
        """
        X = np.asarray(X, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64).ravel()
        Y = np.asarray(Y, dtype=np.float64).ravel()

        # Propensity
        e = self._estimate_propensity(X, T)

        # Outcome models: E[Y|T=1,X] and E[Y|T=0,X]
        mask1 = T > 0.5
        mu1 = self._fit_outcome(X, Y, mask1)
        mu0 = self._fit_outcome(X, Y, ~mask1)

        # Doubly robust formula
        eps = 1e-8
        dr = mu1 - mu0
        dr += T * (Y - mu1) / np.clip(e, eps, 1 - eps)
        dr -= (1 - T) * (Y - mu0) / np.clip(1 - e, eps, 1 - eps)

        ate = float(np.mean(dr))
        se = float(np.std(dr) / np.sqrt(len(Y)))

        return ATEEstimate(ate=ate, se=se, ci_lower=ate - 1.96 * se,
                           ci_upper=ate + 1.96 * se, method="doubly_robust")

    # ── 反事实预测 ──────────────────────────────────────────────────

    def predict_counterfactual(
        self, X: np.ndarray, T: np.ndarray, Y: np.ndarray,
        T_counterfactual: np.ndarray,
    ) -> np.ndarray:
        """预测反事实结果: E[Y|do(T=t')]。

        用线性回归拟合 Y ~ T + X, 然后代入 T_counterfactual。
        """
        X = np.asarray(X, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64).reshape(-1, 1)
        Y = np.asarray(Y, dtype=np.float64).reshape(-1, 1)
        n = len(Y)
        design = np.column_stack([np.ones((n, 1)), T, X])
        try:
            DtD = design.T @ design
            beta = np.linalg.solve(DtD, design.T @ Y)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(design, Y, rcond=None)[0]

        # Predict under counterfactual T
        T_cf = np.asarray(T_counterfactual, dtype=np.float64).reshape(-1, 1)
        cf_design = np.column_stack([np.ones((len(T_cf), 1)), T_cf, X])
        return (cf_design @ beta).ravel()

    # ── 完整对比报告 ────────────────────────────────────────────────

    def compare_all(
        self, X: np.ndarray, T: np.ndarray, Y: np.ndarray
    ) -> CausalEstimateReport:
        """运行所有估计方法并生成对比报告。"""
        report = CausalEstimateReport()
        report.naive_ate = self.naive_ate(T, Y).ate

        methods = [
            ("linear_regression", self.estimate_ate_linear),
            ("ipw", self.estimate_ate_ipw),
            ("doubly_robust", self.estimate_ate_doubly_robust),
        ]

        for name, method in methods:
            try:
                result = method(X, T, Y)
                report.estimates.append(result)
            except Exception as e:
                logger.warning("吞异常", exc_info=True)
        return report

    # ── 内部方法 ─────────────────────────────────────────────────────

    def _estimate_propensity(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        """用 logistic 回归估计倾向得分 e(x) = P(T=1|X=x)。"""
        X = np.asarray(X, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64).ravel()
        _n, d = X.shape

        # 标准化 X
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8
        X_norm = (X - X_mean) / X_std

        # SGD for logistic regression
        w = np.zeros(d + 1)
        lr = 0.1
        for _ in range(500):
            logits = w[0] + X_norm @ w[1:]
            p = 1.0 / (1.0 + np.exp(-np.clip(logits, -10, 10)))
            grad0 = np.mean(p - T)
            grad = np.mean((p - T)[:, None] * X_norm, axis=0)
            w[0] -= lr * grad0
            w[1:] -= lr * grad

        logits = w[0] + X_norm @ w[1:]
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -10, 10)))

    def _fit_outcome(
        self, X: np.ndarray, Y: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Fit E[Y|X] on the subset defined by mask."""
        X_sub = X[mask]
        Y_sub = Y[mask].reshape(-1, 1)
        n = len(Y_sub)
        if n < 3:
            return np.full(len(Y), float(np.mean(Y_sub))) if n > 0 else np.zeros(len(Y))

        design = np.column_stack([np.ones((n, 1)), X_sub])
        try:
            DtD = design.T @ design
            beta = np.linalg.solve(DtD, design.T @ Y_sub)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(design, Y_sub, rcond=None)[0]

        full_design = np.column_stack([np.ones((len(X), 1)), X])
        return (full_design @ beta).ravel()

    def _bootstrap_se(
        self, X: np.ndarray, T: np.ndarray, Y: np.ndarray, n_bootstrap: int = 200
    ) -> float:
        """Bootstrap 标准误。"""
        n = len(Y)
        ates = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            idx = self._rng.choice(n, n, replace=True)
            try:
                est = self.estimate_ate_linear(X[idx], T[idx], Y[idx])
                ates[b] = est.ate
            except Exception:
                logger.warning("异常降级", exc_info=True)
                ates[b] = np.nan
        valid = ates[~np.isnan(ates)]
        return float(np.std(valid)) if len(valid) > 10 else 0.1
