"""MCI World Model — DifferentiableCausalInference 可微因果推断
=============================================================

将因果推断转化为可优化问题——通过梯度下降优化因果参数，
使得因果效应估计在观测数据上最优。

核心能力:
    CausalParameter      — 可优化的因果参数
    DifferentiableCausalInference — 可微因果推断引擎

设计原则:
    - 与 DoCalculus + SpectralCausal 正交
    - 纯 numpy 梯度下降 (无 autograd)
    - 可微 = 可优化 = 可学习
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CausalParameter — 可优化的因果参数
# =============================================================================


@dataclass
class CausalParameter:
    """可优化的因果参数。

    Attributes:
        name: 参数名称
        value: 当前值
        gradient: 当前梯度
        learning_rate: 学习率
    """

    name: str
    value: float = 0.0
    gradient: float = 0.0
    learning_rate: float = 0.01

    def step(self) -> float:
        """执行一步梯度下降。

        Returns:
            更新后的值
        """
        self.value -= self.learning_rate * self.gradient
        self.gradient = 0.0
        return self.value


# =============================================================================
# OptimizationResult — 优化结果
# =============================================================================


@dataclass
class OptimizationResult:
    """可微因果推断优化结果。

    Attributes:
        n_iterations: 迭代次数
        initial_loss: 初始损失
        final_loss: 最终损失
        converged: 是否收敛
        parameters: 最终参数值
    """

    n_iterations: int = 0
    initial_loss: float = 0.0
    final_loss: float = 0.0
    converged: bool = False
    parameters: dict[str, float] = field(default_factory=dict)


# =============================================================================
# DifferentiableCausalInference — 可微因果推断引擎
# =============================================================================


class DifferentiableCausalInference:
    """可微因果推断引擎 — 梯度下降优化因果参数。

    简化实现:
      - 线性因果模型: Y = β*X + γ*Z + ε
      - 目标: 最小化 ||Y_obs - Y_pred||²
      - 参数: β (treatment effect), γ (confounder effect)

    用法:
        >>> dci = DifferentiableCausalInference()
        >>> dci.set_data(treatment=X, outcome=Y, confounders=Z)
        >>> result = dci.optimize(n_iterations=100)
        >>> print(f"ATE = {result.parameters['beta']:.4f}")
    """

    def __init__(self, learning_rate: float = 0.01, convergence_threshold: float = 1e-4):
        if learning_rate <= 0:
            raise ValueError("learning_rate 必须 > 0")
        self._lr = learning_rate
        self._conv_threshold = convergence_threshold
        self._params: dict[str, CausalParameter] = {}
        self._data: dict[str, np.ndarray] = {}
        self._loss_history: list[float] = []

    def set_data(
        self,
        treatment: np.ndarray,
        outcome: np.ndarray,
        confounders: np.ndarray | None = None,
    ) -> None:
        """设置观测数据。

        Args:
            treatment: 处理变量 (n_samples,)
            outcome: 结果变量 (n_samples,)
            confounders: 混杂变量 (n_samples, n_confounders)
        """
        self._data["treatment"] = np.atleast_1d(np.asarray(treatment, dtype=float))
        self._data["outcome"] = np.atleast_1d(np.asarray(outcome, dtype=float))
        if confounders is not None:
            self._data["confounders"] = np.atleast_2d(np.asarray(confounders, dtype=float))
        else:
            self._data["confounders"] = np.zeros((len(treatment), 0))

        # 初始化参数
        self._params["beta"] = CausalParameter("beta", value=0.0, learning_rate=self._lr)
        self._params["gamma"] = CausalParameter("gamma", value=0.0, learning_rate=self._lr)
        self._params["intercept"] = CausalParameter("intercept", value=0.0, learning_rate=self._lr)

        self._loss_history.clear()

    def predict(self, treatment: np.ndarray | None = None) -> np.ndarray:
        """预测结果。

        Args:
            treatment: 处理变量 (None = 使用训练数据)

        Returns:
            预测结果
        """
        if treatment is None:
            X = self._data["treatment"]
        else:
            X = np.atleast_1d(np.asarray(treatment, dtype=float))

        Z = self._data.get("confounders", np.zeros((len(X), 0)))
        beta = self._params["beta"].value
        gamma = self._params["gamma"].value
        intercept = self._params["intercept"].value

        pred = beta * X + intercept
        if Z.shape[1] > 0:
            pred = pred + gamma * np.mean(Z, axis=1)
        return pred

    def compute_loss(self) -> float:
        """计算当前损失 (MSE)。"""
        if "outcome" not in self._data:
            return 0.0
        pred = self.predict()
        actual = self._data["outcome"]
        if len(pred) != len(actual):
            min_len = min(len(pred), len(actual))
            pred = pred[:min_len]
            actual = actual[:min_len]
        return float(np.mean((actual - pred) ** 2))

    def compute_gradients(self) -> dict[str, float]:
        """计算参数梯度 (解析)。"""
        if "outcome" not in self._data:
            return {}

        X = self._data["treatment"]
        Y = self._data["outcome"]
        Z = self._data.get("confounders", np.zeros((len(X), 0)))

        pred = self.predict()
        residual = pred - Y

        n = max(len(residual), 1)
        grad_beta = 2.0 * np.dot(residual, X) / n
        grad_intercept = 2.0 * np.mean(residual)

        grad_gamma = 0.0
        if Z.shape[1] > 0:
            z_mean = np.mean(Z, axis=1)
            grad_gamma = 2.0 * np.dot(residual, z_mean) / n

        return {
            "beta": float(grad_beta),
            "gamma": float(grad_gamma),
            "intercept": float(grad_intercept),
        }

    def optimize(self, n_iterations: int = 100) -> OptimizationResult:
        """执行梯度下降优化。

        Args:
            n_iterations: 迭代次数

        Returns:
            OptimizationResult
        """
        if not self._params:
            return OptimizationResult()

        initial_loss = self.compute_loss()

        for i in range(n_iterations):
            # 计算梯度
            grads = self.compute_gradients()

            # 更新参数
            for name, grad in grads.items():
                if name in self._params:
                    self._params[name].gradient = grad
                    self._params[name].step()

            loss = self.compute_loss()
            self._loss_history.append(loss)

            # 收敛检查
            if i > 0 and abs(self._loss_history[-2] - self._loss_history[-1]) < self._conv_threshold:
                logger.info("可微因果推断: 第 %d 步收敛", i)
                break

        final_loss = self._loss_history[-1] if self._loss_history else initial_loss
        converged = initial_loss - final_loss > self._conv_threshold

        return OptimizationResult(
            n_iterations=len(self._loss_history),
            initial_loss=initial_loss,
            final_loss=final_loss,
            converged=converged,
            parameters={name: p.value for name, p in self._params.items()},
        )

    @property
    def loss_history(self) -> list[float]:
        return list(self._loss_history)

    @property
    def treatment_effect(self) -> float:
        """估计的平均处理效应 (ATE)。"""
        return self._params.get("beta", CausalParameter("beta")).value

    def statistics(self) -> dict[str, Any]:
        return {
            "n_params": len(self._params),
            "learning_rate": self._lr,
            "convergence_threshold": self._conv_threshold,
            "treatment_effect": self.treatment_effect,
            "loss_history_length": len(self._loss_history),
        }
