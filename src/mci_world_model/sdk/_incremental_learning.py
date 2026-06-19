from __future__ import annotations

"""增量学习框架 — TASK-B3。

使用 Elastic Weight Consolidation (EWC) 在保留旧知识的同时学习新数据。

核心公式:
    L_total = L_new_task + λ/2 × Σ F_i × (θ_i - θ*_i)²

    其中:
        L_new_task: 新任务损失
        F_i: Fisher 信息矩阵对角元素 (参数重要性)
        θ*_i: 旧任务最优参数
        λ: EWC 正则化强度

Fisher 信息矩阵估计:
    F_i = E[ (∂ log p(y|x,θ) / ∂θ_i)² ]
    在训练数据上计算梯度的平方期望

多任务组织:
    - 参数级共享: 所有任务共享参数, EWC 惩罚重要参数偏移
    - 任务注册: 每个任务完成后注册 (保存 θ* 和 F)
    - 支持多个任务的连续学习

验收标准:
    - 旧任务遗忘率 < 15% (EWC vs 无 EWC)
    - 新任务学习速度差异 < 30% (EWC vs 从头训练)
    - 支持 ≥ 5 个连续任务的增量学习不崩溃
"""


import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 核心数据结构
# =============================================================================


@dataclass
class TaskSpec:
    """任务描述。

    Attributes:
        name: 任务名称
        input_dim: 输入维度
        output_dim: 输出维度
        n_samples: 训练样本数
    """

    name: str = ""
    input_dim: int = 2
    output_dim: int = 2
    n_samples: int = 0


@dataclass
class TaskRecord:
    """已完成任务的记录。

    Attributes:
        task: 任务描述
        optimal_params: 任务完成时的最优参数 {name: ndarray}
        fisher_diagonal: Fisher 信息矩阵对角线 {name: ndarray}
        final_loss: 最终训练损失
        accuracy: 任务准确率 [0, 1]
    """

    task: TaskSpec = field(default_factory=TaskSpec)
    optimal_params: dict[str, np.ndarray] = field(default_factory=dict)
    fisher_diagonal: dict[str, np.ndarray] = field(default_factory=dict)
    final_loss: float = 0.0
    accuracy: float = 0.0


@dataclass
class EWCConfig:
    """EWC 增量学习配置。

    Attributes:
        ewc_lambda: EWC 正则化强度 λ 基值
        adaptive_ewc: 是否启用自适应 λ (随任务数增长)
        ewc_lambda_growth: λ 增长率 — λ_adaptive = ewc_lambda * (1 + growth * n_tasks)
        hidden_dim: MLP 隐层维度
        lr: 学习率
        n_epochs: 每个任务训练轮数
        fisher_samples: Fisher 信息矩阵估计的采样数
        seed: 随机种子
        max_tasks: 最大支持任务数
    """

    ewc_lambda: float = 100.0
    adaptive_ewc: bool = True
    ewc_lambda_growth: float = 0.2
    hidden_dim: int = 32
    lr: float = 0.01
    n_epochs: int = 50
    fisher_samples: int = 50
    seed: int = 42
    max_tasks: int = 10


# =============================================================================
# IncrementalMLP — 支持增量学习的简单 MLP
# =============================================================================


class IncrementalMLP:
    """支持增量学习的 MLP。

    架构: Input → Hidden → ReLU → Output
    手写梯度 + SGD, 兼容 EWC 正则化。

    用法:
        >>> mlp = IncrementalMLP(input_dim=2, output_dim=2, hidden_dim=32)
        >>> loss = mlp.train_step(x, y, lr=0.01)
        >>> pred = mlp.forward(x)
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Xavier 初始化
        self.W1 = rng.randn(input_dim, hidden_dim).astype(np.float64) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.W2 = rng.randn(hidden_dim, output_dim).astype(np.float64) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b2 = np.zeros(output_dim, dtype=np.float64)

        # 缓存
        self._cache: dict[str, Any] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播。"""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        h = x @ self.W1 + self.b1
        h_act = np.maximum(h, 0)  # ReLU
        out = h_act @ self.W2 + self.b2

        self._cache = {"x": x, "h": h, "h_act": h_act}
        return out

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        """单步训练 (SGD)。

        Args:
            x: 输入 (batch, input_dim)
            y: 目标 (batch, output_dim)
            lr: 学习率

        Returns:
            MSE loss
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)

        pred = self.forward(x)
        loss = float(np.mean((pred - y) ** 2))

        # 反向传播
        batch = x.shape[0]
        d_out = 2.0 * (pred - y) / batch  # (batch, output_dim)

        h_act = self._cache["h_act"]
        x_input = self._cache["x"]

        # W2, b2 梯度
        d_W2 = h_act.T @ d_out  # (hidden, output)
        d_b2 = d_out.sum(axis=0)  # (output,)

        # ReLU 反向
        d_h_act = d_out @ self.W2.T  # (batch, hidden)
        d_h = d_h_act * (self._cache["h"] > 0).astype(np.float64)

        # W1, b1 梯度
        d_W1 = x_input.T @ d_h  # (input, hidden)
        d_b1 = d_h.sum(axis=0)  # (hidden,)

        # 梯度裁剪
        for g in [d_W1, d_b1, d_W2, d_b2]:
            np.clip(g, -5, 5, out=g)

        # SGD 更新
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2

        return loss

    def train_step_with_ewc(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lr: float,
        task_records: list[TaskRecord],
        ewc_lambda: float,
    ) -> float:
        """带 EWC 正则化的训练步。

        L_total = L_MSE + λ/2 × Σ_tasks Σ_params F_i × (θ_i - θ*_i)²

        Args:
            x, y, lr: 同 train_step
            task_records: 已完成任务记录列表
            ewc_lambda: EWC 正则化强度

        Returns:
            总损失 (MSE + EWC penalty)
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)

        pred = self.forward(x)
        mse_loss = float(np.mean((pred - y) ** 2))

        # EWC 惩罚
        ewc_penalty = 0.0
        for record in task_records:
            for param_name in ["W1", "b1", "W2", "b2"]:
                if param_name not in record.optimal_params:
                    continue
                current = getattr(self, param_name)
                optimal = record.optimal_params[param_name]
                fisher = record.fisher_diagonal.get(param_name, np.ones_like(current))

                ewc_penalty += float(np.sum(fisher * (current - optimal) ** 2))

        ewc_penalty *= ewc_lambda / 2.0
        total_loss = mse_loss + ewc_penalty

        # 反向传播 (MSE 部分, 同 train_step)
        batch = x.shape[0]
        d_out = 2.0 * (pred - y) / batch

        h_act = self._cache["h_act"]
        x_input = self._cache["x"]

        d_W2 = h_act.T @ d_out
        d_b2 = d_out.sum(axis=0)

        d_h_act = d_out @ self.W2.T
        d_h = d_h_act * (self._cache["h"] > 0).astype(np.float64)

        d_W1 = x_input.T @ d_h
        d_b1 = d_h.sum(axis=0)

        # EWC 梯度: ∂/∂θ [λ/2 × Σ F_i (θ_i - θ*_i)²] = λ × Σ F_i × (θ_i - θ*_i)
        for record in task_records:
            for param_name, d_param in [("W1", d_W1), ("b1", d_b1), ("W2", d_W2), ("b2", d_b2)]:
                if param_name not in record.optimal_params:
                    continue
                current = getattr(self, param_name)
                optimal = record.optimal_params[param_name]
                fisher = record.fisher_diagonal.get(param_name, np.ones_like(current))
                d_param += ewc_lambda * fisher * (current - optimal)

        # 梯度裁剪
        for g in [d_W1, d_b1, d_W2, d_b2]:
            np.clip(g, -5, 5, out=g)

        # SGD 更新
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2

        return total_loss

    def get_params(self) -> dict[str, np.ndarray]:
        """获取所有参数。"""
        return {"W1": self.W1.copy(), "b1": self.b1.copy(), "W2": self.W2.copy(), "b2": self.b2.copy()}

    def compute_fisher(self, X: np.ndarray, n_samples: int = 50) -> dict[str, np.ndarray]:
        """估计 Fisher 信息矩阵对角线 — 解析梯度法 O(N)。

        使用反向传播解析梯度替代有限差分, 复杂度从 O(N²) 降至 O(N):
            F_i ≈ (1/N) Σ_n (∂L/∂θ_i)²

        Args:
            X: 数据样本 (n, input_dim)
            n_samples: 采样数

        Returns:
            {param_name: fisher_diagonal}
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n = min(n_samples, X.shape[0])
        fisher = {name: np.zeros_like(param) for name, param in self.get_params().items()}

        for idx in range(n):
            x = X[idx : idx + 1]
            # 用模型自身输出作为目标, 计算 MSE 梯度
            y_target = self.forward(x).copy()
            grads = self._compute_gradients(x, y_target)
            for name, grad in grads.items():
                fisher[name] += grad**2

        # 归一化
        for name in fisher:
            fisher[name] /= n

        return fisher

    def _compute_gradients(self, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        """计算 MSE 损失对参数的解析梯度 (反向传播)。

        与 train_step 共享梯度逻辑, 但不更新参数。

        Args:
            x: 输入 (1, input_dim)
            y: 目标 (1, output_dim)

        Returns:
            {param_name: gradient}
        """
        pred = self.forward(x)
        # MSE 梯度: ∂/∂pred [mean((pred - y)²)] = 2(pred - y) / batch
        batch = x.shape[0]
        d_out = 2.0 * (pred - y) / batch

        h_act = self._cache["h_act"]
        x_input = self._cache["x"]

        # W2, b2 梯度
        d_W2 = h_act.T @ d_out
        d_b2 = d_out.sum(axis=0)

        # ReLU 反向
        d_h_act = d_out @ self.W2.T
        d_h = d_h_act * (self._cache["h"] > 0).astype(np.float64)

        # W1, b1 梯度
        d_W1 = x_input.T @ d_h
        d_b1 = d_h.sum(axis=0)

        return {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2}


# =============================================================================
# IncrementalLearningEngine — 增量学习引擎
# =============================================================================


class IncrementalLearningEngine:
    """EWC 增量学习引擎。

    工作流:
        1. 注册任务 → learn_task(task, X, y)
        2. 训练: 带有 EWC 正则化的 SGD
        3. 保存最优参数 + Fisher 信息
        4. 后续任务训练时, EWC 惩罚防止重要参数偏移

    用法:
        >>> engine = IncrementalLearningEngine(EWCConfig())
        >>> engine.learn_task(TaskSpec(name="task1", input_dim=2, output_dim=2), X1, y1)
        >>> engine.learn_task(TaskSpec(name="task2", input_dim=2, output_dim=2), X2, y2)
        >>> pred = engine.predict(X_new)

    验收标准:
        - 旧任务遗忘率 < 15%
        - 新任务学习速度差异 < 30%
        - 支持 ≥ 5 个连续任务
    """

    def __init__(self, config: EWCConfig | None = None):
        """
        Args:
            config: EWC 配置
        """
        self._config = config or EWCConfig()
        self._task_records: list[TaskRecord] = []
        self._model: IncrementalMLP | None = None
        self._current_task: TaskSpec | None = None
        self._loss_history: list[list[float]] = []

    @property
    def n_tasks(self) -> int:
        """已完成任务数。"""
        return len(self._task_records)

    @property
    def task_names(self) -> list[str]:
        """已完成任务名称列表。"""
        return [r.task.name for r in self._task_records]

    @property
    def model(self) -> IncrementalMLP | None:
        """当前模型。"""
        return self._model

    @property
    def task_records(self) -> list[TaskRecord]:
        """任务记录列表。"""
        return list(self._task_records)

    @property
    def config(self) -> EWCConfig:
        """配置。"""
        return self._config

    @property
    def _adaptive_lambda(self) -> float:
        """自适应 EWC lambda — 随已完成任务数增长。

        公式: λ_adaptive = λ_base × (1 + growth × n_completed_tasks)

        5 任务后: λ = 100 × (1 + 0.2 × 4) = 180 (vs 固定 100)
        """
        if not self._config.adaptive_ewc:
            return self._config.ewc_lambda
        n_completed = len(self._task_records)  # 当前训练中的任务尚未注册
        return self._config.ewc_lambda * (1.0 + self._config.ewc_lambda_growth * n_completed)

    # -----------------------------------------------------------------
    # 公开 API
    # -----------------------------------------------------------------

    def learn_task(
        self,
        task: TaskSpec,
        X: np.ndarray,
        y: np.ndarray,
        use_ewc: bool = True,
    ) -> dict[str, Any]:
        """学习一个新任务。

        Args:
            task: 任务描述
            X: 训练输入 (n, input_dim)
            y: 训练目标 (n, output_dim)
            use_ewc: 是否使用 EWC 正则化

        Returns:
            {"final_loss": float, "accuracy": float, "n_epochs": int, "ewc_enabled": bool}
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if len(self._task_records) >= self._config.max_tasks:
            logger.warning("已达到最大任务数 %d, 不再学习新任务", self._config.max_tasks)
            return {"final_loss": float("inf"), "accuracy": 0.0, "n_epochs": 0, "ewc_enabled": use_ewc}

        # 初始化或复用模型
        if self._model is None:
            self._model = IncrementalMLP(
                input_dim=task.input_dim,
                output_dim=task.output_dim,
                hidden_dim=self._config.hidden_dim,
                seed=self._config.seed,
            )
        elif self._model.input_dim != task.input_dim or self._model.output_dim != task.output_dim:
            raise ValueError(
                f"维度不匹配: 模型 ({self._model.input_dim}→{self._model.output_dim}) "
                f"vs 任务 ({task.input_dim}→{task.output_dim})"
            )

        # 训练
        losses: list[float] = []
        for epoch in range(self._config.n_epochs):
            epoch_loss = 0.0
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)

            for idx in indices:
                x_i = X[idx : idx + 1]
                y_i = y[idx : idx + 1]

                if use_ewc and self._task_records:
                    # 自适应 lambda: 随已完成任务数增长, 抑制遗忘
                    current_lambda = self._adaptive_lambda
                    loss = self._model.train_step_with_ewc(
                        x_i,
                        y_i,
                        lr=self._config.lr,
                        task_records=self._task_records,
                        ewc_lambda=current_lambda,
                    )
                else:
                    loss = self._model.train_step(x_i, y_i, lr=self._config.lr)

                epoch_loss += loss

            avg_loss = epoch_loss / X.shape[0]
            losses.append(avg_loss)

        self._loss_history.append(losses)

        # 计算准确率 (MSE < 0.1 视为正确)
        preds = self._model.forward(X)
        accuracy = float(np.mean(np.sum((preds - y) ** 2, axis=1) < 0.1))

        # 计算 Fisher 信息矩阵
        fisher = self._model.compute_fisher(X, n_samples=self._config.fisher_samples)

        # 保存任务记录
        record = TaskRecord(
            task=task,
            optimal_params=self._model.get_params(),
            fisher_diagonal=fisher,
            final_loss=losses[-1] if losses else 0.0,
            accuracy=accuracy,
        )
        self._task_records.append(record)
        self._current_task = task

        return {
            "final_loss": record.final_loss,
            "accuracy": accuracy,
            "n_epochs": self._config.n_epochs,
            "ewc_enabled": use_ewc,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """使用当前模型预测。

        Args:
            X: 输入 (n, input_dim)

        Returns:
            预测 (n, output_dim)
        """
        if self._model is None:
            raise RuntimeError("尚未训练任何任务")
        return self._model.forward(X)

    def evaluate_on_task(self, task_idx: int, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """评估模型在指定任务上的表现。

        Args:
            task_idx: 任务索引
            X: 测试输入
            y: 测试目标

        Returns:
            {"mse": float, "accuracy": float}
        """
        if self._model is None:
            raise RuntimeError("尚未训练任何任务")

        preds = self._model.forward(X)
        mse = float(np.mean((preds - y) ** 2))
        accuracy = float(np.mean(np.sum((preds - y) ** 2, axis=1) < 0.1))

        return {"mse": mse, "accuracy": accuracy}

    def forgetting_rate(self, task_idx: int, X: np.ndarray, y: np.ndarray) -> float:
        """计算指定任务的遗忘率。

        forgetting_rate = max(0, old_accuracy - current_accuracy) / old_accuracy

        Args:
            task_idx: 任务索引
            X: 测试输入
            y: 测试目标

        Returns:
            遗忘率 [0, 1]
        """
        if task_idx >= len(self._task_records):
            return 0.0

        old_accuracy = self._task_records[task_idx].accuracy
        if old_accuracy <= 0:
            return 0.0

        current = self.evaluate_on_task(task_idx, X, y)
        current_accuracy = current["accuracy"]

        return max(0.0, (old_accuracy - current_accuracy) / old_accuracy)
