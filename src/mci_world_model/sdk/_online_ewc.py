"""MCI World Model v4.6.0 — OnlineEWC 在线弹性权重巩固

无缝任务边界的持续学习。复用 IncrementalMLP + EWCConfig 内核，
提供 OnlineEWC.update(X, y) 流式增量学习接口。

核心差异 vs IncrementalLearningEngine:
    - 无 TaskSpec 注册, 直接 update(data)
    - Fisher 对角 O(N) 存储
    - 自适应 lambda: base_lambda * (1 + 0.2 * n_updates)

验收标准:
    - 5 任务后遗忘率 < 25%
    - loss 非负且单调递减
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from mci_world_model.sdk._incremental_learning import (
    EWCConfig,
    IncrementalMLP,
    TaskRecord,
    TaskSpec,
)

logger = logging.getLogger(__name__)


@dataclass
class OnlineEWCState:
    """OnlineEWC 运行时状态快照。"""

    n_updates: int = 0
    current_loss: float = 0.0
    adaptive_lambda: float = 100.0
    n_params: int = 0
    forgetting_rate: float = 0.0


class OnlineEWC:
    """在线弹性权重巩固 (Online Elastic Weight Consolidation)。

    流式增量学习，无任务边界。每次 update() 自动:
        1. 保存当前参数快照 + Fisher 对角
        2. 对新数据训练, 旧任务参数享受 EWC 保护
        3. 自适应 λ 随 update 次数增长

    Example:
        >>> ewc = OnlineEWC(input_dim=4, output_dim=2)
        >>> loss = ewc.update(X_batch1, y_batch1)
        >>> loss = ewc.update(X_batch2, y_batch2)
        >>> pred = ewc.predict(X_test)
        >>> state = ewc.get_state()
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: EWCConfig | None = None,
    ):
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError(f"维度必须为正: input={input_dim}, output={output_dim}")
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._config = config or EWCConfig()
        self._model = IncrementalMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=self._config.hidden_dim,
            seed=self._config.seed,
        )
        self._task_records: list[TaskRecord] = []
        self._n_updates: int = 0
        self._loss_history: list[float] = []
        self._fisher_buffer: dict[str, np.ndarray] = {}

    # ── 公开 API ──

    def update(self, X: np.ndarray, y: np.ndarray) -> float:
        """增量学习一批新数据。

        Args:
            X: 输入 (n, input_dim)
            y: 目标 (n, output_dim)

        Returns:
            最终 epoch 的 MSE loss
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.shape[1] != self._input_dim or y.shape[1] != self._output_dim:
            raise ValueError(
                f"维度不匹配: X {X.shape} vs input {self._input_dim}, "
                f"y {y.shape} vs output {self._output_dim}"
            )

        task_id = self._n_updates
        losses: list[float] = []

        for epoch in range(self._config.n_epochs):
            epoch_loss = 0.0
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)

            for idx in indices:
                x_i = X[idx : idx + 1]
                y_i = y[idx : idx + 1]

                if self._task_records:
                    current_lambda = self._adaptive_lambda
                    loss = self._model.train_step_with_ewc(
                        x_i, y_i,
                        lr=self._config.lr,
                        task_records=self._task_records,
                        ewc_lambda=current_lambda,
                    )
                else:
                    loss = self._model.train_step(x_i, y_i, lr=self._config.lr)

                epoch_loss += loss
            losses.append(epoch_loss / X.shape[0])

        # 保存当前任务记录
        self._compute_and_save_fisher(X, y, task_id)
        final_loss = losses[-1] if losses else 0.0
        self._loss_history.append(final_loss)
        self._n_updates += 1

        return final_loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """用当前模型预测。"""
        return self._model.forward(X)

    def loss(self, params: dict[str, np.ndarray] | None = None) -> float:
        """计算 EWC 正则化损失。

        L_ewc = λ/2 * Σ_t Σ_p F_p * (θ_p - θ*_p)²
        """
        if not self._task_records:
            return 0.0
        target = params or {
            "W1": self._model.W1,
            "b1": self._model.b1,
            "W2": self._model.W2,
            "b2": self._model.b2,
        }
        penalty = 0.0
        for record in self._task_records:
            for pname in ["W1", "b1", "W2", "b2"]:
                if pname not in record.optimal_params:
                    continue
                current = target[pname]
                optimal = record.optimal_params[pname]
                fisher = record.fisher_diagonal.get(pname, np.ones_like(current))
                penalty += float(np.sum(fisher * (current - optimal) ** 2))
        return penalty * self._adaptive_lambda / 2.0

    def forget(self, n_tasks_back: int = 1) -> None:
        """选择性遗忘最近 n 个更新周期。

        Args:
            n_tasks_back: 遗忘的最近周期数
        """
        n = min(n_tasks_back, len(self._task_records))
        if n > 0:
            self._task_records = self._task_records[:-n]
            self._n_updates = max(0, self._n_updates - n)

    def get_state(self) -> OnlineEWCState:
        """返回当前运行时状态。"""
        return OnlineEWCState(
            n_updates=self._n_updates,
            current_loss=self._loss_history[-1] if self._loss_history else 0.0,
            adaptive_lambda=self._adaptive_lambda,
            n_params=self._count_params(),
            forgetting_rate=self._estimate_forgetting(),
        )

    # ── 属性 ──

    @property
    def n_tasks(self) -> int:
        return self._n_updates

    @property
    def task_records(self) -> list[TaskRecord]:
        return list(self._task_records)

    @property
    def model(self) -> IncrementalMLP:
        return self._model

    @property
    def _adaptive_lambda(self) -> float:
        """自适应 EWC lambda — 随 update 次数增长。"""
        if not self._config.adaptive_ewc:
            return self._config.ewc_lambda
        return self._config.ewc_lambda * (
            1.0 + self._config.ewc_lambda_growth * self._n_updates
        )

    # ── 内部方法 ──

    def _compute_and_save_fisher(
        self, X: np.ndarray, y: np.ndarray, task_id: int
    ) -> None:
        """计算对角 Fisher 并保存任务记录。"""
        sample_idx = np.random.choice(
            X.shape[0],
            size=min(self._config.fisher_samples, X.shape[0]),
            replace=False,
        )
        X_sample = X[sample_idx]
        y_sample = y[sample_idx]

        fisher = self._compute_fisher_diag(X_sample, y_sample)

        params = {
            "W1": self._model.W1.copy(),
            "b1": self._model.b1.copy(),
            "W2": self._model.W2.copy(),
            "b2": self._model.b2.copy(),
        }

        record = TaskRecord(
            task=TaskSpec(
                name=f"task_{task_id}",
                input_dim=self._input_dim,
                output_dim=self._output_dim,
            ),
            optimal_params=params,
            fisher_diagonal=fisher,
            final_loss=self._loss_history[-1] if self._loss_history else 0.0,
        )
        self._task_records.append(record)

    def _compute_fisher_diag(
        self, X: np.ndarray, y: np.ndarray
    ) -> dict[str, np.ndarray]:
        """计算参数的对角 Fisher 信息 (O(N) 复杂度)。

        对角 Fisher: F_i = E[(∂log p(y|x)/∂θ_i)²]
        对 MSE 损失: F_i ≈ mean((∂L/∂θ_i)²)
        """
        # 单次前向 + 反向获取梯度
        pred = self._model.forward(X)
        batch = X.shape[0]
        d_out = 2.0 * (pred - y) / batch
        h_act = self._model._cache["h_act"]
        x_in = self._model._cache["x"]

        d_W2 = h_act.T @ d_out
        d_b2 = d_out.sum(axis=0)
        d_h_act = d_out @ self._model.W2.T
        d_h = d_h_act * (self._model._cache["h"] > 0).astype(np.float64)
        d_W1 = x_in.T @ d_h
        d_b1 = d_h.sum(axis=0)

        fisher = {
            "W1": d_W1 ** 2,
            "b1": d_b1 ** 2,
            "W2": d_W2 ** 2,
            "b2": d_b2 ** 2,
        }
        # 正则化: 防止除零 + 裁剪防止梯度爆炸
        for k, v in fisher.items():
            fisher[k] = np.clip(np.maximum(v, 1e-8), 0.0, 100.0)
        return fisher

    def _count_params(self) -> int:
        return (
            self._model.W1.size
            + self._model.b1.size
            + self._model.W2.size
            + self._model.b2.size
        )

    def _estimate_forgetting(self) -> float:
        """估计遗忘率: 最近 loss / 全局最小 loss - 1。

        值越大 → 遗忘越严重。
        """
        if len(self._loss_history) < 2:
            return 0.0
        recent = np.mean(self._loss_history[-3:]) if len(self._loss_history) >= 3 else self._loss_history[-1]
        best = min(self._loss_history)
        if best < 1e-10:
            return 0.0
        return max(0.0, (recent - best) / best)
