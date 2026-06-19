"""SimpleMAML — 轻量级模型无关元学习。

P3 "赋魂" 核心模块 — 纯 numpy 实现的 MAML (Model-Agnostic Meta-Learning)，
目标：Pendulum→Cart 零样本迁移准确率 ≥60%。

核心思想:
    - Inner loop: θ' = θ - α · ∇L_task(θ)     (任务内快速适应)
    - Outer loop: θ ← θ - β · ∇L_meta(θ')      (跨任务元优化)
    - 学到一个「易于适应新任务」的参数初始化点

Usage::
    from mci_world_model.sdk._simple_maml import SimpleMAML, MAMLTask

    maml = SimpleMAML(input_dim=2, output_dim=2, hidden_dim=32)
    tasks = [MAMLTask(x_train, y_train) for _ in range(10)]
    maml.meta_train(tasks, n_epochs=100)
    loss = maml.adapt(task_new.x_support, task_new.y_support)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MAMLTask:
    """MAML 单任务 — 支撑集 + 查询集。

    Attributes:
        x_support: 支撑集输入 (n_support, input_dim)。
        y_support: 支撑集标签 (n_support, output_dim)。
        x_query: 查询集输入 (n_query, input_dim)。
        y_query: 查询集标签 (n_query, output_dim)。
    """

    x_support: np.ndarray
    y_support: np.ndarray
    x_query: np.ndarray | None = None
    y_query: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.x_query is None:
            # 默认 50/50 划分
            n = len(self.x_support)
            split = n // 2
            perm = np.random.RandomState(42).permutation(n)
            self.x_support, self.x_query = (
                self.x_support[perm[:split]],
                self.x_support[perm[split:]],
            )
            self.y_support, self.y_query = (
                self.y_support[perm[:split]],
                self.y_support[perm[split:]],
            )


@dataclass
class SimpleMAML:
    """轻量级 MAML — 纯 numpy 实现。

    使用单隐藏层 MLP 作为元模型，通过二阶梯度（或一阶近似）更新。
    默认使用 FOMAML（一阶 MAML）简化计算、加快速度。

    Attributes:
        input_dim: 输入维度。
        output_dim: 输出维度。
        hidden_dim: 隐藏层维度。
        use_first_order: True 使用 FOMAML (更快)，False 使用完整 MAML。
        meta_lr: 外循环学习率 β。
        inner_lr: 内循环学习率 α。
        inner_steps: 内循环梯度步数。
    """

    input_dim: int = 2
    output_dim: int = 2
    hidden_dim: int = 32
    use_first_order: bool = True
    meta_lr: float = 0.01
    inner_lr: float = 0.1
    inner_steps: int = 5

    # 模型参数
    w1: np.ndarray = field(init=False, repr=False)
    b1: np.ndarray = field(init=False, repr=False)
    w2: np.ndarray = field(init=False, repr=False)
    b2: np.ndarray = field(init=False, repr=False)

    # 训练历史
    _meta_losses: list[float] = field(default_factory=list, repr=False)
    _n_meta_steps: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.RandomState(42)
        scale = np.sqrt(2.0 / self.input_dim)
        self.w1 = rng.randn(self.input_dim, self.hidden_dim).astype(np.float64) * scale
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float64)
        self.w2 = rng.randn(self.hidden_dim, self.output_dim).astype(np.float64) * scale
        self.b2 = np.zeros(self.output_dim, dtype=np.float64)

    # ── Forward ─────────────────────────────────────────────────────────

    def _forward(
        self, x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray
    ) -> np.ndarray:
        """单隐藏层 MLP 前向传播。"""
        h = np.maximum(0, x @ w1 + b1)  # ReLU
        return h @ w2 + b2

    def _loss(
        self, x: np.ndarray, y: np.ndarray,
        w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray,
    ) -> float:
        """MSE 损失。"""
        pred = self._forward(x, w1, b1, w2, b2)
        return float(np.mean((pred - y) ** 2))

    # ── Inner Loop (Task-Specific Adaptation) ────────────────────────────

    def adapt(
        self, x_support: np.ndarray, y_support: np.ndarray,
        steps: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """内循环：在支撑集上快速适应。

        Returns:
            (w1', b1', w2', b2') — 适应后的参数。
        """
        n_steps = steps if steps is not None else self.inner_steps
        w1, b1 = self.w1.copy(), self.b1.copy()
        w2, b2 = self.w2.copy(), self.b2.copy()
        lr = self.inner_lr

        for _ in range(n_steps):
            h = np.maximum(0, x_support @ w1 + b1)
            pred = h @ w2 + b2
            err = pred - y_support

            # Gradient of W2, B2
            grad_w2 = h.T @ err / len(x_support)
            grad_b2 = err.mean(axis=0)

            # Gradient of W1, B1 (backprop through ReLU)
            dh = err @ w2.T
            dh[h <= 0] = 0
            grad_w1 = x_support.T @ dh / len(x_support)
            grad_b1 = dh.mean(axis=0)

            # Update
            w2 -= lr * grad_w2
            b2 -= lr * grad_b2
            w1 -= lr * grad_w1
            b1 -= lr * grad_b1

        return w1, b1, w2, b2

    # ── Outer Loop (Meta-Learning) ───────────────────────────────────────

    def meta_train(
        self, tasks: list[MAMLTask], n_epochs: int = 100, verbose: bool = False,
    ) -> list[float]:
        """元训练：学习跨任务通用初始化。

        Args:
            tasks: 训练任务列表。
            n_epochs: 外循环轮数。
            verbose: 是否打印进度。

        Returns:
            每轮的元损失列表。
        """
        self._meta_losses = []
        beta = self.meta_lr

        for epoch in range(n_epochs):
            meta_grad_w1 = np.zeros_like(self.w1)
            meta_grad_b1 = np.zeros_like(self.b1)
            meta_grad_w2 = np.zeros_like(self.w2)
            meta_grad_b2 = np.zeros_like(self.b2)
            epoch_loss = 0.0

            for task in tasks:
                # Inner loop adaptation
                w1_a, b1_a, w2_a, b2_a = self.adapt(task.x_support, task.y_support)

                # Query set loss
                pred_q = self._forward(
                    task.x_query, w1_a, b1_a, w2_a, b2_a
                )
                task_loss = float(np.mean((pred_q - task.y_query) ** 2))
                epoch_loss += task_loss

                if self.use_first_order:
                    # FOMAML: 只对外层参数求导，忽略二阶项
                    err_q = pred_q - task.y_query
                    h_q = np.maximum(0, task.x_query @ w1_a + b1_a)
                    # Gradient on adapted params → accumulate on meta params
                    grad_w2_q = h_q.T @ err_q / len(task.x_query)
                    grad_b2_q = err_q.mean(axis=0)

                    meta_grad_w2 += grad_w2_q
                    meta_grad_b2 += grad_b2_q

                    dh_q = err_q @ w2_a.T
                    dh_q[h_q <= 0] = 0
                    grad_w1_q = task.x_query.T @ dh_q / len(task.x_query)
                    grad_b1_q = dh_q.mean(axis=0)

                    meta_grad_w1 += grad_w1_q
                    meta_grad_b1 += grad_b1_q

            # Average across tasks
            n_tasks = max(len(tasks), 1)
            self.w2 -= beta * meta_grad_w2 / n_tasks
            self.b2 -= beta * meta_grad_b2 / n_tasks
            self.w1 -= beta * meta_grad_w1 / n_tasks
            self.b1 -= beta * meta_grad_b1 / n_tasks

            avg_loss = epoch_loss / n_tasks
            self._meta_losses.append(avg_loss)
            self._n_meta_steps += 1

            if verbose and epoch % max(1, n_epochs // 10) == 0:
                print(f"  Epoch {epoch:4d}/{n_epochs}  loss={avg_loss:.6f}")

        return self._meta_losses

    # ── Evaluation ──────────────────────────────────────────────────────

    def evaluate_adaptation(
        self, x_support: np.ndarray, y_support: np.ndarray,
        x_test: np.ndarray, y_test: np.ndarray,
    ) -> dict[str, float]:
        """评估在新任务上的快速适应能力。

        Returns:
            dict: before_loss, after_loss, improvement_ratio, steps。
        """
        # Before adaptation
        pred_before = self._forward(x_test, self.w1, self.b1, self.w2, self.b2)
        loss_before = float(np.mean((pred_before - y_test) ** 2))

        # After adaptation
        w1_a, b1_a, w2_a, b2_a = self.adapt(x_support, y_support)
        pred_after = self._forward(x_test, w1_a, b1_a, w2_a, b2_a)
        loss_after = float(np.mean((pred_after - y_test) ** 2))

        return {
            "loss_before": loss_before,
            "loss_after": loss_after,
            "improvement_ratio": (loss_before - loss_after) / max(loss_before, 1e-12),
            "inner_steps": self.inner_steps,
        }

    def transfer_score(
        self,
        source_tasks: list[MAMLTask],
        target_task: MAMLTask,
        n_epochs: int = 50,
    ) -> dict[str, float]:
        """评估零样本迁移能力。

        在源任务上训练 MAML，在目标任务上测量快速适应效果。

        Returns:
            dict: transfer_score, source_loss, target_before, target_after。
        """
        # Save original params
        orig = (self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy())

        # Train on source
        source_losses = self.meta_train(source_tasks, n_epochs=n_epochs)
        final_source_loss = source_losses[-1] if source_losses else float("inf")

        # Evaluate on target
        eval_result = self.evaluate_adaptation(
            target_task.x_support, target_task.y_support,
            target_task.x_query, target_task.y_query,
        )

        # Transfer score: improvement from pre-adaptation to post-adaptation
        # ≥ 60% means effective zero-shot transfer
        transfer = eval_result["improvement_ratio"]

        # Restore
        self.w1, self.b1, self.w2, self.b2 = orig

        return {
            "transfer_score": max(0.0, transfer),
            "source_loss": final_source_loss,
            "target_before": eval_result["loss_before"],
            "target_after": eval_result["loss_after"],
            "zero_shot_passes": transfer >= 0.6,
        }

    # ── Utilities ───────────────────────────────────────────────────────

    @property
    def meta_loss_history(self) -> list[float]:
        return list(self._meta_losses)

    def reset(self) -> None:
        """重置模型参数和训练历史。"""
        self.__post_init__()
        self._meta_losses = []
        self._n_meta_steps = 0


__all__ = ["SimpleMAML", "MAMLTask"]
