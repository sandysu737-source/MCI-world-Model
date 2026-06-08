"""
MCI World Model v3.2.0 — ActionConditionedPredictor
=====================================================

动作条件化预测器——MCI 世界模型"预测环"的核心抽象。

与 JEPAPredictor 的关键区别:
    JEPAPredictor.predict(s_t) → s_{t+1}              (被动观察, 无动作)
    ActionConditionedPredictor.predict(s_t, a_t, n) → [s_{t+1}, ..., s_{t+n}]  (主动干预, 多步推演)

这是修复 MCI 世界模型架构缺陷 D2 的核心产物:
    世界模型必须能回答"如果我推一下会发生什么",
    而不仅仅是"我观察到什么变化"。

基线实现:
    PendulumPhysicsPredictor   — 利用已知物理公式 (ground truth 金标准)
    PendulumJEPAPredictor      — 纯 numpy MLP 学习器 (验证 JEPA 学习能力)

设计原则:
    - 与 JEPAPredictor 并行不冲突，独立文件独立接口
    - Future 不依赖 torch/MLX，纯 numpy 实现
    - 单摆验证: PhysicsPredictor == ground truth, JEPAPredictor → 逼近 PhysicsPredictor
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mci_world_model.sdk._world_state import Action, WorldState

logger = logging.getLogger(__name__)


# =============================================================================
# ActionConditionedPredictor — 抽象基类
# =============================================================================


class ActionConditionedPredictor(ABC):
    """动作条件化预测器抽象基类。

    与 JEPAPredictor 的区别:
        JEPAPredictor:  predict(s_t) → s_{t+1}           无动作参数
        ActionConditionedPredictor: predict(s_t, a_t, n_steps) → [s_{t+1}, ..., s_{t+n}]

    核心方法:
        predict(state, action, n_steps) → 多步预测轨迹
        rollout(state, actions)        → 动作序列完整推演
    """

    def __init__(self, name: str = "base"):
        self._name = name
        self._prediction_count: int = 0

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def predict(
        self,
        state: WorldState,
        action: Action | None,
        n_steps: int = 1,
    ) -> list[WorldState]:
        """动作条件化多步预测。

        Args:
            state: 当前世界状态 s_t
            action: 施加的动作 a_t (None = 无外力自然演化)
            n_steps: 预测步数

        Returns:
            预测的未来状态序列 [s_{t+1}, s_{t+2}, ..., s_{t+n_steps}]
            action 在每一步都施加相同动作（零阶保持）
        """
        ...

    def rollout(
        self,
        state: WorldState,
        actions: list[Action],
    ) -> list[WorldState]:
        """执行动作序列的完整推演轨迹。

        Args:
            state: 初始状态 s_0
            actions: 动作序列 [a_0, a_1, ..., a_{n-1}]

        Returns:
            状态轨迹 [s_1, s_2, ..., s_n]
        """
        trajectory = []
        current = state.copy()
        for action in actions:
            preds = self.predict(current, action, n_steps=1)
            current = preds[0]
            trajectory.append(current)
        return trajectory

    def evaluate(
        self,
        dataset: list,
    ) -> dict:
        """在 (s_t, a_t, s_{t+1}_true) 数据集上评估预测精度。

        Args:
            dataset: [(state, action, ground_truth), ...]

        Returns:
            评估统计 {"avg_distance": float, "n": int, ...}
        """

        distances = []
        for state, action, gt in dataset:
            preds = self.predict(state, action, n_steps=1)
            d = preds[0].distance(gt)
            distances.append(d)
            self._prediction_count += 1

        if not distances:
            return {"avg_distance": 1.0, "n": 0, "predictor": self._name}

        return {
            "avg_distance": round(float(np.mean(distances)), 6),
            "min_distance": round(float(np.min(distances)), 6),
            "max_distance": round(float(np.max(distances)), 6),
            "n": len(distances),
            "predictor": self._name,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name!r})"


# =============================================================================
# PendulumPhysicsPredictor — 物理公式金标准
# =============================================================================


class PendulumPhysicsPredictor(ActionConditionedPredictor):
    """单摆物理公式预测器——利用已知物理定律做 ground truth 预测。

    这是 MCI 世界模型预测环的"金标准"：
    - 不做任何学习，直接用 Euler 积分 + sin 公式
    - 任何学习型预测器必须以此精度为上限
    - 如果 PendulumJEPAPredictor 不能逼近此精度，说明架构有问题

    使用:
        >>> pred = PendulumPhysicsPredictor()
        >>> state = PendulumState(theta=0.5, omega=0.0)
        >>> push = PendulumAction(torque=3.0)
        >>> trajectory = pred.predict(state, action=push, n_steps=10)
        >>> # trajectory[0] 是 1 步后的物理正确状态
    """

    def __init__(self):
        super().__init__(name="pendulum_physics")

    def predict(
        self,
        state: WorldState,
        action: Action | None,
        n_steps: int = 1,
    ) -> list[WorldState]:
        """用物理公式做 Euler 积分多步预测。"""
        from mci_world_model.sdk._world_state import PendulumAction, PendulumState

        if not isinstance(state, PendulumState):
            raise TypeError(f"PendulumPhysicsPredictor 只能预测 PendulumState，收到 {type(state).__name__}")

        # 零阶保持: 同一个 action 施加 n_steps 次
        pendulum_action = action if isinstance(action, PendulumAction) else None

        trajectory = []
        current = state.copy()
        for _ in range(n_steps):
            if pendulum_action is not None:
                current = pendulum_action.apply(current)
            else:
                current = current.step_physics()
            trajectory.append(current)

        return trajectory


# =============================================================================
# PendulumJEPAPredictor — 学习型两隐层 MLP 预测器
# =============================================================================


class PendulumJEPAPredictor(ActionConditionedPredictor):
    """单摆 JEPA 学习型预测器——纯 numpy 线性模型。

    不依赖 torch/MLX/tensorflow。CPU 友好，仅 6 个可学习参数。

    模型: 线性变换 (theta, omega, torque) → (theta', omega')
        theta' = w1*theta + w2*omega + w3*torque + b1
        omega' = w4*theta + w5*omega + w6*torque + b2

    物理 ground truth（小角度近似）:
        theta' ≈ theta + omega*dt
        omega' ≈ omega - (g/L)*theta*dt + torque/L²*dt

    因此理想权重约为:
        w1≈1, w2≈dt, w3≈0
        w4≈-(g/L)*dt, w5≈1, w6≈dt/L²

    使用最小二乘解析解（正规方程）一次性训练，无需梯度下降。
    这确保 100% 收敛到全局最优。
    """

    def __init__(self, seed: int = 42):
        super().__init__(name="pendulum_jepa")
        rng = np.random.RandomState(seed)
        # 6 个参数: W(2,3) + b(2), 但用 8 个参数便于实现
        self._W = rng.randn(2, 3).astype(np.float64) * 0.01  # [2, 3]
        self._b = np.zeros(2, dtype=np.float64)
        self._trained: bool = False
        self._train_loss: float = float("inf")

    # ── 前向传播 ──

    def _forward(self, theta: float, omega: float, torque: float) -> tuple[float, float]:
        """线性前向: (θ, ω, τ) → (θ̂, ω̂)."""
        x = np.array([theta, omega, torque], dtype=np.float64)
        out = self._W @ x + self._b
        return float(out[0]), float(out[1])

    # ── 预测接口 ──

    def predict(
        self,
        state: WorldState,
        action: Action | None,
        n_steps: int = 1,
    ) -> list[WorldState]:
        """JEPA 学习型多步预测。"""
        from mci_world_model.sdk._world_state import PendulumAction, PendulumState

        if not isinstance(state, PendulumState):
            raise TypeError(f"PendulumJEPAPredictor 只能预测 PendulumState，收到 {type(state).__name__}")

        torque = action.torque if isinstance(action, PendulumAction) else 0.0
        trajectory = []
        current_theta, current_omega = state.theta, state.omega

        for _ in range(n_steps):
            theta_hat, omega_hat = self._forward(current_theta, current_omega, torque)
            trajectory.append(
                PendulumState(
                    theta=theta_hat,
                    omega=omega_hat,
                    g=state.g,
                    L=state.L,
                    dt=state.dt,
                )
            )
            current_theta, current_omega = theta_hat, omega_hat

        return trajectory

    # ── 训练: 最小二乘解析解 ──

    @property
    def is_trained(self) -> bool:
        return self._trained

    def train(
        self,
        n_samples: int = 2000,
        noise_std: float = 0.0,
    ) -> dict:
        """用最小二乘法（正规方程）训练线性模型。

        使用 PendulumPhysicsPredictor 生成训练数据，
        然后解析求解 W, b = argmin ||XW + b - Y||²。

        Args:
            n_samples: 训练样本数
            noise_std: 输出噪声标准差（0 = 无噪声）

        Returns:
            训练报告
        """
        from mci_world_model.sdk._world_state import PendulumAction, PendulumState

        physics = PendulumPhysicsPredictor()
        rng = np.random.RandomState(42)

        X = np.zeros((n_samples, 3), dtype=np.float64)  # (theta, omega, torque)
        Y = np.zeros((n_samples, 2), dtype=np.float64)  # (theta', omega')

        for i in range(n_samples):
            theta0 = rng.uniform(-np.pi * 0.8, np.pi * 0.8)
            omega0 = rng.uniform(-3.0, 3.0)
            torque = rng.uniform(-10.0, 10.0)

            s = PendulumState(theta=theta0, omega=omega0)
            a = PendulumAction(torque=torque)
            gt_list = physics.predict(s, a, n_steps=1)
            gt = gt_list[0]

            X[i] = [theta0, omega0, torque]
            Y[i] = [gt.theta + rng.randn() * noise_std, gt.omega + rng.randn() * noise_std]

        # 正规方程: W = (X^T X)^{-1} X^T Y
        # 加偏置: X_aug = [X, 1], 然后解 (X^T X) W_all = X^T Y
        X_aug = np.column_stack([X, np.ones(n_samples, dtype=np.float64)])
        W_aug = np.linalg.lstsq(X_aug, Y, rcond=None)[0]  # [4, 2]

        self._W = W_aug[:3, :].T  # [2, 3]
        self._b = W_aug[3, :]  # [2]

        # 计算训练损失
        Y_pred = X_aug @ W_aug
        mse = float(np.mean((Y_pred - Y) ** 2))
        self._train_loss = mse
        self._trained = True

        return {
            "final_loss": round(mse, 6),
            "n_samples": n_samples,
            "converged": mse < 0.1,
            "method": "least_squares",
        }
