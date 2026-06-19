from __future__ import annotations

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
    CartPhysicsPredictor       — 小车物理公式预测器 (v4.4.0 Phase 0 泛化验证)

设计原则:
    - 与 JEPAPredictor 并行不冲突，独立文件独立接口
    - Future 不依赖 torch/MLX，纯 numpy 实现
    - 单摆验证: PhysicsPredictor == ground truth, JEPAPredictor → 逼近 PhysicsPredictor
"""


import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

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

    def __init__(self, name: str = "base") -> None:
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
        dataset: list[Any],
    ) -> dict[str, Any]:
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

    def __init__(self) -> None:
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

    def __init__(self, seed: int = 42) -> None:
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
    ) -> dict[str, Any]:
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


# =============================================================================
# PendulumNeuralPredictor — ≥10K 参数 MLP 预测器 (F10 修复)
# =============================================================================


class PendulumNeuralPredictor(ActionConditionedPredictor):
    """单摆神经网络预测器 — ≥10K 参数 3 层 MLP。

    替代 PendulumJEPAPredictor 的 6 参数线性模型,
    使用 3 层 MLP (3→64→128→64→2) 捕获非线性动力学。

    参数量: 3*64 + 64 + 64*128 + 128 + 128*64 + 64 + 64*2 + 2 = 16,962

    与 PendulumJEPAPredictor 的关键区别:
        - 非线性激活 (ReLU) — 可捕获 sin/cos 非线性
        - 3 层隐层 — 万能逼近定理保证
        - 梯度下降训练 — 支持任意损失函数

    保留 PendulumJEPAPredictor 作为线性基线 fallback。
    """

    def __init__(self, seed: int = 42) -> None:
        super().__init__(name="pendulum_neural")
        rng = np.random.RandomState(seed)

        # 3-layer MLP: 3 → 64 → 128 → 64 → 2
        self._W1 = self._xavier_init(rng, 3, 64)
        self._b1 = np.zeros(64, dtype=np.float64)
        self._W2 = self._xavier_init(rng, 64, 128)
        self._b2 = np.zeros(128, dtype=np.float64)
        self._W3 = self._xavier_init(rng, 128, 64)
        self._b3 = np.zeros(64, dtype=np.float64)
        self._W4 = self._xavier_init(rng, 64, 2)
        self._b4 = np.zeros(2, dtype=np.float64)

        self._trained: bool = False
        self._train_loss: float = float("inf")
        self._cache: dict[str, Any] = {}

    @staticmethod
    def _xavier_init(rng: np.random.RandomState, fan_in: int, fan_out: int) -> np.ndarray:
        """Xavier/Glorot 初始化。"""
        return rng.randn(fan_in, fan_out).astype(np.float64) * np.sqrt(2.0 / (fan_in + fan_out))

    @property
    def n_params(self) -> int:
        """可学习参数总数。"""
        return sum(
            p.size
            for p in [
                self._W1,
                self._b1,
                self._W2,
                self._b2,
                self._W3,
                self._b3,
                self._W4,
                self._b4,
            ]
        )

    # ── 前向传播 ──

    def _forward_batch(self, x: np.ndarray) -> np.ndarray:
        """批量前向传播: (batch, 3) → (batch, 2)。"""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        h1 = x @ self._W1 + self._b1
        h1_act = np.maximum(h1, 0)  # ReLU
        h2 = h1_act @ self._W2 + self._b2
        h2_act = np.maximum(h2, 0)
        h3 = h2_act @ self._W3 + self._b3
        h3_act = np.maximum(h3, 0)
        out = h3_act @ self._W4 + self._b4

        self._cache = {"x": x, "h1": h1, "h1_act": h1_act, "h2": h2, "h2_act": h2_act, "h3": h3, "h3_act": h3_act}
        return out

    def _forward_single(self, theta: float, omega: float, torque: float) -> tuple[float, float]:
        """单样本前向: (θ, ω, τ) → (θ̂, ω̂)。"""
        x = np.array([[theta, omega, torque]], dtype=np.float64)
        out = self._forward_batch(x)
        return float(out[0, 0]), float(out[0, 1])

    # ── 预测接口 ──

    def predict(
        self,
        state: WorldState,
        action: Action | None,
        n_steps: int = 1,
    ) -> list[WorldState]:
        """神经网络多步预测。"""
        from mci_world_model.sdk._world_state import PendulumAction, PendulumState

        if not isinstance(state, PendulumState):
            raise TypeError(f"PendulumNeuralPredictor 只能预测 PendulumState，收到 {type(state).__name__}")

        torque = action.torque if isinstance(action, PendulumAction) else 0.0
        trajectory = []
        current_theta, current_omega = state.theta, state.omega

        for _ in range(n_steps):
            theta_hat, omega_hat = self._forward_single(current_theta, current_omega, torque)
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

    # ── 训练: Mini-batch SGD ──

    @property
    def is_trained(self) -> bool:
        return self._trained

    def train(
        self,
        n_samples: int = 2000,
        noise_std: float = 0.0,
        lr: float = 0.001,
        n_epochs: int = 100,
        batch_size: int = 32,
    ) -> dict[str, Any]:
        """用 Mini-batch SGD 训练 MLP 预测器。

        使用 PendulumPhysicsPredictor 生成训练数据。

        Args:
            n_samples: 训练样本数
            noise_std: 输出噪声标准差
            lr: 学习率
            n_epochs: 训练轮数
            batch_size: 批大小

        Returns:
            训练报告
        """
        from mci_world_model.sdk._world_state import PendulumAction, PendulumState

        physics = PendulumPhysicsPredictor()
        rng = np.random.RandomState(42)

        # 生成训练数据
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

        # Mini-batch SGD
        for epoch in range(n_epochs):
            indices = np.arange(n_samples)
            rng.shuffle(indices)

            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start : start + batch_size]
                x_batch = X[batch_idx]
                y_batch = Y[batch_idx]

                # 前向
                h1 = x_batch @ self._W1 + self._b1
                h1_act = np.maximum(h1, 0)
                h2 = h1_act @ self._W2 + self._b2
                h2_act = np.maximum(h2, 0)
                h3 = h2_act @ self._W3 + self._b3
                h3_act = np.maximum(h3, 0)
                pred = h3_act @ self._W4 + self._b4

                # 反向传播
                bs = x_batch.shape[0]
                d_out = 2.0 * (pred - y_batch) / bs

                d_W4 = h3_act.T @ d_out
                d_b4 = d_out.sum(axis=0)

                d_h3_act = d_out @ self._W4.T
                d_h3 = d_h3_act * (h3 > 0).astype(np.float64)

                d_W3 = h2_act.T @ d_h3
                d_b3 = d_h3.sum(axis=0)

                d_h2_act = d_h3 @ self._W3.T
                d_h2 = d_h2_act * (h2 > 0).astype(np.float64)

                d_W2 = h1_act.T @ d_h2
                d_b2 = d_h2.sum(axis=0)

                d_h1_act = d_h2 @ self._W2.T
                d_h1 = d_h1_act * (h1 > 0).astype(np.float64)

                d_W1 = x_batch.T @ d_h1
                d_b1 = d_h1.sum(axis=0)

                # 梯度裁剪
                for g in [d_W1, d_b1, d_W2, d_b2, d_W3, d_b3, d_W4, d_b4]:
                    np.clip(g, -5, 5, out=g)

                # SGD 更新
                self._W1 -= lr * d_W1
                self._b1 -= lr * d_b1
                self._W2 -= lr * d_W2
                self._b2 -= lr * d_b2
                self._W3 -= lr * d_W3
                self._b3 -= lr * d_b3
                self._W4 -= lr * d_W4
                self._b4 -= lr * d_b4

        # 计算最终损失
        final_pred = self._forward_batch(X)
        self._train_loss = float(np.mean((final_pred - Y) ** 2))
        self._trained = True

        return {
            "final_loss": round(self._train_loss, 6),
            "n_samples": n_samples,
            "n_params": self.n_params,
            "converged": self._train_loss < 0.1,
            "method": "sgd_mlp",
        }


# =============================================================================
# CartPhysicsPredictor — 小车物理公式预测器 (v4.4.0 Phase 0)
# =============================================================================


class CartPhysicsPredictor(ActionConditionedPredictor):
    """小车物理公式预测器——利用已知物理定律做 ground truth 预测。

    v4.4.0 Phase 0: CEWM 架构泛化的第二种预测器验证器。
    证明 PlanAgent / MultiBranchPredictor / cewm_step() 不依赖 Pendulum 特有属性。

    物理定律（Euler 积分，F=ma, m=1kg）:
        x_{t+1} = x_t + v_t · dt
        v_{t+1} = v_t + force · dt

    使用:
        >>> pred = CartPhysicsPredictor()
        >>> state = CartState(x=0.0, v=1.0)
        >>> push = CartAction(force=2.0)
        >>> trajectory = pred.predict(state, action=push, n_steps=10)
        >>> # trajectory[0] 是 1 步后的物理正确状态
    """

    def __init__(self) -> None:
        super().__init__(name="cart_physics")

    def predict(
        self,
        state: WorldState,
        action: Action | None,
        n_steps: int = 1,
    ) -> list[WorldState]:
        """用物理公式做 Euler 积分多步预测。"""
        from mci_world_model.sdk._world_state import CartAction, CartState

        if not isinstance(state, CartState):
            raise TypeError(f"CartPhysicsPredictor 只能预测 CartState，收到 {type(state).__name__}")

        # 零阶保持: 同一个 action 施加 n_steps 次
        cart_action = action if isinstance(action, CartAction) else None

        trajectory = []
        current = state.copy()
        for _ in range(n_steps):
            if cart_action is not None:
                current = cart_action.apply(current)
            else:
                current = current.step_physics()
            trajectory.append(current)

        return trajectory
