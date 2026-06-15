"""
MCI World Model v5.0.0 — Learned Dynamics Predictor
=====================================================

可学习动力学预测器：从 (state, action) → next_state 的观测数据中学习动力学映射。

架构:
    Input (state_dim + action_dim) → Linear(128) → ReLU → Linear(64) → ReLU → Linear(state_dim)
    4 层全连接 MLP, ~20K 参数, 手写梯度 SGD

训练:
    数据来源: DynamicsDataGenerator 使用 PhysicsPredictor 生成训练数据
    损失: MSE(predicted_state, ground_truth_state)
    优化: 手写梯度反向传播 → SGD

关键设计:
    - 继承 ActionConditionedPredictor 接口
    - n 步自回归推演（上一步输出作为下一步输入）
    - teacher forcing 训练模式
    - 与 PendulumPhysicsPredictor 并行，可互相切换

用法:
    from mci_world_model.sdk._learned_dynamics_predictor import LearnedDynamicsPredictor

    predictor = LearnedDynamicsPredictor(state_dim=2, action_dim=1)
    trajectory = predictor.predict(state, action, n_steps=5)
    predictor.train_step(state_vec, action_vec, next_state_vec, lr=0.01)
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from mci_world_model.sdk._action_conditioned_predictor import ActionConditionedPredictor

logger = logging.getLogger(__name__)


class LearnedDynamicsPredictor(ActionConditionedPredictor):
    """
    可学习动力学预测器：从数据学习 (state, action) → next_state 映射。

    Example:
        >>> pred = LearnedDynamicsPredictor(state_dim=2, action_dim=1)
        >>> # 训练...
        >>> from mci_world_model.sdk._world_state import PendulumState, PendulumAction
        >>> state = PendulumState(theta=0.5, omega=0.0)
        >>> action = PendulumAction(torque=1.0)
        >>> trajectory = pred.predict(state, action, n_steps=5)
    """

    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 1,
        hidden_dim: int = 128,
        seed: int = 42,
        l2_reg: float = 0.0,
    ):
        super().__init__(name="learned_dynamics")
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if action_dim < 0:
            raise ValueError(f"action_dim must be non-negative, got {action_dim}")

        self._state_dim = state_dim
        self._action_dim = action_dim
        self._hidden_dim = hidden_dim
        self._l2_reg = l2_reg
        self._rng = np.random.RandomState(seed)

        input_dim = state_dim + action_dim
        H = hidden_dim

        # ── 参数初始化 (Xavier/Glorot) ──
        self.W1: np.ndarray = self._rng.randn(input_dim, H).astype(np.float64) * np.sqrt(2.0 / (input_dim + H))
        self.b1: np.ndarray = np.zeros(H, dtype=np.float64)
        self.W2: np.ndarray = self._rng.randn(H, H // 2).astype(np.float64) * np.sqrt(2.0 / (H + H // 2))
        self.b2: np.ndarray = np.zeros(H // 2, dtype=np.float64)
        self.W3: np.ndarray = self._rng.randn(H // 2, state_dim).astype(np.float64) * np.sqrt(
            2.0 / (H // 2 + state_dim)
        )
        self.b3: np.ndarray = np.zeros(state_dim, dtype=np.float64)

        # ── 前向缓存 ──
        self._cache: dict[str, Any] = {}
        self._cache_lock = threading.Lock()

        # ── 训练统计 ──
        self._train_steps: int = 0

    # =====================================================================
    # 属性
    # =====================================================================

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def train_steps(self) -> int:
        return self._train_steps

    @property
    def n_params(self) -> int:
        return sum(p.size for p in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])

    # =====================================================================
    # ActionConditionedPredictor 接口
    # =====================================================================

    def predict(
        self,
        state,
        action=None,
        n_steps: int = 1,
    ) -> list:
        """
        动作条件化多步自回归预测。

        Args:
            state: 当前世界状态 (WorldState 或 ndarray)
            action: 施加的动作 (Action 或 ndarray)
            n_steps: 预测步数

        Returns:
            预测的未来状态列表
        """
        state_vec = self._to_state_vector(state)
        action_vec = self._to_action_vector(action)

        trajectory = []
        current_vec = state_vec.copy()

        for step in range(n_steps):
            next_vec = self._forward_inference(current_vec, action_vec)
            next_state = self._from_state_vector(next_vec, state)
            trajectory.append(next_state)
            current_vec = next_vec

        return trajectory

    # =====================================================================
    # 前向传播
    # =====================================================================

    def _forward_inference(self, state_vec: np.ndarray, action_vec: np.ndarray) -> np.ndarray:
        """推理模式前向传播。"""
        x = np.concatenate([state_vec, action_vec]).astype(np.float64)
        h1 = x @ self.W1 + self.b1
        a1 = np.maximum(h1, 0.0)  # ReLU
        h2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(h2, 0.0)  # ReLU
        next_state = a2 @ self.W3 + self.b3
        return next_state

    def training_forward(self, state_vec: np.ndarray, action_vec: np.ndarray) -> np.ndarray:
        """
        训练模式前向传播（缓存中间值）。

        Args:
            state_vec: shape (state_dim,)
            action_vec: shape (action_dim,)

        Returns:
            预测的下一状态向量 shape (state_dim,)
        """
        state_vec = np.asarray(state_vec, dtype=np.float64).ravel()
        action_vec = np.asarray(action_vec, dtype=np.float64).ravel()

        x = np.concatenate([state_vec, action_vec])
        h1 = x @ self.W1 + self.b1
        a1 = np.maximum(h1, 0.0)
        h2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(h2, 0.0)
        pred = a2 @ self.W3 + self.b3

        with self._cache_lock:
            self._cache = {
                "x_input": x,
                "state_input": state_vec,
                "action_input": action_vec,
                "h1": h1,
                "a1": a1,
                "h2": h2,
                "a2": a2,
                "pred": pred,
            }

        return pred.copy()

    # =====================================================================
    # 梯度计算
    # =====================================================================

    def compute_gradients(self, target_state: np.ndarray) -> dict[str, Any]:
        """
        计算 MSE 损失 + 手写反向传播梯度。

        L = mean((pred - target)^2) + l2_reg * sum(||W||^2)

        Args:
            target_state: 目标下一状态 shape (state_dim,)

        Returns:
            {"loss": float, "mse": float, "l2": float, "grads": {...}}
        """
        with self._cache_lock:
            cache = self._cache.copy()

        if not cache:
            return {"loss": 0.0, "mse": 0.0, "l2": 0.0, "grads": self._zero_grads()}

        target = np.asarray(target_state, dtype=np.float64).ravel()
        pred = cache["pred"]
        diff = pred - target
        D = self._state_dim

        # MSE
        mse = float(np.mean(diff**2))

        # L2
        l2 = 0.0
        if self._l2_reg > 0:
            for W in [self.W1, self.W2, self.W3]:
                l2 += float(np.sum(W**2))
            l2 *= self._l2_reg

        loss = mse + l2

        # ── 反向传播 ──
        dpred = (2.0 / D) * diff  # (state_dim,)

        # Layer 3: pred = a2 @ W3 + b3
        dW3 = np.outer(cache["a2"], dpred)
        db3 = dpred
        da2 = dpred @ self.W3.T  # (H//2,)

        # ReLU backward
        dh2 = da2 * (cache["h2"] > 0).astype(np.float64)

        # Layer 2: h2 = a1 @ W2 + b2
        dW2 = np.outer(cache["a1"], dh2)
        db2 = dh2
        da1 = dh2 @ self.W2.T  # (H,)

        # ReLU backward
        dh1 = da1 * (cache["h1"] > 0).astype(np.float64)

        # Layer 1: h1 = x @ W1 + b1
        dW1 = np.outer(cache["x_input"], dh1)
        db1 = dh1

        # L2 正则梯度
        if self._l2_reg > 0:
            dW1 += 2.0 * self._l2_reg * self.W1
            dW2 += 2.0 * self._l2_reg * self.W2
            dW3 += 2.0 * self._l2_reg * self.W3

        grads = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
            "W3": dW3,
            "b3": db3,
        }

        return {"loss": loss, "mse": mse, "l2": l2, "grads": grads}

    def apply_gradients(self, grads: dict[str, np.ndarray], lr: float = 0.01) -> None:
        """应用梯度更新参数 (SGD)。"""
        for name in ["W1", "b1", "W2", "b2", "W3", "b3"]:
            if name in grads:
                param = getattr(self, name)
                grad = np.asarray(grads[name], dtype=np.float64)
                if param.shape != grad.shape:
                    raise ValueError(f"Shape mismatch for {name}: {param.shape} vs {grad.shape}")
                setattr(self, name, param - lr * grad)
        self._train_steps += 1

    # =====================================================================
    # 训练
    # =====================================================================

    def train_step(
        self,
        state_vec: np.ndarray,
        action_vec: np.ndarray,
        target_vec: np.ndarray,
        lr: float = 0.01,
    ) -> float:
        """单步训练。"""
        self.training_forward(state_vec, action_vec)
        result = self.compute_gradients(target_vec)
        self.apply_gradients(result["grads"], lr=lr)
        return result["mse"]

    def train_on_dataset(
        self,
        dataset: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        n_epochs: int = 100,
        lr: float = 0.01,
    ) -> dict[str, Any]:
        """在数据集上训练多轮。"""
        loss_history: list[float] = []

        for epoch in range(n_epochs):
            indices = np.random.permutation(len(dataset))
            epoch_losses: list[float] = []

            for idx in indices:
                s, a, s_next = dataset[idx]
                loss = self.train_step(s, a, s_next, lr=lr)
                epoch_losses.append(loss)

            avg_loss = float(np.mean(epoch_losses))
            loss_history.append(avg_loss)

            if (epoch + 1) % 50 == 0:
                logger.info("Dynamics Epoch %d/%d | MSE: %.6f", epoch + 1, n_epochs, avg_loss)

        return {
            "final_loss": loss_history[-1] if loss_history else 0.0,
            "loss_history": loss_history,
            "n_epochs": n_epochs,
        }

    # =====================================================================
    # 评估
    # =====================================================================

    def evaluate_physics(
        self,
        physics_predictor: ActionConditionedPredictor,
        test_states: list,
        n_steps: int = 1,
    ) -> dict[str, float]:
        """
        对比 Learned vs Physics 的预测误差。

        Args:
            physics_predictor: 物理公式预测器 (ground truth)
            test_states: 测试状态列表
            n_steps: 预测步数

        Returns:
            {"learned_mse": float, "learned_mae": float, "n": int}
        """
        errors_sq = []
        errors_abs = []

        for state in test_states:
            action = None  # 无外力
            learned_traj = self.predict(state, action, n_steps=n_steps)
            physics_traj = physics_predictor.predict(state, action, n_steps=n_steps)

            for l_state, p_state in zip(learned_traj, physics_traj):
                l_vec = self._to_state_vector(l_state)
                p_vec = self._to_state_vector(p_state)
                errors_sq.append(float(np.mean((l_vec - p_vec) ** 2)))
                errors_abs.append(float(np.mean(np.abs(l_vec - p_vec))))

        return {
            "learned_mse": float(np.mean(errors_sq)) if errors_sq else float("inf"),
            "learned_mae": float(np.mean(errors_abs)) if errors_abs else float("inf"),
            "n": len(errors_sq),
        }

    # =====================================================================
    # 序列化
    # =====================================================================

    def save_params(self, path: str) -> None:
        """保存参数到 .npz。"""
        params = {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "W3": self.W3,
            "b3": self.b3,
            "state_dim": np.array([self._state_dim]),
            "action_dim": np.array([self._action_dim]),
            "train_steps": np.array([self._train_steps]),
        }
        np.savez_compressed(path, **params)

    def load_params(self, path: str) -> None:
        """从 .npz 加载参数。"""
        data = np.load(path)
        if "state_dim" in data and int(data["state_dim"][0]) != self._state_dim:
            raise ValueError(f"state_dim mismatch: model={self._state_dim}, file={int(data['state_dim'][0])}")
        params = {k: v for k, v in data.items() if k not in ("state_dim", "action_dim", "train_steps")}
        for name in ["W1", "b1", "W2", "b2", "W3", "b3"]:
            if name in params:
                setattr(self, name, np.asarray(params[name], dtype=np.float64))
        if "train_steps" in data:
            self._train_steps = int(data["train_steps"][0])

    # =====================================================================
    # 辅助方法
    # =====================================================================

    def _to_state_vector(self, state) -> np.ndarray:
        """将 WorldState 转换为向量。"""
        if isinstance(state, np.ndarray):
            return state.astype(np.float64).ravel()
        if hasattr(state, "to_vector"):
            vec = state.to_vector()
            return np.asarray(vec, dtype=np.float64).ravel()
        raise TypeError(f"Cannot convert {type(state)} to state vector")

    def _to_action_vector(self, action) -> np.ndarray:
        """将 Action 转换为向量。"""
        if action is None:
            return np.zeros(self._action_dim, dtype=np.float64)
        if isinstance(action, np.ndarray):
            return action.astype(np.float64).ravel()
        if hasattr(action, "to_vector"):
            vec = action.to_vector()
            return np.asarray(vec, dtype=np.float64).ravel()
        if isinstance(action, (int, float)):
            return np.array([float(action)], dtype=np.float64)
        return np.zeros(self._action_dim, dtype=np.float64)

    def _from_state_vector(self, vec: np.ndarray, template_state):
        """从向量重建 WorldState。"""
        if isinstance(template_state, np.ndarray):
            return vec.copy()
        if hasattr(template_state, "from_vector"):
            try:
                return template_state.from_vector(vec)
            except Exception:
                return vec.copy()
        return vec.copy()

    def _zero_grads(self) -> dict[str, np.ndarray]:
        return {
            "W1": np.zeros_like(self.W1),
            "b1": np.zeros_like(self.b1),
            "W2": np.zeros_like(self.W2),
            "b2": np.zeros_like(self.b2),
            "W3": np.zeros_like(self.W3),
            "b3": np.zeros_like(self.b3),
        }

    def __repr__(self) -> str:
        return (
            f"LearnedDynamicsPredictor(state_dim={self._state_dim}, "
            f"action_dim={self._action_dim}, n_params={self.n_params}, "
            f"train_steps={self._train_steps})"
        )


# =============================================================================
# DynamicsDataGenerator — 使用 PhysicsPredictor 生成训练数据
# =============================================================================


class DynamicsDataGenerator:
    """
    动力学训练数据生成器。

    使用硬编码的 PhysicsPredictor 生成 (s_t, a_t, s_{t+1}) 三元组。
    支持 PendulumState, CartState。
    """

    def __init__(self, state_type: str = "pendulum", seed: int = 42):
        """
        Args:
            state_type: "pendulum" 或 "cart"
            seed: 随机种子
        """
        self._state_type = state_type
        self._rng = np.random.RandomState(seed)

    def generate_dataset(
        self,
        n_trajectories: int = 100,
        steps_per_trajectory: int = 50,
        noise_std: float = 0.01,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        生成训练数据集。

        Args:
            n_trajectories: 轨迹数量
            steps_per_trajectory: 每条轨迹步数
            noise_std: 添加的观测噪声标准差

        Returns:
            [(state_vec, action_vec, next_state_vec), ...] 列表
        """
        dataset: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        if self._state_type == "pendulum":
            from mci_world_model.sdk._world_state import PendulumAction, PendulumState

            predictor_pendulum = __import__(
                "mci_world_model.sdk._action_conditioned_predictor",
                fromlist=["PendulumPhysicsPredictor"],
            ).PendulumPhysicsPredictor()

            for _ in range(n_trajectories):
                theta = self._rng.uniform(-np.pi / 4, np.pi / 4)
                omega = self._rng.uniform(-2.0, 2.0)
                state = PendulumState(theta=theta, omega=omega)

                for _ in range(steps_per_trajectory):
                    torque = self._rng.uniform(-2.0, 2.0)
                    action = PendulumAction(torque=torque)

                    next_states = predictor_pendulum.predict(state, action, n_steps=1)
                    if next_states:
                        s_vec = np.array([state.theta, state.omega], dtype=np.float64)
                        a_vec = np.array([torque], dtype=np.float64)
                        ns = next_states[0]
                        ns_vec = np.array([ns.theta, ns.omega], dtype=np.float64)

                        # 添加观测噪声
                        if noise_std > 0:
                            ns_vec += self._rng.randn(*ns_vec.shape) * noise_std

                        dataset.append((s_vec, a_vec, ns_vec))
                        state = ns

        elif self._state_type == "cart":
            from mci_world_model.sdk._action_conditioned_predictor import CartPhysicsPredictor
            from mci_world_model.sdk._world_state import CartAction, CartState

            predictor_cart = CartPhysicsPredictor()

            for _ in range(n_trajectories):
                x = self._rng.uniform(-2.0, 2.0)
                v = self._rng.uniform(-1.0, 1.0)
                state = CartState(x=x, v=v)

                for _ in range(steps_per_trajectory):
                    force = self._rng.uniform(-5.0, 5.0)
                    action = CartAction(force=force)

                    next_states = predictor_cart.predict(state, action, n_steps=1)
                    if next_states:
                        s_vec = np.array([state.x, state.v], dtype=np.float64)
                        a_vec = np.array([force], dtype=np.float64)
                        ns = next_states[0]
                        ns_vec = np.array([ns.x, ns.v], dtype=np.float64)

                        if noise_std > 0:
                            ns_vec += self._rng.randn(*ns_vec.shape) * noise_std

                        dataset.append((s_vec, a_vec, ns_vec))
                        state = ns

        else:
            raise ValueError(f"Unsupported state_type: {self._state_type}")

        return dataset
