"""MCI World Model — 时间感知临床动态学预测器（TemporalClinicalDynamicsPredictor）

============================================================

D6 升级模块：在 ClinicalDynamicsPredictor（无状态 MLP）基础上引入时间记忆。

为什么需要时间感知？
    原 ClinicalDynamicsPredictor 的底层是 LearnedDynamicsPredictor（纯 MLP），
    每步预测独立：s_{t+1} = f(s_t, a_t)，**无隐状态**。
    这导致它无法学习"药物起效延迟"、"体征惯性回落"等时序依赖，
    在波形自然演化（零动作）场景方向准确率仅 ~47%。

时间感知方案：
    本模块用简单 vanilla RNN（纯 NumPy，无 GPU 依赖）替代 MLP：
        h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b_h)
        s_{t+1} = W_hy · h_t + b_y

    其中 x_t = [state_t, action_t]，h 是隐状态（跨时间步传递）。

    训练：BPTT 截断到 1 步（等价于带隐状态初始化的 SGD），
    避免长序列梯度消失/爆炸。

设计原则（与 ClinicalDynamicsPredictor 一致）：
    - 继承 ActionConditionedPredictor，可互换使用
    - 所有随机源设 seed
    - 无持久化（记忆归 su-memory-sdk）
    - 数值健壮：隐状态 tanh 裁剪到 [-1,1]，梯度全局范数裁剪
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mci_world_model.sdk._action_conditioned_predictor import ActionConditionedPredictor
from mci_world_model.sdk._clinical_dynamics import (
    _DOSE_SCALE,
    _LAB_SCALE,
    _VITAL_HALF,
    _VITAL_MID,
    UncertainPrediction,
)
from mci_world_model.sdk._clinical_world_state import (
    DRUG_EFFECT_TABLE,
    N_VITALS,
    STATE_VECTOR_DIM,
    VITAL_NAMES,
    VITAL_NORMAL_RANGES,
    MedicalAction,
    PatientState,
)
from mci_world_model.sdk._world_state import Action, WorldState


class _SimpleRNNCore:
    """纯 NumPy vanilla RNN 核心（隐状态跨时间步传递）。

    架构：
        输入 x = [state(state_dim) + action(action_dim)]
        隐状态 h（hidden_dim）跨时间步传递
        h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b_h)
        y_t = h_t @ W_hy + b_y
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        hidden_dim: int = 64,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self._seed = seed
        rng = np.random.default_rng(seed)
        # Xavier 初始化（数值稳定）
        scale_xh = np.sqrt(2.0 / max(input_dim + hidden_dim, 1))
        scale_hy = np.sqrt(2.0 / max(hidden_dim + state_dim, 1))
        self.W_xh = rng.normal(0, scale_xh, size=(input_dim, hidden_dim))
        self.W_hh = np.eye(hidden_dim) * 0.9  # 对角初始化（接近恒等映射，缓解梯度消失）
        self.b_h = np.zeros(hidden_dim)
        self.W_hy = rng.normal(0, scale_hy, size=(hidden_dim, state_dim))
        self.b_y = np.zeros(state_dim)
        self._train_steps = 0

    def forward(
        self,
        x_seq: np.ndarray,
        h0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """前向传播整个序列，返回 (输出序列, h序列, tanh前a序列, x序列缓存)。

        Args:
            x_seq: 输入序列 shape (T, input_dim)。
            h0: 初始隐状态（None 时置零）。

        Returns:
            (y_seq, hs, a_pre, xs) 用于 BPTT。
        """
        T = x_seq.shape[0]
        if h0 is None:
            h_prev = np.zeros(self.hidden_dim)
        else:
            h_prev = np.asarray(h0, dtype=np.float64).copy()
        xs: list[np.ndarray] = []
        a_pre: list[np.ndarray] = []  # tanh 前的值
        hs: list[np.ndarray] = [h_prev.copy()]
        ys = np.zeros((T, self.state_dim))
        for t in range(T):
            x_t = x_seq[t]
            a = x_t @ self.W_xh + h_prev @ self.W_hh + self.b_h
            h = np.tanh(a)  # 隐状态裁剪到 [-1, 1]
            y = h @ self.W_hy + self.b_y
            xs.append(x_t)
            a_pre.append(a)
            hs.append(h.copy())
            ys[t] = y
            h_prev = h
        return ys, hs, a_pre, xs

    def predict_seq(
        self,
        x0: np.ndarray,
        n_steps: int,
        h0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """自回归预测：重复输入 x0，传递隐状态（零阶保持动作）。

        Args:
            x0: 初始输入向量（单步）shape (input_dim,)。
            n_steps: 预测步数。
            h0: 初始隐状态。

        Returns:
            (输出序列 (n_steps, state_dim), 最终隐状态)。
        """
        if h0 is None:
            h_prev = np.zeros(self.hidden_dim)
        else:
            h_prev = np.asarray(h0, dtype=np.float64).copy()
        ys = np.zeros((n_steps, self.state_dim))
        x_t = np.asarray(x0, dtype=np.float64)
        for t in range(n_steps):
            a = x_t @ self.W_xh + h_prev @ self.W_hh + self.b_h
            h = np.tanh(a)
            y = h @ self.W_hy + self.b_y
            ys[t] = y
            h_prev = h
        return ys, h_prev

    def bptt_step(
        self,
        x_t: np.ndarray,
        h_prev: np.ndarray,
        h_t: np.ndarray,
        a_t: np.ndarray,
        target: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """单步 BPTT（截断到 1 步）：计算该时间步的梯度。

        损失 L = 0.5 * ||y - target||^2 / state_dim
        """
        y = h_t @ self.W_hy + self.b_y
        diff = y - target
        D = self.state_dim
        # 输出层梯度
        dW_hy = np.outer(h_t, diff) / D
        db_y = diff / D
        dh = diff @ self.W_hy.T  # (hidden_dim,)
        # tanh 反向
        da = dh * (1.0 - h_t**2)
        # 隐层梯度（截断：不继续往 h_prev 反传，用恒等近似）
        dW_xh = np.outer(x_t, da)
        dW_hh = np.outer(h_prev, da)
        db_h = da
        return {
            "W_xh": dW_xh,
            "W_hh": dW_hh,
            "b_h": db_h,
            "W_hy": dW_hy,
            "b_y": db_y,
        }

    def apply_grads(self, grads: dict[str, np.ndarray], lr: float) -> None:
        """应用梯度（含全局范数裁剪 + NaN 防护）。"""
        names = ["W_xh", "W_hh", "b_h", "W_hy", "b_y"]
        norm_sq = 0.0
        bad = False
        for n in names:
            if n in grads:
                g = np.asarray(grads[n], dtype=np.float64)
                if not np.all(np.isfinite(g)):
                    bad = True
                    break
                norm_sq += float(np.sum(g * g))
        if bad:
            self._train_steps += 1
            return
        max_norm = 10.0
        norm = norm_sq**0.5
        scale = max_norm / norm if (norm > max_norm and norm > 0) else 1.0
        for n in names:
            if n in grads:
                cur = getattr(self, n)
                g = np.asarray(grads[n], dtype=np.float64)
                setattr(self, n, cur - lr * scale * g)
        self._train_steps += 1


class TemporalClinicalDynamicsPredictor(ActionConditionedPredictor):
    """时间感知临床动态学预测器 — 带 RNN 隐状态的转移模型 T。

    与 ClinicalDynamicsPredictor 的区别：
        - 后者：MLP 无状态，每步独立，无法学习时序依赖
        - 本类：RNN 带隐状态，跨时间步传递，能学习药物延迟效应/体征惯性

    接口与 ClinicalDynamicsPredictor 兼容（可互换）：
        predict(state, action, n_steps) → [PatientState', ...]
        fit_from_effect_table(...) / fit_from_trajectories(...)
        predict_with_uncertainty(...)
        evaluate_direction_accuracy(...)

    Example:
        >>> predictor = TemporalClinicalDynamicsPredictor(seed=42)
        >>> predictor.fit_from_effect_table(n_samples=500, n_epochs=100)
        >>> state = PatientState(vital_signs=np.array([[75,120,80,98,16,36.8,15]]))
        >>> action = MedicalAction(target="dopamine", magnitude=5.0)
        >>> preds = predictor.predict(state, action, n_steps=3)
    """

    def __init__(
        self,
        state_dim: int = STATE_VECTOR_DIM,
        action_dim: int = 11,
        hidden_dim: int = 64,
        seed: int = 42,
    ) -> None:
        super().__init__(name="temporal_clinical_dynamics")
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._hidden_dim = hidden_dim
        self._seed = seed
        self._fitted = False
        input_dim = state_dim + action_dim
        self._rnn = _SimpleRNNCore(
            input_dim=input_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            seed=seed,
        )

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    # ── 标准化（与 ClinicalDynamicsPredictor 一致）──────────────

    @staticmethod
    def _normalize_state(vec: np.ndarray) -> np.ndarray:
        result = vec.copy().astype(np.float64)
        result[:N_VITALS] = (result[:N_VITALS] - _VITAL_MID) / _VITAL_HALF
        lab_start = N_VITALS
        lab_end = min(N_VITALS + len(_LAB_SCALE), len(result))
        result[lab_start:lab_end] = result[lab_start:lab_end] / _LAB_SCALE[: lab_end - lab_start]
        return result

    @staticmethod
    def _denormalize_state(vec: np.ndarray) -> np.ndarray:
        result = vec.copy().astype(np.float64)
        result[:N_VITALS] = result[:N_VITALS] * _VITAL_HALF + _VITAL_MID
        lab_start = N_VITALS
        lab_end = min(N_VITALS + len(_LAB_SCALE), len(result))
        result[lab_start:lab_end] = result[lab_start:lab_end] * _LAB_SCALE[: lab_end - lab_start]
        return result

    @staticmethod
    def _normalize_action(vec: np.ndarray) -> np.ndarray:
        result = vec.copy().astype(np.float64)
        if len(result) > 4:
            result[-1] = result[-1] / _DOSE_SCALE
        return result

    @staticmethod
    def _clip_to_feasible(vec: np.ndarray) -> np.ndarray:
        result = vec.copy()
        for i, vname in enumerate(VITAL_NAMES):
            if i >= len(result):
                break
            lo, hi = VITAL_NORMAL_RANGES[vname]
            margin = (hi - lo) * 2.0
            result[i] = np.clip(result[i], lo - margin, hi + margin)
        return result

    # ── ActionConditionedPredictor 契约 ────────────────────────────

    def predict(
        self,
        state: WorldState,
        action: Action | None,
        n_steps: int = 1,
    ) -> list[WorldState]:
        """动作条件化多步预测（RNN 隐状态跨步传递）。"""
        if not isinstance(state, PatientState):
            raise TypeError(f"需要 PatientState，收到 {type(state)}")

        s_vec = self._normalize_state(state.to_vector())
        a_vec = (
            self._normalize_action(action.to_vector())
            if isinstance(action, MedicalAction)
            else np.zeros(self._action_dim, dtype=np.float64)
        )
        x0 = np.concatenate([s_vec, a_vec])
        y_seq, _ = self._rnn.predict_seq(x0, n_steps=n_steps)

        results: list[WorldState] = []
        for vec in y_seq:
            vec = np.asarray(vec, dtype=np.float64).ravel()
            vec = self._denormalize_state(vec)
            vec = self._clip_to_feasible(vec)
            ps = PatientState.from_vector(vec)
            ps.patient_id = state.patient_id
            ps.age = state.age
            ps.gender = state.gender
            ps.diagnoses = list(state.diagnoses)
            results.append(ps)
        return results

    # ── 训练接口 ────────────────────────────────────────────────────

    def fit_from_effect_table(
        self,
        n_samples: int = 1000,
        n_epochs: int = 300,
        lr: float = 0.01,
        noise_std: float = 0.5,
    ) -> dict[str, Any]:
        """从药效基线表训练 RNN（单步预测监督）。"""
        rng = np.random.default_rng(self._seed)
        states, actions, next_states = self._generate_effect_table_data(rng, n_samples, noise_std)
        losses: list[float] = []
        for _epoch in range(n_epochs):
            indices = rng.permutation(n_samples)
            epoch_loss = 0.0
            for idx in indices:
                s_vec = states[idx]
                a_vec = actions[idx]
                target = next_states[idx]
                x_t = np.concatenate([s_vec, a_vec]).reshape(1, -1)
                # 单步前向（h0=0）
                ys, hs, a_pre, xs = self._rnn.forward(x_t, h0=np.zeros(self._hidden_dim))
                # 单步 BPTT
                grads = self._rnn.bptt_step(
                    x_t=xs[0],
                    h_prev=hs[0],
                    h_t=hs[1],
                    a_t=a_pre[0],
                    target=target,
                )
                self._rnn.apply_grads(grads, lr=lr)
                diff = ys[0] - target
                epoch_loss += float(np.dot(diff, diff) / self._state_dim)
            epoch_loss /= n_samples
            losses.append(epoch_loss)
        final_loss = losses[-1] if losses else 1.0
        self._fitted = True
        return {
            "final_loss": round(float(final_loss), 6),
            "n_samples": n_samples,
            "n_epochs": n_epochs,
            "converged": final_loss < 0.05,
            "backend": "rnn",
        }

    def fit_from_trajectories(
        self,
        trajectories: list[np.ndarray],
        n_epochs: int = 300,
        lr: float = 0.01,
    ) -> dict[str, Any]:
        """从真实时序轨迹训练 RNN（真正利用时序依赖）。

        与 ClinicalDynamicsPredictor 的关键区别：本方法把整条轨迹作为序列输入，
        RNN 隐状态在轨迹内传递，学习时序动态。
        """
        rng = np.random.default_rng(self._seed)
        zero_action = np.zeros(self._action_dim, dtype=np.float64)
        # 构造序列样本：每条轨迹作为一个序列
        seqs_x: list[np.ndarray] = []
        seqs_y: list[np.ndarray] = []
        for traj in trajectories:
            traj = np.asarray(traj, dtype=np.float64)
            if traj.ndim != 2 or traj.shape[0] < 2:
                continue
            traj = np.nan_to_num(traj, nan=0.0)
            T = traj.shape[0]
            n_copy = min(traj.shape[1], N_VITALS)
            # 构造 (T-1, input_dim) 输入和 (T-1, state_dim) 目标
            x_seq = np.zeros((T - 1, self._state_dim + self._action_dim))
            y_seq = np.zeros((T - 1, self._state_dim))
            for t in range(T - 1):
                cur = np.zeros(N_VITALS)
                nxt = np.zeros(N_VITALS)
                cur[:n_copy] = traj[t, :n_copy]
                nxt[:n_copy] = traj[t + 1, :n_copy]
                s = self._normalize_state(PatientState(vital_signs=cur.reshape(1, -1)).to_vector())
                sn = self._normalize_state(PatientState(vital_signs=nxt.reshape(1, -1)).to_vector())
                x_seq[t] = np.concatenate([s, zero_action])
                y_seq[t] = sn
            seqs_x.append(x_seq)
            seqs_y.append(y_seq)

        n_seqs = len(seqs_x)
        if n_seqs == 0:
            self._fitted = True
            return {"final_loss": 1.0, "n_samples": 0, "converged": False, "backend": "rnn"}

        total_steps = sum(s.shape[0] for s in seqs_x)
        losses: list[float] = []
        for _epoch in range(n_epochs):
            order = rng.permutation(n_seqs)
            epoch_loss = 0.0
            for si in order:
                x_seq = seqs_x[si]
                y_seq = seqs_y[si]
                T = x_seq.shape[0]
                # 前向整条序列（隐状态跨步传递）
                ys, hs, a_pre, xs = self._rnn.forward(x_seq, h0=np.zeros(self._hidden_dim))
                # 逐步 BPTT（截断到 1 步，用前一步真实 h）
                for t in range(T):
                    grads = self._rnn.bptt_step(
                        x_t=xs[t],
                        h_prev=hs[t],
                        h_t=hs[t + 1],
                        a_t=a_pre[t],
                        target=y_seq[t],
                    )
                    self._rnn.apply_grads(grads, lr=lr)
                    diff = ys[t] - y_seq[t]
                    epoch_loss += float(np.dot(diff, diff) / self._state_dim)
            epoch_loss /= max(total_steps, 1)
            losses.append(epoch_loss)
        final_loss = losses[-1] if losses else 1.0
        self._fitted = True
        return {
            "final_loss": round(float(final_loss), 6),
            "n_samples": total_steps,
            "n_epochs": n_epochs,
            "converged": final_loss < 0.05,
            "backend": "rnn",
        }

    def _generate_effect_table_data(
        self,
        rng: np.random.Generator,
        n_samples: int,
        noise_std: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """从 DRUG_EFFECT_TABLE 生成训练三元组（与 MLP 版一致）。"""
        drugs = list(DRUG_EFFECT_TABLE.keys())
        states = np.zeros((n_samples, self._state_dim), dtype=np.float64)
        actions = np.zeros((n_samples, self._action_dim), dtype=np.float64)
        next_states = np.zeros((n_samples, self._state_dim), dtype=np.float64)
        for i in range(n_samples):
            vitals = np.zeros(N_VITALS, dtype=np.float64)
            for j, vname in enumerate(VITAL_NAMES):
                lo, hi = VITAL_NORMAL_RANGES[vname]
                vitals[j] = rng.uniform(lo, hi)
            states[i, :N_VITALS] = vitals
            drug = drugs[rng.integers(len(drugs))]
            dose = rng.uniform(1.0, 10.0)
            action = MedicalAction(target=drug, magnitude=dose)
            actions[i] = action.to_vector()
            patient = PatientState(vital_signs=vitals.reshape(1, -1))
            true_next = action.apply(patient)
            next_vec = true_next.to_vector()
            next_vec += rng.normal(0, noise_std, size=next_vec.shape)
            next_states[i] = next_vec
        for i in range(n_samples):
            states[i] = self._normalize_state(states[i])
            next_states[i] = self._normalize_state(next_states[i])
            actions[i] = self._normalize_action(actions[i])
        return states, actions, next_states

    def evaluate_direction_accuracy(
        self,
        test_cases: list[tuple[PatientState, MedicalAction, PatientState]],
    ) -> dict[str, Any]:
        """评估预测的趋势方向准确率（与 MLP 版接口一致）。"""
        correct = 0
        total_mae = 0.0
        for state, action, true_next in test_cases:
            preds = self.predict(state, action, n_steps=1)
            pred_next = preds[0]
            for i, _vname in enumerate(VITAL_NAMES):
                s_val = state.vital_signs[-1][i]
                true_delta = true_next.vital_signs[-1][i] - s_val
                pred_delta = pred_next.vital_signs[-1][i] - s_val
                true_sign = 1 if true_delta > 0.5 else (-1 if true_delta < -0.5 else 0)
                pred_sign = 1 if pred_delta > 0.5 else (-1 if pred_delta < -0.5 else 0)
                if true_sign == pred_sign:
                    correct += 1
                total_mae += abs(true_delta - pred_delta)
        total = len(test_cases) * N_VITALS
        return {
            "direction_accuracy": round(correct / max(total, 1), 4),
            "mae": round(total_mae / max(total, 1), 4),
            "n": len(test_cases),
            "backend": "rnn",
        }

    def predict_with_uncertainty(
        self,
        state: PatientState,
        action: MedicalAction | None = None,
        n_steps: int = 1,
        n_bootstrap: int = 50,
        seed: int = 42,
    ) -> UncertainPrediction:
        """带不确定性量化的预测（贝叶斯 bootstrap）。"""
        rng = np.random.default_rng(seed)
        point_preds = self.predict(state, action, n_steps=n_steps)
        boot_preds: list[list[np.ndarray]] = []
        state_vec = self._normalize_state(state.to_vector())
        for _ in range(n_bootstrap):
            noise = rng.normal(0, 0.05, size=state_vec.shape)
            perturbed_vec = self._denormalize_state(state_vec + noise)
            perturbed_vec = self._clip_to_feasible(perturbed_vec)
            ps = PatientState.from_vector(perturbed_vec)
            ps.patient_id = state.patient_id
            try:
                preds = self.predict(ps, action, n_steps=n_steps)
                boot_preds.append([p.vital_signs[-1].copy() for p in preds])
            except (ValueError, RuntimeError):
                continue
        ci_lower_steps: list[np.ndarray] = []
        ci_upper_steps: list[np.ndarray] = []
        std_steps: list[np.ndarray] = []
        for step in range(n_steps):
            if not boot_preds or step >= len(boot_preds[0]):
                ci_lower_steps.append(point_preds[step].vital_signs[-1].copy())
                ci_upper_steps.append(point_preds[step].vital_signs[-1].copy())
                std_steps.append(np.zeros(N_VITALS))
                continue
            step_preds = np.array([bp[step] for bp in boot_preds if step < len(bp)])
            ci_lower_steps.append(np.percentile(step_preds, 2.5, axis=0))
            ci_upper_steps.append(np.percentile(step_preds, 97.5, axis=0))
            std_steps.append(np.std(step_preds, axis=0))
        return UncertainPrediction(
            point_estimates=point_preds,
            ci_lower=ci_lower_steps,
            ci_upper=ci_upper_steps,
            std=std_steps,
            n_bootstrap=len(boot_preds),
        )
