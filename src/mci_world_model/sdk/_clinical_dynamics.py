"""MCI World Model — 临床动态学预测器（ClinicalDynamicsPredictor）

============================================================

Phase 1 核心模块：医疗世界模型的转移模型 T(s, a) → s'。

这是世界模型五要素中的第三个（转移 T），回答核心问题：
    "如果对患者施加动作 a，患者状态会如何变化？"

架构：
    PatientState.to_vector() ──┐
                               ├─→ LearnedDynamicsPredictor ──→ 预测向量 ──→ PatientState'
    MedicalAction.to_vector() ─┘    （JEPA 潜空间 MLP）         （from_vector 重建）

与 JEPAPhysicsAdapter 的关键区别：
    - JEPAPhysicsAdapter: 零动作假设（只预测自然演化），输入是原始矩阵
    - ClinicalDynamicsPredictor: 动作条件化（药物干预效应），输入是结构化状态+动作

训练数据来源（两种模式）：
    1. DRUG_EFFECT_TABLE 基线：从药效表生成 (s, a, s') 三元组（ground truth）
    2. 真实时序数据：从 MIMIC 波形切片学习（Phase 1 验证用）

设计原则：
    - 继承 ActionConditionedPredictor，融入已有 world model 闭环
    - 所有随机源设 seed（AGENTS.md 要求）
    - 无状态计算：不持久化训练数据（记忆归 su-memory-sdk）
    - 可审计：predict 返回完整审计信息
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mci_world_model.sdk._action_conditioned_predictor import ActionConditionedPredictor
from mci_world_model.sdk._clinical_world_state import (
    DRUG_EFFECT_TABLE,
    N_VITALS,
    STATE_VECTOR_DIM,
    VITAL_NAMES,
    VITAL_NORMAL_RANGES,
    MedicalAction,
    PatientState,
)

# 标准化常量：将体征值映射到 ~[-1, 1] 范围，消除量纲差异
# 用正常范围中点和半宽做标准化：normalized = (val - mid) / half_width
_VITAL_MID = np.array([(VITAL_NORMAL_RANGES[v][0] + VITAL_NORMAL_RANGES[v][1]) / 2.0 for v in VITAL_NAMES])
_VITAL_HALF = np.array([max((VITAL_NORMAL_RANGES[v][1] - VITAL_NORMAL_RANGES[v][0]) / 2.0, 1.0) for v in VITAL_NAMES])
# 检验值标准化基准（典型范围估计）
_LAB_SCALE = np.array([1.0, 0.5, 5.0, 3.0, 100.0, 2.0])  # K/Cr/WBC/Hb/Plt/Lac
# 动作 magnitude 标准化基准
_DOSE_SCALE = 10.0
from mci_world_model.sdk._learned_dynamics_predictor import LearnedDynamicsPredictor
from mci_world_model.sdk._world_state import Action, WorldState

# =============================================================================
# ClinicalDynamicsPredictor — 临床动态学预测器
# =============================================================================


@dataclass
class UncertainPrediction:
    """带不确定性量化的预测结果。

    Attributes:
        point_estimates: 点估计的 PatientState 序列（每步一个）。
        ci_lower: 每步每个体征的 95% CI 下界，shape (n_steps, N_VITALS)。
        ci_upper: 每步每个体征的 95% CI 上界，shape (n_steps, N_VITALS)。
        std: 每步每个体征的预测标准差，shape (n_steps, N_VITALS)。
        n_bootstrap: 实际完成的 bootstrap 次数。
    """

    point_estimates: list[PatientState]
    ci_lower: list[np.ndarray]
    ci_upper: list[np.ndarray]
    std: list[np.ndarray]
    n_bootstrap: int = 0

    def uncertainty_score(self, step: int = 0) -> float:
        """计算指定步数的平均不确定性分数（CI 宽度归一化）。

        Args:
            step: 步数索引。

        Returns:
            平均 CI 宽度占正常范围的比例 ∈ [0, ∞)。越小越确定。
        """
        if step >= len(self.ci_lower):
            return 1.0
        widths = self.ci_upper[step] - self.ci_lower[step]
        spans = np.array([hi - lo for lo, hi in VITAL_NORMAL_RANGES.values()])
        return float(np.mean(widths / np.maximum(spans, 1e-6)))

    def to_dict(self, step: int = 0) -> dict[str, Any]:
        """序列化指定步数的预测+CI（审计用）。"""
        if step >= len(self.point_estimates):
            return {}
        point = self.point_estimates[step]
        return {
            "step": step,
            "point_estimate": point.to_dict(),
            "ci_lower": {name: round(float(self.ci_lower[step][i]), 2) for i, name in enumerate(VITAL_NAMES)},
            "ci_upper": {name: round(float(self.ci_upper[step][i]), 2) for i, name in enumerate(VITAL_NAMES)},
            "uncertainty_score": round(self.uncertainty_score(step), 4),
            "n_bootstrap": self.n_bootstrap,
        }


class ClinicalDynamicsPredictor(ActionConditionedPredictor):
    """临床动态学预测器 — 医疗世界模型的转移模型 T。

    给定患者状态和临床干预动作，预测未来患者状态。

    核心接口（继承 ActionConditionedPredictor）：
        predict(state, action, n_steps) → [PatientState', ...]  多步预测
        rollout(state, actions) → [PatientState', ...]          动作序列推演

    训练接口：
        fit_from_effect_table(n_samples, n_epochs)  从药效基线表训练
        fit_from_trajectories(trajectories)         从真实时序轨迹训练

    Example:
        >>> predictor = ClinicalDynamicsPredictor(seed=42)
        >>> predictor.fit_from_effect_table(n_samples=1000, n_epochs=300)
        >>> state = PatientState(vital_signs=np.array([[75, 120, 80, 98, 16, 36.8, 15]]))
        >>> action = MedicalAction(target="dopamine", magnitude=5.0)
        >>> preds = predictor.predict(state, action, n_steps=3)
        >>> print(preds[0])  # 预测的下一时刻患者状态
    """

    def __init__(
        self,
        state_dim: int = STATE_VECTOR_DIM,
        action_dim: int = 11,  # 4 type + 6 drug + 1 magnitude
        hidden_dim: int = 128,
        seed: int = 42,
    ) -> None:
        """初始化临床动态学预测器。

        Args:
            state_dim: 状态向量维度（默认 STATE_VECTOR_DIM = 7 体征 + 6 检验 = 13）。
            action_dim: 动作向量维度（默认 5 = 4 onehot + 1 magnitude）。
            hidden_dim: MLP 隐藏层维度。
            seed: 随机种子（可复现性）。
        """
        super().__init__(name="clinical_dynamics")
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._seed = seed
        self._fitted = False

        # 底层 JEPA 学习器（纯 NumPy MLP，无 GPU 依赖）
        self._learner = LearnedDynamicsPredictor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            seed=seed,
        )

    @property
    def is_fitted(self) -> bool:
        """是否已完成训练。"""
        return self._fitted

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    # ── ActionConditionedPredictor 契约 ────────────────────────────

    def predict(
        self,
        state: WorldState,
        action: Action | None,
        n_steps: int = 1,
    ) -> list[WorldState]:
        """动作条件化多步预测。

        Args:
            state: 当前 PatientState。
            action: MedicalAction（None = 无干预，自然演化）。
            n_steps: 预测步数。

        Returns:
            预测的未来 PatientState 序列 [s', s'', ...]。
        """
        if not isinstance(state, PatientState):
            raise TypeError(f"需要 PatientState，收到 {type(state)}")

        s_vec = self._normalize_state(state.to_vector())
        a_vec = (
            self._normalize_action(action.to_vector())
            if isinstance(action, MedicalAction)
            else np.zeros(self._action_dim, dtype=np.float64)
        )

        # 多步链式预测（每步施加相同动作 = 零阶保持）
        trajectory_vecs = self._learner.predict(s_vec, a_vec, n_steps=n_steps)

        # 向量 → PatientState 重建
        results: list[WorldState] = []
        for vec in trajectory_vecs:
            vec = np.asarray(vec, dtype=np.float64).ravel()
            # 反标准化：潜空间输出 → 原始量纲
            vec = self._denormalize_state(vec)
            # 物理约束：裁剪到生理可行范围
            vec = self._clip_to_feasible(vec)
            predicted_state = PatientState.from_vector(vec)
            # 保留原状态的元信息
            predicted_state.patient_id = state.patient_id
            predicted_state.age = state.age
            predicted_state.gender = state.gender
            predicted_state.diagnoses = list(state.diagnoses)
            results.append(predicted_state)

        return results

    # ── 训练接口 ────────────────────────────────────────────────────

    def fit_from_effect_table(
        self,
        n_samples: int = 1000,
        n_epochs: int = 300,
        lr: float = 0.01,
        noise_std: float = 0.5,
    ) -> dict[str, Any]:
        """从药效基线表生成训练数据并训练。

        生成逻辑：
            1. 随机采样患者状态（体征在正常范围内波动）
            2. 随机选择药物 + 剂量作为动作
            3. 用 DRUG_EFFECT_TABLE 计算 ground truth 下一状态
            4. 训练 MLP 学习 (s, a) → s' 映射

        Args:
            n_samples: 训练样本数。
            n_epochs: 训练轮数。
            lr: 学习率。
            noise_std: ground truth 添加的高斯噪声标准差。

        Returns:
            训练信息 {"final_loss", "n_samples", "n_epochs", "converged"}。
        """
        rng = np.random.default_rng(self._seed)

        # 生成 (s, a, s') 三元组
        states, actions, next_states = self._generate_effect_table_data(rng, n_samples, noise_std)

        # 逐样本 SGD 训练
        losses: list[float] = []
        for epoch in range(n_epochs):
            indices = rng.permutation(n_samples)
            epoch_loss = 0.0
            for idx in indices:
                s_vec = states[idx]
                a_vec = actions[idx]
                target = next_states[idx]
                self._learner.training_forward(s_vec, a_vec)
                result = self._learner.compute_gradients(target)
                self._learner.apply_gradients(result["grads"], lr=lr)
                epoch_loss += result["mse"]
            epoch_loss /= n_samples
            losses.append(epoch_loss)

        final_loss = losses[-1] if losses else 1.0
        self._fitted = True
        return {
            "final_loss": round(float(final_loss), 6),
            "n_samples": n_samples,
            "n_epochs": n_epochs,
            "converged": final_loss < 0.05,
        }

    def fit_from_trajectories(
        self,
        trajectories: list[np.ndarray],
        n_epochs: int = 300,
        lr: float = 0.01,
    ) -> dict[str, Any]:
        """从真实时序轨迹训练（MIMIC 波形数据用）。

        每条轨迹是 (T, V) 体征矩阵，按时间窗切分为 (s_t, s_{t+1}) 对。
        动作向量置零（真实数据中动作信息需额外提供）。

        Args:
            trajectories: 患者体征时序轨迹列表，每条 shape (T, V)。
            n_epochs: 训练轮数。
            lr: 学习率。

        Returns:
            训练信息。
        """
        rng = np.random.default_rng(self._seed)

        # 从轨迹切分 (s_t, s_{t+1}) 对
        states: list[np.ndarray] = []
        next_states: list[np.ndarray] = []
        zero_action = np.zeros(self._action_dim, dtype=np.float64)

        for traj in trajectories:
            traj = np.asarray(traj, dtype=np.float64)
            if traj.ndim != 2 or traj.shape[0] < 2:
                continue
            for t in range(traj.shape[0] - 1):
                # 截取/补零到 N_VITALS 列（处理任意列数的轨迹）
                cur_window = traj[t : t + 1]
                next_window = traj[t + 1 : t + 2]
                # NaN 填充为 0（训练阶段简化处理）
                cur_window = np.nan_to_num(cur_window, nan=0.0)
                next_window = np.nan_to_num(next_window, nan=0.0)
                # 重塑到 (1, N_VITALS)
                cur_vitals = np.zeros((1, N_VITALS))
                next_vitals = np.zeros((1, N_VITALS))
                n_copy = min(cur_window.shape[1], N_VITALS)
                cur_vitals[0, :n_copy] = cur_window[0, :n_copy]
                next_vitals[0, :n_copy] = next_window[0, :n_copy]
                s = PatientState(vital_signs=cur_vitals)
                s_next = PatientState(vital_signs=next_vitals)
                states.append(s.to_vector())
                next_states.append(s_next.to_vector())

        n_samples = len(states)
        if n_samples == 0:
            self._fitted = True
            return {"final_loss": 1.0, "n_samples": 0, "converged": False}

        states_arr = np.array(states, dtype=np.float64)
        next_arr = np.array(next_states, dtype=np.float64)

        # 预计算标准化数据
        norm_states = np.array([self._normalize_state(s) for s in states_arr])
        norm_next = np.array([self._normalize_state(s) for s in next_arr])

        losses: list[float] = []
        for _epoch in range(n_epochs):
            indices = rng.permutation(n_samples)
            epoch_loss = 0.0
            for idx in indices:
                self._learner.training_forward(norm_states[idx], zero_action)
                result = self._learner.compute_gradients(norm_next[idx])
                self._learner.apply_gradients(result["grads"], lr=lr)
                epoch_loss += result["mse"]
            epoch_loss /= n_samples
            losses.append(epoch_loss)

        final_loss = losses[-1] if losses else 1.0
        self._fitted = True
        return {
            "final_loss": round(float(final_loss), 6),
            "n_samples": n_samples,
            "n_epochs": n_epochs,
            "converged": final_loss < 0.05,
        }

    # ── 评估接口 ────────────────────────────────────────────────────

    def evaluate_direction_accuracy(
        self,
        test_cases: list[tuple[PatientState, MedicalAction, PatientState]],
    ) -> dict[str, float]:
        """评估预测的趋势方向准确率。

        对每个 (state, action, true_next) 三元组，检查预测的体征变化方向
        是否与真实方向一致（升/降/平）。

        Args:
            test_cases: [(state, action, true_next), ...] 测试三元组。

        Returns:
            {"direction_accuracy": float, "mae": float, "n": int}
        """
        correct = 0
        total_mae = 0.0
        for state, action, true_next in test_cases:
            preds = self.predict(state, action, n_steps=1)
            pred_next = preds[0]
            # 比较每个体征的变化方向
            for i, vname in enumerate(VITAL_NAMES):
                s_val = state.vital_signs[-1][i]
                true_delta = true_next.vital_signs[-1][i] - s_val
                pred_delta = pred_next.vital_signs[-1][i] - s_val
                # 方向一致判断（同号或都接近零）
                true_sign = 1 if true_delta > 0.5 else (-1 if true_delta < -0.5 else 0)
                pred_sign = 1 if pred_delta > 0.5 else (-1 if pred_delta < -0.5 else 0)
                if true_sign == pred_sign:
                    correct += 1
                total_mae += abs(true_delta - pred_delta)

        total_comparisons = len(test_cases) * N_VITALS
        return {
            "direction_accuracy": round(correct / max(total_comparisons, 1), 4),
            "mae": round(total_mae / max(total_comparisons, 1), 4),
            "n": len(test_cases),
        }

    # ── 不确定性量化（贝叶斯 bootstrap）──────────────────────────

    def predict_with_uncertainty(
        self,
        state: PatientState,
        action: MedicalAction | None = None,
        n_steps: int = 1,
        n_bootstrap: int = 50,
        seed: int = 42,
    ) -> UncertainPrediction:
        """带不确定性量化的预测（贝叶斯 bootstrap）。

        对同一 (state, action) 做 n_bootstrap 次扰动预测，
        通过给输入状态添加微小高斯噪声模拟认知不确定性，
        用预测分布的百分位计算 95% 置信区间。

        Args:
            state: 当前 PatientState。
            action: MedicalAction（None = 无干预）。
            n_steps: 预测步数。
            n_bootstrap: bootstrap 重采样次数（默认 50）。
            seed: 随机种子。

        Returns:
            UncertainPrediction（含点估计、95% CI、不确定性分数）。
        """
        rng = np.random.default_rng(seed)

        # 点估计（无扰动）
        point_preds = self.predict(state, action, n_steps=n_steps)

        # Bootstrap 扰动预测
        boot_preds: list[list[np.ndarray]] = []
        state_vec = self._normalize_state(state.to_vector())
        for _ in range(n_bootstrap):
            noise = rng.normal(0, 0.05, size=state_vec.shape)
            perturbed_vec = self._denormalize_state(state_vec + noise)
            perturbed_vec = self._clip_to_feasible(perturbed_vec)
            perturbed_state = PatientState.from_vector(perturbed_vec)
            perturbed_state.patient_id = state.patient_id
            try:
                preds = self.predict(perturbed_state, action, n_steps=n_steps)
                boot_preds.append([p.vital_signs[-1].copy() for p in preds])
            except (ValueError, RuntimeError):
                continue

        # 计算每步每个体征的置信区间
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

    # ── 标准化 ──────────────────────────────────────────────────

    @staticmethod
    def _normalize_state(vec: np.ndarray) -> np.ndarray:
        """将状态向量标准化到 ~[-1, 1] 范围。"""
        result = vec.copy().astype(np.float64)
        # 体征部分：减中点除半宽
        result[:N_VITALS] = (result[:N_VITALS] - _VITAL_MID) / _VITAL_HALF
        # 检验部分：除以尺度
        lab_start = N_VITALS
        lab_end = min(N_VITALS + len(_LAB_SCALE), len(result))
        result[lab_start:lab_end] = result[lab_start:lab_end] / _LAB_SCALE[: lab_end - lab_start]
        return result

    @staticmethod
    def _denormalize_state(vec: np.ndarray) -> np.ndarray:
        """将标准化向量还原为原始量纲。"""
        result = vec.copy().astype(np.float64)
        result[:N_VITALS] = result[:N_VITALS] * _VITAL_HALF + _VITAL_MID
        lab_start = N_VITALS
        lab_end = min(N_VITALS + len(_LAB_SCALE), len(result))
        result[lab_start:lab_end] = result[lab_start:lab_end] * _LAB_SCALE[: lab_end - lab_start]
        return result

    @staticmethod
    def _normalize_action(vec: np.ndarray) -> np.ndarray:
        """将动作向量标准化。onehot 不变，magnitude（最后一维）除以 _DOSE_SCALE。"""
        result = vec.copy().astype(np.float64)
        if len(result) > 4:
            result[-1] = result[-1] / _DOSE_SCALE  # magnitude 在最后一维
        return result

    # ── 内部方法 ────────────────────────────────────────────────────

    def _generate_effect_table_data(
        self,
        rng: np.random.Generator,
        n_samples: int,
        noise_std: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """从 DRUG_EFFECT_TABLE 生成训练三元组。"""
        drugs = list(DRUG_EFFECT_TABLE.keys())
        states = np.zeros((n_samples, self._state_dim), dtype=np.float64)
        actions = np.zeros((n_samples, self._action_dim), dtype=np.float64)
        next_states = np.zeros((n_samples, self._state_dim), dtype=np.float64)

        for i in range(n_samples):
            # 随机患者状态（正常范围波动 + 噪声）
            vitals = np.zeros(N_VITALS, dtype=np.float64)
            for j, vname in enumerate(VITAL_NAMES):
                lo, hi = VITAL_NORMAL_RANGES[vname]
                vitals[j] = rng.uniform(lo, hi)
            states[i, :N_VITALS] = vitals
            # 检验值置零（训练阶段不关注）

            # 随机药物 + 剂量
            drug = drugs[rng.integers(len(drugs))]
            dose = rng.uniform(1.0, 10.0)
            action = MedicalAction(target=drug, magnitude=dose)
            actions[i] = action.to_vector()

            # Ground truth: 用 apply() 计算
            patient = PatientState(vital_signs=vitals.reshape(1, -1))
            true_next = action.apply(patient)
            next_vec = true_next.to_vector()
            # 加噪声
            next_vec += rng.normal(0, noise_std, size=next_vec.shape)
            next_states[i] = next_vec

        # 标准化训练数据（消除量纲差异，防止梯度爆炸）
        for i in range(n_samples):
            states[i] = self._normalize_state(states[i])
            next_states[i] = self._normalize_state(next_states[i])
            actions[i] = self._normalize_action(actions[i])
        return states, actions, next_states

    @staticmethod
    def _clip_to_feasible(vec: np.ndarray) -> np.ndarray:
        """将预测向量裁剪到生理可行范围。"""
        result = vec.copy()
        for i, vname in enumerate(VITAL_NAMES):
            if i >= len(result):
                break
            lo, hi = VITAL_NORMAL_RANGES[vname]
            # 放宽到 2 倍范围（允许异常但合理的预测值）
            margin = (hi - lo) * 2.0
            result[i] = np.clip(result[i], lo - margin, hi + margin)
        return result
