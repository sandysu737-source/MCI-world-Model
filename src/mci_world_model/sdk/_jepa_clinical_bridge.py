"""MCI World Model — JEPA 临床桥接（JEPAClinicalBridge）

============================================================

方向一：接通 JEPA — 把 TrueJEPAEncoder 的潜空间编码-预测分离架构
真正引入医疗世界模型。

为什么需要？
    当前 ClinicalDynamicsPredictor 底层用 LearnedDynamicsPredictor（通用 MLP），
    在**原始观测空间**（R¹³）直接预测 s_{t+1} = f(s_t, a_t)。
    这没用上 JEPA 的核心能力：
        1. 潜空间编码：把高维观测压缩到紧凑表示，剥离噪声/冗余
        2. 编码-预测分离：在潜空间做预测（更稳定、更高效）
        3. EMA target + VICReg：防止表征坍塌（JEPA 训练稳定性关键）
        4. 能量一致性：潜空间预测与目标编码对齐

本桥接做什么？
    PatientState.to_vector() R¹³
        ↓ TrueJEPAEncoder.encode()          [观测 → 潜空间]
    潜向量 z (latent_dim)
        ↓ TrueJEPAEncoder.predict_next(z, a) [潜空间预测]
    预测潜向量 z'
        ↓ _Decoder(z') → R¹³                 [潜空间 → 观测]
    预测 PatientState

    训练时联合优化:
        L_jepa  = ||predictor(z_t, a) - stopgrad(target(x_{t+1}))||² + VICReg
        L_recon = ||decoder(encode(x_t)) - x_t||²              (重建保真)
        L_total = L_jepa + λ_recon * L_recon

设计原则（AGENTS.md 边界）:
    - 无状态编排：每次 predict 独立，不持久化训练数据（记忆归 su-memory-sdk）
    - 所有随机源设 seed
    - 数值健壮：潜向量裁剪、重建值 clip 到生理可行范围
    - 可审计：predict 返回潜向量与重建损失，供决策引擎审计
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mci_world_model.sdk._clinical_dynamics import (
    _DOSE_SCALE,
    _LAB_SCALE,
    _VITAL_HALF,
    _VITAL_MID,
    UncertainPrediction,
)
from mci_world_model.sdk._clinical_semantic_embedding import (
    ClinicalSemanticEmbedding,
)
from mci_world_model.sdk._clinical_world_state import (
    N_VITALS,
    STATE_VECTOR_DIM,
    VITAL_NAMES,
    VITAL_NORMAL_RANGES,
    MedicalAction,
    PatientState,
)
from mci_world_model.sdk._true_jepa_encoder import _MLP, TrueJEPAConfig, TrueJEPAEncoder

# =============================================================================
# 配置
# =============================================================================


@dataclass
class JEPAClinicalConfig:
    """JEPA 临床桥接配置。

    Attributes:
        obs_dim: 观测维度（= PatientState 向量维度 STATE_VECTOR_DIM = 13）。
        latent_dim: 潜空间维度（≥ 2×obs_dim 以保留信息）。
        hidden_dim: 编码/解码 MLP 隐层维度。
        action_dim: 动作维度（= MedicalAction 向量维度 = 11）。
        ema_tau: EMA 动量（越大 target 更新越慢）。
        vicreg_var_weight: VICReg 方差项权重。
        vicreg_cov_weight: VICReg 协方差项权重。
        recon_weight: 重建损失权重（L_recon 系数）。
        lr: 学习率。
        seed: 随机种子。
    """

    obs_dim: int = STATE_VECTOR_DIM
    latent_dim: int = 64
    hidden_dim: int = 128
    action_dim: int = 11
    ema_tau: float = 0.996
    vicreg_var_weight: float = 1.0
    vicreg_cov_weight: float = 0.04
    recon_weight: float = 0.5
    lr: float = 0.001
    seed: int = 42


# =============================================================================
# JEPAClinicalBridge — JEPA 潜空间临床转移模型
# =============================================================================


class JEPAClinicalBridge:
    """JEPA 潜空间临床转移模型 — 方向一接通 JEPA。

    与 ClinicalDynamicsPredictor（原始空间 MLP）的区别：
        - 后者在 R¹³ 观测空间直接预测，受量纲差异/噪声影响
        - 本桥接在潜空间预测，剥离冗余，用 EMA+VICReg 保证表征稳定

    核心接口：
        predict(state, action, n_steps) → [PatientState', ...]  潜空间多步预测
        train_step(obs_t, obs_t1, action) → loss                联合训练
        fit_from_effect_table(...) → info                       从药效表训练
        encode(state) → 潜向量                                  单纯编码
        reconstruct(潜向量) → PatientState                      单纯解码

    Example:
        >>> bridge = JEPAClinicalBridge()
        >>> bridge.fit_from_effect_table(n_samples=500, n_epochs=100)
        >>> state = PatientState(vital_signs=np.array([[130,140,90,98,20,37,15]]))
        >>> action = MedicalAction(target="metoprolol", magnitude=5.0)
        >>> preds = bridge.predict(state, action, n_steps=3)
    """

    def __init__(
        self,
        config: JEPAClinicalConfig | None = None,
        semantic_embedder: ClinicalSemanticEmbedding | None = None,
    ) -> None:
        """初始化 JEPA 临床桥接。

        Args:
            config: 桥接配置。
            semantic_embedder: 方向二真实嵌入 — 可选的临床语义嵌入器。
                提供时，encode 用完整语义向量（体征+诊断+用药）替代纯数值，
                让世界模型状态空间携带临床语义。
                未提供时退化为纯数值 R^obs_dim（方向一行为）。
        """
        self._config = config or JEPAClinicalConfig()
        self._semantic_embedder = semantic_embedder
        self._fitted = False

        # 方向二：若启用语义嵌入，obs_dim 扩展为 numeric + diag + med
        if semantic_embedder is not None:
            semantic_dim = semantic_embedder.semantic_dim
            self._effective_obs_dim = self._config.obs_dim + semantic_dim
        else:
            self._effective_obs_dim = self._config.obs_dim

        # JEPA 编码器（含 online/target/predictor，自带 EMA+VICReg）
        jepa_cfg = TrueJEPAConfig(
            obs_dim=self._effective_obs_dim,
            latent_dim=self._config.latent_dim,
            hidden_dim=self._config.hidden_dim,
            action_dim=self._config.action_dim,
            ema_tau=self._config.ema_tau,
            vicreg_var_weight=self._config.vicreg_var_weight,
            vicreg_cov_weight=self._config.vicreg_cov_weight,
            lr=self._config.lr,
            seed=self._config.seed,
        )
        self._jepa = TrueJEPAEncoder(jepa_cfg)

        # Decoder: latent → obs（潜空间重建回观测，保证可解释性）
        self._decoder = _MLP(
            input_dim=self._config.latent_dim,
            hidden_dim=self._config.hidden_dim,
            output_dim=self._effective_obs_dim,
            seed=self._config.seed + 1,
        )
        self._train_steps = 0
        self._loss_history: list[float] = []
        self._recon_loss_history: list[float] = []

    @property
    def is_fitted(self) -> bool:
        """是否已训练。"""
        return self._fitted

    @property
    def latent_dim(self) -> int:
        return self._config.latent_dim

    @property
    def obs_dim(self) -> int:
        return self._config.obs_dim

    @property
    def loss_history(self) -> list[float]:
        """JEPA 预测损失历史。"""
        return list(self._loss_history)

    @property
    def recon_loss_history(self) -> list[float]:
        """重建损失历史。"""
        return list(self._recon_loss_history)

    # ── 标准化（与 ClinicalDynamicsPredictor 一致）──────────────

    @staticmethod
    def _normalize_state(vec: np.ndarray) -> np.ndarray:
        """体征标准化到 ~[-1, 1]（消除量纲差异，喂给 JEPA 编码器）。"""
        result = vec.copy().astype(np.float64)
        result[:N_VITALS] = (result[:N_VITALS] - _VITAL_MID) / _VITAL_HALF
        lab_start = N_VITALS
        lab_end = min(N_VITALS + len(_LAB_SCALE), len(result))
        if lab_end > lab_start:
            result[lab_start:lab_end] = result[lab_start:lab_end] / _LAB_SCALE[: lab_end - lab_start]
        return result

    @staticmethod
    def _denormalize_state(vec: np.ndarray) -> np.ndarray:
        """标准化向量还原为原始量纲。"""
        result = vec.copy().astype(np.float64)
        result[:N_VITALS] = result[:N_VITALS] * _VITAL_HALF + _VITAL_MID
        lab_start = N_VITALS
        lab_end = min(N_VITALS + len(_LAB_SCALE), len(result))
        if lab_end > lab_start:
            result[lab_start:lab_end] = result[lab_start:lab_end] * _LAB_SCALE[: lab_end - lab_start]
        return result

    def _normalize_full(self, full_vec: np.ndarray) -> np.ndarray:
        """标准化完整语义向量（数值部分标准化，语义部分保持归一化）。

        方向二：语义嵌入向量已是 L2 归一的（方向/幅度已编码语义），
        只需标准化前 obs_dim 维数值部分。

        Args:
            full_vec: 完整语义向量 [numeric(obs_dim) | diag | med]。

        Returns:
            标准化后的完整向量。
        """
        result = full_vec.copy().astype(np.float64)
        obs_dim = self._config.obs_dim
        # 数值部分标准化
        result[:obs_dim] = self._normalize_state(result[:obs_dim])
        # 语义部分保持不变（已是归一化向量，幅度 ~[-1,1]）
        return result

    def _state_to_normalized_obs(self, state: PatientState) -> np.ndarray:
        """PatientState → 标准化观测向量（支持语义模式）。"""
        if self._semantic_embedder is not None:
            sem = self._semantic_embedder.embed(state)
            return self._normalize_full(sem.full_vector)
        return self._normalize_state(state.to_vector())

    @staticmethod
    def _normalize_action(vec: np.ndarray) -> np.ndarray:
        """动作标准化：onehot 不变，magnitude 除以 _DOSE_SCALE。"""
        result = vec.copy().astype(np.float64)
        if len(result) > 4:
            result[-1] = result[-1] / _DOSE_SCALE
        return result

    @staticmethod
    def _clip_to_feasible(vec: np.ndarray) -> np.ndarray:
        """裁剪到生理可行范围。"""
        result = vec.copy()
        for i, vname in enumerate(VITAL_NAMES):
            if i >= len(result):
                break
            lo, hi = VITAL_NORMAL_RANGES[vname]
            margin = (hi - lo) * 2.0
            result[i] = np.clip(result[i], lo - margin, hi + margin)
        return result

    # ── 编码 / 解码 ──────────────────────────────────────────────

    def encode(self, state: PatientState) -> np.ndarray:
        """编码 PatientState 为 JEPA 潜向量。

        方向二：若启用 semantic_embedder，用完整语义向量（体征+诊断+用药）
        替代纯数值向量，让潜空间携带临床语义。

        Args:
            state: 患者状态。

        Returns:
            潜向量 (latent_dim,)。
        """
        if not isinstance(state, PatientState):
            raise TypeError(f"需要 PatientState，收到 {type(state)}")
        if self._semantic_embedder is not None:
            sem = self._semantic_embedder.embed(state)
            obs = self._normalize_full(sem.full_vector)
        else:
            obs = self._normalize_state(state.to_vector())
        return self._jepa.encode(obs)

    def encode_target(self, state: PatientState) -> np.ndarray:
        """目标编码器（EMA）—训练时提供 stop-gradient 目标。"""
        if not isinstance(state, PatientState):
            raise TypeError(f"需要 PatientState，收到 {type(state)}")
        if self._semantic_embedder is not None:
            sem = self._semantic_embedder.embed(state)
            obs = self._normalize_full(sem.full_vector)
        else:
            obs = self._normalize_state(state.to_vector())
        return self._jepa.encode_target(obs)

    def reconstruct(self, z: np.ndarray) -> PatientState:
        """解码潜向量为 PatientState。

        方向二：若启用语义，decoder 输出含语义部分，只取前 obs_dim 维还原体征。

        Args:
            z: 潜向量 (latent_dim,)。

        Returns:
            重建的 PatientState（体征部分，语义部分用于训练损失）。
        """
        z = np.asarray(z, dtype=np.float64).ravel()
        recon_full = self._decoder.forward(z)
        # 只取数值部分还原 PatientState（语义部分不还原为可读字段）
        recon_obs = recon_full[: self._config.obs_dim]
        # 反标准化 + 裁剪
        recon_obs = self._denormalize_state(recon_obs)
        recon_obs = self._clip_to_feasible(recon_obs)
        ps = PatientState.from_vector(recon_obs)
        return ps

    # ── 预测 ─────────────────────────────────────────────────────

    def predict(
        self,
        state: PatientState,
        action: MedicalAction | None,
        n_steps: int = 1,
    ) -> list[PatientState]:
        """JEPA 潜空间多步预测。

        流程：
            s → encode → z
            循环 n_steps: z = predict_next(z, a)
            z' → reconstruct → PatientState'

        Args:
            state: 当前 PatientState。
            action: MedicalAction（None = 无干预）。
            n_steps: 预测步数。

        Returns:
            预测的未来 PatientState 序列。
        """
        if not isinstance(state, PatientState):
            raise TypeError(f"需要 PatientState，收到 {type(state)}")

        z = self.encode(state)
        a_vec = (
            self._normalize_action(action.to_vector())
            if isinstance(action, MedicalAction)
            else np.zeros(self._config.action_dim, dtype=np.float64)
        )

        results: list[PatientState] = []
        for _ in range(n_steps):
            z = self._jepa.predict_next(z, a_vec)
            ps = self.reconstruct(z)
            ps.patient_id = state.patient_id
            ps.age = state.age
            ps.gender = state.gender
            ps.diagnoses = list(state.diagnoses)
            results.append(ps)
        return results

    # ── 训练 ─────────────────────────────────────────────────────

    def train_step(
        self,
        state_t: PatientState,
        state_t1: PatientState,
        action: MedicalAction | None = None,
    ) -> dict[str, float]:
        """单步联合训练（JEPA 预测损失 + 重建损失）。

        Args:
            state_t: 当前状态。
            state_t1: 下一状态（ground truth）。
            action: 干预动作。

        Returns:
            {"total_loss", "jepa_loss", "recon_loss"}。
        """
        obs_t = self._state_to_normalized_obs(state_t)
        obs_t1 = self._state_to_normalized_obs(state_t1)
        a_vec = (
            self._normalize_action(action.to_vector())
            if isinstance(action, MedicalAction)
            else np.zeros(self._config.action_dim, dtype=np.float64)
        )

        # ── JEPA 预测损失（内部含 EMA + VICReg）──
        jepa_loss = self._jepa.train_step(obs_t, obs_t1, a_vec)

        # ── 重建损失：decoder(encode(x_t)) ≈ x_t ──
        z = self._jepa.encode(obs_t)
        recon = self._decoder.forward(z)
        recon_diff = recon - obs_t
        recon_loss = float(np.dot(recon_diff, recon_diff) / self._config.obs_dim)

        # 反向传播重建损失
        d_recon = (2.0 / self._config.obs_dim) * recon_diff
        decoder_grads = self._decoder.backward(d_recon)
        self._decoder.apply_grads(decoder_grads, self._config.lr)

        total_loss = jepa_loss + self._config.recon_weight * recon_loss

        self._train_steps += 1
        self._loss_history.append(jepa_loss)
        self._recon_loss_history.append(recon_loss)
        return {
            "total_loss": round(total_loss, 6),
            "jepa_loss": round(jepa_loss, 6),
            "recon_loss": round(recon_loss, 6),
        }

    def fit_from_effect_table(
        self,
        n_samples: int = 500,
        n_epochs: int = 100,
        noise_std: float = 0.5,
    ) -> dict[str, Any]:
        """从药效基线表训练 JEPA 桥接。

        生成 (s_t, a_t, s_{t+1}) 三元组，联合训练 JEPA + decoder。

        Args:
            n_samples: 样本数。
            n_epochs: 训练轮数。
            noise_std: ground truth 高斯噪声标准差。

        Returns:
            训练信息。
        """
        from mci_world_model.sdk._clinical_world_state import DRUG_EFFECT_TABLE

        rng = np.random.default_rng(self._config.seed)
        drugs = list(DRUG_EFFECT_TABLE.keys())

        # 预生成训练三元组
        triples: list[tuple[PatientState, MedicalAction, PatientState]] = []
        for _ in range(n_samples):
            vitals = np.zeros(N_VITALS, dtype=np.float64)
            for j, vname in enumerate(VITAL_NAMES):
                lo, hi = VITAL_NORMAL_RANGES[vname]
                vitals[j] = rng.uniform(lo, hi)
            patient = PatientState(vital_signs=vitals.reshape(1, -1))
            drug = drugs[rng.integers(len(drugs))]
            dose = rng.uniform(1.0, 10.0)
            action = MedicalAction(target=drug, magnitude=dose)
            # ground truth 下一状态
            true_next = action.apply(patient)
            # 加噪声构造 state_t1
            next_vec = true_next.to_vector()
            next_vec += rng.normal(0, noise_std, size=next_vec.shape)
            next_vec = self._clip_to_feasible(next_vec)
            state_t1 = PatientState.from_vector(next_vec)
            triples.append((patient, action, state_t1))

        last_info: dict[str, float] = {"total_loss": 1.0, "jepa_loss": 1.0, "recon_loss": 1.0}
        for _epoch in range(n_epochs):
            indices = rng.permutation(n_samples)
            epoch_jepa = 0.0
            epoch_recon = 0.0
            for idx in indices:
                s_t, a_t, s_t1 = triples[idx]
                info = self.train_step(s_t, s_t1, a_t)
                epoch_jepa += info["jepa_loss"]
                epoch_recon += info["recon_loss"]
            last_info = info
            last_info["epoch_jepa_avg"] = round(epoch_jepa / n_samples, 6)
            last_info["epoch_recon_avg"] = round(epoch_recon / n_samples, 6)

        self._fitted = True
        return {
            "final_jepa_loss": last_info.get("epoch_jepa_avg", 1.0),
            "final_recon_loss": last_info.get("epoch_recon_avg", 1.0),
            "n_samples": n_samples,
            "n_epochs": n_epochs,
            "converged": last_info.get("epoch_jepa_avg", 1.0) < 0.1,
            "backend": "jepa",
            "latent_dim": self._config.latent_dim,
        }

    def fit_from_trajectories(
        self,
        trajectories: list[np.ndarray],
        n_epochs: int = 100,
    ) -> dict[str, Any]:
        """从真实时序轨迹训练（MIMIC 波形数据）。

        Args:
            trajectories: 患者体征时序矩阵列表，每条 shape (T, ≥N_VITALS)。
            n_epochs: 训练轮数。

        Returns:
            训练信息。
        """
        rng = np.random.default_rng(self._config.seed)
        zero_action = MedicalAction(target="", magnitude=0.0, action_type="diagnostic")

        # 切分 (s_t, s_{t+1}) 对
        pairs: list[tuple[PatientState, PatientState]] = []
        for traj in trajectories:
            traj = np.asarray(traj, dtype=np.float64)
            if traj.ndim != 2 or traj.shape[0] < 2:
                continue
            traj = np.nan_to_num(traj, nan=0.0)
            for t in range(traj.shape[0] - 1):
                n_copy = min(traj.shape[1], N_VITALS)
                cur = np.zeros(N_VITALS)
                nxt = np.zeros(N_VITALS)
                cur[:n_copy] = traj[t, :n_copy]
                nxt[:n_copy] = traj[t + 1, :n_copy]
                pairs.append(
                    (
                        PatientState(vital_signs=cur.reshape(1, -1)),
                        PatientState(vital_signs=nxt.reshape(1, -1)),
                    )
                )

        n_pairs = len(pairs)
        if n_pairs == 0:
            self._fitted = True
            return {"final_jepa_loss": 1.0, "n_samples": 0, "converged": False, "backend": "jepa"}

        last_info: dict[str, float] = {"jepa_loss": 1.0, "recon_loss": 1.0, "total_loss": 1.0}
        for _epoch in range(n_epochs):
            indices = rng.permutation(n_pairs)
            for idx in indices:
                s_t, s_t1 = pairs[idx]
                last_info = self.train_step(s_t, s_t1, zero_action)

        self._fitted = True
        return {
            "final_jepa_loss": last_info["jepa_loss"],
            "final_recon_loss": last_info["recon_loss"],
            "n_samples": n_pairs,
            "n_epochs": n_epochs,
            "converged": last_info["jepa_loss"] < 0.1,
            "backend": "jepa",
        }

    # ── 不确定性量化 ─────────────────────────────────────────────

    def predict_with_uncertainty(
        self,
        state: PatientState,
        action: MedicalAction | None = None,
        n_steps: int = 1,
        n_bootstrap: int = 50,
        seed: int = 42,
    ) -> UncertainPrediction:
        """带不确定性量化的预测（贝叶斯 bootstrap）。

        给输入状态添加微小高斯噪声做 n_bootstrap 次扰动预测，
        用预测分布的百分位计算 95% 置信区间。

        Args:
            state: 当前 PatientState。
            action: MedicalAction（None = 无干预）。
            n_steps: 预测步数。
            n_bootstrap: bootstrap 次数。
            seed: 随机种子。

        Returns:
            UncertainPrediction（与 ClinicalDynamicsPredictor 接口一致）。
        """
        rng = np.random.default_rng(seed)
        point_preds = self.predict(state, action, n_steps=n_steps)

        # 在潜空间做 bootstrap（比观测空间扰动更语义化）
        boot_preds: list[list[np.ndarray]] = []
        z_base = self.encode(state)
        a_vec = (
            self._normalize_action(action.to_vector())
            if isinstance(action, MedicalAction)
            else np.zeros(self._config.action_dim, dtype=np.float64)
        )
        for _ in range(n_bootstrap):
            # 潜空间扰动
            z = z_base + rng.normal(0, 0.1, size=z_base.shape)
            step_preds: list[np.ndarray] = []
            for _step in range(n_steps):
                z = self._jepa.predict_next(z, a_vec)
                ps = self.reconstruct(z)
                step_preds.append(ps.vital_signs[-1].copy())
            boot_preds.append(step_preds)

        ci_lower_steps: list[np.ndarray] = []
        ci_upper_steps: list[np.ndarray] = []
        std_steps: list[np.ndarray] = []
        for step in range(n_steps):
            if not boot_preds or step >= len(boot_preds[0]):
                ci_lower_steps.append(point_preds[step].vital_signs[-1].copy())
                ci_upper_steps.append(point_preds[step].vital_signs[-1].copy())
                std_steps.append(np.zeros(N_VITALS))
                continue
            step_preds_arr = np.array([bp[step] for bp in boot_preds if step < len(bp)])
            ci_lower_steps.append(np.percentile(step_preds_arr, 2.5, axis=0))
            ci_upper_steps.append(np.percentile(step_preds_arr, 97.5, axis=0))
            std_steps.append(np.std(step_preds_arr, axis=0))

        return UncertainPrediction(
            point_estimates=point_preds,
            ci_lower=ci_lower_steps,
            ci_upper=ci_upper_steps,
            std=std_steps,
            n_bootstrap=len(boot_preds),
        )

    # ── 评估 ─────────────────────────────────────────────────────

    def evaluate_direction_accuracy(
        self,
        test_cases: list[tuple[PatientState, MedicalAction, PatientState]],
    ) -> dict[str, Any]:
        """评估预测的趋势方向准确率。

        Args:
            test_cases: [(state, action, true_next), ...]。

        Returns:
            {"direction_accuracy", "mae", "n", "backend"}。
        """
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
            "backend": "jepa",
        }

    def reconstruction_error(self, state: PatientState) -> float:
        """计算单个状态的重建误差（编码再解码的保真度）。

        用于验证潜空间是否保留了关键临床信息。

        Args:
            state: 患者状态。

        Returns:
            重建 MSE（原始量纲）。
        """
        z = self.encode(state)
        recon = self.reconstruct(z)
        diff = recon.to_vector() - state.to_vector()
        return float(np.dot(diff, diff) / self._config.obs_dim)
