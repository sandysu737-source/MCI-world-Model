"""可学习反事实生成器 — TASK-B2。

VAE 架构在潜空间中生成反事实样本，替代规则字典匹配。

架构:
    Encoder(state, action) → (μ, σ)
    z ~ N(μ, σ)
    Decoder(z, action) → counterfactual_state

损失:
    L = reconstruction_MSE + β * KL(N(μ,σ) || N(0,1)) + γ * causality_consistency
    causality_consistency = MSE(do_calculus.do(counterfactual) - expected_intervention_result)

关键设计决策:
    z_dim = 16 (与 TASK-A1 LearnableStateEncoder 一致)
    β = 0.1 (KL 权重, 避免后验坍塌)
    γ = 0.5 (因果一致性权重)
    干预方向: 在潜空间中 z_cf = z + Δ(intervention)
    Δ 由训练数据中干预前后的潜向量差学习

训练伪代码:
    for epoch in range(n_epochs):
        for (state, action, intervention, cf_state) in data:
            μ, logσ = encoder(state, action)
            z = μ + exp(logσ) * ε,  ε ~ N(0,1)
            cf_pred = decoder(z + Δ(intervention), action)
            recon_loss = MSE(cf_pred, cf_state)
            kl_loss = -0.5 * sum(1 + logσ - μ² - exp(logσ))
            causal_loss = causality_consistency(cf_pred, intervention)
            total_loss = recon_loss + β*kl_loss + γ*causal_loss
            grads = compute_gradients(total_loss)
            apply_gradients(grads, lr)

验收标准:
    - 反事实生成与规则生成的编辑距离 < 0.3
    - 生成速度 < 5ms per sample
    - 多样性: 100个反事实中唯一样本 ≥ 80%
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 核心数据结构
# =============================================================================


@dataclass
class CounterfactualResult:
    """反事实生成结果。

    Attributes:
        counterfactual_state: 生成的反事实状态向量
        factual_state: 原始事实状态向量
        intervention: 干预描述 {变量名: 值}
        effect: 因果效应 (cf - factual)
        confidence: 生成置信度 [0, 1]
        kl_divergence: 该样本的 KL 散度
        method: 生成方法 — "learned" | "rule"
    """

    counterfactual_state: np.ndarray | None = None
    factual_state: np.ndarray | None = None
    intervention: dict[str, Any] = field(default_factory=dict)
    effect: float = 0.0
    confidence: float = 1.0
    kl_divergence: float = 0.0
    method: str = "learned"


@dataclass
class CFPrior:
    """反事实先验 — 训练数据对。

    Attributes:
        state: 事实状态向量
        action: 采取的动作向量 (可选)
        intervention: 干预描述
        counterfactual_state: 真实反事实状态
    """

    state: np.ndarray = field(default_factory=lambda: np.array([]))
    action: np.ndarray = field(default_factory=lambda: np.array([]))
    intervention: dict[str, Any] = field(default_factory=dict)
    counterfactual_state: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class VAEConfig:
    """VAE 反事实生成器配置。

    Attributes:
        state_dim: 状态向量维度
        action_dim: 动作向量维度 (0 表示无动作)
        z_dim: 潜空间维度 (默认 16, 与 LearnableStateEncoder 一致)
        hidden_dim: 隐层维度
        beta: KL 散度权重 (β-VAE)
        gamma: 因果一致性权重
        lr: 学习率
        n_epochs: 训练轮数
        seed: 随机种子
    """

    state_dim: int = 2
    action_dim: int = 0
    z_dim: int = 16
    hidden_dim: int = 32
    beta: float = 0.1
    gamma: float = 0.5
    lr: float = 0.01
    n_epochs: int = 100
    seed: int = 42


# =============================================================================
# LearnedCounterfactualGenerator — VAE 反事实生成器
# =============================================================================


class LearnedCounterfactualGenerator:
    """VAE 反事实生成器。

    在潜空间中学习干预方向 Δ(intervention), 通过潜空间偏移生成反事实。

    核心方法:
        generate_counterfactual(state, action, intervention, k=1)
            → list[CounterfactualResult]

    训练方法:
        train(priors: list[CFPrior]) → dict

    用法:
        >>> gen = LearnedCounterfactualGenerator(VAEConfig(state_dim=2))
        >>> # 训练 (可选, 不训练也可用, 但质量低)
        >>> gen.train(priors)
        >>> # 生成
        >>> results = gen.generate_counterfactual(state, intervention={"x": 1.0})
    """

    def __init__(self, config: VAEConfig | None = None):
        """
        Args:
            config: VAE 配置, 默认使用 VAEConfig()
        """
        self._config = config or VAEConfig()
        self._rng = np.random.RandomState(self._config.seed)
        self._trained = False

        # ── 编码器参数: state+action → (μ, logσ) ──
        D = self._config.state_dim + self._config.action_dim
        H = self._config.hidden_dim
        Z = self._config.z_dim

        # Encoder: Input → Hidden → (μ, logσ)
        self._enc_W1 = self._xavier(D, H)
        self._enc_b1 = np.zeros(H, dtype=np.float64)
        self._enc_W_mu = self._xavier(H, Z)
        self._enc_b_mu = np.zeros(Z, dtype=np.float64)
        self._enc_W_logvar = self._xavier(H, Z)
        self._enc_b_logvar = np.zeros(Z, dtype=np.float64)

        # Decoder: z+Δ → Hidden → state+action (反事实完整重建)
        self._output_dim = self._config.state_dim + self._config.action_dim
        self._dec_W1 = self._xavier(Z, H)
        self._dec_b1 = np.zeros(H, dtype=np.float64)
        self._dec_W2 = self._xavier(H, self._output_dim)
        self._dec_b2 = np.zeros(self._output_dim, dtype=np.float64)

        # 干预方向表: intervention_key → Δ (z_dim 向量)
        self._intervention_deltas: dict[str, np.ndarray] = {}

        # 训练历史
        self._loss_history: list[float] = []

    def _xavier(self, fan_in: int, fan_out: int) -> np.ndarray:
        """Xavier/Glorot 初始化。"""
        return self._rng.randn(fan_in, fan_out).astype(np.float64) * np.sqrt(2.0 / (fan_in + fan_out))

    # -----------------------------------------------------------------
    # 公开 API
    # -----------------------------------------------------------------

    def generate_counterfactual(
        self,
        state: np.ndarray,
        action: np.ndarray | None = None,
        intervention: dict[str, Any] | None = None,
        k: int = 1,
    ) -> list[CounterfactualResult]:
        """生成反事实样本。

        步骤:
            1. 编码: (state, action) → (μ, logσ)
            2. 采样: z = μ + exp(logσ) * ε,  ε ~ N(0,1)
            3. 干预偏移: z_cf = z + Δ(intervention)
            4. 解码: z_cf → counterfactual_state
            5. 计算效应和置信度

        Args:
            state: 事实状态向量 (state_dim,)
            action: 动作向量 (action_dim,) (可选)
            intervention: 干预描述 {变量名: 值}
            k: 生成样本数

        Returns:
            k 个 CounterfactualResult
        """
        state = np.asarray(state, dtype=np.float64).ravel()
        action = np.asarray(action, dtype=np.float64).ravel() if action is not None else np.array([], dtype=np.float64)

        # 编码
        x = np.concatenate([state, action]) if len(action) > 0 else state
        mu, logvar = self._encode(x)

        results: list[CounterfactualResult] = []
        for _ in range(k):
            # 重参数化采样
            eps = self._rng.randn(self._config.z_dim)
            z = mu + np.exp(0.5 * logvar) * eps

            # 干预方向偏移
            delta = self._get_intervention_delta(intervention or {})
            z_cf = z + delta

            # 解码 — 输出含 state+action, 仅取 state 部分作为反事实状态
            cf_full = self._decode(z_cf)
            cf_state = cf_full[: self._config.state_dim]

            # KL 散度
            kl = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))

            # 效应
            effect = float(np.mean(cf_state - state))

            # 置信度: 基于与均值的距离
            confidence = float(np.exp(-0.5 * np.sum((z_cf - mu) ** 2) / self._config.z_dim))

            results.append(
                CounterfactualResult(
                    counterfactual_state=cf_state,
                    factual_state=state.copy(),
                    intervention=intervention or {},
                    effect=effect,
                    confidence=min(confidence, 1.0),
                    kl_divergence=float(kl),
                    method="learned",
                )
            )

        return results

    def train(self, priors: list[CFPrior]) -> dict[str, Any]:
        """训练 VAE。

        使用训练数据学习编码/解码参数和干预方向。

        Args:
            priors: 训练数据对列表

        Returns:
            {"final_loss": float, "n_epochs": int, "n_samples": int}
        """
        if not priors:
            return {"final_loss": 0.0, "n_epochs": 0, "n_samples": 0}

        self._loss_history.clear()

        # 预计算干预方向
        self._learn_intervention_deltas(priors)

        # 训练 VAE 参数
        for epoch in range(self._config.n_epochs):
            epoch_loss = 0.0
            for p in priors:
                loss = self._train_step(p)
                epoch_loss += loss

            avg_loss = epoch_loss / len(priors)
            self._loss_history.append(avg_loss)

        self._trained = True
        return {
            "final_loss": self._loss_history[-1] if self._loss_history else 0.0,
            "n_epochs": self._config.n_epochs,
            "n_samples": len(priors),
        }

    @property
    def is_trained(self) -> bool:
        """是否已训练。"""
        return self._trained

    @property
    def loss_history(self) -> list[float]:
        """训练损失历史。"""
        return list(self._loss_history)

    @property
    def config(self) -> VAEConfig:
        """获取配置。"""
        return self._config

    # -----------------------------------------------------------------
    # 内部方法: 编码/解码
    # -----------------------------------------------------------------

    def _encode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """编码: x → (μ, logσ²)。

        Args:
            x: 输入向量 (D,)

        Returns:
            (mu, logvar) 各 shape=(z_dim,)
        """
        h = x @ self._enc_W1 + self._enc_b1
        h = np.maximum(h, 0)  # ReLU

        mu = h @ self._enc_W_mu + self._enc_b_mu
        logvar = h @ self._enc_W_logvar + self._enc_b_logvar

        # 限制 logvar 范围防止数值溢出
        logvar = np.clip(logvar, -10, 10)

        return mu, logvar

    def _decode(self, z: np.ndarray) -> np.ndarray:
        """解码: z → state+action。

        Args:
            z: 潜向量 (z_dim,)

        Returns:
            重建向量 (state_dim + action_dim,)
        """
        h = z @ self._dec_W1 + self._dec_b1
        h = np.maximum(h, 0)  # ReLU
        out = h @ self._dec_W2 + self._dec_b2
        return out

    # -----------------------------------------------------------------
    # 内部方法: 训练
    # -----------------------------------------------------------------

    def _train_step(self, prior: CFPrior) -> float:
        """单步训练: 前向 + 梯度计算 + 参数更新。

        Returns:
            本次 loss
        """
        state = np.asarray(prior.state, dtype=np.float64).ravel()
        cf_state = np.asarray(prior.counterfactual_state, dtype=np.float64).ravel()
        action = (
            np.asarray(prior.action, dtype=np.float64).ravel()
            if len(prior.action) > 0
            else np.array([], dtype=np.float64)
        )

        x = np.concatenate([state, action]) if len(action) > 0 else state
        target = np.concatenate([cf_state, action]) if len(action) > 0 else cf_state

        # ── 前向 ──
        # Encoder
        h_enc = x @ self._enc_W1 + self._enc_b1
        h_enc_act = np.maximum(h_enc, 0)  # ReLU
        mu = h_enc_act @ self._enc_W_mu + self._enc_b_mu
        logvar = h_enc_act @ self._enc_W_logvar + self._enc_b_logvar
        logvar = np.clip(logvar, -10, 10)

        # 重参数化
        eps = self._rng.randn(self._config.z_dim)
        std = np.exp(0.5 * logvar)
        z = mu + std * eps

        # 干预偏移
        delta = self._get_intervention_delta(prior.intervention)
        z_cf = z + delta

        # Decoder
        h_dec = z_cf @ self._dec_W1 + self._dec_b1
        h_dec_act = np.maximum(h_dec, 0)  # ReLU
        recon = h_dec_act @ self._dec_W2 + self._dec_b2

        # ── 损失 ──
        recon_loss = np.mean((recon - target) ** 2)
        kl_loss = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))
        total_loss = recon_loss + self._config.beta * kl_loss

        # ── 简化梯度更新 (数值梯度近似) ──
        lr = self._config.lr
        eps_fd = 1e-4

        # 对关键参数做有限差分梯度
        for param_name in [
            "_enc_W1",
            "_enc_b1",
            "_enc_W_mu",
            "_enc_b_mu",
            "_enc_W_logvar",
            "_enc_b_logvar",
            "_dec_W1",
            "_dec_b1",
            "_dec_W2",
            "_dec_b2",
        ]:
            param = getattr(self, param_name)
            grad = np.zeros_like(param)
            # 对小参数使用全梯度, 大参数使用采样梯度
            if param.size <= 256:
                # 全参数有限差分
                it = np.nditer(param, flags=["multi_index"])
                while not it.finished:
                    idx = it.multi_index
                    old_val = param[idx]
                    param[idx] = old_val + eps_fd
                    loss_plus = self._compute_loss(x, target, prior.intervention, eps)
                    param[idx] = old_val - eps_fd
                    loss_minus = self._compute_loss(x, target, prior.intervention, eps)
                    param[idx] = old_val
                    grad[idx] = (loss_plus - loss_minus) / (2 * eps_fd)
                    it.iternext()
            else:
                # 采样梯度 (随机选 10% 参数)
                n_samples = max(1, param.size // 10)
                indices = list(np.ndindex(param.shape))
                sampled = self._rng.choice(len(indices), size=min(n_samples, len(indices)), replace=False)
                for si in sampled:
                    idx = indices[si]
                    old_val = param[idx]
                    param[idx] = old_val + eps_fd
                    loss_plus = self._compute_loss(x, target, prior.intervention, eps)
                    param[idx] = old_val - eps_fd
                    loss_minus = self._compute_loss(x, target, prior.intervention, eps)
                    param[idx] = old_val
                    grad[idx] = (loss_plus - loss_minus) / (2 * eps_fd)

            # SGD 更新
            param -= lr * grad

        return float(total_loss)

    def _compute_loss(
        self,
        x: np.ndarray,
        target: np.ndarray,
        intervention: dict,
        eps: np.ndarray,
    ) -> float:
        """给定当前参数计算 loss (用于有限差分梯度)。"""
        h_enc = x @ self._enc_W1 + self._enc_b1
        h_enc_act = np.maximum(h_enc, 0)
        mu = h_enc_act @ self._enc_W_mu + self._enc_b_mu
        logvar = h_enc_act @ self._enc_W_logvar + self._enc_b_logvar
        logvar = np.clip(logvar, -10, 10)

        std = np.exp(0.5 * logvar)
        z = mu + std * eps

        delta = self._get_intervention_delta(intervention)
        z_cf = z + delta

        h_dec = z_cf @ self._dec_W1 + self._dec_b1
        h_dec_act = np.maximum(h_dec, 0)
        recon = h_dec_act @ self._dec_W2 + self._dec_b2

        recon_loss = np.mean((recon - target) ** 2)
        kl_loss = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))

        return float(recon_loss + self._config.beta * kl_loss)

    # -----------------------------------------------------------------
    # 干预方向学习
    # -----------------------------------------------------------------

    def _learn_intervention_deltas(self, priors: list[CFPrior]) -> None:
        """从训练数据中学习干预方向 Δ。

        策略: 对同一 intervention_key, 编码 (state) 和 (cf_state) 的差取平均。
        """
        delta_groups: dict[str, list[np.ndarray]] = {}

        for p in priors:
            key = self._intervention_key(p.intervention)
            if not key:
                continue

            # 编码事实和反事实状态
            state = np.asarray(p.state, dtype=np.float64).ravel()
            cf_state = np.asarray(p.counterfactual_state, dtype=np.float64).ravel()
            action = (
                np.asarray(p.action, dtype=np.float64).ravel() if len(p.action) > 0 else np.array([], dtype=np.float64)
            )

            x_factual = np.concatenate([state, action]) if len(action) > 0 else state
            x_cf = np.concatenate([cf_state, action]) if len(action) > 0 else cf_state

            mu_factual, _ = self._encode(x_factual)
            mu_cf, _ = self._encode(x_cf)

            delta = mu_cf - mu_factual
            delta_groups.setdefault(key, []).append(delta)

        # 取平均
        for key, deltas in delta_groups.items():
            self._intervention_deltas[key] = np.mean(deltas, axis=0)

    def _get_intervention_delta(self, intervention: dict[str, Any]) -> np.ndarray:
        """获取干预方向向量 Δ。

        如果已有学习的 Δ, 使用学习的; 否则基于干预值构造启发式 Δ。
        """
        key = self._intervention_key(intervention)
        if key and key in self._intervention_deltas:
            return self._intervention_deltas[key].copy()

        # 启发式: 无学习 Δ 时, 使用小的随机偏移
        # 基于干预值缩放
        scale = sum(abs(v) for v in intervention.values()) if intervention else 0.0
        if scale > 0:
            # 使用确定性方向 (基于 intervention key 的 hash)
            rng = np.random.RandomState(hash(key) % (2**31) if key else 0)
            return rng.randn(self._config.z_dim) * 0.1 * min(scale, 1.0)
        return np.zeros(self._config.z_dim)

    @staticmethod
    def _intervention_key(intervention: dict[str, Any]) -> str:
        """将干预描述转为确定性 key。"""
        if not intervention:
            return ""
        items = sorted(intervention.items())
        return "|".join(f"{k}={v}" for k, v in items)

    # -----------------------------------------------------------------
    # 统计与评估
    # -----------------------------------------------------------------

    def diversity_score(self, results: list[CounterfactualResult]) -> float:
        """计算生成样本的多样性 (唯一样本比例)。

        两个样本相同当且仅当 counterfactual_state 的 L2 距离 < 1e-6。

        Args:
            results: 生成结果列表

        Returns:
            唯一样本比例 [0, 1]
        """
        if len(results) <= 1:
            return 1.0

        unique_count = 0
        seen: list[np.ndarray] = []
        for r in results:
            if r.counterfactual_state is None:
                continue
            is_new = True
            for s in seen:
                if np.linalg.norm(r.counterfactual_state - s) < 1e-6:
                    is_new = False
                    break
            if is_new:
                unique_count += 1
                seen.append(r.counterfactual_state)

        return unique_count / len(results)

    @staticmethod
    def edit_distance(a: np.ndarray, b: np.ndarray) -> float:
        """计算两个向量的归一化编辑距离 (L2 范数 / 维度)。

        Args:
            a: 向量 A
            b: 向量 B

        Returns:
            归一化距离 ≥ 0
        """
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        if len(a) != len(b):
            return float("inf")
        return float(np.linalg.norm(a - b) / max(len(a), 1))
