from __future__ import annotations

from typing import Any

"""
MCI World Model v4.6.0 — Parametric Memory Engine
===================================================

基于 CausalMLP 的参数化记忆训练引擎（MLX Native）。
替代 v3.6.0 的 Qwen2.5-1.5B + QLoRA 路线，彻底移除 transformers/torch/peft 依赖。

核心能力:
- CausalMLP 小型因果推断网络 (~15K params)
- 纯 numpy 手写梯度 SGD 训练（零自动微分依赖）
- 五范畴因果分类 (semantic / causal / spacetime / generative / trust)
- Adapter 保存/加载 (.npz + .json)
- 路径白名单安全机制（防御路径穿越）

架构原则:
- 零 transformers / PyTorch / peft 硬依赖
- 纯 CPU + numpy + scipy
- 复用 JEPA 手写梯度范式

用法:
    from mci_world_model.sdk._parametric_memory import ParametricMemory, ParametricMemoryConfig

    config = ParametricMemoryConfig()
    pm = ParametricMemory(config)
    pm.prepare_training_data(qa_pairs)
    stats = pm.train()
    results = pm.predict("物价上涨导致货币贬值")
    pm.save_adapter("./checkpoints/mci-world-model-v0.3.7")
"""


import hashlib
import json
import logging
import os
from dataclasses import dataclass

import numpy as np

from mci_world_model.sdk._causal_mlp import (
    CATEGORY_TO_INDEX,
    ENERGY_CATEGORIES,
    CausalMLP,
    SimpleTextEmbedder,
    energy_relation_to_category,
    energy_relation_to_rho,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ParametricMemoryConfig
# =============================================================================


@dataclass
class ParametricMemoryConfig:
    """
    V3.0.7: CausalMLP 参数化记忆训练配置。

    对比 v3.6.0：移除 QLoRA 专属字段（lora_*, quant_*, base_model, use_bfloat16, bnb_*）。
    新增 CausalMLP 架构配置字段。
    """

    # ── CausalMLP 架构 ──
    input_dim: int = 128
    hidden_dims: tuple[int, ...] = (64, 32)
    num_categories: int = 5

    # ── 训练参数 ──
    batch_size: int = 8
    learning_rate: float = 0.01
    num_epochs: int = 10
    rho_weight: float = 0.1  # rho 回归损失权重 β

    # ── 能量一致性损失 ──
    energy_loss_alpha: float = 0.1
    use_energy_loss: bool = True

    # ── Checkpoint ──
    logging_steps: int = 50
    output_dir: str = "./checkpoints/mci-world-model"

    # ── 训练数据 ──
    min_training_pairs: int = 10  # V3.0.7: 降低门槛（小模型快速验证）
    min_confidence: float = 0.4
    max_training_pairs: int = 30000

    # ── 随机种子 ──
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "4.6.0",
            "model_type": "CausalMLP",
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "num_categories": self.num_categories,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "rho_weight": self.rho_weight,
            "energy_loss_alpha": self.energy_loss_alpha,
            "use_energy_loss": self.use_energy_loss,
        }


# =============================================================================
# TrainingDataFormat
# =============================================================================


@dataclass
class TrainingSample:
    """单条训练样本。"""

    instruction: str
    input_text: str
    output_text: str
    energy_relation: str  # enhance / suppress / same / neutral
    confidence: float
    sample_id: str


# =============================================================================
# ParametricMemory
# =============================================================================


class ParametricMemory:
    """
    V3.0.7: 参数化记忆引擎 — CausalMLP 因果推理。

    工作流:
    1. prepare_training_data(qa_pairs) → 转换 QA 对为训练格式
    2. train() → CausalMLP 手写梯度 SGD 训练
    3. save_adapter() / load_adapter() → 持久化 (.npz)
    4. predict() → 五范畴因果概率预测
    """

    def __init__(self, config: ParametricMemoryConfig | None = None) -> None:
        """
        Args:
            config: 训练配置（None 时使用默认值）
        """
        self.config = config or ParametricMemoryConfig()
        self._model: CausalMLP | None = None
        self._embedder: SimpleTextEmbedder = SimpleTextEmbedder(output_dim=self.config.input_dim)
        self._training_data: list[TrainingSample] = []
        self._is_trained: bool = False
        self._training_stats: dict[str, Any] = {}

    # ────────────────────────────────────────────────
    # 模型初始化
    # ────────────────────────────────────────────────

    def _init_model(self) -> CausalMLP:
        """初始化或返回已有的 CausalMLP 模型。"""
        if self._model is None:
            self._model = CausalMLP(
                input_dim=self.config.input_dim,
                hidden_dims=self.config.hidden_dims,
                num_categories=self.config.num_categories,
                seed=self.config.seed,
            )
            logger.info(
                "CausalMLP 已初始化: %d params, arch=%s",
                self._model.n_trainable_params,
                self._model,
            )
        return self._model

    # ────────────────────────────────────────────────
    # 训练数据准备
    # ────────────────────────────────────────────────

    def prepare_training_data(
        self,
        qa_pairs: list[Any],
    ) -> tuple[int, dict]:  # type: ignore
        """
        将 Reflection QA 对转换为训练格式。

        支持的数据源:
        - SynthesizedQAPair (ReflectionSynthesizer 输出)
        - dict (含 cause_text, effect_text, energy_relation, confidence)

        Args:
            qa_pairs: QA 对列表

        Returns:
            (n_samples, quality_report)
        """
        self._training_data = []
        skipped = 0
        energy_dist: dict[str, int] = {}
        confidences: list[float] = []

        for i, pair in enumerate(qa_pairs):
            # 兼容 SynthesizedQAPair 和 dict 两种格式
            if hasattr(pair, "cause_text"):
                cause = pair.cause_text
                effect = pair.effect_text
                energy_rel = getattr(pair, "energy_relation", "neutral")
                conf = getattr(pair, "confidence", 0.5)
            elif isinstance(pair, dict):
                cause = pair.get("cause_text", pair.get("cause", ""))
                effect = pair.get("effect_text", pair.get("effect", ""))
                energy_rel = pair.get("energy_relation", "neutral")
                conf = pair.get("confidence", 0.5)
            else:
                skipped += 1
                continue

            if not cause or not effect:
                skipped += 1
                continue

            if conf < self.config.min_confidence:
                skipped += 1
                continue

            sample = TrainingSample(
                instruction="分析以下原因和效应之间的因果关系。",
                input_text=cause,
                output_text=effect,
                energy_relation=energy_rel,
                confidence=conf,
                sample_id=_hash_sample_id(cause, effect, i),
            )
            self._training_data.append(sample)

            energy_dist[energy_rel] = energy_dist.get(energy_rel, 0) + 1
            confidences.append(conf)

            if len(self._training_data) >= self.config.max_training_pairs:
                break

        # 质量报告
        n = len(self._training_data)
        report = {
            "total_samples": n,
            "skipped": skipped,
            "avg_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
            "energy_distribution": dict(energy_dist),
            "meets_minimum": n >= self.config.min_training_pairs,
        }

        logger.info(
            "训练数据准备完成: %d 条样本 (跳过 %d 条), 平均置信度 %.4f",
            n,
            skipped,
            report["avg_confidence"],
        )
        return n, report

    def get_training_format(self) -> list[dict[str, Any]]:
        """获取转换后的训练格式数据。"""
        return [
            {
                "instruction": s.instruction,
                "input": s.input_text,
                "output": s.output_text,
                "energy_relation": s.energy_relation,
                "confidence": s.confidence,
            }
            for s in self._training_data
        ]

    # ────────────────────────────────────────────────
    # 嵌入转换
    # ────────────────────────────────────────────────

    def _embed_training_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        将训练数据转换为 CausalMLP 可用的嵌入和标签。

        Returns:
            (X, y_categories, y_rhos):
            - X: (N, input_dim) 嵌入矩阵
            - y_categories: (N,) 类别索引
            - y_rhos: (N,) rho 代理值
        """
        if not self._training_data:
            return (
                np.zeros((0, self.config.input_dim), dtype=np.float32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.float32),
            )

        n = len(self._training_data)
        X = np.zeros((n, self.config.input_dim), dtype=np.float32)
        y_categories = np.zeros(n, dtype=np.int32)
        y_rhos = np.zeros(n, dtype=np.float32)

        for i, sample in enumerate(self._training_data):
            X[i] = self._embedder.embed(sample.input_text)
            y_categories[i] = energy_relation_to_category(sample.energy_relation)
            y_rhos[i] = energy_relation_to_rho(sample.energy_relation)

        return X, y_categories, y_rhos

    # ────────────────────────────────────────────────
    # 训练
    # ────────────────────────────────────────────────

    def train(  # type: ignore
        self,
        training_data: list | None = None,  # type: ignore
        energy_loss_fn=None,
    ) -> dict[str, Any]:
        """
        执行 CausalMLP 参数化训练。

        Args:
            training_data: 训练数据（None 时使用 prepare_training_data 的数据）
            energy_loss_fn: EnergyConsistencyLoss 实例（保留接口兼容，当前未使用）

        Returns:
            训练统计字典
        """
        if training_data is not None:
            self.prepare_training_data(training_data)

        if len(self._training_data) < self.config.min_training_pairs:
            logger.warning(
                "训练数据不足: %d 条 (最低 %d 条)，训练可能效果不佳",
                len(self._training_data),
                self.config.min_training_pairs,
            )

        # ── 初始化模型 ──
        model = self._init_model()

        # ── 嵌入训练数据 ──
        X, y_categories, y_rhos = self._embed_training_data()
        if X.shape[0] == 0:
            logger.warning("无有效训练数据")
            return {"error": "no_training_data", "backend": "numpy_sgd"}

        logger.info(
            "开始 CausalMLP 训练: %d samples, %d epochs, lr=%.4f",
            X.shape[0],
            self.config.num_epochs,
            self.config.learning_rate,
        )

        # ── 训练 ──
        stats = model.train(
            X,
            y_categories,
            y_rhos,
            n_epochs=self.config.num_epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            rho_weight=self.config.rho_weight,
        )

        stats["backend"] = "numpy_sgd"
        stats["n_trainable_params"] = model.n_trainable_params
        stats["n_samples"] = X.shape[0]

        self._is_trained = True
        self._training_stats = stats

        logger.info(
            "CausalMLP 训练完成: %d epochs, final loss %.6f",
            self.config.num_epochs,
            stats["final_loss"],
        )
        return stats

    def train_on_signals(
        self,
        cause_texts: list[str],
        true_categories: list[int],
        learning_rate: float = 0.01,
    ) -> dict[str, Any]:
        """
        快速单轮训练：用于 JEPATrainer 集成场景。

        Args:
            cause_texts: 原因文本列表
            true_categories: 真实类别索引列表
            learning_rate: 学习率

        Returns:
            训练统计
        """
        model = self._init_model()
        n = min(len(cause_texts), len(true_categories))

        losses: list[float] = []
        for i in range(n):
            x = self._embedder.embed(cause_texts[i])
            loss = model.train_step(
                x,
                int(true_categories[i]),
                y_rho=0.5,
                learning_rate=learning_rate,
                rho_weight=self.config.rho_weight,
            )
            losses.append(loss)

        self._is_trained = True
        avg_loss = float(np.mean(losses)) if losses else 0.0
        return {"n_steps": n, "final_loss": round(avg_loss, 6), "backend": "numpy_sgd"}

    # ────────────────────────────────────────────────
    # Adapter 持久化
    # ────────────────────────────────────────────────

    # F2-P0-2: 路径白名单常量 — 限制 adapter 可落盘/加载的根目录
    _ALLOWED_ADAPTER_ROOTS = (
        "~/.mci_world_model/adapters",
        "~/.cache/mci_world_model/adapters",
        "./adapters",
        "./checkpoints",
    )

    @classmethod
    def _validate_adapter_path(cls, path: str) -> str:
        """
        F2-P0-2: 验证 adapter 路径合法性，防御路径穿越与任意写入。

        Returns:
            规范化后的绝对路径

        Raises:
            ValueError: 路径不合法
        """
        import os.path as _osp

        if not isinstance(path, str) or not path.strip():
            raise ValueError("adapter path must be a non-empty string")

        expanded = _osp.expanduser(path)
        abs_path = _osp.abspath(expanded)

        # 拒绝路径穿越
        if ".." in _osp.normpath(expanded).split(_osp.sep):
            raise ValueError(f"adapter path traversal not allowed: {path!r} (resolved: {abs_path})")

        # 展开允许根目录
        allowed_roots = [_osp.abspath(_osp.expanduser(r)) for r in cls._ALLOWED_ADAPTER_ROOTS]

        # 兼容旧用法：项目根目录或 CWD 也允许
        cwd = os.getcwd()
        project_root_marker = _osp.abspath(_osp.dirname(_osp.dirname(_osp.dirname(_osp.dirname(__file__)))))
        allowed_roots.extend([cwd, project_root_marker])

        for root in allowed_roots:
            try:
                if _osp.commonpath([abs_path, root]) == root:
                    return abs_path
            except ValueError:
                continue

        raise ValueError(f"adapter path not in whitelist: {abs_path}. Allowed roots: {cls._ALLOWED_ADAPTER_ROOTS}")

    def save_adapter(self, path: str) -> bool:
        """
        保存 CausalMLP 模型到磁盘 (.npz + adapter_config.json)。

        Args:
            path: 输出目录路径

        Returns:
            True 如果保存成功
        """
        try:
            safe_path = self._validate_adapter_path(path)
        except ValueError as e:
            logger.error("save_adapter 拒绝非法路径: %s", e)
            return False

        os.makedirs(safe_path, exist_ok=True)

        try:
            # ── 保存模型权重 ──
            model = self._init_model()
            model_path = os.path.join(safe_path, "causal_mlp_weights")
            if not model.save(model_path):
                return False

            # ── 保存配置 ──
            config_dict = self.config.to_dict()
            config_dict["adapter_type"] = "causal_mlp"
            config_dict["n_params"] = model.n_trainable_params
            with open(os.path.join(safe_path, "adapter_config.json"), "w") as f:
                json.dump(config_dict, f, indent=2)

            # ── 保存训练统计 ──
            with open(os.path.join(safe_path, "training_stats.json"), "w") as f:
                json.dump(self._training_stats, f, indent=2)

            logger.info(
                "CausalMLP adapter 已保存到: %s (%d 训练样本, %d params)",
                safe_path,
                len(self._training_data),
                model.n_trainable_params,
            )
            return True
        except Exception as e:
            logger.error("保存 adapter 失败: %s", e)
            return False

    def load_adapter(self, path: str) -> bool:
        """
        从磁盘加载预训练的 CausalMLP adapter。

        Args:
            path: adapter 目录路径

        Returns:
            True 如果加载成功
        """
        try:
            safe_path = self._validate_adapter_path(path)
        except ValueError as e:
            logger.error("load_adapter 拒绝非法路径: %s", e)
            return False

        config_path = os.path.join(safe_path, "adapter_config.json")
        if not os.path.exists(config_path):
            logger.error("adapter 配置文件不存在: %s", config_path)
            return False

        try:
            with open(config_path) as f:
                adapter_config = json.load(f)

            # 更新配置
            for key in ("input_dim", "hidden_dims", "num_categories"):
                if key in adapter_config:
                    value = adapter_config[key]
                    if key == "hidden_dims":
                        value = tuple(value)
                    setattr(self.config, key, value)

            # 加载模型
            model_path = os.path.join(safe_path, "causal_mlp_weights")
            self._model = CausalMLP.load(model_path)
            if self._model is None:
                logger.error("CausalMLP 权重加载失败: %s", model_path)
                return False

            self._is_trained = True
            logger.info(
                "adapter 加载成功: %s (version=%s)",
                path,
                adapter_config.get("version", "unknown"),
            )
            return True
        except Exception as e:
            logger.error("加载 adapter 失败: %s", e)
            return False

    # ────────────────────────────────────────────────
    # 推理
    # ────────────────────────────────────────────────

    def predict(
        self,
        cause: str,
        target_category: str | None = None,
        top_k: int = 3,
        max_new_tokens: int = 128,  # 保留接口兼容，CausalMLP 不使用
    ) -> list[dict[str, Any]]:
        """
        参数化因果推理：给定原因文本，预测五范畴因果分布。

        Args:
            cause: 原因文本
            target_category: 目标状态类别（可选，约束输出排序）
            top_k: 返回前 K 个预测
            max_new_tokens: 保留接口兼容（CausalMLP 不使用）

        Returns:
            [{"effect": str, "confidence": float, "energy_relation": str}, ...]
        """
        model = self._init_model()
        if model is None:
            logger.warning("模型初始化失败，返回空预测")
            return []

        # ── 嵌入 + 前向 ──
        x = self._embedder.embed(cause)
        probs = model.forward(x)

        # ── 构建排序结果 ──
        category_order = list(ENERGY_CATEGORIES)
        if target_category and target_category in CATEGORY_TO_INDEX:
            # 目标类别优先
            category_order = [target_category] + [c for c in category_order if c != target_category]

        results = []
        for cat in category_order[: min(top_k, len(category_order))]:
            idx = CATEGORY_TO_INDEX[cat]
            confidence = round(float(probs[idx]), 4)

            # 将类别映射回 energy_relation
            if cat == "causal":
                relation = "enhance" if confidence > 0.5 else "suppress"
            elif cat == "semantic":
                relation = "same"
            else:
                relation = "neutral"

            results.append(
                {
                    "effect": f"[CausalMLP]: 基于因果先验的效应推断 → {cat}",
                    "confidence": confidence,
                    "energy_relation": relation,
                    "category": cat,
                }
            )

        return results

    def predict_probs(self, cause: str) -> dict[str, float]:
        """返回所有五范畴的概率分布。"""
        model = self._model
        if model is None:
            return dict.fromkeys(ENERGY_CATEGORIES, 0.0)

        x = self._embedder.embed(cause)
        probs = model.forward(x)
        return {cat: round(float(probs[i]), 4) for i, cat in enumerate(ENERGY_CATEGORIES)}

    # ────────────────────────────────────────────────
    # 状态查询
    # ────────────────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def training_stats(self) -> dict[str, Any]:
        return self._training_stats.copy()

    @property
    def n_training_samples(self) -> int:
        return len(self._training_data)

    @property
    def model(self) -> CausalMLP | None:
        return self._model

    def health_check(self) -> dict[str, Any]:
        """健康检查。"""
        return {
            "model_initialized": self._model is not None,
            "model_type": "CausalMLP" if self._model is not None else None,
            "n_trainable_params": self._model.n_trainable_params if self._model else 0,
            "is_trained": self._is_trained,
            "n_training_samples": self.n_training_samples,
            "backend": "numpy_sgd",
            "input_dim": self.config.input_dim,
            "hidden_dims": list(self.config.hidden_dims),
            "adapters": [],
        }


# =============================================================================
# 工具函数
# =============================================================================


def _hash_sample_id(cause: str, effect: str, idx: int) -> str:
    """生成训练样本唯一 ID。"""
    content = f"{cause}::{effect}::{idx}"
    h = hashlib.sha256(content.encode()).hexdigest()
    return f"pm_{h[:12]}"


def estimate_training_time(n_samples: int, backend: str = "numpy_sgd") -> dict[str, Any]:
    """
    估算训练时间。

    Args:
        n_samples: 训练样本数
        backend: "numpy_sgd" (仅支持此模式)

    Returns:
        {"hours": float, "minutes": float, "steps": int}
    """
    steps_per_sample = 0.001  # numpy SGD ~1ms per sample
    total_seconds = n_samples * steps_per_sample
    return {
        "hours": round(total_seconds / 3600, 3),
        "minutes": round(total_seconds / 60, 3),
        "steps": int(n_samples),
        "backend": backend,
    }
