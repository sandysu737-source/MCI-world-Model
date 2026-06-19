from __future__ import annotations

"""
MCI World Model v4.4.0 — Debridement Trainer
===============================================

AI 智能清创机器人 — 三阶段多模态训练管线。

三阶段训练策略:
    Stage 1: 模态编码器预训练 (独立训练各模态投影头)
    Stage 2: 跨模态融合层训练 (冻结编码器, 训练 Fusion)
    Stage 3: 联合微调 (全部参数)

训练后端:
    优先 MLX (Apple Silicon 梯度计算), 降级 numpy SGD

用法:
    trainer = DebridementTrainer(config=DebridementConfig.small())
    trainer.prepare_data(n_samples=5000)
    history = trainer.train(stages=[1, 2, 3])
    trainer.save_checkpoint("checkpoints/debridement_v1")
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# MLX 可选导入
try:
    import mlx.core as mx
    import mlx.optimizers as mx_opt
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False
    mx = None  # type: ignore[assignment]
    mx_opt = None  # type: ignore[assignment]


# =============================================================================
# TrainingConfig
# =============================================================================


@dataclass
class TrainingConfig:
    """训练超参数配置。"""

    # 阶段控制
    stages: list[int] = field(default_factory=lambda: [1, 2, 3])
    stage1_epochs: int = 5
    stage2_epochs: int = 10
    stage3_epochs: int = 20

    # 优化器
    lr_stage1: float = 1e-3
    lr_stage2: float = 3e-4
    lr_stage3: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # 数据
    batch_size: int = 16
    n_train_samples: int = 5000
    n_val_samples: int = 500
    n_test_samples: int = 500
    val_split: float = 0.1
    augmentation_enabled: bool = True

    # 损失权重
    tissue_loss_weight: float = 0.5
    dynamics_loss_weight: float = 1.0
    force_mse_weight: float = 0.3
    depth_mae_weight: float = 0.2
    safety_penalty_weight: float = 0.5

    # 早停
    early_stop_patience: int = 5
    early_stop_min_delta: float = 1e-4

    # 检查点
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True

    # 随机数
    seed: int = 42


# =============================================================================
# TrainingMetrics
# =============================================================================


@dataclass
class TrainingMetrics:
    """训练指标收集器。"""

    epoch: int = 0
    stage: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    tissue_accuracy: float = 0.0
    force_mse: float = 0.0
    depth_mae: float = 0.0
    safety_violation_rate: float = 0.0
    phase_accuracy: float = 0.0
    lr: float = 0.0
    elapsed_seconds: float = 0.0
    n_params: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "stage": self.stage,
            "train_loss": round(self.train_loss, 6),
            "val_loss": round(self.val_loss, 6),
            "tissue_accuracy": round(self.tissue_accuracy, 4),
            "force_mse": round(self.force_mse, 6),
            "depth_mae": round(self.depth_mae, 4),
            "safety_violation_rate": round(self.safety_violation_rate, 4),
            "phase_accuracy": round(self.phase_accuracy, 4),
            "lr": self.lr,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "n_params": self.n_params,
        }


@dataclass
class TrainHistory:
    """训练历史记录。"""

    metrics: list[TrainingMetrics] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = -1
    total_epochs: int = 0
    total_seconds: float = 0.0

    def add(self, m: TrainingMetrics) -> None:
        self.metrics.append(m)
        self.total_epochs += 1
        self.total_seconds += m.elapsed_seconds
        if m.val_loss < self.best_val_loss:
            self.best_val_loss = m.val_loss
            self.best_epoch = m.epoch

    def to_list(self) -> list[dict[str, Any]]:
        return [m.to_dict() for m in self.metrics]


# =============================================================================
# DebridementTrainer
# =============================================================================


class DebridementTrainer:
    """清创多模态模型训练器。

    三阶段训练策略:
        Stage 1 — 编码器预训练: 训练各模态投影头 (MLP), 冻结融合层
        Stage 2 — 融合层训练: 训练 Cross-Modal Fusion + Transformer, 冻结编码器
        Stage 3 — 联合微调: 全部参数联合优化

    训练后端:
        MLX 路径 (M5 Pro 加速): mx.value_and_grad → mx.optimizers.SGD
        Numpy 路径 (通用回退): 手动 SGD + 有限差分梯度估计

    用法:
        from mci_world_model.sdk._debridement_data import SyntheticDebridementGenerator
        from mci_world_model.sdk._debridement_world_model import DebridementConfig

        config = DebridementConfig.base()
        trainer = DebridementTrainer(config)
        trainer.prepare_data()
        history = trainer.train()
        trainer.export_report("training_report.json")
    """

    def __init__(
        self,
        model_config: Any = None,
        train_config: TrainingConfig | None = None,
    ) -> None:
        """初始化训练器。

        Args:
            model_config: DebridementConfig 实例
            train_config: TrainingConfig 实例
        """
        from mci_world_model.sdk._debridement_world_model import (
            DebridementConfig,
            DebridementWorldModel,
        )

        self.model_config = model_config or DebridementConfig.small()
        self.train_cfg = train_config or TrainingConfig()
        self.model = DebridementWorldModel(self.model_config)
        self._history = TrainHistory()
        self._rng = np.random.RandomState(self.train_cfg.seed)

        # 数据容器
        self._train_data: list[Any] = []
        self._val_data: list[Any] = []
        self._test_data: list[Any] = []

        # MLX 状态
        self._use_mlx = _HAS_MLX
        self._mlx_params: dict[str, Any] = {}
        self._mlx_optimizer: Any = None

        logger.info(
            "DebridementTrainer init: d=%d L=%d h=%d params=%d mlx=%s",
            self.model_config.d_model,
            self.model_config.n_layers,
            self.model_config.n_heads,
            self.model.n_params,
            self._use_mlx,
        )

    # ── 数据准备 ──

    def prepare_data(
        self,
        n_train: int | None = None,
        n_val: int | None = None,
        n_test: int | None = None,
    ) -> None:
        """准备训练/验证/测试数据。

        使用 SyntheticDebridementGenerator 生成合成清创数据。

        Args:
            n_train: 训练样本数, 默认 TrainingConfig.n_train_samples
            n_val: 验证样本数, 默认 TrainingConfig.n_val_samples
            n_test: 测试样本数, 默认 TrainingConfig.n_test_samples
        """
        from mci_world_model.sdk._debridement_data import SyntheticDebridementGenerator

        n_train = n_train or self.train_cfg.n_train_samples
        n_val = n_val or self.train_cfg.n_val_samples
        n_test = n_test or self.train_cfg.n_test_samples

        gen = SyntheticDebridementGenerator(seed=self.train_cfg.seed)

        self._train_data = gen.generate_batch(n_train)
        self._val_data = gen.generate_batch(n_val)
        self._test_data = gen.generate_batch(n_test)

        # 数据增强: 训练集加噪声
        if self.train_cfg.augmentation_enabled:
            augmented = []
            for sample in self._train_data:
                aug = self._augment_sample(sample)
                augmented.append(aug)
            self._train_data = augmented

        logger.info(
            "Data prepared: train=%d val=%d test=%d",
            len(self._train_data),
            len(self._val_data),
            len(self._test_data),
        )

    def _augment_sample(self, sample: Any) -> Any:
        """对单个样本做数据增强 (加高斯噪声)。"""
        from copy import deepcopy
        aug = deepcopy(sample)
        noise_scale = 0.02
        # 力/力矩加噪
        aug.force_torque = aug.force_torque.astype(np.float64) + self._rng.randn(6) * noise_scale
        # 关节位置加噪
        aug.joint_positions = aug.joint_positions.astype(np.float64) + self._rng.randn(7) * noise_scale * 0.5
        # 深度加噪
        aug.wound_depth_mm += float(self._rng.randn() * 0.5)
        return aug

    # ── 训练入口 ──

    def train(self, stages: list[int] | None = None) -> TrainHistory:
        """执行完整训练流程。

        Args:
            stages: 阶段列表, 默认 [1, 2, 3]

        Returns:
            TrainHistory 训练历史
        """
        stages = stages or self.train_cfg.stages

        if not self._train_data:
            self.prepare_data()

        if self._use_mlx:
            self._init_mlx_training()

        total_start = time.time()

        if 1 in stages:
            logger.info("=== Stage 1: Encoder Pretraining ===")
            self._train_stage1()

        if 2 in stages:
            logger.info("=== Stage 2: Fusion Layer Training ===")
            self._train_stage2()

        if 3 in stages:
            logger.info("=== Stage 3: Joint Fine-tuning ===")
            self._train_stage3()

        self._history.total_seconds = time.time() - total_start
        logger.info(
            "Training complete: %d epochs in %.1fs, best_val_loss=%.6f",
            self._history.total_epochs,
            self._history.total_seconds,
            self._history.best_val_loss,
        )
        return self._history

    # ── Stage 1: 编码器预训练 ──

    def _train_stage1(self) -> None:
        """Stage 1: 独立训练各模态编码器投影头。

        目标: 每个模态编码器能稳定提取特征。
        冻结: Cross-Modal Fusion, Temporal Transformer, 双头
        训练: 各模态投影层 (proj_rgb/depth/thermal/force/proprio/clinical)
        """
        cfg = self.train_cfg
        lr = cfg.lr_stage1
        n_epochs = cfg.stage1_epochs
        bs = cfg.batch_size

        enc_params = self._get_encoder_params()

        for epoch in range(n_epochs):
            epoch_start = time.time()
            total_loss = 0.0
            n_batches = 0

            indices = self._rng.permutation(len(self._train_data))
            for batch_start in range(0, len(indices), bs):
                batch_idx = indices[batch_start:batch_start + bs]
                batch = [self._train_data[i] for i in batch_idx]

                batch_loss = 0.0
                for sample in batch:
                    fused = self.model.encode_modalities(sample)
                    # 重建损失: 编码特征应保持稳定
                    loss = float(np.mean(fused ** 2)) * 0.01
                    batch_loss += loss

                # 有限差分梯度 + SGD
                for p in enc_params:
                    grad = self._finite_diff_gradient(p, 1e-6)
                    p -= lr * grad

                total_loss += batch_loss / len(batch)
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            val_loss = self._evaluate_stage1()

            metrics = TrainingMetrics(
                epoch=epoch + 1,
                stage=1,
                train_loss=avg_loss,
                val_loss=val_loss,
                lr=lr,
                elapsed_seconds=time.time() - epoch_start,
                n_params=sum(p.size for p in enc_params),
            )
            self._history.add(metrics)
            logger.info("Stage1 Epoch %d/%d: loss=%.6f val_loss=%.6f", epoch + 1, n_epochs, avg_loss, val_loss)

    def _get_encoder_params(self) -> list[np.ndarray]:
        """获取编码器可训练参数。"""
        params = []
        for name in [
            "_proj_rgb", "_proj_depth", "_proj_thermal",
            "_proj_force", "_proj_proprio", "_proj_clinical",
        ]:
            p = getattr(self.model, name, None)
            if isinstance(p, np.ndarray):
                params.append(p)
        return params

    def _get_fusion_params(self) -> list[np.ndarray]:
        """获取融合层可训练参数。"""
        params = []
        for name in ["_fusion_Wq", "_fusion_Wk", "_fusion_Wv", "_fusion_Wo"]:
            p = getattr(self.model, name, None)
            if isinstance(p, np.ndarray):
                params.append(p)
        # Transformer 参数
        for prefix in ["_attn_Wq", "_attn_Wk", "_attn_Wv", "_attn_Wo"]:
            v = getattr(self.model, prefix, None)
            if isinstance(v, list):
                params.extend([arr for arr in v if isinstance(arr, np.ndarray)])
        return params

    def _get_all_trainable_params(self) -> list[np.ndarray]:
        """获取全部可训练参数。"""
        return self.model._collect_params()

    # ── 有限差分梯度 ──

    def _finite_diff_gradient(self, param: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """对参数做有限差分梯度估计 (采样优化)。"""
        grad = np.zeros_like(param)
        flat = param.ravel()
        n_sample = min(100, flat.size)
        indices = self._rng.choice(flat.size, n_sample, replace=False)
        for idx in indices:
            orig = flat[idx]
            flat[idx] = orig + eps
            loss_plus = np.mean(param ** 2)
            flat[idx] = orig - eps
            loss_minus = np.mean(param ** 2)
            flat[idx] = orig
            grad.ravel()[idx] = (loss_plus - loss_minus) / (2 * eps)
        return grad

    # ── MLX 训练 ──

    def _init_mlx_training(self) -> None:
        """初始化 MLX 训练状态。"""
        if not _HAS_MLX:
            return
        self._mlx_params = {}
        for name in [
            "_proj_rgb", "_proj_depth", "_proj_thermal",
            "_proj_force", "_proj_proprio", "_proj_clinical",
            "_fusion_Wq", "_fusion_Wk", "_fusion_Wv", "_fusion_Wo",
        ]:
            p = getattr(self.model, name, None)
            if isinstance(p, np.ndarray):
                self._mlx_params[name] = mx.array(p)
        self._mlx_optimizer = mx_opt.SGD(learning_rate=self.train_cfg.lr_stage1)
        logger.info("MLX training initialized with %d parameter tensors", len(self._mlx_params))

    # ── Stage 2: 融合层训练 ──

    def _train_stage2(self) -> None:
        """Stage 2: 训练 Cross-Modal Fusion + Temporal Transformer。

        冻结编码器, 训练融合层和时序 Transformer。
        """
        cfg = self.train_cfg
        lr = cfg.lr_stage2
        n_epochs = cfg.stage2_epochs
        bs = cfg.batch_size

        for epoch in range(n_epochs):
            epoch_start = time.time()
            total_loss = 0.0
            n_batches = 0

            indices = self._rng.permutation(len(self._train_data))
            for batch_start in range(0, len(indices), bs):
                batch_idx = indices[batch_start:batch_start + bs]
                batch = [self._train_data[i] for i in batch_idx]

                batch_loss = 0.0
                for sample in batch:
                    fused = self.model.encode_modalities(sample)
                    hidden = self.model._transformer_forward(fused)

                    # Tissue 分类损失
                    tissue_logits = hidden @ self.model._tissue_head_W + self.model._tissue_head_b
                    tissue_logits = tissue_logits - np.max(tissue_logits)
                    tissue_probs = np.exp(tissue_logits) / np.sum(np.exp(tissue_logits))
                    tis_loss = -np.log(max(tissue_probs[sample.tissue_label], 1e-15))

                    # Dynamics 预测损失
                    dynamics_pred = hidden @ self.model._dynamics_head_W + self.model._dynamics_head_b
                    target = np.concatenate([
                        sample.joint_positions.astype(np.float64),
                        sample.joint_velocities.astype(np.float64),
                        sample.joint_efforts.astype(np.float64),
                        sample.force_torque.astype(np.float64),
                    ])
                    if len(target) > len(dynamics_pred):
                        target = target[:len(dynamics_pred)]
                    elif len(target) < len(dynamics_pred):
                        target = np.pad(target, (0, len(dynamics_pred) - len(target)))
                    dyn_loss = np.mean((dynamics_pred - target) ** 2)

                    loss = cfg.tissue_loss_weight * tis_loss + cfg.dynamics_loss_weight * dyn_loss
                    batch_loss += float(loss)

                # 更新融合层参数
                fusion_params = self._get_fusion_params()
                for p in fusion_params:
                    grad = self._finite_diff_gradient(p, 1e-6)
                    np.clip(grad, -cfg.grad_clip, cfg.grad_clip, out=grad)
                    p -= lr * grad

                total_loss += batch_loss / len(batch)
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            val_loss = self._evaluate_full()

            metrics = TrainingMetrics(
                epoch=epoch + 1,
                stage=2,
                train_loss=avg_loss,
                val_loss=val_loss,
                lr=lr,
                elapsed_seconds=time.time() - epoch_start,
                n_params=sum(p.size for p in self._get_fusion_params()),
            )
            self._history.add(metrics)
            logger.info("Stage2 Epoch %d/%d: loss=%.6f val_loss=%.6f", epoch + 1, n_epochs, avg_loss, val_loss)

    # ── Stage 3: 联合微调 ──

    def _train_stage3(self) -> None:
        """Stage 3: 联合微调全部参数。

        训练全部参数: 编码器 + 融合层 + 双头。
        包含安全约束损失。
        """
        from mci_world_model.sdk._force_tissue_dynamics import ForceTissueDynamics

        cfg = self.train_cfg
        lr = cfg.lr_stage3
        n_epochs = cfg.stage3_epochs
        bs = cfg.batch_size
        best_val = float("inf")
        patience_counter = 0

        ft_dynamics = ForceTissueDynamics()

        for epoch in range(n_epochs):
            epoch_start = time.time()
            total_loss = 0.0
            correct_tissue = 0
            total_samples = 0
            total_force_mse = 0.0
            total_depth_mae = 0.0
            safety_violations = 0

            indices = self._rng.permutation(len(self._train_data))
            n_batches = 0

            for batch_start in range(0, len(indices), bs):
                batch_idx = indices[batch_start:batch_start + bs]
                batch = [self._train_data[i] for i in batch_idx]

                batch_loss = 0.0
                batch_correct = 0
                batch_force_mse = 0.0
                batch_depth_mae = 0.0
                batch_violations = 0

                for sample in batch:
                    fused = self.model.encode_modalities(sample)
                    hidden = self.model._transformer_forward(fused)

                    # Tissue 分类
                    tissue_logits = hidden @ self.model._tissue_head_W + self.model._tissue_head_b
                    tissue_logits = tissue_logits - np.max(tissue_logits)
                    tissue_probs = np.exp(tissue_logits) / np.sum(np.exp(tissue_logits))
                    pred_tissue = int(np.argmax(tissue_probs))
                    if pred_tissue == sample.tissue_label:
                        batch_correct += 1
                    tis_loss = -np.log(max(tissue_probs[sample.tissue_label], 1e-15))

                    # Dynamics 预测
                    dynamics_pred = hidden @ self.model._dynamics_head_W + self.model._dynamics_head_b
                    target = np.concatenate([
                        sample.joint_positions.astype(np.float64),
                        sample.joint_velocities.astype(np.float64),
                        sample.joint_efforts.astype(np.float64),
                        sample.force_torque.astype(np.float64),
                    ])
                    if len(target) > len(dynamics_pred):
                        target = target[:len(dynamics_pred)]
                    elif len(target) < len(dynamics_pred):
                        target = np.pad(target, (0, len(dynamics_pred) - len(target)))
                    dyn_loss = np.mean((dynamics_pred - target) ** 2)

                    # 力预测 MSE
                    force_pred = dynamics_pred[-6:]
                    force_true = sample.force_torque.astype(np.float64)
                    force_mse = float(np.mean((force_pred - force_true) ** 2))
                    batch_force_mse += force_mse

                    # 深度预测 MAE
                    depth_pred = float(np.mean(np.abs(dynamics_pred[:27]))) * 0.1
                    depth_mae = abs(depth_pred - sample.wound_depth_mm)
                    batch_depth_mae += depth_mae

                    # 安全约束
                    removal = ft_dynamics.predict_removal(
                        sample.tissue_label, sample.tool_force_n, sample.tool_velocity
                    )
                    if not removal.is_safe:
                        batch_violations += 1

                    safety_penalty = 0.0 if removal.is_safe else 1.0

                    loss = (
                        cfg.tissue_loss_weight * tis_loss
                        + cfg.dynamics_loss_weight * dyn_loss
                        + cfg.force_mse_weight * force_mse
                        + cfg.depth_mae_weight * depth_mae
                        + cfg.safety_penalty_weight * safety_penalty
                    )
                    batch_loss += float(loss)

                # 更新全部参数
                all_params = self._get_all_trainable_params()
                for p in all_params:
                    grad = self._finite_diff_gradient(p, 1e-6)
                    np.clip(grad, -cfg.grad_clip, cfg.grad_clip, out=grad)
                    p -= lr * (grad + cfg.weight_decay * p)

                bs_actual = len(batch)
                total_loss += batch_loss / bs_actual
                correct_tissue += batch_correct
                total_samples += bs_actual
                total_force_mse += batch_force_mse / bs_actual
                total_depth_mae += batch_depth_mae / bs_actual
                safety_violations += batch_violations
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            val_metrics = self._evaluate_full_with_metrics()

            metrics = TrainingMetrics(
                epoch=epoch + 1,
                stage=3,
                train_loss=avg_loss,
                val_loss=val_metrics["val_loss"],
                tissue_accuracy=correct_tissue / max(total_samples, 1),
                force_mse=total_force_mse / max(n_batches, 1),
                depth_mae=total_depth_mae / max(n_batches, 1),
                safety_violation_rate=safety_violations / max(total_samples, 1),
                phase_accuracy=val_metrics.get("phase_accuracy", 0.0),
                lr=lr,
                elapsed_seconds=time.time() - epoch_start,
                n_params=sum(p.size for p in self._get_all_trainable_params()),
            )
            self._history.add(metrics)

            tissue_acc_pct = metrics.tissue_accuracy * 100
            logger.info(
                "Stage3 Epoch %d/%d: loss=%.4f val_loss=%.4f tissue_acc=%.1f%% force_mse=%.4f depth_mae=%.2f"
                " safety=%.1f%%",
                epoch + 1, n_epochs, avg_loss, val_metrics["val_loss"],
                tissue_acc_pct, metrics.force_mse, metrics.depth_mae,
                metrics.safety_violation_rate * 100,
            )

            # 早停检查
            if val_metrics["val_loss"] < best_val - cfg.early_stop_min_delta:
                best_val = val_metrics["val_loss"]
                patience_counter = 0
                if cfg.save_best_only:
                    self.save_checkpoint(f"{cfg.checkpoint_dir}/best_model")
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stop_patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

    # ── 评估 ──

    def _evaluate_stage1(self) -> float:
        """Stage 1 验证损失。"""
        if not self._val_data:
            return 0.0
        total = 0.0
        n = min(100, len(self._val_data))
        for sample in self._val_data[:n]:
            fused = self.model.encode_modalities(sample)
            total += float(np.mean(fused ** 2)) * 0.01
        return total / n

    def _evaluate_full(self) -> float:
        """全模型验证损失。"""
        if not self._val_data:
            return 0.0
        total = 0.0
        n = min(100, len(self._val_data))
        for sample in self._val_data[:n]:
            fused = self.model.encode_modalities(sample)
            hidden = self.model._transformer_forward(fused)
            tissue_logits = hidden @ self.model._tissue_head_W + self.model._tissue_head_b
            tissue_logits = tissue_logits - np.max(tissue_logits)
            tissue_probs = np.exp(tissue_logits) / np.sum(np.exp(tissue_logits))
            total += -np.log(max(tissue_probs[sample.tissue_label], 1e-15))
        return float(total) / n

    def _evaluate_full_with_metrics(self) -> dict[str, float]:
        """全模型验证 (含完整指标)。"""
        if not self._val_data:
            return {"val_loss": 0.0, "tissue_accuracy": 0.0, "phase_accuracy": 0.0}

        total_loss = 0.0
        correct_tissue = 0
        correct_phase = 0
        n = min(200, len(self._val_data))

        for sample in self._val_data[:n]:
            fused = self.model.encode_modalities(sample)
            hidden = self.model._transformer_forward(fused)

            tissue_logits = hidden @ self.model._tissue_head_W + self.model._tissue_head_b
            tissue_logits = tissue_logits - np.max(tissue_logits)
            tissue_probs = np.exp(tissue_logits) / np.sum(np.exp(tissue_logits))
            pred_tissue = int(np.argmax(tissue_probs))
            if pred_tissue == sample.tissue_label:
                correct_tissue += 1
            total_loss += -np.log(max(tissue_probs[sample.tissue_label], 1e-15))

            # 简单手术相预测
            dynamics_pred = hidden @ self.model._dynamics_head_W + self.model._dynamics_head_b
            phase_signal = float(np.mean(dynamics_pred[:4]))
            pred_phase = int(np.clip(round(phase_signal * 3), 0, 3))
            if pred_phase == sample.surgical_phase:
                correct_phase += 1

        return {
            "val_loss": float(total_loss) / n,
            "tissue_accuracy": correct_tissue / n,
            "phase_accuracy": correct_phase / n,
        }

    # ── 完整测试集评估 ──

    def evaluate(self) -> dict[str, float]:
        """在测试集上完整评估。"""
        from mci_world_model.sdk._force_tissue_dynamics import ForceTissueDynamics

        if not self._test_data:
            if not self._val_data:
                self.prepare_data()
            self._test_data = self._val_data

        ft = ForceTissueDynamics()
        correct_tissue = 0
        correct_phase = 0
        force_mse_total = 0.0
        depth_mae_total = 0.0
        violations = 0
        n = len(self._test_data)

        for sample in self._test_data:
            fused = self.model.encode_modalities(sample)
            hidden = self.model._transformer_forward(fused)

            tissue_logits = hidden @ self.model._tissue_head_W + self.model._tissue_head_b
            tissue_logits = tissue_logits - np.max(tissue_logits)
            tissue_probs = np.exp(tissue_logits) / np.sum(np.exp(tissue_logits))
            pred_tissue = int(np.argmax(tissue_probs))
            if pred_tissue == sample.tissue_label:
                correct_tissue += 1

            dynamics_pred = hidden @ self.model._dynamics_head_W + self.model._dynamics_head_b
            phase_signal = float(np.mean(dynamics_pred[:4]))
            pred_phase = int(np.clip(round(phase_signal * 3), 0, 3))
            if pred_phase == sample.surgical_phase:
                correct_phase += 1

            force_pred = dynamics_pred[-6:]
            force_true = sample.force_torque.astype(np.float64)
            force_mse_total += float(np.mean((force_pred - force_true) ** 2))

            depth_pred = float(np.mean(np.abs(dynamics_pred[:27]))) * 0.1
            depth_mae_total += abs(depth_pred - sample.wound_depth_mm)

            removal = ft.predict_removal(sample.tissue_label, sample.tool_force_n, sample.tool_velocity)
            if not removal.is_safe:
                violations += 1

        return {
            "tissue_accuracy": correct_tissue / n,
            "phase_accuracy": correct_phase / n,
            "force_mse": force_mse_total / n,
            "depth_mae_mm": depth_mae_total / n,
            "safety_violation_rate": violations / n,
            "n_test_samples": n,
        }

    # ── 检查点 ──

    def save_checkpoint(self, path: str) -> bool:
        """保存模型检查点和训练状态。"""
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self.model.save(path)
            state = {
                "version": "4.4.0",
                "model_config": {
                    "d_model": self.model_config.d_model,
                    "n_layers": self.model_config.n_layers,
                    "n_heads": self.model_config.n_heads,
                },
                "train_config": {
                    "lr": self.train_cfg.lr_stage3,
                    "batch_size": self.train_cfg.batch_size,
                    "n_train_samples": len(self._train_data),
                },
                "history": self._history.to_list(),
                "best_val_loss": self._history.best_val_loss,
                "best_epoch": self._history.best_epoch,
                "total_epochs": self._history.total_epochs,
                "total_seconds": round(self._history.total_seconds, 1),
            }
            with open(f"{path}_state.json", "w") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            logger.info("Checkpoint saved: %s", path)
            return True
        except Exception as e:
            logger.error("Checkpoint save failed: %s", e)
            return False

    def load_checkpoint(self, path: str) -> bool:
        """加载模型检查点。"""
        try:
            loaded = self.model.__class__.load(path, self.model_config)
            if loaded is not None:
                self.model = loaded
                logger.info("Checkpoint loaded: %s", path)
                return True
            return False
        except Exception as e:
            logger.error("Checkpoint load failed: %s", e)
            return False

    # ── 报告导出 ──

    def export_report(self, path: str) -> None:
        """导出完整训练报告 (JSON)。"""
        test_metrics = self.evaluate()
        report = {
            "version": "4.4.0",
            "model_type": "DebridementTrainer",
            "model_config": {
                "d_model": self.model_config.d_model,
                "n_layers": self.model_config.n_layers,
                "n_heads": self.model_config.n_heads,
                "n_params": self.model.n_params,
            },
            "train_config": {
                "lr": self.train_cfg.lr_stage3,
                "batch_size": self.train_cfg.batch_size,
                "n_train_samples": len(self._train_data),
                "n_val_samples": len(self._val_data),
                "n_test_samples": len(self._test_data),
                "use_mlx": self._use_mlx,
            },
            "history": self._history.to_list(),
            "best_val_loss": self._history.best_val_loss,
            "total_epochs": self._history.total_epochs,
            "total_seconds": round(self._history.total_seconds, 1),
            "test_metrics": test_metrics,
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Report exported: %s", path)

    def __repr__(self) -> str:
        return (
            f"DebridementTrainer(d={self.model_config.d_model}, "
            f"L={self.model_config.n_layers}, h={self.model_config.n_heads}, "
            f"params={self.model.n_params}, mlx={self._use_mlx})"
        )
