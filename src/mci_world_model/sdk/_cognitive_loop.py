from __future__ import annotations

"""
MCI World Model v3.4.0 — CognitiveLoopBus Wiener 四环跨层反馈总线
=================================================================

实现 Wiener 四层嵌套反馈闭环的跨层误差传播总线。

四层闭环（从底到顶）:
    1. PERCEPTION (感知环) — 物理信号 → 世界状态编码
    2. COGNITION  (认知环) — 世界状态 → 因果推理 / 信念更新
    3. PREDICTION (预测环) — 世界状态 + 动作 → 未来状态预测
    4. ACTION     (行动环) — 预测 vs 目标 → 动作搜索 / 执行

跨层误差传播方程:
    Δθ_l(t) = −α · ∇‖e_l(t)‖² + β · e_{l−1}(t)

    其中:
        e_l(t)  = 第 l 层当前时刻误差信号
        α       = 学习率（梯度下降步长）
        β       = 跨层耦合系数（下层误差对上层的影响）
        γ       = 衰减因子（防止跨层反馈震荡）

理论对标:
    - Wiener 控制论四层嵌套反馈
    - Ashby 必要多样性定律 H(C) ≥ H(S)
    - Beer VSM System 1-5 层间通信

设计原则:
    - 纯 numpy，零外部依赖
    - 与 WorldState ABC / SurpriseDetector 正交组合
    - 衰减因子 γ ∈ (0,1) 控制跨层传播幅度（R4 风险缓解）
"""


import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CognitiveLayer — 四层闭环枚举
# =============================================================================


class CognitiveLayer(Enum):
    """Wiener 四层嵌套反馈闭环层级。

    层级编号从底到顶:
        PERCEPTION(0) → COGNITION(1) → PREDICTION(2) → ACTION(3)
    """

    PERCEPTION = 0  # 感知环: 信号采集 → 世界状态编码
    COGNITION = 1  # 认知环: 因果推理 → 信念更新
    PREDICTION = 2  # 预测环: 未来状态推演
    ACTION = 3  # 行动环: 动作搜索 → 执行

    @property
    def index(self) -> int:
        return self.value

    @property
    def label(self) -> str:
        _labels = {
            0: "感知环",
            1: "认知环",
            2: "预测环",
            3: "行动环",
        }
        return _labels[self.value]


# =============================================================================
# ErrorSignal — 层误差信号
# =============================================================================


@dataclass
class ErrorSignal:
    """单层误差信号。

    Attributes:
        layer: 所属层级
        magnitude: 误差幅值 ‖e‖（标量，非负）
        gradient: 误差梯度向量 ∇‖e‖²（可选，用于梯度传播）
        source: 误差来源描述
        timestamp: 产生时间
        metadata: 附加信息
    """

    layer: CognitiveLayer
    magnitude: float
    gradient: np.ndarray | None = None
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.magnitude < 0:
            self.magnitude = abs(self.magnitude)

    @property
    def is_significant(self) -> bool:
        """误差是否显著（> 机器精度）。"""
        return self.magnitude > 1e-8


# =============================================================================
# PropagationResult — 传播结果
# =============================================================================


@dataclass
class PropagationResult:
    """单次跨层传播的完整结果。

    Attributes:
        layer_errors: 传播后各层误差信号
        deltas: 各层参数调整量 Δθ
        cross_coupling: 跨层耦合贡献矩阵 (4×4)
        total_energy: 系统总误差能量 Σ‖e_l‖²
        converged: 是否收敛（总能量 < 收敛阈值）
        step: 传播步数
    """

    layer_errors: dict[CognitiveLayer, ErrorSignal] = field(default_factory=dict)
    deltas: dict[CognitiveLayer, np.ndarray] = field(default_factory=dict)
    cross_coupling: np.ndarray = field(default_factory=lambda: np.zeros((4, 4)))
    total_energy: float = 0.0
    converged: bool = False
    step: int = 0


# =============================================================================
# LoopHealthReport — 闭环健康度报告
# =============================================================================


@dataclass
class LoopHealthReport:
    """四层闭环健康度评估报告。

    Attributes:
        layer_health: 各层健康度 [0,1]，1 = 完全健康
        overall_health: 整体健康度
        bottleneck_layer: 瓶颈层（健康度最低）
        oscillation_detected: 是否检测到震荡
        coupling_balance: 耦合平衡度（各层误差方差倒数）
    """

    layer_health: dict[CognitiveLayer, float] = field(default_factory=dict)
    overall_health: float = 0.0
    bottleneck_layer: CognitiveLayer | None = None
    oscillation_detected: bool = False
    coupling_balance: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# CognitiveLoopBus — 跨层反馈总线
# =============================================================================


class CognitiveLoopBus:
    """Wiener 四环跨层反馈总线 — 认知增强世界模型的闭环通信骨干。

    核心职责:
        1. 收集各层误差信号
        2. 执行跨层误差传播方程
        3. 输出各层参数调整量
        4. 监测闭环健康度

    跨层传播方程:
        Δθ_l(t) = −α · ∇‖e_l(t)‖² + β · e_{l−1}(t) · γ

    参数:
        learning_rate (α): 梯度下降步长，默认 0.01
        coupling_coeff (β): 跨层耦合系数，默认 0.3
        decay_factor (γ): 衰减因子，默认 0.9（R4 风险缓解）
        convergence_threshold: 收敛判定阈值，默认 1e-4
        max_history: 历史记录最大长度，默认 1000

    Example:
        >>> bus = CognitiveLoopBus()
        >>> bus.inject_error(CognitiveLayer.PREDICTION, magnitude=0.5)
        >>> bus.inject_error(CognitiveLayer.COGNITION, magnitude=0.3)
        >>> result = bus.propagate()
        >>> print(result.total_energy, result.converged)
        >>> health = bus.health_report()
        >>> print(health.bottleneck_layer)
    """

    # 默认参数
    DEFAULT_LEARNING_RATE = 0.01
    DEFAULT_COUPLING_COEFF = 0.3
    DEFAULT_DECAY_FACTOR = 0.9
    DEFAULT_CONVERGENCE_THRESHOLD = 1e-4
    DEFAULT_MAX_HISTORY = 1000

    # 四层数量
    N_LAYERS = 4

    def __init__(
        self,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        coupling_coeff: float = DEFAULT_COUPLING_COEFF,
        decay_factor: float = DEFAULT_DECAY_FACTOR,
        convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
        max_history: int = DEFAULT_MAX_HISTORY,
    ):
        """初始化 CognitiveLoopBus。

        Args:
            learning_rate: 梯度下降步长 (α)
            coupling_coeff: 跨层耦合系数 (β)
            decay_factor: 衰减因子 (γ)，防止跨层传播震荡
            convergence_threshold: 收敛判定阈值
            max_history: 历史记录最大长度
        """
        self._alpha = learning_rate
        self._beta = coupling_coeff
        self._gamma = decay_factor
        self._convergence_threshold = convergence_threshold
        self._max_history = max_history

        # 各层当前误差信号
        self._layer_errors: dict[CognitiveLayer, ErrorSignal] = {}
        for layer in CognitiveLayer:
            self._layer_errors[layer] = ErrorSignal(layer=layer, magnitude=0.0)

        # 各层参数调整量
        self._deltas: dict[CognitiveLayer, np.ndarray] = {}
        for layer in CognitiveLayer:
            self._deltas[layer] = np.zeros(1, dtype=np.float64)

        # 跨层耦合矩阵 (4×4)，初始化为单位耦合
        self._coupling_matrix = np.eye(self.N_LAYERS, dtype=np.float64)

        # 历史记录
        self._energy_history: list[float] = []
        self._propagation_results: list[PropagationResult] = []
        self._step_count = 0

    # -----------------------------------------------------------------
    # 属性
    # -----------------------------------------------------------------

    @property
    def learning_rate(self) -> float:
        """梯度下降步长 (α)。"""
        return self._alpha

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self._alpha = max(1e-6, min(1.0, value))

    @property
    def coupling_coeff(self) -> float:
        """跨层耦合系数 (β)。"""
        return self._beta

    @coupling_coeff.setter
    def coupling_coeff(self, value: float) -> None:
        self._beta = max(0.0, min(1.0, value))

    @property
    def decay_factor(self) -> float:
        """衰减因子 (γ)。"""
        return self._gamma

    @decay_factor.setter
    def decay_factor(self, value: float) -> None:
        self._gamma = max(0.0, min(1.0, value))

    @property
    def step_count(self) -> int:
        """已执行的传播步数。"""
        return self._step_count

    @property
    def energy_history(self) -> list[float]:
        """总误差能量历史记录。"""
        return list(self._energy_history)

    # -----------------------------------------------------------------
    # inject_error — 注入层误差
    # -----------------------------------------------------------------

    def inject_error(
        self,
        layer: CognitiveLayer,
        magnitude: float,
        gradient: np.ndarray | None = None,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ErrorSignal:
        """向指定层注入误差信号。

        误差信号将被累加到该层当前误差上（非替换）。

        Args:
            layer: 目标层级
            magnitude: 误差幅值
            gradient: 误差梯度向量（可选）
            source: 误差来源描述
            metadata: 附加信息

        Returns:
            注入后的该层 ErrorSignal
        """
        signal = ErrorSignal(
            layer=layer,
            magnitude=magnitude,
            gradient=gradient,
            source=source,
            metadata=metadata or {},
        )

        # 累加到当前层误差
        current = self._layer_errors[layer]
        combined_magnitude = current.magnitude + signal.magnitude

        # 梯度合并（加权平均）
        if signal.gradient is not None:
            if current.gradient is not None:
                combined_gradient = (current.gradient + signal.gradient) / 2.0
            else:
                combined_gradient = signal.gradient.copy()
        else:
            combined_gradient = current.gradient

        self._layer_errors[layer] = ErrorSignal(
            layer=layer,
            magnitude=combined_magnitude,
            gradient=combined_gradient,
            source=f"{current.source};{source}".strip(";"),
            metadata={**current.metadata, **signal.metadata},
        )

        return self._layer_errors[layer]

    # -----------------------------------------------------------------
    # inject_from_surprise — 从 SurpriseSignal 注入
    # -----------------------------------------------------------------

    def inject_from_surprise(
        self,
        surprise_score: float,
        breakdown: dict[str, float] | None = None,
    ) -> None:
        """从 SurpriseDetector 的惊奇信号自动注入误差到对应层。

        映射规则:
            - state_distance → PERCEPTION 层
            - vector_deviation → PREDICTION 层
            - direction_error → COGNITION 层
            - 总分 → ACTION 层（作为行动触发信号）

        Args:
            surprise_score: 惊奇度总分 [0,1]
            breakdown: 惊奇度分解 {"state_distance", "vector_deviation", "direction_error"}
        """
        if breakdown is None:
            # 无分解信息时，均匀分配到四层
            for layer in CognitiveLayer:
                self.inject_error(layer, magnitude=surprise_score / 4.0, source="surprise")
            return

        # 按维度映射到对应层
        if "state_distance" in breakdown:
            self.inject_error(
                CognitiveLayer.PERCEPTION,
                magnitude=breakdown["state_distance"],
                source="surprise:state_distance",
            )
        if "direction_error" in breakdown:
            self.inject_error(
                CognitiveLayer.COGNITION,
                magnitude=breakdown["direction_error"],
                source="surprise:direction_error",
            )
        if "vector_deviation" in breakdown:
            self.inject_error(
                CognitiveLayer.PREDICTION,
                magnitude=breakdown["vector_deviation"],
                source="surprise:vector_deviation",
            )
        # 总分作为行动环触发
        self.inject_error(
            CognitiveLayer.ACTION,
            magnitude=surprise_score,
            source="surprise:total",
        )

    # -----------------------------------------------------------------
    # propagate — 核心跨层误差传播
    # -----------------------------------------------------------------

    def propagate(self) -> PropagationResult:
        """执行一次跨层误差传播。

        传播方程:
            Δθ_l(t) = −α · ∇‖e_l(t)‖² + β · e_{l−1}(t) · γ

        传播顺序: PERCEPTION → COGNITION → PREDICTION → ACTION
        （自底向上，每层的输出通过耦合矩阵影响上一层）

        Returns:
            PropagationResult 包含各层调整量和系统状态
        """
        self._step_count += 1
        new_errors: dict[CognitiveLayer, ErrorSignal] = {}
        new_deltas: dict[CognitiveLayer, np.ndarray] = {}
        cross_coupling = np.zeros((self.N_LAYERS, self.N_LAYERS), dtype=np.float64)

        # 按层级顺序传播
        layers_ordered = sorted(CognitiveLayer, key=lambda lyr: lyr.index)

        for layer in layers_ordered:
            current_error = self._layer_errors[layer]
            idx = layer.index

            # ── 梯度项: −α · ∇‖e_l‖² ──
            if current_error.gradient is not None:
                grad_term = -self._alpha * current_error.gradient
            else:
                # 无显式梯度时，使用幅值的平方作为标量梯度近似
                grad_term = -self._alpha * np.array([current_error.magnitude**2], dtype=np.float64)

            # ── 跨层耦合项: β · e_{l−1}(t) · γ ──
            coupling_term = np.zeros_like(grad_term)
            if idx > 0:
                # 下层误差
                prev_layer = CognitiveLayer(idx - 1)
                prev_error = self._layer_errors[prev_layer]

                # 耦合贡献
                coupling_weight = self._coupling_matrix[idx, idx - 1] * self._beta * self._gamma
                if prev_error.gradient is not None:
                    coupling_term = coupling_weight * prev_error.gradient
                else:
                    coupling_term = coupling_weight * np.array([prev_error.magnitude], dtype=np.float64)
                cross_coupling[idx, idx - 1] = coupling_weight * prev_error.magnitude

            # 记录所有跨层耦合对（包括非相邻层）
            for other_layer in layers_ordered:
                if other_layer.index != idx:
                    other_error = self._layer_errors[other_layer]
                    w = self._coupling_matrix[idx, other_layer.index]
                    cross_coupling[idx, other_layer.index] = w * other_error.magnitude

            # ── 合成调整量 ──
            delta = grad_term + coupling_term
            new_deltas[layer] = delta

            # ── 更新层误差（施加衰减） ──
            # 新误差 = 旧误差 + 调整量的幅值贡献，然后施加 γ 衰减
            residual_magnitude = abs(current_error.magnitude + float(np.mean(delta)))
            decayed_magnitude = residual_magnitude * self._gamma

            new_errors[layer] = ErrorSignal(
                layer=layer,
                magnitude=decayed_magnitude,
                gradient=delta,
                source=f"propagated:{current_error.source}",
            )

        # ── 计算总能量 ──
        total_energy = sum(sig.magnitude**2 for sig in new_errors.values())

        # ── 收敛判定 ──
        converged = total_energy < self._convergence_threshold

        # ── 更新状态 ──
        self._layer_errors = new_errors
        self._deltas = new_deltas

        # 记录历史
        self._energy_history.append(total_energy)
        if len(self._energy_history) > self._max_history:
            self._energy_history = self._energy_history[-self._max_history :]

        result = PropagationResult(
            layer_errors=dict(new_errors),
            deltas=dict(new_deltas),
            cross_coupling=cross_coupling,
            total_energy=round(total_energy, 8),
            converged=converged,
            step=self._step_count,
        )
        self._propagation_results.append(result)

        logger.debug(
            "CognitiveLoopBus propagate step=%d, energy=%.6f, converged=%s",
            self._step_count,
            total_energy,
            converged,
        )

        return result

    # -----------------------------------------------------------------
    # propagate_n — 多步传播
    # -----------------------------------------------------------------

    def propagate_n(self, n: int, early_stop: bool = True) -> list[PropagationResult]:
        """连续执行 n 步传播。

        Args:
            n: 传播步数
            early_stop: 若收敛则提前终止

        Returns:
            PropagationResult 列表
        """
        results: list[PropagationResult] = []
        for _ in range(n):
            result = self.propagate()
            results.append(result)
            if early_stop and result.converged:
                logger.info("CognitiveLoopBus converged at step %d", result.step)
                break
        return results

    # -----------------------------------------------------------------
    # get_layer_error — 查询层误差
    # -----------------------------------------------------------------

    def get_layer_error(self, layer: CognitiveLayer) -> ErrorSignal:
        """获取指定层的当前误差信号。"""
        return self._layer_errors[layer]

    def get_all_errors(self) -> dict[CognitiveLayer, ErrorSignal]:
        """获取所有层的当前误差信号。"""
        return dict(self._layer_errors)

    def get_delta(self, layer: CognitiveLayer) -> np.ndarray:
        """获取指定层的最近一次参数调整量。"""
        return self._deltas[layer]

    # -----------------------------------------------------------------
    # set_coupling — 设置耦合矩阵
    # -----------------------------------------------------------------

    def set_coupling(self, from_layer: CognitiveLayer, to_layer: CognitiveLayer, weight: float) -> None:
        """设置层间耦合权重。

        Args:
            from_layer: 源层
            to_layer: 目标层
            weight: 耦合权重 [0,1]
        """
        weight = max(0.0, min(1.0, weight))
        self._coupling_matrix[to_layer.index, from_layer.index] = weight

    def get_coupling_matrix(self) -> np.ndarray:
        """获取完整耦合矩阵副本。"""
        return self._coupling_matrix.copy()

    # -----------------------------------------------------------------
    # adapt_parameters — 自适应参数调整
    # -----------------------------------------------------------------

    def adapt_parameters(self, strategy: str = "energy") -> dict[str, float]:
        """基于历史数据自适应调整传播参数。

        策略:
            - "energy": 基于能量变化趋势调整 α 和 γ
            - "oscillation": 检测震荡并自动降低 β

        Args:
            strategy: 自适应策略

        Returns:
            调整后的参数字典
        """
        params = {
            "alpha": self._alpha,
            "beta": self._beta,
            "gamma": self._gamma,
        }

        if len(self._energy_history) < 3:
            return params

        recent = self._energy_history[-10:]
        energy_arr = np.array(recent, dtype=np.float64)

        if strategy == "energy":
            # 如果能量在上升，降低学习率
            if len(recent) >= 3 and recent[-1] > recent[-3]:
                self._alpha = max(1e-5, self._alpha * 0.9)
                params["alpha"] = self._alpha
            # 如果能量很低，提高衰减以加速收敛
            if energy_arr[-1] < self._convergence_threshold * 10:
                self._gamma = min(0.99, self._gamma * 1.01)
                params["gamma"] = self._gamma

        elif strategy == "oscillation":
            # 检测震荡：能量交替上升下降
            oscillation_count = 0
            for i in range(1, len(recent) - 1):
                if ((recent[i] > recent[i - 1]) and (recent[i] > recent[i + 1])) or (
                    (recent[i] < recent[i - 1]) and (recent[i] < recent[i + 1])
                ):
                    oscillation_count += 1

            if oscillation_count >= 2:
                # 检测到震荡，降低耦合系数
                self._beta = max(0.01, self._beta * 0.8)
                self._gamma = max(0.5, self._gamma * 0.95)
                params["beta"] = self._beta
                params["gamma"] = self._gamma
                logger.warning(
                    "CognitiveLoopBus oscillation detected, reducing β=%.4f, γ=%.4f",
                    self._beta,
                    self._gamma,
                )

        return params

    # -----------------------------------------------------------------
    # health_report — 闭环健康度评估
    # -----------------------------------------------------------------

    def health_report(self) -> LoopHealthReport:
        """生成四层闭环健康度评估报告。

        健康度计算:
            - 各层健康度 = 1 - min(1, 当前误差 / 参考阈值)
            - 整体健康度 = 加权均值（认知环权重最高）
            - 震荡检测 = 近期能量方差异常

        Returns:
            LoopHealthReport
        """
        # 各层参考阈值（经验值）
        thresholds = {
            CognitiveLayer.PERCEPTION: 0.5,
            CognitiveLayer.COGNITION: 0.3,
            CognitiveLayer.PREDICTION: 0.4,
            CognitiveLayer.ACTION: 0.6,
        }

        # 各层权重（认知环最重要）
        weights = {
            CognitiveLayer.PERCEPTION: 0.2,
            CognitiveLayer.COGNITION: 0.35,
            CognitiveLayer.PREDICTION: 0.25,
            CognitiveLayer.ACTION: 0.2,
        }

        layer_health: dict[CognitiveLayer, float] = {}
        for layer in CognitiveLayer:
            error = self._layer_errors[layer]
            threshold = thresholds[layer]
            health = max(0.0, 1.0 - min(1.0, error.magnitude / threshold))
            layer_health[layer] = round(health, 4)

        # 整体健康度（加权均值）
        overall = sum(layer_health[lyr] * weights[lyr] for lyr in CognitiveLayer)

        # 瓶颈层
        bottleneck = min(layer_health, key=lambda lyr: layer_health[lyr])

        # 震荡检测
        oscillation_detected = False
        if len(self._energy_history) >= 5:
            recent = np.array(self._energy_history[-10:], dtype=np.float64)
            diffs = np.diff(recent)
            sign_changes = sum(1 for i in range(1, len(diffs)) if diffs[i] * diffs[i - 1] < 0)
            oscillation_detected = sign_changes >= 3

        # 耦合平衡度
        magnitudes = [self._layer_errors[lyr].magnitude for lyr in CognitiveLayer]
        if max(magnitudes) > 1e-8:
            mag_std = float(np.std(magnitudes))
            mag_mean = float(np.mean(magnitudes))
            coupling_balance = 1.0 - min(1.0, mag_std / max(mag_mean, 1e-8))
        else:
            coupling_balance = 1.0

        return LoopHealthReport(
            layer_health=layer_health,
            overall_health=round(overall, 4),
            bottleneck_layer=bottleneck,
            oscillation_detected=oscillation_detected,
            coupling_balance=round(coupling_balance, 4),
            details={
                "step_count": self._step_count,
                "total_energy": self._energy_history[-1] if self._energy_history else 0.0,
                "convergence_threshold": self._convergence_threshold,
                "parameters": {
                    "alpha": self._alpha,
                    "beta": self._beta,
                    "gamma": self._gamma,
                },
            },
        )

    # -----------------------------------------------------------------
    # layer_connectivity — 层间连通性检查
    # -----------------------------------------------------------------

    def layer_connectivity(self) -> dict[str, bool]:
        """检查四层闭环互联状态。

        Returns:
            {
                "perception_to_cognition": bool,
                "cognition_to_prediction": bool,
                "prediction_to_action": bool,
                "action_to_perception": bool,  # 反馈环
                "all_connected": bool,
            }
        """
        cm = self._coupling_matrix
        connectivity = {
            "perception_to_cognition": bool(cm[1, 0] > 0),
            "cognition_to_prediction": bool(cm[2, 1] > 0),
            "prediction_to_action": bool(cm[3, 2] > 0),
            "action_to_perception": bool(cm[0, 3] > 0),
        }
        connectivity["all_connected"] = all(v for k, v in connectivity.items())
        return connectivity

    # -----------------------------------------------------------------
    # reset — 重置
    # -----------------------------------------------------------------

    def reset(self, clear_history: bool = True) -> None:
        """重置总线状态。

        Args:
            clear_history: 是否同时清空历史记录
        """
        for layer in CognitiveLayer:
            self._layer_errors[layer] = ErrorSignal(layer=layer, magnitude=0.0)
            self._deltas[layer] = np.zeros(1, dtype=np.float64)

        self._coupling_matrix = np.eye(self.N_LAYERS, dtype=np.float64)

        if clear_history:
            self._energy_history.clear()
            self._propagation_results.clear()
            self._step_count = 0

        logger.debug("CognitiveLoopBus reset (clear_history=%s)", clear_history)

    # -----------------------------------------------------------------
    # running_statistics — 运行统计
    # -----------------------------------------------------------------

    def running_statistics(self) -> dict[str, Any]:
        """返回总线运行统计信息。"""
        if not self._energy_history:
            return {
                "steps": 0,
                "mean_energy": 0.0,
                "max_energy": 0.0,
                "convergence_rate": 0.0,
                "avg_step_energy_reduction": 0.0,
            }

        arr = np.array(self._energy_history, dtype=np.float64)

        # 收敛率（能量低于阈值的比例）
        convergence_rate = float(np.sum(arr < self._convergence_threshold) / len(arr))

        # 平均步间能量变化
        if len(arr) > 1:
            diffs = np.diff(arr)
            avg_reduction = float(-np.mean(diffs))
        else:
            avg_reduction = 0.0

        return {
            "steps": self._step_count,
            "mean_energy": round(float(np.mean(arr)), 6),
            "max_energy": round(float(np.max(arr)), 6),
            "convergence_rate": round(convergence_rate, 4),
            "avg_step_energy_reduction": round(avg_reduction, 6),
            "current_energy": round(float(arr[-1]), 6),
            "parameters": {
                "alpha": self._alpha,
                "beta": self._beta,
                "gamma": self._gamma,
            },
        }

    # -----------------------------------------------------------------
    # __repr__
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        energy = self._energy_history[-1] if self._energy_history else 0.0
        return (
            f"CognitiveLoopBus(steps={self._step_count}, "
            f"α={self._alpha:.4f}, β={self._beta:.4f}, γ={self._gamma:.4f}, "
            f"energy={energy:.6f})"
        )
