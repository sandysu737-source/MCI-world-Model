from __future__ import annotations

"""
MCI World Model v3.3.0 — SurpriseDetector 惊奇误差检测器
==========================================================

预测 vs 观测的偏差量化——世界模型的"惊奇信号"。

当预测状态与实际观测状态偏差超过阈值时，触发惊奇信号，
驱动 PlanAgent 重规划或 Configurator 调整模型参数。

核心能力:
    compute_surprise(predicted, actual) — 单次惊奇度计算
    detect_anomalies(history)           — 批量异常检测
    running_statistics()                — 滚动统计
    adapt_threshold(n_std)              — 自适应阈值调整

惊奇度量化三维度:
    1. 状态距离 — WorldState.distance()
    2. 向量偏差 — to_vector() 的 L2 距离
    3. 方向偏差 — 向量余弦相似度反转

设计原则:
    - 纯 numpy，零外部依赖
    - 与 WorldState ABC 正交组合
    - 阈值自适应：基于历史惊奇度的均值 + n_std * 标准差
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mci_world_model.sdk._world_state import WorldState

logger = logging.getLogger(__name__)


# =============================================================================
# SurpriseSignal — 惊奇信号
# =============================================================================


@dataclass
class SurpriseSignal:
    """惊奇信号 — 预测 vs 观测的偏差量化结果。

    Attributes:
        score: 惊奇度 [0, 1]，0 = 完美预测，1 = 极度偏离
        predicted: 预测状态
        actual: 实际状态
        breakdown: 各维度惊奇分解
            - "state_distance": WorldState.distance() 归一化
            - "vector_deviation": L2 向量偏差归一化
            - "direction_error": 余弦相似度偏差
        is_anomaly: 是否超过阈值
        threshold: 当前阈值
    """

    score: float
    predicted: WorldState
    actual: WorldState
    breakdown: dict[str, float] = field(default_factory=dict)
    is_anomaly: bool = False
    threshold: float = 0.5


# =============================================================================
# SurpriseDetector — 惊奇误差检测器
# =============================================================================


class SurpriseDetector:
    """惊奇误差检测器 — 预测 vs 观测的偏差量化。

    Example:
        >>> from mci_world_model.sdk import PendulumState
        >>> detector = SurpriseDetector(threshold=0.3)
        >>> predicted = PendulumState(theta=0.5, omega=1.0)
        >>> actual = PendulumState(theta=0.6, omega=0.8)
        >>> signal = detector.compute_surprise(predicted, actual)
        >>> print(signal.score, signal.is_anomaly)
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Args:
            threshold: 惊奇度阈值，超过此值判定为异常
        """
        self._threshold = threshold
        self._history: list[float] = []

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = max(0.0, min(1.0, value))

    @property
    def history(self) -> list[float]:
        return list(self._history)

    @property
    def n_observations(self) -> int:
        return len(self._history)

    # -----------------------------------------------------------------
    # compute_surprise — 单次惊奇度
    # -----------------------------------------------------------------

    def compute_surprise(
        self,
        predicted: WorldState,
        actual: WorldState,
    ) -> SurpriseSignal:
        """计算单次惊奇度。

        三维度融合:
            1. state_distance: WorldState.distance() 归一化到 [0,1]
            2. vector_deviation: to_vector() L2 距离归一化到 [0,1]
            3. direction_error: 1 - cosine_similarity 归一化到 [0,1]

        最终 score = 0.4 * state_distance + 0.3 * vector_deviation + 0.3 * direction_error

        Args:
            predicted: 预测状态
            actual: 实际状态

        Returns:
            SurpriseSignal 惊奇信号
        """
        # 维度1: 状态距离 (通过 WorldState.distance())
        raw_dist = predicted.distance(actual)
        # 归一化: 使用 sigmoid 映射到 [0,1]
        state_distance = 1.0 - np.exp(-raw_dist)

        # 维度2: 向量 L2 偏差
        pred_vec = predicted.to_vector()
        actual_vec = actual.to_vector()
        l2_dist = float(np.linalg.norm(pred_vec - actual_vec))
        # 归一化
        vector_deviation = 1.0 - np.exp(-l2_dist)

        # 维度3: 方向偏差 (余弦相似度)
        pred_norm = float(np.linalg.norm(pred_vec))
        actual_norm = float(np.linalg.norm(actual_vec))
        if pred_norm > 1e-10 and actual_norm > 1e-10:
            cosine_sim = float(np.dot(pred_vec, actual_vec) / (pred_norm * actual_norm))
            cosine_sim = max(-1.0, min(1.0, cosine_sim))
        else:
            cosine_sim = 1.0  # 零向量视为完全一致
        direction_error = (1.0 - cosine_sim) / 2.0  # [0, 1]

        # 加权融合
        score = 0.4 * state_distance + 0.3 * vector_deviation + 0.3 * direction_error
        score = float(np.clip(score, 0.0, 1.0))

        is_anomaly = score >= self._threshold
        self._history.append(score)

        breakdown = {
            "state_distance": round(float(state_distance), 6),
            "vector_deviation": round(float(vector_deviation), 6),
            "direction_error": round(float(direction_error), 6),
        }

        return SurpriseSignal(
            score=round(score, 6),
            predicted=predicted,
            actual=actual,
            breakdown=breakdown,
            is_anomaly=is_anomaly,
            threshold=self._threshold,
        )

    # -----------------------------------------------------------------
    # detect_anomalies — 批量异常检测
    # -----------------------------------------------------------------

    def detect_anomalies(
        self,
        history: list[tuple[WorldState, WorldState]],
    ) -> list[SurpriseSignal]:
        """批量检测 (predicted, actual) 对中的异常。

        Args:
            history: [(predicted_state, actual_state), ...] 预测-实际对列表

        Returns:
            SurpriseSignal 列表（只返回超过阈值的异常信号）
        """
        anomalies: list[SurpriseSignal] = []
        for predicted, actual in history:
            signal = self.compute_surprise(predicted, actual)
            if signal.is_anomaly:
                anomalies.append(signal)
        return anomalies

    # -----------------------------------------------------------------
    # running_statistics — 滚动统计
    # -----------------------------------------------------------------

    def running_statistics(self) -> dict[str, Any]:
        """返回惊奇度历史滚动统计。

        Returns:
            {
                "n": 观测总数,
                "mean": 均值,
                "std": 标准差,
                "max": 最大惊奇度,
                "min": 最小惊奇度,
                "anomaly_rate": 异常率 (is_anomaly 的比例),
            }
        """
        if not self._history:
            return {
                "n": 0,
                "mean": 0.0,
                "std": 0.0,
                "max": 0.0,
                "min": 0.0,
                "anomaly_rate": 0.0,
            }

        arr = np.array(self._history, dtype=np.float64)
        n_anomalies = int(np.sum(arr >= self._threshold))

        return {
            "n": len(self._history),
            "mean": round(float(np.mean(arr)), 6),
            "std": round(float(np.std(arr)), 6),
            "max": round(float(np.max(arr)), 6),
            "min": round(float(np.min(arr)), 6),
            "anomaly_rate": round(n_anomalies / len(self._history), 6),
        }

    # -----------------------------------------------------------------
    # adapt_threshold — 自适应阈值
    # -----------------------------------------------------------------

    def adapt_threshold(self, n_std: float = 2.0) -> float:
        """基于历史数据自适应调整阈值。

        新阈值 = mean + n_std * std

        Args:
            n_std: 标准差倍数（默认 2.0，约 95% 置信）

        Returns:
            新的阈值（已自动应用到 self._threshold）
        """
        if len(self._history) < 2:
            return self._threshold

        arr = np.array(self._history, dtype=np.float64)
        new_threshold = float(np.mean(arr) + n_std * np.std(arr))
        new_threshold = max(0.01, min(0.99, new_threshold))

        self._threshold = new_threshold
        logger.info(
            "SurpriseDetector threshold adapted: %.4f (n=%d, mean=%.4f, std=%.4f)",
            new_threshold,
            len(self._history),
            float(np.mean(arr)),
            float(np.std(arr)),
        )
        return new_threshold

    # -----------------------------------------------------------------
    # reset — 重置
    # -----------------------------------------------------------------

    def reset(self) -> None:
        """清空历史记录，不改变阈值。"""
        self._history.clear()

    # -----------------------------------------------------------------
    # diagnose — 根因分析链 (v3.4.0, VSM System 3*)
    # -----------------------------------------------------------------

    def diagnose(
        self,
        signal: SurpriseSignal,
        causal_graph: dict[str, list[str]] | None = None,
        context: dict | None = None,
    ) -> dict[str, Any]:
        """对单个惊奇信号执行根因分析链。

        VSM System 3* (异常审计):
            SurpriseSignal → 维度分解 → 异常层定位 → 因果图追溯 → 根因报告

        Args:
            signal: 惊奇信号
            causal_graph: 因果图邻接表 {node: [children]}（可选）
            context: 附加上下文信息（可选）

        Returns:
            {
                "root_cause_layer": str,       # 主要异常层
                "dimension_analysis": dict[str, Any],     # 三维度分析
                "causal_chain": list[str],      # 因果链
                "severity": float,              # 综合严重度
                "recommendation": str,          # 建议
                "details": dict[str, Any],
            }
        """
        breakdown = signal.breakdown
        score = signal.score

        # 1. 维度分析
        dimension_analysis = self._analyze_dimensions(breakdown)

        # 2. 定位主要异常层
        root_cause_layer = self._locate_anomaly_layer(breakdown)

        # 3. 因果图追溯
        causal_chain: list[str] = []
        if causal_graph:
            causal_chain = self._trace_causal_chain(breakdown, causal_graph)

        # 4. 生成建议
        recommendation = self._generate_recommendation(root_cause_layer, score)

        # 5. 严重度评估
        severity = self._assess_severity(signal, dimension_analysis)

        return {
            "root_cause_layer": root_cause_layer,
            "dimension_analysis": dimension_analysis,
            "causal_chain": causal_chain,
            "severity": round(severity, 6),
            "recommendation": recommendation,
            "details": {
                "score": score,
                "threshold": signal.threshold,
                "is_anomaly": signal.is_anomaly,
                "history_stats": self.running_statistics(),
                "context": context or {},
            },
        }

    def _analyze_dimensions(self, breakdown: dict[str, float]) -> dict[str, dict[str, Any]]:
        """三维度独立分析。"""
        result = {}
        for dim, value in breakdown.items():
            level = "normal"
            if value > 0.6:
                level = "critical"
            elif value > 0.3:
                level = "warning"
            elif value > 0.1:
                level = "mild"
            result[dim] = {
                "value": value,
                "level": level,
                "interpretation": self._interpret_dimension(dim, value),
            }
        return result

    def _interpret_dimension(self, dim: str, value: float) -> str:
        """解读维度含义。"""
        interpretations = {
            "state_distance": {
                "critical": "世界状态发生剧烈变化，可能是外部干预或系统跳变",
                "warning": "世界状态偏差显著，需检查感知通道是否异常",
                "mild": "世界状态小幅偏离预期，属于正常波动范围",
                "normal": "世界状态与预期一致",
            },
            "vector_deviation": {
                "critical": "预测向量严重偏离实际，预测器可能需要重新校准",
                "warning": "预测向量偏差显著，建议检查模型参数",
                "mild": "预测向量轻微偏离，在容许范围内",
                "normal": "预测向量与实际一致",
            },
            "direction_error": {
                "critical": "预测方向完全错误，因果模型可能存在结构性问题",
                "warning": "预测方向偏差显著，因果推断可能需要更新",
                "mild": "预测方向轻微偏差，不影响整体推断",
                "normal": "预测方向正确",
            },
        }
        dim_map = interpretations.get(dim, {})
        if value > 0.6:
            return dim_map.get("critical", "异常")
        elif value > 0.3:
            return dim_map.get("warning", "偏差")
        elif value > 0.1:
            return dim_map.get("mild", "轻微波动")
        return dim_map.get("normal", "正常")

    def _locate_anomaly_layer(self, breakdown: dict[str, float]) -> str:
        """基于惊奇度分解定位主要异常层。"""
        if not breakdown:
            return "unknown"

        max_dim = max(breakdown, key=lambda k: breakdown[k])
        layer_map = {
            "state_distance": "perception",  # 感知环异常
            "vector_deviation": "prediction",  # 预测环异常
            "direction_error": "cognition",  # 认知环异常
        }
        return layer_map.get(max_dim, "unknown")

    def _trace_causal_chain(self, breakdown: dict[str, float], graph: dict[str, list[str]]) -> list[str]:
        """在因果图上追溯异常传播链。"""
        chain: list[str] = []

        # 从最大异常维度对应的节点开始追溯
        max_dim = max(breakdown, key=lambda k: breakdown[k])
        start_nodes = self._dim_to_nodes(max_dim)

        visited: set[str] = set()
        queue = list(start_nodes)

        while queue and len(chain) < 5:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            # 查找因果图中的相关边
            for parent, children in graph.items():
                if node in children or node == parent:
                    chain.append(f"{parent} → {node}")
                    if parent not in visited:
                        queue.append(parent)

        return chain

    def _dim_to_nodes(self, dim: str) -> list[str]:
        """将异常维度映射到可能的因果图节点。"""
        mapping = {
            "state_distance": ["sensor", "encoder", "perception"],
            "vector_deviation": ["predictor", "jepa", "forward_model"],
            "direction_error": ["causal_graph", "do_calculus", "belief"],
        }
        return mapping.get(dim, ["unknown"])

    def _generate_recommendation(self, layer: str, score: float) -> str:
        """基于异常层和严重度生成建议。"""
        recommendations = {
            "perception": {
                "high": "建议: 检查传感器校准状态，验证感知通道信号质量",
                "low": "建议: 感知轻微偏差，可在下次校准周期时调整",
            },
            "cognition": {
                "high": "建议: 因果模型可能需更新，建议重新运行因果发现算法",
                "low": "建议: 因果推断轻微偏差，可积累更多观测后自动修正",
            },
            "prediction": {
                "high": "建议: 预测器严重偏离，建议触发 PlanAgent 重规划",
                "low": "建议: 预测器小幅偏差，可在自然演化中恢复",
            },
            "action": {
                "high": "建议: 动作执行可能失败，需检查执行器状态",
                "low": "建议: 动作效果轻微偏离，可微调控制参数",
            },
        }
        layer_recs = recommendations.get(layer, recommendations["prediction"])
        level = "high" if score > 0.5 else "low"
        return layer_recs[level]

    def _assess_severity(self, signal: SurpriseSignal, dimension_analysis: dict[str, Any]) -> float:
        """综合评估严重度。"""
        # 基础严重度 = 惊奇度
        base_severity = signal.score

        # 维度加权：关键维度贡献更大
        critical_count = sum(1 for d in dimension_analysis.values() if d.get("level") == "critical")
        warning_count = sum(1 for d in dimension_analysis.values() if d.get("level") == "warning")

        # 多维度同时异常 → 严重度上升
        multi_factor = 1.0 + 0.1 * critical_count + 0.05 * warning_count

        # 历史异常率修正
        stats = self.running_statistics()
        anomaly_rate = stats.get("anomaly_rate", 0.0)
        if anomaly_rate > 0.5:
            # 频繁异常 → 可能系统性问题
            history_factor = 1.2
        else:
            history_factor = 1.0

        severity = base_severity * multi_factor * history_factor
        return float(min(1.0, severity))

    # -----------------------------------------------------------------
    # batch_diagnose — 批量根因分析 (v3.4.0)
    # -----------------------------------------------------------------

    def batch_diagnose(
        self,
        signals: list[SurpriseSignal],
        causal_graph: dict[str, list[str]] | None = None,
    ) -> list[dict[str, Any]]:
        """批量根因分析。

        Args:
            signals: 惊奇信号列表
            causal_graph: 因果图（可选）

        Returns:
            诊断结果列表
        """
        return [self.diagnose(s, causal_graph) for s in signals]

    # -----------------------------------------------------------------
    # 字符串表示
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        return f"SurpriseDetector(threshold={self._threshold:.3f}, n_observations={self.n_observations})"
