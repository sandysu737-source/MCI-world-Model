from __future__ import annotations

"""
MCI World Model v3.4.0 — CognitiveDiversity 认知多样性度量
==========================================================

实现 Ashby 必要多样性定律的形式化度量: H(C) ≥ H(S)

其中:
    H(S) = 系统状态的 Shannon 熵（物理状态多样性）
    H(C) = 认知控制器的多样性（认知状态多样性）

CEWM 扩展:
    H_CEWM = H_physics + H_causal + H_temporal + H_modal + H_meta

五维认知多样性向量:
    1. H_physics  — 物理状态空间覆盖度（WorldState 向量分布熵）
    2. H_causal   — 因果图拓扑多样性（因果边密度 + 路径多样性）
    3. H_temporal — 时序预测覆盖度（多步预测的方差分布）
    4. H_modal    — 模态融合多样性（多模态信号覆盖度）
    5. H_meta     — 元认知自省深度（认知空洞检测覆盖率）

Ashby 条件验证:
    H_CEWM > H_physics → 认知增强有效（多样性被扩展）
    H_CEWM < H_physics → 认知增强不足（多样性坍缩，违反必要多样性定律）

理论对标:
    - Ashby 必要多样性定律
    - CEWM 论文 §4.2 "认知多样性度量"
    - Kant 先验范畴计算化（量/质/关系/模态）

设计原则:
    - 纯 numpy，零外部依赖
    - 与 WorldState ABC 正交组合
    - 增量计算（新数据不需要重算全部历史）
"""


import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mci_world_model.sdk._world_state import WorldState

logger = logging.getLogger(__name__)


# =============================================================================
# DiversityVector — 五维认知多样性向量
# =============================================================================


@dataclass
class DiversityVector:
    """五维认知多样性向量。

    Attributes:
        h_physics: 物理状态空间覆盖度
        h_causal: 因果图拓扑多样性
        h_temporal: 时序预测覆盖度
        h_modal: 模态融合多样性
        h_meta: 元认知自省深度
        total: H_CEWM = 五维之和
        ashby_satisfied: H_CEWM > H_physics
    """

    h_physics: float = 0.0
    h_causal: float = 0.0
    h_temporal: float = 0.0
    h_modal: float = 0.0
    h_meta: float = 0.0

    @property
    def total(self) -> float:
        """H_CEWM = 五维认知多样性之和。"""
        return self.h_physics + self.h_causal + self.h_temporal + self.h_modal + self.h_meta

    @property
    def ashby_satisfied(self) -> bool:
        """Ashby 条件: H_CEWM > H_physics。"""
        return self.total > self.h_physics

    @property
    def ashby_ratio(self) -> float:
        """H_CEWM / H_physics 比值（>1 = 满足 Ashby 条件）。"""
        if self.h_physics < 1e-10:
            return float("inf") if self.total > 0 else 1.0
        return self.total / self.h_physics

    def to_vector(self) -> np.ndarray:
        """编码为 5 维向量。"""
        return np.array(
            [self.h_physics, self.h_causal, self.h_temporal, self.h_modal, self.h_meta],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "h_physics": round(self.h_physics, 6),
            "h_causal": round(self.h_causal, 6),
            "h_temporal": round(self.h_temporal, 6),
            "h_modal": round(self.h_modal, 6),
            "h_meta": round(self.h_meta, 6),
            "total": round(self.total, 6),
            "ashby_satisfied": self.ashby_satisfied,
            "ashby_ratio": round(self.ashby_ratio, 4),
        }


# =============================================================================
# DiversityHistory — 多样性历史统计
# =============================================================================


@dataclass
class DiversityHistory:
    """多样性度量历史记录统计。"""

    n_samples: int = 0
    diversity_history: list[DiversityVector] = field(default_factory=list)

    @property
    def latest(self) -> DiversityVector | None:
        return self.diversity_history[-1] if self.diversity_history else None

    def trend(self, dimension: str = "total") -> float:
        """计算指定维度的趋势（线性回归斜率）。

        Args:
            dimension: 维度名称 (h_physics/h_causal/.../total)

        Returns:
            趋势斜率（正 = 增长，负 = 衰减）
        """
        if len(self.diversity_history) < 2:
            return 0.0

        if dimension == "total":
            values = [dv.total for dv in self.diversity_history]
        else:
            values = [getattr(dv, dimension, 0.0) for dv in self.diversity_history]

        n = len(values)
        x = np.arange(n, dtype=np.float64)
        y = np.array(values, dtype=np.float64)
        # 简单线性回归斜率
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        if denominator < 1e-10:
            return 0.0
        return float(numerator / denominator)


# =============================================================================
# CognitiveDiversity — 认知多样性计算器
# =============================================================================


class CognitiveDiversity:
    """Ashby 必要多样性定律的形式化度量器。

    计算五维认知多样性向量 H_CEWM = [H_physics, H_causal, H_temporal, H_modal, H_meta]

    核心方法:
        compute(states, ...) → DiversityVector: 一次性计算五维多样性
        update(state) → None: 增量更新（新数据进入）
        current() → DiversityVector: 获取当前多样性估计

    Example:
        >>> cd = CognitiveDiversity()
        >>> from mci_world_model.sdk import PendulumState
        >>> states = [PendulumState(theta=i*0.1, omega=0.0) for i in range(20)]
        >>> dv = cd.compute(states)
        >>> print(dv.total, dv.ashby_satisfied)
    """

    def __init__(self, n_bins: int = 10, max_history: int = 5000):
        """
        Args:
            n_bins: 直方图分箱数（用于熵计算）
            max_history: 最大历史状态数
        """
        self._n_bins = n_bins
        self._max_history = max_history

        # 历史状态向量缓存
        self._state_vectors: list[np.ndarray] = []

        # 因果图信息
        self._causal_edges: list[tuple[str, str, float]] = []
        self._causal_nodes: set[str] = set()

        # 时序预测信息
        self._prediction_errors: list[float] = []

        # 模态信息
        self._modality_counts: dict[str, int] = {}

        # 元认知信息
        self._gap_count: int = 0
        self._total_checks: int = 0

        # 多样性历史
        self._history = DiversityHistory()

    # -----------------------------------------------------------------
    # compute — 一次性计算五维多样性
    # -----------------------------------------------------------------

    def compute(
        self,
        states: list[WorldState] | None = None,
        causal_edges: list[tuple[str, str, float]] | None = None,
        prediction_errors: list[float] | None = None,
        modality_counts: dict[str, int] | None = None,
        gap_count: int = 0,
        total_checks: int = 0,
    ) -> DiversityVector:
        """计算五维认知多样性向量。

        Args:
            states: 世界状态列表（用于 H_physics）
            causal_edges: 因果边列表 [(from, to, weight), ...]（用于 H_causal）
            prediction_errors: 预测误差列表（用于 H_temporal）
            modality_counts: 各模态数据计数（用于 H_modal）
            gap_count: 已发现的认知空洞数（用于 H_meta）
            total_checks: 总检查次数（用于 H_meta）

        Returns:
            DiversityVector 五维认知多样性向量
        """
        # 1. H_physics — 物理状态空间覆盖度
        h_physics = self._compute_physics_diversity(states)

        # 2. H_causal — 因果图拓扑多样性
        h_causal = self._compute_causal_diversity(causal_edges)

        # 3. H_temporal — 时序预测覆盖度
        h_temporal = self._compute_temporal_diversity(prediction_errors)

        # 4. H_modal — 模态融合多样性
        h_modal = self._compute_modal_diversity(modality_counts)

        # 5. H_meta — 元认知自省深度
        h_meta = self._compute_meta_diversity(gap_count, total_checks)

        dv = DiversityVector(
            h_physics=round(h_physics, 6),
            h_causal=round(h_causal, 6),
            h_temporal=round(h_temporal, 6),
            h_modal=round(h_modal, 6),
            h_meta=round(h_meta, 6),
        )

        # 存入历史
        self._history.diversity_history.append(dv)
        self._history.n_samples += 1
        if len(self._history.diversity_history) > self._max_history:
            self._history.diversity_history = self._history.diversity_history[-self._max_history :]

        return dv

    # -----------------------------------------------------------------
    # update — 增量更新
    # -----------------------------------------------------------------

    def update(
        self,
        state: WorldState | None = None,
        causal_edge: tuple[str, str, float] | None = None,
        prediction_error: float | None = None,
        modality: str | None = None,
        gap_detected: bool = False,
    ) -> None:
        """增量更新多样性度量（单个数据点）。

        Args:
            state: 新世界状态
            causal_edge: 新因果边
            prediction_error: 新预测误差
            modality: 新模态名称
            gap_detected: 本次检查是否发现空洞
        """
        if state is not None:
            vec = state.to_vector()
            self._state_vectors.append(vec)
            if len(self._state_vectors) > self._max_history:
                self._state_vectors = self._state_vectors[-self._max_history :]

        if causal_edge is not None:
            self._causal_edges.append(causal_edge)
            self._causal_nodes.add(causal_edge[0])
            self._causal_nodes.add(causal_edge[1])

        if prediction_error is not None:
            self._prediction_errors.append(prediction_error)
            if len(self._prediction_errors) > self._max_history:
                self._prediction_errors = self._prediction_errors[-self._max_history :]

        if modality is not None:
            self._modality_counts[modality] = self._modality_counts.get(modality, 0) + 1

        self._total_checks += 1
        if gap_detected:
            self._gap_count += 1

    def current(self) -> DiversityVector:
        """基于当前缓存数据计算多样性向量。"""
        return self.compute(
            states=None,  # 使用内部缓存
            causal_edges=self._causal_edges if self._causal_edges else None,
            prediction_errors=self._prediction_errors if self._prediction_errors else None,
            modality_counts=self._modality_counts if self._modality_counts else None,
            gap_count=self._gap_count,
            total_checks=self._total_checks,
        )

    # -----------------------------------------------------------------
    # 五维计算内部方法
    # -----------------------------------------------------------------

    def _compute_physics_diversity(self, states: list[WorldState] | None) -> float:
        """H_physics: 物理状态空间覆盖度（Shannon 熵）。"""
        vectors = []
        if states is not None:
            vectors = [s.to_vector() for s in states]
        elif self._state_vectors:
            vectors = self._state_vectors

        if len(vectors) < 2:
            return 0.0

        # 拼接所有向量
        all_vecs = np.stack(vectors)

        # 对每个维度计算直方图熵，取均值
        total_entropy = 0.0
        n_dims = all_vecs.shape[1]

        for d in range(n_dims):
            col = all_vecs[:, d]
            col_range = col.max() - col.min()
            if col_range < 1e-10:
                continue
            # 归一化到 [0, 1]
            normalized = (col - col.min()) / col_range
            # 直方图
            hist, _ = np.histogram(normalized, bins=self._n_bins, range=(0.0, 1.0))
            # 概率分布
            probs = hist / hist.sum()
            # Shannon 熵
            entropy = -sum(p * math.log2(p + 1e-10) for p in probs if p > 0)
            total_entropy += entropy

        # 归一化：除以最大可能熵 log2(n_bins) × n_dims
        max_entropy = math.log2(self._n_bins) * n_dims
        if max_entropy < 1e-10:
            return 0.0

        return total_entropy / max_entropy

    def _compute_causal_diversity(self, edges: list[tuple[str, str, float]] | None) -> float:
        """H_causal: 因果图拓扑多样性。"""
        edge_list = edges if edges is not None else self._causal_edges
        if not edge_list:
            return 0.0

        # 收集所有节点
        nodes: set[str] = set()
        for src, dst, _ in edge_list:
            nodes.add(src)
            nodes.add(dst)

        n_nodes = len(nodes)
        n_edges = len(edge_list)

        if n_nodes < 2:
            return 0.0

        # 1. 边密度
        max_edges = n_nodes * (n_nodes - 1)
        density = n_edges / max(max_edges, 1)

        # 2. 入度/出度分布熵
        in_degree: dict[str, int] = {}
        out_degree: dict[str, int] = {}
        for node in nodes:
            in_degree[node] = 0
            out_degree[node] = 0

        for src, dst, _ in edge_list:
            out_degree[src] = out_degree.get(src, 0) + 1
            in_degree[dst] = in_degree.get(dst, 0) + 1

        # 度分布熵
        degree_values = list(in_degree.values()) + list(out_degree.values())
        if not degree_values:
            return density

        max_deg = max(degree_values)
        if max_deg == 0:
            return density

        hist, _ = np.histogram(degree_values, bins=min(self._n_bins, max_deg + 1), range=(0, max_deg + 1))
        probs = hist / hist.sum()
        degree_entropy = -sum(p * math.log2(p + 1e-10) for p in probs if p > 0)
        max_degree_entropy = math.log2(min(self._n_bins, max_deg + 1))

        degree_diversity = degree_entropy / max(max_degree_entropy, 1e-10)

        # 3. 权重多样性
        weights = [w for _, _, w in edge_list]
        weight_std = float(np.std(weights))

        # 综合：0.4*密度 + 0.4*度分布熵 + 0.2*权重多样性
        return 0.4 * density + 0.4 * degree_diversity + 0.2 * min(1.0, weight_std)

    def _compute_temporal_diversity(self, errors: list[float] | None) -> float:
        """H_temporal: 时序预测覆盖度。"""
        error_list = errors if errors is not None else self._prediction_errors
        if len(error_list) < 2:
            return 0.0

        arr = np.array(error_list, dtype=np.float64)

        # 1. 误差分布熵（归一化）
        err_range = arr.max() - arr.min()
        if err_range < 1e-10:
            return 0.0

        normalized = (arr - arr.min()) / err_range
        hist, _ = np.histogram(normalized, bins=self._n_bins, range=(0.0, 1.0))
        probs = hist / hist.sum()
        entropy = -sum(p * math.log2(p + 1e-10) for p in probs if p > 0)
        max_entropy = math.log2(self._n_bins)
        distribution_diversity = entropy / max(max_entropy, 1e-10)

        # 2. 误差趋势多样性（自相关衰减速度）
        mean_err = np.mean(arr)
        centered = arr - mean_err
        variance = np.var(arr)
        if variance < 1e-10:
            return distribution_diversity

        # 一阶自相关
        autocorr = float(np.mean(centered[:-1] * centered[1:])) / variance
        trend_diversity = 1.0 - abs(autocorr)  # 低自相关 = 高多样性

        return 0.6 * distribution_diversity + 0.4 * max(0.0, trend_diversity)

    def _compute_modal_diversity(self, counts: dict[str, int] | None) -> float:
        """H_modal: 模态融合多样性。"""
        modality = counts if counts is not None else self._modality_counts
        if not modality:
            return 0.0

        total = sum(modality.values())
        if total == 0:
            return 0.0

        # Shannon 熵
        probs = [c / total for c in modality.values()]
        entropy = -sum(p * math.log2(p + 1e-10) for p in probs if p > 0)

        # 归一化：除以最大可能熵 log2(n_modalities)
        n_modalities = len(modality)
        if n_modalities < 2:
            return 0.0

        max_entropy = math.log2(n_modalities)
        normalized_entropy = entropy / max(max_entropy, 1e-10)

        # 覆盖率奖励：使用的模态越多越好（假设最多 5 种模态）
        coverage = min(1.0, n_modalities / 5.0)

        return 0.7 * normalized_entropy + 0.3 * coverage

    def _compute_meta_diversity(self, gap_count: int, total_checks: int) -> float:
        """H_meta: 元认知自省深度。"""
        total = total_checks if total_checks > 0 else self._total_checks
        gaps = gap_count if gap_count > 0 else self._gap_count

        if total == 0:
            return 0.0

        # 1. 检测覆盖率（发现空洞的比例）
        detection_rate = gaps / total

        # 2. 检测深度（基于发现的空洞类型数）
        # 简化：使用检测率 * 检测频率
        check_frequency = min(1.0, total / 100.0)  # 100 次检查为满分

        # 综合
        return 0.6 * detection_rate + 0.4 * check_frequency

    # -----------------------------------------------------------------
    # is_sufficient_diversity — QUAL-02 (S-4): Ashby 阈值量化
    # -----------------------------------------------------------------

    # Ashby 必要多样性定律阈值：H(C)/H(S) > 1.0 即满足。
    # 为避免浮点噪声，使用 1.0 作为严格门限。
    ASHBY_SUFFICIENCY_THRESHOLD: float = 1.0

    def is_sufficient_diversity(self, dv: DiversityVector | None = None) -> bool:
        """判断当前认知多样性是否满足 Ashby 必要多样性定律。

        QUAL-02 (S-4): 提供可编程的布尔门禁，便于在 CI 或运行时断言。

        Ashby 必要多样性定律要求 H(C) ≥ H(S)，即 ashby_ratio ≥ 1.0。
        当 H_physics 为零时（无物理状态数据），返回 False 以标记数据不足。

        Args:
            dv: 可选的多样性向量。为 None 时使用 ``current()`` 实时计算。

        Returns:
            True 如果 ashby_ratio ≥ ASHBY_SUFFICIENCY_THRESHOLD 且 H_physics > 0
        """
        if dv is None:
            dv = self.current()

        # H_physics 为零意味着尚未有足够的物理状态数据
        if dv.h_physics < 1e-10:
            return False

        return dv.ashby_ratio >= self.ASHBY_SUFFICIENCY_THRESHOLD

    @property
    def ashby_ratio(self) -> float:
        """QUAL-02 (S-4): 暴露当前多样性向量的 Ashby 比值。

        等价于 ``self.current().ashby_ratio``，便于外部直接引用。
        """
        return self.current().ashby_ratio

    # -----------------------------------------------------------------
    # ashby_check — Ashby 条件验证
    # -----------------------------------------------------------------

    def ashby_check(self, dv: DiversityVector) -> dict[str, Any]:
        """验证 Ashby 必要多样性定律。

        Args:
            dv: 五维认知多样性向量

        Returns:
            {
                "satisfied": bool,
                "ratio": H_CEWM / H_physics,
                "deficit": H_physics - H_CEWM (正数表示不足),
                "verdict": str
            }
        """
        deficit = dv.h_physics - dv.total
        ratio = dv.ashby_ratio

        if deficit <= 0:
            verdict = "Ashby 条件满足：认知多样性 ≥ 物理状态多样性"
        elif deficit < 0.1:
            verdict = "Ashby 条件边缘：认知多样性略低于物理状态多样性"
        else:
            verdict = "Ashby 条件违反：认知多样性严重不足，需增强认知控制能力"

        return {
            "satisfied": dv.ashby_satisfied,
            "ratio": round(ratio, 4),
            "deficit": round(deficit, 6),
            "verdict": verdict,
            "h_cewm": round(dv.total, 6),
            "h_physics": round(dv.h_physics, 6),
        }

    # -----------------------------------------------------------------
    # history — 历史记录
    # -----------------------------------------------------------------

    def get_history(self) -> DiversityHistory:
        """获取多样性度量历史。"""
        return self._history

    # -----------------------------------------------------------------
    # reset — 重置
    # -----------------------------------------------------------------

    def reset(self) -> None:
        """重置所有缓存和历史。"""
        self._state_vectors.clear()
        self._causal_edges.clear()
        self._causal_nodes.clear()
        self._prediction_errors.clear()
        self._modality_counts.clear()
        self._gap_count = 0
        self._total_checks = 0
        self._history = DiversityHistory()

    # -----------------------------------------------------------------
    # __repr__
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        latest = self._history.latest
        if latest:
            return (
                f"CognitiveDiversity(H_CEWM={latest.total:.4f}, "
                f"ashby={'✓' if latest.ashby_satisfied else '✗'}, "
                f"n_samples={self._history.n_samples})"
            )
        return f"CognitiveDiversity(n_samples={self._history.n_samples})"
