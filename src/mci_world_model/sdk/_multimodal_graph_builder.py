from __future__ import annotations

"""
MCI World Model v3.3.0 — MultimodalGraphBuilder 多模态图构建器
================================================================

将多模态特征时序数据构建为跨模态因果图。

与 PhysicalGraphBuilder 的关系:
    PhysicalGraphBuilder: 标量时序 → Pearson 相关 → 因果边
    MultimodalGraphBuilder: 向量时序 → 向量级相关 → 模态内 + 跨模态因果边

核心方法:
    build_from_features(feature_timeline)  — 多模态时序 → 因果边
    build_cross_modality_edges(...)        — 两个模态间的因果边

设计原则:
    - 纯 numpy，零外部依赖
    - 复用 PhysicalGraphBuilder._compute_lagged_correlation() 逻辑
    - 支持向量级相关（cosine similarity 替代 Pearson）
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MultimodalGraphBuilder:
    """多模态特征时序 → 跨模态因果图。

    Example:
        >>> builder = MultimodalGraphBuilder(min_correlation=0.3)
        >>> timeline = [
        ...     {"vision": np.array([0.1, 0.2]), "audio": np.array([0.3, 0.4])},
        ...     {"vision": np.array([0.2, 0.3]), "audio": np.array([0.4, 0.5])},
        ...     {"vision": np.array([0.3, 0.4]), "audio": np.array([0.5, 0.6])},
        ... ]
        >>> edges = builder.build_from_features(timeline)
    """

    def __init__(self, min_correlation: float = 0.3, max_lag: int = 3) -> None:
        """
        Args:
            min_correlation: 最小相关阈值
            max_lag: 最大滞后步数
        """
        self._min_corr = min_correlation
        self._max_lag = max_lag

    @property
    def min_correlation(self) -> float:
        return self._min_corr

    # -----------------------------------------------------------------
    # build_from_features — 主入口
    # -----------------------------------------------------------------

    def build_from_features(
        self,
        feature_timeline: list[dict[str, np.ndarray]],
    ) -> list[dict[str, Any]]:
        """从多模态特征时序构建因果图。

        Args:
            feature_timeline: [
                {modality_name: feature_vector, ...},  # t=0
                {modality_name: feature_vector, ...},  # t=1
                ...
            ]

        Returns:
            因果边列表 [{cause, effect, correlation, lag, modality_pair, ...}, ...]
        """
        if len(feature_timeline) < 3:
            return []

        # 收集所有模态名
        all_modalities: set[str] = set()
        for step in feature_timeline:
            all_modalities.update(step.keys())
        modalities = sorted(all_modalities)

        edges: list[dict[str, Any]] = []

        # 模态内因果边（同一模态不同时间步的自回归）
        for mod in modalities:
            series = self._extract_series(feature_timeline, mod)
            if series is None or len(series) < 3:
                continue
            auto_edges = self._build_auto_edges(series, mod)
            edges.extend(auto_edges)

        # 跨模态因果边
        for i, mod_a in enumerate(modalities):
            for mod_b in modalities[i + 1 :]:
                series_a = self._extract_series(feature_timeline, mod_a)
                series_b = self._extract_series(feature_timeline, mod_b)
                if series_a is None or series_b is None:
                    continue
                if len(series_a) < 3 or len(series_b) < 3:
                    continue
                cross_edges = self.build_cross_modality_edges(
                    series_a,
                    series_b,
                    mod_a,
                    mod_b,
                )
                edges.extend(cross_edges)

        return edges

    # -----------------------------------------------------------------
    # build_cross_modality_edges — 跨模态因果边
    # -----------------------------------------------------------------

    def build_cross_modality_edges(
        self,
        features_a: np.ndarray,
        features_b: np.ndarray,
        modality_a: str,
        modality_b: str,
    ) -> list[dict[str, Any]]:
        """两个模态之间的因果边检测。

        使用滞后余弦相似度检测因果关系。

        Args:
            features_a: (T, D_a) 模态 A 的特征时序
            features_b: (T, D_b) 模态 B 的特征时序
            modality_a: 模态 A 名称
            modality_b: 模态 B 名称

        Returns:
            因果边列表
        """
        T = min(len(features_a), len(features_b))
        if T < 3:
            return []

        fa = features_a[:T]
        fb = features_b[:T]

        edges: list[dict[str, Any]] = []

        # A → B (各 lag)
        for lag in range(0, self._max_lag + 1):
            corr_ab = self._vector_lagged_correlation(fa, fb, lag)
            if abs(corr_ab) >= self._min_corr:
                edges.append(
                    {
                        "cause": f"{modality_a}_feature",
                        "effect": f"{modality_b}_feature",
                        "correlation": round(corr_ab, 6),
                        "lag": lag,
                        "modality_pair": (modality_a, modality_b),
                        "confidence": round(abs(corr_ab), 4),
                        "direction": f"{modality_a}→{modality_b}",
                    }
                )

        # B → A (各 lag, lag > 0 避免重复)
        for lag in range(1, self._max_lag + 1):
            corr_ba = self._vector_lagged_correlation(fb, fa, lag)
            if abs(corr_ba) >= self._min_corr:
                edges.append(
                    {
                        "cause": f"{modality_b}_feature",
                        "effect": f"{modality_a}_feature",
                        "correlation": round(corr_ba, 6),
                        "lag": lag,
                        "modality_pair": (modality_b, modality_a),
                        "confidence": round(abs(corr_ba), 4),
                        "direction": f"{modality_b}→{modality_a}",
                    }
                )

        return edges

    # -----------------------------------------------------------------
    # 辅助方法
    # -----------------------------------------------------------------

    def _extract_series(
        self,
        timeline: list[dict[str, np.ndarray]],
        modality: str,
    ) -> np.ndarray | None:
        """从时序中提取指定模态的特征矩阵 (T, D)。"""
        series = []
        for step in timeline:
            if modality in step:
                vec = np.asarray(step[modality], dtype=np.float64).flatten()
                series.append(vec)
        if not series:
            return None
        # 确保所有向量维度一致
        d = len(series[0])
        for s in series[1:]:
            if len(s) != d:
                return None
        return np.array(series)

    def _build_auto_edges(
        self,
        series: np.ndarray,
        modality: str,
    ) -> list[dict[str, Any]]:
        """构建自回归因果边（t → t+1）。"""
        if len(series) < 3:
            return []

        # 使用差分作为变化量
        diffs = np.diff(series, axis=0)  # (T-1, D)
        if diffs.shape[0] < 2:
            return []

        # 计算当前值与变化量的相关（均值化到标量）
        mean_vals = np.mean(series[:-1], axis=1)  # (T-1,)
        mean_diffs = np.mean(diffs, axis=1)  # (T-1,)

        corr = self._scalar_correlation(mean_vals, mean_diffs)
        if abs(corr) >= self._min_corr:
            return [
                {
                    "cause": f"{modality}_auto",
                    "effect": f"{modality}_delta",
                    "correlation": round(corr, 6),
                    "lag": 1,
                    "modality_pair": (modality, modality),
                    "confidence": round(abs(corr), 4),
                    "direction": f"{modality}→{modality}_next",
                }
            ]
        return []

    @staticmethod
    def _vector_lagged_correlation(
        a: np.ndarray,
        b: np.ndarray,
        lag: int,
    ) -> float:
        """计算两个特征矩阵之间的滞后余弦相似度。

        Args:
            a: (T, D_a)
            b: (T, D_b)
            lag: 滞后步数 (b 相对于 a 的滞后)

        Returns:
            相关系数 [-1, 1]
        """
        T = min(len(a), len(b))
        if lag + 1 >= T:
            return 0.0

        # 对齐: a[:T-lag], b[lag:T]
        a_slice = a[: T - lag]
        b_slice = b[lag:T]

        # 降维到标量（取每帧向量均值）
        a_scalar = np.mean(a_slice, axis=1)
        b_scalar = np.mean(b_slice, axis=1)

        if len(a_scalar) < 2:
            return 0.0

        # Pearson 相关
        a_centered = a_scalar - np.mean(a_scalar)
        b_centered = b_scalar - np.mean(b_scalar)
        denom = np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2))
        if denom < 1e-10:
            return 0.0
        return float(np.sum(a_centered * b_centered) / denom)

    @staticmethod
    def _scalar_correlation(a: np.ndarray, b: np.ndarray) -> float:
        """标量 Pearson 相关。"""
        if len(a) < 2 or len(b) < 2:
            return 0.0
        n = min(len(a), len(b))
        a = a[:n]
        b = b[:n]
        a_c = a - np.mean(a)
        b_c = b - np.mean(b)
        denom = np.sqrt(np.sum(a_c**2) * np.sum(b_c**2))
        if denom < 1e-10:
            return 0.0
        return float(np.sum(a_c * b_c) / denom)

    def __repr__(self) -> str:
        return f"MultimodalGraphBuilder(min_corr={self._min_corr}, max_lag={self._max_lag})"
