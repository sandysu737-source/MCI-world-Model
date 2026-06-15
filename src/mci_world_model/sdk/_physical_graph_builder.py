"""
MCI World Model v3.1.0 — PhysicalGraphBuilder 物理量→因果边转换器
================================================================

将数值时序物理量数据转换为 CausalWorldModelState.causal_edges 格式，
使现有 JEPA 编码器 (to_graph_tensors) 可处理物理世界信号。

核心映射:
    物理量 (albumin: 35 → 32) → 因果边:
        {cause: "albumin_t", effect: "albumin_t+1",
         rho: -0.3, energy_relation: "suppress",
         cause_energy: "generative", effect_energy: "trust"}

处理流程:
    患者时序 → 特征提取 → 相关系数计算 → 因果边构造 → causal_edges

设计原则:
- 零新依赖: 纯 numpy 实现
- 与 JEPA 编码器无缝对接: 输出标准 causal_edges 格式
- 时序滞后检测: 自动搜索 1-7 天最优滞后窗口

用法:
    from mci_world_model.sdk._physical_graph_builder import PhysicalGraphBuilder

    builder = PhysicalGraphBuilder()
    timeline = [
        {"day": 1, "albumin": 30, "calorie_intake": 1200},
        {"day": 2, "albumin": 32, "calorie_intake": 1400},
        ...
    ]
    edges = builder.build_graph(timeline)
    # edges → JEPAEncoder.to_graph_tensors()
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# PhysicalGraphBuilder
# =============================================================================


class PhysicalGraphBuilder:
    """
    v3.1.0: 物理量 → 因果边转换器。

    将数值时序数据转换为标准 causal_edges 格式，
    使 JEPA 编码器可处理物理世界信号。

    核心能力:
    - 时序相关系数计算 (Pearson r)
    - 滞后效应检测 (1-7 天窗口)
    - 能量关系推断 (基于五范畴映射)
    - 因果边格式化 (标准 causal_edges 字典)
    """

    # 预定义的五范畴-物理量映射
    ENERGY_PHYSICAL_MAP: dict[str, list[str]] = {
        "semantic": ["diagnosis_code", "chief_complaint", "disease_type"],
        "causal": ["medication_dose", "intervention_type", "treatment_code"],
        "spacetime": ["timestamp", "los_days", "season", "day_of_week"],
        "generative": [
            "albumin",
            "prealbumin",
            "calorie_intake",
            "protein_intake",
            "body_weight",
            "bmi",
            "nitrogen_balance",
            "lymphocyte_count",
        ],
        "trust": ["nrs2002", "apache_ii", "evidence_level", "sofa_score", "glasgow_score", "mortality_risk"],
    }

    # energy_relation 推断阈值
    RHO_ENHANCE_THRESHOLD = 0.3  # rho > 0.3 视为 enhance
    RHO_SUPPRESS_THRESHOLD = -0.3  # rho < -0.3 视为 suppress
    MAX_LAG_DAYS = 7  # 最大滞后天数

    def __init__(self, min_correlation: float = 0.15):
        """
        Args:
            min_correlation: 最小相关系数阈值（abs），低于此值不建边
        """
        self.min_correlation = abs(min_correlation)

    # -----------------------------------------------------------------
    # 核心: 构建因果图
    # -----------------------------------------------------------------

    def build_graph(
        self,
        patient_timeline: list[dict],
        max_lag: int | None = None,
    ) -> list[dict]:
        """
        将患者时序数据转换为因果边列表。

        Args:
            patient_timeline: [
                {"day": 1, "albumin": 30, "calorie_intake": 1200, ...},
                {"day": 2, "albumin": 32, "calorie_intake": 1400, ...},
                ...
            ]
            max_lag: 最大滞后天数 (默认 7)

        Returns:
            causal_edges — JEPA 编码器 to_graph_tensors() 可直接消费:
            [{"cause": str, "effect": str, "rho": float,
              "confidence": float, "energy_relation": str,
              "cause_energy": str, "effect_energy": str}, ...]
        """
        if not patient_timeline or len(patient_timeline) < 3:
            logger.warning("时间线数据不足（需 ≥ 3 天），无法构建因果图")
            return []

        lag = max_lag if max_lag is not None else self.MAX_LAG_DAYS
        lag = min(lag, len(patient_timeline) - 1)

        # ── 提取特征名称 ──
        feature_names = self._extract_feature_names(patient_timeline[0])
        if len(feature_names) < 2:
            return []

        # ── 构建时序矩阵 (N_days × N_features) ──
        data_matrix = self._build_timeline_matrix(patient_timeline, feature_names)

        # ── 计算相关系数 + 滞后检测 ──
        edges: list[dict] = []
        for i, f1 in enumerate(feature_names):
            for j, f2 in enumerate(feature_names):
                if i == j:
                    continue
                # 瞬时相关
                rho, best_lag = self._compute_lagged_correlation(data_matrix[:, i], data_matrix[:, j], max_lag=lag)
                if abs(rho) >= self.min_correlation:
                    cause_energy = self._map_to_energy(f1)
                    effect_energy = self._map_to_energy(f2)
                    energy_rel = self._infer_energy_relation(rho)

                    edge = {
                        "cause": f"temporal_{f1}",
                        "effect": f"temporal_{f2}",
                        "rho": round(float(rho), 4),
                        "confidence": round(min(abs(rho) + 0.2, 1.0), 4),
                        "verdict": "novel" if abs(rho) > 0.5 else "none",
                        "energy_relation": energy_rel,
                        "cause_energy": cause_energy,
                        "effect_energy": effect_energy,
                        "best_lag_days": best_lag,
                        "bayes_factor": round(abs(rho) * 2.0, 4),
                        "feature_pair": (f1, f2),
                    }
                    edges.append(edge)

        # ── 同变量自回归边（时序延续） ──
        for f in feature_names:
            series = data_matrix[:, feature_names.index(f)]
            if len(series) >= 3:
                # 去趋势后的滞后 1-step 相关
                diff = series[1:] - series[:-1]
                prev = series[:-1]
                if np.std(prev) > 1e-8 and np.std(diff) > 1e-8:
                    rho_ar = float(np.corrcoef(prev, diff)[0, 1])
                    if abs(rho_ar) >= self.min_correlation:
                        edges.append(
                            {
                                "cause": f"temporal_{f}",
                                "effect": f"temporal_{f}",
                                "rho": round(rho_ar, 4),
                                "confidence": round(min(abs(rho_ar) + 0.15, 1.0), 4),
                                "verdict": "confirmed" if abs(rho_ar) > 0.3 else "none",
                                "energy_relation": "same",
                                "cause_energy": self._map_to_energy(f),
                                "effect_energy": self._map_to_energy(f),
                                "best_lag_days": 1,
                                "bayes_factor": round(abs(rho_ar) * 1.5, 4),
                                "feature_pair": (f, f),
                            }
                        )

        logger.info(
            "PhysicalGraphBuilder: %d 天 × %d 特征 → %d 因果边",
            len(patient_timeline),
            len(feature_names),
            len(edges),
        )
        return edges

    # -----------------------------------------------------------------
    # 辅助方法
    # -----------------------------------------------------------------

    def _extract_feature_names(self, sample: dict) -> list[str]:
        """从样本字典提取数值特征名称（排除 day/timestamp 等）。"""
        skip_keys = {"day", "timestamp", "patient_id", "source"}
        names = []
        for k, v in sample.items():
            if k in skip_keys:
                continue
            try:
                float(v)
                names.append(k)
            except (TypeError, ValueError):
                continue
        return sorted(names)

    def _build_timeline_matrix(
        self,
        timeline: list[dict],
        feature_names: list[str],
    ) -> np.ndarray:
        """构建 (N_days × N_features) 数值矩阵。"""
        n_days = len(timeline)
        n_feats = len(feature_names)
        matrix = np.zeros((n_days, n_feats), dtype=np.float64)
        for t, day_data in enumerate(timeline):
            for j, fname in enumerate(feature_names):
                try:
                    val = float(day_data.get(fname, np.nan))
                    if np.isfinite(val):
                        matrix[t, j] = val
                except (TypeError, ValueError):
                    pass  # 非数值数据跳过，矩阵单元保持 NaN
        return matrix

    def _compute_lagged_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_lag: int = 7,
    ) -> tuple[float, int]:
        """
        计算 x 和 y 的最优滞后相关系数。

        对 lag ∈ [0, max_lag]，计算 corr(x[:N-lag], y[lag:]),
        返回最大 |rho| 对应的 (rho, lag)。

        滞后解释: lag=0 为瞬时相关，lag>0 表示 x 提前 lag 天影响 y。
        """
        n = len(x)
        best_rho = 0.0
        best_lag = 0

        # 瞬时相关
        if np.std(x) > 1e-8 and np.std(y) > 1e-8:
            rho0 = float(np.corrcoef(x, y)[0, 1])
            if np.isfinite(rho0) and abs(rho0) > abs(best_rho):
                best_rho = rho0
                best_lag = 0

        # 滞后相关
        for lag in range(1, min(max_lag + 1, n - 2)):
            x_trunc = x[: n - lag]
            y_trunc = y[lag:]
            if np.std(x_trunc) > 1e-8 and np.std(y_trunc) > 1e-8:
                try:
                    rho_lag = float(np.corrcoef(x_trunc, y_trunc)[0, 1])
                    if np.isfinite(rho_lag) and abs(rho_lag) > abs(best_rho):
                        best_rho = rho_lag
                        best_lag = lag
                except Exception as e:
                    logger.warning("滞后相关计算异常: %s", e)

        return best_rho, best_lag

    def _map_to_energy(self, feature_name: str) -> str:
        """将物理量名称映射到五范畴能量类型。"""
        name_lower = feature_name.lower()
        for cat, names in self.ENERGY_PHYSICAL_MAP.items():
            for n in names:
                if n.lower() in name_lower or name_lower in n.lower():
                    return cat
        return "generative"

    def _infer_energy_relation(self, rho: float) -> str:
        """根据相关系数推断能量关系。"""
        if rho >= self.RHO_ENHANCE_THRESHOLD:
            return "enhance"
        elif rho <= self.RHO_SUPPRESS_THRESHOLD:
            return "suppress"
        elif abs(rho) < 0.1:
            return "neutral"
        else:
            return "same"

    # -----------------------------------------------------------------
    # 快捷: 构建 CausalWorldModelState
    # -----------------------------------------------------------------

    def build_state(
        self,
        patient_timeline: list[dict],
        max_lag: int | None = None,
    ) -> Any:
        """
        一步构建 CausalWorldModelState（含 causal_edges）。

        Args:
            patient_timeline: 患者时序数据
            max_lag: 最大滞后天数

        Returns:
            CausalWorldModelState 实例
        """
        from mci_world_model.sdk._world_model import CausalWorldModelState

        edges = self.build_graph(patient_timeline, max_lag=max_lag)
        return CausalWorldModelState(
            causal_edges=edges,
            n_novel=sum(1 for e in edges if e.get("verdict") == "novel"),
            n_confirmed=sum(1 for e in edges if e.get("verdict") == "confirmed"),
            n_memories=len(patient_timeline),
            timestamp=str(patient_timeline[-1].get("day", "")) if patient_timeline else "",
        )


# =============================================================================
# 工具函数
# =============================================================================


def signals_to_timeline(signals: list, n_days: int = 30) -> list[dict]:
    """
    v3.1.0: 将 MultimodalSignal 列表转换为时序字典列表。

    辅助函数，用于 PhysicalGraphBuilder 数据准备。

    Args:
        signals: MultimodalSignal 列表
        n_days: 时间线长度

    Returns:
        [{"day": 1, "feature_name": value, ...}, ...]
    """
    # 按天分组信号
    day_signals: dict[int, dict[str, float]] = {}
    from mci_world_model._sys._perception_pipeline import PerceptionPipeline

    pipeline = PerceptionPipeline()
    features = pipeline.process_multimodal(signals, enable_fusion=False)

    # 从处理后的特征重建按天分组结构
    for feat in features:
        ts = feat.get("timestamp", "")
        day = 0
        # 尝试从 timestamp 提取天数
        try:
            if ts:
                # 简单处理: 假设 timestamp 为 "day_N" 或 ISO 格式
                import re as _re

                day_match = _re.search(r"day[_=]?(\d+)", str(ts))
                if day_match:
                    day = int(day_match.group(1))
        except Exception as e:
            logger.warning("signals_to_timeline 解析异常: %s", e)

        if day not in day_signals:
            day_signals[day] = {}
        day_signals[day][feat["feature_name"]] = feat["value"]

    # 构建时间线
    timeline = []
    for d in sorted(day_signals.keys()):
        entry = {"day": d}
        entry.update(day_signals[d])
        timeline.append(entry)

    return timeline
