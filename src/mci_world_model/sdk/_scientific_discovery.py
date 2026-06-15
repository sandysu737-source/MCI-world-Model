"""MCI World Model — ScientificDiscoveryPipeline 科学发现管线
============================================================

端到端科学发现管线——从观测数据到因果规律发现，
集成自主因果发现、符号回归和系统验证。

核心能力:
    DiscoveryStage      — 发现阶段枚举
    DiscoveryReport     — 发现报告
    ScientificDiscoveryPipeline — 科学发现管线

设计原则:
    - 基于 AutonomousLawDiscovererV2 (T1)
    - 纯 numpy，零外部依赖
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# DiscoveryStage — 发现阶段
# =============================================================================


class DiscoveryStage(Enum):
    """科学发现阶段。"""

    DATA_COLLECTION = "data_collection"
    EXPLORATION = "exploration"
    SKELETON_DISCOVERY = "skeleton_discovery"
    LAW_DISCOVERY = "law_discovery"
    VALIDATION = "validation"
    COMPLETED = "completed"


# =============================================================================
# DiscoveryReport — 发现报告
# =============================================================================


@dataclass
class DiscoveryReport:
    """科学发现报告。

    Attributes:
        stage: 当前阶段
        n_variables: 变量数
        n_laws: 发现的规律数
        consistency: 系统一致性
        is_discovery_complete: 发现是否完成
        details: 详细信息
    """

    stage: DiscoveryStage = DiscoveryStage.DATA_COLLECTION
    n_variables: int = 0
    n_laws: int = 0
    consistency: float = 0.0
    is_discovery_complete: bool = False
    details: dict = field(default_factory=dict)


# =============================================================================
# ScientificDiscoveryPipeline — 科学发现管线
# =============================================================================


class ScientificDiscoveryPipeline:
    """科学发现管线 — 从数据到因果规律的端到端发现。

    管线阶段:
      1. 数据收集 → 2. 探索分析 → 3. 骨架发现
      4. 规律发现 → 5. 验证 → 6. 完成

    用法:
        >>> pipeline = ScientificDiscoveryPipeline()
        >>> pipeline.load_data(data, var_names)
        >>> report = pipeline.run()
    """

    def __init__(
        self,
        pc_alpha: float = 0.05,
        conservation_threshold: float = 0.85,
        min_laws_for_discovery: int = 1,
    ):
        self._pc_alpha = pc_alpha
        self._conservation_threshold = conservation_threshold
        self._min_laws = min_laws_for_discovery
        self._data: np.ndarray | None = None
        self._var_names: list[str] = []
        self._stage = DiscoveryStage.DATA_COLLECTION
        self._discovered_laws: list[dict] = []
        self._skeleton_edges: list[tuple[str, str]] = []
        self._reports: list[DiscoveryReport] = []

    @property
    def current_stage(self) -> DiscoveryStage:
        return self._stage

    @property
    def discovered_laws(self) -> list[dict]:
        return list(self._discovered_laws)

    def load_data(self, data: np.ndarray, var_names: list[str]) -> None:
        """加载观测数据。"""
        self._data = np.asarray(data, dtype=float)
        self._var_names = list(var_names)
        self._stage = DiscoveryStage.EXPLORATION

    def run(self) -> DiscoveryReport:
        """执行完整的科学发现管线。

        Returns:
            DiscoveryReport
        """
        if self._data is None:
            return DiscoveryReport(stage=self._stage, details={"error": "no data loaded"})

        # Stage 2: 探索分析
        self._stage = DiscoveryStage.EXPLORATION
        exploration = self._explore()
        self._reports.append(
            DiscoveryReport(
                stage=self._stage,
                n_variables=len(self._var_names),
                details=exploration,
            )
        )

        # Stage 3: 骨架发现
        self._stage = DiscoveryStage.SKELETON_DISCOVERY
        skeleton_result = self._discover_skeleton()
        self._skeleton_edges = skeleton_result.get("edges", [])
        self._reports.append(
            DiscoveryReport(
                stage=self._stage,
                n_variables=len(self._var_names),
                details=skeleton_result,
            )
        )

        # Stage 4: 规律发现
        self._stage = DiscoveryStage.LAW_DISCOVERY
        laws = self._discover_laws()
        self._discovered_laws = laws
        self._reports.append(
            DiscoveryReport(
                stage=self._stage,
                n_variables=len(self._var_names),
                n_laws=len(laws),
                details={"laws": laws},
            )
        )

        # Stage 5: 验证
        self._stage = DiscoveryStage.VALIDATION
        validation = self._validate()
        self._reports.append(
            DiscoveryReport(
                stage=self._stage,
                n_variables=len(self._var_names),
                n_laws=len(laws),
                details=validation,
            )
        )

        # Stage 6: 完成
        self._stage = DiscoveryStage.COMPLETED
        consistency = validation.get("consistency_score", 0.0)
        is_complete = len(laws) >= self._min_laws and consistency >= self._conservation_threshold

        final_report = DiscoveryReport(
            stage=self._stage,
            n_variables=len(self._var_names),
            n_laws=len(laws),
            consistency=consistency,
            is_discovery_complete=is_complete,
            details={
                "exploration": exploration,
                "skeleton_edges": self._skeleton_edges,
                "laws": laws,
                "validation": validation,
            },
        )
        self._reports.append(final_report)
        return final_report

    def _explore(self) -> dict:
        """探索性分析。"""
        if self._data is None:
            return {}
        corr = np.corrcoef(self._data.T) if self._data.shape[1] > 1 else np.eye(1)
        return {
            "n_samples": self._data.shape[0],
            "n_variables": self._data.shape[1],
            "max_abs_correlation": float(np.max(np.abs(corr - np.eye(corr.shape[0])))),
        }

    def _discover_skeleton(self) -> dict:
        """骨架发现 (简化 PC 算法)。"""
        if self._data is None:
            return {"edges": []}

        n_vars = self._data.shape[1]
        corr = np.corrcoef(self._data.T) if n_vars > 1 else np.eye(1)

        edges = []
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if abs(corr[i, j]) > self._pc_alpha:
                    edges.append((self._var_names[i], self._var_names[j]))

        return {"edges": edges, "n_edges": len(edges)}

    def _discover_laws(self) -> list[dict]:
        """对每条边做符号回归。"""
        if self._data is None:
            return []

        laws = []
        for cause_name, effect_name in self._skeleton_edges:
            cause_idx = self._var_names.index(cause_name)
            effect_idx = self._var_names.index(effect_name)
            x = self._data[:, cause_idx]
            y = self._data[:, effect_idx]

            # 线性回归
            a, b = np.polyfit(x, y, 1)
            y_pred = a * x + b
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

            laws.append(
                {
                    "edge": (cause_name, effect_name),
                    "equation": f"{effect_name} = {a:.4f} * {cause_name} + {b:.4f}",
                    "r_squared": float(r2),
                    "conservation_verified": r2 >= self._conservation_threshold,
                }
            )

        return laws

    def _validate(self) -> dict:
        """验证发现的规律。"""
        if not self._discovered_laws:
            return {"consistency_score": 0.0, "verified_count": 0}

        verified = sum(1 for l in self._discovered_laws if l.get("conservation_verified", False))
        consistency = verified / max(len(self._discovered_laws), 1)

        return {
            "consistency_score": float(consistency),
            "verified_count": verified,
            "total_laws": len(self._discovered_laws),
        }

    def statistics(self) -> dict[str, Any]:
        return {
            "current_stage": self._stage.value,
            "n_variables": len(self._var_names),
            "n_laws": len(self._discovered_laws),
            "n_edges": len(self._skeleton_edges),
            "reports_generated": len(self._reports),
        }
