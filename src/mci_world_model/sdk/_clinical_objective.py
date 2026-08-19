"""MCI World Model — 临床目标函数（ClinicalObjective）

============================================================

Phase 2 模块：医疗世界模型的评估函数 R。

这是世界模型五要素中的第四个（评估 R），回答核心问题：
    "患者状态是好是坏？"

评估维度（多目标加权）：
    1. 体征稳定度：越接近正常范围越好
    2. 器官功能：SOFA 评分越低越好
    3. 安全约束违反惩罚：越界体征重罚

设计原则：
    - 基于 PatientState 的客观临床指标（SOFA/正常范围），不用自造指标
    - reward ∈ [0, 1]：1 = 完全健康，0 = 危重
    - 硬安全约束：is_safe() 为 False 时 reward 大幅降低
    - 无状态：纯函数，不持有跨调用状态

与 MCTSPlanner 的集成：
    ClinicalMCTSPlanner 用 ClinicalObjective.reward(state) 替代默认的 distance(goal)
"""

from __future__ import annotations

import numpy as np

from mci_world_model.sdk._clinical_world_state import (
    N_VITALS,
    VITAL_NAMES,
    VITAL_NORMAL_RANGES,
    PatientState,
)

# =============================================================================
# ClinicalObjective — 临床目标函数
# =============================================================================


class ClinicalObjective:
    """临床目标函数 — 评估患者状态好坏（世界模型评估层 R）。

    组合三个维度的评分：

    1. **体征稳定度**（权重 w_stability）：
       每个体征值与正常范围中点的标准化距离，取平均。
       完全正常 → 1.0，偏离越远 → 越低。

    2. **器官功能**（权重 w_organ）：
       基于 SOFA 评分（PatientState.sofa_score()），归一化到 [0, 1]。
       SOFA=0 → 1.0，SOFA 越高 → 越低。

    3. **安全约束**（权重 w_safety）：
       is_safe() 为 True → 1.0，每项违规扣分。

    最终 reward = w_stability × stability + w_organ × organ + w_safety × safety

    Example:
        >>> obj = ClinicalObjective()
        >>> state = PatientState(vital_signs=np.array([[75, 120, 80, 98, 16, 36.8, 15]]))
        >>> reward = obj.reward(state)  # → ~0.95（接近正常）
        >>> assert obj.is_safe(state) is True
    """

    def __init__(
        self,
        w_stability: float = 0.4,
        w_organ: float = 0.35,
        w_safety: float = 0.25,
    ) -> None:
        """初始化临床目标函数。

        Args:
            w_stability: 体征稳定度权重。
            w_organ: 器官功能权重。
            w_safety: 安全约束权重。

        三者之和应为 1.0（自动归一化）。
        """
        total = w_stability + w_organ + w_safety
        self._w_stability = w_stability / total
        self._w_organ = w_organ / total
        self._w_safety = w_safety / total

        # 预计算正常范围中点和半宽（用于稳定度评分）
        self._vital_mids = np.array(
            [(VITAL_NORMAL_RANGES[v][0] + VITAL_NORMAL_RANGES[v][1]) / 2.0 for v in VITAL_NAMES]
        )
        self._vital_halfs = np.array(
            [max((VITAL_NORMAL_RANGES[v][1] - VITAL_NORMAL_RANGES[v][0]) / 2.0, 1e-6) for v in VITAL_NAMES]
        )

    @property
    def weights(self) -> dict[str, float]:
        """当前权重配置。"""
        return {
            "stability": round(self._w_stability, 3),
            "organ": round(self._w_organ, 3),
            "safety": round(self._w_safety, 3),
        }

    def reward(self, state: PatientState) -> float:
        """评估患者状态好坏。

        Args:
            state: 待评估的 PatientState。

        Returns:
            reward ∈ [0, 1]。1 = 完全健康，0 = 危重。
        """
        return (
            self._w_stability * self._stability_score(state)
            + self._w_organ * self._organ_score(state)
            + self._w_safety * self._safety_score(state)
        )

    def is_safe(self, state: PatientState) -> bool:
        """安全约束检查（硬约束，规划器剪枝用）。

        调用 PatientState.is_physiologically_valid()：
        超出生理可行范围（致命值）的状态不安全。
        """
        return state.is_physiologically_valid()

    def detail(self, state: PatientState) -> dict[str, float]:
        """返回详细评分分解（审计/调试用）。

        Returns:
            {"stability", "organ", "safety", "sofa", "reward"} 字典。
        """
        stability = self._stability_score(state)
        organ = self._organ_score(state)
        safety = self._safety_score(state)
        return {
            "stability": round(stability, 4),
            "organ": round(organ, 4),
            "safety": round(safety, 4),
            "sofa": round(state.sofa_score(), 2),
            "reward": round(self.reward(state), 4),
            "is_safe": 1.0 if self.is_safe(state) else 0.0,
        }

    # ── 子评分函数 ──────────────────────────────────────────────

    def _stability_score(self, state: PatientState) -> float:
        """体征稳定度评分。

        每个体征与正常中点的标准化距离，转换为 [0, 1] 分数。
        完全正常 = 1.0，偏离一个半宽 = 0.5，偏离两个半宽 ≈ 0。
        """
        latest = state.vital_signs[-1]
        # 标准化偏差（绝对值，单位 = 半宽）
        deviations = np.abs(latest - self._vital_mids) / self._vital_halfs
        # 转换为分数：1/(1+dev^2)，dev=0→1, dev=1→0.5, dev=2→0.2
        scores = 1.0 / (1.0 + deviations**2)
        return float(np.mean(scores))

    @staticmethod
    def _organ_score(state: PatientState) -> float:
        """器官功能评分（基于 SOFA）。

        SOFA 评分范围 0-24+，归一化为 1 - sofa/24。
        SOFA=0 → 1.0, SOFA=12 → 0.5, SOFA=24 → 0.0。
        """
        sofa = state.sofa_score()
        return max(0.0, 1.0 - sofa / 24.0)

    @staticmethod
    def _safety_score(state: PatientState) -> float:
        """安全约束评分。

        is_safe() 为 True → 1.0。
        每项违规扣 1/N_VITALS 分。
        """
        if state.is_safe():
            return 1.0
        violations = state.safety_violations()
        return max(0.0, 1.0 - len(violations) / N_VITALS)
