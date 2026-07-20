"""MCI World Model — 临床因果发现（ClinicalCausalDiscovery）

============================================================

D4 模块：从患者体征时序数据发现体征间的因果结构。

为什么需要？
    决策引擎原有 MedicalCausalSDK 是基于规则/证据的因果诊断（定性），
    无法从数据中发现"哪些体征因果驱动哪些体征"。
    本模块用 PC 算法的偏相关条件独立性检验，从患者多时间窗数据
    学习体征间因果结构，为决策引擎提供数据驱动的因果归因。

算法（PC 算法简化版）:
    1. 构建完全无向图（所有体征两两相连）
    2. 对每条边 X-Y，检验给定 conditioning set S 时 X⊥Y|S
       （高斯假设下用偏相关系数 = 0 检验）
    3. 若条件独立则删边，得到骨架（无向图）
    4. 用时间滞后方向（X_t→Y_{t+1} 比 Y_t→X_{t+1} 强则定向）确定因果方向

设计原则（AGENTS.md 边界）:
    - 输入是本次推理的患者状态数据（非持久化记忆）
    - 不依赖 su-memory-sdk / lite_pro
    - 所有随机源设 seed
    - 数值健壮：相关系数防 NaN，小样本降级
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mci_world_model.sdk._clinical_world_state import N_VITALS, VITAL_NAMES


@dataclass
class CausalLink:
    """发现的因果边。

    Attributes:
        cause: 因果源体征名。
        effect: 因果目标体征名。
        strength: 因果效应强度 |偏相关系数| ∈ [0, 1]。
        direction: 方向置信度 ∈ [-1, 1]，正=cause→effect，负=反向。
        p_value: 条件独立性检验 p 值（越小越显著）。
    """

    cause: str
    effect: str
    strength: float
    direction: float
    p_value: float = 1.0


@dataclass
class CausalStructure:
    """发现的因果结构。

    Attributes:
        links: 因果边列表。
        adjacency: 邻接矩阵（有向加权），shape (N_VITALS, N_VITALS)。
        n_samples: 学习用样本数。
        method: 发现方法名。
    """

    links: list[CausalLink] = field(default_factory=list)
    adjacency: np.ndarray | None = None
    n_samples: int = 0
    method: str = "pc_partial_corr"

    def to_dict(self) -> dict[str, Any]:
        """序列化（审计用）。"""
        return {
            "links": [
                {
                    "cause": link.cause,
                    "effect": link.effect,
                    "strength": round(link.strength, 4),
                    "direction": round(link.direction, 4),
                    "p_value": round(link.p_value, 4),
                }
                for link in self.links
            ],
            "n_samples": self.n_samples,
            "method": self.method,
            "n_strong_links": sum(1 for link in self.links if link.strength > 0.3),
        }


class ClinicalCausalDiscovery:
    """临床因果发现器 — 从体征时序数据学习因果结构。

    Example:
        >>> discovery = ClinicalCausalDiscovery(significance=0.05)
        >>> # vitals_history: 多时间窗体征矩阵 (T, N_VITALS)
        >>> structure = discovery.discover(vitals_history)
        >>> print(structure.to_dict())
    """

    def __init__(self, significance: float = 0.05, min_samples: int = 10) -> None:
        """初始化。

        Args:
            significance: 条件独立性检验显著性水平（默认 0.05）。
            min_samples: 最少样本数（不足则返回空结构，避免过拟合）。
        """
        self._alpha = significance
        self._min_samples = min_samples

    def discover(
        self,
        vitals_history: np.ndarray,
        max_conditioning_size: int = 1,
    ) -> CausalStructure:
        """从体征时序矩阵发现因果结构。

        Args:
            vitals_history: 体征矩阵 shape (T, N_VITALS)，T 是时间窗数。
            max_conditioning_size: PC 算法 conditioning set 最大规模
               （默认 1，即一阶偏相关；增大则更精确但需更多数据）。

        Returns:
            CausalStructure。
        """
        data = np.asarray(vitals_history, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"vitals_history 需 2D (T, N_VITALS)，收到 {data.shape}")
        T = data.shape[0]
        n_vars = min(data.shape[1], N_VITALS)

        if self._min_samples > T:
            return CausalStructure(n_samples=T, method="insufficient_data")

        # 截取到 N_VITALS 列（忽略检验列）
        data = data[:, :n_vars]

        # Step 1: 计算相关系数矩阵
        corr = self._safe_corr(data)
        if corr is None:
            return CausalStructure(n_samples=T, method="degenerate_data")

        # Step 2: PC 骨架学习（条件独立性检验删边）
        # adj[i][j]=1 表示有边（待定向）
        skeleton = np.ones((n_vars, n_vars), dtype=bool)
        np.fill_diagonal(skeleton, False)

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if not skeleton[i, j]:
                    continue
                # 检验 i⊥j | S（S 遍历其他变量子集）
                independent, p_val = self._test_conditional_independence(corr, i, j, n_vars, max_conditioning_size, T)
                if independent:
                    skeleton[i, j] = False
                    skeleton[j, i] = False

        # Step 3: 用时间滞后方向定向（Granger 思想）
        # 若 corr(X_t, Y_{t+1}) > corr(Y_t, X_{t+1}) 则 X→Y
        adjacency = np.zeros((n_vars, n_vars), dtype=np.float64)
        links: list[CausalLink] = []

        for i in range(n_vars):
            for j in range(n_vars):
                if i == j or not skeleton[i, j]:
                    continue
                # 偏相关强度
                strength = abs(self._partial_corr(corr, i, j, set()))
                if strength < 1e-3:
                    continue
                # 时间滞后方向
                direction = self._lag_direction(data, i, j)
                # 只保留方向为正的边（避免双向重复）
                if direction >= 0:
                    adjacency[i, j] = strength
                    p_val = self._corr_p_value(strength, T)
                    links.append(
                        CausalLink(
                            cause=VITAL_NAMES[i],
                            effect=VITAL_NAMES[j],
                            strength=round(float(strength), 4),
                            direction=round(float(direction), 4),
                            p_value=round(float(p_val), 4),
                        )
                    )

        # 按 strength 降序
        links.sort(key=lambda link: link.strength, reverse=True)
        return CausalStructure(
            links=links,
            adjacency=adjacency,
            n_samples=T,
            method="pc_partial_corr",
        )

    @staticmethod
    def _safe_corr(data: np.ndarray) -> np.ndarray | None:
        """计算相关系数矩阵（防 NaN/常数列）。"""
        try:
            # 移除常数列（相关系数未定义）
            std = np.std(data, axis=0)
            valid = std > 1e-10
            if valid.sum() < 2:
                return None
            data_valid = data[:, valid]
            corr_valid = np.corrcoef(data_valid.T)
            if corr_valid.ndim == 0:
                return None
            # 还原到完整矩阵
            n = data.shape[1]
            full = np.zeros((n, n))
            vi = 0
            for i in range(n):
                vj = 0
                for j in range(n):
                    if valid[i] and valid[j]:
                        full[i, j] = corr_valid[vi, vj]
                    elif i == j:
                        full[i, j] = 1.0
                    if valid[j]:
                        vj += 1
                if valid[i]:
                    vi += 1
            # NaN 填 0（无条件独立）
            full = np.nan_to_num(full, nan=0.0)
            return full
        except (ValueError, np.linalg.LinAlgError):
            return None

    @staticmethod
    def _partial_corr(corr: np.ndarray, i: int, j: int, cond_set: set[int]) -> float:
        r"""计算偏相关系数 ρ_{ij|S}。

        用递归公式: ρ_{ij|S} = (ρ_{ij|S\k} - ρ_{ik|S\k}·ρ_{jk|S\k}) /
                                sqrt((1-ρ_{ik|S\k}²)(1-ρ_{jk|S\k}²))
        """
        if not cond_set:
            val = corr[i, j]
            return float(val) if np.isfinite(val) else 0.0
        # 取一个 conditioning 变量
        k_set = set(cond_set)
        k = k_set.pop()
        rest = k_set
        rij = ClinicalCausalDiscovery._partial_corr(corr, i, j, rest)
        rik = ClinicalCausalDiscovery._partial_corr(corr, i, k, rest)
        rjk = ClinicalCausalDiscovery._partial_corr(corr, j, k, rest)
        denom = np.sqrt(max((1 - rik**2) * (1 - rjk**2), 1e-12))
        if denom < 1e-12:
            return 0.0
        return float((rij - rik * rjk) / denom)

    def _test_conditional_independence(
        self,
        corr: np.ndarray,
        i: int,
        j: int,
        n_vars: int,
        max_cond_size: int,
        n_samples: int,
    ) -> tuple[bool, float]:
        """检验 X_i ⊥ X_j | S 对所有 |S| ≤ max_cond_size。

        Returns:
            (是否条件独立, 最小 p 值)。
        """
        from itertools import combinations

        others = [k for k in range(n_vars) if k not in (i, j)]
        min_p = 1.0
        for size in range(0, max_cond_size + 1):
            for cond in combinations(others, size):
                cond_set = set(cond)
                pcorr = self._partial_corr(corr, i, j, cond_set)
                p_val = self._corr_p_value(abs(pcorr), n_samples)
                min_p = min(min_p, p_val)
                if p_val > self._alpha:
                    # 不能拒绝独立性（p 大）→ 存在条件独立 → 删边
                    return True, p_val
        # 所有 conditioning set 下都显著相关（min_p 小）→ 不独立 → 保留边
        return False, min_p

    @staticmethod
    def _corr_p_value(r: float, n: int) -> float:
        """相关系数的 p 值（Fisher z 变换近似）。

        Args:
            r: 相关系数绝对值。
            n: 样本数。

        Returns:
            双侧 p 值。
        """
        if n <= 3 or r >= 0.9999:
            return 0.0 if r >= 0.9999 else 1.0
        # Fisher z
        r_clip = min(max(r, 0.0), 0.9999)
        z = 0.5 * np.log((1 + r_clip) / max(1 - r_clip, 1e-12))
        se = 1.0 / np.sqrt(max(n - 3, 1))
        # 标准正态 |z/se| 的双侧 p
        from math import erf, sqrt

        z_stat = abs(z) / se
        # p = 2*(1 - Φ(z_stat))
        p = 2.0 * (1.0 - 0.5 * (1.0 + erf(z_stat / sqrt(2.0))))
        return float(max(min(p, 1.0), 0.0))

    @staticmethod
    def _lag_direction(data: np.ndarray, i: int, j: int) -> float:
        """时间滞后方向：比较 corr(X_t, Y_{t+1}) vs corr(Y_t, X_{t+1})。

        Returns:
            方向分 ∈ [-1, 1]。正=i→j（i 驱动 j），负=j→i。
        """
        if data.shape[0] < 3:
            return 0.0
        x_t = data[:-1, i]
        y_t = data[:-1, j]
        x_next = data[1:, i]
        y_next = data[1:, j]
        # corr(X_t, Y_{t+1}): i 驱动 j 的证据
        c_xy = ClinicalCausalDiscovery._safe_2corr(x_t, y_next)
        # corr(Y_t, X_{t+1}): j 驱动 i 的证据
        c_yx = ClinicalCausalDiscovery._safe_2corr(y_t, x_next)
        if c_xy is None or c_yx is None:
            return 0.0
        # 归一化到 [-1, 1]
        diff = c_xy - c_yx
        denom = abs(c_xy) + abs(c_yx) + 1e-12
        return float(diff / denom)

    @staticmethod
    def _safe_2corr(x: np.ndarray, y: np.ndarray) -> float | None:
        """安全计算两个向量的相关系数。"""
        sx, sy = np.std(x), np.std(y)
        if sx < 1e-10 or sy < 1e-10:
            return None
        c = float(np.corrcoef(x, y)[0, 1])
        return c if np.isfinite(c) else None
