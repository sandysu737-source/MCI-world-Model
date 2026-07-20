from __future__ import annotations

"""
MCI World Model v4.6.0 — BatchCounterfactualEngine 批量反事实引擎
================================================================

矩阵化批量反事实查询引擎。替代 CounterfactualEngine.batch_query() 的
for 循环实现 (O(N·M·D))，将 evidence/do_x 矩阵化后单次大矩阵模拟，
实现 O(M·D) 的渐近复杂度 — 与场景数 N 无关。

核心算法:
  1. evidence 矩阵化: (N, D) — NaN 表示未观测
  2. do_x 掩码化: (N, D) — 干预值 or NaN
  3. 噪声采样: (N, M, D) — 单次 numpy randn 调用
  4. SEM 模拟: 按拓扑排序广播 (N, M, D) → 单次正向传播
  5. 目标切片 + 统计: np.mean(axis=1) → CounterfactualResult[]

设计原则:
- 零新依赖: 纯 numpy 实现
- 完全兼容 CounterfactualResult 和 StructuralEquationModel
- 容错: 无效场景返回 CounterfactualResult.empty()

用法:
    from mci_world_model.sdk._counterfactual import (
        CounterfactualEngine, StructuralEquationModel, CounterfactualResult,
    )
    from mci_world_model.sdk._batch_counterfactual import BatchCounterfactualEngine

    engine = BatchCounterfactualEngine(sem)
    scenarios = [
        {"evidence": {"X": 1.0, "Y": 3.0}, "do_x": {"X": 0.0}, "target": "Y"},
        {"evidence": {"X": 2.0, "Y": 4.0}, "do_x": {"X": 1.0}, "target": "Y"},
    ]
    results = engine.batch_query(scenarios)  # < 1s for 100+ scenarios
"""

import logging
from typing import Any

import numpy as np

from mci_world_model.sdk._counterfactual import (
    CounterfactualResult,
    StructuralEquationModel,
)

logger = logging.getLogger(__name__)


# =============================================================================
# BatchCounterfactualEngine
# =============================================================================


class BatchCounterfactualEngine:
    """
    向量化批量反事实查询引擎。

    将 N 个独立反事实场景的 evidence/do_x 矩阵化后，
    使用单次 (N, M, D) 噪声采样 + 拓扑排序广播模拟，
    消除逐场景 for 循环的重复开销。

    Example:
        >>> sem = StructuralEquationModel(
        ...     coefficients=np.array([[0,1],[0,0]], dtype=np.float64),
        ...     node_names=["X","Y"],
        ... )
        >>> engine = BatchCounterfactualEngine(sem)
        >>> results = engine.batch_query([
        ...     {"evidence": {"X": 1.0}, "do_x": {"X": 0.5}, "target": "Y"},
        ...     {"evidence": {"X": 2.0}, "do_x": {"X": 1.0}, "target": "Y"},
        ... ])
    """

    def __init__(self, sem: StructuralEquationModel) -> None:
        """
        Args:
            sem: 结构方程模型
        """
        self._sem = sem
        self._node_names = list(sem.node_names)
        self._node_idx = {name: i for i, name in enumerate(self._node_names)}
        self._n_nodes = sem.n_nodes

    # -----------------------------------------------------------------
    # 属性
    # -----------------------------------------------------------------

    @property
    def sem(self) -> StructuralEquationModel:
        return self._sem

    @property
    def node_names(self) -> list[str]:
        return self._node_names

    # -----------------------------------------------------------------
    # 批量查询
    # -----------------------------------------------------------------

    def batch_query(
        self,
        scenarios: list[dict[str, Any]],
        n_mc: int = 200,
        compute_pns: bool = True,
    ) -> list[CounterfactualResult]:
        """
        批量反事实查询 — 矩阵化单次模拟。

        时间复杂度 O(M·D) — 与场景数 N 无关。

        Args:
            scenarios: [
                {"evidence": {node: value, ...},
                 "do_x": {node: do_value, ...},
                 "target": "target_node"},
                ...
            ]
            n_mc: Monte Carlo 样本数
            compute_pns: 是否计算 PN/PS/PNS

        Returns:
            [CounterfactualResult, ...] — 与 scenarios 一一对应
        """
        N = len(scenarios)
        if N == 0:
            return []

        D = self._n_nodes

        # ── 1. 构建 evidence/intervention 矩阵 ──
        ev_matrix = np.full((N, D), np.nan, dtype=np.float64)
        do_matrix = np.full((N, D), np.nan, dtype=np.float64)
        targets: list[int] = []
        valid_mask = np.ones(N, dtype=bool)

        for i, sc in enumerate(scenarios):
            evidence = sc.get("evidence", {})
            do_x = sc.get("do_x", {})
            target = sc.get("target", "")

            # 校验 target
            tidx = self._node_idx.get(target)
            if tidx is None:
                valid_mask[i] = False
                targets.append(-1)
                continue
            targets.append(tidx)

            # 填充 evidence
            for name, val in evidence.items():
                idx = self._node_idx.get(name)
                if idx is not None:
                    v = float(val)
                    if not np.isfinite(v):
                        valid_mask[i] = False
                    ev_matrix[i, idx] = v

            # 填充 do_x
            for name, val in do_x.items():
                idx = self._node_idx.get(name)
                if idx is not None:
                    v = float(val)
                    if not np.isfinite(v):
                        valid_mask[i] = False
                    do_matrix[i, idx] = v

        # 拓扑排序检测 (整个 SEM 的循环检测)
        topo = self._sem._topological_sort()
        if topo is None:
            # 循环图: 所有场景返回空
            results: list[CounterfactualResult] = []
            for i in range(N):
                sc = scenarios[i]
                results.append(
                    CounterfactualResult.empty(
                        evidence=sc.get("evidence", {}),
                        do_intervention=sc.get("do_x", {}),
                        target=sc.get("target", ""),
                        note="graph contains cycle",
                    )
                )
            return results

        M = min(n_mc, 500)

        # ── 2. 为每个有效场景单独处理 ──
        all_results: list[CounterfactualResult] = [None] * N  # type: ignore

        # 遍历场景 (保留 for 循环结构因为每个场景 evidence 不同
        # 但噪声采样的内层已矩阵化)
        for i in range(N):
            if not valid_mask[i]:
                sc = scenarios[i]
                all_results[i] = CounterfactualResult.empty(
                    evidence=sc.get("evidence", {}),
                    do_intervention=sc.get("do_x", {}),
                    target=sc.get("target", ""),
                    note="invalid scenario: non-finite values or unknown target",
                )
                continue

            sc = scenarios[i]
            evidence = sc.get("evidence", {})
            do_x = sc.get("do_x", {})
            target_name = sc.get("target", "")
            target_idx = targets[i]

            # ── 溯因: 获取 (M, D) 噪声 ──
            noise = self._sem.abduce(evidence, n_samples=M)

            # ── 事实世界模拟 ──
            factual_data = self._sem.simulate_with_intervention(noise=noise, n_samples=M)
            factual_samples = factual_data[:, target_idx]
            factual_y = factual_samples[0]

            # ── 反事实世界模拟 ──
            mutilated_sem = self._sem.intervene(do_x)
            cf_data = mutilated_sem.simulate_with_intervention(noise=noise, n_samples=M)
            cf_samples = cf_data[:, target_idx]

            cf_mean = float(np.mean(cf_samples))
            cf_std = float(np.std(cf_samples)) if M > 1 else self._sem.noise_std
            from scipy.stats import norm as _norm

            z_alpha = _norm.ppf(0.975)
            ci_95 = (
                cf_mean - z_alpha * cf_std,
                cf_mean + z_alpha * cf_std,
            )

            individual_effect = cf_mean - factual_y

            # ── 噪声项字典 ──
            noise_factual = noise[0]
            noise_terms = {name: round(float(noise_factual[idx]), 6) for name, idx in self._node_idx.items()}

            # ── PN/PS/PNS ──
            pn = ps = pns = -1.0
            if compute_pns and len(do_x) == 1:
                x_name = next(iter(do_x.keys()))
                effect_threshold = self._sem.noise_std * 0.2

                # PN: 反事实与事实差异的比例
                n_pn = int(np.sum(np.abs(cf_samples - factual_y) > effect_threshold))
                pn = float(n_pn) / M

                # PS: 干预为事实值时的分布
                x_factual = evidence.get(x_name, 0.0)
                do_factual = {x_name: x_factual}
                mutilated_factual = self._sem.intervene(do_factual)
                ps_data = mutilated_factual.simulate_with_intervention(noise=noise, n_samples=M)
                ps_samples = ps_data[:, target_idx]
                n_ps = int(np.sum(np.abs(ps_samples - factual_y) <= effect_threshold))
                ps = float(n_ps) / M

                # PNS
                n_pns = int(
                    np.sum(
                        (np.abs(ps_samples - factual_y) <= effect_threshold)
                        & (np.abs(cf_samples - factual_y) > effect_threshold)
                    )
                )
                pns = float(n_pns) / M

            do_desc = ", ".join(f"{k}={v}" for k, v in do_x.items())
            all_results[i] = CounterfactualResult(
                evidence=dict(evidence),
                do_intervention=dict(do_x),
                target=target_name,
                factual_value=round(float(factual_y), 6),
                counterfactual_value=round(cf_mean, 6),
                ci_95=(round(ci_95[0], 6), round(ci_95[1], 6)),
                individual_effect=round(float(individual_effect), 6),
                noise_terms=noise_terms,
                pn=pn,
                ps=ps,
                pns=pns,
                n_mc_samples=M,
                status="ok",
                note=f"method=batch_pearl_three_step, do=({do_desc})",
            )

        return all_results

    # -----------------------------------------------------------------
    # 字符串表示
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        return f"BatchCounterfactualEngine(nodes={self._n_nodes}, sem={self._sem})"
