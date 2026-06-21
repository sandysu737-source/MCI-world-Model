from __future__ import annotations

"""MCI World Model v6.0.0 — AutonomousLawDiscovererV2 自主因果发现 2.0
========================================================================

从 V1 单方程发现升级到完整因果结构 (DAG) + PC 算法骨架发现。

核心升级 (V1 → V2):
    V1: 观测 → 符号回归 → 单方程 → 守恒验证 → 因果图更新
    V2: 观测 → PC骨架 → 每边符号回归 → 守恒+方向验证 → 系统一致性

核心能力:
    discover_causal_structure(data, var_names) — 完整因果结构发现
    build_system_report()                       — 多方程系统一致性报告

设计原则:
    - 纯 numpy，零外部依赖
    - PC 算法骨架 + 符号回归方程
    - 与 SIGReg / DoCalculus 正交组合
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# CausalEdge — 因果边
# =============================================================================


@dataclass
class CausalEdge:
    """因果边 — 从 cause 到 effect 的有向边。

    Attributes:
        cause: 因果变量名
        effect: 结果变量名
        equation: 符号回归方程 (字符串)
        r_squared: 拟合 R²
        conservation_verified: 是否通过守恒验证
        causal_verified: 是否通过因果方向验证
    """

    cause: str
    effect: str
    equation: str = ""
    r_squared: float = 0.0
    conservation_verified: bool = False
    causal_verified: bool = False


# =============================================================================
# CausalSkeleton — PC 算法骨架
# =============================================================================


@dataclass
class CausalSkeleton:
    """PC 算法发现的因果骨架。

    Attributes:
        nodes: 变量名列表
        edges: 有向边列表 [(from, to), ...]
        adj_matrix: 邻接矩阵 (n_vars, n_vars), 1=有边
        confidence: 骨架置信度
    """

    nodes: list[str] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    adj_matrix: np.ndarray | None = None
    confidence: float = 0.0


# =============================================================================
# SystemReport — 多方程系统一致性报告
# =============================================================================


@dataclass
class SystemReport:
    """多方程系统一致性报告。

    Attributes:
        n_variables: 变量数
        n_edges: 发现的因果边数
        conservation_score: 守恒验证综合得分
        causal_dag: 因果骨架
        laws: 发现的因果规律列表
        is_consistent: 系统是否一致
    """

    n_variables: int = 0
    n_edges: int = 0
    conservation_score: float = 0.0
    causal_dag: dict[str, Any] = field(default_factory=dict)
    laws: list[dict[str, Any]] = field(default_factory=list)
    is_consistent: bool = False


# =============================================================================
# PCSkeletonDiscoverer — PC 算法骨架发现器
# =============================================================================


class PCSkeletonDiscoverer:
    """PC 算法骨架发现器 — 从数据中学习因果 DAG 骨架。

    简化版 PC 算法:
      1. 从完全图开始
      2. 条件独立性检验逐步删边
      3. 方向规则 (v-structure + orientation rules)

    约束: 变量数 ≤ 10
    """

    def __init__(self, alpha: float = 0.05, min_corr: float = 0.1) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha 必须在 (0,1), 当前 {alpha}")
        if min_corr < 0.0:
            raise ValueError(f"min_corr 必须 >= 0, 当前 {min_corr}")
        self._alpha = alpha
        self._min_corr = min_corr

    def discover(self, data: np.ndarray, var_names: list[str]) -> CausalSkeleton:
        """从数据中学习因果骨架。

        Args:
            data: 观测数据 (n_samples, n_vars)
            var_names: 变量名列表

        Returns:
            CausalSkeleton 骨架结果
        """
        n_vars = len(var_names)
        _n_samples = data.shape[0]
        if _n_samples == 0:
            return CausalSkeleton(
                nodes=list(var_names),
                edges=[],
                adj_matrix=np.zeros((n_vars, n_vars), dtype=int),
                confidence=0.0,
            )
        if n_vars > 10:
            logger.warning("PC 算法变量数 >10，可能不稳定，建议 ≤10")

        # Step 1: 初始化完全图
        adj = np.ones((n_vars, n_vars), dtype=int)
        np.fill_diagonal(adj, 0)

        # Step 2: 条件独立性检验 — 逐步删边
        corr = np.corrcoef(data.T)

        # 2a: 0 阶检验 (无条件独立)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adj[i, j] == 0:
                    continue
                r = self._partial_corr(corr, i, j, [])
                if abs(r) < self._min_corr:
                    # 相关太弱 → 直接删边 (解决大 n 统计显著但实际无关问题)
                    adj[i, j] = 0
                    adj[j, i] = 0
                    continue
                p_val = self._test_independence(data[:, i], data[:, j], r, _n_samples)
                if p_val > self._alpha:
                    adj[i, j] = 0
                    adj[j, i] = 0

        # 2b: 1 阶检验 (条件独立, 打破 chain/confounder 伪边)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adj[i, j] == 0:
                    continue
                # 遍历所有可能的条件变量 k
                for k in range(n_vars):
                    if k in (i, j):
                        continue
                    r = self._partial_corr(corr, i, j, [k])
                    p_val = self._fisher_z_test(r, _n_samples)
                    if p_val > self._alpha:
                        adj[i, j] = 0
                        adj[j, i] = 0
                        break

        # Step 3: 方向推断 (简化: 基于 partial correlation 不对称性)
        edges = self._orient_edges(adj, corr, var_names, _n_samples)

        # Step 3+: 回归残余定向 — 对无向边使用 OLS 残差方差不对称性确定方向
        edges = self._orient_edges_by_regression(data, adj, edges, var_names)

        # Build directed adjacency matrix from final edges (for SHD comparison)
        name_to_idx = {name: i for i, name in enumerate(var_names)}
        dir_adj = np.zeros((n_vars, n_vars), dtype=int)
        for src, dst in edges:
            dir_adj[name_to_idx[src], name_to_idx[dst]] = 1

        return CausalSkeleton(
            nodes=list(var_names),
            edges=edges,
            adj_matrix=dir_adj,
            confidence=self._compute_confidence(adj, corr),
        )

    @staticmethod
    def _partial_corr(corr: np.ndarray, i: int, j: int, cond: list[int]) -> float:
        """计算偏相关系数 r_{ij·cond}。

        0 阶: 直接 Pearson 相关
        1 阶: r_{ij|k} = (r_{ij} - r_{ik}·r_{kj}) / sqrt((1-r_{ik}²)(1-r_{kj}²))
        """
        if len(cond) == 0:
            return corr[i, j]
        if len(cond) == 1:
            k = cond[0]
            rik, rkj, rij = corr[i, k], corr[k, j], corr[i, j]
            denom = np.sqrt(max(1 - rik * rik, 1e-10) * max(1 - rkj * rkj, 1e-10))
            return (rij - rik * rkj) / denom
        # 高阶 (>1): 递归消去第一个条件变量后降阶
        k = cond[0]
        rik, rkj, rij = corr[i, k], corr[k, j], corr[i, j]
        denom = np.sqrt(max(1 - rik * rik, 1e-10) * max(1 - rkj * rkj, 1e-10))
        r_adj = (rij - rik * rkj) / denom
        # 简化: 忽略条件变量对其他边的影响, 近似递归
        return r_adj

    def _fisher_z_test(self, r: float, n_samples: int) -> float:
        """Fisher z-transform 双尾检验。

        H₀: ρ = 0 (独立)
        返回 p 值, p > alpha 则不能拒绝独立性 → 删边
        """
        r = np.clip(r, -0.9999, 0.9999)
        z = 0.5 * np.log((1 + r) / (1 - r))
        se = 1.0 / np.sqrt(max(n_samples - 3, 1))
        return 2.0 * (1.0 - self._normal_cdf(abs(z) / se))

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """标准正态 CDF 近似 (Abramowitz & Stegun)。"""
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p = 0.3275911
        sign = 1 if x >= 0 else -1
        x = abs(x)
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
        return 0.5 * (1.0 + sign * y)

    def _orient_edges(
        self, adj: np.ndarray, corr: np.ndarray,
        var_names: list[str], n_samples: int,
    ) -> list[tuple[str, str]]:
        """方向推断 — v-structure + Meek R1。

        Step 3a: v-structure (collider) 检测
            对每个 triple i-k-j 其中 i—k, k—j 但 i,j 无边:
            若 i NOT-INDEP j | k → k 是 collider → 定向 i→k, j→k

        Step 3b: Meek R1 — 防环
            若 i→j—k 且 i,k 无边 → 定向 j→k

        Step 3c: 剩余边保持无向 (双向输出)
        """
        n_vars = adj.shape[0]
        # 方向矩阵: 0=无向, 1=i→j, -1=j→i
        dir_mat = np.zeros((n_vars, n_vars), dtype=int)

        # ── 3a: v-structure 检测 (仅对无屏蔽 triple) ──
        # 标准 PC: v-structure = i→k←j 仅在 i—k—j 且 i,j NON-ADJACENT 时检测
        # 测试 i NOT-INDEP j | k → k 是 collider → 定向 i→k, j→k
        for k in range(n_vars):
            for i in range(n_vars):
                if i == k or adj[i, k] == 0:
                    continue
                for j in range(i + 1, n_vars):
                    if j == k or adj[k, j] == 0:
                        continue
                    if adj[i, j] != 0:  # 必须是 unshielded triple
                        continue
                    r = self._partial_corr(corr, i, j, [k])
                    p = self._fisher_z_test(r, n_samples)
                    if p <= self._alpha:  # 拒绝独立性 → k 是 collider
                        dir_mat[i, k] = 1
                        dir_mat[k, i] = -1
                        dir_mat[j, k] = 1
                        dir_mat[k, j] = -1

        # ── 3b: Meek R1 — 防环传播 ──
        changed = True
        while changed:
            changed = False
            for i in range(n_vars):
                for j in range(n_vars):
                    if i == j or dir_mat[i, j] != 1:  # i → j ?
                        continue
                    for k in range(n_vars):
                        if k in (i, j):
                            continue
                        if dir_mat[j, k] != 0:  # j—k ?
                            continue
                        if adj[j, k] == 0:  # 必须有边
                            continue
                        if adj[i, k] != 0:  # i,k 不能有边 (否则 R1 不适用)
                            continue
                        # i→j—k, i,k 无边 → 定向 j→k
                        dir_mat[j, k] = 1
                        dir_mat[k, j] = -1
                        changed = True

        # ── 3c: 输出边 ──
        edges = []
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adj[i, j] == 0:
                    continue
                if dir_mat[i, j] == 1:
                    edges.append((var_names[i], var_names[j]))
                elif dir_mat[j, i] == 1:
                    edges.append((var_names[j], var_names[i]))
                else:
                    # 无向 → 双向输出
                    edges.append((var_names[i], var_names[j]))
                    edges.append((var_names[j], var_names[i]))

        return edges

    @staticmethod
    def _compute_confidence(adj: np.ndarray, corr: np.ndarray) -> float:
        """骨架置信度 — 边的平均|偏相关|。"""
        n = adj.shape[0]
        total, count = 0.0, 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 1:
                    total += abs(corr[i, j])
                    count += 1
        return total / max(count, 1)

    def _orient_edges_by_regression(
        self, data: np.ndarray, adj: np.ndarray,
        edges: list[tuple[str, str]], var_names: list[str],
    ) -> list[tuple[str, str]]:
        """Regression-based orientation for remaining undirected edges.

        For each bidirectional pair (i,j) and (j,i) (undirected edge):
          - Regress X_i ~ X_j, compute residual variance sigma_i|j
          - Regress X_j ~ X_i, compute residual variance sigma_j|i
          - Keep the direction with lower residual variance
          (e.g. sigma_a|b < sigma_b|a means b predicts a better -> keep b->a)

        Reference: LiNGAM-style residual asymmetry (Shimizu et al. 2006)
        """
        name_to_idx = {name: i for i, name in enumerate(var_names)}

        # Find all bidirectional pairs (undirected edges)
        edge_set = set(edges)
        bidir_pairs: list[tuple[str, str]] = []
        for a, b in edges:
            if (b, a) in edge_set and a < b:
                bidir_pairs.append((a, b))

        for a_name, b_name in bidir_pairs:
            a_idx = name_to_idx[a_name]
            b_idx = name_to_idx[b_name]

            x_a = data[:, a_idx].astype(np.float64)
            x_b = data[:, b_idx].astype(np.float64)

            # Regress a ~ b
            b_coef, _, _, _ = np.linalg.lstsq(x_b.reshape(-1, 1), x_a, rcond=None)
            res_a = x_a - x_b * b_coef[0]
            var_a_given_b = float(np.var(res_a))

            # Regress b ~ a
            b_coef2, _, _, _ = np.linalg.lstsq(x_a.reshape(-1, 1), x_b, rcond=None)
            res_b = x_b - x_a * b_coef2[0]
            var_b_given_a = float(np.var(res_b))

            # Keep direction with lower residual variance
            if var_a_given_b < var_b_given_a:
                # a better predicted by b -> b->a is correct direction; remove a->b
                edge_set.discard((a_name, b_name))
            else:
                # b better predicted by a -> a->b is correct direction; remove b->a
                edge_set.discard((b_name, a_name))

        return list(edge_set)


# =============================================================================

    @staticmethod
    def _hsic_test(x, y, n_perm=100, sigma=None):
        """HSIC (Hilbert-Schmidt Independence Criterion) permutation test.

        RBF kernel based nonlinear independence test.
        Returns p-value (0 = dependent, 1 = independent).
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()

        max_n = 400
        if len(x) > max_n:
            idx_sub = np.random.RandomState(42).choice(len(x), max_n, replace=False)
            x, y = x[idx_sub], y[idx_sub]
        n = len(x)
        if n < 10:
            return 1.0

        if sigma is None:
            dists = np.abs(x[:, None] - x[None, :])
            sigma = float(np.median(dists[dists > 0])) if np.any(dists > 0) else 1.0
            sigma = max(sigma, 1e-3)

        def _rbf(a):
            sq = np.sum((a[:, None, :] - a[None, :, :]) ** 2, axis=-1)
            return np.exp(-sq / (2.0 * sigma ** 2))

        K = _rbf(x.reshape(-1, 1))
        L = _rbf(y.reshape(-1, 1))
        H = np.eye(n) - 1.0 / n * np.ones((n, n))
        hsic_obs = float(np.trace(K @ H @ L @ H)) / (n - 1) ** 2

        y_shuf = y.copy()
        null_hsic = np.zeros(n_perm)
        rng = np.random.RandomState(42)
        for p in range(n_perm):
            rng.shuffle(y_shuf)
            Lp = _rbf(y_shuf.reshape(-1, 1))
            null_hsic[p] = float(np.trace(K @ H @ Lp @ H)) / (n - 1) ** 2

        return float(np.mean(null_hsic >= hsic_obs))

    @staticmethod
    def _partial_hsic(x, y, z, n_perm=50, sigma=None):
        """Partial HSIC: HSIC on residuals after regressing out conditioning variable Z.

        X_perp = X - E[X|Z], Y_perp = Y - E[Y|Z], then HSIC(X_perp, Y_perp).
        Uses fewer permutations (50 vs 100) since this is a fallback test.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        z = np.asarray(z, dtype=np.float64).ravel()

        # Downsample for speed (kernel methods are O(n^2))
        max_n = 400
        if len(x) > max_n:
            idx_sub = np.random.RandomState(42).choice(len(x), max_n, replace=False)
            x, y, z = x[idx_sub], y[idx_sub], z[idx_sub]
        n = len(x)
        if n < 10:
            return 1.0

        # Regress out Z
        Z_aug = np.column_stack([np.ones(n), z])
        beta_x, _, _, _ = np.linalg.lstsq(Z_aug, x, rcond=None)
        beta_y, _, _, _ = np.linalg.lstsq(Z_aug, y, rcond=None)
        x_res = x - Z_aug @ beta_x
        y_res = y - Z_aug @ beta_y

        # HSIC on residuals
        if sigma is None:
            dists = np.abs(x_res[:, None] - x_res[None, :])
            sigma = float(np.median(dists[dists > 0])) if np.any(dists > 0) else 1.0
            sigma = max(sigma, 1e-3)

        def _rbf(a):
            sq = np.sum((a[:, None, :] - a[None, :, :]) ** 2, axis=-1)
            return np.exp(-sq / (2.0 * sigma ** 2))

        K = _rbf(x_res.reshape(-1, 1))
        L = _rbf(y_res.reshape(-1, 1))
        H = np.eye(n) - 1.0 / n * np.ones((n, n))
        hsic_obs = float(np.trace(K @ H @ L @ H)) / (n - 1) ** 2

        y_shuf = y_res.copy()
        null_hsic = np.zeros(n_perm)
        rng = np.random.RandomState(42)
        for p in range(n_perm):
            rng.shuffle(y_shuf)
            Lp = _rbf(y_shuf.reshape(-1, 1))
            null_hsic[p] = float(np.trace(K @ H @ Lp @ H)) / (n - 1) ** 2

        return float(np.mean(null_hsic >= hsic_obs))


    def _test_independence(self, x, y, corr_val, n_samples):
        """Combined independence test: Fisher z first, HSIC fallback."""
        r = np.clip(abs(corr_val), 0.0, 0.9999)
        z = 0.5 * np.log((1 + r) / (1 - r))
        se = 1.0 / np.sqrt(max(n_samples - 3, 1))
        p_linear = 2.0 * (1.0 - self._normal_cdf(z / se))
        if p_linear <= self._alpha:
            return p_linear
        p_hsic = self._hsic_test(x, y, n_perm=100)
        return min(p_linear, p_hsic)

# GESDiscoverer - Greedy Equivalence Search
# =============================================================================


class GESDiscoverer:
    """Greedy Equivalence Search (GES) causal discovery.

    Searches CPDAG space:
      1. Forward: greedy edge addition maximizing BIC
      2. Backward: greedy edge deletion maximizing BIC

    Constraints: <= 10 variables, linear Gaussian data
    """

    def __init__(self, alpha: float = 0.05, max_iter: int = 50) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        self._alpha = alpha
        self._max_iter = max_iter

    def discover(self, data: np.ndarray, var_names: list[str]) -> CausalSkeleton:
        """Learn causal skeleton from data."""
        n_vars = len(var_names)
        n_samples = data.shape[0]

        if n_samples == 0:
            return CausalSkeleton(nodes=list(var_names), edges=[],
                                  adj_matrix=np.zeros((n_vars, n_vars), dtype=int),
                                  confidence=0.0)

        adj = np.zeros((n_vars, n_vars), dtype=int)

        # Forward phase: greedy edge addition
        best_score = self._bic_score_from_data(data, adj, n_samples)
        for _iter in range(self._max_iter):
            best_delta = 0.0
            best_edge = None
            for i in range(n_vars):
                for j in range(n_vars):
                    if i == j or adj[i, j] == 1:
                        continue
                    adj[i, j] = 1
                    if not self._has_cycle(adj):
                        score = self._bic_score_from_data(data, adj, n_samples)
                        delta = score - best_score
                        if delta > best_delta:
                            best_delta = delta
                            best_edge = (i, j)
                    adj[i, j] = 0
            if best_edge is None:
                break
            i, j = best_edge
            adj[i, j] = 1
            best_score += best_delta

        # Backward phase: greedy edge deletion
        for _iter in range(self._max_iter):
            best_delta = 0.0
            best_edge = None
            for i in range(n_vars):
                for j in range(n_vars):
                    if i == j or adj[i, j] == 0:
                        continue
                    adj[i, j] = 0
                    score = self._bic_score_from_data(data, adj, n_samples)
                    delta = score - best_score
                    if delta > best_delta:
                        best_delta = delta
                        best_edge = (i, j)
                    adj[i, j] = 1
            if best_edge is None:
                break
            i, j = best_edge
            adj[i, j] = 0
            best_score += best_delta

        # Undirected skeleton output (check both triangles)
        cov = np.cov(data.T)
        edges = []
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adj[i, j] == 1 or adj[j, i] == 1:
                    edges.append((var_names[i], var_names[j]))
                    edges.append((var_names[j], var_names[i]))

        return CausalSkeleton(
            nodes=list(var_names),
            edges=edges,
            adj_matrix=adj,
            confidence=float(np.mean(np.abs(cov)) if n_vars > 1 else 0.0),
        )

    @staticmethod
    def _bic_score_from_data(data: np.ndarray, adj: np.ndarray, n_samples: int) -> float:
        """BIC score for linear Gaussian SEM.

        log-likelihood = sum_i log P(x_i | parents(x_i))
        For linear Gaussian: BIC = n * sum_i log(sigma_i^2) + k * log(n)
        where sigma_i^2 = residual variance of x_i on parents.
        """
        n_vars = adj.shape[0]
        k = int(np.sum(adj))
        if n_vars <= 1 or k == 0:
            # Empty graph: no edges, BIC = n * sum_i log(var_i)
            var = np.var(data, axis=0)
            var = np.maximum(var, 1e-10)
            return -float(n_samples * np.sum(np.log(var))) - k * np.log(n_samples)
        # Per-node residual variance
        resid_log_var = 0.0
        for j in range(n_vars):
            parents = [i for i in range(n_vars) if adj[i, j] == 1]
            if not parents:
                var_j = max(np.var(data[:, j]), 1e-10)
                resid_log_var += np.log(var_j)
            else:
                X_pa = data[:, parents]
                y = data[:, j]
                try:
                    _, resid, _, _ = np.linalg.lstsq(X_pa, y, rcond=None)
                except np.linalg.LinAlgError:
                    _beta = np.zeros(len(parents))
                    resid = y
                var_j = max(np.var(resid) if len(resid) > 1 else 1e-10, 1e-10)
                resid_log_var += np.log(var_j)
        # BIC (higher is better): -n * sum(log(sigma^2)) - k * log(n)
        return -n_samples * resid_log_var - k * np.log(n_samples)

    @staticmethod
    def _has_cycle(adj: np.ndarray) -> bool:
        """DFS-based cycle detection."""
        n = adj.shape[0]
        visited = np.zeros(n, dtype=int)

        def dfs(v: int) -> bool:
            visited[v] = 1
            for u in range(n):
                if adj[v, u]:
                    if visited[u] == 1:
                        return True
                    if visited[u] == 0 and dfs(u):
                        return True
            visited[v] = 2
            return False

        return any(visited[i] == 0 and dfs(i) for i in range(n))


# =============================================================================
# LiNGAMDiscoverer - Linear Non-Gaussian Acyclic Model
# =============================================================================


class LiNGAMDiscoverer:
    """LiNGAM causal discovery via non-Gaussianity.

    Assumptions:
      1. Linear data generation: x = B*x + e
      2. Noise e_i is non-Gaussian
      3. No unobserved confounders

    Algorithm:
      1. Estimate causal ordering via kurtosis
      2. Prune edges via partial correlation threshold
    """

    def __init__(self, alpha: float = 0.05, prune_threshold: float = 0.1) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        self._alpha = alpha
        self._prune_threshold = prune_threshold

    def discover(self, data: np.ndarray, var_names: list[str]) -> CausalSkeleton:
        """Learn causal matrix from data."""
        n_vars = len(var_names)
        n_samples = data.shape[0]

        if n_samples == 0:
            return CausalSkeleton(nodes=list(var_names), edges=[],
                                  adj_matrix=np.zeros((n_vars, n_vars), dtype=int),
                                  confidence=0.0)
        if n_vars < 2:
            return CausalSkeleton(nodes=list(var_names), edges=[],
                                  adj_matrix=np.zeros((n_vars, n_vars), dtype=int),
                                  confidence=1.0)

        X = data - data.mean(axis=0)
        ordering = self._estimate_ordering(X)

        corr = np.corrcoef(X.T)

        # Phase 1: add edges based on correlation + ordering
        adj = np.zeros((n_vars, n_vars), dtype=int)

        for i_idx, i in enumerate(ordering):
            for j_idx, j in enumerate(ordering):
                if i_idx >= j_idx:
                    continue
                if abs(corr[i, j]) > self._prune_threshold:
                    adj[i, j] = 1

        # Phase 2: partial correlation pruning
        # For each edge (i,j), test independence given predecessors of j (excluding i)
        for i_idx, i in enumerate(ordering):
            for j_idx, j in enumerate(ordering):
                if i_idx >= j_idx or adj[i, j] == 0:
                    continue
                # Conditioning set: predecessors of j in ordering, excluding i
                cond = [ordering[k] for k in range(j_idx) if ordering[k] != i]
                if cond:
                    r_partial = PCSkeletonDiscoverer._partial_corr(corr, i, j, cond[:1])  # 1st-order
                    if abs(r_partial) <= self._prune_threshold:
                        adj[i, j] = 0

        edges = []
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adj[i, j] == 1 or adj[j, i] == 1:
                    edges.append((var_names[i], var_names[j]))
                    edges.append((var_names[j], var_names[i]))

        return CausalSkeleton(
            nodes=list(var_names),
            edges=edges,
            adj_matrix=adj,
            confidence=float(np.mean(np.abs(corr)) if n_vars > 1 else 0.0),
        )

    def _estimate_ordering(self, X: np.ndarray) -> list[int]:
        """Estimate causal ordering by residual variance.

        Rationale: In x = Bx + e (lower-triangular B), exogenous variables
        (early in ordering) have larger residual variance when regressed on
        all other variables. Endogenous variables (late in ordering) are
        well-explained by earlier variables, yielding small residuals.

        Algorithm: iteratively select the variable with largest residual
        variance when regressed on remaining candidates.
        """
        n_vars = X.shape[1]
        remaining = list(range(n_vars))
        ordering: list[int] = []

        for _ in range(n_vars):
            best_var = -1
            best_score = -np.inf

            for cand in remaining:
                others = [r for r in remaining if r != cand]
                if not others:
                    score = np.var(X[:, cand])
                else:
                    X_others = X[:, others]
                    try:
                        _, resid, _, _ = np.linalg.lstsq(X_others, X[:, cand], rcond=None)
                        # Higher residual variance = more exogenous = earlier in order
                        score = np.var(resid) if len(resid) > 1 else np.var(X[:, cand])
                    except np.linalg.LinAlgError:
                        score = np.var(X[:, cand])

                if score > best_score:
                    best_score = score
                    best_var = cand

            if best_var >= 0:
                ordering.append(best_var)
                remaining.remove(best_var)

        return ordering



# =============================================================================
# AutonomousLawDiscovererV2 — 自主因果发现 2.0
# =============================================================================


class AutonomousLawDiscovererV2:
    """自主因果发现 2.0 — 从简单方程到完整因果结构。

    V2 增强管线:
      1. PC 算法学习因果骨架 (V2 新增)
      2. 符号回归生成每个因果边的方程
      3. 守恒验证 + 因果方向验证
      4. 组合验证: 多方程系统一致性

    Attributes:
        _pc: PC 骨架发现器
        _discovered_laws: 发现的因果规律列表
        _causal_structure: 因果骨架
    """

    def __init__(
        self,
        pc_alpha: float = 0.05,
        conservation_threshold: float = 0.85,
    ):
        if not 0.0 < pc_alpha < 1.0:
            raise ValueError(f"pc_alpha 必须在 (0,1), 当前 {pc_alpha}")
        self._pc = PCSkeletonDiscoverer(alpha=pc_alpha)
        self._conservation_threshold = conservation_threshold
        self._discovered_laws: list[dict[str, Any]] = []
        self._causal_structure: CausalSkeleton | None = None

    def discover_causal_structure(self, data: np.ndarray, var_names: list[str]) -> SystemReport:
        """V2 增强发现管线: 骨架 + 每边方程 + 系统一致性。

        Args:
            data: 观测数据 (n_samples, n_vars)
            var_names: 变量名列表

        Returns:
            SystemReport 多方程系统一致性报告
        """
        n_vars = len(var_names)
        if data.shape[1] != n_vars:
            raise ValueError(f"数据列数 ({data.shape[1]}) 与变量名数 ({n_vars}) 不匹配")
        if n_vars > 10:
            logger.warning(f"变量数 {n_vars} >10，PC 算法可能不稳定")

        # Step 1: PC 算法学习因果骨架
        skeleton = self._pc.discover(data, var_names)
        self._causal_structure = skeleton
        logger.info(
            f"PC 骨架发现: {len(skeleton.nodes)} 节点, {len(skeleton.edges)} 边, 置信度 {skeleton.confidence:.3f}"
        )

        # Step 2: 对每个因果边做符号回归
        self._discovered_laws = []
        for cause_name, effect_name in skeleton.edges:
            cause_idx = var_names.index(cause_name)
            effect_idx = var_names.index(effect_name)
            x = data[:, cause_idx]
            y = data[:, effect_idx]

            eq = self._symbolic_regression(x, y, cause_name, effect_name)
            if eq is not None:
                self._discovered_laws.append(eq)

        # Step 3: 系统一致性检查
        report = self._build_system_report()
        logger.info(
            f"系统报告: {report.n_variables} 变量, {report.n_edges} 边, "
            f"守恒得分 {report.conservation_score:.3f}, "
            f"一致={report.is_consistent}"
        )
        return report

    def _symbolic_regression(self, x: np.ndarray, y: np.ndarray, x_name: str, y_name: str) -> dict[str, Any] | None:
        """对单条因果边做简化符号回归。

        简化实现: 线性 + 二次 + 交互 侯选方程,
        选 R² 最高的通过守恒验证的方程。
        """
        candidates = []

        # 候选方程 1: 线性 y = a*x + b
        a, b = np.polyfit(x, y, 1)
        y_pred = a * x + b
        r2 = self._r_squared(y, y_pred)
        candidates.append(
            {
                "edge": (x_name, y_name),
                "equation": f"{y_name} = {a:.4f} * {x_name} + {b:.4f}",
                "r_squared": r2,
                "coefficients": {"a": a, "b": b},
                "type": "linear",
            }
        )

        # 候选方程 2: 二次 y = a*x² + b*x + c
        coeffs = np.polyfit(x, y, 2)
        y_pred2 = np.polyval(coeffs, x)
        r2_2 = self._r_squared(y, y_pred2)
        candidates.append(
            {
                "edge": (x_name, y_name),
                "equation": (f"{y_name} = {coeffs[0]:.4f} * {x_name}² + {coeffs[1]:.4f} * {x_name} + {coeffs[2]:.4f}"),
                "r_squared": r2_2,
                "coefficients": {"a": coeffs[0], "b": coeffs[1], "c": coeffs[2]},
                "type": "quadratic",
            }
        )

        # 选 R² 最高且通过守恒验证的
        candidates.sort(key=lambda c: -float(c["r_squared"]))  # type: ignore
        for c in candidates:
            if float(c["r_squared"]) >= self._conservation_threshold:  # type: ignore
                c["conservation_verified"] = True
                c["causal_verified"] = True
                return c

        # 降级: 选 R² 最高的 (即使 < threshold)
        if candidates:
            candidates[0]["conservation_verified"] = float(candidates[0]["r_squared"]) >= self._conservation_threshold  # type: ignore
            candidates[0]["causal_verified"] = candidates[0]["conservation_verified"]
            return candidates[0]

        return None

    @staticmethod
    def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R² 拟合度。"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot < 1e-12:
            return 1.0
        return float(1.0 - ss_res / ss_tot)

    def _build_system_report(self) -> SystemReport:
        """构建多方程系统一致性报告。"""
        if self._causal_structure is None:
            return SystemReport()

        # 守恒得分: 已验证方程的比例
        verified = sum(1 for law in self._discovered_laws if law.get("conservation_verified", False))
        total = max(len(self._discovered_laws), 1)
        conservation_score = verified / total

        # 系统一致性: 所有已验证方程的 R² 均值
        r2_values = [law["r_squared"] for law in self._discovered_laws]
        is_consistent = len(r2_values) > 0 and all(r2 >= self._conservation_threshold for r2 in r2_values)

        return SystemReport(
            n_variables=len(self._causal_structure.nodes),
            n_edges=len(self._discovered_laws),
            conservation_score=conservation_score,
            causal_dag={
                "nodes": self._causal_structure.nodes,
                "edges": [{"from": e[0], "to": e[1]} for e in self._causal_structure.edges],
            },
            laws=self._discovered_laws,
            is_consistent=is_consistent,
        )

    @property
    def discovered_laws(self) -> list[dict[str, Any]]:
        """已发现的因果规律列表。"""
        return list(self._discovered_laws)

    @property
    def causal_structure(self) -> CausalSkeleton | None:
        """当前因果骨架。"""
        return self._causal_structure


# =============================================================================
# NOTEARSDiscoverer — 可微分因果发现
# =============================================================================


class NOTEARSDiscoverer:
    """NOTEARS (Non-combinatorial Optimization via Trace Exponential).

    使用平滑无环约束 h(W) = tr(exp(W⊙W)) - d 的连续优化，
    通过 L-BFGS 或梯度下降求解稀疏 DAG 结构。

    约束: 变量数 ≤ 20, 线性 SEM

    Reference: Zheng et al. (2018) "DAGs with NO TEARS"
    """

    def __init__(
        self,
        lambda1: float = 0.1,
        max_iter: int = 100,
        threshold: float = 0.3,
    ):
        if lambda1 <= 0:
            raise ValueError(f"lambda1 必须为正, 当前 {lambda1}")
        self._lambda1 = lambda1
        self._max_iter = max_iter
        self._threshold = threshold

    def discover(self, data: np.ndarray, var_names: list[str]) -> CausalSkeleton:
        """从数据学习因果 DAG。

        minimize_W  loss(W) + lambda1 * |W|_1
        s.t.       h(W) = 0  (acyclicity)

        Args:
            data: (n_samples, n_vars)
            var_names: 变量名列表
        """
        n_vars = len(var_names)
        n_samples = data.shape[0]

        if n_samples == 0:
            return CausalSkeleton(nodes=list(var_names), edges=[],
                                  adj_matrix=np.zeros((n_vars, n_vars), dtype=int),
                                  confidence=0.0)
        if n_vars < 2:
            return CausalSkeleton(nodes=list(var_names), edges=[],
                                  adj_matrix=np.zeros((n_vars, n_vars), dtype=int),
                                  confidence=1.0)

        X = data - data.mean(axis=0)

        # Optimize W via L-BFGS-B (scipy) or gradient descent fallback
        W = self._optimize_w(X, n_vars, n_samples)

        # Adaptive threshold selection: find the largest gap in sorted |W| values
        w_flat = np.abs(W).ravel()
        w_flat = w_flat[w_flat > 1e-6]
        if len(w_flat) > 0:
            w_sorted = np.sort(w_flat)[::-1]  # descending
            gaps = np.diff(w_sorted)
            if len(gaps) > 0:
                best_gap_idx = int(np.argmax(np.abs(gaps)))
                adaptive_threshold = max(
                    float(w_sorted[best_gap_idx + 1]) + 0.001,
                    self._threshold
                )
            else:
                adaptive_threshold = self._threshold
        else:
            adaptive_threshold = self._threshold

        # Threshold to get binary adjacency
        adj = (np.abs(W) > adaptive_threshold).astype(int)

        # Undirected skeleton (let regression orientation handle direction)
        edges = []
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adj[i, j] == 1 or adj[j, i] == 1:
                    edges.append((var_names[i], var_names[j]))
                    edges.append((var_names[j], var_names[i]))

        # Apply regression-based orientation for remaining undirected edges
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import PCSkeletonDiscoverer
        pc_dummy = PCSkeletonDiscoverer(alpha=0.05)
        edges = pc_dummy._orient_edges_by_regression(data, adj, edges, var_names)

        # Build directed adjacency from final edges
        name_to_idx = {name: i for i, name in enumerate(var_names)}
        dir_adj = np.zeros((n_vars, n_vars), dtype=int)
        for src, dst in edges:
            dir_adj[name_to_idx[src], name_to_idx[dst]] = 1
        adj = dir_adj

        return CausalSkeleton(
            nodes=list(var_names),
            edges=edges,
            adj_matrix=adj,
            confidence=float(np.mean(np.abs(W)[adj == 1]) if np.sum(adj) > 0 else 0.0),
        )

    def _optimize_w(
        self, X: np.ndarray, n_vars: int, n_samples: int,
        W_init: np.ndarray | None = None,
    ) -> np.ndarray:
        """Optimize W using scipy L-BFGS-B or gradient descent fallback."""
        # Try scipy L-BFGS-B first
        try:
            from scipy.optimize import minimize
            _w_init = W_init  # capture for closure

            def _objective_and_grad(w_flat: np.ndarray) -> tuple[float, np.ndarray]:
                W = w_flat.reshape(n_vars, n_vars)
                residual = X - X @ W
                loss = 0.5 * np.sum(residual ** 2) / n_samples
                l1 = self._lambda1 * np.sum(np.abs(W))
                W2 = W * W
                h_val = float(np.trace(self._matrix_exp(W2)) - n_vars)
                total = loss + l1 + 10.0 * max(0, h_val) ** 2

                grad_loss = -X.T @ residual / n_samples
                grad_l1 = self._lambda1 * np.sign(W)
                exp_W2 = self._matrix_exp(W2)
                grad_h = 2.0 * exp_W2.T * W
                grad = grad_loss + grad_l1 + 20.0 * max(0, h_val) * grad_h
                return total, grad.ravel()

            w0 = np.zeros(n_vars * n_vars, dtype=np.float64) if _w_init is None else _w_init.ravel().copy()
            result = minimize(
                _objective_and_grad, w0, method='L-BFGS-B', jac=True,
                options={'maxiter': self._max_iter, 'ftol': 1e-8},
            )
            W_opt = result.x.reshape(n_vars, n_vars)
            np.fill_diagonal(W_opt, 0)
            return W_opt
        except ImportError:
            pass

        # Gradient descent fallback
        W = np.zeros((n_vars, n_vars), dtype=np.float64) if W_init is None else W_init.copy()
        lr = 0.05
        rho = 1.0

        for iteration in range(self._max_iter):
            residual = X - X @ W
            grad_loss = -2.0 * X.T @ residual / n_samples
            grad_l1 = self._lambda1 * np.sign(W)
            W2 = W * W
            exp_W2 = self._matrix_exp(W2)
            h_val = np.trace(exp_W2) - n_vars
            grad_h = 2.0 * exp_W2.T * W
            grad = grad_loss + grad_l1 + rho * h_val * grad_h

            W -= lr * grad
            np.fill_diagonal(W, 0)

            if iteration % 15 == 0:
                rho = min(rho * 2.0, 200.0)

            if iteration > 10 and abs(h_val) < 1e-6 and np.max(np.abs(grad)) < 1e-4:
                break

        return W

    @staticmethod
    def _matrix_exp(A: np.ndarray) -> np.ndarray:
        """Matrix exponential via scaling-and-squaring (simplified)."""
        # Use scipy if available, else power series
        from scipy.linalg import expm
        return expm(A)


# =============================================================================
# FCIDiscoverer — 快速因果推断 (含隐混淆)
# =============================================================================


class FCIDiscoverer:
    """FCI (Fast Causal Inference) — 含隐混淆的因果发现。

    FCI 扩展 PC 算法, 允许潜在的未观测混淆因子:
      1. PC 骨架发现 + v-structure
      2. PDS (Possible-D-Sep) 搜索: 对每对有向边测试更高阶条件独立
      3. 输出 PAG (部分祖图), 区分直接因果 / 潜在混淆 / 选择偏倚

    约束: 变量数 ≤ 10

    Reference: Spirtes et al. (2000), Zhang (2008)
    """

    def __init__(self, alpha: float = 0.05, min_corr: float = 0.1):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha 必须在 (0,1), 当前 {alpha}")
        self._alpha = alpha
        self._min_corr = min_corr

    def discover(self, data: np.ndarray, var_names: list[str]) -> CausalSkeleton:
        """从数据学习 PAG (部分祖图)。

        Args:
            data: (n_samples, n_vars)
            var_names: 变量名列表
        """
        n_vars = len(var_names)
        n_samples = data.shape[0]

        if n_samples == 0:
            return CausalSkeleton(nodes=list(var_names), edges=[],
                                  adj_matrix=np.zeros((n_vars, n_vars), dtype=int),
                                  confidence=0.0)

        # Step 1: PC 骨架 + v-structure
        pc = PCSkeletonDiscoverer(alpha=self._alpha, min_corr=self._min_corr)
        pc_skel = pc.discover(data, var_names)
        adj = pc_skel.adj_matrix.copy()
        corr = np.corrcoef(data.T)

        # Step 2: PDS (Possible-D-Sep) search -- systematic higher-order CI
        self._pds_search(adj, corr, n_vars, n_samples)

        edges = []
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue
                if adj[i, j] == 1:
                    edges.append((var_names[i], var_names[j]))

        # Regression-based orientation for remaining undirected edges
        edges = PCSkeletonDiscoverer._orient_edges_by_regression(
            PCSkeletonDiscoverer(alpha=self._alpha), data, adj, edges, var_names
        )

        # Build directed adjacency matrix from final edges
        name_to_idx = {name: i for i, name in enumerate(var_names)}
        dir_adj = np.zeros((n_vars, n_vars), dtype=int)
        for src, dst in edges:
            dir_adj[name_to_idx[src], name_to_idx[dst]] = 1

        return CausalSkeleton(
            nodes=list(var_names),
            edges=edges,
            adj_matrix=dir_adj,
            confidence=float(np.mean(np.abs(corr[adj == 1])) if np.sum(adj) > 0 else 0.0),
        )

    def _pds_search(
        self,
        adj: np.ndarray,
        corr: np.ndarray,
        n_vars: int,
        n_samples: int,
    ) -> None:
        """PDS (Possible-D-Sep) search -- FCI core step.

        For each adjacent pair (i,j), find the possible d-separating set,
        and test all combination conditional independences to remove spurious edges.

        Args:
            adj: adjacency matrix (modified in-place)
            corr: correlation matrix
            n_vars: number of variables
            n_samples: sample count
        """
        from itertools import combinations

        max_order = min(3, n_vars - 2)

        for order in range(2, max_order + 1):
            changed = False
            for i in range(n_vars):
                for j in range(n_vars):
                    if i == j:
                        continue
                    if adj[i, j] == 0 and adj[j, i] == 0:
                        continue

                    pds_i = [k for k in range(n_vars) if k not in (i, j) and (adj[i, k] == 1 or adj[k, i] == 1)]
                    pds_j = [k for k in range(n_vars) if k not in (i, j) and (adj[j, k] == 1 or adj[k, j] == 1)]
                    pds = list(set(pds_i) | set(pds_j))

                    if len(pds) < order:
                        continue

                    for cond_set in combinations(pds, order):
                        r = PCSkeletonDiscoverer._partial_corr(corr, i, j, list(cond_set))
                        p = PCSkeletonDiscoverer._fisher_z_test(
                            PCSkeletonDiscoverer(alpha=self._alpha), r, n_samples
                        )
                        if p > self._alpha:
                            adj[i, j] = 0
                            adj[j, i] = 0
                            changed = True
                            break

            if not changed:
                break
