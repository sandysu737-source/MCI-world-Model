from __future__ import annotations

from typing import Any

"""
MCI World Model v4.6.0 — Pearl Do-Calculus 干预引擎 (M1)
====================================================

基于 Pearl (2009) do-calculus 的因果干预推理，
支持后门调整、前门调整、平均处理效应估计。

核心公式:
- 后门调整: P(Y | do(X=x)) = Σ_z P(Y | X=x, Z=z) · P(Z=z)
- 前门调整: P(Y | do(X=x)) = Σ_m P(m | X=x) · Σ_x' P(Y | x', m) · P(x')
- ATE: E[Y | do(X=1)] - E[Y | do(X=0)]

设计原则:
- 零新依赖: 纯 numpy 实现
- 图容错: 空图/单节点/循环图均有守卫
- 与 GaussianDAG 无缝对接: build_from_gaussian_dag() 静态工厂

用法:
    from mci_world_model.sdk._do_calculus import DoCalculus, CausalGraph

    cg = CausalGraph(
        nodes=["Z", "X", "Y"],
        edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
    )
    dc = DoCalculus(cg)
    result = dc.estimate_ate("X", "Y")
    logger.info(f"ATE: {result.ate:.4f} [{result.confidence_interval}]")
"""


import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


# =============================================================================
# InterventionResult — 干预分析结果
# =============================================================================


@dataclass
class InterventionResult:
    """
    do-operator 干预分析结果。

    包含:
    - ATE (平均处理效应)
    - 95% 置信区间
    - 使用的调整变量集
    - 干预方法标识
    """

    intervention: str = ""  # do 干预描述 (如 "do(X=1.5)")
    target: str = ""  # 目标变量
    ate: float = 0.0  # 平均处理效应
    confidence_interval: tuple[float, float] = (0.0, 0.0)  # 95% CI
    ci_level: float = 0.95  # 置信水平
    adjustment_set: list[str] = field(default_factory=list)  # 调整变量
    method: str = "none"  # "backdoor" | "frontdoor" | "direct" | "none"
    p_value: float = 1.0  # 双尾 p-value
    effect_direction: str = "unknown"  # "positive" | "negative" | "neutral"
    effect_magnitude: str = "unknown"  # "large" | "medium" | "small" | "negligible"
    sample_size: int = 0  # 有效样本量
    note: str = ""  # 附加说明

    def to_dict(self) -> dict[str, Any]:
        return {
            "intervention": self.intervention,
            "target": self.target,
            "ate": round(self.ate, 6),
            "confidence_interval_95": (
                round(self.confidence_interval[0], 6),
                round(self.confidence_interval[1], 6),
            ),
            "adjustment_set": self.adjustment_set,
            "method": self.method,
            "p_value": round(self.p_value, 6),
            "effect_direction": self.effect_direction,
            "effect_magnitude": self.effect_magnitude,
            "sample_size": self.sample_size,
            "note": self.note,
        }

    @staticmethod
    def empty(method: str = "none") -> InterventionResult:
        return InterventionResult(method=method, note="insufficient_data")


# =============================================================================
# CausalGraph — 因果图数据结构
# =============================================================================


@dataclass
class CausalGraph:
    """
    因果有向图 (DAG) 数据结构。

    支持:
    - 邻接矩阵高效查询
    - 父节点/子节点/后代的图算法
    - 从 GaussianDAG 边列表构建
    """

    nodes: list[str] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    adjacency: np.ndarray | None = None  # n×n 邻接矩阵, adj[i,j] = 边权重 or 1

    def __post_init__(self) -> None:
        n = len(self.nodes)
        if self.adjacency is None:
            self.adjacency = np.zeros((n, n), dtype=np.float32)
            node_idx = {name: i for i, name in enumerate(self.nodes)}
            for src, dst in self.edges:
                i = node_idx.get(src)
                j = node_idx.get(dst)
                if i is not None and j is not None:
                    self.adjacency[i, j] = 1.0

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def node_index(self, name: str) -> int | None:
        """获取节点名称对应的索引。"""
        try:
            return self.nodes.index(name)
        except ValueError:
            return None

    def has_edge(self, src: str, dst: str) -> bool:
        """检查是否存在有向边 src → dst。"""
        if self.adjacency is None:
            return False
        i = self.node_index(src)
        j = self.node_index(dst)
        if i is None or j is None:
            return False
        return self.adjacency[i, j] > 0

    def get_parents(self, node: str) -> list[str]:
        """获取节点的所有父节点 (指向 node 的节点)。"""
        idx = self.node_index(node)
        if idx is None or self.adjacency is None:
            return []
        parents = []
        for i in range(self.n_nodes):
            if self.adjacency[i, idx] > 0:
                parents.append(self.nodes[i])
        return parents

    def get_children(self, node: str) -> list[str]:
        """获取节点的所有子节点 (node 指向的节点)。"""
        idx = self.node_index(node)
        if idx is None or self.adjacency is None:
            return []
        children = []
        for j in range(self.n_nodes):
            if self.adjacency[idx, j] > 0:
                children.append(self.nodes[j])
        return children

    def to_dag(self):
        """转换为 algebra 层的 CausalDAG (用于 d-separation 等图论运算)。

        桥接业务层 CausalGraph (邻接矩阵表示) 到纯数学层 CausalDAG
        (边列表表示), 使 do-calculus 能调用 algebra 的 d-separation /
        后门准则等图论算法。
        """
        from mci_world_model.algebra.causal_graph import CausalDAG

        dag = CausalDAG()
        for n in self.nodes:
            dag.add_node(n)
        for src, dst in self.edges:
            dag.add_edge(src, dst, weight=1.0)
        return dag

    def get_descendants(self, node: str) -> set[str]:
        """获取节点的所有后代 (BFS)。"""
        result: set[str] = set()
        idx = self.node_index(node)
        if idx is None:
            return result
        queue = deque([node])
        while queue:
            current = queue.popleft()
            for child in self.get_children(current):
                if child not in result:
                    result.add(child)
                    queue.append(child)
        return result

    def get_mediators(self, src: str, dst: str) -> list[str]:
        """
        获取 src → dst 路径上的所有中间节点 (中介变量)。

        使用 BFS 从 src 的 children 出发，检查是否可达 dst。
        """
        mediators = []
        src_children = self.get_children(src)
        for child in src_children:
            if child == dst:
                continue
            descendants = self.get_descendants(child)
            if dst in descendants:
                mediators.append(child)
        return mediators

    def __repr__(self) -> str:
        return f"CausalGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"

    # -----------------------------------------------------------------
    # CausalGraph ↔ SEM 双向转换 (v3.0.8)
    # -----------------------------------------------------------------

    def to_sem(  # type: ignore[no-untyped-def]
        self,
        noise_std: float = 0.5,
        activation: str = "linear",
        seed: int | None = None,
    ):
        """
        将 CausalGraph 的邻接矩阵转为 StructuralEquationModel。

        邻接矩阵 → SEM 系数矩阵: 非零边保留权重，零边保持零。

        Args:
            noise_std: SEM 噪声标准差
            activation: 激活函数 — "linear" | "tanh" | "relu" | "sigmoid"
            seed: 随机种子

        Returns:
            StructuralEquationModel 实例
        """
        from mci_world_model.sdk._counterfactual import StructuralEquationModel

        coeff = (
            np.array(self.adjacency, dtype=np.float64)
            if self.adjacency is not None
            else np.zeros((len(self.nodes), len(self.nodes)), dtype=np.float64)
        )
        return StructuralEquationModel(
            coefficients=coeff,
            node_names=list(self.nodes),
            noise_std=noise_std,
            activation=activation,
            seed=seed,
        )

    @staticmethod
    def from_sem(sem: Any) -> CausalGraph:
        """
        从 StructuralEquationModel 系数矩阵反向构建 CausalGraph。

        非零系数 → 边存在 (保留权重)。

        Args:
            sem: StructuralEquationModel 实例

        Returns:
            CausalGraph 实例
        """
        cg = CausalGraph(nodes=list(sem.node_names))
        n = sem.n_nodes
        adj = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                w = sem.coefficients[i, j]
                if w != 0:
                    adj[i, j] = float(w)
                    cg.add_edge(sem.node_names[i], sem.node_names[j])
        cg.adjacency = adj
        return cg

    def add_edge(self, src: str, dst: str, weight: float = 1.0) -> None:
        """添加因果边。"""
        if src not in self.nodes:
            self.nodes.append(src)
        if dst not in self.nodes:
            self.nodes.append(dst)
        if (src, dst) not in self.edges:
            self.edges.append((src, dst))
        # 更新邻接矩阵
        n = len(self.nodes)
        if self.adjacency is None or self.adjacency.shape[0] < n:
            old = self.adjacency
            self.adjacency = np.zeros((n, n), dtype=np.float32)
            if old is not None:
                self.adjacency[: old.shape[0], : old.shape[1]] = old
        i = self.node_index(src)
        j = self.node_index(dst)
        if i is not None and j is not None:
            self.adjacency[i, j] = float(weight)


# =============================================================================
# DoCalculus — Pearl do-calculus 干预推理引擎
# =============================================================================


class DoCalculus:
    """
    Pearl do-calculus 干预推理引擎。

    实现后门调整和前门调整，自动选择最优调整策略。

    Example:
        >>> cg = CausalGraph(
        ...     nodes=["Z", "X", "Y"],
        ...     edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        ... )
        >>> dc = DoCalculus(cg)
        >>> adj = dc.identify_adjustment_set("X", "Y")
        >>> print(adj)  # ["Z"]
        >>> result = dc.estimate_ate("X", "Y")
        >>> print(f"ATE={result.ate:.4f}, method={result.method}")
    """

    def __init__(
        self,
        graph: CausalGraph | None = None,
        data: dict[str, np.ndarray] | None = None,
        seed: int = 42,
    ):
        """
        Args:
            graph: 因果图 (CausalGraph 或 None)
            data: 观测数据 {node_name: values_array} (可选)
            seed: 随机种子 (用于模拟数据生成)
        """
        self._graph = graph
        self._data = data or {}
        self._rng = np.random.RandomState(seed)
        self._is_simulated: bool = len(self._data) == 0

    # -----------------------------------------------------------------
    # 图管理
    # -----------------------------------------------------------------

    def set_graph(self, graph: CausalGraph) -> None:
        """设置/更新因果图。"""
        self._graph = graph

    def set_data(self, data: dict[str, np.ndarray]) -> None:
        """设置观测数据。"""
        self._data = data
        self._is_simulated = False

    # -----------------------------------------------------------------
    # 后门准则 — 调整变量集识别
    # -----------------------------------------------------------------

    def identify_adjustment_set(self, X: str, Y: str) -> list[str] | None:
        """
        基于后门准则识别有效的调整变量集。

        后门准则 (Pearl, 2009):
        Z 是一个有效的调整集，当且仅当:
        1. Z 不包含 X 的后代
        2. Z 阻断 X 和 Y 之间所有含有指向 X 的箭头的路径 (后门路径)

        实现: 委托 algebra 层 CausalDAG 的图论算法做真正的后门准则验证。
        先取 X 的父节点作为候选 (常见的充分调整集), 再用 d-separation
        验证它确实阻断所有后门路径。若父节点集无效则回退到 None。
        """
        if self._graph is None:
            return None

        # 候选调整集: X 的父节点 (排除 X, Y 自身)
        parents = [p for p in self._graph.get_parents(X) if p not in (Y, X)]

        # 用 algebra 层验证后门准则
        dag = self._graph.to_dag()
        if dag.is_valid_adjustment_set(X, Y, set(parents)):
            return parents if parents else None
        # 父节点集无效: 尝试找最小有效调整集
        minimal = dag.find_minimal_adjustment_set(X, Y)
        if minimal is not None:
            return sorted(minimal)
        return None

    # -----------------------------------------------------------------
    # 前门准则 — 中介变量识别
    # -----------------------------------------------------------------

    def identify_frontdoor_mediators(self, X: str, Y: str) -> list[str] | None:
        """
        识别前门调整可用的中介变量集。

        前门准则 (Pearl, 2009):
        M 满足前门准则，当且仅当:
        1. M 阻断所有 X → Y 的有向路径
        2. 不存在从 X 到 M 的后门路径 (即 X 和 M 之间无混杂)
        3. 所有从 M 到 Y 的后门路径都被 X 阻断

        简化实现: 返回 X→Y 路径上直接的中间节点。

        Args:
            X: 干预变量
            Y: 目标变量

        Returns:
            中介变量名列表，或 None (前门调整不可用)
        """
        if self._graph is None:
            return None

        mediators = self._graph.get_mediators(X, Y)
        # 前门准则的条件 2: 不存在从 X 到 M 的后门路径
        # 简化: 检查 X 和 M 之间是否有共同原因 (混杂)
        valid_mediators = []
        for m in mediators:
            x_parents = set(self._graph.get_parents(X))
            m_parents = set(self._graph.get_parents(m))
            common_causes = x_parents & m_parents
            if not common_causes:
                valid_mediators.append(m)

        return valid_mediators if valid_mediators else None

    # -----------------------------------------------------------------
    # 后门调整
    # -----------------------------------------------------------------

    def backdoor_adjustment(
        self,
        X: str,
        Y: str,
        Z_set: list[str],
        x_value: float = 1.0,
        x_baseline: float = 0.0,
    ) -> InterventionResult:
        """
        后门调整公式:

            P(Y | do(X=x)) = Σ_z P(Y | X=x, Z=z) · P(Z=z)

        实现:
        1. 对 Z 的每个取值组合，计算条件期望 E[Y | X=x, Z=z]
        2. 按 P(Z=z) 加权求和
        3. ATE = E[Y | do(X=x)] - E[Y | do(X=x_baseline)]

        Args:
            X: 干预变量
            Y: 目标变量
            Z_set: 调整变量集
            x_value: do(X) 的干预值
            x_baseline: 基线值 (通常是 do(X=0) 或 do(X=自然值))

        Returns:
            InterventionResult 含 ATE 估计

        Warns:
            当无真实观测数据时, 结果基于因果图模拟的 SEM 数据 (循环验证),
            ATE 反映图的拓扑结构而非真实因果效应。结果 method 字段标注为
            "backdoor_simulated" 以提醒用户。
        """
        if self._is_simulated and self._graph is not None:
            result = self._backdoor_simulated(X, Y, Z_set, x_value, x_baseline)
            # D6 修复: 明确标注结果来自模拟 (循环验证), 避免误导用户
            if result.method != "none":
                result.method = "backdoor_simulated"
            return result

        # ── 基于观测数据的后门调整 ──
        if X not in self._data or Y not in self._data:
            return InterventionResult.empty(method="backdoor")

        x_data = self._data[X]
        y_data = self._data[Y]
        n_samples = len(x_data)

        if n_samples < 5:
            return InterventionResult.empty(method="backdoor")

        # 后门调整公式: E[Y | do(X=x)] = Σ_z E[Y | X=x, Z=z] · P(Z=z)
        # 用 OLS 回归 Y ~ X + Z 估计条件期望 E[Y|X=x,Z=z], 再按 P(Z) 加权求和。
        # 这是线性/可加假设下的标准后门调整实现; 对连续和离散 X/Z 均适用。
        z_names = [z for z in Z_set if z in self._data]
        if not z_names:
            # 无调整变量: ATE = E[Y|X=x_value] - E[Y|X=x_baseline]
            def _ey_given_x(x_level: float) -> float:
                mask = np.abs(x_data - x_level) < 1e-6
                return float(np.mean(y_data[mask])) if np.any(mask) else float(np.mean(y_data))

            ate = _ey_given_x(x_value) - _ey_given_x(x_baseline)
        else:
            # 构建设计矩阵 [1, X, Z1, Z2, ...] 做 OLS
            Z_mat = np.column_stack([self._data[z] for z in z_names])
            design = np.column_stack([np.ones(n_samples), x_data, Z_mat])
            try:
                beta, _, _, _ = np.linalg.lstsq(design, y_data, rcond=None)
                # beta = [截距, coef_X, coef_Z1, ...]
                coef_x = beta[1]
                # 线性 SEM 下 E[Y|do(X=x)] = β_X · x + const, ATE = β_X · (x_value - x_baseline)
                # const 项 (含 Z 的边际贡献) 在两次 do 中相消
                ate = float(coef_x * (x_value - x_baseline))
            except np.linalg.LinAlgError:
                ate = 0.0

        # 置信区间 (Wald-type)
        y_std = np.std(y_data) if n_samples > 1 else 0.1
        se = y_std / np.sqrt(n_samples)
        z_alpha = norm.ppf(0.975)  # 1.96 for 95% CI
        ci_lower = ate - z_alpha * se
        ci_upper = ate + z_alpha * se

        # p-value (近似)
        if se > 1e-10:
            z_stat = abs(ate) / se
            p_value = 2.0 * (1.0 - norm.cdf(z_stat))
        else:
            p_value = 1.0

        return self._build_result(
            X=X,
            Y=Y,
            x_value=x_value,
            x_baseline=x_baseline,
            ate=ate,
            ci=(ci_lower, ci_upper),
            adjustment_set=Z_set,
            method="backdoor",
            p_value=p_value,
            sample_size=n_samples,
        )

    def _backdoor_simulated(
        self,
        X: str,
        Y: str,
        Z_set: list[str],
        x_value: float,
        x_baseline: float,
    ) -> InterventionResult:
        """
        基于因果图生成模拟数据的后门调整。

        当没有真实观测数据时，使用因果图的线性 SEM 模拟数据。
        """
        n_sim = 500
        assert self._graph is not None, "Graph not initialized"
        n_nodes = self._graph.n_nodes
        node_idx = {name: i for i, name in enumerate(self._graph.nodes)}

        # 线性 SEM: Y = B^T · Y + ε, ε ~ N(0, 1)
        # 显式拓扑排序后逐节点采样
        X_idx = node_idx.get(X)
        Y_idx = node_idx.get(Y)
        if X_idx is None or Y_idx is None:
            return InterventionResult.empty(method="backdoor")

        # 拓扑排序
        topo_order = self._topological_sort()
        if topo_order is None:
            return InterventionResult.empty(method="backdoor")

        # 模拟数据 (n_sim 个样本)
        sim_data = np.zeros((n_sim, n_nodes), dtype=np.float64)
        for node_i in topo_order:
            # 噪声
            noise = self._rng.randn(n_sim) * 0.5
            parent_vals = np.zeros(n_sim)
            for p_idx in range(n_nodes):
                if self._graph.adjacency[p_idx, node_i] > 0:  # type: ignore
                    weight = self._graph.adjacency[p_idx, node_i]  # type: ignore
                    parent_vals += weight * sim_data[:, p_idx]
            sim_data[:, node_i] = parent_vals + noise

        # ── 后门调整 ──
        y_total = 0.0
        z_weight_sum = 0.0

        for z_name in Z_set:
            z_idx = node_idx.get(z_name)
            if z_idx is None:
                continue

            z_vals = sim_data[:, z_idx]
            # 离散化 Z
            z_edges = np.percentile(z_vals, [0, 25, 50, 75, 100])
            for k in range(len(z_edges) - 1):
                mask = (z_vals >= z_edges[k]) & (z_vals < z_edges[k + 1])
                n_z = np.sum(mask)
                if n_z < 5:
                    continue
                p_z = n_z / n_sim

                # 在 Z=z_k 的条件下，X 接近 x_value 时的 Y 期望
                x_vals = sim_data[:, X_idx]
                y_vals = sim_data[:, Y_idx]
                # 线性回归: E[Y | X=x, Z=z] ≈ intercept + slope * x
                if n_z > 2:
                    slope = np.cov(x_vals[mask], y_vals[mask])[0, 1] / (np.var(x_vals[mask]) + 1e-10)
                    intercept = np.mean(y_vals[mask]) - slope * np.mean(x_vals[mask])
                    y_given_do = intercept + slope * x_value
                else:
                    y_given_do = np.mean(y_vals[mask]) if n_z > 0 else 0.0

                y_total += y_given_do * p_z
                z_weight_sum += p_z

        y_do = y_total / max(z_weight_sum, 1e-10)

        # 基线: 自然观测下的 E[Y]
        y_baseline = np.mean(sim_data[:, Y_idx])

        ate = y_do - y_baseline

        # 置信区间 (bootstrap)
        ate_boots = []
        for _ in range(200):
            idx = self._rng.randint(0, n_sim, n_sim)
            ate_boots.append(np.mean(sim_data[idx, Y_idx]) - np.mean(sim_data[:, Y_idx]))
        se = np.std(ate_boots)
        z_alpha = norm.ppf(0.975)
        ci = (ate - z_alpha * se, ate + z_alpha * se)
        p_value = 2.0 * (1.0 - norm.cdf(abs(ate) / max(se, 1e-10)))

        return self._build_result(
            X=X,
            Y=Y,
            x_value=x_value,
            x_baseline=x_baseline,
            ate=ate,
            ci=ci,
            adjustment_set=Z_set,
            method="backdoor",
            p_value=p_value,
            sample_size=n_sim,
            note="simulated_data",
        )

    # -----------------------------------------------------------------
    # 前门调整
    # -----------------------------------------------------------------

    def frontdoor_adjustment(
        self,
        X: str,
        Y: str,
        M_set: list[str],
        x_value: float = 1.0,
        x_baseline: float = 0.0,
    ) -> InterventionResult:
        """
        前门调整公式:

            P(Y | do(X=x)) = Σ_m P(m | X=x) · Σ_x' P(Y | x', m) · P(x')

        两步:
        1. 估计 X → M 的因果效应 (P(m | do(X=x)) = P(m | X=x) 因为无后门路径)
        2. 估计 M → Y 的因果效应 (控制 X 阻断后门路径)
        3. 合成: P(Y | do(X=x)) = Σ_m P(m | X=x) · Σ_x' P(Y | x', m) · P(x')

        Args:
            X: 干预变量
            Y: 目标变量
            M_set: 中介变量集 (满足前门准则)
            x_value: do(X) 的干预值
            x_baseline: 基线值

        Returns:
            InterventionResult
        """
        if self._is_simulated and self._graph is not None:
            return self._frontdoor_simulated(X, Y, M_set, x_value, x_baseline)

        # ── 基于观测数据的前门调整 ──
        if X not in self._data or Y not in self._data:
            return InterventionResult.empty(method="frontdoor")

        x_data = self._data[X]
        y_data = self._data[Y]
        n_samples = len(x_data)

        if n_samples < 10:
            return InterventionResult.empty(method="frontdoor")

        y_do_total = 0.0
        n_valid_m = 0

        for m_name in M_set:
            if m_name not in self._data:
                continue
            m_data = self._data[m_name]

            # 步骤 1: P(M | X=x) — 在干预值下的 M 分布
            x_disc = self._discretize(x_data, 6)
            x_close_mask = np.abs(x_data - x_value) < (x_disc[1] - x_disc[0]) * 2
            m_given_x = m_data[x_close_mask] if np.any(x_close_mask) else m_data

            # 步骤 2: Σ_x' P(Y | x', m) · P(x')
            x_unique_vals = np.unique(x_data)
            x_probs = np.array([np.mean(x_data == xv) for xv in x_unique_vals])

            y_expected = 0.0
            for xi, xv in enumerate(x_unique_vals):
                m_disc = self._discretize(m_data, 5)
                for k in range(len(m_disc) - 1):
                    m_mask = (m_data >= m_disc[k]) & (m_data < m_disc[k + 1])
                    x_mask = np.abs(x_data - float(xv)) < 0.1
                    joint = m_mask & x_mask
                    if np.sum(joint) < 3:
                        continue
                    y_cond = np.mean(y_data[joint])
                    p_m = (
                        np.mean(m_given_x >= m_disc[k]) - np.mean(m_given_x >= m_disc[k + 1])
                        if k < len(m_disc) - 2
                        else 1.0 / len(m_disc)
                    )
                    y_expected += y_cond * p_m * x_probs[xi]

            y_do_total += y_expected
            n_valid_m += 1

        y_do = y_do_total / max(n_valid_m, 1)
        y_baseline = np.mean(y_data)
        ate = y_do - y_baseline

        y_std = np.std(y_data) if n_samples > 1 else 0.1
        se = y_std / np.sqrt(n_samples)
        z_alpha = norm.ppf(0.975)
        ci = (ate - z_alpha * se, ate + z_alpha * se)
        p_value = 2.0 * (1.0 - norm.cdf(abs(ate) / max(se, 1e-10)))

        return self._build_result(
            X=X,
            Y=Y,
            x_value=x_value,
            x_baseline=x_baseline,
            ate=ate,
            ci=ci,
            adjustment_set=M_set,
            method="frontdoor",
            p_value=p_value,
            sample_size=n_samples,
        )

    def _frontdoor_simulated(
        self,
        X: str,
        Y: str,
        M_set: list[str],
        x_value: float,
        x_baseline: float,
    ) -> InterventionResult:
        """
        基于模拟数据的前门调整。
        """
        n_sim = 500
        node_idx = {name: i for i, name in enumerate(self._graph.nodes)}  # type: ignore

        X_idx = node_idx.get(X)
        Y_idx = node_idx.get(Y)
        if X_idx is None or Y_idx is None:
            return InterventionResult.empty(method="frontdoor")

        topo_order = self._topological_sort()
        if topo_order is None:
            return InterventionResult.empty(method="frontdoor")

        n_nodes = self._graph.n_nodes  # type: ignore

        # 模拟自然数据
        sim_natural = np.zeros((n_sim, n_nodes), dtype=np.float64)
        for node_i in topo_order:
            noise = self._rng.randn(n_sim) * 0.5
            parent_vals = np.zeros(n_sim)
            for p_idx in range(n_nodes):
                if self._graph.adjacency[p_idx, node_i] > 0:  # type: ignore[union-attr,index]
                    parent_vals += self._graph.adjacency[p_idx, node_i] * sim_natural[:, p_idx]  # type: ignore
            sim_natural[:, node_i] = parent_vals + noise

        # 干预模拟: 固定 X = x_value, 再模拟下游
        sim_do = sim_natural.copy()
        sim_do[:, X_idx] = x_value
        for node_i in topo_order:
            if node_i == X_idx:
                continue
            noise = self._rng.randn(n_sim) * 0.5
            parent_vals = np.zeros(n_sim)
            for p_idx in range(n_nodes):
                if self._graph.adjacency[p_idx, node_i] > 0:  # type: ignore
                    parent_vals += self._graph.adjacency[p_idx, node_i] * sim_do[:, p_idx]  # type: ignore
            sim_do[:, node_i] = parent_vals + noise

        y_do = np.mean(sim_do[:, Y_idx])
        y_baseline_do = np.mean(sim_natural[:, Y_idx])

        ate = y_do - y_baseline_do
        se = np.std(sim_natural[:, Y_idx]) / np.sqrt(n_sim)
        z_alpha = norm.ppf(0.975)
        ci = (ate - z_alpha * se, ate + z_alpha * se)
        p_value = 2.0 * (1.0 - norm.cdf(abs(ate) / max(se, 1e-10)))

        return self._build_result(
            X=X,
            Y=Y,
            x_value=x_value,
            x_baseline=x_baseline,
            ate=ate,
            ci=ci,
            adjustment_set=M_set,
            method="frontdoor",
            p_value=p_value,
            sample_size=n_sim,
            note="simulated_data_(do-intervention)",
        )

    # -----------------------------------------------------------------
    # ATE 估计 — 自动方法选择
    # -----------------------------------------------------------------

    def estimate_ate(
        self,
        X: str,
        Y: str,
        x_value: float = 1.0,
        x_baseline: float = 0.0,
        method: str = "auto",
    ) -> InterventionResult:
        """
        估计平均处理效应 ATE，自动选择最优方法。

        方法选择逻辑:
        1. 尝试后门调整 (需要有效的调整集)
        2. 如果后门不可用，尝试前门调整
        3. 如果都不可用，回退到直接效应估计

        Args:
            X: 干预变量 (原因)
            Y: 目标变量 (效果)
            x_value: do(X) 干预值
            x_baseline: 基线值
            method: "auto" | "backdoor" | "frontdoor"

        Returns:
            InterventionResult
        """
        # F4-P1-3: NaN/Inf 边界守卫 — 拒绝非有限输入，保证 ATE 计算洁污
        if not np.isfinite(x_value):
            return InterventionResult(
                intervention=f"do({X}={x_value})",
                target=Y,
                method="rejected",
                note=f"x_value must be finite, got {x_value}",
            )
        if not np.isfinite(x_baseline):
            return InterventionResult(
                intervention=f"do({X}={x_value})",
                target=Y,
                method="rejected",
                note=f"x_baseline must be finite, got {x_baseline}",
            )

        if self._graph is None:
            return InterventionResult.empty(method="none")

        # X / Y 节点存在性检查
        if X not in self._graph.nodes or Y not in self._graph.nodes:
            return InterventionResult(
                intervention=f"do({X}={x_value})",
                target=Y,
                method="rejected",
                note=f"X ({X}) or Y ({Y}) not in graph nodes: {self._graph.nodes}",
            )

        # ── 方法选择 ──
        if method in ("auto", "backdoor"):
            adj_set = self.identify_adjustment_set(X, Y)
            if adj_set:
                return self.backdoor_adjustment(X, Y, adj_set, x_value, x_baseline)

        if method in ("auto", "frontdoor"):
            mediators = self.identify_frontdoor_mediators(X, Y)
            if mediators:
                return self.frontdoor_adjustment(X, Y, mediators, x_value, x_baseline)

        # ── 回退: 直接效应 (无调整) ──
        return self.direct_effect(X, Y, x_value, x_baseline)

    # -----------------------------------------------------------------
    # 受控直接效应
    # -----------------------------------------------------------------

    def direct_effect(
        self,
        X: str,
        Y: str,
        x_value: float = 1.0,
        x_baseline: float = 0.0,
    ) -> InterventionResult:
        """
        受控直接效应 (Controlled Direct Effect):

            CDE = E[Y | do(X=x₁)] - E[Y | do(X=x₀)]

        当没有观测数据可用时，使用因果关系图进行模拟。
        """
        if self._is_simulated and self._graph is not None:
            return self._direct_effect_simulated(X, Y, x_value, x_baseline)

        # 基于观测数据的简单估计
        if X in self._data and Y in self._data:
            x_data = self._data[X]
            y_data = self._data[Y]
            n = len(x_data)

            # 简单协方差估计
            cov_xy = np.cov(x_data, y_data)[0, 1] if n > 1 else 0.0
            var_x = np.var(x_data) if n > 1 else 1.0
            slope = cov_xy / max(var_x, 1e-10)
            ate = slope * (x_value - x_baseline)

            se = np.std(y_data) / np.sqrt(n) if n > 1 else 0.1
            z_alpha = norm.ppf(0.975)
            ci = (ate - z_alpha * se, ate + z_alpha * se)
            p_value = 2.0 * (1.0 - norm.cdf(abs(ate) / max(se, 1e-10)))

            return self._build_result(
                X=X,
                Y=Y,
                x_value=x_value,
                x_baseline=x_baseline,
                ate=ate,
                ci=ci,
                adjustment_set=[],
                method="direct",
                p_value=p_value,
                sample_size=n,
                note="no_adjustment_(confounded_estimate)",
            )

        return InterventionResult.empty(method="direct")

    def _direct_effect_simulated(
        self,
        X: str,
        Y: str,
        x_value: float,
        x_baseline: float,
    ) -> InterventionResult:
        """模拟数据的直接效应。"""
        n_sim = 500
        node_idx = {name: i for i, name in enumerate(self._graph.nodes)}  # type: ignore

        X_idx = node_idx.get(X)
        Y_idx = node_idx.get(Y)
        if X_idx is None or Y_idx is None:
            return InterventionResult.empty(method="direct")

        topo_order = self._topological_sort()
        if topo_order is None:
            return InterventionResult.empty(method="direct")

        n_nodes = self._graph.n_nodes  # type: ignore

        sim = np.zeros((n_sim, n_nodes), dtype=np.float64)
        for node_i in topo_order:
            noise = self._rng.randn(n_sim) * 0.5
            parent_vals = np.zeros(n_sim)
            for p_idx in range(n_nodes):
                if self._graph.adjacency[p_idx, node_i] > 0:  # type: ignore
                    parent_vals += self._graph.adjacency[p_idx, node_i] * sim[:, p_idx]  # type: ignore
            sim[:, node_i] = parent_vals + noise

        # do-intervention
        sim_do = sim.copy()
        sim_do[:, X_idx] = x_value
        for node_i in topo_order:
            if node_i == X_idx:
                continue
            noise = self._rng.randn(n_sim) * 0.5
            parent_vals = np.zeros(n_sim)
            for p_idx in range(n_nodes):
                if self._graph.adjacency[p_idx, node_i] > 0:  # type: ignore
                    parent_vals += self._graph.adjacency[p_idx, node_i] * sim_do[:, p_idx]  # type: ignore
            sim_do[:, node_i] = parent_vals + noise

        y_do = np.mean(sim_do[:, Y_idx])
        y_natural = np.mean(sim[:, Y_idx])
        ate = y_do - y_natural

        se = np.std(sim[:, Y_idx]) / np.sqrt(n_sim)
        z_alpha = norm.ppf(0.975)
        ci = (ate - z_alpha * se, ate + z_alpha * se)
        p_value = 2.0 * (1.0 - norm.cdf(abs(ate) / max(se, 1e-10)))

        return self._build_result(
            X=X,
            Y=Y,
            x_value=x_value,
            x_baseline=x_baseline,
            ate=ate,
            ci=ci,
            adjustment_set=[],
            method="direct",
            p_value=p_value,
            sample_size=n_sim,
            note="simulated_do-intervention_(no_adjustment)",
        )

    # -----------------------------------------------------------------
    # 静态工厂: 从 GaussianDAG 边列表构建
    # -----------------------------------------------------------------

    @staticmethod
    def build_from_gaussian_dag(
        edges: list[dict[str, Any]],
        n_nodes: int,
        min_confidence: float = 0.3,
    ) -> CausalGraph:
        """
        从 GaussianDAG.discover_hidden_edges() 输出构建因果图。

        只保留置信度 >= min_confidence 的边。

        Args:
            edges: discover_hidden_edges() 返回的边列表
            n_nodes: 节点总数
            min_confidence: 边置信度最低阈值

        Returns:
            CausalGraph
        """
        adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        edge_list: list[tuple[str, str]] = []

        for e in edges:
            conf = e.get("confidence", 0)
            if conf < min_confidence:
                continue
            cause_idx = e.get("cause_idx")
            effect_idx = e.get("effect_idx")
            if cause_idx is None or effect_idx is None:
                continue
            if not (0 <= cause_idx < n_nodes and 0 <= effect_idx < n_nodes):
                continue
            adj[cause_idx, effect_idx] = float(conf)
            edge_list.append((f"V{cause_idx}", f"V{effect_idx}"))

        return CausalGraph(
            nodes=[f"V{i}" for i in range(n_nodes)],
            edges=edge_list,
            adjacency=adj,
        )

    # -----------------------------------------------------------------
    # 工具方法
    # -----------------------------------------------------------------

    @staticmethod
    def _discretize(values: np.ndarray, n_bins: int = 10) -> np.ndarray:
        """等频分桶，返回桶边界。"""
        if len(values) < n_bins:
            return np.sort(np.unique(values))
        return np.percentile(values, np.linspace(0, 100, n_bins))

    def _topological_sort(self) -> list[int] | None:
        """
        因果图的拓扑排序 (Kahn 算法)。

        Returns:
            节点索引列表 (按拓扑序)，或 None (含环或图无效)

        注意:
            F4-P1-1 修复: 含环时显式返回 None，不再以 list(range(n)) 静默掩盖。
            下游调用方应检查 None 并返回 InterventionResult.empty()。
        """
        if self._graph is None or self._graph.adjacency is None:
            return None

        n = self._graph.n_nodes
        adj = self._graph.adjacency

        in_degree = np.zeros(n, dtype=int)
        for i in range(n):
            for j in range(n):
                if adj[i, j] > 0:
                    in_degree[j] += 1

        queue = deque([i for i in range(n) if in_degree[i] == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for j in range(n):
                if adj[node, j] > 0:
                    in_degree[j] -= 1
                    if in_degree[j] == 0:
                        queue.append(j)

        # F4-P1-1: 含环 → 返回 None (而非 list(range(n)) 静默错位)
        if len(result) < n:
            logger.warning(
                "因果图含环/不连通: 已处理节点 %d / 总节点 %d，返回 None (F4-P1-1)",
                len(result),
                n,
            )
            return None

        return result

    def _build_result(
        self,
        X: str,
        Y: str,
        x_value: float,
        x_baseline: float,
        ate: float,
        ci: tuple[float, float],
        adjustment_set: list[str],
        method: str,
        p_value: float,
        sample_size: int,
        note: str = "",
    ) -> InterventionResult:
        """构建标准化的 InterventionResult。"""
        direction = "neutral"
        if ate > 0.05:
            direction = "positive"
        elif ate < -0.05:
            direction = "negative"

        abs_ate = abs(ate)
        if abs_ate > 0.5:
            magnitude = "large"
        elif abs_ate > 0.2:
            magnitude = "medium"
        elif abs_ate > 0.05:
            magnitude = "small"
        else:
            magnitude = "negligible"

        return InterventionResult(
            intervention=f"do({X}={x_value})",
            target=Y,
            ate=float(ate),
            confidence_interval=(float(ci[0]), float(ci[1])),
            adjustment_set=adjustment_set,
            method=method,
            p_value=float(p_value),
            effect_direction=direction,
            effect_magnitude=magnitude,
            sample_size=sample_size,
            note=note,
        )

    # -----------------------------------------------------------------
    # 批量 API
    # -----------------------------------------------------------------

    def batch_estimate_ate(
        self,
        pairs: list[tuple[str, str]],
        x_value: float = 1.0,
        x_baseline: float = 0.0,
    ) -> list[InterventionResult]:
        """批量估计多对 (X, Y) 的 ATE。

        Args:
            pairs: [(X1, Y1), (X2, Y2), ...]
            x_value: do(X) 干预值
            x_baseline: 基线值

        Returns:
            [InterventionResult, ...] 与 pairs 顺序一致
        """
        return [self.estimate_ate(X, Y, x_value=x_value, x_baseline=x_baseline) for X, Y in pairs]

    def batch_identify_adjustment_sets(
        self,
        pairs: list[tuple[str, str]],
    ) -> dict[tuple[str, str], list[str] | None]:
        """批量识别调整变量集。

        Args:
            pairs: [(X1, Y1), (X2, Y2), ...]

        Returns:
            {(X, Y): [Z1, Z2, ...] or None} 映射
        """
        result: dict[tuple[str, str], list[str] | None] = {}
        for X, Y in pairs:
            adj = self.identify_adjustment_set(X, Y)
            # Try frontdoor if backdoor not available
            if adj is None:
                med = self.identify_frontdoor_mediators(X, Y)
                result[(X, Y)] = med  # mediators as pseudo-adjustment
            else:
                result[(X, Y)] = adj
        return result

    def batch_query(
        self,
        queries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """批量干预查询 (兼容高层 API)。

        Args:
            queries: [{"X": "x1", "Y": "y1", "x_value": 1.0}, ...]

        Returns:
            [{"ate": float, "method": str, ...}, ...]
        """
        results: list[dict[str, Any]] = []
        for q in queries:
            X = q["X"]
            Y = q["Y"]
            xv = q.get("x_value", 1.0)
            xb = q.get("x_baseline", 0.0)
            r = self.estimate_ate(X, Y, x_value=xv, x_baseline=xb)
            results.append(
                {
                    "X": X,
                    "Y": Y,
                    "ate": r.ate,
                    "method": r.method,
                    "adjustment_set": r.adjustment_set,
                    "ci_low": r.confidence_interval[0],
                    "ci_high": r.confidence_interval[1],
                    "p_value": r.p_value,
                }
            )
        return results

    def __repr__(self) -> str:
        g = "graph" if self._graph else "no_graph"
        d = f"{len(self._data)} vars" if self._data else "simulated"
        return f"DoCalculus({g}, data={d})"
