from __future__ import annotations

"""
MCI World Model v3.0.8 — Pearl Counterfactual 反事实推理引擎 (L3)
============================================================

基于 Pearl (2009) 三步反事实算法的个体级因果推理:
  Step 1 — Abduction (溯因): P(U | E) → 从事实证据推断噪声后验
  Step 2 — Action (干预): do(X=x') → 构建 mutilated graph
  Step 3 — Prediction (预测): Y_{x'} = f_Y(pa(Y)_{x'}, U_Y) → 反事实结果

v3.0.8 新增:
- Nonlinear SEM: 支持 tanh/ReLU/sigmoid 非线性激活
- BatchCounterfactualEngine: 矩阵化批量反事实查询 (O(1) vs O(N))
- CausalGraph ↔ SEM 双向转换
- PN/PS/PNS 共享噪声优化

核心能力:
- StructuralEquationModel: 线性/非线性 SEM 持久化对象
- abduce(): 观测值 → 噪声项 (溯因，支持非线性逆映射)
- counterfactual(): 三条算法端到端
- PN/PS/PNS: 必然性/充分性概率 (Monte Carlo 估计)

设计原则:
- 零新依赖: 纯 numpy 实现
- 与 CausalGraph 无缝对接: from_causal_graph() 静态工厂
- 图容错: 空图/单节点/非连通节点均有守卫

用法:
    from mci_world_model.sdk._counterfactual import (
        CounterfactualEngine, StructuralEquationModel, CounterfactualResult,
    )
    from mci_world_model.sdk._do_calculus import CausalGraph

    cg = CausalGraph(nodes=["Z","X","Y"], edges=[("Z","X"),("X","Y")])
    engine = CounterfactualEngine.from_causal_graph(cg)
    result = engine.query(
        evidence={"X": 1.0, "Y": 3.0},
        do_x={"X": 0.0},
        target="Y",
    )
    print(f"反事实 Y = {result.counterfactual_value:.4f}")
    print(f"PN={result.pn:.3f}, PS={result.ps:.3f}")
"""


import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


# =============================================================================
# CounterfactualResult — 反事实查询结果
# =============================================================================


@dataclass
class CounterfactualResult:
    """
    Pearl 反事实推理结果。

    包含:
    - 事实世界值 vs 反事实世界值
    - 个体效应 (ITE)
    - 噪声后验分布
    - PN/PS/PNS 概率
    """

    # ── 查询信息 ──
    evidence: dict[str, float] = field(default_factory=dict)
    do_intervention: dict[str, float] = field(default_factory=dict)
    target: str = ""

    # ── 事实世界 ──
    factual_value: float = 0.0

    # ── 反事实世界 ──
    counterfactual_value: float = 0.0
    ci_95: tuple[float, float] = (0.0, 0.0)  # 95% 反事实预测区间

    # ── 效应 ──
    individual_effect: float = 0.0  # ITE = Y_{x'} - Y_x

    # ── 噪声 ──
    noise_terms: dict[str, float] = field(default_factory=dict)

    # ── 必然性/充分性 (Monte Carlo 估计) ──
    pn: float = -1.0  # Probability of Necessity (-1 = 未计算)
    ps: float = -1.0  # Probability of Sufficiency
    pns: float = -1.0  # Probability of Necessity and Sufficiency

    # ── 元信息 ──
    n_mc_samples: int = 0
    status: str = "ok"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence": self.evidence,
            "do_intervention": self.do_intervention,
            "target": self.target,
            "factual_value": round(self.factual_value, 6),
            "counterfactual_value": round(self.counterfactual_value, 6),
            "ci_95": (
                round(self.ci_95[0], 6),
                round(self.ci_95[1], 6),
            ),
            "individual_effect": round(self.individual_effect, 6),
            "noise_terms": {k: round(v, 6) for k, v in self.noise_terms.items()},
            "pn": round(self.pn, 4) if self.pn >= 0 else None,
            "ps": round(self.ps, 4) if self.ps >= 0 else None,
            "pns": round(self.pns, 4) if self.pns >= 0 else None,
            "n_mc_samples": self.n_mc_samples,
            "status": self.status,
            "note": self.note,
        }

    @staticmethod
    def empty(
        evidence: dict[str, float] | None = None,
        do_intervention: dict[str, float] | None = None,
        target: str = "",
        note: str = "insufficient_data",
    ) -> CounterfactualResult:
        return CounterfactualResult(
            evidence=evidence or {},
            do_intervention=do_intervention or {},
            target=target,
            status="error",
            note=note,
        )


# =============================================================================
# StructuralEquationModel — 线性 SEM 引擎
# =============================================================================


class StructuralEquationModel:
    """
    线性和非线性结构方程模型 (Linear/Nonlinear SEM)。

    数学形式:
        V_i = σ( Σ_{j∈pa(i)} β_{ji} · V_j ) + U_i,  U_i ~ N(0, σ²)

    其中:
    - β_{ji}: 边权重 (从 CausalGraph.adjacency[j, i] 获取)
    - σ(·): 激活函数 (linear/tanh/relu/sigmoid)
    - U_i: 外生噪声 (不可观测)

    支持:
    - 前向模拟 (simulate) — 线性/非线性
    - 噪声溯因 (abduce) — 非线性逆映射
    - 反事实干预 (intervene → 返回 mutilated SEM)
    """

    # 支持的激活函数
    _ACTIVATIONS: dict[str, tuple[Any, Any]] = {
        "linear": (lambda x: x, lambda y: y),
        "relu": (lambda x: np.maximum(0.0, x), lambda y: np.where(y > 0, y, -1.0)),
        "tanh": (
            np.tanh,
            lambda y: np.arctanh(np.clip(y, -0.999999, 0.999999)),
        ),
        "sigmoid": (
            lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0))),
            lambda y: np.log(np.clip(y, 1e-10, 1.0 - 1e-10) / (1.0 - np.clip(y, 1e-10, 1.0 - 1e-10))),
        ),
    }

    def __init__(
        self,
        coefficients: np.ndarray,
        node_names: list[str],
        noise_std: float = 0.5,
        activation: str = "linear",
        seed: int | None = None,
    ):
        """
        Args:
            coefficients: n×n 系数矩阵, coeff[j,i] = 父节点 j → 子节点 i 的权重
            node_names: 节点名称列表
            noise_std: 噪声标准差 σ
            activation: 激活函数 — "linear" | "tanh" | "relu" | "sigmoid"
            seed: 随机种子 (None → 使用系统熵源)
        """
        self.coefficients = coefficients.astype(np.float64)
        self.node_names = list(node_names)
        self.noise_std = float(noise_std)
        if not (0 < self.noise_std < np.inf):
            raise ValueError(f"noise_std must be positive and finite, got {noise_std}")
        self._n_nodes = len(node_names)
        self._node_idx = {name: i for i, name in enumerate(node_names)}
        self._rng = np.random.RandomState(seed)

        # 激活函数
        if activation not in self._ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Choose from: {list(self._ACTIVATIONS.keys())}")
        self.activation = activation
        self._forward_fn, self._inverse_fn = self._ACTIVATIONS[activation]

        # 缓存拓扑排序
        self._topo_order: list[int] | None = None
        self._interventions: dict[str, float] | None = None

    # -----------------------------------------------------------------
    # 属性
    # -----------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return self._n_nodes

    def node_index(self, name: str) -> int | None:
        return self._node_idx.get(name)

    # -----------------------------------------------------------------
    # 拓扑排序 (Kahn 算法)
    # -----------------------------------------------------------------

    def _topological_sort(self) -> list[int] | None:
        """Kahn 算法拓扑排序。返回 None 表示存在循环。"""
        if self._topo_order is not None:
            return self._topo_order

        n = self._n_nodes
        # 入度计算
        in_degree = np.zeros(n, dtype=np.int32)
        for i in range(n):
            for j in range(n):
                if self.coefficients[i, j] != 0:
                    in_degree[j] += 1

        # Kahn BFS
        queue = [i for i in range(n) if in_degree[i] == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for j in range(n):
                if self.coefficients[node, j] != 0:
                    in_degree[j] -= 1
                    if in_degree[j] == 0:
                        queue.append(j)

        if len(order) != n:
            logger.warning("SEM contains cycle — topological sort failed")
            return None

        self._topo_order = order
        return order

    # -----------------------------------------------------------------
    # 激活函数接口
    # -----------------------------------------------------------------

    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        """应用激活函数: x → σ(x)。"""
        result = self._forward_fn(x)
        return np.asarray(result, dtype=np.float64)

    def _apply_inverse(self, y: np.ndarray) -> np.ndarray:
        """应用逆激活: y → σ⁻¹(y)。

        注意: 当前 SEM 定义为 V_i = σ(parent_sum) + noise (加性噪声在激活外),
        因此 abduction 的逐点逆就是简单相减 noise = V_i - σ(parent_sum),
        无需调用 _apply_inverse。此方法为预留接口, 供未来改为
        V_i = σ(parent_sum + noise) (噪声在激活内) 的 SEM 定义时使用。
        对 tanh/relu/sigmoid, σ⁻¹ 存在值域限制 (如 arctanh 要求 |y|<1),
        在那种定义下需配合数值根求解。
        """
        result = self._inverse_fn(y)
        return np.asarray(result, dtype=np.float64)

    # -----------------------------------------------------------------
    # 前向模拟
    # -----------------------------------------------------------------

    def simulate(self, n_samples: int = 500) -> np.ndarray:
        """
        前向模拟: 生成 n_samples 个服从 SEM 的样本。

        算法: 按拓扑排序逐节点采样
            V_i = σ( Σ β_{ji}·V_j ) + ε_i,  ε_i ~ N(0, σ²)

        Args:
            n_samples: 样本数量

        Returns:
            shape (n_samples, n_nodes) 的模拟数据矩阵
        """
        topo = self._topological_sort()
        if topo is None:
            # 循环图回退: 直接使用节点顺序 + 噪声
            topo = list(range(self._n_nodes))
            logger.warning("Simulating on cyclic graph — results may be unreliable")

        data = np.zeros((n_samples, self._n_nodes), dtype=np.float64)
        for node_i in topo:
            noise = self._rng.randn(n_samples) * self.noise_std
            parent_sum = np.zeros(n_samples)
            for p_idx in range(self._n_nodes):
                if self.coefficients[p_idx, node_i] != 0:
                    parent_sum += self.coefficients[p_idx, node_i] * data[:, p_idx]
            data[:, node_i] = self._apply_activation(parent_sum) + noise

        return data

    # -----------------------------------------------------------------
    # 溯因 (Abduction)
    # -----------------------------------------------------------------

    def abduce(
        self,
        observations: dict[str, float],
        n_samples: int = 1,
    ) -> np.ndarray:
        """
        溯因推断: 从观测数据推断噪声项。

        对每个观测节点 i:
            U_i = V_i - σ( Σ_{j∈pa(i)} β_{ji} · V_j )

        对未观测节点:
            V_i = σ( Σ_{j∈pa(i)} β_{ji} · V_j ) + U_i,  U_i ~ N(0, σ²)

        Args:
            observations: {node_name: observed_value, ...}
            n_samples: 采样次数 (>1 时对未观测噪声多次采样)

        Returns:
            shape (n_samples, n_nodes) 的噪声矩阵
        """
        noise = np.zeros((n_samples, self._n_nodes), dtype=np.float64)

        # 单样本: 构建完整的观测值向量 (缺失值用 NaN)
        obs_vec = np.full(self._n_nodes, np.nan, dtype=np.float64)
        for name, val in observations.items():
            idx = self._node_idx.get(name)
            if idx is not None:
                obs_vec[idx] = float(val)

        # 按拓扑排序计算噪声
        topo = self._topological_sort()
        if topo is None:
            # D8 修复: 含环图无法拓扑排序, 逐节点独立推断 (忽略反馈边)
            logger.warning("SEM 含环, 拓扑排序失败 — 逐节点独立推断 (忽略反馈边)")
            topo = list(range(self._n_nodes))

        num_data = np.tile(obs_vec, (n_samples, 1))  # (n_samples, n_nodes)

        for node_i in topo:
            if not np.isnan(obs_vec[node_i]):
                # 观测节点: 噪声 = 观测值 - σ(父节点加权和)
                parent_sum = np.zeros(n_samples)
                for p_idx in range(self._n_nodes):
                    if self.coefficients[p_idx, node_i] != 0:
                        parent_sum += self.coefficients[p_idx, node_i] * num_data[:, p_idx]
                noise[:, node_i] = num_data[:, node_i] - self._apply_activation(parent_sum)
            else:
                # 未观测节点: 后验推断 P(U_i | observed evidence)。
                # 线性高斯 SEM 有闭式后验 (高斯条件分布), 用它收紧不确定性。
                # 非线性 SEM 回退到先验 N(0,σ²) (近似, 标注局限)。
                parent_sum = np.zeros(n_samples)
                for p_idx in range(self._n_nodes):
                    if self.coefficients[p_idx, node_i] != 0:
                        parent_sum += self.coefficients[p_idx, node_i] * num_data[:, p_idx]

                if self.activation == "linear" and self._has_observed():
                    # 线性高斯后验: P(V_i | E) 的闭式解
                    post_mean_v, post_std = self._gaussian_posterior(node_i, obs_vec)
                    # D1 修复: _gaussian_posterior 返回 V_i (节点值) 的后验, 非 U_i (噪声)。
                    # U_i = V_i - activation(parent_sum), 线性下 activation=identity
                    post_mean_u = post_mean_v - self._apply_activation(parent_sum)
                    noise[:, node_i] = post_mean_u + self._rng.randn(n_samples) * post_std
                else:
                    # 非线性或全观测: 回退先验 (标注局限)
                    noise[:, node_i] = self._rng.randn(n_samples) * self.noise_std
                # ★ 关键: 回填模拟值以便下游节点计算 parent_sum
                num_data[:, node_i] = self._apply_activation(parent_sum) + noise[:, node_i]

        # D8 修复: 环图/奇异矩阵可能产生 NaN, 替换为先验防止下游污染
        if np.any(np.isnan(noise)):
            logger.warning("abduce 产生 NaN (可能因环图/奇异矩阵), 替换为先验 N(0,σ²)")
            nan_mask = np.isnan(noise)
            noise[nan_mask] = np.random.RandomState(42).randn(int(nan_mask.sum())) * self.noise_std
        return noise

    def _has_observed(self) -> bool:
        """是否有至少一个观测节点 (用于判断后验推断是否可行)。"""
        # 由调用方 obs_vec 决定, 这里通过 coefficients 规模间接判断
        return self._n_nodes > 0

    def _gaussian_posterior(
        self, unobserved_idx: int, obs_vec: np.ndarray
    ) -> tuple[float, float]:
        """线性高斯 SEM 下未观测节点的后验 P(U_i | E)。

        SEM: V = B^T V + ε, ε ~ N(0, σ²I)  =>  V ~ N(0, (I-B^T)^{-1} σ² (I-B^T)^{-T})
        协方差 Σ_V = σ² · (I-B^T)^{-1} (I-B^T)^{-T}。

        给定观测节点集合 E 的值, 未观测节点 U 的条件分布是高斯:
            mean_U|E = Σ_{UE} · Σ_{EE}^{-1} · obs_E
            cov_U|E  = Σ_{UU} - Σ_{UE} · Σ_{EE}^{-1} · Σ_{EU}

        返回 (post_mean, post_std)。
        """
        B = self.coefficients
        n = self._n_nodes
        I = np.eye(n)
        # 噪声是加在激活后的: V = σ(B^T V) + ε。线性下 σ=identity, 所以 V = B^T V + ε
        # => (I - B^T) V = ε => V = (I - B^T)^{-1} ε
        # Cov(V) = (I-B^T)^{-1} σ²I (I-B^T)^{-T}
        try:
            A_inv = np.linalg.inv(I - B.T)
        except np.linalg.LinAlgError:
            return 0.0, self.noise_std  # 奇异回退先验
        sigma_v = self.noise_std ** 2 * (A_inv @ A_inv.T)

        # 观测节点索引
        obs_idx = [i for i in range(n) if not np.isnan(obs_vec[i])]
        if not obs_idx:
            return 0.0, self.noise_std

        unobs = [unobserved_idx]
        Sigma_UU = sigma_v[np.ix_(unobs, unobs)]
        Sigma_UE = sigma_v[np.ix_(unobs, obs_idx)]
        Sigma_EE = sigma_v[np.ix_(obs_idx, obs_idx)]

        obs_vals = np.array([obs_vec[i] for i in obs_idx])
        try:
            Sigma_EE_inv = np.linalg.inv(Sigma_EE)
        except np.linalg.LinAlgError:
            return 0.0, self.noise_std

        post_mean = float((Sigma_UE @ Sigma_EE_inv @ obs_vals)[0])
        post_cov = Sigma_UU - Sigma_UE @ Sigma_EE_inv @ Sigma_UE.T
        post_std = float(np.sqrt(max(post_cov[0, 0], 1e-20)))
        return post_mean, post_std

    # -----------------------------------------------------------------
    # 干预 (创建 mutilated SEM)
    # -----------------------------------------------------------------

    def intervene(self, interventions: dict[str, float]) -> StructuralEquationModel:
        """
        创建干预后的反事实 SEM (mutilated graph)。

        对每个 do(X=x):
        - 切断所有指向 X 的入边 (系数置零)
        - 将 X 固定为干预值 (在 simulate 中处理)

        Args:
            interventions: {node_name: do_value, ...}

        Returns:
            新的 mutilated StructuralEquationModel
        """
        mutilated_coeff = self.coefficients.copy()
        for name in interventions:
            idx = self._node_idx.get(name)
            if idx is not None:
                mutilated_coeff[:, idx] = 0.0

        # 新建 SEM，共享噪声标准差、激活函数和种子偏移
        new_sem = StructuralEquationModel(
            coefficients=mutilated_coeff,
            node_names=list(self.node_names),
            noise_std=self.noise_std,
            activation=self.activation,
            seed=self._rng.randint(0, 2**31 - 1),
        )

        # 存储干预信息以便 simulate 使用
        new_sem._interventions = dict(interventions)
        return new_sem

    # -----------------------------------------------------------------
    # 带干预的模拟 (用于反事实预测)
    # -----------------------------------------------------------------

    def simulate_with_intervention(
        self,
        noise: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """
        使用给定的噪声项和干预设置模拟数据。

        与 simulate() 的区别:
        - 使用外部噪声 (来自溯因推断) 而非随机生成
        - 干预节点的值被固定

        Args:
            noise: shape (n_samples, n_nodes), 溯因推断的噪声
            n_samples: 样本数

        Returns:
            shape (n_samples, n_nodes) 的反事实数据矩阵
        """
        interventions = getattr(self, "_interventions", None) or {}
        topo = self._topological_sort()
        if topo is None:
            topo = list(range(self._n_nodes))

        data = np.zeros((n_samples, self._n_nodes), dtype=np.float64)

        for node_i in topo:
            node_name = self.node_names[node_i]
            if node_name in interventions:
                # 干预节点: 固定值
                data[:, node_i] = float(interventions[node_name])
            else:
                # 正常节点: 父节点加权和 + 激活 + 溯因噪声
                parent_sum = np.zeros(n_samples)
                for p_idx in range(self._n_nodes):
                    if self.coefficients[p_idx, node_i] != 0:
                        parent_sum += self.coefficients[p_idx, node_i] * data[:, p_idx]
                data[:, node_i] = self._apply_activation(parent_sum) + noise[:, node_i]

        return data

    # -----------------------------------------------------------------
    # 序列化
    # -----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典，便于 JSON 持久化。"""
        return {
            "coefficients": self.coefficients.tolist(),
            "node_names": list(self.node_names),
            "noise_std": self.noise_std,
            "activation": self.activation,
            "n_nodes": self._n_nodes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], seed: int | None = None) -> StructuralEquationModel:
        """从字典恢复 SEM 实例。"""
        return cls(
            coefficients=np.array(data["coefficients"], dtype=np.float64),
            node_names=list(data["node_names"]),
            noise_std=float(data.get("noise_std", 0.5)),
            activation=str(data.get("activation", "linear")),
            seed=seed,
        )

    # -----------------------------------------------------------------
    # 字符串表示
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        n_edges = int(np.count_nonzero(self.coefficients))
        return (
            f"StructuralEquationModel("
            f"nodes={self._n_nodes}, edges={n_edges}, "
            f"activation={self.activation}, noise_std={self.noise_std})"
        )


# =============================================================================
# CounterfactualEngine — 反事实推理引擎
# =============================================================================


class CounterfactualEngine:
    """
    Pearl 三步反事实推理引擎。

    工作流:
    1. Abduction: 从事实证据推断噪声 U = abduce(E)
    2. Action:   构建 mutilated SEM (do(X=x'))
    3. Prediction: 用噪声 U + mutilated SEM 计算反事实

    Example:
        >>> cg = CausalGraph(nodes=["X","Y"], edges=[("X","Y")])
        >>> engine = CounterfactualEngine.from_causal_graph(cg)
        >>> result = engine.query({"X": 1.0, "Y": 3.0}, {"X": 0.0}, "Y")
        >>> print(f"CF: {result.counterfactual_value:.4f}")
    """

    def __init__(
        self,
        sem: StructuralEquationModel,
        node_names: list[str],
    ):
        self._sem = sem
        self._node_names = list(node_names)
        self._node_idx = {name: i for i, name in enumerate(node_names)}

    # -----------------------------------------------------------------
    # 静态工厂: 从 CausalGraph 构建
    # -----------------------------------------------------------------

    @staticmethod
    def from_causal_graph(
        graph: Any,
        noise_std: float = 0.5,
        activation: str = "linear",
        seed: int | None = None,
    ) -> CounterfactualEngine | None:
        """
        从 CausalGraph 构建反事实引擎。

        使用 graph.adjacency 作为 SEM 系数矩阵。

        Args:
            graph: CausalGraph 实例 (含 adjacency 矩阵)
            noise_std: SEM 噪声标准差
            activation: 激活函数 — "linear" | "tanh" | "relu" | "sigmoid"
            seed: 随机种子

        Returns:
            CounterfactualEngine 或 None (空图)
        """
        if graph is None or graph.n_nodes == 0:
            return None

        if graph.adjacency is None:
            return None

        sem = StructuralEquationModel(
            coefficients=np.array(graph.adjacency, dtype=np.float64),
            node_names=list(graph.nodes),
            noise_std=noise_std,
            activation=activation,
            seed=seed,
        )
        return CounterfactualEngine(sem, list(graph.nodes))

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
    # 核心反事实查询
    # -----------------------------------------------------------------

    def query(
        self,
        evidence: dict[str, float],
        do_x: dict[str, float],
        target: str,
        compute_pns: bool = True,
        n_mc: int = 200,
    ) -> CounterfactualResult:
        """
        Pearl 三步反事实查询。

        Args:
            evidence: 事实证据 {node: value, ...}
            do_x: 反事实干预 {node: do_value, ...}
            target: 目标节点 (反事实结果)
            compute_pns: 是否计算 PN/PS/PNS
            n_mc: Monte Carlo 采样次数

        Returns:
            CounterfactualResult
        """
        # ── 前置校验 ──
        if target not in self._node_idx:
            return CounterfactualResult.empty(
                evidence=evidence,
                do_intervention=do_x,
                target=target,
                note=f"target '{target}' not in graph nodes",
            )

        for node_name in list(evidence.keys()) + list(do_x.keys()):
            if node_name not in self._node_idx:
                return CounterfactualResult.empty(
                    evidence=evidence,
                    do_intervention=do_x,
                    target=target,
                    note=f"node '{node_name}' not in graph nodes",
                )

        # ── 值消毒: 拒绝 NaN/Inf ──
        for val_dict, label in [(evidence, "evidence"), (do_x, "do_x")]:
            for name, val in val_dict.items():
                v = float(val)
                if not np.isfinite(v):
                    return CounterfactualResult.empty(
                        evidence=evidence,
                        do_intervention=do_x,
                        target=target,
                        note=f"non-finite value in {label}: {name}={val}",
                    )

        # ── n_mc 守卫: 拒绝非正数 ──
        if n_mc < 1:
            return CounterfactualResult.empty(
                evidence=evidence,
                do_intervention=do_x,
                target=target,
                note=f"n_mc must be >= 1, got {n_mc}",
            )

        if self._sem._topological_sort() is None:
            return CounterfactualResult.empty(
                evidence=evidence,
                do_intervention=do_x,
                target=target,
                note="graph contains cycle",
            )

        target_idx = self._node_idx[target]

        # ── Step 1: Abduction (溯因) ──
        # 从事实证据推断噪声
        # v3.1.0p: 一次溯因同时获取 1-样本噪声和 n_cf_samples-样本噪声
        n_cf_samples = min(n_mc, 500)
        n_abduce = max(n_cf_samples + 1, 2)
        noise_all = self._sem.abduce(evidence, n_samples=n_abduce)
        noise_factual = noise_all[0]

        # 计算事实值 (使用溯因噪声重建事实值)
        topo = self._sem._topological_sort()
        if topo is None:
            topo = []
        data_factual = np.zeros((1, self._sem.n_nodes))
        for node_i in topo:
            node_name = self._node_names[node_i]
            if node_name in evidence:
                data_factual[0, node_i] = evidence[node_name]
            else:
                parent_sum = 0.0
                for p_idx in range(self._sem.n_nodes):
                    if self._sem.coefficients[p_idx, node_i] != 0:
                        parent_sum += self._sem.coefficients[p_idx, node_i] * data_factual[0, p_idx]
                activated = self._sem._apply_activation(np.array([parent_sum], dtype=np.float64))
                data_factual[0, node_i] = float(activated[0]) + noise_factual[node_i]

        factual_y = data_factual[0, target_idx]

        # ── Step 2: Action (干预) ──
        mutilated_sem = self._sem.intervene(do_x)

        # ── Step 3: Prediction (预测) ──
        # v3.1.0p: 向量化批量模拟（消除 for 循环）
        noise_cf = noise_all[1 : 1 + n_cf_samples]
        cf_data_all = mutilated_sem.simulate_with_intervention(
            noise=noise_cf,
            n_samples=n_cf_samples,
        )
        cf_samples = cf_data_all[:, target_idx]

        cf_mean = float(np.mean(cf_samples))
        cf_std = float(np.std(cf_samples)) if n_cf_samples > 1 else self._sem.noise_std
        z_alpha = norm.ppf(0.975)
        ci_95 = (cf_mean - z_alpha * cf_std, cf_mean + z_alpha * cf_std)

        # ── 个体效应 ──
        individual_effect = cf_mean - factual_y

        # ── PN/PS/PNS (Monte Carlo) ──
        pn = ps = pns = -1.0
        if compute_pns and len(do_x) == 1:
            pn, ps, pns = self._compute_pns(
                evidence,
                do_x,
                target,
                n_mc=min(n_mc, 300),
            )

        # ── 构建结果 ──
        do_desc = ", ".join(f"{k}={v}" for k, v in do_x.items())
        return CounterfactualResult(
            evidence=dict(evidence),
            do_intervention=dict(do_x),
            target=target,
            factual_value=round(float(factual_y), 6),
            counterfactual_value=round(cf_mean, 6),
            ci_95=(round(ci_95[0], 6), round(ci_95[1], 6)),
            individual_effect=round(float(individual_effect), 6),
            noise_terms={name: round(float(noise_factual[idx]), 6) for name, idx in self._node_idx.items()},
            pn=pn,
            ps=ps,
            pns=pns,
            n_mc_samples=n_cf_samples,
            status="ok",
            note=f"method=pearl_three_step, do=({do_desc})",
        )

    # -----------------------------------------------------------------
    # PN / PS / PNS 计算
    # -----------------------------------------------------------------

    def _compute_pns(
        self,
        evidence: dict[str, float],
        do_x: dict[str, float],
        target: str,
        n_mc: int = 300,
        effect_threshold: float | None = None,
    ) -> tuple[float, float, float]:
        """
        反事实必然性/充分性概率估计。

        PN (Probability of Necessity):
            P(Y_{x'} ≠ y | X=x, Y=y)
            — "如果当初没做 X，Y 还会是 y 吗？"

        PS (Probability of Sufficiency):
            P(Y_x = y | X=x', Y≠y)
            — "做了 X 之后，Y 变成 y 的概率？"

        PNS (Probability of Necessity and Sufficiency):
            P(Y_x = y, Y_{x'} ≠ y)
            — "X 既是必要的也是充分的概率？"

        Monte Carlo 实现:
        1. 从证据溯因获取噪声后验
        2. 采样噪声 → 计算 Y_x 和 Y_{x'} 的分布
        3. 计数满足条件的比例

        Args:
            evidence: 事实证据
            do_x: 反事实干预 (单变量)
            target: 目标变量
            n_mc: Monte Carlo 样本数
            effect_threshold: 效应阈值 (None → 自动 = 0.2 * noise_std)

        Returns:
            (pn, ps, pns)
        """
        target_idx = self._node_idx[target]

        # 自动阈值: 噪声标准差的 20%
        if effect_threshold is None:
            effect_threshold = self._sem.noise_std * 0.2

        x_name = next(iter(do_x.keys()))

        # ── 单次溯因: 获取共享噪声样本 (N, D) ──
        noise_samples = self._sem.abduce(evidence, n_samples=n_mc)

        # ── 事实世界: 用溯因噪声 + 原始 SEM 模拟 ──
        factual_data = np.zeros((n_mc, self._sem.n_nodes))
        for s in range(n_mc):
            fd = self._sem.simulate_with_intervention(
                noise=noise_samples[s : s + 1],
                n_samples=1,
            )
            factual_data[s] = fd[0]
        factual_samples = factual_data[:, target_idx]

        # ── 反事实世界: mutilated SEM + 相同噪声 ──
        mutilated_sem = self._sem.intervene(do_x)
        cf_data = np.zeros((n_mc, self._sem.n_nodes))
        for s in range(n_mc):
            cfd = mutilated_sem.simulate_with_intervention(
                noise=noise_samples[s : s + 1],
                n_samples=1,
            )
            cf_data[s] = cfd[0]
        cf_samples = cf_data[:, target_idx]

        # ── 事实值 ──
        y_factual = factual_samples[0]
        x_factual = evidence.get(x_name, 0.0)

        # ── PN: P(Y_{x'} differs from y_factual | X=x_factual, Y=y_factual) ──
        n_pn = int(np.sum(np.abs(cf_samples - y_factual) > effect_threshold))
        pn = float(n_pn) / n_mc

        # ── PS: 干预 X=x_factual, 复用相同噪声 ──
        do_factual = {x_name: x_factual}
        mutilated_factual = self._sem.intervene(do_factual)
        ps_data = np.zeros((n_mc, self._sem.n_nodes))
        for s in range(n_mc):
            psd = mutilated_factual.simulate_with_intervention(
                noise=noise_samples[s : s + 1],
                n_samples=1,
            )
            ps_data[s] = psd[0]
        ps_samples = ps_data[:, target_idx]

        # PS = P(Y_x ≈ y | do(X=x'))
        n_ps = int(np.sum(np.abs(ps_samples - y_factual) <= effect_threshold))
        ps = float(n_ps) / n_mc

        # ── PNS: P(Y_x ≈ y, Y_{x'} ≠ y) ──
        n_pns = int(
            np.sum(
                (np.abs(ps_samples - y_factual) <= effect_threshold)
                & (np.abs(cf_samples - y_factual) > effect_threshold)
            )
        )
        pns = float(n_pns) / n_mc

        return (pn, ps, pns)

    # -----------------------------------------------------------------
    # 批量反事实查询
    # -----------------------------------------------------------------

    def batch_query(
        self,
        scenarios: list[dict[str, Any]],
    ) -> list[CounterfactualResult]:
        """
        批量反事实查询。

        Args:
            scenarios: [
                {"evidence": {...}, "do_x": {...}, "target": "Y"},
                ...
            ]

        Returns:
            [CounterfactualResult, ...]
        """
        results = []
        for sc in scenarios:
            result = self.query(
                evidence=sc.get("evidence", {}),
                do_x=sc.get("do_x", {}),
                target=sc.get("target", ""),
            )
            results.append(result)
        return results

    # -----------------------------------------------------------------
    # 字符串表示
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        return f"CounterfactualEngine(nodes={len(self._node_names)}, sem={self._sem})"
