from __future__ import annotations

from typing import Any, ClassVar

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

## Formal Guarantees

    - 后门调整无偏性: 当调整集 Z 满足后门准则（阻断 X→Y 全部后门路径，且不与 X 的后代相交）时，
      P(Y | do(X=x)) = Σ_z P(Y | X=x, Z=z) · P(Z=z) 成立
    - 前门调整有效性: 当 M 满足前门准则（M 阻断 X→Y 全部中介路径，X→M 无后门路径）时，前门公式给出一致估计
    - ATE = E[Y|do(X=1)] − E[Y|do(X=0)] 在有效调整集下为一致估计
"""


import hashlib
import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class UnknownNodeError(ValueError):
    """结构化节点解析错误。"""

    query: str | int
    kind: str = "unknown_node"

    def __str__(self) -> str:
        return f"{self.kind}: {self.query!r}"


@dataclass
class ObservationDataset:
    """显式观测数据集，携带来源、标识和确定性哈希。"""

    values: dict[str, np.ndarray]
    dataset_id: str | None = None
    source: str = "observed"
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.source not in {"observed", "simulated"}:
            raise ValueError("数据来源必须是 observed 或 simulated")
        self.values = {str(name): np.asarray(values, dtype=np.float64) for name, values in self.values.items()}
        for name, values in self.values.items():
            if values.size == 0:
                raise ValueError(f"观测数据不能为空: {name}")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"观测数据包含非有限值: {name}")

    @property
    def is_observed(self) -> bool:
        return self.source == "observed"

    @property
    def dataset_hash(self) -> str:
        """返回数据内容、来源与标识的确定性哈希。"""
        digest = hashlib.sha256()
        digest.update(self.source.encode("utf-8"))
        digest.update(str(self.dataset_id or "").encode("utf-8"))
        for name in sorted(self.values):
            values = np.ascontiguousarray(self.values[name], dtype=np.float64)
            digest.update(name.encode("utf-8"))
            digest.update(str(values.shape).encode("utf-8"))
            digest.update(values.tobytes(order="C"))
        return digest.hexdigest()


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
    do_x: dict[str, float] = field(default_factory=dict)
    x_baseline: float = 0.0
    estimator: str = ""
    mode: str = "no_data"
    dataset_hash: str | None = None
    seed: int | None = None
    is_conclusive: bool = False

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
            "do_x": dict(self.do_x),
            "x_baseline": self.x_baseline,
            "estimator": self.estimator,
            "mode": self.mode,
            "dataset_hash": self.dataset_hash,
            "seed": self.seed,
            "is_conclusive": self.is_conclusive,
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
    node_aliases: dict[str, list[str]] = field(default_factory=dict)
    edge_mode: str = "causal"
    dataset_id: str | None = None

    CAUSAL_EDGE_MODES: ClassVar[frozenset[str]] = frozenset(
        {
            "orientation_by_intervention",
            "orientation_by_temporal_order",
            "orientation_by_expert_prior",
        }
    )

    def __post_init__(self) -> None:
        n = len(self.nodes)
        self._validate_aliases()
        if self.adjacency is None:
            self.adjacency = np.zeros((n, n), dtype=np.float32)
            node_idx = {name: i for i, name in enumerate(self.nodes)}
            for src, dst in self.edges:
                i = node_idx.get(src)
                j = node_idx.get(dst)
                if i is not None and j is not None:
                    self.adjacency[i, j] = 1.0

    def _validate_aliases(self) -> None:
        """确保别名到业务名节点的一一归属，禁止跨节点冲突。"""
        owner: dict[str, str] = {}
        for canonical, aliases in self.node_aliases.items():
            if canonical not in self.nodes:
                raise ValueError(f"别名映射的节点不存在: {canonical!r}")
            for alias in aliases:
                if alias in owner and owner[alias] != canonical:
                    raise ValueError(f"别名冲突: {alias!r}")
                owner[alias] = canonical

    @property
    def idx_to_name(self) -> dict[int, str]:
        """返回图内索引到业务名节点的只读映射。"""
        return dict(enumerate(self.nodes))

    @property
    def is_causal_graph(self) -> bool:
        """只有显式定向证据构成的图才能参与 do-operator 推理。"""
        return self.edge_mode == "causal"

    def resolve_node(self, node: str | int) -> str:
        """解析业务名、别名或索引到图内唯一节点。"""
        if isinstance(node, int) and not isinstance(node, bool):
            if not 0 <= node < len(self.nodes):
                raise UnknownNodeError(node)
            return self.nodes[node]
        if node in self.nodes:
            return node
        for canonical, aliases in self.node_aliases.items():
            if node in aliases:
                return canonical
        raise UnknownNodeError(node)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def node_index(self, name: str | int) -> int | None:
        """获取业务名、别名或索引对应的图内索引。"""
        try:
            return self.nodes.index(self.resolve_node(name))
        except UnknownNodeError:
            return None

    def has_edge(self, src: str | int, dst: str | int) -> bool:
        """检查是否存在有向边 src → dst。"""
        if self.adjacency is None:
            return False
        i = self.node_index(src)
        j = self.node_index(dst)
        if i is None or j is None:
            return False
        return self.adjacency[i, j] > 0

    def get_parents(self, node: str | int) -> list[str]:
        """获取节点的所有父节点 (指向 node 的节点)。"""
        idx = self.node_index(node)
        if idx is None or self.adjacency is None:
            return []
        parents = []
        for i in range(self.n_nodes):
            if self.adjacency[i, idx] > 0:
                parents.append(self.nodes[i])
        return parents

    def get_children(self, node: str | int) -> list[str]:
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

    def get_descendants(self, node: str | int) -> set[str]:
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

    def get_mediators(self, src: str | int, dst: str | int) -> list[str]:
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

    def to_dict(self) -> dict[str, Any]:
        """序列化图结构与名称映射。"""
        payload: dict[str, Any] = {
            "nodes": list(self.nodes),
            "node_aliases": {canonical: list(aliases) for canonical, aliases in self.node_aliases.items()},
            "edges": [list(edge) for edge in self.edges],
            "edge_mode": self.edge_mode,
            "dataset_id": self.dataset_id,
        }
        if self.adjacency is not None:
            payload["adjacency"] = np.asarray(self.adjacency).tolist()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CausalGraph:
        """反序列化图结构，确保名称映射不丢失。"""
        adjacency = payload.get("adjacency")
        return cls(
            nodes=[str(node) for node in payload.get("nodes", [])],
            edges=[(str(edge[0]), str(edge[1])) for edge in payload.get("edges", [])],
            adjacency=np.asarray(adjacency, dtype=np.float32) if adjacency is not None else None,
            node_aliases={
                str(canonical): [str(alias) for alias in aliases]
                for canonical, aliases in payload.get("node_aliases", {}).items()
            },
            edge_mode=str(payload.get("edge_mode", "causal")),
            dataset_id=payload.get("dataset_id"),
        )

    # -----------------------------------------------------------------
    # 静态工厂: 从 GaussianDAG 边列表构建
    # -----------------------------------------------------------------

    @staticmethod
    def build_from_gaussian_dag(
        edges: list[dict[str, Any]],
        n_nodes: int,
        min_confidence: float = 0.3,
        node_names: Sequence[str] | None = None,
        memories: Sequence[dict[str, Any]] | None = None,
        dataset_id: str | None = None,
    ) -> CausalGraph:
        """从 GaussianDAG.discover_hidden_edges() 输出构建候选因果图。"""
        if node_names is not None and memories is not None and len(node_names) != len(memories):
            raise ValueError("node_names 与 memories 数量不一致")
        if node_names is None:
            if memories is not None:
                node_names = [
                    str(memory.get("node_name") or memory.get("id") or memory.get("content", f"V{index}")).strip()
                    for index, memory in enumerate(memories)
                ]
            else:
                node_names = [f"V{index}" for index in range(n_nodes)]
        if len(node_names) != n_nodes:
            raise ValueError("node_names 数量必须等于 n_nodes")
        if any(not name for name in node_names):
            raise ValueError("node_names 不能包含空值")
        if len(set(node_names)) != len(node_names):
            raise ValueError("node_names 不能重复")

        adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        edge_list: list[tuple[str, str]] = []
        retained_edge_modes: set[str] = set()

        for edge in edges:
            confidence = float(edge.get("confidence", 0))
            if confidence < min_confidence:
                continue
            cause_idx = edge.get("cause_idx")
            effect_idx = edge.get("effect_idx")
            if cause_idx is None or effect_idx is None:
                continue
            if not (0 <= cause_idx < n_nodes and 0 <= effect_idx < n_nodes):
                continue
            edge_mode = str(edge.get("edge_mode", "correlation"))
            if edge_mode not in {"correlation", *CausalGraph.CAUSAL_EDGE_MODES}:
                raise ValueError(f"不支持的 edge_mode: {edge_mode!r}")
            adjacency[cause_idx, effect_idx] = confidence
            edge_list.append((node_names[cause_idx], node_names[effect_idx]))
            retained_edge_modes.add(edge_mode)

        node_aliases = {name: [f"V{index}"] for index, name in enumerate(node_names) if name != f"V{index}"}

        return CausalGraph(
            nodes=list(node_names),
            edges=edge_list,
            adjacency=adjacency,
            node_aliases=node_aliases,
            edge_mode=(
                "causal"
                if retained_edge_modes and retained_edge_modes <= CausalGraph.CAUSAL_EDGE_MODES
                else "correlation"
            ),
            dataset_id=dataset_id,
        )

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
        if not self.is_causal_graph:
            raise ValueError("关联图禁止转换为结构方程模型")
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
        dataset: ObservationDataset | None = None,
    ):
        """
        Args:
            graph: 因果图 (CausalGraph 或 None)
            data: 观测数据 {node_name: values_array} (可选)
            seed: 随机种子 (用于模拟数据生成)
            dataset: 显式观测数据集；传入时优先于 data
        """
        if dataset is not None and data:
            raise ValueError("dataset 与 data 不能同时传入")
        self._graph = graph
        self._dataset = dataset or ObservationDataset(values=data or {}, source="observed")
        self._data = self._dataset.values
        self._rng = np.random.RandomState(seed)
        self._seed = seed
        self._is_simulated: bool = self._dataset.source == "simulated"

    # -----------------------------------------------------------------
    # 图管理
    # -----------------------------------------------------------------

    def set_graph(self, graph: CausalGraph) -> None:
        """设置/更新因果图。"""
        self._graph = graph

    def set_data(self, data: dict[str, np.ndarray] | ObservationDataset, dataset_id: str | None = None) -> None:
        """设置观测数据。"""
        self._dataset = (
            data if isinstance(data, ObservationDataset) else ObservationDataset(values=data, dataset_id=dataset_id)
        )
        self._data = self._dataset.values
        self._is_simulated = not self._dataset.is_observed

    @property
    def dataset(self) -> ObservationDataset:
        """返回当前绑定的观测数据集。"""
        return self._dataset

    @property
    def data_mode(self) -> str:
        """返回 observed、simulated 或 no_data。"""
        if not self._dataset.values:
            return "no_data"
        return self._dataset.source

    def _resolve_node(self, node: str | int) -> str:
        """将业务名、别名或字符串索引解析到图内节点。"""
        if self._graph is None:
            return str(node)
        return self._graph.resolve_node(node)

    def _no_data_result(self, X: str | int, Y: str | int, x_value: float, x_baseline: float) -> InterventionResult:
        """构建显式 no_data 结果，禁止用图模拟伪装观测证据。"""
        return InterventionResult(
            intervention=f"do({X}={x_value})",
            target=str(Y),
            adjustment_set=[],
            method="no_data",
            do_x={str(X): x_value},
            x_baseline=x_baseline,
            estimator="none",
            mode="no_data",
            seed=self._seed,
            is_conclusive=False,
            note="no_data",
            sample_size=0,
        )

    def _unsupported_result(self, X: str, Y: str, x_value: float, x_baseline: float, reason: str) -> InterventionResult:
        """构建多变量或无效干预的显式拒绝结果。"""
        return InterventionResult(
            intervention=f"do({X}={x_value})",
            target=Y,
            adjustment_set=[],
            method="unsupported",
            do_x={X: x_value},
            x_baseline=x_baseline,
            estimator="none",
            mode=self.data_mode,
            dataset_hash=self._dataset.dataset_hash if self._dataset.values else None,
            seed=self._seed,
            is_conclusive=False,
            note=reason,
        )

    def _association_graph_result(
        self, X: str | int, Y: str | int, x_value: float, x_baseline: float
    ) -> InterventionResult:
        """关联图不构成 do-operator 证据，统一显式拒绝。"""
        return InterventionResult(
            intervention=f"do({X}={x_value})",
            target=str(Y),
            adjustment_set=[],
            method="rejected",
            do_x={str(X): x_value},
            x_baseline=x_baseline,
            estimator="none",
            mode=self.data_mode,
            dataset_hash=self._dataset.dataset_hash if self._dataset.values else None,
            seed=self._seed,
            is_conclusive=False,
            note="association_graph_not_causal",
        )

    def _require_causal_graph(self) -> bool:
        """返回 false 表示当前图是关联图，调用方必须停止因果估计。"""
        return self._graph is None or self._graph.is_causal_graph

    def estimate_intervention(
        self,
        do_x: dict[str, float],
        target: str,
        x_baseline: float = 0.0,
        method: str = "auto",
    ) -> InterventionResult:
        """按统一契约处理干预映射；多变量在闭式实现前显式拒绝。"""
        if not self._require_causal_graph():
            first_name = next(iter(do_x), "")
            first_value = float(next(iter(do_x.values()), 0.0))
            return self._association_graph_result(first_name, target, first_value, x_baseline)
        if not isinstance(do_x, dict) or not do_x:
            raise ValueError("do_x 必须是非空 dict")
        if not np.isfinite(x_baseline):
            raise ValueError("x_baseline 必须是有限值")
        try:
            resolved = {
                self._resolve_node(name): float(value)
                for name, value in sorted(do_x.items(), key=lambda item: str(self._resolve_node(item[0])))
            }
        except UnknownNodeError as error:
            first_name = next(iter(do_x), "")
            return self._unsupported_result(
                str(first_name),
                target,
                float(next(iter(do_x.values()), 0.0)),
                x_baseline,
                f"unknown_node: {error}",
            )
        if not np.all(np.isfinite(np.fromiter(resolved.values(), dtype=np.float64))):
            return self._unsupported_result(
                next(iter(resolved), ""), target, 0.0, x_baseline, "do_x value must be finite"
            )
        if len(resolved) != 1:
            return self._unsupported_result(
                next(iter(resolved)),
                target,
                next(iter(resolved.values())),
                x_baseline,
                "multivariate_do_not_implemented",
            )
        treatment, value = next(iter(resolved.items()))
        return self.estimate_ate(treatment, target, x_value=value, x_baseline=x_baseline, method=method)

    # -----------------------------------------------------------------
    # 后门准则 — 调整变量集识别
    # -----------------------------------------------------------------

    def identify_adjustment_set(self, X: str | int, Y: str | int) -> list[str] | None:
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
        if not self._graph.is_causal_graph:
            return None

        X = self._resolve_node(X)
        Y = self._resolve_node(Y)

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
        if not self._graph.is_causal_graph:
            return None

        X = self._resolve_node(X)
        Y = self._resolve_node(Y)

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
        Z_set: Sequence[str | int],
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

        """
        if not self._require_causal_graph():
            return self._association_graph_result(X, Y, x_value, x_baseline)

        X_name = self._resolve_node(X)
        Y_name = self._resolve_node(Y)
        Z_names = [self._resolve_node(z) for z in Z_set]

        if self.data_mode == "no_data":
            return self._no_data_result(X_name, Y_name, x_value, x_baseline)

        # ── 基于观测数据的后门调整 ──
        if X_name not in self._data or Y_name not in self._data:
            return self._no_data_result(X_name, Y_name, x_value, x_baseline)

        x_data = self._data[X_name]
        y_data = self._data[Y_name]
        n_samples = len(x_data)

        if n_samples < 5:
            return self._no_data_result(X_name, Y_name, x_value, x_baseline)

        # 后门调整公式: E[Y | do(X=x)] = Σ_z E[Y | X=x, Z=z] · P(Z=z)
        # 用 OLS 回归 Y ~ X + Z 估计条件期望 E[Y|X=x,Z=z], 再按 P(Z) 加权求和。
        # 这是线性/可加假设下的标准后门调整实现; 对连续和离散 X/Z 均适用。
        z_names = [z for z in Z_names if z in self._data]
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
            X=X_name,
            Y=Y_name,
            x_value=x_value,
            x_baseline=x_baseline,
            ate=ate,
            ci=(ci_lower, ci_upper),
            adjustment_set=Z_names,
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
        X: str | int,
        Y: str | int,
        M_set: Sequence[str | int],
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
        if self._graph is None:
            return self._no_data_result(X, Y, x_value, x_baseline)

        if not self._require_causal_graph():
            return self._association_graph_result(X, Y, x_value, x_baseline)

        X_name = self._resolve_node(X)
        Y_name = self._resolve_node(Y)
        M_names = [self._resolve_node(m) for m in M_set]

        if self.data_mode == "no_data":
            return self._no_data_result(X_name, Y_name, x_value, x_baseline)

        # ── 基于观测数据的前门调整 ──
        if X_name not in self._data or Y_name not in self._data:
            return self._no_data_result(X_name, Y_name, x_value, x_baseline)

        x_data = self._data[X_name]
        y_data = self._data[Y_name]
        n_samples = len(x_data)

        if n_samples < 10:
            return self._no_data_result(X_name, Y_name, x_value, x_baseline)

        y_do_total = 0.0
        n_valid_m = 0

        for m_name in M_names:
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
            X=X_name,
            Y=Y_name,
            x_value=x_value,
            x_baseline=x_baseline,
            ate=ate,
            ci=ci,
            adjustment_set=M_names,
            method="frontdoor",
            p_value=p_value,
            sample_size=n_samples,
        )

    def _frontdoor_simulated(
        self,
        X: str,
        Y: str,
        M_set: Sequence[str],
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
            adjustment_set=list(M_set),
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
        X: str | int,
        Y: str | int,
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
        if not self._require_causal_graph():
            return self._association_graph_result(X, Y, x_value, x_baseline)

        if self._graph is None:
            return self._no_data_result(X, Y, x_value, x_baseline)

        try:
            X_name = self._resolve_node(X)
            Y_name = self._resolve_node(Y)
        except UnknownNodeError as error:
            return InterventionResult(
                intervention=f"do({X}={x_value})",
                target=str(Y),
                method="rejected",
                note=f"unknown_node: {error}",
            )

        # F4-P1-3: NaN/Inf 边界守卫 — 拒绝非有限输入，保证 ATE 计算洁污
        if not np.isfinite(x_value):
            return InterventionResult(
                intervention=f"do({X_name}={x_value})",
                target=Y_name,
                method="rejected",
                note=f"x_value must be finite, got {x_value}",
            )
        if not np.isfinite(x_baseline):
            return InterventionResult(
                intervention=f"do({X_name}={x_value})",
                target=Y_name,
                method="rejected",
                note=f"x_baseline must be finite, got {x_baseline}",
            )

        # ── 方法选择 ──
        if method in ("auto", "backdoor"):
            adj_set = self.identify_adjustment_set(X_name, Y_name)
            if adj_set:
                return self.backdoor_adjustment(X_name, Y_name, adj_set, x_value, x_baseline)

        if method in ("auto", "frontdoor"):
            mediators = self.identify_frontdoor_mediators(X_name, Y_name)
            if mediators:
                return self.frontdoor_adjustment(X_name, Y_name, mediators, x_value, x_baseline)

        # ── 回退: 直接效应 (无调整) ──
        return self.direct_effect(X_name, Y_name, x_value, x_baseline)

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
        if not self._require_causal_graph():
            return self._association_graph_result(X, Y, x_value, x_baseline)

        X = self._resolve_node(X)
        Y = self._resolve_node(Y)

        if self.data_mode == "no_data":
            return self._no_data_result(X, Y, x_value, x_baseline)

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

        return self._no_data_result(X, Y, x_value, x_baseline)

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

    def simulate(
        self,
        n_samples: int = 500,
        seed: int | None = None,
        bind: bool = True,
    ) -> ObservationDataset:
        """显式生成研究型模拟数据；未调用本方法时 estimate_ate 返回 no_data。"""
        if self._graph is None or self._graph.adjacency is None:
            raise ValueError("模拟数据必须绑定有效因果图")
        if not self._graph.is_causal_graph:
            raise ValueError("关联图禁止用于模拟因果干预数据")
        topo_order = self._topological_sort()
        if topo_order is None:
            raise ValueError("模拟数据要求无环因果图")
        if n_samples <= 0:
            raise ValueError("n_samples 必须为正整数")

        effective_seed = self._seed if seed is None else seed
        rng = np.random.RandomState(effective_seed)
        n_nodes = self._graph.n_nodes
        values = np.zeros((n_samples, n_nodes), dtype=np.float64)
        for node_index in topo_order:
            parent_values = np.zeros(n_samples, dtype=np.float64)
            for parent_index in range(n_nodes):
                weight = self._graph.adjacency[parent_index, node_index]
                if weight > 0:
                    parent_values += float(weight) * values[:, parent_index]
            values[:, node_index] = parent_values + rng.randn(n_samples) * 0.5

        dataset = ObservationDataset(
            values={name: values[:, index] for index, name in enumerate(self._graph.nodes)},
            dataset_id=self._graph.dataset_id,
            source="simulated",
            seed=effective_seed,
        )
        if bind:
            self.set_data(dataset)
        return dataset

    # -----------------------------------------------------------------
    # 静态工厂: 从 GaussianDAG 边列表构建
    # -----------------------------------------------------------------

    @staticmethod
    def build_from_gaussian_dag(
        edges: list[dict[str, Any]],
        n_nodes: int,
        min_confidence: float = 0.3,
        node_names: Sequence[str] | None = None,
        memories: Sequence[dict[str, Any]] | None = None,
        dataset_id: str | None = None,
    ) -> CausalGraph:
        """兼容旧调用入口，实际构建逻辑归属 CausalGraph。"""
        return CausalGraph.build_from_gaussian_dag(
            edges=edges,
            n_nodes=n_nodes,
            min_confidence=min_confidence,
            node_names=node_names,
            memories=memories,
            dataset_id=dataset_id,
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
        if not np.isfinite(ate) or not np.isfinite(ci[0]) or not np.isfinite(ci[1]) or not np.isfinite(p_value):
            return InterventionResult(
                intervention=f"do({X}={x_value})",
                target=Y,
                adjustment_set=list(adjustment_set),
                method="rejected",
                do_x={X: float(x_value)},
                x_baseline=float(x_baseline),
                estimator=method,
                mode=self.data_mode,
                dataset_hash=self._dataset.dataset_hash if self._dataset.values else None,
                seed=self._seed,
                is_conclusive=False,
                note="non_finite_result",
            )

        mode = self.data_mode
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
            do_x={X: float(x_value)},
            x_baseline=float(x_baseline),
            estimator=method,
            mode=mode,
            dataset_hash=self._dataset.dataset_hash if self._dataset.values else None,
            seed=self._seed,
            is_conclusive=mode == "observed" and sample_size > 0,
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
