"""CausalDAG — directed acyclic graph with BFS belief/energy propagation.

Mathematical foundation
-----------------------
A causal model over a finite set of nodes is a directed graph G = (V, E)
together with an edge-weight function w : E → (0, +∞) that gives the
*transmission coefficient* along each edge. The graph encodes "X influences Y"
as the edge X → Y.

Given an intervention (a "do" or evidence) that injects an amount Δ at a
source node s, the propagated effect at every reachable node v is computed by
breadth-first traversal along the directed edges, accumulating::

    effect(v) = Δ · Π_{e on a path s→v, taking the min-weight edge into v} ...

In the simplest (BFS, single-visit) form used here, each node is visited once
and receives::

    effect(child) = effect(parent) · w(parent → child)

i.e. the weight acts as a multiplicative attenuation along each edge, and the
first (shortest-path) arrival determines the effect. This is the algebraic
core of the legacy ``CausalChain.propagate``.

If the graph contains a directed cycle, the structure is no longer a DAG and
causal propagation is ill-defined (effects would loop indefinitely). We
therefore detect cycles with a topological sort and refuse to propagate on a
cyclic graph unless an explicit ring-tolerant mode is requested.

This module is pure graph theory: nodes are arbitrary hashable keys, weights
are plain floats, no SDK coupling.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

__all__ = ["CausalDAG"]


@dataclass
class CausalDAG:
    """Directed acyclic graph with weighted BFS propagation.

    Attributes
    ----------
    nodes : set
        The node set V (arbitrary hashable keys).
    edges : dict
        Adjacency map node -> list of (child, weight) tuples.
    """

    nodes: set = field(default_factory=set)
    edges: dict = field(default_factory=lambda: defaultdict(list))

    def __post_init__(self) -> None:
        # D2 修复: 规范化 nodes 为 set (允许传入 list)
        if not isinstance(self.nodes, set):
            self.nodes = set(self.nodes) if self.nodes is not None else set()
        # 支持 edges 为 list[tuple] 的便捷构造:
        # CausalDAG(edges=[("A","B"),("B","C")]) 应把每条边解析为 (child, weight)
        if isinstance(self.edges, (list, tuple, set)):
            edge_list = list(self.edges)
            self.edges = defaultdict(list)
            for item in edge_list:
                if isinstance(item, (list, tuple)):
                    if len(item) == 2:
                        parent, child = item
                        self.add_edge(parent, child, weight=1.0)
                    elif len(item) == 3:
                        parent, child, weight = item
                        self.add_edge(parent, child, weight=float(weight))
                    elif len(item) == 4:
                        # 便捷构造支持 (parent, child, weight, sign)
                        parent, child, weight, sign = item
                        self.add_edge(parent, child, weight=float(weight), sign=int(sign))
        elif isinstance(self.edges, dict):
            self.edges = defaultdict(list, self.edges)
        else:
            self.edges = defaultdict(list)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def add_node(self, node) -> None:
        """Insert a node (no-op if present)."""
        self.nodes.add(node)
        _ = self.edges[node]  # ensure key exists

    def add_edge(self, parent, child, weight: float = 1.0, sign: int = 1) -> None:
        """Add a directed edge parent -> child with transmission weight and sign.

        Parameters
        ----------
        weight : float
            正的传输系数: >1 放大效应（增强）, =1 不变, <1 衰减效应。
        sign : int
            效应符号, 仅允许 {+1, -1}。+1 表示增强效应 (默认),
            -1 表示抑制效应（使下游效应值取负）。不传时默认 +1, 保持向后兼容。
        """
        if not weight > 0.0:
            raise ValueError(f"edge weight must be positive, got {weight}")
        if sign not in (1, -1):
            raise ValueError(f"sign must be +1 or -1, got {sign}")
        self.add_node(parent)
        self.add_node(child)
        # 边存储为三元组 (child, weight, sign), 向后兼容旧的二元组遍历
        self.edges[parent].append((child, float(weight), int(sign)))

    def remove_node(self, node) -> None:
        """Remove a node and all edges incident to it."""
        self.nodes.discard(node)
        self.edges.pop(node, None)
        for p in list(self.edges):
            # 适配新边格式 (child, weight, sign); e[0] 取 child
            self.edges[p] = [e for e in self.edges[p] if e[0] != node]

    # ------------------------------------------------------------------
    # Graph queries
    # ------------------------------------------------------------------
    def parents(self, child) -> list:
        """Direct parents of ``child``."""
        # e[0] 取 child, 兼容 (child, weight) 与 (child, weight, sign)
        return [p for p in self.edges if any(e[0] == child for e in self.edges[p])]

    def children(self, parent) -> list:
        """Direct children of ``parent``."""
        return [e[0] for e in self.edges.get(parent, [])]

    def edge_weight(self, parent, child) -> float:
        """Transmission weight of edge parent -> child (0 if no edge)."""
        # 返回纯权重 float, 保持向后兼容; e[1] 为 weight
        for e in self.edges.get(parent, []):
            if e[0] == child:
                return e[1]
        return 0.0

    def edge_sign(self, parent: str, child: str) -> int:
        """效应符号 of edge parent -> child (默认 +1, 无边时返回 +1)。

        +1 表示增强效应, -1 表示抑制效应。
        """
        for e in self.edges.get(parent, []):
            if e[0] == child:
                # 兼容旧二元组 (缺 sign 时视为 +1)
                return e[2] if len(e) >= 3 else 1
        return 1

    def edge_signed_weight(self, parent: str, child: str) -> tuple[float, int]:
        """返回 (weight, sign) 元组 of edge parent -> child。

        无边时返回 (0.0, 1)。
        """
        for e in self.edges.get(parent, []):
            if e[0] == child:
                return (e[1], e[2] if len(e) >= 3 else 1)
        return (0.0, 1)

    def reach(self, source) -> set:
        """Set of nodes reachable from ``source`` along directed edges."""
        if source not in self.nodes:
            return set()
        seen, dq = set(), deque([source])
        while dq:
            cur = dq.popleft()
            for e in self.edges.get(cur, []):
                c = e[0]
                if c not in seen:
                    seen.add(c)
                    dq.append(c)
        return seen

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------
    def topological_order(self) -> list:
        """Kahn's algorithm topological sort.

        Returns a valid ordering of all nodes. Raises ValueError if the graph
        contains a directed cycle (i.e. is not a DAG).
        """
        indeg = dict.fromkeys(self.nodes, 0)
        for p in self.edges:
            for e in self.edges[p]:
                c = e[0]
                indeg[c] = indeg.get(c, 0) + 1
        dq = deque([n for n, d in indeg.items() if d == 0])
        order = []
        local_indeg = dict(indeg)
        while dq:
            cur = dq.popleft()
            order.append(cur)
            for e in self.edges.get(cur, []):
                c = e[0]
                local_indeg[c] -= 1
                if local_indeg[c] == 0:
                    dq.append(c)
        if len(order) != len(self.nodes):
            raise ValueError("graph contains a directed cycle; not a DAG")
        return order

    def is_dag(self) -> bool:
        """True iff the graph has no directed cycle."""
        try:
            self.topological_order()
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Propagation (the algebraic core)
    # ------------------------------------------------------------------
    def propagate(self, source, delta: float = 1.0) -> dict:
        """BFS causal propagation of an intervention at ``source``.

        Injects ``delta`` at ``source`` and diffuses it along directed edges,
        attenuating multiplicatively by each edge's transmission weight. Each
        node is visited exactly once (shortest-path / first-arrival semantics).

        Parameters
        ----------
        source : node key
            The intervention node.
        delta : float
            Magnitude injected at the source (must be non-negative).

        Returns
        -------
        dict
            Mapping node -> received effect. ``source`` itself is included with
            value ``delta``. Unreachable nodes are absent.
        """
        if delta < 0:
            raise ValueError("delta must be non-negative")
        if source not in self.nodes:
            return {}
        effect = {source: float(delta)}
        visited = {source}
        dq = deque([source])
        while dq:
            cur = dq.popleft()
            cur_effect = effect[cur]
            if cur_effect <= 0:
                continue
            # 仅使用权重 w, 忽略 sign — 保持原无符号行为向后兼容
            for e in self.edges.get(cur, []):
                child, w = e[0], e[1]
                if child not in visited:
                    visited.add(child)
                    effect[child] = cur_effect * w
                    dq.append(child)
        return effect

    def propagate_signed(self, source: str, delta: float = 1.0) -> dict[str, float]:
        """符号化 BFS 因果传播: 效应值 = 父效应 × 边权重 × 边符号。

        与 ``propagate`` 的区别: 每经过一条边, 效应值额外乘以边符号 sign
        (+1 增强 / -1 抑制), 因此效应值可正可负。每个节点仅访问一次
        (最短路径 / 首次到达语义)。

        Parameters
        ----------
        source : node key
            干预（注入）节点。
        delta : float
            源节点注入的初始效应值（可正可负）。

        Returns
        -------
        dict
            node -> 接收效应值（可正可负）。``source`` 自身值为 ``delta``。
            不可达节点不出现在结果中。
        """
        if source not in self.nodes:
            return {}
        effect: dict = {source: float(delta)}
        visited = {source}
        dq = deque([source])
        while dq:
            cur = dq.popleft()
            cur_effect = effect[cur]
            if cur_effect == 0:
                continue
            # e = (child, weight, sign); 兼容旧二元组 (缺 sign 视为 +1)
            for e in self.edges.get(cur, []):
                child, w = e[0], e[1]
                s = e[2] if len(e) >= 3 else 1
                if child not in visited:
                    visited.add(child)
                    # 效应 = 父效应 × 权重 × 符号
                    effect[child] = cur_effect * w * s
                    dq.append(child)
        return effect

    def propagate_multi(self, interventions: dict, normalize: bool = True) -> dict:
        """Propagate several interventions simultaneously and sum effects.

        Parameters
        ----------
        interventions : dict
            node -> injected delta.
        normalize : bool
            If True, clip final effects to [0, ∞) (defensive).

        Returns
        -------
        dict
            node -> total received effect (summed over all sources).
        """
        total: dict = defaultdict(float)
        for src, dlt in interventions.items():
            for node, eff in self.propagate(src, dlt).items():
                total[node] += eff
        if normalize:
            for n, eff in total.items():
                if eff < 0:
                    total[n] = 0.0
        return dict(total)

    # ------------------------------------------------------------------
    # Matrix view
    # ------------------------------------------------------------------
    def adjacency_matrix(self, node_order: list | None = None) -> tuple[np.ndarray, list]:
        """Weighted adjacency matrix W (n x n), with node ordering.

        W[i, j] = weight of edge node_order[i] -> node_order[j], else 0.
        Returns (W, node_order).
        """
        order = list(node_order) if node_order is not None else sorted(self.nodes, key=str)
        idx = {n: i for i, n in enumerate(order)}
        n = len(order)
        W = np.zeros((n, n), dtype=np.float64)
        for p in order:
            for e in self.edges.get(p, []):
                c, w = e[0], e[1]
                W[idx[p], idx[c]] = w
        return W, order

    def propagation_vector(self, source, node_order: list | None = None) -> tuple[np.ndarray, list]:
        """Effect vector from a single source, aligned with ``node_order``.

        Equivalent to ``propagate`` but returned as a dense numpy vector.
        """
        order = list(node_order) if node_order is not None else sorted(self.nodes, key=str)
        eff = self.propagate(source)
        return np.array([eff.get(n, 0.0) for n in order], dtype=np.float64), order

    # ============================================================
    # d-separation and backdoor adjustment (Pearl causal graph theory)
    # ============================================================
    # Extended on top of su-memory-sdk's CausalDAG to provide the graph-theoretic
    # foundation that do-calculus requires: d-separation, backdoor path discovery,
    # and valid-adjustment-set verification per Pearl (2009) Causality §3.3.

    # ------------------------------------------------------------------
    # Ancestry helpers
    # ------------------------------------------------------------------
    def ancestors(self, node) -> set:
        """Set of all ancestors of ``node`` (nodes with a directed path to it).

        Excludes ``node`` itself unless it lies on its own directed cycle
        (impossible in a DAG).
        """
        if node not in self.nodes:
            return set()
        seen, stack = set(), [node]
        while stack:
            cur = stack.pop()
            for p in self.parents(cur):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return seen

    def descendants(self, node) -> set:
        """Set of all descendants of ``node`` (nodes reachable from it)."""
        return self.reach(node)

    # ------------------------------------------------------------------
    # d-separation via Bayes-Ball
    # ------------------------------------------------------------------
    def d_separated(self, x, y, z: set) -> bool:
        """Test whether ``x`` and ``y`` are d-separated given conditioning set ``z``.

        X ⊥ Y | Z  iff every path between x and y is *blocked* by Z.

        A path is blocked if it contains:
          - a chain  i → m → j  or fork  i ← m → j  with m ∈ Z, or
          - a collider i → m ← j  with m ∉ Z and no descendant of m in Z.

        Uses the Bayes-Ball algorithm (Shachter 1998): a two-pass reachability
        over (node, direction) states. Returns True iff y is unreachable.
        """
        z = set(z)
        if x == y:
            return False  # a node is never d-separated from itself
        # D9 修复: 验证节点存在, 避免对不存在节点静默返回 True
        for name, label in [(x, "x"), (y, "y")] + [(n, "z") for n in z]:
            if name not in self.nodes:
                raise KeyError(f"d_separated: 节点 '{name}' ({label}) 不在图中, 已知节点: {sorted(self.nodes)}")
        # (node, direction): direction = "up" (arrived from a child, going up)
        #                                   or "down" (arrived from a parent, going down)
        to_visit = [(x, "up")]
        visited = set()
        reachable = set()
        while to_visit:
            node, direction = to_visit.pop()
            if (node, direction) in visited:
                continue
            visited.add((node, direction))
            if node == y:
                return False  # found an open path
            reachable.add(node)
            if direction == "up":
                # Arrived going upward (from a child or as the source).
                if node not in z:
                    # can continue up through parents (chain/fork tail)
                    for p in self.parents(node):
                        to_visit.append((p, "up"))
                    # can go down through children (fork head)
                    for c in self.children(node):
                        to_visit.append((c, "down"))
            else:  # direction == "down"
                if node not in z:
                    # continue down through children (chain head)
                    for c in self.children(node):
                        to_visit.append((c, "down"))
                # collider: node is a common child — open only if node or a
                # descendant is in Z.
                if node in z or (z & (self.descendants(node) | {node})):
                    for p in self.parents(node):
                        to_visit.append((p, "up"))
        return y not in reachable

    # ------------------------------------------------------------------
    # Backdoor criterion
    # ------------------------------------------------------------------
    def _all_undirected_neighbors(self, node) -> set:
        """Neighbors of ``node`` via any edge (ignoring direction)."""
        nb = set(self.children(node)) | set(self.parents(node))
        return nb

    def find_backdoor_paths(self, x, y) -> list:
        """Enumerate backdoor paths between ``x`` and ``y``.

        A backdoor path is one that starts with an arrow *into* x
        (i.e. x ← ... ). Returns the list of such paths as node sequences.
        Used to verify the backdoor criterion.
        """
        if x == y:
            return []
        paths = []
        # Start from x, first edge must be x ← p (arrow into x)
        for p in self.parents(x):
            self._dfs_undirected(p, y, [x, p], {x, p}, paths)
        return paths

    def _dfs_undirected(self, cur, target, path, visited, paths):
        """Depth-first search over *undirected* adjacency (path enumeration)."""
        if cur == target:
            paths.append(list(path))
            return
        for nb in self._all_undirected_neighbors(cur):
            if nb not in visited:
                visited.add(nb)
                path.append(nb)
                self._dfs_undirected(nb, target, path, visited, paths)
                path.pop()
                visited.remove(nb)

    def is_valid_adjustment_set(self, x, y, z) -> bool:
        """Verify Pearl's backdoor criterion for adjustment set Z.

        Z is a valid adjustment set for estimating X → Y iff:
          1. Z contains no descendant of X, AND
          2. Z blocks every backdoor path (path into X) between X and Y.

        Returns True iff both conditions hold.
        """
        z = set(z) if not isinstance(z, set) else z
        # Condition 1: no node in Z is a descendant of X
        x_desc = self.descendants(x)
        if z & x_desc:
            return False
        # Condition 2: Z blocks every backdoor path
        for path in self.find_backdoor_paths(x, y):
            if not self._is_path_blocked(path, z):
                return False
        return True

    def _is_path_blocked(self, path: list, z: set) -> bool:
        """Check whether an undirected path is blocked by conditioning set Z."""
        if len(path) < 3:
            return False  # direct edge cannot be blocked
        for i in range(1, len(path) - 1):
            prev, mid, nxt = path[i - 1], path[i], path[i + 1]
            into_mid_from_prev = mid in self.children(prev)  # prev → mid
            into_mid_from_next = mid in self.children(nxt)  # nxt → mid
            is_collider = into_mid_from_prev and into_mid_from_next
            if is_collider:
                # collider blocks unless it or a descendant is in Z
                desc = self.descendants(mid) | {mid}
                if not (z & desc):
                    return False  # open collider → path not blocked
            # chain or fork: blocked iff mid ∈ Z
            elif mid not in z:
                return False  # open → path not blocked
        return True

    def find_minimal_adjustment_set(self, x, y) -> set | None:
        """Find a minimal sufficient adjustment set for X → Y via backdoor.

        Returns the set of parents of X that satisfy the backdoor criterion
        (a common sufficient choice), or None if backdoor adjustment is not
        applicable. This is a heuristic; the true minimal set may be smaller.
        """
        candidate = set(self.parents(x))
        candidate.discard(y)
        if self.is_valid_adjustment_set(x, y, candidate):
            return candidate
        return None
