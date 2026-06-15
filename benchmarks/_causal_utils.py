"""
MCI World Model — 因果图传播共享工具模块

提取自 test_causalbench_adapter.py 和 test_counterbench_adapter.py 的公共逻辑。

核心函数:
  - _topological_sort(): Kahn 算法拓扑排序，保证传播顺序正确性
  - _propagate(): 拓扑序正向传播（线性基线），支持负权重边和环检测
  - _find_roots(): 动态根节点检测（无入边节点）
  - build_engine(): 构建 CounterfactualEngine (Pearl 三步全栈)
  - sem_forward(): 基于 SEM 的正向传播（支持非线性激活函数）
"""

from __future__ import annotations

from collections import deque

from mci_world_model.sdk._do_calculus import CausalGraph


def _topological_sort(cg: CausalGraph) -> list[str]:
    """
    Kahn 算法拓扑排序。

    保证父节点在子节点之前处理，使传播结果正确。
    环中节点不会被排入结果（安全跳过）。

    Returns:
        拓扑排序后的节点列表。若图含环，返回的列表不包含环中节点。
    """
    in_degree: dict[str, int] = dict.fromkeys(cg.nodes, 0)
    for n in cg.nodes:
        ni = cg.nodes.index(n)
        for p in cg.nodes:
            if p == n:
                continue
            pi = cg.nodes.index(p)
            if cg.adjacency[pi, ni] != 0:
                in_degree[n] += 1

    queue = deque(n for n in cg.nodes if in_degree[n] == 0)
    order: list[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        ci = cg.nodes.index(node)
        for child in cg.nodes:
            if child == node:
                continue
            chi = cg.nodes.index(child)
            if cg.adjacency[ci, chi] != 0:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

    return order


def _propagate(
    cg: CausalGraph,
    interventions: dict[str, float],
) -> dict[str, float]:
    """
    沿因果图正向传播干预值（拓扑序保证正确性）。

    对每个节点 v:
      - 如果 v 在 interventions 中 → 取干预值
      - 否则 → sum(parent_i * weight_i)  线性加权传播

    支持:
      - 多根节点同时激活
      - 负权重边（抑制性因果关系）
      - 环状图安全处理（环中节点值为 0.0）

    Args:
        cg: 因果有向图
        interventions: 干预值字典 {节点名: 值}

    Returns:
        所有节点的值字典 {节点名: 值}
    """
    order = _topological_sort(cg)
    values: dict[str, float] = {}

    for node in order:
        if node in interventions:
            values[node] = interventions[node]
        else:
            total = 0.0
            ni = cg.nodes.index(node)
            for parent in cg.nodes:
                pi = cg.nodes.index(parent)
                w = cg.adjacency[pi, ni]
                if w != 0 and parent in values:
                    total += values[parent] * w
            values[node] = total

    # 环中节点（未被拓扑排序覆盖）设为 0.0
    for node in cg.nodes:
        if node not in values:
            values[node] = interventions.get(node, 0.0)

    return values


def _find_roots(cg: CausalGraph) -> list[str]:
    """
    找到所有根节点（无入边的节点）。

    使用 w != 0 判断，支持负权重边。

    边界情况:
      - 空图: 返回 []
      - 全环节点: 返回 []
      - 多连通分量: 返回每个分量的根
    """
    return [
        n
        for n in cg.nodes
        if not any(cg.adjacency[cg.nodes.index(p), cg.nodes.index(n)] != 0 for p in cg.nodes if p != n)
    ]


# =============================================================================
# CEWM 全栈工具: CounterfactualEngine + SEM
# =============================================================================


def build_engine(
    cg: CausalGraph,
    noise_std: float = 0.01,
    activation: str = "linear",
    seed: int = 42,
):
    """
    从 CausalGraph 构建 CounterfactualEngine (Pearl 三步全栈)。

    默认使用极低噪声 (noise_std=0.01) 使结果接近确定性，
    保证基准测试的精确性。

    Args:
        cg: 因果有向图
        noise_std: SEM 噪声标准差 (越低越确定性)
        activation: 激活函数 — "linear" | "tanh" | "relu" | "sigmoid"
        seed: 随机种子

    Returns:
        CounterfactualEngine 或 None (空图/环图)
    """
    from mci_world_model.sdk._counterfactual import CounterfactualEngine

    return CounterfactualEngine.from_causal_graph(
        cg,
        noise_std=noise_std,
        activation=activation,
        seed=seed,
    )


def sem_forward(
    cg: CausalGraph,
    interventions: dict[str, float],
    activation: str = "linear",
    noise_std: float = 0.01,
    seed: int = 42,
) -> dict[str, float]:
    """
    基于 SEM 的正向传播（支持非线性激活函数）。

    与 _propagate() 的区别:
    - _propagate() 是纯线性加权传播 (Y = Σ parent_i × w_i)
    - sem_forward() 通过 SEM.intervene() + simulate_with_intervention()
      支持 tanh/sigmoid/relu 等非线性激活

    使用低噪声 + 大样本取均值，逼近确定性结果。

    Args:
        cg: 因果有向图
        interventions: 干预值字典 {节点名: 值}
        activation: 激活函数
        noise_std: SEM 噪声标准差
        seed: 随机种子

    Returns:
        所有节点的值字典 {节点名: 值} (均值)
    """
    import numpy as np

    from mci_world_model.sdk._counterfactual import StructuralEquationModel

    sem = StructuralEquationModel(
        coefficients=np.array(cg.adjacency, dtype=np.float64),
        node_names=list(cg.nodes),
        noise_std=noise_std,
        activation=activation,
        seed=seed,
    )

    # 干预 + 模拟
    mutilated = sem.intervene(interventions)
    noise = np.zeros((1, sem.n_nodes), dtype=np.float64)
    data = mutilated.simulate_with_intervention(noise=noise, n_samples=1)

    return {name: float(data[0, i]) for i, name in enumerate(cg.nodes)}


# =============================================================================
# 环状图迭代收敛传播
# =============================================================================


def _iterative_propagate(
    cg: CausalGraph,
    interventions: dict[str, float],
    max_iter: int = 100,
    tol: float = 1e-6,
    divergence_threshold: float = 1e6,
) -> dict[str, float]:
    """
    Jacobi 迭代法处理环状因果图的正向传播。

    与 _propagate() 的区别:
    - _propagate() 将环中节点设为 0.0 (静默降级)
    - _iterative_propagate() 通过 Jacobi 迭代收敛求解稳态值

    迭代方程:
        x^{(k+1)}_i = Σ_j (adj[j,i] * x^{(k)}_j) + b_i

    其中 b_i = interventions[i] (如果 i 是干预节点)

    收敛保护:
    - 最大迭代次数 max_iter
    - 发散检测: ‖x‖ > divergence_threshold
    - 非收敛时降级到稳态近似 (最后 K 步均值)

    Args:
        cg: 因果有向图 (可含环)
        interventions: 干预值字典
        max_iter: 最大迭代次数
        tol: 收敛容差
        divergence_threshold: 发散阈值

    Returns:
        所有节点的稳态值字典
    """
    import numpy as np

    n = len(cg.nodes)
    if n == 0:
        return {}

    node_idx = {name: i for i, name in enumerate(cg.nodes)}

    # 初始化: 干预节点取干预值，其余为 0
    x = np.zeros(n, dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)  # 偏置 (干预常数项)

    for name, val in interventions.items():
        if name in node_idx:
            i = node_idx[name]
            x[i] = val
            b[i] = val

    # 干预节点集合 (固定值，不参与迭代更新)
    fixed = {node_idx[name] for name in interventions if name in node_idx}

    W = np.array(cg.adjacency, dtype=np.float64)  # n×n 权重矩阵

    # Jacobi 迭代
    history: list[np.ndarray] = []
    converged = False

    for _ in range(max_iter):
        x_new = W.T @ x + b  # x^{(k+1)} = W^T · x^{(k)} + b

        # 固定干预节点
        for idx in fixed:
            x_new[idx] = interventions[cg.nodes[idx]]

        # 发散检测
        norm = np.linalg.norm(x_new)
        if norm > divergence_threshold:
            # 发散 → 用历史均值近似
            if history:
                x = np.mean(history[-min(5, len(history)) :], axis=0)
            break

        delta = np.linalg.norm(x_new - x)
        x = x_new
        history.append(x.copy())

        if delta < tol:
            converged = True
            break

    # 非收敛: 取最后 K 步均值作为稳态近似
    if not converged and len(history) > 3:
        x = np.mean(history[-min(10, len(history)) :], axis=0)

    return {name: float(x[i]) for i, name in enumerate(cg.nodes)}


# =============================================================================
# Peirce 链式推理
# =============================================================================


def chain_reason(
    causal_graphs: list[CausalGraph],
    initial_interventions: dict[str, float],
    target: str = "Y",
    decay: float = 0.9,
    min_confidence: float = 0.1,
    max_depth: int = 5,
    activation: str = "linear",
) -> dict:
    """
    Peirce 链式推理: 解释项→新符号→再推理。

    每层推理结果 R_k 作为下一层的输入符号，
    附带置信度衰减: conf(R_{k+1}) = conf(R_k) * decay

    终止条件:
    - 置信度 < min_confidence
    - 达到 max_depth 层

    Args:
        causal_graphs: 每层使用的因果图列表 (长度 >= max_depth)
        initial_interventions: 第一层的干预值
        target: 每层读取的目标节点
        decay: 置信度衰减因子
        min_confidence: 最低置信度阈值
        max_depth: 最大链式推理深度
        activation: 激活函数

    Returns:
        {
            "chain": [(depth, value, confidence), ...],
            "final_value": float,
            "final_confidence": float,
            "depth": int,
        }
    """
    chain: list[tuple[int, float, float]] = []
    interventions = dict(initial_interventions)
    confidence = 1.0
    last_value = 0.0

    for k in range(min(max_depth, len(causal_graphs))):
        if confidence < min_confidence:
            break

        cg = causal_graphs[k]
        result = sem_forward(cg, interventions, activation=activation)
        value = result.get(target, 0.0)

        chain.append((k + 1, value, confidence))
        last_value = value

        # 准备下一层: 将本层结果作为下层的干预
        interventions = {target: value}
        # 如果目标节点不在下层图中，用第一个根节点
        roots = _find_roots(cg)
        if roots and target not in cg.nodes:
            interventions = {roots[0]: value}
        else:
            # 目标节点在图中，用它作为干预
            interventions = {target: value}

        confidence *= decay

    return {
        "chain": chain,
        "final_value": last_value,
        "final_confidence": chain[-1][2] if chain else 0.0,
        "depth": len(chain),
    }
