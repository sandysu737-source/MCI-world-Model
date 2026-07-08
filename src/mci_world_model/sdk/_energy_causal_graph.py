"""能量加权因果图 — 桥接 EnergyCore 矩阵动力学与 algebra.CausalDAG 图传播。

把五行生克关系 (EnergyCore 的守恒 Flow 矩阵) 编码为 CausalDAG 的有向边
权重, 使因果图传播自动遵循能量守恒律, 而非硬编码系数。

数学基础:
  EnergyCore 的 Flow 矩阵 F 编码能量流转: x_{t+1} = x + F·x。
  F[dst, src] > 0 表示 src → dst 的正向能量流入 (生或克夺取)。
  将这些正向流作为 CausalDAG 的边权重, 传播效应 = Π(沿路径边权重),
  等价于能量沿生克环的几何衰减。

这统一了两套原本平行的实现:
  - _sys/causal.py CausalChain.propagate (硬编码 1.1/0.3 系数)
  - _sys/_energy_core.py EnergyCore.simulate_energy_flow (矩阵迭代)
为基于 algebra 层的单一守恒传播。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from mci_world_model.algebra.causal_graph import CausalDAG


class EnergyWeightedCausalGraph:
    """能量加权的因果图: 用五行 Flow 矩阵作为边权重。

    将 EnergyCore 的生克关系编码为 CausalDAG:
      - 生 (enhance): src → tgt, 权重 = ENHANCE_FLOW_RATE
      - 克 (suppress): tgt → src, 权重 = SUPPRESS_FLOW_RATE (能量夺取方向)

    传播沿这些加权边进行, 自动遵循能量守恒结构。
    """

    def __init__(self, energy_core: Any, min_weight: float = 0.01) -> None:
        """从 EnergyCore 构建能量加权因果图。

        Args:
            energy_core: EnergyCore 实例 (需有 flow_matrix() 方法)
            min_weight: 最小边权重阈值 (低于此值不入图)
        """
        self._core = energy_core
        self._min_weight = min_weight
        self._categories = list(energy_core.ENERGY_ORDER)
        self._idx = {c: i for i, c in enumerate(self._categories)}
        self._dag = self._build_dag()

    def _build_dag(self) -> CausalDAG:
        """从 Flow 矩阵构建加权 CausalDAG。"""
        F = self._core.flow_matrix()
        dag = CausalDAG()
        for c in self._categories:
            dag.add_node(c)
        for src in self._categories:
            for dst in self._categories:
                if src == dst:
                    continue
                w = float(F[self._idx[dst], self._idx[src]])
                if w > self._min_weight:
                    dag.add_edge(src, dst, weight=round(w, 6))
        return dag

    @property
    def dag(self) -> CausalDAG:
        """底层 CausalDAG (algebra 层)。"""
        return self._dag

    @property
    def categories(self) -> list[str]:
        return list(self._categories)

    def propagate(self, source: str, delta: float = 1.0) -> dict[str, float]:
        """沿能量加权因果图传播干预效应。

        等价于能量沿生克环的几何衰减传播。
        每个节点接收 = 父节点效应 × 边权重 (生克系数)。

        Args:
            source: 干预源 (五行维度名)
            delta: 注入量

        Returns:
            {维度: 接收效应} (source 自身含 delta)
        """
        if source not in self._idx:
            raise ValueError(f"未知维度 '{source}', 已知: {self._categories}")
        return self._dag.propagate(source, delta)

    def propagation_vector(self, source: str) -> np.ndarray:
        """传播效应向量 (按 categories 顺序, 供矩阵运算)。"""
        v, _ = self._dag.propagation_vector(source, self._categories)
        return v

    def edge_weights(self) -> np.ndarray:
        """加权邻接矩阵 W[i,j] = 边 i→j 的权重 (algebra 层)。"""
        W, _ = self._dag.adjacency_matrix(self._categories)
        return W

    def relation(self, src: str, dst: str) -> str:
        """分类 src→dst 的关系 (生/克夺取/无)。"""
        from mci_world_model._sys._terms import ENERGY_ENHANCE, ENERGY_SUPPRESS
        if ENERGY_ENHANCE.get(src) == dst:
            return "生"
        if ENERGY_SUPPRESS.get(src) == dst:
            return "克"
        w = self._dag.edge_weight(src, dst)
        if w > 0:
            return "克夺取"  # 克关系的能量夺取方向
        return "无"
