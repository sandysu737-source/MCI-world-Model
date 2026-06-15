"""MCI World Model — CausalGradientPropagation 因果梯度传播
=============================================================

在因果图上传播梯度信号——将端到端损失反传到因果图的
每条边和节点，实现因果结构的端到端优化。

核心能力:
    CausalGradient       — 因果梯度
    CausalGradientPropagation — 因果梯度传播器

设计原则:
    - 基于 NeuralSymbolicFusionV2 (T20) 的融合结果
    - 纯 numpy，零外部依赖
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CausalGradient:
    """因果梯度。

    Attributes:
        source: 梯度源节点
        target: 梯度目标节点
        gradient: 梯度值
        path: 传播路径
    """

    source: str
    target: str
    gradient: float = 0.0
    path: list[str] = field(default_factory=list)


class CausalGradientPropagation:
    """因果梯度传播器 — 在因果图上反向传播梯度。

    用法:
        >>> cgp = CausalGradientPropagation()
        >>> cgp.set_graph(adj_matrix, node_names)
        >>> cgp.set_loss_gradient(loss_grad)
        >>> gradients = cgp.propagate()
    """

    def __init__(self, learning_rate: float = 0.01):
        self._lr = learning_rate
        self._adj: np.ndarray | None = None
        self._node_names: list[str] = []
        self._loss_grad: np.ndarray | None = None
        self._propagation_history: list[list[CausalGradient]] = []

    def set_graph(self, adj_matrix: np.ndarray, node_names: list[str]) -> None:
        """设置因果图。"""
        self._adj = np.atleast_2d(np.asarray(adj_matrix, dtype=float))
        self._node_names = list(node_names)

    def set_loss_gradient(self, gradient: np.ndarray) -> None:
        """设置端到端损失梯度。"""
        self._loss_grad = np.atleast_1d(np.asarray(gradient, dtype=float))

    def propagate(self, n_steps: int = 5) -> list[CausalGradient]:
        """在因果图上传播梯度。

        Args:
            n_steps: 传播步数

        Returns:
            CausalGradient 列表
        """
        if self._adj is None or self._loss_grad is None:
            return []

        n_nodes = self._adj.shape[0]
        grad = self._loss_grad.copy()
        if len(grad) < n_nodes:
            padded = np.zeros(n_nodes)
            padded[: len(grad)] = grad
            grad = padded
        elif len(grad) > n_nodes:
            grad = grad[:n_nodes]

        gradients = []
        for step in range(n_steps):
            # 反向传播: grad = adj^T @ grad
            grad = self._adj.T @ grad

            # 记录显著梯度
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if abs(self._adj[i, j]) > 1e-8 and abs(grad[j]) > 1e-4:
                        name_i = self._node_names[i] if i < len(self._node_names) else f"n{i}"
                        name_j = self._node_names[j] if j < len(self._node_names) else f"n{j}"
                        gradients.append(
                            CausalGradient(
                                source=name_i,
                                target=name_j,
                                gradient=float(grad[j]),
                                path=[name_i, name_j],
                            )
                        )

        self._propagation_history.append(gradients)
        return gradients

    def get_node_gradients(self) -> dict[str, float]:
        """获取各节点的累积梯度。"""
        if self._adj is None or self._loss_grad is None:
            return {}

        n_nodes = self._adj.shape[0]
        grad = self._loss_grad.copy()
        if len(grad) < n_nodes:
            padded = np.zeros(n_nodes)
            padded[: len(grad)] = grad
            grad = padded

        result = {}
        for i in range(min(n_nodes, len(self._node_names))):
            result[self._node_names[i]] = float(grad[i])

        return result

    def update_graph(self, gradients: list[CausalGradient]) -> np.ndarray:
        """根据梯度更新因果图权重。

        Args:
            gradients: 因果梯度列表

        Returns:
            更新后的邻接矩阵
        """
        if self._adj is None:
            return np.array([])

        updated = self._adj.copy()
        name_to_idx = {name: i for i, name in enumerate(self._node_names)}

        for g in gradients:
            i = name_to_idx.get(g.source)
            j = name_to_idx.get(g.target)
            if i is not None and j is not None:
                updated[i, j] -= self._lr * g.gradient

        self._adj = updated
        return updated

    @property
    def propagation_count(self) -> int:
        return len(self._propagation_history)

    def statistics(self) -> dict[str, Any]:
        return {
            "propagation_count": self.propagation_count,
            "n_nodes": len(self._node_names),
            "learning_rate": self._lr,
        }
