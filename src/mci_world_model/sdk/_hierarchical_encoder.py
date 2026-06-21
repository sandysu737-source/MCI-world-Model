from __future__ import annotations

"""
MCI World Model v3.0.2 — Hierarchical JEPA Encoder
====================================================

LeCun H-JEPA (Hierarchical Joint Embedding Predictive Architecture)
分层潜空间编码器，三层堆叠:

    Level 1 (Entity):   X_t   → GAT_L1 → A¹_enc → s¹_t  ──predict──→ s¹_{t+1}
    Level 2 (Energy):   s¹_t  → GAT_L2 → A²_enc → s²_t  ──predict──→ s²_{t+1}
    Level 3 (Causal):   s²_t  → GNN    → A³_enc → s³_t  ──predict──→ s³_{t+1}
                            ↑ 各层误差信号从上层往下传播（top-down guidance）

核心特性:
- 三层 GAT 编码器独立可训练
- Level-3 GNN 预测器复用现有 M2 架构
- 保持 M1/M2/M3 模式不变，分层模式作为 M4 新入口
- 状态机: IDLE → ENCODING_L1 → ENCODING_L2 → ENCODING_L3 → PREDICTING → COMPLETE

用法:
    from mci_world_model.sdk._hierarchical_encoder import (
        HierarchicalJEPAEncoder, HierarchicalState,
    )

    encoder = HierarchicalJEPAEncoder(world_model, key_dim=16, hidden_dim=16)
    h_state = encoder.encode(memories)
    h_pred = encoder.predict(h_state)
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from mci_world_model.sdk._world_model import CausalWorldModelState

logger = logging.getLogger(__name__)


# =============================================================================
# HierarchicalState — 三层潜状态快照
# =============================================================================


@dataclass
class HierarchicalState:
    """
    三层潜状态快照。

    Attributes:
        level_1: Entity-level 因果世界状态（细粒度，~每条实体边）
        level_2: Energy-level 因果世界状态（中粒度，能量关系分组）
        level_3: Causal-level 因果世界状态（粗粒度，全局因果结构）
        timestamp: 编码时间戳
    """

    level_1: CausalWorldModelState
    level_2: CausalWorldModelState
    level_3: CausalWorldModelState
    timestamp: str = ""

    @classmethod
    def empty(cls) -> HierarchicalState:
        return cls(
            level_1=CausalWorldModelState.empty(),
            level_2=CausalWorldModelState.empty(),
            level_3=CausalWorldModelState.empty(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "l1_edges": len(self.level_1.causal_edges),
            "l2_edges": len(self.level_2.causal_edges),
            "l3_edges": len(self.level_3.causal_edges),
            "l1_confirmed": self.level_1.n_confirmed,
            "l2_confirmed": self.level_2.n_confirmed,
            "l3_confirmed": self.level_3.n_confirmed,
            "timestamp": self.timestamp,
        }


# =============================================================================
# HierarchicalJEPAEncoder — H-JEPA 分层编码器
# =============================================================================


class HierarchicalJEPAEncoder:
    """
    H-JEPA 三层分层编码器。

    每层使用独立的 GAT 编码器 + Level-3 使用 GNN 预测器，
    实现 LeCun 风格的逐层抽象与预测。

    六态流转：IDLE → ENCODING_L1 → ENCODING_L2 → ENCODING_L3 → PREDICTING → COMPLETE
    """

    def __init__(  # type: ignore
        self,
        world_model,
        key_dim: int = 16,
        hidden_dim: int = 16,
        seed: int = 42,
    ):
        """
        Args:
            world_model: MCIWorldModel 实例
            key_dim: GAT 注意力键维度（L1/L2 共用）
            hidden_dim: GNN 隐层维度（L3）
            seed: 随机种子
        """
        self._wm = world_model
        self._state: str = "IDLE"  # IDLE → ENCODING_L1 → ... → COMPLETE

        # ── L1: Entity-level GAT ──
        from mci_world_model.sdk._jepa_gat_encoder import GATEncoder

        self._gat_l1 = GATEncoder(input_dim=8, key_dim=key_dim, seed=seed)
        self._gat_l2 = GATEncoder(input_dim=8, key_dim=key_dim, seed=seed + 100)

        # ── L3: GNN 预测器 ──
        from mci_world_model.sdk._jepa_gnn import GNNPredictor

        self._gnn_l3 = GNNPredictor(hidden_dim=hidden_dim, seed=seed + 200)

        # ── 训练缓存 ──
        self._cache: dict[str, Any] = {}
        self._cache_lock = threading.Lock()

        # ── 统计 ──
        self._encode_count: int = 0
        self._predict_count: int = 0
        self._train_steps: int = 0

    # -----------------------------------------------------------------
    # 属性
    # -----------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def encode_count(self) -> int:
        return self._encode_count

    @property
    def predict_count(self) -> int:
        return self._predict_count

    # -----------------------------------------------------------------
    # 编码 — memories → HierarchicalState
    # -----------------------------------------------------------------

    def encode(
        self,
        memories: list[dict[str, Any]],
    ) -> HierarchicalState:
        """
        三层编码：memories → L1 → L2 → L3。

        Args:
            memories: 记忆列表

        Returns:
            HierarchicalState 三层潜状态
        """
        if not memories:
            self._state = "COMPLETE"
            return HierarchicalState.empty()

        from datetime import datetime

        # ── L1: Entity-level GAT ──
        self._state = "ENCODING_L1"
        s_l1 = self._encode_level_1(memories)

        # ── L2: Energy-level GAT（从 L1 状态取特征）──
        self._state = "ENCODING_L2"
        s_l2 = self._encode_level_2(s_l1)

        # ── L3: Causal-level GNN（从 L2 状态取特征）──
        self._state = "ENCODING_L3"
        s_l3 = self._encode_level_3(s_l2)

        self._encode_count += 1
        self._state = "COMPLETE"

        ts = datetime.now().isoformat()
        return HierarchicalState(
            level_1=s_l1,
            level_2=s_l2,
            level_3=s_l3,
            timestamp=ts,
        )

    def _encode_level_1(self, memories: list[dict[str, Any]]) -> CausalWorldModelState:
        """L1: Entity-level GAT 编码。"""
        try:
            from mci_world_model.sdk._jepa_gat_encoder import (
                features_to_state,
                preprocess_memories_to_features,
            )

            X, node_index, _ = preprocess_memories_to_features(self._wm, memories)

            if X.shape[0] == 0:
                return CausalWorldModelState.empty()

            A_enc = self._gat_l1.forward(X)
            state = features_to_state(A_enc, node_index, self._wm._state)
            return state

        except Exception as e:
            logger.warning("H-JEPA L1 编码失败: %s, 降级为 discover()", e)
            try:
                return self._wm.discover(memories)
            except Exception:
                return CausalWorldModelState.empty()

    def _encode_level_2(self, s_l1: CausalWorldModelState) -> CausalWorldModelState:
        """L2: Energy-level GAT 编码（从 L1 状态提取粗粒度特征）。"""
        if not s_l1.causal_edges:
            return CausalWorldModelState.empty()

        try:
            node_index = s_l1._build_node_index()
            X = s_l1.to_node_feature_matrix()

            if X.shape[0] == 0:
                return CausalWorldModelState.empty()

            A_enc = self._gat_l2.forward(X)
            from mci_world_model.sdk._jepa_gat_encoder import features_to_state

            state = features_to_state(A_enc, node_index, s_l1)
            return state

        except Exception as e:
            logger.warning("H-JEPA L2 编码失败: %s, 直接透传 L1", e)
            return s_l1  # 降级：透传

    def _encode_level_3(self, s_l2: CausalWorldModelState) -> CausalWorldModelState:
        """L3: Causal-level GNN 预测（最粗粒度，全局因果结构）。"""
        if not s_l2.causal_edges:
            return CausalWorldModelState.empty()

        try:
            return self._gnn_l3.predict(s_l2)
        except Exception as e:
            logger.warning("H-JEPA L3 GNN 预测失败: %s, 直接透传 L2", e)
            return s_l2  # 降级：透传

    # -----------------------------------------------------------------
    # 分层预测 — HierarchicalState → HierarchicalState
    # -----------------------------------------------------------------

    def predict(self, h_state: HierarchicalState) -> HierarchicalState:
        """
        分层预测下一时刻状态。

        独立于 encode()，可用于训练循环中的 s_t → s_{t+1} 预测。

        Args:
            h_state: 当前时刻三层状态

        Returns:
            预测的下一时刻三层状态
        """
        self._state = "PREDICTING"

        s_l1_pred = self._gnn_l3.predict(h_state.level_1)
        s_l2_pred = self._gnn_l3.predict(h_state.level_2)
        s_l3_pred = self._gnn_l3.predict(h_state.level_3)

        self._predict_count += 1
        self._state = "COMPLETE"

        return HierarchicalState(
            level_1=s_l1_pred if s_l1_pred.causal_edges else h_state.level_1,
            level_2=s_l2_pred if s_l2_pred.causal_edges else h_state.level_2,
            level_3=s_l3_pred if s_l3_pred.causal_edges else h_state.level_3,
            timestamp=h_state.timestamp,
        )

    # -----------------------------------------------------------------
    # 分层距离度量
    # -----------------------------------------------------------------

    def hierarchical_distance(
        self,
        s_pred: HierarchicalState,
        s_target: HierarchicalState,
        alpha_l1: float = 0.2,
        alpha_l2: float = 0.3,
        alpha_l3: float = 0.5,
    ) -> float:
        """
        计算三层加权距离。

        高层（L3）权重更大，因为粗粒度结构更重要。

        Args:
            s_pred: 预测状态
            s_target: 目标状态
            alpha_l1/l2/l3: 各层权重 (sum=1.0)

        Returns:
            加权距离 [0, 1]
        """
        d1 = s_pred.level_1.state_distance(s_target.level_1)
        d2 = s_pred.level_2.state_distance(s_target.level_2)
        d3 = s_pred.level_3.state_distance(s_target.level_3)

        return alpha_l1 * d1 + alpha_l2 * d2 + alpha_l3 * d3

    # -----------------------------------------------------------------
    # 训练接口（M4 端到端可微）
    # -----------------------------------------------------------------

    def training_encode(
        self,
        memories: list[dict[str, Any]],
    ) -> tuple[np.ndarray, dict[str, int]]:
        """
        训练模式编码：返回 L3 的 (A_enc, node_index) 用于端到端训练。

        内部走 L1→L2→L3 全链，但仅输出 L3 张量用于损失计算。

        Args:
            memories: 记忆列表

        Returns:
            (A_enc, node_index) 用于 GNN 损失计算
        """
        if not memories:
            return np.zeros((0, 0)), {}

        from mci_world_model.sdk._jepa_gat_encoder import (
            preprocess_memories_to_features,
        )

        X, node_index, _ = preprocess_memories_to_features(self._wm, memories)
        if X.shape[0] == 0:
            return np.zeros((0, 0)), {}

        # L1: 训练前向
        A1 = self._gat_l1.training_forward(X)

        # L2: 从 L1 邻接矩阵构建特征
        X2 = self._l1_to_l2_features(A1, X)
        A2 = self._gat_l2.training_forward(X2)

        # L3: 输出给 GNN 预测器
        self._train_steps += 1

        with self._cache_lock:
            self._cache = {
                "A1": A1,
                "A2": A2,
                "X": X,
                "X2": X2,
                "node_index": node_index,
            }

        return A2, node_index

    def _l1_to_l2_features(self, A1: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        L1→L2 特征转换：用 L1 邻接矩阵对原始特征做消息传递。

        X2 = relu(A1 @ X)  — 简单的一阶邻居聚合

        Args:
            A1: L1 GAT 输出的 [N, N] 邻接矩阵
            X: 原始节点特征 [N, D]

        Returns:
            L2 输入特征 [N, D]
        """
        H = A1 @ X  # [N, D]
        X2 = np.maximum(H, 0)  # ReLU
        # 归一化
        norms = np.linalg.norm(X2, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        X2 = X2 / norms
        return X2

    # -----------------------------------------------------------------
    # 评估
    # -----------------------------------------------------------------

    def evaluate(
        self,
        dataset: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    ) -> dict[str, Any]:
        """
        在时序数据集上评估分层预测精度。

        Args:
            dataset: [(mem_t, mem_{t+1}), ...]

        Returns:
            评估统计字典
        """
        distances: list[float] = []
        for mem_t, mem_t1 in dataset:
            h_t = self.encode(mem_t)
            h_pred = self.predict(h_t)
            h_t1 = self.encode(mem_t1)
            d = self.hierarchical_distance(h_pred, h_t1)
            distances.append(d)

        if not distances:
            return {"avg_distance": 1.0, "n": 0}

        return {
            "avg_distance": round(float(np.mean(distances)), 6),
            "min_distance": round(float(np.min(distances)), 6),
            "max_distance": round(float(np.max(distances)), 6),
            "n": len(distances),
        }
