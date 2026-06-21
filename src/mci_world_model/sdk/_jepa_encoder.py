from __future__ import annotations

"""
MCI World Model v3.1.0 — JEPA Encoder
=========================================

将观测（记忆列表）编码为结构化潜状态（CausalWorldModelState）。

核心组件:
- JEPAEncoder: JEPA 标准编码器接口
  - encode(memories) → CausalWorldModelState
  - 第一期委托给 MCIWorldModel.discover()（不可微）
  - _differentiable 开关预留 M2 可微版本

用法:
    from mci_world_model.sdk._jepa_encoder import JEPAEncoder

    encoder = JEPAEncoder(world_model)
    s_t = encoder.encode(memories_at_t)
    adj, node_feat = encoder.to_graph_tensors(s_t)
"""


import logging
from typing import Any

import numpy as np

from mci_world_model.sdk._world_model import CausalWorldModelState

logger = logging.getLogger(__name__)


class JEPAEncoder:
    """
    JEPA 编码器: 观测 → 结构化潜状态。

    **线程模型**: 非线程安全。_encode_count 为非原子计数器，
    多线程共享实例需外部同步。

    两种模式:
    - 不可微 (M1/M2): discover() 三层统计因果发现
    - 可微 (M3): GAT 图注意力编码器替代统计推断

    M3 端到端训练流:
        memories → preprocess → X[N,D] → GATEncoder → A_enc → GNNPredictor → A_pred
                                        ↑__________________↓
                                             loss + backward
    """

    def __init__(  # type: ignore
        self,
        world_model,
        differentiable: bool = False,
        gat_key_dim: int = 16,
        learnable_encoder=None,
    ):
        """
        Args:
            world_model: MCIWorldModel 实例（含 discover() 能力）
            differentiable: 是否启用 M3 可微 GAT 编码器
            gat_key_dim: GAT 注意力键维度
            learnable_encoder: v5.0.0 LearnableStateEncoder 实例（可选）
        """
        self._wm = world_model
        self._differentiable = differentiable
        self._encode_count: int = 0

        # M3: GAT 可微编码器
        self._gat_encoder = None
        if differentiable:
            self._init_gat_encoder(key_dim=gat_key_dim)

        # v5.0.0: 可学习状态编码器
        self._learnable_encoder = learnable_encoder

    @property
    def is_differentiable(self) -> bool:
        return self._differentiable

    # -----------------------------------------------------------------
    # 核心编码
    # -----------------------------------------------------------------

    def encode(
        self,
        memories: list[dict[str, Any]] = None,  # type: ignore
        signals: list[dict[str, Any]] = None,  # type: ignore
        use_parametric: bool = False,
    ) -> CausalWorldModelState:
        """
        将观测记忆编码为因果世界状态。

        v3.1.0: 新增 signals 参数支持物理世界信号输入。

        Args:
            memories: 记忆列表 [{"content": ..., "timestamp": ...}, ...]
            signals: 多模态信号列表 (v3.1.0 新增物理世界路径)
                     [{"signal_type": ..., "value": ..., "timestamp": ..., ...}, ...]
                     或 dict 列表格式 (每条为一天的物理量快照)
            use_parametric: 是否启用参数化增强（M2 可微时才生效）

        Returns:
            CausalWorldModelState 潜状态表示

        Raises:
            RuntimeError: 如果 world_model 未就绪
        """
        if signals is not None:
            # ── v3.1.0 物理世界路径 ──
            # v5.0.0: 优先使用可学习编码器
            if self._learnable_encoder is not None:
                return self._encode_differentiable_physical(signals)
            return self._encode_via_physical_builder(signals)

        if memories is not None:
            return self._encode_via_memories(memories, use_parametric)

        return CausalWorldModelState.empty()

    def _encode_differentiable_physical(self, signals: list[dict[str, Any]]) -> CausalWorldModelState:
        """
        v5.0.0: 使用 LearnableStateEncoder 编码物理信号。

        流程:
        1. 从 signals 提取状态向量 (to_vector)
        2. 调用 LearnableStateEncoder.forward()
        3. 将潜向量填充到 CausalWorldModelState 的 parametric_cache 字段
        4. 同时使用 PhysicalGraphBuilder 构建因果边（保持因果推理能力）
        """
        if not signals:
            return CausalWorldModelState.empty()

        from mci_world_model.sdk._physical_graph_builder import PhysicalGraphBuilder

        # 1. 使用 PhysicalGraphBuilder 获取因果边（不可微但提供因果结构）
        builder = PhysicalGraphBuilder()

        # 检测输入格式
        if signals and hasattr(signals[0], "signal_type"):
            from mci_world_model._sys._perception_pipeline import MultimodalSignal

            if isinstance(signals[0], MultimodalSignal):
                from mci_world_model.sdk._physical_graph_builder import signals_to_timeline

                timeline = signals_to_timeline(signals)
            else:
                timeline = signals
        else:
            timeline = signals

        edges = builder.build_graph(timeline)
        timestamp = signals[0].get("timestamp", "") if isinstance(signals[0], dict) else ""

        # 2. 提取状态向量并编码
        latent = None
        try:
            if isinstance(signals[0], dict):
                numeric_values = []
                for sig in signals:
                    for k, v in sig.items():
                        if isinstance(v, (int, float)) and k not in ("day", "timestamp"):
                            numeric_values.append(float(v))
                if numeric_values:
                    state_vector = np.array(numeric_values[: self._learnable_encoder.state_dim], dtype=np.float64)
                    if len(state_vector) < self._learnable_encoder.state_dim:
                        padded = np.zeros(self._learnable_encoder.state_dim, dtype=np.float64)
                        padded[: len(state_vector)] = state_vector
                        state_vector = padded
                    latent = self._learnable_encoder.forward(state_vector)
        except Exception as e:
            logger.debug("可学习编码跳过: %s", e)

        # 3. 构建状态
        state = CausalWorldModelState(
            causal_edges=edges,
            n_novel=sum(1 for e in edges if e.get("verdict") == "novel"),
            n_confirmed=sum(1 for e in edges if e.get("verdict") == "confirmed"),
            n_memories=len(timeline),
            timestamp=timestamp,
        )

        # 4. 将潜向量存入 parametric_cache
        if latent is not None:
            if not hasattr(state, "parametric_cache") or state.parametric_cache is None:
                state.parametric_cache = {}  # type: ignore
            state.parametric_cache["learnable_latent"] = latent.tolist()  # type: ignore

        return state

    @property
    def learnable_encoder(self) -> None:
        """返回 v5.0.0 可学习编码器实例。"""
        return self._learnable_encoder

    def _encode_via_physical_builder(self, signals: list[dict[str, Any]]) -> CausalWorldModelState:
        """
        v3.1.0: 通过 PhysicalGraphBuilder 编码物理信号。

        支持两种输入格式:
        1. dict 列表 (由 generate_synthetic_patient 生成):
           [{"day": 1, "albumin": 30, ...}, ...]
        2. MultimodalSignal 对象列表（先转为 dict 列表）
        """
        from mci_world_model.sdk._physical_graph_builder import PhysicalGraphBuilder

        # v3.1.0a QC: 空 signals 保护
        if not signals:
            return CausalWorldModelState.empty()

        builder = PhysicalGraphBuilder()

        # 检测输入格式
        if signals and hasattr(signals[0], "signal_type"):
            # MultimodalSignal 对象列表 → 先转为 timeline
            from mci_world_model._sys._perception_pipeline import MultimodalSignal

            if isinstance(signals[0], MultimodalSignal):
                from mci_world_model.sdk._physical_graph_builder import signals_to_timeline

                timeline = signals_to_timeline(signals)
            else:
                timeline = signals
        else:
            timeline = signals

        edges = builder.build_graph(timeline)
        timestamp = signals[0].get("timestamp", "") if isinstance(signals[0], dict) else ""

        return CausalWorldModelState(
            causal_edges=edges,
            n_novel=sum(1 for e in edges if e.get("verdict") == "novel"),
            n_confirmed=sum(1 for e in edges if e.get("verdict") == "confirmed"),
            n_memories=len(timeline),  # v3.1.0: 使物理状态通过 JEPADataset 过滤
            timestamp=timestamp,
        )

    def _encode_via_memories(
        self,
        memories: list[dict[str, Any]],
        use_parametric: bool = False,
    ) -> CausalWorldModelState:
        """原有路径: 通过 discover() 编码文本记忆。"""
        if self._wm is None:
            raise RuntimeError("JEPAEncoder: world_model 未初始化")

        # ── 1. 证据收集 ──
        memories = self._collect_evidence(memories)

        # ── 2. 时空标注 ──
        memories = self._annotate_temporal(memories)

        # ── 3. 因果发现 ──
        if self._differentiable:
            state = self._encode_differentiable(memories)
        else:
            state = self._wm.discover(memories, use_parametric=use_parametric)

        self._encode_count += 1
        return state

    def _collect_evidence(self, memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """通过 EvidenceCollector 预处理记忆。"""
        try:
            from mci_world_model._sys.bayesian import BayesianEngine
            from mci_world_model._sys.evidence import EvidenceCollector

            engine = BayesianEngine()
            collector = EvidenceCollector(engine)
            for mem in memories:
                content = mem.get("content", "")
                if content:
                    collector.add_observation(  # type: ignore
                        source_id=mem.get("id", f"mem_{hash(content) % 100000}"),
                        observation_type="memory_content",
                        value=True,
                        confidence=mem.get("confidence", 0.7),
                        metadata={"content": content[:200]},
                    )
        except Exception as e:
            logger.debug("EvidenceCollector 跳过: %s", e)
        return memories

    def _annotate_temporal(self, memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """通过 TemporalSystem 标注时间信息。"""
        try:
            from mci_world_model._sys.chrono import TemporalSystem

            ts = TemporalSystem()  # type: ignore
            for mem in memories:
                timestamp = mem.get("timestamp", "")
                if timestamp:
                    try:
                        temporal_info = ts.encode(timestamp)  # type: ignore
                        mem["_temporal"] = temporal_info
                    except Exception as e:
                        logger.debug("时序编码跳过单条: %s", e)
        except Exception as e:
            logger.debug("TemporalSystem 跳过: %s", e)
        return memories

    def _encode_differentiable(self, memories: list[dict[str, Any]]) -> CausalWorldModelState:
        """
        M3: 可微 GAT 编码器。

        1. discover() 做实体提取 + 节点特征（不可微预处理，仅一次）
        2. GAT 编码器将节点特征映射为因果边（可微）
        3. 从预测邻接矩阵重建 CausalWorldModelState
        """
        if self._gat_encoder is None:
            # 回退到不可微版本
            logger.warning("GAT 编码器未初始化，回退到 discover()")
            return self._wm.discover(memories)

        from mci_world_model.sdk._jepa_gat_encoder import (
            features_to_state,
            preprocess_memories_to_features,
        )

        # ── 1. 预处理: 实体提取 + 节点特征 ──
        X, node_index, _node_names = preprocess_memories_to_features(self._wm, memories)

        if X.shape[0] == 0:
            return CausalWorldModelState.empty()

        # ── 2. GAT 前向 ──
        A_enc = self._gat_encoder.forward(X)

        # ── 3. 重建状态 ──
        template = self._wm._state
        state = features_to_state(A_enc, node_index, template)
        state.timestamp = template.timestamp

        self._encode_count += 1
        return state

    # -----------------------------------------------------------------
    # M3: 可微编码器接口
    # -----------------------------------------------------------------

    def _init_gat_encoder(self, key_dim: int = 16) -> None:
        """初始化 GAT 可微编码器。"""
        try:
            from mci_world_model.sdk._jepa_gat_encoder import GATEncoder

            self._gat_encoder = GATEncoder(input_dim=8, key_dim=key_dim)  # type: ignore
            logger.info("M3 GAT 编码器初始化完成 (key_dim=%d)", key_dim)
        except ImportError as e:
            logger.warning("GAT 编码器不可用: %s", e)
            self._differentiable = False

    @property
    def gat_encoder(self) -> None:
        """返回 GAT 编码器实例（M3 使用）。"""
        return self._gat_encoder

    def training_encode(  # type: ignore
        self,
        memories: list[dict[str, Any]],
        state=None,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """
        M3 训练模式编码（返回张量用于端到端反向传播）。

        与 encode() 的区别:
        - 不返回 CausalWorldModelState，返回原始张量 (A_enc, node_index)
        - 内部调用 GAT.training_forward() 缓存中间值
        - 梯度可从 GNN 预测器损失链式反向传播

        Args:
            memories: 记忆列表
            state: 预计算的 CausalWorldModelState（缓存优化）

        Returns:
            (A_enc, node_index):
            - A_enc: [N, N] float64 边预测邻接矩阵
            - node_index: {"node_name": idx}
        """
        if self._gat_encoder is None:
            raise RuntimeError("training_encode: GAT 编码器未初始化")

        from mci_world_model.sdk._jepa_gat_encoder import preprocess_memories_to_features

        X, node_index, _ = preprocess_memories_to_features(
            self._wm,
            memories,
            state=state,
        )
        if X.shape[0] == 0:
            return np.zeros((0, 0)), {}

        A_enc = self._gat_encoder.training_forward(X)
        return A_enc, node_index

    # -----------------------------------------------------------------
    # 图张量转换
    # -----------------------------------------------------------------

    def to_graph_tensors(self, state: CausalWorldModelState) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        将 CausalWorldModelState 转换为 GNN 可用的张量。

        Args:
            state: 因果世界状态

        Returns:
            (adjacency, node_features, edge_features):
            - adjacency: N×N float32 邻接矩阵
            - node_features: N×D float32 节点特征
            - edge_features: E×F float32 边特征（当前为 None）
        """
        adj = state.to_adjacency_matrix()
        node_feat = state.to_node_feature_matrix()
        # 边特征（rho/confidence/bayes_factor）
        edge_feat = self._build_edge_features(state)
        return adj, node_feat, edge_feat

    def _build_edge_features(self, state: CausalWorldModelState) -> np.ndarray | None:
        """构建边特征矩阵。"""
        edges = state.causal_edges
        if not edges:
            return None
        # 每条边: [rho, confidence, bayes_factor(0 if missing)]
        feat = np.zeros((len(edges), 3), dtype=np.float32)
        for i, e in enumerate(edges):
            feat[i, 0] = float(e.get("rho", 0.0))
            feat[i, 1] = float(e.get("confidence", 0.5))
            feat[i, 2] = float(e.get("bayes_factor", 0.0))
        return feat

    def from_graph_tensors(
        self,
        adj: np.ndarray,
        node_names: list[str],
        edge_feat: np.ndarray | None = None,
        metadata: dict | None = None,  # type: ignore
    ) -> CausalWorldModelState:
        """
        从 GNN 预测的图张量重建 CausalWorldModelState。

        这是 to_graph_tensors() 的逆操作。

        Args:
            adj: N×N 邻接矩阵
            node_names: 节点名称列表
            edge_feat: 边特征矩阵（可选）
            metadata: 额外元信息

        Returns:
            重建的 CausalWorldModelState
        """
        n = adj.shape[0]
        causal_edges: list[dict[str, Any]] = []

        for i in range(n):
            for j in range(n):
                rho = float(adj[i, j])
                if rho > 0.01 and i != j:  # 过滤弱边和自环
                    edge = {
                        "cause": node_names[i] if i < len(node_names) else f"node_{i}",
                        "effect": node_names[j] if j < len(node_names) else f"node_{j}",
                        "rho": round(rho, 4),
                        "confidence": round(min(rho + 0.15, 1.0), 4),
                        "verdict": "novel" if rho > 0.5 else "none",
                        "energy_relation": "neutral",
                        "bayes_factor": 0.0,
                    }
                    # 从边特征获取细节
                    if edge_feat is not None:
                        edge_idx = i * n + j
                        if edge_idx < edge_feat.shape[0]:
                            edge["rho"] = round(float(edge_feat[edge_idx, 0]), 4)
                            edge["confidence"] = round(float(edge_feat[edge_idx, 1]), 4)
                            edge["bayes_factor"] = round(float(edge_feat[edge_idx, 2]), 4)
                    causal_edges.append(edge)

        return CausalWorldModelState(
            causal_edges=causal_edges,
            n_confirmed=sum(1 for e in causal_edges if e.get("verdict") == "confirmed"),
            n_novel=sum(1 for e in causal_edges if e.get("verdict") == "novel"),
            n_suppressed=sum(1 for e in causal_edges if e.get("verdict") == "suppressed"),
            timestamp=metadata.get("timestamp", "") if metadata else "",
            # v3.1.0: 从 metadata 注入时空与认知元数据
            temporal_info=metadata.get("temporal_info") if metadata else None,
            cognitive_gaps=metadata.get("cognitive_gaps", []) if metadata else [],
        )

    # -----------------------------------------------------------------
    # 统计
    # -----------------------------------------------------------------

    @property
    def encode_count(self) -> int:
        return self._encode_count
