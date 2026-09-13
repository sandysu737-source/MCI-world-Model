"""MCI 世界模型的纯状态契约，禁止反向依赖主编排类。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["CausalWorldModelState", "TrajectoryStep", "WorkingMemory"]

logger = logging.getLogger(__name__)


# =============================================================================
# CausalWorldModelState
# =============================================================================


@dataclass
class CausalWorldModelState:
    """
    因果世界模型状态 — 事实世界 + 反事实世界双图结构。

    事实世界 G: 观测到的因果关系图
    反事实世界 G_not_X: 干预后的反事实图
    """

    # ── 因果图 ──
    causal_edges: list[dict[str, Any]] = field(default_factory=list)
    # [{"cause": str, "effect": str, "rho": float, "confidence": float,
    #   "verdict": str, "energy_relation": str, "bayes_factor": float}, ...]

    # ── 状态覆盖 ──
    active_states: set[str] = field(default_factory=set)
    # 当前活跃的五范畴状态

    # ── 置信度统计 ──
    n_confirmed: int = 0
    n_novel: int = 0
    n_suppressed: int = 0

    # ── 元信息 ──
    n_memories: int = 0
    n_qa_pairs: int = 0
    parametric_enhanced: bool = False
    timestamp: str = ""

    # ── 反事实世界（v3.0.8 L3）─
    counterfactual_graph: dict | None = None  # type: ignore
    do_interventions: list[dict[str, Any]] = field(default_factory=list)

    # ── v3.1.0 JEPA: 时空 + 信念 + 元认知元数据 ──
    temporal_info: object | None = None
    # TemporalInfo from _sys/chrono.py（干支持信息）
    belief_tracker: object | None = None
    # BayesianBeliefTracker from _sys/states.py（信念演化）
    cognitive_gaps: list[Any] = field(default_factory=list)
    # list[CognitiveGap] from _sys/awareness.py（认知空洞）

    # ── v3.0.1 STM: 工作记忆轨迹缓冲区 ──
    working_memory: object | None = None
    # WorkingMemory 短期轨迹缓冲

    # ── v3.0.1 Cost: 最近代价信号快照 ──
    latest_cost_signal: object | None = None
    # CostSignal 最近一次评估结果

    # ── v3.0.4: 五维能量分布快照 ──
    energy_ratios: dict[str, float] | None = None
    # {"semantic": 0.25, "causal": 0.30, ...}  从因果边聚合

    # ── v3.2.0: 独立世界状态（桥接新旧架构）──
    world_state: object | None = None
    # WorldState 实例，独立于推理过程的世界内在表征
    # None = 旧模式（纯因果图推理），非 None = 新模式（世界状态 + 因果图并行）

    @classmethod
    def empty(cls) -> CausalWorldModelState:
        return cls()

    def to_dict(self) -> dict[str, Any]:
        result = {
            "n_causal_edges": len(self.causal_edges),
            "n_confirmed": self.n_confirmed,
            "n_novel": self.n_novel,
            "n_suppressed": self.n_suppressed,
            "active_states": list(self.active_states),
            "n_memories": self.n_memories,
            "n_qa_pairs": self.n_qa_pairs,
            "parametric_enhanced": self.parametric_enhanced,
            "timestamp": self.timestamp,
            "has_counterfactual_graph": self.counterfactual_graph is not None,
            "n_do_interventions": len(self.do_interventions),
            "has_temporal_info": self.temporal_info is not None,
            "has_belief_tracker": self.belief_tracker is not None,
            "n_cognitive_gaps": len(self.cognitive_gaps),
            "has_working_memory": self.working_memory is not None,
            "has_cost_signal": self.latest_cost_signal is not None,
            "has_world_state": self.world_state is not None,
        }
        if self.world_state is not None and hasattr(self.world_state, "to_dict"):
            result["world_state"] = self.world_state.to_dict()
        return result

    # ────────────────────────────────────────────────
    # v3.1.0 JEPA: 因果图距离度量
    # ────────────────────────────────────────────────

    def _build_node_index(self) -> dict[str, int]:
        """从 causal_edges 中提取所有唯一节点并建立索引。

        支持两种边格式:
        - entity-level: cause="entity_name", effect="entity_name"
        - memory-level: cause_idx=0, effect_idx=1 (GaussianDAG 输出)
        """
        nodes: dict[str, int] = {}
        has_named_edges = False
        for e in self.causal_edges:
            for key in ("cause", "effect"):
                name = str(e.get(key, ""))
                if name and name not in nodes:
                    nodes[name] = len(nodes)
                    has_named_edges = True
        # Fallback: index-based edges from GaussianDAG
        if not has_named_edges:
            for e in self.causal_edges:
                for key in ("cause_idx", "effect_idx"):
                    idx = e.get(key, -1)
                    if idx >= 0:
                        name = f"n{idx}"
                        if name not in nodes:
                            nodes[name] = len(nodes)
        return nodes

    def _get_node_name(self, e: dict[str, Any], key: str) -> str:
        """从边中提取节点名称，兼容 cause/effect 和 cause_idx/effect_idx 两种格式。"""
        name = str(e.get(key, ""))
        if name:
            return name
        idx_key = f"{key}_idx"
        idx = e.get(idx_key, -1)
        if idx >= 0:
            return f"n{idx}"
        return ""

    def to_adjacency_matrix(self) -> np.ndarray:
        """
        构建 N×N 加权邻接矩阵。

        有因果边 → 权重 = rho（偏相关系数）
        无因果边 → 权重 = 0.0
        自环 → 0.0

        Returns:
            shape=(N, N) 的 float32 邻接矩阵
        """
        node_index = self._build_node_index()
        n = len(node_index)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32)
        adj = np.zeros((n, n), dtype=np.float32)
        for e in self.causal_edges:
            cause_name = self._get_node_name(e, "cause")
            effect_name = self._get_node_name(e, "effect")
            if cause_name in node_index and effect_name in node_index:
                i = node_index[cause_name]
                j = node_index[effect_name]
                adj[i, j] = float(e.get("rho", 0.0))
        return adj

    def to_node_feature_matrix(self) -> np.ndarray:
        """
        构建 N×D 节点特征矩阵。

        特征维度 D = 5 (活跃状态 one-hot) + 3 (度统计)
        - 活跃状态 one-hot (5): semantic/causal/spacetime/generative/trust
        - 出度 (1): 该节点作为 cause 的次数
        - 入度 (1): 该节点作为 effect 的次数
        - 度中心性 (1): (出度+入度) / (2*N)

        Returns:
            shape=(N, 8) 的 float32 特征矩阵
        """
        node_index = self._build_node_index()
        n = len(node_index)
        if n == 0:
            return np.zeros((0, 8), dtype=np.float32)

        features = np.zeros((n, 8), dtype=np.float32)
        five_states = ["semantic", "causal", "spacetime", "generative", "trust"]
        state_to_idx = {s: i for i, s in enumerate(five_states)}

        # 统计度
        out_degree = dict.fromkeys(node_index, 0)
        in_degree = dict.fromkeys(node_index, 0)
        for e in self.causal_edges:
            cause_name = self._get_node_name(e, "cause")
            effect_name = self._get_node_name(e, "effect")
            if cause_name in out_degree:
                out_degree[cause_name] += 1
            if effect_name in in_degree:
                in_degree[effect_name] += 1

        for name, idx in node_index.items():
            d_out = out_degree[name]
            d_in = in_degree[name]
            features[idx, 5] = float(d_out)
            features[idx, 6] = float(d_in)
            features[idx, 7] = (d_out + d_in) / max(2 * n, 1)

            # 活跃状态 one-hot（基于已有的 active_states）
            for state_name, si in state_to_idx.items():
                features[idx, si] = 1.0 if state_name in self.active_states else 0.0

        return features

    def state_distance(
        self,
        other: CausalWorldModelState,
        alpha_edges: float = 0.5,
        alpha_structure: float = 0.3,
        alpha_energy: float = 0.2,
        alpha_temporal: float = 0.15,
        alpha_belief: float = 0.15,
    ) -> float:
        """
        计算两个因果世界状态之间的距离。

        JEPA 训练损失的主项：
            L_pred = state_distance(s_pred, s_actual)

        v3.1.0: 融合因果图距离 + 时空距离 + 信念距离。
            L_total = (α_causal·L_causal + α_temporal·L_temporal + α_belief·L_belief) / Σα

        因果图距离子项：
        1. 边权重 L1 距离 (alpha_edges): 同一条边在两个状态间 rho 的差异
        2. 图结构 Jaccard 差异 (alpha_structure): 边集合的重叠度
        3. 能量守恒差异 (alpha_energy): 因果图总能量变化率

        时空/信念距离（仅在双方均有数据时激活）：
        4. 时空距离 (alpha_temporal): energy_type 不匹配比例
        5. 信念距离 (alpha_belief): 信念轨迹置信度变化

        Args:
            other: 另一个 CausalWorldModelState
            alpha_edges: 边权重距离权重
            alpha_structure: 图结构差异权重
            alpha_energy: 能量守恒差异权重
            alpha_temporal: 时空距离权重
            alpha_belief: 信念距离权重

        Returns:
            0.0 到 1.0 之间的距离标量
        """
        if not self.causal_edges and not other.causal_edges:
            return 0.0
        if not self.causal_edges or not other.causal_edges:
            return 1.0

        # ── 1. 边权重 L1 距离 ──
        self_adj = self.to_adjacency_matrix()
        other_adj = other.to_adjacency_matrix()
        n_max = max(self_adj.shape[0], other_adj.shape[0])
        if self_adj.shape[0] < n_max:
            padded = np.zeros((n_max, n_max), dtype=np.float32)
            padded[: self_adj.shape[0], : self_adj.shape[1]] = self_adj
            self_adj = padded
        if other_adj.shape[0] < n_max:
            padded = np.zeros((n_max, n_max), dtype=np.float32)
            padded[: other_adj.shape[0], : other_adj.shape[1]] = other_adj
            other_adj = padded

        edge_l1 = float(np.sum(np.abs(self_adj - other_adj)))
        total_rho = max(float(np.sum(self_adj) + np.sum(other_adj)), 1e-10)
        dist_edges = min(edge_l1 / total_rho, 1.0)

        # ── 2. 图结构 Jaccard 差异 ──
        self_edges_set = {
            (self._get_node_name(e, "cause"), self._get_node_name(e, "effect")) for e in self.causal_edges
        }
        other_edges_set = {
            (other._get_node_name(e, "cause"), other._get_node_name(e, "effect")) for e in other.causal_edges
        }
        intersection = len(self_edges_set & other_edges_set)
        union = len(self_edges_set | other_edges_set)
        if union > 0:
            jaccard_sim = intersection / union
            dist_structure = 1.0 - jaccard_sim
        else:
            dist_structure = 0.0

        # ── 3. 能量守恒差异 ──
        self_total_energy = sum(abs(e.get("rho", 0.0)) for e in self.causal_edges)
        other_total_energy = sum(abs(e.get("rho", 0.0)) for e in other.causal_edges)
        max_energy = max(self_total_energy, other_total_energy, 1e-10)
        dist_energy = abs(self_total_energy - other_total_energy) / max_energy

        # ── 4. v3.1.0: 时空距离 (energy_type 对齐) ──
        dist_temporal = 0.0
        has_temporal = False
        if self.temporal_info is not None and other.temporal_info is not None:
            has_temporal = True
            try:
                self_et = getattr(self.temporal_info, "energy_type", "")
                other_et = getattr(other.temporal_info, "energy_type", "")
                dist_temporal = 0.0 if self_et == other_et else 1.0
            except (AttributeError, TypeError):
                logger.warning("temporal distance calc failed", exc_info=True)

        # ── 5. v3.1.0: 信念距离 (置信度轨迹差异) ──
        dist_belief = 0.0
        has_belief = False
        if self.belief_tracker is not None and other.belief_tracker is not None:
            has_belief = True
            try:
                self_states = getattr(self.belief_tracker, "belief_states", {})
                other_states = getattr(other.belief_tracker, "belief_states", {})
                all_keys = set(self_states.keys()) | set(other_states.keys())
                if all_keys:
                    diffs = []
                    for k in all_keys:
                        sc = (
                            getattr(self_states.get(k), "confidence", 0.5)
                            if isinstance(self_states.get(k), object)
                            else 0.5
                        )
                        oc = (
                            getattr(other_states.get(k), "confidence", 0.5)
                            if isinstance(other_states.get(k), object)
                            else 0.5
                        )
                        diffs.append(abs(sc - oc))
                    dist_belief = sum(diffs) / len(diffs) if diffs else 0.0
            except (AttributeError, TypeError, ZeroDivisionError):
                logger.warning("belief distance calc failed", exc_info=True)

        # ── 加权求和 ──
        # 归一化: 只对活跃的组件分配权重
        causal_weight = alpha_edges + alpha_structure + alpha_energy
        total_weight = causal_weight
        distance = alpha_edges * dist_edges + alpha_structure * dist_structure + alpha_energy * dist_energy
        if has_temporal:
            total_weight += alpha_temporal
            distance += alpha_temporal * dist_temporal
        if has_belief:
            total_weight += alpha_belief
            distance += alpha_belief * dist_belief

        return min(float(distance / total_weight), 1.0)

    def __sub__(self, other: CausalWorldModelState) -> float:
        """
        操作符重载：`distance = abs(s_t1 - s_t)` 返回距离标量。

        等价于 self.state_distance(other)。
        """
        if not isinstance(other, CausalWorldModelState):
            return NotImplemented
        return self.state_distance(other)


# =============================================================================
# v3.0.1: WorkingMemory — Short-Term Memory 轨迹缓冲区
# =============================================================================


@dataclass
class TrajectoryStep:
    """
    单步轨迹记录。

    Attributes:
        state: 该时刻的因果世界状态
        cost_signal: 该时刻的代价评估结果
        step_index: 全局步序号
        timestamp: 时间戳 (time.time())
    """

    state: object  # CausalWorldModelState
    cost_signal: object | None = None  # CostSignal
    step_index: int = 0
    timestamp: float = field(default_factory=time.time)

    # v3.0.4: 时空间编码
    stem_branch_code: object | None = None  # StemBranchCode
    energy_state: object | None = None  # EnergyState
    temporal_weight: float = 1.0  # 时变重要性


@dataclass
class WorkingMemory:
    """
    LeCun 风格的短期记忆 / 工作记忆缓冲区。

    六态流转：IDLE → RECORDING → FULL → FLUSHING → IDLE

    Attributes:
        max_length: 最大轨迹步数（默认 10）
        trajectory: 轨迹步骤列表（FIFO）
    """

    max_length: int = 10
    trajectory: list[TrajectoryStep] = field(default_factory=list)
    _state: str = "IDLE"  # IDLE → RECORDING → FULL → FLUSHING → IDLE
    _energy_core: object | None = None  # v3.0.4
    _temporal_core: object | None = None  # v3.0.4

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_full(self) -> bool:
        return len(self.trajectory) >= self.max_length

    def push(self, step: TrajectoryStep) -> None:
        """压入一步轨迹，加权淘汰（v3.0.4: 优先淘汰 temporal_weight 最低的记录）。"""
        self._state = "RECORDING"

        # ── v3.0.4: 自动注入时空编码 ──
        from datetime import datetime

        if step.stem_branch_code is None and self._temporal_core is not None:
            now = datetime.now()
            step.stem_branch_code = self._temporal_core.create_code(  # type: ignore[attr-defined]
                stem_idx=now.year % 10,
                branch_idx=now.month - 1,
            )

        if step.energy_state is None and self._energy_core is not None:
            step.energy_state = self._energy_core.get_energy_state("spacetime", datetime.now().month - 1)  # type: ignore[attr-defined]
            strength_name = getattr(getattr(step.energy_state, "strength", None), "name", "")
            step.temporal_weight = {
                "WANG": 1.5,
                "XIANG": 1.2,
                "XIU": 0.8,
                "QIU": 0.5,
                "SI": 0.2,
            }.get(strength_name, 1.0)

        self.trajectory.append(step)

        # ── v3.0.4: 加权淘汰 — 优先淘汰 temporal_weight 最低的记录 ──
        if len(self.trajectory) > self.max_length:
            self.trajectory.sort(key=lambda s: getattr(s, "temporal_weight", 1.0))
            self.trajectory.pop(0)

        if self.is_full:
            self._state = "FULL"

    def get_recent(self, n: int = 3) -> list[TrajectoryStep]:
        """获取最近 n 步轨迹。"""
        if not self.trajectory:
            return []
        return self.trajectory[-n:]

    def get_recent_weighted(self, n: int = 3) -> list[TrajectoryStep]:
        """
        v3.0.4: 按时间距离 × 旺衰权重检索最近轨迹。

        候选池翻倍 (n*2)，用循环距离衰减 + temporal_weight 排序，
        返回 Top-N 步。

        Args:
            n: 返回步数

        Returns:
            加权排序后的轨迹步骤列表
        """
        import math

        if not self.trajectory:
            return []
        steps = self.trajectory[-n * 2 :]  # 候选池翻倍
        if self._temporal_core is not None and steps:
            last = steps[-1]
            now_idx = last.stem_branch_code.cycle_index if last.stem_branch_code is not None else 0  # type: ignore[attr-defined]
            for s in steps:
                if s.stem_branch_code is not None:
                    dist = self._temporal_core.get_cycle_distance(now_idx, s.stem_branch_code.cycle_index)  # type: ignore[attr-defined]
                    s.temporal_weight *= math.exp(-0.1 * dist)
        steps.sort(key=lambda s: getattr(s, "temporal_weight", 1.0), reverse=True)
        return steps[:n]

    def clear(self) -> None:
        """清空缓冲区，重置为 IDLE。"""
        self._state = "FLUSHING"
        self.trajectory.clear()
        self._state = "IDLE"

    def flush_to_experience_db(  # type: ignore[no-untyped-def]
        self,
        experience_db,
        tags: list[str] | None = None,
        context: dict | None = None,  # type: ignore
    ) -> list[str]:
        """将工作记忆轨迹刷入 ExperienceDB 作为经验记忆。

        v3.5.0: WorkingMemory ↔ ExperienceDB 集成。
        每步轨迹转化为一条 Experience，包含代价信号和时空间编码。

        Args:
            experience_db: ExperienceDB 实例
            tags: 语义标签（附加到所有经验）
            context: 上下文信息

        Returns:
            存储的经验 ID 列表
        """
        from mci_world_model.sdk._experience_memory import Experience, ExperienceType

        if not self.trajectory:
            return []

        exp_ids = []
        base_tags = tags or []
        base_context = context or {}

        for step in self.trajectory:
            # 根据代价信号判断经验类型
            cost = step.cost_signal
            if cost is not None and hasattr(cost, "total"):
                exp_type = ExperienceType.SUCCESS if cost.total < 0.5 else ExperienceType.FAILURE
            else:
                exp_type = ExperienceType.TRANSITION

            # 构建因果边（从状态属性推断）
            causal_edges = []
            state = step.state
            if state is not None:
                state_attrs = [a for a in dir(state) if not a.startswith("_")]
                for attr in state_attrs[:3]:  # 最多取 3 个属性作为因果边
                    causal_edges.append(("state", attr))

            # 构建标签
            step_tags = list(base_tags)
            if cost is not None and hasattr(cost, "total"):
                step_tags.append(f"cost_{cost.total:.1f}")

            # 构建上下文
            step_context = dict(base_context)
            step_context["step_index"] = step.step_index
            if step.stem_branch_code is not None:
                step_context["stem_branch"] = str(step.stem_branch_code)

            exp = Experience(
                experience_type=exp_type,
                tags=step_tags,
                causal_edges=causal_edges,
                outcome=f"step_{step.step_index}_cost_{cost.total:.4f}"
                if cost and hasattr(cost, "total")
                else f"step_{step.step_index}",
                context=step_context,
                timestamp=step.timestamp,
                importance=step.temporal_weight,
                state_snapshot=state,
            )
            exp_id = experience_db.store(exp)
            exp_ids.append(exp_id)

        # 清空工作记忆
        self.clear()
        return exp_ids

    def retrieve_experience_hints(  # type: ignore[no-untyped-def]
        self,
        experience_db,
        query_tags: list[str],
        top_k: int = 3,
    ):
        """从 ExperienceDB 检索相关经验以指导当前决策。

        v3.5.0: WorkingMemory ↔ ExperienceDB 集成。

        Args:
            experience_db: ExperienceDB 实例
            query_tags: 查询语义标签
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        return experience_db.retrieve(query_tags=query_tags, top_k=top_k)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_length": self.max_length,
            "n_steps": len(self.trajectory),
            "state": self._state,
            "recent_costs": [round(s.cost_signal.total, 6) if s.cost_signal else None for s in self.get_recent(3)],  # type: ignore[attr-defined]
        }


# =============================================================================
# MCIWorldModel
# =============================================================================
