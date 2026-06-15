"""MCI World Model — 因果图自适应更新器 (CausalUpdater)

CEWM v3.6.0 新增组件 (N4)：
基于新证据自动修正因果图结构（添加/删除/修正因果边）。

理论基础：
    1. Pearl Do-Calculus — 因果图结构决定可识别性
    2. Spirtes PC/FCI 算法 — 从条件独立检验发现因果结构
    3. Kalman 滤波思想 — 新证据增量更新信念

核心能力：
    - update(new_evidence) — 处理新观测证据
    - add_evidence(cause, effect, confidence, weight) — 手动添加证据
    - detect_inconsistencies() — 检测因果图中的不一致
    - auto_correct() — 自动修正低置信度边
    - history() — 查看更新历史

更新策略：
    - 高置信度 (> threshold_high): 确认边
    - 低置信度 (< threshold_low): 删除边
    - 冲突证据: 加权合并 → 重新评估置信度
    - 新边: 置信度从初始值开始累积
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# =============================================================================
# 数据类型
# =============================================================================


class EdgeAction(Enum):
    """边操作类型。"""

    ADD = "add"
    REMOVE = "remove"
    STRENGTHEN = "strengthen"
    WEAKEN = "weaken"
    CONFIRM = "confirm"
    CORRECT = "correct"  # 方向修正


@dataclass
class CausalEdge:
    """因果边及其置信度信息。

    Attributes:
        cause: 因节点
        effect: 果节点
        weight: 边权重 [0, 1]
        confidence: 置信度 [0, 1]
        evidence_count: 支持证据数
        contradiction_count: 矛盾证据数
        first_seen: 首次观测时间
        last_seen: 最近观测时间
        source: 来源标识
    """

    cause: str
    effect: str
    weight: float = 0.5
    confidence: float = 0.5
    evidence_count: int = 0
    contradiction_count: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    source: str = "default"

    @property
    def support_ratio(self) -> float:
        """支持比例 = evidence / (evidence + contradiction)。"""
        total = self.evidence_count + self.contradiction_count
        return self.evidence_count / total if total > 0 else 0.5

    def update_confidence(self) -> float:
        """基于证据重新计算置信度。

        公式：
            confidence = support_ratio × log(1 + evidence_count) / log(1 + evidence_count + contradiction_count)

        证据越多、矛盾越少，置信度越高。
        """
        if self.evidence_count == 0 and self.contradiction_count == 0:
            return self.confidence

        support = self.support_ratio
        ev_factor = math.log(1 + self.evidence_count) / math.log(2 + self.evidence_count + self.contradiction_count)
        self.confidence = support * ev_factor
        return self.confidence


@dataclass
class UpdateRecord:
    """一次更新操作的记录。"""

    action: EdgeAction
    cause: str
    effect: str
    old_confidence: float = 0.0
    new_confidence: float = 0.0
    reason: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class CausalUpdaterStats:
    """更新器统计信息。"""

    total_edges: int = 0
    avg_confidence: float = 0.0
    avg_weight: float = 0.0
    high_confidence_edges: int = 0
    low_confidence_edges: int = 0
    total_updates: int = 0
    edges_added: int = 0
    edges_removed: int = 0
    edges_strengthened: int = 0
    edges_weakened: int = 0
    edges_corrected: int = 0
    inconsistencies_found: int = 0


# =============================================================================
# CausalUpdater 主类
# =============================================================================


@dataclass
class CausalUpdater:
    """因果图自适应更新器。

    基于新证据自动修正因果图结构：
    - 高置信度证据 → 确认/加强边
    - 低置信度或矛盾证据 → 削弱/删除边
    - 新观测到的因果关系 → 添加新边
    - 方向错误 → 修正边方向

    Example:
        >>> updater = CausalUpdater()
        >>> updater.init_from_edges([("A", "B"), ("B", "C")])
        >>> updater.add_evidence("A", "B", confidence=0.9)
        >>> updater.add_evidence("D", "A", confidence=0.8)  # 新边
        >>> updater.add_contradiction("B", "C")  # 矛盾证据
        >>> changes = updater.auto_correct()
        >>> print(f"修正了 {len(changes)} 条边")
    """

    threshold_high: float = 0.7  # 高置信度阈值
    threshold_low: float = 0.2  # 低置信度阈值（低于此值将被删除）
    initial_confidence: float = 0.4  # 新边初始置信度
    learning_rate: float = 0.1  # 证据学习率
    max_edges: int = 10000  # 最大边数

    # 内部状态
    _edges: dict[tuple[str, str], CausalEdge] = field(default_factory=dict)
    _nodes: set[str] = field(default_factory=set)
    _history: list[UpdateRecord] = field(default_factory=list)
    _stats: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # ── 初始化 ──

    def init_from_edges(
        self,
        edges: list[tuple[str, str]],
        weights: list[float] | None = None,
        confidence: float = 0.8,
    ) -> None:
        """从边列表初始化因果图。

        Args:
            edges: [(cause, effect), ...]
            weights: 可选权重列表
            confidence: 初始置信度
        """
        for i, (cause, effect) in enumerate(edges):
            w = weights[i] if weights and i < len(weights) else 0.5
            self._add_edge(cause, effect, weight=w, confidence=confidence, source="init")

    def init_from_causal_graph(self, graph: Any) -> None:
        """从 CausalGraph 对象初始化。

        Args:
            graph: CausalGraph 实例
        """
        for node in graph.nodes:
            self._nodes.add(node)
        for cause, effect in graph.edges:
            self._add_edge(cause, effect, weight=0.5, confidence=0.7, source="causal_graph")

    # ── 核心更新 ──

    def update(self, new_evidence: dict[str, Any] | None = None, **kwargs: Any) -> list[UpdateRecord]:
        """处理新观测证据，自动修正因果图。

        支持多种输入格式：
            1. update({"cause": "A", "effect": "B", "confidence": 0.9})
            2. update(cause="A", effect="B", confidence=0.9)
            3. update({"edges": [("A", "B")], "confidence": 0.8})
            4. update({"contradiction": ("B", "C")})

        Returns:
            更新操作记录列表
        """
        if new_evidence is None:
            new_evidence = kwargs

        records: list[UpdateRecord] = []

        # 处理矛盾证据
        if "contradiction" in new_evidence:
            cause, effect = new_evidence["contradiction"]
            records.extend(self.add_contradiction(cause, effect))
            return records

        # 批量边
        if "edges" in new_evidence:
            confidence = new_evidence.get("confidence", 0.8)
            for cause, effect in new_evidence["edges"]:
                records.extend(self.add_evidence(cause, effect, confidence=confidence))
            return records

        # 单条证据
        cause = new_evidence.get("cause", kwargs.get("cause", ""))
        effect = new_evidence.get("effect", kwargs.get("effect", ""))
        confidence = new_evidence.get("confidence", kwargs.get("confidence", 0.8))
        weight = new_evidence.get("weight", kwargs.get("weight"))

        if cause and effect:
            records.extend(self.add_evidence(cause, effect, confidence=confidence, weight=weight))

        return records

    def add_evidence(
        self,
        cause: str,
        effect: str,
        confidence: float = 0.8,
        weight: float | None = None,
    ) -> list[UpdateRecord]:
        """添加一条支持性因果证据。

        Args:
            cause: 因节点
            effect: 果节点
            confidence: 证据置信度 [0, 1]
            weight: 可选边权重

        Returns:
            更新操作记录列表
        """
        records: list[UpdateRecord] = []
        key = (cause, effect)
        reverse_key = (effect, cause)

        if key in self._edges:
            # 已存在的边 → 加强
            edge = self._edges[key]
            old_conf = edge.confidence
            edge.evidence_count += 1
            edge.last_seen = time.time()

            # 贝叶斯更新
            edge.confidence = min(1.0, edge.confidence + self.learning_rate * confidence)
            if weight is not None:
                edge.weight = 0.7 * edge.weight + 0.3 * weight  # EMA 平滑

            edge.update_confidence()

            records.append(
                UpdateRecord(
                    action=EdgeAction.STRENGTHEN,
                    cause=cause,
                    effect=effect,
                    old_confidence=old_conf,
                    new_confidence=edge.confidence,
                    reason=f"支持证据 (conf={confidence:.2f})",
                )
            )
            self._stats["strengthened"] += 1

        elif reverse_key in self._edges:
            # 反向边存在 → 方向冲突，评估是否修正
            reverse_edge = self._edges[reverse_key]
            if confidence > reverse_edge.confidence + 0.2:
                # 新证据更强 → 修正方向
                old_conf = reverse_edge.confidence
                self._remove_edge_internal(reverse_key)
                self._add_edge(cause, effect, weight=weight or 0.5, confidence=confidence * 0.6, source="corrected")
                records.append(
                    UpdateRecord(
                        action=EdgeAction.CORRECT,
                        cause=cause,
                        effect=effect,
                        old_confidence=old_conf,
                        new_confidence=confidence * 0.6,
                        reason=f"方向修正: {effect}→{cause} → {cause}→{effect}",
                    )
                )
                self._stats["corrected"] += 1
            else:
                # 新证据不够强 → 记录矛盾
                reverse_edge.contradiction_count += 1
                records.append(
                    UpdateRecord(
                        action=EdgeAction.CONFIRM,
                        cause=effect,
                        effect=cause,
                        old_confidence=reverse_edge.confidence,
                        new_confidence=reverse_edge.confidence,
                        reason=f"反向证据不足 (conf={confidence:.2f} < {reverse_edge.confidence + 0.2:.2f})",
                    )
                )

        # 新边 → 添加
        elif len(self._edges) < self.max_edges:
            self._add_edge(
                cause,
                effect,
                weight=weight or 0.5,
                confidence=self.initial_confidence * confidence,
                source="evidence",
            )
            records.append(
                UpdateRecord(
                    action=EdgeAction.ADD,
                    cause=cause,
                    effect=effect,
                    old_confidence=0.0,
                    new_confidence=self.initial_confidence * confidence,
                    reason=f"新边发现 (conf={confidence:.2f})",
                )
            )
            self._stats["added"] += 1

        self._stats["total_updates"] += 1
        return records

    def add_contradiction(self, cause: str, effect: str) -> list[UpdateRecord]:
        """添加一条矛盾证据（削弱对应边）。

        Args:
            cause: 因节点
            effect: 果节点

        Returns:
            更新操作记录列表
        """
        records: list[UpdateRecord] = []
        key = (cause, effect)

        if key in self._edges:
            edge = self._edges[key]
            old_conf = edge.confidence
            edge.contradiction_count += 1
            edge.confidence = max(0.0, edge.confidence - self.learning_rate * 0.5)
            edge.update_confidence()

            records.append(
                UpdateRecord(
                    action=EdgeAction.WEAKEN,
                    cause=cause,
                    effect=effect,
                    old_confidence=old_conf,
                    new_confidence=edge.confidence,
                    reason="矛盾证据",
                )
            )
            self._stats["weakened"] += 1
        else:
            # 边不存在，矛盾证据无影响
            pass

        self._stats["total_updates"] += 1
        return records

    # ── 自动修正 ──

    def auto_correct(self) -> list[UpdateRecord]:
        """自动修正因果图：删除低置信度边，检测不一致。

        策略：
        1. 删除置信度 < threshold_low 的边
        2. 检测并报告不一致

        Returns:
            修正操作记录列表
        """
        records: list[UpdateRecord] = []

        # 1. 删除低置信度边
        to_remove = []
        for key, edge in self._edges.items():
            edge.update_confidence()
            if edge.confidence < self.threshold_low and edge.evidence_count > 0:
                to_remove.append(key)

        for key in to_remove:
            edge = self._edges[key]
            records.append(
                UpdateRecord(
                    action=EdgeAction.REMOVE,
                    cause=edge.cause,
                    effect=edge.effect,
                    old_confidence=edge.confidence,
                    new_confidence=0.0,
                    reason=f"低置信度 ({edge.confidence:.3f} < {self.threshold_low})",
                )
            )
            self._remove_edge_internal(key)
            self._stats["removed"] += 1

        # 2. 检测不一致
        inconsistencies = self.detect_inconsistencies()
        self._stats["inconsistencies_found"] = len(inconsistencies)

        self._history.extend(records)
        return records

    def detect_inconsistencies(self) -> list[dict[str, Any]]:
        """检测因果图中的不一致。

        检查项：
        1. 双向边（A→B 且 B→A）
        2. 自环（A→A）
        3. 孤立节点（无入边无出边）
        4. 置信度与证据不匹配

        Returns:
            不一致列表 [{type, nodes, detail}, ...]
        """
        issues: list[dict[str, Any]] = []

        # 1. 双向边
        for c, e in list(self._edges.keys()):
            if (e, c) in self._edges:
                issues.append(
                    {
                        "type": "bidirectional",
                        "nodes": [c, e],
                        "detail": f"双向边: {c}↔{e}",
                    }
                )

        # 2. 自环
        for c, e in self._edges:
            if c == e:
                issues.append(
                    {
                        "type": "self_loop",
                        "nodes": [c],
                        "detail": f"自环: {c}→{c}",
                    }
                )

        # 3. 孤立节点
        connected = set()
        for c, e in self._edges:
            connected.add(c)
            connected.add(e)
        for node in self._nodes - connected:
            issues.append(
                {
                    "type": "isolated_node",
                    "nodes": [node],
                    "detail": f"孤立节点: {node}",
                }
            )

        # 4. 置信度异常
        for key, edge in self._edges.items():
            if edge.evidence_count > 3 and edge.confidence < 0.3:
                issues.append(
                    {
                        "type": "low_confidence_high_evidence",
                        "nodes": [edge.cause, edge.effect],
                        "detail": (
                            f"高证据低置信: {edge.cause}→{edge.effect} "
                            f"(evidence={edge.evidence_count}, conf={edge.confidence:.3f})"
                        ),
                    }
                )

        return issues

    # ── 查询 ──

    def get_edge(self, cause: str, effect: str) -> CausalEdge | None:
        """获取指定因果边。"""
        return self._edges.get((cause, effect))

    def get_edges(self) -> list[CausalEdge]:
        """获取所有因果边。"""
        return list(self._edges.values())

    def get_nodes(self) -> set[str]:
        """获取所有节点。"""
        return set(self._nodes)

    def get_parents(self, node: str) -> list[str]:
        """获取节点的所有父节点。"""
        return [edge.cause for edge in self._edges.values() if edge.effect == node]

    def get_children(self, node: str) -> list[str]:
        """获取节点的所有子节点。"""
        return [edge.effect for edge in self._edges.values() if edge.cause == node]

    def has_edge(self, cause: str, effect: str) -> bool:
        """检查是否存在因果边。"""
        return (cause, effect) in self._edges

    @property
    def n_edges(self) -> int:
        """边数量。"""
        return len(self._edges)

    @property
    def n_nodes(self) -> int:
        """节点数量。"""
        return len(self._nodes)

    def history(self) -> list[UpdateRecord]:
        """获取更新历史。"""
        return list(self._history)

    def statistics(self) -> CausalUpdaterStats:
        """更新器统计信息。"""
        edges = list(self._edges.values())
        if not edges:
            return CausalUpdaterStats()

        n = len(edges)
        total_conf = sum(e.confidence for e in edges)
        total_weight = sum(e.weight for e in edges)
        high_conf = sum(1 for e in edges if e.confidence >= self.threshold_high)
        low_conf = sum(1 for e in edges if e.confidence < self.threshold_low)

        return CausalUpdaterStats(
            total_edges=n,
            avg_confidence=total_conf / n,
            avg_weight=total_weight / n,
            high_confidence_edges=high_conf,
            low_confidence_edges=low_conf,
            total_updates=self._stats.get("total_updates", 0),
            edges_added=self._stats.get("added", 0),
            edges_removed=self._stats.get("removed", 0),
            edges_strengthened=self._stats.get("strengthened", 0),
            edges_weakened=self._stats.get("weakened", 0),
            edges_corrected=self._stats.get("corrected", 0),
            inconsistencies_found=self._stats.get("inconsistencies_found", 0),
        )

    def to_causal_graph(self) -> Any:
        """导出为 CausalGraph 对象。

        Returns:
            CausalGraph 实例
        """
        from mci_world_model.sdk._do_calculus import CausalGraph

        nodes = sorted(self._nodes)
        edges = [(e.cause, e.effect) for e in self._edges.values()]
        return CausalGraph(nodes=nodes, edges=edges)

    def clear(self) -> None:
        """清空更新器。"""
        self._edges.clear()
        self._nodes.clear()
        self._history.clear()
        self._stats.clear()

    # ── 内部方法 ──

    def _add_edge(
        self,
        cause: str,
        effect: str,
        weight: float = 0.5,
        confidence: float = 0.5,
        source: str = "default",
    ) -> None:
        """内部添加因果边。"""
        key = (cause, effect)
        if key not in self._edges:
            self._edges[key] = CausalEdge(
                cause=cause,
                effect=effect,
                weight=weight,
                confidence=confidence,
                evidence_count=1,
                source=source,
            )
            self._nodes.add(cause)
            self._nodes.add(effect)

    def _remove_edge_internal(self, key: tuple[str, str]) -> None:
        """内部移除因果边。"""
        self._edges.pop(key, None)

    def reset_stats(self) -> None:
        """重置统计计数器。"""
        self._stats.clear()
        self._history.clear()
