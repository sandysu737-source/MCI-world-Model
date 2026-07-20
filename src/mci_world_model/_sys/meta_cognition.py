"""
MCI World Model v4.6.0 — MetaCognition 统一元认知系统
=====================================================

v3.4.0: 合并重写 — 统一 meta_cognition.py (56行) + awareness.py (314行)
       新增: 根因分析链 + 认知评分 + 策略推荐 + 置信度冲突检测

核心能力:
    1. discover_gaps()    — 发现认知空洞（领域/时序/因果）
    2. detect_conflicts() — 检测信念冲突（内容矛盾 + 置信度冲突）
    3. get_aging()        — 知识老化预警
    4. get_suggestions()  — 主动建议（基于当前空洞）
    5. root_cause_analysis() — 根因分析链（VSM System 3*）
    6. cognitive_score()  — 认知健康度评分（六维）
    7. recommend_strategy()  — 策略推荐（学习型策略，VSM System 5）

理论对标:
    - Beer VSM System 5 (政策与身份) — 学习型策略
    - Beer VSM System 3* (异常审计) — 根因分析链
    - CEWM 认知诊断统一接口

向后兼容:
    - discover_gaps(types, domains, memories) → list[dict] (v3.3.0 接口)
    - discover_gaps(memory_types=..., user_domains=..., memory_list=...) → list[CognitiveGap] (awareness 接口)
    - detect_conflicts(beliefs) → list[dict]
    - get_aging(memories) → list[dict]
    - _contradicts(text_a, text_b) → bool
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

# =============================================================================
# 数据结构
# =============================================================================


@dataclass
class CognitiveGap:
    """认知空洞"""

    gap_id: str
    gap_type: str  # "domain" | "temporal" | "causal"
    description: str
    severity: float  # 0-1
    suggestions: list[str]
    discovered_at: float

    def __getitem__(self, key: str) -> Any:
        """支持 dict 式访问（向后兼容 v3.3.0 测试）。"""
        _map = {
            "type": self.gap_type,
            "gap_type": self.gap_type,
            "gap_id": self.gap_id,
            "severity": self.severity,
            "description": self.description,
            "suggestions": self.suggestions,
            "discovered_at": self.discovered_at,
        }
        if key in _map:
            return _map[key]
        raise KeyError(key)


@dataclass
class KnowledgeAging:
    """知识老化"""

    memory_id: str
    days_since_update: int
    current_stage: str
    severity: str  # "normal" | "warning" | "critical"
    suggestion: str


@dataclass
class RootCauseNode:
    """根因分析链节点"""

    node_id: str
    layer: str  # "signal" | "state" | "causal" | "prediction" | "action"
    description: str
    contribution: float  # 0-1，对最终异常的贡献度
    children: list[RootCauseNode] = field(default_factory=list)


@dataclass
class CognitiveScoreCard:
    """六维认知健康度评分卡"""

    causal_discovery: float = 0.0  # 因果发现能力
    counterfactual: float = 0.0  # 反事实推理能力
    ood_generalization: float = 0.0  # OOD 泛化能力
    explainability: float = 0.0  # 可解释性
    memory_reuse: float = 0.0  # 记忆复用能力
    anomaly_detection: float = 0.0  # 异常检测能力

    @property
    def total(self) -> float:
        return (
            self.causal_discovery
            + self.counterfactual
            + self.ood_generalization
            + self.explainability
            + self.memory_reuse
            + self.anomaly_detection
        ) / 6.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "causal_discovery": round(self.causal_discovery, 4),
            "counterfactual": round(self.counterfactual, 4),
            "ood_generalization": round(self.ood_generalization, 4),
            "explainability": round(self.explainability, 4),
            "memory_reuse": round(self.memory_reuse, 4),
            "anomaly_detection": round(self.anomaly_detection, 4),
            "total": round(self.total, 4),
        }


# =============================================================================
# MetaCognition — 统一元认知系统
# =============================================================================


class MetaCognition:
    """
    元认知系统 — 统一接口

    功能：
    1. 发现认知空洞（用户知识的盲区）
    2. 检测信念冲突
    3. 预警知识老化
    4. 提供主动建议
    5. 根因分析链（v3.4.0）
    6. 认知健康度评分（v3.4.0）
    7. 策略推荐（v3.4.0）

    对外隐藏：发现算法、阈值配置
    """

    # 认知空洞检测阈值
    DOMAIN_COVERAGE_THRESHOLD = 0.7
    TEMPORAL_GAP_DAYS = 90
    CONFLICT_SEVERITY_THRESHOLD = 0.8

    # 知识老化阈值
    AGING_WARNING_DAYS = 30
    AGING_CRITICAL_DAYS = 60

    def __init__(self):
        self._gaps: list[CognitiveGap] = []
        self._last_scan = 0
        self._scan_interval = 3600  # 每小时最多扫描一次

        # v3.4.0: 根因分析历史
        self._root_cause_history: list[RootCauseNode] = []

        # v3.4.0: 认知评分历史
        self._score_history: list[CognitiveScoreCard] = []

        # v3.4.0: 策略推荐记忆
        self._strategy_log: list[dict[str, Any]] = []

    # =================================================================
    # discover_gaps — 认知空洞发现（双接口兼容）
    # =================================================================

    def discover_gaps(
        self,
        types: dict | None = None,
        domains: list | None = None,
        memories: list | None = None,
        *,
        memory_types: dict[str, int] | None = None,
        user_domains: list[str] | None = None,
        memory_list: list[dict] | None = None,
    ) -> list:
        """
        发现认知空洞（双接口兼容）。

        v3.3.0 兼容接口: discover_gaps(types, domains, memories) → list[dict]
        v3.4.0 统一接口: discover_gaps(memory_types=..., user_domains=..., memory_list=...) → list[CognitiveGap]

        当使用关键字参数时返回 list[CognitiveGap]；
        当使用位置参数且参数为旧格式时返回 list[dict]。
        """
        # 接口路由：优先使用关键字参数
        if memory_types is not None or user_domains is not None or memory_list is not None:
            return self._discover_gaps_typed(
                memory_types or {},
                user_domains or [],
                memory_list or [],
            )

        # 旧接口兼容
        _types = types or {}
        _domains = domains or []
        _memories = memories or []
        return self._discover_gaps_legacy(_types, _domains, _memories)

    def _discover_gaps_legacy(self, types: dict, domains: list, memories: list[dict]) -> list[CognitiveGap]:
        """旧版 discover_gaps 接口 (v3.3.0 兼容，返回 CognitiveGap 对象)。"""
        # 委托给 typed 路径（统一检测逻辑）
        return self._discover_gaps_typed(
            memory_types=types,
            user_domains=domains,
            memory_list=memories,
        )

    def _discover_gaps_typed(
        self, memory_types: dict[str, int], user_domains: list[str], memory_list: list[dict]
    ) -> list[CognitiveGap]:
        """新版 discover_gaps 接口 (v3.4.0 统一版)。"""
        gaps: list[CognitiveGap] = []
        now = time.time()

        # 1. 领域覆盖空洞
        domain_gaps = self._detect_domain_gaps(memory_types, user_domains)
        gaps.extend(domain_gaps)

        # 2. 时序空洞
        temporal_gaps = self._detect_temporal_gaps(memory_list)
        gaps.extend(temporal_gaps)

        # 3. 因果空洞
        causal_gaps = self._detect_causal_gaps(memory_list)
        gaps.extend(causal_gaps)

        # 更新存储
        self._gaps = gaps
        self._last_scan = now

        return gaps

    def _detect_domain_gaps(self, memory_types: dict[str, int], user_domains: list[str]) -> list[CognitiveGap]:
        """检测领域覆盖空洞"""
        gaps: list[CognitiveGap] = []
        total_memories = sum(memory_types.values())
        if total_memories == 0:
            return gaps

        type_ratios = {k: v / total_memories for k, v in memory_types.items()}

        # 事实类记忆过少
        if type_ratios.get("fact", 0) < 0.3:
            gaps.append(
                CognitiveGap(
                    gap_id=f"domain_fact_{int(time.time())}",
                    gap_type="domain",
                    description="事实类记忆偏少，可能影响判断准确性",
                    severity=0.7,
                    suggestions=["补充更多基础事实信息", "建立知识库"],
                    discovered_at=time.time(),
                )
            )

        # 偏好类记忆过少
        if type_ratios.get("preference", 0) < 0.1:
            gaps.append(
                CognitiveGap(
                    gap_id=f"domain_pref_{int(time.time())}",
                    gap_type="domain",
                    description="用户偏好信息不足，可能影响个性化服务",
                    severity=0.6,
                    suggestions=["收集用户偏好", "记录用户选择"],
                    discovered_at=time.time(),
                )
            )

        # 事件类记忆过多（可能有噪音）
        if type_ratios.get("event", 0) > 0.5:
            gaps.append(
                CognitiveGap(
                    gap_id=f"domain_event_{int(time.time())}",
                    gap_type="domain",
                    description="事件记忆占比过高，可能需要整理",
                    severity=0.5,
                    suggestions=["对事件进行归纳总结", "提取关键规律"],
                    discovered_at=time.time(),
                )
            )

        return gaps

    def _detect_temporal_gaps(self, memory_list: list[dict]) -> list[CognitiveGap]:
        """检测时序空洞"""
        gaps: list[CognitiveGap] = []
        now = time.time()

        type_last_update: dict[str, float] = defaultdict(lambda: 0)
        type_counts: dict[str, int] = defaultdict(int)

        for mem in memory_list:
            mem_type = mem.get("type", "fact")
            timestamp = mem.get("timestamp", 0)
            type_counts[mem_type] += 1
            type_last_update[mem_type] = max(type_last_update[mem_type], timestamp)

        for mem_type, last_update in type_last_update.items():
            days_elapsed = (now - last_update) / (24 * 3600)

            if days_elapsed > self.AGING_CRITICAL_DAYS and type_counts[mem_type] > 3:
                gaps.append(
                    CognitiveGap(
                        gap_id=f"temporal_{mem_type}_{int(time.time())}",
                        gap_type="temporal",
                        description=f"{mem_type}类记忆已{days_elapsed:.0f}天未更新，可能已过时",
                        severity=0.8,
                        suggestions=[f"更新{mem_type}类信息", "检查最新动态"],
                        discovered_at=now,
                    )
                )

        return gaps

    def _detect_causal_gaps(self, memory_list: list[dict]) -> list[CognitiveGap]:
        """检测因果空洞（孤立的记忆节点）"""
        gaps: list[CognitiveGap] = []

        isolated_count = 0
        for mem in memory_list:
            if not mem.get("causal_parents") and not mem.get("causal_children"):
                isolated_count += 1

        total = len(memory_list)
        if total > 10 and isolated_count / total > 0.8:
            gaps.append(
                CognitiveGap(
                    gap_id=f"causal_isolated_{int(time.time())}",
                    gap_type="causal",
                    description="大量记忆缺乏关联，建议建立记忆间的因果联系",
                    severity=0.6,
                    suggestions=["主动建立记忆关联", "分析记忆间的因果关系"],
                    discovered_at=time.time(),
                )
            )

        return gaps

    # =================================================================
    # detect_conflicts — 信念冲突检测（双接口兼容）
    # =================================================================

    def detect_conflicts(self, beliefs: dict) -> list[dict[str, Any]]:
        """
        检测信念冲突（双接口兼容）。

        v3.3.0 格式: beliefs = {id: {"content": str}}
        v3.4.0 格式: beliefs = {id: {"content": str, "confidence": float, "stage": str}}
        """
        # 检测输入格式
        first_val = next(iter(beliefs.values()), None) if beliefs else None
        has_typed_format = first_val is not None and "confidence" in first_val and "stage" in first_val

        if has_typed_format:
            return self._detect_conflicts_typed(beliefs)

        return self._detect_conflicts_legacy(beliefs)

    def _detect_conflicts_legacy(self, beliefs: dict) -> list[dict[str, Any]]:
        """旧版冲突检测 (v3.3.0 兼容)。"""
        conflicts: list[dict[str, Any]] = []
        ids = list(beliefs.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a_content = beliefs[ids[i]].get("content", "")
                b_content = beliefs[ids[j]].get("content", "")
                if self._contradicts(a_content, b_content):
                    conflicts.append({"memory_a": ids[i], "memory_b": ids[j], "severity": 0.8})
        return sorted(conflicts, key=lambda x: -x["severity"])

    def _detect_conflicts_typed(self, beliefs: dict[str, dict]) -> list[dict[str, Any]]:
        """新版冲突检测 (v3.4.0 统一版)。"""
        conflicts: list[dict[str, Any]] = []
        memory_ids = list(beliefs.keys())

        for i, id_a in enumerate(memory_ids):
            for id_b in memory_ids[i + 1 :]:
                state_a = beliefs[id_a]
                state_b = beliefs[id_b]

                # 两者置信度都高但阶段对立
                if (
                    state_a.get("confidence", 0) > 0.7
                    and state_b.get("confidence", 0) > 0.7
                    and state_a.get("stage", "") in ["强化", "确认"]
                    and state_b.get("stage", "") in ["强化", "确认"]
                ):
                    content_a = state_a.get("content", "")[:100].lower()
                    content_b = state_b.get("content", "")[:100].lower()

                    if self._is_contradictory(content_a, content_b):
                        conflicts.append(
                            {
                                "memory_a": id_a,
                                "memory_b": id_b,
                                "severity": (state_a["confidence"] + state_b["confidence"]) / 2,
                                "description": "两个高置信度信念存在内容矛盾",
                                "stage_a": state_a["stage"],
                                "stage_b": state_b["stage"],
                            }
                        )

        conflicts.sort(key=lambda x: x["severity"], reverse=True)
        return conflicts

    def _contradicts(self, text_a: str, text_b: str) -> bool:
        """简单判断内容是否矛盾（v3.3.0 兼容接口）。"""
        pos = ["是", "有", "正确", "知道"]
        neg = ["不是", "没有", "错误", "不知道"]
        a_pos = sum(1 for p in pos if p in text_a)
        b_pos = sum(1 for p in pos if p in text_b)
        a_neg = sum(1 for n in neg if n in text_a)
        b_neg = sum(1 for n in neg if n in text_b)
        return (a_pos and b_neg) or (a_neg and b_pos) > 0

    def _is_contradictory(self, content_a: str, content_b: str) -> bool:
        """判断内容是否矛盾（v3.4.0 增强版）。"""
        positive = ["是", "有", "正确", "可以", "知道", "喜欢", "能"]
        negative = ["不是", "没有", "错误", "不能", "不知道", "讨厌", "否"]

        pos_a = any(w in content_a for w in positive)
        neg_a = any(w in content_a for w in negative)
        pos_b = any(w in content_b for w in positive)
        neg_b = any(w in content_b for w in negative)

        return bool((pos_a and neg_b) or (pos_b and neg_a))

    # =================================================================
    # get_aging — 知识老化预警（双接口兼容）
    # =================================================================

    def get_aging(self, memories: list[dict]) -> list[dict]:
        """
        知识老化预警 (v3.3.0 兼容接口)。

        Returns:
            [{"id": str, "days": int, "severity": "warning"|"critical"}]
        """
        aging: list[dict] = []
        now = time.time()
        for m in memories:
            days = (now - m.get("timestamp", now)) / 86400
            if 14 < days < 30:
                aging.append({"id": m.get("id", ""), "days": round(days), "severity": "warning"})
            elif days >= 30:
                aging.append({"id": m.get("id", ""), "days": round(days), "severity": "critical"})
        return aging

    def get_aging_warnings(self, memory_list: list[dict]) -> list[KnowledgeAging]:
        """获取知识老化预警 (v3.4.0 增强版，返回 KnowledgeAging 对象)。"""
        warnings: list[KnowledgeAging] = []
        now = time.time()

        for mem in memory_list:
            days_elapsed = (now - mem.get("timestamp", now)) / (24 * 3600)

            if days_elapsed > self.AGING_CRITICAL_DAYS:
                severity = "critical"
            elif days_elapsed > self.AGING_WARNING_DAYS:
                severity = "warning"
            else:
                continue

            warnings.append(
                KnowledgeAging(
                    memory_id=mem.get("id", ""),
                    days_since_update=int(days_elapsed),
                    current_stage=mem.get("stage", "未知"),
                    severity=severity,
                    suggestion="建议更新或验证此记忆的准确性",
                )
            )

        return warnings

    # =================================================================
    # get_suggestions — 主动建议
    # =================================================================

    def get_suggestions(self) -> list[str]:
        """获取主动建议（基于当前空洞）"""
        suggestions: list[str] = []
        for gap in self._gaps:
            suggestions.extend(gap.suggestions)

        # 去重，保持顺序
        seen: set[str] = set()
        unique: list[str] = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique.append(s)

        return unique[:5]

    # =================================================================
    # root_cause_analysis — 根因分析链 (v3.4.0, VSM System 3*)
    # =================================================================

    def root_cause_analysis(
        self,
        surprise_signals: list[dict[str, Any]],
        causal_graph: dict[str, list[str]] | None = None,
        world_state_info: dict[str, Any] | None = None,
    ) -> RootCauseNode:
        """
        根因分析链 — 从惊奇信号追溯到根因。

        VSM System 3* (异常审计):
            surprise → 定位异常层 → 因果图追溯 → 根因节点

        Args:
            surprise_signals: 惊奇信号列表
                [{"score": float, "breakdown": {...}, "source": str}, ...]
            causal_graph: 因果图邻接表 {node: [children]}
            world_state_info: 世界状态附加信息

        Returns:
            RootCauseNode 根因分析树
        """
        if not surprise_signals:
            root = RootCauseNode(
                node_id="no_signal",
                layer="signal",
                description="无惊奇信号，无需分析",
                contribution=0.0,
            )
            return root

        # 1. 聚合惊奇信号
        avg_score = sum(s.get("score", 0) for s in surprise_signals) / len(surprise_signals)
        max_signal = max(surprise_signals, key=lambda s: s.get("score", 0))

        # 2. 定位异常层
        breakdown = max_signal.get("breakdown", {})
        anomaly_layer = self._identify_anomaly_layer(breakdown)

        # 3. 构建根因树
        root = RootCauseNode(
            node_id=f"root_{int(time.time())}",
            layer=anomaly_layer,
            description=f"综合惊奇度 {avg_score:.4f}，主要异常在{anomaly_layer}层",
            contribution=1.0,
        )

        # 4. 分解子原因
        # 4a. 信号层原因
        if breakdown.get("state_distance", 0) > 0.3:
            root.children.append(
                RootCauseNode(
                    node_id="signal_distance",
                    layer="signal",
                    description=f"状态距离偏大 (state_distance={breakdown['state_distance']:.4f})",
                    contribution=breakdown["state_distance"],
                )
            )

        # 4b. 预测层原因
        if breakdown.get("vector_deviation", 0) > 0.3:
            root.children.append(
                RootCauseNode(
                    node_id="prediction_deviation",
                    layer="prediction",
                    description=f"预测偏差显著 (vector_deviation={breakdown['vector_deviation']:.4f})",
                    contribution=breakdown["vector_deviation"],
                )
            )

        # 4c. 认知层原因
        if breakdown.get("direction_error", 0) > 0.3:
            root.children.append(
                RootCauseNode(
                    node_id="cognition_direction",
                    layer="causal",
                    description=f"方向误差偏高 (direction_error={breakdown['direction_error']:.4f})",
                    contribution=breakdown["direction_error"],
                )
            )

        # 5. 因果图追溯
        if causal_graph:
            for child in root.children:
                self._trace_causal_graph(child, causal_graph, depth=2)

        # 6. 记录历史
        self._root_cause_history.append(root)
        if len(self._root_cause_history) > 100:
            self._root_cause_history = self._root_cause_history[-100:]

        return root

    def _identify_anomaly_layer(self, breakdown: dict[str, float]) -> str:
        """基于惊奇度分解识别主要异常层。"""
        if not breakdown:
            return "unknown"

        max_key = max(breakdown, key=lambda k: breakdown[k])
        layer_map = {
            "state_distance": "signal",
            "vector_deviation": "prediction",
            "direction_error": "causal",
        }
        return layer_map.get(max_key, "unknown")

    def _trace_causal_graph(self, node: RootCauseNode, graph: dict[str, list[str]], depth: int) -> None:
        """在因果图上递归追溯根因。"""
        if depth <= 0:
            return

        # 查找与当前节点相关的因果链
        for parent_id, children in graph.items():
            for child_id in children:
                if child_id in node.node_id or parent_id in node.node_id:
                    child_node = RootCauseNode(
                        node_id=f"causal_{parent_id}_{child_id}",
                        layer="causal",
                        description=f"因果链: {parent_id} → {child_id}",
                        contribution=node.contribution * 0.7,
                    )
                    node.children.append(child_node)
                    self._trace_causal_graph(child_node, graph, depth - 1)
                    break

    # =================================================================
    # cognitive_score — 认知健康度评分 (v3.4.0)
    # =================================================================

    def cognitive_score(
        self,
        causal_edges_count: int = 0,
        total_memories: int = 0,
        counterfactual_queries: int = 0,
        prediction_accuracy: float = 0.0,
        anomaly_detection_rate: float = 0.0,
        memory_reuse_rate: float = 0.0,
        explainability_score: float = 0.0,
    ) -> CognitiveScoreCard:
        """
        六维认知健康度评分。

        Args:
            causal_edges_count: 因果图中的边数
            total_memories: 总记忆数
            counterfactual_queries: 反事实查询次数
            prediction_accuracy: 预测准确率 [0,1]
            anomaly_detection_rate: 异常检测率 [0,1]
            memory_reuse_rate: 记忆复用率 [0,1]
            explainability_score: 可解释性评分 [0,1]

        Returns:
            CognitiveScoreCard
        """
        # 1. 因果发现能力
        causal_score = min(1.0, causal_edges_count / max(total_memories, 1))
        if total_memories > 0:
            causal_score = min(1.0, causal_edges_count / total_memories * 2)
        else:
            causal_score = 0.0

        # 2. 反事实推理能力
        cf_score = min(1.0, counterfactual_queries / 10.0)

        # 3. OOD 泛化能力（基于预测准确率的代理指标）
        ood_score = prediction_accuracy * 0.8 + 0.2 * min(1.0, total_memories / 100.0)

        # 4. 可解释性
        exp_score = explainability_score

        # 5. 记忆复用能力
        reuse_score = memory_reuse_rate

        # 6. 异常检测能力
        anomaly_score = anomaly_detection_rate

        scorecard = CognitiveScoreCard(
            causal_discovery=round(causal_score, 4),
            counterfactual=round(cf_score, 4),
            ood_generalization=round(ood_score, 4),
            explainability=round(exp_score, 4),
            memory_reuse=round(reuse_score, 4),
            anomaly_detection=round(anomaly_score, 4),
        )

        self._score_history.append(scorecard)
        if len(self._score_history) > 100:
            self._score_history = self._score_history[-100:]

        return scorecard

    # =================================================================
    # recommend_strategy — 策略推荐 (v3.4.0, VSM System 5)
    # =================================================================

    def recommend_strategy(
        self,
        gaps: list | None = None,
        conflicts: list[dict] | None = None,
        scorecard: CognitiveScoreCard | None = None,
    ) -> list[dict[str, Any]]:
        """
        策略推荐 — VSM System 5 学习型策略。

        基于当前认知空洞、冲突和评分，推荐改进策略。

        Args:
            gaps: 认知空洞列表
            conflicts: 冲突列表
            scorecard: 认知评分卡

        Returns:
            策略推荐列表 [{"strategy": str, "priority": float, "reason": str, "expected_impact": str}]
        """
        strategies: list[dict[str, Any]] = []

        # 基于空洞推荐
        if gaps:
            for gap in gaps:
                gap_type = gap.get("type", "") if isinstance(gap, dict) else getattr(gap, "gap_type", "")
                severity = gap.get("severity", 0.5) if isinstance(gap, dict) else getattr(gap, "severity", 0.5)

                if gap_type == "domain":
                    strategies.append(
                        {
                            "strategy": "知识补全",
                            "priority": severity,
                            "reason": "领域覆盖空洞，需补充基础事实",
                            "expected_impact": "提升因果发现能力 10-20%",
                        }
                    )
                elif gap_type == "temporal":
                    strategies.append(
                        {
                            "strategy": "时序刷新",
                            "priority": severity,
                            "reason": "时序空洞，需更新过时知识",
                            "expected_impact": "提升预测准确率 5-15%",
                        }
                    )
                elif gap_type == "causal":
                    strategies.append(
                        {
                            "strategy": "因果关联建设",
                            "priority": severity,
                            "reason": "因果空洞，需建立记忆间关联",
                            "expected_impact": "提升推理深度 2-3 层",
                        }
                    )

        # 基于冲突推荐
        if conflicts:
            for conflict in conflicts:
                strategies.append(
                    {
                        "strategy": "冲突消解",
                        "priority": conflict.get("severity", 0.5),
                        "reason": f"信念冲突: {conflict.get('memory_a', '?')} vs {conflict.get('memory_b', '?')}",
                        "expected_impact": "消除矛盾信念，提升一致性",
                    }
                )

        # 基于评分推荐
        if scorecard:
            scores = scorecard.to_dict()
            min_dim = min(
                [
                    "causal_discovery",
                    "counterfactual",
                    "ood_generalization",
                    "explainability",
                    "memory_reuse",
                    "anomaly_detection",
                ],
                key=lambda d: scores.get(d, 0),
            )
            dim_labels = {
                "causal_discovery": "因果发现",
                "counterfactual": "反事实推理",
                "ood_generalization": "OOD 泛化",
                "explainability": "可解释性",
                "memory_reuse": "记忆复用",
                "anomaly_detection": "异常检测",
            }
            strategies.append(
                {
                    "strategy": f"强化{dim_labels.get(min_dim, min_dim)}",
                    "priority": 1.0 - scores.get(min_dim, 0),
                    "reason": f"{dim_labels.get(min_dim, min_dim)}是当前最弱维度 ({scores.get(min_dim, 0):.2f})",
                    "expected_impact": "提升综合认知评分 5-10%",
                }
            )

        # 按优先级排序
        strategies.sort(key=lambda s: s["priority"], reverse=True)

        # 记录
        self._strategy_log.extend(strategies)
        if len(self._strategy_log) > 200:
            self._strategy_log = self._strategy_log[-200:]

        return strategies[:5]

    # =================================================================
    # get_score_history / get_root_cause_history — 历史查询
    # =================================================================

    def get_score_history(self) -> list[CognitiveScoreCard]:
        """获取认知评分历史。"""
        return list(self._score_history)

    def get_root_cause_history(self) -> list[RootCauseNode]:
        """获取根因分析历史。"""
        return list(self._root_cause_history)

    def get_strategy_log(self) -> list[dict[str, Any]]:
        """获取策略推荐日志。"""
        return list(self._strategy_log)

    # =================================================================
    # reset — 重置
    # =================================================================

    def reset(self) -> None:
        """重置所有状态。"""
        self._gaps.clear()
        self._last_scan = 0
        self._root_cause_history.clear()
        self._score_history.clear()
        self._strategy_log.clear()

    def __repr__(self) -> str:
        return (
            f"MetaCognition(gaps={len(self._gaps)}, "
            f"scores={len(self._score_history)}, "
            f"root_causes={len(self._root_cause_history)})"
        )
