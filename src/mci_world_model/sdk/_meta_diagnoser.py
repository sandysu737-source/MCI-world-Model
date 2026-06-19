from __future__ import annotations

"""MCI World Model — 学习型认知诊断系统 (MetaDiagnoser)

CEWM v3.7.0 新增组件 (N5)：
VSM System 5 完备化 — 学习型认知诊断 + 根因分析链。

理论基础：
    1. Beer VSM System 5 — 策略层完备化诊断
    2. Lakatos 科学研究纲领 — 硬核保护 + 保护带修正
    3. 认知故障树分析 (Cognitive FTA) — 多层根因追溯

核心能力：
    - diagnose(surprise_signals) — 输入惊奇信号，输出根因分析
    - match_patterns(signals) — 匹配已知失败模式
    - trace_root_cause(diagnosis) — 多层根因链追溯
    - cognitive_health_score() — 六维认知健康度评估

失败模式库 (≥8 种)：
    1. PERCEPTION_DRIFT — 感知漂移（传感器退化/噪声增大）
    2. PREDICTION_BIAS — 预测偏差（模型过拟合/欠拟合）
    3. CAUSAL_COLLAPSE — 因果坍缩（因果图连通性下降）
    4. MEMORY_DECAY — 记忆衰减（经验库覆盖不足）
    5. ACTION_OSCILLATION — 行动振荡（策略在两个状态间反复切换）
    6. ENERGY_IMBALANCE — 能量失衡（五范畴能量分布异常）
    7. COGNITIVE_OVERLOAD — 认知过载（信息输入超过处理能力）
    8. FEEDBACK_LOOP_BROKEN — 反馈环断裂（误差无法传播到行动层）
    9. MODEL_DRIFT — 模型漂移（长期统计特性变化）
    10. CONFOUNDER_INTRUSION — 混杂因子入侵（未建模的因果关系）

Example:
    >>> diagnoser = MetaDiagnoser()
    >>> signals = [SurpriseSignal(score=0.8, ...)]
    >>> diagnosis = diagnoser.diagnose(signals)
    >>> print(diagnosis["pattern"])
    'PREDICTION_BIAS'
    >>> print(diagnosis["root_cause_chain"])
    ['PREDICTION层异常', '模型参数偏移', '训练数据分布漂移']
"""


import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# =============================================================================
# 数据类型
# =============================================================================


class FailurePattern(Enum):
    """认知失败模式枚举。"""

    PERCEPTION_DRIFT = "perception_drift"
    PREDICTION_BIAS = "prediction_bias"
    CAUSAL_COLLAPSE = "causal_collapse"
    MEMORY_DECAY = "memory_decay"
    ACTION_OSCILLATION = "action_oscillation"
    ENERGY_IMBALANCE = "energy_imbalance"
    COGNITIVE_OVERLOAD = "cognitive_overload"
    FEEDBACK_LOOP_BROKEN = "feedback_loop_broken"
    MODEL_DRIFT = "model_drift"
    CONFOUNDER_INTRUSION = "confounder_intrusion"


class SeverityLevel(Enum):
    """严重度等级。"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SurpriseSignal:
    """惊奇信号数据类。

    Attributes:
        score: 惊奇度 [0, 1]，越高越异常
        source: 信号来源
        layer: 所属层级
        features: 特征字典
        breakdown: 信号分解（自动从 features 构建）
    """

    score: float = 0.0
    source: str = ""
    layer: str = ""
    features: dict[str, Any] = field(default_factory=dict)

    @property
    def breakdown(self) -> dict[str, float]:
        return {k: v for k, v in self.features.items() if isinstance(v, (int, float))}


@dataclass
class PatternMatch:
    """模式匹配结果。

    Attributes:
        pattern: 匹配的失败模式
        confidence: 匹配置信度 [0, 1]
        evidence: 支持证据列表
        layer: 主要影响层级
    """

    pattern: FailurePattern
    confidence: float
    evidence: list[str] = field(default_factory=list)
    layer: str = ""


@dataclass
class RootCauseChain:
    """根因分析链。

    Attributes:
        chain: 从表象到根因的因果链 [表象, 中间原因, ..., 根因]
        depth: 链深度
        primary_cause: 首要根因
        contribution: 各节点贡献度
    """

    chain: list[str] = field(default_factory=list)
    depth: int = 0
    primary_cause: str = ""
    contribution: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain": self.chain,
            "depth": self.depth,
            "primary_cause": self.primary_cause,
            "contribution": {k: round(v, 4) for k, v in self.contribution.items()},
        }


@dataclass
class DiagnosisResult:
    """诊断结果。

    Attributes:
        pattern: 主要失败模式
        severity: 严重度等级
        confidence: 诊断置信度
        matches: 所有匹配的模式
        root_cause_chain: 根因分析链
        recommendation: 修复建议
        health_scores: 六维认知健康度
        details: 详细分析
    """

    pattern: FailurePattern | None = None
    severity: SeverityLevel = SeverityLevel.LOW
    confidence: float = 0.0
    matches: list[PatternMatch] = field(default_factory=list)
    root_cause_chain: RootCauseChain = field(default_factory=RootCauseChain)
    recommendation: str = ""
    health_scores: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern.value if self.pattern else None,
            "severity": self.severity.value,
            "confidence": round(self.confidence, 4),
            "matches": [
                {
                    "pattern": m.pattern.value,
                    "confidence": round(m.confidence, 4),
                    "evidence": m.evidence,
                    "layer": m.layer,
                }
                for m in self.matches
            ],
            "root_cause_chain": self.root_cause_chain.to_dict(),
            "recommendation": self.recommendation,
            "health_scores": {k: round(v, 4) for k, v in self.health_scores.items()},
        }


@dataclass
class MetaDiagnoserStats:
    """诊断器统计。"""

    total_diagnoses: int = 0
    pattern_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    avg_confidence: float = 0.0
    total_root_cause_depth: int = 0
    accuracy_samples: int = 0
    correct_diagnoses: int = 0


# =============================================================================
# 失败模式特征库
# =============================================================================

# 每种失败模式的特征签名 (signature)
# signature: {dimension: (min, max)} — 惊奇度分解维度的典型范围
_PATTERN_SIGNATURES: dict[FailurePattern, dict[str, tuple[float, float]]] = {
    FailurePattern.PERCEPTION_DRIFT: {
        "state_distance": (0.5, 1.0),
        "vector_deviation": (0.3, 0.7),
        "direction_error": (0.1, 0.4),
    },
    FailurePattern.PREDICTION_BIAS: {
        "state_distance": (0.3, 0.7),
        "vector_deviation": (0.5, 1.0),
        "direction_error": (0.4, 0.8),
    },
    FailurePattern.CAUSAL_COLLAPSE: {
        "state_distance": (0.2, 0.5),
        "vector_deviation": (0.2, 0.5),
        "direction_error": (0.6, 1.0),
    },
    FailurePattern.MEMORY_DECAY: {
        "state_distance": (0.4, 0.8),
        "vector_deviation": (0.4, 0.8),
        "direction_error": (0.3, 0.6),
    },
    FailurePattern.ACTION_OSCILLATION: {
        "state_distance": (0.6, 1.0),
        "vector_deviation": (0.6, 1.0),
        "direction_error": (0.5, 0.9),
    },
    FailurePattern.ENERGY_IMBALANCE: {
        "state_distance": (0.3, 0.6),
        "vector_deviation": (0.3, 0.6),
        "direction_error": (0.3, 0.6),
    },
    FailurePattern.COGNITIVE_OVERLOAD: {
        "state_distance": (0.7, 1.0),
        "vector_deviation": (0.7, 1.0),
        "direction_error": (0.7, 1.0),
    },
    FailurePattern.FEEDBACK_LOOP_BROKEN: {
        "state_distance": (0.1, 0.3),
        "vector_deviation": (0.1, 0.3),
        "direction_error": (0.8, 1.0),
    },
    FailurePattern.MODEL_DRIFT: {
        "state_distance": (0.4, 0.7),
        "vector_deviation": (0.4, 0.7),
        "direction_error": (0.2, 0.5),
    },
    FailurePattern.CONFOUNDER_INTRUSION: {
        "state_distance": (0.5, 0.9),
        "vector_deviation": (0.3, 0.6),
        "direction_error": (0.7, 1.0),
    },
}

# 根因链模板
_ROOT_CAUSE_TEMPLATES: dict[FailurePattern, list[str]] = {
    FailurePattern.PERCEPTION_DRIFT: [
        "感知层异常",
        "传感器信号偏移",
        "环境条件变化/传感器退化",
    ],
    FailurePattern.PREDICTION_BIAS: [
        "预测层异常",
        "模型参数偏移",
        "训练数据分布漂移",
    ],
    FailurePattern.CAUSAL_COLLAPSE: [
        "认知层异常",
        "因果图连通性下降",
        "关键因果边缺失/弱化",
    ],
    FailurePattern.MEMORY_DECAY: [
        "记忆层异常",
        "经验库覆盖不足",
        "经验巩固/遗忘策略失衡",
    ],
    FailurePattern.ACTION_OSCILLATION: [
        "行动层异常",
        "策略在两个状态间反复切换",
        "行动空间探索不足/代价函数平坦",
    ],
    FailurePattern.ENERGY_IMBALANCE: [
        "能量层异常",
        "五范畴能量分布失衡",
        "特定维度信号长期缺失",
    ],
    FailurePattern.COGNITIVE_OVERLOAD: [
        "认知层过载",
        "信息输入超过处理能力",
        "注意力分配策略失效",
    ],
    FailurePattern.FEEDBACK_LOOP_BROKEN: [
        "反馈环断裂",
        "误差信号无法传播",
        "跨层耦合系数衰减至零",
    ],
    FailurePattern.MODEL_DRIFT: [
        "模型漂移",
        "长期统计特性变化",
        "环境非平稳性超出模型适应范围",
    ],
    FailurePattern.CONFOUNDER_INTRUSION: [
        "混杂因子入侵",
        "未建模的因果关系干扰",
        "外部变量未被纳入因果图",
    ],
}

# 修复建议模板
_RECOMMENDATION_TEMPLATES: dict[FailurePattern, str] = {
    FailurePattern.PERCEPTION_DRIFT: "重新校准传感器或切换至备用感知通道",
    FailurePattern.PREDICTION_BIAS: "触发 JEPA 增量训练，更新预测器参数",
    FailurePattern.CAUSAL_COLLAPSE: "执行因果发现重扫描，补充缺失因果边",
    FailurePattern.MEMORY_DECAY: "注入新经验样本，调整巩固/遗忘阈值",
    FailurePattern.ACTION_OSCILLATION: "增大行动空间探索范围，引入随机扰动",
    FailurePattern.ENERGY_IMBALANCE: "定向采集缺失维度的信号数据",
    FailurePattern.COGNITIVE_OVERLOAD: "启用注意力过滤，降低低优先级通道采样率",
    FailurePattern.FEEDBACK_LOOP_BROKEN: "检查跨层耦合系数，重置 CognitiveLoopBus",
    FailurePattern.MODEL_DRIFT: "触发全量重训练，使用近期数据更新模型",
    FailurePattern.CONFOUNDER_INTRUSION: "扩展因果图节点集，纳入潜在混杂因子",
}


# =============================================================================
# MetaDiagnoser 主类
# =============================================================================


@dataclass(eq=False)
class MetaDiagnoser:
    """学习型认知诊断系统。

    基于惊奇信号的多维分解，匹配已知失败模式，
    追溯根因分析链，输出结构化诊断报告。

    Example:
        >>> diagnoser = MetaDiagnoser()
        >>> # 模拟惊奇信号
        >>> signals = [{"score": 0.8, "breakdown": {
        ...     "state_distance": 0.4,
        ...     "vector_deviation": 0.9,
        ...     "direction_error": 0.6,
        ... }}]
        >>> result = diagnoser.diagnose(signals)
        >>> print(result.pattern)
        FailurePattern.PREDICTION_BIAS
    """

    confidence_threshold: float = 0.3  # 最低匹配置信度
    max_chain_depth: int = 5  # 最大根因链深度

    _history: list[DiagnosisResult] = field(default_factory=list)
    _stats: MetaDiagnoserStats = field(default_factory=MetaDiagnoserStats)

    # ── 核心诊断 ──

    def diagnose(
        self,
        surprise_signals: list[dict[str, Any]] | list[Any],
        context: dict[str, Any] | None = None,
    ) -> DiagnosisResult:
        """对惊奇信号执行认知诊断。

        Args:
            surprise_signals: 惊奇信号列表，支持 dict 或 SurpriseSignal 对象
            context: 附加上下文

        Returns:
            DiagnosisResult 结构化诊断结果
        """
        # 标准化信号格式
        signals = [self._normalize_signal(s) for s in surprise_signals]
        if not signals:
            return DiagnosisResult()

        # 1. 模式匹配
        matches = self.match_patterns(signals)

        # 2. 确定主要模式
        primary = matches[0] if matches else None
        pattern = primary.pattern if primary else None

        # 3. 严重度评估
        severity = self._assess_severity(signals, matches)

        # 4. 根因链追溯
        root_cause = self.trace_root_cause(pattern, signals)

        # 5. 生成建议
        recommendation = (
            _RECOMMENDATION_TEMPLATES.get(pattern, "无法确定具体失败模式，建议全系统检查") if pattern else "无明确诊断"
        )

        # 6. 健康度评估
        health = self.cognitive_health_score(signals)

        result = DiagnosisResult(
            pattern=pattern,
            severity=severity,
            confidence=primary.confidence if primary else 0.0,
            matches=matches,
            root_cause_chain=root_cause,
            recommendation=recommendation,
            health_scores=health,
            details={"n_signals": len(signals), "context": context or {}},
        )

        self._history.append(result)
        self._update_stats(result)
        return result

    def match_patterns(
        self,
        signals: list[dict[str, float]],
    ) -> list[PatternMatch]:
        """匹配已知失败模式。

        使用高斯似然函数计算每种模式的匹配度：
            P(pattern | signal) ∝ exp(-0.5 * Σ((d_i - μ_i) / σ_i)²)

        Args:
            signals: 标准化信号列表

        Returns:
            按置信度降序排列的匹配结果
        """
        # 聚合信号维度（取均值）
        if not signals:
            return []

        avg = self._aggregate_signals(signals)

        matches: list[PatternMatch] = []
        for pattern, signature in _PATTERN_SIGNATURES.items():
            score = self._compute_pattern_likelihood(avg, signature)
            if score >= self.confidence_threshold:
                evidence = self._collect_evidence(avg, pattern, signature)
                layer = self._pattern_to_layer(pattern)
                matches.append(
                    PatternMatch(
                        pattern=pattern,
                        confidence=score,
                        evidence=evidence,
                        layer=layer,
                    )
                )

        # 按置信度降序
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches

    def trace_root_cause(
        self,
        pattern: FailurePattern | None,
        signals: list[dict[str, float]],
    ) -> RootCauseChain:
        """多层根因链追溯。

        链结构: [表象症状, 功能层异常, 机制故障, 深层根因]
        深度 ≥ 3 层

        Args:
            pattern: 主要失败模式
            signals: 信号列表

        Returns:
            RootCauseChain
        """
        if pattern is None:
            return RootCauseChain(
                chain=["未知异常", "无法确定根因"],
                depth=2,
                primary_cause="未知",
            )

        template = _ROOT_CAUSE_TEMPLATES.get(pattern, [])
        if not template:
            return RootCauseChain(
                chain=[f"{pattern.value} 异常", "根因未知"],
                depth=2,
                primary_cause=pattern.value,
            )

        # 构建因果链
        chain = list(template)

        # 根据信号强度调整贡献度
        self._aggregate_signals(signals)
        contribution = {}
        for i, node in enumerate(chain):
            # 越靠近根因，贡献度越高
            weight = (i + 1) / len(chain)
            contribution[node] = weight

        # 确保深度 ≥ 3
        while len(chain) < 3:
            chain.append("需进一步调查")

        chain = chain[: self.max_chain_depth]

        return RootCauseChain(
            chain=chain,
            depth=len(chain),
            primary_cause=chain[-1] if chain else "未知",
            contribution=contribution,
        )

    def cognitive_health_score(
        self,
        signals: list[dict[str, float]] | None = None,
    ) -> dict[str, float]:
        """六维认知健康度评估。

        维度:
            1. causal_discovery: 因果发现能力 (direction_error 的反面)
            2. counterfactual: 反事实推理能力 (基于因果图完整性)
            3. ood_generalization: OOD 泛化能力 (state_distance 的反面)
            4. explainability: 可解释性 (根因链完整度)
            5. memory_reuse: 记忆复用能力 (基于记忆衰减信号)
            6. anomaly_detection: 异常检测能力 (检测灵敏度)

        Returns:
            六维评分 dict, 每项 [0, 1], 越高越健康
        """
        if signals is None:
            signals = []

        avg = self._aggregate_signals(signals) if signals else {}

        sd = avg.get("state_distance", 0.0)
        vd = avg.get("vector_deviation", 0.0)
        de = avg.get("direction_error", 0.0)
        score = avg.get("score", 0.0)

        return {
            # 因果发现: direction_error 低 → 因果发现好
            "causal_discovery": max(0.0, 1.0 - de),
            # 反事实: 综合偏差低 → 因果图完整
            "counterfactual": max(0.0, 1.0 - (sd + de) / 2.0),
            # OOD 泛化: state_distance 低 → 泛化好
            "ood_generalization": max(0.0, 1.0 - sd),
            # 可解释性: 向量偏差低 → 可解释
            "explainability": max(0.0, 1.0 - vd * 0.8),
            # 记忆复用: 总分低 → 记忆有效
            "memory_reuse": max(0.0, 1.0 - score * 0.7),
            # 异常检测: 能检测到异常 → 检测能力存在
            "anomaly_detection": min(1.0, score * 1.2) if score > 0 else 0.5,
        }

    # ── 批量诊断 ──

    def batch_diagnose(
        self,
        signal_batches: list[list[dict[str, Any]]],
    ) -> list[DiagnosisResult]:
        """批量诊断多组信号。"""
        return [self.diagnose(batch) for batch in signal_batches]

    # ── 准确率评估 ──

    def evaluate_accuracy(
        self,
        test_cases: list[tuple[list[dict[str, Any]], FailurePattern]],
    ) -> dict[str, float]:
        """评估诊断准确率。

        Args:
            test_cases: [(signals, expected_pattern), ...]

        Returns:
            {"accuracy": float, "correct": int, "total": int}
        """
        correct = 0
        total = len(test_cases)

        for signals, expected in test_cases:
            result = self.diagnose(signals)
            if result.pattern == expected:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        self._stats.accuracy_samples += total
        self._stats.correct_diagnoses += correct

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }

    # ── 统计 ──

    def statistics(self) -> MetaDiagnoserStats:
        """诊断器统计。"""
        return self._stats

    def history(self) -> list[DiagnosisResult]:
        """诊断历史。"""
        return list(self._history)

    def reset_stats(self) -> None:
        """重置统计。"""
        self._stats = MetaDiagnoserStats()
        self._history.clear()

    @property
    def stats(self) -> MetaDiagnoserStats:
        """统计信息属性。"""
        return self._stats

    @property
    def n_patterns(self) -> int:
        """已知失败模式数。"""
        return len(FailurePattern)

    @property
    def pattern_names(self) -> list[str]:
        """所有失败模式名称。"""
        return [p.value for p in FailurePattern]

    # ── 内部方法 ──

    def _normalize_signal(self, signal: Any) -> dict[str, float]:
        """标准化信号为 dict 格式。"""
        if isinstance(signal, dict):
            result = dict(signal)
            # 确保 breakdown 中的维度被提升到顶层
            breakdown = result.pop("breakdown", {})
            if isinstance(breakdown, dict):
                for k, v in breakdown.items():
                    if k not in result:
                        result[k] = v
            return result

        # SurpriseSignal 对象
        if hasattr(signal, "breakdown") and hasattr(signal, "score"):
            result = {"score": signal.score}
            if isinstance(signal.breakdown, dict):
                result.update(signal.breakdown)
            return result

        return {}

    def _aggregate_signals(self, signals: list[dict[str, float]]) -> dict[str, float]:
        """聚合多条信号为均值向量。"""
        if not signals:
            return {}

        keys = set()
        for s in signals:
            keys.update(s.keys())

        avg = {}
        for key in keys:
            values = [s.get(key, 0.0) for s in signals if key in s]
            if values:
                avg[key] = sum(values) / len(values)

        return avg

    def _compute_pattern_likelihood(
        self,
        observed: dict[str, float],
        signature: dict[str, tuple[float, float]],
    ) -> float:
        """计算模式似然度 (高斯)。

        P(pattern | obs) ∝ exp(-0.5 * Σ((obs_i - midpoint_i) / range_i)²)
        """
        total_sq = 0.0
        n_dims = 0

        for dim, (lo, hi) in signature.items():
            if dim in observed:
                midpoint = (lo + hi) / 2.0
                spread = (hi - lo) / 2.0
                if spread > 1e-8:
                    z = (observed[dim] - midpoint) / spread
                    total_sq += z * z
                    n_dims += 1

        if n_dims == 0:
            return 0.0

        # 归一化 → [0, 1]
        avg_sq = total_sq / n_dims
        likelihood = math.exp(-0.5 * avg_sq)
        return min(1.0, max(0.0, likelihood))

    def _collect_evidence(
        self,
        observed: dict[str, float],
        pattern: FailurePattern,
        signature: dict[str, tuple[float, float]],
    ) -> list[str]:
        """收集支持证据。"""
        evidence = []
        for dim, (lo, hi) in signature.items():
            if dim in observed:
                val = observed[dim]
                if lo <= val <= hi:
                    evidence.append(f"{dim}={val:.3f} 在典型范围 [{lo:.1f}, {hi:.1f}] 内")
                elif val > hi:
                    evidence.append(f"{dim}={val:.3f} 超出上限 {hi:.1f}")
        return evidence

    def _pattern_to_layer(self, pattern: FailurePattern) -> str:
        """映射失败模式到主要影响层。"""
        layer_map = {
            FailurePattern.PERCEPTION_DRIFT: "perception",
            FailurePattern.PREDICTION_BIAS: "prediction",
            FailurePattern.CAUSAL_COLLAPSE: "cognition",
            FailurePattern.MEMORY_DECAY: "cognition",
            FailurePattern.ACTION_OSCILLATION: "action",
            FailurePattern.ENERGY_IMBALANCE: "cognition",
            FailurePattern.COGNITIVE_OVERLOAD: "cognition",
            FailurePattern.FEEDBACK_LOOP_BROKEN: "action",
            FailurePattern.MODEL_DRIFT: "prediction",
            FailurePattern.CONFOUNDER_INTRUSION: "cognition",
        }
        return layer_map.get(pattern, "unknown")

    def _assess_severity(
        self,
        signals: list[dict[str, float]],
        matches: list[PatternMatch],
    ) -> SeverityLevel:
        """评估严重度等级。"""
        if not signals:
            return SeverityLevel.LOW

        avg_score = sum(s.get("score", 0.0) for s in signals) / len(signals)
        max_conf = matches[0].confidence if matches else 0.0

        if avg_score > 0.8 or max_conf > 0.9:
            return SeverityLevel.CRITICAL
        elif avg_score > 0.5 or max_conf > 0.7:
            return SeverityLevel.HIGH
        elif avg_score > 0.3 or max_conf > 0.4:
            return SeverityLevel.MEDIUM
        return SeverityLevel.LOW

    def _update_stats(self, result: DiagnosisResult) -> None:
        """更新统计。"""
        self._stats.total_diagnoses += 1
        if result.pattern:
            self._stats.pattern_counts[result.pattern.value] += 1
        self._stats.total_root_cause_depth += result.root_cause_chain.depth

        n = self._stats.total_diagnoses
        self._stats.avg_confidence = (self._stats.avg_confidence * (n - 1) + result.confidence) / n
