from __future__ import annotations

"""LLM ↔ CEWM 互校准闭环 — TASK-B1。

实现双向互校准:
  LLM 推断因果边 → do-calculus 验证 → 贝叶斯后验置信度更新 → CEWM 图修正
  CEWM 高置信发现 → 注入 LLM 下一轮推理 prompt

核心公式 (贝叶斯更新):
    posterior = (likelihood × prior) / (likelihood × prior + (1-likelihood) × (1-prior))
    where likelihood = P(do_calculus_positive | llm_positive, H) 从校准矩阵估计

架构:
    LLMCEWMBridge
        ├── bidirectional_calibrate()  — 一轮互校准迭代
        ├── multi_round_calibrate()    — 多轮迭代 (收敛检测)
        ├── _validate_edge()           — do-calculus 单边验证
        ├── _bayesian_update()         — 贝叶斯后验计算
        ├── _build_context_injection() — CEWM→LLM 反向注入
        └── _parse_llm_edges()         — LLM 输出解析
"""


import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 核心数据结构
# =============================================================================


@dataclass
class InferredEdge:
    """LLM 推断的因果边。

    Attributes:
        cause: 原因变量名
        effect: 结果变量名
        direction: 因果方向 — "positive" | "negative" | "neutral"
        llm_confidence: LLM 给出的置信度 [0, 1]
        source: 来源标识 — "llm" | "cewm" | "human"
    """

    cause: str
    effect: str
    direction: str = "positive"
    llm_confidence: float = 0.5
    source: str = "llm"


@dataclass
class CalibrationRecord:
    """单条边的校准记录。

    Attributes:
        edge: 被校准的因果边
        prior: 校准前置信度
        posterior: 校准后置信度
        do_ate: do-calculus 估计的 ATE
        do_significant: do-calculus 是否显著 (p < α)
        do_p_value: do-calculus p 值
        likelihood: 似然值 P(evidence | H)
        method: 使用的校准方法 — "bayesian" | "hard_filter"
        direction_agreement: LLM 推断方向与 ATE 方向是否一致
    """

    edge: InferredEdge
    prior: float = 0.5
    posterior: float = 0.5
    do_ate: float = 0.0
    do_significant: bool = False
    do_p_value: float = 1.0
    likelihood: float = 0.5
    method: str = "bayesian"
    direction_agreement: bool = False


@dataclass
class CalibrationStats:
    """一轮校准的汇总统计。

    Attributes:
        n_edges: 校准的边数
        n_significant: do-calculus 显著的边数
        n_direction_agreement: 方向一致的边数
        avg_prior: 平均前置信度
        avg_posterior: 平均后验置信度
        n_confidence_increased: 置信度上升的边数
        n_confidence_decreased: 置信度下降的边数
        elapsed_ms: 校准耗时 (毫秒)
        round_idx: 第几轮校准
    """

    n_edges: int = 0
    n_significant: int = 0
    n_direction_agreement: int = 0
    avg_prior: float = 0.0
    avg_posterior: float = 0.0
    n_confidence_increased: int = 0
    n_confidence_decreased: int = 0
    elapsed_ms: float = 0.0
    round_idx: int = 0


@dataclass
class BridgeConfig:
    """互校准桥配置。

    Attributes:
        significance_level: do-calculus 显著性水平 α (默认 0.05)
        prior_default: 无历史置信度时的默认先验 (默认 0.5)
        confidence_cap: 置信度上限 (防止过度自信, 默认 0.99)
        confidence_floor: 置信度下限 (默认 0.01)
        calibration_matrix: 4 条件似然表
            key = (llm_positive: bool, do_positive: bool) → P(correct)
        max_rounds: 最大校准轮数 (默认 5)
        convergence_threshold: 收敛阈值 — 后验变化 < 此值时停止
        context_injection_threshold: CEWM→LLM 反向注入的置信度阈值
    """

    significance_level: float = 0.05
    prior_default: float = 0.5
    confidence_cap: float = 0.99
    confidence_floor: float = 0.01
    calibration_matrix: dict[tuple[bool, bool], float] = field(
        default_factory=lambda: {
            (True, True): 0.85,  # LLM+, do+ → 高概率正确
            (True, False): 0.30,  # LLM+, do- → 低概率正确
            (False, True): 0.60,  # LLM-, do+ → 中概率正确
            (False, False): 0.70,  # LLM-, do- → 中高概率正确
        }
    )
    max_rounds: int = 5
    convergence_threshold: float = 0.01
    context_injection_threshold: float = 0.8


# =============================================================================
# LLMCEWMBridge — LLM ↔ CEWM 互校准闭环
# =============================================================================


class LLMCEWMBridge:
    """LLM ↔ CEWM 双向互校准闭环。

    校准流程 (一轮迭代):
        1. LLM 推断因果边列表 → llm_inferred_edges
        2. 对每条边执行 do-calculus 验证: estimate_ate(cause, effect)
        3. 贝叶斯后验更新: posterior = (likelihood × prior) / evidence
        4. 更新 CEWM 因果图置信度
        5. 提取高置信发现 → 构建 LLM 下一轮 context injection prompt

    用法:
        >>> bridge = LLMCEWMBridge(do_calculus=dc, causal_graph=cg)
        >>> edges = [InferredEdge("dopamine", "heart_rate", "positive", 0.7)]
        >>> graph, prompt, stats = bridge.bidirectional_calibrate(edges)

    验收标准:
        - 互校准后因果边 F1 提升 ≥ 5%
        - 双向校准延迟 < 500ms (含 LLM API 调用)
        - 3 轮互校准后后续轮 F1 不再下降
    """

    def __init__(
        self,
        do_calculus: Any | None = None,
        causal_graph: Any | None = None,
        llm_adapter: Any | None = None,
        config: BridgeConfig | None = None,
    ):
        """
        Args:
            do_calculus: DoCalculus 实例 (用于 estimate_ate 验证)
            causal_graph: CausalGraph 实例 (置信度存储载体)
            llm_adapter: MultiLLMAdapter 实例 (可选, 用于 LLM 推理)
            config: 互校准配置
        """
        self._do_calculus = do_calculus
        self._causal_graph = causal_graph
        self._llm_adapter = llm_adapter
        self._config = config or BridgeConfig()

        # 置信度存储: (cause, effect) → float
        self._confidence_store: dict[tuple[str, str], float] = {}
        self._round_history: list[CalibrationStats] = []

    # -----------------------------------------------------------------
    # 公开 API
    # -----------------------------------------------------------------

    def bidirectional_calibrate(
        self,
        llm_inferred_edges: list[InferredEdge],
        round_idx: int = 0,
    ) -> tuple[Any, str, CalibrationStats]:
        """互校准闭环 — 一轮迭代。

        Args:
            llm_inferred_edges: LLM 推断的因果边列表
            round_idx: 当前轮次 (用于统计)

        Returns:
            (causal_graph, context_injection_prompt, stats)
            - causal_graph: 更新后的因果图
            - context_injection_prompt: CEWM→LLM 反向注入的提示词
            - stats: 校准统计
        """
        t0 = time.perf_counter()
        records: list[CalibrationRecord] = []

        # 1. 逐边校准: do-calculus 验证 + 贝叶斯更新
        for edge in llm_inferred_edges:
            record = self._validate_and_update(edge)
            records.append(record)

        # 2. 汇总统计
        elapsed_ms = (time.perf_counter() - t0) * 1000
        stats = self._compute_stats(records, round_idx, elapsed_ms)
        self._round_history.append(stats)

        # 3. CEWM→LLM 反向注入: 提取高置信发现构建 prompt
        context_prompt = self._build_context_injection()

        return self._causal_graph, context_prompt, stats

    def multi_round_calibrate(
        self,
        llm_edge_generator: Any | None = None,
        initial_edges: list[InferredEdge] | None = None,
    ) -> list[CalibrationStats]:
        """多轮互校准 — 自动收敛检测。

        每轮:
            1. 使用上轮的 context_injection_prompt 重新查询 LLM
            2. 执行 bidirectional_calibrate
            3. 检测收敛: 后验变化 < threshold

        Args:
            llm_edge_generator: 可调用对象, 接收 context_prompt 返回 list[InferredEdge]
            initial_edges: 初始 LLM 推断边 (第一轮使用)

        Returns:
            各轮校准统计列表
        """
        all_stats: list[CalibrationStats] = []
        current_edges = initial_edges or []
        context_prompt = ""

        for round_idx in range(self._config.max_rounds):
            # 第一轮用 initial_edges, 后续轮从 LLM 重新获取
            if round_idx > 0 and llm_edge_generator is not None:
                try:
                    current_edges = llm_edge_generator(context_prompt)
                except Exception as e:
                    logger.warning("LLM edge generation failed at round %d: %s", round_idx, e)
                    break

            if not current_edges:
                logger.info("No edges to calibrate at round %d, stopping.", round_idx)
                break

            _, context_prompt, stats = self.bidirectional_calibrate(current_edges, round_idx=round_idx)
            all_stats.append(stats)

            # 收敛检测: 平均后验变化 < threshold
            if round_idx > 0 and len(all_stats) >= 2:
                prev_avg = all_stats[-2].avg_posterior
                curr_avg = all_stats[-1].avg_posterior
                if abs(curr_avg - prev_avg) < self._config.convergence_threshold:
                    logger.info(
                        "Calibration converged at round %d (Δ=%.4f < %.4f)",
                        round_idx,
                        abs(curr_avg - prev_avg),
                        self._config.convergence_threshold,
                    )
                    break

        return all_stats

    def get_confidence(self, cause: str, effect: str) -> float:
        """获取因果边的当前置信度。

        Args:
            cause: 原因变量
            effect: 结果变量

        Returns:
            置信度 [0, 1]，未存储时返回默认先验
        """
        return self._confidence_store.get((cause, effect), self._config.prior_default)

    def get_high_confidence_edges(self, threshold: float | None = None) -> list[InferredEdge]:
        """获取高置信度边列表。

        Args:
            threshold: 置信度阈值 (默认使用配置中的 context_injection_threshold)

        Returns:
            置信度 ≥ threshold 的 InferredEdge 列表
        """
        thr = threshold if threshold is not None else self._config.context_injection_threshold
        result = []
        for (cause, effect), conf in self._confidence_store.items():
            if conf >= thr:
                result.append(
                    InferredEdge(
                        cause=cause,
                        effect=effect,
                        direction="positive",
                        llm_confidence=conf,
                        source="cewm",
                    )
                )
        return result

    @property
    def round_history(self) -> list[CalibrationStats]:
        """获取历史校准统计。"""
        return list(self._round_history)

    @property
    def config(self) -> BridgeConfig:
        """获取当前配置。"""
        return self._config

    # -----------------------------------------------------------------
    # 内部方法
    # -----------------------------------------------------------------

    def _validate_and_update(self, edge: InferredEdge) -> CalibrationRecord:
        """对单条边执行 do-calculus 验证 + 贝叶斯更新。

        步骤:
            1. 调用 do_calculus.estimate_ate(edge.cause, edge.effect)
            2. 判断 ATE 是否显著 (p < α)
            3. 判断方向一致性 (ATE 符号 vs edge.direction)
            4. 计算似然 P(evidence | H)
            5. 贝叶斯后验更新
            6. 存储新置信度
            7. 更新 CausalGraph (如果可用)
        """
        # 先验
        prior = self._confidence_store.get((edge.cause, edge.effect), self._config.prior_default)

        # do-calculus 验证
        do_ate = 0.0
        do_significant = False
        do_p_value = 1.0

        if self._do_calculus is not None:
            try:
                result = self._do_calculus.estimate_ate(
                    X=edge.cause,
                    Y=edge.effect,
                )
                do_ate = getattr(result, "ate", 0.0)
                do_significant = abs(do_ate) > 0.0  # 简化显著性判断

                # 尝试获取 p_value
                ci = getattr(result, "confidence_interval", (0.0, 0.0))
                if isinstance(ci, (tuple, list)) and len(ci) == 2:
                    abs(ci[1] - ci[0])
                    # p 值近似: CI 不含 0 → p < α
                    if ci[0] > 0 or ci[1] < 0:
                        do_significant = True
                        do_p_value = self._config.significance_level * 0.5
                    else:
                        do_p_value = 1.0

            except Exception as e:
                logger.warning("do-calculus validation failed for %s→%s: %s", edge.cause, edge.effect, e)
                do_significant = False
                do_p_value = 1.0

        # 方向一致性检查
        direction_agreement = self._check_direction_agreement(edge.direction, do_ate)

        # 似然 P(evidence | H): 从校准矩阵查询
        llm_positive = edge.llm_confidence >= 0.5
        likelihood = self._config.calibration_matrix.get((llm_positive, do_significant), 0.5)

        # 如果方向不一致, 降低似然
        if not direction_agreement:
            likelihood *= 0.5

        # 贝叶斯后验
        posterior = self._bayesian_update(prior, likelihood)

        # 限幅
        posterior = np.clip(posterior, self._config.confidence_floor, self._config.confidence_cap)

        # 存储新置信度
        self._confidence_store[(edge.cause, edge.effect)] = float(posterior)

        # 更新 CausalGraph (如果可用)
        if self._causal_graph is not None and hasattr(self._causal_graph, "add_edge"):
            if posterior >= self._config.prior_default:
                self._causal_graph.add_edge(edge.cause, edge.effect, weight=float(posterior))

        return CalibrationRecord(
            edge=edge,
            prior=prior,
            posterior=float(posterior),
            do_ate=do_ate,
            do_significant=do_significant,
            do_p_value=do_p_value,
            likelihood=likelihood,
            method="bayesian",
            direction_agreement=direction_agreement,
        )

    def _bayesian_update(self, prior: float, likelihood: float) -> float:
        """贝叶斯后验更新。

        posterior = (likelihood × prior) / (likelihood × prior + (1-likelihood) × (1-prior))

        Args:
            prior: 先验置信度 P(H)
            likelihood: 似然 P(evidence | H)

        Returns:
            后验置信度 P(H | evidence)
        """
        evidence = likelihood * prior + (1 - likelihood) * (1 - prior)
        if evidence < 1e-10:
            return prior  # 避免除零
        return (likelihood * prior) / evidence

    def _check_direction_agreement(self, direction: str, ate: float) -> bool:
        """检查 LLM 推断方向与 ATE 方向是否一致。

        Args:
            direction: LLM 推断方向 ("positive" | "negative" | "neutral")
            ate: do-calculus 估计的 ATE

        Returns:
            是否一致
        """
        if direction == "positive" and ate > 0:
            return True
        if direction == "negative" and ate < 0:
            return True
        return bool(direction == "neutral" and abs(ate) < 1e-06)

    def _build_context_injection(self) -> str:
        """构建 CEWM→LLM 反向注入提示词。

        提取高置信度发现, 构建 LLM 下一轮推理的上下文注入。

        Returns:
            context injection prompt 字符串
        """
        high_conf_edges = self.get_high_confidence_edges()

        if not high_conf_edges:
            return ""

        lines = ["以下因果关系已通过 do-calculus 验证（高置信度）："]
        for edge in high_conf_edges:
            conf = self.get_confidence(edge.cause, edge.effect)
            lines.append(f"  - {edge.cause} → {edge.effect}: 方向={edge.direction}, 置信度={conf:.2f}")

        lines.append("\n请在推理中参考以上已验证的因果关系。")

        return "\n".join(lines)

    def _compute_stats(
        self,
        records: list[CalibrationRecord],
        round_idx: int,
        elapsed_ms: float,
    ) -> CalibrationStats:
        """计算校准统计。"""
        if not records:
            return CalibrationStats(elapsed_ms=elapsed_ms, round_idx=round_idx)

        priors = [r.prior for r in records]
        posteriors = [r.posterior for r in records]

        n_increased = sum(1 for p, q in zip(priors, posteriors) if q > p)
        n_decreased = sum(1 for p, q in zip(priors, posteriors) if q < p)

        return CalibrationStats(
            n_edges=len(records),
            n_significant=sum(1 for r in records if r.do_significant),
            n_direction_agreement=sum(1 for r in records if r.direction_agreement),
            avg_prior=float(np.mean(priors)),
            avg_posterior=float(np.mean(posteriors)),
            n_confidence_increased=n_increased,
            n_confidence_decreased=n_decreased,
            elapsed_ms=elapsed_ms,
            round_idx=round_idx,
        )

    # -----------------------------------------------------------------
    # LLM 输出解析
    # -----------------------------------------------------------------

    @staticmethod
    def parse_llm_edges(llm_output: str) -> list[InferredEdge]:
        """从 LLM 文本输出中解析因果边。

        支持两种格式:
        1. JSON 数组: [{"cause": "X", "effect": "Y", "direction": "positive", "confidence": 0.8}]
        2. 纯文本行: "X → Y (positive, 0.8)"

        Args:
            llm_output: LLM 原始输出文本

        Returns:
            解析出的 InferredEdge 列表
        """
        edges: list[InferredEdge] = []

        # 尝试 JSON 解析
        try:
            # 提取 JSON 数组部分
            text = llm_output.strip()
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                items = json.loads(json_str)
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict) and "cause" in item and "effect" in item:
                            edges.append(
                                InferredEdge(
                                    cause=str(item["cause"]),
                                    effect=str(item["effect"]),
                                    direction=str(item.get("direction", "positive")),
                                    llm_confidence=float(item.get("confidence", 0.5)),
                                    source="llm",
                                )
                            )
                    if edges:
                        return edges
        except (json.JSONDecodeError, ValueError):
            pass

        # 降级: 正则匹配 "X → Y (direction, confidence)"
        import re

        pattern = r"(\w+)\s*[→>]\s*(\w+)\s*\(\s*(\w+)\s*,\s*([\d.]+)\s*\)"
        for match in re.finditer(pattern, llm_output):
            cause, effect, direction, conf = match.groups()
            try:
                edges.append(
                    InferredEdge(
                        cause=cause,
                        effect=effect,
                        direction=direction,
                        llm_confidence=float(conf),
                        source="llm",
                    )
                )
            except ValueError:
                continue

        return edges
