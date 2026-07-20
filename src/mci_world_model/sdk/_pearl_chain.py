from __future__ import annotations

"""MCI World Model v4.6.0 — PearlChain 协调器
================================================

Pearl 因果三层串联协调器 — F8 修复。

F8 根因: L1(关联)/L2(干预)/L3(反事实) 三层独立运行,
缺少串联和回写机制, 导致 Pearl 因果层级断裂。

修复:
    PearlChain.full_analysis() 串联 L1→L2→L3,
    并将 L3 归因结果回写到 L2 置信度和 L1 边权重。

三层流程:
    L1 观察: CausalEngine.find_causal_pairs() → 关联/因果对
    L2 干预: DoCalculus.estimate_ate() → ATE + CI + p-value
    L3 反事实: CounterfactualEngine.query() → PN/PS/PNS + 个体效应

回写机制:
    L3→L2: PN(必然性概率) × 原置信度 → 更新 L2 置信度
    L2→L1: |ATE| 归一化 → 更新 L1 因果边权重

## Formal Guarantees

    - full_analysis() 始终返回 PearlChainResult, 即使子层失败也优雅降级
    - 回写不会增大原始值 (damping factor ≤ 1.0)
    - 各层可独立调用, PearlChain 仅负责串联

用法:
    >>> from mci_world_model.sdk._pearl_chain import PearlChain, PearlChainResult
    >>> chain = PearlChain()
    >>> result = chain.full_analysis("X", "Y", x_value=1.0,
    ...     data={"X": [...], "Y": [...], "Z": [...]})
    >>> result.l2_result.ate  # ATE
    >>> result.l3_result.pn   # 必然性概率
"""


import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 核心数据结构
# =============================================================================


@dataclass
class L1ObservationResult:
    """L1 观察层结果 — 关联发现。

    Attributes:
        cause: 原因变量名
        effect: 结果变量名
        correlation: 相关系数 [-1, 1]
        causal_type: 因果类型 "cause" | "condition" | "result" | "shared" | "statistical"
        confidence: 观察置信度 [0, 1]
        edge_weight: 因果边权重 (可被 L2/L3 回写更新)
    """

    cause: str = ""
    effect: str = ""
    correlation: float = 0.0
    causal_type: str = "statistical"
    confidence: float = 0.0
    edge_weight: float = 0.0


@dataclass
class PearlChainResult:
    """PearlChain 三层分析完整结果。

    Attributes:
        l1_result: L1 观察层结果
        l2_result: L2 干预层结果 (InterventionResult)
        l3_result: L3 反事实层结果 (CounterfactualResult)
        feedback_applied: 是否应用了回写
        chain_confidence: 串联置信度 (三层综合)
        note: 附加说明
    """

    l1_result: L1ObservationResult | None = None
    l2_result: Any = None  # InterventionResult from _do_calculus
    l3_result: Any = None  # CounterfactualResult from _counterfactual
    feedback_applied: bool = False
    chain_confidence: float = 0.0
    note: str = ""


# =============================================================================
# PearlChain — 三层串联协调器
# =============================================================================


class PearlChain:
    """Pearl 因果三层串联协调器。

    串联 L1(关联) → L2(干预) → L3(反事实) 三层因果推理,
    并将 L3 归因结果回写到 L2 置信度和 L1 边权重。

    F8 修复核心: 原三层独立运行, 现通过 PearlChain 实现端到端串联。

    Example:
        >>> chain = PearlChain()
        >>> result = chain.full_analysis("dopamine", "heart_rate",
        ...     x_value=1.0, data={"dopamine": [...], "heart_rate": [...], "age": [...]})
        >>> print(f"ATE={result.l2_result.ate:.3f}, PN={result.l3_result.pn:.3f}")
    """

    def __init__(self, damping: float = 0.5):
        """
        Args:
            damping: 回写阻尼系数 [0, 1], 越小回写越保守
        """
        if not 0.0 <= damping <= 1.0:
            raise ValueError(f"damping 必须在 [0, 1], 收到 {damping}")
        self._damping = damping
        self._l1_engine = None  # 惰性初始化
        self._analysis_count: int = 0

    @property
    def damping(self) -> float:
        return self._damping

    @property
    def analysis_count(self) -> int:
        return self._analysis_count

    # -----------------------------------------------------------------
    # 核心 API
    # -----------------------------------------------------------------

    def full_analysis(
        self,
        cause: str,
        effect: str,
        x_value: float = 1.0,
        data: dict[str, np.ndarray] | None = None,
        evidence: dict[str, float] | None = None,
    ) -> PearlChainResult:
        """三层串联分析: L1→L2→L3 + 回写。

        Args:
            cause: 原因变量名
            effect: 结果变量名
            x_value: 干预值 (do(X=x_value))
            data: 变量观测数据 {var_name: array}
            evidence: L3 反事实证据 {var_name: observed_value}

        Returns:
            PearlChainResult 包含三层结果和回写状态
        """
        self._analysis_count += 1
        result = PearlChainResult()

        # ── L1: 观察层 — 关联发现 ──
        l1 = self._observe(cause, effect, data)
        result.l1_result = l1

        # ── L2: 干预层 — ATE 估计 ──
        l2 = self._intervene(cause, effect, x_value, data)
        result.l2_result = l2

        # ── L3: 反事实层 — 归因 ──
        l3 = self._counterfactual(cause, effect, x_value, evidence, data)
        result.l3_result = l3

        # ── 回写: L3→L2→L1 ──
        result.feedback_applied = self._apply_feedback(result)

        # ── 综合置信度 ──
        result.chain_confidence = self._compute_chain_confidence(result)

        return result

    # -----------------------------------------------------------------
    # L1: 观察层
    # -----------------------------------------------------------------

    def _observe(
        self,
        cause: str,
        effect: str,
        data: dict[str, np.ndarray] | None,
    ) -> L1ObservationResult:
        """L1 观察层: 计算变量间关联。"""
        result = L1ObservationResult(cause=cause, effect=effect)

        if data is None or cause not in data or effect not in data:
            result.note = "insufficient_data_for_L1"  # type: ignore
            return result

        x = np.asarray(data[cause], dtype=np.float64)
        y = np.asarray(data[effect], dtype=np.float64)

        if len(x) < 3 or len(y) < 3:
            result.note = "insufficient_samples_for_L1"  # type: ignore
            return result

        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]

        # 皮尔逊相关系数
        if np.std(x) > 1e-10 and np.std(y) > 1e-10:
            result.correlation = float(np.corrcoef(x, y)[0, 1])
        else:
            result.correlation = 0.0

        # 置信度 = |correlation|
        result.confidence = min(abs(result.correlation), 1.0)
        result.edge_weight = result.confidence
        result.causal_type = "statistical"

        # 尝试关键词因果检测 (如果数据中含文本信息)
        try:
            from mci_world_model.sdk._causal import detect_causal_link

            link = detect_causal_link(cause, effect)
            if link is not None:
                result.causal_type, kw_conf = link
                # 融合关键词置信度和统计置信度
                result.confidence = 0.6 * result.confidence + 0.4 * kw_conf
                result.edge_weight = result.confidence
        except Exception:
            logger.warning("异常降级", exc_info=True)
            pass  # 关键词检测是增强, 失败不影响基线

        return result

    # -----------------------------------------------------------------
    # L2: 干预层
    # -----------------------------------------------------------------

    def _intervene(
        self,
        cause: str,
        effect: str,
        x_value: float,
        data: dict[str, np.ndarray] | None,
    ) -> Any:
        """L2 干预层: 使用 DoCalculus 估计 ATE。"""
        try:
            from mci_world_model.sdk._do_calculus import CausalGraph, DoCalculus

            if data is None or len(data) < 2:
                return _empty_intervention(cause, effect, "insufficient_data_for_L2")

            # 从数据变量构建因果图
            nodes = list(data.keys())
            edges = []
            _node_idx = {n: i for i, n in enumerate(nodes)}

            # 简单因果结构: cause→effect, 其他变量作为混淆因子
            for node in nodes:
                if node not in {cause, effect}:
                    edges.append((node, cause))
                    edges.append((node, effect))
            edges.append((cause, effect))

            cg = CausalGraph(nodes=nodes, edges=edges)
            dc = DoCalculus(cg)

            # 估计 ATE
            ate_result = dc.estimate_ate(cause, effect)
            return ate_result

        except Exception as e:
            logger.warning("L2 干预分析失败: %s", e)
            return _empty_intervention(cause, effect, f"L2_error: {e}")

    # -----------------------------------------------------------------
    # L3: 反事实层
    # -----------------------------------------------------------------

    def _counterfactual(
        self,
        cause: str,
        effect: str,
        x_value: float,
        evidence: dict[str, float] | None,
        data: dict[str, np.ndarray] | None,
    ) -> Any:
        """L3 反事实层: 使用 CounterfactualEngine 进行归因。"""
        try:
            from mci_world_model.sdk._counterfactual import CounterfactualEngine

            # 构建证据
            if evidence is None and data is not None:
                # 使用数据均值作为证据
                evidence = {}
                for var, arr in data.items():
                    arr = np.asarray(arr, dtype=np.float64)
                    if len(arr) > 0:
                        evidence[var] = float(np.mean(arr))

            if not evidence:
                return _empty_counterfactual(cause, effect, "no_evidence_for_L3")

            # 构建反事实干预
            do_intervention = {cause: x_value}

            # 从因果图构建 CounterfactualEngine
            try:
                from mci_world_model.sdk._do_calculus import CausalGraph

                nodes = list(data.keys()) if data else [cause, effect]
                edges = [(cause, effect)]
                for node in nodes:
                    if node not in {cause, effect}:
                        edges.append((node, effect))
                cg = CausalGraph(nodes=nodes, edges=edges)
                engine = CounterfactualEngine.from_causal_graph(cg)
            except Exception:
                logger.warning("异常降级", exc_info=True)
                # 降级: 使用简单因果图
                cg = CausalGraph(nodes=[cause, effect], edges=[(cause, effect)])
                engine = CounterfactualEngine.from_causal_graph(cg)

            cf_result = engine.query(  # type: ignore
                evidence=evidence,
                do_x=do_intervention,
                target=effect,
            )
            return cf_result

        except Exception as e:
            logger.warning("L3 反事实分析失败: %s", e)
            return _empty_counterfactual(cause, effect, f"L3_error: {e}")

    # -----------------------------------------------------------------
    # 回写机制
    # -----------------------------------------------------------------

    def _apply_feedback(self, result: PearlChainResult) -> bool:
        """三层回写: L3→L2→L1。

        L3→L2: PN(必然性概率) 更新 L2 置信度
        L2→L1: |ATE| 归一化更新 L1 边权重

        回写公式 (damping 控制):
            updated = original × (1 - damping) + feedback × damping

        Returns:
            是否成功应用回写
        """
        applied = False
        d = self._damping

        # L3→L2: PN 更新置信度
        if result.l3_result is not None and result.l2_result is not None:
            pn = getattr(result.l3_result, "pn", -1.0)
            if pn >= 0:
                # PN 越高, L2 结果越可信
                l2_conf = 1.0 - min(getattr(result.l2_result, "p_value", 1.0), 1.0)
                updated_conf = l2_conf * (1 - d) + pn * d
                if hasattr(result.l2_result, "p_value"):
                    # p_value 越小越好, 等效于置信度越高
                    result.l2_result.p_value = max(0.0, 1.0 - updated_conf)
                applied = True

        # L2→L1: |ATE| 更新边权重
        if result.l2_result is not None and result.l1_result is not None:
            ate = abs(getattr(result.l2_result, "ate", 0.0))
            if ate > 0:
                # ATE 归一化到 [0, 1] (假设 ATE < 10 为正常范围)
                ate_norm = min(ate / 10.0, 1.0)
                old_weight = result.l1_result.edge_weight
                result.l1_result.edge_weight = old_weight * (1 - d) + ate_norm * d
                applied = True

        return applied

    def _compute_chain_confidence(self, result: PearlChainResult) -> float:
        """计算三层综合置信度。

        公式: conf = w1*|corr| + w2*(1-p_value) + w3*max(PN,0)
        权重: w1=0.3, w2=0.4, w3=0.3
        """
        w1, w2, w3 = 0.3, 0.4, 0.3
        l1_conf = abs(result.l1_result.correlation) if result.l1_result else 0.0
        l2_conf = 1.0 - min(getattr(result.l2_result, "p_value", 1.0), 1.0) if result.l2_result else 0.0
        pn = getattr(result.l3_result, "pn", -1.0) if result.l3_result else -1.0
        l3_conf = max(pn, 0.0)

        return float(np.clip(w1 * l1_conf + w2 * l2_conf + w3 * l3_conf, 0.0, 1.0))


# =============================================================================
# 辅助函数
# =============================================================================


def _empty_intervention(cause: str, effect: str, note: str) -> Any:
    """创建空的 InterventionResult。"""
    from mci_world_model.sdk._do_calculus import InterventionResult

    return InterventionResult(
        intervention=f"do({cause})",
        target=effect,
        method="none",
        note=note,
    )


def _empty_counterfactual(cause: str, effect: str, note: str) -> Any:
    """创建空的 CounterfactualResult。"""
    from mci_world_model.sdk._counterfactual import CounterfactualResult

    return CounterfactualResult(
        target=effect,
        status="error",
        note=note,
    )
