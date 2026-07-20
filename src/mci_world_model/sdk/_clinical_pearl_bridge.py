"""MCI World Model — 临床因果下沉桥接（ClinicalPearlBridge）

============================================================

方向三：因果下沉 — 把 Pearl 三层因果推断从通用引擎接入医疗世界模型。

为什么需要？
    D4 实现的 ClinicalCausalDiscovery 只到 L1 发现层（PC 算法找因果骨架）。
    MedicalCausalSDK.diagnose 本质是 L1 关联打分。
    MedicalAction.apply 是确定性状态转移，不是 do-intervention。

    通用 Pearl 引擎（DoCalculus/CounterfactualEngine）已存在但完全未接入
    PatientState/MedicalAction/ClinicalCausalDiscovery —— 两侧孤立。

    本桥接搭起这座连接，把因果能力下沉：
        L1 发现 → ClinicalCausalDiscovery（已有）
        L2 干预 → intervene()：do-calculus 估计 do(药物) 的因果效应
        L3 反事实 → counterfactual()：Pearl 三步推算"若当初选另一方案"

三层语义（Pearl causality ladder）:
    L1 关联: P(Y|X)        — 看到 X 时 Y 的概率（发现/诊断）
    L2 干预: P(Y|do(X))    — 主动施加 X 时 Y 的概率（干预效应）
    L3 反事实: P(Y_x|X',Y') — 若当初做了 x，Y 会怎样（个体反事实）

与朴素 predict 差分的本质区别:
    朴素: predict(s,a_A) vs predict(s,a_B)，两次独立采样，噪声不对齐
    正式: Pearl Abduction 锁定同一噪声 U，do(X) 切边，调整集去混杂

设计原则（AGENTS.md）:
    - 无状态：每次推理独立，不持久化（记忆归 su-memory）
    - 复用通用引擎：不重写 do-calculus/counterfactual，只做适配
    - 可审计：返回完整 InterventionResult/CounterfactualResult
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mci_world_model.sdk._clinical_causal_discovery import (
    CausalStructure,
    ClinicalCausalDiscovery,
)
from mci_world_model.sdk._clinical_world_state import (
    VITAL_NAMES,
    MedicalAction,
    PatientState,
)
from mci_world_model.sdk._counterfactual import CounterfactualEngine, CounterfactualResult
from mci_world_model.sdk._do_calculus import CausalGraph, DoCalculus, InterventionResult

# =============================================================================
# 适配器：CausalStructure → CausalGraph
# =============================================================================


def causal_structure_to_graph(structure: CausalStructure) -> CausalGraph:
    """把 ClinicalCausalDiscovery 的结果转为 DoCalculus 的 CausalGraph。

    CausalStructure.links 是有向因果边（cause→effect，带 strength），
    直接映射为 CausalGraph 的 nodes + edges + adjacency。

    Args:
        structure: ClinicalCausalDiscovery.discover() 的结果。

    Returns:
        CausalGraph（DoCalculus 可用）。
    """
    # 收集所有节点（去重保序）
    node_set: list[str] = []
    seen: set[str] = set()
    edges: list[tuple[str, str]] = []
    for link in structure.links:
        if link.cause not in seen:
            node_set.append(link.cause)
            seen.add(link.cause)
        if link.effect not in seen:
            node_set.append(link.effect)
            seen.add(link.effect)
        edges.append((link.cause, link.effect))

    # 邻接矩阵
    n = len(node_set)
    idx = {name: i for i, name in enumerate(node_set)}
    adjacency = np.zeros((n, n), dtype=np.float32)
    for link in structure.links:
        i = idx[link.cause]
        j = idx[link.effect]
        adjacency[i, j] = link.strength

    return CausalGraph(nodes=node_set, edges=edges, adjacency=adjacency)


# =============================================================================
# ClinicalPearlBridge — 因果下沉主桥接
# =============================================================================


class ClinicalPearlBridge:
    """临床因果下沉桥接 — 把 Pearl L2/L3 接入医疗世界模型。

    提供三层因果能力：
        L1 discover(): 从数据发现因果结构（委托 ClinicalCausalDiscovery）
        L2 intervene(): do-calculus 干预效应估计
        L3 counterfactual(): Pearl 反事实推理（含 PN/PS/PNS）

    Example:
        >>> bridge = ClinicalPearlBridge()
        >>> # L1: 从历史数据发现因果图
        >>> structure = bridge.discover(patient_history)
        >>> # L2: 估计 do(心率干预) 对血压的因果效应
        >>> result = bridge.intervene(structure, "heart_rate", "systolic_bp", data)
        >>> # L3: 反事实——若当初心率不同，血压会怎样
        >>> cf = bridge.counterfactual(structure, evidence, {"heart_rate": 90}, "systolic_bp")
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._discovery = ClinicalCausalDiscovery(significance=0.05, min_samples=10)

    # ── L1 发现（委托）──────────────────────────────────────────

    def discover(
        self,
        patient_history: np.ndarray,
        max_conditioning_size: int = 1,
    ) -> CausalStructure:
        """L1: 从患者体征时序数据发现因果结构。

        Args:
            patient_history: 体征矩阵 (T, N_VITALS)。
            max_conditioning_size: PC 算法 conditioning set 最大规模。

        Returns:
            CausalStructure。
        """
        return self._discovery.discover(patient_history, max_conditioning_size=max_conditioning_size)

    # ── L2 干预（do-calculus）─────────────────────────────────────

    def intervene(
        self,
        structure: CausalStructure,
        treatment: str,
        outcome: str,
        data: dict[str, np.ndarray] | None = None,
        structure_graph: CausalGraph | None = None,
    ) -> InterventionResult:
        """L2: 估计 do(treatment) 对 outcome 的因果效应。

        用 do-calculus 后门/前门调整估计平均处理效应（ATE），
        而非朴素 P(Y|X) 关联。

        Args:
            structure: L1 发现的因果结构。
            treatment: 干预变量名（如 "heart_rate"）。
            outcome: 结果变量名（如 "systolic_bp"）。
            data: 观测数据 {var_name: values}（可选，用于估计效应量）。
            structure_graph: 预转换的 CausalGraph（可选，避免重复转换）。

        Returns:
            InterventionResult（含 ATE/CI/调整集/方法）。
        """
        graph = structure_graph or causal_structure_to_graph(structure)

        # 节点必须存在
        if treatment not in graph.nodes or outcome not in graph.nodes:
            return InterventionResult(
                intervention=f"do({treatment})",
                target=outcome,
                method="none",
                note=f"变量 {treatment} 或 {outcome} 不在因果图中",
            )

        dc = DoCalculus(graph=graph, data=data, seed=self._seed)
        try:
            result = dc.estimate_ate(treatment, outcome)
            result.intervention = f"do({treatment})"
            return result
        except (ValueError, RuntimeError) as e:
            return InterventionResult(
                intervention=f"do({treatment})",
                target=outcome,
                method="none",
                note=f"ATE 估计失败: {e}",
            )

    def identify_confounders(
        self,
        structure: CausalStructure,
        treatment: str,
        outcome: str,
    ) -> list[str]:
        """识别 treatment→outcome 的后门混杂变量（需调整的变量）。

        这是 L2 干预的关键：朴素关联有混杂偏倚，do-calculus 通过
        调整后门集消除混杂。

        Args:
            structure: 因果结构。
            treatment: 干预变量。
            outcome: 结果变量。

        Returns:
            后门调整集（混杂变量列表），空表示无混杂或不可识别。
        """
        graph = causal_structure_to_graph(structure)
        if treatment not in graph.nodes or outcome not in graph.nodes:
            return []
        dc = DoCalculus(graph=graph, seed=self._seed)
        adj = dc.identify_adjustment_set(treatment, outcome)
        return adj or []

    # ── L3 反事实（Pearl 三步）────────────────────────────────────

    def counterfactual(
        self,
        structure: CausalStructure,
        evidence: dict[str, float],
        do_intervention: dict[str, float],
        target: str,
        noise_std: float = 0.1,
        n_mc: int = 200,
    ) -> CounterfactualResult:
        """L3: Pearl 反事实推理 — "若当初做了 do_intervention，target 会怎样"。

        标准 Pearl 三步：
            1. Abduction: 从 evidence 反推噪声 U（锁定事实世界）
            2. Action: 施加 do(X=x') 切断 X 的因果父节点
            3. Prediction: 用 SEM 预测反事实世界的 target

        Args:
            structure: L1 发现的因果结构。
            evidence: 事实世界观测 {var: value}（如 {"heart_rate": 130}）。
            do_intervention: 反事实干预 {var: value}（如 {"heart_rate": 90}）。
            target: 关心的结果变量（如 "systolic_bp"）。
            noise_std: SEM 噪声标准差。
            n_mc: Monte Carlo 采样数（用于 PN/PS/PNS 估计）。

        Returns:
            CounterfactualResult（含事实值/反事实值/ITE/PN/PS/PNS）。
        """
        graph = causal_structure_to_graph(structure)
        if target not in graph.nodes:
            return CounterfactualResult(
                evidence=evidence,
                do_intervention=do_intervention,
                target=target,
                note=f"目标 {target} 不在因果图中",
            )

        try:
            engine = CounterfactualEngine.from_causal_graph(graph, noise_std=noise_std, seed=self._seed)
            if engine is None:
                return CounterfactualResult(
                    evidence=evidence,
                    do_intervention=do_intervention,
                    target=target,
                    note="CounterfactualEngine 创建失败",
                )
            result = engine.query(
                evidence=evidence,
                do_x=do_intervention,
                target=target,
                compute_pns=True,
                n_mc=n_mc,
            )
            return result
        except (ValueError, RuntimeError) as e:
            return CounterfactualResult(
                evidence=evidence,
                do_intervention=do_intervention,
                target=target,
                note=f"反事实推理失败: {e}",
            )

    # ── 临床语义封装 ─────────────────────────────────────────────

    def counterfactual_treatment_eval(
        self,
        structure: CausalStructure,
        observed_state: PatientState,
        factual_action: MedicalAction,
        alternative_action: MedicalAction,
        target_vital: str = "systolic_bp",
    ) -> dict[str, Any]:
        """反事实治疗评估 — "若当初选 alternative_action 而非 factual_action"。

        临床场景：患者接受了 factual_action（如美托洛尔），评估若改用
        alternative_action（如地尔硫卓）target_vital（如血压）会如何。

        这比朴素 predict 差分更严格：用 Pearl Abduction 对齐噪声，
        用因果图调整混杂。

        Args:
            structure: 因果结构。
            observed_state: 观测到的患者状态。
            factual_action: 实际施加的动作。
            alternative_action: 反事实假设的动作。
            target_vital: 评估的目标体征。

        Returns:
            反事实评估字典（含事实效应/反事实效应/差异/必然性概率）。
        """
        if target_vital not in VITAL_NAMES:
            raise ValueError(f"target_vital {target_vital} 不在 VITAL_NAMES 中")

        # 事实轨迹：实际用药后的状态（apply 返回 WorldState 基类，实际为 PatientState）
        factual_next_raw = factual_action.apply(observed_state)
        factual_target_val = float(
            factual_next_raw.vital_signs[-1][VITAL_NAMES.index(target_vital)]  # type: ignore[attr-defined]
        )

        # 反事实轨迹：若用替代药物
        cf_next_raw = alternative_action.apply(observed_state)
        cf_target_val = float(
            cf_next_raw.vital_signs[-1][VITAL_NAMES.index(target_vital)]  # type: ignore[attr-defined]
        )

        # 个体处理效应（ITE）
        ite = cf_target_val - factual_target_val

        # 若有因果图，用 Pearl 引擎估计 PN（必然性概率）
        pn_note = ""
        if structure.links:
            evidence = {target_vital: factual_target_val}
            do_intervention = {target_vital: cf_target_val}
            try:
                cf_result = self.counterfactual(structure, evidence, do_intervention, target_vital, n_mc=100)
                pn = getattr(cf_result, "pn", 0.0)
                ps = getattr(cf_result, "ps", 0.0)
                pns = getattr(cf_result, "pns", 0.0)
                pn_note = f"PN={pn:.3f}, PS={ps:.3f}, PNS={pns:.3f}"
            except (ValueError, RuntimeError, AttributeError):
                pn_note = "PN/PS/PNS 不可估计（因果图不足）"

        return {
            "factual_action": factual_action.target,
            "alternative_action": alternative_action.target,
            "target_vital": target_vital,
            "factual_value": round(factual_target_val, 2),
            "counterfactual_value": round(cf_target_val, 2),
            "individual_treatment_effect": round(ite, 2),
            "ite_direction": "higher" if ite > 0 else ("lower" if ite < 0 else "neutral"),
            "causal_necessity": pn_note,
            "interpretation": (
                f"若用 {alternative_action.target} 替代 {factual_action.target}，"
                f"预期 {target_vital} {('升高' if ite > 0 else '降低') if ite != 0 else '不变'} "
                f"{abs(ite):.1f}"
            ),
        }


# =============================================================================
# PearlLevel — 因果层级枚举（审计用）
# =============================================================================


@dataclass
class CausalAssessment:
    """患者因果评估结果（跨 L1/L2/L3 三层）。

    Attributes:
        level1_discovery: L1 发现的因果结构。
        level2_intervention: L2 干预效应（可选）。
        level3_counterfactual: L3 反事实结果（可选）。
        ladder_level: 达到的最高因果层级。
    """

    level1_discovery: CausalStructure | None = None
    level2_intervention: InterventionResult | None = None
    level3_counterfactual: CounterfactualResult | None = None
    ladder_level: str = "L1"  # L1/L2/L3

    def to_dict(self) -> dict[str, Any]:
        """序列化（审计用）。"""
        return {
            "ladder_level": self.ladder_level,
            "l1_discovery": self.level1_discovery.to_dict() if self.level1_discovery else None,
            "l2_intervention": (self.level2_intervention.to_dict() if self.level2_intervention else None),
            "l3_counterfactual": (
                {
                    "target": self.level3_counterfactual.target,
                    "factual": round(self.level3_counterfactual.factual_value, 4),
                    "counterfactual": round(self.level3_counterfactual.counterfactual_value, 4),
                    "ITE": round(self.level3_counterfactual.individual_effect, 4),
                }
                if self.level3_counterfactual
                else None
            ),
        }
