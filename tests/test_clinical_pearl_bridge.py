"""ClinicalPearlBridge 单元测试 — 方向三因果下沉验证。

验证 Pearl 三层因果推断接入医疗世界模型的核心契约：
    1. 适配器：CausalStructure → CausalGraph 正确映射
    2. L2 干预：do-calculus ATE 估计 + 后门调整集识别
    3. L3 反事实：Pearl 三步（Abduction/Action/Prediction）+ ITE
    4. 反事实治疗评估：药物 A vs 药物 B 的个体效应
    5. 决策引擎集成：causal_intervention_effect / counterfactual_evaluation
    6. 数值健壮性 + 边界合规
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._clinical_causal_discovery import CausalLink, CausalStructure
from mci_world_model.sdk._clinical_pearl_bridge import (
    CausalAssessment,
    ClinicalPearlBridge,
    causal_structure_to_graph,
)
from mci_world_model.sdk._clinical_world_state import (
    VITAL_NAMES,
    MedicalAction,
    PatientState,
)
from mci_world_model.sdk._counterfactual import CounterfactualResult
from mci_world_model.sdk._do_calculus import CausalGraph, InterventionResult

SEED = 42


def make_correlated_history(n=150, seed=SEED):
    """构造 HR→SBP→DBP 因果链的合成数据。"""
    rng = np.random.default_rng(seed)
    hr = rng.normal(80, 10, n)
    sbp = 120 + 0.6 * hr + rng.normal(0, 3, n)
    dbp = 80 + 0.4 * sbp + rng.normal(0, 2, n)
    spo2 = rng.normal(98, 1, n)
    rr = rng.normal(16, 2, n)
    temp = rng.normal(36.8, 0.3, n)
    gcs = np.full(n, 15.0)
    return np.stack([hr, sbp, dbp, spo2, rr, temp, gcs], axis=1)


# =============================================================================
# 1. 适配器：CausalStructure → CausalGraph
# =============================================================================


class TestCausalStructureToGraph:
    """验证因果结构到图的适配。"""

    def test_basic_conversion(self):
        """基本转换正确。"""
        structure = CausalStructure(
            links=[
                CausalLink(cause="A", effect="B", strength=0.8, direction=1.0),
                CausalLink(cause="B", effect="C", strength=0.5, direction=1.0),
            ],
            n_samples=100,
        )
        graph = causal_structure_to_graph(structure)
        assert isinstance(graph, CausalGraph)
        assert set(graph.nodes) == {"A", "B", "C"}
        assert graph.has_edge("A", "B")
        assert graph.has_edge("B", "C")
        assert not graph.has_edge("A", "C")

    def test_adjacency_weights_preserved(self):
        """邻接矩阵权重保留。"""
        structure = CausalStructure(links=[CausalLink(cause="X", effect="Y", strength=0.9, direction=1.0)])
        graph = causal_structure_to_graph(structure)
        assert graph.adjacency is not None
        assert graph.adjacency[0, 1] == pytest.approx(0.9)

    def test_empty_structure(self):
        """空结构产生空图。"""
        structure = CausalStructure()
        graph = causal_structure_to_graph(structure)
        assert len(graph.nodes) == 0

    def test_dedup_nodes(self):
        """重复节点去重。"""
        structure = CausalStructure(
            links=[
                CausalLink(cause="A", effect="B", strength=0.5, direction=1.0),
                CausalLink(cause="A", effect="C", strength=0.3, direction=1.0),
            ]
        )
        graph = causal_structure_to_graph(structure)
        assert len(graph.nodes) == 3  # A, B, C（A 不重复）


# =============================================================================
# 2. L1 发现（委托验证）
# =============================================================================


class TestLevel1Discovery:
    """验证 L1 发现（委托 ClinicalCausalDiscovery）。"""

    def test_discover_finds_correlated(self):
        """发现相关变量的因果边。"""
        bridge = ClinicalPearlBridge()
        history = make_correlated_history()
        structure = bridge.discover(history)
        assert isinstance(structure, CausalStructure)
        assert structure.n_samples == 150
        # HR-SBP 强相关，应找到边
        connected = {(lk.cause, lk.effect) for lk in structure.links}
        assert any(("heart_rate" in pair and "systolic_bp" in pair) for pair in connected)


# =============================================================================
# 3. L2 干预（do-calculus）
# =============================================================================


class TestLevel2Intervention:
    """验证 L2 干预效应估计。"""

    def test_intervene_returns_intervention_result(self):
        """intervene 返回 InterventionResult。"""
        bridge = ClinicalPearlBridge()
        history = make_correlated_history()
        structure = bridge.discover(history)
        data_dict = {name: history[:, i] for i, name in enumerate(VITAL_NAMES)}
        # 找图中存在的边
        # make_correlated_history 合成强相关数据, 因果发现应检出边;
        # 若未检出属于回归, 提前失败而非 silent skip。
        assert structure.links, "因果发现应检出边 (合成数据强相关)"
        result = bridge.intervene(structure, structure.links[0].cause, structure.links[0].effect, data_dict)
        assert isinstance(result, InterventionResult)
        assert result.intervention.startswith("do(")

    def test_intervene_unknown_var_returns_none_method(self):
        """未知变量返回 method=none。"""
        bridge = ClinicalPearlBridge()
        structure = CausalStructure(links=[CausalLink(cause="A", effect="B", strength=0.5, direction=1.0)])
        result = bridge.intervene(structure, "nonexistent", "B")
        assert result.method == "none"

    def test_identify_confounders(self):
        """识别后门混杂变量。"""
        bridge = ClinicalPearlBridge()
        history = make_correlated_history()
        structure = bridge.discover(history)
        # HR→SBP 的混杂（若有回边）
        confs = bridge.identify_confounders(structure, "heart_rate", "systolic_bp")
        assert isinstance(confs, list)  # 可能为空或含变量

    def test_intervene_ate_finite(self):
        """ATE 是有限值。"""
        bridge = ClinicalPearlBridge()
        history = make_correlated_history(n=200)
        structure = bridge.discover(history)
        data_dict = {name: history[:, i] for i, name in enumerate(VITAL_NAMES)}
        if structure.links:
            result = bridge.intervene(structure, structure.links[0].cause, structure.links[0].effect, data_dict)
            if result.method != "none":
                assert np.isfinite(result.ate)


# =============================================================================
# 4. L3 反事实（Pearl 三步）
# =============================================================================


class TestLevel3Counterfactual:
    """验证 L3 反事实推理。"""

    def test_counterfactual_returns_result(self):
        """counterfactual 返回 CounterfactualResult。"""
        bridge = ClinicalPearlBridge()
        history = make_correlated_history()
        structure = bridge.discover(history)
        # 合成强相关数据, 因果发现应检出边
        assert structure.links, "因果发现应检出边 (合成数据强相关)"
        cause, effect = structure.links[0].cause, structure.links[0].effect
        result = bridge.counterfactual(
            structure,
            evidence={cause: 100.0},
            do_intervention={cause: 60.0},
            target=effect,
            n_mc=50,
        )
        assert isinstance(result, CounterfactualResult)
        assert result.target == effect

    def test_counterfactual_unknown_target(self):
        """未知目标返回 note。"""
        bridge = ClinicalPearlBridge()
        structure = CausalStructure(links=[CausalLink(cause="A", effect="B", strength=0.5, direction=1.0)])
        result = bridge.counterfactual(structure, {"A": 1.0}, {"A": 2.0}, "nonexistent")
        assert "不在因果图中" in (result.note or "") or result.target == "nonexistent"

    def test_counterfactual_ite_direction(self):
        """反事实 ITE 方向合理。"""
        bridge = ClinicalPearlBridge()
        history = make_correlated_history(n=200)
        structure = bridge.discover(history)
        # 合成强相关数据, 因果发现应检出边
        assert structure.links, "因果发现应检出边 (合成数据强相关)"
        cause, effect = structure.links[0].cause, structure.links[0].effect
        # 大幅干预
        result = bridge.counterfactual(
            structure,
            evidence={cause: 100.0},
            do_intervention={cause: 50.0},
            target=effect,
            n_mc=50,
        )
        # ITE 应为有限值
        assert np.isfinite(result.individual_effect)


# =============================================================================
# 5. 反事实治疗评估
# =============================================================================


class TestCounterfactualTreatmentEval:
    """验证药物 A vs 药物 B 的反事实评估。"""

    def test_metoprolol_vs_dopamine_heart_rate(self):
        """美托洛尔（降心率）vs 多巴胺（升心率）的反事实。"""
        bridge = ClinicalPearlBridge()
        history = make_correlated_history()
        structure = bridge.discover(history)
        state = PatientState(vital_signs=np.array([[130, 140, 90, 98, 20, 37, 15]]))
        factual = MedicalAction(target="metoprolol", magnitude=5.0)  # 降心率
        alt = MedicalAction(target="dopamine", magnitude=5.0)  # 升心率
        result = bridge.counterfactual_treatment_eval(structure, state, factual, alt, "heart_rate")
        # 多巴胺替代美托洛尔 → 心率应升高（ITE > 0）
        assert result["ite_direction"] == "higher"
        assert result["individual_treatment_effect"] > 0
        assert "interpretation" in result
        # 任务2-B: 额外验证 Pearl 三步真跑通——causal_necessity 字段应非空
        # (含 PN/PS/PNS 概率, 而非 "PN/PS/PNS 不可估计")
        assert "causal_necessity" in result, "反事实评估应含 causal_necessity 字段"
        cn = result["causal_necessity"]
        assert isinstance(cn, str) and len(cn) > 0, f"causal_necessity 应非空 (Pearl 三步应跑通), 实际: {cn!r}"
        assert "不可估计" not in cn, f"Pearl 反事实应成功估计 PN/PS/PNS, 实际: {cn!r}"

    def test_invalid_target_raises(self):
        """无效体征名抛错。"""
        bridge = ClinicalPearlBridge()
        structure = CausalStructure()
        state = PatientState(vital_signs=np.array([[80, 120, 80, 98, 16, 36.8, 15]]))
        with pytest.raises(ValueError, match="不在 VITAL_NAMES"):
            bridge.counterfactual_treatment_eval(
                structure,
                state,
                MedicalAction(target="metoprolol", magnitude=5.0),
                MedicalAction(target="dopamine", magnitude=5.0),
                "invalid_vital",
            )


# =============================================================================
# 6. 决策引擎集成
# =============================================================================


class TestDecisionEngineIntegration:
    """验证决策引擎的因果下沉方法。"""

    def test_causal_intervention_effect(self):
        """决策引擎的 L2 干预效应估计。"""
        from mci_world_model.sdk import ClinicalDecisionEngine

        engine = ClinicalDecisionEngine()
        history = make_correlated_history(n=100)
        result = engine.causal_intervention_effect(history, "systolic_bp", "diastolic_bp")
        assert isinstance(result, dict)
        # 任务2-A: 拆掉永真 `or` 断言。InterventionResult.to_dict() 必含
        # intervention/target/ate/method 四个键 (见 _do_calculus.py)。
        assert "intervention" in result, "干预效应 dict 应含 intervention"
        assert "target" in result, "干预效应 dict 应含 target"
        assert "method" in result, "干预效应 dict 应含 method"
        assert "ate" in result, "干预效应 dict 应含 ate"
        assert isinstance(result["ate"], (int, float))
        # 合成强相关数据, systolic_bp→diastolic_bp 应能估计 (method != none)
        assert result["method"] != "none", f"强相关数据应可估计 ATE, method={result['method']!r}"
        assert np.isfinite(result["ate"]), "ATE 应为有限值"

    def test_counterfactual_evaluation(self):
        """决策引擎的 L3 反事实治疗评估。"""
        from mci_world_model.sdk import ClinicalDecisionEngine

        engine = ClinicalDecisionEngine()
        history = make_correlated_history(n=100)
        state = PatientState(vital_signs=np.array([[130, 140, 90, 98, 20, 37, 15]]))
        result = engine.counterfactual_evaluation(
            history,
            state,
            MedicalAction(target="metoprolol", magnitude=5.0),
            MedicalAction(target="dopamine", magnitude=5.0),
            "heart_rate",
        )
        assert "interpretation" in result
        assert "individual_treatment_effect" in result


# =============================================================================
# 7. CausalAssessment 数据类
# =============================================================================


class TestCausalAssessment:
    """验证跨层级评估结果。"""

    def test_to_dict_serializable(self):
        """CausalAssessment.to_dict 可序列化。"""
        assessment = CausalAssessment(ladder_level="L2")
        d = assessment.to_dict()
        assert d["ladder_level"] == "L2"
        assert d["l1_discovery"] is None


# =============================================================================
# 8. 数值健壮性
# =============================================================================


class TestNumericRobustness:
    """验证数值健壮性。"""

    def test_insufficient_data_handled(self):
        """数据不足不崩溃。"""
        bridge = ClinicalPearlBridge()
        structure = bridge.discover(np.array([[80, 120, 80, 98, 16, 36.8, 15]]))
        # 少量数据，intervene 不应崩溃
        result = bridge.intervene(structure, "heart_rate", "systolic_bp")
        assert isinstance(result, InterventionResult)

    def test_constant_data_handled(self):
        """常数数据不崩溃。"""
        bridge = ClinicalPearlBridge()
        constant = np.full((30, 7), 80.0)
        structure = bridge.discover(constant)
        assert len(structure.links) == 0  # 常数无因果
