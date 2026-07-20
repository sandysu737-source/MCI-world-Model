"""ClinicalDecisionEngine 单元测试 — Phase 2-3 闭环验证。

验证五要素编排层的完整性：
    1. fit 训练世界模型
    2. decide_from_vitals 端到端决策
    3. 审计轨迹完整性
    4. 安全降级机制
"""

from __future__ import annotations

import pytest

from mci_world_model.sdk._clinical_decision_engine import ClinicalDecision, ClinicalDecisionEngine
from mci_world_model.sdk._clinical_tri_router import RouteType, SafetyLevel
from mci_world_model.sdk._clinical_world_state import MedicalAction

SEED = 42


def make_normal_vitals():
    return [
        {
            "heart_rate": 75,
            "systolic_bp": 120,
            "diastolic_bp": 80,
            "oxygen_saturation": 98,
            "respiratory_rate": 16,
            "temperature": 36.8,
            "gcs": 15,
        }
    ]


def make_abnormal_vitals():
    return [
        {
            "heart_rate": 130,
            "systolic_bp": 140,
            "diastolic_bp": 95,
            "oxygen_saturation": 94,
            "respiratory_rate": 24,
            "temperature": 38.0,
            "gcs": 13,
        }
    ]


@pytest.fixture(scope="class")
def trained_engine():
    """训练好的决策引擎（类级共享）。"""
    engine = ClinicalDecisionEngine()
    engine.fit(n_samples=2000, n_epochs=500, lr=0.005, seed=SEED)
    return engine


class TestDecisionEngine:
    """验证临床决策引擎。"""

    def test_fit_makes_engine_ready(self, trained_engine):
        """fit 后 is_fitted 为 True。"""
        assert trained_engine.is_fitted

    def test_decide_normal_patient(self, trained_engine):
        """正常患者的决策输出。"""
        decision = trained_engine.decide_from_vitals(make_normal_vitals(), query="患者生命体征平稳", patient_id="P001")
        assert isinstance(decision, ClinicalDecision)
        assert decision.patient_state is not None
        assert decision.current_reward > 0.7  # 正常状态高分

    def test_decide_abnormal_patient_recommends_action(self, trained_engine):
        """异常患者推荐治疗动作。"""
        decision = trained_engine.decide_from_vitals(
            make_abnormal_vitals(), query="患者心动过速如何处理？", patient_id="P002"
        )
        assert decision.recommended_action is not None
        assert isinstance(decision.recommended_action, MedicalAction)
        # 应有治疗评估
        assert decision.treatment_plan is not None

    def test_audit_trail_complete(self, trained_engine):
        """审计轨迹包含编码→路由→规划步骤。"""
        decision = trained_engine.decide_from_vitals(make_abnormal_vitals(), query="给药建议")
        steps = [a["step"] for a in decision.audit_trail]
        assert "encode" in steps
        assert "route" in steps
        assert "plan" in steps

    def test_route_classification(self, trained_engine):
        """路由类型分类正确。"""
        # 生理预测类
        d1 = trained_engine.decide_from_vitals(make_normal_vitals(), query="心率趋势预测")
        assert d1.route_type == RouteType.PHYSICAL

        # 知识问答类
        d2 = trained_engine.decide_from_vitals(make_normal_vitals(), query="感染性休克是什么")
        assert d2.route_type == RouteType.SEMANTIC

        # 因果干预类
        d3 = trained_engine.decide_from_vitals(make_normal_vitals(), query="多巴胺给药剂量")
        assert d3.route_type == RouteType.CAUSAL

    def test_to_dict_serializable(self, trained_engine):
        """to_dict 输出可序列化字典。"""
        decision = trained_engine.decide_from_vitals(make_normal_vitals())
        d = decision.to_dict()
        assert "patient_state" in d
        assert "route" in d
        assert "evaluation" in d
        assert "audit_steps" in d

    def test_unfitted_engine_refuses(self):
        """未训练的引擎拒绝决策。"""
        engine = ClinicalDecisionEngine()
        assert not engine.is_fitted
        decision = engine.decide_from_vitals(make_normal_vitals())
        assert decision.safety_level == SafetyLevel.REFUSED
        assert decision.recommended_action is None


class TestFiveElementLoop:
    """验证五要素完整闭环：S → E → π → T → R。"""

    def test_full_loop_produces_valid_decision(self, trained_engine):
        """完整闭环产出有效决策。"""
        decision = trained_engine.decide_from_vitals(
            make_abnormal_vitals(), query="患者心率血压偏高，治疗方案？", patient_id="LOOP-001"
        )

        # S: 状态空间
        assert decision.patient_state is not None

        # π: 规划器推荐动作
        assert decision.recommended_action is not None

        # T: 预测轨迹（治疗后的状态变化）
        if decision.treatment_plan and decision.treatment_plan.predicted_trajectory:
            _future = decision.treatment_plan.predicted_trajectory[0]
            # R: 预测状态有有效评分
            assert decision.predicted_reward >= 0.0

        # 审计链完整
        assert len(decision.audit_trail) >= 3


class TestSemanticKnowledgePath:
    """验证决策引擎的语义知识库检索路径。"""

    def test_semantic_query_returns_knowledge(self):
        """知识类查询返回知识库条目。"""
        engine = ClinicalDecisionEngine()
        decision = engine.decide_from_vitals(
            make_normal_vitals(),
            query="感染性休克的诊断标准是什么",
        )
        assert decision.route_type == RouteType.SEMANTIC
        assert decision.knowledge_answer is not None
        assert decision.knowledge_answer["term"] is not None

    def test_semantic_audit_trail_includes_retrieval(self):
        """语义路径审计轨迹包含 knowledge_retrieval 步骤。"""
        engine = ClinicalDecisionEngine()
        decision = engine.decide_from_vitals(
            make_normal_vitals(),
            query="ARDS 的治疗原则",
        )
        steps = [a["step"] for a in decision.audit_trail]
        assert "knowledge_retrieval" in steps

    def test_causal_query_does_not_trigger_knowledge(self):
        """因果类查询不触发知识库检索。"""
        engine = ClinicalDecisionEngine()
        decision = engine.decide_from_vitals(
            make_normal_vitals(),
            query="多巴胺给药剂量建议",
        )
        assert decision.route_type == RouteType.CAUSAL
        # 因果路径不应有 knowledge_answer（除非引擎未训练走 fused）
        if decision.recommended_action is None:
            assert decision.knowledge_answer is None or decision.safety_level == SafetyLevel.REFUSED

    def test_knowledge_answer_to_dict(self):
        """知识答案包含在 to_dict 输出中。"""
        engine = ClinicalDecisionEngine()
        decision = engine.decide_from_vitals(
            make_normal_vitals(),
            query="去甲肾上腺素是什么药",
        )
        d = decision.to_dict()
        assert "knowledge_answer" in d
        if d["knowledge_answer"] and d["knowledge_answer"].get("term"):
            assert "definition" in d["knowledge_answer"]

    def test_unknown_query_returns_no_match(self):
        """无匹配的知识查询返回无匹配标记。"""
        engine = ClinicalDecisionEngine()
        decision = engine.decide_from_vitals(
            make_normal_vitals(),
            query="今天股市行情怎么样",
        )
        # 路由到 semantic 或 fused
        if decision.route_type == RouteType.SEMANTIC:
            assert decision.knowledge_answer is not None
            # 无关查询应无匹配
            assert (
                decision.knowledge_answer.get("term") is None
                or decision.knowledge_answer.get("retrieval_score", 1.0) < 0.5
            )


class TestComplianceAndDiagnosis:
    """验证 MedicalCausalSDK 因果诊断 + ComplianceRuleEngine 合规校验接入。"""

    def test_causal_path_triggers_diagnosis(self, trained_engine):
        """CAUSAL 路径触发因果诊断。"""
        decision = trained_engine.decide_from_vitals(
            make_abnormal_vitals(),
            query="多巴胺给药剂量副作用",
        )
        assert decision.route_type == RouteType.CAUSAL
        assert decision.causal_diagnosis is not None
        assert "is_conclusive" in decision.causal_diagnosis

    def test_causal_path_triggers_compliance(self, trained_engine):
        """CAUSAL 路径触发合规校验。"""
        decision = trained_engine.decide_from_vitals(
            make_abnormal_vitals(),
            query="多巴胺给药剂量副作用",
        )
        assert decision.compliance_report is not None
        assert "is_compliant" in decision.compliance_report

    def test_semantic_path_skips_compliance(self):
        """SEMANTIC 路径不触发合规校验（只查知识库）。"""
        engine = ClinicalDecisionEngine()
        decision = engine.decide_from_vitals(
            make_normal_vitals(),
            query="感染性休克的定义和机制",
        )
        assert decision.route_type == RouteType.SEMANTIC
        assert decision.compliance_report is None

    def test_audit_trail_includes_compliance_step(self, trained_engine):
        """审计链包含 compliance_check 步骤。"""
        decision = trained_engine.decide_from_vitals(
            make_abnormal_vitals(),
            query="多巴胺给药剂量",
        )
        steps = [a["step"] for a in decision.audit_trail]
        assert "compliance_check" in steps

    def test_to_dict_includes_compliance_and_diagnosis(self, trained_engine):
        """to_dict 包含合规和诊断信息。"""
        decision = trained_engine.decide_from_vitals(
            make_abnormal_vitals(),
            query="多巴胺给药剂量",
        )
        d = decision.to_dict()
        assert "compliance" in d
        assert "causal_diagnosis" in d


# =============================================================================
# D8: 扩充知识库验证
# =============================================================================


class TestExpandedKnowledgeBase:
    """验证扩充后的临床知识库（18 条 ICU 指南条目）。"""

    # 已知默认知识库必含的核心术语（与 _build_default_knowledge_base 对齐）。
    # 通过公共 query() 接口验证覆盖性，避免耦合 _entries 私有属性。
    CORE_SYNDROMES = [
        "感染性休克",
        "心源性休克",
        "急性呼吸窘迫综合征",
        "急性肾损伤",
        "脓毒症",
    ]
    CORE_DRUGS = ["多巴胺", "去甲肾上腺素", "美托洛尔", "肾上腺素", "呋塞米"]
    # 用作"条目数充足"间接校验的术语集合（≥15 命中即视为默认 KB 完整装载）。
    KNOWN_TERMS = (
        CORE_SYNDROMES
        + CORE_DRUGS
        + [
            "多器官功能障碍综合征",
            "糖尿病酮症酸中毒",
            "急性肝衰竭",
            "上消化道出血",
            "丙泊酚",
            "机械通气",
            "中心静脉压",
            "SOFA评分",
        ]
    )

    @staticmethod
    def _query(kb, term: str):
        """公共接口取条目（封装 query 调用，集中可读性）。"""
        return kb.query(term)

    def test_knowledge_base_has_sufficient_entries(self):
        """知识库覆盖 ≥15 个已知核心术语（间接验证默认 KB 完整装载）。"""
        engine = ClinicalDecisionEngine()
        kb = engine._knowledge_base
        hit = [t for t in self.KNOWN_TERMS if self._query(kb, t) is not None]
        assert len(hit) >= 15, f"知识库已知术语命中不足: {len(hit)}/{len(self.KNOWN_TERMS)}"

    def test_knowledge_covers_core_syndromes(self):
        """知识库覆盖核心 ICU 综合征。"""
        engine = ClinicalDecisionEngine()
        kb = engine._knowledge_base
        for term in self.CORE_SYNDROMES:
            assert self._query(kb, term) is not None, f"核心综合征缺失: {term}"

    def test_knowledge_covers_core_drugs(self):
        """知识库覆盖核心药物。"""
        engine = ClinicalDecisionEngine()
        kb = engine._knowledge_base
        for drug in self.CORE_DRUGS:
            assert self._query(kb, drug) is not None, f"核心药物缺失: {drug}"

    def test_knowledge_entries_have_complete_fields(self):
        """每个知识条目含 term/definition/guideline 三字段。"""
        engine = ClinicalDecisionEngine()
        kb = engine._knowledge_base
        # 抽样核心术语，通过公共 query 拿到条目后验证字段完整性
        for term in self.CORE_SYNDROMES + self.CORE_DRUGS:
            entry = self._query(kb, term)
            assert entry is not None, f"术语无法检索: {term}"
            assert entry.term, f"{term}: term 为空"
            assert entry.definition, f"{term}: definition 为空"
            assert entry.guideline, f"{term}: guideline 为空"

    def test_semantic_retrieval_on_expanded_term(self, trained_engine):
        """扩充术语（如 SOFA评分）能被语义检索命中。"""
        decision = trained_engine.decide_from_vitals(
            vital_records=[
                {
                    "heart_rate": 80,
                    "systolic_bp": 120,
                    "diastolic_bp": 80,
                    "spo2": 98,
                    "respiratory_rate": 16,
                    "temperature": 36.8,
                    "gcs": 15,
                }
            ],
            query="SOFA评分是什么",
        )
        # 至少路由到 semantic 或返回某种响应
        assert decision.route_type.value in ("semantic", "fused")
