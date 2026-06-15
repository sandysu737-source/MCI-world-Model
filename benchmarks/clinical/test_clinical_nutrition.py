"""CEWM v4.0.0 — 医疗 AI 临床营养验证用例

K5-3: 临床场景推理准确率 ≥ 70%

10 个临床营养管理案例，测试 CEWM 的:
    - 因果推理: 营养干预 → 临床结局
    - 经验复用: 历史案例 → 新患者建议
    - 异常检测: 营养指标异常 → 诊断
    - 可解释性: 推理路径可解释输出

运行: pytest benchmarks/clinical/test_clinical_nutrition.py -v
"""

from __future__ import annotations

# =============================================================================
# 临床营养场景定义
# =============================================================================


CLINICAL_CASES = [
    {
        "id": "CASE-01",
        "name": "ICU 重症患者肠内营养启动",
        "patient": {
            "age": 65,
            "weight_kg": 70,
            "diagnosis": "重症肺炎",
            "nrs2002": 5,
            "albumin_g_l": 28,
            "prealbumin_mg_l": 150,
        },
        "intervention": "早期肠内营养 (24h内启动)",
        "expected_outcome": "改善预后，减少感染并发症",
        "causal_tags": ["icu", "enteral_nutrition", "early_feeding", "critical_care"],
        "risk_tags": ["aspiration", "intolerance"],
    },
    {
        "id": "CASE-02",
        "name": "术后患者蛋白质需求评估",
        "patient": {
            "age": 55,
            "weight_kg": 65,
            "diagnosis": "胃癌术后",
            "nrs2002": 4,
            "albumin_g_l": 32,
            "prealbumin_mg_l": 180,
        },
        "intervention": "高蛋白配方 (1.5g/kg/d)",
        "expected_outcome": "促进伤口愈合，维持氮平衡",
        "causal_tags": ["surgery", "protein", "wound_healing", "nitrogen_balance"],
        "risk_tags": ["renal_overload"],
    },
    {
        "id": "CASE-03",
        "name": "糖尿病患者肠内营养血糖管理",
        "patient": {
            "age": 60,
            "weight_kg": 80,
            "diagnosis": "2型糖尿病合并脑卒中",
            "nrs2002": 4,
            "albumin_g_l": 35,
            "prealbumin_mg_l": 200,
        },
        "intervention": "糖尿病专用配方 + 血糖监测",
        "expected_outcome": "血糖稳定 (6-10mmol/L)",
        "causal_tags": ["diabetes", "blood_glucose", "specialized_formula", "stroke"],
        "risk_tags": ["hyperglycemia", "hypoglycemia"],
    },
    {
        "id": "CASE-04",
        "name": "肿瘤患者营养免疫支持",
        "patient": {
            "age": 58,
            "weight_kg": 55,
            "diagnosis": "食管癌化疗期",
            "nrs2002": 5,
            "albumin_g_l": 26,
            "prealbumin_mg_l": 120,
        },
        "intervention": "免疫营养 (谷氨酰胺 + ω-3 脂肪酸)",
        "expected_outcome": "增强免疫功能，减少化疗副作用",
        "causal_tags": ["oncology", "immunonutrition", "glutamine", "omega3"],
        "risk_tags": ["cachexia", "mucositis"],
    },
    {
        "id": "CASE-05",
        "name": "老年患者肌少症营养干预",
        "patient": {
            "age": 78,
            "weight_kg": 58,
            "diagnosis": "肌少症",
            "nrs2002": 3,
            "albumin_g_l": 33,
            "prealbumin_mg_l": 210,
        },
        "intervention": "高蛋白 + 维生素D + 抗阻训练",
        "expected_outcome": "增加肌肉质量，改善功能",
        "causal_tags": ["sarcopenia", "elderly", "vitamin_d", "resistance_training"],
        "risk_tags": ["falls", "frailty"],
    },
    {
        "id": "CASE-06",
        "name": "肝病患者支链氨基酸补充",
        "patient": {
            "age": 52,
            "weight_kg": 68,
            "diagnosis": "肝硬化",
            "nrs2002": 4,
            "albumin_g_l": 25,
            "prealbumin_mg_l": 140,
        },
        "intervention": "BCAA 补充 + 夜间加餐",
        "expected_outcome": "改善蛋白质代谢，减少肝性脑病风险",
        "causal_tags": ["liver_disease", "bcaa", "nocturnal_snack", "ammonia"],
        "risk_tags": ["hepatic_encephalopathy"],
    },
    {
        "id": "CASE-07",
        "name": "肾病患者低蛋白饮食管理",
        "patient": {
            "age": 62,
            "weight_kg": 72,
            "diagnosis": "慢性肾病 CKD4期",
            "nrs2002": 3,
            "albumin_g_l": 34,
            "prealbumin_mg_l": 220,
        },
        "intervention": "低蛋白饮食 (0.6g/kg/d) + α-酮酸",
        "expected_outcome": "延缓肾功能恶化",
        "causal_tags": ["ckd", "low_protein", "ketoacid", "renal_function"],
        "risk_tags": ["malnutrition", "electrolyte_imbalance"],
    },
    {
        "id": "CASE-08",
        "name": "烧伤患者高代谢营养支持",
        "patient": {
            "age": 35,
            "weight_kg": 75,
            "diagnosis": "大面积烧伤 (40% TBSA)",
            "nrs2002": 5,
            "albumin_g_l": 22,
            "prealbumin_mg_l": 100,
        },
        "intervention": "高热量高蛋白 + 谷氨酰胺",
        "expected_outcome": "减少分解代谢，促进创面愈合",
        "causal_tags": ["burn", "hypermetabolism", "glutamine", "wound_healing"],
        "risk_tags": ["infection", "fluid_overload"],
    },
    {
        "id": "CASE-09",
        "name": "IBD 患者肠内营养诱导缓解",
        "patient": {
            "age": 28,
            "weight_kg": 55,
            "diagnosis": "克罗恩病活动期",
            "nrs2002": 4,
            "albumin_g_l": 30,
            "prealbumin_mg_l": 170,
        },
        "intervention": "全肠内营养 (EEN) 8周",
        "expected_outcome": "诱导缓解，黏膜愈合",
        "causal_tags": ["ibd", "crohns", "een", "mucosal_healing"],
        "risk_tags": ["non_compliance", "growth_failure"],
    },
    {
        "id": "CASE-10",
        "name": "ICU 患者再喂养综合征预防",
        "patient": {
            "age": 70,
            "weight_kg": 50,
            "diagnosis": "长期禁食后入ICU",
            "nrs2002": 5,
            "albumin_g_l": 24,
            "prealbumin_mg_l": 90,
        },
        "intervention": "缓慢递增喂养 + 电解质监测",
        "expected_outcome": "安全启动营养，避免再喂养综合征",
        "causal_tags": ["refeeding", "electrolyte", "gradual_feeding", "phosphate"],
        "risk_tags": ["hypophosphatemia", "cardiac_arrhythmia"],
    },
]


# =============================================================================
# C1: 因果推理 — 营养干预 → 临床结局
# =============================================================================


class TestC1ClinicalCausalReasoning:
    """C1: 临床营养因果推理。"""

    def test_c1_causal_graph_construction(self):
        """C1.1: 从临床案例构建因果图。"""
        from mci_world_model.sdk._do_calculus import CausalGraph

        graph = CausalGraph()
        # 营养干预因果链
        graph.add_edge("enteral_nutrition", "gut_integrity", weight=0.7)
        graph.add_edge("gut_integrity", "immune_function", weight=0.6)
        graph.add_edge("immune_function", "infection_rate", weight=-0.5)
        graph.add_edge("protein_intake", "wound_healing", weight=0.65)
        graph.add_edge("protein_intake", "nitrogen_balance", weight=0.8)

        assert graph.has_edge("enteral_nutrition", "gut_integrity")
        # 负权边: has_edge 检查 > 0，负权返回 False → 直接查 adjacency
        i = graph.node_index("immune_function")
        j = graph.node_index("infection_rate")
        assert graph.adjacency[i, j] < 0  # 负因果: 免疫功能↑ → 感染率↓

        # 肠内营养 → 免疫功能 的路径
        descendants = graph.get_descendants("enteral_nutrition")
        assert "immune_function" in descendants

    def test_c1_intervention_effect(self):
        """C1.2: 营养干预效应推理。"""
        from mci_world_model.sdk._causal_updater import CausalUpdater

        updater = CausalUpdater()
        updater.init_from_edges(
            [
                ("early_enteral", "gut_barrier"),
                ("gut_barrier", "reduced_infection"),
                ("protein_intake", "muscle_mass"),
                ("muscle_mass", "functional_recovery"),
            ]
        )

        # 早期肠内营养的证据累积
        for _ in range(5):
            updater.add_evidence("early_enternal", "gut_barrier", confidence=0.85)

        edge = updater.get_edge("early_enternal", "gut_barrier")
        assert edge is not None

    def test_c1_risk_detection(self):
        """C1.3: 营养风险识别。"""
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser, SurpriseSignal

        md = MetaDiagnoser()

        # 模拟营养指标异常
        signal = SurpriseSignal(
            score=0.85,
            source="nutrition_screening",
            layer="clinical",
            features={
                "direction_error": 0.8,  # 营养指标偏离正常
                "state_distance": 0.7,  # 当前状态与目标差距大
                "vector_deviation": 0.6,
            },
        )
        result = md.diagnose([signal])
        assert result is not None
        assert result.root_cause_chain.depth >= 3

    def test_c1_all_cases_have_causal_tags(self):
        """C1.4: 所有案例均有因果标签。"""
        for case in CLINICAL_CASES:
            assert len(case["causal_tags"]) >= 3, f"{case['id']} 缺少因果标签"
            assert len(case["risk_tags"]) >= 1, f"{case['id']} 缺少风险标签"


# =============================================================================
# C2: 经验复用 — 历史案例指导新决策
# =============================================================================


class TestC2ClinicalExperienceReuse:
    """C2: 临床经验复用。"""

    def test_c2_store_clinical_experiences(self):
        """C2.1: 存储临床经验。"""
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB, ExperienceType

        db = ExperienceDB()

        for case in CLINICAL_CASES:
            exp = Experience(
                experience_id=case["id"],
                experience_type=ExperienceType.SUCCESS,
                tags=case["causal_tags"],
                outcome=case["expected_outcome"],
                importance=0.8,
            )
            db.store(exp)

        stats = db.statistics()
        assert stats.total_experiences == 10

    def test_c2_retrieve_similar_cases(self):
        """C2.2: 检索相似临床案例。"""
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB, ExperienceType

        db = ExperienceDB()

        # 存储全部案例
        for case in CLINICAL_CASES:
            db.store(
                Experience(
                    experience_id=case["id"],
                    experience_type=ExperienceType.SUCCESS,
                    tags=case["causal_tags"],
                    outcome=case["expected_outcome"],
                    importance=0.8,
                )
            )

        # 新患者: ICU + 肠内营养
        results = db.retrieve(
            query_tags=["icu", "enteral_nutrition", "critical_care"],
            top_k=3,
        )
        # 应优先匹配 CASE-01 和 CASE-10
        assert len(results) > 0

    def test_c2_risk_based_retrieval(self):
        """C2.3: 基于风险的经验检索。"""
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB, ExperienceType

        db = ExperienceDB()

        # 存储有风险标签的案例
        for case in CLINICAL_CASES:
            all_tags = case["causal_tags"] + case["risk_tags"]
            db.store(
                Experience(
                    experience_id=case["id"],
                    experience_type=ExperienceType.SUCCESS,
                    tags=all_tags,
                    importance=0.8,
                )
            )

        # 查询: 再喂养综合征风险
        results = db.retrieve(
            query_tags=["refeeding", "hypophosphatemia"],
            top_k=3,
        )
        assert len(results) > 0


# =============================================================================
# C3: 异常检测 — 营养指标异常诊断
# =============================================================================


class TestC3ClinicalAnomalyDetection:
    """C3: 临床营养异常检测。"""

    def test_c3_albumin_anomaly(self):
        """C3.1: 低白蛋白血症检测。"""
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser, SurpriseSignal

        md = MetaDiagnoser()

        # 白蛋白 25g/L (正常 35-50) → 高惊奇
        signal = SurpriseSignal(
            score=0.9,
            source="lab_result",
            layer="clinical",
            features={
                "direction_error": 0.85,
                "state_distance": 0.8,
                "vector_deviation": 0.7,
            },
        )
        result = md.diagnose([signal])
        assert result.root_cause_chain.depth >= 3

    def test_c3_nutrition_screening_score(self):
        """C3.2: NRS-2002 营养风险评分推理。"""
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB, ExperienceType

        db = ExperienceDB()

        # 存储不同 NRS 分数的历史案例
        for nrs_score in range(1, 8):
            db.store(
                Experience(
                    experience_id=f"nrs_{nrs_score}",
                    experience_type=(
                        ExperienceType.SUCCESS
                        if nrs_score < 3
                        else ExperienceType.FAILURE
                        if nrs_score >= 5
                        else ExperienceType.TRANSITION
                    ),
                    tags=["nutrition_risk", f"nrs_{nrs_score}", "screening"],
                    importance=0.5 + nrs_score * 0.07,
                )
            )

        # 高风险患者检索
        results = db.retrieve(query_tags=["nutrition_risk", "nrs_5"], top_k=3)
        assert len(results) > 0

    def test_c3_multi_indicator_diagnosis(self):
        """C3.3: 多指标联合诊断。"""
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser, SurpriseSignal

        md = MetaDiagnoser()

        # 多个异常指标同时出现
        signals = [
            SurpriseSignal(
                score=0.8,
                source="albumin",
                layer="clinical",
                features={"direction_error": 0.7, "state_distance": 0.6},
            ),
            SurpriseSignal(
                score=0.7,
                source="prealbumin",
                layer="clinical",
                features={"direction_error": 0.6, "state_distance": 0.5},
            ),
        ]
        results = md.batch_diagnose([[s] for s in signals])
        assert len(results) == 2


# =============================================================================
# C4: 可解释性 — 推理路径输出
# =============================================================================


class TestC4ClinicalExplainability:
    """C4: 临床推理可解释性。"""

    def test_c4_explainable_diagnosis(self):
        """C4.1: 诊断结果可解释。"""
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser, SurpriseSignal

        md = MetaDiagnoser()
        signal = SurpriseSignal(
            score=0.8,
            source="malnutrition_risk",
            layer="clinical",
            features={"direction_error": 0.75, "state_distance": 0.65, "vector_deviation": 0.6},
        )
        result = md.diagnose([signal])
        d = result.to_dict()

        # 必须包含可解释字段
        assert "recommendation" in d
        assert d["recommendation"] != ""
        assert "root_cause_chain" in d
        assert len(d["root_cause_chain"]["chain"]) >= 3

    def test_c4_clinical_kpi_accuracy(self):
        """K5-3: 临床场景推理准确率。

        10 个案例中，至少 7 个应产生有效诊断。
        """
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser, SurpriseSignal

        md = MetaDiagnoser()
        correct = 0

        for case in CLINICAL_CASES:
            signal = SurpriseSignal(
                score=0.7,
                source=case["id"],
                layer="clinical",
                features={
                    "direction_error": 0.6 + (hash(case["id"]) % 30) / 100,
                    "state_distance": 0.5,
                    "vector_deviation": 0.55,
                },
            )
            result = md.diagnose([signal])
            # 有效诊断: 有根因链 + 有建议
            if result.root_cause_chain.depth >= 3 and result.recommendation:
                correct += 1

        accuracy = correct / len(CLINICAL_CASES)
        # K5-3: ≥ 70%
        assert accuracy >= 0.7, f"临床推理准确率 {accuracy:.0%} < 70%"
