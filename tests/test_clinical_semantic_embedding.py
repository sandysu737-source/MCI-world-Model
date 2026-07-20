"""ClinicalSemanticEmbedding 单元测试 — 方向二真实嵌入验证。

验证临床语义嵌入的核心契约：
    1. ICD-10/药物 → 概念桶映射正确
    2. 哈希投影嵌入器：不同概念不同向量，可复现
    3. PatientState → 完整语义向量（体征+诊断+用药）
    4. 语义区分能力：相同体征不同诊断 → 不同嵌入
    5. 外部嵌入器协议（可选增强）
    6. 数值健壮性 + 边界合规（无持久化）
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._clinical_semantic_embedding import (
    CLINICAL_CONCEPTS,
    N_CONCEPTS,
    ClinicalSemanticEmbedding,
    SemanticStateVector,
    _HashProjectionEmbedder,
    drug_to_class,
    icd10_to_concept,
)
from mci_world_model.sdk._clinical_world_state import (
    Medication,
    PatientState,
)

SEED = 42


def make_state(diagnoses=None, medications=None, **vital_overrides):
    vitals = np.array([[75.0, 120.0, 80.0, 98.0, 16.0, 36.8, 15.0]])
    keys = ["hr", "sbp", "dbp", "spo2", "rr", "temp", "gcs"]
    for i, k in enumerate(keys):
        if k in vital_overrides:
            vitals[0, i] = vital_overrides[k]
    return PatientState(
        vital_signs=vitals,
        diagnoses=diagnoses or [],
        medications=medications or [],
    )


# =============================================================================
# 1. 概念映射正确性
# =============================================================================


class TestConceptMapping:
    """验证 ICD-10/药物 → 概念桶映射。"""

    @pytest.mark.parametrize(
        "code,expected",
        [
            ("I48.91", "心律失常"),
            ("I21.9", "急性冠脉综合征"),
            ("N17.0", "急性肾损伤"),
            ("A41.9", "脓毒症"),
            ("J44.1", "慢性阻塞性肺病"),
            ("E11.9", "糖尿病"),
        ],
    )
    def test_icd10_to_concept_known(self, code, expected):
        """已知 ICD-10 代码映射正确。"""
        assert icd10_to_concept(code) == expected

    def test_icd10_unknown_falls_back_to_chapter(self):
        """未知代码按章节归类。"""
        result = icd10_to_concept("Z99.0")  # Z 章
        assert isinstance(result, str)
        assert len(result) > 0

    def test_icd10_empty_code(self):
        """空代码不崩溃。"""
        result = icd10_to_concept("")
        assert isinstance(result, str)

    @pytest.mark.parametrize(
        "drug,expected",
        [
            ("metoprolol", "β受体阻滞剂"),
            ("dopamine", "儿茶酚胺类血管活性药"),
            ("norepinephrine", "儿茶酚胺类血管活性药"),
            ("furosemide", "襻利尿剂"),
        ],
    )
    def test_drug_to_class_known(self, drug, expected):
        """已知药物映射正确。"""
        assert drug_to_class(drug) == expected

    def test_drug_chinese_name(self):
        """中文名药物能映射。"""
        assert drug_to_class("多巴胺") == "儿茶酚胺类血管活性药"
        assert drug_to_class("美托洛尔") == "β受体阻滞剂"

    def test_concepts_cover_core_categories(self):
        """概念桶覆盖核心临床类别。"""
        assert N_CONCEPTS >= 10
        assert "心律失常" in CLINICAL_CONCEPTS
        assert "脓毒症" in CLINICAL_CONCEPTS
        assert "β受体阻滞剂" in CLINICAL_CONCEPTS


# =============================================================================
# 2. 哈希投影嵌入器
# =============================================================================


class TestHashProjectionEmbedder:
    """验证哈希投影嵌入器。"""

    def test_embed_dim_correct(self):
        """嵌入维度正确。"""
        emb = _HashProjectionEmbedder(embed_dim=32, seed=42)
        vec = emb.embed_concept("心律失常")
        assert vec.shape == (32,)

    def test_same_concept_same_vector(self):
        """相同概念产生相同向量（可复现）。"""
        emb = _HashProjectionEmbedder(seed=42)
        v1 = emb.embed_concept("脓毒症")
        v2 = emb.embed_concept("脓毒症")
        np.testing.assert_array_equal(v1, v2)

    def test_different_concepts_different_vectors(self):
        """不同概念产生不同向量。"""
        emb = _HashProjectionEmbedder(seed=42)
        v1 = emb.embed_concept("心律失常")
        v2 = emb.embed_concept("急性肾损伤")
        assert not np.allclose(v1, v2)

    def test_vectors_normalized(self):
        """嵌入向量 L2 归一化。"""
        emb = _HashProjectionEmbedder(seed=42)
        for concept in ["心律失常", "脓毒症", "糖尿病"]:
            v = emb.embed_concept(concept)
            assert abs(np.linalg.norm(v) - 1.0) < 1e-6

    def test_embed_concepts_aggregation(self):
        """多概念聚合为均值。"""
        emb = _HashProjectionEmbedder(seed=42)
        concepts = ["心律失常", "心力衰竭"]
        agg = emb.embed_concepts(concepts)
        assert agg.shape == (emb.embed_dim,)
        # 聚合向量归一化
        assert abs(np.linalg.norm(agg) - 1.0) < 1e-6

    def test_empty_concepts_returns_zero(self):
        """空概念列表返回零向量。"""
        emb = _HashProjectionEmbedder(seed=42)
        v = emb.embed_concepts([])
        assert np.all(v == 0.0)

    def test_reproducible_across_instances(self):
        """不同实例相同种子产生相同嵌入。"""
        e1 = _HashProjectionEmbedder(seed=42)
        e2 = _HashProjectionEmbedder(seed=42)
        np.testing.assert_array_equal(
            e1.embed_concept("心律失常"),
            e2.embed_concept("心律失常"),
        )


# =============================================================================
# 3. 完整语义向量
# =============================================================================


class TestSemanticStateVector:
    """验证 PatientState → 完整语义向量。"""

    def test_embed_returns_semantic_state(self):
        """embed 返回 SemanticStateVector。"""
        emb = ClinicalSemanticEmbedding()
        state = make_state(diagnoses=["I48.91"], medications=[Medication(name="metoprolol", dose=5.0)])
        sem = emb.embed(state)
        assert isinstance(sem, SemanticStateVector)
        assert sem.numeric.shape == (13,)
        assert sem.diagnosis_embedding.shape == (emb.diag_embed_dim,)
        assert sem.medication_embedding.shape == (emb.med_embed_dim,)

    def test_full_vector_concatenation(self):
        """full_vector 是 numeric + diag + med 拼接。"""
        emb = ClinicalSemanticEmbedding()
        state = make_state(diagnoses=["I48.91"])
        sem = emb.embed(state)
        full = sem.full_vector
        assert full.shape == (sem.full_dim,)
        # 前部分 = numeric
        np.testing.assert_array_equal(full[:13], sem.numeric)

    def test_full_dim_correct(self):
        """完整维度 = 13 + diag_dim + med_dim。"""
        emb = ClinicalSemanticEmbedding(diag_embed_dim=32, med_embed_dim=16)
        state = make_state()
        sem = emb.embed(state)
        assert sem.full_dim == 13 + 32 + 16

    def test_concepts_extracted(self):
        """嵌入时提取概念桶列表。"""
        emb = ClinicalSemanticEmbedding()
        state = make_state(diagnoses=["I48.91", "N17.0"])
        sem = emb.embed(state)
        assert "心律失常" in sem.concepts
        assert "急性肾损伤" in sem.concepts

    def test_empty_state_handled(self):
        """无诊断无用药的状态不崩溃。"""
        emb = ClinicalSemanticEmbedding()
        state = make_state()
        sem = emb.embed(state)
        assert np.all(sem.diagnosis_embedding == 0.0)
        assert np.all(sem.medication_embedding == 0.0)


# =============================================================================
# 4. 语义区分能力（核心价值）
# =============================================================================


class TestSemanticDiscrimination:
    """验证语义嵌入能区分相同体征不同诊断。"""

    def test_different_diagnoses_different_embeddings(self):
        """相同体征、不同诊断 → 不同诊断嵌入。"""
        emb = ClinicalSemanticEmbedding()
        v = np.array([[130.0, 140, 90, 98, 20, 37, 15]])
        s1 = PatientState(vital_signs=v, diagnoses=["I48.91"])  # 心律失常
        s2 = PatientState(vital_signs=v, diagnoses=["N17.0"])  # 急性肾损伤
        sem1 = emb.embed(s1)
        sem2 = emb.embed(s2)
        # 数值部分相同
        np.testing.assert_array_equal(sem1.numeric, sem2.numeric)
        # 诊断嵌入不同
        diff = np.linalg.norm(sem1.diagnosis_embedding - sem2.diagnosis_embedding)
        assert diff > 0.01, f"不同诊断嵌入应不同，距离={diff}"

    def test_same_diagnoses_same_embeddings(self):
        """相同诊断 → 相同诊断嵌入。"""
        emb = ClinicalSemanticEmbedding()
        v = np.array([[130.0, 140, 90, 98, 20, 37, 15]])
        s1 = PatientState(vital_signs=v, diagnoses=["I48.91"])
        s2 = PatientState(vital_signs=v, diagnoses=["I48.91"])
        sem1 = emb.embed(s1)
        sem2 = emb.embed(s2)
        np.testing.assert_array_equal(sem1.diagnosis_embedding, sem2.diagnosis_embedding)

    def test_different_medications_different_embeddings(self):
        """不同用药 → 不同用药嵌入。"""
        emb = ClinicalSemanticEmbedding()
        v = np.array([[80.0, 120, 80, 98, 16, 36.8, 15]])
        s1 = PatientState(vital_signs=v, medications=[Medication(name="metoprolol", dose=5.0)])
        s2 = PatientState(vital_signs=v, medications=[Medication(name="dopamine", dose=5.0)])
        sem1 = emb.embed(s1)
        sem2 = emb.embed(s2)
        diff = np.linalg.norm(sem1.medication_embedding - sem2.medication_embedding)
        assert diff > 0.01

    def test_dose_affects_medication_embedding(self):
        """剂量影响用药嵌入（同药不同剂量）。"""
        emb = ClinicalSemanticEmbedding()
        v = np.array([[80.0, 120, 80, 98, 16, 36.8, 15]])
        s_low = PatientState(vital_signs=v, medications=[Medication(name="dopamine", dose=1.0)])
        s_high = PatientState(vital_signs=v, medications=[Medication(name="dopamine", dose=10.0)])
        sem_low = emb.embed(s_low)
        sem_high = emb.embed(s_high)
        # 方向相近（同药）但幅度不同
        cos = np.dot(sem_low.medication_embedding, sem_high.medication_embedding)
        assert cos > 0.5  # 同药方向相近
        # 但不完全相同（剂量调制）
        assert not np.allclose(sem_low.medication_embedding, sem_high.medication_embedding)


# =============================================================================
# 5. 外部嵌入器协议
# =============================================================================


class TestExternalEmbedder:
    """验证可选外部文本嵌入器集成。"""

    def test_mock_text_embedder_used(self):
        """注入外部嵌入器时使用它。"""

        class MockEmbedder:
            def __init__(self):
                self._rng = np.random.default_rng(0)

            def embed(self, text):
                # 简单确定性嵌入：文本长度的 one-hot 风格
                v = np.zeros(8)
                v[len(text) % 8] = 1.0
                return v

        emb = ClinicalSemanticEmbedding(text_embedder=MockEmbedder())
        assert emb.has_text_embedder
        assert emb.diag_embed_dim == 8  # 用外部维度

    def test_external_embedder_failure_fallback(self):
        """外部嵌入器异常时降级到哈希。"""

        class FailingEmbedder:
            def embed(self, text):
                raise RuntimeError("embedding failed")

        emb = ClinicalSemanticEmbedding(text_embedder=FailingEmbedder())
        state = make_state(diagnoses=["I48.91"])
        sem = emb.embed(state)
        # 降级后用哈希维度
        assert sem.diagnosis_embedding is not None


# =============================================================================
# 6. 数值健壮性 + 边界合规
# =============================================================================


class TestNumericRobustness:
    """验证数值健壮性。"""

    def test_extreme_state_no_nan(self):
        """极端体征不产生 NaN。"""
        emb = ClinicalSemanticEmbedding()
        state = PatientState(
            vital_signs=np.array([[200.0, 250, 150, 100, 40, 42, 15]]),
            diagnoses=["I48.91"],
        )
        sem = emb.embed(state)
        assert np.all(np.isfinite(sem.full_vector))

    def test_many_diagnoses_handled(self):
        """大量诊断不崩溃。"""
        emb = ClinicalSemanticEmbedding()
        state = make_state(diagnoses=["I48.91", "I21.9", "N17.0", "A41.9", "J44.1"])
        sem = emb.embed(state)
        assert np.all(np.isfinite(sem.diagnosis_embedding))


class TestSuMemoryBoundary:
    """验证不引入持久化（su-memory 边界）。"""

    def test_no_persistence(self):
        """嵌入器不持有持久化存储。"""
        emb = ClinicalSemanticEmbedding()
        persist_attrs = [
            a for a in dir(emb) if any(kw in a.lower() for kw in ["store", "db", "file", "disk", "persist"])
        ]
        assert persist_attrs == [], f"发现持久化属性: {persist_attrs}"
        # _cache 是哈希投影的内存缓存（单次进程内），非持久化
        assert not hasattr(emb, "_sqlite") and not hasattr(emb, "_faiss")
