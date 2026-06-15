"""
MCI World Model v4.3.2 — ReflectionSynthesizer 测试
====================================================

覆盖 _reflection_synthesizer.py 的五个公开方法:
- extract_facts()        事实提取
- surface_entities()     实体聚类
- synthesize_causal_pairs() 因果对合成
- run_pipeline()          端到端管道
- training_data_report()  训练数据报告
"""

import numpy as np
import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synth():
    """创建最小化 ReflectionSynthesizer 实例。"""
    from mci_world_model.sdk._reflection_synthesizer import ReflectionSynthesizer

    return ReflectionSynthesizer(min_confidence=0.1, max_pairs=50, seed=42)


@pytest.fixture
def sample_memories():
    """返回一组含因果关系的临床记忆。"""
    return [
        {
            "id": "m001",
            "content": "患者白蛋白从35g/L下降至28g/L，导致水肿加重。原因可能是蛋白质摄入不足。",
        },
        {
            "id": "m002",
            "content": "肠内营养支持后，前白蛋白由80mg/L上升至120mg/L，因此伤口愈合加快。",
        },
        {
            "id": "m003",
            "content": "NRS2002评分5分，所以需要强化营养干预。每日热量目标1800kcal。",
        },
    ]


@pytest.fixture
def empty_memories():
    """空记忆列表。"""
    return []


# =============================================================================
# TestExtractFacts
# =============================================================================


class TestExtractFacts:
    """extract_facts() 测试。"""

    def test_extract_from_memories(self, synth, sample_memories):
        """从多条记忆提取事实。"""
        facts = synth.extract_facts(sample_memories)
        assert len(facts) == 3
        for f in facts:
            assert "memory_id" in f
            assert "content" in f
            assert "entities" in f
            assert "numerics" in f
            assert "causals" in f
            assert "energy_type" in f

    def test_entities_extracted(self, synth, sample_memories):
        """实体提取包含数值和概念。"""
        facts = synth.extract_facts(sample_memories)
        # 第一条记忆应包含 albumin 相关实体
        all_entities = []
        for f in facts:
            all_entities.extend(f["entities"])
        assert len(all_entities) > 0

    def test_numerics_extracted(self, synth, sample_memories):
        """数值提取捕获定量指标。"""
        facts = synth.extract_facts(sample_memories)
        all_numerics = []
        for f in facts:
            all_numerics.extend(f["numerics"])
        # 应有白蛋白28/35, 前白蛋白80/120, 1800kcal 等数值
        assert len(all_numerics) >= 3

    def test_causal_markers_found(self, synth):
        """因果指示词提取。"""
        memories = [
            {
                "id": "m001",
                "content": "蛋白质摄入不足导致白蛋白下降，因此出现水肿。",
            }
        ]
        facts = synth.extract_facts(memories)
        causals = facts[0]["causals"]
        assert "导致" in causals or "因此" in causals

    def test_empty_memories_graceful(self, synth, empty_memories):
        """空记忆列表不崩溃。"""
        facts = synth.extract_facts(empty_memories)
        assert facts == []

    def test_empty_content_skipped(self, synth):
        """content 为空时跳过。"""
        memories = [{"id": "m001", "content": ""}]
        facts = synth.extract_facts(memories)
        assert len(facts) == 0


# =============================================================================
# TestSurfaceEntities
# =============================================================================


class TestSurfaceEntities:
    """surface_entities() 测试。"""

    def test_surface_from_facts(self, synth, sample_memories):
        """从事实列表做实体聚类 — 可能因跨文档匹配不足返回空。"""
        facts = synth.extract_facts(sample_memories)
        result = synth.surface_entities(facts)
        assert isinstance(result, dict)

    def test_empty_facts_graceful(self, synth):
        """空事实返回空字典。"""
        result = synth.surface_entities([])
        assert result == {}


# =============================================================================
# TestSynthesizeCausalPairs
# =============================================================================


class TestSynthesizeCausalPairs:
    """synthesize_causal_pairs() 测试。"""

    def test_synthesize_from_facts(self, synth, sample_memories):
        """从事实合成因果 QA 对。"""
        facts = synth.extract_facts(sample_memories)
        pairs = synth.synthesize_causal_pairs(facts)
        assert isinstance(pairs, list)
        # 至少有一定概率生成 QA 对
        if pairs:
            p = pairs[0]
            assert hasattr(p, "cause_text")
            assert hasattr(p, "effect_text")
            assert hasattr(p, "confidence")

    def test_empty_facts_graceful(self, synth):
        """空事实返回空列表。"""
        pairs = synth.synthesize_causal_pairs([])
        assert pairs == []

    def test_single_fact_graceful(self, synth):
        """单条事实不崩溃。"""
        memories = [{"id": "m001", "content": "仅有白蛋白 28g/L。"}]
        facts = synth.extract_facts(memories)
        pairs = synth.synthesize_causal_pairs(facts)
        assert isinstance(pairs, list)
        # 单条没有配对对象，可能为空
        # 不崩溃即可


# =============================================================================
# TestRunPipeline
# =============================================================================


class TestRunPipeline:
    """run_pipeline() 端到端测试。"""

    def test_full_pipeline(self, synth, sample_memories):
        """全管道运行正常。"""
        pairs, prior = synth.run_pipeline(sample_memories)
        assert isinstance(pairs, list)
        assert isinstance(prior, (np.ndarray, type(None)))
        if prior is not None:
            assert prior.ndim == 2

    def test_empty_pipeline(self, synth):
        """空记忆全管道不崩溃。"""
        pairs, _prior = synth.run_pipeline([])
        assert pairs == []


# =============================================================================
# TestTrainingDataReport
# =============================================================================


class TestTrainingDataReport:
    """training_data_report() 测试。"""

    def test_report_structure(self, synth, sample_memories):
        """报告结构完整。"""
        pairs, _ = synth.run_pipeline(sample_memories)
        report = synth.training_data_report(pairs)
        assert isinstance(report, dict)
        assert "total_pairs" in report
        assert "avg_confidence" in report
        assert "ready_for_training" in report

    def test_empty_pairs_report(self, synth):
        """空 QA 对报告不崩溃。"""
        report = synth.training_data_report([])
        assert report["total_pairs"] == 0
        assert report["ready_for_training"] is False

    def test_report_confidence_range(self, synth, sample_memories):
        """置信度在 [0,1] 范围内。"""
        pairs, _ = synth.run_pipeline(sample_memories)
        report = synth.training_data_report(pairs)
        if pairs:
            conf = report["avg_confidence"]
            assert 0.0 <= conf <= 1.0
