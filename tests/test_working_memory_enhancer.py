"""MCI World Model v5.1.0 — 工作记忆增强器 测试

P1-A 增强: WorkingMemory + 遗忘曲线 + 注意力检索 + 记忆整合。
"""

from __future__ import annotations

import math
import time

import numpy as np

from mci_world_model.sdk._working_memory_enhancer import (
    MemoryRetrievalResult,
    WorkingMemoryEnhancer,
    WorkingMemoryEnhancerConfig,
)
from mci_world_model.sdk._world_model import TrajectoryStep, WorkingMemory
from mci_world_model.sdk._world_state import PendulumState

# ═══════════════════════════════════════════════════════════════════════════
# Test 工作记忆增强器存在性
# ═══════════════════════════════════════════════════════════════════════════


class TestWorkingMemoryEnhancerExistence:
    """增强器可实例化和导入。"""

    def test_enhancer_exists(self):
        wm = WorkingMemory()
        enhancer = WorkingMemoryEnhancer(wm)
        assert enhancer is not None

    def test_config_exists(self):
        config = WorkingMemoryEnhancerConfig()
        assert config.decay_rate > 0
        assert 0 < config.consolidation_threshold <= 1.0

    def test_importable_from_sdk(self):
        from mci_world_model.sdk import (
            MemoryRetrievalResult,
            WorkingMemoryEnhancer,
            WorkingMemoryEnhancerConfig,
        )

        assert WorkingMemoryEnhancer is not None
        assert WorkingMemoryEnhancerConfig is not None
        assert MemoryRetrievalResult is not None


# ═══════════════════════════════════════════════════════════════════════════
# Test 遗忘曲线
# ═══════════════════════════════════════════════════════════════════════════


class TestForgettingCurve:
    """Ebbinghaus 遗忘模型 R(t) = e^(-decay * t)。"""

    def setup_method(self):
        self.wm = WorkingMemory(max_length=20)
        self.config = WorkingMemoryEnhancerConfig(decay_rate=1.0)
        self.enhancer = WorkingMemoryEnhancer(self.wm, config=self.config)

    def test_retention_at_push_time(self):
        """刚 push 的步骤保留率 = 1.0。"""
        step = TrajectoryStep(
            state=PendulumState(theta=0.1, omega=0.0),
        )
        now = time.time()
        self.enhancer.push_enhanced(step, timestamp=now)
        retention = self.enhancer.compute_retention(0, current_time=now)
        assert abs(retention - 1.0) < 0.01

    def test_retention_decays_over_time(self):
        """保留率随时间递减。"""
        step = TrajectoryStep(
            state=PendulumState(theta=0.1, omega=0.0),
        )
        now = time.time()
        self.enhancer.push_enhanced(step, timestamp=now)

        r0 = self.enhancer.compute_retention(0, current_time=now)
        r1 = self.enhancer.compute_retention(0, current_time=now + 1.0)
        r5 = self.enhancer.compute_retention(0, current_time=now + 5.0)

        assert r0 > r1 > r5
        # decay_rate=1.0, t=1.0 → R = e^(-1) ≈ 0.368
        assert abs(r1 - math.exp(-1.0)) < 0.01

    def test_all_retentions(self):
        """compute_all_retentions 返回正确长度。"""
        for i in range(5):
            step = TrajectoryStep(
                state=PendulumState(theta=float(i) * 0.1, omega=0.0),
            )
            self.enhancer.push_enhanced(step, timestamp=time.time())

        retentions = self.enhancer.compute_all_retentions()
        assert len(retentions) == 5
        assert all(0 <= r <= 1.0 for r in retentions)


# ═══════════════════════════════════════════════════════════════════════════
# Test 注意力检索
# ═══════════════════════════════════════════════════════════════════════════


class TestAttentionRetrieve:
    """基于 cosine similarity 的注意力检索。"""

    def setup_method(self):
        self.wm = WorkingMemory(max_length=20)
        self.enhancer = WorkingMemoryEnhancer(self.wm)

    def test_empty_memory_returns_empty(self):
        """空记忆返回空结果。"""
        query = np.array([0.0, 0.0])
        results = self.enhancer.attention_retrieve(query)
        assert results == []

    def test_retrieve_returns_results(self):
        """检索返回 MemoryRetrievalResult。"""
        for i in range(5):
            step = TrajectoryStep(
                state=PendulumState(theta=float(i) * 0.5, omega=0.0),
            )
            self.enhancer.push_enhanced(step)

        query = np.array([0.0, 0.0])  # theta=0
        results = self.enhancer.attention_retrieve(query, top_k=3)
        assert len(results) <= 3
        assert all(isinstance(r, MemoryRetrievalResult) for r in results)

    def test_retrieve_sorted_by_effective_weight(self):
        """结果按 effective_weight 降序排列。"""
        for i in range(5):
            step = TrajectoryStep(
                state=PendulumState(theta=float(i) * 0.5, omega=0.0),
            )
            self.enhancer.push_enhanced(step)

        query = np.array([0.1, 0.0])
        results = self.enhancer.attention_retrieve(query)
        weights = [r.effective_weight for r in results]
        assert weights == sorted(weights, reverse=True)

    def test_query_count_increments(self):
        """每次检索递增 query_count。"""
        step = TrajectoryStep(
            state=PendulumState(theta=0.1, omega=0.0),
        )
        self.enhancer.push_enhanced(step)

        assert self.enhancer.query_count == 0
        self.enhancer.attention_retrieve(np.array([0.0, 0.0]))
        assert self.enhancer.query_count == 1


# ═══════════════════════════════════════════════════════════════════════════
# Test 记忆整合
# ═══════════════════════════════════════════════════════════════════════════


class TestConsolidation:
    """相似轨迹步骤自动合并。"""

    def setup_method(self):
        self.wm = WorkingMemory(max_length=20)
        self.config = WorkingMemoryEnhancerConfig(consolidation_threshold=0.9)
        self.enhancer = WorkingMemoryEnhancer(self.wm, config=self.config)

    def test_consolidate_similar_steps(self):
        """高度相似的连续步骤被合并。"""
        for _ in range(3):
            step = TrajectoryStep(
                state=PendulumState(theta=0.100, omega=0.0),
            )
            self.enhancer.push_enhanced(step)

        initial_size = self.enhancer.memory_size
        merged = self.enhancer.consolidate()
        assert merged > 0
        assert self.enhancer.memory_size < initial_size

    def test_no_consolidate_dissimilar(self):
        """不相似的步骤不被合并。"""
        # 用近似正交的向量确保 cosine similarity < 0.9
        # [0, 1] vs [1, 0] → cos = 0
        steps_data = [
            PendulumState(theta=0.0, omega=1.0),
            PendulumState(theta=1.0, omega=0.0),
            PendulumState(theta=-1.0, omega=0.5),
        ]
        for state in steps_data:
            step = TrajectoryStep(state=state)
            self.enhancer.push_enhanced(step)

        merged = self.enhancer.consolidate()
        assert merged == 0

    def test_empty_memory_no_merge(self):
        """空记忆不合并。"""
        merged = self.enhancer.consolidate()
        assert merged == 0


# ═══════════════════════════════════════════════════════════════════════════
# Test 惊奇优先级
# ═══════════════════════════════════════════════════════════════════════════


class TestSurprisePriority:
    """惊奇驱动的优先返回。"""

    def setup_method(self):
        self.wm = WorkingMemory(max_length=20)
        self.enhancer = WorkingMemoryEnhancer(self.wm)

    def test_surprise_top_returns_highest(self):
        """惊奇度最高的步骤排最前。"""
        for i in range(5):
            step = TrajectoryStep(
                state=PendulumState(theta=float(i) * 0.1, omega=0.0),
            )
            surprise = float(i)  # 0, 1, 2, 3, 4
            self.enhancer.push_enhanced(step, surprise_score=surprise)

        results = self.enhancer.get_surprise_top(top_k=2)
        assert len(results) == 2
        # 最后加入的惊奇度最高 (surprise=4)
        assert results[0].relevance >= results[1].relevance

    def test_empty_memory_returns_empty(self):
        """空记忆返回空。"""
        results = self.enhancer.get_surprise_top()
        assert results == []


# ═══════════════════════════════════════════════════════════════════════════
# Test 属性和摘要
# ═══════════════════════════════════════════════════════════════════════════


class TestPropertiesAndSummary:
    """属性与摘要测试。"""

    def test_memory_size(self):
        wm = WorkingMemory(max_length=20)
        enhancer = WorkingMemoryEnhancer(wm)
        assert enhancer.memory_size == 0

        step = TrajectoryStep(
            state=PendulumState(theta=0.1, omega=0.0),
        )
        enhancer.push_enhanced(step)
        assert enhancer.memory_size == 1

    def test_retention_summary(self):
        wm = WorkingMemory()
        enhancer = WorkingMemoryEnhancer(wm)

        step = TrajectoryStep(
            state=PendulumState(theta=0.1, omega=0.0),
        )
        enhancer.push_enhanced(step)

        summary = enhancer.get_retention_summary()
        assert "memory_size" in summary
        assert "avg_retention" in summary
        assert "query_count" in summary
        assert summary["memory_size"] == 1

    def test_repr(self):
        wm = WorkingMemory()
        enhancer = WorkingMemoryEnhancer(wm)
        r = repr(enhancer)
        assert "WorkingMemoryEnhancer" in r
        assert "size=" in r
