"""
MCI World Model — P1-P3 因果推理性能基准测试

覆盖:
- P1: 直接因果推理 (CausalInference.infer_relation)
- P2: 多跳因果推理 (CausalInference.multi_hop_inference 1-3 hop)
- P3: 推理链构建 (CausalInference.build_reasoning_chain)
- 辅助: CausalChain 大规模操作 (传播/冲突检测/老化)

运行: pytest benchmarks/test_causal_bench.py -m benchmark --benchmark-only
"""

from __future__ import annotations

import time

import numpy as np
import pytest

# =============================================================================
# 共享 fixtures
# =============================================================================

CATEGORIES = ["creative", "lake", "light", "thunder", "wind", "abyss", "mountain", "receptive"]
ENERGY_TYPES = ["wood", "fire", "earth", "metal", "water"]


@pytest.fixture(scope="module")
def causal_inference():
    """CausalInference 引擎实例。"""
    from mci_world_model._sys.causal import CausalInference

    return CausalInference()


@pytest.fixture(scope="module")
def large_memories():
    """200 条记忆集合（覆盖 8 卦 + 5 能量类型）。"""
    _rng = np.random.default_rng(42)
    memories = []
    for i in range(200):
        memories.append(
            {
                "id": f"mem_{i}",
                "category_name": CATEGORIES[i % 8],
                "energy_type": ENERGY_TYPES[i % 5],
            }
        )
    return memories


@pytest.fixture(scope="module")
def large_causal_graph():
    """500 节点因果图（5 层结构）。"""
    from mci_world_model._sys.causal import CausalChain

    cc = CausalChain()
    rng = np.random.default_rng(42)

    # 500 节点
    for i in range(500):
        cc.add(
            f"n{i}",
            category=CATEGORIES[i % 8],
            energy_type=ENERGY_TYPES[i % 5],
        )

    # 线性链 + 随机跳跃边
    for i in range(499):
        cc.link(f"n{i}", f"n{i + 1}")
    # 200 条随机跳跃边
    for _ in range(200):
        a = rng.integers(0, 400)
        b = rng.integers(a + 2, min(a + 50, 500))
        cc.link(f"n{a}", f"n{b}")

    # 时序关联
    for i in range(500):
        cc.link_temporal(f"n{i}", f"branch_{(i % 12) + 1}")

    return cc


# =============================================================================
# P1: 直接因果推理延迟
# =============================================================================


@pytest.mark.benchmark(min_rounds=50, max_time=2.0)
class TestP1DirectInference:
    """CausalInference.infer_relation() 单次推理延迟。"""

    def test_infer_same_category(self, benchmark, causal_inference):
        """同类别推理 (score=1.0)。"""
        result = benchmark(
            causal_inference.infer_relation,
            "creative",
            "metal",
            "creative",
            "metal",
        )
        assert result["relation"] == "same"

    def test_infer_generates(self, benchmark, causal_inference):
        """语义相生推理 (score=0.8)。"""
        result = benchmark(
            causal_inference.infer_relation,
            "creative",
            "metal",
            "light",
            "fire",
        )
        assert result["relation"] == "generates"

    def test_infer_contradicts(self, benchmark, causal_inference):
        """语义相克推理 (score=0.3)。"""
        result = benchmark(
            causal_inference.infer_relation,
            "creative",
            "metal",
            "wind",
            "wood",
        )
        assert result["relation"] == "contradicts"

    def test_infer_energy_enhance(self, benchmark, causal_inference):
        """能量相生推理 (score=0.7)。"""
        result = benchmark(
            causal_inference.infer_relation,
            "abyss",
            "wood",
            "mountain",
            "fire",
        )
        assert result["relation"] == "generates"

    def test_infer_neutral(self, benchmark, causal_inference):
        """无关系推理 (score=0.0)。"""
        result = benchmark(
            causal_inference.infer_relation,
            "creative",
            "wood",
            "abyss",
            "wood",
        )
        assert result["relation"] == "neutral"

    def test_infer_batch_64_combos(self, benchmark, causal_inference):
        """64 种组合批量推理（模拟检索场景）。"""

        def batch_infer():
            results = []
            for qc in CATEGORIES:
                for cc in CATEGORIES:
                    r = causal_inference.infer_relation(qc, "wood", cc, "fire")
                    results.append(r)
            return results

        results = benchmark(batch_infer)
        assert len(results) == 64


# =============================================================================
# P2: 多跳因果推理延迟
# =============================================================================


@pytest.mark.benchmark(min_rounds=5, max_time=5.0)
class TestP2MultiHopInference:
    """CausalInference.multi_hop_inference() 多跳推理延迟。"""

    def test_1_hop_200_memories(self, benchmark, causal_inference, large_memories):
        """1 跳推理 (200 条记忆)。"""
        results = benchmark(
            causal_inference.multi_hop_inference,
            "creative",
            "metal",
            large_memories,
            1,
        )
        assert len(results) == 200

    def test_2_hop_200_memories(self, benchmark, causal_inference, large_memories):
        """2 跳推理 (200 条记忆)。"""
        results = benchmark(
            causal_inference.multi_hop_inference,
            "creative",
            "metal",
            large_memories,
            2,
        )
        assert len(results) == 200

    def test_3_hop_200_memories(self, benchmark, causal_inference, large_memories):
        """3 跳推理 (200 条记忆)。"""
        results = benchmark(
            causal_inference.multi_hop_inference,
            "creative",
            "metal",
            large_memories,
            3,
        )
        assert len(results) == 200

    def test_3_hop_50_memories(self, benchmark, causal_inference):
        """3 跳推理 (50 条记忆，常见规模)。"""
        _rng = np.random.default_rng(99)
        memories = [
            {
                "id": f"m{i}",
                "category_name": CATEGORIES[i % 8],
                "energy_type": ENERGY_TYPES[i % 5],
            }
            for i in range(50)
        ]
        results = benchmark(
            causal_inference.multi_hop_inference,
            "creative",
            "metal",
            memories,
            3,
        )
        assert len(results) == 50


# =============================================================================
# P3: 推理链构建延迟
# =============================================================================


@pytest.mark.benchmark(min_rounds=3, max_time=5.0)
class TestP3ReasoningChain:
    """CausalInference.build_reasoning_chain() 推理链构建延迟。"""

    def test_chain_20_memories(self, benchmark, causal_inference):
        """20 条记忆推理链。"""
        memories = [
            {
                "id": f"m{i}",
                "category_name": CATEGORIES[i % 8],
                "energy_type": ENERGY_TYPES[i % 5],
            }
            for i in range(20)
        ]
        result = benchmark(causal_inference.build_reasoning_chain, memories)
        assert result["coverage"] >= 0

    def test_chain_50_memories(self, benchmark, causal_inference):
        """50 条记忆推理链。"""
        memories = [
            {
                "id": f"m{i}",
                "category_name": CATEGORIES[i % 8],
                "energy_type": ENERGY_TYPES[i % 5],
            }
            for i in range(50)
        ]
        result = benchmark(causal_inference.build_reasoning_chain, memories)
        assert result["coverage"] >= 0

    def test_chain_100_memories(self, benchmark, causal_inference):
        """100 条记忆推理链。"""
        memories = [
            {
                "id": f"m{i}",
                "category_name": CATEGORIES[i % 8],
                "energy_type": ENERGY_TYPES[i % 5],
            }
            for i in range(100)
        ]
        result = benchmark(causal_inference.build_reasoning_chain, memories)
        assert result["coverage"] >= 0


# =============================================================================
# 辅助: CausalChain 大规模操作
# =============================================================================


@pytest.mark.benchmark(min_rounds=5, max_time=3.0)
class TestCausalChainLargeScale:
    """500 节点因果图大规模操作延迟。"""

    def test_propagate_500_nodes(self, benchmark, large_causal_graph):
        """500 节点能量传播。"""
        result = benchmark(large_causal_graph.propagate, "n0", delta=0.1)
        assert len(result) > 0

    def test_coverage_500_nodes(self, benchmark, large_causal_graph):
        """500 节点覆盖率计算。"""
        all_ids = [f"n{i}" for i in range(500)]
        cov = benchmark(large_causal_graph.coverage, all_ids)
        assert 0 <= cov <= 100

    def test_detect_conflicts_50_beliefs(self, benchmark, large_causal_graph):
        """50 条信念冲突检测。"""
        beliefs = [
            {
                "id": f"n{i}",
                "content": f"belief_{i}",
                "energy_type": ENERGY_TYPES[i % 5],
                "category": CATEGORIES[i % 8],
            }
            for i in range(50)
        ]
        conflicts = benchmark(large_causal_graph.detect_conflicts, beliefs)
        assert isinstance(conflicts, list)

    def test_causal_path_long(self, benchmark, large_causal_graph):
        """BFS 长路径查询 (n0 → n200)。"""
        path = benchmark(large_causal_graph.get_causal_path, "n0", "n200")
        assert len(path) > 0

    def test_aging_500_memories(self, benchmark):
        """500 条记忆老化检测。"""
        from mci_world_model._sys.causal import CausalChain

        cc = CausalChain()
        now = time.time()
        memories = [
            {"id": f"m{i}", "timestamp": now - (i * 3600)}  # 每小时递增
            for i in range(500)
        ]
        result = benchmark(cc.get_aging, memories)
        assert isinstance(result, list)
