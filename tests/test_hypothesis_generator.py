"""tests/test_hypothesis_generator.py"""

from __future__ import annotations

import pytest

from mci_world_model.sdk._hypothesis_generator import HypothesisGenerator


@pytest.fixture
def gen():
    return HypothesisGenerator(max_hypotheses=20)


class TestGenerate:
    def test_empty(self, gen):
        hypotheses = gen.generate()
        assert hypotheses == []

    def test_transitive(self, gen):
        gen.add_known_cause("X", "Y", strength=0.8)
        gen.add_known_cause("Y", "Z", strength=0.7)
        hypotheses = gen.generate()
        transitive = [h for h in hypotheses if h.source == "transitive"]
        assert len(transitive) >= 1
        assert transitive[0].cause == "X"
        assert transitive[0].effect == "Z"

    def test_unconnected(self, gen):
        gen.add_known_cause("A", "B", strength=0.9)
        gen.add_known_cause("C", "D", strength=0.8)
        hypotheses = gen.generate()
        unconnected = [h for h in hypotheses if h.source == "unconnected"]
        assert len(unconnected) >= 1

    def test_reverse(self, gen):
        gen.add_known_cause("X", "Y", strength=0.9)
        hypotheses = gen.generate()
        reverse = [h for h in hypotheses if h.source == "reverse"]
        assert len(reverse) >= 1
        assert reverse[0].cause == "Y"
        assert reverse[0].effect == "X"

    def test_dedup(self, gen):
        gen.add_known_cause("A", "B")
        gen.add_known_cause("A", "B")
        hypotheses = gen.generate()
        # 不应有重复 (cause, effect) 对
        keys = [(h.cause, h.effect) for h in hypotheses]
        assert len(keys) == len(set(keys))

    def test_max_hypotheses(self):
        gen = HypothesisGenerator(max_hypotheses=3)
        for i in range(10):
            gen.add_known_cause(f"V{i}", f"V{i + 1}")
        hypotheses = gen.generate()
        assert len(hypotheses) <= 3


class TestRank:
    def test_rank(self, gen):
        gen.add_known_cause("X", "Y", strength=0.8)
        gen.add_known_cause("Y", "Z", strength=0.7)
        gen.generate()
        ranked = gen.rank_hypotheses(weight_plausibility=0.5, weight_testability=0.3, weight_novelty=0.2)
        assert len(ranked) == len(gen._hypotheses)


class TestStatistics:
    def test_stats(self, gen):
        gen.add_known_cause("X", "Y")
        gen.generate()
        stats = gen.statistics()
        assert stats["known_causes"] == 1
        assert stats["variables"] == 2
