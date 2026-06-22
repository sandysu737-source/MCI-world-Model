"""Tests for CounterfactualOracle — what-if scenario reasoning."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from mci_world_model.sdk._counterfactual_oracle import (
    CFRanking,
    CFScenario,
    CounterfactualOracle,
)


@pytest.fixture
def oracle():
    return CounterfactualOracle()


class TestCFScenario:
    def test_creation(self):
        s = CFScenario(name="Test", intervention={"X": 1.0}, target="Y")
        assert s.name == "Test"
        assert s.intervention == {"X": 1.0}
        assert s.target == "Y"

    def test_defaults(self):
        s = CFScenario(name="Default", intervention={}, target="outcome")
        assert s.description == ""


class TestCFRanking:
    def test_defaults(self):
        s = CFScenario(name="S", intervention={}, target="T")
        r = CFRanking(scenario=s)
        assert r.scenario is s
        assert r.rank == -1
        assert r.confidence == 1.0

    def test_uncertain_marker(self):
        s = CFScenario(name="U", intervention={}, target="T")
        r = CFRanking(scenario=s, is_uncertain=True)
        assert r.is_uncertain


class TestCounterfactualOracle:
    def test_initial_query_count(self, oracle):
        assert oracle.query_count == 0

    def test_batch_what_if_empty(self, oracle):
        results = oracle.batch_what_if([])
        assert results == []

    def test_batch_what_if_single(self, oracle):
        s = CFScenario(name="Test", intervention={"X": 1.0}, target="Y")
        results = oracle.batch_what_if([s])
        assert len(results) == 1
        assert isinstance(results[0], CFRanking)

    def test_batch_what_if_multiple(self, oracle):
        scenarios = [
            CFScenario(name=f"S{i}", intervention={"X": i}, target="Y")
            for i in range(5)
        ]
        results = oracle.batch_what_if(scenarios)
        assert len(results) == 5

    def test_rank_scenarios_higher_is_better(self, oracle):
        scenarios = [
            CFScenario(name="Low", intervention={"X": 0.1}, target="Y"),
            CFScenario(name="High", intervention={"X": 0.9}, target="Y"),
        ]
        rankings = oracle.rank_scenarios(scenarios, target_direction="higher_is_better")
        assert len(rankings) == 2
        assert rankings[0].rank == 0

    def test_rank_scenarios_lower_is_better(self, oracle):
        scenarios = [
            CFScenario(name="A", intervention={"X": 0.5}, target="Y"),
            CFScenario(name="B", intervention={"X": 0.1}, target="Y"),
        ]
        rankings = oracle.rank_scenarios(scenarios, target_direction="lower_is_better")
        assert len(rankings) == 2

    def test_query_returns_structure(self, oracle):
        result = oracle.query([
            {"name": "HypA", "intervention": {"X": 1.0}, "target": "Y"},
            {"name": "HypB", "intervention": {"X": 2.0}, "target": "Y"},
        ])
        assert "best_scenario" in result
        assert "rankings" in result
        assert "recommendation" in result
        assert "n_scenarios" in result
        assert result["n_scenarios"] == 2

    def test_query_empty_hypotheses(self, oracle):
        result = oracle.query([])
        assert result["n_scenarios"] == 0
        assert result["best_scenario"] is None

    def test_query_single_hypothesis(self, oracle):
        result = oracle.query([
            {"name": "Single", "intervention": {"A": 0.5}, "target": "B"},
        ])
        assert result["n_scenarios"] == 1
        assert result["best_scenario"] == "Single"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
