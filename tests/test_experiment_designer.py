"""tests/test_experiment_designer.py"""

from __future__ import annotations

import pytest

from mci_world_model.sdk._experiment_designer import (
    ExperimentDesigner,
    ExperimentPlan,
)


@pytest.fixture
def designer():
    return ExperimentDesigner(min_sample_size=30, target_power=0.8, alpha=0.05)


class TestDesign:
    def test_basic_design(self, designer):
        plan = designer.design(
            hypothesis_cause="X",
            hypothesis_effect="Y",
            prior_effect_size=0.5,
        )
        assert isinstance(plan, ExperimentPlan)
        assert plan.plan_id.startswith("EXP")
        assert plan.intervention["treatment_variable"] == "X"
        assert plan.control_group["measure_variable"] == "Y"
        assert plan.sample_size >= 30

    def test_large_effect_small_sample(self, designer):
        plan = designer.design("A", "B", prior_effect_size=1.0)
        # n ≈ 16/1² = 16, but min is 30
        assert plan.sample_size == 30

    def test_small_effect_large_sample(self, designer):
        plan = designer.design("A", "B", prior_effect_size=0.2)
        # n ≈ 16/0.04 = 400
        assert plan.sample_size >= 200

    def test_feasibility(self, designer):
        plan_good = designer.design("X", "Y", prior_effect_size=0.5)
        assert plan_good.is_feasible

    def test_not_feasible_tiny_effect(self, designer):
        plan_bad = designer.design("A", "B", prior_effect_size=0.05)
        assert not plan_bad.is_feasible

    def test_plan_counter_increments(self, designer):
        p1 = designer.design("X", "Y")
        p2 = designer.design("A", "B")
        assert p1.plan_id != p2.plan_id

    def test_hypothesis_id(self, designer):
        plan = designer.design("X", "Y", hypothesis_id="H001")
        assert plan.hypothesis_id == "H001"


class TestDesignBatch:
    def test_batch(self, designer):
        hypotheses = [
            {"cause": "X", "effect": "Y", "id": "H1", "effect_size": 0.5},
            {"cause": "A", "effect": "B", "id": "H2", "effect_size": 0.8},
        ]
        plans = designer.design_batch(hypotheses)
        assert len(plans) == 2
        assert plans[0].hypothesis_id == "H1"
        assert plans[1].hypothesis_id == "H2"


class TestGetFeasiblePlans:
    def test_filter(self, designer):
        designer.design("X", "Y", prior_effect_size=0.5)
        designer.design("A", "B", prior_effect_size=0.05)
        feasible = designer.get_feasible_plans()
        assert all(p.is_feasible for p in feasible)
        assert len(feasible) >= 1


class TestStatistics:
    def test_empty_stats(self):
        d = ExperimentDesigner()
        stats = d.statistics()
        assert stats["plan_count"] == 0
        assert stats["feasibility_rate"] == 0.0

    def test_stats_after_design(self, designer):
        designer.design("X", "Y", prior_effect_size=0.5)
        stats = designer.statistics()
        assert stats["plan_count"] == 1
        assert stats["avg_sample_size"] > 0


class TestValidation:
    def test_min_sample_size_too_small(self):
        with pytest.raises(ValueError, match="min_sample_size"):
            ExperimentDesigner(min_sample_size=5)
