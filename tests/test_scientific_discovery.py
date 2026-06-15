"""tests/test_scientific_discovery.py"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._scientific_discovery import DiscoveryReport, DiscoveryStage, ScientificDiscoveryPipeline


@pytest.fixture
def pipeline():
    return ScientificDiscoveryPipeline(pc_alpha=0.05, conservation_threshold=0.5)


@pytest.fixture
def chain_data():
    rng = np.random.RandomState(42)
    n = 300
    x1 = rng.randn(n)
    x2 = 0.8 * x1 + 0.1 * rng.randn(n)
    x3 = 0.6 * x2 + 0.1 * rng.randn(n)
    return np.column_stack([x1, x2, x3]), ["X1", "X2", "X3"]


class TestLoadData:
    def test_load(self, pipeline, chain_data):
        data, names = chain_data
        pipeline.load_data(data, names)
        assert pipeline.current_stage == DiscoveryStage.EXPLORATION


class TestRun:
    def test_full_pipeline(self, pipeline, chain_data):
        data, names = chain_data
        pipeline.load_data(data, names)
        report = pipeline.run()
        assert isinstance(report, DiscoveryReport)
        assert report.stage == DiscoveryStage.COMPLETED
        assert report.n_variables == 3

    def test_no_data(self, pipeline):
        report = pipeline.run()
        assert "error" in report.details

    def test_discovers_laws(self, pipeline, chain_data):
        data, names = chain_data
        pipeline.load_data(data, names)
        report = pipeline.run()
        assert report.n_laws >= 1


class TestDiscoveredLaws:
    def test_laws_property(self, pipeline, chain_data):
        data, names = chain_data
        pipeline.load_data(data, names)
        pipeline.run()
        laws = pipeline.discovered_laws
        assert isinstance(laws, list)


class TestStatistics:
    def test_stats(self, pipeline, chain_data):
        data, names = chain_data
        pipeline.load_data(data, names)
        pipeline.run()
        stats = pipeline.statistics()
        assert stats["n_variables"] == 3
