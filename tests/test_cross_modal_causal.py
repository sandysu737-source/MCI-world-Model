"""tests/test_cross_modal_causal.py"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._cross_modal_causal import (
    CrossModalCausalReasoner,
    CrossModalCausalResult,
)


@pytest.fixture
def reasoner():
    return CrossModalCausalReasoner(min_strength=0.1)


class TestAddObservation:
    def test_add(self, reasoner):
        reasoner.add_observation("vision", np.random.randn(10), "red_light", timestamp=1.0)
        assert reasoner.observation_count == 1


class TestReason:
    def test_no_observations(self, reasoner):
        result = reasoner.reason("vision:red_light", "audio")
        assert result.is_reliable is False

    def test_with_observations(self, reasoner):
        rng = np.random.RandomState(42)
        reasoner.add_observation("vision", rng.randn(10), "red_light", timestamp=1.0)
        reasoner.add_observation("audio", rng.randn(10), "alarm", timestamp=2.0)
        result = reasoner.reason("vision:red_light", "audio")
        assert isinstance(result, CrossModalCausalResult)

    def test_discovered_links(self, reasoner):
        rng = np.random.RandomState(42)
        x = rng.randn(10)
        reasoner.add_observation("vision", x, "red_light", timestamp=1.0)
        reasoner.add_observation("audio", x * 0.8 + rng.randn(10) * 0.1, "alarm", timestamp=2.0)
        reasoner.reason("vision:red_light", "audio")
        links = reasoner.get_discovered_links()
        # 高相关性应有链
        assert len(links) >= 1


class TestStatistics:
    def test_stats(self, reasoner):
        reasoner.add_observation("vision", np.random.randn(10), "test")
        stats = reasoner.statistics()
        assert stats["observation_count"] == 1
