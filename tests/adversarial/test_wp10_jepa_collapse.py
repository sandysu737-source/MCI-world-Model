"""WP-10 JEPA 潜向量契约与抗坍塌对抗测试。"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mci_world_model.sdk._jepa_encoder import JEPAEncoder
from mci_world_model.sdk._latent_state import LatentState
from mci_world_model.sdk._true_jepa_encoder import TrueJEPAConfig, TrueJEPAEncoder

pytestmark = pytest.mark.contract


@pytest.fixture
def encoder() -> JEPAEncoder:
    return JEPAEncoder(world_model=None)


def test_encode_defaults_to_not_ready_latent_contract(encoder: JEPAEncoder) -> None:
    result = encoder.encode([{"embedding": [1.0, 2.0, 3.0, 4.0]}])

    assert isinstance(result, LatentState)
    assert result.source == "not_ready"
    assert result.status == "not_ready"
    assert result.latent.shape == (0,)


def test_attached_true_jepa_returns_trained_latent(encoder: JEPAEncoder) -> None:
    rng = np.random.RandomState(42)
    observations = rng.randn(64, 4)
    true_jepa = TrueJEPAEncoder(TrueJEPAConfig(obs_dim=4, latent_dim=8, hidden_dim=8, action_dim=0, seed=42))
    training = true_jepa.train(observations, n_epochs=3)
    encoder.attach_true_jepa(true_jepa)

    result = encoder.encode(observations)

    assert isinstance(result, LatentState)
    assert result.source == "true_jepa"
    assert result.status == "ok"
    assert result.is_trained is True
    assert result.latent.shape == (64, 8)
    assert training["is_successful"] is True
    assert training["latent_variance_min"] >= true_jepa.config.variance_floor
    assert training["effective_rank"] >= true_jepa.config.min_effective_rank


def test_constant_input_training_is_rejected() -> None:
    true_jepa = TrueJEPAEncoder(TrueJEPAConfig(obs_dim=4, latent_dim=8, hidden_dim=8, action_dim=0, seed=42))
    result = true_jepa.train(np.ones((16, 4)), n_epochs=2)

    assert result["status"] == "rejected"
    assert result["is_successful"] is False
    assert result["collapse_detected"] is True
    assert result["latent_variance_max"] == 0.0


def test_legacy_graph_mode_is_explicit_transition_only() -> None:
    class GraphModel:
        jepa_mode = "legacy_graph"

        def discover(self, memories, use_parametric=False):
            from mci_world_model.sdk._world_model import CausalWorldModelState

            return CausalWorldModelState(causal_edges=[{"cause": "a", "effect": "b", "rho": 0.8}])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        encoder = JEPAEncoder(GraphModel(), jepa_mode="legacy_graph")

    assert any(issubclass(item.category, DeprecationWarning) for item in caught)
    result = encoder.encode([{"embedding": [1.0, 2.0, 3.0, 4.0]}])
    assert result.source == "legacy_graph"
    assert result.graph_state is not None
    assert result.graph_state.causal_edges
