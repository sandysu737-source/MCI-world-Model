"""tests/test_causal_gradient.py"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._causal_gradient import (
    CausalGradient,
    CausalGradientPropagation,
)


@pytest.fixture
def cgp():
    return CausalGradientPropagation(learning_rate=0.01)


class TestSetGraph:
    def test_set_and_check(self, cgp):
        adj = np.array([[0.0, 0.8], [0.0, 0.0]])
        cgp.set_graph(adj, node_names=["X", "Y"])
        stats = cgp.statistics()
        assert stats["n_nodes"] == 2


class TestPropagate:
    def test_basic_propagation(self, cgp):
        adj = np.array([[0.0, 0.8], [0.0, 0.0]])
        cgp.set_graph(adj, node_names=["X", "Y"])
        cgp.set_loss_gradient(np.array([1.0, 0.5]))
        gradients = cgp.propagate(n_steps=1)
        assert len(gradients) > 0
        assert all(isinstance(g, CausalGradient) for g in gradients)

    def test_no_graph_returns_empty(self, cgp):
        cgp.set_loss_gradient(np.array([1.0]))
        result = cgp.propagate()
        assert result == []

    def test_no_loss_grad_returns_empty(self, cgp):
        adj = np.array([[0.0, 0.8], [0.0, 0.0]])
        cgp.set_graph(adj, node_names=["X", "Y"])
        result = cgp.propagate()
        assert result == []

    def test_propagation_increments_count(self, cgp):
        adj = np.array([[0.0, 0.8], [0.0, 0.0]])
        cgp.set_graph(adj, node_names=["X", "Y"])
        cgp.set_loss_gradient(np.array([1.0, 0.5]))
        assert cgp.propagation_count == 0
        cgp.propagate()
        assert cgp.propagation_count == 1


class TestGetNodeGradients:
    def test_basic(self, cgp):
        adj = np.array([[0.0, 0.8], [0.0, 0.0]])
        cgp.set_graph(adj, node_names=["X", "Y"])
        cgp.set_loss_gradient(np.array([1.0, 0.5]))
        node_grads = cgp.get_node_gradients()
        assert "X" in node_grads
        assert "Y" in node_grads

    def test_no_graph_returns_empty(self, cgp):
        result = cgp.get_node_gradients()
        assert result == {}


class TestUpdateGraph:
    def test_basic_update(self, cgp):
        adj = np.array([[0.0, 0.8], [0.0, 0.0]])
        cgp.set_graph(adj, node_names=["X", "Y"])
        grads = [CausalGradient(source="X", target="Y", gradient=1.0, path=["X", "Y"])]
        updated = cgp.update_graph(grads)
        assert updated.shape == (2, 2)
        # adj[0,1] = 0.8 - 0.01*1.0 = 0.79
        assert updated[0, 1] == pytest.approx(0.79)

    def test_no_graph_returns_empty(self):
        cgp2 = CausalGradientPropagation()
        result = cgp2.update_graph([])
        assert result.shape == (0,)


class TestStatistics:
    def test_initial_stats(self, cgp):
        stats = cgp.statistics()
        assert stats["propagation_count"] == 0
        assert stats["n_nodes"] == 0
        assert stats["learning_rate"] == 0.01
