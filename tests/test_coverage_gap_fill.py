"""
Coverage Gap Fill Tests — Push from 44.78% to 45%+
====================================================

Target modules with highest uncovered-statement ROI:
- _energy_flow_predictor.py (36% → target 75%+, +~12 stmts)
- _sigreg.py (15% → target 35%+, +~18 stmts)
- _hierarchical_encoder.py (24% → target 35%+, +~10 stmts)
- _cost_module.py exception path (96% → 100%, +2 stmts)

Estimated net coverage gain: ~42 covered statements → 45%+ gate
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

# =============================================================================
# EnergyFlowPredictor Tests
# =============================================================================


class TestEnergyFlowPredictor:
    """Cover _energy_flow_predictor.py: 28 stmts, 18 missed → target 75%+"""

    @pytest.fixture
    def mock_energy_core(self) -> MagicMock:
        core = MagicMock()
        core.simulate_energy_flow.return_value = [
            {"fire": 0.25, "earth": 0.25, "metal": 0.20, "water": 0.15, "wood": 0.15},
            {"fire": 0.22, "earth": 0.28, "metal": 0.18, "water": 0.17, "wood": 0.15},
            {"fire": 0.20, "earth": 0.30, "metal": 0.16, "water": 0.18, "wood": 0.16},
        ]
        return core

    @pytest.fixture
    def predictor(self, mock_energy_core: MagicMock):
        from mci_world_model.sdk._energy_flow_predictor import EnergyFlowPredictor

        return EnergyFlowPredictor(mock_energy_core)

    def test_predict_returns_flow_and_updates_history(self, predictor, mock_energy_core):
        """predict() calls simulate_energy_flow and stores history."""
        energy = {"fire": 0.3, "earth": 0.2, "metal": 0.2, "water": 0.15, "wood": 0.15}
        flow = predictor.predict(energy, steps=3)

        mock_energy_core.simulate_energy_flow.assert_called_once_with(energy, 3)
        assert len(flow) == 3
        assert flow[0]["fire"] == 0.25
        assert predictor._history == flow

    def test_validate_returns_mae(self, predictor):
        """validate() computes mean absolute deviation."""
        predicted = {"fire": 0.30, "earth": 0.25}
        actual = {"fire": 0.25, "earth": 0.30}
        mae = predictor.validate(predicted, actual)
        assert mae == pytest.approx(0.05, abs=0.001)

    def test_validate_missing_key_uses_zero(self, predictor):
        """validate() treats missing keys as 0."""
        predicted = {"fire": 0.10, "extra": 0.15}
        actual = {"fire": 0.20}
        mae = predictor.validate(predicted, actual)
        assert mae == pytest.approx(0.125, abs=0.001)

    def test_detect_anomaly_short_history_returns_false(self, predictor):
        """detect_anomaly() with <2 steps returns False."""
        flow = [{"fire": 0.5}]
        assert predictor.detect_anomaly(flow) is False

    def test_detect_anomaly_no_spike_returns_false(self, predictor):
        """detect_anomaly() with gradual change returns False."""
        flow = [
            {"fire": 0.30, "earth": 0.25},
            {"fire": 0.32, "earth": 0.24},
            {"fire": 0.34, "earth": 0.23},
        ]
        assert predictor.detect_anomaly(flow, threshold=0.15) is False

    def test_detect_anomaly_large_spike_returns_true(self, predictor):
        """detect_anomaly() with large single-step change returns True."""
        flow = [
            {"fire": 0.30, "earth": 0.25},
            {"fire": 0.60, "earth": 0.25},  # fire +0.30 > 0.15 threshold
        ]
        assert predictor.detect_anomaly(flow, threshold=0.15) is True

    def test_detect_anomaly_empty_history_returns_false(self, predictor):
        """detect_anomaly() with empty history returns False."""
        assert predictor.detect_anomaly([], threshold=0.10) is False

    def test_history_property_returns_latest(self, predictor, mock_energy_core):
        """history property returns the stored history."""
        flow_data = [{"fire": 0.40}, {"fire": 0.38}]
        mock_energy_core.simulate_energy_flow.return_value = flow_data
        predictor.predict({"fire": 0.40}, steps=2)
        assert predictor.history == flow_data


# =============================================================================
# SIGReg Tests
# =============================================================================


class TestSIGRegInit:
    """Cover SIGReg.__init__ validation paths (lines 50-56)."""

    def test_init_valid_params(self):
        from mci_world_model.sdk._sigreg import SIGReg

        sigreg = SIGReg(lambda_reg=0.05, use_sketch=True, sketch_dim=32)
        assert sigreg.lambda_reg == 0.05
        assert sigreg.use_sketch is True
        assert sigreg.sketch_dim == 32

    def test_init_defaults(self):
        from mci_world_model.sdk._sigreg import SIGReg

        sigreg = SIGReg()
        assert sigreg.lambda_reg == 0.01
        assert sigreg.use_sketch is True
        assert sigreg.sketch_dim == 64

    def test_init_rejects_lambda_out_of_range(self):
        from mci_world_model.sdk._sigreg import SIGReg

        with pytest.raises(ValueError, match="lambda_reg"):
            SIGReg(lambda_reg=1.5)

    def test_init_rejects_sketch_dim_zero(self):
        from mci_world_model.sdk._sigreg import SIGReg

        with pytest.raises(ValueError, match="sketch_dim"):
            SIGReg(sketch_dim=0)

    def test_init_accepts_lambda_zero(self):
        from mci_world_model.sdk._sigreg import SIGReg

        sigreg = SIGReg(lambda_reg=0.0)
        assert sigreg.lambda_reg == 0.0


class TestSIGRegRegularize:
    """Cover SIGReg.regularize() — core method (lines 70-126)."""

    def test_call_delegates_to_regularize(self):
        from mci_world_model.sdk._sigreg import SIGReg

        sigreg = SIGReg(lambda_reg=0.1)
        embeddings = np.random.randn(10, 64).astype(np.float32)
        result = sigreg(embeddings)
        assert result.shape == embeddings.shape

    def test_regularize_n_less_than_2(self):
        """n<2 → skips whitening, only L2-normalize."""
        from mci_world_model.sdk._sigreg import SIGReg

        sigreg = SIGReg()
        emb = np.random.randn(1, 16).astype(np.float32)
        result = sigreg.regularize(emb)
        assert result.shape == emb.shape
        # Should be L2-normalized
        norms = np.linalg.norm(result, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_regularize_rejects_non_2d(self):
        from mci_world_model.sdk._sigreg import SIGReg

        sigreg = SIGReg()
        with pytest.raises(ValueError, match="2D"):
            sigreg.regularize(np.random.randn(10).astype(np.float32))

    def test_regularize_rejects_zero_dim(self):
        from mci_world_model.sdk._sigreg import SIGReg

        sigreg = SIGReg()
        with pytest.raises(ValueError, match="不能为 0"):
            sigreg.regularize(np.zeros((0, 64), dtype=np.float32))

    def test_regularize_full_path_small_dim(self):
        """d <= sketch_dim → full whitening path (lines 111-116)."""
        from mci_world_model.sdk._sigreg import SIGReg

        # sketch_dim=64, d=32 → uses full whitening
        sigreg = SIGReg(use_sketch=True, sketch_dim=64, lambda_reg=0.1)
        emb = np.random.randn(20, 32).astype(np.float32)
        result = sigreg.regularize(emb)
        assert result.shape == emb.shape
        norms = np.linalg.norm(result, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_regularize_sketched_path_large_dim(self):
        """d > sketch_dim → sketched whitening (lines 96-109)."""
        from mci_world_model.sdk._sigreg import SIGReg

        # sketch_dim=8, d=64 → uses sketched path
        sigreg = SIGReg(use_sketch=True, sketch_dim=8, lambda_reg=0.1)
        emb = np.random.randn(15, 64).astype(np.float32)
        result = sigreg.regularize(emb)
        assert result.shape == emb.shape
        norms = np.linalg.norm(result, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_regularize_no_sketch_full_path(self):
        """use_sketch=False → always full whitening."""
        from mci_world_model.sdk._sigreg import SIGReg

        sigreg = SIGReg(use_sketch=False, lambda_reg=0.1)
        emb = np.random.randn(15, 64).astype(np.float32)
        result = sigreg.regularize(emb)
        assert result.shape == emb.shape

    def test_regularize_preserves_dtype(self):
        from mci_world_model.sdk._sigreg import SIGReg

        sigreg = SIGReg(lambda_reg=0.05)
        emb_f32 = np.random.randn(10, 16).astype(np.float32)
        result = sigreg.regularize(emb_f32)
        assert result.dtype == np.float32

    def test_regularize_improves_isotropy(self):
        """SIGReg should increase isotropy score."""
        from mci_world_model.sdk._sigreg import SIGReg

        np.random.seed(42)
        biased = np.abs(np.random.randn(50, 128)).astype(np.float32)
        sigreg = SIGReg(lambda_reg=0.1)
        iso_before = sigreg.compute_isotropy_score(biased)
        regularized = sigreg.regularize(biased)
        iso_after = sigreg.compute_isotropy_score(regularized)
        assert iso_after > iso_before


class TestSIGRegIsotropy:
    """Cover SIGReg.compute_isotropy_score() (lines 128-151)."""

    def test_isotropy_n_less_than_2_returns_zero(self):
        from mci_world_model.sdk._sigreg import SIGReg

        sigreg = SIGReg()
        score = sigreg.compute_isotropy_score(np.random.randn(1, 64).astype(np.float32))
        assert score == 0.0

    def test_isotropy_normal_case(self):
        from mci_world_model.sdk._sigreg import SIGReg

        sigreg = SIGReg()
        emb = np.random.randn(50, 64).astype(np.float32)
        score = sigreg.compute_isotropy_score(emb)
        assert 0.0 <= score <= 1.0

    def test_isotropy_biased_low_score(self):
        """Biased embeddings (all positive quadrant) have low isotropy."""
        from mci_world_model.sdk._sigreg import SIGReg

        np.random.seed(42)
        sigreg = SIGReg()
        biased = np.abs(np.random.randn(30, 64)).astype(np.float32)
        score = sigreg.compute_isotropy_score(biased)
        # Biased embeddings should have relatively low isotropy
        assert score < 0.5


# =============================================================================
# HierarchicalState Tests
# =============================================================================


class TestHierarchicalState:
    """Cover HierarchicalState.empty() and to_dict()."""

    def test_empty_returns_valid_state(self):
        from mci_world_model.sdk._hierarchical_encoder import HierarchicalState

        state = HierarchicalState.empty()
        assert state.timestamp == ""
        assert state.level_1 is not None
        assert state.level_2 is not None
        assert state.level_3 is not None

    def test_to_dict_includes_all_levels(self):
        from mci_world_model.sdk._hierarchical_encoder import HierarchicalState

        state = HierarchicalState.empty()
        d = state.to_dict()
        assert "l1_edges" in d
        assert "l2_edges" in d
        assert "l3_edges" in d
        assert "l1_confirmed" in d
        assert "l2_confirmed" in d
        assert "l3_confirmed" in d
        assert d["timestamp"] == ""


# =============================================================================
# HierarchicalJEPAEncoder Tests
# =============================================================================


class TestHierarchicalJEPAEncoderBasics:
    """Cover properties, empty-memories paths, evaluate."""

    @pytest.fixture
    def mock_wm(self) -> MagicMock:
        wm = MagicMock()
        wm._state = MagicMock()
        wm._state.causal_edges = []
        return wm

    @pytest.fixture
    def encoder(self, mock_wm: MagicMock):
        from mci_world_model.sdk._hierarchical_encoder import HierarchicalJEPAEncoder

        return HierarchicalJEPAEncoder(mock_wm, key_dim=8, hidden_dim=8, seed=42)

    def test_initial_state_is_idle(self, encoder):
        """New encoder starts in IDLE state."""
        assert encoder.state == "IDLE"

    def test_encode_count_starts_zero(self, encoder):
        assert encoder.encode_count == 0

    def test_predict_count_starts_zero(self, encoder):
        assert encoder.predict_count == 0

    def test_encode_with_empty_memories(self, encoder):
        """Empty memories → returns empty HierarchicalState, state=COMPLETE."""
        from mci_world_model.sdk._hierarchical_encoder import HierarchicalState

        result = encoder.encode([])
        assert isinstance(result, HierarchicalState)
        assert result.level_1 is not None
        assert encoder.state == "COMPLETE"

    def test_training_encode_empty_memories(self, encoder):
        """training_encode() with empty → zero array and empty dict."""
        result = encoder.training_encode([])
        assert isinstance(result, tuple)
        A_enc, node_index = result
        assert A_enc.shape == (0, 0)
        assert node_index == {}

    def test_evaluate_empty_dataset(self, encoder):
        """evaluate() with empty dataset → avg_distance=1.0, n=0."""
        result = encoder.evaluate([])
        assert result["avg_distance"] == 1.0
        assert result["n"] == 0


# =============================================================================
# CostModule Exception Path
# =============================================================================


class TestCostModuleExceptionPath:
    """Cover the except Exception fallback in evaluate() (lines 162-165)."""

    def test_evaluate_exception_returns_zero_signal(self):
        """When _compute_X methods raise, evaluate() returns CostSignal.zero()."""
        from mci_world_model.sdk._cost_module import EnergyCostModule

        module = EnergyCostModule()

        # Plain object without causal_edges attr → triggers AttributeError → except path
        class BadState:
            pass

        signal = module.evaluate(BadState())
        assert signal.total == 0.0
        assert signal.energy_balance == 0.0
        assert signal.causal_consistency == 0.0
        assert signal.temporal_coherence == 0.0
        assert module.state == "COMPLETE"
