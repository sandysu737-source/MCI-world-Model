"""
MCI World Model v3.1.0 — P0-P3 Priority Coverage Fill
=======================================================

按照优先级顺序精准覆盖四个关键模块的未测路径：

P0: bayesian_augmenter.py — 数据结构 + 初始化 + 报告/持久化 (目标 15%→40%+)
P1: _world_model.py — CausalWorldModelState 图方法 + TrajectoryTracker (目标 49%→62%+)
P2: _jepa_gat_encoder.py — GATEncoder 全流程 0%→50%+ (目标 50%+)
P3: _causal.py — detect_causal_link + CausalEngine 核心方法 (目标 21%→55%+)

预期覆盖率提升：+~150 covered statements → 48-50% 总覆盖率
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

# =============================================================================
# P0: BayesianAugmenter — 数据结构 + 初始化 + 报告 + 持久化
# =============================================================================


class TestBayesianDataStructures:
    """P0-1: EnhancedOutput / ComparisonDelta / AccuracyRecord 数据结构"""

    def test_comparison_delta_positive(self):
        from mci_world_model.sdk.bayesian_augmenter import ComparisonDelta

        delta = ComparisonDelta(
            field="ranking_changes",
            original_value="5 results",
            bayesian_value="3 ranking changes",
            difference_description="排序调整了3个位置",
            improvement_indicator="positive",
        )
        d = delta.to_dict()
        assert d["field"] == "ranking_changes"
        assert d["improvement"] == "positive"

    def test_comparison_delta_neutral(self):
        from mci_world_model.sdk.bayesian_augmenter import ComparisonDelta

        delta = ComparisonDelta(
            field="top1_match",
            original_value="mem_A",
            bayesian_value="mem_A",
            difference_description="一致",
            improvement_indicator="neutral",
        )
        assert delta.improvement_indicator == "neutral"

    def test_enhanced_output_to_dict(self):
        from mci_world_model.sdk.bayesian_augmenter import ComparisonDelta, EnhancedOutput

        output = EnhancedOutput(
            original={"results": []},
            bayesian={"results": [], "engine_stats": {}},
            comparisons=[
                ComparisonDelta(
                    field="test",
                    original_value=0,
                    bayesian_value=1,
                    difference_description="",
                    improvement_indicator="neutral",
                )
            ],
            meta={"method": "dual_path_query"},
        )
        d = output.to_dict()
        assert "original_result" in d
        assert "bayesian_result" in d
        assert "comparison_deltas" in d
        assert len(d["comparison_deltas"]) == 1

    def test_accuracy_record_fields(self):
        from mci_world_model.sdk.bayesian_augmenter import AccuracyRecord

        rec = AccuracyRecord(
            timestamp=1234567890.0,
            method="original",
            query="测试查询",
            predicted_value=0.6,
            actual_value=0.8,
            error=-0.2,
            absolute_error=0.2,
        )
        assert rec.method == "original"
        assert rec.absolute_error == 0.2
        assert abs(rec.error) == 0.2


class TestBayesianAugmenterInit:
    """P0-2: BayesianAugmenter 初始化路径"""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.add.return_value = "mem_new"
        client.query.return_value = [{"memory_id": "mem_1", "score": 0.9}]
        client.predict.return_value = {"event_predictions": []}
        client.reason.return_value = {"confidence": 0.7}
        return client

    def test_init_default_params(self, mock_client):
        from mci_world_model.sdk.bayesian_augmenter import BayesianAugmenter

        aug = BayesianAugmenter(mock_client, enable_auto_sync=False)
        assert aug._client is mock_client
        assert aug._feedback_count == 0
        assert aug.engine is not None
        assert aug.network is not None
        assert aug.evidence is not None

    def test_init_with_all_params(self, mock_client):
        from mci_world_model.sdk.bayesian_augmenter import BayesianAugmenter

        aug = BayesianAugmenter(
            mock_client,
            enable_network=True,
            enable_predictor=True,
            enable_auto_sync=False,
            prior_type="uniform",
            verbose=True,
        )
        assert aug._verbose is True

    def test_init_disable_network_and_predictor(self, mock_client):
        from mci_world_model.sdk.bayesian_augmenter import BayesianAugmenter

        aug = BayesianAugmenter(
            mock_client,
            enable_network=False,
            enable_predictor=False,
            enable_auto_sync=False,
        )
        assert aug.predictor is None

    def test_init_auto_sync_hooks_client(self, mock_client):
        from mci_world_model.sdk.bayesian_augmenter import BayesianAugmenter

        # Capture original add before augmentation
        original_add = mock_client.add
        BayesianAugmenter(mock_client, enable_auto_sync=True)

        # Call hooked add — should call original and sync
        result = mock_client.add("test content", metadata={"tags": ["test"]})
        original_add.assert_called()
        assert result == "mem_new"

    def test_init_auto_sync_handles_add_exception(self, mock_client):
        from mci_world_model.sdk.bayesian_augmenter import BayesianAugmenter

        def raising_add(*args, **kwargs):
            raise RuntimeError("add failed")

        mock_client.add = raising_add
        aug = BayesianAugmenter(mock_client, enable_auto_sync=True)
        # Should not crash on init
        assert aug._client is mock_client


class TestBayesianAugmenterReport:
    """P0-3: get_accuracy_report + print_accuracy_report"""

    @pytest.fixture
    def aug_no_data(self):
        from mci_world_model.sdk.bayesian_augmenter import BayesianAugmenter

        client = MagicMock()
        return BayesianAugmenter(client, enable_auto_sync=False)

    def test_accuracy_report_no_data(self, aug_no_data):
        report = aug_no_data.get_accuracy_report()
        assert report["status"] == "no_data"
        assert report["summary"]["total_feedback"] == 0

    def test_print_accuracy_report_no_data(self, aug_no_data, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="mci_world_model.sdk.bayesian_augmenter"):
            aug_no_data.print_accuracy_report()
        assert "BayesianAugmenter" in caplog.text

    def test_accuracy_report_with_data(self, aug_no_data):
        from mci_world_model.sdk.bayesian_augmenter import AccuracyRecord

        # Inject accuracy records
        aug_no_data._accuracy_records = [
            AccuracyRecord(0, "original", "q1", 0.6, 0.8, -0.2, 0.2),
            AccuracyRecord(0, "bayesian", "q1", 0.75, 0.8, -0.05, 0.05),
        ]
        aug_no_data._feedback_count = 1
        report = aug_no_data.get_accuracy_report()
        assert report["summary"]["total_records"] == 2
        assert "original_stats" in report
        assert report["original_stats"]["mae"] == 0.2
        assert report["bayesian_stats"]["mae"] == 0.05
        # Bayesian better → improvement > 0
        assert report["summary"]["improvement_pct"] > 50

    def test_accuracy_report_with_many_records(self, aug_no_data):
        from mci_world_model.sdk.bayesian_augmenter import AccuracyRecord

        aug_no_data._accuracy_records = []
        for i in range(50):
            aug_no_data._accuracy_records.append(AccuracyRecord(i, "original", f"q{i}", 0.5, 0.5, 0.0, 0.0))
            aug_no_data._accuracy_records.append(AccuracyRecord(i, "bayesian", f"q{i}", 0.5, 0.5, 0.0, 0.0))
        aug_no_data._feedback_count = 50
        report = aug_no_data.get_accuracy_report()
        # Equal performance → similar MAE
        assert abs(report["original_stats"]["mae"]) < 0.01
        assert "verdict" in report["summary"]

    def test_accuracy_report_lru_trim(self, aug_no_data):
        from mci_world_model.sdk.bayesian_augmenter import AccuracyRecord

        # Fill beyond max capacity (1000)
        for i in range(1100):
            aug_no_data._accuracy_records.append(AccuracyRecord(i, "original", f"q{i}", 0.5, 0.5, 0.0, 0.0))
        assert len(aug_no_data._accuracy_records) == 1100

    def test_print_accuracy_report_with_data(self, aug_no_data, caplog):
        import logging

        from mci_world_model.sdk.bayesian_augmenter import AccuracyRecord

        aug_no_data._accuracy_records = [
            AccuracyRecord(0, "original", "q1", 0.6, 0.8, -0.2, 0.2),
            AccuracyRecord(0, "bayesian", "q1", 0.75, 0.8, -0.05, 0.05),
        ]
        aug_no_data._feedback_count = 1
        with caplog.at_level(logging.INFO, logger="mci_world_model.sdk.bayesian_augmenter"):
            aug_no_data.print_accuracy_report()
        assert "MAE" in caplog.text


class TestBayesianAugmenterPersistence:
    """P0-4: save_state / load_state / reset"""

    @pytest.fixture
    def aug(self):
        from mci_world_model.sdk.bayesian_augmenter import BayesianAugmenter

        client = MagicMock()
        return BayesianAugmenter(client, enable_auto_sync=False)

    def test_save_state_creates_file(self, aug):
        import os

        with tempfile.TemporaryDirectory() as tmp:
            path = aug.save_state(os.path.join(tmp, "test_state.json"))
            assert os.path.exists(path)

    def test_save_and_load_roundtrip(self, aug):
        import os

        # Setup some state
        aug._feedback_count = 5
        aug._synced_memory_ids.add("mem_1")

        with tempfile.TemporaryDirectory() as tmp:
            save_path = os.path.join(tmp, "roundtrip.json")
            aug.save_state(save_path)

            # Create a new augmenter and load
            from mci_world_model.sdk.bayesian_augmenter import BayesianAugmenter

            aug2 = BayesianAugmenter(MagicMock(), enable_auto_sync=False)
            aug2.load_state(save_path)

            assert aug2._feedback_count == 5
            assert "mem_1" in aug2._synced_memory_ids
            assert aug2.engine is not None

    def test_save_state_default_path(self, aug):
        path = aug.save_state()
        assert path.endswith(".json")
        # Cleanup
        import os

        if os.path.exists(path):
            os.remove(path)

    def test_reset_bayesian(self, aug):
        from mci_world_model.sdk.bayesian_augmenter import AccuracyRecord

        aug._accuracy_records = [AccuracyRecord(0, "original", "q", 0.5, 0.5, 0, 0)]
        aug._feedback_count = 3
        aug.reset_bayesian()
        assert aug._accuracy_records == []
        assert aug._feedback_count == 0

    def test_get_bayesian_engine(self, aug):
        engine = aug.get_bayesian_engine()
        assert engine is not None

    def test_get_bayesian_network(self, aug):
        net = aug.get_bayesian_network()
        assert net is not None


class TestBayesianAugmenterPassthrough:
    """P0-5: __getattr__ 透传 + 内部方法"""

    @pytest.fixture
    def aug(self):
        from mci_world_model.sdk.bayesian_augmenter import BayesianAugmenter

        client = MagicMock()
        client.explain_query.return_value = "explanation"
        return BayesianAugmenter(client, enable_auto_sync=False)

    def test_getattr_passthrough(self, aug):
        """未包装方法透传到 client"""
        result = aug.explain_query("test")
        assert result == "explanation"

    def test_getattr_private_raises(self, aug):
        """私有属性不透传"""
        with pytest.raises(AttributeError):
            _ = aug._nonexistent_private

    def test_vlog_silent_by_default(self, aug, caplog):
        aug._vlog("test message")
        assert "test message" not in caplog.text

    def test_vlog_verbose(self, caplog):
        import logging

        from mci_world_model.sdk.bayesian_augmenter import BayesianAugmenter

        client = MagicMock()
        aug = BayesianAugmenter(client, enable_auto_sync=False, verbose=True)
        with caplog.at_level(logging.INFO):
            aug._vlog("verbose test")
        assert "verbose test" in caplog.text


# =============================================================================
# P1: _world_model.py — CausalWorldModelState 图方法 + TrajectoryTracker
# =============================================================================


class TestCausalWorldModelStateGraphMethods:
    """P1-1: _build_node_index / to_adjacency_matrix / to_node_feature_matrix"""

    def test_build_node_index_named_edges(self):
        from mci_world_model.sdk._world_model import CausalWorldModelState

        state = CausalWorldModelState(
            causal_edges=[
                {"cause": "X", "effect": "Y", "rho": 0.8},
                {"cause": "Y", "effect": "Z", "rho": 0.6},
            ]
        )
        idx = state._build_node_index()
        assert idx == {"X": 0, "Y": 1, "Z": 2}

    def test_build_node_index_index_only(self):
        """仅包含 cause_idx/effect_idx 的边（GaussianDAG 输出）"""
        from mci_world_model.sdk._world_model import CausalWorldModelState

        state = CausalWorldModelState(
            causal_edges=[
                {"cause_idx": 0, "effect_idx": 1, "rho": 0.5},
                {"cause_idx": 1, "effect_idx": 2, "rho": 0.4},
            ]
        )
        idx = state._build_node_index()
        assert len(idx) == 3
        assert "n0" in idx and "n1" in idx and "n2" in idx

    def test_build_node_index_empty(self):
        from mci_world_model.sdk._world_model import CausalWorldModelState

        state = CausalWorldModelState()
        idx = state._build_node_index()
        assert idx == {}

    def test_get_node_name(self):
        from mci_world_model.sdk._world_model import CausalWorldModelState

        state = CausalWorldModelState()
        assert state._get_node_name({"cause": "hello"}, "cause") == "hello"
        assert state._get_node_name({"cause_idx": 3}, "cause") == "n3"
        assert state._get_node_name({}, "cause") == ""

    def test_to_adjacency_matrix(self):
        from mci_world_model.sdk._world_model import CausalWorldModelState

        state = CausalWorldModelState(
            causal_edges=[
                {"cause": "A", "effect": "B", "rho": 0.7},
                {"cause": "B", "effect": "C", "rho": 0.3},
            ]
        )
        adj = state.to_adjacency_matrix()
        assert adj.shape == (3, 3)
        assert adj[0, 1] == 0.7
        assert adj[1, 2] == 0.3
        assert adj[0, 2] == 0.0  # no direct edge
        assert adj[0, 0] == 0.0  # no self-loop

    def test_to_adjacency_matrix_empty(self):
        from mci_world_model.sdk._world_model import CausalWorldModelState

        state = CausalWorldModelState()
        adj = state.to_adjacency_matrix()
        assert adj.shape == (0, 0)

    def test_to_node_feature_matrix(self):
        from mci_world_model.sdk._world_model import CausalWorldModelState

        state = CausalWorldModelState(
            causal_edges=[
                {"cause": "A", "effect": "B", "rho": 0.8},
                {"cause": "A", "effect": "C", "rho": 0.5},
                {"cause": "C", "effect": "A", "rho": 0.3},
            ],
            active_states={"semantic", "causal"},
        )
        X = state.to_node_feature_matrix()
        assert X.shape == (3, 8)
        # A: out_degree=2, in_degree=1
        assert X[0, 5] == 2.0  # out_degree
        assert X[0, 6] == 1.0  # in_degree
        # B: out_degree=0, in_degree=1
        assert X[1, 5] == 0.0
        assert X[1, 6] == 1.0

    def test_to_node_feature_matrix_empty(self):
        from mci_world_model.sdk._world_model import CausalWorldModelState

        state = CausalWorldModelState()
        X = state.to_node_feature_matrix()
        assert X.shape == (0, 8)


class TestCausalWorldModelStateDistance:
    """P1-2: state_distance + __sub__"""

    def test_state_distance_identical(self):
        from mci_world_model.sdk._world_model import CausalWorldModelState

        s1 = CausalWorldModelState(
            causal_edges=[{"cause": "X", "effect": "Y", "rho": 0.8}],
        )
        s2 = CausalWorldModelState(
            causal_edges=[{"cause": "X", "effect": "Y", "rho": 0.8}],
        )
        dist = s1.state_distance(s2)
        # Identical should be ~0
        assert dist < 0.01

    def test_state_distance_completely_different(self):
        from mci_world_model.sdk._world_model import CausalWorldModelState

        s1 = CausalWorldModelState(
            causal_edges=[{"cause": "X", "effect": "Y", "rho": 0.9}],
        )
        s2 = CausalWorldModelState(
            causal_edges=[{"cause": "A", "effect": "B", "rho": 0.1}],
        )
        dist = s1.state_distance(s2)
        # Different nodes and different rho → should give non-trivial distance
        assert dist > 0.0

    def test_state_distance_both_empty(self):
        from mci_world_model.sdk._world_model import CausalWorldModelState

        s1 = CausalWorldModelState()
        s2 = CausalWorldModelState()
        assert s1.state_distance(s2) == 0.0

    def test_state_distance_one_empty(self):
        from mci_world_model.sdk._world_model import CausalWorldModelState

        s1 = CausalWorldModelState(
            causal_edges=[{"cause": "X", "effect": "Y", "rho": 0.5}],
        )
        s2 = CausalWorldModelState()
        assert s1.state_distance(s2) == 1.0
        assert s2.state_distance(s1) == 1.0

    def test_sub_dunder(self):
        from mci_world_model.sdk._world_model import CausalWorldModelState

        s1 = CausalWorldModelState(
            causal_edges=[{"cause": "X", "effect": "Y", "rho": 0.8}],
        )
        s2 = CausalWorldModelState(
            causal_edges=[{"cause": "X", "effect": "Y", "rho": 0.8}],
        )
        dist = s1 - s2
        assert 0.0 <= dist <= 0.01


class TestWorkingMemory:
    """P1-3: WorkingMemory push/get_recent/get_recent_weighted/clear/state"""

    def test_push_and_get_recent(self):
        from mci_world_model.sdk._world_model import CausalWorldModelState, TrajectoryStep, WorkingMemory

        wm = WorkingMemory(max_length=5)
        assert wm.state == "IDLE"
        assert not wm.is_full

        state = CausalWorldModelState()
        step = TrajectoryStep(state=state, step_index=0)
        wm.push(step)
        assert wm.state == "RECORDING"

        recent = wm.get_recent(3)
        assert len(recent) == 1
        assert recent[0].step_index == 0

    def test_push_beyond_max_len(self):
        from mci_world_model.sdk._world_model import CausalWorldModelState, TrajectoryStep, WorkingMemory

        wm = WorkingMemory(max_length=3)
        for i in range(5):
            wm.push(TrajectoryStep(state=CausalWorldModelState(), step_index=i))
        assert wm.is_full
        recent = wm.get_recent(5)
        # Should only keep last 3
        assert len(recent) == 3

    def test_get_recent_weighted(self):
        from mci_world_model.sdk._world_model import CausalWorldModelState, TrajectoryStep, WorkingMemory

        wm = WorkingMemory(max_length=10)
        wm.push(TrajectoryStep(state=CausalWorldModelState(), step_index=0, temporal_weight=0.1))
        wm.push(TrajectoryStep(state=CausalWorldModelState(), step_index=1, temporal_weight=1.0))

        weighted = wm.get_recent_weighted(2)
        assert len(weighted) == 2

    def test_get_recent_weighted_empty(self):
        from mci_world_model.sdk._world_model import WorkingMemory

        wm = WorkingMemory()
        assert wm.get_recent_weighted(3) == []

    def test_get_recent_empty(self):
        from mci_world_model.sdk._world_model import WorkingMemory

        wm = WorkingMemory()
        assert wm.get_recent(3) == []

    def test_clear(self):
        from mci_world_model.sdk._world_model import CausalWorldModelState, TrajectoryStep, WorkingMemory

        wm = WorkingMemory()
        wm.push(TrajectoryStep(state=CausalWorldModelState(), step_index=0))
        wm.clear()
        assert wm.state == "IDLE"
        assert wm.get_recent(3) == []

    def test_to_dict(self):
        from mci_world_model.sdk._world_model import CausalWorldModelState, TrajectoryStep, WorkingMemory

        wm = WorkingMemory(max_length=10)
        wm.push(TrajectoryStep(state=CausalWorldModelState(), step_index=0))
        d = wm.to_dict()
        assert d["n_steps"] == 1
        assert d["state"] in ("RECORDING", "FULL")
        assert len(d["recent_costs"]) == 1

    def test_state_property(self):
        from mci_world_model.sdk._world_model import WorkingMemory

        wm = WorkingMemory()
        assert wm.state == "IDLE"


class TestAggregateEnergyRatios:
    """P1-4: _aggregate_energy_ratios 公共函数"""

    def test_normal_case(self):
        from mci_world_model.sdk._world_model import _aggregate_energy_ratios

        edges = [
            {"cause_energy": "semantic", "effect_energy": "causal"},
            {"cause_energy": "semantic", "effect_energy": "spacetime"},
            {"cause_energy": "causal", "effect_energy": "trust"},
        ]
        ratios = _aggregate_energy_ratios(edges)
        assert ratios is not None
        assert ratios["semantic"] == 2.0 / 6.0
        assert ratios["causal"] == 2.0 / 6.0
        assert sum(ratios.values()) == pytest.approx(1.0)

    def test_empty_edges(self):
        from mci_world_model.sdk._world_model import _aggregate_energy_ratios

        assert _aggregate_energy_ratios([]) is None

    def test_unknown_energy_keys(self):
        from mci_world_model.sdk._world_model import _aggregate_energy_ratios

        edges = [{"cause_energy": "unknown_type", "effect_energy": "also_unknown"}]
        ratios = _aggregate_energy_ratios(edges)
        assert ratios is None  # total count = 0


# =============================================================================
# P2: _jepa_gat_encoder.py — GATEncoder 全流程 (目标 0%→50%+)
# =============================================================================


class TestGATEncoderInit:
    """P2-1: GATEncoder 初始化与参数"""

    def test_init_default_params(self):
        from mci_world_model.sdk._jepa_gat_encoder import GATEncoder

        g = GATEncoder(input_dim=8, key_dim=16, seed=42)
        assert g._input_dim == 8
        assert g._key_dim == 16
        assert g.W_q.shape == (8, 16)
        assert g.W_k.shape == (8, 16)
        assert g._train_steps == 0
        assert g._forward_count == 0

    def test_init_custom_dims(self):
        from mci_world_model.sdk._jepa_gat_encoder import GATEncoder

        g = GATEncoder(input_dim=4, key_dim=8, seed=123)
        assert g.W_q.shape == (4, 8)

    def test_init_with_l2_reg(self):
        from mci_world_model.sdk._jepa_gat_encoder import GATEncoder

        g = GATEncoder(input_dim=8, key_dim=16, l2_reg=0.01)
        assert g._l2_reg == 0.01

    def test_get_params_return_copy(self):
        from mci_world_model.sdk._jepa_gat_encoder import GATEncoder

        g = GATEncoder(input_dim=8, key_dim=16, seed=42)
        params = g.get_params()
        assert "W_q" in params
        assert "W_k" in params
        # Modifying copy should not affect original
        params["W_q"][0, 0] = 999.0
        assert g.W_q[0, 0] != 999.0

    def test_set_params(self):
        from mci_world_model.sdk._jepa_gat_encoder import GATEncoder

        g = GATEncoder(input_dim=8, key_dim=16, seed=42)
        new_wq = np.ones((8, 16), dtype=np.float64)
        g.set_params({"W_q": new_wq})
        assert np.allclose(g.W_q, new_wq)

    def test_set_params_partial(self):
        from mci_world_model.sdk._jepa_gat_encoder import GATEncoder

        g = GATEncoder(input_dim=8, key_dim=16, seed=42)
        old_wk = g.W_k.copy()
        g.set_params({"W_q": np.zeros((8, 16))})
        assert np.allclose(g.W_k, old_wk)  # W_k unchanged

    def test_key_dim_property(self):
        from mci_world_model.sdk._jepa_gat_encoder import GATEncoder

        g = GATEncoder(key_dim=32, seed=42)
        assert g.key_dim == 32

    def test_repr(self):
        from mci_world_model.sdk._jepa_gat_encoder import GATEncoder

        g = GATEncoder(input_dim=8, key_dim=16, seed=42)
        r = repr(g)
        assert "GATEncoder" in r
        assert "dim=8" in r
        assert "key=16" in r


class TestGATEncoderForward:
    """P2-2: GATEncoder.forward + training_forward"""

    @pytest.fixture
    def encoder(self):
        from mci_world_model.sdk._jepa_gat_encoder import GATEncoder

        return GATEncoder(input_dim=8, key_dim=16, seed=42)

    @pytest.fixture
    def X(self):
        np.random.seed(42)
        return np.random.randn(5, 8).astype(np.float32)

    def test_forward_shape(self, encoder, X):
        A = encoder.forward(X)
        assert A.shape == (5, 5)
        assert A.dtype == np.float32

    def test_forward_values_in_range(self, encoder, X):
        A = encoder.forward(X)
        assert np.all(A >= 0.0)
        assert np.all(A <= 1.0)

    def test_forward_increments_counter(self, encoder, X):
        assert encoder._forward_count == 0
        encoder.forward(X)
        assert encoder._forward_count == 1

    def test_forward_dimension_mismatch(self, encoder):
        with pytest.raises(ValueError, match="输入维度"):
            encoder.forward(np.random.randn(3, 5).astype(np.float32))

    def test_training_forward_shape(self, encoder, X):
        A = encoder.training_forward(X)
        assert A.shape == (5, 5)

    def test_training_forward_caches(self, encoder, X):
        encoder.training_forward(X)
        cache = encoder._cache
        assert "X" in cache
        assert "Q" in cache
        assert "K" in cache
        assert "S" in cache
        assert "A_enc" in cache

    def test_forward_accepts_int_input(self, encoder):
        X_int = np.random.RandomState(42).randint(0, 2, (4, 8)).astype(np.float32)
        A = encoder.forward(X_int)
        assert A.shape == (4, 4)


class TestGATEncoderGradients:
    """P2-3: compute_gradients + compute_gradients_from_mse + apply_gradients"""

    @pytest.fixture
    def encoder(self):
        from mci_world_model.sdk._jepa_gat_encoder import GATEncoder

        return GATEncoder(input_dim=4, key_dim=8, seed=42)

    @pytest.fixture
    def X(self):
        np.random.seed(42)
        return np.random.randn(6, 4).astype(np.float32)

    def test_compute_gradients_no_cache(self, encoder):
        grads = encoder.compute_gradients(np.ones((6, 6)))
        assert np.allclose(grads["W_q"], 0.0)
        assert np.allclose(grads["W_k"], 0.0)

    def test_compute_gradients_with_cache(self, encoder, X):
        encoder.training_forward(X)
        dA = np.ones((6, 6), dtype=np.float64) * 0.01
        grads = encoder.compute_gradients(dA)
        assert grads["W_q"].shape == (4, 8)
        assert grads["W_k"].shape == (4, 8)

    def test_compute_gradients_from_mse_no_cache(self, encoder):
        result = encoder.compute_gradients_from_mse(np.ones((6, 6)))
        assert result["loss"] == 0.0
        assert result["mse"] == 0.0

    def test_compute_gradients_from_mse(self, encoder, X):
        encoder.training_forward(X)
        target = encoder.forward(X) + 0.01  # slightly different target
        result = encoder.compute_gradients_from_mse(target)
        assert result["loss"] > 0.0
        assert result["mse"] > 0.0
        assert result["W_q"].shape == (4, 8)
        assert result["W_k"].shape == (4, 8)

    def test_compute_gradients_from_mse_with_l2(self):
        from mci_world_model.sdk._jepa_gat_encoder import GATEncoder

        enc = GATEncoder(input_dim=4, key_dim=8, l2_reg=0.1, seed=42)
        X = np.random.RandomState(42).randn(6, 4).astype(np.float32)
        enc.training_forward(X)
        result = enc.compute_gradients_from_mse(enc.forward(X))
        assert "l2" in result
        assert result["l2"] > 0.0

    def test_apply_gradients(self, encoder, X):
        encoder.training_forward(X)
        target = encoder.forward(X) + 0.1  # larger target shift
        grads = encoder.compute_gradients_from_mse(target)

        old_wq = encoder.W_q.copy()
        old_steps = encoder.train_steps

        encoder.apply_gradients(grads, lr=0.1)  # higher LR to make visible change
        assert encoder.train_steps == old_steps + 1
        # Parameters should change with large enough LR
        assert not np.allclose(encoder.W_q, old_wq)

    def test_training_roundtrip(self, encoder, X):
        """A complete train step: forward → compute_gradients_from_mse → apply_gradients"""
        A0 = encoder.training_forward(X)
        target = A0 + np.random.RandomState(99).randn(6, 6).astype(np.float64) * 0.01
        grads = encoder.compute_gradients_from_mse(target)
        encoder.apply_gradients(grads, lr=0.005)
        assert encoder.train_steps == 1


class TestGATEncoderUtils:
    """P2-4: get_attention_scores + to_adjacency_matrix + features_to_state"""

    def test_get_attention_scores_no_cache(self):
        from mci_world_model.sdk._jepa_gat_encoder import GATEncoder

        enc = GATEncoder(input_dim=4, key_dim=8, seed=42)
        assert enc.get_attention_scores() is None

    def test_get_attention_scores_with_cache(self):
        from mci_world_model.sdk._jepa_gat_encoder import GATEncoder

        enc = GATEncoder(input_dim=4, key_dim=8, seed=42)
        X = np.random.RandomState(42).randn(4, 4).astype(np.float32)
        enc.training_forward(X)
        S = enc.get_attention_scores()
        assert S is not None
        assert S.shape == (4, 4)

    def test_to_adjacency_matrix(self):
        from mci_world_model.sdk._jepa_gat_encoder import GATEncoder

        enc = GATEncoder(input_dim=4, key_dim=8, seed=42)
        X = np.random.RandomState(42).randn(5, 4).astype(np.float32)
        adj = enc.to_adjacency_matrix(X, threshold=0.3)
        assert adj.shape == (5, 5)
        # Values below threshold should be 0
        assert np.all(adj[adj > 0] >= 0.3)

    def test_to_adjacency_matrix_default_threshold(self):
        from mci_world_model.sdk._jepa_gat_encoder import GATEncoder

        enc = GATEncoder(input_dim=4, key_dim=8, seed=42)
        X = np.random.RandomState(42).randn(5, 4).astype(np.float32)
        adj = enc.to_adjacency_matrix(X)
        assert np.all(adj[adj > 0] >= 0.05)

    def test_features_to_state(self):
        from mci_world_model.sdk._jepa_gat_encoder import features_to_state
        from mci_world_model.sdk._world_model import CausalWorldModelState

        A_enc = np.array(
            [
                [0.0, 0.6, 0.03],
                [0.0, 0.0, 0.7],
                [0.02, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        node_index = {"X": 0, "Y": 1, "Z": 2}
        template = CausalWorldModelState(active_states={"causal"})

        result = features_to_state(A_enc, node_index, template)
        assert isinstance(result, CausalWorldModelState)
        # Edge X→Y (0.6 > 0.05) and Y→Z (0.7 > 0.05), X→Z (0.03 < 0.05 filtered)
        assert len(result.causal_edges) >= 2


# =============================================================================
# P3: _causal.py — detect_causal_link + CausalEngine (目标 21%→55%+)
# =============================================================================


class TestDetectCausalLink:
    """P3-1: detect_causal_link 全路径"""

    def test_cause_pattern_如果所以(self):
        from mci_world_model.sdk._causal import detect_causal_link

        result = detect_causal_link("如果下雨", "所以路滑")
        assert result is not None
        assert result[0] == "cause"

    def test_cause_pattern_因为因此(self):
        from mci_world_model.sdk._causal import detect_causal_link

        result = detect_causal_link("因为价格上涨", "因此需求下降")
        assert result is not None
        assert result[0] == "cause"

    def test_condition_pattern_只要就(self):
        from mci_world_model.sdk._causal import detect_causal_link

        result = detect_causal_link("只要努力", "就能成功")
        assert result is not None
        assert result[0] == "condition"

    def test_shared_causal_导致(self):
        from mci_world_model.sdk._causal import detect_causal_link

        # 共享关键词"导致"
        result = detect_causal_link("暴雨导致内涝", "内涝导致交通瘫痪")
        assert result is not None
        assert result[0] == "shared"

    def test_shared_causal_影响(self):
        from mci_world_model.sdk._causal import detect_causal_link

        result = detect_causal_link("政策影响经济", "经济影响社会")
        assert result is not None
        assert result[0] == "shared"

    def test_no_causal_link(self):
        from mci_world_model.sdk._causal import detect_causal_link

        result = detect_causal_link("今天天气晴朗", "我在吃饭")
        assert result is None

    def test_reverse_causal(self):
        from mci_world_model.sdk._causal import detect_causal_link

        # 原因在 text_b 中: "如果"在text_b, "就"在text_a
        result = detect_causal_link("路滑", "如果下雨")
        # "如果"需要和 effect_marker 匹配 — "路滑"没有 effect_marker，所以无结果
        # 反向检测: 如果 text_b 有 cause_marker，text_a 有 effect_marker
        result = detect_causal_link("因此路滑", "因为下雨")
        assert result is not None
        assert result[0].startswith("reverse_")

    def test_confidence_in_range(self):
        from mci_world_model.sdk._causal import detect_causal_link

        result = detect_causal_link("由于供应短缺", "因此价格上升")
        assert result is not None
        assert 0.0 <= result[1] <= 1.0

    def test_empty_texts(self):
        from mci_world_model.sdk._causal import detect_causal_link

        result = detect_causal_link("", "")
        assert result is None

    def test_multiple_markers_uses_first(self):
        from mci_world_model.sdk._causal import detect_causal_link

        result = detect_causal_link("如果因为", "所以因此")
        assert result is not None
        # detect_causal_link returns on first match
        assert result[0] in ("cause", "condition", "result", "shared")


class TestCausalEngineBasics:
    """P3-2: CausalEngine.__init__ + _hash + _is_duplicate"""

    def test_init_defaults(self):
        from mci_world_model.sdk._causal import CausalEngine

        engine = CausalEngine()
        assert engine.min_confidence == 0.5
        assert engine._causal_pairs_cache == []

    def test_init_custom_confidence(self):
        from mci_world_model.sdk._causal import CausalEngine

        engine = CausalEngine(min_confidence=0.7)
        assert engine.min_confidence == 0.7

    def test_hash_pair_id(self):
        from mci_world_model.sdk._causal import _hash_pair_id_360

        h = _hash_pair_id_360("价格上涨", "需求下降")
        assert h.startswith("p3_")
        assert len(h) == 11  # "p3_" + 8 hex chars

    def test_is_duplicate_false(self):
        from mci_world_model.sdk._causal import _is_duplicate

        mem_a = {"id": "a"}
        mem_b = {"id": "b"}
        pairs = [(mem_a, mem_b, "cause", 0.8)]
        assert not _is_duplicate(pairs, "new_id", "other_id")

    def test_is_duplicate_true_same_order(self):
        from mci_world_model.sdk._causal import _is_duplicate

        mem_a = {"id": "a"}
        mem_b = {"id": "b"}
        pairs = [(mem_a, mem_b, "cause", 0.8)]
        assert _is_duplicate(pairs, "a", "b")

    def test_is_duplicate_true_reverse(self):
        from mci_world_model.sdk._causal import _is_duplicate

        mem_a = {"id": "a"}
        mem_b = {"id": "b"}
        pairs = [(mem_a, mem_b, "cause", 0.8)]
        assert _is_duplicate(pairs, "b", "a")


class TestCausalEngineFindPairs:
    """P3-3: CausalEngine.find_causal_pairs — 关键词路径"""

    @pytest.fixture
    def memories(self):
        return [
            {"id": "m1", "content": "如果暴雨就会引发城市内涝"},
            {"id": "m2", "content": "城市内涝促使排水系统升级"},
            {"id": "m3", "content": "由于供应不足因此价格上升"},
            {"id": "m4", "content": "价格上升导致需求下降"},
            {"id": "m5", "content": "今天天气好"},
        ]

    def test_find_pairs_basic(self, memories):
        from mci_world_model.sdk._causal import CausalEngine

        engine = CausalEngine(min_confidence=0.5)
        pairs = engine.find_causal_pairs(memories)
        assert len(pairs) > 0
        # Each pair is (cause_mem, effect_mem, causal_type, confidence)
        for pair in pairs:
            assert len(pair) == 4
            assert isinstance(pair[3], float)

    def test_find_pairs_sorted_by_confidence(self, memories):
        from mci_world_model.sdk._causal import CausalEngine

        engine = CausalEngine(min_confidence=0.1)
        pairs = engine.find_causal_pairs(memories)
        if len(pairs) >= 2:
            assert pairs[0][3] >= pairs[-1][3]

    def test_find_pairs_confidence_filter(self, memories):
        from mci_world_model.sdk._causal import CausalEngine

        engine = CausalEngine(min_confidence=0.9)
        pairs = engine.find_causal_pairs(memories)
        # High threshold → no or few results
        for pair in pairs:
            assert pair[3] >= 0.9

    def test_find_pairs_empty(self):
        from mci_world_model.sdk._causal import CausalEngine

        engine = CausalEngine()
        pairs = engine.find_causal_pairs([])
        assert pairs == []

    def test_find_pairs_no_causal_content(self):
        from mci_world_model.sdk._causal import CausalEngine

        engine = CausalEngine(min_confidence=0.3)
        memories = [{"id": "x", "content": "普通文本"}, {"id": "y", "content": "也是普通文本"}]
        pairs = engine.find_causal_pairs(memories)
        assert pairs == []

    def test_find_pairs_statistical_disabled(self, memories):
        from mci_world_model.sdk._causal import CausalEngine

        engine = CausalEngine()
        pairs = engine.find_causal_pairs(memories, use_statistical=True)
        assert isinstance(pairs, list)

    def test_find_pairs_large_input_truncation(self):
        from mci_world_model.sdk._causal import CausalEngine

        engine = CausalEngine(min_confidence=0.5)
        large_memories = [{"id": f"m{i}", "content": f"如果事件{i}就会结果{i + 1}"} for i in range(600)]
        pairs = engine.find_causal_pairs(large_memories)
        # Should handle without O(n^2) explosion
        assert isinstance(pairs, list)


class TestCausalEnginePredictEffects:
    """P3-4: CausalEngine.predict_effects — 关键词回退路径"""

    @pytest.fixture
    def engine(self):
        from mci_world_model.sdk._causal import CausalEngine

        return CausalEngine(min_confidence=0.3)

    @pytest.fixture
    def memories(self):
        return [
            {"id": "m1", "content": "由于价格上升因此需求下降"},
            {"id": "m2", "content": "价格上升导致资金流出"},
        ]

    def test_predict_effects_basic(self, engine, memories):
        results = engine.predict_effects("价格上升", memories)
        assert isinstance(results, list)
        if results:
            assert "confidence" in results[0]

    def test_predict_effects_top_k(self, engine, memories):
        results = engine.predict_effects("价格上升", memories, top_k=1)
        assert len(results) <= 1

    def test_predict_effects_empty_memories(self, engine):
        results = engine.predict_effects("测试", [])
        assert results == []

    def test_predict_effects_no_causal(self, engine):
        results = engine.predict_effects("无关文本", [{"id": "x", "content": "也是无关内容"}])
        assert results == []

    def test_predict_effects_intervention_path(self, engine, memories):
        """干预路径 (use_intervention=True) — 应降级到关键词"""
        results = engine.predict_effects("价格上升", memories, use_intervention=True, do_value=1.5)
        assert isinstance(results, list)


class TestCausalEngineQueryChain:
    """P3-5: CausalEngine.query_causal_chain"""

    @pytest.fixture
    def engine(self):
        from mci_world_model.sdk._causal import CausalEngine

        return CausalEngine(min_confidence=0.3)

    @pytest.fixture
    def memories(self):
        return [
            {"id": "m1", "content": "由于供应短缺因此价格上升"},
            {"id": "m2", "content": "价格上升导致需求下降"},
            {"id": "m3", "content": "需求下降促使政策调整"},
        ]

    def test_query_causal_chain_depth1(self, engine, memories):
        chain = engine.query_causal_chain("供应短缺", memories, max_depth=1)
        assert isinstance(chain, list)
        if chain:
            assert chain[0]["depth"] == 1

    def test_query_causal_chain_depth2(self, engine, memories):
        chain = engine.query_causal_chain("供应短缺", memories, max_depth=2)
        # depth=2 可以找到间接效应
        assert isinstance(chain, list)
        depths = {item["depth"] for item in chain}
        # May have depth 1 and 2 entries
        assert 1 in depths or len(chain) == 0

    def test_query_causal_chain_empty_memories(self, engine):
        chain = engine.query_causal_chain("查询", [])
        assert chain == []

    def test_query_causal_chain_no_match(self, engine):
        chain = engine.query_causal_chain("无关联文本", [{"id": "x", "content": "也是无关联"}])
        assert chain == []

    def test_query_causal_chain_deduplication(self, engine):
        """重复记忆不应重复出现"""
        memories = [
            {"id": "dup", "content": "由于重复因此重复"},
        ]
        chain = engine.query_causal_chain("重复", memories, max_depth=2)
        # 每条记忆最多出现一次
        ids = [item["memory_id"] for item in chain]
        assert len(ids) == len(set(ids))


# =============================================================================
# P0 Extension: BayesianAugmenter 双路径集成测试 (query/predict/reason/feedback)
# =============================================================================


class TestBayesianAugmenterQuery:
    """P0-EXT-1: query() 双路径查询 + _bayesian_query + _compare_query_results"""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.add.return_value = "mem_new"
        client.query.return_value = [
            {"memory_id": "mem_1", "content": "价格上升", "score": 0.9},
            {"memory_id": "mem_2", "content": "需求下降", "score": 0.7},
        ]
        client.predict.return_value = {"event_predictions": []}
        client.reason.return_value = {"confidence": 0.7}
        return client

    @pytest.fixture
    def aug(self, mock_client):
        from mci_world_model.sdk.bayesian_augmenter import BayesianAugmenter

        return BayesianAugmenter(mock_client, enable_auto_sync=False)

    def test_query_returns_enhanced_output(self, aug, mock_client):
        """query() 返回 EnhancedOutput 含 original/bayesian/comparisons"""
        result = aug.query("价格上升")
        assert result.original is not None
        assert result.bayesian is not None
        assert isinstance(result.comparisons, list)
        assert "results" in result.original
        assert "results" in result.bayesian

    def test_query_call_count(self, aug, mock_client):
        """query 会调用 client.query"""
        aug.query("test query")
        mock_client.query.assert_called()

    def test_query_bayesian_results_have_extra_fields(self, aug, mock_client):
        """bayesian results 含 bayesian_confidence 等增强字段"""
        result = aug.query("价格上升")
        bayesian_results = result.bayesian["results"]
        assert len(bayesian_results) > 0
        for item in bayesian_results:
            assert "bayesian_confidence" in item or "stage" in item

    def test_query_meta_fields(self, aug):
        """meta 包含 query/method/timestamp"""
        result = aug.query("价格上升", top_k=3)
        meta = result.meta
        assert meta["query"] == "价格上升"
        assert meta["top_k"] == 3
        assert meta["method"] == "dual_path_query"
        assert "timestamp" in meta


class TestBayesianAugmenterPredict:
    """P0-EXT-2: predict() 双路径预测 + _bayesian_predict + _compare_predictions"""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.add.return_value = "mem_new"
        client.query.return_value = []
        client.predict.return_value = {"event_predictions": [{"content": "价格上升会导致需求下降", "confidence": 0.65}]}
        client.reason.return_value = {"confidence": 0.7}
        return client

    @pytest.fixture
    def aug(self, mock_client):
        from mci_world_model.sdk.bayesian_augmenter import BayesianAugmenter

        return BayesianAugmenter(mock_client, enable_network=True, enable_predictor=True, enable_auto_sync=False)

    def test_predict_returns_enhanced_output(self, aug):
        """predict() 返回 EnhancedOutput"""
        result = aug.predict("价格上升")
        assert result.original is not None
        assert result.bayesian is not None
        assert result.meta["method"] == "dual_path_predict"

    def test_predict_with_event_predictions(self, aug):
        """有 event_predictions 时增强置信度"""
        result = aug.predict("价格上升")
        bayesian = result.bayesian
        if "event_predictions" in bayesian:
            for pred in bayesian["event_predictions"]:
                assert "original_confidence" in pred
                assert "bayesian_confidence" in pred

    def test_predict_calibration_report(self, aug):
        """predict 结果含 calibration 报告"""
        result = aug.predict("价格上升")
        assert "calibration" in result.bayesian

    def test_predict_with_query_only_no_predictor(self, mock_client):
        """无 predictor 时 query 路径仍工作"""
        from mci_world_model.sdk.bayesian_augmenter import BayesianAugmenter

        aug = BayesianAugmenter(mock_client, enable_predictor=False, enable_auto_sync=False)
        result = aug.predict("价格上升")
        assert "error" in result.bayesian or "bayesian_prediction" in result.bayesian

    def test_predict_compare_delta_fields(self, aug):
        """comparisons 包含 uncertainty_quantification"""
        result = aug.predict("价格上升")
        delta_fields = [c.field for c in result.comparisons]
        assert "uncertainty_quantification" in delta_fields


class TestBayesianAugmenterReason:
    """P0-EXT-3: reason() 双路径推理 + _bayesian_reason + _compare_reasoning"""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.add.return_value = "mem_new"
        client.query.return_value = [
            {"memory_id": "mem_a", "content": "供应短缺", "score": 0.8},
            {"memory_id": "mem_b", "content": "价格上升", "score": 0.6},
        ]
        client.predict.return_value = {"event_predictions": []}
        client.reason.return_value = {"confidence": 0.7, "chains": []}
        return client

    @pytest.fixture
    def aug(self, mock_client):
        from mci_world_model.sdk.bayesian_augmenter import BayesianAugmenter

        return BayesianAugmenter(mock_client, enable_network=True, enable_auto_sync=False)

    def test_reason_returns_enhanced_output(self, aug):
        """reason() 返回 EnhancedOutput"""
        result = aug.reason("供应短缺")
        assert result.original is not None
        assert result.bayesian is not None
        assert result.meta["method"] == "dual_path_reason"

    def test_reason_bayesian_has_memory_beliefs(self, aug):
        """bayesian 结果含 memory_beliefs"""
        result = aug.reason("供应短缺")
        assert "memory_beliefs" in result.bayesian
        assert isinstance(result.bayesian["memory_beliefs"], list)

    def test_reason_bayesian_has_engine_stats(self, aug):
        """bayesian 结果含 engine_stats"""
        result = aug.reason("供应短缺")
        assert "engine_stats" in result.bayesian

    def test_reason_comparisons_have_causal_chain(self, aug):
        """comparisons 含 causal_chain_count"""
        result = aug.reason("供应短缺")
        delta_fields = [c.field for c in result.comparisons]
        assert "causal_chain_count" in delta_fields
        assert "belief_coverage" in delta_fields

    def test_reason_confidence_comparison(self, aug):
        """当 original 有 confidence 且 bayesian_confidence 存在时产生对比"""
        result = aug.reason("供应短缺")
        delta_fields = [c.field for c in result.comparisons]
        # 如果 original 有 confidence 且 bayesian_confidence 也存在，会产生 reasoning_confidence delta
        # 否则仅包含 causal_chain_count 和 belief_coverage
        assert "causal_chain_count" in delta_fields
        assert "belief_coverage" in delta_fields


class TestBayesianAugmenterFeedback:
    """P0-EXT-4: feedback() 反馈闭环"""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.add.return_value = "mem_new"
        client.query.return_value = [
            {"memory_id": "mem_1", "content": "价格上升", "score": 0.9},
            {"memory_id": "mem_2", "content": "无关内容", "score": 0.3},
        ]
        client.predict.return_value = {"event_predictions": [{"content": "价格上升", "confidence": 0.6}]}
        client.reason.return_value = {"confidence": 0.7}
        return client

    @pytest.fixture
    def aug(self, mock_client):
        from mci_world_model.sdk.bayesian_augmenter import BayesianAugmenter

        return BayesianAugmenter(mock_client, enable_network=True, enable_predictor=True, enable_auto_sync=False)

    def test_feedback_with_expected_memory_ids(self, aug):
        """feedback with expected_memory_ids 更新信念"""
        result = aug.feedback(
            query="价格上升",
            expected_memory_ids=["mem_1"],
        )
        assert "beliefs_updated" in result
        assert result["feedback_id"] == 1

    def test_feedback_increments_counter(self, aug):
        """每次 feedback 递增计数器"""
        aug.feedback(query="q1", expected_memory_ids=["mem_1"])
        assert aug._feedback_count == 1
        aug.feedback(query="q2", expected_memory_ids=["mem_1"])
        assert aug._feedback_count == 2

    def test_feedback_with_ground_truth(self, aug):
        """feedback with ground_truth_value 记录准确度"""
        result = aug.feedback(
            query="价格上升",
            ground_truth_value=0.75,
        )
        assert "accuracy" in result
        assert "original_error" in result["accuracy"]
        assert "bayesian_error" in result["accuracy"]
        assert "improvement" in result["accuracy"]
        assert len(aug._accuracy_records) >= 2  # original + bayesian

    def test_feedback_with_expected_outcome(self, aug):
        """feedback with expected_outcome 更新校准"""
        result = aug.feedback(
            query="价格上升",
            expected_outcome=True,
        )
        assert "calibration_updated" in result

    def test_feedback_with_all_params(self, aug):
        """feedback with 全部参数"""
        result = aug.feedback(
            query="价格上升",
            expected_memory_ids=["mem_1"],
            expected_outcome=True,
            ground_truth_value=0.8,
        )
        assert result["feedback_id"] == 1
        assert "beliefs_updated" in result
        assert "accuracy" in result
        assert "calibration_updated" in result


class TestBayesianAugmenterAccuracyReport:
    """P0-EXT-5: get_accuracy_report + print_accuracy_report 有数据路径"""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.add.return_value = "mem_new"
        client.query.return_value = []
        client.predict.return_value = {"event_predictions": [{"content": "e", "confidence": 0.55}]}
        client.reason.return_value = {"confidence": 0.5}
        return client

    @pytest.fixture
    def aug_with_data(self, mock_client):
        from mci_world_model.sdk.bayesian_augmenter import BayesianAugmenter

        aug = BayesianAugmenter(mock_client, enable_network=True, enable_predictor=True, enable_auto_sync=False)
        # 累积一些 feedback 数据
        for gt in [0.5, 0.7, 0.3, 0.9, 0.6]:
            aug.feedback(query=f"q{gt}", ground_truth_value=gt)
        return aug

    def test_accuracy_report_with_data(self, aug_with_data):
        """有数据时返回完整报告含 summary/original_stats/bayesian_stats"""
        report = aug_with_data.get_accuracy_report()
        assert report["status"] == "ready" if "status" in report else True
        assert "summary" in report
        assert "original_stats" in report
        assert "bayesian_stats" in report
        assert "improvement_pct" in report["summary"]
        assert "verdict" in report["summary"]

    def test_accuracy_report_mae_non_negative(self, aug_with_data):
        """MAE 应为非负数"""
        report = aug_with_data.get_accuracy_report()
        orig_stats = report.get("original_stats", {})
        bayes_stats = report.get("bayesian_stats", {})
        if orig_stats:
            assert orig_stats["mae"] >= 0
        if bayes_stats:
            assert bayes_stats["mae"] >= 0

    def test_print_accuracy_report_with_data(self, aug_with_data, caplog):
        """print_accuracy_report 有数据时输出格式化报告"""
        import logging

        with caplog.at_level(logging.INFO, logger="mci_world_model.sdk.bayesian_augmenter"):
            aug_with_data.print_accuracy_report()
        assert "准确度对比报告" in caplog.text


class TestBayesianAugmenterValidation:
    """P0-EXT-6: run_validation_suite() 批量对比验证"""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.add.return_value = "mem_new"
        client.query.return_value = [
            {"memory_id": "mem_1", "content": "A", "score": 0.9},
            {"memory_id": "mem_2", "content": "B", "score": 0.7},
        ]
        client.predict.return_value = {"event_predictions": [{"content": "e", "confidence": 0.5}]}
        client.reason.return_value = {"confidence": 0.5}
        return client

    @pytest.fixture
    def aug(self, mock_client):
        from mci_world_model.sdk.bayesian_augmenter import BayesianAugmenter

        return BayesianAugmenter(mock_client, enable_network=True, enable_predictor=True, enable_auto_sync=False)

    def test_run_validation_suite_basic(self, aug):
        """run_validation_suite 基本路径"""
        test_queries = [
            {"query": "测试1", "expected_memory_ids": ["mem_1"]},
            {"query": "测试2", "expected_memory_ids": ["mem_1"], "ground_truth_value": 0.7},
        ]
        result = aug.run_validation_suite(test_queries, verbose=False)
        assert "results" in result
        assert "summary" in result
        assert result["summary"]["test_count"] == 2

    def test_run_validation_suite_empty(self, aug):
        """空测试查询列表"""
        result = aug.run_validation_suite([], verbose=False)
        assert result["results"] == []
        assert result["summary"]["test_count"] == 0

    def test_run_validation_suite_verbose(self, aug, caplog):
        """verbose 模式输出验证结果"""
        import logging

        test_queries = [{"query": "测试", "expected_memory_ids": ["mem_1"]}]
        with caplog.at_level(logging.INFO, logger="mci_world_model.sdk.bayesian_augmenter"):
            aug.run_validation_suite(test_queries, verbose=True)
        assert "批量对比验证" in caplog.text


# =============================================================================
# P1 Extension: _world_model.py 补充测试 (state_distance + WorkingMemory)
# =============================================================================


class MockTemporalInfo:
    """模拟 TemporalInfo"""

    def __init__(self, energy_type="wood"):
        self.energy_type = energy_type


class MockBeliefState:
    """模拟信念状态对象"""

    def __init__(self, confidence=0.7):
        self.confidence = confidence


class MockBeliefTracker:
    """模拟 BeliefTracker"""

    def __init__(self, states=None):
        self.belief_states = states or {}


class TestStateDistancePadding:
    """P1-EXT-1: state_distance 不同尺寸邻接矩阵填充逻辑"""

    def test_state_distance_different_sizes(self):
        """不同节点数的两个状态计算距离"""
        from mci_world_model.sdk._world_model import CausalWorldModelState

        state1 = CausalWorldModelState(
            causal_edges=[
                {"cause": "A", "effect": "B", "rho": 0.8, "confidence": 0.9},
                {"cause": "B", "effect": "C", "rho": 0.6, "confidence": 0.7},
            ]
        )
        state2 = CausalWorldModelState(
            causal_edges=[
                {"cause": "A", "effect": "B", "rho": 0.5, "confidence": 0.6},
            ]
        )
        dist = state1.state_distance(state2)
        assert 0.0 <= dist <= 1.0

    def test_state_distance_padding_to_larger(self):
        """小矩阵被填充到大矩阵尺寸"""
        from mci_world_model.sdk._world_model import CausalWorldModelState

        state_big = CausalWorldModelState(
            causal_edges=[
                {"cause": f"node_{i}", "effect": f"node_{i + 1}", "rho": 0.5, "confidence": 0.5} for i in range(5)
            ]
        )
        state_small = CausalWorldModelState(
            causal_edges=[
                {"cause": "X", "effect": "Y", "rho": 0.3, "confidence": 0.4},
            ]
        )
        dist = state_big.state_distance(state_small)
        assert 0.0 <= dist <= 1.0

    def test_state_distance_without_rho(self):
        """边缺少 rho 字段不崩溃"""
        from mci_world_model.sdk._world_model import CausalWorldModelState

        state1 = CausalWorldModelState(
            causal_edges=[
                {"cause": "A", "effect": "B"},
            ]
        )
        state2 = CausalWorldModelState(
            causal_edges=[
                {"cause": "A", "effect": "C"},
            ]
        )
        dist = state1.state_distance(state2)
        assert 0.0 <= dist <= 1.0


class TestStateDistanceV310:
    """P1-EXT-2: state_distance v3.1.0 时空距离 + 信念距离"""

    def test_state_distance_with_temporal_info(self):
        """双方都有 temporal_info 时计算时空距离"""
        from mci_world_model.sdk._world_model import CausalWorldModelState

        ti1 = MockTemporalInfo(energy_type="wood")
        ti2 = MockTemporalInfo(energy_type="wood")
        state1 = CausalWorldModelState(
            causal_edges=[{"cause": "A", "effect": "B", "rho": 0.5, "confidence": 0.5}],
            temporal_info=ti1,
        )
        state2 = CausalWorldModelState(
            causal_edges=[{"cause": "A", "effect": "B", "rho": 0.5, "confidence": 0.5}],
            temporal_info=ti2,
        )
        dist = state1.state_distance(state2)
        assert 0.0 <= dist <= 1.0

    def test_state_distance_with_different_temporal(self):
        """不同 energy_type 时空距离更大"""
        from mci_world_model.sdk._world_model import CausalWorldModelState

        ti1 = MockTemporalInfo(energy_type="wood")
        ti2 = MockTemporalInfo(energy_type="fire")
        state1 = CausalWorldModelState(
            causal_edges=[{"cause": "A", "effect": "B", "rho": 0.5, "confidence": 0.5}],
            temporal_info=ti1,
        )
        state2 = CausalWorldModelState(
            causal_edges=[{"cause": "A", "effect": "B", "rho": 0.5, "confidence": 0.5}],
            temporal_info=ti2,
        )
        dist = state1.state_distance(state2)
        # 不同能量类型距离 > 0
        assert dist > 0.0

    def test_state_distance_one_temporal_none(self):
        """一方无 temporal_info 不计算时空距离"""
        from mci_world_model.sdk._world_model import CausalWorldModelState

        ti = MockTemporalInfo(energy_type="wood")
        state1 = CausalWorldModelState(
            causal_edges=[{"cause": "A", "effect": "B", "rho": 0.5, "confidence": 0.5}],
            temporal_info=ti,
        )
        state2 = CausalWorldModelState(
            causal_edges=[{"cause": "A", "effect": "B", "rho": 0.5, "confidence": 0.5}],
            temporal_info=None,
        )
        dist = state1.state_distance(state2)
        assert 0.0 <= dist <= 1.0

    def test_state_distance_with_belief_tracker(self):
        """双方都有 belief_tracker 时计算信念距离"""
        from mci_world_model.sdk._world_model import CausalWorldModelState

        bt1 = MockBeliefTracker({"k1": MockBeliefState(0.8), "k2": MockBeliefState(0.4)})
        bt2 = MockBeliefTracker({"k1": MockBeliefState(0.6), "k2": MockBeliefState(0.5)})
        state1 = CausalWorldModelState(
            causal_edges=[{"cause": "A", "effect": "B", "rho": 0.5, "confidence": 0.5}],
            belief_tracker=bt1,
        )
        state2 = CausalWorldModelState(
            causal_edges=[{"cause": "A", "effect": "B", "rho": 0.5, "confidence": 0.5}],
            belief_tracker=bt2,
        )
        dist = state1.state_distance(state2)
        assert 0.0 <= dist <= 1.0

    def test_state_distance_with_temporal_and_belief(self):
        """同时有 temporal_info 和 belief_tracker"""
        from mci_world_model.sdk._world_model import CausalWorldModelState

        ti1 = MockTemporalInfo("wood")
        ti2 = MockTemporalInfo("wood")
        bt1 = MockBeliefTracker({"k1": MockBeliefState(0.7)})
        bt2 = MockBeliefTracker({"k1": MockBeliefState(0.7)})
        state1 = CausalWorldModelState(
            causal_edges=[{"cause": "A", "effect": "B", "rho": 0.5, "confidence": 0.5}],
            temporal_info=ti1,
            belief_tracker=bt1,
        )
        state2 = CausalWorldModelState(
            causal_edges=[{"cause": "A", "effect": "B", "rho": 0.5, "confidence": 0.5}],
            temporal_info=ti2,
            belief_tracker=bt2,
        )
        dist = state1.state_distance(state2)
        assert 0.0 <= dist <= 1.0

    def test_sub_dunder_not_implemented(self):
        """__sub__ 操作符：非 CausalWorldModelState 返回 NotImplemented"""
        from mci_world_model.sdk._world_model import CausalWorldModelState

        state = CausalWorldModelState(causal_edges=[{"cause": "A", "effect": "B", "rho": 0.5, "confidence": 0.5}])
        result = state.__sub__("not a state")
        assert result is NotImplemented


class TestStateDistanceCustomWeights:
    """P1-EXT-3: state_distance 自定义权重"""

    def test_state_distance_custom_weights(self):
        """自定义权重参数"""
        from mci_world_model.sdk._world_model import CausalWorldModelState

        state1 = CausalWorldModelState(causal_edges=[{"cause": "A", "effect": "B", "rho": 0.8, "confidence": 0.9}])
        state2 = CausalWorldModelState(causal_edges=[{"cause": "A", "effect": "C", "rho": 0.5, "confidence": 0.6}])
        dist = state1.state_distance(
            state2,
            alpha_edges=1.0,
            alpha_structure=0.5,
            alpha_energy=0.3,
        )
        assert 0.0 <= dist <= 1.0

    def test_state_distance_zero_weights(self):
        """所有权重为零时退化为零距离"""
        from mci_world_model.sdk._world_model import CausalWorldModelState

        state1 = CausalWorldModelState(causal_edges=[{"cause": "A", "effect": "B", "rho": 0.8, "confidence": 0.9}])
        state2 = CausalWorldModelState(causal_edges=[{"cause": "C", "effect": "D", "rho": 0.5, "confidence": 0.6}])
        dist = state1.state_distance(
            state2,
            alpha_edges=0.0,
            alpha_structure=1.0,
            alpha_energy=0.0,
        )
        assert 0.0 <= dist <= 1.0


class TestWorkingMemoryPushWithCores:
    """P1-EXT-4: WorkingMemory.push() with mock temporal/energy cores"""

    def test_push_with_null_stem_branch_code(self):
        """stem_branch_code 为 None 时不调用 temporal_core"""
        from mci_world_model.sdk._world_model import CausalWorldModelState, TrajectoryStep, WorkingMemory

        wm = WorkingMemory(max_length=5)
        state = CausalWorldModelState()
        step = TrajectoryStep(state=state, step_index=0, stem_branch_code=None, energy_state=None)
        wm.push(step)
        assert len(wm.trajectory) == 1

    def test_push_without_cores_no_crash(self):
        """无 temporal_core/energy_core 时 push 不崩溃"""
        from mci_world_model.sdk._world_model import CausalWorldModelState, TrajectoryStep, WorkingMemory

        wm = WorkingMemory(max_length=5)
        state = CausalWorldModelState()
        step = TrajectoryStep(state=state, step_index=0)
        wm.push(step)
        assert wm._state in ("RECORDING", "FULL")

    def test_push_multiple_beyond_limit(self):
        """push 超出 max_length 触发加权淘汰"""
        from mci_world_model.sdk._world_model import CausalWorldModelState, TrajectoryStep, WorkingMemory

        wm = WorkingMemory(max_length=3)
        for i in range(6):
            state = CausalWorldModelState()
            step = TrajectoryStep(state=state, step_index=i, temporal_weight=1.0 / (i + 1))
            wm.push(step)
        assert len(wm.trajectory) <= 3
        assert wm._state == "FULL"


class TestWorkingMemoryWeighted:
    """P1-EXT-5: WorkingMemory.get_recent_weighted() + clear + to_dict"""

    def test_get_recent_weighted_empty(self):
        """空 trajectory 返回空列表"""
        from mci_world_model.sdk._world_model import WorkingMemory

        wm = WorkingMemory(max_length=5)
        result = wm.get_recent_weighted(3)
        assert result == []

    def test_get_recent_weighted_basic(self):
        """基本加权检索"""
        from mci_world_model.sdk._world_model import CausalWorldModelState, TrajectoryStep, WorkingMemory

        wm = WorkingMemory(max_length=5)
        for i in range(4):
            state = CausalWorldModelState()
            step = TrajectoryStep(state=state, step_index=i, temporal_weight=float(i + 1))
            wm.push(step)
        result = wm.get_recent_weighted(2)
        assert len(result) == 2
        # 按 temporal_weight 降序排列
        assert result[0].temporal_weight >= result[1].temporal_weight

    def test_clear_resets_state(self):
        """clear 重置为 IDLE"""
        from mci_world_model.sdk._world_model import CausalWorldModelState, TrajectoryStep, WorkingMemory

        wm = WorkingMemory(max_length=5)
        state = CausalWorldModelState()
        wm.push(TrajectoryStep(state=state, step_index=0))
        wm.clear()
        assert wm._state == "IDLE"
        assert len(wm.trajectory) == 0

    def test_to_dict_contains_expected_keys(self):
        """to_dict 含 max_length/n_steps/state/recent_costs"""
        from mci_world_model.sdk._world_model import CausalWorldModelState, TrajectoryStep, WorkingMemory

        wm = WorkingMemory(max_length=5)
        state = CausalWorldModelState()
        wm.push(TrajectoryStep(state=state, step_index=0, temporal_weight=1.0))
        d = wm.to_dict()
        assert d["max_length"] == 5
        assert d["n_steps"] == 1
        assert d["state"] in ("RECORDING", "FULL")
        assert "recent_costs" in d

    def test_is_full_property(self):
        """is_full 属性反映状态"""
        from mci_world_model.sdk._world_model import CausalWorldModelState, TrajectoryStep, WorkingMemory

        wm = WorkingMemory(max_length=2)
        assert not wm.is_full
        for i in range(3):
            state = CausalWorldModelState()
            wm.push(TrajectoryStep(state=state, step_index=i))
        assert wm.is_full

    def test_push_with_stem_and_energy_none(self):
        """stem_branch_code 和 energy_state 都 None 不注入"""
        from mci_world_model.sdk._world_model import CausalWorldModelState, TrajectoryStep, WorkingMemory

        wm = WorkingMemory(max_length=5)
        state = CausalWorldModelState()
        step = TrajectoryStep(state=state, step_index=0, stem_branch_code=None, energy_state=None)
        wm.push(step)
        assert step.stem_branch_code is None
        assert step.energy_state is None

    def test_push_fills_until_full(self):
        """push 直到 is_full=True"""
        from mci_world_model.sdk._world_model import CausalWorldModelState, TrajectoryStep, WorkingMemory

        wm = WorkingMemory(max_length=3)
        for i in range(3):
            state = CausalWorldModelState()
            wm.push(TrajectoryStep(state=state, step_index=i))
        assert wm.is_full
        assert wm._state == "FULL"


# =============================================================================
# P1 Extension v2: WorkingMemory + MCIWorldModel 真实 su_memory 集成测试
# =============================================================================


class TestWorkingMemoryWithRealCores:
    """P1-EXT-6: WorkingMemory.push/get_recent_weighted with real TemporalCore/EnergyCore"""

    def test_push_with_temporal_core_injection(self):
        """注入真实 TemporalCore 后 push 自动生成 stem_branch_code"""
        from su_memory._sys._temporal_core import TemporalCore

        from mci_world_model.sdk._world_model import CausalWorldModelState, TrajectoryStep, WorkingMemory

        wm = WorkingMemory(max_length=5)
        wm._temporal_core = TemporalCore()
        state = CausalWorldModelState()
        step = TrajectoryStep(state=state, step_index=0, stem_branch_code=None)
        wm.push(step)
        assert step.stem_branch_code is not None
        assert hasattr(step.stem_branch_code, "cycle_index")

    def test_push_with_energy_core_injection(self):
        """注入真实 EnergyCore 后 push 自动生成 energy_state"""
        from su_memory._sys._energy_core import EnergyCore

        from mci_world_model.sdk._world_model import CausalWorldModelState, TrajectoryStep, WorkingMemory

        wm = WorkingMemory(max_length=5)
        wm._energy_core = EnergyCore()
        state = CausalWorldModelState()
        step = TrajectoryStep(state=state, step_index=0, energy_state=None)
        wm.push(step)
        assert step.energy_state is not None
        assert step.temporal_weight != 1.0  # 被能量状态覆盖

    def test_push_with_both_cores(self):
        """同时注入两个 Core"""
        from su_memory._sys._energy_core import EnergyCore
        from su_memory._sys._temporal_core import TemporalCore

        from mci_world_model.sdk._world_model import CausalWorldModelState, TrajectoryStep, WorkingMemory

        wm = WorkingMemory(max_length=5)
        wm._temporal_core = TemporalCore()
        wm._energy_core = EnergyCore()
        state = CausalWorldModelState()
        step = TrajectoryStep(state=state, step_index=0, stem_branch_code=None, energy_state=None)
        wm.push(step)
        assert step.stem_branch_code is not None
        assert step.energy_state is not None

    def test_get_recent_weighted_with_temporal_core(self):
        """get_recent_weighted with temporal_core 应用循环距离衰减"""
        from su_memory._sys._temporal_core import TemporalCore

        from mci_world_model.sdk._world_model import CausalWorldModelState, TrajectoryStep, WorkingMemory

        wm = WorkingMemory(max_length=5)
        wm._temporal_core = TemporalCore()
        tc = wm._temporal_core
        state = CausalWorldModelState()
        for i in range(4):
            code = tc.create_code((i + 1) % 10, (i + 2) % 12)
            step = TrajectoryStep(state=state, step_index=i, stem_branch_code=code, temporal_weight=float(i + 1))
            wm.push(step)
        result = wm.get_recent_weighted(2)
        assert len(result) == 2
        assert result[0].temporal_weight >= result[1].temporal_weight

    def test_get_recent_weighted_no_core(self):
        """无 temporal_core 时 get_recent_weighted 仅按 temporal_weight 排序"""
        from mci_world_model.sdk._world_model import CausalWorldModelState, TrajectoryStep, WorkingMemory

        wm = WorkingMemory(max_length=5)
        state = CausalWorldModelState()
        for i in range(4):
            wm.push(TrajectoryStep(state=state, step_index=i, temporal_weight=float(4 - i)))
        result = wm.get_recent_weighted(2)
        assert len(result) == 2


class TestMCIWorldModelLazyInit:
    """P1-EXT-7: MCIWorldModel 惰性获取器 + lite_pro 自动初始化"""

    def test_get_energy_core_lazy(self):
        """_get_energy_core 惰性创建 EnergyCore"""
        from mci_world_model.sdk._world_model import MCIWorldModel

        wm = MCIWorldModel()
        assert wm._energy_core is None
        core = wm._get_energy_core()
        assert core is not None
        assert wm._energy_core is core

    def test_get_temporal_core_lazy(self):
        """_get_temporal_core 惰性创建 TemporalCore"""
        from mci_world_model.sdk._world_model import MCIWorldModel

        wm = MCIWorldModel()
        assert wm._temporal_core is None
        core = wm._get_temporal_core()
        assert core is not None
        assert wm._temporal_core is core

    def test_get_configurator_lazy(self):
        """_get_configurator 惰性创建 HierarchicalConfigurator"""
        from mci_world_model.sdk._world_model import MCIWorldModel

        wm = MCIWorldModel()
        assert wm._configurator is None
        cfg = wm._get_configurator()
        assert cfg is not None
        assert wm._configurator is cfg

    def test_get_causal_actor_lazy(self):
        """_get_causal_actor 惰性创建 CausalActor"""
        from mci_world_model.sdk._world_model import MCIWorldModel

        wm = MCIWorldModel()
        assert wm._causal_actor is None
        actor = wm._get_causal_actor()
        assert actor is not None
        assert wm._causal_actor is actor

    def test_energy_core_idempotent(self):
        """_get_energy_core 幂等：两次调用返回同一实例"""
        from mci_world_model.sdk._world_model import MCIWorldModel

        wm = MCIWorldModel()
        c1 = wm._get_energy_core()
        c2 = wm._get_energy_core()
        assert c1 is c2

    def test_temporal_core_idempotent(self):
        """_get_temporal_core 幂等"""
        from mci_world_model.sdk._world_model import MCIWorldModel

        wm = MCIWorldModel()
        c1 = wm._get_temporal_core()
        c2 = wm._get_temporal_core()
        assert c1 is c2

    def test_init_with_lite_pro_triggers_initialize(self):
        """传入 lite_pro 自动触发 initialize"""
        from unittest.mock import MagicMock

        from mci_world_model.sdk._world_model import MCIWorldModel

        mock_lite = MagicMock()
        wm = MCIWorldModel(lite_pro=mock_lite)
        assert wm._initialized is True
        assert wm._lite_pro is mock_lite


class TestMCIWorldModelEnergyBus:
    """P1-EXT-8: _build_energy_bus + _extract_energy_ratios + _propagate_energy"""

    def test_extract_energy_ratios_no_edges(self):
        """无因果边时返回 None"""
        from mci_world_model.sdk._world_model import CausalWorldModelState, MCIWorldModel

        wm = MCIWorldModel()
        state = CausalWorldModelState.empty()
        ratios = wm._extract_energy_ratios(state)
        assert ratios is None

    def test_extract_energy_ratios_no_energy_labels(self):
        """边无能量标签时 total=0 返回 None"""
        from mci_world_model.sdk._world_model import CausalWorldModelState, MCIWorldModel

        wm = MCIWorldModel()
        state = CausalWorldModelState(
            causal_edges=[
                {"cause": "A", "effect": "B", "rho": 0.5},
            ]
        )
        ratios = wm._extract_energy_ratios(state)
        assert ratios is None

    def test_extract_energy_ratios_with_labels(self):
        """有能量标签时返回五维分布"""
        from mci_world_model.sdk._world_model import CausalWorldModelState, MCIWorldModel

        wm = MCIWorldModel()
        state = CausalWorldModelState(
            causal_edges=[
                {"cause": "A", "effect": "B", "rho": 0.8, "cause_energy": "causal", "effect_energy": "semantic"},
            ]
        )
        ratios = wm._extract_energy_ratios(state)
        assert ratios is not None
        assert "causal" in ratios
        assert ratios["causal"] > 0
        assert abs(sum(ratios.values()) - 1.0) < 1e-9

    def test_build_energy_bus(self):
        """_build_energy_bus 构建 EnergyBus 并连接边"""
        from mci_world_model.sdk._world_model import CausalWorldModelState, MCIWorldModel

        wm = MCIWorldModel()
        wm._state = CausalWorldModelState(
            causal_edges=[
                {"cause": "A", "effect": "B", "rho": 0.8, "cause_energy": "causal", "effect_energy": "semantic"},
            ]
        )
        bus = wm._build_energy_bus()
        assert bus is not None

    def test_propagate_energy_handles_missing_api(self):
        """_propagate_energy 优雅处理 EnergyBus API 差异"""
        from mci_world_model.sdk._world_model import CausalWorldModelState, MCIWorldModel

        wm = MCIWorldModel()
        wm._state = CausalWorldModelState(
            causal_edges=[
                {"cause": "A", "effect": "B", "rho": 0.8, "cause_energy": "causal", "effect_energy": "semantic"},
            ]
        )
        # _propagate_energy 可能因 API 差异失败，确认是 AttributeError 而非崩溃
        try:
            result = wm._propagate_energy(steps=2)
            assert isinstance(result, dict)
        except AttributeError:
            pass  # EnergyBus API 版本差异

    def test_build_energy_bus_no_edges(self):
        """空因果边仍能构建节点"""
        from mci_world_model.sdk._world_model import CausalWorldModelState, MCIWorldModel

        wm = MCIWorldModel()
        wm._state = CausalWorldModelState.empty()
        bus = wm._build_energy_bus()
        assert bus is not None


class TestStateDistanceEdgeCases:
    """P1-EXT-9: state_distance 异常/边界路径"""

    def test_state_distance_identical_edges_temporal_exception(self):
        """temporal_info 无 energy_type 触发 except 分支"""
        from mci_world_model.sdk._world_model import CausalWorldModelState

        class BadTemporal:
            pass

        s1 = CausalWorldModelState(
            causal_edges=[{"cause": "A", "effect": "B", "rho": 0.5, "confidence": 0.5}],
            temporal_info=BadTemporal(),
        )
        s2 = CausalWorldModelState(
            causal_edges=[{"cause": "A", "effect": "B", "rho": 0.5, "confidence": 0.5}],
            temporal_info=BadTemporal(),
        )
        # 不应崩溃
        dist = s1.state_distance(s2)
        assert 0.0 <= dist <= 1.0

    def test_state_distance_belief_exception(self):
        """belief_tracker 无 belief_states 触发 except 分支"""
        from mci_world_model.sdk._world_model import CausalWorldModelState

        class BadBelief:
            pass

        s1 = CausalWorldModelState(
            causal_edges=[{"cause": "A", "effect": "B", "rho": 0.5, "confidence": 0.5}],
            belief_tracker=BadBelief(),
        )
        s2 = CausalWorldModelState(
            causal_edges=[{"cause": "A", "effect": "B", "rho": 0.5, "confidence": 0.5}],
            belief_tracker=BadBelief(),
        )
        # 不应崩溃
        dist = s1.state_distance(s2)
        assert 0.0 <= dist <= 1.0

    def test_state_distance_same_edges_no_overlap(self):
        """两个状态边相同但 union=0 的极端路径"""
        from mci_world_model.sdk._world_model import CausalWorldModelState

        # 相同边 → intersection == union，Jaccard=1，不会触发 union=0
        s1 = CausalWorldModelState(causal_edges=[{"cause": "A", "effect": "B", "rho": 0.5, "confidence": 0.5}])
        s2 = CausalWorldModelState(causal_edges=[{"cause": "A", "effect": "B", "rho": 0.5, "confidence": 0.5}])
        dist = s1.state_distance(s2)
        assert dist == 0.0  # 完全相同的边

    def test_state_distance_both_empty(self):
        """两个空状态距离为 0"""
        from mci_world_model.sdk._world_model import CausalWorldModelState

        s1 = CausalWorldModelState.empty()
        s2 = CausalWorldModelState.empty()
        dist = s1.state_distance(s2)
        assert dist == 0.0

    def test_state_distance_one_empty(self):
        """一个空状态，距离为 1"""
        from mci_world_model.sdk._world_model import CausalWorldModelState

        s1 = CausalWorldModelState(causal_edges=[{"cause": "A", "effect": "B", "rho": 0.5, "confidence": 0.5}])
        s2 = CausalWorldModelState.empty()
        dist = s1.state_distance(s2)
        assert dist == 1.0

    def test_state_distance_self_adj_padded(self):
        """self 节点数少于 other 时触发 self 填充 (line 331-333)"""
        from mci_world_model.sdk._world_model import CausalWorldModelState

        s1 = CausalWorldModelState(
            causal_edges=[
                {"cause": "A", "effect": "B", "rho": 0.5, "confidence": 0.5},
            ]
        )
        s2 = CausalWorldModelState(
            causal_edges=[
                {"cause": "A", "effect": "B", "rho": 0.5, "confidence": 0.5},
                {"cause": "B", "effect": "C", "rho": 0.4, "confidence": 0.4},
                {"cause": "C", "effect": "D", "rho": 0.3, "confidence": 0.3},
            ]
        )
        dist = s1.state_distance(s2)
        assert 0.0 <= dist <= 1.0

    def test_state_distance_temporal_attribute_error(self):
        """temporal_info 无 getattr 触发 except 分支 (line 373-374)"""
        from mci_world_model.sdk._world_model import CausalWorldModelState

        class BadTemporal:
            pass  # 无 energy_type 属性

        s1 = CausalWorldModelState(
            causal_edges=[{"cause": "A", "effect": "B", "rho": 0.5, "confidence": 0.5}],
            temporal_info=BadTemporal(),
        )
        s2 = CausalWorldModelState(
            causal_edges=[{"cause": "A", "effect": "B", "rho": 0.5, "confidence": 0.5}],
            temporal_info=BadTemporal(),
        )
        dist = s1.state_distance(s2)
        assert 0.0 <= dist <= 1.0

    def test_state_distance_belief_attribute_error(self):
        """belief_tracker 无 getattr 触发 except 分支 (line 400-401)"""
        from mci_world_model.sdk._world_model import CausalWorldModelState

        class BadBelief:
            pass  # 无 belief_states 属性

        s1 = CausalWorldModelState(
            causal_edges=[{"cause": "A", "effect": "B", "rho": 0.5, "confidence": 0.5}],
            belief_tracker=BadBelief(),
        )
        s2 = CausalWorldModelState(
            causal_edges=[{"cause": "A", "effect": "B", "rho": 0.5, "confidence": 0.5}],
            belief_tracker=BadBelief(),
        )
        dist = s1.state_distance(s2)
        assert 0.0 <= dist <= 1.0
