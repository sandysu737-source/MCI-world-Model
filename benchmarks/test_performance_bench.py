"""
MCI World Model — 核心推理路径性能基准测试

使用 time.perf_counter 自测，零外部依赖。
"""

from __future__ import annotations

import time

import numpy as np
import pytest


def _bench(fn, *args, rounds=5, **kwargs):
    """Time a function over multiple rounds, return best-case ms."""
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        _result = fn(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1000)
    return min(times)


# ═════════════════════════════════════════════════════════════════════════════
# T1: JEPAEncoder encode 延迟
# ═════════════════════════════════════════════════════════════════════════════

class TestJEPAEncodeLatency:

    def test_jepa_encode_30d_patient(self, patient_timeline_30d):
        from mci_world_model.sdk._jepa_encoder import JEPAEncoder
        encoder = JEPAEncoder(world_model=None)
        ms = _bench(encoder.encode, signals=patient_timeline_30d, rounds=5)
        result = encoder.encode(signals=patient_timeline_30d)
        assert result is not None
        assert len(result.causal_edges) > 0
        assert ms < 500, f"JEPA encode 30d: {ms:.1f}ms > 500ms"

    def test_jepa_encode_small_window(self, patient_timeline_30d):
        from mci_world_model.sdk._jepa_encoder import JEPAEncoder
        encoder = JEPAEncoder(world_model=None)
        window = patient_timeline_30d[:10]
        ms = _bench(encoder.encode, signals=window, rounds=5)
        result = encoder.encode(signals=window)
        assert result is not None
        assert ms < 500, f"JEPA encode small: {ms:.1f}ms > 500ms"


# ═════════════════════════════════════════════════════════════════════════════
# T2: JEPAPredictor predict 延迟
# ═════════════════════════════════════════════════════════════════════════════

class TestJEPAPredictLatency:

    @pytest.fixture(scope="class")
    def encoded_state(self, patient_timeline_30d):
        from mci_world_model.sdk._jepa_encoder import JEPAEncoder
        encoder = JEPAEncoder(world_model=None)
        return encoder.encode(signals=patient_timeline_30d)

    def test_identity_predictor(self, encoded_state):
        from mci_world_model.sdk._jepa_predictor import IdentityPredictor
        predictor = IdentityPredictor()
        ms = _bench(predictor.predict, encoded_state, rounds=5)
        result = predictor.predict(encoded_state)
        assert result is not None
        assert ms < 100, f"IdentityPredictor: {ms:.1f}ms > 100ms"

    def test_belief_propagation_predictor(self, encoded_state):
        from mci_world_model.sdk._jepa_predictor import BeliefPropagationPredictor
        predictor = BeliefPropagationPredictor()
        ms = _bench(predictor.predict, encoded_state, rounds=5)
        result = predictor.predict(encoded_state)
        assert result is not None
        assert ms < 500, f"BeliefPropagation: {ms:.1f}ms > 500ms"


# ═════════════════════════════════════════════════════════════════════════════
# T3: CounterfactualEngine 单次查询延迟
# ═════════════════════════════════════════════════════════════════════════════

class TestCounterfactualSingleLatency:

    def test_cf_single_query(self, counterfactual_engine):
        evidence = {
            "calorie_intake": 1500.0, "albumin": 35.0, "protein_intake": 65.0,
            "medication_dose": 200.0, "nrs2002_score": 3.0, "prealbumin": 250.0,
            "body_weight": 70.0,
        }
        ms = _bench(
            counterfactual_engine.query, rounds=10,
            evidence=evidence, do_x={"calorie_intake": 2000.0}, target="albumin",
        )
        result = counterfactual_engine.query(
            evidence=evidence, do_x={"calorie_intake": 2000.0}, target="albumin",
        )
        assert result is not None
        assert result.counterfactual_value is not None
        assert ms < 200, f"CF single query: {ms:.1f}ms > 200ms"


# ═════════════════════════════════════════════════════════════════════════════
# T4: BatchCounterfactualEngine 批量查询延迟
# ═════════════════════════════════════════════════════════════════════════════

class TestCounterfactualBatchLatency:

    @pytest.fixture(scope="class")
    def batch_engine(self):
        from mci_world_model.sdk._batch_counterfactual import BatchCounterfactualEngine, StructuralEquationModel
        sem = StructuralEquationModel(
            coefficients=np.array([
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ], dtype=np.float64),
            node_names=["calorie_intake", "albumin", "protein",
                        "nrs2002", "body_weight"])
        return BatchCounterfactualEngine(sem)

    @pytest.fixture(scope="class")
    def batch_scenarios(self):
        return [
            {"do_x": {"calorie_intake": v}, "target": "albumin"}
            for v in np.linspace(1200, 2800, 100)
        ]

    def test_cf_batch_100(self, batch_engine, batch_scenarios):
        ms = _bench(batch_engine.batch_query, batch_scenarios, n_mc=200, rounds=3)
        results = batch_engine.batch_query(batch_scenarios, n_mc=200)
        assert len(results) == 100
        assert ms < 5000, f"CF batch 100: {ms:.1f}ms > 5000ms"


# ═════════════════════════════════════════════════════════════════════════════
# T5: PhysicalGraphBuilder 构建延迟
# ═════════════════════════════════════════════════════════════════════════════

class TestPhysicalGraphLatency:

    def test_build_graph_30d(self, patient_timeline_30d):
        from mci_world_model.sdk._physical_graph_builder import PhysicalGraphBuilder
        builder = PhysicalGraphBuilder()
        timeline = patient_timeline_30d
        ms = _bench(builder.build_graph, timeline, rounds=5)
        edges = builder.build_graph(timeline)
        assert isinstance(edges, list)
        assert ms < 500, f"Build graph 30d: {ms:.1f}ms > 500ms"

    def test_build_graph_7d(self, patient_timeline_30d):
        from mci_world_model.sdk._physical_graph_builder import PhysicalGraphBuilder
        builder = PhysicalGraphBuilder()
        ms = _bench(builder.build_graph, patient_timeline_30d[:7], rounds=5)
        edges = builder.build_graph(patient_timeline_30d[:7])
        assert isinstance(edges, list)
        assert ms < 200, f"Build graph 7d: {ms:.1f}ms > 200ms"


# ═════════════════════════════════════════════════════════════════════════════
# T6: PerceptionPipeline 多模态处理延迟
# ═════════════════════════════════════════════════════════════════════════════

class TestPerceptionPipelineLatency:

    @pytest.mark.skip(reason="_perception_pipeline module not yet built")
    def test_process_120_signals(self, multimodal_signals):
        pass  # placeholder for future PerceptionPipeline


# ═════════════════════════════════════════════════════════════════════════════
# T7: EnergyCore 能量模拟延迟
# ═════════════════════════════════════════════════════════════════════════════

class TestEnergyCoreLatency:

    def test_analyze_balance_5_elements(self):
        from mci_world_model._sys._energy_core import EnergyCore, EnergyType
        core = EnergyCore()
        energies = {
            EnergyType.WOOD: 1.2, EnergyType.FIRE: 0.8, EnergyType.EARTH: 1.0,
            EnergyType.METAL: 1.5, EnergyType.WATER: 0.9,
        }
        ms = _bench(core.analyze_balance, energies, rounds=10)
        result = core.analyze_balance(energies)
        assert result is not None
        assert ms < 100, f"EnergyCore balance: {ms:.1f}ms > 100ms"

    def test_simulate_energy_flow_10_steps(self):
        from mci_world_model._sys._energy_core import EnergyCore, EnergyType
        core = EnergyCore()
        initial = {
            EnergyType.WOOD: 1.0, EnergyType.FIRE: 1.0, EnergyType.EARTH: 1.0,
            EnergyType.METAL: 1.0, EnergyType.WATER: 1.0,
        }
        ms = _bench(core.simulate_energy_flow, initial, steps=10, rounds=10)
        result = core.simulate_energy_flow(initial, steps=10)
        assert len(result) > 0
        assert ms < 100, f"EnergyCore flow 10: {ms:.1f}ms > 100ms"

    def test_get_energy_state(self):
        from mci_world_model._sys._energy_core import EnergyCore
        core = EnergyCore()
        ms = _bench(core.get_energy_state, "WOOD", 1, rounds=10)
        result = core.get_energy_state("WOOD", 1)
        assert result is not None
        assert ms < 50, f"EnergyCore get_state: {ms:.1f}ms > 50ms"


# ═════════════════════════════════════════════════════════════════════════════
# T8: CausalChain 因果链传播延迟
# ═════════════════════════════════════════════════════════════════════════════

class TestCausalChainLatency:

    @pytest.fixture(scope="class")
    def large_causal_chain(self):
        from mci_world_model._sys.causal import CausalChain
        cc = CausalChain()
        energy_types = ["wood", "fire", "earth", "metal", "water"]
        for i in range(100):
            cc.add(f"n{i}", energy_type=energy_types[i % 5])
        for i in range(99):
            cc.link(f"n{i}", f"n{i + 1}")
        return cc

    def test_propagate_100_nodes(self, large_causal_chain):
        ms = _bench(large_causal_chain.propagate, "n0", delta=0.1, rounds=10)
        result = large_causal_chain.propagate("n0", delta=0.1)
        assert len(result) > 0
        assert ms < 200, f"CausalChain propagate 100: {ms:.1f}ms > 200ms"

    def test_coverage_100_nodes(self, large_causal_chain):
        all_ids = [f"n{i}" for i in range(100)]
        ms = _bench(large_causal_chain.coverage, all_ids, rounds=10)
        cov = large_causal_chain.coverage(all_ids)
        assert 0 <= cov <= 100
        assert ms < 100, f"CausalChain coverage 100: {ms:.1f}ms > 100ms"

    def test_get_causal_path(self, large_causal_chain):
        ms = _bench(large_causal_chain.get_causal_path, "n0", "n50", rounds=10)
        path = large_causal_chain.get_causal_path("n0", "n50")
        assert len(path) > 0
        assert ms < 50, f"CausalChain path: {ms:.1f}ms > 50ms"


# ═════════════════════════════════════════════════════════════════════════════
# T9: EvidenceCollector 证据处理延迟
# ═════════════════════════════════════════════════════════════════════════════

class TestEvidenceCollectorLatency:

    @pytest.fixture(scope="class")
    def populated_collector(self):
        from mci_world_model._sys.evidence import EvidenceCollector
        ec = EvidenceCollector()
        ec.register_source("lab_report", initial_reliability=0.9)
        ec.register_source("clinical_note", initial_reliability=0.75)
        ec.register_source("patient_report", initial_reliability=0.6)
        for i in range(50):
            source = "lab_report" if i % 3 == 0 else ("clinical_note" if i % 3 == 1 else "patient_report")
            ec.collect(
                belief_id=f"belief_{i % 10}", is_positive=(i % 2 == 0),
                source=source, weight=0.5 + (i % 50) / 100,
            )
        return ec

    def test_detect_conflicts(self, populated_collector):
        ms = _bench(populated_collector.detect_evidence_conflicts, "belief_0", rounds=10)
        conflicts = populated_collector.detect_evidence_conflicts("belief_0")
        assert isinstance(conflicts, list)
        assert ms < 100, f"Evidence conflicts: {ms:.1f}ms > 100ms"

    def test_evidence_strength(self, populated_collector):
        ms = _bench(populated_collector.compute_evidence_strength, "belief_0", rounds=10)
        strength = populated_collector.compute_evidence_strength("belief_0")
        assert isinstance(strength, dict)
        assert ms < 50, f"Evidence strength: {ms:.1f}ms > 50ms"
