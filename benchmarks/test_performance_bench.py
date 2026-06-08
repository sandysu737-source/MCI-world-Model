"""
MCI World Model P1 — 核心推理路径性能基准测试

6 项基准覆盖:
- JEPAEncoder encode + predict
- CounterfactualEngine 单次/批量查询
- PhysicalGraphBuilder 构建
- PerceptionPipeline 多模态处理

运行: pytest benchmarks/test_performance_bench.py -m benchmark --benchmark-only

v2: 新增 _sys/ 核心模块性能基线:
- T7: EnergyCore 能量模拟
- T8: CausalChain 因果链传播
- T9: EvidenceCollector 证据处理
"""

from __future__ import annotations

import numpy as np
import pytest


# =============================================================================
# T1: JEPAEncoder encode 延迟
# =============================================================================


@pytest.mark.benchmark(min_rounds=5, max_time=2.0)
class TestJEPAEncodeLatency:
    """JEPAEncoder.encode(signals=...) 物理信号编码性能。"""

    def test_jepa_encode_30d_patient(self, benchmark, patient_timeline_30d):
        """30 天 × 7 物理量 → CausalWorldModelState。"""
        from mci_world_model.sdk._jepa_encoder import JEPAEncoder

        encoder = JEPAEncoder(world_model=None)
        timeline = patient_timeline_30d

        result = benchmark(encoder.encode, signals=timeline)
        assert result is not None
        assert len(result.causal_edges) > 0

    def test_jepa_encode_small_window(self, benchmark, patient_timeline_30d):
        """10 天窗口编码（JEPA 训练常见场景）。"""
        from mci_world_model.sdk._jepa_encoder import JEPAEncoder

        encoder = JEPAEncoder(world_model=None)
        window = patient_timeline_30d[:10]

        result = benchmark(encoder.encode, signals=window)
        assert result is not None


# =============================================================================
# T2: JEPAPredictor predict 延迟
# =============================================================================


@pytest.mark.benchmark(min_rounds=5, max_time=2.0)
class TestJEPAPredictLatency:
    """JEPAPredictor.predict() 单次预测性能。"""

    @pytest.fixture(scope="class")
    def encoded_state(self, patient_timeline_30d):
        from mci_world_model.sdk._jepa_encoder import JEPAEncoder

        encoder = JEPAEncoder(world_model=None)
        return encoder.encode(signals=patient_timeline_30d)

    def test_identity_predictor(self, benchmark, encoded_state):
        """IdentityPredictor 预测延迟。"""
        from mci_world_model.sdk._jepa_predictor import IdentityPredictor

        predictor = IdentityPredictor()

        result = benchmark(predictor.predict, encoded_state)
        assert result is not None

    def test_belief_propagation_predictor(self, benchmark, encoded_state):
        """BeliefPropagationPredictor 预测延迟。"""
        from mci_world_model.sdk._jepa_predictor import BeliefPropagationPredictor

        predictor = BeliefPropagationPredictor()

        result = benchmark(predictor.predict, encoded_state)
        assert result is not None


# =============================================================================
# T3: CounterfactualEngine 单次查询延迟
# =============================================================================


@pytest.mark.benchmark(min_rounds=10, max_time=2.0)
class TestCounterfactualSingleLatency:
    """CounterfactualEngine.query() 单次反事实查询性能。"""

    def test_cf_single_query(self, benchmark, counterfactual_engine):
        """单次 do(calorie_intake=2000) → albumin。"""
        evidence = {
            "calorie_intake": 1500.0,
            "albumin": 35.0,
            "protein_intake": 65.0,
            "medication_dose": 200.0,
            "nrs2002_score": 3.0,
            "prealbumin": 250.0,
            "body_weight": 70.0,
        }

        result = benchmark(
            counterfactual_engine.query,
            evidence=evidence,
            do_x={"calorie_intake": 2000.0},
            target="albumin",
        )
        assert result is not None
        assert result.counterfactual_value is not None


# =============================================================================
# T4: BatchCounterfactualEngine 批量查询延迟
# =============================================================================


@pytest.mark.benchmark(min_rounds=3, max_time=5.0)
class TestCounterfactualBatchLatency:
    """BatchCounterfactualEngine.batch_query() 批量反事实性能。"""

    @pytest.fixture(scope="class")
    def batch_engine(self):
        from mci_world_model.sdk._batch_counterfactual import BatchCounterfactualEngine
        from mci_world_model.sdk._do_calculus import CausalGraph

        cg = CausalGraph()
        cg.add_edge("calorie_intake", "albumin", weight=0.6)
        cg.add_edge("medication_dose", "albumin", weight=0.4)
        cg.add_edge("protein_intake", "prealbumin", weight=0.5)
        cg.add_edge("albumin", "nrs2002_score", weight=-0.3)
        cg.add_edge("calorie_intake", "body_weight", weight=0.35)

        sem = cg.to_sem(noise_std=0.2, activation="linear", seed=42)
        return BatchCounterfactualEngine(sem)

    @pytest.fixture(scope="class")
    def batch_scenarios(self):
        rng = np.random.default_rng(42)
        scenarios = []
        for _ in range(100):
            base_cal = rng.uniform(1000, 2000)
            evidence = {
                "calorie_intake": round(base_cal, 0),
                "albumin": round(rng.uniform(28, 42), 1),
                "protein_intake": round(rng.uniform(50, 90), 1),
                "medication_dose": round(rng.uniform(100, 300), 1),
                "nrs2002_score": round(rng.uniform(2, 5), 1),
                "prealbumin": round(rng.uniform(150, 350), 1),
                "body_weight": round(rng.uniform(50, 90), 1),
            }
            scenarios.append({
                "evidence": evidence,
                "do_x": {"calorie_intake": round(base_cal + 500, 0)},
                "target": "albumin",
            })
        return scenarios

    def test_cf_batch_100(self, benchmark, batch_engine, batch_scenarios):
        """100 条批量反事实查询。"""
        results = benchmark(batch_engine.batch_query, batch_scenarios, n_mc=200)
        assert len(results) == 100


# =============================================================================
# T5: PhysicalGraphBuilder 构建延迟
# =============================================================================


@pytest.mark.benchmark(min_rounds=5, max_time=2.0)
class TestPhysicalGraphLatency:
    """PhysicalGraphBuilder.build_graph() 因果图构建性能。"""

    def test_build_graph_30d(self, benchmark, patient_timeline_30d):
        """30 天 × 7 物理量 → causal_edges。"""
        from mci_world_model.sdk._physical_graph_builder import PhysicalGraphBuilder

        builder = PhysicalGraphBuilder()
        timeline = patient_timeline_30d

        edges = benchmark(builder.build_graph, timeline)
        assert len(edges) >= 20  # 至少 20 条因果边

    def test_build_graph_7d(self, benchmark, patient_timeline_30d):
        """7 天最小窗口。"""
        from mci_world_model.sdk._physical_graph_builder import PhysicalGraphBuilder

        builder = PhysicalGraphBuilder()
        window = patient_timeline_30d[:7]

        edges = benchmark(builder.build_graph, window)
        assert isinstance(edges, list)


# =============================================================================
# T6: PerceptionPipeline 多模态处理延迟
# =============================================================================


@pytest.mark.benchmark(min_rounds=5, max_time=2.0)
class TestPerceptionPipelineLatency:
    """PerceptionPipeline.process_multimodal() 多模态信号处理性能。"""

    def test_process_120_signals(self, benchmark, multimodal_signals):
        """120 个 MultimodalSignal（30天×4物理量）。"""
        from mci_world_model._sys._perception_pipeline import PerceptionPipeline

        pipeline = PerceptionPipeline()

        features = benchmark(pipeline.process_multimodal, multimodal_signals)
        assert len(features) >= 30


# =============================================================================
# T7: EnergyCore 能量模拟延迟
# =============================================================================


@pytest.mark.benchmark(min_rounds=10, max_time=2.0)
class TestEnergyCoreLatency:
    """EnergyCore 五行能量引擎性能基线。"""

    def test_analyze_balance_5_elements(self, benchmark):
        """5 元素平衡分析延迟。"""
        from mci_world_model._sys._energy_core import EnergyCore, EnergyType

        core = EnergyCore()
        energies = {
            EnergyType.WOOD: 1.2,
            EnergyType.FIRE: 0.8,
            EnergyType.EARTH: 1.0,
            EnergyType.METAL: 1.5,
            EnergyType.WATER: 0.9,
        }

        result = benchmark(core.analyze_balance, energies)
        assert result is not None

    def test_simulate_energy_flow_10_steps(self, benchmark):
        """10 步能量流转模拟延迟。"""
        from mci_world_model._sys._energy_core import EnergyCore, EnergyType

        core = EnergyCore()
        initial = {
            EnergyType.WOOD: 1.0,
            EnergyType.FIRE: 1.0,
            EnergyType.EARTH: 1.0,
            EnergyType.METAL: 1.0,
            EnergyType.WATER: 1.0,
        }

        result = benchmark(core.simulate_energy_flow, initial, steps=10)
        assert len(result) > 0

    def test_get_energy_state(self, benchmark):
        """单元素状态判定延迟。"""
        from mci_world_model._sys._energy_core import EnergyCore, EnergyType

        core = EnergyCore()
        energies = {
            EnergyType.WOOD: 2.0,
            EnergyType.FIRE: 0.5,
            EnergyType.EARTH: 1.0,
            EnergyType.METAL: 0.8,
            EnergyType.WATER: 1.2,
        }

        result = benchmark(core.get_energy_state, EnergyType.WOOD, energies)
        assert result is not None


# =============================================================================
# T8: CausalChain 因果链传播延迟
# =============================================================================


@pytest.mark.benchmark(min_rounds=10, max_time=2.0)
class TestCausalChainLatency:
    """CausalChain 因果图传播性能基线。"""

    @pytest.fixture(scope="class")
    def large_causal_chain(self):
        """100 节点线性因果链。"""
        from mci_world_model._sys.causal import CausalChain

        cc = CausalChain()
        energy_types = ["wood", "fire", "earth", "metal", "water"]
        for i in range(100):
            cc.add(f"n{i}", energy_type=energy_types[i % 5])
        for i in range(99):
            cc.link(f"n{i}", f"n{i+1}")
        return cc

    def test_propagate_100_nodes(self, benchmark, large_causal_chain):
        """100 节点因果链能量传播延迟。"""
        result = benchmark(large_causal_chain.propagate, "n0", delta=0.1)
        assert len(result) > 0

    def test_coverage_100_nodes(self, benchmark, large_causal_chain):
        """100 节点多层覆盖率计算延迟。"""
        all_ids = [f"n{i}" for i in range(100)]
        cov = benchmark(large_causal_chain.coverage, all_ids)
        assert 0 <= cov <= 100

    def test_get_causal_path(self, benchmark, large_causal_chain):
        """BFS 因果路径查询 (n0 → n50) 延迟。"""
        path = benchmark(large_causal_chain.get_causal_path, "n0", "n50")
        assert len(path) > 0


# =============================================================================
# T9: EvidenceCollector 证据处理延迟
# =============================================================================


@pytest.mark.benchmark(min_rounds=10, max_time=2.0)
class TestEvidenceCollectorLatency:
    """EvidenceCollector 证据收集与冲突检测性能基线。"""

    @pytest.fixture(scope="class")
    def populated_collector(self):
        """预填充的证据收集器。"""
        from mci_world_model._sys.evidence import EvidenceCollector

        ec = EvidenceCollector()
        ec.add_source("lab_report", reliability=0.9)
        ec.add_source("clinical_note", reliability=0.75)
        ec.add_source("patient_report", reliability=0.6)
        for i in range(50):
            ec.collect(
                evidence_id=f"ev_{i}",
                content=f"evidence_{i}",
                source="lab_report" if i % 3 == 0 else "clinical_note" if i % 3 == 1 else "patient_report",
                confidence=0.5 + (i % 50) / 100,
                evidence_type="observation",
            )
        return ec

    def test_detect_conflicts(self, benchmark, populated_collector):
        """50 条证据冲突检测延迟。"""
        conflicts = benchmark(populated_collector.detect_conflicts)
        assert isinstance(conflicts, list)

    def test_evidence_strength(self, benchmark, populated_collector):
        """证据强度计算延迟。"""
        strength = benchmark(populated_collector.get_evidence_strength)
        assert isinstance(strength, dict)
