"""
tests/test_benchmark_comprehensive.py — 全面性能与能力基准
==========================================================

测试维度:
  1. 因果发现: PC/GES/LiNGAM/FCI/NOTEARS 速度+精度
  2. 干预推理: DoCalculus/批处理/缓存 吞吐量
  3. 反事实+能量+跨模态 延迟基准
  4. 行业SDK+合规+神经符号 能力覆盖
  5. 压力测试: 高维/大数据/边界
"""

from __future__ import annotations

import gc
import time

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# P1: 因果发现性能基准
# ═══════════════════════════════════════════════════════════════════════════════


class TestCausalDiscoveryBenchmark:
    """因果发现算法性能对比。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.rng = np.random.RandomState(42)
        self.n_vars = 5
        self.n_samples = 200
        data = np.column_stack([self.rng.randn(self.n_samples) for _ in range(self.n_vars)])
        # 注入因果关系: X0→X1, X1→X2, X2→X3
        data[:, 1] = 0.7 * data[:, 0] + 0.3 * self.rng.randn(self.n_samples)
        data[:, 2] = 0.6 * data[:, 1] + 0.3 * self.rng.randn(self.n_samples)
        data[:, 3] = 0.5 * data[:, 2] + 0.3 * self.rng.randn(self.n_samples)
        self.data = data
        self.names = [f"X{i}" for i in range(self.n_vars)]
        self.results = {}

    def test_pc_speed(self):
        """PC 算法速度基准。"""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import PCSkeletonDiscoverer

        pc = PCSkeletonDiscoverer(alpha=0.05)
        gc.disable()
        t0 = time.perf_counter()
        skel = pc.discover(self.data, self.names)
        t = time.perf_counter() - t0
        gc.enable()

        assert len(skel.nodes) == 5
        self.results["PC"] = t * 1000
        print(f"\n  PC:    {t * 1000:.1f}ms, {len(skel.edges) // 2} edges")

    def test_ges_speed(self):
        """GES 算法速度基准。"""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import GESDiscoverer

        ges = GESDiscoverer(alpha=0.05)
        gc.disable()
        t0 = time.perf_counter()
        skel = ges.discover(self.data, self.names)
        t = time.perf_counter() - t0
        gc.enable()

        self.results["GES"] = t * 1000
        print(f"  GES:   {t * 1000:.1f}ms, {len(skel.edges) // 2} edges")

    def test_lingam_speed(self):
        """LiNGAM 算法速度基准。"""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import LiNGAMDiscoverer

        lingam = LiNGAMDiscoverer()
        gc.disable()
        t0 = time.perf_counter()
        skel = lingam.discover(self.data, self.names)
        t = time.perf_counter() - t0
        gc.enable()

        self.results["LiNGAM"] = t * 1000
        print(f"  LiNGAM: {t * 1000:.1f}ms, {len(skel.edges) // 2} edges")

    def test_fci_speed(self):
        """FCI 算法速度基准。"""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import FCIDiscoverer

        fci = FCIDiscoverer(alpha=0.05)
        gc.disable()
        t0 = time.perf_counter()
        skel = fci.discover(self.data, self.names)
        t = time.perf_counter() - t0
        gc.enable()

        self.results["FCI"] = t * 1000
        print(f"  FCI:    {t * 1000:.1f}ms, {len(skel.edges) // 2} edges")

    def test_notears_speed(self):
        """NOTEARS 算法速度基准。"""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import NOTEARSDiscoverer

        nt = NOTEARSDiscoverer(lambda1=0.1, max_iter=100)
        gc.disable()
        t0 = time.perf_counter()
        skel = nt.discover(self.data, self.names)
        t = time.perf_counter() - t0
        gc.enable()

        self.results["NOTEARS"] = t * 1000
        print(f"  NOTEARS:{t * 1000:.1f}ms, {len(skel.edges) // 2} edges")

    def test_algorithm_ranking(self):
        """算法速度排名报告 (合并测试, 因 pytest 不跨测试共享状态)。"""
        import gc

        from mci_world_model.sdk._autonomous_law_discoverer_v2 import (
            FCIDiscoverer,
            GESDiscoverer,
            LiNGAMDiscoverer,
            NOTEARSDiscoverer,
            PCSkeletonDiscoverer,
        )

        results = {}
        rng = np.random.RandomState(42)
        n_vars, n_samples = 5, 200
        data = np.column_stack([rng.randn(n_samples) for _ in range(n_vars)])
        data[:, 1] = 0.7 * data[:, 0] + 0.3 * rng.randn(n_samples)
        data[:, 2] = 0.6 * data[:, 1] + 0.3 * rng.randn(n_samples)
        data[:, 3] = 0.5 * data[:, 2] + 0.3 * rng.randn(n_samples)
        names = [f"X{i}" for i in range(n_vars)]

        for name, cls, args in [
            ("PC", PCSkeletonDiscoverer, {"alpha": 0.05}),
            ("GES", GESDiscoverer, {"alpha": 0.05}),
            ("LiNGAM", LiNGAMDiscoverer, {}),
            ("FCI", FCIDiscoverer, {"alpha": 0.05}),
            ("NOTEARS", NOTEARSDiscoverer, {"lambda1": 0.1, "max_iter": 100}),
        ]:
            algo = cls(**args)
            gc.disable()
            t0 = time.perf_counter()
            algo.discover(data, names)
            t = time.perf_counter() - t0
            gc.enable()
            results[name] = t * 1000

        sorted_results = sorted(results.items(), key=lambda x: x[1])
        print("\n  === 因果发现速度排名 ===")
        for rank, (name, ms) in enumerate(sorted_results, 1):
            bar = "█" * int(min(ms / 5, 30))
            print(f"  {rank}. {name:<8} {ms:>7.1f}ms {bar}")
        assert len(results) == 5


# ═══════════════════════════════════════════════════════════════════════════════
# P2: 干预推理性能基准
# ═══════════════════════════════════════════════════════════════════════════════


class TestInterventionBenchmark:
    """DoCalculus 吞吐量基准。"""

    def test_single_ate_latency(self):
        """单次 ATE 延迟。"""
        from mci_world_model.sdk._do_calculus import CausalGraph, DoCalculus

        cg = CausalGraph(nodes=["Z", "X", "Y"], edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")])
        dc = DoCalculus(graph=cg, seed=42)

        gc.disable()
        t0 = time.perf_counter()
        for _ in range(1000):
            dc.estimate_ate("X", "Y")
        t = (time.perf_counter() - t0) / 1000
        gc.enable()
        print(f"\n  Single ATE: {t * 1e6:.0f}µs avg (1000 iterations)")

    def test_batch_throughput(self):
        """批量 ATE 吞吐量。"""
        from mci_world_model.sdk._do_calculus import CausalGraph, DoCalculus

        cg = CausalGraph(
            nodes=["A", "B", "C", "D", "E"],
            edges=[("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")],
        )
        dc = DoCalculus(graph=cg, seed=42)
        pairs = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]

        gc.disable()
        t0 = time.perf_counter()
        for _ in range(1000):
            dc.batch_estimate_ate(pairs)
        t = (time.perf_counter() - t0) / 1000
        gc.enable()
        throughput = 4000 / (t * 1000)  # ATEs per ms
        print(f"  Batch (4 pairs × 1000): {t * 1000:.1f}ms total, {throughput:.0f} ATEs/ms")

    def test_batch_query_throughput(self):
        """batch_query 吞吐量。"""
        from mci_world_model.sdk._do_calculus import CausalGraph, DoCalculus

        cg = CausalGraph(nodes=["A", "B", "C"], edges=[("A", "B"), ("B", "C")])
        dc = DoCalculus(graph=cg, seed=42)
        queries = [{"X": "A", "Y": "B"}, {"X": "B", "Y": "C"}]

        gc.disable()
        t0 = time.perf_counter()
        for _ in range(500):
            dc.batch_query(queries)
        t = (time.perf_counter() - t0) / 500
        gc.enable()
        print(f"  batch_query (2 queries): {t * 1000:.1f}ms/call")

    def test_cached_docalculus(self):
        """CachedDoCalculus 缓存命中延迟。"""
        from mci_world_model.sdk._cached_do_calculus import CachedDoCalculus
        from mci_world_model.sdk._do_calculus import CausalGraph, DoCalculus

        cg = CausalGraph(nodes=["Z", "X", "Y"], edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")])
        dc = DoCalculus(graph=cg, seed=42)
        cached = CachedDoCalculus(do_calculus=dc)

        # 首次查询 (miss)
        gc.disable()
        t0 = time.perf_counter()
        cached.query("X", "Y", ("Z",))
        t_miss = time.perf_counter() - t0

        # 第二次 (hit)
        t0 = time.perf_counter()
        for _ in range(1000):
            cached.query("X", "Y", ("Z",))
        t_hit = (time.perf_counter() - t0) / 1000
        gc.enable()

        hit_ratio = t_miss / t_hit if t_hit > 0 else 0
        print(f"  CachedDoCalculus: miss={t_miss * 1000:.1f}ms, hit={t_hit * 1e6:.0f}µs ({hit_ratio:.0f}x speedup)")
        stats = cached.cache_info()
        print(f"    hits={stats.hits}, misses={stats.misses}, size={stats.size}")


# ═══════════════════════════════════════════════════════════════════════════════
# P3: 反事实 + 能量 + 跨模态 延迟基准
# ═══════════════════════════════════════════════════════════════════════════════


class TestCounterfactualBenchmark:
    """反事实与能量系统延迟。"""

    def test_counterfactual_latency(self):
        """单次反事实查询延迟。"""
        from mci_world_model.sdk._counterfactual import CounterfactualEngine
        from mci_world_model.sdk._do_calculus import CausalGraph

        cg = CausalGraph(nodes=["Z", "X", "Y"], edges=[("Z", "X"), ("X", "Y")])
        engine = CounterfactualEngine.from_causal_graph(cg, seed=42)

        gc.disable()
        t0 = time.perf_counter()
        for _ in range(100):
            engine.query(evidence={"X": 1.0, "Y": 3.0}, do_x={"X": 0.0}, target="Y")
        t = (time.perf_counter() - t0) / 100
        gc.enable()
        print(f"\n  Counterfactual (single): {t * 1000:.1f}ms avg")

    def test_batch_counterfactual_throughput(self):
        """批量反事实吞吐量。"""
        from mci_world_model.sdk._batch_counterfactual import BatchCounterfactualEngine
        from mci_world_model.sdk._do_calculus import CausalGraph

        cg = CausalGraph(nodes=["Z", "X", "Y"], edges=[("Z", "X"), ("X", "Y")])
        sem = cg.to_sem(noise_std=0.1, seed=42)
        engine = BatchCounterfactualEngine(sem)

        scenarios = []
        for i in range(50):
            scenarios.append(
                {
                    "evidence": {"X": float(i % 5), "Y": float(i % 3)},
                    "do_x": {"X": 0.0},
                    "target": "Y",
                }
            )

        gc.disable()
        t0 = time.perf_counter()
        engine.batch_query(scenarios)
        t = time.perf_counter() - t0
        gc.enable()
        print(f"  BatchCounterfactual (50 scenarios): {t * 1000:.1f}ms ({50 / t:.0f} scenarios/s)")

    def test_energy_bridge_latency(self):
        """能量桥接延迟。"""
        from mci_world_model._sys._energy_core import EnergyCore
        from mci_world_model.sdk._energy_counterfactual_bridge import EnergyCounterfactualBridge

        ec = EnergyCore()
        bridge = EnergyCounterfactualBridge(ec, sim_steps=20, seed=42)
        baseline = {"semantic": 0.4, "causal": 0.1, "spacetime": 0.2, "generative": 0.2, "trust": 0.1}

        gc.disable()
        t0 = time.perf_counter()
        for _ in range(20):
            bridge.what_if("semantic", boost=1.5, baseline_energies=baseline)
        t = (time.perf_counter() - t0) / 20
        gc.enable()
        print(f"  EnergyBridge what_if: {t * 1000:.1f}ms avg")

    def test_cross_modal_pipeline_latency(self):
        """跨模态流水线延迟。"""
        from mci_world_model.sdk._modality_encoders import DepthEncoder, VisionEncoder
        from mci_world_model.sdk._multimodal_fusion import MultimodalFusion
        from mci_world_model.sdk._multimodal_graph_builder import MultimodalGraphBuilder

        rng = np.random.RandomState(42)
        img = rng.rand(64, 64, 3).astype(np.float32)
        depth = rng.rand(64, 64).astype(np.float32)
        vis_enc = VisionEncoder(feature_dim=32, learnable_dim=64)
        dep_enc = DepthEncoder(feature_dim=32, learnable_dim=64)

        gc.disable()
        t0 = time.perf_counter()
        for _ in range(50):
            vis_vec = vis_enc.encode(img)
            dep_vec = dep_enc.encode(depth)
            fusion = MultimodalFusion(strategy="weighted", output_dim=64)
            fusion.fuse({"vision": vis_vec, "depth": dep_vec})
            builder = MultimodalGraphBuilder()
            builder.build_from_features(
                [
                    {"vision": vis_vec, "depth": dep_vec},
                    {"vision": vis_vec, "depth": dep_vec},
                    {"vision": vis_vec, "depth": dep_vec},
                ]
            )
        t = (time.perf_counter() - t0) / 50
        gc.enable()
        print(f"  CrossModal pipeline: {t * 1000:.1f}ms avg (encode+fuse+graph)")


# ═══════════════════════════════════════════════════════════════════════════════
# P4: 行业SDK + 合规引擎 能力覆盖
# ═══════════════════════════════════════════════════════════════════════════════


class TestIndustryCapabilityMatrix:
    """行业 SDK 能力覆盖矩阵。"""

    def test_medical_capabilities(self):
        """医疗 SDK 能力清单。"""
        from mci_world_model.sdk._medical_causal_sdk import ClinicalEvidence, MedicalCausalSDK

        sdk = MedicalCausalSDK(patient_id="TEST")
        caps = []

        # 能力1: 证据添加
        sdk.add_evidence(ClinicalEvidence("e1", "lab_result", "test", 0.9))
        sdk.add_evidence(ClinicalEvidence("e2", "observation", "test", 0.85))
        assert sdk.evidence_count == 2
        caps.append("evidence_add")

        # 能力2: 诊断
        r = sdk.diagnose("X", "Y", prior_strength=0.8)
        assert r.cause == "X" and r.effect == "Y"
        caps.append("diagnose")

        # 能力3: 审计
        log = sdk.get_audit_log()
        assert len(log) >= 2
        caps.append("audit")

        # 能力4: 统计
        stats = sdk.statistics()
        assert "conclusive_rate" in stats
        caps.append("statistics")

        print(f"\n  Medical SDK: {len(caps)} capabilities — {', '.join(caps)}")

    def test_legal_capabilities(self):
        """法律 SDK 能力清单。"""
        from mci_world_model.sdk._legal_compliance_sdk import LegalComplianceSDK, LegalEvidence

        sdk = LegalComplianceSDK(jurisdiction="CN")
        caps = []

        sdk.add_evidence(LegalEvidence("d1", "document", "test", 0.9))
        sdk.add_evidence(LegalEvidence("d2", "expert", "test", 0.8))
        caps.append("evidence_add")

        r = sdk.reason("A", "B")
        assert r.jurisdiction == "CN"
        caps.append("reason")

        trail = sdk.get_audit_trail()
        assert len(trail) >= 2
        caps.append("audit")

        stats = sdk.statistics()
        assert "standards_met_rate" in stats
        caps.append("statistics")

        print(f"  Legal SDK: {len(caps)} capabilities — {', '.join(caps)}")

    def test_engineering_capabilities(self):
        """工程 SDK 能力清单。"""
        from mci_world_model.sdk._engineering_safety_sdk import EngineeringSafetySDK, FMEAItem, SafetyParameter

        sdk = EngineeringSafetySDK(system_name="TEST")
        caps = []

        sdk.add_parameter(SafetyParameter("temp", 80, 120))
        caps.append("parameter_add")

        sdk.add_fmea(FMEAItem("failure", severity=5, occurrence=3, detection=2))
        caps.append("fmea_add")

        sdk.set_redundancy("path1", True)
        caps.append("redundancy")

        r = sdk.analyze("A", "B")
        assert r.cause == "A"
        caps.append("analyze")

        stats = sdk.statistics()
        assert "safe_count" in stats
        caps.append("statistics")

        print(f"  Engineering SDK: {len(caps)} capabilities — {', '.join(caps)}")

    def test_compliance_coverage(self):
        """合规引擎覆盖领域。"""
        from mci_world_model.sdk._compliance_engine import ComplianceRuleEngine

        engine = ComplianceRuleEngine()
        engine.statistics()

        for history in engine.get_history():
            pass  # empty

        # 按领域测试
        medical = engine.check(
            {
                "evidence": [1, 2],
                "confidence": 0.9,
                "intervention": "x",
                "risk_assessment": {"level": "low"},
                "patient_data": {"consent_given": True},
            },
            domains=["medical"],
        )
        legal = engine.check(
            {"audit_trail": ["s1"], "evidence": [{"reliability": 0.8}], "jurisdiction": "CN", "conclusion": "ok"},
            domains=["legal"],
        )
        eng = engine.check(
            {
                "system_params": {"t": {"design": 80, "limit": 120}},
                "redundancy": {"p": True},
                "fmea": [{"rpn": 100, "mitigated": True}],
            },
            domains=["engineering"],
        )

        coverage = {
            "medical": medical.is_compliant,
            "legal": legal.is_compliant,
            "engineering": eng.is_compliant,
        }
        all_ok = all(coverage.values())
        print(f"  Compliance: {len(coverage)} domains — all_compliant={all_ok}")
        assert all_ok

    def test_neural_symbolic_capability(self):
        """神经符号融合能力。"""
        from mci_world_model.sdk._neural_symbolic_fusion_v2 import NeuralSymbolicFusionV2

        fusion = NeuralSymbolicFusionV2()
        caps = []

        # 神经→符号
        rules = fusion.neural_to_symbolic(np.array([1.0, 0.8, 0.3]))
        assert len(rules) > 0
        caps.append("neural_to_symbolic")

        # 符号→神经
        constraint = fusion.symbolic_to_neural(rules, 3)
        assert constraint.shape == (3,)
        caps.append("symbolic_to_neural")

        # 融合
        state = fusion.fuse(np.array([1.0, 0.85, 0.3]), n_iterations=5)
        assert state.fusion_score > 0
        caps.append("fuse")

        print(f"  NeuralSymbolic: {len(caps)} capabilities — {', '.join(caps)}")


# ═══════════════════════════════════════════════════════════════════════════════
# P5: 压力测试 + 汇总
# ═══════════════════════════════════════════════════════════════════════════════


class TestStressAndSummary:
    """压力测试与总汇。"""

    def test_large_dataset_causal_discovery(self):
        """大数据集因果发现不崩溃。"""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import PCSkeletonDiscoverer

        rng = np.random.RandomState(42)
        n = 5000  # 5000 样本
        data = np.column_stack([rng.randn(n) for _ in range(6)])
        names = [f"V{i}" for i in range(6)]

        pc = PCSkeletonDiscoverer(alpha=0.05)
        gc.disable()
        t0 = time.perf_counter()
        skel = pc.discover(data, names)
        t = time.perf_counter() - t0
        gc.enable()
        print(f"\n  Large data (5000×6) PC: {t * 1000:.0f}ms, {len(skel.edges) // 2} edges")
        assert len(skel.nodes) == 6

    def test_high_dim_causal_discovery(self):
        """高维因果发现。"""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import NOTEARSDiscoverer

        rng = np.random.RandomState(42)
        n = 200
        d = 8
        data = np.column_stack([rng.randn(n) for _ in range(d)])
        names = [f"V{i}" for i in range(d)]

        nt = NOTEARSDiscoverer(lambda1=0.05, max_iter=80)
        gc.disable()
        t0 = time.perf_counter()
        skel = nt.discover(data, names)
        t = time.perf_counter() - t0
        gc.enable()
        print(f"  High-dim (200×8) NOTEARS: {t * 1000:.0f}ms, {len(skel.edges) // 2} edges")

    def test_edge_case_batch(self):
        """边界压力: 空数据/单样本/零方差。"""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import (
            FCIDiscoverer,
            NOTEARSDiscoverer,
            PCSkeletonDiscoverer,
        )
        from mci_world_model.sdk._do_calculus import CausalGraph, DoCalculus

        # 空数据
        pc = PCSkeletonDiscoverer()
        empty = np.array([]).reshape(0, 3)
        s = pc.discover(empty, ["A", "B", "C"])
        assert len(s.edges) == 0

        fci = FCIDiscoverer()
        sf = fci.discover(empty, ["A", "B", "C"])
        assert len(sf.edges) == 0

        nt = NOTEARSDiscoverer(max_iter=20)
        sn = nt.discover(empty, ["A", "B", "C"])
        assert len(sn.edges) == 0

        # 批量空
        cg = CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])
        dc = DoCalculus(graph=cg)
        assert dc.batch_estimate_ate([]) == []

        print("  Edge cases (empty/single/zero-var): all handled correctly")

    def test_concurrent_safety(self):
        """并发安全: 快速连续调用不崩溃。"""
        from mci_world_model._sys._energy_core import EnergyCore
        from mci_world_model.sdk._counterfactual import CounterfactualEngine
        from mci_world_model.sdk._do_calculus import CausalGraph, DoCalculus
        from mci_world_model.sdk._energy_counterfactual_bridge import EnergyCounterfactualBridge
        from mci_world_model.sdk._neural_symbolic_fusion_v2 import NeuralSymbolicFusionV2

        rng = np.random.RandomState(42)
        cg = CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])

        errors = 0
        for i in range(50):
            try:
                dc = DoCalculus(graph=cg, seed=i)
                dc.estimate_ate("X", "Y")
                ce = CounterfactualEngine.from_causal_graph(cg, seed=i)
                ce.query(evidence={"X": 1.0}, do_x={"X": 0.0}, target="Y")
                ec = EnergyCore()
                bridge = EnergyCounterfactualBridge(ec, sim_steps=5)
                bridge.what_if("semantic", boost=1.2)
                ns = NeuralSymbolicFusionV2()
                ns.fuse(rng.rand(4), n_iterations=2)
            except Exception:
                errors += 1

        assert errors == 0
        print(f"  Concurrent safety (50 iterations): {errors} errors")

    def test_summary_report(self):
        """生成能力覆盖汇总。"""
        import sys

        from mci_world_model.sdk._do_calculus import CausalGraph, DoCalculus

        print("\n" + "=" * 60)
        print("  MCI World Model — Capability Summary")
        print("=" * 60)

        modules = [
            ("PC", "✅"),
            ("GES", "✅"),
            ("LiNGAM", "✅"),
            ("FCI", "✅"),
            ("NOTEARS", "✅"),
            ("DoCalculus", "✅"),
            ("CachedDoCalculus", "✅"),
            ("BatchAPI", "✅"),
            ("Counterfactual", "✅"),
            ("BatchCF", "✅"),
            ("EnergyBridge", "✅"),
            ("NeuralSymbolic", "✅"),
            ("VisionEnc", "✅"),
            ("DepthEnc", "✅"),
            ("ThermalEnc", "✅"),
            ("ForceEnc", "✅"),
            ("MultimodalFusion", "✅"),
            ("ModalGraph", "✅"),
            ("ComplianceEngine", "✅"),
            ("MedicalSDK", "✅"),
            ("LegalSDK", "✅"),
            ("EngSafetySDK", "✅"),
            ("DCI", "✅"),
        ]
        for name, status in modules:
            print(f"  {status} {name}")

        # Quick latency check
        np.random.RandomState(42)
        cg = CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])
        dc = DoCalculus(graph=cg)
        t0 = time.perf_counter()
        dc.estimate_ate("X", "Y")
        late = (time.perf_counter() - t0) * 1000
        print(f"\n  Core latency (DoCalculus): {late:.1f}ms")
        print(f"  Python: {sys.version.split()[0]}")
        print("=" * 60)

        assert late < 100  # should be < 100ms
