"""
tests/test_integration_p6_p11.py — P6-P11 实现性端到端验证
==========================================================

模拟真实使用场景，验证所有模块的协同工作能力。
"""

from __future__ import annotations

import time

import numpy as np

# =============================================================================
# P6: 因果推理强化
# =============================================================================


class TestP6CausalReasoning:
    """FCI + NOTEARS + DoCalculus batch + EnergyCore bridge 实现性验证。"""

    def test_fci_discovers_causal_chain(self):
        """场景: 医学研究中含隐混淆的因果发现。"""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import FCIDiscoverer

        rng = np.random.RandomState(42)
        n = 300
        # 模拟: 基因表达 → 蛋白质水平 → 疾病指标, 含隐混淆
        U = rng.randn(n)
        gene = 0.5 * U + rng.randn(n)
        protein = 0.7 * gene + rng.randn(n)
        disease = 0.6 * protein + 0.4 * U + rng.randn(n)

        data = np.column_stack([gene, protein, disease])
        fci = FCIDiscoverer(alpha=0.05, min_corr=0.1)
        skel = fci.discover(data, ["gene", "protein", "disease"])

        assert skel.adj_matrix.shape == (3, 3)
        # 应发现至少 gene→protein 或 protein→disease
        adj = skel.adj_matrix
        assert adj[0, 1] == 1 or adj[1, 2] == 1
        print(f"  FCI found {len(skel.edges) // 2} causal links, conf={skel.confidence:.3f}")

    def test_notears_linear_chain(self):
        """场景: 可微分因果发现线性经济指标链。"""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import NOTEARSDiscoverer

        rng = np.random.RandomState(42)
        n = 200
        inflation = rng.randn(n)
        interest = 0.6 * inflation + rng.randn(n)
        gdp = 0.5 * interest + 0.2 * rng.randn(n)
        data = np.column_stack([inflation, interest, gdp])

        nt = NOTEARSDiscoverer(lambda1=0.05, max_iter=500, threshold=0.3)
        skel = nt.discover(data, ["inflation", "interest", "gdp"])

        assert len(skel.nodes) == 3
        assert len(skel.edges) >= 1, f"No edges found: {skel.edges}"
        assert len(skel.nodes) == 3
        print(f"  NOTEARS conf={skel.confidence:.3f}, edges={skel.edges}")

    def test_docalculus_batch_deferral(self):
        """场景: 批量干预分析——贷款政策对多个经济指标的因果效应。"""
        from mci_world_model.sdk._do_calculus import CausalGraph, DoCalculus

        cg = CausalGraph(
            nodes=["rate", "inflation", "gdp", "employment"],
            edges=[
                ("rate", "inflation"),
                ("rate", "gdp"),
                ("inflation", "gdp"),
                ("gdp", "employment"),
            ],
        )
        dc = DoCalculus(graph=cg, seed=42)

        t0 = time.perf_counter()
        results = dc.batch_estimate_ate(
            [
                ("rate", "inflation"),
                ("rate", "gdp"),
                ("rate", "employment"),
            ]
        )
        elapsed = time.perf_counter() - t0

        assert len(results) == 3
        assert elapsed < 1.0  # 批量应在 1s 内完成
        for r in results:
            assert r.ate is not None
        print(f"  Batch {len(results)} ATEs in {elapsed * 1000:.1f}ms")

    def test_energy_bridge_what_if(self):
        """场景: 五行能量系统的 what-if 干预分析。"""
        from mci_world_model._sys._energy_core import EnergyCore
        from mci_world_model.sdk._energy_counterfactual_bridge import (
            EnergyCounterfactualBridge,
        )

        ec = EnergyCore()
        bridge = EnergyCounterfactualBridge(ec, sim_steps=30, seed=42)

        baseline = {"semantic": 0.4, "causal": 0.1, "spacetime": 0.2, "generative": 0.2, "trust": 0.1}
        results = bridge.what_if("semantic", boost=2.0, baseline_energies=baseline)

        assert len(results) == 4
        deltas = [abs(r.delta) for r in results]
        assert max(deltas) > 0.001  # 增强应有非零效应
        print(f"  Energy what_if: max delta={max(deltas):.4f}")


# =============================================================================
# P7: 行业 SDK
# =============================================================================


class TestP7IndustrySDKs:
    """医疗/法律/工程 SDK 实现性验证。"""

    def test_medical_diagnosis_workflow(self):
        """场景: 临床诊断——药物副作用因果推断。"""
        from mci_world_model.sdk._medical_causal_sdk import (
            ClinicalEvidence,
            MedicalCausalSDK,
        )

        sdk = MedicalCausalSDK(patient_id="P-042", strict_mode=True)

        # 收集 5 条临床证据 (全部相关)
        sdk.add_evidence(ClinicalEvidence("lab1", "observation", "drug_X linked to liver_damage", 0.95))
        sdk.add_evidence(ClinicalEvidence("img1", "observation", "drug_X liver inflammation", 0.93))
        sdk.add_evidence(ClinicalEvidence("obs1", "observation", "drug_X patient on drug_X", 0.91))
        sdk.add_evidence(ClinicalEvidence("vital1", "observation", "drug_X ALT elevated", 0.92))
        sdk.add_evidence(ClinicalEvidence("lab2", "observation", "drug_X other causes excluded", 0.90))

        result = sdk.diagnose("drug_X", "liver_damage", prior_strength=0.95)
        assert result.is_conclusive
        assert result.confidence > 0.7
        assert len(result.evidence_ids) >= 4
        print(
            f"  Medical: {result.cause}→{result.effect}, strength={result.causal_strength:.3f}, conf={result.confidence:.3f}"
        )

    def test_legal_reasoning_workflow(self):
        """场景: 法律因果——产品责任因果链。"""
        from mci_world_model.sdk._legal_compliance_sdk import (
            LegalComplianceSDK,
            LegalEvidence,
        )

        sdk = LegalComplianceSDK(jurisdiction="CN", standard="preponderance")

        sdk.add_evidence(LegalEvidence("doc1", "document", "product X safety report", 0.92))
        sdk.add_evidence(LegalEvidence("doc2", "expert", "expert testimony: causal link", 0.75))
        sdk.add_evidence(LegalEvidence("doc3", "data", "incident records database", 0.65))

        result = sdk.reason("product_X", "consumer_injury", prior_strength=0.8)
        assert result.legal_standard_met
        assert result.causal_link_strength > 0.51
        assert len(result.bias_flags) == 0
        print(
            f"  Legal: {result.cause}→{result.effect}, strength={result.causal_link_strength:.3f}, met={result.legal_standard_met}"
        )

    def test_engineering_safety_workflow(self):
        """场景: 工程安全——核电站控制棒故障因果分析。"""
        from mci_world_model.sdk._engineering_safety_sdk import (
            EngineeringSafetySDK,
            FMEAItem,
            SafetyParameter,
        )

        sdk = EngineeringSafetySDK(system_name="Reactor-Core-7", redundancy_required=True)

        sdk.add_parameter(SafetyParameter("core_temp", 280.0, 350.0, "C"))
        sdk.add_parameter(SafetyParameter("coolant_pressure", 140.0, 175.0, "bar"))
        sdk.add_fmea(FMEAItem("rod_stuck", "control rod failure", severity=9, occurrence=2, detection=3))
        sdk.add_fmea(FMEAItem("coolant_leak", "primary loop breach", severity=10, occurrence=1, detection=2))
        sdk.add_parameter(SafetyParameter("backup_pressure", 50.0, 80.0, "bar"))
        sdk.set_redundancy("cooling_path_1", True)
        sdk.set_redundancy("cooling_path_2", True)

        result = sdk.analyze("control_rod_failure", "core_overheat", causal_evidence_strength=0.9)
        assert result.safety_assessment == "safe"
        assert result.causal_confidence > 0.8
        assert result.margin_sufficient
        print(
            f"  Engineering: {result.cause}→{result.effect}, safety={result.safety_assessment}, RPN={result.fmea_rpn_max}"
        )


# =============================================================================
# P8: 神经符号融合
# =============================================================================


class TestP8NeuralSymbolicFusion:
    """神经符号融合实现性验证。"""

    def test_fusion_bidirectional_cycle(self):
        """场景: 神经特征→符号规则→约束→融合的双向循环。"""
        from mci_world_model.sdk._neural_symbolic_fusion_v2 import NeuralSymbolicFusionV2

        fusion = NeuralSymbolicFusionV2(rule_threshold=0.7, max_iterations=15)
        neural_repr = np.array([1.0, 0.85, 0.3, 0.15, 0.05])
        var_names = ["gene_A", "gene_B", "gene_C", "gene_D", "gene_E"]

        state = fusion.fuse(neural_repr, var_names, n_iterations=10)
        assert state.fusion_score > 0.0
        assert state.consistency >= 0.0
        assert len(state.symbolic_rules) > 0

        # 双向验证
        rules = fusion.neural_to_symbolic(neural_repr, var_names)
        constraint = fusion.symbolic_to_neural(rules, len(neural_repr))
        assert np.any(constraint > 0)
        print(
            f"  NS-Fusion: score={state.fusion_score:.3f}, rules={len(state.symbolic_rules)}, consistency={state.consistency:.3f}"
        )


# =============================================================================
# P9: 合规引擎
# =============================================================================


class TestP9ComplianceEngine:
    """合规引擎全领域实现性验证。"""

    def test_full_pipeline_medical(self):
        """场景: 医疗合规全流程——证据→诊断→合规检查。"""
        from mci_world_model.sdk._compliance_engine import ComplianceRuleEngine
        from mci_world_model.sdk._medical_causal_sdk import ClinicalEvidence, MedicalCausalSDK

        sdk = MedicalCausalSDK(patient_id="P-101", strict_mode=True)
        for i in range(6):
            sdk.add_evidence(ClinicalEvidence(f"e{i}", "lab_result", f"drug_A symptom_B evidence {i}", 0.92 + i * 0.01))
        diagnosis = sdk.diagnose("drug_A", "symptom_B", prior_strength=0.9)

        engine = ComplianceRuleEngine()
        context = {
            "evidence": [{"type": "lab"} for _ in range(4)],
            "confidence": diagnosis.confidence,
            "intervention": {"drug": "drug_A"},
            "risk_assessment": {"level": "low"},
            "patient_data": {"consent_given": True},
        }
        result = engine.check(context, domains=["medical"])
        assert result.is_compliant
        print(f"  Compliance-Medical: {result.summary}")

    def test_full_pipeline_engineering(self):
        """场景: 工程合规——安全裕度+冗余+FMEA 全部通过。"""
        from mci_world_model.sdk._compliance_engine import ComplianceRuleEngine
        from mci_world_model.sdk._engineering_safety_sdk import EngineeringSafetySDK, FMEAItem, SafetyParameter

        sdk = EngineeringSafetySDK(system_name="Turbine-Gen-3")
        sdk.add_parameter(SafetyParameter("rpm", 5400, 7200))
        sdk.add_fmea(FMEAItem("blade_fracture", severity=7, occurrence=2, detection=2, mitigated=True))
        sdk.set_redundancy("bearing_path", True)
        sdk.analyze("vibration_spike", "blade_failure", causal_evidence_strength=0.85)

        engine = ComplianceRuleEngine()
        context = {
            "system_params": {"rpm": {"design": 5400, "limit": 7200}},
            "redundancy": {"bearing_path": True},
            "fmea": [{"failure_mode": "blade_fracture", "rpn": 28, "mitigated": True}],
        }
        result = engine.check(context, domains=["engineering"])
        assert result.is_compliant
        print(f"  Compliance-Engineering: {result.summary}")


# =============================================================================
# P10: 跨模态流水线
# =============================================================================


class TestP10CrossModalPipeline:
    """跨模态流水线实现性验证。"""

    def test_vision_depth_pipeline(self):
        """场景: RGB+深度 → 编码 → 融合 → 图构建。"""
        from mci_world_model.sdk._modality_encoders import DepthEncoder, VisionEncoder
        from mci_world_model.sdk._multimodal_fusion import MultimodalFusion
        from mci_world_model.sdk._multimodal_graph_builder import MultimodalGraphBuilder

        rng = np.random.RandomState(42)
        img = rng.rand(64, 64, 3).astype(np.float32)
        depth = rng.rand(64, 64).astype(np.float32)

        vis_enc = VisionEncoder(feature_dim=32, learnable_dim=64)
        dep_enc = DepthEncoder(feature_dim=32, learnable_dim=64)
        vis_vec = vis_enc.encode(img)
        dep_vec = dep_enc.encode(depth)

        fusion = MultimodalFusion(strategy="weighted", output_dim=64)
        fused = fusion.fuse({"vision": vis_vec, "depth": dep_vec})
        assert fused.fused_vector.shape == (64,)

        builder = MultimodalGraphBuilder(min_correlation=0.1)
        timeline = [
            {"vision": vis_vec, "depth": dep_vec},
            {"vision": vis_vec * 0.9, "depth": dep_vec * 1.1},
            {"vision": vis_vec * 1.1, "depth": dep_vec * 0.9},
        ]
        edges = builder.build_from_features(timeline)
        assert isinstance(edges, list)
        print(
            f"  Cross-modal: vis={vis_vec.shape}, depth={dep_vec.shape}, fused={fused.fused_vector.shape}, graph_edges={len(edges)}"
        )


# =============================================================================
# P11: 可微因果 + NOTEARS
# =============================================================================


class TestP11DifferentiableCausal:
    """可微因果推断 + NOTEARS 实现性验证。"""

    def test_dci_learns_causal_effect(self):
        """场景: 学习治疗效应的真实因果参数。"""
        from mci_world_model.sdk._differentiable_causal import DifferentiableCausalInference

        rng = np.random.RandomState(42)
        n = 300
        dosage = rng.randn(n)
        recovery = 3.0 * dosage + 0.5 * rng.randn(n)  # true ATE = 3.0

        dci = DifferentiableCausalInference(learning_rate=0.005)
        dci.set_data(treatment=dosage, outcome=recovery)
        result = dci.optimize(n_iterations=500)

        effect = dci.treatment_effect
        assert abs(effect) > 1.0
        assert abs(effect - 3.0) < 1.5  # should be close to 3.0
        print(f"  DCI: learned ATE={effect:.3f} (true=3.0), loss {result.initial_loss:.3f}→{result.final_loss:.3f}")

    def test_notears_dci_bridge(self):
        """场景: NOTEARS 发现结构 → DCI 估计效应。"""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import NOTEARSDiscoverer
        from mci_world_model.sdk._differentiable_causal import DifferentiableCausalInference

        rng = np.random.RandomState(42)
        n = 250
        smoking = rng.randn(n)
        cancer_risk = 0.6 * smoking + 0.3 * rng.randn(n)
        X = np.column_stack([smoking, cancer_risk])

        nt = NOTEARSDiscoverer(lambda1=0.05, max_iter=200, threshold=0.3)
        skel = nt.discover(X, ["smoking", "cancer_risk"])
        assert np.sum(skel.adj_matrix) > 0

        health_score = 5.0 - 2.0 * smoking + 0.3 * rng.randn(n)
        dci = DifferentiableCausalInference(learning_rate=0.005)
        dci.set_data(treatment=smoking, outcome=health_score)
        dci.optimize(n_iterations=300)
        effect = dci.treatment_effect
        assert abs(effect) > 0.5
        print(f"  NOTEARS→DCI: discovered {np.sum(skel.adj_matrix)} edges, ATE={effect:.3f}")

    def test_batch_pipeline(self):
        """场景: 一次运行全部模块，验证零崩溃。"""
        from mci_world_model._sys._energy_core import EnergyCore
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import FCIDiscoverer, NOTEARSDiscoverer
        from mci_world_model.sdk._do_calculus import CausalGraph, DoCalculus
        from mci_world_model.sdk._energy_counterfactual_bridge import EnergyCounterfactualBridge
        from mci_world_model.sdk._neural_symbolic_fusion_v2 import NeuralSymbolicFusionV2

        rng = np.random.RandomState(42)
        data = np.column_stack([rng.randn(200) for _ in range(4)])

        t0 = time.perf_counter()

        # P6
        FCIDiscoverer().discover(data, ["A", "B", "C", "D"])
        NOTEARSDiscoverer(max_iter=50).discover(data, ["A", "B", "C", "D"])
        cg = CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])
        dc = DoCalculus(graph=cg)
        dc.batch_estimate_ate([("X", "Y")])
        ec = EnergyCore()
        bridge = EnergyCounterfactualBridge(ec, sim_steps=15)
        bridge.what_if("semantic", boost=1.5)

        # P8
        ns = NeuralSymbolicFusionV2()
        ns.fuse(np.array([1.0, 0.8, 0.3]), n_iterations=5)

        # P11
        from mci_world_model.sdk._differentiable_causal import DifferentiableCausalInference

        dci = DifferentiableCausalInference()
        dci.set_data(treatment=rng.randn(100), outcome=rng.randn(100))
        dci.optimize(n_iterations=30)

        elapsed = time.perf_counter() - t0
        print(f"  Full P6-P11 pipeline: completed in {elapsed * 1000:.0f}ms (zero crash)")
        assert elapsed < 5.0
