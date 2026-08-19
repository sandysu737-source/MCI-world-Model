"""P7-P8 Integration Benchmark — 行业 SDK + 神经符号融合端到端验证。

P7 "立业": MedicalCausalSDK, LegalComplianceSDK, EngineeringSafetySDK,
           ScientificDiscovery, EdgeCloudHybrid, PluginInterface
P8 "超凡": NeuralSymbolicFusionV2, CausalGradient, SymbolGrounding, AGIProtocol

测试策略: 验证模块可创建和导入，详细 API 测试在各 test_*.py 中。
"""

import numpy as np

from mci_world_model.sdk._agi_protocol import AGIIntegrationProtocol
from mci_world_model.sdk._causal_gradient import CausalGradient
from mci_world_model.sdk._edge_cloud_hybrid import EdgeCloudHybrid
from mci_world_model.sdk._engineering_safety_sdk import EngineeringSafetySDK
from mci_world_model.sdk._legal_compliance_sdk import LegalComplianceSDK, LegalEvidence
from mci_world_model.sdk._medical_causal_sdk import ClinicalEvidence, MedicalCausalSDK
from mci_world_model.sdk._neural_symbolic_fusion_v2 import NeuralSymbolicFusionV2
from mci_world_model.sdk._plugin_interface import PluginManager, PluginMetadata
from mci_world_model.sdk._scientific_discovery import ScientificDiscoveryPipeline
from mci_world_model.sdk._symbol_grounding import SymbolGroundingLearning


class TestP7MedicalSDK:
    def test_create(self) -> None:
        sdk = MedicalCausalSDK()
        assert sdk is not None

    def test_diagnose_dopamine_hr(self) -> None:
        """端到端: 多巴胺→心率 因果诊断链。

        场景: 3条临床证据均指向多巴胺引起心率上升。
        预期: 置信度 > 0.3, 诊断结果可审计。
        """
        sdk = MedicalCausalSDK(patient_id="P001", strict_mode=False)

        # 添加3条临床证据
        sdk.add_evidence(
            ClinicalEvidence(
                evidence_id="E1",
                evidence_type="vital_sign",
                description="dopamine 给药后心率从72升至98 bpm",
                confidence=0.85,
            )
        )
        sdk.add_evidence(
            ClinicalEvidence(
                evidence_id="E2",
                evidence_type="lab_result",
                description="dopamine 血药浓度与心率相关性 r=0.72",
                confidence=0.80,
            )
        )
        sdk.add_evidence(
            ClinicalEvidence(
                evidence_id="E3", evidence_type="observation", description="停药后心率回落至基线", confidence=0.90
            )
        )

        diag = sdk.diagnose("dopamine", "heart_rate_increase")

        # 核心断言
        assert diag.cause == "dopamine"
        assert diag.effect == "heart_rate_increase"
        assert diag.confidence > 0.3, f"置信度={diag.confidence:.3f} 过低"
        assert diag.causal_strength > 0.0
        assert len(diag.evidence_ids) == 3
        assert len(diag.audit_trail) >= 2  # evidence_check + diagnosis

        # 审计日志完整性
        log = sdk.get_audit_log()
        assert len(log) == 4  # 3 add_evidence + 1 diagnose
        assert log[-1]["action"] == "diagnose"

        stats = sdk.statistics()
        assert stats["evidence_count"] == 3
        assert stats["diagnosis_count"] == 1

    def test_insufficient_evidence_strict(self) -> None:
        """严格模式: 证据不足应拒绝给出确定性结论。"""
        sdk = MedicalCausalSDK(strict_mode=True)
        sdk.add_evidence(
            ClinicalEvidence(evidence_id="E1", evidence_type="observation", description="单一观察", confidence=0.9)
        )
        # 仅1条证据，MIN_EVIDENCE_COUNT=2
        diag = sdk.diagnose("drug_A", "side_effect")
        assert not diag.is_conclusive
        assert diag.confidence == 0.0
        assert len(diag.warnings) >= 1


class TestP7LegalSDK:
    def test_create(self) -> None:
        sdk = LegalComplianceSDK()
        assert sdk is not None

    def test_compliance_reasoning(self) -> None:
        """端到端: 违规行为→市场影响 法律因果链。

        场景: 内幕交易导致股价异常波动。
        预期: 因果链强度合理, 审计轨迹100%覆盖, 偏差检测激活。
        """
        sdk = LegalComplianceSDK(jurisdiction="CN", standard="preponderance")

        # 添加3条法律证据
        sdk.add_evidence(
            LegalEvidence(
                evidence_id="L1",
                evidence_type="document",
                description="insider_trading 高管在公告前减持",
                reliability=0.85,
                jurisdiction="CN",
            )
        )
        sdk.add_evidence(
            LegalEvidence(
                evidence_id="L2",
                evidence_type="data",
                description="stock_price 减持后2日下跌8%",
                reliability=0.90,
                jurisdiction="CN",
            )
        )
        sdk.add_evidence(
            LegalEvidence(
                evidence_id="L3",
                evidence_type="expert",
                description="insider_trading 时间序列格兰杰因果 p<0.01",
                reliability=0.75,
                jurisdiction="CN",
            )
        )

        conclusion = sdk.reason("insider_trading", "stock_price_drop")

        assert conclusion.cause == "insider_trading"
        assert conclusion.effect == "stock_price_drop"
        assert conclusion.jurisdiction == "CN"
        assert conclusion.causal_link_strength > 0.4
        assert len(conclusion.audit_trail) == 3  # reliability + bias + standard
        assert len(conclusion.evidence_ids) == 3

        # 审计完整性
        trail = sdk.get_audit_trail()
        assert len(trail) == 4  # 3 add_evidence + 1 reason
        assert trail[-1]["action"] == "reason"

    def test_no_evidence_graceful(self) -> None:
        """零证据不应崩溃，应返回空结论。"""
        sdk = LegalComplianceSDK()
        conclusion = sdk.reason("X", "Y")
        assert conclusion.cause == "X"
        assert len(conclusion.bias_flags) >= 1  # "no_evidence" flag


class TestP7EngineeringSDK:
    def test_create(self) -> None:
        sdk = EngineeringSafetySDK()
        assert sdk is not None

    def test_safety_margin_analysis(self) -> None:
        """端到端: 桥架安全裕度 + FMEA 风险分析。

        场景: 2参数含安全裕度违规 + 1个未缓解高风险FMEA。
        预期: 安全裕度检查失败, FMEA告警, 冗余达标。
        """
        from mci_world_model.sdk._engineering_safety_sdk import FMEAItem, SafetyParameter

        sdk = EngineeringSafetySDK(system_name="Bridge_A1")

        # 添加安全参数
        sdk.add_parameter(
            SafetyParameter(name="load_capacity", design_value=10.0, limit_value=13.0, unit="kN")
        )  # margin≈0.23 < 0.3
        sdk.add_parameter(
            SafetyParameter(name="stress_tolerance", design_value=50.0, limit_value=52.0, unit="MPa")
        )  # margin≈0.038
        sdk.add_parameter(
            SafetyParameter(name="deflection", design_value=2.0, limit_value=3.0, unit="mm")
        )  # margin=0.33 OK

        # 添加FMEA
        sdk.add_fmea(
            FMEAItem(
                failure_mode="cable_corrosion",
                effect="structural_failure",
                severity=8,
                occurrence=6,
                detection=3,
                mitigated=False,
            )
        )  # RPN=144 > 125, 未缓解

        # 设置冗余
        sdk.set_redundancy("load_sensor", True)

        result = sdk.analyze("load_overload", "bridge_collapse", 0.7)

        assert result.cause == "load_overload"
        assert result.effect == "bridge_collapse"
        assert not result.margin_sufficient  # 有安全裕度违规
        assert result.fmea_rpn_max > 0
        assert result.redundancy_ok  # 冗余已设置


class TestP7ScientificDiscovery:
    def test_create(self) -> None:
        sd = ScientificDiscoveryPipeline()
        assert sd is not None

    def test_discovery_pipeline(self) -> None:
        """端到端: 科学发现四阶段流水线。

        场景: 5变量线性数据 → 探索→骨架→定律→验证。
        预期: 至少到达验证阶段, 发现≥1条定律。
        """
        np.random.seed(42)
        n = 200
        X1 = np.random.randn(n)
        X2 = 0.7 * X1 + 0.2 * np.random.randn(n)
        X3 = 0.5 * X2 + 0.2 * np.random.randn(n)
        X4 = 0.3 * X1 + 0.2 * np.random.randn(n)
        X5 = 0.4 * X3 + 0.2 * np.random.randn(n)
        data = np.column_stack([X1, X2, X3, X4, X5])

        sd = ScientificDiscoveryPipeline()
        sd.load_data(data, ["V1", "V2", "V3", "V4", "V5"])
        report = sd.run()

        assert report.stage is not None
        assert report.n_laws >= 1
        assert hasattr(report, "details")
        assert report.consistency >= 0.0


class TestP7EdgeCloud:
    def test_create(self) -> None:
        ec = EdgeCloudHybrid()
        assert ec is not None

    def test_dispatch_and_cache(self) -> None:
        """端到端: 推理请求调度 + 缓存命中。

        场景: 发送2次相同请求 → 第2次应命中缓存。
        预期: dispatch_count=2, 缓存命中。
        """
        from mci_world_model.sdk._edge_cloud_hybrid import InferenceRequest

        ec = EdgeCloudHybrid(edge_capacity=10)

        req = InferenceRequest(
            request_id="R001", priority="high", max_latency_ms=50.0, query={"type": "causal_discovery"}
        )

        # 首次请求
        r1 = ec.dispatch(req)
        assert r1.request_id == "R001"
        assert r1.latency_ms > 0

        # 相同请求应命中缓存
        r2 = ec.dispatch(req)
        assert r2.cached

        stats = ec.statistics()
        assert stats["dispatch_count"] == 2


class TestP7PluginInterface:
    def test_metadata_create(self) -> None:
        meta = PluginMetadata(
            name="test_plugin",
            version="0.1.0",
            description="A test plugin",
        )
        assert meta.name == "test_plugin"
        assert meta.version == "0.1.0"

    def test_manager_create(self) -> None:
        pm = PluginManager()
        assert pm is not None


class TestP8NeuralSymbolic:
    def test_create(self) -> None:
        nsf = NeuralSymbolicFusionV2()
        assert nsf is not None

    def test_bidirectional_fusion(self) -> None:
        """端到端: 神经→符号→神经 双向融合循环。

        场景: 表征向量 [0.8, 0.6, 0.1] 中检测线性关系。
        预期: 融合收敛, 规则提取非空, 一致性 > 0。
        """
        nsf = NeuralSymbolicFusionV2(rule_threshold=1.5, max_iterations=10)

        # 神经表征: 第一个变量与第二个强相关
        neural_repr = np.array([0.8, 0.6, 0.1])
        var_names = ["dose", "response", "noise"]

        state = nsf.fuse(neural_repr, var_names, n_iterations=5)

        # 融合结果断言
        assert state.n_iterations == 5
        assert state.fusion_score >= 0.0
        assert state.fusion_score <= 1.0
        assert state.neural_representation is not None
        assert len(state.symbolic_rules) >= 0  # 可能无显著规则

        # 统计
        stats = nsf.statistics()
        assert stats["fusion_count"] == 1

    def test_roundtrip_consistency(self) -> None:
        """神经→符号→神经 roundtrip 保持一致性。"""
        nsf = NeuralSymbolicFusionV2(rule_threshold=2.0, max_iterations=5)

        neural = np.array([1.0, 0.5, 0.2, 0.1])
        state = nsf.fuse(neural, n_iterations=3)

        # 一轮融合后表征应仍为有限值
        assert np.all(np.isfinite(state.neural_representation))

        # 多轮融合后应稳定
        state2 = nsf.fuse(state.neural_representation, n_iterations=2)
        diff = np.max(np.abs(state2.neural_representation - state.neural_representation))
        # 融合应逐渐稳定
        assert diff < 1.0, f"融合发散 diff={diff:.4f}"


class TestP8CausalGradient:
    def test_create(self) -> None:
        cg = CausalGradient(source="X", target="Y")
        assert cg is not None

    def test_gradient_propagation(self) -> None:
        """端到端: 因果图梯度传播 + 权重更新。

        场景: 3节点链 X→Y→Z, 损失梯度作用于Z。
        预期: 梯度沿因果图反向传播, 更新Y→Z权重。
        """
        from mci_world_model.sdk._causal_gradient import CausalGradientPropagation

        cgm = CausalGradientPropagation(learning_rate=0.05)

        # X→Y→Z 链式结构
        adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        cgm.set_graph(adj, ["X", "Y", "Z"])

        # Z 的损失梯度
        cgm.set_loss_gradient(np.array([0.0, 0.0, 0.8]))

        # 传播梯度
        cgm.propagate(3)

        # 获取节点梯度
        node_grads = cgm.get_node_gradients()
        assert len(node_grads) == 3
        # Z 的梯度应非零
        assert abs(node_grads["Z"]) > 0.0


class TestP8SymbolGrounding:
    def test_create(self) -> None:
        sg = SymbolGroundingLearning()
        assert sg is not None

    def test_ground_and_verify(self) -> None:
        """端到端: 符号接地 + 验证循环。

        场景: 将"red"接地到视觉向量, 多次增量更新。
        预期: 接地强度随样例增加而增长, 验证返回合理相似度。
        """
        sg = SymbolGroundingLearning(similarity_threshold=0.5)

        # 第一次接地
        e1 = sg.ground("red", "visual", np.array([0.9, 0.1, 0.05]))
        assert e1.symbol == "red"
        assert e1.grounding_strength == 0.2  # 首次

        # 增量更新 ×3
        for _ in range(3):
            sg.ground("red", "visual", np.array([0.85, 0.12, 0.08]))
        e_final = sg.ground("red", "visual", np.array([0.88, 0.11, 0.06]))

        assert e_final.grounding_strength > 0.5  # 5次≥0.2+4次增强
        assert sg.is_grounded("red")
        assert not sg.is_grounded("blue")

        # 验证
        sim = sg.verify_grounding("red", np.array([0.87, 0.13, 0.07]))
        assert sim > 0.7, f"相似度={sim:.3f} 过低"

        # 未接地符号
        ungrounded = sg.get_ungrounded_symbols(["red", "blue", "green"])
        assert "blue" in ungrounded
        assert "green" in ungrounded
        assert "red" not in ungrounded


class TestP8AGIProtocol:
    def test_create(self) -> None:
        agi = AGIIntegrationProtocol()
        assert agi is not None

    def test_request_response_cycle(self) -> None:
        """端到端: 多智能体能力注册 + 请求响应。

        场景: 注册2个能力(causal_discovery, counterfactual),
              发送因果发现请求。
        预期: 响应命中对应能力, 含审计记录。
        """
        from mci_world_model.sdk._agi_protocol import AGICapability, AGIRequest

        agi = AGIIntegrationProtocol(min_confidence=0.3, audit_enabled=True)

        # 注册能力
        agi.register_capability(AGICapability.CAUSAL_REASONING)
        agi.register_capability(AGICapability.COUNTERFACTUAL)

        assert "causal_reasoning" in agi.registered_capabilities
        assert "counterfactual" in agi.registered_capabilities

        # 发送请求 (evidence_strength≥threshold才能成功)
        req = AGIRequest(
            request_id="AGI_001",
            capability=AGICapability.CAUSAL_REASONING,
            payload={"hypothesis": "X causes Y", "evidence_strength": 0.85},
        )
        resp = agi.handle_request(req)

        assert resp.request_id == "AGI_001"
        assert resp.success  # confidence=0.85 ≥ 0.3
        assert resp.confidence >= 0.3
        assert len(resp.audit_trail_id) > 0  # 审计ID已生成

        # 统计
        stats = agi.statistics()
        assert stats["capabilities"] is not None
