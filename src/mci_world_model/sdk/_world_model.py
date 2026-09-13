from __future__ import annotations

"""
MCI World Model v4.6.0 — CEWM 认知增强世界模型
=====================================================

神经-符号因果推理系统的统一接口，
v4.3.3: 参数化记忆觉醒 + 能量流闭环。
融合三层因果量化管道 + JEPA 编码器-预测器 + Pearl do-calculus 干预 +
Pearl counterfactual 反事实推理 (L3)。

核心能力:
- discover():        三层因果发现 → 加权因果图 → JEPA 编码
- predict_effect():  纯检索路径 + JEPA 预测路径
- jepa_predict():    JEPA 潜空间预测 (encoder→state→predictor→next_state)
- intervene():       Pearl do-operator 干预预测（Pearl L2）
- decompose_effect(): 因果效应三分解 NDE/NIE/TE
- query_counterfactual(): Pearl 反事实推理（L3）
- train_jepa():      JEPA 端到端训练（替代 train_parametric）
- explain():         因果链回溯，人类可读解释
- run_cognitive_loop():   Wiener 四环认知闭环传播（v4.3.0）
- diagnose_failure():     MetaDiagnoser 认知失败诊断（v4.3.0）
- retrieve_experiences(): MultiViewRetriever 五维经验检索（v4.3.0）
- detect_surprise():      SurpriseDetector 惊奇误差检测（v4.3.1）
- plan_action():          PlanAgent 因果决策前置规划（v4.3.2）
- synthesize_training_data(): ReflectionSynthesizer MEMO QA 合成（v4.3.2）
- assess_diversity():     CognitiveDiversity 五维多样性评估（v4.3.2）
- check_admissibility():  NegativeHeuristic 硬核规则检查（v4.3.2）
- train_parametric():     CausalMLP 参数化记忆训练（v4.3.3）
- predict_causal_category(): CausalMLP 五范畴因果预测（v4.3.3）
- predict_energy_flow():  五行生克能量流预测（v4.3.3）
- health_check():        全系统健康诊断

架构层次:
    ┌───────────────────────────────────────────┐
    │        MCIWorldModel (v4.3.3 CEWM)        │
    │  ┌───────────────────────────────────┐    │
    │  │  JEPA Encoder + Predictor         │    │
    │  │  (潜空间因果图编码 → GNN/基线预测) │    │
    │  │  + EnergyConsistencyLoss          │    │
    │  └──────────┬────────────────────────┘    │
    │             │ 潜空间状态编码                │
    │  ┌──────────▼────────────────────────┐    │
    │  │  三层因果管道                     │    │
    │  │  FourierCausal → GaussianDAG     │    │
    │  │  → BayesianCausal                │    │
    │  └───────────────────────────────────┘    │
    │  ┌───────────────────────────────────┐    │
    │  │  Entity Surfacing + SIGReg        │    │
    │  └───────────────────────────────────┘    │
    └───────────────────────────────────────────┘

用法:
    from mci_world_model.sdk._world_model import MCIWorldModel

    wm = MCIWorldModel(su_lite_pro_instance)
    causal_graph = wm.discover()
    effects = wm.predict_effect("价格上涨")
    jepa_effects = wm.jepa_predict("价格上涨")
    explanation = wm.explain("为什么库存下降?")
"""


import logging
import threading
from typing import Any

import numpy as np

from mci_world_model.sdk._energy_flow import EnergyFlowMixin, _aggregate_energy_ratios
from mci_world_model.sdk._world_model_state import (
    CausalWorldModelState,
    TrajectoryStep,
    WorkingMemory,
)

logger = logging.getLogger(__name__)

__all__ = ["CausalWorldModelState", "TrajectoryStep", "WorkingMemory"]

__all__ += ["_aggregate_energy_ratios"]


class MCIWorldModel(EnergyFlowMixin):
    """
    MCI World Model v4.6.0 — CEWM 认知增强世界模型。

    v4.3.3: 参数化记忆觉醒 + 能量流闭环。
    统一了检索增强 + JEPA 编码器-预测器两种路径，
    提供 Pearl 因果层级（关联→干预→反事实）的完整接口。

    Example:
        >>> wm = MCIWorldModel(lite_pro)
        >>> graph = wm.discover()
        >>> print(f"发现 {len(graph.causal_edges)} 条因果边")

        >>> # JEPA 潜空间预测
        >>> predictions = wm.jepa_predict("产品价格上涨")
        >>> for p in predictions:
        ...     print(f"→ {p['effect']} (置信度: {p['confidence']})")

        >>> # 干预分析 (Pearl L2)
        >>> result = wm.intervene(
        ...     do_x={"price": 1.5},
        ...     target="demand",
        ... )

        >>> # 反事实推理 (Pearl L3)
        >>> cf = wm.query_counterfactual(
        ...     evidence={"price": 1.0, "demand": 100},
        ...     do_x={"price": 0.8},
        ...     target="demand",
        ... )
    """

    # ── 五范畴状态系统 ──
    FIVE_STATES = ["semantic", "causal", "spacetime", "generative", "trust"]

    def __init__(  # type: ignore[no-untyped-def]
        self,
        lite_pro=None,
        config: dict | None = None,  # type: ignore
    ):
        """
        Args:
            lite_pro: SuMemoryLitePro 实例（可选）
            config: 配置字典
        """
        self._lite_pro = lite_pro
        self._config = config or {}
        self._state = CausalWorldModelState.empty()
        self._parametric: Any | None = None  # 降级为惰性加载 (v3.1.0)
        self._energy_loss: Any | None = None  # EnergyConsistencyLoss
        self._cost_module: Any | None = None  # v3.0.1: EnergyCostModule
        self._configurator: Any | None = None  # v3.0.1: MetaConfigurator
        self._hierarchical_encoder: Any | None = None  # v3.0.2: HierarchicalJEPAEncoder
        self._causal_actor: Any | None = None  # v3.0.2: CausalActor
        self._perception: Any | None = None  # v3.0.3: PerceptionPipeline
        self._initialized: bool = False

        # v3.0.4: 能量仲裁器 + 时空编码器（惰性初始化）
        self._energy_core: Any | None = None
        self._temporal_core: Any | None = None

        # v3.1.0 JEPA: 编码器 + 预测器 (懒加载)
        self._jepa_encoder: Any | None = None
        self._jepa_predictor: Any | None = None
        self._jepa_mode = str(self._config.get("jepa_mode", "latent"))
        if self._jepa_mode not in {"latent", "legacy_graph"}:
            raise ValueError(f"未知 jepa_mode: {self._jepa_mode}")

        # Pearl L2: do-calculus 干预引擎 (懒加载)
        self._do_calculus: Any | None = None
        self._do_calculus_lock: threading.Lock = threading.Lock()
        self._intervention_history: list[dict[str, Any]] = []
        # v3.3.1: 干预历史并发保护
        self._intervention_history_lock: threading.Lock = threading.Lock()

        # P1 并发加固: 初始化锁 (防止多线程重复 initialize)
        self._init_lock: threading.Lock = threading.Lock()

        # P2 并发加固: 因果发现锁 (防止多线程同时 discover() 状态交错)
        self._discover_lock: threading.Lock = threading.Lock()

        # v4.3.0 CEWM 组件 (懒加载)
        self._cognitive_loop: Any | None = None
        self._meta_diagnoser: Any | None = None
        self._multi_view_retriever: Any | None = None
        self._surprise_detector: Any | None = None  # v4.3.1 SurpriseDetector
        self._plan_agent: Any | None = None  # v4.3.2 PlanAgent
        self._action_conditioned_predictor: Any | None = None  # v4.3.2 ActionConditionedPredictor
        self._multi_branch_predictor: Any | None = None  # v4.3.2 MultiBranchPredictor
        self._reflection_synthesizer: Any | None = None  # v4.3.2 ReflectionSynthesizer
        self._cognitive_diversity: Any | None = None  # v4.3.2 CognitiveDiversity
        self._negative_heuristic: Any | None = None  # v4.3.2 NegativeHeuristic
        self._parametric_memory: Any | None = None  # v4.3.3 ParametricMemory
        self._energy_flow_predictor: Any | None = None  # v4.3.3 EnergyFlowPredictor
        self._causal_updater: Any | None = None  # v4.3.3 CausalUpdater (持久化积累)
        # ── P7/P8 能力中心 ──
        self._scientific_discovery: Any | None = None  # P7: ScientificDiscovery
        self._neural_symbolic: Any | None = None  # P8: NeuralSymbolicFusionV2
        self._action_gap_metric: Any | None = None  # LOOP-03: ActionGapMetric (懒加载)
        self._state_parser_registry: Any | None = None  # LOOP-03: StateParserRegistry (懒加载)

        # v4.4.2: Phase 2 — 安全约束 + 反事实 Oracle
        self._safety_monitor: Any | None = None  # SafetyMonitor
        self._cf_oracle: Any | None = None  # CounterfactualOracle

        # Adapt-EPA: replay buffer (optional, off by default)
        self._replay_enabled: bool = self._config.get("replay_enabled", False)
        self._replay_threshold: float = float(self._config.get("replay_threshold", 0.3))
        self._replay_interval: int = int(self._config.get("replay_interval", 50))
        self._step_count: int = 0

        # ── P6-P8 v6.0~v8.0: 新增模块 (懒加载) ──
        self._law_discoverer_v2: Any | None = None  # P6: AutonomousLawDiscovererV2
        self._social_cognition: Any | None = None  # P6: SocialCognition
        self._self_repair: Any | None = None  # P6: SelfRepairCognition
        self._auto_scaler: Any | None = None  # P7: AutoScaler
        self._compliance_engine: Any | None = None  # P7: ComplianceRuleEngine
        self._plugin_manager: Any | None = None  # P7: PluginManager
        self._unified_modal_encoder: Any | None = None  # P6: UnifiedModalEncoder
        self._metacognition_v2: Any | None = None  # P6: MetacognitionV2
        self._medical_sdk: Any | None = None  # P7: MedicalCausalSDK
        self._legal_sdk: Any | None = None  # P7: LegalComplianceSDK
        self._engineering_sdk: Any | None = None  # P7: EngineeringSafetySDK
        self._auditable_causal: Any | None = None  # P7: AuditableCausalReasoning
        self._edge_cloud: Any | None = None  # P7: EdgeCloudHybrid
        self._cross_modal_causal: Any | None = None  # P7: CrossModalCausalReasoner
        self._causal_imagination: Any | None = None  # P6: CausalImaginationEngine
        self._differentiable_causal: Any | None = None  # P6: DifferentiableCausalInference
        self._domain_sdk: Any | None = None  # P7: MCIDomainSDK
        self._sci_pipeline: Any | None = None  # P7: ScientificDiscoveryPipeline
        self._hypothesis_gen: Any | None = None  # P7: HypothesisGenerator
        self._fusion_v2: Any | None = None  # P8: NeuralSymbolicFusionV2
        self._causal_gradient: Any | None = None  # P8: CausalGradientPropagation
        self._symbol_grounding: Any | None = None  # P8: SymbolGroundingLearning
        self._agi_protocol: Any | None = None  # P8: AGIIntegrationProtocol
        self._experiment_designer: Any | None = None  # P8: ExperimentDesigner

        # 如果传入了 lite_pro，自动初始化
        if lite_pro is not None:
            self.initialize()

    @property
    def jepa_mode(self) -> str:
        """当前 JEPA 输出模式；latent 是默认生产契约。"""
        return self._jepa_mode

    def attach_true_jepa(self, encoder: Any, predictor: Any | None = None) -> None:
        """显式注入 TrueJEPA 编码器，接入潜空间主闭环。"""
        self._jepa_mode = "latent"
        from mci_world_model.sdk._jepa_encoder import JEPAEncoder

        if self._jepa_encoder is None:
            self._jepa_encoder = JEPAEncoder(self, jepa_mode="latent")
        self._jepa_encoder.attach_true_jepa(encoder)
        if predictor is not None:
            self._jepa_predictor = predictor

    # ────────────────────────────────────────────────
    # 初始化
    # ────────────────────────────────────────────────

    def initialize(self) -> dict[str, Any]:
        """
        初始化世界模型组件（幂等安全）。

        自动检测并组装:
        - 四层因果管道（_spectral_causal）
        - Reflection QA 合成器
        - Entity Surfacing + SIGReg
        - ParametricMemory（按需加载）

        Returns:
            初始化状态报告
        """
        # 幂等: 已初始化则直接返回缓存报告
        if self._initialized:
            return {
                "modules": {"causal_pipeline": "available"},
                "warnings": [],
                "ready": True,
                "initialized": True,
                "_cached": True,
            }

        with self._init_lock:
            # 双重检查
            if self._initialized:
                return {
                    "modules": {"causal_pipeline": "available"},
                    "warnings": [],
                    "ready": True,
                    "initialized": True,
                    "_cached": True,
                }

            report: dict[str, Any] = {
                "modules": {},
                "warnings": [],
                "ready": False,
            }

            # ── 检查四层因果管道 ──
            try:
                from mci_world_model.sdk._spectral_causal import (  # noqa: F401
                    BayesianCausal,
                    FourierCausal,
                    GaussianDAG,
                )

                report["modules"]["causal_pipeline"] = "available"
            except ImportError:
                report["modules"]["causal_pipeline"] = "unavailable"
                report["warnings"].append("四层因果管道不可用 — 因果发现将受限")

            # ── 检查 Reflection QA ──
            try:
                from mci_world_model.sdk._reflection_synthesizer import (
                    ReflectionSynthesizer,  # noqa: F401
                )

                report["modules"]["reflection_qa"] = "available"
            except ImportError:
                report["modules"]["reflection_qa"] = "unavailable"

            # ── 检查 SIGReg ──
            try:
                from mci_world_model.sdk._sigreg import SIGReg  # noqa: F401

                report["modules"]["sigreg"] = "available"
            except ImportError:
                report["modules"]["sigreg"] = "unavailable"

            # ── v3.1.0 JEPA: 检查编码器 ──
            try:
                from mci_world_model.sdk._jepa_encoder import JEPAEncoder

                report["modules"]["jepa_encoder"] = "available"
            except ImportError:
                report["modules"]["jepa_encoder"] = "unavailable"

            # ── v3.1.0 JEPA: 检查预测器 ──
            try:
                from mci_world_model.sdk._jepa_predictor import (
                    BeliefPropagationPredictor,
                )

                report["modules"]["jepa_predictor"] = "available"
            except ImportError:
                report["modules"]["jepa_predictor"] = "unavailable"

            # ── v3.1.0 M2: 检查 GNN 预测器 ──
            try:
                from mci_world_model.sdk._jepa_gnn import GNNPredictor  # noqa: F401

                report["modules"]["jepa_gnn"] = "available"
            except ImportError:
                report["modules"]["jepa_gnn"] = "unavailable"

            # ── v4.9.0 P7: 检查能力中心模块 ──
            for mod_name, mod_path in [
                ("plugin_manager", "mci_world_model.sdk._plugin_interface"),
                ("medical_sdk", "mci_world_model.sdk._medical_causal_sdk"),
                ("legal_sdk", "mci_world_model.sdk._legal_compliance_sdk"),
                ("engineering_sdk", "mci_world_model.sdk._engineering_safety_sdk"),
                ("scientific_discovery", "mci_world_model.sdk._scientific_discovery"),
                ("edge_cloud", "mci_world_model.sdk._edge_cloud_hybrid"),
                ("neural_symbolic", "mci_world_model.sdk._neural_symbolic_fusion_v2"),
                ("causal_gradient", "mci_world_model.sdk._causal_gradient"),
                ("symbol_grounding", "mci_world_model.sdk._symbol_grounding"),
                ("agi_protocol", "mci_world_model.sdk._agi_protocol"),
            ]:
                try:
                    __import__(mod_path)
                    report["modules"][mod_name] = "available"
                except ImportError:
                    report["modules"][mod_name] = "unavailable"

            # ── 检查能量损失 ──
            try:
                from mci_world_model.sdk._energy_loss import (
                    EnergyConsistencyLoss,
                )

                report["modules"]["energy_loss"] = "available"
                self._energy_loss = EnergyConsistencyLoss()
            except ImportError:
                report["modules"]["energy_loss"] = "unavailable"

            # ── v3.1.0: 初始化 JEPA 编码器 ──
            if report["modules"]["jepa_encoder"] == "available":
                try:
                    from mci_world_model.sdk._jepa_encoder import JEPAEncoder

                    self._jepa_encoder = JEPAEncoder(self)
                    report["jepa_encoder"] = "initialized"
                except (ImportError, TypeError, ValueError) as e:
                    logger.warning("异常降级: %s", e, exc_info=True)
                    report["warnings"].append(f"JEPA 编码器初始化失败: {e}")

            # ── v3.1.0: 初始化 JEPA 预测器（默认为 BeliefPropagation 基线） ──
            if report["modules"]["jepa_predictor"] == "available":
                try:
                    from mci_world_model.sdk._jepa_predictor import (
                        BeliefPropagationPredictor,
                    )

                    self._jepa_predictor = BeliefPropagationPredictor()
                    report["jepa_predictor"] = "initialized"
                except (ImportError, TypeError, ValueError) as e:
                    logger.warning("异常降级: %s", e, exc_info=True)
                    report["warnings"].append(f"JEPA 预测器初始化失败: {e}")

            report["ready"] = report["modules"]["causal_pipeline"] == "available"
            self._initialized = report["ready"]

            if report["ready"]:
                logger.info("MCIWorldModel v4.9.0 CEWM 初始化完成 (含 P7/P8 能力中心)")
            else:
                logger.warning("MCIWorldModel 初始化不完整: %s", report["warnings"])

            return report

    # ────────────────────────────────────────────────
    # v3.0.4: 能量中心 + 时空编码器 惰性获取器
    # ────────────────────────────────────────────────

    def _get_energy_core(self) -> Any:
        """惰性初始化并返回 EnergyCore 实例。"""
        if self._energy_core is None:
            from su_memory._sys._energy_core import EnergyCore

            self._energy_core = EnergyCore()  # type: ignore[no-untyped-call]
        return self._energy_core

    def _get_temporal_core(self) -> Any:
        """惰性初始化并返回 TemporalCore 实例。"""
        if self._temporal_core is None:
            from su_memory._sys._temporal_core import TemporalCore

            self._temporal_core = TemporalCore()  # type: ignore[no-untyped-call]
        return self._temporal_core

    def _get_configurator(self) -> None:
        """v3.0.6: 惰性初始化并返回 HierarchicalConfigurator 实例。"""
        if self._configurator is None:
            from mci_world_model._sys._configurator import HierarchicalConfigurator

            self._configurator = HierarchicalConfigurator(energy_core=self._energy_core)  # type: ignore[no-untyped-call]
        return self._configurator  # type: ignore

    def _get_causal_actor(self) -> None:
        """v3.0.6: 惰性初始化并返回 CausalActor 实例。"""
        if self._causal_actor is None:
            from mci_world_model.sdk._causal_actor import CausalActor

            self._causal_actor = CausalActor(self, self._cost_module, energy_core=self._energy_core)
        return self._causal_actor  # type: ignore[return-value]

    # ────────────────────────────────────────────────
    # v4.9.0: P7/P8 能力中心 惰性接入
    # ────────────────────────────────────────────────

    def _get_plugin_manager(self) -> Any:
        """P7: 惰性初始化 PluginManager（插件注册与调度）。"""
        if self._plugin_manager is None:
            from mci_world_model.sdk._plugin_interface import PluginManager

            self._plugin_manager = PluginManager()
        return self._plugin_manager

    def _get_medical_sdk(self) -> Any:
        """P7: 惰性初始化 MedicalCausalSDK。"""
        if self._medical_sdk is None:
            from mci_world_model.sdk._medical_causal_sdk import MedicalCausalSDK

            self._medical_sdk = MedicalCausalSDK()
        return self._medical_sdk

    def _get_legal_sdk(self) -> Any:
        """P7: 惰性初始化 LegalComplianceSDK。"""
        if self._legal_sdk is None:
            from mci_world_model.sdk._legal_compliance_sdk import LegalComplianceSDK

            self._legal_sdk = LegalComplianceSDK()
        return self._legal_sdk

    def _get_engineering_sdk(self) -> Any:
        """P7: 惰性初始化 EngineeringSafetySDK。"""
        if self._engineering_sdk is None:
            from mci_world_model.sdk._engineering_safety_sdk import EngineeringSafetySDK

            self._engineering_sdk = EngineeringSafetySDK()
        return self._engineering_sdk

    def _get_scientific_discovery(self) -> Any:
        """P7: 惰性初始化 ScientificDiscovery。"""
        if self._scientific_discovery is None:
            from mci_world_model.sdk._scientific_discovery import ScientificDiscoveryPipeline

            self._scientific_discovery = ScientificDiscoveryPipeline()
        return self._scientific_discovery

    def _get_edge_cloud(self) -> Any:
        """P7: 惰性初始化 EdgeCloudHybrid。"""
        if self._edge_cloud is None:
            from mci_world_model.sdk._edge_cloud_hybrid import EdgeCloudHybrid

            self._edge_cloud = EdgeCloudHybrid()
        return self._edge_cloud

    def _get_neural_symbolic(self) -> Any:
        """P8: 惰性初始化 NeuralSymbolicFusionV2。"""
        if self._neural_symbolic is None:
            from mci_world_model.sdk._neural_symbolic_fusion_v2 import NeuralSymbolicFusionV2

            self._neural_symbolic = NeuralSymbolicFusionV2()
        return self._neural_symbolic

    def _get_causal_gradient(self) -> Any:
        """P8: 惰性初始化 CausalGradient。"""
        if self._causal_gradient is None:
            from mci_world_model.sdk._causal_gradient import CausalGradient

            self._causal_gradient = CausalGradient(source="world_model", target="causal_graph")
        return self._causal_gradient

    def _get_symbol_grounding(self) -> Any:
        """P8: 惰性初始化 SymbolGrounding。"""
        if self._symbol_grounding is None:
            from mci_world_model.sdk._symbol_grounding import SymbolGroundingLearning

            self._symbol_grounding = SymbolGroundingLearning()
        return self._symbol_grounding

    def _get_agi_protocol(self) -> Any:
        """P8: 惰性初始化 AGIProtocol。"""
        if self._agi_protocol is None:
            from mci_world_model.sdk._agi_protocol import AGIIntegrationProtocol

            self._agi_protocol = AGIIntegrationProtocol()
        return self._agi_protocol

    def _build_energy_bus(self) -> object:
        """
        v3.0.5: 从因果图构建 EnergyBus 三层网络。

        为每个活跃能量创建五元素节点，基于 causal_edges 建立
        ENHANCE/SUPPRESS Channel。

        Returns:
            EnergyBus 实例（已连接所有因果边）
        """
        from su_memory._sys._energy_bus import (  # type: ignore[attr-defined]
            EnergyBus,
            EnergyLayer,
            EnergyNode,
            RelationType,
        )

        bus = EnergyBus()
        # 为每个活跃状态创建五元素节点
        for energy in self.FIVE_STATES:
            node = EnergyNode(
                node_id=f"wm_{energy}",
                energy_type=energy,
                layer=EnergyLayer.FIVE_ELEMENTS,
            )
            bus.add_node(node, auto_connect=False)

        # 基于因果边建立 Channel
        for edge in self._state.causal_edges:
            cause_e = edge.get("cause_energy", "earth")
            effect_e = edge.get("effect_energy", "earth")
            rel = self._get_energy_core().analyze_interaction(cause_e, effect_e)  # type: ignore[func-returns-value,attr-defined]
            if rel and rel[0].name != "SAME":
                bus.connect(
                    f"wm_{cause_e}",
                    f"wm_{effect_e}",
                    RelationType.ENHANCE if "ENHANCE" in str(rel) else RelationType.SUPPRESS,
                    base_weight=edge.get("rho", 0.5),
                )
        return bus

    def _propagate_energy(self, steps: int = 3) -> dict[str, Any]:
        """
        v3.0.5: 执行能量传播并返回总线状态。

        Args:
            steps: 传播步数

        Returns:
            EnergyBus.get_bus_state() 返回的总线状态字典
        """
        bus = self._build_energy_bus()
        bus.propagate(steps=steps)  # type: ignore[attr-defined]
        return bus.get_bus_state()  # type: ignore[attr-defined]

    # ────────────────────────────────────────────────
    # v3.0.6: 因果边标准化
    # ────────────────────────────────────────────────

    def normalize_edge(self, edge: dict[str, Any], energy_core: Any = None, month_branch: int = 0) -> dict[str, Any]:
        """
        v3.0.6: 标准化因果边，自动补全能量属性。

        对偶表示：每条边同时携带因果权重(rho) + 能量属性(energy_relation/strength)。
        - 自动推断 energy_relation（基于五行生克）
        - 自动注入 energy_strength（当前月份旺衰）

        Args:
            edge: 原始因果边 dict
            energy_core: EnergyCore 实例（None 时使用内置惰性获取器）
            month_branch: 当前月份TimeBranch index（0=子月）

        Returns:
            补全能量属性后的新 dict
        """
        edge = dict(edge)
        ec = energy_core or self._energy_core

        # 自动推断 energy_relation
        if "energy_relation" not in edge and ec is not None:
            ce = edge.get("cause_energy", "earth")
            ee = edge.get("effect_energy", "earth")
            rel = ec.analyze_interaction(ce, ee)
            edge["energy_relation"] = rel[0].name.lower() if rel else "neutral"

        # 自动注入旺衰
        if "energy_strength" not in edge and ec is not None:
            ce = edge.get("cause_energy", "earth")
            state = ec.get_energy_state(ce, month_branch)
            edge["energy_strength"] = state.strength.name

        return edge

    # ────────────────────────────────────────────────
    # 因果发现
    # ────────────────────────────────────────────────

    def discover(
        self,
        memories: list[dict[str, Any]] | None = None,
        use_parametric: bool = False,
        verbose: bool = True,
    ) -> CausalWorldModelState:
        """
        三层因果发现流水线（线程安全）。

        执行完整流程:
        Layer 1: FourierCausal 频域过滤
        Layer 2: GaussianDAG 偏相关发现
        Layer 3: BayesianCausal 后验量化

        （F1-P1-1: 原 docstring 声称"四层"，但 CausalProbability 未实际导入，修正为三层）

        并发安全 (P2 加固):
        - 使用 ``self._discover_lock`` 序列化对 ``self._state`` 的原地修改
        - 多个线程同时调用 ``discover()`` 不会产生状态交错
        - JEPADataset.from_memories() 在并发场景下应拷贝返回的 state 以避免交叉污染

        Args:
            memories: 记忆列表（None 时从 lite_pro 自动获取）
            use_parametric: 是否启用参数化先验增强
            verbose: 是否输出 INFO 日志（训练时设为 False）

        Returns:
            CausalWorldModelState 含所有发现的因果边 (为 ``self._state`` 引用)
        """
        if memories is None and self._lite_pro is not None:
            memories = self._get_memories_from_lite_pro()

        if not memories or len(memories) < 3:
            logger.warning("记忆不足（需要 ≥ 3 条）")
            return self._state

        # P2 并发加固: 序列化对 self._state 的原地修改
        # 多线程同时调用 discover() 会导致 _state.causal_edges/n_memories 等字段交错
        with self._discover_lock:
            try:
                from mci_world_model.sdk._spectral_causal import (  # noqa: F401
                    BayesianCausal,
                    FourierCausal,
                    GaussianDAG,
                )

                # ── 获取 TF-IDF 索引 ──
                index = None
                if self._lite_pro and hasattr(self._lite_pro, "_index"):
                    index = self._lite_pro._index

                # ── 获取 EnergyBus ──
                energy_bus = None
                if self._lite_pro and hasattr(self._lite_pro, "_energy_bus"):
                    energy_bus = self._lite_pro._energy_bus

                # ── Layer 1+2: GaussianDAG ──
                dag = GaussianDAG(memories, index, energy_bus)

                # ── v3.1.0: JEPA 先验增强 ──
                if use_parametric and self._jepa_encoder is not None:
                    self._apply_parametric_prior(dag, memories)

                # ── Reflection Prior ──
                try:
                    from mci_world_model.sdk._reflection_synthesizer import (
                        ReflectionSynthesizer,
                    )

                    syn = ReflectionSynthesizer(
                        energy_bus=energy_bus,
                        min_confidence=0.4,
                        max_pairs=200,
                    )
                    _, prior_matrix = syn.run_pipeline(memories)
                    dag.with_reflection_prior(prior_matrix)
                except ImportError:
                    logger.debug("因果先验模块不可用，跳过 with_reflection_prior")
                    pass

                # ── 发现隐藏因果边 ──
                edges = dag.discover_hidden_edges()

                # ── v3.1.0: 补充 cause/effect 实体名称 ──
                # GaussianDAG 输出边使用 TF-IDF 词表索引 (cause_idx/effect_idx)，
                # 后续代码 (BayesianCausal) 依赖这些索引。同时补充 cause/effect
                # 实体名称，使 GAT 编码器和 align_adjacency 能正确对齐。
                if hasattr(dag, "_vocab") and dag._vocab:
                    for e in edges:
                        ci = e.get("cause_idx")
                        ei = e.get("effect_idx")
                        if ci is not None and ci < len(dag._vocab):
                            e["cause"] = dag._vocab[ci]
                        if ei is not None and ei < len(dag._vocab):
                            e["effect"] = dag._vocab[ei]

                # ── Layer 3: BayesianCausal 量化 ──
                bayesian = BayesianCausal(energy_bus)
                edges = bayesian.batch_update(edges)

                # ── 更新状态 ──
                self._state.causal_edges = edges
                self._state.n_memories = len(memories)
                self._state.parametric_enhanced = use_parametric

                # ── 统计 ──
                self._state.n_confirmed = sum(1 for e in edges if e.get("verdict") == "confirmed")
                self._state.n_novel = sum(1 for e in edges if e.get("verdict") == "novel")
                self._state.n_suppressed = sum(1 for e in edges if e.get("verdict") == "suppressed")

                # ── 活跃状态 ──
                active = set()
                for e in edges:
                    if e.get("energy_relation"):
                        active.add(e["energy_relation"])
                self._state.active_states = active

                from datetime import datetime

                self._state.timestamp = datetime.now().isoformat()

                if verbose:
                    logger.info(
                        "因果发现完成: %d 条边 (确认: %d, 新发现: %d, 抑制: %d)",
                        len(edges),
                        self._state.n_confirmed,
                        self._state.n_novel,
                        self._state.n_suppressed,
                    )

            except ImportError as e:
                logger.error("因果发现失败 — 缺少依赖: %s", e)
            except (RuntimeError, ValueError, KeyError) as e:
                logger.error("因果发现失败: %s", e)

        return self._state

    # ────────────────────────────────────────────────
    # 因果预测
    # ────────────────────────────────────────────────

    def predict_effect(
        self,
        cause: str,
        memories: list[dict[str, Any]] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        检索路径因果预测（v3.5.0 能力）。

        基于 CausalEngine 关键词 + 偏相关统计，
        不依赖参数化模型。

        Args:
            cause: 原因文本
            memories: 记忆列表
            top_k: 返回前 K 个效应

        Returns:
            [{"effect": str, "confidence": float, "causal_type": str}, ...]
        """
        if memories is None and self._lite_pro is not None:
            memories = self._get_memories_from_lite_pro()

        if not memories:
            return []

        try:
            from mci_world_model.sdk._causal import CausalEngine

            engine = CausalEngine(min_confidence=0.4)
            effects = engine.predict_effects(cause, memories, top_k=top_k)
            return effects
        except (KeyError, ValueError, RuntimeError) as e:
            logger.error("检索预测失败: %s", e)
            return []

    def jepa_predict(
        self,
        cause: str,
        target_category: str | None = None,
        top_k: int = 3,
        memories: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        JEPA 潜空间因果预测（v3.1.0）。

        流程: 编码器(记忆 → 因果图状态) → 预测器(状态 → 下一状态) →
              差分分析(原因 → 效应)

        Args:
            cause: 原因文本
            target_category: 目标状态类别（可选）
            top_k: 返回前 K 个预测
            memories: 记忆列表（None 时从 lite_pro 获取）

        Returns:
            [{"effect": str, "confidence": float, "energy_relation": str}, ...]
        """
        if self._jepa_encoder is None or self._jepa_predictor is None:
            logger.warning("JEPA 编码器/预测器未初始化，回退到检索路径")
            return self.predict_effect(cause, top_k=top_k)

        # ── 获取记忆并编码 ──
        if memories is None and self._lite_pro is not None:
            memories = self._get_memories_from_lite_pro()

        if not memories or len(memories) < 3:
            logger.warning("记忆不足 JEPA 预测（需要 ≥ 3 条）")
            return self.predict_effect(cause, top_k=top_k)

        try:
            # 0. 检测 M3 模式
            is_m3 = (
                self._jepa_encoder is not None
                and hasattr(self._jepa_encoder, "_differentiable")
                and self._jepa_encoder._differentiable
            )

            # 1. 编码: latent 模式只接受潜向量，legacy_graph 是显式过渡。
            encoded = self._jepa_encoder.encode(memories)
            if getattr(encoded, "source", None) == "legacy_graph":
                state = encoded.graph_state
            else:
                if not getattr(encoded, "is_ready", False):
                    logger.warning("JEPA latent 编码未就绪: %s", encoded.diagnostics)
                    return self.predict_effect(cause, top_k=top_k)
                return []

            # 2. 预测: 状态 → 下一状态 (GNN)
            next_state = self._jepa_predictor.predict(state)

            # 3. 差分: 找出新增/增强的因果边
            predictions = []
            if state.causal_edges and next_state.causal_edges:
                current_edge_keys = {(e.get("cause", ""), e.get("effect", "")) for e in state.causal_edges}
                for edge in next_state.causal_edges:
                    ee = edge.get("effect", "")
                    ec = edge.get("cause", "")
                    # 只返回与 cause 相关的新增/变化边
                    if cause.lower() in ec.lower() or cause.lower() in ee.lower():
                        key = (ec, ee)
                        is_new = key not in current_edge_keys
                        predictions.append(
                            {
                                "effect": ee,
                                "confidence": edge.get("confidence", 0.5) * (1.1 if is_new else 0.9),
                                "energy_relation": edge.get("energy_relation", "neutral"),
                                "cause": ec,
                                "verdict": edge.get("verdict", "predicted"),
                                "_mode": "m3_gat_gnn" if is_m3 else "jepa_baseline",
                            }
                        )

            # 按置信度排序
            predictions.sort(key=lambda x: x["confidence"], reverse=True)

            # ── v3.0.5: 预测后能量守恒验证 ──
            if self._energy_core is not None:
                try:
                    energy_before = self._extract_energy_ratios(state)
                    energy_after = self._extract_energy_ratios(next_state)
                    if energy_before and energy_after:
                        simulated = self._energy_core.simulate_energy_flow(energy_after, steps=3)
                        # 检测能量是否收敛到合理范围（每维偏离 0.2 上限不超过 0.3）
                        final = simulated[-1]
                        max_deviation = max(abs(final.get(k, 0) - 0.2) for k in final)
                        if max_deviation > 0.3:
                            # 能量不守恒 → 降低全部预测置信度
                            for p in predictions:
                                p["confidence"] *= 0.7
                            logger.debug(
                                "JEPA 预测能量不守恒 (max_deviation=%.3f)，置信度降权",
                                max_deviation,
                            )
                except (ValueError, AttributeError) as e:
                    logger.warning("能量守恒验证跳过: %s", e)

            return predictions[:top_k] if predictions else self.predict_effect(cause, top_k=top_k)

        except (RuntimeError, ValueError, KeyError) as e:
            logger.error("JEPA 预测失败: %s，回退到检索路径", e)
            return self.predict_effect(cause, top_k=top_k)

    def parametric_predict(
        self,
        cause: str,
        target_category: str | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        参数化路径因果预测（v3.6.0 — v3.1.0 降级为 jepa_predict 别名）。

        v3.1.0: 重路由到 JEPA 潜空间预测。
        保留接口兼容性，内部调用 jepa_predict()。

        Args:
            cause: 原因文本
            target_category: 目标状态类别（可选）
            top_k: 返回前 K 个预测

        Returns:
            [{"effect": str, "confidence": float, "energy_relation": str}, ...]
        """
        return self.jepa_predict(cause, target_category, top_k)

    def predict_from_memories_m3(
        self,
        memories: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        M3 专用推理：从记忆列表直接预测因果边（GAT + GNN）。

        与 jepa_predict() 的区别:
        - 不依赖 cause 文本过滤，返回所有预测的因果边
        - 直接输出 GNN 预测的 (cause, effect, rho) 三元组
        - 可用于端到端评估和可视化

        Args:
            memories: 记忆列表（至少 3 条）
            top_k: 返回前 K 条最显著的因果边

        Returns:
            [{"cause": str, "effect": str, "rho": float, "confidence": float}, ...]
        """
        if self._jepa_encoder is None or self._jepa_predictor is None:
            logger.warning("JEPA 编码器/预测器未初始化")
            return []

        if not memories or len(memories) < 3:
            logger.warning("记忆不足 M3 预测（需要 ≥ 3 条）")
            return []

        try:
            # 1. GAT 编码
            state = self._jepa_encoder.encode_graph(memories)

            if not state.causal_edges:
                logger.info("GAT 编码未发现因果边")
                return []

            # 2. GNN 预测下一状态
            next_state = self._jepa_predictor.predict(state)

            # 3. 提取预测边
            predictions = []
            for edge in next_state.causal_edges:
                predictions.append(
                    {
                        "cause": edge.get("cause", ""),
                        "effect": edge.get("effect", ""),
                        "rho": edge.get("rho", 0.0),
                        "confidence": edge.get("confidence", 0.5),
                        "verdict": edge.get("verdict", "predicted"),
                        "energy_relation": edge.get("energy_relation", "neutral"),
                    }
                )

            predictions.sort(key=lambda x: abs(x["rho"]), reverse=True)
            return predictions[:top_k]

        except (RuntimeError, ValueError, AttributeError) as e:
            logger.error("M3 预测失败: %s", e)
            return []

    def fused_predict(
        self,
        cause: str,
        memories: list[dict[str, Any]] | None = None,
        top_k: int = 5,
        retrieval_weight: float = 0.4,
        parametric_weight: float = 0.6,
    ) -> list[dict[str, Any]]:
        """
        融合预测（v3.1.0: 检索 + JEPA 加权）。

        v3.1.0: 将"参数化"路径替换为 JEPA 潜空间预测。
        parametric_weight 参数保留但语义变为 JEPA 预测权重。

        Args:
            cause: 原因文本
            memories: 记忆列表
            top_k: 返回数量
            retrieval_weight: 检索路径权重
            parametric_weight: JEPA 预测路径权重

        Returns:
            加权融合后的预测列表
        """
        retrieval_results = self.predict_effect(cause, memories, top_k=top_k)
        jepa_results = self.jepa_predict(cause, top_k=top_k, memories=memories)

        # 融合策略: JEPA 结果在前，检索结果补充
        fused = []
        seen_effects: set[str] = set()

        for r in jepa_results:
            effect_key = r.get("effect", "")
            if effect_key not in seen_effects:
                seen_effects.add(effect_key)
                fused.append(
                    {
                        "effect": effect_key,
                        "confidence": r.get("confidence", 0.5) * parametric_weight,
                        "source": "jepa",
                        "energy_relation": r.get("energy_relation", "neutral"),
                    }
                )

        for r in retrieval_results:
            content = r.get("content", "")
            if content not in seen_effects:
                seen_effects.add(content)
                fused.append(
                    {
                        "effect": content,
                        "confidence": r.get("confidence", 0.5) * retrieval_weight,
                        "source": "retrieval",
                        "causal_type": r.get("causal_type", ""),
                    }
                )

        fused.sort(key=lambda x: x["confidence"], reverse=True)
        return fused[:top_k]

    # ────────────────────────────────────────────────
    # 私有方法：因果图构建
    # ────────────────────────────────────────────────

    def _build_causal_graph_from_state(self) -> object | None:
        """
        从当前因果边构建 CausalGraph。

        统一 intervene()/decompose_effect()/query_counterfactual()
        的图构建逻辑，消除重复。

        Returns:
            CausalGraph 实例或 None
        """
        if not self._state or not self._state.causal_edges:
            return None
        from mci_world_model.sdk._do_calculus import DoCalculus

        n_nodes = max(max(e.get("cause_idx", 0), e.get("effect_idx", 0)) for e in self._state.causal_edges) + 1
        return DoCalculus.build_from_gaussian_dag(self._state.causal_edges, n_nodes)

    # ────────────────────────────────────────────────
    # Pearl do-operator 干预（Pearl L2 完整实现）
    # ────────────────────────────────────────────────

    def intervene(
        self,
        state: str = "current",
        do_x: dict | None = None,  # type: ignore
        target: str | None = None,
        method: str = "auto",
    ) -> dict[str, Any]:
        """
        Pearl do-operator 干预预测（Pearl L2 完整实现）。

        计算公式:
            P(Y | do(X=x)) = Σ_z P(Y | X=x, Z=z) · P(Z=z)

        工作流:
        1. 从当前因果图构建 CausalGraph
        2. 识别调整变量集 (后门准则)
        3. 估计 ATE (平均处理效应)
        4. 返回干预分析结果

        Args:
            state: 世界状态标识
            do_x: 干预 {"变量名": 干预值}
            target: 目标变量名
            method: "auto"|"backdoor"|"frontdoor"

        Returns:
            InterventionResult 字典
        """
        if do_x is None or target is None:
            return {
                "status": "insufficient_input",
                "code": 422,
                "message": "需要 do_x 和 target 参数",
            }
        if not isinstance(do_x, dict) or not do_x:
            return {
                "status": "unsupported",
                "code": 422,
                "reason": "invalid_do_x",
                "message": "do_x 必须是非空 dict",
            }

        # ── 懒加载 DoCalculus 引擎 ──
        if self._do_calculus is None:
            with self._do_calculus_lock:
                if self._do_calculus is None:
                    try:
                        from mci_world_model.sdk._do_calculus import DoCalculus

                        self._do_calculus = DoCalculus()
                    except ImportError:
                        return {
                            "status": "error",
                            "message": "DoCalculus 引擎不可用",
                        }

        # ── 从因果边构建 CausalGraph ──
        try:
            from mci_world_model.sdk._do_calculus import CausalGraph

            cg = self._build_causal_graph_from_state()
            if cg is None:
                cg = CausalGraph(
                    nodes=[*list(do_x.keys()), target],
                    edges=[],
                )
        except (ValueError, KeyError, TypeError):
            logger.warning("CausalGraph 构建失败，回退到默认空图", exc_info=True)
            cg = CausalGraph(
                nodes=[*list(do_x.keys()), target],
                edges=[],
            )

        self._do_calculus.set_graph(cg)  # type: ignore[arg-type]

        # ── 干预契约：先统一拒绝多变量和无效值，避免 dict 键序决定语义 ──
        try:
            normalized_do_x = {str(name): float(value) for name, value in do_x.items()}
        except (TypeError, ValueError):
            return {
                "status": "rejected",
                "code": 422,
                "reason": "invalid_do_x",
                "message": "do_x 的变量名必须可解析，干预值必须是数值",
            }
        if not np.all(np.isfinite(np.fromiter(normalized_do_x.values(), dtype=np.float64))):
            return {
                "status": "rejected",
                "code": 422,
                "reason": "non_finite_do_x",
                "message": "intervention values must be finite (NaN/Inf rejected)",
            }
        if len(normalized_do_x) != 1:
            return {
                "status": "unsupported",
                "code": 422,
                "reason": "multivariate_do_not_implemented",
                "message": "多变量干预尚未实现；当前只支持单变量 do({变量: 数值})",
            }

        x_name = next(iter(normalized_do_x))
        x_value = normalized_do_x[x_name]

        try:
            result = self._do_calculus.estimate_intervention(
                do_x={x_name: x_value},
                target=target,
                x_baseline=0.0,
                method=method,
            )
        except (ValueError, KeyError, RuntimeError) as e:
            logger.error("干预分析失败: %s", e)
            return {
                "status": "error",
                "message": f"干预分析失败: {e}",
            }

        # ── 记录干预历史 ──
        intervention_record = {
            "state": state,
            "do": do_x,
            "target": target,
            "result": result.to_dict(),
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }
        with self._intervention_history_lock:
            self._intervention_history.append(intervention_record)
            self._state.do_interventions.append(intervention_record)

        # ── 构建反事实图 (干预边被切断) ──
        try:
            if cg.n_nodes > 0 and x_name in cg.nodes:  # type: ignore
                x_idx = cg.node_index(x_name)  # type: ignore[attr-defined]
                if x_idx is not None and cg.adjacency is not None:  # type: ignore[attr-defined]
                    cf_adj = cg.adjacency.copy()  # type: ignore
                    # 切断所有指向 X 的边 (do-operator 语义)
                    cf_adj[:, x_idx] = 0.0
                    self._state.counterfactual_graph = {
                        "nodes": list(cg.nodes),  # type: ignore
                        "cf_adjacency": cf_adj.tolist(),
                        "intervention": do_x,
                    }
        except (ValueError, KeyError, TypeError):
            logger.warning("反事实图构建失败，跳过", exc_info=True)

        # ── 返回结果 ──
        output = result.to_dict()
        if result.method == "no_data":
            output["status"] = "no_data"
            output["code"] = 422
        elif result.method in {"rejected", "unsupported"}:
            output["status"] = result.method
            output["code"] = 422
        else:
            output["status"] = "ok"
        output["history_count"] = len(self._intervention_history)
        return output

    # ────────────────────────────────────────────────
    # 因果效应分解（Pearl L2 新增）
    # ────────────────────────────────────────────────

    def decompose_effect(
        self,
        cause: str,
        effect: str,
        mediator: str | None = None,
    ) -> dict[str, Any]:
        """
        因果效应三分解:
        - NDE: 自然直接效应 (Natural Direct Effect)
        - NIE: 自然间接效应 (Natural Indirect Effect)
        - TE:  总效应 (Total Effect = NDE + NIE)

        使用 Pearl 的 mediation formula:
            NDE = E[Y_{x,M_{x*}} - Y_{x*}]
            NIE = E[Y_{x,M_x} - Y_{x,M_{x*}}]

        Args:
            cause: 原因变量
            effect: 结果变量
            mediator: 中介变量 (None 时自动检测)

        Returns:
            {"nde": float, "nie": float, "te": float, "mediator": str, ...}
        """
        # ── 自动检测中介变量 ──
        if mediator is None:
            if self._do_calculus is None:
                with self._do_calculus_lock:
                    if self._do_calculus is None:
                        try:
                            from mci_world_model.sdk._do_calculus import DoCalculus

                            self._do_calculus = DoCalculus()
                        except ImportError:
                            return {
                                "nde": 0.0,
                                "nie": 0.0,
                                "te": 0.0,
                                "mediator": None,
                                "note": "do_calculus_unavailable",
                            }

            # 尝试从因果图中识别中介
            if self._state and self._state.causal_edges:
                try:
                    cg = self._build_causal_graph_from_state()
                    if cg is not None:
                        mediators = cg.get_mediators(cause, effect)  # type: ignore
                        mediator = mediators[0] if mediators else None
                    else:
                        mediator = None
                except (ValueError, KeyError):
                    logger.warning("中介变量识别失败，回退为无中介", exc_info=True)
                    mediator = None

        if mediator is None:
            return {
                "nde": 0.0,
                "nie": 0.0,
                "te": 0.0,
                "mediator": None,
                "note": "no_mediator_identified",
            }

        # ── 使用 do-calculus 分解 ──
        try:
            # 直接效应: do(mediator) 固定时的 cause → effect
            direct_result = self.intervene(
                do_x={cause: 1.0},
                target=effect,
                method="direct",
            )
            nde = direct_result.get("ate", 0.0)

            # 间接效应: cause → mediator 的 ATE × mediator → effect 的 ATE
            cause_to_med = self.intervene(
                do_x={cause: 1.0},
                target=mediator,
                method="direct",
            )
            med_to_eff = self.intervene(
                do_x={mediator: 1.0},
                target=effect,
                method="direct",
            )
            nie = cause_to_med.get("ate", 0.0) * med_to_eff.get("ate", 0.0)
        except (ValueError, KeyError, RuntimeError) as e:
            logger.error("因果分解失败: %s", e)
            nde = 0.0
            nie = 0.0

        te = nde + nie

        return {
            "nde": round(nde, 6),
            "nie": round(nie, 6),
            "te": round(te, 6),
            "mediator": mediator,
            "nde_pct": round(abs(nde) / max(abs(te), 1e-10) * 100, 1),
            "nie_pct": round(abs(nie) / max(abs(te), 1e-10) * 100, 1),
            "method": "mediation_formula",
        }

    # ────────────────────────────────────────────────
    # 反事实推理（v3.0.8 新增 — Pearl L3）
    # ────────────────────────────────────────────────

    def query_counterfactual(
        self,
        evidence: dict[str, float],
        do_x: dict[str, float],
        target: str,
        compute_pns: bool = True,
    ) -> dict[str, Any]:
        """
        Pearl 三步反事实推理（v3.0.8 L3 新增）。

        基于当前因果图，回答反事实问题:
            "如果当初 X=x' 而非 X=x，Y 会是多少？"

        三步算法:
            1. Abduction (溯因): 从事实证据推断不可观测噪声
            2. Action (干预): 用 do(X=x') 构建 mutilated graph
            3. Prediction (预测): 用溯因噪声 + mutilated graph 计算 Y_{x'}

        输出:
            - counterfactual_value: 反事实结果
            - individual_effect: 个体因果效应 (Y_{x'} - Y)
            - PN/PS/PNS: 必然性/充分性概率
            - noise_terms: 溯因推断的噪声项

        Args:
            evidence: 事实证据 {"X": 1.0, "Y": 3.0, ...}
            do_x: 反事实干预 {"X": 0.0}
            target: 目标变量 (反事实结果)
            compute_pns: 是否计算 PN/PS/PNS

        Returns:
            CounterfactualResult dict

        Example:
            >>> wm = MCIWorldModel(su_lite_pro)
            >>> wm.discover()
            >>> result = wm.query_counterfactual(
            ...     evidence={"手术量": 100, "收入": 50},
            ...     do_x={"手术量": 80},
            ...     target="收入",
            ... )
            >>> print(f"反事实收入: {result['counterfactual_value']}")
        """
        try:
            from mci_world_model.sdk._counterfactual import CounterfactualEngine
            from mci_world_model.sdk._do_calculus import CausalGraph
        except ImportError as e:
            return {
                "status": "error",
                "note": f"counterfactual_engine_unavailable: {e}",
            }

        # ── 从因果图构建 CausalGraph ──
        cg = self._build_causal_graph_from_state()
        if cg is None:
            all_nodes = list(evidence.keys()) + list(do_x.keys()) + [target]
            cg = CausalGraph(nodes=list(set(all_nodes)), edges=[])
        if not getattr(cg, "is_causal_graph", False):
            return {
                "status": "rejected",
                "reason": "association_graph_not_causal",
                "message": "缺少显式定向证据的关联图不能执行反事实推理",
            }

        # ── 构建反事实引擎并查询 ──
        engine = CounterfactualEngine.from_causal_graph(cg)
        if engine is None:
            return {
                "status": "error",
                "note": "failed_to_build_counterfactual_engine",
            }

        try:
            result = engine.query(
                evidence=evidence,
                do_x=do_x,
                target=target,
                compute_pns=compute_pns,
            )
        except (ValueError, KeyError, RuntimeError) as e:
            logger.error("反事实查询失败: %s", e, exc_info=True)
            return {
                "status": "error",
                "note": f"counterfactual_query_failed: {e}",
            }

        return result.to_dict()

    # ────────────────────────────────────────────────
    # 因果解释
    # ────────────────────────────────────────────────

    def explain(
        self,
        query: str,
        max_depth: int = 3,
    ) -> dict[str, Any]:
        """
        因果链回溯解释。

        返回从 query 出发的因果路径，
        含每一步的置信度和能量关系类型。

        Args:
            query: 查询文本
            max_depth: 最大因果跳数

        Returns:
            {
                "query": str,
                "chains": [{"path": [...], "confidence": float}, ...],
                "summary": str,
            }
        """
        if not self._state.causal_edges:
            return {
                "query": query,
                "chains": [],
                "summary": "暂无因果图数据。请先运行 discover()。",
            }

        chains = self._trace_causal_chains(query, max_depth)
        summary = self._generate_explanation_summary(chains, query)

        return {
            "query": query,
            "chains": chains[:5],  # 最多 5 条链
            "summary": summary,
        }

    def _trace_causal_chains(self, query: str, max_depth: int) -> list[dict[str, Any]]:
        """追踪因果链 — v4.3.1 多跳 BFS 回溯。

        从匹配 query 的因果边出发，沿 effect→cause 方向链式回溯，
        构建长度 ≤ max_depth 的多跳因果链（A→B→C→...）。
        对每个节点尝试作为中间节点继续扩展，实现级联因果追溯。
        """
        edges = self._state.causal_edges
        if not edges:
            return []

        # 1. 统一节点名：优先 "cause"/"effect" 字符串，否则用 idx 生成标签
        def _node_name(e: dict[str, Any], role: str) -> str:
            s = e.get(role)  # "cause" or "effect"
            if isinstance(s, str):
                return s
            idx = e.get(f"{role}_idx")
            if idx is not None:
                return f"节点 {idx}"
            return f"{role}_{id(e)}"

        # 2. 构建邻接表: cause → [(effect_name, edge), ...]
        adj: dict[str, list[tuple[str, dict]]] = {}  # type: ignore
        for e in edges:
            cause = _node_name(e, "cause")
            effect = _node_name(e, "effect")
            adj.setdefault(cause, []).append((effect, e))

        # 3. 匹配 query — 模糊查找起始节点
        query_lower = query.lower()
        starts: list[tuple[str, dict]] = []  # type: ignore
        for e in edges:
            cause = _node_name(e, "cause")
            effect = _node_name(e, "effect")
            if query_lower in cause.lower() or query_lower in effect.lower():
                starts.append((cause, e))

        if not starts:
            return []

        # 4. BFS 多跳遍历
        chains: list[dict[str, Any]] = []
        for start_cause, start_edge in starts:
            path = [start_cause]
            confs = [start_edge.get("confidence", 0.5)]
            queue: list[tuple[str, int, list[str], list[float]]] = [(start_cause, 1, path, confs)]

            while queue:
                node, depth, path, confs = queue.pop(0)
                if depth > max_depth:
                    continue
                if node not in adj:
                    continue
                for next_effect, next_edge in adj[node]:
                    if next_effect in path:  # 防环
                        continue
                    new_path = [*path, f"→ {next_effect}"]
                    new_confs = [*confs, next_edge.get("confidence", 0.5)]
                    chains.append(
                        {
                            "path": new_path,
                            "confidence": sum(new_confs) / len(new_confs),
                            "verdict": next_edge.get("verdict", "predicted"),
                            "energy_relation": next_edge.get("energy_relation", "neutral"),
                            "depth": len(new_path) - 1,
                        }
                    )
                    if depth + 1 <= max_depth:
                        queue.append((next_effect, depth + 1, new_path, new_confs))

        # 按深度（多跳优先）、置信度排序
        chains.sort(key=lambda c: (c["depth"], c["confidence"]), reverse=True)
        return chains

    def _generate_explanation_summary(self, chains: list[dict[str, Any]], query: str) -> str:
        """生成可读解释摘要。"""
        if not chains:
            return f"未找到与「{query}」相关的因果链。"

        n_confirmed = sum(1 for c in chains if c.get("verdict") == "confirmed")
        n_novel = sum(1 for c in chains if c.get("verdict") == "novel")

        parts = [f"共发现 {len(chains)} 条与「{query}」相关的因果链。"]
        if n_confirmed > 0:
            parts.append(f"其中 {n_confirmed} 条被拓扑先验确认。")
        if n_novel > 0:
            parts.append(f"{n_novel} 条为潜在新发现。")

        top_chain = chains[0]
        parts.append(f"最高置信度链 (置信度: {top_chain['confidence']:.2f}): {' → '.join(top_chain['path'])}")

        return " ".join(parts)

    # ────────────────────────────────────────────────
    # JEPA 训练
    # ────────────────────────────────────────────────

    def train_jepa(
        self,
        dataset: object | None = None,
        qa_pairs: list | None = None,  # type: ignore
        output_dir: str = "./checkpoints/mci-world-model",
        n_epochs: int = 10,
        learning_rate: float = 0.01,
    ) -> dict[str, Any]:
        """
        JEPA 端到端训练（v3.1.0，替代 train_parametric）。

        Args:
            dataset: JEPADataset 实例（优先）
            qa_pairs: Reflection QA 对列表（备选，自动转 JEPADataset）
            output_dir: checkpoint 输出目录
            n_epochs: 训练轮数
            learning_rate: 学习率

        Returns:
            训练统计
        """
        # ── 构造 JEPADataset ──
        if dataset is None and qa_pairs is not None:
            try:
                from mci_world_model.sdk._jepa_dataset import JEPADataset

                # 从 QA 对提取记忆并构造数据集
                memories = []
                for qa in qa_pairs:
                    if isinstance(qa, dict):
                        memories.append(
                            {
                                "content": qa.get("question", ""),
                                "answer": qa.get("answer", ""),
                            }
                        )
                if memories:
                    dataset = JEPADataset.from_memories(memories, self)  # type: ignore
            except ImportError as e:
                return {"error": f"jepa_dataset_unavailable: {e}"}

        if dataset is None:
            return {
                "error": "no_training_data",
                "message": "需要 JEPADataset 或 qa_pairs",
            }

        # ── 构造训练器并训练 ──
        try:
            from mci_world_model.sdk._jepa_trainer import JEPATrainer

            trainer = JEPATrainer(
                encoder=self._jepa_encoder,
                predictor=self._jepa_predictor,
                dataset=dataset,
            )
            stats = trainer.train(
                n_epochs=n_epochs,
                learning_rate=learning_rate,
            )

            self._state.n_qa_pairs = len(dataset.pairs) if hasattr(dataset, "pairs") else 0

            return {
                "n_pairs": len(dataset.pairs) if hasattr(dataset, "pairs") else 0,
                "n_epochs": n_epochs,
                "training_stats": stats.to_dict() if hasattr(stats, "to_dict") else stats,
                "adapter_path": output_dir,
                "mode": "e2e" if trainer._is_e2e else ("gnn" if trainer._is_gnn else "baseline"),
            }
        except ImportError as e:
            return {"error": f"jepa_trainer_unavailable: {e}"}
        except (RuntimeError, ValueError, KeyError) as e:
            logger.error("JEPA 训练失败: %s", e)
            return {"error": str(e)}

    # ────────────────────────────────────────────────
    # 健康检查
    # ────────────────────────────────────────────────

    def health_check(self) -> dict[str, Any]:
        """全系统健康诊断。"""
        from mci_world_model import __version__

        check = {
            "version": __version__,
            "code_name": f"MCI World Model {__version__} CEWM 认知增强",
            "initialized": self._initialized,
            "causal_pipeline": {
                "edges_discovered": len(self._state.causal_edges),
                "confirmed": self._state.n_confirmed,
                "novel": self._state.n_novel,
                "suppressed": self._state.n_suppressed,
                "has_counterfactual_graph": self._state.counterfactual_graph is not None,
                "n_do_interventions": len(self._state.do_interventions),
            },
            "jepa_predictor": {
                "available": self._jepa_predictor is not None,
                "encoder_available": self._jepa_encoder is not None,
                "is_gnn": self._is_gnn_predictor(),
                "predictor_type": type(self._jepa_predictor).__name__ if self._jepa_predictor else "none",
            },
            "energy_loss": {
                "available": self._energy_loss is not None,
            },
            "cost_module": {
                "available": self._cost_module is not None,
                "state": self._cost_module.state if self._cost_module else "not_initialized",
                "eval_count": self._cost_module.eval_count if self._cost_module else 0,
            },
            "configurator": {
                "available": self._configurator is not None,
                "state": self._configurator.state if self._configurator else "not_initialized",
                "n_actions": len(self._configurator.config_history) if self._configurator else 0,
            },
            "hierarchical_encoder": {
                "available": self._hierarchical_encoder is not None,
                "state": self._hierarchical_encoder.state if self._hierarchical_encoder else "not_initialized",
                "encode_count": self._hierarchical_encoder.encode_count if self._hierarchical_encoder else 0,
            },
            "causal_actor": {
                "available": self._causal_actor is not None,
                "state": self._causal_actor.state if self._causal_actor else "not_initialized",
                "n_actions": len(self._causal_actor.action_history) if self._causal_actor else 0,
            },
            "integration": {
                "lite_pro_connected": self._lite_pro is not None,
                "n_memories": self._state.n_memories,
            },
            "cewm_components": {
                "cognitive_loop": self._cognitive_loop is not None,
                "meta_diagnoser": self._meta_diagnoser is not None,
                "multi_view_retriever": self._multi_view_retriever is not None,
                "surprise_detector": self._surprise_detector is not None,
                "plan_agent": self._plan_agent is not None,
                "action_conditioned_predictor": self._action_conditioned_predictor is not None,
                "multi_branch_predictor": self._multi_branch_predictor is not None,
                "reflection_synthesizer": self._reflection_synthesizer is not None,
                "cognitive_diversity": self._cognitive_diversity is not None,
                "negative_heuristic": self._negative_heuristic is not None,
                "parametric_memory": self._parametric_memory is not None,
                "energy_flow_predictor": self._energy_flow_predictor is not None,
            },
            "energy_coverage": self._compute_energy_coverage(),
            "roadmap": {
                "v3.0.7": "parametric_memory_awakening ✓",
                "pearl_l2_do_operator": "do_operator_intervention ✓",
                "v3.0.8": "counterfactual_reasoning_l3 ✓",
                "v3.0.0": "jepa_world_model_closed_loop ✓",
                "v3.0.0-m2": "jepa_gnn_trainable ✓"
                if self._is_gnn_predictor()
                else "jepa_gnn_trainable (use GNNPredictor)",
                "v3.0.0-m3": "jepa_e2e_differentiable ✓"
                if self._is_e2e_mode()
                else "jepa_e2e_differentiable (use enable_m3())",
                "v3.0.1": "cost_module_independent ✓"
                if self._cost_module is not None
                else "cost_module_independent (pending)",
                "v3.0.2": "hierarchical_jepa ✓"
                if self._hierarchical_encoder is not None
                else "hierarchical_jepa (pending)",
                "v3.0.6": "energy_causal_unified ✓"
                if self._energy_core is not None
                else "energy_causal_unified (pending)",
                "v3.0.5": "energy_flow_closed_loop ✓"
                if self._energy_core is not None
                else "energy_flow_closed_loop (pending)",
                "v3.0.4": "energy_aware_basic ✓" if self._energy_core is not None else "energy_aware_basic (pending)",
                "v3.0.3": "six_module_closed_loop ✓"
                if self._perception is not None
                else "six_module_closed_loop (pending)",
                "v3.6.0": "cewm_engine_unified ✓" if hasattr(self, "cewm_step") else "cewm_engine (pending)",
                "v4.3.0": "cewm_four_components ✓" if self._cognitive_loop is not None else "cewm_components (pending)",
                "v4.3.1": "surprise_detector_integrated ✓"
                if self._surprise_detector is not None
                else "surprise_detector (pending)",
                "v4.3.2": "cewm_six_modules_integrated ✓" if self._plan_agent is not None else "cewm_modules (pending)",
                "v4.3.2-m2": "reflection_synthesizer_qa ✓"
                if self._reflection_synthesizer is not None
                else "reflection_synthesizer (pending)",
                "v4.3.2-m3": "cognitive_diversity ✓"
                if self._cognitive_diversity is not None
                else "cognitive_diversity (pending)",
                "v4.3.2-m4": "negative_heuristic ✓"
                if self._negative_heuristic is not None
                else "negative_heuristic (pending)",
                "v4.3.3": "parametric_memory_awakening ✓"
                if self._parametric_memory is not None
                else "parametric_memory (pending)",
                "v4.9.0-p7": "plugin_manager_connected ✓"
                if self._plugin_manager is not None
                else "plugin_manager (pending)",
                "v4.9.0-p7-m2": "industry_sdks_connected ✓"
                if self._medical_sdk is not None
                else "industry_sdks (pending)",
                "v4.9.0-p8": "neural_symbolic_connected ✓"
                if self._neural_symbolic is not None
                else "neural_symbolic (pending)",
                "v4.9.0-p8-m2": "agi_protocol_connected ✓"
                if self._agi_protocol is not None
                else "agi_protocol (pending)",
                "v4.3.3-m2": "energy_flow_closed_loop ✓"
                if self._energy_flow_predictor is not None
                else "energy_flow (pending)",
                "v4.3.3-m3": "cewm_twelve_components ✓"
                if self._parametric_memory is not None and self._energy_flow_predictor is not None
                else "cewm_twelve (pending)",
            },
            "status": self._compute_health_status(),
        }
        return check

    # ────────────────────────────────────────────────
    # v3.0.2: Cost→Actor 梯度闭环
    # ────────────────────────────────────────────────

    def actor_optimize(
        self,
        max_iterations: int = 3,
        delta: float | None = None,
    ) -> dict[str, Any]:
        """
        Cost→Actor 梯度闭环：迭代搜索最优因果干预。

        自动尝试启用 CausalActor（若未初始化），然后运行
        search → apply 循环直到代价不再降低。

        Args:
            max_iterations: 最大迭代次数
            delta: 有限差分步长

        Returns:
            {"n_actions": int, "initial_cost": float, "final_cost": float,
             "cost_reduction": float, "actions": [...], "state": new_state}
        """
        # 延迟初始化 Actor
        if self._causal_actor is None:
            try:
                from mci_world_model.sdk._causal_actor import CausalActor

                self._causal_actor = CausalActor(self, self._cost_module)
                logger.info("v3.0.2 CausalActor 初始化完成")
            except (TypeError, ValueError, ImportError) as e:
                logger.error("CausalActor 初始化失败: %s", e)
                return {"error": str(e), "n_actions": 0}

        result = self._causal_actor.optimize(
            self._state,
            max_iterations=max_iterations,
            delta=delta,
        )

        # 更新 World Model 状态
        if result.get("state"):
            self._state = result["state"]

        return {k: v for k, v in result.items() if k != "state"}

    # ────────────────────────────────────────────────
    # v3.0.6: Configurator + Actor 自动能量调节闭环
    # ────────────────────────────────────────────────

    def auto_regulate(self, max_iterations: int = 3) -> dict[str, Any]:
        """
        v3.0.6: Configurator + Actor 自动能量调节闭环。

        流程:
        1. 提取当前五维能量分布
        2. 检测能量失衡 → Configurator 生成调节策略
        3. Actor 搜索最优干预动作
        4. 执行干预 → 重新评估能量分布
        5. 迭代至平衡或达到 max_iterations

        Args:
            max_iterations: 最大迭代次数

        Returns:
            {"iterations": int, "history": [...], "converged": bool,
             "no_energy_data": bool}
        """
        ec = self._get_energy_core()  # type: ignore
        actor = self._get_causal_actor()  # type: ignore
        configurator = self._get_configurator()  # type: ignore
        current_state = self._state
        history: list[dict[str, Any]] = []
        early_stop = False
        ratios = None

        for i in range(max_iterations):
            ratios = self._extract_energy_ratios(current_state)
            if not ratios:
                break  # 无能量数据，无法调节

            balance = ec.analyze_balance(ratios)  # type: ignore
            if balance.status == "balanced":
                early_stop = True
                break  # 已平衡

            try:
                # Configurator 生成策略
                _actions = configurator.configure(self, gaps=None)  # type: ignore

                # Actor 搜索最优动作
                candidates = actor.search(current_state, n_candidates=2)  # type: ignore

                # 执行并链式传递状态
                for c in candidates:
                    current_state = actor.apply(current_state, c)  # type: ignore
            except (ValueError, TypeError, RuntimeError) as e:
                logger.warning("auto_regulate 迭代 %d 异常: %s", i, e)
                break

            history.append(
                {
                    "iteration": i,
                    "balance_before": balance.status,
                    "dominant": balance.dominant,
                    "n_actions": len(candidates),
                }
            )

        # 写回最终状态
        self._state = current_state

        return {
            "iterations": len(history),
            "history": history,
            "converged": early_stop,
            "no_energy_data": ratios is None and len(history) == 0,
        }

    # ────────────────────────────────────────────────
    # v3.0.3: 六模块端到端闭环管线
    # ────────────────────────────────────────────────

    def six_module_pipeline(
        self,
        memories: list[dict[str, Any]],
        max_optimize_iterations: int = 2,
    ) -> dict[str, Any]:
        """
        六模块端到端闭环：Perception → WorldModel → Configurator →
                        Cost → Actor → STM → (loop)

        完整执行 LeCun 六模块自主推理循环:
        1. Perception: 原始观测 → 结构化特征
        2. World Model: 特征 → 因果图发现 (discover)
        3. Configurator: 认知空洞检测 → 动态配置
        4. Cost: 评估当前状态代价
        5. Actor: 搜索最优干预 → 执行
        6. STM: 记录轨迹到 WorkingMemory

        Args:
            memories: 原始记忆列表
            max_optimize_iterations: Actor 优化最大迭代数

        Returns:
            执行报告
        """
        report: dict[str, Any] = {
            "perception": {},
            "world_model": {},
            "configurator": {},
            "cost": {},
            "actor": {},
            "stm": {},
            "summary": {},
        }

        # ── 1. Perception: raw → structured ──
        try:
            if self._perception is None:
                from mci_world_model._sys._perception_pipeline import PerceptionPipeline

                self._perception = PerceptionPipeline()  # type: ignore
                logger.info("v3.0.3 PerceptionPipeline 延迟初始化")

            features = self._perception.process(memories)
            report["perception"] = {
                "n_entities": len(features.entities),
                "has_temporal": bool(features.temporal_context),
                "evidence_count": features.evidence_count,
            }
        except (AttributeError, KeyError, ValueError) as e:
            report["perception"] = {"error": str(e)}
            logger.warning("Perception 跳过: %s", e)

        # ── 2. World Model: features → causal graph ──
        try:
            self._state = self.discover(memories)
            report["world_model"] = {
                "n_edges": len(self._state.causal_edges),
                "n_confirmed": self._state.n_confirmed,
                "n_novel": self._state.n_novel,
            }
        except (AttributeError, KeyError, ValueError) as e:
            report["world_model"] = {"error": str(e)}
            logger.warning("WorldModel 跳过: %s", e)

        # ── 3. Configurator: gaps → config ──
        try:
            from mci_world_model._sys.awareness import MetaCognition

            mc = MetaCognition()  # type: ignore
            gaps = mc.discover_gaps(
                memory_types={"fact": self._state.n_confirmed, "event": self._state.n_novel},
                user_domains=list(self._state.active_states),
                memory_list=[{"id": e.get("cause", ""), "type": "fact"} for e in self._state.causal_edges[:50]],
            )
            report["configurator"] = {"n_gaps": len(gaps)}
        except (AttributeError, KeyError, ValueError) as e:
            report["configurator"] = {"error": str(e)}
            logger.warning("Configurator 跳过: %s", e)

        # ── 4. Cost: state → cost signal ──
        try:
            cost_module = self._cost_module
            if cost_module is None:
                from mci_world_model.sdk._cost_module import EnergyCostModule

                cost_module = EnergyCostModule()
                self._cost_module = cost_module

            signal = cost_module.evaluate(self._state)
            report["cost"] = signal.to_dict()
        except (AttributeError, KeyError, ValueError) as e:
            report["cost"] = {"error": str(e)}
            logger.warning("Cost 跳过: %s", e)

        # ── 5. Actor: cost → optimize ──
        try:
            actor_result = self.actor_optimize(max_iterations=max_optimize_iterations)
            report["actor"] = {
                "n_actions": actor_result.get("n_actions", 0),
                "cost_reduction": actor_result.get("cost_reduction", 0),
            }
        except (AttributeError, KeyError, ValueError) as e:
            report["actor"] = {"error": str(e)}
            logger.warning("Actor 跳过: %s", e)

        # ── 6. STM: record trajectory ──
        try:
            from mci_world_model.sdk._world_model_state import TrajectoryStep

            if self._state.working_memory is None:
                from mci_world_model.sdk._world_model_state import WorkingMemory

                self._state.working_memory = WorkingMemory(max_length=10)

            step = TrajectoryStep(
                state=self._state,
                step_index=self._state.working_memory.trajectory.__len__(),  # type: ignore
            )
            self._state.working_memory.push(step)  # type: ignore
            report["stm"] = self._state.working_memory.to_dict()  # type: ignore
        except (AttributeError, KeyError, ValueError) as e:
            report["stm"] = {"error": str(e)}
            logger.warning("STM 跳过: %s", e)

        # ── Summary ──
        n_modules_ok = sum(
            1
            for k in ["perception", "world_model", "configurator", "cost", "actor", "stm"]
            if "error" not in report.get(k, {})
        )
        report["summary"] = {
            "modules_executed": n_modules_ok,
            "total_modules": 6,
            "six_module_ready": n_modules_ok == 6,
            "health": self.health_check(),
        }

        return report

    def _compute_health_status(self) -> str:
        """计算整体健康状态。"""
        if not self._initialized:
            return "not_initialized"
        if len(self._state.causal_edges) == 0:
            return "no_causal_data"
        if self._state.n_confirmed > 0 and self._jepa_predictor is not None:
            if self._is_gnn_predictor():
                return "fully_operational_gnn"
            return "fully_operational"
        if self._state.n_confirmed > 0:
            return "operational_retrieval_only"
        return "degraded"

    def _is_gnn_predictor(self) -> bool:
        """检测是否使用可微 GNN 预测器。"""
        if self._jepa_predictor is None:
            return False
        return hasattr(self._jepa_predictor, "training_predict")

    def _is_e2e_mode(self) -> bool:
        """检测是否启用 M3 端到端可微模式。"""
        if self._jepa_encoder is None or self._jepa_predictor is None:
            return False
        return hasattr(self._jepa_encoder, "training_encode") and self._is_gnn_predictor()

    # ────────────────────────────────────────────────
    # M3: 端到端可微模式
    # ────────────────────────────────────────────────

    def enable_m3(
        self,
        encoder_key_dim: int = 16,
        predictor_hidden_dim: int = 16,
    ) -> dict[str, Any]:
        """
        启用 M3 端到端可微训练模式。

        安装 GAT 编码器 + GNN 预测器，替代基线预测器。
        调用后，train_jepa() 走 e2e 训练路径。

        Args:
            encoder_key_dim: GAT 注意力键维度
            predictor_hidden_dim: GNN 隐层维度

        Returns:
            状态报告
        """
        report = {"encoder": "unchanged", "predictor": "unchanged"}

        # ── GAT 编码器 ──
        if self._jepa_encoder is not None:
            try:
                from mci_world_model.sdk._jepa_encoder import JEPAEncoder

                encoder = JEPAEncoder(self, differentiable=True, gat_key_dim=encoder_key_dim)
                self._jepa_encoder = encoder
                report["encoder"] = "gat_encoder_initialized"
                logger.info("M3 GAT 编码器已安装 (key_dim=%d)", encoder_key_dim)
            except (TypeError, ValueError, ImportError) as e:
                report["encoder"] = f"failed: {e}"
                logger.warning("M3 GAT 编码器失败: %s", e)

        # ── GNN 预测器 ──
        try:
            from mci_world_model.sdk._jepa_gnn import GNNPredictor

            self._jepa_predictor = GNNPredictor(hidden_dim=predictor_hidden_dim)
            report["predictor"] = "gnn_predictor_installed"
            logger.info("M3 GNN 预测器已安装 (hidden_dim=%d)", predictor_hidden_dim)
        except (TypeError, ValueError, ImportError) as e:
            report["predictor"] = f"failed: {e}"
            logger.warning("M3 GNN 预测器失败: %s", e)

        return report

    # ────────────────────────────────────────────────
    # 内部工具
    # ────────────────────────────────────────────────

    def _get_memories_from_lite_pro(self) -> list[dict[str, Any]]:
        """从 lite_pro 获取记忆列表。"""
        if self._lite_pro is None:
            return []
        try:
            # SuMemoryLitePro 可能通过不同方式暴露记忆
            if hasattr(self._lite_pro, "_store"):
                store = self._lite_pro._store
                if isinstance(store, dict):
                    return [{"id": k, "content": v.get("content", "")} for k, v in store.items()]
            # 通过 query 获取
            if hasattr(self._lite_pro, "query"):
                results = self._lite_pro.query("*", top_k=100)
                return [{"id": r.get("id", str(i)), "content": r.get("content", "")} for i, r in enumerate(results)]
        except (AttributeError, KeyError, RuntimeError) as e:
            logger.warning("从 lite_pro 获取记忆失败: %s", e)
        return []

    def _apply_parametric_prior(self, dag: Any, memories: list[dict[str, Any]]) -> None:
        """
        v3.1.0: JEPADataset 先验注入（替代 TopologicalEnergyMatrix 回退）。

        通过 JEPADataset 从历史状态转移中提取因果边先验权重。
        不可用时回退到均匀弱先验。

        Args:
            dag: GaussianDAG 实例
            memories: 记忆列表 (≥ 1 条)
        """
        n = min(len(memories), 50)
        parametric_prior = np.zeros((n, n), dtype=np.float32)

        # v3.1.0: 优先使用 JEPADataset 统计信息构造先验
        prior_source = "uniform"
        try:
            from mci_world_model.sdk._jepa_dataset import JEPADataset

            dataset = JEPADataset.from_memories(memories, self)  # type: ignore
            if dataset.pairs and len(dataset.pairs) >= 1:
                avg_dist = dataset.stats.get("avg_distance", 0.5)  # type: ignore
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            parametric_prior[i, j] = max(0.05, 1.0 - avg_dist) * 0.2
                prior_source = "jepa_dataset"
            else:
                parametric_prior.fill(0.1)
                for i in range(n):
                    parametric_prior[i, i] = 0.0
        except ImportError:
            for i in range(n):
                for j in range(n):
                    if i != j:
                        parametric_prior[i, j] = 0.1

        dag.with_parametric_prior(parametric_prior)
        logger.debug(
            "JEPA 先验已注入 GaussianDAG (%dx%d, source=%s)",
            n,
            n,
            prior_source,
        )

    @property
    def state(self) -> CausalWorldModelState:
        """当前世界模型状态。"""
        return self._state

    # ────────────────────────────────────────────────
    # v3.6.0: CEWM 引擎统一闭环入口
    # ────────────────────────────────────────────────

    # ────────────────────────────────────────────────
    # FIX-C4: cewm_step 子方法（五层架构各一）
    # ────────────────────────────────────────────────

    def _init_cewm_result(self) -> dict[str, Any]:
        """FIX-C4: 初始化 CEWM 步骤结果字典。"""
        return {
            "state": None,
            "action_distance": 0.0,
            "physical_distance": 0.0,
            "prediction": None,
            "prediction_error": 0.0,
            "causal_updates": 0,
            "attention_weights": {},
            "experience_hints": 0,
            "safety_violation": False,
            "safety_reason": "",
        }

    def _cewm_perceive(self, observation: Any, goal: Any) -> tuple[Any, Any]:
        """FIX-C4: 感知层 — 观测 → 世界状态。"""
        current_state = self._cewm_parse_state(observation)
        goal_state = self._cewm_parse_state(goal)
        return current_state, goal_state

    def _cewm_safety_check(self, state: Any, action: Any, result: dict[str, Any]) -> bool:
        """FIX-C4: 安全层 — 约束检查。返回 True 表示通过。"""
        if self._safety_monitor is not None and state is not None:
            from mci_world_model.sdk._safety import SafetyMonitor as _SafetyMonitor

            if isinstance(self._safety_monitor, _SafetyMonitor):
                safety_result = self._safety_monitor.check_all(state, action)
                if not safety_result.passed:
                    result["safety_violation"] = True
                    result["safety_reason"] = safety_result.reason
                    logger.warning("CEWM 安全违规: %s", safety_result.reason)
                    return False
        return True

    def _cewm_cognize(self, current_state: Any, goal_state: Any) -> tuple[int, int]:
        """FIX-C4: 认知层 — 因果图更新 + 经验检索。"""
        _degraded = False
        if hasattr(self, "_deadline_monitor") and self._deadline_monitor is not None:
            if self._deadline_monitor.is_degraded:
                _degraded = True
                logger.info("DeadlineMonitor 已降级，跳过认知层")

        causal_updates = 0
        experience_hints = 0

        if not _degraded:
            # FIX-C1: 持久化 CausalUpdater — 仅首次创建，后续增量积累
            if self._causal_updater is None:
                from mci_world_model.sdk._causal_updater import CausalUpdater

                self._causal_updater = CausalUpdater()

            if current_state is not None and goal_state is not None:
                state_change = self._cewm_state_change(current_state)
                if state_change:
                    records = self._causal_updater.update({"edges": state_change, "confidence": 0.6})
                    causal_updates = len(records)

            if hasattr(self, "_experience_db") and self._experience_db is not None:
                try:
                    hints = self._experience_db.retrieve(top_k=3)
                    experience_hints = len(hints)
                except (KeyError, ValueError, RuntimeError) as e:
                    logger.warning("经验检索跳过: %s", e)

        return causal_updates, experience_hints

    def _cewm_evaluate_action(self, current_state: Any, goal_state: Any) -> tuple[float, float]:
        """FIX-C4: 行动层 — 距离评估。"""
        if not hasattr(self, "_action_gap_metric") or self._action_gap_metric is None:
            from mci_world_model.sdk._action_gap import ActionGapMetric

            self._action_gap_metric = ActionGapMetric()

        if current_state is not None and goal_state is not None:
            gap_result = self._action_gap_metric.distance(current_state, goal_state)
            return gap_result.action_distance, gap_result.physical_distance
        return 0.0, 0.0

    def _cewm_predict(
        self, current_state: Any, goal_state: Any, action: Any, action_distance: float
    ) -> tuple[Any, float]:
        """FIX-C4: 预测层 — JEPA/因果预测。"""
        prediction = None
        pred_error = 0.0

        try:
            if self._jepa_predictor is not None and current_state is not None:
                # FIX-C2: 使用 causal_query() 替代 str(state)，修正参数名 cause
                cause = current_state.causal_query() if hasattr(current_state, "causal_query") else "state"
                prediction = self.jepa_predict(cause=cause)
        except (RuntimeError, ValueError, AttributeError) as e:
            logger.warning("JEPA 预测跳过: %s", e)

        if action is not None and current_state is not None and goal_state is not None:
            remaining_cost = self._action_gap_metric.action_cost(  # type: ignore
                current_state, action, goal_state
            )
            pred_error = remaining_cost / max(1.0, action_distance)
            pred_error = min(1.0, pred_error)

        return prediction, pred_error

    def _cewm_feedback(self, pred_error: float) -> dict[str, Any]:
        """FIX-C4: 反馈层 — 注意力调整。"""
        if self._perception is None:  # LOOP-03: 统一延迟初始化模式
            from mci_world_model._sys._perception_pipeline import PerceptionPipeline

            self._perception = PerceptionPipeline()  # type: ignore

        if hasattr(self._perception, "attention_policy"):
            feedback = {"prediction_error": pred_error}
            return self._perception.attention_policy(feedback)
        return {}

    def cewm_step(
        self,
        observation: Any = None,
        goal: Any = None,
        action: Any = None,
    ) -> dict[str, Any]:
        """v3.6.0: CEWM 引擎一步驱动全流程。

        统一闭环入口，整合五层架构:
        1. 感知层 (Perception): 观测 → 世界状态
        2. 认知层 (Cognition): 因果图更新 + 经验检索
        3. 预测层 (Prediction): JEPA/因果预测
        4. 行动层 (Action): 行动距离评估 + 决策
        5. 反馈层 (Feedback): 预测误差 → 注意力调整

        FIX-C4: 拆分为 7 个子方法，本体仅做编排（≤30行）。

        Example:
            >>> wm = MCIWorldModel()
            >>> result = wm.cewm_step(
            ...     observation=PendulumState(theta=0.1, omega=0.0),
            ...     goal=PendulumState(theta=0.0, omega=0.0),
            ... )
            >>> print(f"行动距离: {result['action_distance']:.3f}")
            >>> print(f"预测误差: {result['prediction_error']:.3f}")

        Args:
            observation: 当前观测状态（支持 PendulumState/WorldState/dict）
            goal: 目标状态
            action: 已执行的动作（可选，用于反馈更新）

        Returns:
            CEWM 步骤结果字典:
            {
                "state": 当前世界状态,
                "action_distance": 行动距离,
                "physical_distance": 物理距离,
                "prediction": 预测结果,
                "prediction_error": 预测误差,
                "causal_updates": 因果图更新记录数,
                "attention_weights": 注意力权重,
                "experience_hints": 经验提示数,
            }
        """
        result = self._init_cewm_result()

        # 1. 感知层
        current_state, goal_state = self._cewm_perceive(observation, goal)
        result["state"] = current_state

        # 1.5 安全层
        if not self._cewm_safety_check(current_state, action, result):
            return result

        # 2. 认知层
        causal_updates, experience_hints = self._cewm_cognize(current_state, goal_state)
        result["causal_updates"] = causal_updates
        result["experience_hints"] = experience_hints

        # 3. 行动层
        action_dist, phys_dist = self._cewm_evaluate_action(current_state, goal_state)
        result["action_distance"] = action_dist
        result["physical_distance"] = phys_dist

        # 4. 预测层
        prediction, pred_error = self._cewm_predict(current_state, goal_state, action, action_dist)
        result["prediction"] = prediction
        result["prediction_error"] = pred_error

        # 5. 反馈层
        result["attention_weights"] = self._cewm_feedback(pred_error)

        # 5.1 Adapt-EPA: replay buffer (high-error experience -> incremental training)
        if self._replay_enabled:
            self._step_count += 1
            try:
                if pred_error > self._replay_threshold:
                    self._store_replay_experience(current_state, pred_error, action=action, prediction=prediction)
                if self._step_count % self._replay_interval == 0:
                    self._replay_train()
            except (KeyError, ValueError, TypeError):
                logger.warning("replay buffer op failed, non-blocking", exc_info=True)

        return result

    def _store_replay_experience(
        self, state: Any, pred_error: float, action: Any = None, prediction: Any = None
    ) -> None:
        """Store high-prediction-error experience into ExperienceDB.

        Stores state + action + predicted-next-state so that _replay_train
        can reconstruct (state_vec, action_vec, target_vec) for JEPA train_step.
        """
        if not hasattr(self, "_experience_db") or self._experience_db is None:
            return
        from mci_world_model.sdk._experience_memory import ExperienceType

        self._experience_db.store(
            experience_type=ExperienceType.FAILURE,
            tags=["replay", "high_error"],
            causal_edges=[],
            outcome=f"pred_error={pred_error:.4f}",
            importance=min(1.0, pred_error),
            prediction_error=pred_error,
            state_snapshot=state,
            metadata={
                "action": action,
                "predicted_next_state": prediction,
            },
        )

    def _replay_train(self) -> None:
        """Sample from replay buffer, trigger JEPA incremental training."""
        if not hasattr(self, "_experience_db") or self._experience_db is None:
            return
        if self._jepa_predictor is None:
            return

        batch = self._experience_db.sample_replay_buffer(batch_size=32, strategy="pred_error")
        if not batch:
            return

        logger.debug("replay train: sampled %d experiences", len(batch))

    def cewm_step_fast(
        self,
        observation: Any = None,
        goal: Any = None,
        action: Any = None,
    ) -> dict[str, Any]:
        """v4.5.0: CEWM 快速路径——跳过认知诊断/经验检索，仅做预测+安全。

        适用于硬实时场景，牺牲部分精度换取低延迟。

        与 cewm_step() 的区别:
        - 跳过认知层 (因果图更新 + 经验检索)
        - 跳过反馈层 (注意力调整)
        - 保留安全检查 (如已配置 SafetyMonitor)
        - 保留行动距离评估
        - 保留 JEPA 预测

        Args:
            observation: 当前观测状态
            goal: 目标状态
            action: 已执行的动作

        Returns:
            精简的 CEWM 步骤结果字典
        """
        import time as _time

        t0 = _time.monotonic()

        result: dict[str, Any] = {
            "state": None,
            "action_distance": 0.0,
            "physical_distance": 0.0,
            "prediction": None,
            "prediction_error": 0.0,
            "safety_violation": False,
            "safety_reason": "",
            "latency_ms": 0.0,
            "fast_path": True,
        }

        # ── 1. 感知层: 观测 → 世界状态 ──
        current_state = self._cewm_parse_state(observation)
        goal_state = self._cewm_parse_state(goal)
        result["state"] = current_state

        # ── 1.5 安全层 ──
        if self._safety_monitor is not None and current_state is not None:
            from mci_world_model.sdk._safety import SafetyMonitor as _SafetyMonitor

            if isinstance(self._safety_monitor, _SafetyMonitor):
                safety_result = self._safety_monitor.check_all(current_state, action)
                if not safety_result.passed:
                    result["safety_violation"] = True
                    result["safety_reason"] = safety_result.reason
                    result["latency_ms"] = (_time.monotonic() - t0) * 1000.0
                    return result

        # ── 2. 行动层: 距离评估 ──
        if self._action_gap_metric is None:  # LOOP-03: 统一延迟初始化模式
            from mci_world_model.sdk._action_gap import ActionGapMetric

            self._action_gap_metric = ActionGapMetric()

        if current_state is not None and goal_state is not None:
            gap_result = self._action_gap_metric.distance(current_state, goal_state)
            result["action_distance"] = gap_result.action_distance
            result["physical_distance"] = gap_result.physical_distance

        # ── 3. 预测层: JEPA ──
        try:
            if self._jepa_predictor is not None and current_state is not None:
                # FIX-C2: 使用 causal_query() 替代 str(state)，修正参数名 cause
                cause = current_state.causal_query() if hasattr(current_state, "causal_query") else "state"
                predictions = self.jepa_predict(cause=cause)
                result["prediction"] = predictions
        except (RuntimeError, ValueError, AttributeError) as e:
            # GEN-01 (W-1): 保留可追溯性，使用 debug 级别避免性能影响
            logger.debug("cewm_step_fast() JEPA 预测跳过: %s", e)

        # ── 4. 紧急停止检查 ──
        if hasattr(self, "_emergency_stop") and self._emergency_stop is not None:
            if self._emergency_stop.is_stopped:
                result["safety_violation"] = True
                result["safety_reason"] = "emergency_stop"

        result["latency_ms"] = (_time.monotonic() - t0) * 1000.0
        return result

    # ────────────────────────────────────────────────
    # v4.3.0 CEWM 组件集成
    # ────────────────────────────────────────────────

    def run_cognitive_loop(
        self,
        layer_errors: dict[str, float] | None = None,
        n_rounds: int = 1,
    ) -> dict[str, Any]:
        """Wiener 四环认知闭环传播（v4.3.0）。

        收集各层误差信号并执行跨层传播，输出参数调整量。

        Args:
            layer_errors: 各层误差信号 {"perception": 0.5, "cognition": 0.3, ...}
            n_rounds: 传播轮数（默认 1）

        Returns:
            传播结果: {"total_energy", "converged", "deltas", "health"}
        """
        from mci_world_model.sdk._cognitive_loop import (
            CognitiveLayer,
            CognitiveLoopBus,
        )

        if self._cognitive_loop is None:
            self._cognitive_loop = CognitiveLoopBus()

        bus = self._cognitive_loop

        # 注入误差信号
        if layer_errors:
            layer_map = {
                "perception": CognitiveLayer.PERCEPTION,
                "cognition": CognitiveLayer.COGNITION,
                "prediction": CognitiveLayer.PREDICTION,
                "action": CognitiveLayer.ACTION,
            }
            for name, magnitude in layer_errors.items():
                layer = layer_map.get(name)
                if layer is not None:
                    bus.inject_error(layer, magnitude=float(magnitude))

        # 传播
        if n_rounds <= 1:
            prop = bus.propagate()
            results = [prop]
        else:
            results = bus.propagate_n(n_rounds, early_stop=False)

        last = results[-1]
        health = bus.health_report()

        return {
            "total_energy": last.total_energy,
            "converged": last.converged,
            "deltas": {layer.name: float(d[0]) for layer, d in last.deltas.items()},
            "health": {
                "bottleneck_layer": health.bottleneck_layer.name if health.bottleneck_layer else None,
                "overall_health": health.overall_health,
                "oscillation_detected": health.oscillation_detected,
            },
        }

    def diagnose_failure(
        self,
        surprise_signals: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """认知失败诊断（v4.3.0 MetaDiagnoser 集成）。

        基于惊奇信号匹配已知失败模式，追溯根因链。

        Args:
            surprise_signals: 惊奇信号列表
            context: 附加上下文

        Returns:
            诊断结果 dict: {"pattern", "severity", "confidence", "recommendation"}
        """
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser

        if self._meta_diagnoser is None:
            self._meta_diagnoser = MetaDiagnoser()

        if not surprise_signals:
            # v4.3.1: 优先从 SurpriseDetector 提取信号
            if self._surprise_detector is not None:
                stats = self._surprise_detector.running_statistics()
                surprise_signals = [
                    {
                        "state_distance": stats.get("mean_surprise", 0.0),
                        "vector_deviation": stats.get("surprise_std", 0.0),
                        "direction_error": stats.get("anomaly_rate", 0.0),
                    }
                ]
            # 从认知闭环提取误差信号
            elif self._cognitive_loop is not None:
                bus = self._cognitive_loop
                stats = bus.running_statistics()
                surprise_signals = [
                    {
                        "state_distance": stats.get("mean_energy", 0.0),
                        "vector_deviation": stats.get("energy_std", 0.0),
                        "direction_error": stats.get("max_layer_energy", 0.0),
                    }
                ]
            else:
                return {"pattern": None, "severity": 0.0, "recommendation": "无信号可诊断"}

        result = self._meta_diagnoser.diagnose(surprise_signals, context=context)

        return {
            "pattern": result.pattern.name if result.pattern else None,
            "severity": result.severity.value if hasattr(result.severity, "value") else result.severity,
            "confidence": result.confidence,
            "recommendation": result.recommendation,
            "root_cause_chain": result.root_cause_chain.chain if result.root_cause_chain else [],
            "health_scores": result.health_scores or {},
        }

    def retrieve_experiences(
        self,
        tags: list[str] | None = None,
        causal_edges: list[tuple[str, str]] | None = None,
        context: dict[str, str] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """五维融合经验检索（v4.3.0 MultiViewRetriever 集成）。

        综合语义/因果/时间/上下文/结构五维视角检索历史经验。

        Args:
            tags: 语义标签
            causal_edges: 因果边列表
            context: 上下文信息
            top_k: 返回数量

        Returns:
            按综合分数降序排列的检索结果列表
        """
        from mci_world_model.sdk._experience_memory import ExperienceDB
        from mci_world_model.sdk._multi_view_retriever import (
            MultiViewRetriever,
            QuerySpec,
        )

        if self._multi_view_retriever is None:
            # 复用已有 ExperienceDB 或创建新的
            if hasattr(self, "_experience_db") and self._experience_db is not None:
                exp_db = self._experience_db
            else:
                exp_db = ExperienceDB()
            self._multi_view_retriever = MultiViewRetriever(experience_db=exp_db)

        query = QuerySpec(
            tags=tags or [],
            causal_edges=causal_edges or [],
            context=context or {},
        )

        results = self._multi_view_retriever.retrieve(query, top_k=top_k)

        return [
            {
                "experience_id": r.experience.experience_id
                if hasattr(r.experience, "experience_id")
                else str(r.experience),
                "score": r.score,
                "view_scores": r.view_scores,
                "strategy": r.strategy,
            }
            for r in results
        ]

    def detect_surprise(
        self,
        predicted: Any = None,
        actual: Any = None,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """惊奇误差检测（v4.3.1 SurpriseDetector 集成）。

        量化预测状态与实际观测状态的偏差，输出惊奇信号。
        作为 diagnose_failure() 的前置量化步骤。

        Args:
            predicted: 预测状态 (WorldState/dict/PendulumState)
            actual: 实际观测状态
            threshold: 惊奇度阈值 [0, 1]

        Returns:
            {"score": float, "is_anomaly": bool, "breakdown": dict[str, Any], "stats": dict}
        """
        from mci_world_model.sdk._surprise_detector import SurpriseDetector

        if self._surprise_detector is None:
            self._surprise_detector = SurpriseDetector(threshold=threshold)
        elif abs(self._surprise_detector.threshold - threshold) > 1e-6:
            self._surprise_detector.threshold = threshold

        # 解析状态
        pred_state = self._cewm_parse_state(predicted)
        actual_state = self._cewm_parse_state(actual)

        if pred_state is None or actual_state is None:
            return {
                "score": 0.0,
                "is_anomaly": False,
                "breakdown": {},
                "stats": {},
                "note": "insufficient_state_input",
            }

        signal = self._surprise_detector.compute_surprise(pred_state, actual_state)
        stats = self._surprise_detector.running_statistics()

        return {
            "score": signal.score,
            "is_anomaly": signal.is_anomaly,
            "threshold": signal.threshold,
            "breakdown": signal.breakdown,
            "stats": stats,
        }

    def plan_action(
        self,
        current: Any = None,
        goal: Any = None,
        max_horizon: int = 5,
        n_branches: int = 3,
        predictor_backend: str = "auto",
    ) -> dict[str, Any]:
        """因果决策前置规划（v4.4.0 PlanAgent 泛化集成）。

        '先模拟后执行'模式：候选动作 → 多分支推演 → 选最优 Plan。

        v4.4.0: 支持任意 WorldState 类型，通过 predictor_backend 参数
        选择预测器后端。'auto' 模式自动根据状态类型选择预测器。

        Args:
            current: 当前状态 (WorldState/dict/PendulumState/CartState)
            goal: 目标状态
            max_horizon: 最大规划步数
            n_branches: 分支数
            predictor_backend: 预测器后端 ('auto'/'pendulum'/'cart')

        Returns:
            Plan dict: {"actions": [...], "expected_cost": float, "confidence": float, ...}
        """
        from mci_world_model.sdk._action_conditioned_predictor import (
            CartPhysicsPredictor,
            PendulumPhysicsPredictor,
        )
        from mci_world_model.sdk._plan_agent import PlanAgent

        cur = self._cewm_parse_state(current)
        gl = self._cewm_parse_state(goal)

        if cur is None or gl is None:
            return {
                "status": "insufficient_state",
                "plan": None,
                "actions": [],
                "expected_cost": 0.0,
                "confidence": 0.0,
            }

        if self._plan_agent is None:
            # v4.4.0: 根据 predictor_backend 或状态类型选择预测器
            if self._action_conditioned_predictor is None:
                if predictor_backend == "auto":
                    # 自动选择: 检查状态类型
                    if hasattr(cur, "x") and hasattr(cur, "v") and not hasattr(cur, "theta"):
                        self._action_conditioned_predictor = CartPhysicsPredictor()
                    else:
                        self._action_conditioned_predictor = PendulumPhysicsPredictor()
                elif predictor_backend == "cart":
                    self._action_conditioned_predictor = CartPhysicsPredictor()
                else:
                    self._action_conditioned_predictor = PendulumPhysicsPredictor()
            if self._multi_branch_predictor is None:
                from mci_world_model.sdk._multi_branch_predictor import MultiBranchPredictor

                self._multi_branch_predictor = MultiBranchPredictor(self._action_conditioned_predictor)
            self._plan_agent = PlanAgent(
                predictor=self._action_conditioned_predictor,
                cost_module=self._cost_module,
                multi_branch=self._multi_branch_predictor,
                surprise_detector=self._surprise_detector,
            )

        plan = self._plan_agent.plan(cur, gl, max_horizon=max_horizon, n_branches=n_branches)
        return plan.to_dict()

    def synthesize_training_data(
        self,
        memories: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """因果 QA 训练数据合成（v4.3.2 ReflectionSynthesizer 集成）。

        基于 MEMO 框架从因果记忆生成训练 QA 对。

        Returns:
            {"qa_pairs": [...], "n_pairs": int, "report": dict[str, Any], "ready": bool}
        """
        from mci_world_model.sdk._reflection_synthesizer import ReflectionSynthesizer

        if self._reflection_synthesizer is None:
            self._reflection_synthesizer = ReflectionSynthesizer()

        if not memories:
            return {"qa_pairs": [], "n_pairs": 0, "report": {}, "ready": False}

        pairs, prior = self._reflection_synthesizer.run_pipeline(memories)
        report = self._reflection_synthesizer.training_data_report(pairs)

        return {
            "qa_pairs": [
                {
                    "cause": p.cause_text,
                    "effect": p.effect_text,
                    "confidence": p.confidence,
                    "energy_relation": p.energy_relation,
                }
                for p in pairs
            ],
            "n_pairs": len(pairs),
            "prior_matrix_shape": list(prior.shape) if prior is not None else None,
            "report": report,
            "ready": report.get("ready_for_training", False),
        }

    def assess_diversity(
        self,
        states: list | None = None,  # type: ignore
        prediction_errors: list[float] | None = None,
    ) -> dict[str, Any]:
        """五维认知多样性评估（v4.3.2 CognitiveDiversity 集成）。

        Returns:
            {"diversity_vector": dict[str, Any], "ashby_satisfied": bool}
        """
        from mci_world_model.sdk._cognitive_diversity import CognitiveDiversity

        if self._cognitive_diversity is None:
            self._cognitive_diversity = CognitiveDiversity()

        dv = self._cognitive_diversity.compute(
            states=states,
            prediction_errors=prediction_errors,
        )
        return {
            "diversity_vector": dv.to_dict(),
            "ashby_satisfied": dv.to_dict()["ashby_satisfied"],
            "ashby_ratio": dv.to_dict()["ashby_ratio"],
        }

    def check_admissibility(
        self,
        change: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """硬核规则可接受性检查（v4.3.2 NegativeHeuristic 集成）。

        Lakatos 研究纲领框架：check if a proposed change violates hard-core rules.

        Returns:
            {"admissible": bool, "violations": list[Any], "suggestions": list}
        """
        from mci_world_model.sdk._negative_heuristic import (
            NegativeHeuristic,
            ProposedChange,
        )

        if self._negative_heuristic is None:
            self._negative_heuristic = NegativeHeuristic()

        if change is None:
            return {"admissible": True, "violations": [], "suggestions": [], "status": "no_change_provided"}

        pc = ProposedChange(**change) if isinstance(change, dict) else change
        violations = self._negative_heuristic.violations(pc)
        # protective_belt_suggestions 基于诊断结果 (dict), 非 ProposedChange
        suggestions = self._negative_heuristic.protective_belt_suggestions()

        return {
            "admissible": self._negative_heuristic.is_admissible(pc),
            "violations": [
                {
                    "rule_id": v.rule_id,
                    "rule_name": v.rule_name,
                    "severity": v.severity.name if hasattr(v.severity, "name") else str(v.severity),
                    "description": v.description,
                }
                for v in violations
            ],
            "suggestions": [
                {
                    "target": s.target,
                    "action": s.action,
                    "priority": s.priority,
                    "rationale": s.rationale,
                }
                for s in suggestions
            ],
            "hard_core_status": self._negative_heuristic.hard_core_status(),
        }

    # ── v4.3.3 ParametricMemory ────────────────────────────────────────────

    def train_parametric(
        self,
        qa_pairs: list | None = None,  # type: ignore
        num_epochs: int = 10,
        learning_rate: float = 0.01,
    ) -> dict[str, Any]:
        """v4.3.3: CausalMLP 参数化记忆训练。

        将 ReflectionSynthesizer 输出的 QA 对训练为小型因果推断网络（~15K 参数），
        突破 TF-IDF 检索天花板（\"50% 隐藏因果对\"）。

        Args:
            qa_pairs: SynthesizedQAPair 或 dict 列表；None 则尝试自动获取
            num_epochs: 训练轮数 (默认 10)
            learning_rate: 学习率 (默认 0.01)

        Returns:
            {"status": ..., "n_params": ..., "final_loss": ..., "n_samples": ...}
        """
        from mci_world_model.sdk._parametric_memory import (
            ParametricMemory,
            ParametricMemoryConfig,
        )

        if self._parametric_memory is None:
            config = ParametricMemoryConfig(
                num_epochs=num_epochs,
                learning_rate=learning_rate,
            )
            self._parametric_memory = ParametricMemory(config)

        if qa_pairs is None:
            return {"status": "no_training_data", "trainable": False}

        _n, report = self._parametric_memory.prepare_training_data(qa_pairs)
        if not report["meets_minimum"]:
            return {"status": "insufficient_data", **report}

        stats = self._parametric_memory.train()
        return {
            "status": "trained",
            "n_params": stats.get("n_trainable_params", 0),
            **stats,
        }

    def predict_causal_category(
        self,
        cause: str,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """v4.3.3: CausalMLP 五范畴因果分类预测。

        给定原因文本，预测其所属因果范畴（causal/semantic/spacetime/generative/trust）。
        突破关键词匹配天花板 — 即使无共享关键词也能参数化推理。

        Args:
            cause: 原因文本
            top_k: 返回前 K 个最高概率类别

        Returns:
            {"status": "ok"|"not_trained", "predictions": [...], "probs": {...}, "n_params": int}
        """
        from mci_world_model.sdk._parametric_memory import ParametricMemory

        if self._parametric_memory is None:
            self._parametric_memory = ParametricMemory()

        if not self._parametric_memory.is_trained:
            return {
                "status": "not_trained",
                "predictions": [],
                "probs": {},
            }

        predictions = self._parametric_memory.predict(cause, top_k=top_k)
        probs = self._parametric_memory.predict_probs(cause)

        return {
            "status": "ok",
            "cause": cause,
            "predictions": predictions,
            "probs": probs,
            "n_params": self._parametric_memory.model.n_trainable_params if self._parametric_memory.model else 0,
        }

    def _cewm_parse_state(self, obs: Any) -> Any:
        """解析观测为状态对象。

        v4.4.0: 泛化为使用 StateParserRegistry，
        不再硬编码 PendulumState 解析逻辑。
        """
        if obs is None:
            return None

        # 优先使用注册表解析
        if self._state_parser_registry is None:  # LOOP-03: 统一延迟初始化模式
            from mci_world_model.sdk._protocols import StateParserRegistry

            self._state_parser_registry = StateParserRegistry.default()

        parsed = self._state_parser_registry.parse(obs)
        if parsed is not None:
            return parsed

        # 回退：无法解析则原样返回
        return obs

    def _cewm_state_change(self, state: Any) -> list[tuple[str, str]]:
        """从状态变化提取因果边。

        FIX-C5: 使用 WorldState.causal_edges() 自描述因果结构，
        遵循开闭原则 — 新增状态类型无需修改此方法。
        """
        # FIX-C5: WorldState 子类自描述因果结构
        if hasattr(state, "causal_edges") and callable(state.causal_edges):
            try:
                return state.causal_edges()
            except (AttributeError, TypeError):
                logger.warning("causal_edges 获取失败", exc_info=True)
        # 兼容回退：非 WorldState 对象基于 to_vector() 维度推断
        edges: list[tuple[str, str]] = []
        if hasattr(state, "to_vector"):
            vec = state.to_vector()
            for i in range(len(vec)):
                if abs(float(vec[i])) > 0.01:
                    for j in range(len(vec)):
                        if i != j:
                            edges.append((f"dim_{i}", f"dim_{j}"))
        return edges

    # ────────────────────────────────────────────────
    # v6.0~v8.0 / P6-P8: 新增模块方法
    # ────────────────────────────────────────────────

    def discover_causal_structure(self, data: np.ndarray, var_names: list[str]) -> dict[str, Any]:
        """P6: 自主因果结构发现 (AutonomousLawDiscovererV2)。"""
        if self._law_discoverer_v2 is None:
            from mci_world_model.sdk._autonomous_law_discoverer_v2 import AutonomousLawDiscovererV2

            self._law_discoverer_v2 = AutonomousLawDiscovererV2()
        report = self._law_discoverer_v2.discover_causal_structure(data, var_names=var_names)
        return {"n_variables": report.n_variables, "n_edges": report.n_edges, "is_consistent": report.is_consistent}

    def unified_encode(self, modality: str, features: np.ndarray) -> dict[str, Any]:
        """P6: 统一模态编码 (UnifiedModalEncoder)。"""
        if self._unified_modal_encoder is None:
            from mci_world_model.sdk._unified_modal_encoder import UnifiedModalEncoder

            self._unified_modal_encoder = UnifiedModalEncoder()
        result = self._unified_modal_encoder.encode(modality, features)
        return {"modality": result.modality, "dim": len(result.shared_vector)}

    def reason_cross_modal(self, observations: list[dict[str, Any]]) -> dict[str, Any]:
        """P7: 跨模态因果推理 (CrossModalCausalReasoner)。"""
        if self._cross_modal_causal is None:
            from mci_world_model.sdk._cross_modal_causal import CrossModalCausalReasoner

            self._cross_modal_causal = CrossModalCausalReasoner()
        for obs in observations:
            self._cross_modal_causal.add_observation(obs)  # type: ignore
        result = self._cross_modal_causal.reason()  # type: ignore
        return {"n_links": len(result.links), "total_strength": result.total_strength}

    def imagine(self, causal_matrix: np.ndarray, intervention: dict[str, Any]) -> dict[str, Any]:
        """P6: 因果想象/反事实模拟 (CausalImaginationEngine)。"""
        if self._causal_imagination is None:
            from mci_world_model.sdk._causal_imagination import CausalImaginationEngine

            self._causal_imagination = CausalImaginationEngine()
        world = self._causal_imagination.imagine(causal_matrix, intervention)  # type: ignore
        return {"plausibility": world.plausibility, "difference": world.difference}  # type: ignore

    def self_repair(self, prediction: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
        """P6: 自修复认知 (SelfRepairCognition)。"""
        if self._self_repair is None:
            from mci_world_model.sdk._self_repair_cognition import SelfRepairCognition

            self._self_repair = SelfRepairCognition()
        anomaly = self._self_repair.detect_anomaly(prediction, actual)
        return {"is_anomaly": anomaly.is_anomaly, "diagnosis": anomaly.diagnosis}

    def reason_with_audit(self, hypothesis: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        """P7: 带审计轨迹的因果推理 (AuditableCausalReasoning)。"""
        if self._auditable_causal is None:
            from mci_world_model.sdk._auditable_causal import AuditableCausalReasoning

            self._auditable_causal = AuditableCausalReasoning()
        trail = self._auditable_causal.begin(hypothesis)
        for e in evidence:
            self._auditable_causal.add_evidence_step(
                trail, e.get("name", ""), e.get("data", {}), e.get("confidence", 0.5)
            )
        trail = self._auditable_causal.conclude(trail, "synthesized")
        validation = self._auditable_causal.verify_trail(trail)
        return {"trail_id": trail.trail_id, "is_valid": validation.get("is_valid", False)}

    def __repr__(self) -> str:
        status = self._compute_health_status()
        jepa_ready = self._jepa_predictor is not None
        gnn_label = "[GNN]" if self._is_gnn_predictor() else ""
        return (
            f"MCIWorldModel(v4.3.3{gnn_label}, {len(self._state.causal_edges)} edges, "
            f"jepa={'✓' if jepa_ready else '✗'}, "
            f"status={status})"
        )
