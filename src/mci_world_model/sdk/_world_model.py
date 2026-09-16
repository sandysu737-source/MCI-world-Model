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

from mci_world_model.sdk._causal_inference import CausalInferenceMixin
from mci_world_model.sdk._cewm_engine import CEWMEngineMixin
from mci_world_model.sdk._cognitive_components import CognitiveComponentsMixin
from mci_world_model.sdk._energy_flow import EnergyFlowMixin, _aggregate_energy_ratios
from mci_world_model.sdk._prediction_components import PredictionComponentsMixin
from mci_world_model.sdk._world_model_state import (
    CausalWorldModelState,
    TrajectoryStep,
    WorkingMemory,
)

logger = logging.getLogger(__name__)

__all__ = ["CausalWorldModelState", "TrajectoryStep", "WorkingMemory"]

__all__ += ["_aggregate_energy_ratios"]


class MCIWorldModel(
    CognitiveComponentsMixin,
    PredictionComponentsMixin,
    CausalInferenceMixin,
    CEWMEngineMixin,
    EnergyFlowMixin,
):
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
    # JEPA 训练
    # ────────────────────────────────────────────────

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
