"""
mci_world_model.sdk — World Model Core

Pearl 因果三层 + JEPA 世界建模 + 能量一致性
v3.0.1: Cost 模块独立 + STM 工作记忆
v3.0.2: H-JEPA 分层预测 + Actor 动作搜索
v3.0.3: 六模块完整闭环 (Perception + Configurator 分层)
v3.0.4: 能量感知基础集成 (EnergyCore + Cost时变 + STM时空编码 + Actor能量亲和度)
v3.0.5: 能量流深度闭环 (JEPA能量守恒 + EnergyBus传播 + 生克干预推断 + 月度自适应阈值)
v3.0.6: 能量-因果统一 (对偶边表示 + EnergyFlowPredictor + 自动调节闭环 + 五维覆盖度)
v3.0.7: 参数化记忆觉醒 (CausalMLP + MLX Native 训练 + 移除 torch/transformers/peft)
v3.0.8: 反事实推理增强 (非线性SEM + 批量反事实引擎 + CG↔SEM双向转换)
v3.1.0: 物理世界应用 (多模态信号感知 + PhysicalGraphBuilder + JEPA物理编码)
"""
# v4.4.0: Debridement modules
# ── _sys re-export 块移至文件末尾，避免循环导入 ──
# (v3.5.0: _sys/__init__.py 导入 sdk._multi_view_retriever，需先完成 sdk 内部模块加载)
# v3.4.0: 闭环基础设施 (CEWM Phase 1)
# v3.5.0: _sys re-export (v3.3.1: 能量Bus核心能力全面释放 + 全部符号穿透)
# 移至此处避免循环导入：_sys/__init__.py 导入 sdk._multi_view_retriever
from mci_world_model._sys import (
    BRANCH_CHONG,
    BRANCH_CHONG_MAP,
    BRANCH_HE,
    BRANCH_HE_MAP,
    BRANCH_HIDDEN_STEM_MAP,
    BRANCH_SANHE,
    BRANCH_SANHE_MAP,
    BRANCH_XING,
    CATEGORY_ANCHORS,
    ENERGY_ENHANCE_MAP,
    ENERGY_SUPPRESS_MAP,
    ENERGY_TO_CATEGORY,
    FOUR_SYMBOLS_TO_ENERGY,
    KEYWORDS_TO_CATEGORY,
    MEMORY_TYPE_TO_CATEGORY,
    MONTH_ENERGY_STATE,
    RELATION_STRENGTH,
    SEMANTIC_CATEGORY,
    SEMANTIC_CATEGORY_NAMES,
    STATE_STRENGTH_MAP,
    STEM_CHONG,
    STEM_CHONG_MAP,
    STEM_HE,
    STEM_HE_MAP,
    STRENGTH_MULTIPLIER,
    TIME_BRANCH_ENERGY,
    TIME_BRANCHES,
    TIME_STEMS,
    TRIGRAM_BODY_MAP,
    TRIGRAM_ENERGY_MAP,
    ActionCommand,
    ActionPriority,
    ActionResult,
    ActuatorChannel,
    BayesianEngine,
    BayesianNetwork,
    BayesianReasoningSystem,
    BranchRelation,
    CategoryCausalEngine,
    CausalChain,
    CausalInference,
    CognitiveGap,
    CognitiveScoreCard,
    ConfigAction,
    DiZhi,
    DynamicPriority,
    EnergyBalanceResult,
    EnergyBus,
    EnergyChannel,
    EnergyCore,
    EnergyEnumRelation,
    EnergyEnumType,
    EnergyFlow,
    EnergyLayer,
    EnergyMemoryNode,
    EnergyNetwork,
    EnergyNode,
    EnergyPattern,
    EnergyRelation,
    EnergySignal,
    EnergyState,
    EnergyType,
    Experience,
    ExperienceDB,
    ExperienceDBStats,
    ExperienceType,
    FourSymbols,
    FusionStrategy,
    HierarchicalConfigurator,
    KnowledgeAging,
    MemoryNodeEnergy,
    MetaCognition,
    MetaConfigurator,
    MultimodalSignal,
    MultiViewResult,
    MultiViewRetriever,
    MultiViewStats,
    PatternInference,
    PerceptionPipeline,
    PhysicalSignal,
    PropagationConfig,
    QuerySpec,
    RelationType,
    RetrievalResult,
    RetrievalView,
    RootCauseNode,
    Season,
    SemanticCategory,
    SensorModality,
    SignalSubType,
    SignalType,
    StemBranchCode,
    StrengthState,
    TaijiMapper,
    TemporalCore,
    TemporalInfo,
    TemporalSystem,
    ThreePowers,
    TianGan,
    TimeBranch,
    TimeCodeInfo,
    TimeCycle,
    TimeStem,
    TrigramContext,
    TrigramCore,
    TrigramRelation,
    TrigramType,
    UnifiedInfoFactory,
    UnifiedInfoUnit,
    YinYang,
    analyze_balance,
    analyze_relation,
    calculate_link_weight,
    check_state_interaction,
    create_complete_energy_network,
    create_energy_bus,
    create_stem_branch,
    create_time_code,
    create_unified_unit,
    energy_from_category,
    energy_similarity,
    find_reverse_causal_chain,
    get_affinity_score,
    get_branch,
    get_cycle,
    get_cycle_name,
    get_cycle_sequence,
    get_enhance_relation,
    get_enhanced_energy,
    get_enhancing_energy,
    get_stem,
    get_suppress_chain,
    get_suppress_relation,
    get_suppressed_energy,
    get_suppressing_energy,
    is_enhancing,
    is_suppressing,
    surface_entities,
)
from mci_world_model.sdk._absolute_awareness import (
    AbsoluteAwareness,
    AwarenessLevel,
    AwarenessState,
    CausalFieldObservation,
)
from mci_world_model.sdk._absolute_trust import (
    AbsoluteTrust,
    AuditEntry,
    IntegrityCheck,
    TrustChain,
    TrustLevel,
)
from mci_world_model.sdk._action_conditioned_predictor import (
    ActionConditionedPredictor,
    CartPhysicsPredictor,
    PendulumJEPAPredictor,
    PendulumNeuralPredictor,
    PendulumPhysicsPredictor,
)
from mci_world_model.sdk._agi_protocol import (
    AGICapability,
    AGIIntegrationProtocol,
    AGIRequest,
    AGIResponse,
)
from mci_world_model.sdk._auditable_causal import (
    AuditableCausalReasoning,
    AuditStep,
    AuditTrail,
)
from mci_world_model.sdk._auto_scaler import (
    AutoScaler,
    ScaleDecision,
)

# ═══════════════════════════════════════════════════════════════════════════════
# v6.0.0 / P6 "入化": 自主律发现 + 社会认知 + 自修复 + 弹性伸缩
# ═══════════════════════════════════════════════════════════════════════════════
from mci_world_model.sdk._autonomous_law_discoverer_v2 import (
    AutonomousLawDiscovererV2,
    CausalSkeleton,
    FCIDiscoverer,
    GESDiscoverer,
    LiNGAMDiscoverer,
    NOTEARSDiscoverer,
    PCSkeletonDiscoverer,
    SystemReport,
)

# v5.0.0 Phase C: 自主记忆管理
from mci_world_model.sdk._autonomous_memory import (
    AMMConfig,
    AutonomousMemoryManager,
    MemoryAction,
    MemoryDecision,
    MemoryFeatures,
)

# v3.0.8: 批量反事实引擎
from mci_world_model.sdk._batch_counterfactual import BatchCounterfactualEngine
from mci_world_model.sdk._cached_do_calculus import CachedDoCalculus

# 因果引擎基础
from mci_world_model.sdk._causal import CausalEngine, detect_causal_link

# v3.0.2: Causal Actor
from mci_world_model.sdk._causal_actor import ActionCandidate, CausalActor, EnergyGuidedAction

# ═══════════════════════════════════════════════════════════════════════════════
# P11 "无极" 增强: 自主因果意识
# ═══════════════════════════════════════════════════════════════════════════════
from mci_world_model.sdk._causal_consciousness import (
    AutonomousCausalConsciousness,
    CausalSelfModel,
    CivilizationInfra,
    ConsciousnessLevel,
    SelfModelProperty,
)

# ═══════════════════════════════════════════════════════════════════════════════
# v13.0.0 / P13 "造化": 因果创造引擎 + 知识文明 + 因果经济
# ═══════════════════════════════════════════════════════════════════════════════
from mci_world_model.sdk._causal_creation_engine import (
    CausalCreationEngine,
    CreatedTheory,
    CreationStrategy,
    DomainKnowledge,
    TheoryStatus,
)
from mci_world_model.sdk._causal_economy import (
    CausalEconomy,
    CausalKnowledgeMarket,
    CausalKnowledgeValueModel,
    Transaction,
)

# ═══════════════════════════════════════════════════════════════════════════════
# v12.0.0 / P12 "传承": 因果联邦 + 量子推理 + 联邦治理
# ═══════════════════════════════════════════════════════════════════════════════
from mci_world_model.sdk._causal_federation_protocol import (
    CausalFederationProtocol,
    FederationConsensus,
    FederationMessage,
    FederationMessageType,
    FederationState,
    NodeRole,
    PeerInfo,
)
from mci_world_model.sdk._causal_gradient import (
    CausalGradient,
    CausalGradientPropagation,
)
from mci_world_model.sdk._causal_imagination import (
    CausalImaginationEngine,
    ImaginedWorld,
)

# v3.0.7: CausalMLP 因果推断网络
from mci_world_model.sdk._causal_mlp import CausalMLP
from mci_world_model.sdk._causal_mlp import SimpleTextEmbedder as CausalMLPTextEmbedder

# ═══════════════════════════════════════════════════════════════════════════════
# P9 "归真" 增强: 可信因果增强框架
# ═══════════════════════════════════════════════════════════════════════════════
from mci_world_model.sdk._causal_trust import (
    CausalTrustEnhancement,
    TrustCertificate,
    TrustClaim,
    TrustGrade,
    ValidationMethod,
)

# ═══════════════════════════════════════════════════════════════════════════════
# v14.0.0 / P14 "太极": 因果宇宙统一 + 跨维度推理 + 宇宙觉察/信任
# ═══════════════════════════════════════════════════════════════════════════════
from mci_world_model.sdk._causal_unification_formal import (
    Axiom,
    AxiomID,
    CausalUnificationFormal,
    ProofResult,
    Theorem,
)
from mci_world_model.sdk._causal_unification_formal import (
    ProofStatus as UnificationProofStatus,
)

# P14 "太极" 缺失导出补齐: 因果宇宙统一理论 + 终极因果智能
from mci_world_model.sdk._causal_universe_theory import (
    CausalScale,
    CausalUniverseTheory,
    ScaleResult,
)
from mci_world_model.sdk._char_tokenizer import (
    CharTokenizer,
    SimpleTextEmbedderV2,
)
from mci_world_model.sdk._cognitive_diversity import (
    CognitiveDiversity,
    DiversityHistory,
    DiversityVector,
)
from mci_world_model.sdk._cognitive_loop import (
    CognitiveLayer,
    CognitiveLoopBus,
    ErrorSignal,
    LoopHealthReport,
    PropagationResult,
)
from mci_world_model.sdk._compliance_engine import (
    ComplianceCheckResult,
    ComplianceLevel,
    ComplianceReport,
    ComplianceRule,
    ComplianceRuleEngine,
)
from mci_world_model.sdk._cosmic_awareness import (
    AwarenessScope,
    CausalAnomaly,
    CausalDomain,
    CosmicAwareness,
    CosmicMap,
    EvolutionPrediction,
)
from mci_world_model.sdk._cosmic_trust import (
    ConsistencyReport,
    CosmicCertificate,
    CosmicTrust,
    CosmicTrustLevel,
    DimensionalTrust,
    TrustDimension,
)

# v3.0.1: Cost 模块独立
from mci_world_model.sdk._cost_module import CostSignal, EnergyCostModule

# v3.0.8: 反事实推理（P3 增强）
from mci_world_model.sdk._counterfactual import (
    CounterfactualEngine,
    CounterfactualResult,
    StructuralEquationModel,
)

# v4.4.2: Phase 2 — LLM↔CEWM 反馈闭环 + 安全约束
from mci_world_model.sdk._counterfactual_oracle import (
    CFRanking,
    CFScenario,
    CounterfactualOracle,
)
from mci_world_model.sdk._creative_consciousness import (
    CreativeCausalConsciousness,
    CreativeDrive,
    CreativeState,
)
from mci_world_model.sdk._creative_trust import (
    CreativeTrust,
)
from mci_world_model.sdk._cross_dimensional_causal import (
    CrossDimensionalCausal,
)

# ═══════════════════════════════════════════════════════════════════════════════
# P10 "融通" 增强: 跨域因果迁移
# ═══════════════════════════════════════════════════════════════════════════════
from mci_world_model.sdk._cross_domain_transfer import (
    CausalKnowledge,
    CrossDomainCausalTransfer,
    DomainAdapter,
    DomainType,
    TransferResult,
    TransferStatus,
)

# v4.4.0: 具身清创 — 时序 Transformer + 手术相预测
from mci_world_model.sdk._debridement_world_model import (  # noqa: F401
    DebridementWorldModel as DebridementWorldModelV2,
)
from mci_world_model.sdk._differentiable_causal import (
    CausalParameter,
    DifferentiableCausalInference,
    OptimizationResult,
)

# v3.0.0: Pearl Do-Calculus 干预层（V2.0 起点）
from mci_world_model.sdk._do_calculus import (
    CausalGraph,
    DoCalculus,
    InterventionResult,
)
from mci_world_model.sdk._domain_sdk_base import (
    DomainResult,
    MCIDomainSDK,
)
from mci_world_model.sdk._edge_cloud_hybrid import (
    EdgeCloudHybrid,
    InferenceRequest,
    InferenceResult,
)
from mci_world_model.sdk._emergency_stop import (
    EmergencyStop,
    EmergencyStopState,
)

# v3.0.6: 能量流预测器
from mci_world_model.sdk._energy_counterfactual_bridge import (
    EnergyCounterfactualBridge,
    EnergyWhatIfResult,
)
from mci_world_model.sdk._energy_flow_predictor import EnergyFlowPredictor

# 能量一致性损失
from mci_world_model.sdk._energy_loss import (
    EnergyConsistencyLoss,
    TopologicalEnergyMatrix,
    build_energy_matrix_from_energy_bus,
)
from mci_world_model.sdk._engineering_safety_sdk import (
    EngineeringCausalResult,
    EngineeringSafetySDK,
    FMEAItem,
    SafetyParameter,
)
from mci_world_model.sdk._enhanced_perception import EnhancedPerception
from mci_world_model.sdk._eternal_protocol import (
    CausalConservationLaw,
    EternalProtocol,
    GenerationGovernance,
    ProtocolLevel,
    ProtocolViolation,
)
from mci_world_model.sdk._existence_axioms import (
    ExistenceAxiom,
    ExistenceAxiomSystem,
)
from mci_world_model.sdk._existence_realization import (
    ExistenceConfidence,
    ExistenceRealization,
    RealizationInsight,
    RealizationLevel,
)
from mci_world_model.sdk._existence_theorem import (
    ExistenceTheorem,
    TheoremProof,
    TheoremStatus,
)
from mci_world_model.sdk._existence_verify import (
    ExistenceVerify,
    IndependentVerification,
    VerificationPerspective,
    VerificationResult,
    VerificationStatus,
)
from mci_world_model.sdk._experiment_designer import (
    ExperimentDesigner,
    ExperimentPlan,
)
from mci_world_model.sdk._federated_agent_market import (
    AgentSpec,
    FederatedAgentMarket,
    TradeRecord,
)
from mci_world_model.sdk._federated_consciousness import (
    FederatedCausalConsciousness,
    FederationAwarenessState,
    FederationSelfModel,
    ReflectionResult,
    SelfModel,
)
from mci_world_model.sdk._federated_trust import (
    FederatedTrust,
    LocalTrust,
)
from mci_world_model.sdk._federated_trust import (
    TrustLevel as FederatedTrustLevel,
)
from mci_world_model.sdk._federation_arch import (
    CausalFederationArchitecture,
    CausalShard,
)
from mci_world_model.sdk._federation_audit import (
    AuditEntry as FederationAuditEntry,
)
from mci_world_model.sdk._federation_audit import (
    AuditSeverity,
    AuditStatus,
    FederationAudit,
)
from mci_world_model.sdk._final_community import (
    CommunityMember,
    CommunityState,
    EternalDeclaration,
    FinalCommunity,
    MemberRole,
    Proposal,
    ProposalStatus,
)
from mci_world_model.sdk._final_theorem import (
    Corollary,
    FinalTheorem,
    FormalPremise,
    FormalProof,
    ProofStatus,
    ProofStep,
)
from mci_world_model.sdk._force_tissue_dynamics import (
    ForceTissueDynamics,
    RemovalPrediction,
    SafetyVerdict,
)
from mci_world_model.sdk._generalized_physics import (
    GeneralizedPhysicsPredictor,
    cart_dynamics,
    double_pendulum_dynamics,
    euler_step,
    fluid_flow_dynamics,
    pendulum_dynamics,
    projectile_dynamics,
    spring_mass_dynamics,
)

# JEPA 全套
from mci_world_model.sdk._hierarchical_encoder import HierarchicalJEPAEncoder, HierarchicalState
from mci_world_model.sdk._hypothesis_generator import (
    CausalHypothesis,
    HypothesisGenerator,
)

# v5.0.0 Phase B: 增量学习框架
from mci_world_model.sdk._incremental_learning import (
    EWCConfig,
    IncrementalLearningEngine,
    IncrementalMLP,
    TaskRecord,
    TaskSpec,
)
from mci_world_model.sdk._jepa_dataset import JEPADataset
from mci_world_model.sdk._jepa_encoder import JEPAEncoder
from mci_world_model.sdk._jepa_gat_encoder import GATEncoder
from mci_world_model.sdk._jepa_gnn import GNNPredictor
from mci_world_model.sdk._jepa_predictor import (
    BeliefPropagationPredictor,
    EnergyPropagationPredictor,
    IdentityPredictor,
    JEPAPredictor,
)
from mci_world_model.sdk._jepa_trainer import JEPATrainer, JEPATrainingStats
from mci_world_model.sdk._knowledge_civilization import (
    AutonomousKnowledgeCivilization,
    CivilizationMetrics,
    KnowledgeRepository,
)

# v5.0.0: 可学习状态编码器
from mci_world_model.sdk._learnable_encoder import LearnableStateEncoder

# v5.0.0 Phase B: 可学习反事实生成
from mci_world_model.sdk._learned_counterfactual import (
    CFPrior,
    LearnedCounterfactualGenerator,
    VAEConfig,
)

# v5.0.0: 可学习动力学预测器
from mci_world_model.sdk._learned_dynamics_predictor import (
    DynamicsDataGenerator,
    LearnedDynamicsPredictor,
)
from mci_world_model.sdk._legal_compliance_sdk import (
    LegalCausalConclusion,
    LegalComplianceSDK,
    LegalEvidence,
)

# v5.0.0 Phase B: LLM↔CEWM 互校准闭环
from mci_world_model.sdk._llm_cewm_bridge import (
    BridgeConfig,
    CalibrationRecord,
    CalibrationStats,
    InferredEdge,
    LLMCEWMBridge,
)

# v5.1.0: MCTS 规划器 (F11 修复)
from mci_world_model.sdk._mcts_planner import (
    MCTSConfig,
    MCTSNode,
    MCTSPlanner,
)
from mci_world_model.sdk._medical_causal_sdk import (
    CausalDiagnosis,
    ClinicalEvidence,
    MedicalCausalSDK,
)

# P3 "赋魂": LRU 缓存 Do-Calculus
from mci_world_model.sdk._meta_learners import SLearner, TLearner
from mci_world_model.sdk._metacognition_v2 import (
    MetacognitionState,
    MetacognitionV2,
)

# v3.3.0: 多模态因果世界模型
from mci_world_model.sdk._modality_encoders import (
    AudioEncoder,
    DepthEncoder,
    ForceEncoder,
    LearnableMixin,
    ThermalEncoder,
    VisionEncoder,
)

# v3.2.0: Phase 1 P0 — 多分支推演/惊奇检测/PlanAgent
from mci_world_model.sdk._multi_branch_predictor import (
    BranchEvaluation,
    MultiBranchPredictor,
)

# v5.1.0: P2 SDK 桥接 — MultiLLM 适配器
from mci_world_model.sdk._multillm_adapter import (
    MultiLLMAdapter,
    OllamaProvider,
    OpenAIProvider,
    register_provider,
)
from mci_world_model.sdk._multimodal_fusion import (
    FusedRepresentation,
    MultimodalFusion,
)
from mci_world_model.sdk._multimodal_graph_builder import (
    MultimodalGraphBuilder,
)

# v8.0.0 / P8 "超凡": 神经符号融合V2 + 因果梯度 + 符号接地 + AGI协议 + 实验设计
from mci_world_model.sdk._neural_symbolic_fusion_v2 import (
    FusionState,
    NeuralSymbolicFusionV2,
)

# v5.0.0 Phase C: 神经符号融合世界模型
from mci_world_model.sdk._neurosymbolic_world_model import (
    NeurosymbolicConfig,
    NeurosymbolicWorldModel,
    RouteDecision,
    RouteType,
    TripleRepresentation,
)
from mci_world_model.sdk._novelty_verifier import NoveltyResult, NoveltyVerifier
from mci_world_model.sdk._online_ewc import OnlineEWC, OnlineEWCState

# P3 "赋魂": 在线弹性权重巩固
# v5.1.0: P2 SDK 桥接 — Orchestrator 桥接
from mci_world_model.sdk._orchestrator_bridge import (
    AgentResult,
    OrchestratorBridge,
    register_intent,
)

# ═══════════════════════════════════════════════════════════════════════════════
# v15.0.0 / P15 "无量" 桥接模块: 因果宇宙扩展 + 多宇宙联邦 + 跨宇宙推理
# ═══════════════════════════════════════════════════════════════════════════════
from mci_world_model.sdk._p15_causal_universe_bridge import (
    CausalUniverseExpansion,
    CrossUniverseCausal,
    ExpansionPhase,
    FederationBridge,
    MultiUniverseFederation,
    UniverseScale,
    UniverseSpec,
)

# ═══════════════════════════════════════════════════════════════════════════════
# v16.0.0 / P16 "永恒" 桥接模块: 永恒因果智能 + 时间因果推理 + 自复制
# ═══════════════════════════════════════════════════════════════════════════════
from mci_world_model.sdk._p16_eternal_intelligence_bridge import (
    EternalCausalIntelligence,
    EternalKnowledgeSpec,
    EternalPhase,
    SelfReplicatingCausal,
    TemporalCausalReasoning,
    TemporalScope,
)

# ═══════════════════════════════════════════════════════════════════════════════
# v17.0.0 / P17 "共演" 桥接模块: 因果物理共演化 + 因果力理论 + 统一场
# ═══════════════════════════════════════════════════════════════════════════════
from mci_world_model.sdk._p17_coevolution_bridge import (
    CausalForceTheory,
    CausalPhysicalCoevolution,
    CausalPhysicalUnifiedField,
    CoevolutionMode,
    CoevolutionState,
    ForceType,
)

# ═══════════════════════════════════════════════════════════════════════════════
# v18.0.0 / P18 "创生" 桥接模块: 因果宇宙创生 + 创世论 + 多实相拓扑
# ═══════════════════════════════════════════════════════════════════════════════
from mci_world_model.sdk._p18_genesis_bridge import (
    CausalCosmogony,
    CausalUniverseGenesis,
    CreatedUniverse,
    GenesisMode,
    GenesisSpec,
    MultiRealityTopology,
    RealityTopology,
)

# ═══════════════════════════════════════════════════════════════════════════════
# v19.0.0 / P19 "超因" 桥接模块: 元因果推理 + 超越因果 + 前因果存在
# ═══════════════════════════════════════════════════════════════════════════════
from mci_world_model.sdk._p19_transcendence_bridge import (
    BeyondCausality,
    BeyondDomain,
    BeyondObservation,
    MetaCausalPattern,
    MetaCausalReasoning,
    PreCausalExistence,
    ReasoningTier,
)

# 参数化记忆
from mci_world_model.sdk._parametric_memory import (
    ParametricMemory,
    ParametricMemoryConfig,
    TrainingSample,
)
from mci_world_model.sdk._parametric_memory import (  # type: ignore
    SimpleTextEmbedder as MemoryTextEmbedder,
)

# v5.1.0: PearlChain 协调器 (F8 修复)
from mci_world_model.sdk._pearl_chain import (
    L1ObservationResult,
    PearlChain,
    PearlChainResult,
)

# v5.0.0: 持久化经验记忆
from mci_world_model.sdk._persistent_memory import (
    PersistentExperienceMemory,
    PersistentMemoryConfig,
    VectorStore,
)

# v3.1.0: PhysicalGraphBuilder 物理量→因果边转换器
from mci_world_model.sdk._physical_graph_builder import (
    PhysicalGraphBuilder,
    signals_to_timeline,
)
from mci_world_model.sdk._plan_agent import (
    Plan,
    PlanAgent,
)
from mci_world_model.sdk._plugin_interface import (
    PluginContext,
    PluginHook,
    PluginInterface,
    PluginManager,
    PluginMetadata,
)
from mci_world_model.sdk._protocols import (
    CartStateParser,
    GenericStateParser,
    PendulumStateParser,
    PredictorProtocol,
    StateParserProtocol,
    StateParserRegistry,
)

# P12 "传承" 缺失导出补齐: 量子因果推理 + 量子-经典桥接
from mci_world_model.sdk._quantum_causal_inference import (
    CausalEffectResult,
    QuantumCausalInference,
)
from mci_world_model.sdk._quantum_causal_inference import (  # type: ignore
    QuantumCircuit as QCIQuantumCircuit,
)
from mci_world_model.sdk._quantum_causal_inference import (  # type: ignore
    QuantumClassicalBridge as QCIClassicalBridge,
)
from mci_world_model.sdk._quantum_classical_bridge import (
    EncodingMethod,
    QuantumBackend,
    QuantumErrorMitigator,
)
from mci_world_model.sdk._quantum_classical_bridge import (
    QuantumCircuit as QCBQuantumCircuit,
)
from mci_world_model.sdk._quantum_classical_bridge import (
    QuantumClassicalBridge as QCBClassicalBridge,
)
from mci_world_model.sdk._quantum_classical_bridge import (
    QuantumResult as QCBQuantumResult,
)
from mci_world_model.sdk._quantum_classical_bridge import (
    QuantumResult as QCIQuantumResult,
)

# Reflection QA 合成器
from mci_world_model.sdk._reflection_synthesizer import (
    ReflectionSynthesizer,
    SynthesizedQAPair,
)

# v4.5.0: Phase 3 — 手术机器人桥接 + 硬实时
from mci_world_model.sdk._robot_state import (
    RobotAction,
    RobotWorldState,
)
from mci_world_model.sdk._ros2_bridge import (
    BridgeState,
    ROS2Bridge,
    ROS2BridgeConfig,
)

# 扩展安全约束 (Phase 3)
from mci_world_model.sdk._safety import (
    AccelerationLimitConstraint,
    ForceLimitConstraint,
    JointLimitConstraint,
    PositionBoundConstraint,
    SafetyCheckResult,
    SafetyConstraint,
    SafetyMonitor,
    SelfCollisionConstraint,
    ToolForceConstraint,
    VelocityLimitConstraint,
    WorkspaceBoundConstraint,
)

# v5.1.0: 认知/语义安全约束 (F7 修复: 8→13类)
from mci_world_model.sdk._safety_cognitive import (
    CognitiveSafetyConstraint,
    ContentSafetyConstraint,
    SocialSafetyConstraint,
    TemporalSafetyConstraint,
    ValueAlignmentConstraint,
)
from mci_world_model.sdk._scientific_discovery import (
    DiscoveryReport,
    ScientificDiscoveryPipeline,
)
from mci_world_model.sdk._scientific_discovery import (
    DiscoveryStage as MCI_DiscoveryStage,
)
from mci_world_model.sdk._self_repair_cognition import (
    AnomalyReport,
    RepairAction,
    SelfRepairCognition,
)

# SIGReg 嵌入正则化
from mci_world_model.sdk._sigreg import SIGReg, apply_sigreg_to_index
from mci_world_model.sdk._simple_maml import MAMLTask, SimpleMAML
from mci_world_model.sdk._social_cognition import (
    AgentAction,
    AgentModel,
    NashEquilibriumResult,
    SocialCognition,
)
from mci_world_model.sdk._spectral_causal import BayesianCausal, FourierCausal, GaussianDAG
from mci_world_model.sdk._temporal_causal import GrangerCausality, LaggedCorrelationScanner, TemporalCausalReport
from mci_world_model.sdk._the_absolute import (
    AbsoluteProperty,
    GeneratedStructure,
    TheAbsolute,
)

# v4.4.0: 小模型 Transformer (清创文本推理)
from mci_world_model.sdk._tissue_classifier import (
    TISSUE_EPITHELIAL,
    TISSUE_GRANULATION,
    TISSUE_NAMES,
    TISSUE_NECROTIC,
    TISSUE_SLOUGH,
    TissueClassifier,
    TissueResult,
)

# v5.1.0: TrueJEPA 编码器 (F6 修复)
from mci_world_model.sdk._true_jepa_encoder import (
    TrueJEPAConfig,
    TrueJEPAEncoder,
)
from mci_world_model.sdk._ultimate_causal_intelligence import (
    AutonomousAction,
    Capability,
    CapabilityStatus,
    ExistenceMode,
    ExistenceReport,
    UltimateCausalIntelligence,
)

# v3.2.0: 动作条件化预测器
# ═══════════════════════════════════════════════════════════════════════════════
# v20.0.0 / P20 "归一": 终极统一 + 存在定理 + 绝对存在 + 归一意识 + 永恒协议
# ═══════════════════════════════════════════════════════════════════════════════
from mci_world_model.sdk._ultimate_unification import (
    ExistenceInvariant,
    FieldTensor,
    UltimateUnification,
    UnificationLevel,
    UnificationReport,
)
from mci_world_model.sdk._unified_consciousness import (
    UnifiedCausalConsciousness,
    UnifiedState,
)

# v6.0.0 / P6 "入化" + v7.0.0 / P7 "立业": 统一编码 + 元认知 + 领域SDK
from mci_world_model.sdk._unified_modal_encoder import (
    AlignmentResult,
    EncodingResult,
    ModalityProjection,
    UnifiedModalEncoder,
)

# v5.0.0 Phase C: 多模态物理感知编码器
from mci_world_model.sdk._visual_encoder import (
    MultimodalPair,
    VisualEncoder,
    VisualEncoderConfig,
)

# v5.1.0: 工作记忆增强器
from mci_world_model.sdk._working_memory_enhancer import (
    MemoryRetrievalResult,
    WorkingMemoryEnhancer,
    WorkingMemoryEnhancerConfig,
)
from mci_world_model.sdk._world_model import (
    CausalWorldModelState,
    MCIWorldModel,
    # v3.0.1: STM 工作记忆
    TrajectoryStep,
    WorkingMemory,
)

# v3.2.0: 独立世界状态抽象
from mci_world_model.sdk._world_state import (
    Action,
    CartAction,
    CartState,
    DoublePendulumAction,
    DoublePendulumState,
    MultimodalWorldState,
    PendulumAction,
    PendulumState,
    WorldState,
)
from mci_world_model.sdk._zvec_store import (
    EmbeddingStoreConfig,
    ZvecEmbeddingStore,
)

# 贝叶斯增强器
from mci_world_model.sdk.bayesian_augmenter import (
    AccuracyRecord,
    BayesianAugmenter,
    ComparisonDelta,
    EnhancedOutput,
)

__all__ = [
    "AccuracyRecord",
    # v3.0.2: Causal Actor
    "ActionCandidate",
    # Bayesian Augmenter
    "BayesianAugmenter",
    # Spectral Causal
    "BayesianCausal",
    # Bayesian (re-export from _sys)
    "BayesianNetwork",
    "BayesianReasoningSystem",
    # v3.4.0: 闭环基础设施
    "CognitiveLayer",
    "CognitiveLoopBus",
    "ErrorSignal",
    "LoopHealthReport",
    "PropagationResult",
    "CognitiveScoreCard",
    "CognitiveDiversity",
    "DiversityHistory",
    "DiversityVector",
    "RootCauseNode",
    # v3.5.0: 经验记忆系统
    "Experience",
    "ExperienceDB",
    "ExperienceDBStats",
    "ExperienceType",
    "FusionStrategy",
    "MultiViewResult",
    "MultiViewRetriever",
    "MultiViewStats",
    "QuerySpec",
    "RetrievalResult",
    "RetrievalView",
    # v3.6.0: 认知环闭环
    "ActionCostResult",
    "ActionGapConfig",
    "ActionGapMetric",
    "CausalEdge",
    "CausalUpdater",
    "CausalUpdaterStats",
    "EdgeAction",
    "UpdateRecord",
    # v3.7.0: 认知诊断系统
    "ChangeType",
    "DiagnosisResult",
    "FailurePattern",
    "HardCoreViolation",
    "MetaDiagnoser",
    "MetaDiagnoserStats",
    "NegativeHeuristic",
    "NegativeHeuristicStats",
    "PatternMatch",
    "ProposedChange",
    "ProtectiveBeltSuggestion",
    "RootCauseChain",
    "RuleSeverity",
    "SeverityLevel",
    "SurpriseSignal",
    # v3.0.8: BatchCounterfactualEngine
    "BatchCounterfactualEngine",
    "BeliefPropagationPredictor",
    # v4.4.2: Phase 2 — LLM↔CEWM 反馈闭环
    "CFScenario",
    "CFRanking",
    "CounterfactualOracle",
    # v4.4.2: Phase 2 — 安全约束层
    "ForceLimitConstraint",
    "PositionBoundConstraint",
    "SafetyCheckResult",
    "SafetyConstraint",
    "SafetyMonitor",
    "VelocityLimitConstraint",
    # 因果引擎基础
    "CausalEngine",
    "CausalGraph",
    "CausalWorldModelState",
    "ComparisonDelta",
    # v3.0.1/v3.0.3: Configurator
    "ConfigAction",
    # Counterfactual
    "CounterfactualEngine",
    "CounterfactualResult",
    "EnergyGuidedAction",
    # v3.0.1: Cost Module
    "CostSignal",
    # v3.0.2: Causal Actor
    "CausalActor",
    # v3.0.7: CausalMLP
    "CausalMLP",
    # Pearl Do-Calculus
    "DoCalculus",
    "CachedDoCalculus",
    "SLearner",
    "TLearner",
    "GrangerCausality",
    "LaggedCorrelationScanner",
    "TemporalCausalReport",
    # v3.2.0: 独立世界状态
    "Action",
    "ActionConditionedPredictor",
    # v3.3.0: 模态编码器
    "AudioEncoder",
    "MultimodalWorldState",
    "PendulumAction",
    "PendulumJEPAPredictor",
    "PendulumNeuralPredictor",
    "PendulumPhysicsPredictor",
    "PendulumState",
    "CartAction",
    "CartPhysicsPredictor",
    "CartState",
    "CartStateParser",
    "GenericStateParser",
    "DoublePendulumAction",
    "DoublePendulumState",
    "GeneralizedPhysicsPredictor",
    "cart_dynamics",
    "double_pendulum_dynamics",
    "euler_step",
    "fluid_flow_dynamics",
    "pendulum_dynamics",
    "projectile_dynamics",
    "spring_mass_dynamics",
    # v4.5.0: Phase 3 — 手术机器人桥接
    "RobotAction",
    "RobotWorldState",
    "EmergencyStop",
    "EmergencyStopState",
    "DeadlineConfig",
    "DeadlineMonitor",
    "DeadlineStats",
    "BridgeState",
    "ROS2Bridge",
    "ROS2BridgeConfig",
    # v4.5.0: Phase 3 扩展安全约束
    "AccelerationLimitConstraint",
    "JointLimitConstraint",
    "SelfCollisionConstraint",
    "ToolForceConstraint",
    "WorkspaceBoundConstraint",
    # v3.2.0: Phase 1 P0
    "MultiBranchPredictor",
    "BranchEvaluation",
    "PendulumStateParser",
    "PredictorProtocol",
    "StateParserProtocol",
    "StateParserRegistry",
    "SurpriseDetector",
    "SurpriseSignal",
    "PlanAgent",
    "Plan",
    # v3.3.0: 多模态
    "MultimodalFusion",
    "FusedRepresentation",
    "MultimodalGraphBuilder",
    "VisionEncoder",
    "ThermalEncoder",
    "WorldState",
    # Energy Loss
    "EnergyConsistencyLoss",
    "EnergyCostModule",
    "EnergyCounterfactualBridge",
    "EnergyWhatIfResult",
    "EnergyFlowPredictor",
    "EnergyPropagationPredictor",
    "EnhancedOutput",
    "FourierCausal",
    "GNNPredictor",
    "GaussianDAG",
    "GaussianDistribution",
    # v3.0.2: H-JEPA
    "HierarchicalJEPAEncoder",
    # v3.0.3: Configurator 分层
    "HierarchicalConfigurator",
    "HierarchicalState",
    "IdentityPredictor",
    "JEPADataset",
    # JEPA
    "JEPAEncoder",
    "JEPAPredictor",
    "JEPATrainer",
    # v5.0.0: Learnable State Encoder
    "LearnableStateEncoder",
    # v5.1.0: TrueJEPA Encoder (F6 fix)
    "TrueJEPAConfig",
    "TrueJEPAEncoder",
    # v5.1.0: PearlChain (F8 fix)
    "L1ObservationResult",
    "PearlChain",
    "PearlChainResult",
    # v5.1.0: 认知/语义安全约束 (F7 fix: 8→13类)
    "CognitiveSafetyConstraint",
    "ContentSafetyConstraint",
    "SocialSafetyConstraint",
    "TemporalSafetyConstraint",
    "ValueAlignmentConstraint",
    # v5.1.0: MCTS 规划器 (F11 fix)
    "MCTSConfig",
    "MCTSNode",
    "MCTSPlanner",
    # v5.1.0: 工作记忆增强器
    "MemoryRetrievalResult",
    "WorkingMemoryEnhancer",
    "WorkingMemoryEnhancerConfig",
    # v5.1.0: P2 SDK 桥接 — MultiLLM
    "MultiLLMAdapter",
    "OllamaProvider",
    "OpenAIProvider",
    "register_provider",
    # v5.1.0: P2 SDK 桥接 — Orchestrator
    "AgentResult",
    "OrchestratorBridge",
    # v5.0.0: Learned Dynamics Predictor
    "DynamicsDataGenerator",
    "LearnedDynamicsPredictor",
    # v5.0.0: Persistent Experience Memory
    "PersistentExperienceMemory",
    "PersistentMemoryConfig",
    "VectorStore",
    # v5.0.0 Phase B: LLM↔CEWM Bridge
    "BridgeConfig",
    "CalibrationRecord",
    "CalibrationStats",
    "InferredEdge",
    "LLMCEWMBridge",
    # v5.0.0 Phase B: Learned Counterfactual
    "CFPrior",
    "CounterfactualResult",
    "LearnedCounterfactualGenerator",
    "VAEConfig",
    # v5.0.0 Phase B: Incremental Learning
    "EWCConfig",
    "IncrementalLearningEngine",
    "OnlineEWC",
    "OnlineEWCState",
    "SimpleMAML",
    "MAMLTask",
    "IncrementalMLP",
    "TaskRecord",
    "TaskSpec",
    # v5.0.0 Phase C: Neurosymbolic World Model
    "InferenceResult",
    "NeurosymbolicConfig",
    "NeurosymbolicWorldModel",
    "RouteType",
    "TripleRepresentation",
    # v5.0.0 Phase C: Visual Encoder
    "MultimodalPair",
    "VisualEncoder",
    "VisualEncoderConfig",
    # v5.0.0 Phase C: Autonomous Memory Manager
    "AMMConfig",
    "AutonomousMemoryManager",
    "MemoryAction",
    "MemoryDecision",
    "MemoryFeatures",
    # V3.0.0 核心
    "MCIWorldModel",
    # v3.1.0: 多模态信号
    "MultimodalSignal",
    # v3.0.1: Configurator
    "MetaConfigurator",
    # Parametric Memory
    "ParametricMemory",
    "ParametricMemoryConfig",
    # ── v4.3.3 补齐: 未导出符号选择性导出 ──
    "BeyondObservation",
    "build_energy_matrix_from_energy_bus",
    "CausalMLPTextEmbedder",
    "ComplianceRule",
    "detect_causal_link",
    "EnhancedPerception",
    "GATEncoder",
    "InterventionResult",
    "JEPATrainingStats",
    "LearnableMixin",
    "MemoryTextEmbedder",
    "MCI_DiscoveryStage",
    "NoveltyResult",
    "NoveltyVerifier",
    "register_intent",
    "RouteDecision",
    "TrainingSample",
    # Reflection
    "ReflectionSynthesizer",
    # v3.0.3: Perception
    "PerceptionPipeline",
    # v3.1.0: PhysicalGraphBuilder
    "PhysicalGraphBuilder",
    # SIGReg
    "SIGReg",
    "SynthesizedQAPair",
    # v3.0.8: StructuralEquationModel
    "StructuralEquationModel",
    # v3.1.0: signals_to_timeline 工具函数
    "signals_to_timeline",
    # v3.1.0: SignalType 枚举
    "SignalType",
    "TopologicalEnergyMatrix",
    # v3.2.0: 感知-行动接口范式
    "ActionCommand",
    "ActionPriority",
    "ActionResult",
    "ActuatorChannel",
    "PhysicalSignal",
    "SensorModality",
    "SignalSubType",
    # v3.0.1: STM
    "TrajectoryStep",
    "WorkingMemory",
    "apply_sigreg_to_index",
    # ── v3.3.1: 能量Bus核心能力全面释放 ──
    # 人层 (Energy/Human)
    "EnergyBus",
    "EnergyChannel",
    "EnergyCore",
    "EnergyLayer",
    "EnergyNode",
    "EnergySignal",
    "EnergyState",
    "EnergyFlow",
    "EnergyBalanceResult",
    "PropagationConfig",
    "create_complete_energy_network",
    "create_energy_bus",
    # 五行生克
    "RelationType",
    "analyze_relation",
    "analyze_balance",
    "get_enhance_relation",
    "get_suppress_relation",
    "get_enhancing_energy",
    "get_suppressing_energy",
    "get_enhanced_energy",
    "get_suppressed_energy",
    "is_enhancing",
    "is_suppressing",
    "get_affinity_score",
    "calculate_link_weight",
    "find_reverse_causal_chain",
    "surface_entities",
    "get_cycle_sequence",
    "FOUR_SYMBOLS_TO_ENERGY",
    "RELATION_STRENGTH",
    "MemoryNodeEnergy",
    # 能量网络 (c2)
    "EnergyNetwork",
    "check_state_interaction",
    "energy_from_category",
    "energy_similarity",
    "ENERGY_ENHANCE_MAP",
    "ENERGY_SUPPRESS_MAP",
    "STATE_STRENGTH_MAP",
    # 基础枚举
    "EnergyType",
    "EnergyRelation",
    "EnergyPattern",
    "FourSymbols",
    "Season",
    "StrengthState",
    "ThreePowers",
    # 天层 (Temporal/Sky)
    "TemporalCore",
    "StemBranchCode",
    "create_stem_branch",
    "get_cycle_name",
    # 地层 (Spatial/Earth)
    "MetaCognition",
    "CognitiveGap",
    "CognitiveScoreCard",
    "KnowledgeAging",
    "PatternInference",
    "TrigramContext",
    # ── v3.3.1: 时空/时序/分类/因果/枚举 全面穿透 ──
    # 时空数据表
    "BRANCH_CHONG",
    "BRANCH_CHONG_MAP",
    "BRANCH_HE",
    "BRANCH_HE_MAP",
    "BRANCH_HIDDEN_STEM_MAP",
    "BRANCH_SANHE",
    "BRANCH_SANHE_MAP",
    "BRANCH_XING",
    "MONTH_ENERGY_STATE",
    "SEMANTIC_CATEGORY",
    "SEMANTIC_CATEGORY_NAMES",
    "STEM_CHONG",
    "STEM_CHONG_MAP",
    "STEM_HE",
    "STEM_HE_MAP",
    "STRENGTH_MULTIPLIER",
    "TIME_BRANCH_ENERGY",
    "TIME_BRANCHES",
    "TIME_STEMS",
    "TRIGRAM_BODY_MAP",
    "TRIGRAM_ENERGY_MAP",
    # 时序编码
    "TimeBranch",
    "TimeCodeInfo",
    "TimeCycle",
    "TimeStem",
    "create_time_code",
    "get_branch",
    "get_cycle",
    "get_stem",
    # 时序系统
    "DiZhi",
    "DynamicPriority",
    "TemporalInfo",
    "TemporalSystem",
    "TianGan",
    # 维度映射
    "TaijiMapper",
    # 分类/语义
    "CATEGORY_ANCHORS",
    "ENERGY_TO_CATEGORY",
    "KEYWORDS_TO_CATEGORY",
    "MEMORY_TYPE_TO_CATEGORY",
    "SemanticCategory",
    # 因果引擎
    "BayesianEngine",
    "CategoryCausalEngine",
    "CausalChain",
    "CausalInference",
    "EnergyMemoryNode",
    # 基础枚举
    "BranchRelation",
    "TrigramRelation",
    "TrigramType",
    "YinYang",
    # 兼容别名
    "EnergyEnumRelation",
    "EnergyEnumType",
    # 三才合一
    "UnifiedInfoFactory",
    "UnifiedInfoUnit",
    "create_unified_unit",
    # 地层
    "TrigramCore",
    # 能量关系
    "get_suppress_chain",
    # ── v6.0.0 / P6 "入化" ──
    "AutonomousLawDiscovererV2",
    "CausalSkeleton",
    "CAMDiscoverer",
    "FCIDiscoverer",
    "GESDiscoverer",
    "LiNGAMDiscoverer",
    "NOTEARSDiscoverer",
    "PCSkeletonDiscoverer",
    "SystemReport",
    "AgentAction",
    "AgentModel",
    "NashEquilibriumResult",
    "SocialCognition",
    "AnomalyReport",
    "RepairAction",
    "SelfRepairCognition",
    "AutoScaler",
    "ScaleDecision",
    "ComplianceCheckResult",
    "ComplianceLevel",
    "ComplianceReport",
    "ComplianceRuleEngine",
    "PluginContext",
    "PluginHook",
    "PluginInterface",
    "PluginManager",
    "PluginMetadata",
    # ── v6.0.0 / P6 + v7.0.0 / P7 ──
    "AlignmentResult",
    "EncodingResult",
    "ModalityProjection",
    "UnifiedModalEncoder",
    "MetacognitionState",
    "MetacognitionV2",
    "CausalDiagnosis",
    "ClinicalEvidence",
    "MedicalCausalSDK",
    "LegalCausalConclusion",
    "LegalComplianceSDK",
    "LegalEvidence",
    "EngineeringCausalResult",
    "EngineeringSafetySDK",
    "FMEAItem",
    "SafetyParameter",
    "AuditStep",
    "AuditTrail",
    "AuditableCausalReasoning",
    "EdgeCloudHybrid",
    "InferenceRequest",
    "InferenceResult",
    # ── v7.0.0 / P7 + v8.0.0 / P8 ──
    "CrossModalCausalLink",
    "CrossModalCausalReasoner",
    "CrossModalCausalResult",
    "CausalImaginationEngine",
    "ImaginedWorld",
    "CausalParameter",
    "DifferentiableCausalInference",
    "OptimizationResult",
    "DomainResult",
    "MCIDomainSDK",
    "DiscoveryReport",
    "ScientificDiscoveryPipeline",
    "CausalHypothesis",
    "HypothesisGenerator",
    # ── v8.0.0 / P8 "超凡" ──
    "FusionState",
    "NeuralSymbolicFusionV2",
    "CausalGradient",
    "CausalGradientPropagation",
    "GroundingEntry",
    "SymbolGroundingLearning",
    "AGICapability",
    "AGIIntegrationProtocol",
    "AGIRequest",
    "AGIResponse",
    "ExperimentDesigner",
    "ExperimentPlan",
    # ── v20.0.0 / P20 "归一" ──
    "AbsoluteAwareness",
    "AbsoluteProperty",
    "AbsoluteTrust",
    "AuditEntry",
    "AwarenessLevel",
    "AwarenessState",
    "CausalConservationLaw",
    "CausalFieldObservation",
    "Corollary",
    "EternalProtocol",
    "ExistenceAxiom",
    "ExistenceAxiomSystem",
    "ExistenceConfidence",
    "ExistenceInvariant",
    "ExistenceRealization",
    "ExistenceTheorem",
    "ExistenceVerify",
    "FieldTensor",
    "FinalTheorem",
    "FormalPremise",
    "FormalProof",
    "GeneratedStructure",
    "GenerationGovernance",
    "IndependentVerification",
    "IntegrityCheck",
    "ProtocolLevel",
    "ProtocolViolation",
    "ProofStatus",
    "ProofStep",
    "RealizationInsight",
    "RealizationLevel",
    "TheAbsolute",
    "TheoremProof",
    "TheoremStatus",
    "TrustChain",
    "TrustLevel",
    "UltimateUnification",
    "UnifiedCausalConsciousness",
    "UnifiedState",
    "UnificationLevel",
    "UnificationReport",
    "VerificationPerspective",
    "VerificationResult",
    "VerificationStatus",
    # ── v12.0.0 / P12 "传承" ──
    "AgentSpec",
    "CausalFederationArchitecture",
    "CausalFederationProtocol",
    "CausalShard",
    "FederationAudit",
    "FederationAuditEntry",
    "FederatedAgentMarket",
    "FederatedCausalConsciousness",
    "FederatedTrust",
    "FederatedTrustLevel",
    "FederationAwarenessState",
    "FederationConsensus",
    "FederationMessage",
    "FederationMessageType",
    "FederationSelfModel",
    "FederationState",
    "LocalTrust",
    "NodeRole",
    "PeerInfo",
    "ReflectionResult",
    "SelfModel",
    "TradeRecord",
    "TrustCertificate",
    "AuditSeverity",
    "AuditStatus",
    # ── P12 "传承" quantum 补齐 ──
    "CausalEffectResult",
    "QuantumCausalInference",
    "QCIQuantumCircuit",
    "QCIClassicalBridge",
    "QCIQuantumResult",
    "EncodingMethod",
    "QuantumBackend",
    "QCBQuantumCircuit",
    "QCBClassicalBridge",
    "QuantumErrorMitigator",
    "QCBQuantumResult",
    # ── v13.0.0 / P13 "造化" ──
    "AutonomousKnowledgeCivilization",
    "CausalCreationEngine",
    "CausalEconomy",
    "CausalKnowledgeMarket",
    "CausalKnowledgeValueModel",
    "CivilizationMetrics",
    "CreatedTheory",
    "CreativeCausalConsciousness",
    "CreativeDrive",
    "CreativeState",
    "CreativeTrust",
    "CreationStrategy",
    "DomainKnowledge",
    "KnowledgeRepository",
    "TheoryStatus",
    "Transaction",
    # ── v14.0.0 / P14 "太极" ──
    "AwarenessScope",
    "Axiom",
    "AxiomID",
    "CausalAnomaly",
    "CausalDomain",
    "CausalUnificationFormal",
    "ConsistencyReport",
    "CosmicAwareness",
    "CosmicCertificate",
    "CosmicMap",
    "CosmicTrust",
    "CosmicTrustLevel",
    "CrossDimensionalCausal",
    "DimensionalTrust",
    "EvolutionPrediction",
    "ProofResult",
    "Theorem",
    "TrustDimension",
    "UnificationProofStatus",
    # ── P14 "太极" 补齐 ──
    "CausalScale",
    "CausalUniverseTheory",
    "ScaleResult",
    "AutonomousAction",
    "Capability",
    "CapabilityStatus",
    "ExistenceMode",
    "ExistenceReport",
    "UltimateCausalIntelligence",
    # ── v20.0.0 / P20 "归一" 补全 ──
    "CommunityMember",
    "CommunityState",
    "EternalDeclaration",
    "FinalCommunity",
    "MemberRole",
    "Proposal",
    "ProposalStatus",
    # ── P15 桥接导出 ──
    "CausalUniverseExpansion",
    "CrossUniverseCausal",
    "ExpansionPhase",
    "FederationBridge",
    "MultiUniverseFederation",
    "UniverseScale",
    "UniverseSpec",
    # ── P16 桥接导出 ──
    "EternalCausalIntelligence",
    "EternalKnowledgeSpec",
    "EternalPhase",
    "SelfReplicatingCausal",
    "TemporalCausalReasoning",
    "TemporalScope",
    # ── P17 桥接导出 ──
    "CausalForceTheory",
    "CausalPhysicalCoevolution",
    "CausalPhysicalUnifiedField",
    "CoevolutionMode",
    "CoevolutionState",
    "ForceType",
    # ── P18 桥接导出 ──
    "CausalCosmogony",
    "CausalUniverseGenesis",
    "CreatedUniverse",
    "GenesisMode",
    "GenesisSpec",
    "MultiRealityTopology",
    "RealityTopology",
    # ── P19 桥接导出 ──
    "BeyondCausality",
    "BeyondDomain",
    "MetaCausalPattern",
    "MetaCausalReasoning",
    "PreCausalExistence",
    "ReasoningTier",
    # ── P9 增强导出 ──
    "CausalTrustEnhancement",
    "TrustCertificate",
    "TrustClaim",
    "TrustGrade",
    "ValidationMethod",
    # ── P10 增强导出 ──
    "CausalKnowledge",
    "CrossDomainCausalTransfer",
    "DomainAdapter",
    "DomainType",
    "TransferResult",
    "TransferStatus",
    # ── P11 增强导出 ──
    "AutonomousCausalConsciousness",
    "CausalSelfModel",
    "CivilizationInfra",
    "ConsciousnessLevel",
    "SelfModelProperty",    "DebridementSample",
    "DepthEncoder",
    "EmbeddingStoreConfig",
    "ForceEncoder",
    "SyntheticDebridementGenerator",
    "TISSUE_NECROTIC",
    "TISSUE_SLOUGH",
    "TISSUE_GRANULATION",
    "TISSUE_EPITHELIAL",
    "TISSUE_NAMES",
    "TissueClassifier",
    "TissueResult",
    "WoundDatasetAdapter",
    "ZvecEmbeddingStore",
    "CharTokenizer",
    "DebridementConfig",
    "DebridementWorldModel",
    "ForceTissueDynamics",
    "RemovalPrediction",
    "SafetyVerdict",
    "SimpleTextEmbedderV2",

]

# ═══════════════════════════════════════════════════════════════════════════════
# v3.5.0: _sys 延迟注入 (Late-binding injection)
# ───────────────────────────────────────────────────────────────────────────────
# 将 v3.5.0 新增符号注入到 _sys 命名空间，替换 None 占位符。
# 必须在此处执行，因为 _sys/__init__.py 不能直接导入 sdk 模块（循环依赖）。
# 注入后，world_model.py 从 _sys 导入时将获得真实类而非 None。
import mci_world_model._sys as _sys_module

# 直接从源模块导入（不能用 sdk 命名空间中的变量，因为它们是 None 占位符）
from mci_world_model.sdk._experience_memory import (
    Experience as _exp_Experience,
)
from mci_world_model.sdk._experience_memory import (
    ExperienceDB as _exp_ExperienceDB,
)
from mci_world_model.sdk._experience_memory import (
    ExperienceDBStats as _exp_ExperienceDBStats,
)
from mci_world_model.sdk._experience_memory import (
    ExperienceType as _exp_ExperienceType,
)
from mci_world_model.sdk._experience_memory import (
    RetrievalResult as _exp_RetrievalResult,
)
from mci_world_model.sdk._multi_view_retriever import (
    FusionStrategy as _mvr_FusionStrategy,
)
from mci_world_model.sdk._multi_view_retriever import (
    MultiViewResult as _mvr_MultiViewResult,
)
from mci_world_model.sdk._multi_view_retriever import (
    MultiViewRetriever as _mvr_MultiViewRetriever,
)
from mci_world_model.sdk._multi_view_retriever import (
    MultiViewStats as _mvr_MultiViewStats,
)
from mci_world_model.sdk._multi_view_retriever import (
    QuerySpec as _mvr_QuerySpec,
)
from mci_world_model.sdk._multi_view_retriever import (
    RetrievalView as _mvr_RetrievalView,
)

_V350_INJECTIONS = {
    "Experience": _exp_Experience,
    "ExperienceDB": _exp_ExperienceDB,
    "ExperienceDBStats": _exp_ExperienceDBStats,
    "ExperienceType": _exp_ExperienceType,
    "FusionStrategy": _mvr_FusionStrategy,
    "MultiViewResult": _mvr_MultiViewResult,
    "MultiViewRetriever": _mvr_MultiViewRetriever,
    "MultiViewStats": _mvr_MultiViewStats,
    "QuerySpec": _mvr_QuerySpec,
    "RetrievalResult": _exp_RetrievalResult,
    "RetrievalView": _mvr_RetrievalView,
}

# 注入到 _sys 命名空间
for _name, _cls in _V350_INJECTIONS.items():
    setattr(_sys_module, _name, _cls)

# 重新绑定 sdk 层引用（注入前 sdk 层已从 _sys 拿到 None 占位符）
import mci_world_model.sdk as _sdk_self

for _name, _cls in _V350_INJECTIONS.items():
    setattr(_sdk_self, _name, _cls)

# ═══════════════════════════════════════════════════════════════════════════════
# v3.6.0: _sys 延迟注入 (Late-binding injection) — 因果图自适应 + 行动距离
# ═══════════════════════════════════════════════════════════════════════════════
from mci_world_model.sdk._action_gap import (
    ActionCostResult as _ag_ActionCostResult,
)
from mci_world_model.sdk._action_gap import (
    ActionGapConfig as _ag_ActionGapConfig,
)
from mci_world_model.sdk._action_gap import (
    ActionGapMetric as _ag_ActionGapMetric,
)
from mci_world_model.sdk._causal_updater import (
    CausalEdge as _cu_CausalEdge,
)
from mci_world_model.sdk._causal_updater import (
    CausalUpdater as _cu_CausalUpdater,
)
from mci_world_model.sdk._causal_updater import (
    CausalUpdaterStats as _cu_CausalUpdaterStats,
)
from mci_world_model.sdk._causal_updater import (
    EdgeAction as _cu_EdgeAction,
)
from mci_world_model.sdk._causal_updater import (
    UpdateRecord as _cu_UpdateRecord,
)

_V360_INJECTIONS = {
    "CausalUpdater": _cu_CausalUpdater,
    "CausalEdge": _cu_CausalEdge,
    "CausalUpdaterStats": _cu_CausalUpdaterStats,
    "EdgeAction": _cu_EdgeAction,
    "UpdateRecord": _cu_UpdateRecord,
    "ActionGapMetric": _ag_ActionGapMetric,
    "ActionCostResult": _ag_ActionCostResult,
    "ActionGapConfig": _ag_ActionGapConfig,
}

# 注入到 _sys 命名空间
for _name, _cls in _V360_INJECTIONS.items():
    setattr(_sys_module, _name, _cls)

# 重新绑定 sdk 层引用
for _name, _cls in _V360_INJECTIONS.items():
    setattr(_sdk_self, _name, _cls)

# ═══════════════════════════════════════════════════════════════════════════════
# v3.7.0: _sys 延迟注入 (Late-binding injection) — 认知诊断 + 负面启发法
# ═══════════════════════════════════════════════════════════════════════════════
from mci_world_model.sdk._meta_diagnoser import (
    DiagnosisResult as _md_DiagnosisResult,
)
from mci_world_model.sdk._meta_diagnoser import (
    FailurePattern as _md_FailurePattern,
)
from mci_world_model.sdk._meta_diagnoser import (
    MetaDiagnoser as _md_MetaDiagnoser,
)
from mci_world_model.sdk._meta_diagnoser import (
    MetaDiagnoserStats as _md_MetaDiagnoserStats,
)
from mci_world_model.sdk._meta_diagnoser import (
    PatternMatch as _md_PatternMatch,
)
from mci_world_model.sdk._meta_diagnoser import (
    RootCauseChain as _md_RootCauseChain,
)
from mci_world_model.sdk._meta_diagnoser import (
    SeverityLevel as _md_SeverityLevel,
)
from mci_world_model.sdk._meta_diagnoser import (
    SurpriseSignal as _md_SurpriseSignal,
)
from mci_world_model.sdk._negative_heuristic import (
    ChangeType as _nh_ChangeType,
)
from mci_world_model.sdk._negative_heuristic import (
    HardCoreViolation as _nh_HardCoreViolation,
)
from mci_world_model.sdk._negative_heuristic import (
    NegativeHeuristic as _nh_NegativeHeuristic,
)
from mci_world_model.sdk._negative_heuristic import (
    NegativeHeuristicStats as _nh_NegativeHeuristicStats,
)
from mci_world_model.sdk._negative_heuristic import (
    ProposedChange as _nh_ProposedChange,
)
from mci_world_model.sdk._negative_heuristic import (
    ProtectiveBeltSuggestion as _nh_ProtectiveBeltSuggestion,
)
from mci_world_model.sdk._negative_heuristic import (
    RuleSeverity as _nh_RuleSeverity,
)

_V370_INJECTIONS = {
    "MetaDiagnoser": _md_MetaDiagnoser,
    "SurpriseSignal": _md_SurpriseSignal,
    "FailurePattern": _md_FailurePattern,
    "SeverityLevel": _md_SeverityLevel,
    "PatternMatch": _md_PatternMatch,
    "RootCauseChain": _md_RootCauseChain,
    "DiagnosisResult": _md_DiagnosisResult,
    "MetaDiagnoserStats": _md_MetaDiagnoserStats,
    "NegativeHeuristic": _nh_NegativeHeuristic,
    "ProposedChange": _nh_ProposedChange,
    "ChangeType": _nh_ChangeType,
    "HardCoreViolation": _nh_HardCoreViolation,
    "ProtectiveBeltSuggestion": _nh_ProtectiveBeltSuggestion,
    "NegativeHeuristicStats": _nh_NegativeHeuristicStats,
    "RuleSeverity": _nh_RuleSeverity,
}

for _name, _cls in _V370_INJECTIONS.items():
    setattr(_sys_module, _name, _cls)

for _name, _cls in _V370_INJECTIONS.items():
    setattr(_sdk_self, _name, _cls)
