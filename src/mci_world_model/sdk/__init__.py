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

# 贝叶斯系统（从 _sys re-export 到 sdk）
# v3.0.1: Configurator (re-export from _sys)
# v3.0.3: Perception Pipeline (re-export from _sys)
from mci_world_model._sys import (
    # v3.2.0: 感知-行动接口范式
    ActionCommand,
    ActionPriority,
    ActionResult,
    ActuatorChannel,
    BayesianNetwork,
    BayesianReasoningSystem,
    ConfigAction,
    HierarchicalConfigurator,
    MetaConfigurator,
    # v3.1.0: 多模态信号
    MultimodalSignal,
    PerceptionPipeline,
    PhysicalSignal,
    SensorModality,
    SignalSubType,
    # v3.1.0: 信号类型枚举
    SignalType,
)

# v3.2.0: 动作条件化预测器
from mci_world_model.sdk._action_conditioned_predictor import (
    ActionConditionedPredictor,
    PendulumJEPAPredictor,
    PendulumPhysicsPredictor,
)

# v3.0.8: 批量反事实引擎
from mci_world_model.sdk._batch_counterfactual import BatchCounterfactualEngine

# 因果引擎基础
from mci_world_model.sdk._causal import CausalEngine

# v3.0.2: Causal Actor
from mci_world_model.sdk._causal_actor import ActionCandidate, CausalActor, EnergyGuidedAction

# v3.0.7: CausalMLP 因果推断网络
from mci_world_model.sdk._causal_mlp import CausalMLP

# v3.0.1: Cost 模块独立
from mci_world_model.sdk._cost_module import CostSignal, EnergyCostModule

# v3.0.8: 反事实推理（P3 增强）
from mci_world_model.sdk._counterfactual import (
    CounterfactualEngine,
    CounterfactualResult,
    StructuralEquationModel,
)

# v3.0.0: Pearl Do-Calculus 干预层（V2.0 起点）
from mci_world_model.sdk._do_calculus import (
    CausalGraph,
    DoCalculus,
)

# v3.0.6: 能量流预测器
from mci_world_model.sdk._energy_flow_predictor import EnergyFlowPredictor

# 能量一致性损失
from mci_world_model.sdk._energy_loss import (
    EnergyConsistencyLoss,
    TopologicalEnergyMatrix,
)

# JEPA 全套
from mci_world_model.sdk._hierarchical_encoder import HierarchicalJEPAEncoder, HierarchicalState
from mci_world_model.sdk._jepa_dataset import JEPADataset
from mci_world_model.sdk._jepa_encoder import JEPAEncoder
from mci_world_model.sdk._jepa_gnn import GNNPredictor
from mci_world_model.sdk._jepa_predictor import (
    BeliefPropagationPredictor,
    EnergyPropagationPredictor,
    IdentityPredictor,
    JEPAPredictor,
)
from mci_world_model.sdk._jepa_trainer import JEPATrainer

# 参数化记忆
from mci_world_model.sdk._parametric_memory import (
    ParametricMemory,
    ParametricMemoryConfig,
)

# v3.1.0: PhysicalGraphBuilder 物理量→因果边转换器
from mci_world_model.sdk._physical_graph_builder import (
    PhysicalGraphBuilder,
    signals_to_timeline,
)

# Reflection QA 合成器
from mci_world_model.sdk._reflection_synthesizer import (
    ReflectionSynthesizer,
    SynthesizedQAPair,
)

# SIGReg 嵌入正则化
from mci_world_model.sdk._sigreg import SIGReg, apply_sigreg_to_index

# 频谱因果引擎（P1 关联层）
from mci_world_model.sdk._spectral_causal import (
    BayesianCausal,
    FourierCausal,
    GaussianDAG,
    GaussianDistribution,
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
    MultimodalWorldState,
    PendulumAction,
    PendulumState,
    WorldState,
)

# v3.2.0: Phase 1 P0 — 多分支推演/惊奇检测/PlanAgent
from mci_world_model.sdk._multi_branch_predictor import (
    BranchEvaluation,
    MultiBranchPredictor,
)
from mci_world_model.sdk._surprise_detector import (
    SurpriseDetector,
    SurpriseSignal,
)
from mci_world_model.sdk._plan_agent import (
    Plan,
    PlanAgent,
)

# v3.3.0: 多模态因果世界模型
from mci_world_model.sdk._modality_encoders import (
    AudioEncoder,
    ThermalEncoder,
    VisionEncoder,
)
from mci_world_model.sdk._multimodal_fusion import (
    FusedRepresentation,
    MultimodalFusion,
)
from mci_world_model.sdk._multimodal_graph_builder import (
    MultimodalGraphBuilder,
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
    # v3.0.8: BatchCounterfactualEngine
    "BatchCounterfactualEngine",
    "BeliefPropagationPredictor",
    # Causal Engine
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
    # v3.2.0: 独立世界状态
    "Action",
    "ActionConditionedPredictor",
    # v3.3.0: 模态编码器
    "AudioEncoder",
    "MultimodalWorldState",
    "PendulumAction",
    "PendulumJEPAPredictor",
    "PendulumPhysicsPredictor",
    "PendulumState",
    # v3.2.0: Phase 1 P0
    "MultiBranchPredictor",
    "BranchEvaluation",
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
    # V3.0.0 核心
    "MCIWorldModel",
    # v3.1.0: 多模态信号
    "MultimodalSignal",
    # v3.0.1: Configurator
    "MetaConfigurator",
    # Parametric Memory
    "ParametricMemory",
    "ParametricMemoryConfig",
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
]
