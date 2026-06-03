"""
mci_world_model.sdk — World Model Core

Pearl 因果三层 + JEPA 世界建模 + 能量一致性
"""

# 贝叶斯系统（从 _sys re-export 到 sdk）
from mci_world_model._sys import (
    BayesianNetwork,
    BayesianReasoningSystem,
)

# 因果引擎基础
from mci_world_model.sdk._causal import CausalEngine

# v3.0.0: 反事实推理（P3）
from mci_world_model.sdk._counterfactual import (
    CounterfactualEngine,
    CounterfactualResult,
)

# v3.0.0: Pearl Do-Calculus 干预层（V2.0 起点）
from mci_world_model.sdk._do_calculus import (
    CausalGraph,
    DoCalculus,
)

# 能量一致性损失
from mci_world_model.sdk._energy_loss import (
    EnergyConsistencyLoss,
    TopologicalEnergyMatrix,
)

# JEPA 全套
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
    # Bayesian Augmenter
    "BayesianAugmenter",
    # Spectral Causal
    "BayesianCausal",
    # Bayesian (re-export from _sys)
    "BayesianNetwork",
    "BayesianReasoningSystem",
    "BeliefPropagationPredictor",
    # Causal Engine
    "CausalEngine",
    "CausalGraph",
    "CausalWorldModelState",
    "ComparisonDelta",
    # Counterfactual
    "CounterfactualEngine",
    "CounterfactualResult",
    # Pearl Do-Calculus
    "DoCalculus",
    # Energy Loss
    "EnergyConsistencyLoss",
    "EnergyPropagationPredictor",
    "EnhancedOutput",
    "FourierCausal",
    "GNNPredictor",
    "GaussianDAG",
    "GaussianDistribution",
    "IdentityPredictor",
    "JEPADataset",
    # JEPA
    "JEPAEncoder",
    "JEPAPredictor",
    "JEPATrainer",
    # V3.0.0 核心
    "MCIWorldModel",
    # Parametric Memory
    "ParametricMemory",
    "ParametricMemoryConfig",
    # Reflection
    "ReflectionSynthesizer",
    # SIGReg
    "SIGReg",
    "SynthesizedQAPair",
    "TopologicalEnergyMatrix",
    "apply_sigreg_to_index",
]
