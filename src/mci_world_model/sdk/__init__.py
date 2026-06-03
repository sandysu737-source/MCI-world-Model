"""
mci_world_model.sdk — World Model Core

Pearl 因果三层 + JEPA 世界建模 + 能量一致性
"""

from mci_world_model.sdk._world_model import (
    CausalWorldModelState,
    MCIWorldModel,
)

# v3.0.0: Pearl Do-Calculus 干预层（V2.0 起点）
from mci_world_model.sdk._do_calculus import (
    CausalGraph,
    DoCalculus,
)

# v3.0.0: 反事实推理（P3）
from mci_world_model.sdk._counterfactual import (
    CounterfactualEngine,
    CounterfactualResult,
)

# 频谱因果引擎（P1 关联层）
from mci_world_model.sdk._spectral_causal import (
    BayesianCausal,
    FourierCausal,
    GaussianDAG,
    GaussianDistribution,
)

# 因果引擎基础
from mci_world_model.sdk._causal import CausalEngine

# JEPA 全套
from mci_world_model.sdk._jepa_dataset import JEPADataset
from mci_world_model.sdk._jepa_encoder import JEPAEncoder
from mci_world_model.sdk._jepa_gat_encoder import JEPAGATEncoder
from mci_world_model.sdk._jepa_gnn import GNNPredictor
from mci_world_model.sdk._jepa_predictor import JEPAPredictor
from mci_world_model.sdk._jepa_trainer import JEPATrainer

# SIGReg 嵌入正则化
from mci_world_model.sdk._sigreg import SIGReg, apply_sigreg_to_index

# 贝叶斯增强器
from mci_world_model.sdk.bayesian_augmenter import (
    AccuracyRecord,
    BayesianAugmenter,
    ComparisonDelta,
    EnhancedOutput,
)

# 能量一致性损失
from mci_world_model.sdk._energy_loss import (
    EnergyConsistencyLoss,
    TopologicalEnergyMatrix,
)

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

__all__ = [
    # V3.0.0 核心
    "MCIWorldModel",
    "CausalWorldModelState",
    # Pearl Do-Calculus
    "DoCalculus",
    "CausalGraph",
    # Counterfactual
    "CounterfactualEngine",
    "CounterfactualResult",
    # Spectral Causal
    "BayesianCausal",
    "FourierCausal",
    "GaussianDAG",
    "GaussianDistribution",
    # Causal Engine
    "CausalEngine",
    # JEPA
    "JEPAEncoder",
    "JEPAGATEncoder",
    "GNNPredictor",
    "JEPAPredictor",
    "JEPATrainer",
    "JEPADataset",
    # SIGReg
    "SIGReg",
    "apply_sigreg_to_index",
    # Bayesian Augmenter
    "BayesianAugmenter",
    "AccuracyRecord",
    "ComparisonDelta",
    "EnhancedOutput",
    # Energy Loss
    "EnergyConsistencyLoss",
    "TopologicalEnergyMatrix",
    # Parametric Memory
    "ParametricMemory",
    "ParametricMemoryConfig",
    # Reflection
    "ReflectionSynthesizer",
    "SynthesizedQAPair",
]
