"""
MCI World Model — 独立世界模型引擎

从 su-memory-sdk（记忆引擎 V3.5.1）分离而来的因果世界模型项目。
本项目聚焦 Pearl 三层因果推理 + JEPA 世界建模 + 能量中心。

🎯 核心定位
============

MCI World Model 是强劲的「CPU」:
- 独立运行的通用因果推理引擎
- 不依赖 Transformer 或 GPU
- Transformer 仅作为可选的「GPU」（文本生成加速器）插在 CPU 后端，可随时更换或移除

🧬 三大支柱
============

1. **Pearl Causal Hierarchy** — 关联 / 干预 / 反事实
   - P1 (Association): 频谱因果引擎 + 贝叶斯增强器
   - P2 (Intervention): Do-Calculus 干预层（MCI World Model V2.0 起点）
   - P3 (Counterfactual): 反事实推理引擎

2. **JEPA World Modeling** — 世界建模
   - JEPA Encoder（CNN + GAT + GNN）
   - JEPA Predictor（可微 GNN 预测器）
   - SIGReg 嵌入正则化
   - Reflection QA 合成器

3. **Energy Center** — 能量中心三才合一
   - 天层 (Temporal/Sky): 时空感知与六十甲子
   - 地层 (Spatial/Earth): 64 卦三层推断
   - 人层 (Energy/Human): 五行能量 + 加权因果

📦 版本映射
============

| 项目 | 版本号 |
|------|--------|
| su-memory-sdk（记忆引擎） | V3.5.1 |
| MCI World Model（世界模型） | V4.3.3（CEWM 认知增强世界模型 — 参数化记忆觉醒+能量流闭环） |

🔗 上下游
==========

- 上游依赖：su-memory-sdk ≥ 3.5.1（记忆引擎）
- 下游产品：MCI·焕 (mci-huan) / mci-kernel-integrations
"""

__version__ = "4.6.0"
__project__ = "MCI World Model"
__brand__ = "MCI"

# 顶层统一入口：从 _sys 与 sdk 暴露关键能力
from mci_world_model._sys import (
    # 贝叶斯系统
    BayesianNetwork,
    BayesianReasoningSystem,
    # 因果引擎
    CategoryCausalEngine,
    CognitiveGap,
    # v3.0.1/v3.0.3: Configurator 分层决策
    ConfigAction,
    EnergyBus,
    EnergyCore,
    FourSymbols,
    HierarchicalConfigurator,
    KnowledgeAging,
    # 元认知
    MetaCognition,
    MetaConfigurator,
    PatternInference,
    # v3.0.1/v3.0.3: Perception 前端信号处理
    PerceptionPipeline,
    StemBranchCode,
    # 时空感知
    TemporalCore,
    TemporalSystem,
    ThreePowers,
    # 64 卦推断
    TrigramCore,
    # 能量中心（v3.5.0 三才合一）
    YinYang,
)
from mci_world_model.sdk import (
    # v3.6.0: 因果图自适应 + 行动距离
    ActionConditionedPredictor,
    ActionGapMetric,
    # v3.0.8: BatchCounterfactualEngine
    BatchCounterfactualEngine,
    # 贝叶斯增强
    BayesianAugmenter,
    BayesianCausal,
    CachedDiscoverer,
    CachedDoCalculus,
    # v4.6.0: CausalDataFrame (user-friendly API)
    CausalDataFrame,
    CausalEngine,
    CausalGraph,
    CausalGraphResult,
    # v3.0.7: CausalMLP
    CausalMLP,
    CausalUpdater,
    CausalWorldModelState,
    # v3.4.0: 闭环基础设施
    CognitiveDiversity,
    CognitiveLoopBus,
    CounterfactualEngine,
    # Pearl 三层
    DoCalculus,
    # 能量一致性
    EnergyConsistencyLoss,
    # v3.5.0: 经验记忆系统
    Experience,
    ExperienceDB,
    FourierCausal,
    GaussianDAG,
    GNNPredictor,
    IdentityPredictor,
    JEPADataset,
    # JEPA 全套
    JEPAEncoder,
    JEPAPredictor,
    JEPATrainer,
    # V3.0.0 核心
    MCIWorldModel,
    # v3.7.0: 认知诊断系统
    MetaDiagnoser,
    MultiBranchPredictor,
    # v3.5.0: 五维融合检索器
    MultiViewRetriever,
    NegativeHeuristic,
    # 参数化记忆
    ParametricMemory,
    ParametricMemoryConfig,
    # v3.1.0: 物理量→因果边转换器
    PhysicalGraphBuilder,
    # v4.3.2: PlanAgent
    PlanAgent,
    # v3.5.0: 查询规格
    QuerySpec,
    # Reflection QA
    ReflectionSynthesizer,
    # SIGReg
    SIGReg,
    SLearner,
    # v3.0.8: StructuralEquationModel
    StructuralEquationModel,
    SynthesizedQAPair,
    TLearner,
    TopologicalEnergyMatrix,
    apply_sigreg_to_index,
)

__all__ = [
    # sdk — 贝叶斯增强
    "BayesianAugmenter",
    "BayesianCausal",
    # _sys — 贝叶斯系统
    "BayesianNetwork",
    "BayesianReasoningSystem",
    # _sys — 因果引擎
    "CategoryCausalEngine",
    "CausalEngine",
    "CausalGraph",
    # v3.0.8: BatchCounterfactualEngine
    "BatchCounterfactualEngine",
    "CausalWorldModelState",
    # v3.4.0: 闭环基础设施
    "CognitiveDiversity",
    "CognitiveLoopBus",
    "CognitiveGap",
    # v3.0.1/v3.0.3: Configurator
    "ConfigAction",
    "CounterfactualEngine",
    # sdk — Pearl 三层
    "DoCalculus",
    "EnergyBus",
    # sdk — 能量一致性
    "EnergyConsistencyLoss",
    "EnergyCore",
    # v3.5.0: 经验记忆系统
    "Experience",
    "ExperienceDB",
    "FourSymbols",
    "FourierCausal",
    "GNNPredictor",
    "GaussianDAG",
    # v3.0.3: Configurator 分层
    "HierarchicalConfigurator",
    "IdentityPredictor",
    "JEPADataset",
    # sdk — JEPA 全套
    "JEPAEncoder",
    "JEPAPredictor",
    "JEPATrainer",
    "KnowledgeAging",
    # sdk — V3.0.0 核心
    "MCIWorldModel",
    "MultiBranchPredictor",
    # v3.5.0: 五维融合检索器
    "MultiViewRetriever",
    # _sys — 元认知
    "MetaCognition",
    "MetaConfigurator",
    # v3.6.0: 认知环闭环
    "ActionConditionedPredictor",
    "ActionGapMetric",
    "CausalUpdater",
    # v3.7.0: 认知诊断系统
    "MetaDiagnoser",
    "NegativeHeuristic",
    # v4.3.2: PlanAgent
    "PlanAgent",
    # v3.0.7: CausalMLP
    "CausalMLP",
    # sdk — 参数化记忆
    "ParametricMemory",
    "ParametricMemoryConfig",
    "PatternInference",
    # v3.0.3: Perception
    "PerceptionPipeline",
    # v3.1.0: PhysicalGraphBuilder
    "PhysicalGraphBuilder",
    # v3.5.0: 查询规格
    "QuerySpec",
    # sdk — Reflection QA
    "ReflectionSynthesizer",
    # sdk — SIGReg
    "SIGReg",
    "StemBranchCode",
    "SynthesizedQAPair",
    # v3.0.8: StructuralEquationModel
    "StructuralEquationModel",
    # _sys — 时空感知
    "TemporalCore",
    "TemporalSystem",
    "ThreePowers",
    "TopologicalEnergyMatrix",
    # _sys — 64 卦推断
    "TrigramCore",
    # _sys — 能量中心
    "YinYang",
    "__brand__",
    "__project__",
    "__version__",
    "apply_sigreg_to_index",
    # v4.6.0: CausalDataFrame
    "CausalDataFrame",
    "CausalGraphResult",
    "CachedDiscoverer",
    "CachedDoCalculus",
    "TLearner",
    "SLearner",
]
