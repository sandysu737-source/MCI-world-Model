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
| MCI World Model（世界模型） | V3.0.0（更名前 V4.0.0） |

🔗 上下游
==========

- 上游依赖：su-memory-sdk ≥ 3.5.1（记忆引擎）
- 下游产品：MCI·焕 (mci-huan) / mci-kernel-integrations
"""

__version__ = "3.0.0"
__project__ = "MCI World Model"
__brand__ = "MCI"

# 顶层统一入口：从 _sys 与 sdk 暴露关键能力
from mci_world_model._sys import (
    # 能量中心（v3.5.0 三才合一）
    YinYang,
    ThreePowers,
    FourSymbols,
    EnergyCore,
    EnergyBus,
    # 时空感知
    TemporalCore,
    StemBranchCode,
    TemporalSystem,
    # 64 卦推断
    TrigramCore,
    PatternInference,
    # 因果引擎
    CausalEngine,
    CategoryCausalEngine,
    # 贝叶斯系统
    BayesianNetwork,
    BayesianReasoner,
    # 元认知
    MetaCognition,
    CognitiveGap,
    KnowledgeAging,
)

from mci_world_model.sdk import (
    # V3.0.0 核心
    MCIWorldModel,
    CausalWorldModelState,
    # Pearl 三层
    DoCalculus,
    CausalGraph,
    CounterfactualEngine,
    BayesianCausal,
    FourierCausal,
    GaussianDAG,
    # JEPA 全套
    JEPAEncoder,
    GNNPredictor,
    JEPATrainer,
    JEPADataset,
    JEPAGATEncoder,
    # SIGReg
    SIGReg,
    apply_sigreg_to_index,
    # 贝叶斯增强
    BayesianAugmenter,
    # 能量一致性
    EnergyConsistencyLoss,
    TopologicalEnergyMatrix,
    # 参数化记忆
    ParametricMemory,
    ParametricMemoryConfig,
    # Reflection QA
    ReflectionSynthesizer,
    SynthesizedQAPair,
)

__all__ = [
    "__version__",
    "__project__",
    "__brand__",
    # 上面显式导出的所有符号
]
