"""
mci_world_model._sys — System Layer

底层系统能力：能量中心、贝叶斯系统、时空感知、因果引擎、模式推断。
"""

# 基础类型层
from mci_world_model._sys._terms import (
    BranchRelation,
    EnergyPattern,
    FourSymbols,
    Season,
    SemanticCategory,
    StrengthState,
    ThreePowers,
    TimeBranch,
    TimeStem,
    TrigramRelation,
    TrigramType,
    YinYang,
)
from mci_world_model._sys._unified_unit import (
    UnifiedInfoFactory,
    UnifiedInfoUnit,
)

# 天层 (Temporal/Sky)
from mci_world_model._sys._temporal_core import (
    StemBranchCode,
    TemporalCore,
    TemporalSystem,
    TimeCycle,
    TimeCodeInfo,
)

# 地层 (Spatial/Earth)
from mci_world_model._sys._dimension_map import (
    PatternInference,
    TaijiMapper,
    TrigramCore,
)

# 人层 (Energy/Human)
from mci_world_model._sys._energy_core import EnergyCore
from mci_world_model._sys._energy_bus import EnergyBus
from mci_world_model._sys._energy_relations import analyze_balance
from mci_world_model._sys._causal_engine import (
    CategoryCausalEngine,
    CausalEngine,
)

# 贝叶斯系统
from mci_world_model._sys.bayesian import BayesianInference
from mci_world_model._sys.bayesian_network import BayesianNetwork
from mci_world_model._sys.bayesian_reasoning import BayesianReasoner

# 因果图与工具
from mci_world_model._sys.causal import CausalChain, CausalInference

# 时空与时序
from mci_world_model._sys.chrono import ChronoAxis, TimePoint
from mci_world_model._sys.awareness import AwarenessSignal, SelfAwareness

# 元认知
from mci_world_model._sys.meta_cognition import (
    CognitiveGap,
    KnowledgeAging,
    MetaCognition,
)

__all__ = [
    # 基础类型
    "YinYang",
    "ThreePowers",
    "FourSymbols",
    "Season",
    "TimeStem",
    "TimeBranch",
    "BranchRelation",
    "TrigramType",
    "TrigramRelation",
    "StrengthState",
    "EnergyPattern",
    "SemanticCategory",
    # 统一信息单元
    "UnifiedInfoUnit",
    "UnifiedInfoFactory",
    # 天层
    "TemporalCore",
    "StemBranchCode",
    "TemporalSystem",
    "TimeCycle",
    "TimeCodeInfo",
    # 地层
    "TrigramCore",
    "TaijiMapper",
    "PatternInference",
    # 人层 / 能量中心
    "EnergyCore",
    "EnergyBus",
    "analyze_balance",
    "CategoryCausalEngine",
    "CausalEngine",
    # 贝叶斯
    "BayesianInference",
    "BayesianNetwork",
    "BayesianReasoner",
    # 因果
    "CausalChain",
    "CausalInference",
    # 时空
    "ChronoAxis",
    "TimePoint",
    "AwarenessSignal",
    "SelfAwareness",
    # 元认知
    "MetaCognition",
    "CognitiveGap",
    "KnowledgeAging",
]
