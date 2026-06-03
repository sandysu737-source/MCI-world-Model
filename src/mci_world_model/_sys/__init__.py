"""
mci_world_model._sys — System Layer

底层系统能力：能量中心、贝叶斯系统、时空感知、因果引擎、模式推断。
"""

# 基础类型层 — 枚举类（来自 _enums）
# 基础类型层 — 语义分类（来自 _c1）
from mci_world_model._sys._c1 import SemanticCategory
from mci_world_model._sys._enums import (
    BranchRelation,
    EnergyPattern,
    EnergyRelation,
    EnergyType,
    FourSymbols,
    Season,
    StrengthState,
    ThreePowers,
    TimeBranch,
    TimeStem,
    TrigramRelation,
    TrigramType,
    YinYang,
)

# 基础类型层 — 八卦推断（来自 _pattern_inference）
from mci_world_model._sys._pattern_inference import TrigramContext

# 贝叶斯系统（world_model.py + sdk 端 re-export 需要）
from mci_world_model._sys.bayesian_network import BayesianNetwork
from mci_world_model._sys.bayesian_reasoning import BayesianReasoningSystem

# 兼容别名：world_model.py 期望 EnergyEnumType/EnergyEnumRelation
EnergyEnumType = EnergyType
EnergyEnumRelation = EnergyRelation

# 基础类型层 — 数据字典（来自 _terms）
# 基础类型层 — 语义/能量分类（_c1 + _c2 扩展）
from mci_world_model._sys._c1 import (
    CATEGORY_ANCHORS,
    ENERGY_TO_CATEGORY,
    KEYWORDS_TO_CATEGORY,
    MEMORY_TYPE_TO_CATEGORY,
)
from mci_world_model._sys._c2 import (
    ENERGY_ENHANCE_MAP,
    ENERGY_SUPPRESS_MAP,
    STATE_STRENGTH_MAP,
    EnergyNetwork,
    check_state_interaction,
    energy_from_category,
    energy_similarity,
)
from mci_world_model._sys._causal_engine import EnergyMemoryNode

# 人层 — 能量网络与核心
from mci_world_model._sys._energy_bus import (
    EnergyChannel,
    EnergyLayer,
    EnergyNode,
    EnergySignal,
    PropagationConfig,
    create_complete_energy_network,
    create_energy_bus,
)
from mci_world_model._sys._energy_core import (
    EnergyBalanceResult,
    EnergyFlow,
    EnergyState,
)

# 能量关系
from mci_world_model._sys._energy_relations import (
    FOUR_SYMBOLS_TO_ENERGY,
    RELATION_STRENGTH,
    MemoryNodeEnergy,
    RelationType,
    analyze_relation,
    calculate_link_weight,
    find_reverse_causal_chain,
    get_affinity_score,
    get_cycle_sequence,
    get_enhance_relation,
    get_enhanced_energy,
    get_enhancing_energy,
    get_suppress_chain,
    get_suppress_relation,
    get_suppressed_energy,
    get_suppressing_energy,
    is_enhancing,
    is_suppressing,
    surface_entities,
)
from mci_world_model._sys._terms import (
    BRANCH_CHONG_MAP,
    BRANCH_HE_MAP,
    BRANCH_HIDDEN_STEM_MAP,
    BRANCH_SANHE_MAP,
    MONTH_ENERGY_STATE,
    SEMANTIC_CATEGORY,
    SEMANTIC_CATEGORY_NAMES,
    STEM_CHONG_MAP,
    STEM_HE_MAP,
    STRENGTH_MULTIPLIER,
    TIME_BRANCH_ENERGY,
    TIME_BRANCHES,
    TIME_STEMS,
    TRIGRAM_BODY_MAP,
    TRIGRAM_ENERGY_MAP,
)

# 天层 — 时空关系数据 + 工具
from mci_world_model._sys._time_code import (
    BRANCH_CHONG,
    BRANCH_HE,
    BRANCH_SANHE,
    BRANCH_XING,
    STEM_CHONG,
    STEM_HE,
    get_branch,
    get_cycle,
    get_stem,
)

# 三才合一 + 检索融合
from mci_world_model._sys._unified_unit import create_unified_unit

# 兼容占位：world_model.py 中引用但本版本不提供的过时符号
get_seasonal_energy_state = None  # type: ignore[assignment]
EnergyRelationsType = None  # type: ignore[assignment]

# MultiViewRetriever 占位：5 维融合检索器在未来版本提供，当前为 stub
MultiViewRetriever = None  # type: ignore[assignment]

# 别名：保持 world_model.py 的历史接口名仍可导入
EnergyStateInfo = EnergyState  # type: ignore[assignment]

# 地层 (Spatial/Earth)
from mci_world_model._sys._category_core import TrigramCore
from mci_world_model._sys._causal_engine import CategoryCausalEngine

# v3.0.1: Configurator
from mci_world_model._sys._configurator import ConfigAction, HierarchicalConfigurator, MetaConfigurator
from mci_world_model._sys._dimension_map import TaijiMapper
from mci_world_model._sys._energy_bus import EnergyBus

# 人层 (Energy/Human)
from mci_world_model._sys._energy_core import EnergyCore
from mci_world_model._sys._energy_relations import analyze_balance
from mci_world_model._sys._pattern_inference import PatternInference

# v3.0.3: Perception Pipeline
from mci_world_model._sys._perception_pipeline import PerceptionPipeline

# 天层 (Temporal/Sky) — 六十花甲编码核心
from mci_world_model._sys._temporal_core import (
    StemBranchCode,
    TemporalCore,
    create_stem_branch,
    get_cycle_name,
)

# 天层 (Temporal/Sky) — 独立时空量化系统
from mci_world_model._sys._time_code import (
    TimeCodeInfo,
    TimeCycle,
    create_time_code,
)
from mci_world_model._sys._unified_unit import (
    UnifiedInfoFactory,
    UnifiedInfoUnit,
)

# 元认知
from mci_world_model._sys.awareness import CognitiveGap, KnowledgeAging

# 贝叶斯系统
from mci_world_model._sys.bayesian import BayesianEngine

# 因果图与工具
from mci_world_model._sys.causal import CausalChain, CausalInference

# 时空与时序
from mci_world_model._sys.chrono import (
    DiZhi,
    DynamicPriority,
    TemporalInfo,
    TemporalSystem,
    TianGan,
)
from mci_world_model._sys.meta_cognition import MetaCognition

__all__ = [
    # 贝叶斯
    "BayesianEngine",
    "BayesianNetwork",
    "BayesianReasoningSystem",
    # 地支
    "BRANCH_CHONG",
    "BRANCH_CHONG_MAP",
    "BRANCH_HE",
    "BRANCH_HE_MAP",
    "BRANCH_HIDDEN_STEM_MAP",
    "BRANCH_SANHE",
    "BRANCH_SANHE_MAP",
    "BRANCH_XING",
    "BranchRelation",
    # 分类/能量映射
    "CATEGORY_ANCHORS",
    "CategoryCausalEngine",
    # 因果
    "CausalChain",
    "CausalInference",
    "CognitiveGap",
    # v3.0.1: Configurator
    "ConfigAction",
    "DiZhi",
    "DynamicPriority",
    # 能量总线
    "ENERGY_ENHANCE_MAP",
    "ENERGY_SUPPRESS_MAP",
    "ENERGY_TO_CATEGORY",
    "EnergyBalanceResult",
    "EnergyBus",
    "EnergyChannel",
    # 人层 / 能量中心
    "EnergyCore",
    "EnergyEnumRelation",
    "EnergyEnumType",
    "EnergyFlow",
    "EnergyLayer",
    "EnergyMemoryNode",
    "EnergyNetwork",
    "EnergyNode",
    "EnergyPattern",
    "EnergyRelation",
    "EnergySignal",
    "EnergyState",
    "EnergyType",
    # 四象
    "FOUR_SYMBOLS_TO_ENERGY",
    "FourSymbols",
    # 关键词映射
    "KEYWORDS_TO_CATEGORY",
    "KnowledgeAging",
    # 记忆类型
    "MEMORY_TYPE_TO_CATEGORY",
    "MONTH_ENERGY_STATE",
    "MemoryNodeEnergy",
    # 元认知
    "MetaConfigurator",
    "MetaCognition",
    "PatternInference",
    # v3.0.3: Perception
    "PerceptionPipeline",
    "PropagationConfig",
    # 关系强度
    "RELATION_STRENGTH",
    "RelationType",
    # 语义分类
    "SEMANTIC_CATEGORY",
    "SEMANTIC_CATEGORY_NAMES",
    "STATE_STRENGTH_MAP",
    "STEM_CHONG",
    "STEM_CHONG_MAP",
    "STEM_HE",
    "STEM_HE_MAP",
    "STRENGTH_MULTIPLIER",
    "Season",
    "SemanticCategory",
    "StemBranchCode",
    "StrengthState",
    # 八卦
    "TIME_BRANCH_ENERGY",
    "TIME_BRANCHES",
    "TIME_STEMS",
    "TRIGRAM_BODY_MAP",
    "TRIGRAM_ENERGY_MAP",
    "TaijiMapper",
    # 天层
    "TemporalCore",
    # v3.0.3: H-JEPA
    "HierarchicalConfigurator",
    "TemporalInfo",
    "TemporalSystem",
    "ThreePowers",
    # 时空
    "TianGan",
    "TimeBranch",
    "TimeCodeInfo",
    "TimeCycle",
    "TimeStem",
    # 地层
    "TrigramContext",
    "TrigramCore",
    "TrigramRelation",
    "TrigramType",
    # 统一信息单元
    "UnifiedInfoFactory",
    "UnifiedInfoUnit",
    # 基础类型
    "YinYang",
    # 函数
    "analyze_balance",
    "analyze_relation",
    "calculate_link_weight",
    "check_state_interaction",
    "create_complete_energy_network",
    "create_energy_bus",
    "create_stem_branch",
    "create_time_code",
    "create_unified_unit",
    "energy_from_category",
    "energy_similarity",
    "find_reverse_causal_chain",
    "get_affinity_score",
    "get_branch",
    "get_cycle",
    "get_cycle_name",
    "get_cycle_sequence",
    "get_enhance_relation",
    "get_enhanced_energy",
    "get_enhancing_energy",
    "get_stem",
    "get_suppress_chain",
    "get_suppress_relation",
    "get_suppressed_energy",
    "get_suppressing_energy",
    "is_enhancing",
    "is_suppressing",
    "surface_entities",
]
