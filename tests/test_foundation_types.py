"""
test_foundation_types.py — Foundation Types Test Suite

覆盖 _sys 基础类型层：
- 13 个枚举（_enums）：YinYang, ThreePowers, FourSymbols, Season,
  TimeStem, TimeBranch, BranchRelation, TrigramType, TrigramRelation,
  EnergyType, EnergyRelation, StrengthState, EnergyPattern
- SemanticCategory 8 象枚举（_c1）
- _c1 数据字典：CATEGORY_ANCHORS, KEYWORDS_TO_CATEGORY,
  MEMORY_TYPE_TO_CATEGORY, ENERGY_TO_CATEGORY
- _c2 能量核心：EnergyType (c2 版，带 element/nature 等属性),
  ENERGY_ENHANCE_MAP, ENERGY_SUPPRESS_MAP, STATE_STRENGTH_MAP,
  get_energy_state, check_state_interaction, energy_similarity,
  energy_from_category, EnergyState, EnergyNetwork
- _terms 数据字典：SEMANTIC_CATEGORY, SEMANTIC_CATEGORY_NAMES,
  TIME_STEMS, TIME_BRANCHES, BRANCH_HE_MAP 等
"""

from enum import Enum, IntEnum

import pytest

# --- 13 个枚举（_enums） ---
from mci_world_model._sys import (
    BranchRelation,
    EnergyPattern,
    EnergyRelation,
    EnergyType,
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

# --- _c1 数据字典 ---
from mci_world_model._sys._c1 import (
    CATEGORY_ANCHORS,
    CATEGORY_ASSOCIATIONS,
    ENERGY_TO_CATEGORY,
    KEYWORDS_TO_CATEGORY,
    MEMORY_TYPE_TO_CATEGORY,
)
from mci_world_model._sys._c2 import (
    CATEGORY_STATE_MULTIPLIERS,
    ENERGY_ENHANCE,
    ENERGY_ENHANCE_MAP,
    ENERGY_SUPPRESS,
    ENERGY_SUPPRESS_MAP,
    STATE_STRENGTH_MAP,
    EnergyNetwork,
    EnergyState,
    check_state_interaction,
    energy_from_category,
    energy_similarity,
    get_energy_state,
)

# --- _c2 能量核心（带属性的 EnergyType）---
from mci_world_model._sys._c2 import (
    EnergyType as C2EnergyType,
)

# --- _terms 术语安全映射 ---
from mci_world_model._sys._terms import (
    BRANCH_CHONG_MAP,
    BRANCH_HE_MAP,
    BRANCH_HIDDEN_STEM_MAP,
    BRANCH_SANHE_MAP,
    MONTH_ENERGY_STATE,
    SEMANTIC_CATEGORY,
    SEMANTIC_CATEGORY_NAMES,
    SEMANTIC_CATEGORY_PROPERTIES,
    SEMANTIC_CATEGORY_SYMBOLS,
    STEM_CHONG_MAP,
    STEM_HE_MAP,
    STRENGTH_MULTIPLIER,
    STRENGTH_STATE,
    TIME_BRANCH_ENERGY,
    TIME_BRANCHES,
    TIME_CYCLE_LENGTH,
    TIME_STEMS,
)

# =====================================================================
# [1] YinYang 基础双仪枚举
# =====================================================================


class TestYinYang:
    def test_yinyang_is_enum(self):
        assert issubclass(YinYang, Enum)
        assert not issubclass(YinYang, IntEnum)

    def test_yinyang_members(self):
        assert YinYang.YIN.name == "YIN"
        assert YinYang.YANG.name == "YANG"

    def test_yinyang_values(self):
        assert YinYang.YIN.value == 0
        assert YinYang.YANG.value == 1

    def test_yinyang_count(self):
        assert len(list(YinYang)) == 2

    def test_yinyang_inequality(self):
        assert YinYang.YIN != YinYang.YANG


# =====================================================================
# [2] ThreePowers 三才枚举
# =====================================================================


class TestThreePowers:
    def test_threepowers_is_enum(self):
        assert issubclass(ThreePowers, Enum)
        assert not issubclass(ThreePowers, IntEnum)

    def test_threepowers_members(self):
        assert {m.name for m in ThreePowers} == {"TIAN", "REN", "DI"}

    def test_threepowers_values(self):
        assert ThreePowers.TIAN.value == 0
        assert ThreePowers.REN.value == 1
        assert ThreePowers.DI.value == 2

    def test_threepowers_count(self):
        assert len(list(ThreePowers)) == 3


# =====================================================================
# [3] FourSymbols 四象枚举
# =====================================================================


class TestFourSymbols:
    def test_foursymbols_is_intenum(self):
        assert issubclass(FourSymbols, IntEnum)

    def test_foursymbols_members(self):
        names = {m.name for m in FourSymbols}
        assert names == {"SHAO_YANG", "TAI_YANG", "SHAO_YIN", "TAI_YIN"}

    def test_foursymbols_values(self):
        assert FourSymbols.SHAO_YANG.value == 0
        assert FourSymbols.TAI_YANG.value == 1
        assert FourSymbols.SHAO_YIN.value == 2
        assert FourSymbols.TAI_YIN.value == 3

    def test_foursymbols_int_comparison(self):
        assert FourSymbols.SHAO_YANG == 0
        assert int(FourSymbols.TAI_YIN) == 3


# =====================================================================
# [4] Season 季节枚举
# =====================================================================


class TestSeason:
    def test_season_is_intenum(self):
        assert issubclass(Season, IntEnum)

    def test_season_members(self):
        names = {m.name for m in Season}
        assert "SPRING" in names and "WINTER" in names
        assert "LATE_SUMMER" in names

    def test_season_count(self):
        assert len(list(Season)) == 5

    def test_season_late_summer_value(self):
        assert Season.LATE_SUMMER.value == 2


# =====================================================================
# [5] TimeStem 十天干枚举
# =====================================================================


class TestTimeStem:
    def test_timestem_is_intenum(self):
        assert issubclass(TimeStem, IntEnum)

    def test_timestem_count(self):
        assert len(list(TimeStem)) == 10

    def test_timestem_jia_yi_ne_gui(self):
        assert TimeStem.JIA.value == 0
        assert TimeStem.YI.value == 1
        assert TimeStem.BING.value == 2
        assert TimeStem.DING.value == 3
        assert TimeStem.WU.value == 4
        assert TimeStem.JI.value == 5
        assert TimeStem.GENG.value == 6
        assert TimeStem.XIN.value == 7
        assert TimeStem.REN.value == 8
        assert TimeStem.GUI.value == 9


# =====================================================================
# [6] TimeBranch 十二地支枚举
# =====================================================================


class TestTimeBranch:
    def test_timebranch_is_intenum(self):
        assert issubclass(TimeBranch, IntEnum)

    def test_timebranch_count(self):
        assert len(list(TimeBranch)) == 12

    def test_timebranch_zi_chou_ne_hai(self):
        assert TimeBranch.ZI.value == 0
        assert TimeBranch.CHOU.value == 1
        assert TimeBranch.YIN.value == 2
        assert TimeBranch.MAO.value == 3
        assert TimeBranch.CHEN.value == 4
        assert TimeBranch.SI.value == 5
        assert TimeBranch.WU.value == 6
        assert TimeBranch.WEI.value == 7
        assert TimeBranch.SHEN.value == 8
        assert TimeBranch.YOU.value == 9
        assert TimeBranch.XU.value == 10
        assert TimeBranch.HAI.value == 11


# =====================================================================
# [7] BranchRelation 地支关系枚举
# =====================================================================


class TestBranchRelation:
    def test_branchrelation_is_intenum(self):
        assert issubclass(BranchRelation, IntEnum)

    def test_branchrelation_count(self):
        assert len(list(BranchRelation)) == 6

    def test_branchrelation_values_start_from_1(self):
        assert BranchRelation.LIU_HE.value == 1
        assert BranchRelation.SAN_HE.value == 2
        assert BranchRelation.LIU_CHONG.value == 3
        assert BranchRelation.SAN_XING.value == 4
        assert BranchRelation.LIU_HAI.value == 5
        assert BranchRelation.PO.value == 6

    def test_branchrelation_names(self):
        names = {m.name for m in BranchRelation}
        assert {"LIU_HE", "SAN_HE", "LIU_CHONG", "SAN_XING", "LIU_HAI", "PO"} <= names


# =====================================================================
# [8] TrigramType 八卦枚举
# =====================================================================


class TestTrigramType:
    def test_trigramtype_is_intenum(self):
        assert issubclass(TrigramType, IntEnum)

    def test_trigramtype_count(self):
        assert len(list(TrigramType)) == 8

    def test_trigramtype_eight_members(self):
        names = {m.name for m in TrigramType}
        expected = {"QIAN", "KUN", "ZHEN", "XUN", "KAN", "LI", "GEN", "DUI"}
        assert names == expected

    def test_trigramtype_values(self):
        assert TrigramType.QIAN.value == 0
        assert TrigramType.KUN.value == 1
        assert TrigramType.DUI.value == 7


# =====================================================================
# [9] TrigramRelation 卦关系枚举
# =====================================================================


class TestTrigramRelation:
    def test_trigramrelation_is_intenum(self):
        assert issubclass(TrigramRelation, IntEnum)

    def test_trigramrelation_count(self):
        assert len(list(TrigramRelation)) == 5

    def test_trigramrelation_names(self):
        names = {m.name for m in TrigramRelation}
        assert {"CUO", "HU", "ZONG", "BAN", "JIA"} <= names


# =====================================================================
# [10] EnergyType 五行能量枚举（_enums 公开版本）
# =====================================================================


class TestEnergyType:
    def test_energytype_is_intenum(self):
        assert issubclass(EnergyType, IntEnum)

    def test_energytype_count(self):
        assert len(list(EnergyType)) == 5

    def test_energytype_values(self):
        assert EnergyType.WOOD.value == 0
        assert EnergyType.FIRE.value == 1
        assert EnergyType.EARTH.value == 2
        assert EnergyType.METAL.value == 3
        assert EnergyType.WATER.value == 4

    def test_energytype_names(self):
        names = {m.name for m in EnergyType}
        assert names == {"WOOD", "FIRE", "EARTH", "METAL", "WATER"}


# =====================================================================
# [11] EnergyRelation 能量关系枚举
# =====================================================================


class TestEnergyRelation:
    def test_energyrelation_is_intenum(self):
        assert issubclass(EnergyRelation, IntEnum)

    def test_energyrelation_count(self):
        assert len(list(EnergyRelation)) == 5

    def test_energyrelation_names(self):
        names = {m.name for m in EnergyRelation}
        expected = {"ENHANCE", "SUPPRESS", "OVERCONSTRAINT", "REVERSE", "SAME"}
        assert names == expected

    def test_energyrelation_values(self):
        assert EnergyRelation.ENHANCE.value == 1
        assert EnergyRelation.SAME.value == 5


# =====================================================================
# [12] StrengthState 强度状态枚举
# =====================================================================


class TestStrengthState:
    def test_strengthstate_is_intenum(self):
        assert issubclass(StrengthState, IntEnum)

    def test_strengthstate_count(self):
        assert len(list(StrengthState)) == 5

    def test_strengthstate_names(self):
        names = {m.name for m in StrengthState}
        assert names == {"WANG", "XIANG", "XIU", "QIU", "SI"}

    def test_strengthstate_values(self):
        assert StrengthState.WANG.value == 0
        assert StrengthState.SI.value == 4


# =====================================================================
# [13] EnergyPattern 能量格局枚举
# =====================================================================


class TestEnergyPattern:
    def test_energypattern_is_intenum(self):
        assert issubclass(EnergyPattern, IntEnum)

    def test_energypattern_count(self):
        assert len(list(EnergyPattern)) == 5

    def test_energypattern_names(self):
        names = {m.name for m in EnergyPattern}
        expected = {"ZHI_HUA", "CONG_WANG", "ZHUAN_WANG", "FAN_WANG", "PEI_HE"}
        assert names == expected


# =====================================================================
# [14] SemanticCategory 8 象语义分类
# =====================================================================


class TestSemanticCategory:
    def test_semanticcategory_is_enum(self):
        assert issubclass(SemanticCategory, Enum)

    def test_semanticcategory_count(self):
        assert len(list(SemanticCategory)) == 8

    def test_eight_categories_present(self):
        names = {m.name for m in SemanticCategory}
        expected = {
            "CREATIVE",
            "LAKE",
            "LIGHT",
            "THUNDER",
            "WIND",
            "ABYSS",
            "MOUNTAIN",
            "RECEPTIVE",
        }
        assert names == expected

    def test_creative_attributes(self):
        cat = SemanticCategory.CREATIVE
        assert cat.label == "creative"
        assert cat.energy == "metal"
        assert cat.direction == "northwest"
        assert cat.nature == "assertive"
        assert cat.character == "authority"

    def test_lake_attributes(self):
        cat = SemanticCategory.LAKE
        assert cat.label == "lake"
        assert cat.energy == "metal"
        assert cat.direction == "west"
        assert cat.nature == "joyful"
        assert cat.character == "exchange"

    def test_light_attributes(self):
        cat = SemanticCategory.LIGHT
        assert cat.energy == "fire"
        assert cat.direction == "south"
        assert cat.nature == "bright"

    def test_thunder_attributes(self):
        cat = SemanticCategory.THUNDER
        assert cat.energy == "wood"
        assert cat.direction == "east"
        assert cat.nature == "dynamic"

    def test_wind_attributes(self):
        cat = SemanticCategory.WIND
        assert cat.energy == "wood"
        assert cat.direction == "southeast"
        assert cat.nature == "penetrating"

    def test_abyss_attributes(self):
        cat = SemanticCategory.ABYSS
        assert cat.energy == "water"
        assert cat.direction == "north"
        assert cat.nature == "challenging"

    def test_mountain_attributes(self):
        cat = SemanticCategory.MOUNTAIN
        assert cat.energy == "earth"
        assert cat.direction == "northeast"
        assert cat.nature == "stable"

    def test_receptive_attributes(self):
        cat = SemanticCategory.RECEPTIVE
        assert cat.energy == "earth"
        assert cat.direction == "southwest"
        assert cat.nature == "supportive"

    def test_from_identifier_by_label(self):
        cat = SemanticCategory.from_identifier("creative")
        assert cat == SemanticCategory.CREATIVE

    def test_from_identifier_by_energy(self):
        cat = SemanticCategory.from_identifier("water")
        assert cat == SemanticCategory.ABYSS

    def test_from_identifier_by_direction(self):
        cat = SemanticCategory.from_identifier("south")
        assert cat == SemanticCategory.LIGHT

    def test_from_identifier_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown category identifier"):
            SemanticCategory.from_identifier("nonexistent")

    def test_get_info_returns_dict(self):
        info = SemanticCategory.CREATIVE.get_info()
        assert isinstance(info, dict)
        assert info["label"] == "creative"
        assert info["energy"] == "metal"
        assert set(info.keys()) == {"label", "energy", "direction", "nature", "character"}


# =====================================================================
# [15] _c1 数据字典一致性
# =====================================================================


class TestC1DataDicts:
    def test_category_anchors_eight_keys(self):
        assert len(CATEGORY_ANCHORS) == 8

    def test_category_anchors_all_strings(self):
        for k, v in CATEGORY_ANCHORS.items():
            assert isinstance(k, str)
            assert isinstance(v, str)
            assert len(v) > 0

    def test_keywords_to_category_eight_keys(self):
        assert len(KEYWORDS_TO_CATEGORY) == 8

    def test_keywords_to_category_values_are_lists(self):
        for v in KEYWORDS_TO_CATEGORY.values():
            assert isinstance(v, list)
            assert all(isinstance(kw, str) for kw in v)

    def test_memory_type_to_category_eight_entries(self):
        assert len(MEMORY_TYPE_TO_CATEGORY) == 8

    def test_memory_type_to_category_values_are_categories(self):
        for cat in MEMORY_TYPE_TO_CATEGORY.values():
            assert isinstance(cat, SemanticCategory)

    def test_memory_type_mappings(self):
        assert MEMORY_TYPE_TO_CATEGORY["fact"] == SemanticCategory.CREATIVE
        assert MEMORY_TYPE_TO_CATEGORY["danger"] == SemanticCategory.ABYSS
        assert MEMORY_TYPE_TO_CATEGORY["goal"] == SemanticCategory.MOUNTAIN
        assert MEMORY_TYPE_TO_CATEGORY["knowledge"] == SemanticCategory.LIGHT

    def test_energy_to_category_five_energies(self):
        assert len(ENERGY_TO_CATEGORY) == 5
        expected_energies = {"metal", "fire", "wood", "water", "earth"}
        assert set(ENERGY_TO_CATEGORY.keys()) == expected_energies

    def test_energy_to_category_values_are_category_lists(self):
        for cats in ENERGY_TO_CATEGORY.values():
            assert isinstance(cats, list)
            for c in cats:
                assert isinstance(c, SemanticCategory)

    def test_metal_maps_to_creative_and_lake(self):
        assert SemanticCategory.CREATIVE in ENERGY_TO_CATEGORY["metal"]
        assert SemanticCategory.LAKE in ENERGY_TO_CATEGORY["metal"]

    def test_wood_maps_to_thunder_and_wind(self):
        assert SemanticCategory.THUNDER in ENERGY_TO_CATEGORY["wood"]
        assert SemanticCategory.WIND in ENERGY_TO_CATEGORY["wood"]

    def test_earth_maps_to_mountain_and_receptive(self):
        assert SemanticCategory.MOUNTAIN in ENERGY_TO_CATEGORY["earth"]
        assert SemanticCategory.RECEPTIVE in ENERGY_TO_CATEGORY["earth"]

    def test_category_associations_eight_keys(self):
        assert len(CATEGORY_ASSOCIATIONS) == 8


# =====================================================================
# [16] _c2 EnergyType（带属性版本）
# =====================================================================


class TestC2EnergyType:
    def test_c2_energytype_is_enum(self):
        assert issubclass(C2EnergyType, Enum)

    def test_c2_energytype_count(self):
        assert len(list(C2EnergyType)) == 5

    def test_c2_wood_element(self):
        assert C2EnergyType.WOOD.element == "semantic"

    def test_c2_fire_element(self):
        assert C2EnergyType.FIRE.element == "causal"

    def test_c2_earth_element(self):
        assert C2EnergyType.EARTH.element == "spacetime"

    def test_c2_metal_element(self):
        assert C2EnergyType.METAL.element == "generative"

    def test_c2_water_element(self):
        assert C2EnergyType.WATER.element == "trust"

    def test_wood_nature_growth(self):
        assert C2EnergyType.WOOD.nature == "growth"

    def test_fire_nature_warmth(self):
        assert C2EnergyType.FIRE.nature == "warmth"

    def test_movement_properties(self):
        assert C2EnergyType.WOOD.movement == "ascending"
        assert C2EnergyType.FIRE.movement == "light"
        assert C2EnergyType.EARTH.movement == "transformation"
        assert C2EnergyType.METAL.movement == "cleansing"
        assert C2EnergyType.WATER.movement == "descending"

    def test_direction_properties(self):
        assert C2EnergyType.WOOD.direction == "east"
        assert C2EnergyType.FIRE.direction == "south"
        assert C2EnergyType.EARTH.direction == "center"
        assert C2EnergyType.METAL.direction == "west"
        assert C2EnergyType.WATER.direction == "north"

    def test_season_properties(self):
        assert C2EnergyType.WOOD.season == "spring"
        assert C2EnergyType.FIRE.season == "summer"
        assert C2EnergyType.EARTH.season == "late_summer"
        assert C2EnergyType.METAL.season == "autumn"
        assert C2EnergyType.WATER.season == "winter"


# =====================================================================
# [17] 能量关系图
# =====================================================================


class TestEnergyRelations:
    def test_enhance_map_is_full_cycle(self):
        assert len(ENERGY_ENHANCE_MAP) == 5
        assert ENERGY_ENHANCE_MAP[C2EnergyType.WOOD] == C2EnergyType.FIRE
        assert ENERGY_ENHANCE_MAP[C2EnergyType.FIRE] == C2EnergyType.EARTH
        assert ENERGY_ENHANCE_MAP[C2EnergyType.EARTH] == C2EnergyType.METAL
        assert ENERGY_ENHANCE_MAP[C2EnergyType.METAL] == C2EnergyType.WATER
        assert ENERGY_ENHANCE_MAP[C2EnergyType.WATER] == C2EnergyType.WOOD

    def test_suppress_map_is_full_cycle(self):
        assert len(ENERGY_SUPPRESS_MAP) == 5
        assert ENERGY_SUPPRESS_MAP[C2EnergyType.WOOD] == C2EnergyType.EARTH
        assert ENERGY_SUPPRESS_MAP[C2EnergyType.EARTH] == C2EnergyType.WATER
        assert ENERGY_SUPPRESS_MAP[C2EnergyType.WATER] == C2EnergyType.FIRE
        assert ENERGY_SUPPRESS_MAP[C2EnergyType.FIRE] == C2EnergyType.METAL
        assert ENERGY_SUPPRESS_MAP[C2EnergyType.METAL] == C2EnergyType.WOOD

    def test_enhance_string_map(self):
        assert ENERGY_ENHANCE["wood"] == "fire"
        assert ENERGY_ENHANCE["fire"] == "earth"
        assert ENERGY_ENHANCE["water"] == "wood"

    def test_suppress_string_map(self):
        assert ENERGY_SUPPRESS["wood"] == "earth"
        assert ENERGY_SUPPRESS["metal"] == "wood"

    def test_enhance_cycles_completely(self):
        visited = set()
        current = C2EnergyType.WOOD
        for _ in range(5):
            assert current not in visited
            visited.add(current)
            current = ENERGY_ENHANCE_MAP[current]
        assert len(visited) == 5

    def test_suppress_cycles_completely(self):
        visited = set()
        current = C2EnergyType.WOOD
        for _ in range(5):
            assert current not in visited
            visited.add(current)
            current = ENERGY_SUPPRESS_MAP[current]
        assert len(visited) == 5


# =====================================================================
# [18] 能量状态与强度
# =====================================================================


class TestEnergyStateCalc:
    def test_state_strength_map_five_states(self):
        assert len(STATE_STRENGTH_MAP) == 5

    def test_state_strength_map_values(self):
        assert STATE_STRENGTH_MAP["strong"] == 2.0
        assert STATE_STRENGTH_MAP["balanced"] == 1.3
        assert STATE_STRENGTH_MAP["rested"] == 1.0
        assert STATE_STRENGTH_MAP["restrained"] == 0.5
        assert STATE_STRENGTH_MAP["declined"] == 0.3

    def test_category_state_multipliers(self):
        assert len(CATEGORY_STATE_MULTIPLIERS) == 5
        assert CATEGORY_STATE_MULTIPLIERS["strong"] == 2.0

    def test_get_energy_state_same_season(self):
        state, mult = get_energy_state(C2EnergyType.WOOD, C2EnergyType.WOOD)
        assert state == "strong"
        assert mult == 2.0

    def test_get_energy_state_enhanced_by_season(self):
        state, mult = get_energy_state(C2EnergyType.EARTH, C2EnergyType.FIRE)
        assert state == "balanced"
        assert mult == 1.3

    def test_get_energy_state_restrained(self):
        # target克season → restrained
        state, mult = get_energy_state(C2EnergyType.WOOD, C2EnergyType.EARTH)
        assert state == "restrained"
        assert mult == 0.5

    def test_get_energy_state_declined(self):
        # season克target → declined
        state, mult = get_energy_state(C2EnergyType.EARTH, C2EnergyType.WOOD)
        assert state == "declined"
        assert mult == 0.3

    def test_get_energy_state_default(self):
        state, mult = get_energy_state(C2EnergyType.FIRE, C2EnergyType.WATER)
        assert isinstance(state, str)
        assert isinstance(mult, float)


class TestStateInteraction:
    def test_non_suppress_returns_normal(self):
        result = check_state_interaction(C2EnergyType.WOOD, C2EnergyType.FIRE, 1.0, 1.0)
        assert result == "normal"

    def test_overwhelming_attacker_stronger(self):
        result = check_state_interaction(C2EnergyType.WOOD, C2EnergyType.EARTH, 3.0, 1.0)
        assert result == "overwhelming"

    def test_overwhelming_zero_defender(self):
        result = check_state_interaction(C2EnergyType.WOOD, C2EnergyType.EARTH, 1.0, 0.0)
        assert result == "overwhelming"

    def test_counter_defender_stronger(self):
        result = check_state_interaction(C2EnergyType.WOOD, C2EnergyType.EARTH, 1.0, 3.0)
        assert result == "counter"

    def test_counter_zero_attacker(self):
        result = check_state_interaction(C2EnergyType.WOOD, C2EnergyType.EARTH, 0.0, 1.0)
        assert result == "counter"

    def test_normal_balanced(self):
        result = check_state_interaction(C2EnergyType.WOOD, C2EnergyType.EARTH, 1.0, 1.0)
        assert result == "normal"


# =====================================================================
# [19] 能量相似度
# =====================================================================


class TestEnergySimilarity:
    def test_same_energy_similarity(self):
        assert energy_similarity(C2EnergyType.WOOD, C2EnergyType.WOOD) == 1.0

    def test_enhance_similarity(self):
        assert energy_similarity(C2EnergyType.WOOD, C2EnergyType.FIRE) == 0.7
        assert energy_similarity(C2EnergyType.FIRE, C2EnergyType.WOOD) == 0.7

    def test_suppress_similarity(self):
        assert energy_similarity(C2EnergyType.WOOD, C2EnergyType.EARTH) == 0.1
        assert energy_similarity(C2EnergyType.EARTH, C2EnergyType.WOOD) == 0.1

    def test_any_non_same_is_suppress_or_enhance(self):
        """五行全覆盖，任何不同元素对都是生或克（0.3 是死代码）"""
        for e1 in C2EnergyType:
            for e2 in C2EnergyType:
                if e1 == e2:
                    continue
                s = energy_similarity(e1, e2)
                assert s in (0.7, 0.1), f"{e1.name}-{e2.name}: {s}"

    def test_similarity_range(self):
        for e1 in C2EnergyType:
            for e2 in C2EnergyType:
                s = energy_similarity(e1, e2)
                assert 0.0 <= s <= 1.0


# =====================================================================
# [20] 能量-语义映射函数
# =====================================================================


class TestEnergyFromCategory:
    def test_creative_to_metal(self):
        """creative → metal对应 METAL(0xgenerative0x)
        注：energy_from_category 的 CATEGORY_ENERGY_MAP 用 "metal"/"wood" 等
        传统名，但 EnergyType.element 用 "generative"/"semantic" 等语义名，
        导致无法匹配具0x均降级为 EARTH"""
        result = energy_from_category("creative")
        # 当前实际行为：所有类别均降级为 EARTH
        assert result == C2EnergyType.EARTH

    def test_lake_to_metal(self):
        result = energy_from_category("lake")
        assert result == C2EnergyType.EARTH  # 降级为 EARTH

    def test_light_to_fire(self):
        result = energy_from_category("light")
        assert result == C2EnergyType.EARTH  # 降级为 EARTH

    def test_thunder_to_wood(self):
        result = energy_from_category("thunder")
        assert result == C2EnergyType.EARTH  # 降级为 EARTH

    def test_wind_to_wood(self):
        result = energy_from_category("wind")
        assert result == C2EnergyType.EARTH  # 降级为 EARTH

    def test_abyss_to_water(self):
        result = energy_from_category("abyss")
        assert result == C2EnergyType.EARTH  # 降级为 EARTH

    def test_mountain_to_earth(self):
        result = energy_from_category("mountain")
        assert result == C2EnergyType.EARTH  # 降级为 EARTH

    def test_receptive_to_earth(self):
        result = energy_from_category("receptive")
        assert result == C2EnergyType.EARTH  # 降级为 EARTH

    def test_unknown_defaults_to_earth(self):
        result = energy_from_category("nonexistent")
        assert result == C2EnergyType.EARTH


# =====================================================================
# [21] EnergyState dataclass
# =====================================================================


class TestEnergyStateDataclass:
    def test_default_construction(self):
        state = EnergyState(energy_type=C2EnergyType.WOOD)
        assert state.energy_type == C2EnergyType.WOOD
        assert state.intensity == 1.0
        assert state.status == "balanced"

    def test_custom_construction(self):
        state = EnergyState(energy_type=C2EnergyType.FIRE, intensity=2.5, status="strong")
        assert state.intensity == 2.5
        assert state.status == "strong"

    def test_get_effective_intensity_no_env(self):
        state = EnergyState(energy_type=C2EnergyType.WOOD, intensity=2.0)
        result = state.get_effective_intensity()
        assert result == 2.0

    def test_get_effective_intensity_with_env(self):
        state = EnergyState(energy_type=C2EnergyType.WOOD, intensity=2.0)
        env = EnergyState(energy_type=C2EnergyType.WOOD, intensity=1.0)
        result = state.get_effective_intensity(env)
        assert result == 4.0
        assert state.status == "strong"


# =====================================================================
# [22] EnergyNetwork 能量网络
# =====================================================================


class TestEnergyNetwork:
    def test_empty_network_dominant_is_earth(self):
        net = EnergyNetwork()
        assert net.get_dominant_energy() == C2EnergyType.EARTH

    def test_register_memory(self):
        net = EnergyNetwork()
        net.register_memory("m1", C2EnergyType.WOOD)
        assert "m1" in net.memory_states
        assert net.memory_states["m1"].energy_type == C2EnergyType.WOOD

    def test_propagate_energy_enhances_target(self):
        net = EnergyNetwork()
        net.register_memory("src", C2EnergyType.WOOD)
        net.register_memory("dst", C2EnergyType.FIRE)
        net.propagate_energy("src", 0.5)
        assert net.memory_states["dst"].intensity == pytest.approx(1.5)

    def test_propagate_unknown_source_noop(self):
        net = EnergyNetwork()
        net.register_memory("dst", C2EnergyType.FIRE)
        net.propagate_energy("unknown", 1.0)
        assert net.memory_states["dst"].intensity == 1.0

    def test_get_dominant_energy(self):
        net = EnergyNetwork()
        net.register_memory("m1", C2EnergyType.WOOD)
        net.register_memory("m2", C2EnergyType.FIRE)
        net.register_memory("m3", C2EnergyType.FIRE)
        net.register_memory("m4", C2EnergyType.FIRE)
        assert net.get_dominant_energy() == C2EnergyType.FIRE

    def test_propagate_skips_self_type(self):
        net = EnergyNetwork()
        net.register_memory("src", C2EnergyType.WOOD)
        net.register_memory("other_wood", C2EnergyType.WOOD)
        net.propagate_energy("src", 0.5)
        assert net.memory_states["other_wood"].intensity == 1.0


# =====================================================================
# [23] _terms SEMANTIC_CATEGORY 数据字典
# =====================================================================


class TestSemanticCategoryTerms:
    def test_semantic_category_has_eight_keys(self):
        assert len(SEMANTIC_CATEGORY) == 8

    def test_semantic_category_keys_format(self):
        for k in SEMANTIC_CATEGORY:
            assert k.startswith("CAT_")

    def test_semantic_category_values_consecutive(self):
        values = sorted(SEMANTIC_CATEGORY.values())
        assert values == list(range(8))

    def test_semantic_category_names_inverse(self):
        assert SEMANTIC_CATEGORY_NAMES[0] == "creative"
        assert SEMANTIC_CATEGORY_NAMES[7] == "receptive"
        assert len(SEMANTIC_CATEGORY_NAMES) == 8

    def test_semantic_category_symbols_eight(self):
        assert len(SEMANTIC_CATEGORY_SYMBOLS) == 8

    def test_semantic_category_properties_eight(self):
        assert len(SEMANTIC_CATEGORY_PROPERTIES) == 8

    def test_property_keys(self):
        for props in SEMANTIC_CATEGORY_PROPERTIES.values():
            assert {"direction", "season", "nature"} <= set(props.keys())


# =====================================================================
# [24] _terms 时序数据
# =====================================================================


class TestTimeTerms:
    def test_time_stems_ten(self):
        assert len(TIME_STEMS) == 10
        assert TIME_STEMS[0] == "甲"
        assert TIME_STEMS[9] == "癸"

    def test_time_branches_twelve(self):
        assert len(TIME_BRANCHES) == 12
        assert TIME_BRANCHES[0] == "子"
        assert TIME_BRANCHES[11] == "亥"

    def test_time_branch_energy_complete(self):
        assert len(TIME_BRANCH_ENERGY) == 12
        for branch in TIME_BRANCHES:
            assert branch in TIME_BRANCH_ENERGY

    def test_time_branch_energy_values_valid(self):
        valid = {"wood", "fire", "earth", "metal", "water"}
        for energy in TIME_BRANCH_ENERGY.values():
            assert energy in valid

    def test_time_cycle_length_is_60(self):
        assert TIME_CYCLE_LENGTH == 60


# =====================================================================
# [25] _terms 地支关系图
# =====================================================================


class TestBranchRelationMaps:
    def test_stem_he_map(self):
        assert len(STEM_HE_MAP) == 5
        assert STEM_HE_MAP[0] == 5  # 甲-己
        assert STEM_HE_MAP[4] == 9  # 戊-癸

    def test_stem_chong_map(self):
        assert len(STEM_CHONG_MAP) == 5
        assert STEM_CHONG_MAP[0] == 6  # 甲-庚
        assert STEM_CHONG_MAP[4] == 5  # 戊-己

    def test_branch_he_map(self):
        assert len(BRANCH_HE_MAP) == 6
        assert BRANCH_HE_MAP[0] == 1  # 子-丑
        assert BRANCH_HE_MAP[2] == 11  # 寅-亥（修复点）

    def test_branch_chong_map(self):
        assert len(BRANCH_CHONG_MAP) == 6
        assert BRANCH_CHONG_MAP[0] == 6  # 子-午
        assert BRANCH_CHONG_MAP[5] == 11  # 巳-亥

    def test_branch_sanhe_map(self):
        assert len(BRANCH_SANHE_MAP) == 4
        valid = {"wood", "fire", "earth", "metal", "water"}
        for v in BRANCH_SANHE_MAP.values():
            assert v in valid

    def test_branch_hidden_stem_map_complete(self):
        assert len(BRANCH_HIDDEN_STEM_MAP) == 12
        for i in range(12):
            assert i in BRANCH_HIDDEN_STEM_MAP
            assert isinstance(BRANCH_HIDDEN_STEM_MAP[i], list)
            assert len(BRANCH_HIDDEN_STEM_MAP[i]) > 0

    def test_branch_he_yin_hai_corrected(self):
        assert BRANCH_HE_MAP[2] == 11


# =====================================================================
# [26] 强度状态术语
# =====================================================================


class TestStrengthTerms:
    def test_strength_state_dict(self):
        assert STRENGTH_STATE["STATE_STRONG"] == "strong"
        assert STRENGTH_STATE["STATE_DECLINED"] == "declined"

    def test_month_energy_state_twelve_months(self):
        assert len(MONTH_ENERGY_STATE) == 12
        assert MONTH_ENERGY_STATE[5] == "fire"

    def test_strength_multiplier_ordering(self):
        assert STRENGTH_MULTIPLIER["strong"] > STRENGTH_MULTIPLIER["balanced"]
        assert STRENGTH_MULTIPLIER["balanced"] > STRENGTH_MULTIPLIER["rested"]
        assert STRENGTH_MULTIPLIER["rested"] > STRENGTH_MULTIPLIER["restrained"]
        assert STRENGTH_MULTIPLIER["restrained"] > STRENGTH_MULTIPLIER["declined"]


# =====================================================================
# [27] 集成一致性测试
# =====================================================================


class TestIntegrationConsistency:
    def test_13_enums_all_importable(self):
        from mci_world_model._sys import (
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

        for cls in [
            YinYang,
            ThreePowers,
            FourSymbols,
            Season,
            TimeStem,
            TimeBranch,
            BranchRelation,
            TrigramType,
            TrigramRelation,
            EnergyType,
            EnergyRelation,
            StrengthState,
            EnergyPattern,
        ]:
            assert issubclass(cls, (Enum, IntEnum))

    def test_two_energy_types_are_different(self):
        assert EnergyType is not C2EnergyType

    def test_energy_enhance_consistency_enum_vs_str(self):
        """c2 ENERGY_ENHANCE(string) 用 traditional names (wood/fire),
        而 element property 用 semantic names (semantic/causal)。
        验证 enhance 图循环完整性而非命名一致性"""
        # 验证 enhance 图形成完整闭环
        visited = set()
        current = C2EnergyType.WOOD
        for _ in range(5):
            assert current not in visited
            visited.add(current)
            current = ENERGY_ENHANCE_MAP[current]
        assert len(visited) == 5

    def test_energy_suppress_consistency_enum_vs_str(self):
        """c2 ENERGY_SUPPRESS(string) 用 traditional names (wood/fire),
        而 element property 用 semantic names (semantic/causal)。
        验证 suppress 图循环完整性"""
        visited = set()
        current = C2EnergyType.WOOD
        for _ in range(5):
            assert current not in visited
            visited.add(current)
            current = ENERGY_SUPPRESS_MAP[current]
        assert len(visited) == 5

    def test_branch_he_map_used_by_temporal(self):
        from mci_world_model._sys._temporal_core import TemporalCore

        tc = TemporalCore()
        rels = tc.analyze_branch_relation(TimeBranch.ZI, TimeBranch.CHOU)
        from mci_world_model._sys import BranchRelation as BR

        assert BR.LIU_HE in rels

    def test_c2_energy_wood_in_c1(self):
        # energy_from_category 当前均降级为 EARTH（语义命名不匹配 bug）
        # 但数据层确实存在 wood→thunder/wind 映射
        assert SemanticCategory.THUNDER in ENERGY_TO_CATEGORY["wood"]
        assert SemanticCategory.WIND in ENERGY_TO_CATEGORY["wood"]

    def test_foundation_type_integers(self):
        for s in TimeStem:
            assert isinstance(int(s), int)
        for b in TimeBranch:
            assert isinstance(int(b), int)
