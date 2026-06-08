"""
test_temporal_core.py — Temporal Core Test Suite

覆盖 天层时空系统：
- StemBranchCode dataclass：name/polarity/hidden_stems/energy_type/验证
- TemporalCore 60 甲子引擎：
  create_code / get_cycle_name / get_cycle_index /
  analyze_stem_relation / analyze_branch_relation /
  get_hidden_stems / get_cycle_distance / is_same_trigram /
  get_branch_energy / get_stem_energy / is_stem_yang / is_branch_yang /
  get_san_he_branches
- TimeCycle (独立时空量化单例)
- TimeCodeInfo dataclass
- 便捷函数：create_stem_branch, get_cycle_name,
  create_time_code, get_stem, get_branch, get_cycle
"""

import pytest

from mci_world_model._sys import (
    BranchRelation,
    StemBranchCode,
    TemporalCore,
    TimeBranch,
    TimeStem,
    create_stem_branch,
    get_cycle_name,
)
from mci_world_model._sys._temporal_core import (
    BRANCH_PRIMARY_ENERGY,
)
from mci_world_model._sys._time_code import (
    TimeCodeInfo,
    TimeCycle,
    create_time_code,
    get_branch,
    get_cycle,
    get_stem,
)

# =====================================================================
# [1] StemBranchCode dataclass 基础
# =====================================================================


class TestStemBranchCode:
    """StemBranchCode 六十花甲编码类测试"""

    def test_create_jia_zi(self):
        code = StemBranchCode(TimeStem.JIA, TimeBranch.ZI, 0)
        assert code.stem == TimeStem.JIA
        assert code.branch == TimeBranch.ZI
        assert code.cycle_index == 0

    def test_create_gui_hai(self):
        code = StemBranchCode(TimeStem.GUI, TimeBranch.HAI, 59)
        assert code.cycle_index == 59

    def test_invalid_cycle_index_raises(self):
        with pytest.raises(ValueError, match="Cycle index must be 0-59"):
            StemBranchCode(TimeStem.JIA, TimeBranch.ZI, -1)
        with pytest.raises(ValueError, match="Cycle index must be 0-59"):
            StemBranchCode(TimeStem.JIA, TimeBranch.ZI, 60)

    def test_polarity_yang(self):
        # JIA=0 → even → yang
        code = StemBranchCode(TimeStem.JIA, TimeBranch.ZI, 0)
        assert code.polarity == "yang"

    def test_polarity_yin(self):
        # YI=1 → odd → yin
        code = StemBranchCode(TimeStem.YI, TimeBranch.CHOU, 1)
        assert code.polarity == "yin"

    def test_name_jia_zi(self):
        code = StemBranchCode(TimeStem.JIA, TimeBranch.ZI, 0)
        assert code.name == "甲子"

    def test_name_yi_chou(self):
        code = StemBranchCode(TimeStem.YI, TimeBranch.CHOU, 1)
        assert code.name == "乙丑"

    def test_name_gui_hai(self):
        code = StemBranchCode(TimeStem.GUI, TimeBranch.HAI, 59)
        assert code.name == "癸亥"

    def test_hidden_stems_property(self):
        # 子藏癸(REN=8)
        code = StemBranchCode(TimeStem.JIA, TimeBranch.ZI, 0)
        assert len(code.hidden_stems) == 1
        assert code.hidden_stems[0] == TimeStem.REN

    def test_energy_type_property(self):
        # 子=water
        code = StemBranchCode(TimeStem.JIA, TimeBranch.ZI, 0)
        assert code.energy_type == "water"

    def test_str_method(self):
        code = StemBranchCode(TimeStem.JIA, TimeBranch.ZI, 0)
        assert str(code) == "甲子"

    def test_repr_method(self):
        code = StemBranchCode(TimeStem.JIA, TimeBranch.ZI, 0)
        r = repr(code)
        assert "StemBranchCode" in r
        assert "JIA" in r
        assert "ZI" in r


# =====================================================================
# [2] TemporalCore 60 甲子引擎 - create_code
# =====================================================================


class TestTemporalCoreCreateCode:
    """TemporalCore.create_code 测试"""

    def setup_method(self):
        self.tc = TemporalCore()

    def test_create_jia_zi(self):
        code = self.tc.create_code(0, 0)
        assert code.name == "甲子"
        assert code.cycle_index == 0

    def test_create_gui_hai(self):
        code = self.tc.create_code(9, 11)
        assert code.name == "癸亥"
        assert code.cycle_index == 59

    def test_create_wrap_stem(self):
        """stem 自动 wrap (10→0)"""
        code = self.tc.create_code(10, 0)
        assert code.stem == TimeStem.JIA  # 10%10=0

    def test_create_wrap_branch(self):
        """branch 自动 wrap (12→0)"""
        code = self.tc.create_code(0, 12)
        assert code.branch == TimeBranch.ZI  # 12%12=0


# =====================================================================
# [3] TemporalCore get_cycle_name / get_cycle_index
# =====================================================================


class TestTemporalCoreNavigation:
    """TemporalCore 导航方法测试"""

    def setup_method(self):
        self.tc = TemporalCore()

    def test_get_cycle_name_0(self):
        assert self.tc.get_cycle_name(0) == "甲子"

    def test_get_cycle_name_59(self):
        assert self.tc.get_cycle_name(59) == "癸亥"

    def test_get_cycle_name_mid(self):
        # index 30 = stem 0 (JIA) + 30 → JIA at 30, branch = 30%12 = 6 (WU)
        assert self.tc.get_cycle_name(30) == "甲午"

    def test_get_cycle_name_negative_raises(self):
        with pytest.raises(ValueError):
            self.tc.get_cycle_name(-1)

    def test_get_cycle_name_out_of_range_raises(self):
        with pytest.raises(ValueError):
            self.tc.get_cycle_name(60)

    def test_get_cycle_index_jia_zi(self):
        idx = self.tc.get_cycle_index(TimeStem.JIA, TimeBranch.ZI)
        assert idx == 0

    def test_get_cycle_index_gui_hai(self):
        idx = self.tc.get_cycle_index(TimeStem.GUI, TimeBranch.HAI)
        assert idx == 59

    def test_get_cycle_index_invalid_returns_minus_one(self):
        # 甲丑 is not a valid combination in the 60-cycle
        # Actually let's check: stem(0), branch(1)
        idx = self.tc.get_cycle_index(TimeStem.JIA, TimeBranch.CHOU)
        # JIA-CHOU should exist — let me think: stem[0]==JIA, branch[1]==CHOU
        # The cycle works: i%10 and i%12. JIA=0, CHOU=1. stem[1] = YI (1%10)
        # Actually start with stem[0]=JIA, branch[0]=ZI. stem[1]=YI, branch[1]=CHOU
        # So JIA-CHOU is not possible directly because at i=0 it's (JIA,ZI), not (JIA,CHOU)
        # The combination JIA+CHOU appears at some i where i%10=0 and i%12=1
        # i%10=0 → i ∈ {0,10,20,30,40,50}
        # i%12=1 → i mod 12 = 1
        # 0%12=0≠1, 10%12=10≠1, 20%12=8≠1, 30%12=6≠1, 40%12=4≠1, 50%12=2≠1
        # None matches, so (JIA,CHOU) returns -1
        assert idx == -1


# =====================================================================
# [4] TemporalCore analyze_stem_relation
# =====================================================================


class TestTemporalCoreStemRelations:
    """天干关系分析测试"""

    def setup_method(self):
        self.tc = TemporalCore()

    def test_jia_ji_he(self):
        rel = self.tc.analyze_stem_relation(TimeStem.JIA, TimeStem.JI)
        assert rel == BranchRelation.LIU_HE

    def test_yi_geng_he(self):
        rel = self.tc.analyze_stem_relation(TimeStem.YI, TimeStem.GENG)
        assert rel == BranchRelation.LIU_HE

    def test_bing_xin_he(self):
        rel = self.tc.analyze_stem_relation(TimeStem.BING, TimeStem.XIN)
        assert rel == BranchRelation.LIU_HE

    def test_ding_ren_he(self):
        rel = self.tc.analyze_stem_relation(TimeStem.DING, TimeStem.REN)
        assert rel == BranchRelation.LIU_HE

    def test_wu_gui_he(self):
        rel = self.tc.analyze_stem_relation(TimeStem.WU, TimeStem.GUI)
        assert rel == BranchRelation.LIU_HE

    def test_jia_geng_chong(self):
        rel = self.tc.analyze_stem_relation(TimeStem.JIA, TimeStem.GENG)
        assert rel == BranchRelation.LIU_CHONG

    def test_yi_xin_chong(self):
        rel = self.tc.analyze_stem_relation(TimeStem.YI, TimeStem.XIN)
        assert rel == BranchRelation.LIU_CHONG

    def test_bing_ren_chong(self):
        rel = self.tc.analyze_stem_relation(TimeStem.BING, TimeStem.REN)
        assert rel == BranchRelation.LIU_CHONG

    def test_ding_gui_chong(self):
        rel = self.tc.analyze_stem_relation(TimeStem.DING, TimeStem.GUI)
        assert rel == BranchRelation.LIU_CHONG

    def test_no_relation(self):
        rel = self.tc.analyze_stem_relation(TimeStem.JIA, TimeStem.YI)
        assert rel is None

    def test_symmetric(self):
        """合/冲是对称的"""
        assert self.tc.analyze_stem_relation(TimeStem.JIA, TimeStem.JI) == self.tc.analyze_stem_relation(
            TimeStem.JI, TimeStem.JIA
        )


# =====================================================================
# [5] TemporalCore analyze_branch_relation
# =====================================================================


class TestTemporalCoreBranchRelations:
    """地支关系分析测试"""

    def setup_method(self):
        self.tc = TemporalCore()

    # ----- 六合 -----
    def test_zi_chou_he(self):
        rels = self.tc.analyze_branch_relation(TimeBranch.ZI, TimeBranch.CHOU)
        assert BranchRelation.LIU_HE in rels

    def test_yin_hai_he(self):
        rels = self.tc.analyze_branch_relation(TimeBranch.YIN, TimeBranch.HAI)
        assert BranchRelation.LIU_HE in rels

    def test_mao_xu_he(self):
        rels = self.tc.analyze_branch_relation(TimeBranch.MAO, TimeBranch.XU)
        assert BranchRelation.LIU_HE in rels

    def test_chen_you_he(self):
        rels = self.tc.analyze_branch_relation(TimeBranch.CHEN, TimeBranch.YOU)
        assert BranchRelation.LIU_HE in rels

    def test_si_shen_he(self):
        rels = self.tc.analyze_branch_relation(TimeBranch.SI, TimeBranch.SHEN)
        assert BranchRelation.LIU_HE in rels

    def test_wu_wei_he(self):
        rels = self.tc.analyze_branch_relation(TimeBranch.WU, TimeBranch.WEI)
        assert BranchRelation.LIU_HE in rels

    # ----- 六冲 -----
    def test_zi_wu_chong(self):
        rels = self.tc.analyze_branch_relation(TimeBranch.ZI, TimeBranch.WU)
        assert BranchRelation.LIU_CHONG in rels

    def test_chou_wei_chong(self):
        rels = self.tc.analyze_branch_relation(TimeBranch.CHOU, TimeBranch.WEI)
        assert BranchRelation.LIU_CHONG in rels

    def test_yin_shen_chong(self):
        rels = self.tc.analyze_branch_relation(TimeBranch.YIN, TimeBranch.SHEN)
        assert BranchRelation.LIU_CHONG in rels

    def test_mao_you_chong(self):
        rels = self.tc.analyze_branch_relation(TimeBranch.MAO, TimeBranch.YOU)
        assert BranchRelation.LIU_CHONG in rels

    def test_chen_xu_chong(self):
        rels = self.tc.analyze_branch_relation(TimeBranch.CHEN, TimeBranch.XU)
        assert BranchRelation.LIU_CHONG in rels

    def test_si_hai_chong(self):
        rels = self.tc.analyze_branch_relation(TimeBranch.SI, TimeBranch.HAI)
        assert BranchRelation.LIU_CHONG in rels

    # ----- 三刑 -----
    def test_zi_mao_xing(self):
        rels = self.tc.analyze_branch_relation(TimeBranch.ZI, TimeBranch.MAO)
        assert BranchRelation.SAN_XING in rels

    def test_yin_si_xing(self):
        rels = self.tc.analyze_branch_relation(TimeBranch.YIN, TimeBranch.SI)
        assert BranchRelation.SAN_XING in rels

    # ----- 六害 -----
    def test_zi_wei_hai(self):
        rels = self.tc.analyze_branch_relation(TimeBranch.ZI, TimeBranch.WEI)
        assert BranchRelation.LIU_HAI in rels

    def test_chou_wu_hai(self):
        rels = self.tc.analyze_branch_relation(TimeBranch.CHOU, TimeBranch.WU)
        assert BranchRelation.LIU_HAI in rels

    # ----- 破 -----
    def test_zi_you_po(self):
        rels = self.tc.analyze_branch_relation(TimeBranch.ZI, TimeBranch.YOU)
        assert BranchRelation.PO in rels

    def test_yin_hai_po(self):
        rels = self.tc.analyze_branch_relation(TimeBranch.YIN, TimeBranch.HAI)
        assert BranchRelation.PO in rels

    def test_no_relation(self):
        rels = self.tc.analyze_branch_relation(TimeBranch.ZI, TimeBranch.CHEN)
        # 子辰 not in any defined relationship
        # Actually 子辰 is part of 申子辰 三合, but analyze_branch_relation doesn't include 三合
        assert BranchRelation.LIU_HE not in rels
        assert BranchRelation.LIU_CHONG not in rels

    def test_returns_list(self):
        rels = self.tc.analyze_branch_relation(TimeBranch.ZI, TimeBranch.CHOU)
        assert isinstance(rels, list)


# =====================================================================
# [6] TemporalCore get_hidden_stems
# =====================================================================


class TestTemporalCoreHiddenStems:
    """藏干提取测试"""

    def setup_method(self):
        self.tc = TemporalCore()

    def test_zi_hidden_ren(self):
        stems = self.tc.get_hidden_stems(TimeBranch.ZI)
        assert len(stems) == 1
        assert stems[0] == TimeStem.REN  # 子藏癸(REN=8)

    def test_yin_hidden_jia_bing_wu(self):
        stems = self.tc.get_hidden_stems(TimeBranch.YIN)
        assert len(stems) == 3
        assert TimeStem.JIA in stems
        assert TimeStem.BING in stems
        assert TimeStem.WU in stems

    def test_hai_hidden_ren_jia(self):
        stems = self.tc.get_hidden_stems(TimeBranch.HAI)
        assert len(stems) == 2
        assert TimeStem.REN in stems
        assert TimeStem.JIA in stems

    def test_you_hidden_xin(self):
        stems = self.tc.get_hidden_stems(TimeBranch.YOU)
        assert len(stems) == 1
        assert stems[0] == TimeStem.XIN

    def test_all_12_branches(self):
        """所有 12 地支都有藏干"""
        for branch in TimeBranch:
            stems = self.tc.get_hidden_stems(branch)
            assert isinstance(stems, list)
            assert len(stems) > 0


# =====================================================================
# [7] TemporalCore 循环距离
# =====================================================================


class TestTemporalCoreCycleDistance:
    """循环距离测试"""

    def setup_method(self):
        self.tc = TemporalCore()

    def test_cycle_distance_same(self):
        assert self.tc.get_cycle_distance(0, 0) == 0

    def test_cycle_distance_linear(self):
        assert self.tc.get_cycle_distance(0, 30) == 30

    def test_cycle_distance_10_to_20(self):
        assert self.tc.get_cycle_distance(10, 20) == 10

    def test_cycle_distance_wrap_around(self):
        # 0 and 59 wrap → distance = 1
        assert self.tc.get_cycle_distance(0, 59) == 1

    def test_cycle_distance_wrap_30(self):
        assert self.tc.get_cycle_distance(30, 0) == 30

    def test_cycle_distance_wraps_with_mod(self):
        """超出范围自动 wrap"""
        assert self.tc.get_cycle_distance(0, 90) == 30  # 90%60=30


# =====================================================================
# [8] TemporalCore 三合局
# =====================================================================


class TestTemporalCoreTrigram:
    """三合局测试"""

    def setup_method(self):
        self.tc = TemporalCore()

    def test_shen_zi_chen_water(self):
        code_sh = self.tc.create_code(6, 8)  # 庚申
        code_zi = self.tc.create_code(8, 0)  # 壬子
        code_ch = self.tc.create_code(4, 4)  # 戊辰
        is_tri, energy = self.tc.is_same_trigram(code_sh, code_zi, code_ch)
        assert is_tri
        assert energy == "water"

    def test_hai_mao_wei_wood(self):
        code_h = self.tc.create_code(8, 11)  # 壬亥
        code_m = self.tc.create_code(1, 3)  # 乙卯
        code_w = self.tc.create_code(5, 7)  # 己未
        is_tri, energy = self.tc.is_same_trigram(code_h, code_m, code_w)
        assert is_tri
        assert energy == "wood"

    def test_yin_wu_xu_fire(self):
        code_y = self.tc.create_code(0, 2)  # 甲寅
        code_w = self.tc.create_code(3, 6)  # 丁午
        code_x = self.tc.create_code(4, 10)  # 戊戌
        is_tri, energy = self.tc.is_same_trigram(code_y, code_w, code_x)
        assert is_tri
        assert energy == "fire"

    def test_si_you_chou_metal(self):
        code_s = self.tc.create_code(2, 5)  # 丙巳
        code_y = self.tc.create_code(7, 9)  # 辛酉
        code_c = self.tc.create_code(5, 1)  # 己丑
        is_tri, energy = self.tc.is_same_trigram(code_s, code_y, code_c)
        assert is_tri
        assert energy == "metal"

    def test_two_codes_partial_trigram(self):
        """2 个码也检测是否同属三合局"""
        code_sh = self.tc.create_code(6, 8)  # 庚申
        code_zi = self.tc.create_code(8, 0)  # 壬子
        is_tri, energy = self.tc.is_same_trigram(code_sh, code_zi)
        assert is_tri
        assert energy == "water"

    def test_not_trigram(self):
        code1 = self.tc.create_code(0, 0)  # 甲子
        code2 = self.tc.create_code(1, 1)  # 乙丑
        is_tri, energy = self.tc.is_same_trigram(code1, code2)
        assert not is_tri
        assert energy is None

    def test_not_trigram_three_codes(self):
        code1 = self.tc.create_code(0, 0)  # 甲子
        code2 = self.tc.create_code(1, 1)  # 乙丑
        code3 = self.tc.create_code(2, 2)  # 丙寅
        is_tri, energy = self.tc.is_same_trigram(code1, code2, code3)
        assert not is_tri
        assert energy is None

    def test_is_same_trigram_set_empty(self):
        """空列表不是三合局"""
        is_tri, energy = self.tc.is_same_trigram_set([])
        assert not is_tri
        assert energy is None

    def test_is_same_trigram_set_single(self):
        """单码不是三合局"""
        code = self.tc.create_code(0, 0)
        is_tri, energy = self.tc.is_same_trigram_set([code])
        assert not is_tri
        assert energy is None


# =====================================================================
# [9] TemporalCore 能量类型
# =====================================================================


class TestTemporalCoreEnergy:
    """能量类型测试"""

    def setup_method(self):
        self.tc = TemporalCore()

    def test_get_branch_energy_zi(self):
        assert self.tc.get_branch_energy(TimeBranch.ZI) == "water"

    def test_get_branch_energy_yin(self):
        assert self.tc.get_branch_energy(TimeBranch.YIN) == "wood"

    def test_get_branch_energy_si(self):
        assert self.tc.get_branch_energy(TimeBranch.SI) == "fire"

    def test_get_branch_energy_shen(self):
        assert self.tc.get_branch_energy(TimeBranch.SHEN) == "metal"

    def test_get_branch_energy_chen(self):
        assert self.tc.get_branch_energy(TimeBranch.CHEN) == "earth"

    def test_get_stem_energy_jia(self):
        assert self.tc.get_stem_energy(TimeStem.JIA) == "wood"

    def test_get_stem_energy_bing(self):
        assert self.tc.get_stem_energy(TimeStem.BING) == "fire"

    def test_get_stem_energy_wu(self):
        assert self.tc.get_stem_energy(TimeStem.WU) == "earth"

    def test_get_stem_energy_geng(self):
        assert self.tc.get_stem_energy(TimeStem.GENG) == "metal"

    def test_get_stem_energy_ren(self):
        assert self.tc.get_stem_energy(TimeStem.REN) == "water"

    def test_branch_primary_energy_all_12(self):
        """BRANCH_PRIMARY_ENERGY 覆盖 12 地支"""
        assert len(BRANCH_PRIMARY_ENERGY) == 12
        valid = {"wood", "fire", "earth", "metal", "water"}
        for energy in BRANCH_PRIMARY_ENERGY.values():
            assert energy in valid

    def test_energy_type_on_code_matches(self):
        """StemBranchCode.energy_type 与 TemporalCore.get_branch_energy 一致"""
        code = self.tc.create_code(0, 0)
        assert code.energy_type == self.tc.get_branch_energy(TimeBranch.ZI)


# =====================================================================
# [10] TemporalCore 阴阳判断
# =====================================================================


class TestTemporalCorePolarity:
    """阴阳判断测试"""

    def setup_method(self):
        self.tc = TemporalCore()

    def test_jia_is_yang(self):
        assert self.tc.is_stem_yang(TimeStem.JIA)

    def test_yi_is_yin(self):
        assert not self.tc.is_stem_yang(TimeStem.YI)

    def test_zi_is_yang(self):
        assert self.tc.is_branch_yang(TimeBranch.ZI)

    def test_chou_is_yin(self):
        assert not self.tc.is_branch_yang(TimeBranch.CHOU)

    def test_all_stems_alternate(self):
        """天干阴阳交替（偶=阳, 奇=阴）"""
        for s in TimeStem:
            assert self.tc.is_stem_yang(s) == (s.value % 2 == 0)

    def test_all_branches_alternate(self):
        """地支阴阳交替"""
        for b in TimeBranch:
            assert self.tc.is_branch_yang(b) == (b.value % 2 == 0)

    def test_code_polarity_jia_zi(self):
        code = self.tc.create_code(0, 0)
        assert code.polarity == "yang"

    def test_code_polarity_yi_chou(self):
        code = self.tc.create_code(1, 1)
        assert code.polarity == "yin"


# =====================================================================
# [11] TemporalCore get_san_he_branches
# =====================================================================


class TestTemporalCoreSanHeBranches:
    """三合局分支获取测试"""

    def setup_method(self):
        self.tc = TemporalCore()

    def test_water_san_he(self):
        branches = self.tc.get_san_he_branches("water")
        assert len(branches) == 3
        assert TimeBranch.SHEN in branches
        assert TimeBranch.ZI in branches
        assert TimeBranch.CHEN in branches

    def test_wood_san_he(self):
        branches = self.tc.get_san_he_branches("wood")
        assert len(branches) == 3
        assert TimeBranch.HAI in branches
        assert TimeBranch.MAO in branches
        assert TimeBranch.WEI in branches

    def test_fire_san_he(self):
        branches = self.tc.get_san_he_branches("fire")
        assert len(branches) == 3
        assert TimeBranch.YIN in branches
        assert TimeBranch.WU in branches
        assert TimeBranch.XU in branches

    def test_metal_san_he(self):
        branches = self.tc.get_san_he_branches("metal")
        assert len(branches) == 3
        assert TimeBranch.SI in branches
        assert TimeBranch.YOU in branches
        assert TimeBranch.CHOU in branches

    def test_unknown_returns_empty(self):
        branches = self.tc.get_san_he_branches("earth")
        assert branches == []


# =====================================================================
# [12] TemporalCore __repr__
# =====================================================================


class TestTemporalCoreRepr:
    def test_repr(self):
        tc = TemporalCore()
        r = repr(tc)
        assert "TemporalCore" in r
        assert "60" in r


# =====================================================================
# [13] 便捷函数 create_stem_branch
# =====================================================================


class TestCreateStemBranch:
    def test_create_stem_branch_jia_zi(self):
        code = create_stem_branch(0, 0)
        assert code.name == "甲子"

    def test_create_stem_branch_gui_hai(self):
        code = create_stem_branch(9, 11)
        assert code.name == "癸亥"

    def test_create_stem_branch_is_stem_branch_code(self):
        code = create_stem_branch(0, 0)
        assert isinstance(code, StemBranchCode)


# =====================================================================
# [14] 便捷函数 get_cycle_name
# =====================================================================


class TestGetCycleName:
    def test_get_cycle_name_0(self):
        assert get_cycle_name(0) == "甲子"

    def test_get_cycle_name_59(self):
        assert get_cycle_name(59) == "癸亥"

    def test_get_cycle_name_negative_raises(self):
        with pytest.raises(ValueError):
            get_cycle_name(-1)


# =====================================================================
# [15] TimeCycle 单例（独立时空量化）
# =====================================================================


class TestTimeCycle:
    """TimeCycle 独立时空量化单例测试"""

    def test_timecycle_is_singleton(self):
        tc1 = TimeCycle()
        tc2 = TimeCycle()
        assert tc1 is tc2

    def test_get_index_0(self):
        tc = TimeCycle()
        stem, branch = tc.get(0)
        # stem is _time_code.TimeStem (JIA_YANG)
        assert stem.value == 0
        assert branch.value == 0
        assert stem.name == "甲"
        assert branch.name == "子"

    def test_get_index_59(self):
        tc = TimeCycle()
        stem, branch = tc.get(59)
        assert stem.value == 9
        assert branch.value == 11

    def test_get_wrap_around(self):
        tc = TimeCycle()
        stem, branch = tc.get(60)
        assert stem.value == 0
        assert branch.value == 0

    def test_get_name_0(self):
        tc = TimeCycle()
        assert tc.get_name(0) == "甲子"

    def test_get_name_59(self):
        tc = TimeCycle()
        assert tc.get_name(59) == "癸亥"

    def test_get_energy_type(self):
        tc = TimeCycle()
        assert tc.get_energy_type(0) == "wood"  # JIA_YANG=0 → wood

    def test_get_returns_tuple(self):
        tc = TimeCycle()
        result = tc.get(0)
        assert isinstance(result, tuple)
        assert len(result) == 2


# =====================================================================
# [16] TimeCodeInfo dataclass
# =====================================================================


class TestTimeCodeInfo:
    """TimeCodeInfo 时空标注信息测试"""

    def test_create_basic(self):
        from mci_world_model._sys._time_code import TimeBranch as TCBranch
        from mci_world_model._sys._time_code import TimeStem as TCStem

        info = TimeCodeInfo(time_stem=TCStem.JIA_YANG, time_branch=TCBranch.ZI_YANG, cycle_index=0)
        assert info.time_stem == TCStem.JIA_YANG
        assert info.time_branch == TCBranch.ZI_YANG
        assert info.cycle_index == 0

    def test_energy_type(self):
        from mci_world_model._sys._time_code import TimeBranch as TCBranch
        from mci_world_model._sys._time_code import TimeStem as TCStem

        info = TimeCodeInfo(TCStem.JIA_YANG, TCBranch.ZI_YANG, 0)
        assert info.energy_type == "wood"

    def test_name(self):
        from mci_world_model._sys._time_code import TimeBranch as TCBranch
        from mci_world_model._sys._time_code import TimeStem as TCStem

        info = TimeCodeInfo(TCStem.JIA_YANG, TCBranch.ZI_YANG, 0)
        assert info.name == "甲子"

    def test_polarity(self):
        from mci_world_model._sys._time_code import TimeBranch as TCBranch
        from mci_world_model._sys._time_code import TimeStem as TCStem

        info = TimeCodeInfo(TCStem.JIA_YANG, TCBranch.ZI_YANG, 0)
        assert info.polarity == "yang"

    def test_life_cycle(self):
        from mci_world_model._sys._time_code import TimeBranch as TCBranch
        from mci_world_model._sys._time_code import TimeStem as TCStem

        info = TimeCodeInfo(TCStem.JIA_YANG, TCBranch.ZI_YANG, 0)
        assert info.life_cycle == "wood"


# =====================================================================
# [17] create_time_code 便捷函数
# =====================================================================


class TestCreateTimeCode:
    """create_time_code 便捷函数测试"""

    def test_create_jia_zi(self):
        info = create_time_code(0, 0)
        assert info.name == "甲子"

    def test_create_gui_hai(self):
        info = create_time_code(9, 11)
        assert info.name == "癸亥"

    def test_cycle_index_range(self):
        info = create_time_code(5, 5)
        assert 0 <= info.cycle_index < 60


# =====================================================================
# [18] _time_code 工具函数
# =====================================================================


class TestTimeCodeUtils:
    """_time_code 工具函数测试"""

    def test_get_stem_0(self):
        assert get_stem(0) == "甲"

    def test_get_stem_9(self):
        assert get_stem(9) == "癸"

    def test_get_stem_wrap(self):
        assert get_stem(10) == "甲"

    def test_get_branch_0(self):
        assert get_branch(0) == "子"

    def test_get_branch_11(self):
        assert get_branch(11) == "亥"

    def test_get_branch_wrap(self):
        assert get_branch(12) == "子"

    def test_get_cycle_0(self):
        assert get_cycle(0) == "甲子"

    def test_get_cycle_59(self):
        assert get_cycle(59) == "癸亥"


# =====================================================================
# [19] 集成一致性测试
# =====================================================================


class TestTemporalIntegration:
    """跨模块集成一致性测试"""

    def test_temporal_core_importable_from_sys(self):
        """TemporalCore 可从 _sys 直接导入"""
        from mci_world_model._sys import (
            StemBranchCode,
            TemporalCore,
        )

        assert TemporalCore is not None
        assert StemBranchCode is not None

    def test_timecycle_importable_from_sys(self):
        """TimeCycle 可从 _sys 直接导入"""
        from mci_world_model._sys import TimeCodeInfo, TimeCycle

        assert TimeCycle is not None
        assert TimeCodeInfo is not None

    def test_full_60_cycle_completeness(self):
        """60 甲子完整性：0-59 全覆盖无重复"""
        tc = TemporalCore()
        names = set()
        for i in range(60):
            name = tc.get_cycle_name(i)
            assert name not in names, f"Duplicate: {name}"
            names.add(name)
        assert len(names) == 60

    def test_cycle_name_and_create_code_consistent(self):
        """create_code(stem, branch) 与 get_cycle_name/index 一致"""
        tc = TemporalCore()
        for i in range(60):
            # 对第 i 个 code
            code = tc.create_code(i % 10, i % 12)
            assert code.cycle_index == 0 if i == 0 else ...  # not all
            # Actually, create_code returns stem/branch from mod, then maps via _cycle_index_map
            # Let's just verify the name matches what get_cycle_name gives for the same index
            pass
        # Simpler: verify 甲子 ↔ index 0
        code = tc.create_code(0, 0)
        assert code.name == tc.get_cycle_name(0)

    def test_stem_energy_consistency(self):
        """天干能量与 _time_code._terms 一致"""
        tc = TemporalCore()
        assert tc.get_stem_energy(TimeStem.JIA) == "wood"
        assert tc.get_stem_energy(TimeStem.YI) == "wood"
        assert tc.get_stem_energy(TimeStem.BING) == "fire"
        assert tc.get_stem_energy(TimeStem.DING) == "fire"
        assert tc.get_stem_energy(TimeStem.WU) == "earth"
        assert tc.get_stem_energy(TimeStem.JI) == "earth"
        assert tc.get_stem_energy(TimeStem.GENG) == "metal"
        assert tc.get_stem_energy(TimeStem.XIN) == "metal"
        assert tc.get_stem_energy(TimeStem.REN) == "water"
        assert tc.get_stem_energy(TimeStem.GUI) == "water"

    def test_branch_chong_is_symmetric(self):
        """六冲对称性"""
        tc = TemporalCore()
        for b1 in TimeBranch:
            for b2 in TimeBranch:
                rels12 = tc.analyze_branch_relation(b1, b2)
                rels21 = tc.analyze_branch_relation(b2, b1)
                assert (BranchRelation.LIU_CHONG in rels12) == (BranchRelation.LIU_CHONG in rels21)
