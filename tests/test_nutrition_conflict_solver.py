"""专利B 多疾病并存营养冲突求解 —— S700 安全校验闭环测试。

覆盖说明书实施例 1-6 数值复现、S700 容差边界、钳制边界与异常输入。
"""

from __future__ import annotations

import pytest

from mci_world_model.nutrition import (
    DiseaseConstraint,
    PairRule,
    SafetyConfig,
    SolvedState,
    solve_nutrition_plan,
)


def _rule(
    name: str,
    pmin: float,
    pmax: float,
    stress: float = 1.0,
    fluid_rule: str = "normal",
    fluid_factor: float = 30.0,
    priority: int = 1,
    specialist: bool = False,
) -> DiseaseConstraint:
    """构造单关键词测试约束。"""
    return DiseaseConstraint(
        name=name,
        keywords=(name,),
        protein_min=pmin,
        protein_max=pmax,
        stress=stress,
        fluid_rule=fluid_rule,
        fluid_factor=fluid_factor,
        priority=priority,
        specialist=specialist,
    )


class TestEmbodiments:
    """说明书实施例 1-6 数值复现。"""

    def test_example1_tumor_ckd(self) -> None:
        """胃癌+慢性肾病3期：区间无交集，取最严格上限 0.8，S700 蛋白校验通过。"""
        plan = solve_nutrition_plan("胃癌合并慢性肾病 3 期", 60.0)
        assert plan.matched
        assert plan.protein_coef == pytest.approx(0.8)
        assert plan.protein_check_pass is True
        assert plan.protein_target_g == pytest.approx(48.0)
        assert any("蛋白质冲突" in w for w in plan.warnings)

    def test_example2_dialysis_cover(self) -> None:
        """透析覆盖非透析肾病：系数取透析区间中点 1.1。"""
        plan = solve_nutrition_plan("慢性肾衰竭，行血液透析治疗", 60.0)
        assert plan.dialysis_covered
        assert plan.protein_coef == pytest.approx(1.1)
        assert any("透析覆盖" in w for w in plan.warnings)

    def test_example3_liver_diabetes(self) -> None:
        """肝硬化+糖尿病：区间有交集，取中点 1.0。"""
        plan = solve_nutrition_plan("肝硬化，2 型糖尿病", 60.0)
        assert plan.protein_coef == pytest.approx(1.0)
        assert plan.protein_check_pass is True

    def test_example4_hf_burn(self) -> None:
        """心衰+烧伤：蛋白冲突取 1.2，液体整体 restrict、因子取 min(25,40)=25。"""
        plan = solve_nutrition_plan("心力衰竭，烧伤 30%", 70.0)
        assert plan.protein_coef == pytest.approx(1.2)
        assert plan.fluid_check_pass is True
        assert plan.liquid_target_ml == pytest.approx(1750.0)

    def test_example5_ffm_ascites(self) -> None:
        """肝硬化+腹水：FFM 级联修正 45×0.85=38.25，蛋白目标 38.25×1.1≈42.1。"""
        plan = solve_nutrition_plan("肝硬化", 70.0, ffm_kg=45.0, ffm_correction=0.85)
        assert plan.protein_coef == pytest.approx(1.1)
        assert plan.protein_target_g == pytest.approx(42.075, abs=1e-3)

    def test_example6_parkland_fail_rollback(self) -> None:
        """心衰+烧伤急性期：Parkland 8400 ml/d > 限液约束 2100 ml/d，校验失败并回退。"""
        plan = solve_nutrition_plan("心力衰竭，烧伤 30%", 70.0, burn_area_pct=30)
        assert plan.final_fluid_volume_ml == pytest.approx(8400.0)
        assert plan.fluid_check_pass is False
        assert plan.safety_pass is False
        assert plan.fallback is True
        assert any("液体冲突" in w for w in plan.warnings)
        assert any("回退" in w for w in plan.warnings)


class TestSafetyBoundaries:
    """S700 容差边界与钳制边界。"""

    def test_protein_tolerance_boundary_pass(self) -> None:
        """蛋白质热量恰为上限热量×1.15 判通过：60kg×1.15g/kg×4=276 kcal ≤ 1.15×60×4×1.15=317.4 kcal。"""
        rules = (_rule("甲病", 1.15, 1.15),)
        plan = solve_nutrition_plan("甲病", 60.0, disease_rules=rules)
        assert plan.protein_coef == pytest.approx(1.15)
        assert plan.protein_check_pass is True

    def test_protein_rule_override_fail(self) -> None:
        """S500 规则将最终蛋白系数上调时，热量校验（480 kcal > 1.2×60×4×1.15=331.2 kcal）失败并回退。"""

        def override_coef(state: SolvedState) -> None:
            state.protein_coef = 2.0  # 模拟高蛋白策略上调，超过 1.2×1.15=1.38

        rules = (_rule("甲病", 1.0, 1.2),)
        pairs = (PairRule(("甲病",), "TEST", "测试规则", override_coef),)
        plan = solve_nutrition_plan("甲病", 60.0, disease_rules=rules, pair_rules=pairs)
        assert plan.protein_check_pass is False
        assert plan.fallback is True

    def test_ffm_heat_check_matches_final_output(self) -> None:
        """FFM 修正场景：系数口径判失败（1.4 > 1.38），热量口径校验最终输出仍通过
        （50kg×0.85×1.4×4=238 kcal ≤ 1.2×70×4×1.15=386.4 kcal），体现"校验上一步输出"语义。"""

        def override_coef(state: SolvedState) -> None:
            state.protein_coef = 1.4

        rules = (_rule("甲病", 1.0, 1.2),)
        pairs = (PairRule(("甲病",), "T-FFM", "测试规则", override_coef),)
        plan = solve_nutrition_plan(
            "甲病", 70.0, disease_rules=rules, pair_rules=pairs, ffm_kg=50.0, ffm_correction=0.85
        )
        assert plan.protein_target_g == pytest.approx(59.5)
        assert plan.protein_kcal == pytest.approx(238.0)
        assert plan.protein_check_pass is True

    def test_fluid_tolerance_boundary_pass(self) -> None:
        """限液模式下液体量恰为 限液因子×体重+5×体重 判通过。"""

        def set_at_limit(state: SolvedState) -> None:
            state.final_fluid_volume_ml = 25.0 * state.weight_kg + 5.0 * state.weight_kg

        rules = (_rule("心衰", 1.0, 1.2, fluid_rule="restrict", fluid_factor=25.0),)
        pairs = (PairRule(("心衰",), "T-LIMIT", "测试规则", set_at_limit),)
        plan = solve_nutrition_plan("心衰", 70.0, disease_rules=rules, pair_rules=pairs)
        assert plan.fluid_check_pass is True

    def test_fluid_tolerance_exceed_fail(self) -> None:
        """限液模式下液体量超过 限液因子×体重+5×体重 判失败并回退。"""

        def set_over_limit(state: SolvedState) -> None:
            state.final_fluid_volume_ml = 25.0 * state.weight_kg + 5.0 * state.weight_kg + 1.0

        rules = (_rule("心衰", 1.0, 1.2, fluid_rule="restrict", fluid_factor=25.0),)
        pairs = (PairRule(("心衰",), "T-OVER", "测试规则", set_over_limit),)
        plan = solve_nutrition_plan("心衰", 70.0, disease_rules=rules, pair_rules=pairs)
        assert plan.fluid_check_pass is False
        assert plan.fallback is True

    def test_liquid_clamp_upper(self) -> None:
        """液体目标钳制上限 4000 ml/d。"""
        plan = solve_nutrition_plan("甲病", 150.0, disease_rules=(_rule("甲病", 1.0, 1.0),))
        assert plan.liquid_target_ml == pytest.approx(4000.0)

    def test_energy_clamp_upper(self) -> None:
        """能量目标钳制上限 4000 kcal/d。"""
        rules = (_rule("甲病", 1.0, 1.0, stress=1.4),)
        plan = solve_nutrition_plan("甲病", 200.0, disease_rules=rules)
        assert plan.energy_target_kcal == pytest.approx(4000.0)

    def test_macro_nutrient_split(self) -> None:
        """三大营养素分配：蛋白热量=蛋白目标×4，脂肪=能量×25%，碳水=能量-蛋白-脂肪。"""
        rules = (_rule("甲病", 1.0, 1.0, stress=1.0),)
        plan = solve_nutrition_plan("甲病", 60.0, disease_rules=rules)
        assert plan.protein_kcal == pytest.approx(240.0)
        assert plan.fat_kcal == pytest.approx(1500.0 * 0.25)
        assert plan.carb_kcal == pytest.approx(1500.0 - 240.0 - 1500.0 * 0.25)


class TestExceptions:
    """异常与空输入。"""

    def test_empty_diagnosis(self) -> None:
        plan = solve_nutrition_plan("", 60.0)
        assert plan.matched is False
        assert any("未匹配" in w for w in plan.warnings)

    def test_no_match(self) -> None:
        plan = solve_nutrition_plan("完全不相关的文本", 60.0)
        assert plan.matched is False

    def test_invalid_weight(self) -> None:
        with pytest.raises(ValueError, match="体重"):
            solve_nutrition_plan("甲病", 0.0, disease_rules=(_rule("甲病", 1.0, 1.0),))

    def test_invalid_ffm(self) -> None:
        with pytest.raises(ValueError, match="FFM"):
            solve_nutrition_plan("甲病", 60.0, ffm_kg=-1.0, disease_rules=(_rule("甲病", 1.0, 1.0),))

    def test_custom_config_injection(self) -> None:
        """配置注入：调整蛋白容差后边界判定随之改变。"""
        cfg = SafetyConfig(protein_tolerance=0.1)
        rules = (_rule("甲病", 1.0, 1.0),)
        plan = solve_nutrition_plan("甲病", 60.0, disease_rules=rules, config=cfg)
        assert plan.protein_check_pass is True
