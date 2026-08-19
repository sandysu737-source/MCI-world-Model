"""多疾病并存营养冲突求解 —— 专利B 可复现示例实现。

对应专利：《基于区间交集代数的多疾病并存营养冲突求解方法及系统》。
本模块按说明书步骤 S100-S800 实现核心流程，重点复现 S700 安全校验闭环：

- 蛋白质：最终输出的蛋白质热量(kcal)（蛋白质目标×4 kcal/g，含 FFM 修正）
  不得超过最严格蛋白质上限(g/kg) × 体重(kg) × 4 kcal/g × (1 + 15%)；
- 液体：疾病要求限液时，最终输出的液体量(ml/d) 不得超过
  限液液体因子 × 体重 + 5 ml/kg × 体重；
- 任一校验失败 → 标记不通过并回退规则引擎。

说明：内置规则库数值取自说明书实施例 1-6；实施例未给出处（如 NRS 评分、
部分液体因子）为示例值，非临床指南。生产使用须替换为经临床审核的配置。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = [
    "DiseaseConstraint",
    "PairRule",
    "SafetyConfig",
    "SolvedState",
    "NutritionPlan",
    "solve_nutrition_plan",
    "DEFAULT_DISEASE_RULES",
    "DEFAULT_PAIR_RULES",
]

_SPLIT_PATTERN = re.compile(r"[，,；;。\n]|合并|伴|伴有|及|并发")


@dataclass(frozen=True)
class DiseaseConstraint:
    """疾病-营养约束（说明书 S200 词典条目）。"""

    name: str
    keywords: tuple[str, ...]
    protein_min: float
    protein_max: float
    stress: float
    fluid_rule: str  # normal / restrict / liberal
    fluid_factor: float  # ml/kg
    priority: int
    specialist: bool = False  # 是否为专科约束（触发专科覆盖通用过滤）
    score: float = 1.0  # NRS-2002 严重度评分（实施例未给出，示例值）


@dataclass(frozen=True)
class PairRule:
    """循证疾病对规则（说明书 S500），回调可改写求解中间状态。"""

    pair: tuple[str, ...]
    rule_id: str
    evidence: str
    apply: Callable[[SolvedState], None]


@dataclass(frozen=True)
class SafetyConfig:
    """S700 安全校验与目标计算参数（配置与代码分离）。"""

    protein_tolerance: float = 0.15
    fluid_tolerance_ml_per_kg: float = 5.0
    liquid_min_ml: float = 1000.0
    liquid_max_ml: float = 4000.0
    energy_min_kcal: float = 800.0
    energy_max_kcal: float = 4000.0
    default_fluid_factor: float = 30.0
    bmr_kcal_per_kg: float = 25.0
    protein_kcal_per_g: float = 4.0
    fat_ratio: float = 0.25
    protein_target_floor_per_kg: float = 0.5


@dataclass
class SolvedState:
    """S400-S700 各步共享的求解中间状态。"""

    weight_kg: float
    burn_area_pct: int | None = None
    activity_factor: float = 1.0
    ffm_kg: float | None = None
    ffm_correction: float = 1.0
    stress: float = 1.0
    protein_coef: float | None = None
    protein_lower: float | None = None
    protein_upper: float | None = None
    fluid_rule: str = "normal"
    fluid_factor: float = 30.0
    energy_target_kcal: float | None = None
    protein_target_g: float | None = None
    liquid_target_ml: float | None = None
    final_fluid_volume_ml: float | None = None
    protein_kcal: float | None = None
    fat_kcal: float | None = None
    carb_kcal: float | None = None
    protein_check_pass: bool | None = None
    fluid_check_pass: bool | None = None
    protein_conflict: bool = False
    dialysis_covered: bool = False
    pair_rules_hit: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class NutritionPlan:
    """S800 输出的结构化营养方案。"""

    matched: bool = False
    diseases: list[str] = field(default_factory=list)
    protein_coef: float | None = None
    protein_target_g: float | None = None
    energy_target_kcal: float | None = None
    liquid_target_ml: float | None = None
    protein_kcal: float | None = None
    fat_kcal: float | None = None
    carb_kcal: float | None = None
    final_fluid_volume_ml: float | None = None
    protein_check_pass: bool | None = None
    fluid_check_pass: bool | None = None
    safety_pass: bool = False
    fallback: bool = False
    dialysis_covered: bool = False
    warnings: list[str] = field(default_factory=list)


def _default_rules() -> tuple[DiseaseConstraint, ...]:
    """示例约束词典（数值取自说明书实施例 1-6）。"""
    return (
        DiseaseConstraint("恶性肿瘤", ("肿瘤", "癌"), 1.0, 1.5, 1.2, "normal", 30.0, 3, True, 3.0),
        DiseaseConstraint("慢性肾病", ("肾病", "肾衰"), 0.6, 0.8, 1.05, "normal", 30.0, 1, True, 2.0),
        DiseaseConstraint("透析", ("透析",), 1.0, 1.2, 1.0, "normal", 30.0, 1, True, 2.0),
        DiseaseConstraint("肝硬化", ("肝硬",), 1.0, 1.2, 1.0, "normal", 30.0, 2, True, 2.0),
        DiseaseConstraint("糖尿病", ("糖尿病",), 0.8, 1.0, 1.0, "normal", 30.0, 4, False, 1.0),
        DiseaseConstraint("心力衰竭", ("心衰", "心力衰竭"), 1.0, 1.2, 1.15, "restrict", 25.0, 3, True, 3.0),
        DiseaseConstraint("烧伤", ("烧伤",), 1.5, 2.0, 1.4, "liberal", 40.0, 10, False, 3.0),
    )


def _parkland_fluid(state: SolvedState) -> None:
    """CM003：心衰+烧伤急性期按 Parkland 公式补液，改写最终输出液体量。"""
    if state.burn_area_pct is not None:
        state.final_fluid_volume_ml = 4.0 * state.weight_kg * float(state.burn_area_pct)
        state.warnings.append(
            "疾病对规则 CM003（心衰+烧伤，证据等级 B）：急性期按 Parkland 公式补液，"
            f"24h 补液量 = 4 × {state.weight_kg:g} × {state.burn_area_pct} "
            f"= {state.final_fluid_volume_ml:g} ml"
        )


def _keep_protein(state: SolvedState) -> None:
    """CM001：恶性肿瘤+慢性肾病保持最严格蛋白上限并代偿。"""
    state.warnings.append(
        "疾病对规则 CM001（恶性肿瘤+慢性肾病，证据等级 A）：保持 0.8 g/kg 保护肾功能，推荐 α-酮酸 + EPA 代偿"
    )


def _default_pair_rules() -> tuple[PairRule, ...]:
    """示例循证疾病对规则库（S500）。"""
    return (
        PairRule(("恶性肿瘤", "慢性肾病"), "CM001", "证据等级 A", _keep_protein),
        PairRule(("心力衰竭", "烧伤"), "CM003", "证据等级 B", _parkland_fluid),
    )


DEFAULT_DISEASE_RULES: tuple[DiseaseConstraint, ...] = _default_rules()
DEFAULT_PAIR_RULES: tuple[PairRule, ...] = _default_pair_rules()


def parse_diagnosis(text: str) -> list[str]:
    """S100：按分隔符与中文医学连接词切分诊断文本。"""
    entries: list[str] = []
    for part in _SPLIT_PATTERN.split(text):
        item = part.strip()
        if len(item) >= 2:
            entries.append(item)
    return entries


def match_constraints(entries: list[str], rules: tuple[DiseaseConstraint, ...]) -> list[DiseaseConstraint]:
    """S200：子串匹配 + 同标准名去重（保留优先级最小）+ 专科覆盖通用。"""
    by_name: dict[str, DiseaseConstraint] = {}
    for entry in entries:
        for rule in rules:
            if any(keyword in entry for keyword in rule.keywords):
                prev = by_name.get(rule.name)
                if prev is None or rule.priority < prev.priority:
                    by_name[rule.name] = rule
    matched = list(by_name.values())
    has_specialist = any(r.specialist and r.priority <= 2 for r in matched)
    if has_specialist:
        matched = [r for r in matched if not (r.priority >= 5 and not r.specialist)]
    return matched


def apply_dialysis_cover(matched: list[DiseaseConstraint], text: str) -> tuple[list[DiseaseConstraint], bool]:
    """S300：透析治疗状态覆盖非透析性肾病约束。"""
    if "透析" in text:
        kept = [r for r in matched if r.name != "慢性肾病"]
        if len(kept) != len(matched):
            return kept, True
    return matched, False


def solve_protein_interval(
    matched: list[DiseaseConstraint],
) -> tuple[float, float, float, bool]:
    """S430：蛋白质区间交集求解，返回 (系数, lower, upper, 冲突标记)。"""
    lower = max(r.protein_min for r in matched)
    upper = min(r.protein_max for r in matched)
    if lower <= upper:
        return (lower + upper) / 2.0, lower, upper, False
    return upper, lower, upper, True


def solve_fluid_strategy(matched: list[DiseaseConstraint], config: SafetyConfig) -> tuple[str, float]:
    """S440：液体策略求解，返回 (整体策略, 液体因子)。"""
    restrict = [r.fluid_factor for r in matched if r.fluid_rule == "restrict"]
    if restrict:
        return "restrict", min(restrict)
    liberal = [r.fluid_factor for r in matched if r.fluid_rule == "liberal"]
    if liberal:
        return "liberal", max(liberal)
    return "normal", config.default_fluid_factor


def apply_pair_rules(matched: list[DiseaseConstraint], state: SolvedState, pair_rules: tuple[PairRule, ...]) -> None:
    """S500：循证疾病对规则叠加，回调改写中间状态。"""
    names = {r.name for r in matched}
    for rule in pair_rules:
        if all(name in names for name in rule.pair):
            state.pair_rules_hit.append(rule.rule_id)
            rule.apply(state)


def compute_targets(state: SolvedState, matched: list[DiseaseConstraint], config: SafetyConfig) -> None:
    """S600：能量/蛋白质/液体目标与三大营养素分配。"""
    assert state.protein_coef is not None
    stress = max(r.stress for r in matched)
    state.stress = stress
    energy = config.bmr_kcal_per_kg * state.weight_kg * stress * state.activity_factor
    state.energy_target_kcal = min(max(energy, config.energy_min_kcal), config.energy_max_kcal)

    basis_kg = state.ffm_kg if state.ffm_kg is not None else state.weight_kg
    state.protein_target_g = max(
        basis_kg * state.ffm_correction * state.protein_coef,
        state.weight_kg * config.protein_target_floor_per_kg,
    )

    liquid = state.weight_kg * state.fluid_factor
    state.liquid_target_ml = min(max(liquid, config.liquid_min_ml), config.liquid_max_ml)
    if state.final_fluid_volume_ml is None:
        state.final_fluid_volume_ml = state.liquid_target_ml

    state.protein_kcal = state.protein_target_g * config.protein_kcal_per_g
    state.fat_kcal = state.energy_target_kcal * config.fat_ratio
    state.carb_kcal = state.energy_target_kcal - state.protein_kcal - state.fat_kcal


def safety_check(state: SolvedState, matched: list[DiseaseConstraint], config: SafetyConfig) -> None:
    """S700：安全校验闭环。校验对象为上一步 S620 输出的最终值（热量/液体量）。"""
    assert state.protein_kcal is not None
    protein_limit_kcal = min(r.protein_max for r in matched) * state.weight_kg * config.protein_kcal_per_g
    state.protein_check_pass = state.protein_kcal <= protein_limit_kcal * (1.0 + config.protein_tolerance)

    if state.fluid_rule == "restrict":
        assert state.final_fluid_volume_ml is not None
        fluid_limit = state.fluid_factor * state.weight_kg + config.fluid_tolerance_ml_per_kg * state.weight_kg
        state.fluid_check_pass = state.final_fluid_volume_ml <= fluid_limit
        if not state.fluid_check_pass:
            state.warnings.append(
                f"液体冲突：最终液体量 {state.final_fluid_volume_ml:g} ml/d "
                f"> 限液约束 {state.fluid_factor:g} × {state.weight_kg:g} + "
                f"{config.fluid_tolerance_ml_per_kg:g} × {state.weight_kg:g} "
                f"= {fluid_limit:g} ml/d，校验不通过"
            )
    else:
        state.fluid_check_pass = True

    if not state.protein_check_pass:
        state.warnings.append(
            f"蛋白质热量校验不通过：输出热量 {state.protein_kcal:g} kcal > 最严格蛋白上限对应热量 "
            f"{protein_limit_kcal:g} × (1+15%) = "
            f"{protein_limit_kcal * (1.0 + config.protein_tolerance):g} kcal"
        )


def build_warnings(state: SolvedState, matched: list[DiseaseConstraint], config: SafetyConfig) -> list[str]:
    """S800：冲突预警列表组装。"""
    warnings = list(state.warnings)
    if state.protein_conflict:
        warnings.append(f"蛋白质冲突：疾病蛋白质区间无交集，取最严格上限 {state.protein_coef:g} g/kg（安全优先）")
    if state.dialysis_covered:
        warnings.append("透析覆盖：KDOQI 2020 血液透析 1.0-1.2 g/kg，高于非透析 CKD 的 0.6-0.8")
    if len(matched) >= 3:
        warnings.append("多病复杂度预警：并存疾病数 ≥ 3")
    if not (state.protein_check_pass and state.fluid_check_pass):
        warnings.append("安全校验不通过，触发回退至规则引擎")
    return warnings


def solve_nutrition_plan(
    diagnosis_text: str,
    weight_kg: float,
    *,
    ffm_kg: float | None = None,
    ffm_correction: float = 1.0,
    burn_area_pct: int | None = None,
    activity_factor: float = 1.0,
    disease_rules: tuple[DiseaseConstraint, ...] | None = None,
    pair_rules: tuple[PairRule, ...] | None = None,
    config: SafetyConfig | None = None,
) -> NutritionPlan:
    """S100-S800 全流程入口。"""
    cfg = config if config is not None else SafetyConfig()
    rules = disease_rules if disease_rules is not None else DEFAULT_DISEASE_RULES
    pair_rule_set = pair_rules if pair_rules is not None else DEFAULT_PAIR_RULES

    if weight_kg <= 0:
        raise ValueError("体重必须大于 0")
    if ffm_kg is not None and ffm_kg <= 0:
        raise ValueError("去脂体重 FFM 必须大于 0")

    entries = parse_diagnosis(diagnosis_text)
    matched = match_constraints(entries, rules)
    if not matched:
        return NutritionPlan(matched=False, warnings=["未匹配到任何疾病约束"])

    matched, dialysis_covered = apply_dialysis_cover(matched, diagnosis_text)
    state = SolvedState(
        weight_kg=weight_kg,
        burn_area_pct=burn_area_pct,
        activity_factor=activity_factor,
        ffm_kg=ffm_kg,
        ffm_correction=ffm_correction,
        dialysis_covered=dialysis_covered,
    )

    if not matched:
        return NutritionPlan(matched=False, warnings=["透析覆盖后无剩余疾病约束"])

    state.protein_coef, state.protein_lower, state.protein_upper, state.protein_conflict = solve_protein_interval(
        matched
    )
    state.fluid_rule, state.fluid_factor = solve_fluid_strategy(matched, cfg)
    apply_pair_rules(matched, state, pair_rule_set)
    compute_targets(state, matched, cfg)
    safety_check(state, matched, cfg)

    plan = NutritionPlan(
        matched=True,
        diseases=[r.name for r in matched],
        protein_coef=state.protein_coef,
        protein_target_g=state.protein_target_g,
        energy_target_kcal=state.energy_target_kcal,
        liquid_target_ml=state.liquid_target_ml,
        protein_kcal=state.protein_kcal,
        fat_kcal=state.fat_kcal,
        carb_kcal=state.carb_kcal,
        final_fluid_volume_ml=state.final_fluid_volume_ml,
        protein_check_pass=state.protein_check_pass,
        fluid_check_pass=state.fluid_check_pass,
        safety_pass=bool(state.protein_check_pass and state.fluid_check_pass),
        fallback=not (state.protein_check_pass and state.fluid_check_pass),
        dialysis_covered=state.dialysis_covered,
        warnings=build_warnings(state, matched, cfg),
    )
    return plan
