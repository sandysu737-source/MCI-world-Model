"""专利B 多疾病并存营养冲突求解 —— 可复现示例实现。"""

from mci_world_model.nutrition.conflict_solver import (
    DEFAULT_DISEASE_RULES,
    DEFAULT_PAIR_RULES,
    DiseaseConstraint,
    NutritionPlan,
    PairRule,
    SafetyConfig,
    SolvedState,
    solve_nutrition_plan,
)

__all__ = [
    "DEFAULT_DISEASE_RULES",
    "DEFAULT_PAIR_RULES",
    "DiseaseConstraint",
    "NutritionPlan",
    "PairRule",
    "SafetyConfig",
    "SolvedState",
    "solve_nutrition_plan",
]
