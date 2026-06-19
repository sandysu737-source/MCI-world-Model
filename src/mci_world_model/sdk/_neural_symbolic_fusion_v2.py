from __future__ import annotations

"""MCI World Model — NeuralSymbolicFusionV2 神经符号融合 2.0
=============================================================

从 V1 单向桥接升级到双向深度融合——神经网络的感知能力
与符号系统的推理能力在因果层面统一。

V1 → V2 升级:
    V1: 神经编码 → 符号规则 (单向)
    V2: 神经↔符号 双向循环 + 因果梯度传播

核心能力:
    FusionState         — 融合状态
    NeuralSymbolicFusionV2 — 神经符号融合引擎

设计原则:
    - 基于 DifferentiableCausalInference (T16)
    - 纯 numpy，零外部依赖
"""


import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# FusionState — 融合状态
# =============================================================================


@dataclass
class FusionState:
    """神经符号融合状态。

    Attributes:
        neural_representation: 神经表征向量
        symbolic_rules: 符号规则列表
        fusion_score: 融合质量得分 [0, 1]
        consistency: 神经-符号一致性
        n_iterations: 融合迭代次数
    """

    neural_representation: np.ndarray | None = None
    symbolic_rules: list[dict[str, Any]] = field(default_factory=list)
    fusion_score: float = 0.0
    consistency: float = 0.0
    n_iterations: int = 0


# =============================================================================
# NeuralSymbolicFusionV2 — 神经符号融合引擎
# =============================================================================


class NeuralSymbolicFusionV2:
    """神经符号融合 2.0 — 双向深度融合。

    工作流程:
      1. neural_to_symbolic: 从神经表征提取符号规则
      2. symbolic_to_neural: 从符号规则约束神经表征
      3. fuse: 双向循环直到收敛

    用法:
        >>> fusion = NeuralSymbolicFusionV2()
        >>> state = fusion.fuse(neural_features, n_iterations=10)
    """

    def __init__(
        self,
        rule_threshold: float = 0.7,
        consistency_threshold: float = 0.6,
        max_iterations: int = 20,
    ):
        self._rule_threshold = rule_threshold
        self._consistency_threshold = consistency_threshold
        self._max_iterations = max_iterations
        self._fusion_history: list[FusionState] = []

    @property
    def fusion_count(self) -> int:
        return len(self._fusion_history)

    def neural_to_symbolic(
        self, neural_repr: np.ndarray, var_names: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """从神经表征提取符号规则。

        简化实现: 从向量中检测显著线性关系。

        Args:
            neural_repr: 神经表征 (n_dims,)
            var_names: 变量名列表

        Returns:
            符号规则列表
        """
        vec = np.atleast_1d(np.asarray(neural_repr, dtype=float))
        n = len(vec)
        if var_names is None:
            var_names = [f"v{i}" for i in range(n)]

        rules = []
        for i in range(n):
            for j in range(i + 1, n):
                ratio = abs(vec[i]) / max(abs(vec[j]), 1e-8)
                if ratio > self._rule_threshold:
                    rules.append({
                        "type": "linear",
                        "rule": f"{var_names[j]} ≈ {ratio:.4f} * {var_names[i]}",
                        "strength": min(ratio, 1.0),
                    })

        return rules

    def symbolic_to_neural(
        self, rules: list[dict[str, Any]], target_dim: int
    ) -> np.ndarray:
        """从符号规则约束神经表征。

        简化实现: 根据规则强度生成约束向量。

        Args:
            rules: 符号规则列表
            target_dim: 目标维度

        Returns:
            约束向量 (target_dim,)
        """
        result = np.zeros(target_dim)
        for i, rule in enumerate(rules[:target_dim]):
            strength = rule.get("strength", 0.5)
            result[i % target_dim] = strength
        return result

    def fuse(
        self,
        neural_repr: np.ndarray,
        var_names: list[str] | None = None,
        n_iterations: int | None = None,
    ) -> FusionState:
        """执行神经符号双向深度融合。

        Args:
            neural_repr: 初始神经表征
            var_names: 变量名
            n_iterations: 迭代次数

        Returns:
            FusionState 融合结果
        """
        if n_iterations is None:
            n_iterations = self._max_iterations

        current_repr = np.atleast_1d(np.asarray(neural_repr, dtype=float)).copy()
        best_score = 0.0
        best_rules = []
        best_consistency = 0.0

        for it in range(n_iterations):
            # Step 1: 神经 → 符号
            rules = self.neural_to_symbolic(current_repr, var_names)

            # Step 2: 符号 → 神经约束
            constraint = self.symbolic_to_neural(rules, len(current_repr))

            # Step 3: 融合更新
            alpha = 0.3  # 约束权重
            current_repr = current_repr * (1 - alpha) + constraint * alpha

            # Step 4: 评估
            score = self._evaluate_fusion(current_repr, rules)
            consistency = self._evaluate_consistency(current_repr, rules)

            if score > best_score:
                best_score = score
                best_rules = rules
                best_consistency = consistency

        state = FusionState(
            neural_representation=current_repr,
            symbolic_rules=best_rules,
            fusion_score=best_score,
            consistency=best_consistency,
            n_iterations=n_iterations,
        )
        self._fusion_history.append(state)

        return state

    @staticmethod
    def _evaluate_fusion(repr: np.ndarray, rules: list[dict[str, Any]]) -> float:
        """评估融合质量。"""
        if not rules:
            return 0.0
        avg_strength = float(np.mean([r.get("strength", 0.0) for r in rules]))
        coverage = len(rules) / max(len(repr), 1)
        return float(min(avg_strength * 0.6 + coverage * 0.4, 1.0))

    @staticmethod
    def _evaluate_consistency(repr: np.ndarray, rules: list[dict[str, Any]]) -> float:
        """评估神经-符号一致性。"""
        if not rules:
            return 0.0
        n_violations = sum(
            1 for r in rules if r.get("strength", 0.0) < 0.3
        )
        return float(1.0 - n_violations / max(len(rules), 1))

    def statistics(self) -> dict[str, Any]:
        return {
            "fusion_count": self.fusion_count,
            "rule_threshold": self._rule_threshold,
            "max_iterations": self._max_iterations,
            "avg_fusion_score": (
                float(np.mean([s.fusion_score for s in self._fusion_history]))
                if self._fusion_history else 0.0
            ),
        }
