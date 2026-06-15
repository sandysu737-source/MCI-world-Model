"""MCI World Model — CausalImaginationEngine 因果想象力引擎
=========================================================

在因果模型上进行反事实想象——给定假设干预，模拟出可能的
替代世界状态，支持"如果...会怎样"的因果推理。

核心能力:
    ImaginedWorld        — 想象世界状态
    CausalImaginationEngine — 因果想象力引擎

设计原则:
    - 基于 counterfactual 推理框架
    - 依赖 UnifiedModalEncoder 提供多模态表征
    - 纯 numpy，零外部依赖
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# ImaginedWorld — 想象世界状态
# =============================================================================


@dataclass
class ImaginedWorld:
    """想象世界状态 — 反事实推理后的替代世界。

    Attributes:
        intervention: 干预变量和值
        original_state: 原始状态
        imagined_state: 想象状态
        difference: 差异度量
        plausibility: 可信度 [0, 1]
        narrative: 叙述描述
    """

    intervention: dict[str, Any] = field(default_factory=dict)
    original_state: np.ndarray | None = None
    imagined_state: np.ndarray | None = None
    difference: float = 0.0
    plausibility: float = 0.0
    narrative: str = ""


# =============================================================================
# CausalImaginationEngine — 因果想象力引擎
# =============================================================================


class CausalImaginationEngine:
    """因果想象力引擎 — 反事实推理 + 替代世界模拟。

    用法:
        >>> engine = CausalImaginationEngine()
        >>> engine.set_current_state(state_vector)
        >>> world = engine.imagine(intervention={"X": 0.5})
        >>> print(world.narrative)
    """

    def __init__(
        self,
        state_dim: int = 10,
        n_imagination_steps: int = 5,
        plausibility_threshold: float = 0.3,
    ):
        if state_dim < 1:
            raise ValueError("state_dim 必须 ≥ 1")
        self._state_dim = state_dim
        self._n_steps = n_imagination_steps
        self._plausibility_threshold = plausibility_threshold
        self._current_state: np.ndarray | None = None
        self._causal_matrix: np.ndarray | None = None
        self._imagination_history: list[ImaginedWorld] = []

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def imagination_count(self) -> int:
        return len(self._imagination_history)

    def set_current_state(self, state: np.ndarray) -> None:
        """设置当前世界状态。"""
        self._current_state = np.atleast_1d(np.asarray(state, dtype=float))

    def set_causal_matrix(self, matrix: np.ndarray) -> None:
        """设置因果结构矩阵 (有向邻接矩阵)。

        Args:
            matrix: (state_dim, state_dim) 因果影响矩阵
        """
        mat = np.atleast_2d(np.asarray(matrix, dtype=float))
        self._causal_matrix = mat

    def imagine(
        self,
        intervention: dict[str, Any],
        n_worlds: int = 1,
    ) -> list[ImaginedWorld]:
        """执行因果想象力推理。

        Args:
            intervention: 干预变量 {var_name: value}
            n_worlds: 想象世界数量

        Returns:
            ImaginedWorld 列表
        """
        if self._current_state is None:
            logger.warning("因果想象: 未设置当前状态, 使用零向量")
            self._current_state = np.zeros(self._state_dim)

        results = []
        for _ in range(n_worlds):
            world = self._imagine_single(intervention)
            results.append(world)
            self._imagination_history.append(world)

        return results

    def _imagine_single(self, intervention: dict[str, Any]) -> ImaginedWorld:
        """单次因果想象。"""
        original = self._current_state.copy()
        imagined = original.copy()

        # 应用干预
        for var_key, var_val in intervention.items():
            if isinstance(var_key, int) and 0 <= var_key < len(imagined):
                imagined[var_key] = float(var_val)
            elif isinstance(var_key, str):
                try:
                    idx = int(var_key)
                    if 0 <= idx < len(imagined):
                        imagined[idx] = float(var_val)
                except ValueError:
                    pass

        # 因果传播: 使用因果矩阵传播干预影响
        if self._causal_matrix is not None:
            for step in range(self._n_steps):
                imagined = self._propagate(imagined, original, intervention)

        # 计算差异
        difference = float(np.linalg.norm(imagined - original))

        # 可信度评估
        plausibility = self._assess_plausibility(original, imagined, difference)

        # 生成叙述
        narrative = self._generate_narrative(intervention, original, imagined)

        return ImaginedWorld(
            intervention=intervention,
            original_state=original,
            imagined_state=imagined,
            difference=difference,
            plausibility=plausibility,
            narrative=narrative,
        )

    def _propagate(
        self,
        current: np.ndarray,
        original: np.ndarray,
        intervention: dict,
    ) -> np.ndarray:
        """因果传播——用因果矩阵迭代传播干预影响。"""
        if self._causal_matrix is None:
            return current

        dim = min(current.shape[0], self._causal_matrix.shape[0], self._causal_matrix.shape[1])
        delta = current[:dim] - original[:dim]
        propagated = current[:dim] + 0.1 * (self._causal_matrix[:dim, :dim] @ delta)
        return propagated

    @staticmethod
    def _assess_plausibility(original: np.ndarray, imagined: np.ndarray, difference: float) -> float:
        """评估想象世界的可信度。"""
        # 简化: 差异越小越可信
        if difference < 0.01:
            return 1.0
        plausibility = max(0.0, 1.0 - difference / (np.linalg.norm(original) + 1.0))
        return float(np.clip(plausibility, 0.0, 1.0))

    @staticmethod
    def _generate_narrative(intervention: dict, original: np.ndarray, imagined: np.ndarray) -> str:
        """生成想象世界的叙述。"""
        if not intervention:
            return "无干预的想象世界"
        parts = [f"如果 {k}={v}" for k, v in intervention.items()]
        diff = float(np.linalg.norm(imagined - original))
        return f"{' ,'.join(parts)}, 则世界状态变化 {diff:.3f}"

    def explore_counterfactuals(
        self,
        variable_index: int,
        values: list[float],
    ) -> list[ImaginedWorld]:
        """探索反事实空间——对某个变量尝试不同值。

        Args:
            variable_index: 变量索引
            values: 要尝试的值列表

        Returns:
            ImaginedWorld 列表
        """
        worlds = []
        for val in values:
            intervention = {variable_index: val}
            result = self.imagine(intervention)
            worlds.extend(result)
        return worlds

    def statistics(self) -> dict[str, Any]:
        return {
            "state_dim": self._state_dim,
            "imagination_count": self.imagination_count,
            "n_steps": self._n_steps,
            "avg_plausibility": (
                float(np.mean([w.plausibility for w in self._imagination_history]))
                if self._imagination_history
                else 0.0
            ),
        }
