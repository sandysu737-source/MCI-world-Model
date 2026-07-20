from __future__ import annotations

"""
MCI World Model v4.6.0 — Energy Cost Module
============================================

LeCun 六模块架构中的 Cost 模块：评估世界状态的"好坏"，
提供梯度信号驱动 Actor 进行动作搜索。

职责：
- 接收 CausalWorldModelState，输出多维度代价信号
- 不依赖 Trainer，可独立被 Actor/Configurator 调用
- 原子化设计：每次 evaluate() 返回不可变 CostSignal

状态机：IDLE → COMPUTING → COMPLETE
异常降级：检测到空状态或计算异常时返回 CostSignal.zero()

用法:
    from mci_world_model.sdk._cost_module import EnergyCostModule, CostSignal

    cost_module = EnergyCostModule(alpha_energy=0.5, beta_causal=0.3, gamma_temporal=0.2)
    signal = cost_module.evaluate(state)
    logger.info(f"Total cost: {signal.total:.4f}")
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# CostSignal — 不可变代价信号
# =============================================================================


@dataclass(frozen=True)
class CostSignal:
    """
    不可变代价信号，由 EnergyCostModule.evaluate() 返回。

    Attributes:
        total: 加权总代价 (0 = 最优状态)
        energy_balance: 五行均衡代价 (偏离黄金比例惩罚)
        causal_consistency: 因果一致性代价 (增强/抑制违规惩罚)
        temporal_coherence: 时序连贯性代价 (能量总量守恒偏离)
        breakdown: 各维度原始值字典
    """

    total: float
    energy_balance: float
    causal_consistency: float
    temporal_coherence: float
    breakdown: dict[str, Any]

    @classmethod
    def zero(cls) -> CostSignal:
        """返回零代价信号（空状态或异常降级）。"""
        return cls(
            total=0.0,
            energy_balance=0.0,
            causal_consistency=0.0,
            temporal_coherence=0.0,
            breakdown={"eb": 0.0, "cc": 0.0, "tc": 0.0},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": round(self.total, 6),
            "energy_balance": round(self.energy_balance, 6),
            "causal_consistency": round(self.causal_consistency, 6),
            "temporal_coherence": round(self.temporal_coherence, 6),
        }


# =============================================================================
# EnergyCostModule — 能量代价评估引擎
# =============================================================================


class EnergyCostModule:
    """
    独立的能量代价评估模块。

    评估三个维度：
    1. 能量均衡代价 — 增强/抑制关系是否合理（边权重偏离阈值）
    2. 因果一致性代价 — 图结构是否违反能量流转模式
    3. 时序连贯性代价 — 能量总量是否守恒

    六态流转：IDLE → COMPUTING → COMPLETE（异常时 COMPUTING → COMPLETE 降级为 zero）
    """

    def __init__(  # type: ignore
        self,
        alpha_energy: float = 0.5,
        beta_causal: float = 0.3,
        gamma_temporal: float = 0.2,
        energy_core=None,  # v3.0.4: 外部注入 EnergyCore
        month_branch: int = 0,  # v3.0.4: 当前月份地支索引
    ):
        """
        Args:
            alpha_energy: 能量均衡维度权重
            beta_causal: 因果一致性维度权重
            gamma_temporal: 时序连贯性维度权重
            energy_core: EnergyCore 实例，启用时变代价 (v3.0.4)
            month_branch: 当前月份地支索引 0-11 (v3.0.4)
        """
        self._alpha = alpha_energy
        self._beta = beta_causal
        self._gamma = gamma_temporal
        self._energy_core = energy_core  # v3.0.4
        self._month_branch = month_branch  # v3.0.4
        self._state: str = "IDLE"  # IDLE → COMPUTING → COMPLETE
        self._eval_count: int = 0

    @property
    def state(self) -> str:
        return self._state

    @property
    def eval_count(self) -> int:
        return self._eval_count

    # -----------------------------------------------------------------
    # 核心评估
    # -----------------------------------------------------------------

    def evaluate(self, state: Any) -> CostSignal:
        """
        评估世界状态的代价。

        对 CausalWorldModelState 中的每条因果边进行三维度评估，
        返回加权总代价。

        Args:
            state: CausalWorldModelState 当前世界状态

        Returns:
            CostSignal 不可变代价信号
        """
        self._state = "COMPUTING"

        try:
            eb = self._energy_balance_cost(state)
            cc = self._causal_consistency_cost(state)
            tc = self._temporal_coherence_cost(state)
            oc = self._overconstraint_cost(state)  # v3.0.4: 乘侮异常代价

            total = self._alpha * eb + self._beta * cc + self._gamma * tc + 0.15 * oc

            self._eval_count += 1
            self._state = "COMPLETE"

            return CostSignal(
                total=total,
                energy_balance=eb,
                causal_consistency=cc,
                temporal_coherence=tc,
                breakdown={"eb": eb, "cc": cc, "tc": tc, "oc": oc},
            )
        except Exception:
            logger.warning("CostModule.evaluate() 异常降级为 zero", exc_info=True)
            self._state = "COMPLETE"
            return CostSignal.zero()

    # -----------------------------------------------------------------
    # 维度计算（私有方法，从 Trainer 提取）
    # -----------------------------------------------------------------

    # v3.0.4: 改为实例方法，支持时变能量惩罚
    def _energy_balance_cost(self, state: Any) -> float:
        """
        计算能量均衡代价。

        对每条因果边检查增强/抑制模式违规：
        - 增强边权重过低 → 惩罚
        - 抑制边权重过高 → 惩罚

        v3.0.4: 引入时变惩罚系数 — 基于 EnergyCore 旺衰状态
        - WANG 态 → 惩罚 ×1.2 (能量过旺时的违规更严重)
        - SI 态  → 惩罚 ×0.3 (能量极弱时违规容忍度更高)

        Args:
            state: CausalWorldModelState

        Returns:
            归一化违规分数 [0, ∞)
        """
        if not state.causal_edges:
            return 0.0

        violations = 0.0
        n_edges = len(state.causal_edges)

        for edge in state.causal_edges:
            energy_rel = edge.get("energy_relation", "neutral")
            rho = edge.get("rho", 0.0)

            # ── v3.0.4: 时变惩罚系数 ──
            time_factor = 1.0
            if self._energy_core is not None:
                try:
                    cause_energy = edge.get("cause_energy", "earth")
                    energy_state = self._energy_core.get_energy_state(cause_energy, self._month_branch)
                    time_factor = self._energy_core.STRENGTH_MULTIPLIER.get(energy_state.strength, 1.0)
                except Exception:
                    logger.warning("energy state time_factor fallback", exc_info=True)

            # 增强模式下的低权重惩罚
            if energy_rel == "enhance" and rho < 0.3:
                violations += (0.3 - rho) * 2.0 * time_factor
            # 抑制模式下的高权重惩罚
            elif energy_rel == "suppress" and rho > 0.7:
                violations += (rho - 0.7) * 2.0 * time_factor

        return violations / max(n_edges, 1)

    def _overconstraint_cost(self, state: Any) -> float:
        """
        v3.0.4: 检测过度克制（相乘）和反向克制（相侮）异常。

        相乘 (overconstraint): 克方过强，被克方过弱 → 异常制衡
        相侮 (reverse): 被克方反向克制克方 → 关系倒置

        Returns:
            乘侮异常分数 [0, ∞)
        """
        if self._energy_core is None or not state.causal_edges:
            return 0.0
        violations = 0.0
        for edge in state.causal_edges:
            cause_energy = edge.get("cause_energy", "earth")
            effect_energy = edge.get("effect_energy", "earth")
            if self._energy_core.get_overconstraint_relation(cause_energy, effect_energy):
                violations += 0.15
            if self._energy_core.get_reverse_relation(cause_energy, effect_energy):
                violations += 0.10
        return violations / max(len(state.causal_edges), 1)

    @staticmethod
    def _causal_consistency_cost(state: Any) -> float:
        """
        计算因果一致性代价。

        检查因果边 verdict 的一致性和置信度分布：
        - 'none' verdict 边 → 弱惩罚（未确认的因果关系）
        - 低置信度 novel 边 → 中等惩罚
        - 仅在有足够边时计算，避免噪声放大

        Args:
            state: CausalWorldModelState

        Returns:
            一致性违规分数 [0, 1]
        """
        if not state.causal_edges:
            return 0.0

        n_edges = len(state.causal_edges)
        inconsistent = 0.0

        for edge in state.causal_edges:
            verdict = edge.get("verdict", "none")
            confidence = edge.get("confidence", 0.0)

            if verdict == "none":
                inconsistent += 0.1  # 未确认的边
            elif verdict == "novel" and confidence < 0.5:
                inconsistent += 0.05  # 低置信新发现

        return inconsistent / max(n_edges, 1)

    @staticmethod
    def _temporal_coherence_cost(state: Any) -> float:
        """
        计算时序连贯性代价。

        评估能量总量的自洽性：因果图中所有边的 rho 绝对值之和
        应与能量关系模式一致。当前使用能量总量自洽性作为代理指标：
        - 边权重分布的标准差作为衡量指标
        - 标准差过大 → 能量分布不均衡

        Args:
            state: CausalWorldModelState

        Returns:
            时序不连贯分数 [0, 1]
        """
        if not state.causal_edges:
            return 0.0

        import numpy as np

        rhos = [abs(e.get("rho", 0.0)) for e in state.causal_edges]
        n_edges = len(rhos)

        if n_edges < 2:
            return 0.0

        mean_rho = np.mean(rhos)
        std_rho = np.std(rhos)

        # 变异系数 (CV) 作为不连贯度量
        cv = std_rho / max(mean_rho, 1e-10)
        return min(cv, 1.0)
