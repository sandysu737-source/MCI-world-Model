"""
MCI World Model v4.5.0 — EnergyCore Counterfactual Bridge
==========================================================

将五行能量系统桥接到反事实推理引擎，
支持 "what-if" 能量干预查询。

核心能力:
- what_if(): 单变量能量干预 → 后果预测 (基于 EnergyCore 仿真)
- batch_what_if(): 批量能量干预查询
- systemic_impact(): 全系统能量干预影响分析
- counterfactual_energy(): 给定当前状态 → 反事实能量分布

设计说明:
    五行生克图是强循环图 (semantic→causal→spacetime→generative→trust→semantic)，
    不适合直接使用 Pearl do-calculus (假设 DAG)。
    因此 what_if 直接基于 EnergyCore.simulate_energy_flow() 的
    差值方法: (干预后稳定态) - (基线稳定态)。

用法:
    from mci_world_model._sys._energy_core import EnergyCore
    from mci_world_model.sdk._energy_counterfactual_bridge import (
        EnergyCounterfactualBridge,
    )

    ec = EnergyCore()
    bridge = EnergyCounterfactualBridge(ec)
    results = bridge.what_if("semantic", boost=1.5)
    impact = bridge.systemic_impact("causal", boost=2.0)
    cf = bridge.counterfactual_energy(
        {"semantic": 0.4, "causal": 0.1, "spacetime": 0.2, "generative": 0.15, "trust": 0.15},
        do={"semantic": 0.8},
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# EnergyWhatIfResult — 能量干预结果
# =============================================================================


@dataclass
class EnergyWhatIfResult:
    """单变量能量干预的结果。

    Attributes:
        intervention: 干预描述
        target_energy: 目标能量
        baseline: 基线值
        counterfactual: 反事实值
        delta: 变化量
        mechanism: 因果机制
    """

    intervention: str = ""
    target_energy: str = ""
    baseline: float = 0.0
    counterfactual: float = 0.0
    delta: float = 0.0
    mechanism: str = ""

    @property
    def effect_ratio(self) -> float:
        """效应比率。"""
        if self.baseline == 0:
            return 0.0
        return self.delta / self.baseline

    @property
    def direction(self) -> str:
        if self.delta > 0.01:
            return "增强"
        elif self.delta < -0.01:
            return "减弱"
        return "无影响"


# =============================================================================
# EnergyCounterfactualBridge
# =============================================================================


class EnergyCounterfactualBridge:
    """五行能量 ↔ 反事实推理桥接器。

    利用 EnergyCore.simulate_energy_flow() 进行差值法
    能量干预预测，自动检测 EnergyCore 的五维类别名。
    """

    def __init__(self, energy_core: Any, sim_steps: int = 30, seed: int = 42):
        """初始化桥接器。

        Args:
            energy_core: EnergyCore 实例
            sim_steps: 仿真步数 (越大收敛越稳，默认 30)
            seed: 随机种子 (保留接口)
        """
        self._energy_core = energy_core
        self._sim_steps = sim_steps
        self._seed = seed

        # 自动检测五维类别名
        self._categories = self._detect_categories()

    def _detect_categories(self) -> list[str]:
        """从 EnergyCore 检测五维类别名。"""
        # 尝试调用 get_energy_cycle
        try:
            cycle = self._energy_core.get_energy_cycle()
            categories: list[str] = []
            seen: set[str] = set()
            for src, _ in cycle:
                if src not in seen:
                    categories.append(src)
                    seen.add(src)
            if len(categories) >= 5:
                return categories[:5]
        except Exception:
            pass

        # 回退到默认五维
        return ["semantic", "causal", "spacetime", "generative", "trust"]

    # ── 公共 API ──

    @property
    def categories(self) -> list[str]:
        """五维类别名。"""
        return self._categories

    def what_if(
        self,
        energy: str,
        boost: float = 1.0,
        baseline_energies: dict[str, float] | None = None,
    ) -> list[EnergyWhatIfResult]:
        """单变量能量干预查询。

        "如果增强/减弱某一个能量维度，其他维度如何变化？"

        使用差值法: 比较干预前后的 simulate_energy_flow() 稳定态。

        Args:
            energy: 干预的能量类型
            boost: 干预倍数 (>1 增强, <1 减弱)
            baseline_energies: 基线能量分布 (None → 默认均衡)

        Returns:
            受影响的所有维度的 WhatIf 结果列表
        """
        if energy not in self._categories:
            raise ValueError(
                f"未知能量类型 '{energy}'，已知: {self._categories}"
            )

        if baseline_energies is None:
            baseline_energies = self._default_energies()

        # 基线仿真 → 稳定态
        baseline_stable = self._simulate_to_stable(baseline_energies)

        # 干预仿真 → 稳定态
        intervened = dict(baseline_energies)
        intervened[energy] *= boost
        intervened_stable = self._simulate_to_stable(intervened)

        # 计算差值
        results: list[EnergyWhatIfResult] = []
        for cat in self._categories:
            if cat == energy:
                continue
            b = baseline_stable.get(cat, 0.0)
            c = intervened_stable.get(cat, 0.0)
            mechanism = self._classify_mechanism(energy, cat)
            results.append(EnergyWhatIfResult(
                intervention=f"do({energy}=×{boost:.1f})",
                target_energy=cat,
                baseline=b,
                counterfactual=c,
                delta=c - b,
                mechanism=mechanism,
            ))

        return results

    def batch_what_if(
        self,
        interventions: list[dict[str, Any]],
        baseline_energies: dict[str, float] | None = None,
    ) -> list[list[EnergyWhatIfResult]]:
        """批量能量干预查询。

        Args:
            interventions: [{"energy":"semantic","boost":1.5}, ...]
            baseline_energies: 基线能量分布

        Returns:
            每个干预的结果列表
        """
        return [
            self.what_if(
                i["energy"],
                boost=i.get("boost", 1.0),
                baseline_energies=baseline_energies,
            )
            for i in interventions
        ]

    def systemic_impact(
        self,
        energy: str,
        boost: float = 1.5,
        baseline_energies: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """全系统能量干预影响分析。

        Returns:
            包含所有维度的基线/反事实/Δ 的完整映射
        """
        results = self.what_if(energy, boost=boost, baseline_energies=baseline_energies)
        impact: dict[str, Any] = {
            "intervention": f"{energy} × {boost:.1f}",
            "affected": {},
        }
        for r in results:
            impact["affected"][r.target_energy] = {
                "baseline": round(r.baseline, 4),
                "counterfactual": round(r.counterfactual, 4),
                "delta": round(r.delta, 4),
                "direction": r.direction,
                "mechanism": r.mechanism,
            }
        return impact

    def counterfactual_energy(
        self,
        current_state: dict[str, float],
        do: dict[str, float],
    ) -> dict[str, float]:
        """给定当前五行状态，计算反事实能量分布。

        Args:
            current_state: 当前能量分布
            do: 能量干预 {"semantic": 0.8}

        Returns:
            反事实稳态能量分布 (全维度)
        """
        cf_state = dict(current_state)
        cf_state.update(do)
        return self._simulate_to_stable(cf_state)

    # ── 内部方法 ──

    def _default_energies(self) -> dict[str, float]:
        v = 1.0 / len(self._categories)
        return dict.fromkeys(self._categories, v)

    def _simulate_to_stable(self, energies: dict[str, float]) -> dict[str, float]:
        """运行 EnergyCore 仿真至稳态，返回尾部均值。

        Args:
            energies: 初始能量分布

        Returns:
            稳态能量分布
        """
        try:
            flow = self._energy_core.simulate_energy_flow(
                energies, steps=self._sim_steps
            )
        except Exception:
            logger.warning("EnergyCore 仿真失败，返回初始值", exc_info=True)
            return dict(energies)

        if not flow:
            return dict(energies)

        # 取尾部 min(5, steps) 步的均值作为稳态
        tail = flow[-min(5, len(flow)):]
        stable: dict[str, float] = {}
        for k in energies:
            vals = [step.get(k, 0.0) for step in tail]
            stable[k] = float(np.mean(vals))
        return stable

    def _classify_mechanism(self, src: str, dst: str) -> str:
        """分类因果机制 (生/克/间接)。"""
        try:
            is_enhance = self._energy_core.get_enhance_relation(src, dst)
            if is_enhance:
                return "相生"
            is_suppress = self._energy_core.get_suppress_relation(src, dst)
            if is_suppress:
                return "相克"
        except Exception:
            pass
        return "间接"
