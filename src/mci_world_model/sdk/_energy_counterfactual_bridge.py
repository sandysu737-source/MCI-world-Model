"""
MCI World Model v4.6.0 — EnergyCore Counterfactual Bridge
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
        except Exception as e:
            logger.warning("吞异常", exc_info=True)
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
        mode: str = "propagation",
    ) -> list[EnergyWhatIfResult]:
        """单变量能量干预查询。

        "如果增强/减弱某一个能量维度，其他维度如何变化？"

        两种模式:
          - mode="propagation" (默认, 推荐): 基于 Flow 矩阵的单步传播效应。
            计算 δ = do向量 - 基线向量, 然后效应 = F·δ。这是矩阵动力学的
            反事实语义——能量守恒系统中, 真正的因果效应是"干预脉冲沿生克
            环的单步传播", 而非"两个终态的差"(守恒系统终态都趋均匀, 无区分度)。
          - mode="steady_state": 旧的稳态差值法 (守恒修复后区分度低, 保留
            供对比)。

        Args:
            energy: 干预的能量类型
            boost: 干预倍数 (>1 增强, <1 减弱)
            baseline_energies: 基线能量分布 (None → 默认均衡)
            mode: "propagation" (矩阵传播) 或 "steady_state" (稳态差值)

        Returns:
            受影响的所有维度的 WhatIf 结果列表
        """
        if energy not in self._categories:
            raise ValueError(
                f"未知能量类型 '{energy}'，已知: {self._categories}"
            )
        if mode not in ("propagation", "steady_state"):
            raise ValueError(f"mode 必须是 'propagation' 或 'steady_state', got '{mode}'")

        if baseline_energies is None:
            baseline_energies = self._default_energies()

        if mode == "propagation":
            return self._what_if_propagation(energy, boost, baseline_energies)
        return self._what_if_steady_state(energy, boost, baseline_energies)

    def _what_if_propagation(
        self, energy: str, boost: float, baseline: dict[str, float]
    ) -> list[EnergyWhatIfResult]:
        """矩阵传播反事实: 效应 = F·(do向量 - 基线向量)。

        这是守恒能量系统中正确的反事实语义。Flow 矩阵的列结构编码了
        每个维度的生克出边, 单步传播给出干预的即时因果效应。
        """
        # 构建干预向量 (归一化: 干预维度 boost, 其余按比例缩减以保持总量)
        base_vec = self._to_ordered_vec(baseline)
        do_vec = base_vec.copy()
        idx = self._categories.index(energy)
        do_vec[idx] *= boost
        if do_vec.sum() > 0:
            do_vec = do_vec / do_vec.sum() * base_vec.sum()  # 保持总能量
        delta_input = do_vec - base_vec

        # Flow 矩阵单步传播
        F = self._energy_matrix()
        effect = F @ delta_input

        results: list[EnergyWhatIfResult] = []
        for i, cat in enumerate(self._categories):
            if cat == energy:
                continue
            mechanism = self._classify_mechanism(energy, cat)
            results.append(EnergyWhatIfResult(
                intervention=f"do({energy}=×{boost:.1f})",
                target_energy=cat,
                baseline=float(base_vec[i]),
                counterfactual=float(base_vec[i] + effect[i]),
                delta=float(effect[i]),
                mechanism=mechanism,
            ))
        return results

    def _what_if_steady_state(
        self, energy: str, boost: float, baseline_energies: dict[str, float]
    ) -> list[EnergyWhatIfResult]:
        """旧稳态差值法 (守恒修复后区分度低, 保留供对比)。"""
        baseline_stable = self._simulate_to_stable(baseline_energies)
        intervened = dict(baseline_energies)
        intervened[energy] *= boost
        intervened_stable = self._simulate_to_stable(intervened)
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

    def _to_ordered_vec(self, energies: dict[str, float]) -> np.ndarray:
        """dict → 按 categories 顺序的 length-5 向量。"""
        v = np.zeros(len(self._categories), dtype=np.float64)
        for i, cat in enumerate(self._categories):
            v[i] = float(energies.get(cat, 0.0))
        return v

    def _energy_matrix(self) -> np.ndarray:
        """获取 EnergyCore 的 Flow 矩阵 (兼容有无 flow_matrix 方法)。"""
        if hasattr(self._energy_core, "flow_matrix"):
            return self._energy_core.flow_matrix()
        # 回退: 手动构建 (旧 EnergyCore)
        return self._build_flow_matrix_fallback()

    def _build_flow_matrix_fallback(self) -> np.ndarray:
        """无 flow_matrix 方法时手动构建守恒 Flow 矩阵。"""
        from mci_world_model._sys._terms import ENERGY_ENHANCE, ENERGY_SUPPRESS
        n = len(self._categories)
        F = np.zeros((n, n), dtype=np.float64)
        enh = getattr(self._energy_core, "ENHANCE_FLOW_RATE", 0.15)
        supp = abs(getattr(self._energy_core, "SUPPRESS_FLOW_RATE", 0.10))
        idx = {c: i for i, c in enumerate(self._categories)}
        for src, tgt in ENERGY_ENHANCE.items():
            if src in idx and tgt in idx:
                F[idx[tgt], idx[src]] += enh
                F[idx[src], idx[src]] -= enh
        for src, tgt in ENERGY_SUPPRESS.items():
            if src in idx and tgt in idx:
                F[idx[src], idx[tgt]] += supp
                F[idx[tgt], idx[tgt]] -= supp
        return F

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
        except Exception as e:
            logger.warning("吞异常", exc_info=True)
        return "间接"
