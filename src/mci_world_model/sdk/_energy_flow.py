from __future__ import annotations

"""MCI World Model 能量流边界。

本模块只承载五维能量聚合、EnergyBus 传播与能量流预测，
不承载因果发现、CEWM 或世界模型编排逻辑。
"""

from typing import TYPE_CHECKING, Any, ClassVar


def _aggregate_energy_ratios(causal_edges: list[dict[str, Any]]) -> dict[str, float] | None:
    """v3.0.6: 从因果边列表聚合五维能量比率（公共逻辑）。"""
    if not causal_edges:
        return None
    energy_counts: dict[str, int] = {
        "semantic": 0,
        "causal": 0,
        "spacetime": 0,
        "generative": 0,
        "trust": 0,
    }
    for edge in causal_edges:
        for key in ("cause_energy", "effect_energy"):
            e = edge.get(key, "")
            if e in energy_counts:
                energy_counts[e] += 1
    total = sum(energy_counts.values())
    if total == 0:
        return None
    return {k: v / total for k, v in energy_counts.items()}


class EnergyFlowMixin:
    """能量流行为 Mixin。

    宿主类必须提供 ``FIVE_STATES``、``_state``、``_energy_core``、
    ``_energy_flow_predictor`` 与 ``_get_energy_core()``。
    """

    if TYPE_CHECKING:
        FIVE_STATES: ClassVar[list[str]]
        _state: Any
        _energy_core: Any | None
        _energy_flow_predictor: Any | None

        def _get_energy_core(self) -> Any: ...

    def _extract_energy_ratios(self, state: Any) -> dict[str, float] | None:
        """
        v3.0.5: 从因果图状态提取五维能量分布比率。

        委托到公共函数 ``_aggregate_energy_ratios`` 避免逻辑重复。

        Args:
            state: CausalWorldModelState 实例

        Returns:
            五维能量比率字典，若 causal_edges 无能量标签返回 None
        """
        if not hasattr(state, "causal_edges") or not state.causal_edges:
            return None
        return _aggregate_energy_ratios(state.causal_edges)

    def _compute_energy_coverage(self) -> dict[str, Any]:
        """
        v3.0.6: 计算五维能量覆盖度。

        从当前因果图状态提取能量分布，计算覆盖评分。
        coverage_score = 有能量标签(>5%)的维度数 / 5。

        Returns:
            {"ratios": {...}, "coverage_score": float, "warning": str|None}
        """
        energy_ratios = self._extract_energy_ratios(self._state)
        active_dims = len([v for v in (energy_ratios or {}).values() if v > 0.05])
        coverage_score = active_dims / 5.0
        warning = None
        if coverage_score < 0.6:
            warning = "能量维度覆盖不足，建议丰富数据源"
        return {
            "ratios": energy_ratios or {},
            "coverage_score": round(coverage_score, 3),
            "warning": warning,
        }

    def predict_energy_flow(
        self,
        steps: int = 5,
    ) -> dict[str, Any]:
        """v4.3.3: 基于五行生克的能量流多步预测。

        闭合 JEPA 在能量维度上的预测盲区，模拟能量在五维空间的流转趋势。

        Args:
            steps: 预测步数 (默认 5)

        Returns:
            {"steps": int, "flow": [...], "anomaly_detected": bool, "current_ratios": {...}}
        """
        from mci_world_model.sdk._energy_flow_predictor import EnergyFlowPredictor

        if self._energy_flow_predictor is None:
            if self._energy_core is None:
                self._get_energy_core()
            self._energy_flow_predictor = EnergyFlowPredictor(self._energy_core)

        current_ratios = self._compute_energy_coverage()
        ratios = current_ratios.get("ratios", {})
        if not ratios:
            ratios = {
                "semantic": 0.2,
                "causal": 0.2,
                "spacetime": 0.2,
                "generative": 0.2,
                "trust": 0.2,
            }

        flow = self._energy_flow_predictor.predict(ratios, steps=steps)
        anomaly = self._energy_flow_predictor.detect_anomaly(flow)

        return {
            "steps": steps,
            "flow": flow,
            "anomaly_detected": anomaly,
            "current_ratios": ratios,
        }
