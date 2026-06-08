"""
MCI World Model v3.0.6 — Energy Flow Predictor
===============================================

基于 EnergyCore.simulate_energy_flow() 的能量流预测器。
补全 JEPA 在能量维度上的预测盲区：因果图结构预测 + 能量流转预测 = 完整世界模型。

核心功能:
- predict(): 预测未来 N 步的五维能量分布
- validate(): 验证预测能量与实际观测的偏差
- detect_anomaly(): 检测能量流异常（单步突变 > 阈值）

用法:
    from mci_world_model.sdk._energy_flow_predictor import EnergyFlowPredictor

    efp = EnergyFlowPredictor(energy_core)
    flow = efp.predict({"semantic": 0.3, "causal": 0.2, ...}, steps=5)
    anomaly = efp.detect_anomaly(flow, threshold=0.15)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class EnergyFlowPredictor:
    """
    能量流预测器 — 基于五行生克的能量演化预测。

    利用 EnergyCore 的五行生克关系模拟能量流转，
    计算多步前瞻的能量分布序列。

    Attributes:
        _energy_core: EnergyCore 实例
        _history: 能量流历史记录
    """

    def __init__(self, energy_core):
        """
        Args:
            energy_core: EnergyCore 实例（来自 su_memory._sys._energy_core）
        """
        self._energy_core = energy_core
        self._history: list[dict[str, float]] = []

    def predict(self, energy_ratios: dict[str, float], steps: int = 5) -> list[dict[str, float]]:
        """
        预测未来 N 步的能量分布。

        调用 EnergyCore.simulate_energy_flow() 进行多步模拟，
        返回包含当前状态的完整序列（len = steps + 1）。

        Args:
            energy_ratios: 当前五维能量比率 {"semantic": 0.3, ...}
            steps: 预测步数

        Returns:
            [当前分布, 第1步预测, 第2步预测, ...] 长度 steps+1
        """
        flow = self._energy_core.simulate_energy_flow(energy_ratios, steps)
        self._history = flow
        return flow

    def validate(self, predicted: dict[str, float], actual: dict[str, float]) -> float:
        """
        验证预测能量与实际观测的偏差。

        Args:
            predicted: 预测的能量分布
            actual: 实际观测的能量分布

        Returns:
            平均绝对偏差 (MAE)，越小越准确
        """
        diff = 0.0
        for k in predicted:
            diff += abs(predicted.get(k, 0) - actual.get(k, 0))
        return diff / max(len(predicted), 1)

    def detect_anomaly(self, flow_history: list[dict[str, float]], threshold: float = 0.15) -> bool:
        """
        检测能量流异常。

        判定标准：相邻两步间，任五维在单步内突变超过阈值。

        Args:
            flow_history: 能量流序列
            threshold: 单步最大允许变化率（默认 0.15）

        Returns:
            True 表示检测到异常
        """
        if len(flow_history) < 2:
            return False
        prev, curr = flow_history[-2], flow_history[-1]
        for k in prev:
            if abs(prev.get(k, 0) - curr.get(k, 0)) > threshold:
                logger.debug(
                    "能量流异常检测: %s 突变 %.3f → %.3f",
                    k,
                    prev.get(k, 0),
                    curr.get(k, 0),
                )
                return True
        return False

    @property
    def history(self) -> list[dict[str, float]]:
        """最近一次预测的能量流历史。"""
        return self._history
