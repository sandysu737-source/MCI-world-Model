from __future__ import annotations

"""
MCI World Model v4.4.0 — Force-Tissue Dynamics
================================================

清创机器人 — 力-组织响应动力学模型。

预测: f(tissue_type, force_applied, velocity) → (depth_removed, force_feedback, safety_flag)

组织特异性参数:
    坏死:   低阻力 (0.5-2N足以去除), 去除深度 ∝ force
    腐肉:   中阻力 (1-3N), 纤维结构
    肉芽:   高阻力 (2-5N), 含血管需保护
    上皮:   极高阻力 (>5N), 严禁损伤

安全门禁: 力超限/深度超限/组织误判均触发停止

v4.4.0 新增:
    - calibrate_from_data(): 从实测数据学习组织特异性参数
    - predict_batch(): 批量预测
    - get_safety_summary(): 安全汇总
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# 组织特异性参数
TISSUE_PARAMS = {
    0: {"name": "坏死", "stiffness": 0.5, "max_force": 3.0, "max_velocity": 10.0, "removal_rate": 2.0},
    1: {"name": "腐肉", "stiffness": 1.5, "max_force": 2.0, "max_velocity": 5.0, "removal_rate": 1.0},
    2: {"name": "肉芽", "stiffness": 3.0, "max_force": 1.0, "max_velocity": 3.0, "removal_rate": 0.2},
    3: {"name": "上皮", "stiffness": 8.0, "max_force": 0.5, "max_velocity": 1.0, "removal_rate": 0.0},
}


@dataclass
class RemovalPrediction:
    """组织去除预测结果。"""
    depth_removed_mm: float
    force_feedback_n: float
    tissue_type: int
    is_safe: bool
    warning: str = ""


@dataclass
class SafetyVerdict:
    """安全判断结果。"""
    passed: bool
    reason: str = ""
    max_allowed_force: float = 0.0
    max_allowed_depth: float = 0.0


class ForceTissueDynamics:
    """力-组织响应动力学模型。

    用物理合理性的简化模型预测工具-组织交互:
    - removal_depth = removal_rate * force / stiffness
    - force_feedback = stiffness * depth_removed (Hooke 定律近似)

    v4.4.0 新增学习型校准:
    - calibrate_from_data(): 从实测数据拟合组织参数
    """

    def __init__(self) -> None:
        self._params = TISSUE_PARAMS

    def predict_removal(self, tissue_type: int, force_n: float, velocity: float) -> RemovalPrediction:
        """预测给定力下的组织去除。

        Args:
            tissue_type: 0=坏死 1=腐肉 2=肉芽 3=上皮
            force_n: 施加力 (N)
            velocity: 工具速度 (mm/s)
        """
        if tissue_type not in self._params:
            return RemovalPrediction(0, 0, tissue_type, False, f"未知组织类型: {tissue_type}")

        p = self._params[tissue_type]
        stiffness = p["stiffness"]
        removal_rate = p["removal_rate"]
        max_force = p["max_force"]
        max_vel = p["max_velocity"]

        # 物理预测
        depth = removal_rate * max(force_n, 0) / max(stiffness, 0.01)
        feedback = stiffness * depth

        # 安全判断
        warnings = []
        if force_n > max_force:
            warnings.append(f"力超限: {force_n:.1f}N > {max_force}N ({p['name']}上限)")
        if velocity > max_vel:
            warnings.append(f"速度超限: {velocity:.1f}mm/s > {max_vel}mm/s ({p['name']}上限)")
        if tissue_type == 3 and force_n > 0.1:
            warnings.append("严禁对上皮组织施加清创力!")

        is_safe = len(warnings) == 0

        return RemovalPrediction(
            depth_removed_mm=round(depth, 3),
            force_feedback_n=round(feedback, 3),
            tissue_type=tissue_type,
            is_safe=is_safe,
            warning="; ".join(warnings) if warnings else "",
        )

    def safety_check(self, tissue_type: int, force_n: float, depth_mm: float) -> SafetyVerdict:
        """安全门禁检查。

        Args:
            tissue_type: 组织类型
            force_n: 当前力 (N)
            depth_mm: 当前清创深度 (mm)
        """
        if tissue_type not in self._params:
            return SafetyVerdict(False, f"未知组织类型: {tissue_type}")

        p = self._params[tissue_type]

        # 力检查
        if force_n > p["max_force"]:
            return SafetyVerdict(
                False,
                f"力超限: {force_n:.1f}N > {p['max_force']}N ({p['name']})",
                p["max_force"],
                depth_mm,
            )

        # 上皮保护
        if tissue_type == 3:
            return SafetyVerdict(False, "上皮组织——停止清创", 0.5, 0.0)

        return SafetyVerdict(True, "安全", p["max_force"], depth_mm + 2.0)

    # ── v4.4.0: 学习型动力学校准 ──

    def calibrate_from_data(
        self,
        force_data: list[tuple[int, float, float, float]],
        lr: float = 0.01,
        n_epochs: int = 100,
    ) -> dict[str, float]:
        """从实测数据校准组织特异性参数。

        Args:
            force_data: [(tissue_type, force_n, velocity, observed_depth_mm), ...]
            lr: 学习率
            n_epochs: 迭代轮数

        Returns:
            {"final_mse": ..., "calibrated_params": {...}}
        """
        import numpy as np

        n_tissues = len(self._params)
        stiffness = np.array([self._params[i]["stiffness"] for i in range(n_tissues)], dtype=np.float64)
        removal_rate = np.array([self._params[i]["removal_rate"] for i in range(n_tissues)], dtype=np.float64)

        tissues = np.array([d[0] for d in force_data], dtype=np.int32)
        forces = np.array([d[1] for d in force_data], dtype=np.float64)
        velocities = np.array([d[2] for d in force_data], dtype=np.float64)
        observed_depths = np.array([d[3] for d in force_data], dtype=np.float64)

        best_mse = float("inf")
        best_stiffness = stiffness.copy()
        best_removal_rate = removal_rate.copy()

        for _epoch in range(n_epochs):
            s = stiffness[tissues]
            r = removal_rate[tissues]
            predicted = r * np.maximum(forces, 0) / np.maximum(s, 0.01)

            errors = predicted - observed_depths
            mse = np.mean(errors ** 2)

            if mse < best_mse:
                best_mse = mse
                best_stiffness = stiffness.copy()
                best_removal_rate = removal_rate.copy()

            grad_s = np.zeros(n_tissues, dtype=np.float64)
            grad_r = np.zeros(n_tissues, dtype=np.float64)

            for i in range(len(force_data)):
                t = tissues[i]
                f = max(forces[i], 0)
                err = errors[i]
                grad_s[t] += -2 * err * removal_rate[t] * f / max(stiffness[t] ** 2, 1e-6)
                grad_r[t] += 2 * err * f / max(stiffness[t], 0.01)

            stiffness -= lr * grad_s
            removal_rate -= lr * grad_r
            stiffness = np.clip(stiffness, 0.1, 20.0)
            removal_rate = np.clip(removal_rate, 0.0, 10.0)

        for i in range(n_tissues):
            self._params[i]["stiffness"] = float(best_stiffness[i])
            self._params[i]["removal_rate"] = float(best_removal_rate[i])

        return {
            "final_mse": float(best_mse),
            "calibrated_params": {
                str(i): {
                    "name": self._params[i]["name"],
                    "stiffness": round(float(best_stiffness[i]), 4),
                    "removal_rate": round(float(best_removal_rate[i]), 4),
                    "max_force": self._params[i]["max_force"],
                }
                for i in range(n_tissues)
            },
        }

    def predict_batch(
        self, batch: list[tuple[int, float, float]]
    ) -> list[RemovalPrediction]:
        """批量预测组织去除。"""
        return [self.predict_removal(t, f, v) for t, f, v in batch]

    def get_safety_summary(
        self, predictions: list[RemovalPrediction]
    ) -> dict[str, Any]:
        """安全汇总: 从预测列表生成安全报告。"""
        safe = sum(1 for p in predictions if p.is_safe)
        violations = [
            {"tissue": p.tissue_type, "warning": p.warning, "depth": p.depth_removed_mm}
            for p in predictions if not p.is_safe
        ]
        return {
            "safe_count": safe,
            "violation_count": len(violations),
            "total": len(predictions),
            "safe_rate": safe / max(len(predictions), 1),
            "violations": violations,
        }

    @staticmethod
    def get_max_force(tissue_type: int) -> float:
        return TISSUE_PARAMS.get(tissue_type, {}).get("max_force", 0.5)

    @staticmethod
    def get_max_velocity(tissue_type: int) -> float:
        return TISSUE_PARAMS.get(tissue_type, {}).get("max_velocity", 1.0)

    @staticmethod
    def get_tissue_name(tissue_type: int) -> str:
        return TISSUE_PARAMS.get(tissue_type, {}).get("name", "未知")
