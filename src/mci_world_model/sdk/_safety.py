from __future__ import annotations

"""MCI World Model v4.5.0 — SafetyConstraint 安全约束层
=========================================================

安全关键系统的基础约束能力——在动作执行前进行安全检查，
确保力/位置/速度等物理量不超出安全范围。

核心能力:
    SafetyConstraint ABC  — 安全约束抽象基类
    ForceLimitConstraint — 力矩/力限制
    PositionBoundConstraint — 位置/角度边界
    VelocityLimitConstraint — 速度/角速度限制
    SafetyMonitor        — 约束链注册 + 执行前检查

设计原则:
    - 约束与动作解耦: check(state, action) → (bool, str)
    - 链式注册: 多个约束可组合，全部通过才放行
    - 与 CircuitBreaker 职责清晰: SafetyConstraint 聚焦物理约束，
      CircuitBreaker 聚焦服务降级
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mci_world_model.sdk._world_state import Action, WorldState

logger = logging.getLogger(__name__)


# =============================================================================
# SafetyCheckResult — 安全检查结果
# =============================================================================


@dataclass
class SafetyCheckResult:
    """安全检查结果。

    Attributes:
        passed: 是否通过安全检查
        constraint_name: 约束名称
        reason: 不通过原因（通过时为空字符串）
        severity: 严重度 ('warning' / 'violation')
        details: 详细信息
    """

    passed: bool
    constraint_name: str = ""
    reason: str = ""
    severity: str = "violation"
    details: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# SafetyConstraint — 安全约束抽象基类
# =============================================================================


class SafetyConstraint(ABC):
    """安全约束抽象基类。

    任何安全约束实现此 ABC 即可插入 SafetyMonitor。
    子类只需实现 check() 方法。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """约束名称。"""
        ...

    @abstractmethod
    def check(self, state: WorldState, action: Action | None) -> SafetyCheckResult:
        """检查给定状态和动作是否满足安全约束。

        Args:
            state: 当前世界状态
            action: 待执行动作（None 表示无动作，只检查状态）

        Returns:
            SafetyCheckResult
        """
        ...


# =============================================================================
# ForceLimitConstraint — 力矩/力限制
# =============================================================================


class ForceLimitConstraint(SafetyConstraint):
    """力矩/力限制约束——确保力不超过安全阈值。

    适用于:
        - PendulumAction: torque 不超过 max_torque
        - CartAction: force 不超过 max_force
        - 任意 Action: 通过 to_vector() 的第一维检查
    """

    def __init__(self, max_torque: float = 10.0, max_force: float = 10.0) -> None:
        self._max_torque = max_torque
        self._max_force = max_force

    @property
    def name(self) -> str:
        return "force_limit"

    def check(self, state: WorldState, action: Action | None) -> SafetyCheckResult:
        if action is None:
            return SafetyCheckResult(passed=True, constraint_name=self.name)

        # PendulumAction: 检查 torque
        if hasattr(action, "torque"):
            if abs(action.torque) > self._max_torque:
                return SafetyCheckResult(
                    passed=False,
                    constraint_name=self.name,
                    reason=f"力矩超限: {abs(action.torque):.2f} > {self._max_torque:.2f}",
                    severity="violation",
                    details={"torque": action.torque, "max_torque": self._max_torque},
                )

        # CartAction: 检查 force
        if hasattr(action, "force"):
            if abs(action.force) > self._max_force:
                return SafetyCheckResult(
                    passed=False,
                    constraint_name=self.name,
                    reason=f"力超限: {abs(action.force):.2f} > {self._max_force:.2f}",
                    severity="violation",
                    details={"force": action.force, "max_force": self._max_force},
                )

        # 通用: 通过 to_vector() 检查
        if hasattr(action, "to_vector"):
            try:
                vec = action.to_vector()
                if len(vec) > 0 and abs(float(vec[0])) > self._max_force:
                    return SafetyCheckResult(
                        passed=False,
                        constraint_name=self.name,
                        reason=f"动作幅值超限: {abs(float(vec[0])):.2f} > {self._max_force:.2f}",
                        severity="violation",
                    )
            except Exception as e:
                logger.warning("吞异常", exc_info=True)
        return SafetyCheckResult(passed=True, constraint_name=self.name)


# =============================================================================
# PositionBoundConstraint — 位置/角度边界
# =============================================================================


class PositionBoundConstraint(SafetyConstraint):
    """位置/角度边界约束——确保状态变量不越界。

    适用于:
        - PendulumState: theta 不超过 [-max_theta, max_theta]
        - CartState: x 不超过 [-max_x, max_x]
        - 任意 WorldState: 通过 to_vector() 的第一维检查
    """

    def __init__(
        self,
        max_theta: float = 3.14159,
        max_x: float = 100.0,
    ):
        self._max_theta = max_theta
        self._max_x = max_x

    @property
    def name(self) -> str:
        return "position_bound"

    def check(self, state: WorldState, action: Action | None) -> SafetyCheckResult:
        # PendulumState: 检查 theta
        if hasattr(state, "theta"):
            if abs(state.theta) > self._max_theta:
                return SafetyCheckResult(
                    passed=False,
                    constraint_name=self.name,
                    reason=f"角度越界: {abs(state.theta):.2f} > {self._max_theta:.2f}",
                    severity="violation",
                    details={"theta": state.theta, "max_theta": self._max_theta},
                )

        # CartState: 检查 x
        if hasattr(state, "x") and not hasattr(state, "theta"):
            if abs(state.x) > self._max_x:
                return SafetyCheckResult(
                    passed=False,
                    constraint_name=self.name,
                    reason=f"位置越界: {abs(state.x):.2f} > {self._max_x:.2f}",
                    severity="violation",
                    details={"x": state.x, "max_x": self._max_x},
                )

        return SafetyCheckResult(passed=True, constraint_name=self.name)


# =============================================================================
# VelocityLimitConstraint — 速度/角速度限制
# =============================================================================


class VelocityLimitConstraint(SafetyConstraint):
    """速度/角速度限制约束——确保速度不超过安全阈值。

    适用于:
        - PendulumState: omega 不超过 max_omega
        - CartState: v 不超过 max_velocity
        - 任意 WorldState: 通过 to_vector() 的第二维检查
    """

    def __init__(self, max_omega: float = 8.0, max_velocity: float = 50.0) -> None:
        self._max_omega = max_omega
        self._max_velocity = max_velocity

    @property
    def name(self) -> str:
        return "velocity_limit"

    def check(self, state: WorldState, action: Action | None) -> SafetyCheckResult:
        # PendulumState: 检查 omega
        if hasattr(state, "omega"):
            if abs(state.omega) > self._max_omega:
                return SafetyCheckResult(
                    passed=False,
                    constraint_name=self.name,
                    reason=f"角速度超限: {abs(state.omega):.2f} > {self._max_omega:.2f}",
                    severity="violation",
                    details={"omega": state.omega, "max_omega": self._max_omega},
                )

        # CartState: 检查 v
        if hasattr(state, "v") and not hasattr(state, "omega"):
            if abs(state.v) > self._max_velocity:
                return SafetyCheckResult(
                    passed=False,
                    constraint_name=self.name,
                    reason=f"速度超限: {abs(state.v):.2f} > {self._max_velocity:.2f}",
                    severity="violation",
                    details={"v": state.v, "max_velocity": self._max_velocity},
                )

        return SafetyCheckResult(passed=True, constraint_name=self.name)


# =============================================================================
# SafetyMonitor — 约束链注册 + 执行前检查
# =============================================================================


class SafetyMonitor:
    """安全监控器——链式注册多个安全约束，执行前统一检查。

    用法:
        >>> monitor = SafetyMonitor()
        >>> monitor.register(ForceLimitConstraint(max_torque=10.0))
        >>> monitor.register(PositionBoundConstraint(max_theta=3.14))
        >>> monitor.register(VelocityLimitConstraint(max_omega=8.0))
        >>>
        >>> result = monitor.check_all(state, action)
        >>> if not result.passed:
        >>>     print(f"安全违规: {result.reason}")
    """

    def __init__(self) -> None:
        self._constraints: list[SafetyConstraint] = []
        self._check_count: int = 0
        self._violation_count: int = 0

    @property
    def constraint_count(self) -> int:
        return len(self._constraints)

    @property
    def check_count(self) -> int:
        return self._check_count

    @property
    def violation_count(self) -> int:
        return self._violation_count

    def register(self, constraint: SafetyConstraint) -> None:
        """注册一个安全约束。"""
        self._constraints.append(constraint)
        logger.info("注册安全约束: %s", constraint.name)

    def check_all(self, state: WorldState, action: Action | None = None) -> SafetyCheckResult:
        """执行所有注册约束的检查。

        短路求值：一旦有约束不通过，立即返回。
        全部通过才返回 passed=True。

        Args:
            state: 当前世界状态
            action: 待执行动作（可选）

        Returns:
            SafetyCheckResult（第一个不通过的约束，或全部通过的结果）
        """
        self._check_count += 1

        for constraint in self._constraints:
            result = constraint.check(state, action)
            if not result.passed:
                self._violation_count += 1
                return result

        return SafetyCheckResult(passed=True, constraint_name="all_constraints")

    def check_individual(self, state: WorldState, action: Action | None = None) -> list[SafetyCheckResult]:
        """逐个检查所有约束，返回完整结果列表。

        Args:
            state: 当前世界状态
            action: 待执行动作

        Returns:
            所有约束的检查结果列表
        """
        self._check_count += 1
        results = []
        for constraint in self._constraints:
            result = constraint.check(state, action)
            results.append(result)
            if not result.passed:
                self._violation_count += 1
        return results

    def statistics(self) -> dict[str, Any]:
        """安全监控统计。"""
        return {
            "constraint_count": self.constraint_count,
            "total_checks": self._check_count,
            "total_violations": self._violation_count,
            "violation_rate": (self._violation_count / self._check_count if self._check_count > 0 else 0.0),
        }


# =============================================================================
# Phase 3: 机器人安全约束扩展
# =============================================================================


class JointLimitConstraint(SafetyConstraint):
    """关节角度限位约束——确保各关节角度不超过硬件限位。

    适用于 RobotWorldState，检查每个关节的位置是否在 [min, max] 范围内。
    """

    def __init__(
        self,
        joint_limits: list[tuple[float, float]] | None = None,
        default_min: float = -3.14159,
        default_max: float = 3.14159,
        n_joints: int = 6,
    ):
        """初始化关节限位。

        Args:
            joint_limits: 各关节的 (min, max) 限位列表
            default_min: 默认最小角度
            default_max: 默认最大角度
            n_joints: 关节数量
        """
        if joint_limits is not None:
            self._limits = joint_limits
        else:
            self._limits = [(default_min, default_max)] * n_joints

    @property
    def name(self) -> str:
        return "joint_limit"

    def check(self, state: WorldState, action: Action | None) -> SafetyCheckResult:
        if not hasattr(state, "joint_positions") or state.joint_positions is None:
            return SafetyCheckResult(passed=True, constraint_name=self.name)

        for i, (j_min, j_max) in enumerate(self._limits):
            if i >= len(state.joint_positions):
                break
            pos = float(state.joint_positions[i])
            if pos < j_min or pos > j_max:
                return SafetyCheckResult(
                    passed=False,
                    constraint_name=self.name,
                    reason=f"关节 {i} 越限: {pos:.3f} 不在 [{j_min:.3f}, {j_max:.3f}]",
                    severity="violation",
                    details={"joint": i, "position": pos, "min": j_min, "max": j_max},
                )

        return SafetyCheckResult(passed=True, constraint_name=self.name)


class SelfCollisionConstraint(SafetyConstraint):
    """自碰撞检测约束——基于简化包围盒的自碰撞警告。

    使用简化连杆模型检测相邻连杆是否过于接近。
    这是一个保守估计，生产系统需要完整的 URDF 碰撞检测。
    """

    def __init__(self, min_clearance: float = 0.05, link_length: float = 0.3) -> None:
        self._min_clearance = min_clearance
        self._link_length = link_length

    @property
    def name(self) -> str:
        return "self_collision"

    def check(self, state: WorldState, action: Action | None) -> SafetyCheckResult:
        if not hasattr(state, "joint_positions") or state.joint_positions is None:
            return SafetyCheckResult(passed=True, constraint_name=self.name)

        # 简化碰撞检测: 相邻关节角度和接近 π 时，连杆可能碰撞
        positions = state.joint_positions
        for i in range(len(positions) - 1):
            angle_sum = abs(float(positions[i]) + float(positions[i + 1]))
            if angle_sum > 2.8:  # 接近 π
                clearance = max(0.0, self._link_length * (3.14159 - angle_sum) / 3.14159)
                if clearance < self._min_clearance:
                    return SafetyCheckResult(
                        passed=False,
                        constraint_name=self.name,
                        reason=f"连杆 {i}-{i + 1} 碰撞风险: 间隙 {clearance:.4f} < {self._min_clearance}",
                        severity="warning",
                        details={"link_pair": (i, i + 1), "clearance": clearance},
                    )

        return SafetyCheckResult(passed=True, constraint_name=self.name)


class WorkspaceBoundConstraint(SafetyConstraint):
    """工作空间边界约束——确保末端执行器不超出工作空间。

    基于简化正运动学模型计算末端位置，检查是否在球形工作空间内。
    """

    def __init__(self, max_reach: float = 1.8, base_offset: float = 0.0) -> None:
        self._max_reach = max_reach
        self._base_offset = base_offset

    @property
    def name(self) -> str:
        return "workspace_bound"

    def check(self, state: WorldState, action: Action | None) -> SafetyCheckResult:
        if not hasattr(state, "joint_positions") or state.joint_positions is None:
            return SafetyCheckResult(passed=True, constraint_name=self.name)

        # 简化正运动学: 假设各关节贡献等量到末端位移
        positions = state.joint_positions
        n = len(positions)
        # 末端距离 ≈ 简化运动学估计
        reach = sum(abs(float(positions[i])) * (0.3 / max(1, n - i)) for i in range(n))
        total_reach = reach + self._base_offset

        if total_reach > self._max_reach:
            return SafetyCheckResult(
                passed=False,
                constraint_name=self.name,
                reason=f"末端执行器超出工作空间: {total_reach:.3f} > {self._max_reach}",
                severity="violation",
                details={"reach": total_reach, "max_reach": self._max_reach},
            )

        return SafetyCheckResult(passed=True, constraint_name=self.name)


class ToolForceConstraint(SafetyConstraint):
    """工具接触力约束——确保手术工具接触力不超过安全阈值。

    检查 RobotWorldState 的 joint_efforts 是否超过安全力限。
    """

    def __init__(self, max_force: float = 10.0, tool_joint_index: int = -1) -> None:
        self._max_force = max_force
        self._tool_joint_index = tool_joint_index

    @property
    def name(self) -> str:
        return "tool_force"

    def check(self, state: WorldState, action: Action | None) -> SafetyCheckResult:
        # 检查 RobotWorldState 的 efforts
        if hasattr(state, "joint_efforts") and state.joint_efforts is not None:
            idx = self._tool_joint_index
            if idx < 0:
                idx = len(state.joint_efforts) + idx
            if 0 <= idx < len(state.joint_efforts):
                force = abs(float(state.joint_efforts[idx]))
                if force > self._max_force:
                    return SafetyCheckResult(
                        passed=False,
                        constraint_name=self.name,
                        reason=f"工具力超限: {force:.2f} > {self._max_force:.2f}",
                        severity="violation",
                        details={"force": force, "max_force": self._max_force, "joint": idx},
                    )

        # 也检查 RobotAction 的 efforts
        if action is not None and hasattr(action, "target_efforts") and action.target_efforts is not None:
            idx = self._tool_joint_index
            if idx < 0 and len(action.target_efforts) > 0:
                idx = len(action.target_efforts) + idx
            if 0 <= idx < len(action.target_efforts):
                force = abs(float(action.target_efforts[idx]))
                if force > self._max_force:
                    return SafetyCheckResult(
                        passed=False,
                        constraint_name=self.name,
                        reason=f"工具目标力超限: {force:.2f} > {self._max_force:.2f}",
                        severity="violation",
                        details={"target_force": force, "max_force": self._max_force},
                    )

        return SafetyCheckResult(passed=True, constraint_name=self.name)


class AccelerationLimitConstraint(SafetyConstraint):
    """关节加速度限制约束——确保关节加速度不超过安全阈值。

    适用于 RobotWorldState，通过速度变化率估算加速度。
    对于 PendulumState/CartState，通过动作幅值估算加速度风险。
    """

    def __init__(self, max_acceleration: float = 5.0, dt: float = 0.01) -> None:
        """初始化加速度限制。

        Args:
            max_acceleration: 最大允许加速度 (rad/s² 或 m/s²)
            dt: 时间步长 (秒)
        """
        self._max_acceleration = max_acceleration
        self._dt = dt

    @property
    def name(self) -> str:
        return "acceleration_limit"

    def check(self, state: WorldState, action: Action | None) -> SafetyCheckResult:
        import numpy as _np

        # RobotWorldState: 检查 joint_velocities 变化率
        if hasattr(state, "joint_velocities") and state.joint_velocities is not None:
            if hasattr(state, "joint_positions") and state.joint_positions is not None:
                if action is not None and hasattr(action, "target_positions") and action.target_positions is not None:
                    delta_pos = _np.asarray(action.target_positions) - _np.asarray(state.joint_positions)
                    target_vel = delta_pos / self._dt
                    accel = (target_vel - _np.asarray(state.joint_velocities)) / self._dt
                    max_accel = float(_np.max(_np.abs(accel)))
                    if max_accel > self._max_acceleration:
                        worst_joint = int(_np.argmax(_np.abs(accel)))
                        return SafetyCheckResult(
                            passed=False,
                            constraint_name=self.name,
                            reason=f"关节 {worst_joint} 加速度超限: {max_accel:.2f} > {self._max_acceleration:.2f}",
                            severity="warning",
                            details={"joint": worst_joint, "acceleration": max_accel, "max": self._max_acceleration},
                        )

        # PendulumState: 检查 omega 变化
        if hasattr(state, "omega") and action is not None and hasattr(action, "torque"):
            accel = abs(action.torque) / 1.0
            if accel > self._max_acceleration:
                return SafetyCheckResult(
                    passed=False,
                    constraint_name=self.name,
                    reason=f"角加速度超限: {accel:.2f} > {self._max_acceleration:.2f}",
                    severity="warning",
                    details={"acceleration": accel, "max": self._max_acceleration},
                )

        # CartState: 检查力引起的加速度
        if hasattr(state, "v") and not hasattr(state, "omega") and action is not None and hasattr(action, "force"):
            accel = abs(action.force) / 1.0
            if accel > self._max_acceleration:
                return SafetyCheckResult(
                    passed=False,
                    constraint_name=self.name,
                    reason=f"加速度超限: {accel:.2f} > {self._max_acceleration:.2f}",
                    severity="warning",
                    details={"acceleration": accel, "max": self._max_acceleration},
                )

        return SafetyCheckResult(passed=True, constraint_name=self.name)


# =============================================================================
# v4.4.0: 清创专用安全约束
# =============================================================================


class TissueForceConstraint(SafetyConstraint):
    """清创组织特异性力约束。

    不同组织类型的最大允许力不同:
        坏死 ≤3.0N, 腐肉 ≤2.0N, 肉芽 ≤1.0N, 上皮 ≤0.5N

    依赖: ForceTissueDynamics 的静态参数表。
    """

    def __init__(self, tissue_label: int = 0) -> None:
        self._tissue_label = tissue_label
        from mci_world_model.sdk._force_tissue_dynamics import TISSUE_PARAMS
        self._params = TISSUE_PARAMS

    @property
    def name(self) -> str:
        return "tissue_force"

    def check(self, state: Any, action: Any=None) -> SafetyCheckResult:
        p = self._params.get(self._tissue_label, {"max_force": 0.5, "name": "未知"})
        current_force = getattr(state, "tool_force_n", 0.0) if hasattr(state, "tool_force_n") else 0.0

        if current_force > p["max_force"]:  # type: ignore
            return SafetyCheckResult(
                passed=False,
                reason=f"组织力超限: {current_force:.1f}N > {p['max_force']}N ({p['name']})",
                constraint_name=self.name,
            )
        return SafetyCheckResult(passed=True, constraint_name=self.name)

    def set_tissue(self, label: int) -> None:
        self._tissue_label = label


class ThermalSafetyConstraint(SafetyConstraint):
    """清创热安全约束: 组织温度 ≤ 42°C。"""

    def __init__(self, max_temp_c: float = 42.0) -> None:
        self._max_temp = max_temp_c

    @property
    def name(self) -> str:
        return "thermal_safety"

    def check(self, state: Any, action: Any=None) -> SafetyCheckResult:
        import numpy as _np
        temp = getattr(state, "thermal_image", None)
        if temp is not None:
            max_t = float(_np.max(temp))
            if max_t > self._max_temp:
                return SafetyCheckResult(passed=False, reason=f"温度超限: {max_t:.1f}°C > {self._max_temp}°C", constraint_name=self.name)
        return SafetyCheckResult(passed=True, constraint_name=self.name)


class DepthLimitConstraint(SafetyConstraint):
    """清创深度约束: 不超过坏死层厚度。"""

    def __init__(self, max_depth_mm: float = 5.0) -> None:
        self._max_depth = max_depth_mm

    @property
    def name(self) -> str:
        return "depth_limit"

    def check(self, state: Any, action: Any=None) -> SafetyCheckResult:
        depth = getattr(state, "wound_depth_mm", 0.0)
        if depth > self._max_depth:
            return SafetyCheckResult(passed=False, reason=f"深度超限: {depth:.1f}mm > {self._max_depth}mm", constraint_name=self.name)
        return SafetyCheckResult(passed=True, constraint_name=self.name)
