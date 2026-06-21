from __future__ import annotations

from typing import Any

"""MCI World Model v4.5.0 — RobotWorldState + RobotAction
========================================================

手术机器人桥接预研——6-DOF 机械臂状态空间和动作空间，
为 ROS2 桥接和 CEWM 闭环驱动机器人做准备。

重要声明: Phase 3 交付的是桥接原型和架构验证，不是生产级手术机器人控制系统。

核心能力:
    RobotWorldState  — 6-DOF 关节状态 (joint_positions + joint_velocities + joint_efforts)
    RobotAction      — N-DOF 关节目标动作
"""


from dataclasses import dataclass

import numpy as np

from mci_world_model.sdk._world_state import Action, WorldState

# =============================================================================
# RobotWorldState — 6-DOF 机械臂状态
# =============================================================================


@dataclass
class RobotWorldState(WorldState):
    """机器人关节状态——6-DOF 机械臂的状态空间。

    状态空间:
        joint_positions:  [j1, j2, j3, j4, j5, j6] (rad)
        joint_velocities: [v1, v2, v3, v4, v5, v6] (rad/s)
        joint_efforts:    [e1, e2, e3, e4, e5, e6] (N·m)

    to_vector() 返回 (N_joints * 3,) 向量。
    """

    joint_positions: np.ndarray | None = None  # (6,) rad
    joint_velocities: np.ndarray | None = None  # (6,) rad/s
    joint_efforts: np.ndarray | None = None  # (6,) N·m
    n_joints: int = 6
    dt: float = 0.01

    def _ensure_arrays(self) -> None:
        n = self.n_joints
        if self.joint_positions is None:
            self.joint_positions = np.zeros(n, dtype=np.float64)
        if self.joint_velocities is None:
            self.joint_velocities = np.zeros(n, dtype=np.float64)
        if self.joint_efforts is None:
            self.joint_efforts = np.zeros(n, dtype=np.float64)

    def to_vector(self) -> np.ndarray:
        self._ensure_arrays()
        return np.concatenate(
            [
                self.joint_positions,
                self.joint_velocities,
                self.joint_efforts,
            ]
        ).astype(np.float64)

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> RobotWorldState:
        vec = np.asarray(vec, dtype=np.float64)
        n = len(vec) // 3
        return cls(
            joint_positions=vec[:n],
            joint_velocities=vec[n : 2 * n],
            joint_efforts=vec[2 * n : 3 * n],
            n_joints=n,
        )

    def distance(self, other: WorldState) -> float:
        if not isinstance(other, RobotWorldState):
            if hasattr(other, "to_vector"):
                return float(np.linalg.norm(self.to_vector() - other.to_vector()))
            return float("inf")
        return float(np.linalg.norm(self.to_vector() - other.to_vector()))

    def copy(self) -> RobotWorldState:
        return RobotWorldState(
            joint_positions=self.joint_positions.copy() if self.joint_positions is not None else None,
            joint_velocities=self.joint_velocities.copy() if self.joint_velocities is not None else None,
            joint_efforts=self.joint_efforts.copy() if self.joint_efforts is not None else None,
            n_joints=self.n_joints,
            dt=self.dt,
        )

    def step_physics(self) -> RobotWorldState:
        """简单 Euler 演化（无动力学，仅位置+=速度*dt）。"""
        self._ensure_arrays()
        new_pos = self.joint_positions + self.joint_velocities * self.dt  # type: ignore
        return RobotWorldState(
            joint_positions=new_pos,  # type: ignore
            joint_velocities=self.joint_velocities.copy(),  # type: ignore
            joint_efforts=self.joint_efforts.copy(),  # type: ignore
            n_joints=self.n_joints,
            dt=self.dt,
        )

    def to_dict(self) -> dict[str, Any]:
        self._ensure_arrays()
        return {
            "type": "RobotWorldState",
            "n_joints": self.n_joints,
            "joint_positions": self.joint_positions.tolist(),  # type: ignore
            "joint_velocities": self.joint_velocities.tolist(),  # type: ignore
            "joint_efforts": self.joint_efforts.tolist(),  # type: ignore
        }


# =============================================================================
# RobotAction — N-DOF 关节目标动作
# =============================================================================


@dataclass
class RobotAction(Action):
    """机器人关节动作——N-DOF 关节目标位置/力矩。

    动作空间:
        target_positions: 目标关节位置 (rad)
        target_efforts:   目标关节力矩 (N·m) (可选)
    """

    target_positions: np.ndarray | None = None
    target_efforts: np.ndarray | None = None
    dt: float = 0.01

    def apply(self, state: WorldState) -> RobotWorldState:
        if not isinstance(state, RobotWorldState):
            raise TypeError("RobotAction 只能作用于 RobotWorldState")
        state._ensure_arrays()

        if self.target_positions is not None:
            # 简单一步到位（无轨迹规划）
            new_vel = (self.target_positions - state.joint_positions) / self.dt
            return RobotWorldState(
                joint_positions=self.target_positions.copy(),
                joint_velocities=new_vel,
                joint_efforts=self.target_efforts.copy()
                if self.target_efforts is not None
                else state.joint_efforts.copy(),  # type: ignore
                n_joints=state.n_joints,
                dt=state.dt,
            )
        return state.copy()

    def to_vector(self) -> np.ndarray:
        parts = []
        if self.target_positions is not None:
            parts.append(np.asarray(self.target_positions, dtype=np.float64))
        if self.target_efforts is not None:
            parts.append(np.asarray(self.target_efforts, dtype=np.float64))
        if not parts:
            return np.zeros(1, dtype=np.float64)
        return np.concatenate(parts)

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> RobotAction:
        n = len(vec) // 2
        return cls(
            target_positions=vec[:n],
            target_efforts=vec[n:] if len(vec) > n else None,
        )
