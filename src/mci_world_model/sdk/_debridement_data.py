from __future__ import annotations

"""
MCI World Model v4.4.0 — Debridement Data Pipeline
====================================================

AI 智能清创机器人数据采集与增强管线。

支持三数据源:
1. 物理模拟器 (Pendulum/Cart/DoublePendulum/Robot)
2. 合成伤口数据集 (程序化生成六模态清创帧)
3. 公开伤口数据集格式适配

输出: DebridementSample (六模态单帧统一格式)

六模态:
    M1 - RGB 视觉 (224×224×3 创面图像)
    M2 - 深度图   (224×224 距离, mm)
    M3 - 热成像   (224×224 温度, °C)
    M4 - 力触觉   (6D: fx,fy,fz,tx,ty,tz)
    M5 - 本体感知 (21D: 7-DOF pos+vel+effort)
    M6 - 临床元数据 (伤口类型, 阶段, 患者信息)
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# DebridementSample — 清创数据单帧
# =============================================================================


@dataclass
class DebridementSample:
    """单帧清创多模态数据样本。

    Attributes:
        rgb_image:         (224, 224, 3) uint8  创面 RGB 图像
        depth_image:       (224, 224)    float32 深度图 (mm)
        thermal_image:     (224, 224)    float32 热成像 (°C)
        force_torque:      (6,)          float32 力/力矩
        joint_positions:   (7,)          float32 关节位置 (rad)
        joint_velocities:  (7,)          float32 关节速度 (rad/s)
        joint_efforts:     (7,)          float32 关节力矩 (N·m)
        tissue_label:      int                   组织标签 (0=坏死 1=腐肉 2=肉芽 3=上皮)
        wound_depth_mm:    float                 创面深度
        surgical_phase:    int                   手术相 (0=探查 1=清创 2=止血 3=验证)
        tool_force_n:      float                 工具施加力 (N)
        tool_velocity:     float                 工具速度 (mm/s)
        sample_id:         str                   样本唯一 ID
    """

    # 视觉模态
    rgb_image: np.ndarray = field(
        default_factory=lambda: np.zeros((224, 224, 3), dtype=np.uint8)
    )
    depth_image: np.ndarray = field(
        default_factory=lambda: np.zeros((224, 224), dtype=np.float32)
    )
    thermal_image: np.ndarray = field(
        default_factory=lambda: np.full((224, 224), 37.0, dtype=np.float32)
    )

    # 力触觉
    force_torque: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )

    # 本体感知 (7-DOF)
    joint_positions: np.ndarray = field(
        default_factory=lambda: np.zeros(7, dtype=np.float32)
    )
    joint_velocities: np.ndarray = field(
        default_factory=lambda: np.zeros(7, dtype=np.float32)
    )
    joint_efforts: np.ndarray = field(
        default_factory=lambda: np.zeros(7, dtype=np.float32)
    )

    # 临床标注
    tissue_label: int = 0  # 0=坏死 1=腐肉 2=肉芽 3=上皮
    wound_depth_mm: float = 0.0
    surgical_phase: int = 0  # 0=探查 1=清创 2=止血 3=验证
    tool_force_n: float = 0.0
    tool_velocity: float = 0.0
    sample_id: str = ""

    # ── TISSUE_LABEL 常量 ──
    TISSUE_NECROTIC: int = 0
    TISSUE_SLOUGH: int = 1
    TISSUE_GRANULATION: int = 2
    TISSUE_EPITHELIAL: int = 3

    TISSUE_NAMES: dict[str, Any] = field(
        default_factory=lambda: {0: "坏死", 1: "腐肉", 2: "肉芽", 3: "上皮"},
        init=False,
        repr=False,
    )

    # ── SURGICAL_PHASE 常量 ──
    PHASE_EXPLORE: int = 0
    PHASE_DEBRIDE: int = 1
    PHASE_HEMOSTASIS: int = 2
    PHASE_VERIFY: int = 3

    PHASE_NAMES: dict[str, Any] = field(
        default_factory=lambda: {0: "探查", 1: "清创", 2: "止血", 3: "验证"},
        init=False,
        repr=False,
    )

    @property
    def tissue_name(self) -> str:
        return self.TISSUE_NAMES.get(self.tissue_label, "未知")

    @property
    def phase_name(self) -> str:
        return self.PHASE_NAMES.get(self.surgical_phase, "未知")

    def to_vector(self) -> np.ndarray:
        """将所有模态拼接为统一向量 (用于模型输入)。"""
        rgb_flat = self.rgb_image.astype(np.float32).ravel() / 255.0
        depth_flat = self.depth_image.ravel() / 200.0  # 归一化到 ~[0,1]
        thermal_flat = (self.thermal_image.ravel() - 30.0) / 15.0  # 归一化

        proprio = np.concatenate(
            [self.joint_positions, self.joint_velocities, self.joint_efforts]
        )

        return np.concatenate(
            [
                rgb_flat,
                depth_flat,
                thermal_flat,
                self.force_torque,
                proprio,
            ]
        ).astype(np.float32)

    @property
    def n_features(self) -> int:
        """特征向量总维度。"""
        return len(self.to_vector())

    def copy(self) -> DebridementSample:
        return DebridementSample(
            rgb_image=self.rgb_image.copy(),
            depth_image=self.depth_image.copy(),
            thermal_image=self.thermal_image.copy(),
            force_torque=self.force_torque.copy(),
            joint_positions=self.joint_positions.copy(),
            joint_velocities=self.joint_velocities.copy(),
            joint_efforts=self.joint_efforts.copy(),
            tissue_label=self.tissue_label,
            wound_depth_mm=self.wound_depth_mm,
            surgical_phase=self.surgical_phase,
            tool_force_n=self.tool_force_n,
            tool_velocity=self.tool_velocity,
            sample_id=self.sample_id,
        )


# =============================================================================
# SyntheticDebridementGenerator — 合成数据生成
# =============================================================================


class SyntheticDebridementGenerator:
    """程序化合成清创训练数据。

    无需真实伤口图像——生成具有物理合理性的模拟数据。
    用途: 编码器预训练 / 模型架构验证 / 单元测试。

    物理合理性:
    - 坏死组织: RGB 暗色 (黑/棕), 温度低 (30-34°C), 低阻力 (0.5-2N)
    - 腐肉:     RGB 黄色, 温度中 (33-36°C), 中阻力 (1-3N)
    - 肉芽:     RGB 红色, 温度高 (35-38°C), 高阻力 (2-5N)
    - 上皮:     RGB 粉色, 温度正常 (36-37°C), 极高阻力 (>5N)
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.RandomState(seed)
        self._counter = 0

    def generate_sample(self, tissue_label: int = -1) -> DebridementSample:
        """生成单个随机清创样本。

        Args:
            tissue_label: -1=随机, 0=坏死, 1=腐肉, 2=肉芽, 3=上皮
        """
        if tissue_label < 0:
            tissue_label = self._rng.randint(0, 4)

        # 组织特异性参数
        params = self._tissue_params(tissue_label)

        # 生成模态数据
        rgb = self._generate_rgb(tissue_label)
        depth = self._generate_depth(params["depth_range"])
        thermal = self._generate_thermal(params["temp_range"])

        ft = self._generate_force_torque(params["force_range"])

        jpos = self._rng.uniform(-np.pi * 0.5, np.pi * 0.5, 7).astype(np.float32)
        jvel = self._rng.uniform(-0.5, 0.5, 7).astype(np.float32)
        jeff = self._rng.uniform(-2.0, 2.0, 7).astype(np.float32)

        phase = self._rng.choice([0, 1, 2, 3], p=[0.2, 0.5, 0.2, 0.1])

        self._counter += 1
        return DebridementSample(
            rgb_image=rgb,
            depth_image=depth,
            thermal_image=thermal,
            force_torque=ft,
            joint_positions=jpos,
            joint_velocities=jvel,
            joint_efforts=jeff,
            tissue_label=tissue_label,
            wound_depth_mm=params["depth"],
            surgical_phase=phase,
            tool_force_n=float(ft[2]),  # fz
            tool_velocity=float(np.linalg.norm(jvel[:3])),
            sample_id=f"syn_{self._counter:06d}",
        )

    def generate_batch(self, n: int, balanced: bool = True) -> list[DebridementSample]:
        """批量生成合成样本。

        Args:
            n: 样本数
            balanced: 是否四类均衡
        """
        if balanced:
            per_class = n // 4
            samples = []
            for label in range(4):
                for _ in range(per_class):
                    samples.append(self.generate_sample(label))
            # 填充剩余
            for _ in range(n - len(samples)):
                samples.append(self.generate_sample())
            self._rng.shuffle(samples)
            return samples
        return [self.generate_sample() for _ in range(n)]

    def _tissue_params(self, label: int) -> dict[str, Any]:
        """组织特异性物理参数。"""
        params = {
            0: {"color": (40, 20, 10), "temp_range": (30, 34), "depth_range": (2, 8), "force_range": (0.5, 2.0), "depth": 5.0},
            1: {"color": (180, 160, 80), "temp_range": (33, 36), "depth_range": (1, 5), "force_range": (1.0, 3.0), "depth": 3.0},
            2: {"color": (200, 60, 50), "temp_range": (35, 38), "depth_range": (0.5, 3), "force_range": (2.0, 5.0), "depth": 1.5},
            3: {"color": (230, 180, 170), "temp_range": (36, 37.5), "depth_range": (0, 0.5), "force_range": (4.0, 10.0), "depth": 0.2},
        }
        return params[label]

    def _generate_rgb(self, tissue_label: int) -> np.ndarray:
        """生成合成创面 RGB 图像。"""
        base = np.array(self._tissue_params(tissue_label)["color"], dtype=np.uint8)
        img = np.full((224, 224, 3), base, dtype=np.uint8)

        # 加纹理噪声
        noise = self._rng.randint(-20, 20, (224, 224, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 中心"创面区域" (圆形, 不同组织颜色)
        cy, cx = 112, 112
        radius = self._rng.randint(50, 100)
        y, x = np.ogrid[:224, :224]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2

        wound_color = np.array(self._tissue_params(tissue_label)["color"], dtype=np.uint8)
        for c in range(3):
            img[:, :, c][mask] = np.clip(
                wound_color[c] + self._rng.randint(-15, 15, mask.sum()), 0, 255
            ).astype(np.uint8)

        return img

    def _generate_depth(self, depth_range: tuple) -> np.ndarray:
        """生成合成深度图 (创面凹陷)。"""
        dmin, dmax = depth_range
        depth = np.full((224, 224), 2.0, dtype=np.float32)  # 背景 2mm

        # 中心创面凹陷
        cy, cx = 112, 112
        radius = self._rng.randint(40, 90)
        y, x = np.ogrid[:224, :224]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2

        # 深度从中心向外递减
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max(radius, 1)
        depth_val = self._rng.uniform(dmin, dmax)
        depth[mask] = depth_val * (1.0 - r[mask] * 0.3)
        depth += self._rng.randn(224, 224).astype(np.float32) * 0.3

        return depth

    def _generate_thermal(self, temp_range: tuple) -> np.ndarray:
        """生成合成热成像 (温度分布)。"""
        tmin, tmax = temp_range
        baseline = 36.0  # 正常体温

        thermal = np.full((224, 224), baseline, dtype=np.float32)

        # 创面区域温度异常
        cy, cx = 112, 112
        radius = self._rng.randint(40, 90)
        y, x = np.ogrid[:224, :224]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2

        wound_temp = self._rng.uniform(tmin, tmax)
        thermal[mask] = wound_temp

        # 传感器噪声
        thermal += self._rng.randn(224, 224).astype(np.float32) * 0.5

        return thermal

    def _generate_force_torque(self, force_range: tuple) -> np.ndarray:
        """生成合成力/力矩数据。"""
        fmin, fmax = force_range
        return self._rng.uniform(fmin, fmax, 6).astype(np.float32)


# =============================================================================
# 公开数据集适配器 (占位)
# =============================================================================


class WoundDatasetAdapter:
    """公开伤口数据集 → DebridementSample 适配器。

    待适配数据集:
    - Medetec Wound Database
    - AZH Wound Database
    - WoundDB (chronic wounds)

    当前仅提供接口定义。
    """

    @staticmethod
    def from_medetec(image: np.ndarray, label: int) -> DebridementSample:
        """从 Medetec 格式转换。"""
        sample = DebridementSample()
        sample.rgb_image = image
        sample.tissue_label = label
        sample.sample_id = f"medetec_{hash(str(image.shape))}"
        return sample

    @staticmethod
    def from_dict(data: dict[str, Any]) -> DebridementSample:
        """从字典恢复 DebridementSample。"""
        return DebridementSample(
            rgb_image=np.array(data.get("rgb", np.zeros((224, 224, 3)))),
            depth_image=np.array(data.get("depth", np.zeros((224, 224)))),
            thermal_image=np.array(data.get("thermal", np.full((224, 224), 37.0))),
            force_torque=np.array(data.get("ft", np.zeros(6))),
            joint_positions=np.array(data.get("jpos", np.zeros(7))),
            joint_velocities=np.array(data.get("jvel", np.zeros(7))),
            joint_efforts=np.array(data.get("jeff", np.zeros(7))),
            tissue_label=int(data.get("label", 0)),
            wound_depth_mm=float(data.get("depth", 0)),
            surgical_phase=int(data.get("phase", 0)),
            tool_force_n=float(data.get("force", 0)),
            tool_velocity=float(data.get("vel", 0)),
            sample_id=str(data.get("id", "")),
        )
