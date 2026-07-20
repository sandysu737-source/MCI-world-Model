"""
MCI World Model v4.6.0 — 感知-行动接口范式
===========================================

MCI 世界模型（大脑）对外部世界的输入输出契约。

MCI 不关心传感器是通过 USB 还是 I²C 连接的，不关心执行器是舵机还是继电器。
它只定义三类抽象契约:

    1. 信号契约 (SignalContract)      — 外部世界 → MCI 的输入格式
    2. 动作契约 (ActionContract)      — MCI → 外部世界的输出格式
    3. 反馈契约 (FeedbackContract)    — 动作执行后 → MCI 的再输入

上层智能体/躯壳层负责:
    - 将真实硬件信号适配为 PhysicalSignal
    - 将 ActionCommand 翻译为具体硬件指令
    - 将执行结果 (ActionResult) 注入回感知管道

与现有模块的关系:
    _perception_pipeline.py:  ProcessMultimodal(signals) —— 消费 PhysicalSignal
    _world_state.py (未来):    WorldState —— PhysicalSignal 编码的目标
    _jepa_predictor.py:       JEPAPredictor —— 预测的输入源
    _causal_actor.py:         CausalActor —— 动作的产生者
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# ═══════════════════════════════════════════════════════════════════════════
# 一、信号契约 — 外部世界 → MCI
# ═══════════════════════════════════════════════════════════════════════════


class SensorModality(Enum):
    """六感感官模态 —— MCI 能理解的外部世界信号种类。"""

    VISION = "vision"
    AUDITION = "audition"
    TACTITION = "tactition"
    OLFACTION = "olfaction"
    GUSTATION = "gustation"
    PROPRIOCEPTION = "proprioception"


class SignalSubType(Enum):
    """
    每类感官下的具体信号格式 —— MCI 期望的数据形状。

    上层躯壳层必须将硬件原始读数映射到这些标准子类型之一。
    """

    # 视觉
    RGB_FRAME = "rgb_frame"  # (H, W, 3) uint8
    DEPTH_FRAME = "depth_frame"  # (H, W, 1) float32
    THERMAL_FRAME = "thermal_frame"  # (H, W, 1) float32

    # 听觉
    AUDIO_WAVEFORM = "audio_waveform"  # (N_samples, N_channels) float32
    ULTRASOUND_RANGE = "ultrasound_range"  # float (米)

    # 触觉
    PRESSURE_VALUE = "pressure_value"  # float (N)
    TEMPERATURE_VALUE = "temperature_value"  # float (°C)
    HUMIDITY_VALUE = "humidity_value"  # float (%RH)

    # 嗅觉
    GAS_MULTICHANNEL = "gas_multichannel"  # dict[str, float] (ppm)
    PM25_VALUE = "pm25_value"  # float (μg/m³)

    # 味觉
    PH_LEVEL = "ph_level"  # float
    TDS_LEVEL = "tds_level"  # float (ppm)

    # 本体感觉
    IMU_9AXIS = "imu_9axis"  # (accel_3, gyro_3, mag_3)
    GNSS_FIX = "gnss_fix"  # (lat, lon, alt, speed)
    ENCODER_POSITION = "encoder_position"  # float (rad or m)
    POWER_STATE = "power_state"  # (voltage, current, remaining_pct)


@dataclass
class PhysicalSignal:
    """
    MCI 接收的标准化物理信号 —— 六感输入的通用载体。

    上层躯壳层将传感器原始数据封装为此格式后，
    直接送入 PerceptionPipeline.process_multimodal()。

    与现有 MultimodalSignal 的关系:
        MultimodalSignal 定义数据处理方式 (NUMERICAL/TEMPORAL_SERIES/...)
        PhysicalSignal 定义数据来自哪个感官、哪种物理量
        两者正交 —— PhysicalSignal.to_multimodal() 做桥接
    """

    modality: SensorModality
    sub_type: SignalSubType
    value: object  # float | np.ndarray | dict[str, float]
    timestamp: float = 0.0  # Unix 秒
    source_id: str = ""  # 来源设备标识
    confidence: float = 1.0  # [0, 1]
    metadata: dict = field(default_factory=dict)

    def to_multimodal(self):
        """桥接到 v3.1.0 的 MultimodalSignal 格式。"""
        from mci_world_model._sys._perception_pipeline import MultimodalSignal, SignalType

        _subtype_to_signaltype = {
            SignalSubType.PRESSURE_VALUE: SignalType.NUMERICAL,
            SignalSubType.TEMPERATURE_VALUE: SignalType.NUMERICAL,
            SignalSubType.HUMIDITY_VALUE: SignalType.NUMERICAL,
            SignalSubType.PH_LEVEL: SignalType.NUMERICAL,
            SignalSubType.TDS_LEVEL: SignalType.NUMERICAL,
            SignalSubType.ULTRASOUND_RANGE: SignalType.NUMERICAL,
            SignalSubType.PM25_VALUE: SignalType.NUMERICAL,
            SignalSubType.ENCODER_POSITION: SignalType.NUMERICAL,  # v3.3.0
            SignalSubType.AUDIO_WAVEFORM: SignalType.TEMPORAL_SERIES,
            SignalSubType.IMU_9AXIS: SignalType.LAB_STRUCTURED,
            SignalSubType.GNSS_FIX: SignalType.LAB_STRUCTURED,
            SignalSubType.RGB_FRAME: SignalType.IMAGE,  # v3.3.0
            SignalSubType.DEPTH_FRAME: SignalType.IMAGE,  # v3.3.0
            SignalSubType.THERMAL_FRAME: SignalType.IMAGE,  # v3.3.0
            SignalSubType.GAS_MULTICHANNEL: SignalType.LAB_STRUCTURED,
            SignalSubType.POWER_STATE: SignalType.LAB_STRUCTURED,
        }

        st = _subtype_to_signaltype.get(self.sub_type, SignalType.NUMERICAL)
        return MultimodalSignal(
            signal_type=st,
            value=self.value,
            timestamp=str(self.timestamp),
            source=f"{self.source_id}:{self.modality.value}",
            metadata={**self.metadata, "sub_type": self.sub_type.value, "confidence": self.confidence},
        )


# ═══════════════════════════════════════════════════════════════════════════
# 二、动作契约 — MCI → 外部世界
# ═══════════════════════════════════════════════════════════════════════════


class ActuatorChannel(Enum):
    """六路输出通道 —— MCI 能发出的动作种类。"""

    DISPLAY = "display"  # 视觉输出 (文字/图像/仪表盘)
    AUDIO_OUT = "audio_out"  # 声音输出 (TTS/警告音)
    ACTUATION = "actuation"  # 物理执行 (开关/运动/力)
    ALERTING = "alerting"  # 数字告警 (Webhook/Email/MQTT)
    API_CALL = "api_call"  # 外部API (REST/I²C/SPI)
    ROBOTIC = "robotic"  # 机器人本体 (导航/操作)


class ActionPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class ActionCommand:
    """
    MCI 发出的标准化动作命令 —— 六路输出的通用载体。

    上层躯壳层负责将此类命令翻译为具体硬件指令:
        ActionCommand("switch_on", {"relay": 1})
        → RPi.GPIO.output(17, HIGH)
    """

    channel: ActuatorChannel
    command: str  # "show_text" | "speak" | "switch_on" | "move_to" | ...
    params: dict = field(default_factory=dict)  # {"text": "...", "speed": 0.5, ...}
    priority: ActionPriority = ActionPriority.NORMAL
    action_id: str = ""  # 唯一ID，用于追踪
    metadata: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# 三、反馈契约 — 动作执行后 → MCI (闭环)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ActionResult:
    """
    动作执行反馈 —— 闭环的关键。

    躯壳层执行 ActionCommand 后必须返回此结构。
    side_effects 中包含执行后捕获到的传感器变化，
    会被注入回 SignalBus → PerceptionPipeline → WorldState，
    形成 感知→认知→预测→行动→新感知 的完整闭环。

    示例:
        执行 "开灯" → ActionResult(
            success=True,
            side_effects=[PhysicalSignal(VISION, RGB_FRAME, brighter_frame)]
        )
    """

    action_id: str
    success: bool
    error_message: str = ""
    actual_value: object = None  # 实际执行值
    side_effects: list[PhysicalSignal] = field(default_factory=list)
    latency_ms: float = 0.0
    timestamp: float = 0.0
