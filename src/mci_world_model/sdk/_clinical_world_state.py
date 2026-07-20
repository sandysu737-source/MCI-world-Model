"""MCI World Model — 临床世界状态空间（PatientState + MedicalAction）

============================================================

Phase 0 地基模块：为医疗智能体世界模型定义状态空间 S 和动作空间 A。

这是世界模型五要素（S, A, T, R, π）的前两个，是转移模型 T 的前提。

核心类:
    PatientState    — 异构临床时序世界状态（继承 WorldState）
    MedicalAction   — 临床干预动作（继承 Action）
    Medication      — 用药记录数据结构
    ClinicalRanges  — 体征生理正常范围（用于安全约束）

设计原则:
    - 继承 WorldState/Action ABC，无缝融入已有世界模型闭环
    - 异构数据融合：连续体征 + 离散检验 + 事件用药 + 类别诊断
    - 无状态：纯数据结构，不持久化（记忆归 su-memory-sdk）
    - 可审计：to_dict() 输出完整可追溯信息
    - 安全约束：体征生理范围硬编码，is_safe() 用于规划器剪枝

与 su-memory-sdk 的边界:
    本模块是纯数据结构 + 无状态计算，不涉及记忆存储/检索/持久化。
    跨调用的经验/案例库归 su-memory-sdk，通过 adapters/ 桥接。
"""

from __future__ import annotations

import copy as _copy
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mci_world_model.sdk._world_state import Action, WorldState

# =============================================================================
# 常量定义 — 7 维核心生命体征（与 ClinicalQuery.patient_state 对齐）
# =============================================================================

#: 体征变量名（顺序固定，索引对齐）
VITAL_NAMES: list[str] = [
    "heart_rate",  # 心率 bpm
    "systolic_bp",  # 收缩压 mmHg
    "diastolic_bp",  # 舒张压 mmHg
    "oxygen_saturation",  # 血氧饱和度 %
    "respiratory_rate",  # 呼吸频率 /min
    "temperature",  # 体温 °C
    "gcs",  # 格拉斯哥昏迷评分 3-15
]

#: 体征维度数
N_VITALS: int = len(VITAL_NAMES)

#: 体征单位（用于显示和审计）
VITAL_UNITS: dict[str, str] = {
    "heart_rate": "bpm",
    "systolic_bp": "mmHg",
    "diastolic_bp": "mmHg",
    "oxygen_saturation": "%",
    "respiratory_rate": "/min",
    "temperature": "°C",
    "gcs": "score",
}

#: 体征生理正常范围 [min, max]（用于安全约束 is_safe）
#: 来源：临床常规参考值（教科书级共识）
VITAL_NORMAL_RANGES: dict[str, tuple[float, float]] = {
    "heart_rate": (60.0, 100.0),
    "systolic_bp": (90.0, 140.0),
    "diastolic_bp": (60.0, 90.0),
    "oxygen_saturation": (95.0, 100.0),
    "respiratory_rate": (12.0, 20.0),
    "temperature": (36.0, 37.5),
    "gcs": (13.0, 15.0),
}

#: 体征生理可行范围 [min, max]（超出即为不可信/致命）
#: 来源：临床极端值（用于 is_physiologically_valid 硬约束）
VITAL_FEASIBLE_RANGES: dict[str, tuple[float, float]] = {
    "heart_rate": (20.0, 220.0),
    "systolic_bp": (40.0, 250.0),
    "diastolic_bp": (20.0, 150.0),
    "oxygen_saturation": (50.0, 100.0),
    "respiratory_rate": (4.0, 50.0),
    "temperature": (33.0, 42.0),
    "gcs": (3.0, 15.0),
}


# =============================================================================
# Medication — 用药记录
# =============================================================================


@dataclass
class Medication:
    """用药记录数据结构。

    Attributes:
        name: 药物名称（通用名，如 "多巴胺" / "dopamine"）。
        dose: 剂量数值。
        unit: 剂量单位（如 "μg/kg/min", "mg", "mL/h"）。
        route: 给药途径（"IV" 静脉 / "PO" 口服 / "IM" 肌注 / "SC" 皮下）。
        start_time: 给药开始时间戳（Unix epoch）。
    """

    name: str
    dose: float
    unit: str = ""
    route: str = "IV"
    start_time: float = field(default_factory=time.time)


# =============================================================================
# PatientState — 患者临床世界状态
# =============================================================================


@dataclass
class PatientState(WorldState):
    """患者临床状态 — 异构时序世界状态。

    继承 WorldState，融入 MCI 世界模型闭环：
        to_vector() / from_vector() → JEPA 编码器接口
        distance()                   → Cost 模块评估预测误差
        copy()                       → Actor what-if 推演

    异构数据融合：
        - vital_signs: 连续体征时序矩阵 (T, 7)，T=时间窗数
        - lab_values:  离散检验值字典（最新快照）
        - medications: 当前用药列表
        - diagnoses:   诊断标签（ICD-10 代码）

    无状态设计：纯数据容器，不持有跨调用状态。记忆/经验归 su-memory-sdk。
    """

    #: 连续体征时序矩阵 (T, 7)，每行一个时间窗，7 列对应 VITAL_NAMES
    vital_signs: np.ndarray = field(default_factory=lambda: np.zeros((1, N_VITALS)))

    #: 离散检验值（最新快照）：检验项名 → 数值
    #: 常见键：potassium(血钾), creatinine(肌酐), wbc(白细胞),
    #:         hemoglobin(血红蛋白), platelet(血小板), lactate(乳酸)
    lab_values: dict[str, float] = field(default_factory=dict)

    #: 当前用药列表
    medications: list[Medication] = field(default_factory=list)

    #: 诊断标签（ICD-10 代码列表）
    diagnoses: list[str] = field(default_factory=list)

    #: 患者元信息
    patient_id: str = ""
    age: int = 0
    gender: str = ""

    #: 时间戳（临床时间线锚点，Unix epoch）
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """校验 vital_signs 形状。"""
        vs = np.asarray(self.vital_signs, dtype=np.float64)
        if vs.ndim == 1:
            vs = vs.reshape(1, -1)
        if vs.shape[1] != N_VITALS:
            raise ValueError(f"vital_signs 列数必须为 {N_VITALS}（对应 VITAL_NAMES），当前为 {vs.shape[1]}")
        self.vital_signs = vs

    # ── WorldState 契约实现 ──────────────────────────────────────────

    def to_vector(self) -> np.ndarray:
        """编码为固定维度向量（JEPA Encoder 输入）。

        取最近一个时间窗的体征 + 检验值，拼接为 1D 向量。
        维度 = N_VITALS + len(STANDARD_LAB_NAMES) = 7 + 6 = 13。
        """
        latest_vitals = self.vital_signs[-1].astype(np.float64)
        lab_vec = np.array(
            [self.lab_values.get(name, 0.0) for name in STANDARD_LAB_NAMES],
            dtype=np.float64,
        )
        return np.concatenate([latest_vitals, lab_vec])


    @classmethod
    def from_vector(cls, vec: np.ndarray) -> PatientState:
        """从向量解码为患者状态（JEPA Decoder 输出）。

        Args:
            vec: to_vector() 格式的 1D 向量 R^13。

        Returns:
            重建的 PatientState（体征填充为单时间窗）。
        """
        vec = np.asarray(vec, dtype=np.float64)
        n_vitals = N_VITALS
        if len(vec) < n_vitals:
            raise ValueError(f"向量维度 {len(vec)} 不足以解码 {n_vitals} 维体征")
        vitals = vec[:n_vitals].reshape(1, n_vitals)
        lab_names = STANDARD_LAB_NAMES
        lab_values: dict[str, float] = {}
        for i, name in enumerate(lab_names):
            idx = n_vitals + i
            if idx < len(vec) and vec[idx] != 0.0:
                lab_values[name] = float(vec[idx])
        return cls(vital_signs=vitals, lab_values=lab_values)

    def distance(self, other: WorldState) -> float:
        """两个患者状态的距离（Cost 模块评估预测误差）。

        使用标准化欧氏距离：体征按正常范围宽度标准化，消除量纲差异。

        Args:
            other: 另一个 PatientState。

        Returns:
            非负浮点数，0 表示体征完全相同。
        """
        if not isinstance(other, PatientState):
            raise TypeError(f"distance 需要 PatientState，收到 {type(other)}")
        # 取最近时间窗对比
        v1 = self.vital_signs[-1]
        v2 = other.vital_signs[-1]
        # 标准化：除以正常范围宽度（消除 bpm / mmHg / % 的量纲差异）
        diffs = np.zeros(N_VITALS, dtype=np.float64)
        for i, name in enumerate(VITAL_NAMES):
            lo, hi = VITAL_NORMAL_RANGES[name]
            span = max(hi - lo, 1e-6)
            diffs[i] = abs(v1[i] - v2[i]) / span
        return float(np.sqrt(np.mean(diffs**2)))

    def copy(self) -> PatientState:
        """深拷贝（Actor what-if 推演用）。"""
        return PatientState(
            vital_signs=self.vital_signs.copy(),
            lab_values=dict(self.lab_values),
            medications=[_copy.copy(m) for m in self.medications],
            diagnoses=list(self.diagnoses),
            patient_id=self.patient_id,
            age=self.age,
            gender=self.gender,
            timestamp=self.timestamp,
        )

    # ── 安全约束 ────────────────────────────────────────────────────

    def is_safe(self) -> bool:
        """体征是否在生理正常范围内（软约束，用于评估）。

        Returns:
            True 如果所有体征均在正常范围内。
        """
        latest = self.vital_signs[-1]
        for i, name in enumerate(VITAL_NAMES):
            lo, hi = VITAL_NORMAL_RANGES[name]
            if not (lo <= latest[i] <= hi):
                return False
        return True

    def is_physiologically_valid(self) -> bool:
        """体征是否在生理可行范围内（硬约束，超出即不可信）。

        Returns:
            True 如果所有体征均在可行范围内（不致命）。
        """
        latest = self.vital_signs[-1]
        for i, name in enumerate(VITAL_NAMES):
            lo, hi = VITAL_FEASIBLE_RANGES[name]
            val = latest[i]
            if not (lo <= val <= hi) or not np.isfinite(val):
                return False
        return True

    def safety_violations(self) -> list[str]:
        """返回超出正常范围的体征列表（用于审计和告警）。

        Returns:
            异常体征描述列表，如 ["heart_rate=120.0 超出 (60, 100)"]。
        """
        latest = self.vital_signs[-1]
        violations: list[str] = []
        for i, name in enumerate(VITAL_NAMES):
            lo, hi = VITAL_NORMAL_RANGES[name]
            val = latest[i]
            if val < lo:
                violations.append(f"{name}={val:.1f} 低于正常 ({lo}, {hi})")
            elif val > hi:
                violations.append(f"{name}={val:.1f} 高于正常 ({lo}, {hi})")
        return violations

    # ── 临床评分 ────────────────────────────────────────────────────

    def sofa_score(self) -> float:
        """简化版 SOFA 评分（序贯器官衰竭评估）。

        基于体征近似计算（完整 SOFA 需检验值，此处用可用数据估算）。
        评分越高，器官功能越差。

        Returns:
            SOFA 评分估计值 [0, 24+]。
        """
        latest = self.vital_signs[-1]
        score = 0.0

        # 心血管（基于血压）
        sbp = latest[VITAL_NAMES.index("systolic_bp")]
        dbp = latest[VITAL_NAMES.index("diastolic_bp")]
        # MAP = DBP + (SBP - DBP) / 3（标准临床公式）
        map_val = dbp + (sbp - dbp) / 3.0
        if map_val < 70:
            score += 1.0
        if sbp < 90:
            score += 1.0

        # 呼吸（基于血氧）
        spo2 = latest[VITAL_NAMES.index("oxygen_saturation")]
        if spo2 < 90:
            score += 3.0
        elif spo2 < 92:
            score += 2.0
        elif spo2 < 95:
            score += 1.0

        # 中枢神经（基于 GCS）
        gcs = latest[VITAL_NAMES.index("gcs")]
        if gcs < 6:
            score += 4.0
        elif gcs < 10:
            score += 3.0
        elif gcs < 13:
            score += 2.0
        elif gcs < 15:
            score += 1.0

        # 有检验值时补充
        if "creatinine" in self.lab_values:
            cr = self.lab_values["creatinine"]
            if cr >= 5.0:
                score += 4.0
            elif cr >= 3.5:
                score += 3.0
            elif cr >= 2.0:
                score += 2.0
            elif cr >= 1.2:
                score += 1.0

        return score

    # ── 序列化 ──────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典（审计日志用）。"""
        return {
            "type": "PatientState",
            "patient_id": self.patient_id,
            "age": self.age,
            "gender": self.gender,
            "vital_signs_latest": {
                name: round(float(self.vital_signs[-1][i]), 2) for i, name in enumerate(VITAL_NAMES)
            },
            "lab_values": {k: round(float(v), 2) for k, v in self.lab_values.items()},
            "medications": [
                {"name": m.name, "dose": m.dose, "unit": m.unit, "route": m.route} for m in self.medications
            ],
            "diagnoses": list(self.diagnoses),
            "is_safe": self.is_safe(),
            "sofa_score": round(self.sofa_score(), 1),
            "timestamp": self.timestamp,
        }

    def causal_query(self) -> str:
        """JEPA 预测的因果查询字符串。"""
        return "patient"

    def __repr__(self) -> str:
        v = self.vital_signs[-1]
        hr = v[VITAL_NAMES.index("heart_rate")]
        sbp = v[VITAL_NAMES.index("systolic_bp")]
        return f"PatientState(hr={hr:.0f}, sbp={sbp:.0f}, safe={self.is_safe()}, sofa={self.sofa_score():.0f})"


# =============================================================================
# 标准检验项名（to_vector/from_vector 编码用）
# =============================================================================

#: 标准检验项名（顺序固定，用于 to_vector 拼接）
STANDARD_LAB_NAMES: list[str] = [
    "potassium",  # 血钾 mmol/L
    "creatinine",  # 肌酐 mg/dL
    "wbc",  # 白细胞 10^9/L
    "hemoglobin",  # 血红蛋白 g/dL
    "platelet",  # 血小板 10^9/L
    "lactate",  # 乳酸 mmol/L
]

#: to_vector 输出维度
STATE_VECTOR_DIM: int = N_VITALS + len(STANDARD_LAB_NAMES)  # 7 + 6 = 13


# =============================================================================
# MedicalAction — 临床干预动作
# =============================================================================


class ActionType(str):
    """动作类型枚举基类（str 子类，可序列化）。"""


# 用 Enum-like 常量（避免 str Enum 序列化问题）
class ActionTypeConstants:
    """动作类型常量。"""

    DRUG = "drug"  # 药物干预
    PROCEDURE = "procedure"  # 操作（插管/透析/手术）
    VITAL_ADJUST = "vital_adjust"  # 生命支持参数调整（呼吸机/补液速度）
    DIAGNOSTIC = "diagnostic"  # 诊断操作（检验/影像）


@dataclass
class MedicalAction(Action):
    """临床干预动作 — 施加于患者状态的干预操作。

    继承 Action，融入 MCI 世界模型动作系统：
        apply(state) → 新 PatientState（转移模型的 ground truth）

    动作空间 A 是离散+连续混合：
        - action_type: 离散（药物/操作/参数调整/诊断）
        - target: 离散（具体药物名/操作名）
        - magnitude: 连续（剂量/强度）

    Attributes:
        action_type: 动作类型（见 ActionTypeConstants）。
        target: 作用对象名称（药物通用名/操作名）。
        magnitude: 剂量或强度（连续值）。
        unit: 单位（如 "μg/kg/min", "mL/h"）。
        route: 给药途径（"IV"/"PO"/"IM"/"SC"）。
    """

    action_type: str = ActionTypeConstants.DRUG
    target: str = ""
    magnitude: float = 0.0
    unit: str = ""
    route: str = "IV"

    def apply(self, state: WorldState, use_emax: bool = False) -> WorldState:
        """对患者状态施加干预，返回新状态。

        注意：此处提供的是**药理效应基线模型**，用于 world model 闭环验证。
        真实药代动力学应由 ClinicalDynamicsPredictor（Phase 1）从数据学习，
        而非在此硬编码。此 apply() 作为 ground truth 基线。

        不修改原状态（函数式语义）。

        Args:
            state: 当前 PatientState。
            use_emax: 是否启用 Emax 饱和模型（默认 False 用线性近似，向后兼容）。
                设为 True 时用 DRUG_PKPD_TABLE 做非线性剂量-响应，
                更符合真实药理学（高剂量饱和）。

        Returns:
            施加干预后的新 PatientState。
        """
        if not isinstance(state, PatientState):
            raise TypeError(f"MedicalAction.apply 需要 PatientState，收到 {type(state)}")

        new_state = state.copy()
        # 记录用药
        if self.action_type == ActionTypeConstants.DRUG:
            new_state.medications.append(
                Medication(
                    name=self.target,
                    dose=self.magnitude,
                    unit=self.unit,
                    route=self.route,
                )
            )
        # 药理效应基线：DRUG_EFFECT_TABLE 查表 + 可选 Emax 饱和
        # （完整动态学预测在 Phase 1 的 ClinicalDynamicsPredictor）
        effects = DRUG_EFFECT_TABLE.get(self.target, {})
        latest = new_state.vital_signs[-1].copy()
        for vital_name, effect_per_unit in effects.items():
            idx = VITAL_NAMES.index(vital_name)
            if use_emax:
                latest[idx] += emax_effect(effect_per_unit, self.magnitude, self.target)
            else:
                latest[idx] += effect_per_unit * self.magnitude
        new_state.vital_signs = latest.reshape(1, -1)
        new_state.timestamp = time.time()
        return new_state

    def to_vector(self) -> np.ndarray:
        """编码动作为向量。

        格式：[action_type_onehot(4)] + [drug_onehot(N_DRUGS)] + [magnitude(1)]

        药物 onehot 使不同药物产生不同动作向量，
        让转移模型能区分多巴胺 vs 美托洛尔的效应差异。
        """
        type_map = {
            ActionTypeConstants.DRUG: 0,
            ActionTypeConstants.PROCEDURE: 1,
            ActionTypeConstants.VITAL_ADJUST: 2,
            ActionTypeConstants.DIAGNOSTIC: 3,
        }
        type_onehot = np.zeros(4, dtype=np.float64)
        type_onehot[type_map.get(self.action_type, 0)] = 1.0
        drug_onehot = np.zeros(N_DRUGS, dtype=np.float64)
        if self.action_type == ActionTypeConstants.DRUG and self.target in DRUG_TO_ID:
            drug_onehot[DRUG_TO_ID[self.target]] = 1.0
        mag = np.array([self.magnitude], dtype=np.float64)
        return np.concatenate([type_onehot, drug_onehot, mag])

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> MedicalAction:
        """从向量解码动作。"""
        vec = np.asarray(vec, dtype=np.float64)
        types = [
            ActionTypeConstants.DRUG,
            ActionTypeConstants.PROCEDURE,
            ActionTypeConstants.VITAL_ADJUST,
            ActionTypeConstants.DIAGNOSTIC,
        ]
        idx = int(np.argmax(vec[:4])) if len(vec) >= 4 else 0
        drug_start = 4
        drug_end = 4 + N_DRUGS
        target = ""
        if len(vec) > drug_end and idx == 0:
            drug_idx = int(np.argmax(vec[drug_start:drug_end]))
            if 0 <= drug_idx < len(DRUG_REGISTRY):
                target = DRUG_REGISTRY[drug_idx]
        mag = float(vec[drug_end]) if len(vec) > drug_end else 0.0
        return cls(action_type=types[idx], target=target, magnitude=mag)


    def to_dict(self) -> dict[str, Any]:
        """序列化为字典（审计日志用）。"""
        return {
            "type": "MedicalAction",
            "action_type": self.action_type,
            "target": self.target,
            "magnitude": self.magnitude,
            "unit": self.unit,
            "route": self.route,
        }

    def __repr__(self) -> str:
        return f"MedicalAction({self.action_type}: {self.target} {self.magnitude}{self.unit} {self.route})"


# =============================================================================
# DRUG_EFFECT_TABLE — 简化药理效应基线表
# =============================================================================

#: 药物即时效应基线表（每单位剂量的体征变化量，线性化 Emax 近似）
#:
#: 用于 MedicalAction.apply() 的 ground truth 基线。
#: Phase 1 的 ClinicalDynamicsPredictor 将学习更精确的动态学。
#:
#: 数值来源与精度说明（诚实披露）:
#:   - 这些数值是 ``Emax`` 模型在低剂量区间的线性斜率（dE/dD @ D≈0），
#:     即 "每 1 个标准化剂量单位引起的体征变化量"。
#:   - 量级来源于 Goodman & Gilman 药理学教科书、UpToDate 临床共识的定性方向，
#:     经人工校准为 ``量级正确 + 方向正确`` 的研究级近似值，非个体化精确 PK/PD。
#:   - 真实临床剂量-响应是非线性饱和曲线（见 DRUG_PKPD_TABLE 的 Emax/EC50）。
#:   - 升级到精确 PK/PD 需要: ①从 EHR/MIMIC 拟合个体参数，或 ②接入开源 PK 库。
DRUG_EFFECT_TABLE: dict[str, dict[str, float]] = {
    "dopamine": {  # 多巴胺（剂量依赖性：低剂量扩肾血管，高剂量升压强心）
        "heart_rate": 2.0,  # 心率↑
        "systolic_bp": 3.0,  # 收缩压↑
        "diastolic_bp": 1.5,  # 舒张压↑
    },
    "norepinephrine": {  # 去甲肾上腺素（强 α 激动，一线升压）
        "systolic_bp": 5.0,  # 收缩压↑↑（强升压）
        "diastolic_bp": 3.0,  # 舒张压↑
        "heart_rate": -0.5,  # 心率反射性↓（压力反射）
    },
    "metoprolol": {  # 美托洛尔（β1 阻滞剂）
        "heart_rate": -3.0,  # 心率↓↓
        "systolic_bp": -2.0,  # 收缩压↓
        "diastolic_bp": -1.0,  # 舒张压↓
    },
    "epinephrine": {  # 肾上腺素（非选择性 α/β 激动）
        "heart_rate": 3.0,  # 心率↑↑
        "systolic_bp": 4.0,  # 收缩压↑
        "diastolic_bp": -1.0,  # 舒张压↓（β2 扩血管，双相反应）
    },
    "furosemide": {  # 呋塞米（襻利尿剂）
        "systolic_bp": -1.5,  # 血压↓（容量减少）
        "diastolic_bp": -1.0,
    },
    "fluid_resuscitation": {  # 液体复苏（晶体液/胶体液）
        "systolic_bp": 2.0,  # 血压↑（容量扩充）
        "diastolic_bp": 1.0,
        "heart_rate": -1.0,  # 心率↓（容量充足反射）
    },
}

#: 药物 PK/PD 结构化参数表（Emax 饱和模型 + 起效时间常数）
#:
#: 每个药物记录其 Emax 模型参数，使效应可从线性近似升级为非线性饱和曲线:
#:
#:     effect(D) = sign * Emax * D / (EC50 + D)
#:
#: 其中 D 是标准化剂量（0~1+），EC50 是半数有效剂量，Emax 是最大效应。
#: 起效时间常数 tau（分钟）描述达稳态的时间，用于时间感知转移模型。
#:
#: 数值来源: Goodman & Gilman 第14版 + 临床药理学手册共识量级（研究级近似）。
#: 真实个体化参数需从 EHR 数据拟合，当前值为群体典型值。
DRUG_PKPD_TABLE: dict[str, dict[str, float]] = {
    "dopamine": {"EC50": 0.5, "Emax": 8.0, "tau": 2.0},  # 起效快（分钟级）
    "norepinephrine": {"EC50": 0.4, "Emax": 12.0, "tau": 1.5},  # 起效很快
    "metoprolol": {"EC50": 0.6, "Emax": 6.0, "tau": 15.0},  # 起效较慢（口服）
    "epinephrine": {"EC50": 0.3, "Emax": 10.0, "tau": 1.0},  # 起效极快（IV推注）
    "furosemide": {"EC50": 0.7, "Emax": 4.0, "tau": 5.0},  # IV 起效数分钟
    "fluid_resuscitation": {"EC50": 0.5, "Emax": 5.0, "tau": 10.0},  # 容量再分布
}


def emax_effect(linear_slope: float, dose: float, drug: str) -> float:
    """Emax 饱和模型计算药物效应（非线性剂量-响应）。

    在低剂量区间退化为线性近似 ``slope * dose``（与 DRUG_EFFECT_TABLE 一致），
    在高剂量区间趋于饱和，更符合真实药理学。

    若药物不在 DRUG_PKPD_TABLE 中，退化为纯线性（向后兼容）。

    模型:
        effect(D) = slope_linear * D * EC50 / (EC50 + D)

    其中 ``slope_linear`` 是低剂量斜率，``EC50`` 控制饱和速度。
    当 D << EC50 时退化为 ``slope_linear * D``；
    当 D >> EC50 时趋于 ``slope_linear * EC50``（饱和上限）。

    Args:
        linear_slope: 来自 DRUG_EFFECT_TABLE 的每单位剂量效应量（低剂量斜率）。
        dose: 标准化剂量（≥0）。
        drug: 药物名（查 DRUG_PKPD_TABLE 获取 EC50）。

    Returns:
        该剂量下的体征变化量（与 linear_slope 同量纲）。
    """
    pkpd = DRUG_PKPD_TABLE.get(drug)
    if pkpd is None or dose <= 0:
        return linear_slope * dose
    ec50 = max(pkpd["EC50"], 1e-6)
    # 线性区: slope*D；饱和因子 EC50/(EC50+D) 使高剂量趋于 slope*EC50
    return linear_slope * dose * ec50 / (ec50 + dose)


# 药物注册表：药物名 -> ID（用于动作向量 one-hot 编码）
# 动作向量格式 = [action_type_onehot(4)] + [drug_onehot(N)] + [magnitude(1)]
DRUG_REGISTRY: list[str] = sorted(DRUG_EFFECT_TABLE.keys())
DRUG_TO_ID: dict[str, int] = {drug: i for i, drug in enumerate(DRUG_REGISTRY)}
N_DRUGS: int = len(DRUG_REGISTRY)
