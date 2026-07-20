"""MCI World Model — 临床状态编码器（ClinicalStateEncoder）

============================================================

Phase 0 地基模块：将异构临床原始数据编码为 PatientState（世界模型状态空间）。

这是世界模型五要素中"感知 E"的实现：
    observation → state（E: 临床数据 → PatientState）

支持三种输入源：
    1. 原始体征字典列表（最通用，任何系统都能转成此格式）
    2. FHIR Observation 列表（医疗标准格式，P0-2 FHIR 适配层的基础）
    3. numpy 时序矩阵（MIMIC/benchmark 格式，P0-3 验证的基础）

设计原则:
    - 无状态：纯转换函数，不持有跨调用状态
    - 容错：缺失字段用 0 或 NaN 填充，不崩溃
    - 可审计：每步编码记录来源
    - 与 su-memory-sdk 严格区隔：只做数据转换，不做存储

核心类:
    ClinicalStateEncoder — 临床数据 → PatientState 编码器
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from mci_world_model.sdk._clinical_world_state import (
    N_VITALS,
    VITAL_NAMES,
    Medication,
    PatientState,
)

# =============================================================================
# LOINC 代码 → 体征变量名映射（FHIR Observation 编码用）
# =============================================================================

#: LOINC 代码 → 内部体征变量名（FHIR Observation.code → VITAL_NAMES）
LOINC_TO_VITAL: dict[str, str] = {
    "8867-4": "heart_rate",  # 心率
    "85354-9": "systolic_bp",  # 收缩压
    "8462-4": "diastolic_bp",  # 舒张压
    "59408-5": "oxygen_saturation",  # 血氧饱和度
    "9279-1": "respiratory_rate",  # 呼吸频率
    "8310-5": "temperature",  # 体温
    "35094-2": "gcs",  # 格拉斯哥昏迷评分
}

#: LOINC 代码 → 内部检验项名（FHIR Observation.code → STANDARD_LAB_NAMES）
LOINC_TO_LAB: dict[str, str] = {
    "2823-3": "potassium",  # 血钾
    "2160-0": "creatinine",  # 肌酐
    "6690-2": "wbc",  # 白细胞
    "718-7": "hemoglobin",  # 血红蛋白
    "777-3": "platelet",  # 血小板
    "32693-4": "lactate",  # 乳酸
}

#: 体征变量名 → LOINC 代码（反向映射，用于序列化）
VITAL_TO_LOINC: dict[str, str] = {v: k for k, v in LOINC_TO_VITAL.items()}


# =============================================================================
# ClinicalStateEncoder — 临床数据编码器
# =============================================================================


def _is_icd10_code(code: str) -> bool:
    """判断字符串是否像 ICD-10 代码（字母+数字+可选小数点）。"""
    if not code or len(code) < 3:
        return False
    c = code[0]
    return c.isalpha() and code[1:3].isdigit()


class ClinicalStateEncoder:
    """临床数据 → PatientState 编码器（世界模型感知层）。

    将异构临床原始数据转换为统一的 PatientState，供世界模型转移模型 T 使用。

    三种输入模式：
        encode_from_dicts()   — 体征字典列表（最通用）
        encode_from_fhir()    — FHIR Observation 列表（医疗标准）
        encode_from_matrix()  — numpy 时序矩阵（benchmark 格式）
    """

    @staticmethod
    def encode_from_dicts(
        vital_records: list[dict[str, Any]],
        lab_records: list[dict[str, Any]] | None = None,
        medications: list[Medication] | None = None,
        diagnoses: list[str] | None = None,
        patient_id: str = "",
        age: int = 0,
        gender: str = "",
    ) -> PatientState:
        """从体征字典列表编码为 PatientState。

        Args:
            vital_records: 体征记录列表，每条是一个时间窗。
                每条字典的键为 VITAL_NAMES 中的变量名，值为数值。
                例：[{"heart_rate": 75, "systolic_bp": 120, ...}, ...]
            lab_records: 检验记录列表（可选）。每条含 "name" 和 "value"。
            medications: 当前用药列表（可选）。
            diagnoses: 诊断标签列表（可选，ICD-10 代码）。
            patient_id: 患者ID。
            age: 年龄。
            gender: 性别。

        Returns:
            编码后的 PatientState。
        """
        if not vital_records:
            return PatientState(
                vital_signs=np.zeros((1, N_VITALS)),
                patient_id=patient_id,
                age=age,
                gender=gender,
            )

        # 构建体征时序矩阵 (T, N_VITALS)
        n_timesteps = len(vital_records)
        vital_matrix = np.full((n_timesteps, N_VITALS), np.nan, dtype=np.float64)

        for t, record in enumerate(vital_records):
            for i, vname in enumerate(VITAL_NAMES):
                if vname in record:
                    vital_matrix[t, i] = float(record[vname])

        # NaN 处理：前向填充 → 均值填充 → 0
        vital_matrix = ClinicalStateEncoder._fill_nan(vital_matrix)

        # 检验值（取最新）
        lab_values: dict[str, float] = {}
        if lab_records:
            for rec in lab_records:
                name = rec.get("name", "")
                value = rec.get("value")
                if name and value is not None:
                    lab_values[name] = float(value)

        return PatientState(
            vital_signs=vital_matrix,
            lab_values=lab_values,
            medications=medications or [],
            diagnoses=diagnoses or [],
            patient_id=patient_id,
            age=age,
            gender=gender,
        )

    @staticmethod
    def encode_from_fhir(
        observations: list[dict[str, Any]],
        patient_resource: dict[str, Any] | None = None,
    ) -> PatientState:
        """从 FHIR Observation 列表编码为 PatientState。

        FHIR R4 Observation 格式：
            {
                "code": {"coding": [{"code": "8867-4", ...}]},
                "valueQuantity": {"value": 75.0, "unit": "bpm"},
                "effectiveDateTime": "2024-01-01T10:00:00Z"
            }

        Args:
            observations: FHIR Observation 资源列表。
            patient_resource: FHIR Patient 资源（可选，提取年龄/性别）。

        Returns:
            编码后的 PatientState。
        """
        vital_dicts: list[dict[str, float]] = [{}]
        lab_values: dict[str, float] = {}

        for obs in observations:
            # 提取 LOINC 代码
            loinc_code = ClinicalStateEncoder._extract_loinc_code(obs)
            if loinc_code is None:
                continue

            # 提取数值
            value = ClinicalStateEncoder._extract_value(obs)
            if value is None:
                continue

            # 映射到内部变量名
            if loinc_code in LOINC_TO_VITAL:
                vname = LOINC_TO_VITAL[loinc_code]
                vital_dicts[0][vname] = value
            elif loinc_code in LOINC_TO_LAB:
                lname = LOINC_TO_LAB[loinc_code]
                lab_values[lname] = value

        # 提取患者元信息
        patient_id = ""
        age = 0
        gender = ""
        if patient_resource:
            patient_id = patient_resource.get("id", "")
            gender = patient_resource.get("gender", "")
            birth_date = patient_resource.get("birthDate", "")
            if birth_date:
                age = ClinicalStateEncoder._estimate_age(birth_date)

        return ClinicalStateEncoder.encode_from_dicts(
            vital_records=vital_dicts,
            lab_records=[{"name": k, "value": v} for k, v in lab_values.items()],
            patient_id=patient_id,
            age=age,
            gender=gender,
        )

    @staticmethod
    def encode_from_fhir_bundle(
        bundle: dict[str, Any],
    ) -> PatientState:
        """从 FHIR R4 Bundle 编码为 PatientState（完整资源集）。

        方向五：扩展 encode_from_fhir，支持完整 Bundle 解析：
            - Observation → 体征/检验值（复用 encode_from_fhir）
            - Condition → 诊断（ICD-10 代码）
            - MedicationRequest/MedicationAdministration → 用药记录
            - Patient → 元信息

        这是方向五临床验证的数据入口，让世界模型能吃真实 EHR 的 FHIR 导出。

        Args:
            bundle: FHIR R4 Bundle，含 entry 列表，每项有 resource。

        Returns:
            编码后的 PatientState（含诊断、用药）。

        Example:
            >>> bundle = {
            ...     "resourceType": "Bundle",
            ...     "entry": [
            ...         {"resource": {"resourceType": "Observation", ...}},
            ...         {"resource": {"resourceType": "Condition", "code": {"coding": [{"code": "I48.91"}]}}},
            ...     ]
            ... }
            >>> state = ClinicalStateEncoder.encode_from_fhir_bundle(bundle)
        """
        entries = bundle.get("entry", [])
        observations: list[dict[str, Any]] = []
        conditions: list[str] = []
        medications: list[Medication] = []
        patient_resource: dict[str, Any] | None = None

        for entry in entries:
            resource = entry.get("resource", {})
            rtype = resource.get("resourceType", "")
            if rtype == "Observation":
                observations.append(resource)
            elif rtype == "Condition":
                # 提取 ICD-10 代码
                coding = resource.get("code", {}).get("coding", [])
                for c in coding:
                    code = c.get("code", "")
                    system = c.get("system", "")
                    if "icd" in system.lower() or _is_icd10_code(code):
                        conditions.append(code)
            elif rtype in ("MedicationRequest", "MedicationAdministration"):
                med = ClinicalStateEncoder._extract_medication_from_resource(resource)
                if med is not None:
                    medications.append(med)
            elif rtype == "Patient":
                patient_resource = resource

        # 用 encode_from_fhir 处理 Observation + Patient
        state = ClinicalStateEncoder.encode_from_fhir(observations, patient_resource)
        # 补充诊断和用药
        state.diagnoses = conditions
        state.medications = medications
        return state

    @staticmethod
    def _extract_medication_from_resource(
        resource: dict[str, Any],
    ) -> Medication | None:
        """从 MedicationRequest/Administration 提取用药信息。"""
        # 药物名
        med_name = ""
        if "medicationCodeableConcept" in resource:
            coding = resource["medicationCodeableConcept"].get("coding", [])
            for c in coding:
                med_name = c.get("display", "") or c.get("code", "")
                if med_name:
                    break
        elif "medicationReference" in resource:
            med_name = resource["medicationReference"].get("display", "")

        # 剂量
        dose = 0.0
        unit = ""
        if "dosageInstruction" in resource:
            for dosage in resource["dosageInstruction"]:
                dose_and_rate = dosage.get("doseAndRate", [{}])
                if dose_and_rate:
                    qty = dose_and_rate[0].get("doseQuantity", {})
                    dose = float(qty.get("value", 0.0))
                    unit = qty.get("unit", "")
                    break
        elif "dosage" in resource:
            qty = resource["dosage"].get("dose", {})
            dose = float(qty.get("value", 0.0))
            unit = qty.get("unit", "")

        if not med_name and dose == 0.0:
            return None
        return Medication(name=med_name, dose=dose, unit=unit)

    @staticmethod
    def encode_from_matrix(
        vital_matrix: np.ndarray,
        variable_names: list[str] | None = None,
        lab_values: dict[str, float] | None = None,
        medications: list[Medication] | None = None,
        patient_id: str = "",
    ) -> PatientState:
        """从 numpy 时序矩阵编码为 PatientState（benchmark/MIMIC 格式）。

        Args:
            vital_matrix: 体征时序矩阵 (T, V)。列顺序由 variable_names 指定。
            variable_names: 列名列表。若为 None，默认按 VITAL_NAMES 顺序。
            lab_values: 检验值字典（可选）。
            medications: 用药列表（可选）。
            patient_id: 患者ID。

        Returns:
            编码后的 PatientState。
        """
        matrix = np.asarray(vital_matrix, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        var_names = variable_names or list(VITAL_NAMES)

        # 重排列到 VITAL_NAMES 顺序
        reordered = np.full((matrix.shape[0], N_VITALS), np.nan, dtype=np.float64)
        for target_idx, target_name in enumerate(VITAL_NAMES):
            if target_name in var_names:
                source_idx = var_names.index(target_name)
                if source_idx < matrix.shape[1]:
                    reordered[:, target_idx] = matrix[:, source_idx]

        reordered = ClinicalStateEncoder._fill_nan(reordered)

        return PatientState(
            vital_signs=reordered,
            lab_values=lab_values or {},
            medications=medications or [],
            patient_id=patient_id,
        )

    # ── 内部辅助方法 ──────────────────────────────────────────────

    @staticmethod
    def _fill_nan(matrix: np.ndarray) -> np.ndarray:
        """NaN 处理：前向填充 → 后向填充 → 列均值 → 0。"""
        result = matrix.copy()
        n_rows, n_cols = result.shape

        for col in range(n_cols):
            # 前向填充
            for row in range(1, n_rows):
                if np.isnan(result[row, col]) and not np.isnan(result[row - 1, col]):
                    result[row, col] = result[row - 1, col]
            # 后向填充
            for row in range(n_rows - 2, -1, -1):
                if np.isnan(result[row, col]) and not np.isnan(result[row + 1, col]):
                    result[row, col] = result[row + 1, col]
            # 列均值填充
            valid = result[:, col][~np.isnan(result[:, col])]
            if len(valid) > 0:
                col_mean = np.mean(valid)
            else:
                col_mean = 0.0
            result[:, col][np.isnan(result[:, col])] = col_mean

        return result

    @staticmethod
    def _extract_loinc_code(obs: dict[str, Any]) -> str | None:
        """从 FHIR Observation 提取 LOINC 代码。"""
        code_block = obs.get("code", {})
        coding_list = code_block.get("coding", [])
        for coding in coding_list:
            system = coding.get("system", "")
            code = coding.get("code", "")
            if "loinc" in system.lower() or code in LOINC_TO_VITAL or code in LOINC_TO_LAB:
                return code
        return None

    @staticmethod
    def _extract_value(obs: dict[str, Any]) -> float | None:
        """从 FHIR Observation 提取数值。"""
        # valueQuantity（数值型）
        vq = obs.get("valueQuantity", {})
        if vq and "value" in vq:
            try:
                return float(vq["value"])
            except (TypeError, ValueError):
                pass
        # valueInteger（整数型，如 GCS）
        vi = obs.get("valueInteger")
        if vi is not None:
            try:
                return float(vi)
            except (TypeError, ValueError):
                pass
        return None

    @staticmethod
    def _estimate_age(birth_date: str) -> int:
        """从出生日期估算年龄（粗略）。"""
        try:
            year = int(birth_date[:4])
            return max(0, int(time.strftime("%Y")) - year)
        except (ValueError, IndexError):
            return 0
