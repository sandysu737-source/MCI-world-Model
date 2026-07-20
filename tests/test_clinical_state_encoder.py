"""ClinicalStateEncoder 单元测试 — Phase 0-D2。

验证三种输入模式的编码正确性：
    1. encode_from_dicts — 体征字典列表
    2. encode_from_fhir — FHIR Observation 列表
    3. encode_from_matrix — numpy 时序矩阵
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from mci_world_model.sdk._clinical_state_encoder import (
    LOINC_TO_VITAL,
    ClinicalStateEncoder,
)
from mci_world_model.sdk._clinical_world_state import (
    N_VITALS,
    VITAL_NAMES,
    Medication,
    PatientState,
)

# =============================================================================
# 1. encode_from_dicts — 字典列表模式
# =============================================================================


class TestEncodeFromDicts:
    """验证体征字典列表编码。"""

    def test_basic_encoding(self):
        """基本体征字典正确编码。"""
        records = [
            {
                "heart_rate": 75,
                "systolic_bp": 120,
                "diastolic_bp": 80,
                "oxygen_saturation": 98,
                "respiratory_rate": 16,
                "temperature": 36.8,
                "gcs": 15,
            },
        ]
        state = ClinicalStateEncoder.encode_from_dicts(records, patient_id="P001")
        assert isinstance(state, PatientState)
        assert state.patient_id == "P001"
        assert state.vital_signs.shape == (1, N_VITALS)

    def test_multi_timestep(self):
        """多时间窗体征正确编码。"""
        records = [
            {
                "heart_rate": 75,
                "systolic_bp": 120,
                "diastolic_bp": 80,
                "oxygen_saturation": 98,
                "respiratory_rate": 16,
                "temperature": 36.8,
                "gcs": 15,
            },
            {
                "heart_rate": 80,
                "systolic_bp": 125,
                "diastolic_bp": 82,
                "oxygen_saturation": 97,
                "respiratory_rate": 18,
                "temperature": 37.0,
                "gcs": 15,
            },
            {
                "heart_rate": 85,
                "systolic_bp": 130,
                "diastolic_bp": 85,
                "oxygen_saturation": 96,
                "respiratory_rate": 20,
                "temperature": 37.2,
                "gcs": 14,
            },
        ]
        state = ClinicalStateEncoder.encode_from_dicts(records)
        assert state.vital_signs.shape == (3, N_VITALS)
        # 心率趋势递增
        hr_idx = VITAL_NAMES.index("heart_rate")
        assert state.vital_signs[0][hr_idx] == 75
        assert state.vital_signs[2][hr_idx] == 85

    def test_missing_fields_filled(self):
        """缺失字段被填充（不崩溃）。"""
        records = [{"heart_rate": 75}]  # 只有心率
        state = ClinicalStateEncoder.encode_from_dicts(records)
        assert state.vital_signs.shape == (1, N_VITALS)
        # 心率正确
        assert state.vital_signs[0][VITAL_NAMES.index("heart_rate")] == 75
        # 其他填充为 0（无可用值时）
        assert state.vital_signs[0][VITAL_NAMES.index("systolic_bp")] == 0.0

    def test_with_labs_and_medications(self):
        """体征+检验+用药+诊断完整编码。"""
        records = [
            {
                "heart_rate": 75,
                "systolic_bp": 120,
                "diastolic_bp": 80,
                "oxygen_saturation": 98,
                "respiratory_rate": 16,
                "temperature": 36.8,
                "gcs": 15,
            }
        ]
        labs = [{"name": "creatinine", "value": 1.2}, {"name": "potassium", "value": 4.0}]
        meds = [Medication(name="aspirin", dose=100, unit="mg", route="PO")]
        dx = ["I10"]

        state = ClinicalStateEncoder.encode_from_dicts(
            records,
            lab_records=labs,
            medications=meds,
            diagnoses=dx,
            patient_id="P002",
            age=70,
            gender="F",
        )
        assert state.lab_values["creatinine"] == 1.2
        assert state.lab_values["potassium"] == 4.0
        assert len(state.medications) == 1
        assert state.diagnoses == ["I10"]
        assert state.age == 70

    def test_empty_records(self):
        """空记录列表返回默认状态。"""
        state = ClinicalStateEncoder.encode_from_dicts([])
        assert state.vital_signs.shape == (1, N_VITALS)


# =============================================================================
# 2. encode_from_fhir — FHIR 标准模式
# =============================================================================


class TestEncodeFromFHIR:
    """验证 FHIR Observation 编码。"""

    def _make_fhir_observations(self) -> list[dict]:
        """构造标准 FHIR R4 Observation 列表。"""
        return [
            {
                "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4"}]},
                "valueQuantity": {"value": 75.0, "unit": "bpm"},
            },
            {
                "code": {"coding": [{"system": "http://loinc.org", "code": "85354-9"}]},
                "valueQuantity": {"value": 120.0, "unit": "mmHg"},
            },
            {
                "code": {"coding": [{"system": "http://loinc.org", "code": "8462-4"}]},
                "valueQuantity": {"value": 80.0, "unit": "mmHg"},
            },
            {
                "code": {"coding": [{"system": "http://loinc.org", "code": "59408-5"}]},
                "valueQuantity": {"value": 98.0, "unit": "%"},
            },
            {
                "code": {"coding": [{"system": "http://loinc.org", "code": "9279-1"}]},
                "valueQuantity": {"value": 16.0, "unit": "/min"},
            },
            {
                "code": {"coding": [{"system": "http://loinc.org", "code": "8310-5"}]},
                "valueQuantity": {"value": 36.8, "unit": "°C"},
            },
            {
                "code": {"coding": [{"system": "http://loinc.org", "code": "35094-2"}]},
                "valueInteger": 15,
            },
        ]

    def test_fhir_vital_encoding(self):
        """FHIR 体征正确映射到 VITAL_NAMES。"""
        obs = self._make_fhir_observations()
        state = ClinicalStateEncoder.encode_from_fhir(obs)
        assert state.vital_signs.shape == (1, N_VITALS)
        # 心率 8867-4 → heart_rate = 75
        assert state.vital_signs[0][VITAL_NAMES.index("heart_rate")] == 75.0
        # 收缩压 85354-9 → systolic_bp = 120
        assert state.vital_signs[0][VITAL_NAMES.index("systolic_bp")] == 120.0
        # GCS 35094-2 → gcs = 15（valueInteger）
        assert state.vital_signs[0][VITAL_NAMES.index("gcs")] == 15.0

    def test_fhir_lab_encoding(self):
        """FHIR 检验值正确映射。"""
        obs = self._make_fhir_observations()
        obs.append(
            {
                "code": {"coding": [{"system": "http://loinc.org", "code": "2160-0"}]},
                "valueQuantity": {"value": 1.5, "unit": "mg/dL"},
            }
        )
        state = ClinicalStateEncoder.encode_from_fhir(obs)
        assert state.lab_values.get("creatinine") == 1.5

    def test_fhir_with_patient_resource(self):
        """FHIR Patient 资源提取元信息。"""
        patient = {
            "id": "PAT-001",
            "gender": "male",
            "birthDate": "1955-01-01",
        }
        state = ClinicalStateEncoder.encode_from_fhir(
            self._make_fhir_observations(),
            patient_resource=patient,
        )
        assert state.patient_id == "PAT-001"
        assert state.gender == "male"
        # 基于系统时间动态断言：1955-01-01 出生，年龄随当前年份递增
        assert state.age == datetime.now().year - 1955

    def test_fhir_missing_code_skipped(self):
        """无 LOINC 代码的 Observation 被跳过。"""
        obs = [
            {"code": {"coding": [{"system": "other", "code": "unknown"}]}, "valueQuantity": {"value": 999}},
            *self._make_fhir_observations(),
        ]
        state = ClinicalStateEncoder.encode_from_fhir(obs)
        # 仍能正确编码标准体征
        assert state.vital_signs[0][VITAL_NAMES.index("heart_rate")] == 75.0

    def test_loinc_mapping_completeness(self):
        """LOINC 映射覆盖全部 7 项体征。"""
        assert len(LOINC_TO_VITAL) == N_VITALS
        for vname in VITAL_NAMES:
            assert vname in LOINC_TO_VITAL.values(), f"{vname} 无 LOINC 映射"


# =============================================================================
# 3. encode_from_matrix — numpy 矩阵模式
# =============================================================================


class TestEncodeFromMatrix:
    """验证 numpy 时序矩阵编码（MIMIC/benchmark 格式）。"""

    def test_matrix_default_order(self):
        """默认列顺序 = VITAL_NAMES。"""
        rng = np.random.default_rng(42)
        matrix = rng.normal(80, 10, size=(12, N_VITALS))
        state = ClinicalStateEncoder.encode_from_matrix(matrix, patient_id="M001")
        assert state.vital_signs.shape == (12, N_VITALS)
        np.testing.assert_allclose(state.vital_signs, matrix, atol=1e-10)

    def test_matrix_custom_order(self):
        """自定义列名正确重排。"""
        # 打乱列顺序
        custom_names = [
            "gcs",
            "temperature",
            "respiratory_rate",
            "oxygen_saturation",
            "diastolic_bp",
            "systolic_bp",
            "heart_rate",
        ]
        rng = np.random.default_rng(42)
        matrix = rng.normal(80, 10, size=(5, N_VITALS))
        state = ClinicalStateEncoder.encode_from_matrix(
            matrix,
            variable_names=custom_names,
        )
        # heart_rate 在 custom 中是最后一列(6)，在 VITAL_NAMES 中是第一列(0)
        assert state.vital_signs[0][VITAL_NAMES.index("heart_rate")] == pytest.approx(matrix[0][6])

    def test_matrix_nan_filled(self):
        """含 NaN 的矩阵被填充。"""
        matrix = np.array(
            [
                [75, 120, 80, 98, 16, 36.8, 15],
                [np.nan, 125, 82, 97, 18, 37.0, 15],
                [80, np.nan, np.nan, 96, 20, 37.2, 14],
            ]
        )
        state = ClinicalStateEncoder.encode_from_matrix(matrix)
        # 无 NaN 残留
        assert not np.any(np.isnan(state.vital_signs))
        # 前向填充：第二行心率 = 第一行心率
        assert state.vital_signs[1][0] == 75.0

    def test_matrix_subset_columns(self):
        """矩阵列数 < N_VITALS 时正确处理。"""
        matrix = np.array([[75, 120, 80]])  # 只有 3 列
        state = ClinicalStateEncoder.encode_from_matrix(
            matrix,
            variable_names=["heart_rate", "systolic_bp", "diastolic_bp"],
        )
        assert state.vital_signs.shape == (1, N_VITALS)
        # 提供的 3 列正确
        assert state.vital_signs[0][0] == 75.0  # heart_rate
        # 缺失列填充为 0
        assert state.vital_signs[0][3] == 0.0  # oxygen_saturation（缺失）


# =============================================================================
# 4. 编码一致性验证
# =============================================================================


class TestEncodingConsistency:
    """验证不同输入模式产出一致的 PatientState。"""

    def test_dict_and_matrix_produce_same_state(self):
        """相同数据，字典模式和矩阵模式产出相同状态。"""
        # 字典模式
        records = [
            {
                "heart_rate": 75,
                "systolic_bp": 120,
                "diastolic_bp": 80,
                "oxygen_saturation": 98,
                "respiratory_rate": 16,
                "temperature": 36.8,
                "gcs": 15,
            }
        ]
        state_dict = ClinicalStateEncoder.encode_from_dicts(records)

        # 矩阵模式（同样数据）
        matrix = np.array([[75, 120, 80, 98, 16, 36.8, 15]])
        state_matrix = ClinicalStateEncoder.encode_from_matrix(matrix)

        np.testing.assert_allclose(state_dict.vital_signs, state_matrix.vital_signs, atol=1e-10)

    def test_to_vector_stable_across_encodings(self):
        """不同编码方式产出的 to_vector() 一致。"""
        records = [
            {
                "heart_rate": 80,
                "systolic_bp": 110,
                "diastolic_bp": 70,
                "oxygen_saturation": 96,
                "respiratory_rate": 18,
                "temperature": 37.0,
                "gcs": 14,
            }
        ]
        state1 = ClinicalStateEncoder.encode_from_dicts(records)

        matrix = np.array([[80, 110, 70, 96, 18, 37.0, 14]])
        state2 = ClinicalStateEncoder.encode_from_matrix(matrix)

        np.testing.assert_allclose(state1.to_vector(), state2.to_vector(), atol=1e-10)


# =============================================================================
# 4. encode_from_fhir_bundle — 方向五：完整 FHIR R4 Bundle
# =============================================================================


class TestEncodeFromFhirBundle:
    """方向五：从完整 FHIR Bundle 解析体征+诊断+用药+患者元信息。"""

    @staticmethod
    def _make_bundle() -> dict:
        """构造一个含 4 类资源的 FHIR R4 Bundle（ICU 感染性休克样例）。"""
        return {
            "resourceType": "Bundle",
            "entry": [
                # 患者
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "FHIR-BUNDLE-001",
                        "gender": "female",
                        "birthDate": "1950-05-01",
                    }
                },
                # 体征：心率 8867-4 = 110
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4"}]},
                        "valueQuantity": {"value": 110, "unit": "/min"},
                    }
                },
                # 体征：收缩压 85354-9 = 85
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"coding": [{"system": "http://loinc.org", "code": "85354-9"}]},
                        "valueQuantity": {"value": 85, "unit": "mmHg"},
                    }
                },
                # 诊断：ICD-10 A41.9 脓毒症
                {
                    "resource": {
                        "resourceType": "Condition",
                        "code": {
                            "coding": [
                                {
                                    "system": "http://hl7.org/fhir/sid/icd-10",
                                    "code": "A41.9",
                                    "display": "Sepsis, unspecified",
                                }
                            ]
                        },
                    }
                },
                # 用药：norepinephrine
                {
                    "resource": {
                        "resourceType": "MedicationRequest",
                        "medicationCodeableConcept": {
                            "coding": [
                                {
                                    "display": "Norepinephrine",
                                    "code": "1646825",
                                }
                            ]
                        },
                        "dosageInstruction": [
                            {
                                "doseAndRate": [
                                    {
                                        "doseQuantity": {
                                            "value": 0.2,
                                            "unit": "mcg/kg/min",
                                        }
                                    }
                                ]
                            }
                        ],
                    }
                },
            ],
        }

    def test_bundle_parses_all_resource_types(self):
        """Bundle 含 4 类资源时全部正确解析。"""
        state = ClinicalStateEncoder.encode_from_fhir_bundle(self._make_bundle())

        # 患者元信息
        assert state.patient_id == "FHIR-BUNDLE-001"
        assert state.gender == "female"
        # 基于系统时间动态断言：1950-05-01 出生，年龄随当前年份递增
        assert state.age == datetime.now().year - 1950

        # 体征：encode_from_fhir 输出为单行聚合 (1, N_VITALS)
        # 多个 Observation 按体征位写入同一行
        assert state.vital_signs.shape == (1, N_VITALS)
        assert state.vital_signs[0][VITAL_NAMES.index("heart_rate")] == 110.0
        assert state.vital_signs[0][VITAL_NAMES.index("systolic_bp")] == 85.0

        # 诊断
        assert state.diagnoses == ["A41.9"]

        # 用药
        assert len(state.medications) == 1
        assert state.medications[0].name == "Norepinephrine"
        assert state.medications[0].dose == 0.2
        assert state.medications[0].unit == "mcg/kg/min"

    def test_bundle_empty_returns_empty_state(self):
        """空 Bundle 返回无诊断无用药的空状态（不抛错）。"""
        state = ClinicalStateEncoder.encode_from_fhir_bundle({"resourceType": "Bundle", "entry": []})
        assert state.diagnoses == []
        assert state.medications == []
        # encode_from_fhir 上游行为：无 Observation 时返回单行 0 填充
        assert state.vital_signs.shape == (1, N_VITALS)
        assert state.vital_signs.sum() == 0.0

    def test_bundle_condition_without_icd_skipped(self):
        """非 ICD-10 诊断码被跳过。"""
        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Condition",
                        "code": {
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "code": "12345",
                                }
                            ]
                        },
                    }
                }
            ],
        }
        state = ClinicalStateEncoder.encode_from_fhir_bundle(bundle)
        assert state.diagnoses == []

    def test_bundle_medication_administration(self):
        """MedicationAdministration 资源也正确解析（dosage 字段）。"""
        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "MedicationAdministration",
                        "medicationCodeableConcept": {"coding": [{"display": "Furosemide"}]},
                        "dosage": {"dose": {"value": 40, "unit": "mg"}},
                    }
                }
            ],
        }
        state = ClinicalStateEncoder.encode_from_fhir_bundle(bundle)
        assert len(state.medications) == 1
        assert state.medications[0].name == "Furosemide"
        assert state.medications[0].dose == 40.0

    def test_bundle_unknown_resource_type_ignored(self):
        """未知 resourceType 静默跳过，不影响其他资源解析。"""
        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {"resource": {"resourceType": "Encounter", "id": "E1"}},
                {
                    "resource": {
                        "resourceType": "Condition",
                        "code": {"coding": [{"code": "I48.91"}]},
                    }
                },
            ],
        }
        state = ClinicalStateEncoder.encode_from_fhir_bundle(bundle)
        # Encounter 被忽略，Condition 正常解析（I48.91 符合 ICD-10 格式）
        assert state.diagnoses == ["I48.91"]
