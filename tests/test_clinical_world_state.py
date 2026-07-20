"""PatientState + MedicalAction 单元测试 — Phase 0 地基验证。

验证世界模型状态空间的五个核心契约：
    1. WorldState 契约：to_vector / from_vector / distance / copy
    2. Action 契约：apply 返回新状态（不修改原状态）
    3. 安全约束：is_safe / is_physiologically_valid / safety_violations
    4. 临床评分：SOFA 评分逻辑
    5. 异构数据融合：体征 + 检验 + 用药 + 诊断
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._clinical_world_state import (
    DRUG_EFFECT_TABLE,
    DRUG_PKPD_TABLE,
    N_VITALS,
    STATE_VECTOR_DIM,
    VITAL_NAMES,
    ActionTypeConstants,
    MedicalAction,
    Medication,
    PatientState,
    emax_effect,
)
from mci_world_model.sdk._world_state import Action, WorldState

# =============================================================================
# 辅助函数
# =============================================================================


def make_normal_state(patient_id: str = "P001") -> PatientState:
    """构造生理正常范围的 PatientState。"""
    vitals = np.array([[75.0, 120.0, 80.0, 98.0, 16.0, 36.8, 15.0]])
    return PatientState(
        vital_signs=vitals,
        patient_id=patient_id,
        age=65,
        gender="M",
    )


def make_abnormal_state() -> PatientState:
    """构造异常体征的 PatientState（心动过速+低血压+低血氧）。"""
    vitals = np.array([[130.0, 80.0, 50.0, 88.0, 28.0, 38.5, 10.0]])
    return PatientState(vital_signs=vitals, patient_id="P002")


# =============================================================================
# 1. WorldState 契约验证
# =============================================================================


class TestWorldStateContract:
    """验证 PatientState 正确实现 WorldState ABC 契约。"""

    def test_is_worldstate_subclass(self):
        """PatientState 是 WorldState 子类。"""
        assert issubclass(PatientState, WorldState)

    def test_to_vector_dimension(self):
        """to_vector 输出维度 = 7 体征 + 6 检验 = 13。"""
        state = make_normal_state()
        vec = state.to_vector()
        assert vec.shape == (STATE_VECTOR_DIM,)
        assert vec.dtype == np.float64

    def test_to_from_vector_roundtrip(self):
        """to_vector → from_vector 往返一致性。"""
        state = make_normal_state()
        state.lab_values = {"potassium": 4.0, "creatinine": 1.0}
        vec = state.to_vector()
        restored = PatientState.from_vector(vec)
        # 体征应精确恢复
        np.testing.assert_allclose(restored.vital_signs[-1], state.vital_signs[-1], atol=1e-10)
        # 检验值应恢复
        assert restored.lab_values["potassium"] == 4.0
        assert restored.lab_values["creatinine"] == 1.0

    def test_distance_zero_for_identical_states(self):
        """相同状态的距离为 0。"""
        s1 = make_normal_state()
        s2 = make_normal_state()
        assert s1.distance(s2) == pytest.approx(0.0, abs=1e-10)

    def test_distance_positive_for_different_states(self):
        """不同状态的距离 > 0。"""
        s1 = make_normal_state()
        s2 = make_abnormal_state()
        d = s1.distance(s2)
        assert d > 0.0
        # 异常程度大，距离应较大
        assert d > 0.5

    def test_distance_symmetric(self):
        """距离度量对称：d(a,b) == d(b,a)。"""
        s1 = make_normal_state()
        s2 = make_abnormal_state()
        assert s1.distance(s2) == pytest.approx(s2.distance(s1))

    def test_copy_is_deep(self):
        """copy() 是深拷贝，修改副本不影响原状态。"""
        state = make_normal_state()
        state.medications.append(Medication(name="dopamine", dose=5.0))
        copied = state.copy()
        # 修改副本
        copied.vital_signs[-1][0] = 200.0
        copied.medications[0].dose = 999.0
        # 原状态不受影响
        assert state.vital_signs[-1][0] == 75.0
        assert state.medications[0].dose == 5.0

    def test_copy_preserves_all_fields(self):
        """copy() 保留所有字段。"""
        state = make_normal_state()
        state.lab_values = {"potassium": 4.5}
        state.diagnoses = ["I50.9"]  # 心衰
        copied = state.copy()
        assert copied.lab_values == state.lab_values
        assert copied.diagnoses == state.diagnoses
        assert copied.patient_id == state.patient_id


# =============================================================================
# 2. 安全约束验证
# =============================================================================


class TestSafetyConstraints:
    """验证生理范围安全约束。"""

    def test_normal_state_is_safe(self):
        """正常体征 is_safe 返回 True。"""
        assert make_normal_state().is_safe() is True

    def test_abnormal_state_not_safe(self):
        """异常体征 is_safe 返回 False。"""
        assert make_abnormal_state().is_safe() is False

    def test_safety_violations_lists_abnormal(self):
        """safety_violations 列出所有异常体征。"""
        state = make_abnormal_state()  # HR=130, SBP=80, SpO2=88, ...
        violations = state.safety_violations()
        assert len(violations) > 0
        # 心率 130 > 100 应被报告
        assert any("heart_rate" in v for v in violations)

    def test_normal_state_no_violations(self):
        """正常体征无违规。"""
        assert len(make_normal_state().safety_violations()) == 0

    def test_physiologically_valid_rejects_extreme(self):
        """极端不可信值被 is_physiologically_valid 拒绝。"""
        vitals = np.array([[999.0, 120.0, 80.0, 98.0, 16.0, 36.8, 15.0]])
        state = PatientState(vital_signs=vitals)
        assert state.is_physiologically_valid() is False

    def test_physiologically_valid_accepts_abnormal_but_feasible(self):
        """异常但生理可行的值通过 is_physiologically_valid。"""
        # HR=130 异常但 < 220 可行
        assert make_abnormal_state().is_physiologically_valid() is True


# =============================================================================
# 3. SOFA 评分验证
# =============================================================================


class TestSOFAScore:
    """验证简化 SOFA 评分逻辑。"""

    def test_normal_state_low_sofa(self):
        """正常体征 SOFA 评分接近 0。"""
        score = make_normal_state().sofa_score()
        assert score < 1.0

    def test_abnormal_state_higher_sofa(self):
        """异常体征 SOFA 评分更高。"""
        normal = make_normal_state().sofa_score()
        abnormal = make_abnormal_state().sofa_score()
        assert abnormal > normal

    def test_severe_hypoxia_increases_sofa(self):
        """严重低氧血症增加 SOFA。"""
        vitals = np.array([[75.0, 120.0, 80.0, 85.0, 16.0, 36.8, 15.0]])
        state = PatientState(vital_signs=vitals)
        assert state.sofa_score() >= 3.0  # SpO2 < 90 → +3

    def test_low_gcs_increases_sofa(self):
        """低 GCS 增加神经系统 SOFA。"""
        vitals = np.array([[75.0, 120.0, 80.0, 98.0, 16.0, 36.8, 5.0]])  # GCS=5
        state = PatientState(vital_signs=vitals)
        assert state.sofa_score() >= 4.0  # GCS < 6 → +4


# =============================================================================
# 4. MedicalAction 契约验证
# =============================================================================


class TestMedicalAction:
    """验证 MedicalAction 正确实现 Action ABC 契约。"""

    def test_is_action_subclass(self):
        """MedicalAction 是 Action 子类。"""
        assert issubclass(MedicalAction, Action)

    def test_apply_returns_new_state(self):
        """apply 返回新 PatientState。"""
        state = make_normal_state()
        action = MedicalAction(
            action_type=ActionTypeConstants.DRUG,
            target="dopamine",
            magnitude=5.0,
            unit="μg/kg/min",
        )
        new_state = action.apply(state)
        assert isinstance(new_state, PatientState)
        assert new_state is not state

    def test_apply_does_not_mutate_original(self):
        """apply 不修改原状态（函数式语义）。"""
        state = make_normal_state()
        original_hr = state.vital_signs[-1][0]
        action = MedicalAction(target="dopamine", magnitude=5.0)
        action.apply(state)
        assert state.vital_signs[-1][0] == original_hr

    def test_dopamine_increases_heart_rate(self):
        """多巴胺增加心率（药理效应正确性）。"""
        state = make_normal_state()
        original_hr = state.vital_signs[-1][VITAL_NAMES.index("heart_rate")]
        action = MedicalAction(target="dopamine", magnitude=5.0)
        new_state = action.apply(state)
        new_hr = new_state.vital_signs[-1][VITAL_NAMES.index("heart_rate")]
        assert new_hr > original_hr

    def test_metoprolol_decreases_heart_rate(self):
        """美托洛尔降低心率（β受体阻滞剂效应）。"""
        state = make_normal_state()
        original_hr = state.vital_signs[-1][VITAL_NAMES.index("heart_rate")]
        action = MedicalAction(target="metoprolol", magnitude=5.0)
        new_state = action.apply(state)
        new_hr = new_state.vital_signs[-1][VITAL_NAMES.index("heart_rate")]
        assert new_hr < original_hr

    def test_norepinephrine_increases_bp(self):
        """去甲肾上腺素升高收缩压（强升压药）。"""
        state = make_normal_state()
        original_sbp = state.vital_signs[-1][VITAL_NAMES.index("systolic_bp")]
        action = MedicalAction(target="norepinephrine", magnitude=3.0)
        new_state = action.apply(state)
        new_sbp = new_state.vital_signs[-1][VITAL_NAMES.index("systolic_bp")]
        assert new_sbp > original_sbp

    def test_apply_records_medication(self):
        """药物干预记录到用药列表。"""
        state = make_normal_state()
        action = MedicalAction(target="dopamine", magnitude=5.0, unit="μg/kg/min")
        new_state = action.apply(state)
        assert len(new_state.medications) == 1
        assert new_state.medications[0].name == "dopamine"
        assert new_state.medications[0].dose == 5.0

    def test_action_to_vector_dimension(self):
        """动作向量维度 = 4 onehot + 1 magnitude = 5。"""
        action = MedicalAction(target="dopamine", magnitude=5.0)
        vec = action.to_vector()
        assert vec.shape == (11,)

    def test_action_from_vector_roundtrip(self):
        """动作向量往返一致性。"""
        action = MedicalAction(
            action_type=ActionTypeConstants.PROCEDURE,
            target="intubation",
            magnitude=1.0,
        )
        vec = action.to_vector()
        restored = MedicalAction.from_vector(vec)
        assert restored.action_type == ActionTypeConstants.PROCEDURE
        assert restored.magnitude == pytest.approx(1.0)


# =============================================================================
# 5. 异构数据融合验证
# =============================================================================


class TestHeterogeneousData:
    """验证体征+检验+用药+诊断的异构融合。"""

    def test_full_state_construction(self):
        """完整异构状态构造。"""
        vitals = np.array([[80.0, 118.0, 78.0, 97.0, 15.0, 36.6, 15.0]])
        state = PatientState(
            vital_signs=vitals,
            lab_values={"potassium": 4.2, "creatinine": 0.9, "wbc": 8.5},
            medications=[Medication(name="aspirin", dose=100.0, unit="mg", route="PO")],
            diagnoses=["I10"],  # 高血压
            patient_id="P003",
            age=70,
            gender="F",
        )
        assert len(state.lab_values) == 3
        assert len(state.medications) == 1
        assert state.diagnoses == ["I10"]

    def test_to_dict_includes_all_fields(self):
        """to_dict 输出完整审计信息。"""
        state = make_normal_state()
        state.lab_values = {"creatinine": 2.5}
        d = state.to_dict()
        assert d["type"] == "PatientState"
        assert "vital_signs_latest" in d
        assert "lab_values" in d
        assert "is_safe" in d
        assert "sofa_score" in d
        assert d["lab_values"]["creatinine"] == 2.5

    def test_vital_signs_shape_validation(self):
        """体征矩阵列数必须为 7。"""
        with pytest.raises(ValueError, match="列数必须为"):
            PatientState(vital_signs=np.zeros((3, 5)))

    def test_single_row_vital_signs_reshaped(self):
        """1D 体征自动 reshape 为 (1, 7)。"""
        state = PatientState(vital_signs=np.array([75.0, 120, 80, 98, 16, 36.8, 15]))
        assert state.vital_signs.shape == (1, N_VITALS)

    def test_multi_timestep_vital_signs(self):
        """多时间窗体征矩阵正确存储。"""
        vitals = np.random.default_rng(42).normal(0, 1, size=(12, N_VITALS))
        state = PatientState(vital_signs=vitals)
        assert state.vital_signs.shape == (12, N_VITALS)
        # 全量数据完整保留（所有时间窗）
        np.testing.assert_array_equal(state.vital_signs, vitals)
        # to_vector 取最后一个时间窗
        np.testing.assert_allclose(state.to_vector()[:N_VITALS], vitals[-1])


# =============================================================================
# 6. 药效表完整性
# =============================================================================


class TestDrugEffectTable:
    """验证 DRUG_EFFECT_TABLE 的完整性。"""

    def test_all_effects_target_valid_vitals(self):
        """所有药效目标体征都是合法变量名。"""
        for drug, effects in DRUG_EFFECT_TABLE.items():
            for vital_name in effects:
                assert vital_name in VITAL_NAMES, f"药物 {drug} 的效应目标 {vital_name} 不在 VITAL_NAMES 中"

    def test_core_drugs_present(self):
        """核心药物都在药效表中。"""
        core_drugs = ["dopamine", "norepinephrine", "metoprolol", "epinephrine"]
        for drug in core_drugs:
            assert drug in DRUG_EFFECT_TABLE, f"核心药物 {drug} 缺失"


# =============================================================================
# 7. Emax 饱和药理模型（D7 升级）
# =============================================================================


class TestEmaxPharmacology:
    """验证 Emax 饱和剂量-响应模型（DRUG_PKPD_TABLE + emax_effect）。"""

    def test_pkpd_table_covers_all_drugs(self):
        """DRUG_PKPD_TABLE 覆盖 DRUG_EFFECT_TABLE 所有药物。"""
        for drug in DRUG_EFFECT_TABLE:
            assert drug in DRUG_PKPD_TABLE, f"药物 {drug} 缺少 PK/PD 参数"

    def test_pkpd_params_positive_and_finite(self):
        """所有 PK/PD 参数为正有限值（药理学约束）。"""
        for drug, params in DRUG_PKPD_TABLE.items():
            for key in ("EC50", "Emax", "tau"):
                val = params[key]
                assert np.isfinite(val) and val > 0, f"{drug}.{key}={val} 非正有限"

    def test_emax_low_dose_approximates_linear(self):
        """低剂量（D << EC50）退化为线性近似 slope*D。"""
        # dopamine EC50=0.5，剂量 0.01 远小于 EC50
        linear = emax_effect(2.0, 0.01, "dopamine")
        expected = 2.0 * 0.01
        # 相对误差应 < 5%
        assert abs(linear - expected) / expected < 0.05

    def test_emax_high_dose_saturates(self):
        """高剂量趋于饱和，效应远小于线性外推。"""
        # dopamine 高剂量 10.0
        saturated = emax_effect(2.0, 10.0, "dopamine")
        linear = 2.0 * 10.0
        # 饱和值应显著小于线性外推（避免毒性量级）
        assert saturated < linear * 0.1, f"高剂量未饱和: {saturated} vs {linear}"
        # 且为正有限
        assert saturated > 0 and np.isfinite(saturated)

    def test_emax_zero_dose_returns_zero(self):
        """零剂量无效应。"""
        assert emax_effect(2.0, 0.0, "dopamine") == 0.0

    def test_emax_unknown_drug_fallback_linear(self):
        """未知药物退化为纯线性（向后兼容）。"""
        result = emax_effect(1.5, 3.0, "nonexistent_drug")
        assert result == 1.5 * 3.0

    def test_apply_with_emax_changes_state(self):
        """MedicalAction.apply(use_emax=True) 正确应用饱和效应。"""
        state = PatientState(vital_signs=np.array([[80.0] * N_VITALS]))
        action = MedicalAction(
            action_type=ActionTypeConstants.DRUG,
            target="dopamine",
            magnitude=5.0,
        )
        new_linear = action.apply(state, use_emax=False)
        new_emax = action.apply(state, use_emax=True)
        hr_idx = VITAL_NAMES.index("heart_rate")
        # Emax 模式心率变化应小于线性模式（饱和）
        delta_linear = new_linear.vital_signs[-1, hr_idx] - state.vital_signs[-1, hr_idx]
        delta_emax = new_emax.vital_signs[-1, hr_idx] - state.vital_signs[-1, hr_idx]
        assert delta_emax < delta_linear, "Emax 应使高剂量效应小于线性"
        assert delta_emax > 0, "多巴胺应升高心率"

    def test_apply_default_backward_compatible(self):
        """默认 use_emax=False 保持线性（向后兼容）。"""
        state = PatientState(vital_signs=np.array([[80.0] * N_VITALS]))
        action = MedicalAction(
            action_type=ActionTypeConstants.DRUG,
            target="norepinephrine",
            magnitude=2.0,
        )
        new_state = action.apply(state)
        sbp_idx = VITAL_NAMES.index("systolic_bp")
        expected_delta = DRUG_EFFECT_TABLE["norepinephrine"]["systolic_bp"] * 2.0
        actual_delta = new_state.vital_signs[-1, sbp_idx] - state.vital_signs[-1, sbp_idx]
        assert abs(actual_delta - expected_delta) < 1e-9
