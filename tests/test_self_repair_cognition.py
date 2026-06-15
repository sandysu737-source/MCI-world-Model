"""
tests/test_self_repair_cognition.py — SelfRepairCognition 测试
==============================================================

覆盖:
    - detect_anomaly: 异常检测 + 诊断
    - repair: 4层修复策略
    - repair_and_verify: 完整检测→诊断→修复→验证循环
    - 边界: 无异常/严重异常/未知层
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._self_repair_cognition import (
    AnomalyReport,
    RepairAction,
    SelfRepairCognition,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def repair_cog():
    return SelfRepairCognition(anomaly_threshold=2.0, max_repair_attempts=3)


@pytest.fixture
def close_prediction():
    """预测与实际接近(无异常)。"""
    return np.array([1.0, 2.0, 3.0]), np.array([1.05, 2.1, 3.05])


@pytest.fixture
def far_prediction():
    """预测与实际差距大(有异常)。"""
    return np.array([1.0, 2.0, 3.0]), np.array([5.0, -1.0, 10.0])


# =============================================================================
# TestAnomalyReport
# =============================================================================


class TestAnomalyReport:
    """AnomalyReport 数据类。"""

    def test_creation_no_anomaly(self):
        report = AnomalyReport(is_anomaly=False, error=0.5, error_threshold=2.0)
        assert report.is_anomaly is False
        assert report.error == 0.5
        assert report.diagnosis == {}

    def test_creation_with_anomaly(self):
        report = AnomalyReport(
            is_anomaly=True,
            error=5.0,
            error_threshold=2.0,
            diagnosis={"layer": "perception", "severity": "medium"},
        )
        assert report.is_anomaly is True
        assert report.diagnosis["layer"] == "perception"

    def test_default_fields(self):
        report = AnomalyReport()
        assert report.is_anomaly is False
        assert report.error == 0.0
        assert report.error_threshold == 0.0
        assert report.diagnosis == {}


# =============================================================================
# TestRepairAction
# =============================================================================


class TestRepairAction:
    """RepairAction 数据类。"""

    def test_creation(self):
        action = RepairAction(
            action="recalibrate_encoder",
            layer="perception",
            success=True,
        )
        assert action.action == "recalibrate_encoder"
        assert action.layer == "perception"
        assert action.success is True

    def test_default_fields(self):
        action = RepairAction()
        assert action.action == ""
        assert action.success is False


# =============================================================================
# TestDetectAnomaly
# =============================================================================


class TestDetectAnomaly:
    """异常检测测试。"""

    def test_no_anomaly(self, repair_cog, close_prediction):
        """预测接近实际 → 无异常。"""
        pred, actual = close_prediction
        report = repair_cog.detect_anomaly(pred, actual)
        assert report.is_anomaly is False
        assert report.error < repair_cog.anomaly_threshold

    def test_detects_anomaly(self, repair_cog, far_prediction):
        """预测远离实际 → 有异常。"""
        pred, actual = far_prediction
        report = repair_cog.detect_anomaly(pred, actual)
        assert report.is_anomaly is True
        assert report.error > repair_cog.anomaly_threshold

    def test_diagnosis_dict(self, repair_cog, far_prediction):
        """异常报告应包含诊断字典。"""
        pred, actual = far_prediction
        report = repair_cog.detect_anomaly(pred, actual)
        if report.is_anomaly:
            assert "layer" in report.diagnosis
            assert report.diagnosis["layer"] in ("perception", "prediction", "causal", "unknown")

    def test_error_threshold_stored(self, repair_cog, far_prediction):
        """异常报告应存储阈值。"""
        pred, actual = far_prediction
        report = repair_cog.detect_anomaly(pred, actual)
        assert report.error_threshold == 2.0

    def test_identical_vectors(self, repair_cog):
        """完全相同的向量 → 无异常。"""
        vec = np.array([1.0, 2.0, 3.0])
        report = repair_cog.detect_anomaly(vec, vec)
        assert report.is_anomaly is False
        assert report.error == 0.0

    def test_zero_vectors(self, repair_cog):
        """零向量处理。"""
        zero = np.zeros(5)
        report = repair_cog.detect_anomaly(zero, zero)
        assert report.is_anomaly is False

    def test_single_dim(self, repair_cog):
        """单维度处理。"""
        pred = np.array([1.0])
        actual = np.array([10.0])
        report = repair_cog.detect_anomaly(pred, actual)
        assert report.is_anomaly is True
        assert report.error > 2.0


# =============================================================================
# TestRepair
# =============================================================================


class TestRepair:
    """修复策略测试。"""

    def test_perception_layer_repair(self, repair_cog):
        """感知层异常 → 重新校准编码器。"""
        report = AnomalyReport(
            is_anomaly=True,
            error=5.0,
            error_threshold=2.0,
            diagnosis={"layer": "perception", "severity": "medium"},
        )
        action = repair_cog.repair(report)
        assert action.action == "recalibrate_encoder"
        assert action.layer == "perception"

    def test_prediction_layer_repair(self, repair_cog):
        """预测层异常 → 增加推理步数。"""
        report = AnomalyReport(
            is_anomaly=True,
            error=4.0,
            error_threshold=2.0,
            diagnosis={"layer": "prediction", "severity": "medium"},
        )
        action = repair_cog.repair(report)
        assert action.action == "increase_prediction_steps"
        assert action.layer == "prediction"

    def test_causal_layer_repair(self, repair_cog):
        """因果层异常 → 重新学习结构。"""
        report = AnomalyReport(
            is_anomaly=True,
            error=6.0,
            error_threshold=2.0,
            diagnosis={"layer": "causal", "severity": "high"},
        )
        action = repair_cog.repair(report)
        assert action.action == "relearn_causal_structure"
        assert action.layer == "causal"

    def test_unknown_layer_repair(self, repair_cog):
        """未知层异常 → 安全回退。"""
        report = AnomalyReport(
            is_anomaly=True,
            error=3.0,
            error_threshold=2.0,
            diagnosis={"layer": "unknown"},
        )
        action = repair_cog.repair(report)
        assert action.action == "fallback_to_safe_state"

    def test_no_repair_for_no_anomaly(self, repair_cog):
        """无异常时无需修复。"""
        report = AnomalyReport(is_anomaly=False, error=0.5)
        action = repair_cog.repair(report)
        assert action.action == "none"
        assert action.success is True


# =============================================================================
# TestRepairAndVerify
# =============================================================================


class TestRepairAndVerify:
    """完整修复验证循环测试。"""

    def test_no_anomaly_flow(self, repair_cog, close_prediction):
        """无异常 → 无需修复。"""
        pred, actual = close_prediction
        result = repair_cog.repair_and_verify(pred, actual)
        assert result["anomaly"] is False
        assert result["repair_needed"] is False

    def test_anomaly_with_repair(self, repair_cog, far_prediction):
        """有异常 → 触发修复。"""
        pred, actual = far_prediction
        result = repair_cog.repair_and_verify(pred, actual)
        assert result["anomaly"] is True
        assert result["repair_needed"] is True
        assert "original_error" in result
        assert "repair_action" in result
        assert "verified" in result


# =============================================================================
# TestSelfRepairCognitionMisc
# =============================================================================


class TestSelfRepairCognitionMisc:
    """SelfRepairCognition 杂项测试。"""

    def test_creation(self, repair_cog):
        assert repair_cog.anomaly_threshold == 2.0

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="anomaly_threshold"):
            SelfRepairCognition(anomaly_threshold=0)
        with pytest.raises(ValueError, match="anomaly_threshold"):
            SelfRepairCognition(anomaly_threshold=-1)

    def test_repair_history(self, repair_cog, far_prediction):
        """修复历史跟踪。"""
        pred, actual = far_prediction
        repair_cog.repair_and_verify(pred, actual)
        assert len(repair_cog.repair_history) >= 1

    def test_repair_success_rate(self, repair_cog, far_prediction):
        """修复成功率。"""
        pred, actual = far_prediction
        repair_cog.repair_and_verify(pred, actual)
        rate = repair_cog.repair_success_rate
        assert 0.0 <= rate <= 1.0

    def test_success_rate_no_repairs(self, repair_cog):
        """无修复时成功率为0。"""
        assert repair_cog.repair_success_rate == 0.0
