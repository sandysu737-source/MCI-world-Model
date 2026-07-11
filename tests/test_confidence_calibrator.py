"""ConfidenceCalibrator 测试 — Adapt-EPA 借鉴校准层。"""

from __future__ import annotations

import numpy as np

from mci_world_model.sdk._confidence_calibrator import ConfidenceCalibrator


class TestPlattScaling:
    """Platt Scaling 校准测试。"""

    def test_unfitted_returns_raw(self):
        """未拟合时返回原始值。"""
        cal = ConfidenceCalibrator(method="platt")
        assert cal.calibrate(0.8) == 0.8

    def test_fit_basic(self):
        """拟合后参数更新。"""
        cal = ConfidenceCalibrator(method="platt")
        # 构造数据: raw=0.3 → 正确率 0.2; raw=0.8 → 正确率 0.6
        # confidence 系统性偏高, 校准应降低
        rng = np.random.RandomState(42)
        history = []
        for _ in range(50):
            history.append((0.3, rng.random() < 0.2))  # 低 confidence, 低正确率
        for _ in range(50):
            history.append((0.8, rng.random() < 0.6))  # 高 confidence, 中等正确率
        cal.fit(history)
        assert cal.is_fitted

    def test_calibrate_reduces_overconfident(self):
        """过高的 confidence 被校准降低 (保守原则)。"""
        cal = ConfidenceCalibrator(method="platt")
        rng = np.random.RandomState(42)
        # raw=0.9 但实际正确率只有 0.5 → 校准应降低
        history = [(0.9, rng.random() < 0.5) for _ in range(100)]
        cal.fit(history)
        calibrated = cal.calibrate(0.9)
        assert calibrated <= 0.9, f"校准后 {calibrated} > 原始 0.9, 违反保守原则"
        assert calibrated < 0.9, f"校准后 {calibrated} 未降低, 预期降低"

    def test_calibrate_never_exceeds_raw(self):
        """校准后值永远 ≤ 原始值 (医疗安全)。"""
        cal = ConfidenceCalibrator(method="platt")
        rng = np.random.RandomState(123)
        # 构造正确率高于 confidence 的场景 (理论上校准想抬高, 但保守原则阻止)
        history = [(0.3, rng.random() < 0.8) for _ in range(100)]
        cal.fit(history)
        for raw in [0.1, 0.3, 0.5, 0.7, 0.9]:
            calibrated = cal.calibrate(raw)
            assert calibrated <= raw + 1e-9, f"raw={raw}: calibrated={calibrated} > raw"

    def test_ece_decreases_after_calibration(self):
        """校准后 ECE 应有改善趋势。"""
        rng = np.random.RandomState(42)
        # 构造系统性偏高的 confidence
        history = []
        for _ in range(200):
            actual = rng.random() < 0.4
            raw = 0.75 if actual else 0.65  # 都偏高
            history.append((raw, actual))

        cal = ConfidenceCalibrator(method="platt")
        cal.fit(history)
        ece_after = cal.expected_calibration_error()

        # 模拟未校准的 ECE (直接用 raw confidence)
        raws = np.array([h[0] for h in history])
        outcomes = np.array([1.0 if h[1] else 0.0 for h in history])
        bins = np.linspace(0, 1, 11)
        ece_before = 0.0
        for i in range(10):
            mask = (raws >= bins[i]) & (raws < bins[i + 1] if i < 9 else raws <= bins[i + 1])
            n = np.sum(mask)
            if n > 0:
                ece_before += (n / len(raws)) * abs(np.mean(raws[mask]) - np.mean(outcomes[mask]))

        # Platt 校准应该降低 ECE (至少不恶化)
        assert ece_after <= ece_before + 0.01


class TestIsotonicRegression:
    """Isotonic Regression 校准测试。"""

    def test_fit_and_calibrate(self):
        cal = ConfidenceCalibrator(method="isotonic")
        rng = np.random.RandomState(42)
        history = []
        for _ in range(50):
            history.append((0.2, rng.random() < 0.1))
        for _ in range(50):
            history.append((0.9, rng.random() < 0.85))
        cal.fit(history)
        assert cal.is_fitted
        # 校准后的值应该单调
        vals = [cal.calibrate(r) for r in [0.1, 0.3, 0.5, 0.7, 0.9]]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1] + 0.01, f"非单调: {vals}"

    def test_conservative_principle(self):
        """Isotonic 也遵守保守原则。"""
        cal = ConfidenceCalibrator(method="isotonic")
        rng = np.random.RandomState(42)
        history = [(0.8, rng.random() < 0.4) for _ in range(100)]
        cal.fit(history)
        calibrated = cal.calibrate(0.8)
        assert calibrated <= 0.8


class TestOnlineUpdate:
    """在线增量更新测试。"""

    def test_update_stores_data(self):
        cal = ConfidenceCalibrator(method="platt")
        for i in range(15):
            cal.update(0.5 + i * 0.01, i % 2 == 0)
        assert cal.sample_count == 15

    def test_refit_uses_accumulated_data(self):
        cal = ConfidenceCalibrator(method="platt")
        for i in range(20):
            cal.update(0.7, i < 10)  # 正确率 50%
        assert not cal.is_fitted
        cal.refit()
        assert cal.is_fitted
        calibrated = cal.calibrate(0.7)
        assert calibrated <= 0.7


class TestDegradation:
    """降级安全性测试。"""

    def test_calibrate_failure_returns_raw(self):
        """校准计算出错时返回原始值。"""
        cal = ConfidenceCalibrator(method="platt")
        cal._fitted = True
        cal._platt_a = float("inf")
        cal._platt_b = float("nan")
        # 不应抛异常
        result = cal.calibrate(0.5)
        assert result == 0.5

    def test_none_method_returns_raw(self):
        cal = ConfidenceCalibrator(method="none")
        cal.fit([(0.5, True)] * 20)
        assert cal.calibrate(0.8) == 0.8

    def test_insufficient_data_no_fit(self):
        """数据不足时不拟合。"""
        cal = ConfidenceCalibrator(method="platt")
        cal.fit([(0.5, True)] * 5)  # < 10
        assert not cal.is_fitted
        assert cal.calibrate(0.8) == 0.8


class TestSDKIntegration:
    """MedicalCausalSDK 集成测试。"""

    def test_set_calibrator(self):
        from mci_world_model.sdk._medical_causal_sdk import MedicalCausalSDK

        sdk = MedicalCausalSDK()
        cal = ConfidenceCalibrator(method="platt")
        sdk.set_calibrator(cal)
        assert sdk._calibrator is cal

    def test_diagnose_with_calibrator(self):
        from mci_world_model.sdk._medical_causal_sdk import (
            ClinicalEvidence,
            MedicalCausalSDK,
        )

        sdk = MedicalCausalSDK()
        cal = ConfidenceCalibrator(method="platt")
        # 拟合: confidence 系统性偏高
        rng = np.random.RandomState(42)
        history = [(0.75, rng.random() < 0.4) for _ in range(100)]
        cal.fit(history)
        sdk.set_calibrator(cal)

        for i in range(5):
            sdk.add_evidence(
                ClinicalEvidence(
                    evidence_id=f"E{i}",
                    description="白蛋白 低",
                    confidence=0.85,
                )
            )
        diag = sdk.diagnose("低白蛋白", "营养不良", 0.5)
        # 校准后 confidence 应 ≤ 未校准值
        assert diag.confidence <= 0.85

    def test_diagnose_without_calibrator_unchanged(self):
        from mci_world_model.sdk._medical_causal_sdk import (
            ClinicalEvidence,
            MedicalCausalSDK,
        )

        sdk = MedicalCausalSDK()
        for i in range(5):
            sdk.add_evidence(
                ClinicalEvidence(
                    evidence_id=f"E{i}",
                    description="白蛋白",
                    confidence=0.85,
                )
            )
        diag = sdk.diagnose("低白蛋白", "营养不良", 0.5)
        # 无校准器, confidence 应为原始公式值
        assert diag.confidence > 0.7
        assert diag.is_conclusive

    def test_record_outcome(self):
        from mci_world_model.sdk._medical_causal_sdk import (
            ClinicalEvidence,
            MedicalCausalSDK,
        )

        sdk = MedicalCausalSDK()
        cal = ConfidenceCalibrator(method="platt")
        sdk.set_calibrator(cal)

        for i in range(5):
            sdk.add_evidence(
                ClinicalEvidence(
                    evidence_id=f"E{i}",
                    description="test",
                    confidence=0.8,
                )
            )
        sdk.diagnose("A", "B", 0.5)
        sdk.record_outcome(0, True)
        assert cal.sample_count == 1


class TestThresholdAware:
    """threshold_aware 分段校准测试。"""

    def test_below_threshold_no_change(self):
        """低于阈值的不校准。"""
        cal = ConfidenceCalibrator(method="threshold_aware")
        cal._platt_a = 2.0
        cal._platt_b = -0.5
        cal._fitted = True
        assert cal.calibrate(0.3) == 0.3
        assert cal.calibrate(0.5) == 0.5
        assert cal.calibrate(0.69) == 0.69

    def test_at_threshold_no_drop(self):
        """阈值处不降 (0.70 → 0.70)。"""
        cal = ConfidenceCalibrator(method="threshold_aware")
        cal._platt_a = 1.5
        cal._platt_b = -0.3
        cal._fitted = True
        assert cal.calibrate(0.70) == 0.70

    def test_near_threshold_minimal_drop(self):
        """阈值附近最多降 2%。"""
        cal = ConfidenceCalibrator(method="threshold_aware")
        cal._platt_a = 1.5
        cal._platt_b = -0.3
        cal._fitted = True
        calibrated = cal.calibrate(0.75)
        assert calibrated >= 0.73, f"降幅过大: 0.75 → {calibrated}"
        assert calibrated <= 0.75

    def test_high_confidence_full_platt(self):
        """高 confidence (>0.85) 用完整 Platt 校准。"""
        cal = ConfidenceCalibrator(method="threshold_aware")
        cal._platt_a = 1.0
        cal._platt_b = -0.2
        cal._fitted = True
        calibrated = cal.calibrate(0.95)
        # Platt: sigmoid(1.0*0.95 - 0.2) = sigmoid(0.75) ≈ 0.679
        assert calibrated < 0.95  # 明显降低
        assert calibrated <= 0.95  # 保守原则

    def test_conservative_principle(self):
        """threshold_aware 也遵守保守原则。"""
        cal = ConfidenceCalibrator(method="threshold_aware")
        cal._platt_a = 0.5
        cal._platt_b = 0.3
        cal._fitted = True
        for raw in [0.1, 0.3, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            calibrated = cal.calibrate(raw)
            assert calibrated <= raw + 1e-9, f"raw={raw}: {calibrated} > raw"

    def test_fit_with_threshold_aware(self):
        """threshold_aware 可以 fit + calibrate。"""
        rng = np.random.RandomState(42)
        history = [(0.8, rng.random() < 0.5) for _ in range(100)]
        cal = ConfidenceCalibrator(method="threshold_aware")
        cal.fit(history)
        assert cal.is_fitted
        # 0.70 应该不降
        assert cal.calibrate(0.70) == 0.70
        # 0.90 应该降
        assert cal.calibrate(0.90) < 0.90
