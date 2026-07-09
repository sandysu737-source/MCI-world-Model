"""MedicalCausalSDK 置信度双重相乘修复回归测试 (局限①)。"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.oracle


class TestMedicalCausalConfidence:
    """验证 diagnose() 的置信度计算: 不双重相乘, 权重 0.3/0.7。"""

    def _build_sdk(self, n_evidence=5, ev_conf=0.85):
        from mci_world_model.sdk._medical_causal_sdk import MedicalCausalSDK, ClinicalEvidence

        sdk = MedicalCausalSDK(patient_id="test", strict_mode=True)
        for i in range(n_evidence):
            sdk.add_evidence(ClinicalEvidence(
                evidence_id=f"E{i}",
                evidence_type="lab_result",
                description="白蛋白 营养不良",
                confidence=ev_conf,
            ))
        return sdk

    def test_high_confidence_is_conclusive(self):
        """ev_conf=0.85, 5条证据 → 应 conclusive。"""
        sdk = self._build_sdk(n_evidence=5, ev_conf=0.85)
        diag = sdk.diagnose("低白蛋白", "营养不良", prior_strength=0.5)
        # cs = 0.5*0.3 + 0.85*0.7 = 0.745
        assert diag.confidence == pytest.approx(0.745, abs=0.01)
        assert diag.is_conclusive is True

    def test_ev_conf_0_8_reaches_threshold(self):
        """ev_conf=0.8 应达到 conclusive 阈值 0.7 (核心修复目标)。"""
        sdk = self._build_sdk(n_evidence=5, ev_conf=0.8)
        diag = sdk.diagnose("A", "B", prior_strength=0.5)
        # cs = 0.5*0.3 + 0.8*0.7 = 0.71
        assert diag.confidence == pytest.approx(0.71, abs=0.01)
        assert diag.is_conclusive is True

    def test_confidence_no_longer_dual_multiplied(self):
        """confidence 应等于 causal_strength, 不再乘 evidence_confidence。"""
        sdk = self._build_sdk(n_evidence=5, ev_conf=0.90)
        diag = sdk.diagnose("A", "B", prior_strength=0.5)
        # cs = 0.5*0.3 + 0.9*0.7 = 0.78
        assert diag.confidence == pytest.approx(0.78, abs=0.01)
        assert diag.is_conclusive is True

    def test_insufficient_evidence_still_strict(self):
        """证据不足时仍应拒绝 (安全约束不变)。"""
        sdk = self._build_sdk(n_evidence=1, ev_conf=1.0)
        diag = sdk.diagnose("A", "B", prior_strength=0.5)
        assert diag.is_conclusive is False
        assert diag.confidence == 0.0

    def test_low_confidence_still_inconclusive(self):
        """ev_conf=0.6 → confidence 应 < 0.7 → inconclusive (保守性保持)。"""
        sdk = self._build_sdk(n_evidence=5, ev_conf=0.6)
        diag = sdk.diagnose("A", "B", prior_strength=0.5)
        # cs = 0.5*0.3 + 0.6*0.7 = 0.57
        assert diag.confidence < 0.7
        assert diag.is_conclusive is False
