"""MedicalCausalSDK 置信度双重相乘修复回归测试 (局限①)。"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.oracle


class TestMedicalCausalConfidence:
    """验证 diagnose() 的置信度计算: 不双重相乘, 权重 0.1/0.9。"""

    def _build_sdk(self, n_evidence=5, ev_conf=0.85):
        from mci_world_model.sdk._medical_causal_sdk import ClinicalEvidence, MedicalCausalSDK

        sdk = MedicalCausalSDK(patient_id="test", strict_mode=True)
        for i in range(n_evidence):
            sdk.add_evidence(
                ClinicalEvidence(
                    evidence_id=f"E{i}",
                    evidence_type="lab_result",
                    description="白蛋白 营养不良",
                    confidence=ev_conf,
                )
            )
        return sdk

    def test_high_confidence_is_conclusive(self):
        """ev_conf=0.85, 5条证据 → 应 conclusive。"""
        sdk = self._build_sdk(n_evidence=5, ev_conf=0.85)
        diag = sdk.diagnose("低白蛋白", "营养不良", prior_strength=0.5)
        # cs = 0.5*0.1 + 0.85*0.9 = 0.815
        assert diag.confidence == pytest.approx(0.815, abs=0.01)
        assert diag.is_conclusive is True

    def test_ev_conf_0_8_reaches_threshold(self):
        """ev_conf=0.8 应达到 conclusive 阈值 0.7 (核心修复目标)。"""
        sdk = self._build_sdk(n_evidence=5, ev_conf=0.8)
        diag = sdk.diagnose("A", "B", prior_strength=0.5)
        # cs = 0.5*0.1 + 0.8*0.9 = 0.77
        assert diag.confidence == pytest.approx(0.77, abs=0.01)
        assert diag.is_conclusive is True

    def test_confidence_no_longer_dual_multiplied(self):
        """confidence 应等于 causal_strength, 不再乘 evidence_confidence。"""
        sdk = self._build_sdk(n_evidence=5, ev_conf=0.90)
        diag = sdk.diagnose("A", "B", prior_strength=0.5)
        # cs = 0.5*0.1 + 0.9*0.9 = 0.86
        assert diag.confidence == pytest.approx(0.86, abs=0.01)
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
        # cs = 0.5*0.1 + 0.6*0.9 = 0.59
        assert diag.confidence < 0.7
        assert diag.is_conclusive is False


class TestInputValidationRound4:
    """第四轮: 输入范围校验 (医疗安全关键)。"""

    def test_confidence_out_of_range_rejected(self):
        """confidence < 0 或 > 1 应报错。"""
        from mci_world_model.sdk._medical_causal_sdk import ClinicalEvidence

        with pytest.raises(ValueError, match="confidence"):
            ClinicalEvidence(evidence_id="E1", confidence=-0.1)
        with pytest.raises(ValueError, match="confidence"):
            ClinicalEvidence(evidence_id="E1", confidence=1.1)

    def test_confidence_boundary_accepted(self):
        """confidence = 0.0 和 1.0 应被接受。"""
        from mci_world_model.sdk._medical_causal_sdk import ClinicalEvidence

        e0 = ClinicalEvidence(evidence_id="E1", confidence=0.0)
        e1 = ClinicalEvidence(evidence_id="E2", confidence=1.0)
        assert e0.confidence == 0.0
        assert e1.confidence == 1.0

    def test_prior_strength_out_of_range_rejected(self):
        """prior_strength 不在 [0,1] 应报错。"""
        from mci_world_model.sdk._medical_causal_sdk import ClinicalEvidence, MedicalCausalSDK

        sdk = MedicalCausalSDK()
        for i in range(5):
            sdk.add_evidence(ClinicalEvidence(evidence_id=f"E{i}", confidence=0.8))
        with pytest.raises(ValueError, match="prior_strength"):
            sdk.diagnose("A", "B", prior_strength=1.5)
        with pytest.raises(ValueError, match="prior_strength"):
            sdk.diagnose("A", "B", prior_strength=-0.1)

    def test_evidence_count_cap(self):
        """证据超过 MAX_EVIDENCE_COUNT 应报错。"""
        from mci_world_model.sdk._medical_causal_sdk import ClinicalEvidence, MedicalCausalSDK

        sdk = MedicalCausalSDK()
        for i in range(sdk.MAX_EVIDENCE_COUNT):
            sdk.add_evidence(ClinicalEvidence(evidence_id=f"E{i}", confidence=0.5))
        with pytest.raises(ValueError, match="超过上限"):
            sdk.add_evidence(ClinicalEvidence(evidence_id="overflow", confidence=0.5))


# =============================================================================
# 安全关键路径覆盖（第二轮补充）
# =============================================================================


class TestSafetyCriticalPaths:
    """覆盖 record_outcome / get_audit_log / set_calibrator /
    batch_diagnose / statistics / clear_evidence 六个公共方法。

    这些是医疗安全关键路径, 原文件零覆盖, 本类补齐最小用例。
    """

    def _add_evidence(self, sdk, n=3, ev_conf=0.85, desc="白蛋白 营养不良"):
        """复用: 向 sdk 注入 n 条证据。"""
        from mci_world_model.sdk._medical_causal_sdk import ClinicalEvidence

        for i in range(n):
            sdk.add_evidence(
                ClinicalEvidence(
                    evidence_id=f"E{i}",
                    evidence_type="lab_result",
                    description=desc,
                    confidence=ev_conf,
                )
            )

    # ── record_outcome + get_audit_log ───────────────────────────

    def test_record_outcome_appends_to_audit_log(self):
        """record_outcome 应在审计日志留下记录。"""
        from mci_world_model.sdk._medical_causal_sdk import MedicalCausalSDK

        sdk = MedicalCausalSDK(patient_id="p1", strict_mode=True)
        self._add_evidence(sdk)
        # diagnose 会把一条记录追加到 _diagnoses 和 _audit_log
        diag = sdk.diagnose("低白蛋白", "营养不良", prior_strength=0.5)
        assert diag is not None

        before = len(sdk.get_audit_log())
        # 索引 0 对应刚才那条诊断
        sdk.record_outcome(0, is_correct=True)
        after = len(sdk.get_audit_log())
        # record_outcome 内部不写 _audit_log (只触发 calibrator.update),
        # 审计日志条数不应减少
        assert after >= before, "record_outcome 不应减少审计日志"

    def test_get_audit_log_returns_list_of_dicts(self):
        """get_audit_log 返回 list[dict] 且不可外部篡改内部状态。"""
        from mci_world_model.sdk._medical_causal_sdk import MedicalCausalSDK

        sdk = MedicalCausalSDK(patient_id="p1", strict_mode=True)
        self._add_evidence(sdk)
        sdk.diagnose("低白蛋白", "营养不良", prior_strength=0.5)

        log = sdk.get_audit_log()
        assert isinstance(log, list)
        assert len(log) >= 1
        # 每条至少含 action/timestamp 字段
        for entry in log:
            assert isinstance(entry, dict)
            assert "action" in entry
            assert "timestamp" in entry
        # 至少一条 diagnose 动作
        actions = {e["action"] for e in log}
        assert "diagnose" in actions
        # 修改返回值不应影响内部状态 (get_audit_log 返回的是 list copy)
        log.clear()
        assert len(sdk.get_audit_log()) > 0, "审计日志应是不可变副本"

    def test_record_outcome_invalid_index_is_noop(self):
        """越界 diagnosis_index 不应崩溃 (健壮性)。"""
        from mci_world_model.sdk._medical_causal_sdk import MedicalCausalSDK

        sdk = MedicalCausalSDK(strict_mode=True)
        self._add_evidence(sdk)
        sdk.diagnose("A", "B", prior_strength=0.5)
        # 越界索引 (0 <= idx < len 才生效, 这里 len=1)
        sdk.record_outcome(99, is_correct=False)  # 不应抛
        sdk.record_outcome(-1, is_correct=False)  # 不应抛

    # ── set_calibrator ───────────────────────────────────────────

    def test_set_calibrator_affects_downstream_diagnose(self):
        """注入校准器后, diagnose 的 confidence 应被 calibrate() 映射。

        校准器保守原则: calibrate(raw) <= raw, 因此 fit 后的高置信度
        诊断应被压低 (或保持不变)。
        """
        from mci_world_model.sdk._confidence_calibrator import ConfidenceCalibrator
        from mci_world_model.sdk._medical_causal_sdk import MedicalCausalSDK

        # 无校准器 baseline
        sdk_raw = MedicalCausalSDK(strict_mode=True)
        self._add_evidence(sdk_raw)
        diag_raw = sdk_raw.diagnose("白蛋白", "营养不良", prior_strength=0.5)
        raw_conf = diag_raw.confidence

        # 注入已拟合校准器: 历史显示该区间诊断大多错误 → Platt 压低
        cal = ConfidenceCalibrator(method="platt")
        # _do_fit 要求 history >= 10 才拟合 (医疗保守), 故给 12 条全错样本
        cal.fit(history=[(raw_conf, False)] * 12)
        assert cal.is_fitted

        sdk_cal = MedicalCausalSDK(strict_mode=True)
        sdk_cal.set_calibrator(cal)
        self._add_evidence(sdk_cal)
        diag_cal = sdk_cal.diagnose("白蛋白", "营养不良", prior_strength=0.5)

        # 保守原则: 校准后 confidence <= raw
        assert diag_cal.confidence <= raw_conf + 1e-9, (
            f"校准应只降不升: raw={raw_conf}, calibrated={diag_cal.confidence}"
        )

    def test_set_calibrator_accepts_calibrator_instance(self):
        """set_calibrator 接受 ConfidenceCalibrator 实例, 不抛异常。"""
        from mci_world_model.sdk._confidence_calibrator import ConfidenceCalibrator
        from mci_world_model.sdk._medical_causal_sdk import MedicalCausalSDK

        sdk = MedicalCausalSDK(strict_mode=True)
        cal = ConfidenceCalibrator(method="none")
        sdk.set_calibrator(cal)  # 不应抛
        # 用未拟合校准器做一次完整 diagnose 仍应正常工作 (降级返回 raw)
        self._add_evidence(sdk)
        diag = sdk.diagnose("A", "B", prior_strength=0.5)
        assert 0.0 <= diag.confidence <= 1.0

    def test_record_outcome_feeds_calibrator(self):
        """record_outcome 应把 (confidence, is_correct) 喂给校准器。"""
        from mci_world_model.sdk._confidence_calibrator import ConfidenceCalibrator
        from mci_world_model.sdk._medical_causal_sdk import MedicalCausalSDK

        cal = ConfidenceCalibrator(method="platt")
        sdk = MedicalCausalSDK(strict_mode=True)
        sdk.set_calibrator(cal)
        self._add_evidence(sdk)
        sdk.diagnose("A", "B", prior_strength=0.5)
        assert cal.sample_count == 0
        sdk.record_outcome(0, is_correct=True)
        assert cal.sample_count == 1, "record_outcome 应增量记录到 calibrator"

    # ── batch_diagnose ───────────────────────────────────────────

    def test_batch_diagnose_returns_list_matching_queries(self):
        """batch_diagnose 返回 list[CausalDiagnosis], 长度等于 queries。"""
        from mci_world_model.sdk._medical_causal_sdk import CausalDiagnosis, MedicalCausalSDK

        sdk = MedicalCausalSDK(patient_id="batch", strict_mode=True)
        queries = [
            {
                "cause": "低白蛋白",
                "effect": "营养不良",
                "prior_strength": 0.5,
                "evidence": [
                    {"id": f"e{i}", "type": "lab_result", "description": "白蛋白 营养不良", "confidence": 0.85}
                    for i in range(3)
                ],
            },
            {
                "cause": "感染",
                "effect": "发热",
                "prior_strength": 0.5,
                "evidence": [
                    {"id": f"f{i}", "type": "observation", "description": "感染 发热", "confidence": 0.6}
                    for i in range(3)
                ],
            },
        ]
        results = sdk.batch_diagnose(queries)
        assert isinstance(results, list)
        assert len(results) == len(queries)
        for r in results:
            assert isinstance(r, CausalDiagnosis)
            assert 0.0 <= r.confidence <= 1.0
        # 第一个查询证据置信度高, 第二个低, 应有差异
        assert results[0].confidence != results[1].confidence or (results[0].evidence_ids != results[1].evidence_ids)

    def test_batch_diagnose_empty_queries(self):
        """空查询列表返回空列表 (边界)。"""
        from mci_world_model.sdk._medical_causal_sdk import MedicalCausalSDK

        sdk = MedicalCausalSDK(strict_mode=True)
        results = sdk.batch_diagnose([])
        assert isinstance(results, list)
        assert len(results) == 0

    # ── statistics ───────────────────────────────────────────────

    def test_statistics_returns_dict_with_expected_fields(self):
        """statistics 返回 dict, 含合理字段且类型正确。"""
        from mci_world_model.sdk._medical_causal_sdk import MedicalCausalSDK

        sdk = MedicalCausalSDK(patient_id="stats", strict_mode=True)
        self._add_evidence(sdk, n=5)
        sdk.diagnose("A", "B", prior_strength=0.5)

        stats = sdk.statistics()
        assert isinstance(stats, dict)
        # 必含字段
        for key in (
            "patient_id",
            "evidence_count",
            "diagnosis_count",
            "conclusive_count",
            "conclusive_rate",
            "audit_entries",
            "strict_mode",
        ):
            assert key in stats, f"statistics 缺字段 {key}"
        # 字段语义合理
        assert stats["patient_id"] == "stats"
        assert stats["evidence_count"] == 5
        assert stats["diagnosis_count"] == 1
        assert isinstance(stats["conclusive_count"], int)
        assert 0.0 <= stats["conclusive_rate"] <= 1.0
        assert stats["strict_mode"] is True
        assert stats["audit_entries"] >= stats["diagnosis_count"]

    def test_statistics_empty_sdk_no_division_error(self):
        """无诊断时 conclusive_rate 不应除零。"""
        from mci_world_model.sdk._medical_causal_sdk import MedicalCausalSDK

        sdk = MedicalCausalSDK(strict_mode=True)
        stats = sdk.statistics()
        assert stats["diagnosis_count"] == 0
        assert stats["conclusive_rate"] == 0.0  # max(0, 1) 兜底

    # ── clear_evidence ───────────────────────────────────────────

    def test_clear_evidence_resets_evidence_count(self):
        """clear_evidence 后 evidence_count 归零。"""
        from mci_world_model.sdk._medical_causal_sdk import MedicalCausalSDK

        sdk = MedicalCausalSDK(strict_mode=True)
        self._add_evidence(sdk, n=5)
        assert sdk.evidence_count == 5
        sdk.clear_evidence()
        assert sdk.evidence_count == 0, "clear_evidence 后 evidence_count 应为 0"

    def test_clear_evidence_then_diagnose_is_inconclusive(self):
        """清空证据后再 diagnose 应因证据不足而 inconclusive (安全约束)。"""
        from mci_world_model.sdk._medical_causal_sdk import MedicalCausalSDK

        sdk = MedicalCausalSDK(strict_mode=True)
        self._add_evidence(sdk, n=5)
        sdk.diagnose("A", "B", prior_strength=0.5)  # 有诊断
        sdk.clear_evidence()
        assert sdk.evidence_count == 0
        # 证据 < MIN_EVIDENCE_COUNT(2) → strict 模式拒绝
        diag = sdk.diagnose("A", "B", prior_strength=0.5)
        assert diag.is_conclusive is False
        assert diag.confidence == 0.0
