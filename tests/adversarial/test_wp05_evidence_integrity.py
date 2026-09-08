"""WP-05 批量诊断与医疗证据完整性对抗测试。"""

from __future__ import annotations

import pytest

from mci_world_model.sdk._medical_causal_sdk import (
    ClinicalEvidence,
    MedicalCausalSDK,
    get_max_batch_queries,
)
from mci_world_model.server.app import MCIAPIHandler
from mci_world_model.server.validation import RequestLimitError

pytestmark = pytest.mark.contract


def test_twenty_thousand_batch_queries_exceed_api_budget() -> None:
    handler = object.__new__(MCIAPIHandler)
    body = {"queries": [{"cause": "A", "effect": "B"} for _ in range(20_000)]}

    with pytest.raises(RequestLimitError, match="批量查询数量超过上限"):
        handler._validate_params("/api/v1/diagnose/batch", body)


def test_sdk_batch_budget_enforced_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCI_MAX_BATCH_QUERIES", "2")
    sdk = MedicalCausalSDK()
    queries = [{"cause": "A", "effect": "B"} for _ in range(3)]

    assert get_max_batch_queries() == 2
    with pytest.raises(ValueError, match="批量查询数量超过上限"):
        sdk.batch_diagnose(queries)


def test_invalid_batch_budget_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCI_MAX_BATCH_QUERIES", "0")

    with pytest.raises(ValueError, match="正整数"):
        get_max_batch_queries()


def test_one_thousand_and_one_evidence_exceeds_api_budget() -> None:
    handler = object.__new__(MCIAPIHandler)
    body = {"evidence": [{"id": str(index)} for index in range(1001)]}

    with pytest.raises(RequestLimitError, match="证据数量超过上限"):
        handler._validate_params("/api/v1/diagnose", body)


def test_five_duplicate_evidence_items_do_not_make_conclusive_result() -> None:
    sdk = MedicalCausalSDK(patient_id="p1", strict_mode=True)
    for index in range(5):
        sdk.add_evidence(
            ClinicalEvidence(
                evidence_id=f"E{index}",
                source=f"source-{index}",
                description="重复的同一段临床描述",
                confidence=0.95,
            )
        )

    diagnosis = sdk.diagnose("低白蛋白", "营养不良")

    assert diagnosis.is_conclusive is False
    assert diagnosis.effective_evidence_count == 1
    assert diagnosis.duplicate_count == 4
    assert "独立证据不足" in " ".join(diagnosis.warnings)


def test_same_content_from_different_sources_is_duplicate() -> None:
    sdk = MedicalCausalSDK(patient_id="p1")
    for index in range(3):
        sdk.add_evidence(
            ClinicalEvidence(
                evidence_id=f"E{index}",
                evidence_type="lab_result",
                source=f"HIS-{index}",
                description="同一份报告内容",
                confidence=0.9,
            )
        )

    unique, duplicate_count, _ = sdk._deduplicate_evidence(list(sdk._evidence), "p1", "A", "B")

    assert len(unique) == 1
    assert duplicate_count == 2


def test_similar_content_is_marked_correlated() -> None:
    sdk = MedicalCausalSDK(patient_id="p1")
    sdk.add_evidence(ClinicalEvidence(evidence_id="A1", description="急性胰腺炎导致上腹剧烈持续疼痛", source="s1"))
    sdk.add_evidence(ClinicalEvidence(evidence_id="A2", description="急性胰腺炎导致上腹剧烈持续疼感", source="s2"))
    sdk.add_evidence(ClinicalEvidence(evidence_id="B1", description="肾功能异常导致蛋白丢失", source="s3"))

    unique, duplicate_count, correlated_ids = sdk._deduplicate_evidence(list(sdk._evidence), "p1", "急性胰腺炎", "腹痛")

    assert len(unique) == 3
    assert duplicate_count == 0
    assert set(correlated_ids) == {"A1", "A2"}


def test_independent_evidence_count_excludes_duplicates_and_correlated_items() -> None:
    sdk = MedicalCausalSDK(patient_id="p1", strict_mode=False)
    for index in range(2):
        sdk.add_evidence(
            ClinicalEvidence(
                evidence_id=f"D{index}",
                description="完全重复的证据内容",
                source=f"source-{index}",
                confidence=0.95,
            )
        )
    sdk.add_evidence(ClinicalEvidence(evidence_id="C1", description="肾功能异常导致蛋白丢失", source="s1"))
    sdk.add_evidence(ClinicalEvidence(evidence_id="C2", description="肾功能异常导致蛋白大量丢失", source="s2"))

    diagnosis = sdk.diagnose("低白蛋白", "营养不良")

    assert diagnosis.duplicate_count == 1
    assert diagnosis.correlated_count == 2
    assert diagnosis.effective_evidence_count == 1
    assert diagnosis.is_conclusive is False
