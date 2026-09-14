"""ZvecEmbeddingStore 与当前 zvec API 的兼容测试。"""

from __future__ import annotations

import pytest

pytest.importorskip("zvec")

from mci_world_model.sdk._zvec_store import EmbeddingStoreConfig, ZvecEmbeddingStore


def test_insert_and_vector_search_uses_current_zvec_api(tmp_path) -> None:
    store = ZvecEmbeddingStore(EmbeddingStoreConfig(dim=16, store_path=str(tmp_path / "qa_store")))
    assert store.is_available is True

    inserted = store.insert_qa_pairs(
        [
            {"cause_text": "蛋白质摄入不足", "effect_text": "肌肉流失", "confidence": 0.9},
            {"cause_text": "维生素D不足", "effect_text": "骨密度下降", "confidence": 0.8},
        ]
    )
    assert inserted == 2
    assert store.n_docs == 2

    results = store.search_similar("蛋白质摄入", topk=1)
    assert len(results) == 1
    assert set(results[0]) == {
        "cause_text",
        "effect_text",
        "energy_relation",
        "confidence",
        "score",
    }
