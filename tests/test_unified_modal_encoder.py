"""tests/test_unified_modal_encoder.py — UnifiedModalEncoder 测试"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._unified_modal_encoder import (
    AlignmentResult,
    EncodingResult,
    ModalityProjection,
    UnifiedModalEncoder,
)


@pytest.fixture
def encoder():
    enc = UnifiedModalEncoder(shared_dim=64)
    enc.register_modality("vision", input_dim=128)
    enc.register_modality("audio", input_dim=64)
    return enc


class TestModalityProjection:
    def test_project(self):
        proj = ModalityProjection("test", input_dim=32, shared_dim=64)
        features = np.random.randn(32)
        out = proj.project(features)
        assert out.shape == (64,)

    def test_project_dimension_mismatch(self):
        proj = ModalityProjection("test", input_dim=32, shared_dim=64)
        # Shorter vector: should pad
        out = proj.project(np.random.randn(16))
        assert out.shape == (64,)
        # Longer vector: should truncate
        out2 = proj.project(np.random.randn(48))
        assert out2.shape == (64,)


class TestUnifiedModalEncoder:
    def test_register(self, encoder):
        assert encoder.modality_count == 2
        assert "vision" in encoder.registered_modalities

    def test_register_duplicate(self, encoder):
        with pytest.raises(ValueError, match="已注册"):
            encoder.register_modality("vision", input_dim=128)

    def test_encode(self, encoder):
        features = np.random.randn(128)
        result = encoder.encode("vision", features)
        assert isinstance(result, EncodingResult)
        assert result.modality == "vision"
        assert result.shared_vector.shape == (64,)

    def test_encode_unregistered(self, encoder):
        with pytest.raises(KeyError, match="未注册"):
            encoder.encode("text", np.random.randn(32))

    def test_encode_batch(self, encoder):
        features_list = [np.random.randn(128) for _ in range(3)]
        results = encoder.encode_batch("vision", features_list)
        assert len(results) == 3

    def test_compute_similarity(self, encoder):
        a = np.random.randn(64)
        b = a + np.random.randn(64) * 0.1
        sim = encoder.compute_similarity(a, b)
        assert -1.0 <= sim <= 1.0

    def test_similarity_identical(self, encoder):
        a = np.random.randn(64)
        sim = encoder.compute_similarity(a, a)
        assert abs(sim - 1.0) < 1e-6

    def test_similarity_orthogonal(self, encoder):
        a = np.array([1.0, 0.0] + [0.0] * 62)
        b = np.array([0.0, 1.0] + [0.0] * 62)
        sim = encoder.compute_similarity(a, b)
        assert abs(sim) < 1e-6

    def test_align(self, encoder):
        result = encoder.align(
            "vision",
            np.random.randn(128),
            "audio",
            np.random.randn(64),
        )
        assert isinstance(result, AlignmentResult)
        assert result.modality_a == "vision"
        assert result.modality_b == "audio"

    def test_cross_modal_retrieve(self, encoder):
        query = np.random.randn(128)
        candidates = [np.random.randn(64) for _ in range(5)]
        results = encoder.cross_modal_retrieve(
            "vision",
            query,
            "audio",
            candidates,
            top_k=3,
        )
        assert len(results) == 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_statistics(self, encoder):
        stats = encoder.statistics()
        assert stats["modality_count"] == 2
        assert stats["shared_dim"] == 64

    def test_invalid_shared_dim(self):
        with pytest.raises(ValueError):
            UnifiedModalEncoder(shared_dim=0)
