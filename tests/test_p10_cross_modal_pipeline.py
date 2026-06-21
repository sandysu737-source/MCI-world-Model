"""
tests/test_p10_cross_modal_pipeline.py — P10 跨模态端到端流水线
==============================================================

覆盖:
    - 多模态编码器 (Vision/Depth/Thermal/Force)
    - 统一模态编码器 (UnifiedModalEncoder)
    - 多模态融合 (MultimodalFusion)
    - 跨模态因果推理 (CrossModalCausalReasoner)
    - 多模态图构建 (MultimodalGraphBuilder)
    - 端到端流水线: encoder → fusion → causal graph
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._cross_modal_causal import CrossModalCausalReasoner
from mci_world_model.sdk._modality_encoders import (
    DepthEncoder,
    ForceEncoder,
    ThermalEncoder,
    VisionEncoder,
)
from mci_world_model.sdk._multimodal_fusion import MultimodalFusion
from mci_world_model.sdk._multimodal_graph_builder import MultimodalGraphBuilder
from mci_world_model.sdk._unified_modal_encoder import (
    EncodingResult,
    UnifiedModalEncoder,
)


@pytest.fixture
def image():
    rng = np.random.RandomState(42)
    return rng.rand(64, 64, 3).astype(np.float32)


@pytest.fixture
def depth_map():
    rng = np.random.RandomState(42)
    return rng.rand(64, 64).astype(np.float32)


@pytest.fixture
def thermal():
    rng = np.random.RandomState(42)
    return rng.rand(32, 32).astype(np.float32)


@pytest.fixture
def force_signal():
    rng = np.random.RandomState(42)
    return rng.rand(16).astype(np.float32)


@pytest.fixture
def unified_encoder():
    ue = UnifiedModalEncoder(shared_dim=32)
    ue.register_modality("vision", 64 * 64 * 3)
    ue.register_modality("depth", 64 * 64)
    ue.register_modality("thermal", 32 * 32)
    return ue


class TestModalityEncoders:
    def test_vision_encoder(self, image):
        enc = VisionEncoder(feature_dim=32, learnable_dim=64)
        out = enc.encode(image)
        assert out.shape == (64,)
        assert enc.feature_dim == 32
        assert enc.output_dim == 64

    def test_depth_encoder(self, depth_map):
        enc = DepthEncoder(feature_dim=16, learnable_dim=64)
        out = enc.encode(depth_map)
        assert out.shape == (64,)

    def test_thermal_encoder(self, thermal):
        enc = ThermalEncoder(feature_dim=8, learnable_dim=32)
        out = enc.encode(thermal)
        assert out.shape == (32,)

    def test_force_encoder(self, force_signal):
        enc = ForceEncoder(feature_dim=16, output_dim=32)
        out = enc.encode(force_signal)
        assert out.shape == (32,)

    def test_force_encoder_history(self):
        rng = np.random.RandomState(42)
        window = rng.rand(10, 16).astype(np.float32)
        enc = ForceEncoder(feature_dim=16, output_dim=32)
        out = enc.encode_history(window)
        assert out.shape == (32,)

    def test_vision_deterministic(self, image):
        enc = VisionEncoder(feature_dim=32, learnable_dim=64)
        out1 = enc.encode(image)
        out2 = enc.encode(image)
        assert np.array_equal(out1, out2)


class TestUnifiedEncoder:
    def test_register_and_encode(self, unified_encoder, image):
        flat = image.flatten()
        result = unified_encoder.encode("vision", flat)
        assert isinstance(result, EncodingResult)
        assert result.shared_vector.shape == (32,)

    def test_encode_batch(self, unified_encoder, image):
        flat = image.flatten()
        results = unified_encoder.encode_batch("vision", [flat, flat])
        assert len(results) == 2

    def test_similarity(self, unified_encoder, image):
        flat = image.flatten()
        vec_a = unified_encoder.encode("vision", flat).shared_vector
        vec_b = unified_encoder.encode("vision", flat).shared_vector
        sim = unified_encoder.compute_similarity(vec_a, vec_b)
        assert 0.0 <= sim <= 1.0

    def test_cross_modal_retrieve(self, unified_encoder, image, depth_map):
        results = unified_encoder.cross_modal_retrieve(
            query_modality="vision",
            query_features=image.flatten(),
            candidate_modality="depth",
            candidate_features_list=[depth_map.flatten()],
            top_k=1,
        )
        assert len(results) == 1

    def test_statistics(self, unified_encoder):
        stats = unified_encoder.statistics()
        assert stats["modality_count"] == 3
        assert stats["shared_dim"] == 32


class TestMultimodalFusion:
    def test_weighted_fusion(self, image, depth_map):
        fusion = MultimodalFusion(strategy="weighted", output_dim=64)
        rgb_vec = VisionEncoder(feature_dim=32, learnable_dim=64).encode(image)
        depth_vec = DepthEncoder(feature_dim=32, learnable_dim=64).encode(depth_map)
        result = fusion.fuse({"vision": rgb_vec, "depth": depth_vec})
        assert hasattr(result, "fused_vector")
        assert result.fused_vector.shape == (64,)

    def test_concat_fusion(self, image, depth_map):
        fusion = MultimodalFusion(strategy="concat", output_dim=64)
        rgb_vec = VisionEncoder(feature_dim=32, learnable_dim=64).encode(image)
        depth_vec = DepthEncoder(feature_dim=32, learnable_dim=64).encode(depth_map)
        result = fusion.fuse({"vision": rgb_vec, "depth": depth_vec})
        assert result.fused_vector.shape == (64,)

    def test_attention_fusion(self, image, depth_map):
        fusion = MultimodalFusion(strategy="attention", output_dim=64)
        rgb_vec = VisionEncoder(feature_dim=32, learnable_dim=64).encode(image)
        depth_vec = DepthEncoder(feature_dim=32, learnable_dim=64).encode(depth_map)
        result = fusion.fuse({"vision": rgb_vec, "depth": depth_vec})
        assert result.fused_vector.shape == (64,)

    def test_single_modality(self, image):
        fusion = MultimodalFusion(strategy="weighted", output_dim=32)
        rgb_vec = VisionEncoder(feature_dim=32, learnable_dim=32).encode(image)
        result = fusion.fuse({"vision": rgb_vec})
        assert result.fused_vector.shape == (32,)


class TestMultimodalGraph:
    def test_build_from_features(self, image, depth_map):
        builder = MultimodalGraphBuilder(min_correlation=0.1, max_lag=3)
        rgb_vec = VisionEncoder(feature_dim=16, learnable_dim=32).encode(image)
        depth_vec = DepthEncoder(feature_dim=16, learnable_dim=32).encode(depth_map)
        # Needs timeline format: list of dicts
        timeline = [
            {"vision": rgb_vec, "depth": depth_vec},
            {"vision": rgb_vec * 0.9, "depth": depth_vec * 1.1},
            {"vision": rgb_vec * 1.1, "depth": depth_vec * 0.9},
        ]
        edges = builder.build_from_features(timeline)
        assert isinstance(edges, list)

    def test_build_cross_modality_edges(self):
        rng = np.random.RandomState(42)
        vision_series = rng.rand(50, 16).astype(np.float64)
        depth_series = rng.rand(50, 16).astype(np.float64)
        builder = MultimodalGraphBuilder(min_correlation=0.1)
        edges = builder.build_cross_modality_edges(
            features_a=vision_series,
            features_b=depth_series,
            modality_a="vision",
            modality_b="depth",
        )
        assert isinstance(edges, list)


class TestCrossModalCausal:
    def test_reasoner_creation(self):
        reasoner = CrossModalCausalReasoner()
        assert reasoner is not None

    def test_cross_modal_pipeline(self, image, depth_map):
        rgb_enc = VisionEncoder(feature_dim=32, learnable_dim=64).encode(image)
        depth_enc = DepthEncoder(feature_dim=32, learnable_dim=64).encode(depth_map)

        fusion = MultimodalFusion(strategy="weighted", output_dim=64)
        fused = fusion.fuse({"vision": rgb_enc, "depth": depth_enc})
        assert fused.fused_vector.shape == (64,)

        builder = MultimodalGraphBuilder(min_correlation=0.1)
        timeline = [
            {"vision": rgb_enc, "depth": depth_enc},
            {"vision": rgb_enc * 0.9, "depth": depth_enc * 1.1},
            {"vision": rgb_enc * 1.1, "depth": depth_enc * 0.9},
        ]
        edges = builder.build_from_features(timeline)
        assert isinstance(edges, list)
