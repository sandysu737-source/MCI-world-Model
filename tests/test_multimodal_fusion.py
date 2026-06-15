"""
tests/test_multimodal_fusion.py — MultimodalFusion 测试
=========================================================

覆盖:
    - 三种融合策略: attention / weighted / concat
    - 空输入/单模态/多模态
    - FusedRepresentation 数据结构
    - encode_to_state
    - 维度对齐
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._multimodal_fusion import (
    FusedRepresentation,
    MultimodalFusion,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fusion_attn():
    return MultimodalFusion(strategy="attention", output_dim=16)


@pytest.fixture
def fusion_weighted():
    return MultimodalFusion(strategy="weighted", output_dim=16)


@pytest.fixture
def fusion_concat():
    return MultimodalFusion(strategy="concat", output_dim=16)


# =============================================================================
# TestFusedRepresentation
# =============================================================================


class TestFusedRepresentation:
    def test_fields(self):
        fr = FusedRepresentation(
            fused_vector=np.zeros(8),
            modality_contributions={"vision": 0.6, "audio": 0.4},
            confidence=0.9,
        )
        assert fr.fused_vector.shape == (8,)
        assert fr.strategy == "attention"

    def test_default_values(self):
        fr = FusedRepresentation(fused_vector=np.zeros(4))
        assert fr.confidence == 1.0
        assert fr.modality_contributions == {}


# =============================================================================
# TestMultimodalFusion
# =============================================================================


class TestMultimodalFusion:
    """三种融合策略测试。"""

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            MultimodalFusion(strategy="invalid")

    def test_empty_input(self, fusion_attn):
        fused = fusion_attn.fuse({})
        assert fused.fused_vector.shape == (16,)
        assert fused.confidence == 0.0

    def test_single_modality(self, fusion_attn):
        features = {"vision": np.random.rand(32)}
        fused = fusion_attn.fuse(features)
        assert fused.fused_vector.shape == (16,)
        assert "vision" in fused.modality_contributions

    def test_two_modalities_attention(self, fusion_attn):
        features = {
            "vision": np.random.rand(32),
            "audio": np.random.rand(16),
        }
        fused = fusion_attn.fuse(features)
        assert fused.fused_vector.shape == (16,)
        assert len(fused.modality_contributions) == 2
        assert fused.strategy == "attention"

    def test_weighted_strategy(self, fusion_weighted):
        features = {
            "vision": np.random.rand(32),
            "audio": np.random.rand(16),
        }
        confidences = {"vision": 0.9, "audio": 0.7}
        fused = fusion_weighted.fuse(features, confidences)
        assert fused.fused_vector.shape == (16,)
        assert fused.strategy == "weighted"
        assert fused.confidence == pytest.approx(0.8, abs=0.01)

    def test_concat_strategy(self, fusion_concat):
        features = {
            "vision": np.random.rand(10),
            "audio": np.random.rand(6),
        }
        fused = fusion_concat.fuse(features)
        assert fused.fused_vector.shape == (16,)
        assert fused.strategy == "concat"
        # 贡献按维度比例
        assert "audio" in fused.modality_contributions
        assert "vision" in fused.modality_contributions

    def test_output_dim_property(self, fusion_attn):
        assert fusion_attn.output_dim == 16

    def test_strategy_property(self, fusion_weighted):
        assert fusion_weighted.strategy == "weighted"

    def test_repr(self, fusion_attn):
        r = repr(fusion_attn)
        assert "MultimodalFusion" in r
        assert "attention" in r

    def test_deterministic(self, fusion_weighted):
        features = {"vision": np.ones(8), "audio": np.ones(4)}
        f1 = fusion_weighted.fuse(features)
        f2 = fusion_weighted.fuse(features)
        np.testing.assert_array_almost_equal(f1.fused_vector, f2.fused_vector)


# =============================================================================
# TestEncodeToState
# =============================================================================


class TestEncodeToState:
    def test_default_state_class(self, fusion_weighted):
        features = {"vision": np.random.rand(16)}
        fused = fusion_weighted.fuse(features)
        state = fusion_weighted.encode_to_state(fused)
        from mci_world_model.sdk._world_state import MultimodalWorldState

        assert isinstance(state, MultimodalWorldState)

    def test_from_vector(self, fusion_weighted):
        fused = FusedRepresentation(fused_vector=np.array([1.0, 2.0, 3.0]))
        state = fusion_weighted.encode_to_state(fused)
        assert state.fused is not None
        np.testing.assert_array_equal(state.fused, [1.0, 2.0, 3.0])
