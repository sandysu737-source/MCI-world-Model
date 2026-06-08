"""
tests/test_modality_encoders.py — 模态特征编码器测试
======================================================

覆盖:
    - VisionEncoder: RGB/灰度/深度帧编码
    - AudioEncoder: 波形编码 + 频谱特征
    - ThermalEncoder: 热成像帧编码
    - 输出维度/类型/可复现性
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._modality_encoders import (
    AudioEncoder,
    ThermalEncoder,
    VisionEncoder,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def vision_enc():
    return VisionEncoder(feature_dim=32)


@pytest.fixture
def audio_enc():
    return AudioEncoder(feature_dim=16)


@pytest.fixture
def thermal_enc():
    return ThermalEncoder(feature_dim=8)


# =============================================================================
# TestVisionEncoder
# =============================================================================


class TestVisionEncoder:
    """VisionEncoder 测试。"""

    def test_output_dim(self, vision_enc):
        frame = np.random.rand(64, 64, 3)
        vec = vision_enc.encode(frame)
        assert vec.shape == (32,)

    def test_grayscale(self, vision_enc):
        gray = np.random.rand(32, 32)
        vec = vision_enc.encode(gray)
        assert vec.shape == (32,)

    def test_depth_frame(self, vision_enc):
        depth = np.random.rand(48, 64) * 10.0
        vec = vision_enc.encode(depth)
        assert vec.shape == (32,)

    def test_deterministic(self, vision_enc):
        frame = np.random.RandomState(42).rand(32, 32, 3)
        v1 = vision_enc.encode(frame)
        v2 = vision_enc.encode(frame)
        np.testing.assert_array_equal(v1, v2)

    def test_different_inputs_different_outputs(self, vision_enc):
        f1 = np.zeros((32, 32, 3))
        f2 = np.ones((32, 32, 3))
        v1 = vision_enc.encode(f1)
        v2 = vision_enc.encode(f2)
        assert not np.allclose(v1, v2)

    def test_feature_dim_property(self, vision_enc):
        assert vision_enc.feature_dim == 32

    def test_small_frame(self, vision_enc):
        """最小帧 2x2。"""
        frame = np.random.rand(2, 2, 3)
        vec = vision_enc.encode(frame)
        assert vec.shape == (32,)

    def test_global_stats(self, vision_enc):
        """全局统计字段正确。"""
        frame = np.full((16, 16), 0.5)
        vec = vision_enc.encode(frame)
        assert vec[0] == pytest.approx(0.5, abs=1e-10)  # mean
        assert vec[1] == pytest.approx(0.0, abs=1e-10)  # std


# =============================================================================
# TestAudioEncoder
# =============================================================================


class TestAudioEncoder:
    """AudioEncoder 测试。"""

    def test_output_dim(self, audio_enc):
        wave = np.random.randn(16000)
        vec = audio_enc.encode(wave)
        assert vec.shape == (16,)

    def test_silence(self, audio_enc):
        """静音 → RMS ≈ 0。"""
        silence = np.zeros(8000)
        vec = audio_enc.encode(silence)
        assert vec[0] == pytest.approx(0.0, abs=1e-10)  # RMS

    def test_sine_wave(self, audio_enc):
        """正弦波编码。"""
        t = np.linspace(0, 1, 16000)
        wave = np.sin(2 * np.pi * 440 * t)
        vec = audio_enc.encode(wave)
        assert vec.shape == (16,)
        assert vec[0] > 0  # RMS > 0

    def test_stereo(self, audio_enc):
        """多通道音频。"""
        stereo = np.random.randn(8000, 2)
        vec = audio_enc.encode(stereo)
        assert vec.shape == (16,)

    def test_deterministic(self, audio_enc):
        wave = np.random.RandomState(42).randn(4000)
        v1 = audio_enc.encode(wave)
        v2 = audio_enc.encode(wave)
        np.testing.assert_array_equal(v1, v2)

    def test_empty_waveform(self, audio_enc):
        vec = audio_enc.encode(np.array([]))
        assert vec.shape == (16,)
        assert np.all(vec == 0)

    def test_custom_sample_rate(self, audio_enc):
        wave = np.random.randn(8000)
        vec = audio_enc.encode(wave, sample_rate=8000)
        assert vec.shape == (16,)


# =============================================================================
# TestThermalEncoder
# =============================================================================


class TestThermalEncoder:
    """ThermalEncoder 测试。"""

    def test_output_dim(self, thermal_enc):
        frame = np.random.rand(32, 32) * 40 + 20
        vec = thermal_enc.encode(frame)
        assert vec.shape == (8,)

    def test_uniform_temp(self, thermal_enc):
        """均匀温度 → std=0, 梯度=0。"""
        frame = np.full((16, 16), 37.0)
        vec = thermal_enc.encode(frame)
        assert vec[0] == pytest.approx(37.0, abs=1e-10)  # mean
        assert vec[1] == pytest.approx(0.0, abs=1e-10)   # std

    def test_hotspot(self, thermal_enc):
        """热点检测。"""
        frame = np.full((16, 16), 25.0)
        frame[8, 8] = 80.0  # 热点
        vec = thermal_enc.encode(frame)
        assert vec[2] == pytest.approx(80.0, abs=1e-10)  # max
        assert vec[4] == pytest.approx(8.0 / 15, abs=1e-2)  # hot_y_norm
        assert vec[5] == pytest.approx(8.0 / 15, abs=1e-2)  # hot_x_norm

    def test_deterministic(self, thermal_enc):
        frame = np.random.RandomState(42).rand(16, 16)
        v1 = thermal_enc.encode(frame)
        v2 = thermal_enc.encode(frame)
        np.testing.assert_array_equal(v1, v2)

    def test_3d_input(self, thermal_enc):
        """3D 输入自动转灰度。"""
        frame = np.random.rand(16, 16, 3)
        vec = thermal_enc.encode(frame)
        assert vec.shape == (8,)
