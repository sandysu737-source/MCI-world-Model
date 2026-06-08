"""
MCI World Model v3.3.0 — 模态特征编码器
==========================================

纯 numpy 实现的多模态特征提取器，将高维传感器信号压缩为
固定维度的特征向量。

三个编码器:
    VisionEncoder   — RGB/Depth 帧 → 32 维特征
    AudioEncoder    — 音频波形 → 16 维特征
    ThermalEncoder  — 热成像帧 → 8 维特征

设计原则:
    - 纯 numpy，零外部依赖（不依赖 torch/opencv/librosa）
    - 统计特征 + 空间池化替代 CNN/ViT
    - 输出固定维度向量，便于 MultimodalFusion 和因果图构建
    - 可复现：相同输入 → 相同输出（无随机性）

性能预期:
    - 不追求 SOTA 精度，重在提供稳定的特征表示
    - 用于因果图构建和惊奇检测，不需要像素级重建能力
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# VisionEncoder — 视觉特征编码器
# =============================================================================


class VisionEncoder:
    """RGB/Depth 帧 → 固定维度特征向量（纯 numpy）。

    特征组成 (feature_dim=32):
        [0:4]   全局统计: mean, std, min, max
        [4:8]   四象限均值 (空间池化)
        [8:16]  边缘强度 (Sobel 近似): 水平+垂直各 4 个块
        [16:24] 颜色/深度直方图 (8 bins)
        [24:32] 梯度方向直方图 (8 bins)

    Example:
        >>> enc = VisionEncoder(feature_dim=32)
        >>> frame = np.random.rand(64, 64, 3)
        >>> vec = enc.encode(frame)
        >>> assert vec.shape == (32,)
    """

    def __init__(self, feature_dim: int = 32):
        self._feature_dim = feature_dim

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def encode(self, frame: np.ndarray) -> np.ndarray:
        """编码视觉帧为特征向量。

        Args:
            frame: 2D 或 3D numpy 数组
                - (H, W) — 灰度/深度/热图
                - (H, W, C) — RGB/多通道

        Returns:
            (feature_dim,) float64 特征向量
        """
        if frame.ndim == 3:
            # 转灰度：简单均值
            gray = np.mean(frame.astype(np.float64), axis=2)
        elif frame.ndim == 2:
            gray = frame.astype(np.float64)
        else:
            # 降维到 2D
            gray = np.mean(frame.astype(np.float64), axis=tuple(range(2, frame.ndim)))

        h, w = gray.shape
        features = np.zeros(self._feature_dim, dtype=np.float64)
        idx = 0

        # [0:4] 全局统计
        if idx + 4 <= self._feature_dim:
            features[idx] = np.mean(gray)
            features[idx + 1] = np.std(gray)
            features[idx + 2] = np.min(gray)
            features[idx + 3] = np.max(gray)
            idx += 4

        # [4:8] 四象限均值 (空间池化)
        if idx + 4 <= self._feature_dim:
            mh, mw = h // 2, w // 2
            if mh > 0 and mw > 0:
                features[idx] = np.mean(gray[:mh, :mw])
                features[idx + 1] = np.mean(gray[:mh, mw:])
                features[idx + 2] = np.mean(gray[mh:, :mw])
                features[idx + 3] = np.mean(gray[mh:, mw:])
            idx += 4

        # [8:16] Sobel 近似边缘强度 (2x2 blocks)
        if idx + 8 <= self._feature_dim and h >= 3 and w >= 3:
            # Sobel-X: [-1 0 1; -2 0 2; -1 0 1]
            gx = np.zeros((h - 2, w - 2), dtype=np.float64)
            gy = np.zeros((h - 2, w - 2), dtype=np.float64)
            for i in range(h - 2):
                for j in range(w - 2):
                    patch = gray[i : i + 3, j : j + 3]
                    gx[i, j] = -patch[0, 0] + patch[0, 2] - 2 * patch[1, 0] + 2 * patch[1, 2] - patch[2, 0] + patch[2, 2]
                    gy[i, j] = -patch[0, 0] - 2 * patch[0, 1] - patch[0, 2] + patch[2, 0] + 2 * patch[2, 1] + patch[2, 2]
            edge_mag = np.sqrt(gx ** 2 + gy ** 2)
            eh, ew = edge_mag.shape
            qh, qw = eh // 2, ew // 2
            if qh > 0 and qw > 0:
                # 8 个块: 2x4 grid
                block_h = max(1, eh // 4)
                for bi in range(4):
                    r_start = bi * block_h
                    r_end = min((bi + 1) * block_h, eh)
                    features[idx + bi] = np.mean(edge_mag[r_start:r_end, :qw])
                    features[idx + bi + 4] = np.mean(edge_mag[r_start:r_end, qw:])
            idx += 8

        # [16:24] 值直方图 (8 bins)
        n_hist_bins = 8
        if idx + n_hist_bins <= self._feature_dim:
            flat = gray.flatten()
            if len(flat) > 0:
                vmin, vmax = float(np.min(flat)), float(np.max(flat))
                if vmax - vmin < 1e-10:
                    hist = np.zeros(n_hist_bins, dtype=np.float64)
                    hist[0] = 1.0
                else:
                    hist, _ = np.histogram(flat, bins=n_hist_bins, range=(vmin, vmax))
                    hist = hist.astype(np.float64) / max(len(flat), 1)
                features[idx : idx + n_hist_bins] = hist
            idx += n_hist_bins

        # [24:32] 梯度方向直方图 (8 bins)
        if idx + 8 <= self._feature_dim and h >= 3 and w >= 3:
            angles = np.arctan2(gy, gx)  # [-pi, pi]
            angle_bins = 8
            hist_angles, _ = np.histogram(
                angles.flatten(), bins=angle_bins, range=(-np.pi, np.pi),
            )
            total = max(np.sum(hist_angles), 1)
            features[idx : idx + 8] = hist_angles.astype(np.float64) / total
            idx += 8

        # 填充剩余维度为零
        return features


# =============================================================================
# AudioEncoder — 音频特征编码器
# =============================================================================


class AudioEncoder:
    """音频波形 → 固定维度统计特征（纯 numpy）。

    特征组成 (feature_dim=16):
        [0:4]   全局: RMS, 峰值, 零交叉率, 能量
        [4:8]   分帧能量包络 (4 段)
        [8:12]  频谱质心 + 带宽 + 滚降 + 平坦度
        [12:16] 过零率分帧 (4 段)

    Example:
        >>> enc = AudioEncoder(feature_dim=16)
        >>> wave = np.random.randn(16000)  # 1秒 @ 16kHz
        >>> vec = enc.encode(wave)
        >>> assert vec.shape == (16,)
    """

    def __init__(self, feature_dim: int = 16):
        self._feature_dim = feature_dim

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def encode(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """编码音频波形为特征向量。

        Args:
            waveform: 1D 音频波形 (N_samples,) 或 2D (N_samples, N_channels)
            sample_rate: 采样率 (Hz)

        Returns:
            (feature_dim,) float64 特征向量
        """
        if waveform.ndim > 1:
            # 多通道取均值
            waveform = np.mean(waveform.astype(np.float64), axis=1)
        else:
            waveform = waveform.astype(np.float64)

        n = len(waveform)
        features = np.zeros(self._feature_dim, dtype=np.float64)
        idx = 0

        if n == 0:
            return features

        # [0:4] 全局统计
        if idx + 4 <= self._feature_dim:
            features[idx] = float(np.sqrt(np.mean(waveform ** 2)))  # RMS
            features[idx + 1] = float(np.max(np.abs(waveform)))  # 峰值
            # 零交叉率
            zc = np.sum(np.abs(np.diff(np.sign(waveform))) > 0)
            features[idx + 2] = float(zc) / max(n - 1, 1)
            features[idx + 3] = float(np.sum(waveform ** 2))  # 总能量
            idx += 4

        # [4:8] 分帧能量包络 (4 段)
        n_segments = 4
        if idx + n_segments <= self._feature_dim:
            seg_len = max(1, n // n_segments)
            for i in range(n_segments):
                start = i * seg_len
                end = min((i + 1) * seg_len, n)
                segment = waveform[start:end]
                features[idx + i] = float(np.mean(segment ** 2))
            idx += n_segments

        # [8:12] 频谱特征 (FFT-based)
        if idx + 4 <= self._feature_dim and n >= 8:
            fft_vals = np.abs(np.fft.rfft(waveform))
            freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
            total_energy = max(np.sum(fft_vals ** 2), 1e-10)

            # 频谱质心
            spectral_centroid = float(np.sum(freqs * fft_vals ** 2) / total_energy)
            features[idx] = spectral_centroid / max(sample_rate / 2, 1)

            # 频谱带宽 (标准差)
            spectral_bw = float(np.sqrt(
                np.sum(((freqs - spectral_centroid) ** 2) * fft_vals ** 2) / total_energy
            ))
            features[idx + 1] = spectral_bw / max(sample_rate / 2, 1)

            # 频谱滚降 (85% 能量截止频率)
            cumsum = np.cumsum(fft_vals ** 2)
            rolloff_idx = np.searchsorted(cumsum, 0.85 * total_energy)
            rolloff_idx = min(rolloff_idx, len(freqs) - 1)
            features[idx + 2] = float(freqs[rolloff_idx]) / max(sample_rate / 2, 1)

            # 频谱平坦度 (几何均值/算术均值)
            fft_pos = fft_vals[fft_vals > 0]
            if len(fft_pos) > 1:
                geo_mean = np.exp(np.mean(np.log(fft_pos + 1e-10)))
                arith_mean = np.mean(fft_pos)
                features[idx + 3] = float(geo_mean / max(arith_mean, 1e-10))
            idx += 4

        # [12:16] 过零率分帧 (4 段)
        if idx + n_segments <= self._feature_dim:
            signs = np.sign(waveform)
            zc_arr = np.abs(np.diff(signs)) > 0
            seg_len = max(1, len(zc_arr) // n_segments)
            for i in range(n_segments):
                start = i * seg_len
                end = min((i + 1) * seg_len, len(zc_arr))
                features[idx + i] = float(np.sum(zc_arr[start:end])) / max(end - start, 1)
            idx += n_segments

        return features


# =============================================================================
# ThermalEncoder — 热感应特征编码器
# =============================================================================


class ThermalEncoder:
    """热成像帧 → 固定维度温度分布特征（纯 numpy）。

    特征组成 (feature_dim=8):
        [0:4]   温度统计: mean, std, max (热点), min
        [4:6]   热点位置: (hot_y_norm, hot_x_norm) 归一化坐标
        [6:8]   梯度统计: mean_gradient, max_gradient

    Example:
        >>> enc = ThermalEncoder(feature_dim=8)
        >>> thermal = np.random.rand(32, 32) * 40 + 20  # 20-60°C
        >>> vec = enc.encode(thermal)
        >>> assert vec.shape == (8,)
    """

    def __init__(self, feature_dim: int = 8):
        self._feature_dim = feature_dim

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def encode(self, thermal_frame: np.ndarray) -> np.ndarray:
        """编码热成像帧为特征向量。

        Args:
            thermal_frame: 2D numpy 数组 (H, W) 温度值

        Returns:
            (feature_dim,) float64 特征向量
        """
        if thermal_frame.ndim > 2:
            thermal_frame = np.mean(thermal_frame.astype(np.float64), axis=2)
        else:
            thermal_frame = thermal_frame.astype(np.float64)

        h, w = thermal_frame.shape
        features = np.zeros(self._feature_dim, dtype=np.float64)
        idx = 0

        # [0:4] 温度统计
        if idx + 4 <= self._feature_dim:
            features[idx] = float(np.mean(thermal_frame))
            features[idx + 1] = float(np.std(thermal_frame))
            features[idx + 2] = float(np.max(thermal_frame))  # 热点温度
            features[idx + 3] = float(np.min(thermal_frame))
            idx += 4

        # [4:6] 热点位置 (归一化坐标)
        if idx + 2 <= self._feature_dim and h > 0 and w > 0:
            max_pos = np.unravel_index(np.argmax(thermal_frame), thermal_frame.shape)
            features[idx] = float(max_pos[0]) / max(h - 1, 1)  # y 归一化
            features[idx + 1] = float(max_pos[1]) / max(w - 1, 1)  # x 归一化
            idx += 2

        # [6:8] 温度梯度
        if idx + 2 <= self._feature_dim and h >= 2 and w >= 2:
            grad_y = np.diff(thermal_frame, axis=0)
            grad_x = np.diff(thermal_frame, axis=1)
            grad_mag = np.sqrt(
                grad_y[:, : w - 1] ** 2 + grad_x[: h - 1, :] ** 2
            )
            features[idx] = float(np.mean(grad_mag))
            features[idx + 1] = float(np.max(grad_mag))
            idx += 2

        return features
