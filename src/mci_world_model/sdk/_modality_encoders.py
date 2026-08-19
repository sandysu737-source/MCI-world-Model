from __future__ import annotations

"""
MCI World Model v4.6.0 — 模态特征编码器
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


import logging

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# VisionEncoder — 视觉特征编码器
# =============================================================================


class LearnableMixin:
    """P0-F5 修复: 为编码器添加可学习参数的混入类。

    原始编码器 (VisionEncoder/AudioEncoder/ThermalEncoder) 纯统计特征，
    无可学习参数，导致不同图像/音频/热成像的嵌入几乎相同 (维度坍塌)。

    本 Mixin 为编码器添加轻量 MLP 投影头，将统计特征映射到更高维空间。
    MLP 参数随机初始化，保证不同输入产生不同输出。
    后续 P2 阶段将用 CLIP 蒸馏进一步优化。
    """

    def _init_learnable(self, stat_dim: int, output_dim: int, seed: int = 42) -> None:
        """初始化可学习投影头。

        Args:
            stat_dim: 统计特征维度 (如 VisionEncoder 的 32)
            output_dim: 目标输出维度 (如 128)
            seed: 随机种子
        """
        rng = np.random.RandomState(seed)
        self._stat_dim = stat_dim
        self._output_dim = output_dim

        # 两层 MLP 投影头: stat_dim → hidden → output_dim
        hidden_dim = max(64, (stat_dim + output_dim) // 2)
        self._proj_W1 = rng.randn(stat_dim, hidden_dim).astype(np.float64) * np.sqrt(2.0 / (stat_dim + hidden_dim))
        self._proj_b1 = np.zeros(hidden_dim, dtype=np.float64)
        self._proj_W2 = rng.randn(hidden_dim, output_dim).astype(np.float64) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self._proj_b2 = np.zeros(output_dim, dtype=np.float64)

    @property
    def n_params(self) -> int:
        """可学习参数数量。"""
        return self._proj_W1.size + self._proj_b1.size + self._proj_W2.size + self._proj_b2.size

    def _project(self, stat_features: np.ndarray) -> np.ndarray:
        """将统计特征投影到高维空间。

        Args:
            stat_features: (stat_dim,) 统计特征向量

        Returns:
            (output_dim,) 投影后的特征向量
        """
        x = stat_features.astype(np.float64)
        h = x @ self._proj_W1 + self._proj_b1
        h = np.maximum(h, 0)  # ReLU
        out = h @ self._proj_W2 + self._proj_b2
        return out.astype(np.float64)


class VisionEncoder(LearnableMixin):
    """RGB/Depth 帧 → 固定维度特征向量（纯 numpy）。

    P0-F5 修复: 添加可学习投影头，解决维度坍塌问题。

    特征组成 (feature_dim=32 统计 + 128 可学习 = 128 总输出):
        统计特征 (32D): 全局统计 + 四象限均值 + Sobel边缘 + 直方图 + 梯度方向
        可学习投影 (128D): 两层 MLP 将统计特征投影到高维空间

    Example:
        >>> enc = VisionEncoder(feature_dim=32, learnable_dim=128)
        >>> frame = np.random.rand(64, 64, 3)
        >>> vec = enc.encode(frame)
        >>> assert vec.shape == (128,)
    """

    def __init__(self, feature_dim: int = 32, learnable_dim: int = 128, seed: int = 42):
        self._feature_dim = feature_dim
        self._learnable_dim = learnable_dim
        self._output_dim = learnable_dim  # 默认输出可学习维度
        self._init_learnable(feature_dim, learnable_dim, seed=seed)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def output_dim(self) -> int:
        """输出向量维度 (可学习投影后)。"""
        return self._output_dim

    def encode(self, frame: np.ndarray) -> np.ndarray:
        """编码视觉帧为特征向量。

        P0-F5 修复: 返回可学习投影后的高维向量。

        Args:
            frame: 2D 或 3D numpy 数组
                - (H, W) — 灰度/深度/热图
                - (H, W, C) — RGB/多通道

        Returns:
            (output_dim,) float64 特征向量
        """
        stat_features = self._extract_stat_features(frame)
        return self._project(stat_features)

    def _extract_stat_features(self, frame: np.ndarray) -> np.ndarray:
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
                    gx[i, j] = (
                        -patch[0, 0] + patch[0, 2] - 2 * patch[1, 0] + 2 * patch[1, 2] - patch[2, 0] + patch[2, 2]
                    )
                    gy[i, j] = (
                        -patch[0, 0] - 2 * patch[0, 1] - patch[0, 2] + patch[2, 0] + 2 * patch[2, 1] + patch[2, 2]
                    )
            edge_mag = np.sqrt(gx**2 + gy**2)
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
                angles.flatten(),
                bins=angle_bins,
                range=(-np.pi, np.pi),
            )
            total = max(np.sum(hist_angles), 1)
            features[idx : idx + 8] = hist_angles.astype(np.float64) / total
            idx += 8

        # 填充剩余维度为零
        return features


# =============================================================================
# AudioEncoder — 音频特征编码器
# =============================================================================


class AudioEncoder(LearnableMixin):
    """音频波形 → 固定维度统计特征（纯 numpy）。

    P0-F5 修复: 添加可学习投影头，输出 64D 可学习特征。

    Example:
        >>> enc = AudioEncoder(feature_dim=16, learnable_dim=64)
        >>> wave = np.random.randn(16000)
        >>> vec = enc.encode(wave)
        >>> assert vec.shape == (64,)
    """

    def __init__(self, feature_dim: int = 16, learnable_dim: int = 64, seed: int = 42):
        self._feature_dim = feature_dim
        self._learnable_dim = learnable_dim
        self._output_dim = learnable_dim
        self._init_learnable(feature_dim, learnable_dim, seed=seed)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def output_dim(self) -> int:
        """输出向量维度 (可学习投影后)。"""
        return self._output_dim

    def encode(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """编码音频波形为特征向量。

        P0-F5 修复: 返回可学习投影后的高维向量。

        Args:
            waveform: 1D 音频波形 (N_samples,) 或 2D (N_samples, N_channels)
            sample_rate: 采样率 (Hz)

        Returns:
            (output_dim,) float64 特征向量
        """
        stat_features = self._extract_stat_features(waveform, sample_rate)
        return self._project(stat_features)

    def _extract_stat_features(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """提取统计特征 (原 encode 逻辑)。"""
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
            features[idx] = float(np.sqrt(np.mean(waveform**2)))  # RMS
            features[idx + 1] = float(np.max(np.abs(waveform)))  # 峰值
            # 零交叉率
            zc = np.sum(np.abs(np.diff(np.sign(waveform))) > 0)
            features[idx + 2] = float(zc) / max(n - 1, 1)
            features[idx + 3] = float(np.sum(waveform**2))  # 总能量
            idx += 4

        # [4:8] 分帧能量包络 (4 段)
        n_segments = 4
        if idx + n_segments <= self._feature_dim:
            seg_len = max(1, n // n_segments)
            for i in range(n_segments):
                start = i * seg_len
                end = min((i + 1) * seg_len, n)
                segment = waveform[start:end]
                features[idx + i] = float(np.mean(segment**2))
            idx += n_segments

        # [8:12] 频谱特征 (FFT-based)
        if idx + 4 <= self._feature_dim and n >= 8:
            fft_vals = np.abs(np.fft.rfft(waveform))
            freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
            total_energy = max(np.sum(fft_vals**2), 1e-10)

            # 频谱质心
            spectral_centroid = float(np.sum(freqs * fft_vals**2) / total_energy)
            features[idx] = spectral_centroid / max(sample_rate / 2, 1)

            # 频谱带宽 (标准差)
            spectral_bw = float(np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft_vals**2) / total_energy))
            features[idx + 1] = spectral_bw / max(sample_rate / 2, 1)

            # 频谱滚降 (85% 能量截止频率)
            cumsum = np.cumsum(fft_vals**2)
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


class ThermalEncoder(LearnableMixin):
    """热成像帧 → 固定维度温度分布特征（纯 numpy）。

    P0-F5 修复: 添加可学习投影头，输出 32D 可学习特征。

    Example:
        >>> enc = ThermalEncoder(feature_dim=8, learnable_dim=32)
        >>> thermal = np.random.rand(32, 32) * 40 + 20
        >>> vec = enc.encode(thermal)
        >>> assert vec.shape == (32,)
    """

    def __init__(self, feature_dim: int = 8, learnable_dim: int = 32, seed: int = 42):
        self._feature_dim = feature_dim
        self._learnable_dim = learnable_dim
        self._output_dim = learnable_dim
        self._init_learnable(feature_dim, learnable_dim, seed=seed)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def output_dim(self) -> int:
        """输出向量维度 (可学习投影后)。"""
        return self._output_dim

    def encode(self, thermal_frame: np.ndarray) -> np.ndarray:
        """编码热成像帧为特征向量。

        P0-F5 修复: 返回可学习投影后的高维向量。

        Args:
            thermal_frame: 2D numpy 数组 (H, W) 温度值

        Returns:
            (output_dim,) float64 特征向量
        """
        stat_features = self._extract_stat_features(thermal_frame)
        return self._project(stat_features)

    def _extract_stat_features(self, thermal_frame: np.ndarray) -> np.ndarray:
        """提取统计特征 (原 encode 逻辑)。"""
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
            grad_mag = np.sqrt(grad_y[:, : w - 1] ** 2 + grad_x[: h - 1, :] ** 2)
            features[idx] = float(np.mean(grad_mag))
            features[idx + 1] = float(np.max(grad_mag))
            idx += 2

        return features


# =============================================================================
# DepthEncoder — 3D 创面深度编码器 (v4.4.0 新增)
# =============================================================================


class DepthEncoder(LearnableMixin):
    """深度图编码器 — 3D 创面重建特征提取。

    输入: (H, W) 深度图 (单位: mm)
    统计特征: 均值、方差、梯度幅值、曲率、边缘锐度 (8维)
    可学习投影: 8D → 32D

    用途: 清创机器人感知创面三维形态，
    判断坏死组织深度和清创边界。
    """

    def __init__(self, feature_dim: int = 8, learnable_dim: int = 32, seed: int = 42):
        self._stat_dim = feature_dim
        super()._init_learnable(feature_dim, learnable_dim, seed)

    @property
    def feature_dim(self) -> int:
        return self._stat_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def encode(self, depth_map: np.ndarray) -> np.ndarray:
        """编码深度图为特征向量。

        Args:
            depth_map: (H, W) float32 深度值 (mm)

        Returns:
            (output_dim,) 归一化深度特征
        """
        stat = self._extract_stat_features(depth_map)
        return self._project(stat)

    def _extract_stat_features(self, depth_map: np.ndarray) -> np.ndarray:
        """提取深度统计特征 (8维)。

        特征:
        [0] 平均深度 (mm)
        [1] 深度标准差
        [2] 最大深度
        [3] 深度梯度幅值均值
        [4] 曲率 (二阶导数)
        [5] 创面面积 (深度 > 背景阈值的像素比例)
        [6] 边缘锐度 (梯度 > 阈值的比例)
        [7] 深度分布的偏度
        """
        dm = depth_map.astype(np.float64)
        features = np.zeros(self._stat_dim, dtype=np.float64)

        # 基础统计
        features[0] = float(np.mean(dm))
        features[1] = float(np.std(dm))
        features[2] = float(np.max(dm))

        # 梯度
        gy, gx = np.gradient(dm)
        grad_mag = np.sqrt(gx**2 + gy**2)
        features[3] = float(np.mean(grad_mag))

        # 曲率 (拉普拉斯)
        if dm.size > 4:
            laplacian = np.gradient(gx, axis=1) + np.gradient(gy, axis=0)
            features[4] = float(np.mean(np.abs(laplacian)))
        else:
            features[4] = 0.0

        # 创面面积: 深度显著偏离背景 (> mean + 1 std)
        bg = np.mean(dm) + np.std(dm)
        features[5] = float(np.mean(dm > bg))

        # 边缘锐度
        features[6] = float(np.mean(grad_mag > np.mean(grad_mag) * 2))

        # 偏度
        if features[1] > 1e-10:
            features[7] = float(np.mean((dm - features[0]) ** 3) / (features[1] ** 3))
        else:
            features[7] = 0.0

        return features


# =============================================================================
# ForceEncoder — 力触觉编码器 (v4.4.0 新增)
# =============================================================================


class ForceEncoder:
    """力触觉编码器 — 工具-组织交互力特征提取。

    输入: (6,) 或 (T, 6) 力/力矩信号
    特征: 均值、方差、峰值、变化率、频谱能量 (16维)
    输出: (32,) 归一化力特征

    用途: 清创机器人感知工具与组织的力交互，
    区分不同组织类型的力响应特征。
    """

    def __init__(self, feature_dim: int = 16, output_dim: int = 32, seed: int = 42):
        self._stat_dim = feature_dim
        self._output_dim = output_dim

        # 轻量投影: 统计特征 → 输出 (无训练参数, 确定性)
        rng = np.random.RandomState(seed)
        self._proj_W = rng.randn(feature_dim, output_dim).astype(np.float64) * np.sqrt(2.0 / (feature_dim + output_dim))
        self._proj_b = np.zeros(output_dim, dtype=np.float64)

    @property
    def feature_dim(self) -> int:
        return self._stat_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def encode(self, ft_signal: np.ndarray) -> np.ndarray:
        """编码力/力矩信号为特征向量。

        Args:
            ft_signal: (6,) 单帧或 (T, 6) 时序力/力矩

        Returns:
            (output_dim,) 归一化力特征向量
        """
        ft = ft_signal.astype(np.float64)
        if ft.ndim == 1:
            ft = ft.reshape(1, -1)

        stat = self._extract_stat_features(ft)
        out = stat @ self._proj_W + self._proj_b
        return out.astype(np.float64)

    def encode_history(self, ft_window: np.ndarray) -> np.ndarray:
        """编码力时序窗口。

        Args:
            ft_window: (T, 6) 力/力矩历史窗口

        Returns:
            (output_dim,) 时序力特征
        """
        return self.encode(ft_window)

    def _extract_stat_features(self, ft: np.ndarray) -> np.ndarray:
        """提取力统计特征 (16维)。

        特征:
        [0-5]   各轴均值
        [6-11]  各轴标准差
        [12]    合力幅值均值
        [13]    合力变化率
        [14]    力峰值 (max)
        [15]    高频能量占比
        """
        features = np.zeros(self._stat_dim, dtype=np.float64)

        for axis in range(min(6, ft.shape[1])):
            values = ft[:, axis]
            features[axis] = float(np.mean(values))
            features[6 + axis] = float(np.std(values))

        # 合力
        resultant = np.sqrt(np.sum(ft[:, :3] ** 2, axis=1))
        features[12] = float(np.mean(resultant))

        # 变化率
        if ft.shape[0] > 1:
            features[13] = float(np.mean(np.abs(np.diff(resultant))))
        else:
            features[13] = 0.0

        # 峰值
        features[14] = float(np.max(resultant))

        # 高频能量 (用相邻差作为高频代理)
        if ft.shape[0] > 2:
            highfreq = np.diff(ft[:, 0])
            total_var = np.var(ft[:, 0])
            if total_var > 1e-10:
                features[15] = float(np.var(highfreq) / total_var)
            else:
                features[15] = 0.0
        else:
            features[15] = 0.0

        return features
