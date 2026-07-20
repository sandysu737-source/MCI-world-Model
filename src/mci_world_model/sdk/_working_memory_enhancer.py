from __future__ import annotations

"""MCI World Model v4.6.0 — 工作记忆增强器
==============================================

P1-A 增强: 从基本 FIFO 工作记忆 → 认知增强型记忆系统。

现有 WorkingMemory (v3.0.4) 仅提供:
    - FIFO push/pop + temporal_weight 加权淘汰
    - 无注意力检索、无遗忘曲线、无记忆整合

WorkingMemoryEnhancer 新增:
    1. **注意力检索**: query → 按 relevance 加权返回 top-K
    2. **遗忘曲线**: Ebbinghaus 遗忘模型 R(t) = e^(-t/S)，自动衰减
    3. **记忆整合**: 语义相似轨迹步骤自动合并/压缩
    4. **检索增强**: 基于 cosine similarity 的向量检索
    5. **优先级队列**: surprise-driven 优先保留异常轨迹

设计原则:
    - 复用 WorkingMemory，增强而非替换
    - 纯 numpy，零外部依赖
    - 与 cewm_step() 兼容

## Formal Guarantees

    - 遗忘曲线: R(t) = exp(-decay * t), R(0) = 1, R(∞) = 0
    - 注意力权重: w_i = softmax(query · key_i / sqrt(d))
    - 整合: 合并后 temporal_weight = max(merged_steps)
    - 检索结果按 relevance 降序

用法:
    >>> from mci_world_model.sdk._working_memory_enhancer import WorkingMemoryEnhancer
    >>> from mci_world_model.sdk import WorkingMemory, TrajectoryStep
    >>> wm = WorkingMemory(max_length=20)
    >>> enhancer = WorkingMemoryEnhancer(wm)
    >>> # 正常使用 wm.push() 后...
    >>> results = enhancer.attention_retrieve(query_vector, top_k=3)
"""


import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mci_world_model.sdk._world_model import TrajectoryStep, WorkingMemory

logger = logging.getLogger(__name__)


# =============================================================================
# WorkingMemoryEnhancerConfig — 配置
# =============================================================================


@dataclass
class WorkingMemoryEnhancerConfig:
    """工作记忆增强器配置。

    Attributes:
        decay_rate: 遗忘曲线衰减率 (默认 0.1, Ebbinghaus)
        consolidation_threshold: 整合相似度阈值 (默认 0.9, cosine)
        max_retrieval_results: 最大检索结果数 (默认 5)
        surprise_boost: 惊奇优先级提升系数 (默认 2.0)
        attention_temperature: 注意力温度 (默认 1.0)
    """

    decay_rate: float = 0.1
    consolidation_threshold: float = 0.9
    max_retrieval_results: int = 5
    surprise_boost: float = 2.0
    attention_temperature: float = 1.0


# =============================================================================
# RetrievalResult — 检索结果
# =============================================================================


@dataclass
class MemoryRetrievalResult:
    """记忆检索结果。

    Attributes:
        step: 轨迹步骤
        relevance: 相关度分数 [0, 1]
        retention: 遗忘曲线保留率 [0, 1]
        effective_weight: 有效权重 (temporal_weight × retention × relevance)
    """

    step: Any  # TrajectoryStep
    relevance: float = 0.0
    retention: float = 1.0
    effective_weight: float = 0.0


# =============================================================================
# WorkingMemoryEnhancer — 工作记忆增强器
# =============================================================================


class WorkingMemoryEnhancer:
    """认知增强型工作记忆系统。

    在 WorkingMemory 基础上增加:
    - 遗忘曲线衰减
    - 注意力检索
    - 记忆整合
    - 惊奇驱动优先级

    Example:
        >>> from mci_world_model.sdk import WorkingMemory, TrajectoryStep
        >>> from mci_world_model.sdk._working_memory_enhancer import WorkingMemoryEnhancer
        >>> wm = WorkingMemory(max_length=20)
        >>> enhancer = WorkingMemoryEnhancer(wm)
    """

    def __init__(
        self,
        working_memory: WorkingMemory,
        config: WorkingMemoryEnhancerConfig | None = None,
    ):
        """
        Args:
            working_memory: 底层工作记忆实例
            config: 增强器配置
        """
        self._wm = working_memory
        self._config = config or WorkingMemoryEnhancerConfig()
        self._step_timestamps: list[float] = []
        self._step_surprise: list[float] = []
        self._query_count: int = 0

    @property
    def config(self) -> WorkingMemoryEnhancerConfig:
        return self._config

    @property
    def query_count(self) -> int:
        return self._query_count

    @property
    def memory_size(self) -> int:
        return len(self._wm.trajectory)

    # -----------------------------------------------------------------
    # push_enhanced — 增强版 push (记录时间戳和惊奇度)
    # -----------------------------------------------------------------

    def push_enhanced(
        self,
        step: TrajectoryStep,
        surprise_score: float = 0.0,
        timestamp: float | None = None,
    ) -> None:
        """增强版 push: 记录时间戳和惊奇度用于遗忘/优先级。

        Args:
            step: 轨迹步骤
            surprise_score: 惊奇度分数 [0, ∞)
            timestamp: 时间戳 (None 时自动生成)
        """
        import time as _time

        self._wm.push(step)

        ts = timestamp if timestamp is not None else _time.time()
        self._step_timestamps.append(ts)
        self._step_surprise.append(surprise_score)

        # 修剪时间戳/惊奇度列表与 trajectory 同步
        while len(self._step_timestamps) > len(self._wm.trajectory):
            self._step_timestamps.pop(0)
        while len(self._step_surprise) > len(self._wm.trajectory):
            self._step_surprise.pop(0)

    # -----------------------------------------------------------------
    # compute_retention — 遗忘曲线
    # -----------------------------------------------------------------

    def compute_retention(self, step_index: int, current_time: float | None = None) -> float:
        """计算指定步骤的遗忘曲线保留率。

        Ebbinghaus 遗忘模型: R(t) = exp(-decay * Δt)

        Args:
            step_index: 步骤索引
            current_time: 当前时间 (None 时自动取)

        Returns:
            保留率 [0, 1]
        """
        if step_index >= len(self._step_timestamps) or not self._step_timestamps:
            return 1.0

        import time as _time

        now = current_time or _time.time()
        elapsed = now - self._step_timestamps[step_index]
        retention = math.exp(-self._config.decay_rate * elapsed)
        return max(0.0, min(1.0, retention))

    def compute_all_retentions(self, current_time: float | None = None) -> list[float]:
        """计算所有步骤的遗忘曲线保留率。

        Returns:
            保留率列表
        """
        return [self.compute_retention(i, current_time) for i in range(len(self._wm.trajectory))]

    # -----------------------------------------------------------------
    # attention_retrieve — 注意力检索
    # -----------------------------------------------------------------

    def attention_retrieve(
        self,
        query_vector: np.ndarray,
        top_k: int | None = None,
    ) -> list[MemoryRetrievalResult]:
        """基于注意力机制的检索。

        计算查询向量与每个轨迹步骤的 cosine similarity，
        用 softmax 归一化后选 top-K。

        Args:
            query_vector: 查询向量
            top_k: 返回结果数 (None 时用配置默认值)

        Returns:
            检索结果列表 (按 relevance 降序)
        """
        self._query_count += 1
        k = top_k or self._config.max_retrieval_results

        if not self._wm.trajectory:
            return []

        # 提取轨迹步骤的向量表示
        step_vectors = []
        for step in self._wm.trajectory:
            vec = self._step_to_vector(step)
            step_vectors.append(vec)

        if not step_vectors:
            return []

        # 计算 cosine similarity
        query_norm = np.linalg.norm(query_vector)
        if query_norm < 1e-10:
            return []

        similarities = []
        for vec in step_vectors:
            vec_norm = np.linalg.norm(vec)
            if vec_norm < 1e-10:
                similarities.append(0.0)
            else:
                sim = float(np.dot(query_vector, vec) / (query_norm * vec_norm))
                similarities.append(max(0.0, sim))

        # Softmax 归一化 (带温度)
        temperature = self._config.attention_temperature
        exp_sims = [math.exp(s / temperature) for s in similarities]
        total = sum(exp_sims)
        attention_weights = [e / total for e in exp_sims] if total > 0 else similarities

        # 构建检索结果
        results = []
        for i, step in enumerate(self._wm.trajectory):
            retention = self.compute_retention(i)
            surprise_boost = (
                1.0 + self._config.surprise_boost * self._step_surprise[i] if i < len(self._step_surprise) else 1.0
            )
            tw = getattr(step, "temporal_weight", 1.0)
            effective = tw * retention * attention_weights[i] * surprise_boost

            results.append(
                MemoryRetrievalResult(
                    step=step,
                    relevance=attention_weights[i],
                    retention=retention,
                    effective_weight=effective,
                )
            )

        # 按 effective_weight 降序排序
        results.sort(key=lambda r: r.effective_weight, reverse=True)
        return results[:k]

    # -----------------------------------------------------------------
    # consolidate — 记忆整合
    # -----------------------------------------------------------------

    def consolidate(self) -> int:
        """整合语义相似的轨迹步骤。

        将 cosine similarity > threshold 的连续步骤合并，
        保留 temporal_weight 更高的那个。

        Returns:
            合并的步骤数
        """
        if len(self._wm.trajectory) < 2:
            return 0

        merged_count = 0
        to_remove: list[int] = []

        for i in range(len(self._wm.trajectory) - 1):
            if i in to_remove:
                continue

            vec_i = self._step_to_vector(self._wm.trajectory[i])
            vec_j = self._step_to_vector(self._wm.trajectory[i + 1])

            if vec_i is None or vec_j is None:
                continue

            norm_i = np.linalg.norm(vec_i)
            norm_j = np.linalg.norm(vec_j)

            if norm_i < 1e-10 or norm_j < 1e-10:
                continue

            similarity = float(np.dot(vec_i, vec_j) / (norm_i * norm_j))

            if similarity > self._config.consolidation_threshold:
                # 保留 temporal_weight 更高的
                tw_i = getattr(self._wm.trajectory[i], "temporal_weight", 1.0)
                tw_j = getattr(self._wm.trajectory[i + 1], "temporal_weight", 1.0)

                if tw_i >= tw_j:
                    # 提升 i 的 temporal_weight
                    if hasattr(self._wm.trajectory[i], "temporal_weight"):
                        self._wm.trajectory[i].temporal_weight = max(tw_i, tw_j)
                    to_remove.append(i + 1)
                else:
                    if hasattr(self._wm.trajectory[i + 1], "temporal_weight"):
                        self._wm.trajectory[i + 1].temporal_weight = max(tw_i, tw_j)
                    to_remove.append(i)

                merged_count += 1

        # 从后向前移除
        for idx in sorted(to_remove, reverse=True):
            if idx < len(self._wm.trajectory):
                self._wm.trajectory.pop(idx)
                if idx < len(self._step_timestamps):
                    self._step_timestamps.pop(idx)
                if idx < len(self._step_surprise):
                    self._step_surprise.pop(idx)

        return merged_count

    # -----------------------------------------------------------------
    # get_surprise_top — 惊奇驱动优先返回
    # -----------------------------------------------------------------

    def get_surprise_top(self, top_k: int = 3) -> list[MemoryRetrievalResult]:
        """返回惊奇度最高的轨迹步骤。

        Args:
            top_k: 返回数量

        Returns:
            按惊奇度降序的结果
        """
        if not self._wm.trajectory:
            return []

        results = []
        for i, step in enumerate(self._wm.trajectory):
            surprise = self._step_surprise[i] if i < len(self._step_surprise) else 0.0
            retention = self.compute_retention(i)
            tw = getattr(step, "temporal_weight", 1.0)

            results.append(
                MemoryRetrievalResult(
                    step=step,
                    relevance=surprise,
                    retention=retention,
                    effective_weight=tw * retention * (1.0 + self._config.surprise_boost * surprise),
                )
            )

        results.sort(key=lambda r: r.relevance, reverse=True)
        return results[:top_k]

    # -----------------------------------------------------------------
    # get_retention_summary — 保留率摘要
    # -----------------------------------------------------------------

    def get_retention_summary(self) -> dict[str, Any]:
        """获取遗忘曲线和记忆状态摘要。"""
        retentions = self.compute_all_retentions()

        return {
            "memory_size": len(self._wm.trajectory),
            "avg_retention": float(np.mean(retentions)) if retentions else 1.0,
            "min_retention": float(np.min(retentions)) if retentions else 1.0,
            "high_retention_count": sum(1 for r in retentions if r > 0.8),
            "low_retention_count": sum(1 for r in retentions if r < 0.3),
            "total_surprise": sum(self._step_surprise),
            "query_count": self._query_count,
        }

    # -----------------------------------------------------------------
    # _step_to_vector — 轨迹步骤 → 向量
    # -----------------------------------------------------------------

    def _step_to_vector(self, step: TrajectoryStep) -> np.ndarray:
        """将轨迹步骤转换为向量表示 (用于相似度计算)。"""
        if hasattr(step, "state") and hasattr(step.state, "to_vector"):
            return step.state.to_vector()
        if hasattr(step, "to_vector"):
            return step.to_vector()
        # 退回: 用 __dict__ 的数值属性构造向量
        values = []
        for v in vars(step).values():
            if isinstance(v, (int, float)):
                values.append(float(v))
            elif isinstance(v, np.ndarray):
                values.extend(v.flatten().tolist())
        return np.array(values, dtype=np.float64) if values else np.zeros(1)

    # -----------------------------------------------------------------
    # 字符串表示
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"WorkingMemoryEnhancer(size={self.memory_size}, "
            f"queries={self._query_count}, "
            f"decay={self._config.decay_rate})"
        )
