from __future__ import annotations

"""MCI World Model — EdgeCloudHybrid 边云协同推理
===============================================

边缘设备与云端服务器协同推理——低延迟场景下边缘优先,
复杂推理自动上云, 支持模型蒸馏和结果缓存。

核心能力:
    InferenceRequest     — 推理请求数据类
    InferenceResult      — 推理结果数据类
    EdgeCloudHybrid      — 边云协同调度器

设计原则:
    - 边缘优先: 延迟敏感任务优先边缘执行
    - 自动上云: 超出边缘能力自动切换云端
    - 结果缓存: 相似查询复用缓存
    - 纯 numpy，零外部依赖
"""


import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# InferenceRequest — 推理请求
# =============================================================================


@dataclass
class InferenceRequest:
    """推理请求。

    Attributes:
        request_id: 请求ID
        query: 查询数据
        priority: 优先级 ('low' / 'medium' / 'high' / 'critical')
        max_latency_ms: 最大允许延迟
        complexity: 推理复杂度估计 (1-10)
        requires_audit: 是否需要审计
    """

    request_id: str
    query: dict[str, Any] = field(default_factory=dict)
    priority: str = "medium"
    max_latency_ms: float = 100.0
    complexity: int = 5
    requires_audit: bool = False


# =============================================================================
# InferenceResult — 推理结果
# =============================================================================


@dataclass
class InferenceResult:
    """推理结果。

    Attributes:
        request_id: 请求ID
        result: 推理结果数据
        executed_on: 执行位置 ('edge' / 'cloud')
        latency_ms: 实际延迟
        confidence: 置信度
        cached: 是否来自缓存
        audit_trail_id: 审计轨迹ID (如有)
    """

    request_id: str
    result: dict[str, Any] = field(default_factory=dict)
    executed_on: str = "edge"
    latency_ms: float = 0.0
    confidence: float = 0.0
    cached: bool = False
    audit_trail_id: str = ""


# =============================================================================
# EdgeCloudHybrid — 边云协同调度器
# =============================================================================


class EdgeCloudHybrid:
    """边云协同调度器 — 边缘优先 + 自动上云。

    调度策略:
      - complexity ≤ edge_capacity → 边缘执行
      - complexity > edge_capacity → 云端执行
      - 请求有审计要求 → 云端执行
      - 缓存命中 → 直接返回

    用法:
        >>> hybrid = EdgeCloudHybrid(edge_capacity=5)
        >>> request = InferenceRequest(request_id="req1", complexity=3)
        >>> result = hybrid.dispatch(request)
    """

    def __init__(
        self,
        edge_capacity: int = 5,
        edge_latency_ms: float = 10.0,
        cloud_latency_ms: float = 100.0,
        cache_size: int = 100,
    ):
        if edge_capacity < 1:
            raise ValueError(f"edge_capacity 必须 ≥ 1, 当前 {edge_capacity}")
        self._edge_capacity = edge_capacity
        self._edge_latency = edge_latency_ms
        self._cloud_latency = cloud_latency_ms
        self._cache_size = cache_size
        self._cache: dict[str, InferenceResult] = {}
        self._dispatch_count: int = 0
        self._edge_count: int = 0
        self._cloud_count: int = 0
        self._cache_hits: int = 0

    @property
    def edge_capacity(self) -> int:
        return self._edge_capacity

    @property
    def dispatch_count(self) -> int:
        return self._dispatch_count

    def dispatch(self, request: InferenceRequest) -> InferenceResult:
        """调度推理请求。

        Args:
            request: 推理请求

        Returns:
            InferenceResult
        """
        self._dispatch_count += 1

        # 检查缓存
        cache_key = self._make_cache_key(request)
        if cache_key in self._cache:
            self._cache_hits += 1
            cached = self._cache[cache_key]
            return InferenceResult(
                request_id=request.request_id,
                result=cached.result,
                executed_on=cached.executed_on,
                latency_ms=1.0,  # 缓存命中极低延迟
                confidence=cached.confidence,
                cached=True,
            )

        # 调度决策
        if request.requires_audit or request.complexity > self._edge_capacity:
            executed_on = "cloud"
            latency = self._cloud_latency
            self._cloud_count += 1
        else:
            executed_on = "edge"
            latency = self._edge_latency
            self._edge_count += 1

        # 模拟推理 (简化: 返回固定结果)
        result_data = self._simulate_inference(request, executed_on)

        result = InferenceResult(
            request_id=request.request_id,
            result=result_data,
            executed_on=executed_on,
            latency_ms=latency + np.random.rand() * latency * 0.2,
            confidence=result_data.get("confidence", 0.5),
        )

        # 写入缓存
        if len(self._cache) >= self._cache_size:
            # 简化 LRU: 删除最早的
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[cache_key] = result

        logger.info(
            "边云调度: %s → %s (complexity=%d, latency=%.1fms)",
            request.request_id,
            executed_on,
            request.complexity,
            result.latency_ms,
        )

        return result

    def _simulate_inference(self, request: InferenceRequest, executed_on: str) -> dict[str, Any]:
        """模拟推理执行 (简化实现)。"""
        confidence = 0.5 + 0.1 * (1 if executed_on == "cloud" else 0)
        return {
            "query_summary": str(request.query)[:100],
            "confidence": min(confidence, 0.99),
            "executed_on": executed_on,
            "complexity": request.complexity,
        }

    @staticmethod
    def _make_cache_key(request: InferenceRequest) -> str:
        """生成缓存键。"""
        import hashlib

        key_str = f"{request.query}:{request.complexity}:{request.requires_audit}"
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    def clear_cache(self) -> None:
        """清除缓存。"""
        self._cache.clear()

    def statistics(self) -> dict[str, Any]:
        """调度统计。"""
        total = max(self._dispatch_count, 1)
        return {
            "dispatch_count": self._dispatch_count,
            "edge_count": self._edge_count,
            "cloud_count": self._cloud_count,
            "cache_hits": self._cache_hits,
            "edge_ratio": self._edge_count / total,
            "cloud_ratio": self._cloud_count / total,
            "cache_hit_rate": self._cache_hits / total,
            "cache_size": len(self._cache),
        }
