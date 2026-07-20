"""MCI World Model — 三元路由共享类型

定义三元融合路由的路径类型与安全等级枚举，供所有路由相关模块共享，
避免重复定义导致的接口不一致（不同模块的 RouteType 互不匹配）。
"""

from __future__ import annotations

from enum import Enum


class RouteType(str, Enum):
    """三元推理路径类型。"""

    PHYSICAL = "physical"  # 生理动力学 / JEPA 潜空间路径
    CAUSAL = "causal"  # do-calculus 因果路径
    SEMANTIC = "semantic"  # 语义 / 知识嵌入路径
    FUSED = "fused"  # 多路径融合


class SafetyLevel(str, Enum):
    """临床安全输出三级降级。"""

    TRUSTED = "trusted"  # uncertainty ≤ 低阈值，可直接输出
    NEEDS_REVIEW = "needs_review"  # 中间区间，输出 + 建议医师确认
    REFUSED = "refused"  # 高不确定性 / 方向矛盾，拒绝输出确定结论
