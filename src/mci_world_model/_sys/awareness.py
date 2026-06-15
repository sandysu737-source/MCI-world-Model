"""
元认知系统 (向后兼容层)

v3.4.0: 核心实现已迁移至 meta_cognition.py
本文件保留为兼容重导出层，供 sdk/_world_model.py 等历史引用。

对外暴露：MetaCognition, CognitiveGap, KnowledgeAging
"""

# 从统一模块重导出
from mci_world_model._sys.meta_cognition import (
    CognitiveGap,
    CognitiveScoreCard,
    KnowledgeAging,
    MetaCognition,
    RootCauseNode,
)

__all__ = [
    "CognitiveGap",
    "CognitiveScoreCard",
    "KnowledgeAging",
    "MetaCognition",
    "RootCauseNode",
]
