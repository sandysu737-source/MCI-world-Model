"""
MCI World Model v3.0.3 — Perception Pipeline
==============================================

LeCun 六模块架构中的 Perception 模块：将原始观测（文本/记忆）转换为
World Model 可消费的结构化特征。

职责：
- 接收原始文本或记忆列表 → 输出结构化感知特征
- 封装 EvidenceCollector + EncoderCore + TemporalSystem 三源预处理
- 提供统一的 "sensor-like" 接口

处理流程:
    raw_input → evidence_collection → temporal_annotation →
    semantic_encoding → feature_extraction → structured_output

状态机: IDLE → COLLECTING → ENCODING → STRUCTURED → COMPLETE

用法:
    from mci_world_model._sys._perception_pipeline import PerceptionPipeline

    perception = PerceptionPipeline()
    features = perception.process(memories)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# PerceivedFeatures — 感知输出
# =============================================================================


@dataclass
class PerceivedFeatures:
    """
    感知管道输出：结构化的多维度特征。

    Attributes:
        entities: 提取的实体列表 [{"name": str, "type": str, "confidence": float}, ...]
        temporal_context: 时序上下文 {"stem": str, "branch": str, "cycle": int, ...}
        energy_profile: 能量分布 {"semantic": float, "causal": float, ...} (五行强度)
        semantic_embeddings: 语义向量 [N, D] 或 None
        evidence_count: 收集的证据条数
        timestamp: 处理时间戳
    """

    entities: list[dict] = field(default_factory=list)
    temporal_context: dict = field(default_factory=dict)
    energy_profile: dict[str, float] = field(default_factory=dict)
    semantic_embeddings: list = field(default_factory=list)
    evidence_count: int = 0
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def empty(cls) -> PerceivedFeatures:
        return cls()

    def to_dict(self) -> dict:
        return {
            "n_entities": len(self.entities),
            "has_temporal": bool(self.temporal_context),
            "energy_profile": self.energy_profile,
            "evidence_count": self.evidence_count,
        }


# =============================================================================
# PerceptionPipeline — 统一感知前端
# =============================================================================


class PerceptionPipeline:
    """
    统一感知前端管道。

    六态流转：IDLE → COLLECTING → ENCODING → STRUCTURED → COMPLETE

    三源融合:
    1. EvidenceCollector: 多源证据收集 + 可靠性评估
    2. SemanticEncoder: 文本→语义向量编码（64卦全息编码）
    3. TemporalSystem: 时间标注（天干地支/六十花甲）
    """

    def __init__(self):
        self._state: str = "IDLE"
        self._process_count: int = 0
        self._last_features: PerceivedFeatures | None = None

    # -----------------------------------------------------------------
    # 属性
    # -----------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def process_count(self) -> int:
        return self._process_count

    @property
    def last_features(self) -> dict | None:
        if self._last_features is None:
            return None
        return self._last_features.to_dict()

    # -----------------------------------------------------------------
    # 核心处理
    # -----------------------------------------------------------------

    def process(
        self,
        memories: list[dict],
        collect_evidence: bool = True,
        encode_semantic: bool = False,
        annotate_temporal: bool = True,
    ) -> PerceivedFeatures:
        """
        处理原始观测 → 结构化感知特征。

        Args:
            memories: 记忆列表 [{"content": str, "timestamp": str, ...}, ...]
            collect_evidence: 是否启用证据收集
            encode_semantic: 是否启用语义编码（较重）
            annotate_temporal: 是否启用时间标注

        Returns:
            PerceivedFeatures 结构化特征
        """
        self._state = "COLLECTING"

        if not memories:
            self._state = "COMPLETE"
            return PerceivedFeatures.empty()

        features = PerceivedFeatures()

        # ── 1. 证据收集 ──
        if collect_evidence:
            features.evidence_count = self._collect_evidence(memories)

        # ── 2. 时间标注 ──
        if annotate_temporal:
            features.temporal_context = self._annotate_temporal(memories)

        # ── 3. 实体提取 ──
        self._state = "ENCODING"
        features.entities = self._extract_entities(memories)

        # ── 4. 能量剖面 ──
        features.energy_profile = self._compute_energy_profile(memories)

        # ── 5. 语义编码（可选）──
        if encode_semantic:
            features.semantic_embeddings = self._semantic_encode(memories)

        self._state = "STRUCTURED"
        self._last_features = features
        self._process_count += 1
        self._state = "COMPLETE"

        logger.debug(
            "PerceptionPipeline 处理完成: %d entities, %d evidence",
            len(features.entities),
            features.evidence_count,
        )
        return features

    # -----------------------------------------------------------------
    # 子管道（私有）
    # -----------------------------------------------------------------

    def _collect_evidence(self, memories: list[dict]) -> int:
        """证据收集：通过 EvidenceCollector 注册来源并收集证据。"""
        try:
            from mci_world_model._sys.bayesian import BayesianEngine
            from mci_world_model._sys.evidence import EvidenceCollector

            engine = BayesianEngine()
            collector = EvidenceCollector(engine)
            count = 0

            for mem in memories:
                content = mem.get("content", "")
                mem_id = mem.get("id", f"mem_{hash(content) % 100000}")
                source = mem.get("source", "memory")

                if content:
                    collector.register_source(mem_id, source_type=source)
                    collector.add_observation(
                        source_id=mem_id,
                        observation_type="memory_content",
                        value=True,
                        confidence=mem.get("confidence", 0.7),
                        metadata={"content": content[:200]},
                    )
                    count += 1

            return count
        except Exception as e:
            logger.debug("Perception evidence collect 跳过: %s", e)
            return 0

    def _annotate_temporal(self, memories: list[dict]) -> dict:
        """时间标注：提取最新的时序上下文。"""
        try:
            from mci_world_model._sys.chrono import TemporalSystem

            ts = TemporalSystem()
            latest_context = {}

            for mem in memories:
                timestamp = mem.get("timestamp", "")
                content = mem.get("content", "")
                if timestamp or content:
                    try:
                        temporal_info = ts.encode(timestamp or content)
                        if temporal_info:
                            latest_context = (
                                temporal_info
                                if isinstance(temporal_info, dict)
                                else {"info": str(temporal_info)}
                            )
                    except Exception:
                        pass

            return latest_context
        except Exception as e:
            logger.debug("Perception temporal 跳过: %s", e)
            return {}

    def _extract_entities(self, memories: list[dict]) -> list[dict]:
        """实体提取：从记忆内容中提取关键词/实体。"""
        entities: list[dict] = []

        for mem in memories:
            content = mem.get("content", "")
            if not content:
                continue

            # 简单关键词分割（中文按语义分割符）
            import re

            words = re.findall(r"[\u4e00-\u9fff]{2,}", str(content))
            seen: set[str] = set()
            for w in words[:10]:  # 每段记忆最多10个实体
                if w not in seen:
                    seen.add(w)
                    entities.append(
                        {
                            "name": w,
                            "type": "keyword",
                            "confidence": 0.5,  # 简单提取，置信度不高
                            "source": mem.get("id", ""),
                        }
                    )

        return entities

    def _compute_energy_profile(self, memories: list[dict]) -> dict[str, float]:
        """
        能量剖面计算：基于记忆内容的关键词统计五行能量分布。

        使用 _sys/_energy_core 的能量映射表，对记忆内容做关键词→能量类型映射。
        """
        try:
            from mci_world_model._sys._c1 import KEYWORDS_TO_CATEGORY

            profile: dict[str, float] = {
                "semantic": 0.0,
                "causal": 0.0,
                "spacetime": 0.0,
                "generative": 0.0,
                "trust": 0.0,
            }

            total: float = 0.0
            for mem in memories:
                content = str(mem.get("content", ""))
                for keyword, category in KEYWORDS_TO_CATEGORY.items():
                    if keyword in content:
                        energy_type = category_to_energy(category)
                        if energy_type in profile:
                            profile[energy_type] += 1.0
                            total += 1.0

            # 归一化
            if total > 0:
                for k, v in profile.items():
                    profile[k] = round(v / total, 4)

            return profile
        except Exception as e:
            logger.debug("Perception energy profile 跳过: %s", e)
            return {
                "semantic": 0.2,
                "causal": 0.2,
                "spacetime": 0.2,
                "generative": 0.2,
                "trust": 0.2,
            }

    def _semantic_encode(self, memories: list[dict]) -> list:
        """
        语义编码：使用全息编码器对内容做向量化。

        降级策略：encoder 不可用时返回空列表。
        """
        try:
            from mci_world_model._sys.encoders import SemanticEncoder

            encoder = SemanticEncoder()
            vectors = []
            for mem in memories:
                content = str(mem.get("content", ""))[:500]
                if content:
                    vec = encoder.encode(content)
                    vectors.append(vec)
            return vectors
        except Exception as e:
            logger.debug("Perception semantic encode 跳过: %s", e)
            return []


def category_to_energy(category: str) -> str:
    """语义分类 → 能量类型映射。"""
    _map = {
        "fact": "semantic",
        "knowledge": "semantic",
        "causal": "causal",
        "reasoning": "causal",
        "temporal": "spacetime",
        "spatial": "spacetime",
        "creative": "generative",
        "generative": "generative",
        "social": "trust",
        "trust": "trust",
        "emotion": "trust",
    }
    return _map.get(category, "semantic")
