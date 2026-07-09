"""
MCI World Model v3.0.3 — Perception Pipeline
==============================================

LeCun 六模块架构中的 Perception 模块：将原始观测（文本/记忆/物理信号）转换为
World Model 可消费的结构化特征。

职责：
- 接收原始文本或记忆列表 → 输出结构化感知特征
- 接收多模态信号 (v3.1.0) → 输出物理感知特征
- 封装 EvidenceCollector + EncoderCore + TemporalSystem 三源预处理
- 提供统一的 "sensor-like" 接口

处理流程:
    raw_input → evidence_collection → temporal_annotation →
    semantic_encoding → feature_extraction → structured_output

多模态流程 (v3.1.0):
    MultimodalSignal[] → signal_type_dispatch → feature_embedding → PerceivedFeatures

状态机: IDLE → COLLECTING → ENCODING → STRUCTURED → COMPLETE

用法:
    from mci_world_model._sys._perception_pipeline import PerceptionPipeline

    perception = PerceptionPipeline()
    features = perception.process(memories)
    mf = perception.process_multimodal(signals)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# v3.1.0: 多模态信号类型
# =============================================================================


class SignalType(Enum):
    """多模态信号类型枚举。"""

    TEXT = "text"
    NUMERICAL = "numerical"  # 单值数值 (血糖 5.6)
    TEMPORAL_SERIES = "temporal"  # 时序数据 (每日白蛋白: [30,32,31,33])
    LAB_STRUCTURED = "lab"  # 实验室检查 (多项结构化指标)
    CATEGORICAL = "categorical"  # 类别变量 (NRS2002 评分: 4)
    IMAGE = "image"  # v3.3.0: 2D 图像 (RGB/Depth/Thermal)
    AUDIO_FEATURES = "audio_features"  # v3.3.0: 音频特征向量


@dataclass
class MultimodalSignal:
    """
    多模态信号数据结构 (v3.1.0)。

    统一表示来自不同源的物理世界信号，
    包括实验室检查、护理记录、营养摄入等。

    Attributes:
        signal_type: 信号类型枚举
        value: 原始值 (数值/字符串/列表)
        timestamp: ISO 格式时间戳
        source: 信号来源 ("lab_report" | "nursing_note" | "diet_record" | ...)
        metadata: 额外元信息
    """

    signal_type: SignalType
    value: object
    timestamp: str = ""
    source: str = "unknown"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "signal_type": self.signal_type.value,
            "value": self.value if not isinstance(self.value, np.ndarray) else self.value.tolist(),
            "timestamp": self.timestamp,
            "source": self.source,
            "metadata": self.metadata,
        }


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

    # v3.2.0: 从物理信号构建的世界状态（感知→认知桥接）
    world_state: object | None = None

    @classmethod
    def empty(cls) -> PerceivedFeatures:
        return cls()

    def to_dict(self) -> dict:
        result = {
            "n_entities": len(self.entities),
            "has_temporal": bool(self.temporal_context),
            "energy_profile": self.energy_profile,
            "evidence_count": self.evidence_count,
            "has_world_state": self.world_state is not None,
        }
        if self.world_state is not None and hasattr(self.world_state, "to_dict"):
            result["world_state"] = self.world_state.to_dict()
        return result


# =============================================================================
# PerceptionPipeline — 统一感知前端
# =============================================================================


class PerceptionPipeline:
    """
    统一感知前端管道。

    **线程模型**: 非线程安全。_state 和 _process_count 为单线程场景设计，
    并发调用需外部加锁。

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
    # v3.1.0: 多模态处理
    # -----------------------------------------------------------------

    # 物理量 → 五范畴能量映射表
    ENERGY_PHYSICAL_MAP: dict[str, list[str]] = {
        "semantic": ["diagnosis_code", "chief_complaint", "disease_type"],
        "causal": ["medication_dose", "intervention_type", "treatment_code"],
        "spacetime": ["timestamp", "los_days", "season", "day_of_week"],
        "generative": ["albumin", "prealbumin", "calorie_intake", "protein_intake", "body_weight", "bmi"],
        "trust": ["nrs2002", "apache_ii", "evidence_level", "sofa_score", "glasgow_score"],
    }

    def process_multimodal(
        self,
        signals: list[MultimodalSignal],
        fusion_strategy: str = "attention",
        enable_fusion: bool = True,
    ) -> list[dict] | PerceivedFeatures:
        """
        v3.1.0/v4.5.0: 将多模态信号统一转换为因果发现可用的结构化特征。

        v4.5.0: 默认启用融合 (enable_fusion=True)，返回 PerceivedFeatures
        其中 world_state 为融合后 MultimodalWorldState。
        如需旧版纯特征列表，设置 enable_fusion=False。

        处理策略:
        - NUMERICAL → 单值特征 + 五范畴映射
        - TEMPORAL_SERIES → 统计量 (mean/std/trend/min/max)
        - LAB_STRUCTURED → {feature_name: value} 展开
        - CATEGORICAL → one-hot 或 ordinal 编码
        - TEXT → 关键词提取（降级至 process()）

        Args:
            signals: MultimodalSignal 列表
            fusion_strategy: 融合策略 "attention" / "weighted" / "concat"
            enable_fusion: 是否默认启用融合 (v4.5.0 默认 True)

        Returns:
            enable_fusion=True  → PerceivedFeatures (含融合后 world_state)
            enable_fusion=False → list[dict] 结构化特征
        """
        if enable_fusion:
            result = self.process_multimodal_fused(signals, fusion_strategy=fusion_strategy)
            return result

        self._state = "COLLECTING"
        if not signals:
            self._state = "COMPLETE"
            return []

        features: list[dict] = []

        for sig in signals:
            try:
                processed = self._dispatch_signal(sig)
                if processed:
                    if isinstance(processed, list):
                        features.extend(processed)
                    else:
                        features.append(processed)
            except Exception as e:
                logger.warning("信号处理异常 [%s]: %s", sig.signal_type.value, e)

        self._state = "COMPLETE"
        self._process_count += 1
        return features

    # -----------------------------------------------------------------
    # v3.2.0: 物理信号 → WorldState（感知→认知桥接）
    # -----------------------------------------------------------------

    def process_physical(
        self,
        signals: list,
        state_class: type | None = None,
    ) -> PerceivedFeatures:
        """v3.2.0: 处理物理信号 (PhysicalSignal) → 构建 WorldState。

        与 process() / process_multimodal() 并列，不替代。

        process():         文本记忆 → 实体提取 + 能量剖面
        process_multimodal(): 多模态信号 → 结构化特征 (v3.1.0)
        process_physical():   物理信号 → WorldState (v3.2.0 新)

        Args:
            signals: PhysicalSignal 列表
            state_class: 期望构建的 WorldState 子类
                        (None = 自动推断，从第一个信号的 modality 判断)

        Returns:
            PerceivedFeatures，其中 world_state 字段被填充
        """
        self._state = "COLLECTING"

        if not signals:
            self._state = "COMPLETE"
            return PerceivedFeatures.empty()

        features = PerceivedFeatures()

        # 自动推断 WorldState 类型
        if state_class is None:
            state_class = self._infer_state_class(signals)

        # 构建世界状态
        if state_class is not None:
            try:
                features.world_state = self._signals_to_world_state(signals, state_class)
            except Exception as e:
                logger.warning("process_physical 状态构建异常: %s", e)

        self._state = "STRUCTURED"
        self._last_features = features
        self._process_count += 1
        self._state = "COMPLETE"

        return features

    @staticmethod
    def _infer_state_class(signals: list) -> type | None:
        """从信号 modality 推断应该构建哪种 WorldState。

        v3.3.0: 注册表模式 — 支持多模态状态推断。
        规则:
            - 有本体感觉信号 (PROPRIOCEPTION) → PendulumState
            - 有多模态信号 (VISION + PROPRIOCEPTION) → MultimodalWorldState
        """

        # 注册表: modality → WorldState 类
        _STATE_REGISTRY: dict[str, type] = {}
        try:
            from mci_world_model.sdk._world_state import PendulumState

            _STATE_REGISTRY["proprioception"] = PendulumState
        except ImportError:
            pass
        try:
            from mci_world_model.sdk._world_state import MultimodalWorldState

            _STATE_REGISTRY["multimodal"] = MultimodalWorldState
        except ImportError:
            pass

        modalities_found: set[str] = set()
        for sig in signals:
            modality_val = str(getattr(sig, "modality", "")).lower()
            # 提取模态名 (enum repr or value)
            for m in ("proprioception", "vision", "audition", "tactition", "olfaction", "gustation"):
                if m in modality_val:
                    modalities_found.add(m)

        # 多模态优先
        if len(modalities_found) > 1 and "multimodal" in _STATE_REGISTRY:
            return _STATE_REGISTRY["multimodal"]

        # 单模态查找
        for m in modalities_found:
            if m in _STATE_REGISTRY:
                return _STATE_REGISTRY[m]

        return None

    @staticmethod
    def _signals_to_world_state(
        signals: list,
        state_class: type,
    ) -> object:
        """将物理信号编码为世界状态。

        委托给 state_class.from_signals() 类方法。
        每个 WorldState 子类自行定义如何从信号构建自身。

        Args:
            signals: PhysicalSignal 列表
            state_class: WorldState 子类

        Returns:
            WorldState 实例
        """
        if hasattr(state_class, "from_signals"):
            return state_class.from_signals(signals)
        raise NotImplementedError(f"{state_class.__name__} 未实现 from_signals() 类方法")

    # -----------------------------------------------------------------
    # v3.3.0: 多模态融合处理
    # -----------------------------------------------------------------

    def process_multimodal_fused(
        self,
        signals: list[MultimodalSignal],
        fusion_strategy: str = "attention",
    ) -> PerceivedFeatures:
        """v3.3.0: 多模态信号 → 各模态编码 → 融合 → PerceivedFeatures。

        完整流水线:
            signals → 各模态编码 → MultimodalFusion → MultimodalWorldState → PerceivedFeatures

        Args:
            signals: MultimodalSignal 列表
            fusion_strategy: 融合策略 "attention" / "weighted" / "concat"

        Returns:
            PerceivedFeatures，world_state 为 MultimodalWorldState
        """
        self._state = "COLLECTING"

        if not signals:
            self._state = "COMPLETE"
            return PerceivedFeatures.empty()

        features = PerceivedFeatures()

        # 1. 先做普通多模态处理（enable_fusion=False 获取纯特征列表，避免递归）
        processed = self.process_multimodal(signals, enable_fusion=False)

        # 2. 按模态分组收集特征向量
        modality_features: dict[str, np.ndarray] = {}
        modality_confidences: dict[str, float] = {}
        for item in processed:
            modality = item.get("modality", "")
            value = item.get("value")
            if modality and value is not None:
                try:
                    vec = np.asarray(value, dtype=np.float64).flatten()
                    modality_features[modality] = vec
                    modality_confidences[modality] = 1.0
                except (TypeError, ValueError):
                    continue

        # 3. 融合
        if modality_features:
            try:
                from mci_world_model.sdk._multimodal_fusion import MultimodalFusion

                fusion = MultimodalFusion(
                    strategy=fusion_strategy,
                    output_dim=32,
                )
                fused = fusion.fuse(modality_features, modality_confidences)

                # 4. 编码为 WorldState
                features.world_state = fusion.encode_to_state(fused)
            except Exception as e:
                logger.warning("process_multimodal_fused 融合异常: %s", e)

        self._state = "STRUCTURED"
        self._last_features = features
        self._process_count += 1
        self._state = "COMPLETE"

        return features

    def _dispatch_signal(self, sig: MultimodalSignal) -> object:
        """根据信号类型分派处理器。"""
        dispatcher = {
            SignalType.NUMERICAL: self._process_numerical,
            SignalType.TEMPORAL_SERIES: self._process_temporal_series,
            SignalType.LAB_STRUCTURED: self._process_lab_structured,
            SignalType.CATEGORICAL: self._process_categorical,
            SignalType.TEXT: self._process_text_signal,
            # v3.3.0: 多模态信号分派
            SignalType.IMAGE: self._process_image,
            SignalType.AUDIO_FEATURES: self._process_audio_features,
        }
        handler = dispatcher.get(sig.signal_type)
        if handler is None:
            logger.warning("未知信号类型: %s", sig.signal_type)
            return None
        return handler(sig)

    def _map_to_energy_category(self, feature_name: str) -> str:
        """将物理量名称映射到五范畴能量类型。"""
        for cat, names in self.ENERGY_PHYSICAL_MAP.items():
            if feature_name.lower() in [n.lower() for n in names]:
                return cat
        return "generative"  # 默认归类为生成类

    def _process_numerical(self, sig: MultimodalSignal) -> dict | None:
        """处理数值信号。"""
        try:
            val = float(sig.value)  # type: ignore
        except (TypeError, ValueError):
            return None
        if not np.isfinite(val):
            return None
        name = sig.metadata.get("name", sig.source)
        return {
            "feature_name": name,
            "value": val,
            "category": self._map_to_energy_category(name),
            "timestamp": sig.timestamp,
            "source": sig.source,
        }

    def _process_temporal_series(self, sig: MultimodalSignal) -> list[dict]:
        """处理时序信号 → 统计特征。"""
        try:
            arr = np.array(sig.value, dtype=np.float64)  # type: ignore
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                return []
        except (TypeError, ValueError):
            return []

        name = sig.metadata.get("name", sig.source)
        cat = self._map_to_energy_category(name)
        base = {"category": cat, "timestamp": sig.timestamp, "source": sig.source}

        result = []
        stats = [
            ("mean", float(np.mean(arr))),
            ("std", float(np.std(arr)) if arr.size > 1 else 0.0),
            ("min", float(np.min(arr))),
            ("max", float(np.max(arr))),
        ]
        # 趋势 (最后3天 vs 前3天的差值)
        if arr.size >= 6:
            trend = float(np.mean(arr[-3:]) - np.mean(arr[:3]))
            stats.append(("trend", trend))

        for stat_name, stat_val in stats:
            result.append(
                {
                    "feature_name": f"{name}_{stat_name}",
                    "value": round(stat_val, 4),
                    **base,
                }
            )
        return result

    def _process_lab_structured(self, sig: MultimodalSignal) -> list[dict]:
        """处理实验室结构化数据 → 逐项展开。"""
        val = sig.value
        if not isinstance(val, dict):
            return []

        result = []
        for item_name, item_val in val.items():
            try:
                fv = float(item_val)  # type: ignore
                if not np.isfinite(fv):
                    continue
            except (TypeError, ValueError):
                continue
            result.append(
                {
                    "feature_name": str(item_name),
                    "value": fv,
                    "category": self._map_to_energy_category(str(item_name)),
                    "timestamp": sig.timestamp,
                    "source": sig.source,
                }
            )
        return result

    def _process_categorical(self, sig: MultimodalSignal) -> dict | None:
        """处理类别信号。"""
        name = sig.metadata.get("name", sig.source)
        raw_val = sig.value
        # 尝试数值化
        try:
            num_val = float(raw_val)  # type: ignore
        except (TypeError, ValueError):
            num_val = float(hash(str(raw_val)) % 100) / 100.0  # hash 映射

        return {
            "feature_name": name,
            "value": num_val,
            "category": self._map_to_energy_category(name),
            "timestamp": sig.timestamp,
            "source": sig.source,
            "raw_label": str(raw_val),
        }

    def _process_image(self, sig: MultimodalSignal) -> dict | None:
        """v3.3.0: 处理图像信号 → VisionEncoder 特征。"""
        try:
            from mci_world_model.sdk._modality_encoders import VisionEncoder

            frame = np.asarray(sig.value, dtype=np.float64)
            if frame.ndim < 2:
                return None
            enc = VisionEncoder(feature_dim=32)
            features = enc.encode(frame)
            name = sig.metadata.get("name", sig.source)
            return {
                "feature_name": f"{name}_vision",
                "value": features.tolist(),
                "category": "generative",
                "timestamp": sig.timestamp,
                "source": sig.source,
                "modality": "vision",
                "feature_dim": enc.feature_dim,
            }
        except Exception as e:
            logger.warning("图像处理异常: %s", e)
            return None

    def _process_audio_features(self, sig: MultimodalSignal) -> dict | None:
        """v3.3.0: 处理音频特征信号 → AudioEncoder 或直接使用特征向量。"""
        try:
            arr = np.asarray(sig.value, dtype=np.float64)
            if arr.size == 0:
                return None
            # 如果已经是特征向量 (1D)，直接使用
            name = sig.metadata.get("name", sig.source)
            if arr.ndim == 1:
                return {
                    "feature_name": f"{name}_audio",
                    "value": arr.tolist(),
                    "category": "generative",
                    "timestamp": sig.timestamp,
                    "source": sig.source,
                    "modality": "audio",
                    "feature_dim": len(arr),
                }
            # 原始波形 → AudioEncoder
            from mci_world_model.sdk._modality_encoders import AudioEncoder

            enc = AudioEncoder(feature_dim=16)
            features = enc.encode(arr.flatten())
            return {
                "feature_name": f"{name}_audio",
                "value": features.tolist(),
                "category": "generative",
                "timestamp": sig.timestamp,
                "source": sig.source,
                "modality": "audio",
                "feature_dim": enc.feature_dim,
            }
        except Exception as e:
            logger.warning("音频处理异常: %s", e)
            return None

    def _process_text_signal(self, sig: MultimodalSignal) -> dict | None:
        """处理文本信号 → 降级为简单特征。"""
        content = str(sig.value) if sig.value else ""
        if not content.strip():
            return None
        words = re.findall(r"[\u4e00-\u9fff]{2,}", content)
        return {
            "feature_name": "text_signal",
            "value": float(len(words)),
            "category": "semantic",
            "timestamp": sig.timestamp,
            "source": sig.source,
            "word_count": len(words),
        }

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
            logger.warning("Perception evidence collect 跳过: %s", e)
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
                                temporal_info if isinstance(temporal_info, dict) else {"info": str(temporal_info)}
                            )
                    except Exception as e:
                        logger.warning("吞异常", exc_info=True)
            return latest_context
        except Exception as e:
            logger.warning("Perception temporal 跳过: %s", e)
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
            logger.warning("Perception energy profile 跳过: %s", e)
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
            logger.warning("Perception semantic encode 跳过: %s", e)
            return []

    # -----------------------------------------------------------------
    # v3.6.0: 感知注意力调整 (Perception Attention Policy)
    # -----------------------------------------------------------------

    def attention_policy(
        self,
        feedback: dict[str, float] | None = None,
        **kwargs,
    ) -> dict[str, float]:
        """v3.6.0: 基于反馈信号动态调整感知通道的采样权重。

        理论基础:
            感知环反向调整 — 失败的通道应获得更高的采样权重，
            以便在下次感知中获得更多信息。成功的通道可以降低
            权重以节省计算资源。

        输入格式:
            feedback = {
                "semantic": -0.3,     # 负值 = 失败信号
                "causal": 0.5,        # 正值 = 成功信号
                "spacetime": -0.8,    # 强失败信号
                "prediction_error": 0.7,  # 预测误差
                "surprise": 0.9,      # 惊奇度
            }

        调整规则:
            1. 失败信号 (feedback < 0) → 提升对应通道权重
            2. 成功信号 (feedback > 0) → 降低对应通道权重
            3. prediction_error → 全局提升所有通道权重
            4. surprise → 提升所有通道 + 重置衰减
            5. 最终权重归一化到 [0, 1]

        Args:
            feedback: 各通道反馈信号

        Returns:
            调整后的通道采样权重 dict
        """
        if feedback is None:
            feedback = kwargs

        # 默认五通道权重
        if not hasattr(self, "_attention_weights"):
            self._attention_weights: dict[str, float] = {
                "semantic": 0.20,
                "causal": 0.20,
                "spacetime": 0.20,
                "generative": 0.20,
                "trust": 0.20,
            }

        weights = dict(self._attention_weights)
        lr = 0.15  # 注意力学习率

        # 1. 处理各通道反馈
        channel_keys = {"semantic", "causal", "spacetime", "generative", "trust"}
        for key in channel_keys:
            if key in feedback:
                signal = feedback[key]
                if signal < 0:
                    # 失败信号 → 提升权重
                    weights[key] = weights[key] + lr * abs(signal)
                elif signal > 0:
                    # 成功信号 → 降低权重（但不低于 0.05）
                    weights[key] = max(0.05, weights[key] - lr * signal * 0.5)

        # 2. 预测误差 → 全局提升
        pred_error = feedback.get("prediction_error", 0.0)
        if pred_error > 0.3:
            boost = lr * pred_error * 0.5
            for key in weights:
                weights[key] += boost

        # 3. 惊奇度 → 全面增强
        surprise = feedback.get("surprise", 0.0)
        if surprise > 0.5:
            for key, value in weights.items():
                weights[key] = min(1.0, value * (1.0 + surprise * 0.3))

        # 4. 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        # 5. 持久化
        self._attention_weights = dict(weights)

        return dict(weights)

    @property
    def attention_weights(self) -> dict[str, float]:
        """当前通道采样权重。"""
        if not hasattr(self, "_attention_weights"):
            return {
                "semantic": 0.20,
                "causal": 0.20,
                "spacetime": 0.20,
                "generative": 0.20,
                "trust": 0.20,
            }
        return dict(self._attention_weights)

    def reset_attention(self) -> None:
        """重置注意力权重为均匀分布。"""
        self._attention_weights = {
            "semantic": 0.20,
            "causal": 0.20,
            "spacetime": 0.20,
            "generative": 0.20,
            "trust": 0.20,
        }


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
