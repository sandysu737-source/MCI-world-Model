from __future__ import annotations

"""
MCI World Model v3.1.1 — 增强感知管道

在 PerceptionPipeline 基础上添加 MultiLLM 语义增强：
- 自由文本 → 结构化信号提取
- 多模态信号 → 因果上下文理解
- 端到端: 原始观测 → 因果状态 (Perception → JEPA)

用法:
    from mci_world_model.sdk._enhanced_perception import EnhancedPerception

    ep = EnhancedPerception(multillm=adapter)
    signals = ep.extract_signals("患者近3天白蛋白从35降至28，食欲不振")
    state = ep.perceive_to_state(signals)
"""


import logging
from typing import Any

logger = logging.getLogger(__name__)


class EnhancedPerception:
    """
    增强感知管道 — 桥接 LLM 语义理解与 MCI 结构化推理。

    三层处理:
    1. 文本提取: LLM 从自由文本中提取结构化信号
    2. 信号处理: PerceptionPipeline 多模态特征提取
    3. 状态编码: JEPAEncoder 编码为因果世界状态
    """

    def __init__(self, multillm: Any = None, pipeline: Any = None) -> None:
        """
        Args:
            multillm: MultiLLMAdapter 实例
            pipeline: PerceptionPipeline 实例 (可选，lazyload)
        """
        self._multillm = multillm
        self._pipeline = pipeline

    # -----------------------------------------------------------------
    # Layer 1: 文本提取
    # -----------------------------------------------------------------

    def extract_signals(self, text: str, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        从自由文本中提取结构化信号。

        Args:
            text: 自然语言描述 (e.g., "患者体重 72kg，白蛋白 28 g/L")
            context: 额外上下文

        Returns:
            [
                {"signal_type": "numerical", "name": "albumin", "value": 28.0, ...},
                {"signal_type": "numerical", "name": "body_weight", "value": 72.0, ...},
            ]
        """
        signals: list[dict[str, Any]] = []

        if self._multillm is not None:
            signals = self._llm_extract(text, context)
        else:
            signals = self._rule_extract(text)

        return signals

    def _llm_extract(self, text: str, context: dict | None) -> list[dict[str, Any]]:  # type: ignore
        """使用 LLM 提取结构化信号。"""
        try:
            prompt = f"""从以下临床文本中提取定量信号。仅输出 JSON 数组格式，每个信号包含:
- name: 指标名称 (如 albumin, body_weight, calorie_intake, nrs2002_score, prealbumin)
- value: 数值
- unit: 单位 (如 g/L, kg, kcal/day)
- trend: rising/falling/stable/unknown

文本: {text}

输出示例:
[{{"name": "albumin", "value": 28.0, "unit": "g/L", "trend": "falling"}}]
"""
            raw = self._multillm.generate(prompt, system="仅输出 JSON 数组，不要输出其他内容。")
            return self._parse_signal_json(raw)
        except Exception as e:
            logger.warning(f"LLM signal extraction failed: {e}")
            return self._rule_extract(text)

    def _rule_extract(self, text: str) -> list[dict[str, Any]]:
        """基于规则的信号提取 (降级方案)。"""
        import re

        signals = []
        seen = set()
        patterns = {
            "albumin": [
                (r"白蛋白\s*[:：]?\s*(\d+\.?\d*)\s*(g/L|g/l)", "g/L"),
                (r"白蛋白\s*[:：]?(?:从\d+)?[降至]+(\d+\.?\d*)", "g/L"),
                (r"albumin\s*[:：]?\s*(\d+\.?\d*)", "g/L"),
            ],
            "prealbumin": [
                (r"前白蛋白\s*[:：]?\s*(\d+\.?\d*)\s*(mg/L|mg/l)", "mg/L"),
                (r"prealbumin\s*[:：]?\s*(\d+\.?\d*)", "mg/L"),
            ],
            "body_weight": [
                (r"体重\s*[:：]?\s*(\d+\.?\d*)\s*(kg)", "kg"),
                (r"(\d+\.?\d*)\s*公斤", "kg"),
            ],
            "calorie_intake": [
                (r"热量\s*[:：]?\s*(\d+\.?\d*)\s*(kcal)", "kcal/day"),
                (r"热量摄入\s*[:：]?\s*(\d+\.?\d*)", "kcal/day"),
            ],
            "nrs2002_score": [
                (r"NRS\s*2002\s*[:：=]?\s*(\d+\.?\d*)", ""),
                (r"NRS2002\s*(?:评分|得分)?[:：=]?\s*(\d+\.?\d*)", ""),
            ],
            "protein_intake": [
                (r"蛋白质\s*[:：]?\s*(\d+\.?\d*)\s*(g/d|g)", "g/day"),
                (r"蛋白(?:质)?摄入\s*[:：]?\s*(\d+\.?\d*)", "g/day"),
            ],
        }

        for name, pats in patterns.items():
            for pat, unit in pats:
                m = re.search(pat, text, re.IGNORECASE)
                if m and name not in seen:
                    seen.add(name)
                    signals.append(
                        {
                            "signal_type": "numerical",
                            "name": name,
                            "value": float(m.group(1)),
                            "unit": unit or "unknown",
                            "source": "text_extraction",
                        }
                    )
                    break

        return signals

    def _parse_signal_json(self, raw: str) -> list[dict[str, Any]]:
        """解析 LLM 返回的 JSON 信号数组。"""
        import json

        # 尝试提取 JSON 块
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start >= 0 and end > start:
            json_str = raw[start:end]
        else:
            json_str = raw.strip()

        try:
            items = json.loads(json_str)
            if isinstance(items, list):
                return [
                    {
                        "signal_type": "numerical",
                        "name": item.get("name", "unknown"),
                        "value": float(item.get("value", 0)),
                        "unit": item.get("unit", ""),
                        "trend": item.get("trend", "unknown"),
                        "source": "llm_extraction",
                    }
                    for item in items
                ]
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON parse failed: {e}")

        return []

    # -----------------------------------------------------------------
    # Layer 2: 信号处理 (lazyload PerceptionPipeline)
    # -----------------------------------------------------------------

    def process_signals(self, signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        信号特征提取 (通过 PerceptionPipeline)。

        自动将 dict 格式转换为 MultimodalSignal 格式。

        Args:
            signals: extract_signals() 或外部输入的结构化信号

        Returns:
            特征列表
        """
        if not signals:
            return []

        if self._pipeline is None:
            from mci_world_model._sys._perception_pipeline import PerceptionPipeline

            self._pipeline = PerceptionPipeline()  # type: ignore

        # 转换为 MultimodalSignal 格式
        multimodal = self._dict_to_multimodal(signals)
        if not multimodal:
            return signals

        try:
            return self._pipeline.process_multimodal(multimodal, enable_fusion=False)
        except Exception as e:
            logger.warning(f"Signal processing failed: {e}")
            return signals  # 降级: 返回原始信号

    def _dict_to_multimodal(self, signals: list[dict[str, Any]]) -> list[Any]:
        """将 dict 信号列表转换为 MultimodalSignal 对象列表。"""
        from mci_world_model._sys._perception_pipeline import MultimodalSignal, SignalType

        result = []
        for sig in signals:
            sig_type = SignalType.NUMERICAL
            if sig.get("signal_type") == "categorical":
                sig_type = SignalType.CATEGORICAL
            elif sig.get("signal_type") == "temporal_series":
                sig_type = SignalType.TEMPORAL_SERIES

            try:
                ms = MultimodalSignal(
                    signal_type=sig_type,
                    value=sig.get("value", 0.0),
                    timestamp=sig.get("timestamp", ""),
                    source=sig.get("source", "perception"),
                    metadata={
                        "name": sig.get("name", "unknown"),
                        "unit": sig.get("unit", ""),
                        "trend": sig.get("trend", "unknown"),
                    },
                )
                result.append(ms)
            except Exception as e:
                logger.warning("Skip signal conversion: %s", e)

        return result

    # -----------------------------------------------------------------
    # Layer 3: 端到端感知→状态
    # -----------------------------------------------------------------

    def perceive_to_state(self, signals: list[dict[str, Any]]) -> Any:
        """
        端到端: 原始信号 → 因果世界状态。

        Pipeline: signals → process_multimodal → JEPAEncoder.encode → CausalWorldModelState

        Args:
            signals: 结构化信号列表

        Returns:
            CausalWorldModelState
        """
        from mci_world_model.sdk._jepa_encoder import JEPAEncoder

        # Layer 2: 信号处理
        features = self.process_signals(signals)
        if not features:
            # 空结果 → 尝试直接编码
            encoder = JEPAEncoder(world_model=None)
            return encoder.encode(signals=signals)

        # Layer 3: 状态编码
        encoder = JEPAEncoder(world_model=None)
        return encoder.encode(signals=features)

    def text_to_state(self, text: str, context: dict[str, Any] | None = None) -> Any:
        """
        端到端: 自然语言 → 因果世界状态。

        Pipeline: text → extract_signals → process_multimodal → JEPAEncoder.encode → State

        Args:
            text: 自然语言描述
            context: 额外上下文

        Returns:
            CausalWorldModelState
        """
        signals = self.extract_signals(text, context)
        if not signals:
            from mci_world_model.sdk._jepa_encoder import CausalWorldModelState  # type: ignore

            return CausalWorldModelState.empty()

        return self.perceive_to_state(signals)
