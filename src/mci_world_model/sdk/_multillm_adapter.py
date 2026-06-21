from __future__ import annotations

"""
MCI World Model v3.1.1 — MultiLLM 适配器

零硬依赖 MultiLLM 接口 — 桥接 ai-native-nutrition-v1 的三环推理能力。
Provider 链: Ollama(qwen3.5) → OpenAI(gpt-4o-mini) → graceful 降级。

设计原则:
- 零硬依赖: 不依赖 torch/transformers，使用标准库 + requests
- 可插拔: 任意 LLM provider 通过 register_provider() 扩展
- 降级优雅: 首个 provider 不可用时自动 fallback，全部不可用返回预设响应

用法:
    from mci_world_model.sdk._multillm_adapter import MultiLLMAdapter

    adapter = MultiLLMAdapter(providers=["ollama", "openai"])
    response = adapter.generate("患者白蛋白 28 g/L，推荐营养方案")
    label = adapter.classify("患者体重下降 5kg", ["低风险", "中风险", "高风险"])
"""


import json
import logging
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# Provider Registry
# =============================================================================

_LLM_REGISTRY: dict[str, Callable] = {}  # type: ignore


def register_provider(name: str, fn: Callable) -> None:  # type: ignore
    """注册自定义 LLM provider。"""
    _LLM_REGISTRY[name] = fn


# =============================================================================
# 降级嵌入 (无 LLM 可用时使用)
# =============================================================================


def _charset_embed(text: str, dim: int = 3) -> np.ndarray:
    """
    字符集嵌入: 基于字符频率的确定性嵌入。
    无模型依赖，所有 language 均可使用。
    """
    char_vec = np.zeros(26 + 10 + 1, dtype=np.float32)  # a-z, 0-9, other
    for ch in text.lower():
        if "a" <= ch <= "z":
            char_vec[ord(ch) - ord("a")] += 1
        elif "0" <= ch <= "9":
            char_vec[26 + ord(ch) - ord("0")] += 1
        else:
            char_vec[-1] += 1

    total = char_vec.sum() + 1e-10
    char_vec /= total

    # 投影到 dim 维
    if dim <= char_vec.shape[0]:
        return char_vec[:dim].astype(np.float64)
    else:
        # 使用简单的 FFT-based 上投影
        fft_features = np.abs(np.fft.rfft(char_vec))[:dim]
        return (fft_features / (fft_features.sum() + 1e-10)).astype(np.float64)


# =============================================================================
# Ollama Provider
# =============================================================================


class OllamaProvider:
    """Ollama 本地推理 provider。"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen3.5", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _is_available(self) -> bool:
        try:
            import urllib.request

            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def generate(self, prompt: str, system: str = "", **kwargs: Any) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system or "你是一个医学营养专家。请用中文简洁回答。",
            "stream": False,
        }
        payload.update(kwargs)

        try:
            import urllib.request

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "")
        except Exception as e:
            logger.warning(f"Ollama generate failed: {e}")
            raise


class OpenAIProvider:
    """OpenAI API provider。"""

    def __init__(self, api_key: str = "", model: str = "gpt-4o-mini", base_url: str = "", timeout: float = 30.0) -> None:
        import os

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.timeout = timeout

    def _is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, system: str = "", **kwargs: Any) -> str:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 500,
        }
        payload.update(kwargs)

        try:
            import urllib.request

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"OpenAI generate failed: {e}")
            raise


# =============================================================================
# MultiLLMAdapter
# =============================================================================


class MultiLLMAdapter:
    """
    Multi-LLM 适配器 — 自动降级链。

    Provider 链:
    1. Ollama (qwen3.5) — 本地优先，低延迟
    2. OpenAI (gpt-4o-mini) — 云兜底
    3. 降级响应 — 全部不可用时返回预设

    Example:
        >>> adapter = MultiLLMAdapter(providers=["ollama", "openai"])
        >>> # 检查可用性
        >>> status = adapter.health_check()
        >>> # 生成回答
        >>> response = adapter.generate("患者营养方案建议")
        >>> # 分类
        >>> label = adapter.classify("体重下降", ["轻微", "中等", "严重"])
    """

    def __init__(self, providers: list[str] | None = None, **kwargs: Any) -> None:
        """
        Args:
            providers: LLM provider 列表，如 ["ollama", "openai"]
            **kwargs: 传递给每个 provider 的配置
        """
        self._provider_instances: dict[str, Any] = {}
        self._available = False
        self._active_provider: str | None = None

        provider_list = providers or ["ollama"]

        # 初始化 providers
        for name in provider_list:
            try:
                inst = self._init_provider(name, **kwargs)
                if inst is not None:
                    self._provider_instances[name] = inst
            except Exception as e:
                logger.warning("Provider '%s' init skipped: %s", name, e)

        # 检测可用性
        self._detect_availability()

    def _init_provider(self, name: str, **kwargs: Any) -> Any:
        """初始化单个 provider。"""
        if name == "ollama":
            return OllamaProvider(
                base_url=kwargs.get("ollama_url", "http://localhost:11434"),
                model=kwargs.get("ollama_model", "qwen3.5"),
                timeout=kwargs.get("timeout", 30.0),
            )
        elif name == "openai":
            return OpenAIProvider(
                api_key=kwargs.get("openai_api_key", ""),
                model=kwargs.get("openai_model", "gpt-4o-mini"),
                timeout=kwargs.get("timeout", 30.0),
            )
        elif name in _LLM_REGISTRY:
            return _LLM_REGISTRY[name](kwargs)
        else:
            logger.warning(f"Unknown provider: {name}")
            return None

    def _detect_availability(self) -> None:
        """按优先级检测 provider 可用性，设置 _active_provider。"""
        for name, inst in self._provider_instances.items():
            if hasattr(inst, "_is_available") and inst._is_available():
                self._active_provider = name
                self._available = True
                logger.info(f"MultiLLM: active provider = {name}")
                return

        self._active_provider = None
        self._available = False
        logger.info("MultiLLM: no LLM available — using fallback mode")

    def generate(self, prompt: str, context: dict[str, Any] | None = None, **kwargs: Any) -> str:
        """
        使用活跃 provider 生成文本。

        Args:
            prompt: 主提示词
            context: 额外上下文 (可选)
            **kwargs: 传递给 provider 的额外参数

        Returns:
            生成的文本
        """
        # 注入上下文
        if context:
            ctx_str = "\n".join(f"{k}: {v}" for k, v in context.items())
            prompt = f"基于以下上下文:\n{ctx_str}\n\n问题: {prompt}"

        # 尝试活跃 provider
        if self._active_provider is not None:
            try:
                inst = self._provider_instances[self._active_provider]
                system = kwargs.pop("system", "你是一个临床营养专家，请用中文简洁回答（不超过 200 字）。")
                return inst.generate(prompt, system=system, **kwargs)
            except Exception as e:
                logger.warning(f"Provider '{self._active_provider}' failed: {e}")

        # 降级响应
        return self._fallback_generate(prompt, context)

    def _fallback_generate(self, prompt: str, context: dict | None = None) -> str:  # type: ignore
        """无 LLM 可用时的降级响应。"""
        return (
            "根据临床营养指南，建议进行综合营养评估（包括白蛋白、前白蛋白、NRS2002评分），"
            "制定个体化营养方案。如需详细建议，请配置 LLM 服务（Ollama 或 OpenAI）。"
        )

    def classify(self, text: str, labels: list[str], **kwargs: Any) -> dict[str, Any]:
        """
        文本分类。

        Args:
            text: 待分类文本
            labels: 候选标签列表
            **kwargs: 传递给 provider 的额外参数

        Returns:
            {"label": "最可能标签", "scores": {"label": 0.8, ...}, "method": "llm"|"rules"}
        """
        if self._active_provider is not None:
            try:
                labels_str = "、".join(labels)
                prompt = f"将以下文本分类到最合适的类别中。类别: {labels_str}。只输出类别名称。\n\n文本: {text}"
                inst = self._provider_instances[self._active_provider]
                result = inst.generate(prompt, system="你是一个分类助手，只输出类别名称。", **kwargs)
                result = result.strip()
                # 匹配最接近的标签
                matched = self._match_label(result, labels)
                return {"label": matched, "scores": {matched: 1.0}, "method": "llm"}
            except Exception as e:
                logger.warning("LLM 分类失败，降级为规则匹配: %s", e)

        # 降级: 规则匹配
        matched = self._rules_classify(text, labels)
        return {
            "label": matched,
            "scores": {lbl: 1.0 if lbl == matched else 0.0 for lbl in labels},
            "method": "rules",
        }

    def _rules_classify(self, text: str, labels: list[str]) -> str:
        """基于关键词的规则分类。"""
        text_lower = text.lower()
        best_label = labels[0]
        best_score = -1.0
        for label in labels:
            score = float(sum(1 for ch in label if ch in text or ch.lower() in text_lower))
            score += len(label) / 20.0  # 轻微偏好更具体的标签
            if score > best_score:
                best_score = score
                best_label = label
        return best_label

    def _match_label(self, result: str, labels: list[str]) -> str:
        """将 LLM 返回结果匹配到最接近的标签。"""
        best = labels[0]
        best_score = -1.0
        for label in labels:
            if label in result:
                return label
            # 模糊匹配
            common = sum(1 for ch in label if ch in result)
            if common > best_score:
                best_score = common
                best = label
        return best

    def embed(self, text: str, dim: int = 3) -> np.ndarray:
        """
        文本嵌入。

        策略: 优先使用 LLM embedding API，不可用时使用字符集嵌入。

        Args:
            text: 输入文本
            dim: 嵌入维度 (默认 3，兼容 JEPA 潜空间)

        Returns:
            numpy 数组 shape=(dim,)
        """
        if self._active_provider is not None and self._active_provider == "ollama":
            try:
                import urllib.request

                inst = self._provider_instances["ollama"]
                url = f"{inst.base_url}/api/embeddings"
                payload = json.dumps({"model": inst.model, "prompt": text}).encode("utf-8")
                req = urllib.request.Request(
                    url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    embedding = np.array(result.get("embedding", []), dtype=np.float64)
                    if len(embedding) > dim:
                        # PCA-like 降维 (取前 dim 维)
                        return embedding[:dim]
                    return embedding
            except Exception as e:
                logger.warning(f"Ollama embed failed: {e}")

        # 降级: 字符集嵌入
        return _charset_embed(text, dim)

    def health_check(self) -> dict[str, Any]:
        """健康检查。返回可用性状态。"""
        providers_status = {}
        for name, inst in self._provider_instances.items():
            if hasattr(inst, "_is_available"):
                try:
                    providers_status[name] = inst._is_available()
                except Exception:
                    providers_status[name] = False
            else:
                providers_status[name] = True  # 无检测方法 = 假定可用

        return {
            "available": self._available,
            "active_provider": self._active_provider,
            "providers": providers_status,
            "mode": "llm" if self._available else "fallback",
        }

    def reason_with_cf(  # type: ignore
        self,
        prompt: str,
        cf_result: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """v4.4.2: 基于反事实推演结果的再推理。

        LLM↔CEWM 双向反馈闭环核心——LLM 接收 CEWM 反事实推演结果，
        将结果融入后续推理，实现"如果选 A 会怎样 → CEWM 推演 → 改为选 B"的决策闭环。

        Args:
            prompt: 原始问题/提示词
            cf_result: CounterfactualOracle.query() 的返回结果
                {"best_scenario": str, "best_effect": float,
                 "rankings": [...], "recommendation": str, "n_scenarios": int}
            context: 额外上下文
            **kwargs: 传递给 provider 的额外参数

        Returns:
            融合了反事实推演结果的推理文本
        """
        if cf_result is None or not cf_result.get("rankings"):
            # 无 CF 结果，退化为普通 generate
            return self.generate(prompt, context=context, **kwargs)

        # 构建 CF 增强提示词
        cf_prompt = self._build_cf_prompt(prompt, cf_result)

        # 尝试活跃 provider
        if self._active_provider is not None:
            try:
                inst = self._provider_instances[self._active_provider]
                system = kwargs.pop(
                    "system",
                    "你是一个临床营养专家。请基于世界模型反事实推演结果，"
                    "用中文给出推理过程和最终建议（不超过 300 字）。",
                )
                return inst.generate(cf_prompt, system=system, **kwargs)
            except Exception as e:
                logger.warning("reason_with_cf provider 失败: %s，降级", e)

        # 降级: 基于 CF 结果的规则推理
        return self._fallback_reason_with_cf(prompt, cf_result)

    def _build_cf_prompt(self, prompt: str, cf_result: dict[str, Any]) -> str:
        """构建 CF 增强提示词。"""
        best = cf_result.get("best_scenario", "未知")
        best_effect = cf_result.get("best_effect")
        recommendation = cf_result.get("recommendation", "")
        rankings = cf_result.get("rankings", [])

        ranking_str = "\n".join(
            f"  - {r['name']}: 效应={r.get('effect', 'N/A'):.3f}, "
            f"置信度={r.get('confidence', 0):.0%}, "
            f"排名={r.get('rank', -1) + 1}"
            for r in rankings
            if isinstance(r, dict)
        )

        return (
            f"原始问题: {prompt}\n\n"
            f"基于世界模型反事实推演结果:\n"
            f"  最优方案: {best}\n"
            f"  最优效应: {best_effect}\n"
            f"  各方案推演:\n{ranking_str}\n"
            f"  推荐理由: {recommendation}\n\n"
            f"请基于以上反事实推演结果，给出你的推理过程和最终建议。"
        )

    def _fallback_reason_with_cf(self, prompt: str, cf_result: dict[str, Any]) -> str:
        """无 LLM 时的降级 CF 推理。"""
        best = cf_result.get("best_scenario", "未知")
        recommendation = cf_result.get("recommendation", "")
        is_uncertain = any(r.get("is_uncertain", False) for r in cf_result.get("rankings", []))

        if is_uncertain:
            return f"基于世界模型反事实推演，{best} 可能较优，但推演结果不确定，建议结合更多信息决策。"
        return recommendation or f"推荐方案: {best}"
