"""
benchmarks/real_world/llm_baseline.py — LLM 因果推理基线 (v5.0.0)

使用 GPT-4o / Claude / Ollama Qwen3 进行 prompt-based 因果推理，
与 CEWM 在相同的 MIMIC 子集上对比。

成本控制策略:
- 主力: Ollama 本地 Qwen3 (零成本)
- 验证: GPT-4o / Claude 仅最终轮 (记录 token 和成本)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

from benchmarks.real_world.mimic_causal_benchmark import (
    BenchmarkResult,
    MIMICCausalBenchmark,
    PatientTimeline,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt 模板
# ─────────────────────────────────────────────────────────────────────────────

CAUSAL_DISCOVERY_PROMPT = """You are a clinical causal reasoning expert. Given the following ICU patient vitals over 48 hours, identify causal relationships between variables.

Patient data summary:
{data_summary}

Known ICU variables: heart_rate, mean_arterial_pressure, dopamine_dose, norepinephrine_dose, lactate, creatinine, albumin, fluid_input, glucose, etc.

Identify causal relationships where one variable directly influences another. Output as a JSON array:
[
  {{"cause": "variable_name", "effect": "variable_name", "direction": "positive", "confidence": 0.8, "reasoning": "brief explanation"}},
  ...
]

Rules:
- "direction" must be "positive" (increase causes increase) or "negative" (increase causes decrease)
- Only include relationships with confidence >= 0.3
- Focus on pharmacological and physiological causal mechanisms
- Consider temporal precedence (cause should precede effect)
"""

# ─────────────────────────────────────────────────────────────────────────────
# LLM 结果解析
# ─────────────────────────────────────────────────────────────────────────────


def parse_llm_causal_output(text: str) -> list[tuple]:
    """
    从 LLM 输出中解析因果边。

    支持格式:
    - JSON 数组: [{"cause": "X", "effect": "Y", "direction": "+/-"}, ...]
    - 自由文本: "X → Y (positive)", "X causes Y to increase"

    Returns:
        [(cause, effect, direction, ate, magnitude), ...]
    """
    edges = []

    # 尝试 JSON 解析
    json_match = re.search(r"\[.*\]", text, re.DOTALL)
    if json_match:
        try:
            items = json.loads(json_match.group())
            for item in items:
                if isinstance(item, dict) and "cause" in item and "effect" in item:
                    cause = str(item["cause"]).lower().strip()
                    effect = str(item["effect"]).lower().strip()
                    raw_dir = str(item.get("direction", "positive")).strip().lower()

                    if raw_dir in ("positive", "+", "increase"):
                        direction = "positive"
                    elif raw_dir in ("negative", "-", "decrease"):
                        direction = "negative"
                    else:
                        direction = "neutral"

                    confidence = float(item.get("confidence", 0.5))
                    edges.append((cause, effect, direction, confidence, "medium"))
            return edges
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: 正则解析
    # 匹配 "X → Y" 或 "X -> Y" 或 "X causes Y"
    patterns = [
        r"(\w+)\s*→\s*(\w+)\s*\((\w+)\)",
        r"(\w+)\s*->\s*(\w+)\s*\((\w+)\)",
        r'"cause":\s*"(\w+)"\s*,\s*"effect":\s*"(\w+)"\s*,\s*"direction":\s*"(\w+)"',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            cause, effect, direction_raw = match.groups()
            direction = (
                "positive"
                if direction_raw.lower() in ("positive", "+", "increase")
                else ("negative" if direction_raw.lower() in ("negative", "-", "decrease") else "neutral")
            )
            edges.append((cause.lower(), effect.lower(), direction, 0.5, "medium"))

    return edges


# ─────────────────────────────────────────────────────────────────────────────
# LLM 客户端抽象
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LLMCallResult:
    """单次 LLM 调用结果。"""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    error: str = ""


def call_ollama(
    prompt: str,
    model: str = "qwen3:8b",
    base_url: str = "http://localhost:11434",
    timeout: int = 60,
) -> LLMCallResult:
    """调用本地 Ollama API (零成本)。"""
    import urllib.error
    import urllib.request

    start = time.time()
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 1024},
        }
    ).encode()

    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode())
            latency = time.time() - start
            return LLMCallResult(
                text=result.get("response", ""),
                model=model,
                latency_seconds=latency,
            )
    except urllib.error.URLError as e:
        return LLMCallResult(
            text="",
            model=model,
            error=str(e),
            latency_seconds=time.time() - start,
        )
    except Exception as e:
        return LLMCallResult(
            text="",
            model=model,
            error=f"Unexpected: {e}",
            latency_seconds=time.time() - start,
        )


def call_openai_api(
    prompt: str,
    model: str = "gpt-4o",
    api_key: str = "",
    base_url: str = "https://api.openai.com/v1",
    timeout: int = 60,
) -> LLMCallResult:
    """调用 OpenAI 兼容 API (记录成本)。"""
    import urllib.error
    import urllib.request

    if not api_key:
        return LLMCallResult(text="", model=model, error="No API key provided")

    start = time.time()
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1024,
        }
    ).encode()

    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode())
            latency = time.time() - start

            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            # 成本估算
            cost = 0.0
            if "gpt-4o" in model:
                cost = prompt_tokens * 2.5e-6 + completion_tokens * 1e-5
            elif "gpt-4" in model:
                cost = prompt_tokens * 3e-5 + completion_tokens * 6e-5
            elif "claude" in model:
                cost = prompt_tokens * 3e-6 + completion_tokens * 1.5e-5

            return LLMCallResult(
                text=content,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost,
                latency_seconds=latency,
            )
    except Exception as e:
        return LLMCallResult(
            text="",
            model=model,
            error=str(e),
            latency_seconds=time.time() - start,
        )


# ─────────────────────────────────────────────────────────────────────────────
# LLM Benchmark Runner
# ─────────────────────────────────────────────────────────────────────────────


class LLMBaselineRunner:
    """
    LLM 因果推理基线运行器。

    Example:
        >>> runner = LLMBaselineRunner(backend="ollama", model="qwen3:8b")
        >>> patients = bench.load_synthetic_dataset()
        >>> result = runner.run_benchmark(patients)
        >>> print(result.metrics.to_dict())
    """

    def __init__(
        self,
        backend: str = "ollama",
        model: str = "qwen3:8b",
        api_key: str = "",
        base_url: str = "",
        timeout: int = 60,
    ):
        self.backend = backend
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    def _call_llm(self, prompt: str) -> LLMCallResult:
        if self.backend == "ollama":
            url = self.base_url or "http://localhost:11434"
            return call_ollama(prompt, model=self.model, base_url=url, timeout=self.timeout)
        elif self.backend in ("openai", "api2d", "custom"):
            url = self.base_url or "https://api.openai.com/v1"
            return call_openai_api(
                prompt,
                model=self.model,
                api_key=self.api_key,
                base_url=url,
                timeout=self.timeout,
            )
        else:
            return LLMCallResult(text="", model=self.model, error=f"Unknown backend: {self.backend}")

    def run_inference(self, patient: PatientTimeline) -> list[tuple]:
        """对单个患者运行 LLM 因果推理。"""
        data_summary = patient.data_summary(max_rows=20)
        prompt = CAUSAL_DISCOVERY_PROMPT.format(data_summary=data_summary)

        result = self._call_llm(prompt)

        if result.error:
            logger.warning("LLM 调用失败: %s", result.error)
            return []

        return parse_llm_causal_output(result.text)

    def run_benchmark(
        self,
        patients: list[PatientTimeline],
        benchmark: MIMICCausalBenchmark | None = None,
    ) -> BenchmarkResult:
        """在患者数据集上运行 LLM 基线 Benchmark。"""
        if benchmark is None:
            benchmark = MIMICCausalBenchmark()

        start_time = time.time()
        total_cost = 0.0
        all_edges: list[tuple] = []
        per_patient = []

        for patient in patients:
            edges = self.run_inference(patient)
            all_edges.extend(edges)

            # 去重
            seen = set()
            unique_edges = []
            for e in edges:
                key = (e[0], e[1])
                if key not in seen:
                    seen.add(key)
                    unique_edges.append(e)

            pm = benchmark.compare_graphs(unique_edges)
            per_patient.append(pm)

        agg = benchmark.compare_graphs(all_edges)
        elapsed = time.time() - start_time

        return BenchmarkResult(
            method="llm",
            model_name=self.model,
            n_patients=len(patients),
            metrics=agg,
            per_patient_metrics=per_patient,
            runtime_seconds=elapsed,
            cost_usd=total_cost,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
