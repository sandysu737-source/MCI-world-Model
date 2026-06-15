"""Cladder 数据集加载与解析模块。

从 HuggingFace 加载 causalnlp/CLadder v1.5 数据集并缓存到本地。
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# =============================================================================
# 数据结构
# =============================================================================

QUERY_TYPE_BY_RUNG: dict[int, list[str]] = {
    1: ["correlation", "marginal", "exp_away"],
    2: ["ate", "backadj", "collider_bias"],
    3: ["ett", "nde", "nie", "det-counterfactual"],
}

GRAPH_TYPES = [
    "IV",
    "arrowhead",
    "chain",
    "collision",
    "confounding",
    "diamond",
    "diamondcut",
    "fork",
    "frontdoor",
    "mediation",
]


@dataclass
class CladderQuestion:
    """单个 Cladder 因果推理问题。"""

    question_id: int
    rung: int
    query_type: str
    graph_id: str
    label: str  # "yes" / "no"
    prompt: str = ""
    reasoning: str = ""
    formal_form: str = ""
    question_property: str = ""

    # 从 reasoning 字段解析的结构化信息
    edges: list[tuple[str, str]] = field(default_factory=list)
    var_names: dict[str, str] = field(default_factory=dict)
    probabilities: dict[str, float] = field(default_factory=dict)
    solution_value: float | None = None  # 从 reasoning 解析的计算结果

    @property
    def label_bool(self) -> bool:
        return self.label == "yes"

    @property
    def solution_sign(self) -> str:
        """计算结果的符号方向 (> 0 或 < 0)。"""
        if self.solution_value is None:
            return "unknown"
        if self.solution_value > 0:
            return ">"
        if self.solution_value < 0:
            return "<"
        return "="


# =============================================================================
# 数据集加载
# =============================================================================

_CLADDER_CACHE: list[CladderQuestion] | None = None
_CLADDER_CACHE_FILE = os.path.join(os.path.dirname(__file__), "data", "cladder_parsed.json")


def load_cladder(use_cache: bool = True) -> list[CladderQuestion]:
    """加载完整的 Cladder v1.5 数据集。

    Args:
        use_cache: 是否使用本地缓存

    Returns:
        10112 道 CladderQuestion
    """
    global _CLADDER_CACHE

    if _CLADDER_CACHE is not None:
        return _CLADDER_CACHE

    # 尝试从缓存加载
    if use_cache and os.path.exists(_CLADDER_CACHE_FILE):
        try:
            with open(_CLADDER_CACHE_FILE) as f:
                raw = json.load(f)
            _CLADDER_CACHE = [CladderQuestion(**d) for d in raw]
            logger.info("从缓存加载 %d 道 Cladder 题", len(_CLADDER_CACHE))
            return _CLADDER_CACHE
        except Exception as e:
            logger.warning("缓存加载失败: %s", e)

    # 从 HuggingFace 加载
    from datasets import load_dataset as hf_load

    ds = hf_load("causalnlp/CLadder", split="full_v1.5_default")

    questions: list[CladderQuestion] = []
    for row in ds:
        q = CladderQuestion(
            question_id=int(row["id"]),
            rung=int(row["rung"]),
            query_type=str(row["query_type"]),
            graph_id=str(row["graph_id"]),
            label=str(row["label"]),
            prompt=str(row.get("prompt", "")),
            reasoning=str(row.get("reasoning", "")),
            formal_form=str(row.get("formal_form", "")),
            question_property=str(row.get("question_property", "")),
        )
        # 解析 reasoning 字段
        _parse_reasoning(q)
        questions.append(q)

    _CLADDER_CACHE = questions

    # 写入缓存
    os.makedirs(os.path.dirname(_CLADDER_CACHE_FILE), exist_ok=True)
    try:
        with open(_CLADDER_CACHE_FILE, "w") as f:
            json.dump(
                [
                    {
                        "question_id": q.question_id,
                        "rung": q.rung,
                        "query_type": q.query_type,
                        "graph_id": q.graph_id,
                        "label": q.label,
                        "prompt": q.prompt,
                        "reasoning": q.reasoning,
                        "formal_form": q.formal_form,
                        "question_property": q.question_property,
                        "edges": q.edges,
                        "var_names": q.var_names,
                        "probabilities": q.probabilities,
                        "solution_value": q.solution_value,
                    }
                    for q in questions
                ],
                f,
                indent=2,
            )
        logger.info("Cladder 数据集已缓存: %d 道题", len(questions))
    except Exception as e:
        logger.warning("缓存写入失败: %s", e)

    return questions


# =============================================================================
# Reasoning 解析
# =============================================================================

_EDGE_PATTERN = re.compile(r"(\w+(?:\s+\w+)?\s*(?:<|)-{1,2}>?\s*(\w+(?:\s+\w+)?))")

_PROB_PATTERN = re.compile(r"P\s*\(\s*([\w\s|,=]+?)\s*\)\s*=\s*([\d.]+)")

_SOL_PATTERN = re.compile(r"([\d.+\-*/() ]+)\s*=\s*([\d.-]+)\s*$")

_VAR_PATTERN = re.compile(r"Let\s+(\w+)\s*=\s*([^;]+?);?")


def _parse_reasoning(q: CladderQuestion) -> None:
    """从 reasoning 字段提取结构化因果信息。

    提取:
    - 变量名映射: "Let X = treatment; V2 = blood pressure; Y = recovery."
    - 因果边: "X->V2,X->Y,V2->Y"
    - 概率表: "P(Y=1 | X=0, V2=0) = 0.08"
    - 计算结果: "0.74 * (0.86 - 0.41) = 0.32" → 0.32
    """
    reasoning = q.reasoning
    if not reasoning or reasoning == "nan":
        return

    lines = reasoning.split("\n")

    # ── 解析变量定义 ──
    var_part = lines[0] if lines else ""
    for m in _VAR_PATTERN.finditer(var_part):
        var_name = m.group(1).strip()
        var_desc = m.group(2).strip()
        q.var_names[var_name] = var_desc

    # ── 解析边 ──
    for line in lines:
        # 查找包含 "->" 的行
        if "->" in line and "P(" not in line:
            # 按逗号分割边声明
            edge_str = line.split("<")[0] if "<" in line else line
            parts = edge_str.replace(" ", "").split(",")
            for part in parts:
                if "->" in part and len(part) >= 3:
                    nodes = part.split("->")
                    if len(nodes) == 2 and nodes[0] and nodes[1]:
                        edge = (nodes[0].strip(), nodes[1].strip())
                        if edge not in q.edges:
                            q.edges.append(edge)
            break  # 只取第一条边声明

    # ── 解析概率表 ──
    for line in lines:
        for m in _PROB_PATTERN.finditer(line):
            cond = m.group(1).strip()
            prob_val = float(m.group(2))
            q.probabilities[cond] = prob_val

    # ── 解析最终计算结果 ──
    for line in reversed(lines):
        line = line.strip()
        if not line or line.startswith("Let"):
            continue
        # 查找形如 "X.XX = Y.YY" 的行 或 最后一行的 "= value"
        eq_match = re.search(r"=\s*([\d.-]+)\s*$", line)
        if eq_match:
            val_str = eq_match.group(1)
            try:
                # 排除概率表行
                if "P(" not in line:
                    q.solution_value = float(val_str)
                    break
            except ValueError:
                pass


# =============================================================================
# 统计工具
# =============================================================================


def dataset_stats(
    questions: list[CladderQuestion] | None = None,
) -> dict:
    """计算数据集统计信息。"""
    if questions is None:
        questions = load_cladder()

    rung_dist = Counter(q.rung for q in questions)
    qt_dist = Counter(q.query_type for q in questions)
    graph_dist = Counter(q.graph_id for q in questions)
    label_dist = Counter(q.label for q in questions)
    prop_dist = Counter(q.question_property for q in questions)
    has_reasoning = sum(1 for q in questions if q.reasoning and q.reasoning != "nan")
    has_edges = sum(1 for q in questions if q.edges)

    return {
        "n_total": len(questions),
        "rung_distribution": dict(sorted(rung_dist.items())),
        "query_type_distribution": dict(sorted(qt_dist.items())),
        "graph_type_distribution": dict(sorted(graph_dist.items())),
        "label_distribution": dict(label_dist),
        "question_property_distribution": dict(prop_dist),
        "with_reasoning": has_reasoning,
        "with_parsed_edges": has_edges,
    }
