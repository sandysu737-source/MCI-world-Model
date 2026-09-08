"""HTTP JSON 请求与响应的安全校验。"""

from __future__ import annotations

import json
import math
import numbers
from typing import Any

DEFAULT_MAX_JSON_DEPTH = 64


class RequestLimitError(ValueError):
    """请求规模超过显式预算。"""


def reject_json_constant(value: str) -> None:
    """拒绝 JSON 规范外的 NaN 与 Infinity 常量。"""
    raise ValueError(f"非标准 JSON 常量: {value}")


def assert_json_depth(raw: bytes | str, max_depth: int = DEFAULT_MAX_JSON_DEPTH) -> None:
    """迭代式扫描 JSON 结构深度，避免递归解析造成的无响应。"""
    if max_depth < 1:
        raise ValueError("max_depth 必须大于 0")

    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    depth = 0
    in_string = False
    escaped = False

    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
            if depth > max_depth:
                raise ValueError(f"JSON 嵌套深度超过 {max_depth}")
        elif char in "}]":
            depth = max(0, depth - 1)


def assert_finite_tree(value: Any, path: str = "$") -> None:
    """递归校验数值树中没有 NaN 或 Inf。"""
    if isinstance(value, numbers.Real):
        if not math.isfinite(float(value)):
            raise ValueError(f"非有限数值: {path}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_tree(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_tree(item, f"{path}[{index}]")


def loads_safe_json(raw: bytes, max_depth: int = DEFAULT_MAX_JSON_DEPTH) -> dict:
    """解析请求 JSON，并在解析前后应用结构深度与有限性门禁。"""
    assert_json_depth(raw, max_depth)
    value = json.loads(raw.decode("utf-8"), parse_constant=reject_json_constant)
    if not isinstance(value, dict):
        raise ValueError("请求体必须是 JSON 对象")
    assert_finite_tree(value)
    return value


__all__: list[str] = [
    "DEFAULT_MAX_JSON_DEPTH",
    "RequestLimitError",
    "assert_finite_tree",
    "assert_json_depth",
    "loads_safe_json",
    "reject_json_constant",
]
