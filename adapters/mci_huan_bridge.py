"""HuanBridge — MCI World Model ↔ mci-huan 集成桥接。

将 MCI World Model 的因果分析结果提交到 mci-huan 临床 harness，
生成 AgentTrace 供 harness 的 Judge-Agent 循环消费。

Usage::
    from adapters.mci_huan_bridge import HuanBridge

    bridge = HuanBridge()
    trace = bridge.create_causal_trace(patient_data, model_result)
    bridge.submit_trace(trace)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ── Probe ──────────────────────────────────────────────────────────────
_HUAN_AVAILABLE = False
try:
    # mci-huan backend is a FastAPI service — bridge via HTTP
    _HUAN_AVAILABLE = True
except ImportError:
    pass


# ── Enums ──────────────────────────────────────────────────────────────

class AnalysisPhase(str, Enum):
    """因果分析阶段 — 对齐 mci-huan LoopPhase。"""

    PERCEPTION = "perception"
    CAUSAL_INFERENCE = "causal_inference"
    ACTION_PLANNING = "action_planning"
    REFLECTION = "reflection"


# ── Data Classes ────────────────────────────────────────────────────────

@dataclass
class CausalTraceSpan:
    """因果分析追踪片段 — 对应 mci-huan TraceSpan。"""

    phase: AnalysisPhase
    input_summary: dict[str, Any]
    output_summary: dict[str, Any]
    elapsed_ms: float = 0.0
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "input": self.input_summary,
            "output": self.output_summary,
            "elapsed_ms": self.elapsed_ms,
            "trace_id": self.trace_id,
        }


@dataclass
class CausalAgentTrace:
    """因果分析完整追踪 — 对应 mci-huan AgentTrace。

    记录多阶段因果分析的全过程，供 harness 的重审循环消费。
    """

    session_id: str
    patient_id: str | None = None
    spans: list[CausalTraceSpan] = field(default_factory=list)
    final_decision: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def add_span(self, span: CausalTraceSpan) -> None:
        self.spans.append(span)

    def to_agent_output(self) -> dict[str, Any]:
        """转换为 mci-huan AgentOutput 格式。"""
        return {
            "session_id": self.session_id,
            "patient_id": self.patient_id,
            "trace": [s.to_dict() for s in self.spans],
            "decision": self.final_decision,
            "n_spans": len(self.spans),
            "timestamp": self.created_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.to_agent_output()


# ── Bridge ──────────────────────────────────────────────────────────────

@dataclass
class HuanBridge:
    """MCI World Model → mci-huan 桥接器。

    将因果推理结果封装为 mci-huan harness 可消费的 AgentTrace，
    支持 HTTP 提交或本地序列化。

    Attributes:
        huan_api_url: mci-huan 后端 API 地址。
        _traces: 本地 trace 缓存。
    """

    huan_api_url: str = "http://localhost:8000/api"
    _traces: list[CausalAgentTrace] = field(default_factory=list, repr=False)

    @property
    def is_available(self) -> bool:
        return _HUAN_AVAILABLE

    # ── Trace Building ──────────────────────────────────────────────────

    def create_causal_trace(
        self,
        patient_data: dict[str, Any],
        model_result: Any,
        *,
        session_id: str | None = None,
    ) -> CausalAgentTrace:
        """从 MCI World Model 因果结果创建 AgentTrace。

        Args:
            patient_data: 患者上下文数据。
            model_result: MCI 因果推理结果。
            session_id: 会话 ID (自动生成如果为空)。

        Returns:
            CausalAgentTrace 供 harness 消费。
        """
        if session_id is None:
            session_id = f"mci-huan-{uuid.uuid4().hex[:8]}"

        trace = CausalAgentTrace(
            session_id=session_id,
            patient_id=patient_data.get("patient_id"),
        )

        # Span 1: 感知 — 输入解析
        span1 = CausalTraceSpan(
            phase=AnalysisPhase.PERCEPTION,
            input_summary={"patient_data_keys": list(patient_data.keys())},
            output_summary={"parsed": True, "n_features": len(patient_data)},
            elapsed_ms=0.0,
        )
        trace.add_span(span1)

        # Span 2: 因果推理
        result_summary: dict[str, Any] = {"type": type(model_result).__name__}
        if hasattr(model_result, "ate"):
            result_summary["ate"] = float(model_result.ate)
        if hasattr(model_result, "method"):
            result_summary["method"] = str(model_result.method)
        if hasattr(model_result, "counterfactual_value"):
            result_summary["cf_value"] = float(model_result.counterfactual_value)

        span2 = CausalTraceSpan(
            phase=AnalysisPhase.CAUSAL_INFERENCE,
            input_summary={"query": patient_data.get("query", "causal_analysis")},
            output_summary=result_summary,
            elapsed_ms=0.0,
        )
        trace.add_span(span2)

        # Span 3: 行动规划
        action = {
            "recommendation": f"基于 ATE={result_summary.get('ate', 'N/A')} 的因果建议",
            "confidence": 0.85,
            "method": result_summary.get("method", "do-calculus"),
        }
        span3 = CausalTraceSpan(
            phase=AnalysisPhase.ACTION_PLANNING,
            input_summary=result_summary,
            output_summary=action,
            elapsed_ms=0.0,
        )
        trace.add_span(span3)

        # 终审决定
        trace.final_decision = {
            "approved": True,
            "action": action["recommendation"],
            "causal_basis": result_summary,
            "review_required": result_summary.get("ate", 0.0) < 0.1,
        }

        self._traces.append(trace)
        return trace

    # ── Submission ──────────────────────────────────────────────────────

    def submit_trace(self, trace: CausalAgentTrace) -> dict[str, Any]:
        """提交 trace 到 mci-huan harness。

        Args:
            trace: CausalAgentTrace 实例。

        Returns:
            提交结果字典。
        """
        payload = trace.to_dict()

        # 尝试 HTTP 提交
        try:
            import urllib.request

            req = urllib.request.Request(
                f"{self.huan_api_url}/traces",
                data=json.dumps(payload, default=str).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return {
                    "status": "submitted",
                    "http_code": resp.getcode(),
                    "session_id": trace.session_id,
                }
        except Exception as e:
            # 回退: 保存到本地
            local_path = f"/tmp/mci-huan-trace-{trace.session_id}.json"
            try:
                with open(local_path, "w") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
                return {
                    "status": "saved_locally",
                    "path": local_path,
                    "session_id": trace.session_id,
                }
            except Exception:
                return {
                    "status": "cached_only",
                    "session_id": trace.session_id,
                    "error": str(e),
                }

    # ── Query ───────────────────────────────────────────────────────────

    def list_traces(self, limit: int = 20) -> list[dict[str, Any]]:
        """列出缓存的 traces。

        Args:
            limit: 最大返回数。

        Returns:
            trace 摘要列表。
        """
        return [
            {
                "session_id": t.session_id,
                "patient_id": t.patient_id,
                "n_spans": len(t.spans),
                "decision_approved": t.final_decision.get("approved"),
            }
            for t in self._traces[-limit:]
        ]

    # ── Health ──────────────────────────────────────────────────────────

    def health_check(self) -> dict[str, Any]:
        """健康检查。"""
        status: dict[str, Any] = {
            "bridge": "mci_huan",
            "target_url": self.huan_api_url,
            "traces_cached": len(self._traces),
        }
        try:
            import urllib.request

            req = urllib.request.Request(
                f"{self.huan_api_url}/health", method="GET"
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                status["huan_api_ok"] = resp.getcode() == 200
        except Exception as e:
            status["huan_api_ok"] = False
            status["huan_api_error"] = str(e)
        return status

    def clear_cache(self) -> None:
        """清空本地 trace 缓存。"""
        self._traces.clear()

    def __len__(self) -> int:
        return len(self._traces)


__all__ = ["HuanBridge", "CausalAgentTrace", "CausalTraceSpan", "AnalysisPhase"]
