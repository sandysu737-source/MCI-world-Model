"""MCI World Model HTTP API 服务。

零第三方依赖, 使用 Python 标准库 http.server。
生产环境建议前置 Nginx/Traefik 做 TLS 终止和负载均衡。

端点:
    GET  /health     — liveness (进程存活)
    GET  /ready      — readiness (模型加载完成)
    POST /api/v1/diagnose         — 医疗因果诊断
    POST /api/v1/backdoor         — 后门调整 ATE
    POST /api/v1/counterfactual   — 反事实推断
    POST /api/v1/energy/what_if   — 能量干预预测
"""
from __future__ import annotations

import json
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

from mci_world_model._logging import setup_logging

logger = logging.getLogger(__name__)

# 全局引擎实例 (线程安全)
_ready = False


def _init_engines():
    """延迟初始化引擎。"""
    global _ready
    try:
        # 验证核心模块可导入
        import mci_world_model  # noqa: F401
        _ready = True
        logger.info("MCI 引擎初始化完成")
    except Exception as e:
        logger.error("MCI 引擎初始化失败: %s", e, exc_info=True)
        _ready = False


class MCIAPIHandler(BaseHTTPRequestHandler):
    """HTTP 请求处理器。"""

    # 请求计数 (简单指标)
    _request_count = 0
    _request_count_lock = threading.Lock()

    def log_message(self, format: str, *args: Any) -> None:
        """用 logging 替代默认 stderr 输出。"""
        logger.info("%s - %s", self.address_string(), format % args)

    def _send_json(self, status: int, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw)

    def do_GET(self) -> None:
        with self._request_count_lock:
            self._request_count += 1

        if self.path == "/health":
            self._send_json(200, {"status": "alive", "timestamp": time.time()})
        elif self.path == "/ready":
            self._send_json(200, {"ready": _ready})
        elif self.path == "/metrics":
            with self._request_count_lock:
                count = self._request_count
            self._send_json(200, {
                "requests_total": count,
                "ready": _ready,
                "metrics_format": "simple",
            })
        else:
            self._send_json(404, {"error": "not found", "path": self.path})

    def do_POST(self) -> None:
        with self._request_count_lock:
            self._request_count += 1

        if not _ready:
            self._send_json(503, {"error": "not ready"})
            return

        try:
            body = self._read_body()
        except json.JSONDecodeError as e:
            self._send_json(400, {"error": "invalid JSON", "detail": str(e)})
            return

        try:
            if self.path == "/api/v1/diagnose":
                result = self._handle_diagnose(body)
                self._send_json(200, result)
            elif self.path == "/api/v1/backdoor":
                result = self._handle_backdoor(body)
                self._send_json(200, result)
            elif self.path == "/api/v1/energy/what_if":
                result = self._handle_energy_whatif(body)
                self._send_json(200, result)
            else:
                self._send_json(404, {"error": "not found", "path": self.path})
        except Exception as e:
            logger.error("API 错误 %s: %s", self.path, e, exc_info=True)
            self._send_json(500, {"error": "internal error", "detail": str(e)})

    # ── 端点处理 ──

    def _handle_diagnose(self, body: dict) -> dict:
        from mci_world_model.sdk._medical_causal_sdk import (
            MedicalCausalSDK,
            ClinicalEvidence,
        )

        sdk = MedicalCausalSDK(patient_id=body.get("patient_id", ""))
        for ev in body.get("evidence", []):
            sdk.add_evidence(ClinicalEvidence(
                evidence_id=ev.get("id", ""),
                evidence_type=ev.get("type", "observation"),
                description=ev.get("description", ""),
                confidence=ev.get("confidence", 0.5),
            ))
        cause = body.get("cause", "")
        effect = body.get("effect", "")
        prior = body.get("prior_strength", 0.5)
        diag = sdk.diagnose(cause, effect, prior)
        return {
            "cause": diag.cause,
            "effect": diag.effect,
            "confidence": diag.confidence,
            "is_conclusive": diag.is_conclusive,
            "warnings": diag.warnings,
        }

    def _handle_backdoor(self, body: dict) -> dict:
        from mci_world_model.sdk._do_calculus import DoCalculus, CausalGraph
        import numpy as np

        cg = CausalGraph()
        for src, dst in body.get("edges", []):
            cg.add_edge(src, dst)
        dc = DoCalculus(graph=cg)

        data = {}
        for node, values in body.get("data", {}).items():
            data[node] = np.array(values, dtype=np.float64)
        if data:
            dc.set_data(data)

        x = body.get("treatment", "")
        y = body.get("outcome", "")
        z_set = body.get("adjustment_set", [])
        result = dc.backdoor_adjustment(x, y, Z_set=z_set)
        return {
            "ate": result.ate,
            "method": result.method,
            "treatment": x,
            "outcome": y,
        }

    def _handle_energy_whatif(self, body: dict) -> dict:
        from mci_world_model._sys._energy_core import EnergyCore
        from mci_world_model.sdk._energy_counterfactual_bridge import (
            EnergyCounterfactualBridge,
        )

        bridge = EnergyCounterfactualBridge(EnergyCore())
        target = body.get("target_energy", "semantic")
        boost = body.get("boost", 1.0)
        results = bridge.what_if(target, boost=boost)
        # what_if 返回 list[EnergyWhatIfResult]
        if not isinstance(results, list):
            results = [results]
        return {
            "results": [
                {
                    "target_energy": r.target_energy,
                    "baseline": r.baseline,
                    "counterfactual": r.counterfactual,
                    "delta": r.delta,
                }
                for r in results
            ],
        }


def create_server(host: str = "0.0.0.0", port: int = 8080) -> HTTPServer:
    """创建 HTTP 服务器。"""
    setup_logging()
    _init_engines()
    server = HTTPServer((host, port), MCIAPIHandler)
    server.timeout = 1
    logger.info("MCI API 服务启动: %s:%d", host, port)
    return server


def run(host: str = "0.0.0.0", port: int = 8080) -> None:
    """启动服务 (阻塞)。"""
    server = create_server(host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("收到中断信号, 关闭服务")
        server.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MCI World Model API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    run(args.host, args.port)
