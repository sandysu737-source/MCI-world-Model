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
import os
import signal as signal_module
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from mci_world_model._logging import setup_logging
from mci_world_model.server.metrics import metrics
from mci_world_model.server.security import (
    get_circuit_breaker,
    get_rate_limiter,
    verify_auth,
)
from mci_world_model.server.storage import list_diagnoses, load_diagnosis, save_diagnosis

logger = logging.getLogger(__name__)

# 全局引擎实例 (线程安全)
_ready = False

# ── 背压控制 ──
# 限制并发请求数, 防止高并发下内存/线程耗尽
_MAX_CONCURRENT = int(os.environ.get("MCI_MAX_CONCURRENT", "50"))
_concurrent_semaphore = threading.BoundedSemaphore(_MAX_CONCURRENT)
# 请求计数 (用于背压指标)
_active_requests = 0
_active_requests_lock = threading.Lock()


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

    MAX_BODY_SIZE = 10 * 1024 * 1024  # 10MB

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        # D12 修复: 限制请求体大小, 防止 DoS
        if length > self.MAX_BODY_SIZE:
            raise ValueError(f"请求体过大: {length} > {self.MAX_BODY_SIZE}")
        raw = self.rfile.read(length)
        return json.loads(raw)

    # 健康检查端点不需要认证
    _PUBLIC_ENDPOINTS = {"/health", "/ready", "/metrics"}

    def _check_security(self) -> bool:
        """认证 + 限流 + 断路器检查。返回 True 表示通过。"""
        # 公开端点跳过认证 (但仍限流)
        if self.path not in self._PUBLIC_ENDPOINTS:
            from mci_world_model.server.security import AuthConfig

            auth_cfg = AuthConfig.from_env()
            hdrs = dict(self.headers)
            logger.debug("AUTH CHECK: keys=%s, headers=%s, path=%s", auth_cfg.api_keys, list(hdrs.keys()), self.path)
            if not verify_auth(hdrs, auth_cfg):
                logger.warning("AUTH FAILED: headers=%s, configured_keys=%s", list(hdrs.keys()), auth_cfg.api_keys)
                self._send_json(401, {"error": "unauthorized"})
                return False

        # 限流
        client_ip = self.client_address[0] if self.client_address else "unknown"
        if not get_rate_limiter().allow(client_ip):
            self._send_json(429, {"error": "rate limit exceeded"})
            return False

        return True

    def do_GET(self) -> None:
        if not self._check_security():
            return
        t0 = time.time()
        endpoint = self.path
        trace_id = uuid.uuid4().hex[:12]
        logger.debug("GET %s [trace=%s]", endpoint, trace_id)

        if self.path == "/health":
            self._send_json(200, {"status": "alive", "timestamp": time.time()})
        elif self.path == "/ready":
            # 增强就绪探针: 检查引擎真正可用 (不只是 import 成功)
            ready = _ready
            checks = {"engine_loaded": ready}
            if ready:
                try:
                    from mci_world_model.sdk._medical_causal_sdk import MedicalCausalSDK

                    MedicalCausalSDK()
                    checks["sdk_instantiable"] = True
                except Exception:
                    checks["sdk_instantiable"] = False
                    ready = False
            self._send_json(200 if ready else 503, {"ready": ready, "checks": checks})
        elif self.path == "/metrics":
            # Prometheus text exposition format
            body = metrics.expose().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/v1/diagnosis/"):
            # GET /api/v1/diagnosis/{record_id} — 查询已保存的诊断
            record_id = self.path.split("/api/v1/diagnosis/", 1)[1].split("?")[0]
            if not record_id:
                self._send_json(400, {"error": "missing record_id"})
            else:
                record = load_diagnosis(record_id)
                if record is not None:
                    self._send_json(200, record)
                else:
                    self._send_json(404, {"error": "diagnosis not found", "record_id": record_id})
        elif self.path.startswith("/api/v1/diagnoses"):
            # GET /api/v1/diagnoses?patient_id=P001 — 列出诊断记录
            from urllib.parse import parse_qs, urlparse

            query = parse_qs(urlparse(self.path).query)
            patient_id = query.get("patient_id", [""])[0]
            keys = list_diagnoses(patient_id)
            self._send_json(200, {"count": len(keys), "record_ids": keys})
        else:
            self._send_json(404, {"error": "not found", "path": self.path})

        metrics.inc_request(endpoint)
        metrics.observe_latency(endpoint, time.time() - t0)

    def do_POST(self) -> None:
        if not self._check_security():
            return
        t0 = time.time()
        endpoint = self.path
        trace_id = uuid.uuid4().hex[:12]

        # 背压: 并发数超限时拒绝 (503 backlog full)
        global _active_requests
        if not _concurrent_semaphore.acquire(blocking=False):
            metrics.inc_error(endpoint)
            self._send_json(
                503,
                {
                    "error": "server busy",
                    "detail": "concurrent request limit reached, retry later",
                    "trace_id": trace_id,
                },
            )
            return

        with _active_requests_lock:
            _active_requests += 1
            metrics.set_gauge("mci_active_requests", float(_active_requests))

        try:
            cb = get_circuit_breaker()
            if not cb.allow():
                self._send_json(503, {"error": "circuit breaker open", "trace_id": trace_id})
                return

            if not _ready:
                metrics.inc_error(endpoint)
                self._send_json(503, {"error": "not ready", "trace_id": trace_id})
                return

            try:
                body = self._read_body()
            except json.JSONDecodeError as e:
                metrics.inc_error(endpoint)
                self._send_json(400, {"error": "invalid JSON", "detail": str(e), "trace_id": trace_id})
                return
            except ValueError as e:
                metrics.inc_error(endpoint)
                self._send_json(400, {"error": "bad request", "detail": str(e), "trace_id": trace_id})
                return

            try:
                if self.path == "/api/v1/diagnose":
                    result = self._handle_diagnose(body)
                    self._send_json(200, result)
                elif self.path == "/api/v1/diagnose/batch":
                    result = self._handle_batch_diagnose(body)
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
                logger.error("API 错误 %s [trace=%s]: %s", self.path, trace_id, e, exc_info=True)
                metrics.inc_error(endpoint)
                cb.record_failure()
                self._send_json(500, {"error": "internal error", "detail": str(e), "trace_id": trace_id})
            else:
                cb.record_success()

            metrics.inc_request(endpoint)
            metrics.observe_latency(endpoint, time.time() - t0)
        finally:
            _concurrent_semaphore.release()
            with _active_requests_lock:
                _active_requests -= 1
                metrics.set_gauge("mci_active_requests", float(_active_requests))

    def _handle_diagnose(self, body: dict) -> dict:
        from mci_world_model.sdk._medical_causal_sdk import (
            ClinicalEvidence,
            MedicalCausalSDK,
        )

        # D15 修复: cause/effect 非空校验
        cause = body.get("cause", "").strip()
        effect = body.get("effect", "").strip()
        if not cause or not effect:
            raise ValueError("cause 和 effect 不能为空")

        sdk = MedicalCausalSDK(patient_id=body.get("patient_id", ""))
        for ev in body.get("evidence", [])[: MedicalCausalSDK.MAX_EVIDENCE_COUNT]:
            sdk.add_evidence(
                ClinicalEvidence(
                    evidence_id=ev.get("id", ""),
                    evidence_type=ev.get("type", "observation"),
                    description=ev.get("description", ""),
                    confidence=ev.get("confidence", 0.5),
                )
            )
        prior = body.get("prior_strength", 0.5)
        diag = sdk.diagnose(cause, effect, prior)
        result = {
            "cause": diag.cause,
            "effect": diag.effect,
            "confidence": diag.confidence,
            "is_conclusive": diag.is_conclusive,
            "warnings": diag.warnings,
        }
        # 持久化诊断结果 (状态外置化, 支持多副本共享)
        patient_id = body.get("patient_id", "unknown")
        try:
            record_id = save_diagnosis(patient_id, result)
            result["record_id"] = record_id
        except Exception:
            logger.warning("诊断结果持久化失败, 不影响响应", exc_info=True)
        return result

    def _handle_batch_diagnose(self, body: dict) -> dict:
        """批量诊断端点。"""
        from mci_world_model.sdk._medical_causal_sdk import MedicalCausalSDK

        sdk = MedicalCausalSDK(patient_id=body.get("patient_id", ""))
        queries = body.get("queries", [])
        results = sdk.batch_diagnose(queries)
        return {
            "count": len(results),
            "diagnoses": [
                {
                    "cause": r.cause,
                    "effect": r.effect,
                    "confidence": r.confidence,
                    "is_conclusive": r.is_conclusive,
                }
                for r in results
            ],
        }

    def _handle_backdoor(self, body: dict) -> dict:
        import numpy as np

        from mci_world_model.sdk._do_calculus import CausalGraph, DoCalculus

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


def create_server(host: str = "0.0.0.0", port: int = 8080) -> ThreadingHTTPServer:
    """创建 HTTP 服务器。"""
    setup_logging()
    _init_engines()
    server = ThreadingHTTPServer((host, port), MCIAPIHandler)
    server.timeout = 1
    server.daemon_threads = True
    logger.info("MCI API 服务启动: %s:%d", host, port)
    return server


def run(host: str = "0.0.0.0", port: int = 8080) -> None:
    """启动服务 (阻塞), 支持 SIGTERM 优雅关闭。"""
    server = create_server(host, port)

    def _graceful_shutdown(signum, frame):
        sig_name = signal_module.Signals(signum).name
        logger.info("收到 %s 信号, 开始优雅关闭...", sig_name)
        server.shutdown()

    signal_module.signal(signal_module.SIGTERM, _graceful_shutdown)
    signal_module.signal(signal_module.SIGINT, _graceful_shutdown)

    metrics.set_gauge("mci_ready", 1.0 if _ready else 0.0)
    metrics.set_gauge("mci_active_requests", 0.0)
    logger.info("MCI API 服务就绪: %s:%d (SIGTERM 优雅关闭已启用)", host, port)
    try:
        server.serve_forever()
    finally:
        metrics.set_gauge("mci_ready", 0.0)
        logger.info("MCI API 服务已关闭")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="MCI World Model API Server")
    parser.add_argument("--host", default=os.environ.get("MCI_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MCI_PORT", "8080")))
    args = parser.parse_args()
    run(args.host, args.port)
