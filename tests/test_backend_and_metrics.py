"""P2: 计算后端抽象层 + 指标系统测试。"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.contract


class TestBackendAbstraction:
    """计算后端抽象层。"""

    def test_numpy_backend_basic_ops(self):
        from mci_world_model._backend import B
        A = B.array([[1, 2], [3, 4]])
        x = B.array([1.0, 0.0])
        assert B.name == "numpy"
        assert B.is_gpu is False
        assert np.allclose(B.matmul(A, x), [1, 3])
        assert np.allclose(B.norm(x), 1.0)

    def test_backend_consistency_with_numpy(self):
        """后端结果应与直接用 numpy 一致。"""
        from mci_world_model._backend import B
        A = np.array([[4, 2], [1, 3]], dtype=np.float64)
        b = np.array([1.0, 2.0])
        # solve
        x = B.solve(A, b)
        assert np.allclose(A @ x, b)
        # inv
        A_inv = B.inv(A)
        assert np.allclose(A @ A_inv, np.eye(2))
        # eigh
        sym = np.array([[2, 1], [1, 3]], dtype=np.float64)
        evals = B.eigvalsh(sym)
        assert evals[0] <= evals[1]

    def test_backend_env_override(self):
        """MCI_BACKEND=numpy 强制 CPU。"""
        import os
        os.environ["MCI_BACKEND"] = "numpy"
        from mci_world_model._backend.selector import get_backend
        b = get_backend()
        assert b.name == "numpy"
        del os.environ["MCI_BACKEND"]


class TestMetricsSystem:
    """Prometheus 兼容指标。"""

    def test_counter_and_histogram(self):
        from mci_world_model.server.metrics import MetricsCollector
        m = MetricsCollector()
        m.inc_request("diagnose")
        m.inc_request("diagnose")
        m.inc_error("diagnose")
        m.observe_latency("diagnose", 0.003)
        m.observe_latency("diagnose", 0.02)
        out = m.expose()
        assert 'requests_total{endpoint="diagnose"} 2.0' in out
        assert 'errors_total{endpoint="diagnose"} 1.0' in out
        assert "request_duration_count" in out
        assert "request_duration_sum" in out

    def test_prometheus_format_valid(self):
        """输出应符合 Prometheus text exposition 格式。"""
        from mci_world_model.server.metrics import MetricsCollector
        m = MetricsCollector()
        m.observe_latency("backdoor", 0.1)
        out = m.expose()
        # 每行要么是注释(#), 要么是 name value
        for line in out.strip().split('\n'):
            if line.startswith('#'):
                continue
            assert ' ' in line, f"无效格式: {line}"
