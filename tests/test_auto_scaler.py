"""
tests/test_auto_scaler.py — AutoScaler 测试
===========================================

覆盖:
    - compute_desired_replicas: 扩缩容决策
    - 扩容/缩容/保持 场景
    - 边界: 零QPS/负值/极值
"""

from __future__ import annotations

import pytest

from mci_world_model.sdk._auto_scaler import AutoScaler, ScaleDecision

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def scaler():
    return AutoScaler(
        min_replicas=1,
        max_replicas=20,
        target_latency_ms=200.0,
        qps_per_replica=100.0,
    )


# =============================================================================
# TestScaleDecision
# =============================================================================


class TestScaleDecision:
    """ScaleDecision 数据类。"""

    def test_creation(self):
        decision = ScaleDecision(
            desired_replicas=5,
            current_replicas=3,
            action="scale_up",
            reason="latency exceeded",
        )
        assert decision.desired_replicas == 5
        assert decision.action == "scale_up"

    def test_hold(self):
        decision = ScaleDecision(
            desired_replicas=3,
            current_replicas=3,
            action="hold",
            reason="within target",
        )
        assert decision.action == "hold"

    def test_with_metrics(self):
        decision = ScaleDecision(
            desired_replicas=5,
            current_replicas=3,
            action="scale_up",
            reason="high latency",
            qps=500.0,
            latency_ms=350.0,
        )
        assert decision.qps == 500.0
        assert decision.latency_ms == 350.0


# =============================================================================
# TestComputeDesiredReplicas
# =============================================================================


class TestComputeDesiredReplicas:
    """扩缩容决策测试。"""

    def test_scale_up_by_latency(self, scaler):
        """延迟超标 → 扩容。"""
        # target=200, 1.5*target=300 → latency=400 > 300 → scale_up
        decision = scaler.compute_desired_replicas(
            current_qps=100.0,
            avg_latency_ms=400.0,
        )
        assert decision.action == "scale_up"
        assert decision.desired_replicas >= 1

    def test_scale_down_by_latency(self, scaler):
        """延迟过低 → 缩容。"""
        # target=200, 0.5*target=100 → latency=50 < 100 → scale_down
        decision = scaler.compute_desired_replicas(
            current_qps=50.0,
            avg_latency_ms=50.0,
        )
        assert decision.action == "scale_down"

    def test_hold(self, scaler):
        """正常范围 → 保持。"""
        # latency=180 在 [100, 300] 范围内 → hold
        decision = scaler.compute_desired_replicas(
            current_qps=100.0,
            avg_latency_ms=180.0,
        )
        assert decision.action == "hold"

    def test_min_replicas_enforced(self, scaler):
        """最小副本数限制。"""
        decision = scaler.compute_desired_replicas(
            current_qps=1.0,
            avg_latency_ms=10.0,
        )
        assert decision.desired_replicas >= 1

    def test_max_replicas_enforced(self, scaler):
        """最大副本数限制。"""
        decision = scaler.compute_desired_replicas(
            current_qps=10000.0,
            avg_latency_ms=5000.0,
        )
        assert decision.desired_replicas <= 20

    def test_zero_qps(self, scaler):
        """零QPS。"""
        decision = scaler.compute_desired_replicas(
            current_qps=0.0,
            avg_latency_ms=0.0,
        )
        assert decision.desired_replicas >= 1

    def test_negative_qps_error(self, scaler):
        """负QPS应报错。"""
        with pytest.raises(ValueError, match="current_qps"):
            scaler.compute_desired_replicas(-1.0, 100.0)

    def test_negative_latency_error(self, scaler):
        """负延迟应报错。"""
        with pytest.raises(ValueError, match="avg_latency_ms"):
            scaler.compute_desired_replicas(100.0, -1.0)

    def test_decision_has_reason(self, scaler):
        """决策应有原因说明。"""
        decision = scaler.compute_desired_replicas(100.0, 400.0)
        assert len(decision.reason) > 0

    def test_qps_based_scaling(self, scaler):
        """高QPS应扩大副本数。"""
        decision = scaler.compute_desired_replicas(
            current_qps=1000.0,
            avg_latency_ms=400.0,
        )
        assert decision.action == "scale_up"
        assert decision.desired_replicas >= 10  # 1000/100 + 1 = 11


# =============================================================================
# TestAutoScalerMisc
# =============================================================================


class TestAutoScalerMisc:
    """AutoScaler 杂项测试。"""

    def test_creation(self, scaler):
        assert scaler.min_replicas == 1
        assert scaler.max_replicas == 20

    def test_invalid_min_replicas(self):
        with pytest.raises(ValueError, match="min_replicas"):
            AutoScaler(min_replicas=0)

    def test_invalid_max_lt_min(self):
        with pytest.raises(ValueError, match="max_replicas"):
            AutoScaler(min_replicas=5, max_replicas=3)

    def test_invalid_target_latency(self):
        with pytest.raises(ValueError, match="target_latency_ms"):
            AutoScaler(target_latency_ms=0)

    def test_metrics_history(self, scaler):
        """指标历史记录。"""
        scaler.compute_desired_replicas(100.0, 200.0)
        scaler.compute_desired_replicas(200.0, 300.0)
        assert len(scaler.metrics_history) == 2

    def test_current_replicas_tracking(self, scaler):
        """当前副本数跟踪。"""
        scaler.compute_desired_replicas(100.0, 400.0)
        assert scaler.current_replicas >= 1

    def test_record_metrics(self, scaler):
        """record_metrics 手动记录。"""
        scaler.record_metrics(50.0, 100.0, 3)
        assert len(scaler.metrics_history) == 1
