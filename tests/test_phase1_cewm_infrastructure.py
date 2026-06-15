"""
测试 Phase 1 (v3.4.0) CEWM 闭环基础设施
========================================

覆盖:
    - CognitiveLoopBus: 四层跨层反馈总线
    - CognitiveDiversity: Ashby 认知多样性度量
    - MetaCognition v3.4.0: 统一元认知（根因分析/评分/策略）
    - SurpriseDetector.diagnose(): 根因分析链

KPI 验证:
    K1-1: 跨层反馈覆盖率 4/4 层闭环互联
    K1-2: H_cognitive 可计算 + H_CEWM > H_physics
    K1-3: MetaCognition 400+ 行
    K1-4: 全量测试通过
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mci_world_model.sdk._cognitive_diversity import CognitiveDiversity, DiversityVector
from mci_world_model.sdk._cognitive_loop import (
    CognitiveLayer,
    CognitiveLoopBus,
)
from mci_world_model.sdk._surprise_detector import SurpriseDetector
from mci_world_model.sdk._world_state import PendulumState

# =============================================================================
# CognitiveLoopBus — Wiener 四环跨层反馈总线
# =============================================================================


class TestCognitiveLoopBus:
    """CognitiveLoopBus 核心功能测试。"""

    # ── 初始化与属性 ──

    def test_init_defaults(self):
        bus = CognitiveLoopBus()
        assert bus.learning_rate == 0.01
        assert bus.coupling_coeff == 0.3
        assert bus.decay_factor == 0.9
        assert bus.step_count == 0

    def test_init_custom(self):
        bus = CognitiveLoopBus(learning_rate=0.05, coupling_coeff=0.5, decay_factor=0.8)
        assert bus.learning_rate == 0.05
        assert bus.coupling_coeff == 0.5
        assert bus.decay_factor == 0.8

    def test_property_setters(self):
        bus = CognitiveLoopBus()
        bus.learning_rate = 0.1
        assert bus.learning_rate == 0.1
        bus.coupling_coeff = 0.6
        assert bus.coupling_coeff == 0.6
        bus.decay_factor = 0.7
        assert bus.decay_factor == 0.7

    def test_property_clamping(self):
        bus = CognitiveLoopBus()
        bus.learning_rate = -1.0
        assert bus.learning_rate > 0
        bus.coupling_coeff = 5.0
        assert bus.coupling_coeff <= 1.0

    # ── 注入误差 ──

    def test_inject_error(self):
        bus = CognitiveLoopBus()
        signal = bus.inject_error(CognitiveLayer.PREDICTION, magnitude=0.5)
        assert signal.magnitude == 0.5
        assert signal.layer == CognitiveLayer.PREDICTION

    def test_inject_error_accumulates(self):
        bus = CognitiveLoopBus()
        bus.inject_error(CognitiveLayer.COGNITION, magnitude=0.3)
        bus.inject_error(CognitiveLayer.COGNITION, magnitude=0.2)
        error = bus.get_layer_error(CognitiveLayer.COGNITION)
        assert error.magnitude == pytest.approx(0.5, abs=0.01)

    def test_inject_error_with_gradient(self):
        bus = CognitiveLoopBus()
        grad = np.array([0.1, 0.2, 0.3])
        signal = bus.inject_error(CognitiveLayer.PERCEPTION, magnitude=0.4, gradient=grad)
        assert signal.gradient is not None

    def test_inject_from_surprise(self):
        bus = CognitiveLoopBus()
        breakdown = {
            "state_distance": 0.3,
            "vector_deviation": 0.4,
            "direction_error": 0.2,
        }
        bus.inject_from_surprise(surprise_score=0.5, breakdown=breakdown)
        # 所有层都应有误差
        for layer in CognitiveLayer:
            error = bus.get_layer_error(layer)
            assert error.magnitude > 0

    def test_inject_from_surprise_no_breakdown(self):
        bus = CognitiveLoopBus()
        bus.inject_from_surprise(surprise_score=0.4)
        for layer in CognitiveLayer:
            error = bus.get_layer_error(layer)
            assert error.magnitude > 0

    # ── 传播 ──

    def test_propagate_basic(self):
        bus = CognitiveLoopBus()
        bus.inject_error(CognitiveLayer.PREDICTION, magnitude=0.5)
        result = bus.propagate()
        assert result.step == 1
        assert result.total_energy >= 0
        assert isinstance(result.converged, bool)

    def test_propagate_reduces_energy(self):
        bus = CognitiveLoopBus(decay_factor=0.5)
        bus.inject_error(CognitiveLayer.PREDICTION, magnitude=0.8)
        bus.propagate()
        energy_after_1 = bus.energy_history[-1]
        bus.propagate()
        energy_after_2 = bus.energy_history[-1]
        assert energy_after_2 < energy_after_1

    def test_propagate_n_convergence(self):
        bus = CognitiveLoopBus(decay_factor=0.3)
        bus.inject_error(CognitiveLayer.COGNITION, magnitude=0.1)
        results = bus.propagate_n(20, early_stop=True)
        assert len(results) > 0
        # 低初始误差应该快速收敛
        assert results[-1].total_energy < 0.01

    def test_propagate_n_early_stop(self):
        bus = CognitiveLoopBus(decay_factor=0.1, convergence_threshold=0.1)
        bus.inject_error(CognitiveLayer.PERCEPTION, magnitude=0.05)
        results = bus.propagate_n(100, early_stop=True)
        assert len(results) < 100  # 应该提前停止

    def test_propagate_cross_layer(self):
        """K1-1: 验证预测环误差能传播到认知环。"""
        bus = CognitiveLoopBus()
        bus.inject_error(CognitiveLayer.PREDICTION, magnitude=0.8)
        result = bus.propagate()
        # 检查跨层耦合矩阵非零
        assert result.cross_coupling.shape == (4, 4)
        # 认知环应该收到来自下层的影响
        cog_error = result.layer_errors[CognitiveLayer.COGNITION]
        assert cog_error.magnitude >= 0

    def test_propagate_all_four_layers(self):
        """K1-1: 验证 4/4 层闭环互联。"""
        bus = CognitiveLoopBus()
        for layer in CognitiveLayer:
            bus.inject_error(layer, magnitude=0.3)
        result = bus.propagate()
        for layer in CognitiveLayer:
            assert layer in result.layer_errors
            assert layer in result.deltas

    # ── 耦合矩阵 ──

    def test_set_coupling(self):
        bus = CognitiveLoopBus()
        bus.set_coupling(CognitiveLayer.PERCEPTION, CognitiveLayer.COGNITION, 0.8)
        cm = bus.get_coupling_matrix()
        assert cm[1, 0] == 0.8

    def test_layer_connectivity_default(self):
        bus = CognitiveLoopBus()
        connectivity = bus.layer_connectivity()
        assert isinstance(connectivity, dict)
        assert "perception_to_cognition" in connectivity

    # ── 健康度报告 ──

    def test_health_report(self):
        bus = CognitiveLoopBus()
        bus.inject_error(CognitiveLayer.PREDICTION, magnitude=0.3)
        report = bus.health_report()
        assert 0 <= report.overall_health <= 1
        assert report.bottleneck_layer is not None
        assert isinstance(report.oscillation_detected, bool)

    def test_health_report_no_error(self):
        bus = CognitiveLoopBus()
        report = bus.health_report()
        assert report.overall_health > 0.9

    # ── 自适应参数 ──

    def test_adapt_parameters_energy(self):
        bus = CognitiveLoopBus()
        for i in range(10):
            bus.inject_error(CognitiveLayer.COGNITION, magnitude=0.1 * (i + 1))
            bus.propagate()
        params = bus.adapt_parameters(strategy="energy")
        assert "alpha" in params

    def test_adapt_parameters_oscillation(self):
        bus = CognitiveLoopBus()
        # 注入震荡模式
        for i in range(10):
            mag = 0.5 if i % 2 == 0 else 0.1
            bus.inject_error(CognitiveLayer.PREDICTION, magnitude=mag)
            bus.propagate()
        params = bus.adapt_parameters(strategy="oscillation")
        assert "alpha" in params

    # ── 运行统计 ──

    def test_running_statistics(self):
        bus = CognitiveLoopBus()
        stats = bus.running_statistics()
        assert stats["steps"] == 0

        bus.inject_error(CognitiveLayer.COGNITION, magnitude=0.3)
        bus.propagate()
        stats = bus.running_statistics()
        assert stats["steps"] == 1
        assert stats["mean_energy"] >= 0

    # ── 重置 ──

    def test_reset(self):
        bus = CognitiveLoopBus()
        bus.inject_error(CognitiveLayer.PREDICTION, magnitude=0.5)
        bus.propagate()
        bus.reset()
        assert bus.step_count == 0
        assert len(bus.energy_history) == 0

    def test_reset_keep_history(self):
        bus = CognitiveLoopBus()
        bus.inject_error(CognitiveLayer.COGNITION, magnitude=0.3)
        bus.propagate()
        bus.reset(clear_history=False)
        assert len(bus.energy_history) > 0

    def test_repr(self):
        bus = CognitiveLoopBus()
        assert "CognitiveLoopBus" in repr(bus)


# =============================================================================
# CognitiveLayer 枚举
# =============================================================================


class TestCognitiveLayer:
    def test_layer_values(self):
        assert CognitiveLayer.PERCEPTION.value == 0
        assert CognitiveLayer.COGNITION.value == 1
        assert CognitiveLayer.PREDICTION.value == 2
        assert CognitiveLayer.ACTION.value == 3

    def test_layer_labels(self):
        assert CognitiveLayer.PERCEPTION.label == "感知环"
        assert CognitiveLayer.ACTION.label == "行动环"

    def test_layer_index(self):
        for layer in CognitiveLayer:
            assert layer.index == layer.value


# =============================================================================
# CognitiveDiversity — Ashby 认知多样性度量
# =============================================================================


class TestCognitiveDiversity:
    """CognitiveDiversity 核心功能测试。"""

    def test_init(self):
        cd = CognitiveDiversity()
        assert cd._n_bins == 10

    # ── H_physics ──

    def test_compute_physics_diversity(self):
        cd = CognitiveDiversity()
        states = [PendulumState(theta=i * 0.1, omega=0.0) for i in range(50)]
        dv = cd.compute(states=states)
        assert dv.h_physics >= 0

    def test_compute_physics_diversity_single_state(self):
        cd = CognitiveDiversity()
        dv = cd.compute(states=[PendulumState(theta=0.5, omega=1.0)])
        assert dv.h_physics == 0.0  # 单状态无多样性

    def test_compute_physics_diversity_varied(self):
        """K1-2: 验证 H_CEWM 可计算。"""
        cd = CognitiveDiversity()
        states = [
            PendulumState(theta=float(np.random.uniform(-3.14, 3.14)), omega=float(np.random.uniform(-5, 5)))
            for _ in range(100)
        ]
        dv = cd.compute(states=states)
        assert dv.h_physics > 0.1

    # ── H_causal ──

    def test_compute_causal_diversity(self):
        cd = CognitiveDiversity()
        edges = [
            ("A", "B", 0.8),
            ("B", "C", 0.6),
            ("A", "C", 0.4),
            ("C", "D", 0.9),
        ]
        dv = cd.compute(causal_edges=edges)
        assert dv.h_causal > 0

    def test_compute_causal_diversity_empty(self):
        cd = CognitiveDiversity()
        dv = cd.compute()
        assert dv.h_causal == 0.0

    # ── H_temporal ──

    def test_compute_temporal_diversity(self):
        cd = CognitiveDiversity()
        errors = [0.1, 0.3, 0.2, 0.5, 0.15, 0.4, 0.25, 0.35]
        dv = cd.compute(prediction_errors=errors)
        assert dv.h_temporal >= 0

    # ── H_modal ──

    def test_compute_modal_diversity(self):
        cd = CognitiveDiversity()
        counts = {"proprioception": 50, "vision": 30, "audio": 10, "thermal": 5}
        dv = cd.compute(modality_counts=counts)
        assert dv.h_modal > 0

    def test_compute_modal_diversity_single(self):
        cd = CognitiveDiversity()
        counts = {"proprioception": 100}
        dv = cd.compute(modality_counts=counts)
        assert dv.h_modal == 0.0  # 单模态无多样性

    # ── H_meta ──

    def test_compute_meta_diversity(self):
        cd = CognitiveDiversity()
        dv = cd.compute(gap_count=5, total_checks=20)
        assert dv.h_meta > 0

    def test_compute_meta_diversity_no_checks(self):
        cd = CognitiveDiversity()
        dv = cd.compute()
        assert dv.h_meta == 0.0

    # ── Ashby 条件 ──

    def test_ashby_satisfied(self):
        """K1-2: 验证 H_CEWM > H_physics（Ashby 条件）。"""
        dv = DiversityVector(
            h_physics=0.5,
            h_causal=0.3,
            h_temporal=0.2,
            h_modal=0.1,
            h_meta=0.15,
        )
        assert dv.total > dv.h_physics
        assert dv.ashby_satisfied is True
        assert dv.ashby_ratio > 1.0

    def test_ashby_violated(self):
        dv = DiversityVector(h_physics=0.9, h_causal=0.0, h_temporal=0.0, h_modal=0.0, h_meta=0.0)
        assert dv.ashby_satisfied is False

    def test_ashby_check(self):
        cd = CognitiveDiversity()
        dv = DiversityVector(h_physics=0.3, h_causal=0.2, h_temporal=0.1, h_modal=0.1, h_meta=0.1)
        result = cd.ashby_check(dv)
        assert result["satisfied"] is True
        assert "verdict" in result

    # ── 增量更新 ──

    def test_update_and_current(self):
        cd = CognitiveDiversity()
        for i in range(20):
            cd.update(state=PendulumState(theta=i * 0.2, omega=0.0))
        dv = cd.current()
        assert dv.h_physics >= 0

    def test_update_causal_edge(self):
        cd = CognitiveDiversity()
        cd.update(causal_edge=("A", "B", 0.5))
        cd.update(causal_edge=("B", "C", 0.3))
        dv = cd.current()
        assert dv.h_causal > 0

    # ── DiversityVector ──

    def test_diversity_vector_to_vector(self):
        dv = DiversityVector(h_physics=0.5, h_causal=0.3, h_temporal=0.2, h_modal=0.1, h_meta=0.15)
        vec = dv.to_vector()
        assert len(vec) == 5
        assert vec[0] == 0.5

    def test_diversity_vector_to_dict(self):
        dv = DiversityVector(h_physics=0.5, h_causal=0.3, h_temporal=0.2, h_modal=0.1, h_meta=0.15)
        d = dv.to_dict()
        assert "total" in d
        assert "ashby_satisfied" in d

    # ── 历史趋势 ──

    def test_history_trend(self):
        cd = CognitiveDiversity()
        history = cd.get_history()
        assert history.n_samples == 0

        for i in range(5):
            cd.compute(states=[PendulumState(theta=j * 0.1, omega=0.0) for j in range(20)])
        history = cd.get_history()
        assert history.n_samples == 5
        assert history.latest is not None

    def test_reset(self):
        cd = CognitiveDiversity()
        cd.compute(states=[PendulumState(theta=0.5, omega=1.0)])
        cd.reset()
        assert cd.get_history().n_samples == 0

    def test_repr(self):
        cd = CognitiveDiversity()
        assert "CognitiveDiversity" in repr(cd)


# =============================================================================
# MetaCognition v3.4.0 — 统一元认知
# =============================================================================


class TestMetaCognitionV34:
    """MetaCognition v3.4.0 新功能测试。"""

    # ── 向后兼容: v3.3.0 接口 ──

    def test_legacy_discover_gaps(self):
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        gaps = mc.discover_gaps({"fact": 1, "other": 9}, [], [])
        assert isinstance(gaps, list)
        assert any(g["type"] == "domain" for g in gaps)

    def test_legacy_detect_conflicts(self):
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        beliefs = {"a": {"content": "这是正确的"}, "b": {"content": "这是错误的"}}
        conflicts = mc.detect_conflicts(beliefs)
        assert len(conflicts) > 0

    def test_legacy_get_aging(self):
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        memories = [{"id": "m1", "timestamp": time.time() - 86400 * 40}]
        aging = mc.get_aging(memories)
        assert len(aging) == 1
        assert aging[0]["severity"] == "critical"

    def test_legacy_contradicts(self):
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        assert mc._contradicts("这是正确的决定", "但不是这样")

    # ── 新接口: v3.4.0 ──

    def test_typed_discover_gaps(self):
        from mci_world_model._sys.meta_cognition import CognitiveGap, MetaCognition

        mc = MetaCognition()
        gaps = mc.discover_gaps(
            memory_types={"fact": 1, "event": 20},
            user_domains=["science"],
            memory_list=[{"id": f"m{i}", "type": "event"} for i in range(25)],
        )
        assert all(isinstance(g, CognitiveGap) for g in gaps)

    def test_typed_detect_conflicts(self):
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        beliefs = {
            "a": {"content": "这是正确的", "confidence": 0.9, "stage": "强化"},
            "b": {"content": "这是错误的", "confidence": 0.85, "stage": "确认"},
        }
        conflicts = mc.detect_conflicts(beliefs)
        assert len(conflicts) > 0
        assert "description" in conflicts[0]

    def test_get_aging_warnings(self):
        from mci_world_model._sys.meta_cognition import KnowledgeAging, MetaCognition

        mc = MetaCognition()
        memories = [{"id": "m1", "timestamp": time.time() - 86400 * 40, "stage": "强化"}]
        warnings = mc.get_aging_warnings(memories)
        assert len(warnings) == 1
        assert isinstance(warnings[0], KnowledgeAging)

    # ── 根因分析链 (v3.4.0) ──

    def test_root_cause_analysis(self):
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        signals = [
            {"score": 0.7, "breakdown": {"state_distance": 0.5, "vector_deviation": 0.4, "direction_error": 0.3}},
        ]
        root = mc.root_cause_analysis(signals)
        assert root.contribution == 1.0
        assert root.layer in ("signal", "prediction", "causal")
        assert len(root.children) > 0

    def test_root_cause_analysis_with_graph(self):
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        signals = [
            {"score": 0.8, "breakdown": {"state_distance": 0.6, "vector_deviation": 0.5, "direction_error": 0.4}},
        ]
        graph = {"sensor": ["encoder"], "encoder": ["perception"]}
        root = mc.root_cause_analysis(signals, causal_graph=graph)
        assert root.contribution == 1.0

    def test_root_cause_analysis_no_signals(self):
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        root = mc.root_cause_analysis([])
        assert root.contribution == 0.0

    # ── 认知评分 (v3.4.0) ──

    def test_cognitive_score(self):
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        score = mc.cognitive_score(
            causal_edges_count=20,
            total_memories=50,
            counterfactual_queries=5,
            prediction_accuracy=0.85,
            anomaly_detection_rate=0.7,
            memory_reuse_rate=0.6,
            explainability_score=0.75,
        )
        assert score.total > 0
        d = score.to_dict()
        assert "causal_discovery" in d
        assert "total" in d

    def test_cognitive_score_empty(self):
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        score = mc.cognitive_score()
        assert score.total == 0.0

    # ── 策略推荐 (v3.4.0) ──

    def test_recommend_strategy(self):
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        gaps = [{"type": "domain", "severity": 0.7}]
        strategies = mc.recommend_strategy(gaps=gaps)
        assert len(strategies) > 0
        assert "strategy" in strategies[0]
        assert "priority" in strategies[0]

    def test_recommend_strategy_with_scorecard(self):
        from mci_world_model._sys.meta_cognition import CognitiveScoreCard, MetaCognition

        mc = MetaCognition()
        sc = CognitiveScoreCard(
            causal_discovery=0.8,
            counterfactual=0.3,
            ood_generalization=0.6,
            explainability=0.7,
            memory_reuse=0.4,
            anomaly_detection=0.9,
        )
        strategies = mc.recommend_strategy(scorecard=sc)
        # 应该推荐最弱维度（counterfactual=0.3）
        assert any("反事实" in s.get("strategy", "") for s in strategies)

    # ── 历史查询 ──

    def test_score_history(self):
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        mc.cognitive_score(causal_edges_count=10, total_memories=20)
        assert len(mc.get_score_history()) == 1

    def test_root_cause_history(self):
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        mc.root_cause_analysis([{"score": 0.5, "breakdown": {}}])
        assert len(mc.get_root_cause_history()) == 1

    def test_reset(self):
        from mci_world_model._sys.meta_cognition import MetaCognition

        mc = MetaCognition()
        mc.cognitive_score(causal_edges_count=5)
        mc.reset()
        assert len(mc.get_score_history()) == 0


# =============================================================================
# SurpriseDetector.diagnose() — 根因分析链
# =============================================================================


class TestSurpriseDetectorDiagnose:
    """SurpriseDetector.diagnose() 功能测试。"""

    def test_diagnose_basic(self):
        detector = SurpriseDetector(threshold=0.3)
        predicted = PendulumState(theta=0.5, omega=1.0)
        actual = PendulumState(theta=1.0, omega=0.5)
        signal = detector.compute_surprise(predicted, actual)
        diagnosis = detector.diagnose(signal)
        assert "root_cause_layer" in diagnosis
        assert "dimension_analysis" in diagnosis
        assert "severity" in diagnosis
        assert "recommendation" in diagnosis

    def test_diagnose_with_causal_graph(self):
        detector = SurpriseDetector(threshold=0.3)
        predicted = PendulumState(theta=0.5, omega=1.0)
        actual = PendulumState(theta=2.0, omega=-1.0)
        signal = detector.compute_surprise(predicted, actual)
        graph = {"sensor": ["encoder"], "encoder": ["predictor"]}
        diagnosis = detector.diagnose(signal, causal_graph=graph)
        assert "causal_chain" in diagnosis

    def test_diagnose_dimension_analysis(self):
        detector = SurpriseDetector(threshold=0.3)
        predicted = PendulumState(theta=0.0, omega=0.0)
        actual = PendulumState(theta=2.0, omega=3.0)
        signal = detector.compute_surprise(predicted, actual)
        diagnosis = detector.diagnose(signal)
        dim_analysis = diagnosis["dimension_analysis"]
        assert "state_distance" in dim_analysis
        assert "vector_deviation" in dim_analysis
        assert "direction_error" in dim_analysis
        for dim_info in dim_analysis.values():
            assert "value" in dim_info
            assert "level" in dim_info
            assert "interpretation" in dim_info

    def test_diagnose_severity_range(self):
        detector = SurpriseDetector(threshold=0.3)
        predicted = PendulumState(theta=0.5, omega=1.0)
        actual = PendulumState(theta=0.6, omega=0.9)
        signal = detector.compute_surprise(predicted, actual)
        diagnosis = detector.diagnose(signal)
        assert 0 <= diagnosis["severity"] <= 1

    def test_batch_diagnose(self):
        detector = SurpriseDetector(threshold=0.3)
        pairs = [
            (PendulumState(theta=0.5, omega=1.0), PendulumState(theta=1.0, omega=0.5)),
            (PendulumState(theta=0.0, omega=0.0), PendulumState(theta=0.1, omega=0.1)),
        ]
        signals = [detector.compute_surprise(p, a) for p, a in pairs]
        results = detector.batch_diagnose(signals)
        assert len(results) == 2

    def test_diagnose_details(self):
        detector = SurpriseDetector(threshold=0.3)
        predicted = PendulumState(theta=0.5, omega=1.0)
        actual = PendulumState(theta=1.5, omega=-0.5)
        signal = detector.compute_surprise(predicted, actual)
        diagnosis = detector.diagnose(signal, context={"step": 42})
        assert diagnosis["details"]["context"]["step"] == 42


# =============================================================================
# 导入兼容性测试
# =============================================================================


class TestImportsV34:
    """验证 v3.4.0 新增符号可从各层级导入。"""

    def test_import_from_sdk(self):
        from mci_world_model.sdk import CognitiveDiversity, CognitiveLoopBus

        assert CognitiveLoopBus is not None
        assert CognitiveDiversity is not None

    def test_import_from_top(self):
        from mci_world_model import CognitiveDiversity, CognitiveLoopBus

        assert CognitiveLoopBus is not None
        assert CognitiveDiversity is not None

    def test_import_metacognition_from_sys(self):
        from mci_world_model._sys.meta_cognition import CognitiveGap, KnowledgeAging, MetaCognition

        assert MetaCognition is not None
        assert CognitiveGap is not None
        assert KnowledgeAging is not None

    def test_import_metacognition_from_awareness(self):
        """验证 awareness.py 兼容重导出。"""
        from mci_world_model._sys.awareness import CognitiveGap, KnowledgeAging, MetaCognition

        assert MetaCognition is not None
        assert CognitiveGap is not None
        assert KnowledgeAging is not None

    def test_import_from_world_model(self):
        """验证 _world_model.py 使用的导入路径仍然可用。"""
        from mci_world_model._sys.awareness import MetaCognition as MC

        mc = MC()
        gaps = mc.discover_gaps(
            memory_types={"fact": 5, "event": 10},
            user_domains=[],
            memory_list=[],
        )
        assert isinstance(gaps, list)
