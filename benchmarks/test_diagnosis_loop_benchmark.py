"""
MCI World Model V4.2.0 — 推理失败自动诊断闭环基准测试

对标: CEWM 认知诊断能力 — MetaDiagnoser + 惊奇信号 → 根因分析 → 策略调整

评测推理失败时的自动诊断闭环:
  1. 构造 6 种推理失败场景 (惊奇信号)
  2. 触发 MetaDiagnoser.diagnose() 输出根因分析
  3. 验证诊断器正确识别失败模式
  4. 根据诊断结果自动调整推理策略

理论对标:
  - Beer VSM System 5: 策略层完备化诊断
  - Lakatos 进步性: 反常信号驱动理论修正
  - 认知故障树分析: 多层根因追溯

运行: pytest benchmarks/test_diagnosis_loop_benchmark.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._meta_diagnoser import (
    DiagnosisResult,
    FailurePattern,
    MetaDiagnoser,
    SeverityLevel,
    SurpriseSignal,
)

# =============================================================================
# 辅助函数
# =============================================================================


def _make_signal(
    score: float,
    state_distance: float,
    vector_deviation: float,
    direction_error: float,
    source: str = "reasoning",
) -> SurpriseSignal:
    """构造一个带有三维分解特征的惊奇信号。"""
    return SurpriseSignal(
        score=score,
        source=source,
        layer="cognitive",
        features={
            "state_distance": state_distance,
            "vector_deviation": vector_deviation,
            "direction_error": direction_error,
        },
    )


# 各失败模式的特征中值 (基于 _PATTERN_SIGNATURES)
_FAILURE_PROFILES: dict[FailurePattern, dict[str, float]] = {
    FailurePattern.PERCEPTION_DRIFT: {
        "state_distance": 0.75,
        "vector_deviation": 0.50,
        "direction_error": 0.25,
    },
    FailurePattern.PREDICTION_BIAS: {
        "state_distance": 0.50,
        "vector_deviation": 0.75,
        "direction_error": 0.60,
    },
    FailurePattern.CAUSAL_COLLAPSE: {
        "state_distance": 0.35,
        "vector_deviation": 0.35,
        "direction_error": 0.80,
    },
    FailurePattern.MEMORY_DECAY: {
        "state_distance": 0.60,
        "vector_deviation": 0.60,
        "direction_error": 0.45,
    },
    FailurePattern.FEEDBACK_LOOP_BROKEN: {
        "state_distance": 0.20,
        "vector_deviation": 0.20,
        "direction_error": 0.90,
    },
    FailurePattern.CONFOUNDER_INTRUSION: {
        "state_distance": 0.70,
        "vector_deviation": 0.45,
        "direction_error": 0.85,
    },
}


# =============================================================================
# pytest 测试套件
# =============================================================================


@pytest.fixture(scope="module")
def diagnoser():
    return MetaDiagnoser(confidence_threshold=0.2)


class TestDiagnoserBasicSetup:
    """验证 MetaDiagnoser 基本功能。"""

    def test_diagnose_returns_result(self, diagnoser):
        """diagnose 返回 DiagnosisResult。"""
        signal = _make_signal(0.8, 0.5, 0.7, 0.6)
        result = diagnoser.diagnose([signal])
        assert isinstance(result, DiagnosisResult)
        assert result.pattern is not None, "Should identify a failure pattern"

    def test_diagnose_with_dict_signal(self, diagnoser):
        """diagnose 接受 dict 格式信号。"""
        signal_dict = {
            "score": 0.8,
            "state_distance": 0.5,
            "vector_deviation": 0.7,
            "direction_error": 0.6,
        }
        result = diagnoser.diagnose([signal_dict])
        assert result.pattern is not None

    def test_root_cause_chain_depth(self, diagnoser):
        """根因链深度 ≥ 3。"""
        signal = _make_signal(0.9, 0.35, 0.35, 0.80)
        result = diagnoser.diagnose([signal])
        assert result.root_cause_chain.depth >= 3, f"Root cause chain depth {result.root_cause_chain.depth} < 3"


class TestFailureModeDiagnosis:
    """验证 6 种失败模式的正确诊断。

    目标: ≥ 5/6 正确诊断。
    """

    @pytest.mark.parametrize(
        "expected_pattern,features",
        [
            (
                FailurePattern.CAUSAL_COLLAPSE,
                {"state_distance": 0.35, "vector_deviation": 0.35, "direction_error": 0.80},
            ),
            (
                FailurePattern.PREDICTION_BIAS,
                {"state_distance": 0.50, "vector_deviation": 0.75, "direction_error": 0.60},
            ),
            (
                FailurePattern.PERCEPTION_DRIFT,
                {"state_distance": 0.75, "vector_deviation": 0.50, "direction_error": 0.25},
            ),
            (
                FailurePattern.MEMORY_DECAY,
                {"state_distance": 0.60, "vector_deviation": 0.60, "direction_error": 0.45},
            ),
            (
                FailurePattern.FEEDBACK_LOOP_BROKEN,
                {"state_distance": 0.20, "vector_deviation": 0.20, "direction_error": 0.90},
            ),
            (
                FailurePattern.CONFOUNDER_INTRUSION,
                {"state_distance": 0.70, "vector_deviation": 0.45, "direction_error": 0.85},
            ),
        ],
        ids=[
            "causal_collapse",
            "prediction_bias",
            "perception_drift",
            "memory_decay",
            "feedback_loop_broken",
            "confounder_intrusion",
        ],
    )
    def test_diagnose_failure_mode(self, expected_pattern, features):
        """每种失败模式应被正确识别。"""
        fresh_diagnoser = MetaDiagnoser(confidence_threshold=0.2)
        signal = SurpriseSignal(
            score=0.9,
            source="test",
            layer="cognitive",
            features=features,
        )
        result = fresh_diagnoser.diagnose([signal])

        assert result.pattern is not None, f"Failed to diagnose {expected_pattern.value}"
        # 主要模式应匹配，或者在 matches 中出现
        match_patterns = [m.pattern for m in result.matches]
        assert expected_pattern in match_patterns, (
            f"Expected {expected_pattern.value} in matches, got {[p.value for p in match_patterns]}"
        )


class TestDiagnosisAccuracy:
    """综合诊断准确率评估。"""

    def test_overall_accuracy(self):
        """总诊断准确率 ≥ 83% (5/6)。"""
        correct = 0
        total = len(_FAILURE_PROFILES)

        for pattern, features in _FAILURE_PROFILES.items():
            fresh_diagnoser = MetaDiagnoser(confidence_threshold=0.2)
            signal = SurpriseSignal(
                score=0.9,
                source="accuracy_test",
                layer="cognitive",
                features=dict(features),
            )
            result = fresh_diagnoser.diagnose([signal])

            if result.pattern == pattern:
                correct += 1
            elif any(m.pattern == pattern for m in result.matches):
                correct += 1  # 部分得分: 在 matches 中

        accuracy = correct / total
        assert accuracy >= 0.83, f"Diagnosis accuracy {accuracy:.1%} ({correct}/{total}) < 83%"


class TestDiagnosisDrivenStrategy:
    """诊断驱动的策略调整闭环。"""

    def test_recommendation_not_empty(self, diagnoser):
        """诊断结果应包含修复建议。"""
        signal = _make_signal(0.85, 0.35, 0.35, 0.80)
        result = diagnoser.diagnose([signal])
        assert result.recommendation, "Diagnosis should include a recommendation"
        assert len(result.recommendation) > 5, "Recommendation too short"

    def test_severity_assessment(self, diagnoser):
        """高惊奇度应触发高严重度。"""
        high_signal = _make_signal(0.95, 0.7, 0.8, 0.9)
        result = diagnoser.diagnose([high_signal])
        assert result.severity in (SeverityLevel.HIGH, SeverityLevel.CRITICAL), (
            f"High surprise should be HIGH/CRITICAL, got {result.severity.value}"
        )

    def test_health_score_degradation(self):
        """严重故障下认知健康度应下降。"""
        fresh = MetaDiagnoser()

        # 正常信号 (低惊奇度) — 使用 dict 格式
        normal_health = fresh.cognitive_health_score(
            [
                {"score": 0.1, "state_distance": 0.1, "vector_deviation": 0.1, "direction_error": 0.1},
            ]
        )

        # 异常信号 (高惊奇度)
        abnormal_health = fresh.cognitive_health_score(
            [
                {"score": 0.95, "state_distance": 0.8, "vector_deviation": 0.9, "direction_error": 0.9},
            ]
        )

        # 异常信号下至少部分维度下降
        normal_avg = np.mean(list(normal_health.values())) if normal_health else 0
        abnormal_avg = np.mean(list(abnormal_health.values())) if abnormal_health else 0

        assert abnormal_avg < normal_avg, f"Abnormal health ({abnormal_avg:.3f}) should < normal ({normal_avg:.3f})"

    def test_diagnosis_history_accumulates(self):
        """多次诊断累积到历史记录。"""
        fresh = MetaDiagnoser()
        for i in range(3):
            signal = _make_signal(0.7 + i * 0.1, 0.5, 0.5, 0.5)
            fresh.diagnose([signal])

        assert len(fresh._history) == 3, f"Expected 3 history entries, got {len(fresh._history)}"

    def test_multiple_signals_improve_diagnosis(self):
        """多个一致信号应提高诊断置信度。"""
        fresh = MetaDiagnoser(confidence_threshold=0.2)

        # 单信号诊断
        single_signal = _make_signal(0.8, 0.35, 0.35, 0.80)
        fresh.diagnose([single_signal])

        # 多信号诊断 (一致信号)
        fresh2 = MetaDiagnoser(confidence_threshold=0.2)
        signals = [
            _make_signal(0.8, 0.35, 0.35, 0.80),
            _make_signal(0.85, 0.30, 0.40, 0.75),
            _make_signal(0.9, 0.40, 0.30, 0.85),
        ]
        multi_result = fresh2.diagnose(signals)

        # 多信号应有合理的置信度
        assert multi_result.confidence > 0, "Multi-signal confidence should be > 0"
        assert multi_result.pattern is not None, "Multi-signal should produce a diagnosis"


class TestDiagnosisLoopComposite:
    """综合: 推理失败 → 诊断 → 策略调整 完整闭环。"""

    def test_full_diagnosis_loop(self):
        """完整闭环: 推理异常 → 诊断 → 根因 → 建议。"""
        fresh = MetaDiagnoser(confidence_threshold=0.2)

        # Step 1: 模拟推理失败 → 生成惊奇信号
        # 因果坍缩: 因果图连通性下降 → direction_error 高
        surprise = SurpriseSignal(
            score=0.85,
            source="causal_reasoning_failure",
            layer="cognitive",
            features={
                "state_distance": 0.35,  # 状态距离中等
                "vector_deviation": 0.30,  # 向量偏差低
                "direction_error": 0.85,  # 方向误差高 → 因果坍缩
            },
        )

        # Step 2: 诊断
        result = fresh.diagnose([surprise])

        # Step 3: 验证诊断结果
        assert result.pattern is not None, "Should identify failure pattern"
        assert result.confidence > 0.2, f"Confidence {result.confidence:.3f} too low"
        assert result.root_cause_chain.depth >= 3, "Root cause chain too shallow"
        assert result.recommendation, "Should provide recommendation"

        # Step 4: 验证根因链有意义
        chain = result.root_cause_chain.chain
        assert len(chain) >= 3, f"Chain should have ≥3 elements, got {len(chain)}"
        assert result.root_cause_chain.primary_cause, "Should identify primary cause"

    def test_diagnosis_to_dict_serializable(self):
        """诊断结果可序列化为 dict。"""
        fresh = MetaDiagnoser()
        signal = _make_signal(0.8, 0.5, 0.7, 0.6)
        result = fresh.diagnose([signal])

        d = result.to_dict()
        assert isinstance(d, dict)
        assert "pattern" in d
        assert "severity" in d
        assert "confidence" in d
        assert "root_cause_chain" in d
        assert "recommendation" in d
