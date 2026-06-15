"""Phase 4 (v3.7.0) 认知诊断系统 — 测试套件

覆盖:
- TestMetaDiagnoser: MetaDiagnoser 学习型认知诊断 (18 测试)
- TestNegativeHeuristic: NegativeHeuristic Lakatos 负面启发法 (14 测试)
- TestHierarchicalConfiguratorUpgrade: 协调层升级 (8 测试)
- TestImportsV37: 导出符号完整性 (6 测试)

目标: 46 个新测试 → 基线 1870 + 46 = 1916 passed
"""

from __future__ import annotations

# =============================================================================
# TestMetaDiagnoser — 学习型认知诊断系统
# =============================================================================


class TestMetaDiagnoser:
    """MetaDiagnoser 学习型认知诊断系统测试。"""

    def test_init(self):
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser

        md = MetaDiagnoser()
        assert md.confidence_threshold == 0.3
        assert md.max_chain_depth == 5

    def test_n_patterns_at_least_8(self):
        """K4-1: 诊断模式数 ≥ 8。"""
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser

        md = MetaDiagnoser()
        assert md.n_patterns >= 8

    def test_failure_pattern_enum(self):
        from mci_world_model.sdk._meta_diagnoser import FailurePattern

        patterns = list(FailurePattern)
        assert len(patterns) >= 8
        assert FailurePattern.PERCEPTION_DRIFT in patterns
        assert FailurePattern.PREDICTION_BIAS in patterns

    def test_surprise_signal_dataclass(self):
        from mci_world_model.sdk._meta_diagnoser import SurpriseSignal

        sig = SurpriseSignal(
            score=0.8,
            source="test",
            layer="prediction",
            features={"error_rate": 0.7},
        )
        assert sig.score == 0.8
        assert sig.source == "test"
        assert sig.breakdown == {"error_rate": 0.7}

    def test_diagnose_high_signal(self):
        """K4-2: 高惊奇信号应产生诊断。"""
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser, SurpriseSignal

        md = MetaDiagnoser()
        signal = SurpriseSignal(
            score=0.85,
            source="pred",
            layer="prediction",
            features={"direction_error": 0.8, "state_distance": 0.6, "vector_deviation": 0.7},
        )
        result = md.diagnose([signal])
        assert result is not None
        assert result.confidence >= 0.0

    def test_diagnose_low_signal(self):
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser, SeverityLevel, SurpriseSignal

        md = MetaDiagnoser()
        signal = SurpriseSignal(
            score=0.1,
            source="noise",
            layer="perception",
            features={"direction_error": 0.05},
        )
        result = md.diagnose([signal])
        assert result.severity in (SeverityLevel.LOW, SeverityLevel.MEDIUM)

    def test_diagnose_with_dict_signals(self):
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser

        md = MetaDiagnoser()
        signals = [{"score": 0.7, "direction_error": 0.6, "state_distance": 0.5}]
        result = md.diagnose(signals)
        assert result is not None

    def test_root_cause_chain_depth_ge_3(self):
        """K4-3: 根因分析链深度 ≥ 3。"""
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser, SurpriseSignal

        md = MetaDiagnoser()
        signal = SurpriseSignal(
            score=0.8,
            source="pred",
            layer="prediction",
            features={"direction_error": 0.7, "state_distance": 0.5, "vector_deviation": 0.6},
        )
        result = md.diagnose([signal])
        assert result.root_cause_chain.depth >= 3

    def test_root_cause_chain_structure(self):
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser, SurpriseSignal

        md = MetaDiagnoser()
        signal = SurpriseSignal(
            score=0.75,
            source="causal",
            layer="causal",
            features={"direction_error": 0.65, "state_distance": 0.4},
        )
        result = md.diagnose([signal])
        chain = result.root_cause_chain
        assert len(chain.chain) == chain.depth
        assert chain.primary_cause != ""

    def test_match_patterns(self):
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser

        md = MetaDiagnoser()
        signals = [{"direction_error": 0.8, "state_distance": 0.7, "score": 0.6}]
        matches = md.match_patterns(signals)
        assert isinstance(matches, list)

    def test_cognitive_health_score_six_dims(self):
        """六维认知健康度。"""
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser

        md = MetaDiagnoser()
        health = md.cognitive_health_score()
        expected = {
            "causal_discovery",
            "counterfactual",
            "ood_generalization",
            "explainability",
            "memory_reuse",
            "anomaly_detection",
        }
        assert set(health.keys()) == expected
        for v in health.values():
            assert 0.0 <= v <= 1.0

    def test_batch_diagnose(self):
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser, SurpriseSignal

        md = MetaDiagnoser()
        batches = [
            [SurpriseSignal(score=0.8, features={"direction_error": 0.7})],
            [SurpriseSignal(score=0.3, features={"direction_error": 0.2})],
        ]
        results = md.batch_diagnose(batches)
        assert len(results) == 2

    def test_evaluate_accuracy(self):
        """K4-2: 诊断准确率测试。"""
        from mci_world_model.sdk._meta_diagnoser import (
            FailurePattern,
            MetaDiagnoser,
            SurpriseSignal,
        )

        md = MetaDiagnoser()
        test_cases = [
            (
                [SurpriseSignal(score=0.9, features={"direction_error": 0.9, "state_distance": 0.3})],
                FailurePattern.PREDICTION_BIAS,
            ),
        ]
        result = md.evaluate_accuracy(test_cases)
        assert "accuracy" in result
        assert result["total"] == 1

    def test_statistics(self):
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser, SurpriseSignal

        md = MetaDiagnoser()
        md.diagnose([SurpriseSignal(score=0.5, features={})])
        stats = md.statistics()
        assert stats.total_diagnoses == 1

    def test_stats_property(self):
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser, SurpriseSignal

        md = MetaDiagnoser()
        md.diagnose([SurpriseSignal(score=0.5, features={})])
        assert md.stats.total_diagnoses == 1

    def test_history(self):
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser, SurpriseSignal

        md = MetaDiagnoser()
        md.diagnose([SurpriseSignal(score=0.5, features={})])
        assert len(md.history()) == 1

    def test_reset_stats(self):
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser, SurpriseSignal

        md = MetaDiagnoser()
        md.diagnose([SurpriseSignal(score=0.5, features={})])
        md.reset_stats()
        assert md.stats.total_diagnoses == 0
        assert len(md.history()) == 0

    def test_diagnosis_result_to_dict(self):
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser, SurpriseSignal

        md = MetaDiagnoser()
        result = md.diagnose([SurpriseSignal(score=0.6, features={"direction_error": 0.5})])
        d = result.to_dict()
        assert "pattern" in d
        assert "severity" in d
        assert "root_cause_chain" in d


# =============================================================================
# TestNegativeHeuristic — Lakatos 负面启发法
# =============================================================================


class TestNegativeHeuristic:
    """NegativeHeuristic Lakatos 负面启发法测试。"""

    def test_init(self):
        from mci_world_model.sdk._negative_heuristic import NegativeHeuristic

        nh = NegativeHeuristic()
        assert len(nh.rules) >= 5

    def test_seven_default_rules(self):
        """K4-4: 硬核规则数 ≥ 5 (实际 7)。"""
        from mci_world_model.sdk._negative_heuristic import NegativeHeuristic

        nh = NegativeHeuristic()
        assert len(nh.rules) == 7
        rule_ids = {r["rule_id"] for r in nh.rules}
        assert "HC-1" in rule_ids
        assert "HC-7" in rule_ids

    def test_violate_remove_causal(self):
        """HC-1: 移除因果组件应被拒绝。"""
        from mci_world_model.sdk._negative_heuristic import (
            ChangeType,
            NegativeHeuristic,
            ProposedChange,
        )

        nh = NegativeHeuristic()
        change = ProposedChange(
            description="移除因果图",
            affected_components=["causal_graph"],
            change_type=ChangeType.REMOVE,
        )
        viols = nh.violations(change)
        assert len(viols) >= 1
        assert any(v.rule_id == "HC-1" for v in viols)

    def test_violate_disable_memory(self):
        """HC-2: 禁用记忆组件应被拒绝。"""
        from mci_world_model.sdk._negative_heuristic import (
            ChangeType,
            NegativeHeuristic,
            ProposedChange,
        )

        nh = NegativeHeuristic()
        change = ProposedChange(
            description="禁用经验库",
            affected_components=["experience_db"],
            change_type=ChangeType.DISABLE,
        )
        viols = nh.violations(change)
        assert any(v.rule_id == "HC-2" for v in viols)

    def test_violate_remove_feedback_loop(self):
        """HC-3: 移除反馈环应被拒绝。"""
        from mci_world_model.sdk._negative_heuristic import (
            ChangeType,
            NegativeHeuristic,
            ProposedChange,
        )

        nh = NegativeHeuristic()
        change = ProposedChange(
            description="移除反馈环",
            affected_components=["cognitive_loop_bus"],
            change_type=ChangeType.REMOVE,
        )
        viols = nh.violations(change)
        assert any(v.rule_id == "HC-3" for v in viols)

    def test_admissible_parameter_tune(self):
        """参数调整应被允许。"""
        from mci_world_model.sdk._negative_heuristic import (
            ChangeType,
            NegativeHeuristic,
            ProposedChange,
        )

        nh = NegativeHeuristic()
        change = ProposedChange(
            description="调整学习率",
            affected_components=["causal_graph"],
            change_type=ChangeType.PARAMETER_TUNE,
        )
        assert nh.is_admissible(change) is True

    def test_not_admissible_remove_causal(self):
        """移除因果组件不可接受。"""
        from mci_world_model.sdk._negative_heuristic import (
            ChangeType,
            NegativeHeuristic,
            ProposedChange,
        )

        nh = NegativeHeuristic()
        change = ProposedChange(
            description="移除因果图",
            affected_components=["causal_graph"],
            change_type=ChangeType.REMOVE,
        )
        assert nh.is_admissible(change) is False

    def test_protective_belt_suggestions_prediction_bias(self):
        from mci_world_model.sdk._negative_heuristic import NegativeHeuristic

        nh = NegativeHeuristic()
        suggestions = nh.protective_belt_suggestions(
            diagnosis={"pattern": "PREDICTION_BIAS", "severity": "high"},
        )
        assert len(suggestions) >= 1
        assert suggestions[0].target == "prediction_weights"

    def test_protective_belt_suggestions_causal_collapse(self):
        from mci_world_model.sdk._negative_heuristic import NegativeHeuristic

        nh = NegativeHeuristic()
        suggestions = nh.protective_belt_suggestions(
            diagnosis={"pattern": "CAUSAL_COLLAPSE", "severity": "high"},
        )
        assert len(suggestions) >= 1

    def test_protective_belt_suggestions_empty(self):
        from mci_world_model.sdk._negative_heuristic import NegativeHeuristic

        nh = NegativeHeuristic()
        assert nh.protective_belt_suggestions(None) == []

    def test_hard_core_status(self):
        from mci_world_model.sdk._negative_heuristic import NegativeHeuristic

        nh = NegativeHeuristic()
        status = nh.hard_core_status()
        assert "HC-1" in status
        assert "HC-7" in status
        assert status["HC-1"]["severity"] == "absolute"

    def test_stats_tracking(self):
        from mci_world_model.sdk._negative_heuristic import (
            ChangeType,
            NegativeHeuristic,
            ProposedChange,
        )

        nh = NegativeHeuristic()
        nh.violations(ProposedChange(description="ok", change_type=ChangeType.MODIFY))
        nh.violations(
            ProposedChange(
                description="bad",
                affected_components=["causal_graph"],
                change_type=ChangeType.REMOVE,
            )
        )
        stats = nh.stats
        assert stats.total_checks == 2
        assert stats.accepted_changes >= 1

    def test_reset_stats(self):
        from mci_world_model.sdk._negative_heuristic import NegativeHeuristic

        nh = NegativeHeuristic()
        nh.reset_stats()
        assert nh.stats.total_checks == 0

    def test_add_change_no_overlap(self):
        """不涉及保护组件的变更无违反。"""
        from mci_world_model.sdk._negative_heuristic import (
            ChangeType,
            NegativeHeuristic,
            ProposedChange,
        )

        nh = NegativeHeuristic()
        change = ProposedChange(
            description="修改日志格式",
            affected_components=["logging_module"],
            change_type=ChangeType.MODIFY,
        )
        assert nh.is_admissible(change) is True


# =============================================================================
# TestHierarchicalConfiguratorUpgrade — 协调层升级
# =============================================================================


class TestHierarchicalConfiguratorUpgrade:
    """HierarchicalConfigurator v3.7.0 升级测试。"""

    def test_multi_objective_optimize_basic(self):
        from mci_world_model._sys._configurator import HierarchicalConfigurator

        hc = HierarchicalConfigurator()
        result = hc.multi_objective_optimize(
            prediction_error=0.3,
            cognitive_gap_score=0.4,
            energy_balance_score=0.7,
        )
        assert "composite_score" in result
        assert "dominant_objective" in result
        assert "pareto_frontier" in result

    def test_multi_objective_optimize_low_scores(self):
        """低分应触发建议。"""
        from mci_world_model._sys._configurator import HierarchicalConfigurator

        hc = HierarchicalConfigurator()
        result = hc.multi_objective_optimize(
            prediction_error=0.9,
            cognitive_gap_score=0.8,
            energy_balance_score=0.1,
        )
        assert len(result["recommendations"]) >= 2

    def test_multi_objective_optimize_pareto(self):
        """高分应达到 Pareto 最优。"""
        from mci_world_model._sys._configurator import HierarchicalConfigurator

        hc = HierarchicalConfigurator()
        result = hc.multi_objective_optimize(
            prediction_error=0.1,
            cognitive_gap_score=0.1,
            energy_balance_score=0.9,
        )
        assert result["pareto_frontier"]["is_optimal"] is True

    def test_multi_objective_optimize_custom_weights(self):
        from mci_world_model._sys._configurator import HierarchicalConfigurator

        hc = HierarchicalConfigurator()
        result = hc.multi_objective_optimize(
            prediction_error=0.5,
            cognitive_gap_score=0.5,
            energy_balance_score=0.5,
            weights={"prediction": 0.6, "cognitive": 0.3, "energy": 0.1},
        )
        assert 0 <= result["composite_score"] <= 1

    def test_multi_objective_dominant_is_worst(self):
        """主导目标应为最差维度。"""
        from mci_world_model._sys._configurator import HierarchicalConfigurator

        hc = HierarchicalConfigurator()
        result = hc.multi_objective_optimize(
            prediction_error=0.9,  # pred_score = 0.1 (worst)
            cognitive_gap_score=0.2,  # cog_score = 0.8
            energy_balance_score=0.8,  # eng_score = 0.8
        )
        assert result["dominant_objective"] == "prediction"

    def test_diagnose_and_configure(self):
        """诊断驱动配置整合流程。"""
        from mci_world_model._sys._configurator import HierarchicalConfigurator
        from mci_world_model.sdk._world_model import MCIWorldModel

        hc = HierarchicalConfigurator()
        wm = MCIWorldModel()

        result = hc.diagnose_and_configure(
            world_model=wm,
            diagnosis_result={
                "pattern": "PREDICTION_BIAS",
                "prediction_error": 0.6,
                "cognitive_gap_score": 0.4,
                "energy_balance_score": 0.7,
            },
        )
        assert "actions" in result
        assert "heuristic_check" in result
        assert "optimization" in result

    def test_diagnose_and_configure_with_violations(self):
        """诊断建议违反硬核时应被标记。"""
        from mci_world_model._sys._configurator import HierarchicalConfigurator
        from mci_world_model.sdk._world_model import MCIWorldModel

        hc = HierarchicalConfigurator()
        wm = MCIWorldModel()

        result = hc.diagnose_and_configure(
            world_model=wm,
            diagnosis_result={
                "pattern": "CAUSAL_COLLAPSE",
                "suggested_changes": [
                    {"description": "移除因果图", "components": ["causal_graph"], "type": "remove"},
                ],
            },
        )
        assert result["heuristic_check"]["is_admissible"] is False

    def test_diagnose_and_configure_no_diagnosis(self):
        """无诊断结果时正常执行。"""
        from mci_world_model._sys._configurator import HierarchicalConfigurator
        from mci_world_model.sdk._world_model import MCIWorldModel

        hc = HierarchicalConfigurator()
        wm = MCIWorldModel()
        result = hc.diagnose_and_configure(world_model=wm)
        assert result["pattern"] == ""


# =============================================================================
# TestImportsV37 — 导出符号完整性
# =============================================================================


class TestImportsV37:
    """v3.7.0 导出符号完整性测试。"""

    def test_sdk_imports(self):
        from mci_world_model.sdk import MetaDiagnoser, NegativeHeuristic

        assert MetaDiagnoser is not None
        assert NegativeHeuristic is not None

    def test_sdk_detail_imports(self):
        from mci_world_model.sdk import (
            DiagnosisResult,
            FailurePattern,
            PatternMatch,
            RootCauseChain,
            SeverityLevel,
            SurpriseSignal,
        )

        assert all(
            x is not None
            for x in [
                SurpriseSignal,
                FailurePattern,
                SeverityLevel,
                DiagnosisResult,
                PatternMatch,
                RootCauseChain,
            ]
        )

    def test_sdk_negative_heuristic_imports(self):
        from mci_world_model.sdk import (
            ChangeType,
            HardCoreViolation,
            ProposedChange,
            ProtectiveBeltSuggestion,
            RuleSeverity,
        )

        assert all(
            x is not None
            for x in [
                ProposedChange,
                ChangeType,
                HardCoreViolation,
                ProtectiveBeltSuggestion,
                RuleSeverity,
            ]
        )

    def test_sys_imports(self):
        from mci_world_model._sys import MetaDiagnoser, NegativeHeuristic

        assert MetaDiagnoser is not None
        assert NegativeHeuristic is not None

    def test_top_level_imports(self):
        from mci_world_model import MetaDiagnoser, NegativeHeuristic

        assert MetaDiagnoser is not None
        assert NegativeHeuristic is not None

    def test_all_in_sdk(self):
        from mci_world_model import sdk

        v37_symbols = [
            "MetaDiagnoser",
            "NegativeHeuristic",
            "SurpriseSignal",
            "FailurePattern",
            "ChangeType",
            "ProposedChange",
        ]
        for sym in v37_symbols:
            assert sym in sdk.__all__, f"{sym} not in sdk.__all__"
