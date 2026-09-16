from __future__ import annotations

"""MCI World Model 认知组件边界。

本模块只承载认知闭环、失败诊断、经验检索、惊奇检测、行动规划、
反思数据合成、多样性评估与启发式检查，不承载健康诊断或世界模型编排。
"""

from typing import TYPE_CHECKING, Any


class CognitiveComponentsMixin:
    """认知组件行为 Mixin。

    宿主类必须提供各认知组件槽位，并通过 CEWM Mixin 提供状态解析。
    """

    if TYPE_CHECKING:
        _cognitive_loop: Any | None
        _meta_diagnoser: Any | None
        _surprise_detector: Any | None
        _experience_db: Any | None
        _multi_view_retriever: Any | None
        _action_conditioned_predictor: Any | None
        _cost_module: Any | None
        _multi_branch_predictor: Any | None
        _plan_agent: Any | None
        _reflection_synthesizer: Any | None
        _cognitive_diversity: Any | None
        _negative_heuristic: Any | None

        def _cewm_parse_state(self, obs: Any) -> Any: ...

    def run_cognitive_loop(
        self,
        layer_errors: dict[str, float] | None = None,
        n_rounds: int = 1,
    ) -> dict[str, Any]:
        """Wiener 四环认知闭环传播（v4.3.0）。

        收集各层误差信号并执行跨层传播，输出参数调整量。

        Args:
            layer_errors: 各层误差信号 {"perception": 0.5, "cognition": 0.3, ...}
            n_rounds: 传播轮数（默认 1）

        Returns:
            传播结果: {"total_energy", "converged", "deltas", "health"}
        """
        from mci_world_model.sdk._cognitive_loop import (
            CognitiveLayer,
            CognitiveLoopBus,
        )

        if self._cognitive_loop is None:
            self._cognitive_loop = CognitiveLoopBus()

        bus = self._cognitive_loop

        # 注入误差信号
        if layer_errors:
            layer_map = {
                "perception": CognitiveLayer.PERCEPTION,
                "cognition": CognitiveLayer.COGNITION,
                "prediction": CognitiveLayer.PREDICTION,
                "action": CognitiveLayer.ACTION,
            }
            for name, magnitude in layer_errors.items():
                layer = layer_map.get(name)
                if layer is not None:
                    bus.inject_error(layer, magnitude=float(magnitude))

        # 传播
        if n_rounds <= 1:
            prop = bus.propagate()
            results = [prop]
        else:
            results = bus.propagate_n(n_rounds, early_stop=False)

        last = results[-1]
        health = bus.health_report()

        return {
            "total_energy": last.total_energy,
            "converged": last.converged,
            "deltas": {layer.name: float(d[0]) for layer, d in last.deltas.items()},
            "health": {
                "bottleneck_layer": health.bottleneck_layer.name if health.bottleneck_layer else None,
                "overall_health": health.overall_health,
                "oscillation_detected": health.oscillation_detected,
            },
        }

    def diagnose_failure(
        self,
        surprise_signals: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """认知失败诊断（v4.3.0 MetaDiagnoser 集成）。

        基于惊奇信号匹配已知失败模式，追溯根因链。

        Args:
            surprise_signals: 惊奇信号列表
            context: 附加上下文

        Returns:
            诊断结果 dict: {"pattern", "severity", "confidence", "recommendation"}
        """
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser

        if self._meta_diagnoser is None:
            self._meta_diagnoser = MetaDiagnoser()

        if not surprise_signals:
            # v4.3.1: 优先从 SurpriseDetector 提取信号
            if self._surprise_detector is not None:
                stats = self._surprise_detector.running_statistics()
                surprise_signals = [
                    {
                        "state_distance": stats.get("mean_surprise", 0.0),
                        "vector_deviation": stats.get("surprise_std", 0.0),
                        "direction_error": stats.get("anomaly_rate", 0.0),
                    }
                ]
            # 从认知闭环提取误差信号
            elif self._cognitive_loop is not None:
                bus = self._cognitive_loop
                stats = bus.running_statistics()
                surprise_signals = [
                    {
                        "state_distance": stats.get("mean_energy", 0.0),
                        "vector_deviation": stats.get("energy_std", 0.0),
                        "direction_error": stats.get("max_layer_energy", 0.0),
                    }
                ]
            else:
                return {"pattern": None, "severity": 0.0, "recommendation": "无信号可诊断"}

        result = self._meta_diagnoser.diagnose(surprise_signals, context=context)

        return {
            "pattern": result.pattern.name if result.pattern else None,
            "severity": result.severity.value if hasattr(result.severity, "value") else result.severity,
            "confidence": result.confidence,
            "recommendation": result.recommendation,
            "root_cause_chain": result.root_cause_chain.chain if result.root_cause_chain else [],
            "health_scores": result.health_scores or {},
        }

    def retrieve_experiences(
        self,
        tags: list[str] | None = None,
        causal_edges: list[tuple[str, str]] | None = None,
        context: dict[str, str] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """五维融合经验检索（v4.3.0 MultiViewRetriever 集成）。

        综合语义/因果/时间/上下文/结构五维视角检索历史经验。

        Args:
            tags: 语义标签
            causal_edges: 因果边列表
            context: 上下文信息
            top_k: 返回数量

        Returns:
            按综合分数降序排列的检索结果列表
        """
        from mci_world_model.sdk._experience_memory import ExperienceDB
        from mci_world_model.sdk._multi_view_retriever import (
            MultiViewRetriever,
            QuerySpec,
        )

        if self._multi_view_retriever is None:
            # 复用已有 ExperienceDB 或创建新的
            if hasattr(self, "_experience_db") and self._experience_db is not None:
                exp_db = self._experience_db
            else:
                exp_db = ExperienceDB()
            self._multi_view_retriever = MultiViewRetriever(experience_db=exp_db)

        query = QuerySpec(
            tags=tags or [],
            causal_edges=causal_edges or [],
            context=context or {},
        )

        results = self._multi_view_retriever.retrieve(query, top_k=top_k)

        return [
            {
                "experience_id": r.experience.experience_id
                if hasattr(r.experience, "experience_id")
                else str(r.experience),
                "score": r.score,
                "view_scores": r.view_scores,
                "strategy": r.strategy,
            }
            for r in results
        ]

    def detect_surprise(
        self,
        predicted: Any = None,
        actual: Any = None,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """惊奇误差检测（v4.3.1 SurpriseDetector 集成）。

        量化预测状态与实际观测状态的偏差，输出惊奇信号。
        作为 diagnose_failure() 的前置量化步骤。

        Args:
            predicted: 预测状态 (WorldState/dict/PendulumState)
            actual: 实际观测状态
            threshold: 惊奇度阈值 [0, 1]

        Returns:
            {"score": float, "is_anomaly": bool, "breakdown": dict[str, Any], "stats": dict}
        """
        from mci_world_model.sdk._surprise_detector import SurpriseDetector

        if self._surprise_detector is None:
            self._surprise_detector = SurpriseDetector(threshold=threshold)
        elif abs(self._surprise_detector.threshold - threshold) > 1e-6:
            self._surprise_detector.threshold = threshold

        # 解析状态
        pred_state = self._cewm_parse_state(predicted)
        actual_state = self._cewm_parse_state(actual)

        if pred_state is None or actual_state is None:
            return {
                "score": 0.0,
                "is_anomaly": False,
                "breakdown": {},
                "stats": {},
                "note": "insufficient_state_input",
            }

        signal = self._surprise_detector.compute_surprise(pred_state, actual_state)
        stats = self._surprise_detector.running_statistics()

        return {
            "score": signal.score,
            "is_anomaly": signal.is_anomaly,
            "threshold": signal.threshold,
            "breakdown": signal.breakdown,
            "stats": stats,
        }

    def plan_action(
        self,
        current: Any = None,
        goal: Any = None,
        max_horizon: int = 5,
        n_branches: int = 3,
        predictor_backend: str = "auto",
    ) -> dict[str, Any]:
        """因果决策前置规划（v4.4.0 PlanAgent 泛化集成）。

        '先模拟后执行'模式：候选动作 → 多分支推演 → 选最优 Plan。

        v4.4.0: 支持任意 WorldState 类型，通过 predictor_backend 参数
        选择预测器后端。'auto' 模式自动根据状态类型选择预测器。

        Args:
            current: 当前状态 (WorldState/dict/PendulumState/CartState)
            goal: 目标状态
            max_horizon: 最大规划步数
            n_branches: 分支数
            predictor_backend: 预测器后端 ('auto'/'pendulum'/'cart')

        Returns:
            Plan dict: {"actions": [...], "expected_cost": float, "confidence": float, ...}
        """
        from mci_world_model.sdk._action_conditioned_predictor import (
            CartPhysicsPredictor,
            PendulumPhysicsPredictor,
        )
        from mci_world_model.sdk._plan_agent import PlanAgent

        cur = self._cewm_parse_state(current)
        gl = self._cewm_parse_state(goal)

        if cur is None or gl is None:
            return {
                "status": "insufficient_state",
                "plan": None,
                "actions": [],
                "expected_cost": 0.0,
                "confidence": 0.0,
            }

        if self._plan_agent is None:
            # v4.4.0: 根据 predictor_backend 或状态类型选择预测器
            if self._action_conditioned_predictor is None:
                if predictor_backend == "auto":
                    # 自动选择: 检查状态类型
                    if hasattr(cur, "x") and hasattr(cur, "v") and not hasattr(cur, "theta"):
                        self._action_conditioned_predictor = CartPhysicsPredictor()
                    else:
                        self._action_conditioned_predictor = PendulumPhysicsPredictor()
                elif predictor_backend == "cart":
                    self._action_conditioned_predictor = CartPhysicsPredictor()
                else:
                    self._action_conditioned_predictor = PendulumPhysicsPredictor()
            if self._multi_branch_predictor is None:
                from mci_world_model.sdk._multi_branch_predictor import MultiBranchPredictor

                self._multi_branch_predictor = MultiBranchPredictor(self._action_conditioned_predictor)
            self._plan_agent = PlanAgent(
                predictor=self._action_conditioned_predictor,
                cost_module=self._cost_module,
                multi_branch=self._multi_branch_predictor,
                surprise_detector=self._surprise_detector,
            )

        plan = self._plan_agent.plan(cur, gl, max_horizon=max_horizon, n_branches=n_branches)
        return plan.to_dict()

    def synthesize_training_data(
        self,
        memories: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """因果 QA 训练数据合成（v4.3.2 ReflectionSynthesizer 集成）。

        基于 MEMO 框架从因果记忆生成训练 QA 对。

        Returns:
            {"qa_pairs": [...], "n_pairs": int, "report": dict[str, Any], "ready": bool}
        """
        from mci_world_model.sdk._reflection_synthesizer import ReflectionSynthesizer

        if self._reflection_synthesizer is None:
            self._reflection_synthesizer = ReflectionSynthesizer()

        if not memories:
            return {"qa_pairs": [], "n_pairs": 0, "report": {}, "ready": False}

        pairs, prior = self._reflection_synthesizer.run_pipeline(memories)
        report = self._reflection_synthesizer.training_data_report(pairs)

        return {
            "qa_pairs": [
                {
                    "cause": p.cause_text,
                    "effect": p.effect_text,
                    "confidence": p.confidence,
                    "energy_relation": p.energy_relation,
                }
                for p in pairs
            ],
            "n_pairs": len(pairs),
            "prior_matrix_shape": list(prior.shape) if prior is not None else None,
            "report": report,
            "ready": report.get("ready_for_training", False),
        }

    def assess_diversity(
        self,
        states: list | None = None,  # type: ignore
        prediction_errors: list[float] | None = None,
    ) -> dict[str, Any]:
        """五维认知多样性评估（v4.3.2 CognitiveDiversity 集成）。

        Returns:
            {"diversity_vector": dict[str, Any], "ashby_satisfied": bool}
        """
        from mci_world_model.sdk._cognitive_diversity import CognitiveDiversity

        if self._cognitive_diversity is None:
            self._cognitive_diversity = CognitiveDiversity()

        dv = self._cognitive_diversity.compute(
            states=states,
            prediction_errors=prediction_errors,
        )
        return {
            "diversity_vector": dv.to_dict(),
            "ashby_satisfied": dv.to_dict()["ashby_satisfied"],
            "ashby_ratio": dv.to_dict()["ashby_ratio"],
        }

    def check_admissibility(
        self,
        change: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """硬核规则可接受性检查（v4.3.2 NegativeHeuristic 集成）。

        Lakatos 研究纲领框架：check if a proposed change violates hard-core rules.

        Returns:
            {"admissible": bool, "violations": list[Any], "suggestions": list}
        """
        from mci_world_model.sdk._negative_heuristic import (
            NegativeHeuristic,
            ProposedChange,
        )

        if self._negative_heuristic is None:
            self._negative_heuristic = NegativeHeuristic()

        if change is None:
            return {"admissible": True, "violations": [], "suggestions": [], "status": "no_change_provided"}

        pc = ProposedChange(**change) if isinstance(change, dict) else change
        violations = self._negative_heuristic.violations(pc)
        # protective_belt_suggestions 基于诊断结果 (dict), 非 ProposedChange
        suggestions = self._negative_heuristic.protective_belt_suggestions()

        return {
            "admissible": self._negative_heuristic.is_admissible(pc),
            "violations": [
                {
                    "rule_id": v.rule_id,
                    "rule_name": v.rule_name,
                    "severity": v.severity.name if hasattr(v.severity, "name") else str(v.severity),
                    "description": v.description,
                }
                for v in violations
            ],
            "suggestions": [
                {
                    "target": s.target,
                    "action": s.action,
                    "priority": s.priority,
                    "rationale": s.rationale,
                }
                for s in suggestions
            ],
            "hard_core_status": self._negative_heuristic.hard_core_status(),
        }
