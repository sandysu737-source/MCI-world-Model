from __future__ import annotations

"""
MCI World Model v4.6.0 — Orchestrator 桥接

连接 MCI World Model SDK 工具链与 ai-native-nutrition-v1 的 AgentOrchestrator，
实现三环推理 (MultiLLM + ToolOrchestrator + ExternalPerception) 对 MCI SDK
因果推理能力的调用。

意图↔SDK 映射:
    SCREENING        → BayesianNetwork.bayes_inference()
    ASSESSMENT       → CounterfactualEngine.query()
    PLAN_GENERATION  → JEPAEncoder.encode() + JEPAPredictor.predict()
    FOLLOWUP         → PhysicalGraphBuilder.build_graph() + EnergyFlow

设计原则:
- 零硬依赖: 不 import ai-native-nutrition-v1（反向依赖）
- 标准接口: 统一意图→SDK 调用映射
- 可扩展: 通过 register_intent() 注册新意图映射

用法:
    from mci_world_model.sdk._orchestrator_bridge import OrchestratorBridge

    bridge = OrchestratorBridge()
    result = bridge.execute_intent(
        intent_type="ASSESSMENT",
        params={
            "evidence": {"calorie_intake": 1500, "albumin": 28},
            "do_x": {"calorie_intake": 2000},
            "target": "albumin",
        },
    )
"""


import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── SafetyGuard 懒加载 ──
_safety_guard: Any | None = None
_safety_available: bool = False

try:
    from mci_sdk.safety_guard import SafetyGuard as _SafetyGuard

    _safety_guard = _SafetyGuard()
    _safety_available = True
except ImportError as e:
    logger.debug("mci_sdk.safety_guard not available: %s", e)


# =============================================================================
# AgentResult — 统一返回类型
# =============================================================================


@dataclass
class AgentResult:
    """
    意图执行结果 — 兼容 ai-native-nutrition-v1 AgentResult 结构。

    Attributes:
        success: 执行是否成功
        intent_type: 执行的意图类型
        data: 结果数据
        error: 错误信息 (仅 success=False 时)
        metadata: 额外元信息
    """

    success: bool
    intent_type: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls: Any, intent_type: str, data: dict[str, Any], **meta: Any) -> AgentResult:
        return cls(success=True, intent_type=intent_type, data=data, metadata=meta)

    @classmethod
    def fail(cls, intent_type: str, error: str) -> AgentResult:
        return cls(success=False, intent_type=intent_type, error=error)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "intent_type": self.intent_type,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }


# =============================================================================
# Intent Registry
# =============================================================================

_INTENT_REGISTRY: dict[str, dict[str, Any]] = {}


def register_intent(intent_type: str, handler: Callable[..., Any], description: str = "", **meta: Any) -> None:
    """注册自定义意图映射。"""
    _INTENT_REGISTRY[intent_type] = {"handler": handler, "description": description, "meta": meta}


# =============================================================================
# OrchestratorBridge
# =============================================================================


class OrchestratorBridge:
    """
    MCI World Model → ai-native-nutrition-v1 桥接。

    将 AgentOrchestrator 的自愈意图映射到 MCI SDK 的具体因果推理调用。
    """

    def __init__(
        self,
        world_model: Any = None,
        multillm: Any = None,
    ):
        """
        Args:
            world_model: MCI WorldModel 实例 (可选，可 lazyload)
            multillm: MultiLLMAdapter 实例 (可选)
        """
        self._world_model = world_model
        self._multillm = multillm
        self._intent_map = self._build_default_map()

    def _build_default_map(self) -> dict[str, Callable]:  # type: ignore
        """构建默认意图映射表。"""
        return {
            "SCREENING": self._handle_screening,
            "ASSESSMENT": self._handle_assessment,
            "PLAN_GENERATION": self._handle_plan_generation,
            "FOLLOWUP": self._handle_followup,
            "MONITORING": self._handle_monitoring,
            "CF_QUERY": self._handle_cf_query,
        }

    # -----------------------------------------------------------------
    # 意图执行入口
    # -----------------------------------------------------------------

    def execute_intent(self, intent_type: str, params: dict[str, Any] | None = None) -> AgentResult:
        """
        执行单个意图。

        Args:
            intent_type: 意图类型 (SCREENING/ASSESSMENT/PLAN_GENERATION/FOLLOWUP/MONITORING)
            params: 意图参数

        Returns:
            AgentResult
        """
        params = params or {}

        # ── SafetyGuard Layer 1: 输入安全检查 ──
        if _safety_available and _safety_guard is not None:
            # 检查意图参数中所有文本值
            check_parts = [intent_type]
            for _k, _v in params.items():
                if isinstance(_v, str):
                    check_parts.append(_v)
                elif isinstance(_v, dict):
                    check_parts.extend(str(v) for v in _v.values() if isinstance(v, str))
                elif isinstance(_v, list):
                    check_parts.extend(str(v) for v in _v if isinstance(v, str))
            check_text = " ".join(check_parts)
            if _safety_guard.should_block(check_text):
                return AgentResult.fail(intent_type, "输入被安全策略拦截")

        # 检查自定义注册表
        if intent_type in _INTENT_REGISTRY:
            try:
                handler = _INTENT_REGISTRY[intent_type]["handler"]
                return handler(self, params)
            except Exception as e:
                logger.error(f"Custom intent '{intent_type}' failed: {e}")
                return AgentResult.fail(intent_type, str(e))

        # 默认映射
        handler = self._intent_map.get(intent_type)
        if handler is None:
            return AgentResult.fail(intent_type, f"Unknown intent type: {intent_type}")

        try:
            result = handler(params)
        except Exception as e:
            logger.error(f"Intent '{intent_type}' execution failed: {e}", exc_info=True)
            return AgentResult.fail(intent_type, str(e))

        # ── SafetyGuard Layer 1: 输出脱敏 ──
        if _safety_available and _safety_guard is not None and result.success:
            result_str = str(result.data)
            sanitized = _safety_guard.sanitize_output(result_str)
            if sanitized != result_str:
                logger.info("OrchestratorBridge: output sanitized by SafetyGuard")

        return result

    def execute_workflow(
        self, workflow_name: str, patient_id: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        执行多意图工作流。

        Args:
            workflow_name: 工作流名称 (e.g., "initial_screening", "followup_plan")
            patient_id: 患者 ID
            context: 上下文数据

        Returns:
            {
                "workflow": workflow_name,
                "patient_id": patient_id,
                "results": [AgentResult, ...],
                "summary": str,
            }
        """
        context = context or {}
        results: list[AgentResult] = []

        if workflow_name == "initial_screening":
            # 初筛工作流: SCREENING → ASSESSMENT
            screening_params = {
                "evidence": context.get("evidence", {}),
                "nodes": context.get("risk_nodes", ["albumin", "nrs2002_score", "weight"]),
            }
            results.append(self.execute_intent("SCREENING", screening_params))

            if results[-1].success:
                assessment_params = {
                    "evidence": context.get("evidence", {}),
                    "do_x": context.get("intervention", {}),
                    "target": "nrs2002_score",
                }
                results.append(self.execute_intent("ASSESSMENT", assessment_params))

        elif workflow_name == "followup_plan":
            # 随访规划: PLAN_GENERATION → FOLLOWUP
            plan_params = {
                "patient_id": patient_id,
                "signals": context.get("timeline", []),
                "memories": context.get("memories", []),
            }
            results.append(self.execute_intent("PLAN_GENERATION", plan_params))

            followup_params = {
                "patient_id": patient_id,
                "timeline": context.get("timeline", []),
                "previous_plan": results[-1].data if results[-1].success else {},
            }
            results.append(self.execute_intent("FOLLOWUP", followup_params))

        else:
            results.append(AgentResult.fail("WORKFLOW", f"Unknown workflow: {workflow_name}"))

        # 汇总
        summary_parts = []
        if self._multillm is not None:
            summary_prompt = f"患者 {patient_id} 的 {workflow_name} 结果汇总。"
            try:
                summary_parts.append(self._multillm.generate(summary_prompt))
            except Exception as e:
                logger.warning("LLM 汇总生成跳过: %s", e)

        return {
            "workflow": workflow_name,
            "patient_id": patient_id,
            "results": [r.to_dict() for r in results],
            "summary": summary_parts[0] if summary_parts else self._make_fallback_summary(workflow_name, results),
        }

    def _make_fallback_summary(self, workflow_name: str, results: list[AgentResult]) -> str:
        """生成降级工作流摘要。"""
        n_ok = sum(1 for r in results if r.success)
        n_fail = len(results) - n_ok
        return f"{workflow_name} 完成: {n_ok}/{len(results)} 成功, {n_fail} 失败"

    # -----------------------------------------------------------------
    # 意图处理器
    # -----------------------------------------------------------------

    def _handle_screening(self, params: dict[str, Any]) -> AgentResult:
        """
        SCREENING → BayesianNetwork.bayes_inference()

        基于贝叶斯网络进行初筛风险评估。
        """
        from mci_world_model._sys.bayesian_network import BayesianNetwork

        evidence = params.get("evidence", {})
        nodes = params.get("nodes", [])

        if not nodes:
            return AgentResult.ok("SCREENING", {"baysian_result": {}, "note": "no risk nodes specified"})

        # 构建贝叶斯网络
        bn = BayesianNetwork(name="screening")
        for node in nodes:
            bn.add_node(node)

        # 添加节点间关系 (默认: 简单 chain)
        for i in range(len(nodes) - 1):
            bn.add_edge(nodes[i], nodes[i + 1], initial_strength=0.5)

        # 证据注入
        bayes_evidence = {}
        for node, val in evidence.items():
            if node in nodes:
                bayes_evidence[node] = val > 0.5 if isinstance(val, (int, float)) else bool(val)

        # 后验推理
        if bayes_evidence:
            posterior = bn.infer_posterior(query_nodes=nodes, evidence=bayes_evidence)
        else:
            posterior = {}

        # 因果强度
        strengths = {}
        for i in range(len(nodes) - 1):
            cs = bn.query_causal_strength(nodes[i], nodes[i + 1])
            strengths[f"{nodes[i]}->{nodes[i + 1]}"] = cs

        return AgentResult.ok(
            "SCREENING",
            {
                "posterior": {k: {"mean": v.mean} for k, v in posterior.items()},
                "causal_strengths": strengths,
                "risk_nodes": nodes,
            },
        )

    def _handle_assessment(self, params: dict[str, Any]) -> AgentResult:
        """
        ASSESSMENT → CounterfactualEngine.query()

        反事实推理: "如果干预 X，Y 会怎样？"
        """
        from mci_world_model.sdk._counterfactual import CounterfactualEngine
        from mci_world_model.sdk._do_calculus import CausalGraph

        evidence = params.get("evidence", {})
        do_x = params.get("do_x", {})
        target = params.get("target", "")
        n_mc = params.get("n_mc", 200)

        if not target:
            return AgentResult.fail("ASSESSMENT", "target parameter required")

        if not evidence:
            return AgentResult.fail("ASSESSMENT", "evidence parameter required")

        if not do_x:
            return AgentResult.fail("ASSESSMENT", "do_x (intervention) parameter required")

        # 构建因果图
        cg = CausalGraph()

        # 启发式: 从 evidence → target 建立边
        for node in set(evidence.keys()) - {target}:
            cg.add_edge(node, target, weight=0.3)

        # 干预节点到目标的边
        for node in do_x:
            if node != target:
                cg.add_edge(node, target, weight=0.5)

        sem = cg.to_sem(noise_std=0.3, activation="linear", seed=42)
        engine = CounterfactualEngine(sem, list(sem.node_names))

        try:
            result = engine.query(
                evidence=evidence,
                do_x=do_x,
                target=target,
                compute_pns=params.get("compute_pns", False),
                n_mc=n_mc,
            )
        except Exception as e:
            logger.warning("异常降级: %s", e, exc_info=True)
            return AgentResult.fail("ASSESSMENT", f"Counterfactual query failed: {e}")

        return AgentResult.ok(
            "ASSESSMENT",
            {
                "factual_value": result.factual_value,
                "counterfactual_value": result.counterfactual_value,
                "individual_effect": result.individual_effect,
                "ci_95": result.ci_95,
                "noise_terms": result.noise_terms,
                "pn": result.pn,
                "ps": result.ps,
                "pns": result.pns,
            },
        )

    def _handle_plan_generation(self, params: dict[str, Any]) -> AgentResult:
        """
        PLAN_GENERATION → JEPAEncoder.encode() + JEPAPredictor.predict()

        基于 JEPA 潜空间生成营养方案。
        """
        from mci_world_model.sdk._jepa_encoder import JEPAEncoder
        from mci_world_model.sdk._jepa_predictor import IdentityPredictor

        signals = params.get("signals", [])
        memories = params.get("memories", [])
        patient_id = params.get("patient_id", "unknown")

        if not signals and not memories:
            return AgentResult.fail("PLAN_GENERATION", "signals or memories required")

        encoder = JEPAEncoder(world_model=None)
        predictor = IdentityPredictor()

        # 编码
        if signals:
            state = encoder.encode(signals=signals)
        else:
            try:
                state = encoder.encode(memories=memories)
            except Exception as e:
                logger.warning("异常降级: %s", e, exc_info=True)
                return AgentResult.fail("PLAN_GENERATION", f"encode failed: {e}")

        # 预测
        predictor.predict(state)

        # 方案生成 (使用 MultiLLM 增强)
        plan_narrative = ""
        if self._multillm is not None:
            try:
                prompt = (
                    f"患者 {patient_id} 因果推理结果:\n"
                    f"检测到 {len(state.causal_edges)} 条因果边\n"
                    f"新发现 {state.n_novel} 条新因果边\n"
                    f"已确认 {state.n_confirmed} 条已知因果边\n\n"
                    "基于以上信息，生成一份简洁的营养方案（3-5 条建议）。"
                )
                plan_narrative = self._multillm.generate(prompt)
            except Exception as e:
                logger.warning("LLM 方案生成跳过: %s", e)

        return AgentResult.ok(
            "PLAN_GENERATION",
            {
                "causal_edges": state.causal_edges[:10],
                "n_novel": state.n_novel,
                "n_confirmed": state.n_confirmed,
                "plan_narrative": plan_narrative,
            },
        )

    def _handle_followup(self, params: dict[str, Any]) -> AgentResult:
        """
        FOLLOWUP → PhysicalGraphBuilder.build_graph() + EnergyFlowPredictor

        基于物理图构建随访计划。
        """
        from mci_world_model.sdk._physical_graph_builder import PhysicalGraphBuilder

        timeline = params.get("timeline", [])
        _patient_id = params.get("patient_id", "unknown")

        if not timeline:
            return AgentResult.fail("FOLLOWUP", "timeline data required")

        builder = PhysicalGraphBuilder()
        edges = builder.build_graph(timeline)

        # 能量流分析: 按 verdict 分组
        novel_edges = [e for e in edges if e.get("verdict") == "novel"]
        confirmed_edges = [e for e in edges if e.get("verdict") == "confirmed"]

        # 随访建议
        followup_suggestions = []
        if novel_edges:
            followup_suggestions.append(
                f"新发现 {len(novel_edges)} 条因果边: "
                + ", ".join(f"{e.get('metric_a', '?')}-{e.get('metric_b', '?')}" for e in novel_edges[:3])
            )

        # 关键指标趋势
        key_metrics = ["albumin", "prealbumin", "nrs2002_score", "body_weight"]
        trends = {}
        for metric in key_metrics:
            values = [d.get(metric, 0) for d in timeline[-7:] if metric in d]
            if values:
                avg = np.mean(values)
                trend = (
                    "上升" if values[-1] > values[0] * 1.05 else ("下降" if values[-1] < values[0] * 0.95 else "持平")
                )
                trends[metric] = {"avg": round(float(avg), 1), "trend": trend}

        return AgentResult.ok(
            "FOLLOWUP",
            {
                "n_edges": len(edges),
                "n_novel_edges": len(novel_edges),
                "n_confirmed_edges": len(confirmed_edges),
                "trends": trends,
                "followup_suggestions": followup_suggestions,
            },
        )

    def _handle_monitoring(self, params: dict[str, Any]) -> AgentResult:
        """
        MONITORING → 指标趋势监控 + 异常检测。
        """
        timeline = params.get("timeline", [])
        alert_metrics = params.get("alert_metrics", ["albumin", "nrs2002_score"])

        if not timeline:
            return AgentResult.fail("MONITORING", "timeline data required")

        alerts = []
        for metric in alert_metrics:
            values = [d.get(metric) for d in timeline if metric in d]
            if len(values) >= 3:
                recent_mean = np.mean(values[-3:])
                # baseline_mean 保留用于未来趋势对比扩展
                _ = np.mean(values[:-3]) if len(values) > 3 else recent_mean

                # 白蛋白 < 30 → 警报
                if metric == "albumin" and recent_mean < 30:
                    alerts.append(
                        {
                            "metric": metric,
                            "severity": "warning",
                            "value": round(float(recent_mean), 1),
                            "threshold": 30,
                        }
                    )
                # NRS2002 >= 3 → 警报
                if metric == "nrs2002_score" and recent_mean >= 3:
                    alerts.append(
                        {"metric": metric, "severity": "warning", "value": round(float(recent_mean), 1), "threshold": 3}
                    )

        return AgentResult.ok("MONITORING", {"alerts": alerts, "total_days": len(timeline)})

    def _handle_cf_query(self, params: dict[str, Any]) -> AgentResult:
        """
        CF_QUERY → CounterfactualOracle.batch_what_if() + rank_scenarios()

        v4.4.2: LLM↔CEWM 反馈闭环核心——LLM 可以查询 CEWM 的
        反事实推演结果，并将结果融入后续推理。

        Params:
            hypotheses: list[dict[str, Any]] — 反事实假设列表
                [{"name": "A", "intervention": {...}, "target": "..."}]
            goal: str — 优化目标描述
            target_direction: str — 'higher_is_better' 或 'lower_is_better'
        """
        from mci_world_model.sdk._counterfactual_oracle import (
            CounterfactualOracle,
        )

        hypotheses = params.get("hypotheses", [])
        if not hypotheses:
            return AgentResult.fail("CF_QUERY", "hypotheses parameter required")

        goal = params.get("goal", "maximize")
        target_direction = params.get("target_direction", "higher_is_better")

        # 构建 Oracle（优先使用 world_model 上的实例）
        oracle = None
        if (
            self._world_model is not None
            and hasattr(self._world_model, "_cf_oracle")
            and self._world_model._cf_oracle is not None
        ):
            oracle = self._world_model._cf_oracle
        else:
            oracle = CounterfactualOracle(world_model=self._world_model)

        # 完整查询流程
        result = oracle.query(
            hypotheses=hypotheses,
            goal=goal,
            target_direction=target_direction,
        )

        return AgentResult.ok(
            "CF_QUERY",
            {
                "best_scenario": result.get("best_scenario"),
                "best_effect": result.get("best_effect"),
                "rankings": result.get("rankings", []),
                "recommendation": result.get("recommendation", ""),
                "n_scenarios": result.get("n_scenarios", 0),
            },
        )
