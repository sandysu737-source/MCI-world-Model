from __future__ import annotations

"""MCI World Model 因果推理边界。

本模块只承载 Pearl L2 干预、L3 反事实、因果效应分解与因果链解释，
不承载因果发现、JEPA、CEWM 或健康诊断逻辑。
"""

import logging
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from mci_world_model.sdk._world_model_state import CausalWorldModelState

logger = logging.getLogger(__name__)


class CausalInferenceMixin:
    """因果推理行为 Mixin。

    宿主类必须提供因果状态、DoCalculus 懒加载槽位、
    干预历史及其同步锁。
    """

    if TYPE_CHECKING:
        _state: CausalWorldModelState
        _do_calculus: Any | None
        _do_calculus_lock: threading.Lock
        _intervention_history: list[dict[str, Any]]
        _intervention_history_lock: threading.Lock

    def _build_causal_graph_from_state(self) -> object | None:
        """
        从当前因果边构建 CausalGraph。

        统一 intervene()/decompose_effect()/query_counterfactual()
        的图构建逻辑，消除重复。

        Returns:
            CausalGraph 实例或 None
        """
        if not self._state or not self._state.causal_edges:
            return None
        from mci_world_model.sdk._do_calculus import DoCalculus

        n_nodes = max(max(e.get("cause_idx", 0), e.get("effect_idx", 0)) for e in self._state.causal_edges) + 1
        return DoCalculus.build_from_gaussian_dag(self._state.causal_edges, n_nodes)

    def intervene(
        self,
        state: str = "current",
        do_x: dict | None = None,  # type: ignore
        target: str | None = None,
        method: str = "auto",
    ) -> dict[str, Any]:
        """
        Pearl do-operator 干预预测（Pearl L2 完整实现）。

        计算公式:
            P(Y | do(X=x)) = Σ_z P(Y | X=x, Z=z) · P(Z=z)

        工作流:
        1. 从当前因果图构建 CausalGraph
        2. 识别调整变量集 (后门准则)
        3. 估计 ATE (平均处理效应)
        4. 返回干预分析结果

        Args:
            state: 世界状态标识
            do_x: 干预 {"变量名": 干预值}
            target: 目标变量名
            method: "auto"|"backdoor"|"frontdoor"

        Returns:
            InterventionResult 字典
        """
        if do_x is None or target is None:
            return {
                "status": "insufficient_input",
                "code": 422,
                "message": "需要 do_x 和 target 参数",
            }
        if not isinstance(do_x, dict) or not do_x:
            return {
                "status": "unsupported",
                "code": 422,
                "reason": "invalid_do_x",
                "message": "do_x 必须是非空 dict",
            }

        # ── 懒加载 DoCalculus 引擎 ──
        if self._do_calculus is None:
            with self._do_calculus_lock:
                if self._do_calculus is None:
                    try:
                        from mci_world_model.sdk._do_calculus import DoCalculus

                        self._do_calculus = DoCalculus()
                    except ImportError:
                        return {
                            "status": "error",
                            "message": "DoCalculus 引擎不可用",
                        }

        # ── 从因果边构建 CausalGraph ──
        try:
            from mci_world_model.sdk._do_calculus import CausalGraph

            cg = self._build_causal_graph_from_state()
            if cg is None:
                cg = CausalGraph(
                    nodes=[*list(do_x.keys()), target],
                    edges=[],
                )
        except (ValueError, KeyError, TypeError):
            logger.warning("CausalGraph 构建失败，回退到默认空图", exc_info=True)
            cg = CausalGraph(
                nodes=[*list(do_x.keys()), target],
                edges=[],
            )

        self._do_calculus.set_graph(cg)  # type: ignore[arg-type]

        # ── 干预契约：先统一拒绝多变量和无效值，避免 dict 键序决定语义 ──
        try:
            normalized_do_x = {str(name): float(value) for name, value in do_x.items()}
        except (TypeError, ValueError):
            return {
                "status": "rejected",
                "code": 422,
                "reason": "invalid_do_x",
                "message": "do_x 的变量名必须可解析，干预值必须是数值",
            }
        if not np.all(np.isfinite(np.fromiter(normalized_do_x.values(), dtype=np.float64))):
            return {
                "status": "rejected",
                "code": 422,
                "reason": "non_finite_do_x",
                "message": "intervention values must be finite (NaN/Inf rejected)",
            }
        if len(normalized_do_x) != 1:
            return {
                "status": "unsupported",
                "code": 422,
                "reason": "multivariate_do_not_implemented",
                "message": "多变量干预尚未实现；当前只支持单变量 do({变量: 数值})",
            }

        x_name = next(iter(normalized_do_x))
        x_value = normalized_do_x[x_name]

        try:
            result = self._do_calculus.estimate_intervention(
                do_x={x_name: x_value},
                target=target,
                x_baseline=0.0,
                method=method,
            )
        except (ValueError, KeyError, RuntimeError) as e:
            logger.error("干预分析失败: %s", e)
            return {
                "status": "error",
                "message": f"干预分析失败: {e}",
            }

        # ── 记录干预历史 ──
        intervention_record = {
            "state": state,
            "do": do_x,
            "target": target,
            "result": result.to_dict(),
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }
        with self._intervention_history_lock:
            self._intervention_history.append(intervention_record)
            self._state.do_interventions.append(intervention_record)

        # ── 构建反事实图 (干预边被切断) ──
        try:
            if cg.n_nodes > 0 and x_name in cg.nodes:  # type: ignore
                x_idx = cg.node_index(x_name)  # type: ignore[attr-defined]
                if x_idx is not None and cg.adjacency is not None:  # type: ignore[attr-defined]
                    cf_adj = cg.adjacency.copy()  # type: ignore
                    # 切断所有指向 X 的边 (do-operator 语义)
                    cf_adj[:, x_idx] = 0.0
                    self._state.counterfactual_graph = {
                        "nodes": list(cg.nodes),  # type: ignore
                        "cf_adjacency": cf_adj.tolist(),
                        "intervention": do_x,
                    }
        except (ValueError, KeyError, TypeError):
            logger.warning("反事实图构建失败，跳过", exc_info=True)

        # ── 返回结果 ──
        output = result.to_dict()
        if result.method == "no_data":
            output["status"] = "no_data"
            output["code"] = 422
        elif result.method in {"rejected", "unsupported"}:
            output["status"] = result.method
            output["code"] = 422
        else:
            output["status"] = "ok"
        output["history_count"] = len(self._intervention_history)
        return output

    def decompose_effect(
        self,
        cause: str,
        effect: str,
        mediator: str | None = None,
    ) -> dict[str, Any]:
        """
        因果效应三分解:
        - NDE: 自然直接效应 (Natural Direct Effect)
        - NIE: 自然间接效应 (Natural Indirect Effect)
        - TE:  总效应 (Total Effect = NDE + NIE)

        使用 Pearl 的 mediation formula:
            NDE = E[Y_{x,M_{x*}} - Y_{x*}]
            NIE = E[Y_{x,M_x} - Y_{x,M_{x*}}]

        Args:
            cause: 原因变量
            effect: 结果变量
            mediator: 中介变量 (None 时自动检测)

        Returns:
            {"nde": float, "nie": float, "te": float, "mediator": str, ...}
        """
        # ── 自动检测中介变量 ──
        if mediator is None:
            if self._do_calculus is None:
                with self._do_calculus_lock:
                    if self._do_calculus is None:
                        try:
                            from mci_world_model.sdk._do_calculus import DoCalculus

                            self._do_calculus = DoCalculus()
                        except ImportError:
                            return {
                                "nde": 0.0,
                                "nie": 0.0,
                                "te": 0.0,
                                "mediator": None,
                                "note": "do_calculus_unavailable",
                            }

            # 尝试从因果图中识别中介
            if self._state and self._state.causal_edges:
                try:
                    cg = self._build_causal_graph_from_state()
                    if cg is not None:
                        mediators = cg.get_mediators(cause, effect)  # type: ignore
                        mediator = mediators[0] if mediators else None
                    else:
                        mediator = None
                except (ValueError, KeyError):
                    logger.warning("中介变量识别失败，回退为无中介", exc_info=True)
                    mediator = None

        if mediator is None:
            return {
                "nde": 0.0,
                "nie": 0.0,
                "te": 0.0,
                "mediator": None,
                "note": "no_mediator_identified",
            }

        # ── 使用 do-calculus 分解 ──
        try:
            # 直接效应: do(mediator) 固定时的 cause → effect
            direct_result = self.intervene(
                do_x={cause: 1.0},
                target=effect,
                method="direct",
            )
            nde = direct_result.get("ate", 0.0)

            # 间接效应: cause → mediator 的 ATE × mediator → effect 的 ATE
            cause_to_med = self.intervene(
                do_x={cause: 1.0},
                target=mediator,
                method="direct",
            )
            med_to_eff = self.intervene(
                do_x={mediator: 1.0},
                target=effect,
                method="direct",
            )
            nie = cause_to_med.get("ate", 0.0) * med_to_eff.get("ate", 0.0)
        except (ValueError, KeyError, RuntimeError) as e:
            logger.error("因果分解失败: %s", e)
            nde = 0.0
            nie = 0.0

        te = nde + nie

        return {
            "nde": round(nde, 6),
            "nie": round(nie, 6),
            "te": round(te, 6),
            "mediator": mediator,
            "nde_pct": round(abs(nde) / max(abs(te), 1e-10) * 100, 1),
            "nie_pct": round(abs(nie) / max(abs(te), 1e-10) * 100, 1),
            "method": "mediation_formula",
        }

    def query_counterfactual(
        self,
        evidence: dict[str, float],
        do_x: dict[str, float],
        target: str,
        compute_pns: bool = True,
    ) -> dict[str, Any]:
        """
        Pearl 三步反事实推理（v3.0.8 L3 新增）。

        基于当前因果图，回答反事实问题:
            "如果当初 X=x' 而非 X=x，Y 会是多少？"

        三步算法:
            1. Abduction (溯因): 从事实证据推断不可观测噪声
            2. Action (干预): 用 do(X=x') 构建 mutilated graph
            3. Prediction (预测): 用溯因噪声 + mutilated graph 计算 Y_{x'}

        输出:
            - counterfactual_value: 反事实结果
            - individual_effect: 个体因果效应 (Y_{x'} - Y)
            - PN/PS/PNS: 必然性/充分性概率
            - noise_terms: 溯因推断的噪声项

        Args:
            evidence: 事实证据 {"X": 1.0, "Y": 3.0, ...}
            do_x: 反事实干预 {"X": 0.0}
            target: 目标变量 (反事实结果)
            compute_pns: 是否计算 PN/PS/PNS

        Returns:
            CounterfactualResult dict

        Example:
            >>> wm = MCIWorldModel(su_lite_pro)
            >>> wm.discover()
            >>> result = wm.query_counterfactual(
            ...     evidence={"手术量": 100, "收入": 50},
            ...     do_x={"手术量": 80},
            ...     target="收入",
            ... )
            >>> print(f"反事实收入: {result['counterfactual_value']}")
        """
        try:
            from mci_world_model.sdk._counterfactual import CounterfactualEngine
            from mci_world_model.sdk._do_calculus import CausalGraph
        except ImportError as e:
            return {
                "status": "error",
                "note": f"counterfactual_engine_unavailable: {e}",
            }

        # ── 从因果图构建 CausalGraph ──
        cg = self._build_causal_graph_from_state()
        if cg is None:
            all_nodes = list(evidence.keys()) + list(do_x.keys()) + [target]
            cg = CausalGraph(nodes=list(set(all_nodes)), edges=[])
        if not getattr(cg, "is_causal_graph", False):
            return {
                "status": "rejected",
                "reason": "association_graph_not_causal",
                "message": "缺少显式定向证据的关联图不能执行反事实推理",
            }

        # ── 构建反事实引擎并查询 ──
        engine = CounterfactualEngine.from_causal_graph(cg)
        if engine is None:
            return {
                "status": "error",
                "note": "failed_to_build_counterfactual_engine",
            }

        try:
            result = engine.query(
                evidence=evidence,
                do_x=do_x,
                target=target,
                compute_pns=compute_pns,
            )
        except (ValueError, KeyError, RuntimeError) as e:
            logger.error("反事实查询失败: %s", e, exc_info=True)
            return {
                "status": "error",
                "note": f"counterfactual_query_failed: {e}",
            }

        return result.to_dict()

    def explain(
        self,
        query: str,
        max_depth: int = 3,
    ) -> dict[str, Any]:
        """
        因果链回溯解释。

        返回从 query 出发的因果路径，
        含每一步的置信度和能量关系类型。

        Args:
            query: 查询文本
            max_depth: 最大因果跳数

        Returns:
            {
                "query": str,
                "chains": [{"path": [...], "confidence": float}, ...],
                "summary": str,
            }
        """
        if not self._state.causal_edges:
            return {
                "query": query,
                "chains": [],
                "summary": "暂无因果图数据。请先运行 discover()。",
            }

        chains = self._trace_causal_chains(query, max_depth)
        summary = self._generate_explanation_summary(chains, query)

        return {
            "query": query,
            "chains": chains[:5],  # 最多 5 条链
            "summary": summary,
        }

    def _trace_causal_chains(self, query: str, max_depth: int) -> list[dict[str, Any]]:
        """追踪因果链 — v4.3.1 多跳 BFS 回溯。

        从匹配 query 的因果边出发，沿 effect→cause 方向链式回溯，
        构建长度 ≤ max_depth 的多跳因果链（A→B→C→...）。
        对每个节点尝试作为中间节点继续扩展，实现级联因果追溯。
        """
        edges = self._state.causal_edges
        if not edges:
            return []

        # 1. 统一节点名：优先 "cause"/"effect" 字符串，否则用 idx 生成标签
        def _node_name(e: dict[str, Any], role: str) -> str:
            s = e.get(role)  # "cause" or "effect"
            if isinstance(s, str):
                return s
            idx = e.get(f"{role}_idx")
            if idx is not None:
                return f"节点 {idx}"
            return f"{role}_{id(e)}"

        # 2. 构建邻接表: cause → [(effect_name, edge), ...]
        adj: dict[str, list[tuple[str, dict]]] = {}  # type: ignore
        for e in edges:
            cause = _node_name(e, "cause")
            effect = _node_name(e, "effect")
            adj.setdefault(cause, []).append((effect, e))

        # 3. 匹配 query — 模糊查找起始节点
        query_lower = query.lower()
        starts: list[tuple[str, dict]] = []  # type: ignore
        for e in edges:
            cause = _node_name(e, "cause")
            effect = _node_name(e, "effect")
            if query_lower in cause.lower() or query_lower in effect.lower():
                starts.append((cause, e))

        if not starts:
            return []

        # 4. BFS 多跳遍历
        chains: list[dict[str, Any]] = []
        for start_cause, start_edge in starts:
            path = [start_cause]
            confs = [start_edge.get("confidence", 0.5)]
            queue: list[tuple[str, int, list[str], list[float]]] = [(start_cause, 1, path, confs)]

            while queue:
                node, depth, path, confs = queue.pop(0)
                if depth > max_depth:
                    continue
                if node not in adj:
                    continue
                for next_effect, next_edge in adj[node]:
                    if next_effect in path:  # 防环
                        continue
                    new_path = [*path, f"→ {next_effect}"]
                    new_confs = [*confs, next_edge.get("confidence", 0.5)]
                    chains.append(
                        {
                            "path": new_path,
                            "confidence": sum(new_confs) / len(new_confs),
                            "verdict": next_edge.get("verdict", "predicted"),
                            "energy_relation": next_edge.get("energy_relation", "neutral"),
                            "depth": len(new_path) - 1,
                        }
                    )
                    if depth + 1 <= max_depth:
                        queue.append((next_effect, depth + 1, new_path, new_confs))

        # 按深度（多跳优先）、置信度排序
        chains.sort(key=lambda c: (c["depth"], c["confidence"]), reverse=True)
        return chains

    def _generate_explanation_summary(self, chains: list[dict[str, Any]], query: str) -> str:
        """生成可读解释摘要。"""
        if not chains:
            return f"未找到与「{query}」相关的因果链。"

        n_confirmed = sum(1 for c in chains if c.get("verdict") == "confirmed")
        n_novel = sum(1 for c in chains if c.get("verdict") == "novel")

        parts = [f"共发现 {len(chains)} 条与「{query}」相关的因果链。"]
        if n_confirmed > 0:
            parts.append(f"其中 {n_confirmed} 条被拓扑先验确认。")
        if n_novel > 0:
            parts.append(f"{n_novel} 条为潜在新发现。")

        top_chain = chains[0]
        parts.append(f"最高置信度链 (置信度: {top_chain['confidence']:.2f}): {' → '.join(top_chain['path'])}")

        return " ".join(parts)
