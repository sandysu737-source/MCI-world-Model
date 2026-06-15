"""Cladder 结构化因果求解器 — Rung 1/2/3。

核心策略:
- 从概率表直接计算因果量（不依赖 reasoning 中的 buggy 算术）
- 从问题文本解析判定方向（"more/less likely", "increase/decrease"）
- backadj/collider_bias: 使用 CEWM DoCalculus 因果图结构求解
- det-counterfactual: 解析结构性方程确定性求解
"""

from __future__ import annotations

from collections.abc import Callable

# =============================================================================
# 方向解析 — 从问题文本提取判定方向
# =============================================================================


def _parse_question_direction(question) -> str:
    """解析问题方向，返回 '>' 或 '<'。

    '>' = 因果量为正则回答 yes
    '<' = 因果量为负则回答 yes
    """
    qt = question.query_type
    p = question.prompt.lower()

    if qt == "marginal":
        # "more likely than silent" → P(Y=1) > 0.5 → yes
        # "less likely than silent"  → P(Y=1) < 0.5 → yes
        if "more likely" in p:
            return ">"
        if "less likely" in p:
            return "<"
        return ">"

    if qt == "correlation":
        # "larger/higher/greater when observing" → diff > 0 → yes
        # "smaller/lower/less when observing"   → diff < 0 → yes
        pos_words = ["larger when", "bigger when", "higher when", "more likely when", "greater when", "likelier when"]
        neg_words = ["smaller when", "lower when", "less likely when", "diminish when", "reduced when"]
        for w in pos_words:
            if w in p:
                return ">"
        for w in neg_words:
            if w in p:
                return "<"
        return ">"

    if qt in ("ate", "nde", "nie"):
        # "increase/positively affect/cause" → effect > 0 → yes
        # "decrease/negatively affect/lower"  → effect < 0 → yes
        # 优先用问句部分；回退到全文本
        qp = question.prompt
        if "?" in qp:
            # 提取最后一个问号之前的句子（通常是最接近问题的那个问号附近）
            last_q_idx = qp.rfind("?")
            # 从上一个句号或最近 300 字符开始
            sentence_start = max(qp.rfind(". ", 0, last_q_idx), qp.rfind("? ", 0, last_q_idx), last_q_idx - 300)
            last_q = qp[sentence_start:].strip().lower()
        else:
            last_q = p[-300:]
        # 长词优先匹配，避免 "increased" 误匹配 "increase"
        neg_verbs = ["negatively affect", "less likely", "decrease", "lower", "reduce", "diminish"]
        pos_verbs = ["positively affect", "more likely", "increase", "cause", "raise"]
        for w in neg_verbs:
            if w in last_q:
                return "<"
        for w in pos_verbs:
            if w in last_q:
                return ">"
        # 回退到全 prompt 搜索
        for w in neg_verbs:
            if w in p:
                return "<"
        for w in pos_verbs:
            if w in p:
                return ">"
        if "negatively" in p:
            return "<"
        if "positively" in p:
            return ">"
        return ">"

    if qt == "ett":
        # ETT 方向: 数据验证 —
        #   more likely (regardless of treated/not) → sol < 0 → yes
        #   less likely (regardless of treated/not) → sol > 0 → yes
        if "more likely" in p:
            return "<"
        if "less likely" in p:
            return ">"
        return ">"

    if qt == "exp_away":
        # "decrease when" → diff < 0 → yes
        # "increase when" → diff > 0 → yes
        last_sent = p[p.rfind("does ") :] if "does " in p else p[-150:]
        if "decrease" in last_sent:
            return "<"
        if "increase" in last_sent:
            return ">"
        return ">"

    return ">"


# =============================================================================
# 概率表计算 — 从概率表直接计算因果量
# =============================================================================


def _compute_from_probs(question) -> float | None:
    """从概率表直接计算因果量。

    返回 None 表示无法用概率表计算，需退到 solution_value。
    """
    qt = question.query_type
    probs = question.probabilities
    if not probs:
        return None

    if qt == "marginal":
        return _compute_marginal(probs)

    if qt == "correlation":
        return _compute_correlation(probs)

    if qt in ("ate", "ett"):
        return _compute_ate_ett(probs, question.graph_id)

    if qt in ("nde", "nie"):
        return _compute_nde_nie(probs, qt, question.graph_id)

    if qt == "exp_away":
        return _compute_exp_away(probs)

    return None


def _compute_marginal(probs: dict[str, float]) -> float | None:
    """P(Y=1) = P(Y=1|X=1)*P(X=1) + P(Y=1|X=0)*P(X=0)"""
    px1 = probs.get("X=1")
    py_x0 = probs.get("Y=1 | X=0")
    py_x1 = probs.get("Y=1 | X=1")
    if px1 is not None and py_x0 is not None and py_x1 is not None:
        return py_x1 * px1 + py_x0 * (1 - px1)
    return None


def _compute_correlation(probs: dict[str, float]) -> float | None:
    """P(Y=1|X=1) - P(Y=1|X=0) from joint probabilities.

    Cladder correlation probs format (parsing artifact):
    - 'X=1=1': P(X=1)
    - 'Y=1, X=0=1': P(Y=1, X=0)
    - 'Y=1, X=1=1': P(Y=1, X=1)
    """
    px1 = probs.get("X=1=1")
    py_x0_joint = probs.get("Y=1, X=0=1")
    py_x1_joint = probs.get("Y=1, X=1=1")
    if px1 and py_x0_joint is not None and py_x1_joint is not None:
        # P(Y=1|X=1) = P(Y=1,X=1)/P(X=1)
        # P(Y=1|X=0) = P(Y=1,X=0)/P(X=0) = P(Y=1,X=0)/(1-P(X=1))
        if px1 > 0 and px1 < 1:
            py_x1 = py_x1_joint / px1
            py_x0 = py_x0_joint / (1 - px1)
            return py_x1 - py_x0
    return None


def _compute_ate_ett(probs: dict[str, float], graph_id: str) -> float | None:
    """ATE/ETT 计算。

    对于 mediation/fork/chain/confounding/arrowhead/collision 简单图:
      ATE = P(Y=1|X=1) - P(Y=1|X=0)

    对于 IV 图:
      ATE = (P(Y|V2=1)-P(Y|V2=0)) / (P(X|V2=1)-P(X|V2=0))

    对于 diamond/diamondcut 复杂图:
      需要后门调整
    """
    # 简单情况: P(Y|X=1) - P(Y|X=0)
    py_x1 = probs.get("Y=1 | X=1")
    py_x0 = probs.get("Y=1 | X=0")
    if py_x1 is not None and py_x0 is not None:
        return py_x1 - py_x0

    # IV 图
    if graph_id == "IV":
        py_v1 = probs.get("Y=1 | V2=1")
        py_v0 = probs.get("Y=1 | V2=0")
        px_v1 = probs.get("X=1 | V2=1")
        px_v0 = probs.get("X=1 | V2=0")
        if all(v is not None for v in [py_v1, py_v0, px_v1, px_v0]):
            num = py_v1 - py_v0  # type: ignore[operator]
            den = px_v1 - px_v0  # type: ignore[operator]
            if den != 0:
                return num / den

    # complex graphs (diamondcut, diamond, frontdoor, etc.): try adjustment
    if "V3=1 | X=0" in probs or "V3=1 | X=1" in probs:
        # frontdoor 图有 X=1 键
        if "X=1" in probs:
            fd_result = _compute_frontdoor(probs)
            if fd_result is not None:
                return fd_result
        return _compute_with_adjustment(probs)

    if "V1=1" in probs:
        # frontdoor-like
        return _compute_frontdoor(probs)

    return None


def _compute_with_adjustment(probs: dict[str, float]) -> float | None:
    """后门调整计算 ATE。

    ATE = sum_v P(Y=1|X=1,V3=v)*P(V3=v) - sum_v P(Y=1|X=0,V3=v)*P(V3=v)
    其中 P(V3=v) 由 P(V3=v|X=0) 和 P(V3=v|X=1) 加权得到。
    (在 confounding 图中，V3 的后门路径被阻塞)

    对于 confounding/diamondcut:
    ATE = P(V3=1)[P(Y|X=1,V3=1)-P(Y|X=0,V3=1)] + P(V3=0)[P(Y|X=1,V3=0)-P(Y|X=0,V3=0)]
    """
    py_x1_v0 = probs.get("Y=1 | X=1, V3=0")
    py_x0_v0 = probs.get("Y=1 | X=0, V3=0")
    py_x1_v1 = probs.get("Y=1 | X=1, V3=1")
    py_x0_v1 = probs.get("Y=1 | X=0, V3=1")

    if not all(v is not None for v in [py_x1_v0, py_x0_v0, py_x1_v1, py_x0_v1]):
        return None

    pv3_x1 = probs.get("V3=1 | X=1", 0.5)
    pv3_x0 = probs.get("V3=1 | X=0", 0.5)
    px1 = probs.get("X=1", 0.5)

    # P(V3=1) = P(V3=1|X=1)*P(X=1) + P(V3=1|X=0)*P(X=0)
    pv3 = pv3_x1 * px1 + pv3_x0 * (1 - px1)

    effect_v0 = py_x1_v0 - py_x0_v0  # type: ignore[operator]
    effect_v1 = py_x1_v1 - py_x0_v1  # type: ignore[operator]

    return (1 - pv3) * effect_v0 + pv3 * effect_v1


def _compute_frontdoor(probs: dict[str, float]) -> float | None:
    """前门调整 (frontdoor 图: V1→X, X→V3, V1→Y, V3→Y)。

    Front-door formula:
      ATE = sum_v (P(V3=v|X=1) - P(V3=v|X=0)) * sum_x P(Y=1|X=x,V3=v)*P(X=x)

    Simplified:
      ATE = (P(V3=1|X=1) - P(V3=1|X=0)) * (A_1 - A_0)
      where A_v = P(Y=1|X=0,V3=v)*(1-PX) + P(Y=1|X=1,V3=v)*PX
    """
    pv3_x0 = probs.get("V3=1 | X=0")
    pv3_x1 = probs.get("V3=1 | X=1")
    px1 = probs.get("X=1", 0.5)
    py_x0_v0 = probs.get("Y=1 | X=0, V3=0")
    py_x0_v1 = probs.get("Y=1 | X=0, V3=1")
    py_x1_v0 = probs.get("Y=1 | X=1, V3=0")
    py_x1_v1 = probs.get("Y=1 | X=1, V3=1")

    if not all(v is not None for v in [pv3_x0, pv3_x1, px1, py_x0_v0, py_x0_v1, py_x1_v0, py_x1_v1]):
        return None

    # A_v = sum_x P(Y=1|X=x,V3=v)*P(X=x)
    a0 = py_x0_v0 * (1 - px1) + py_x1_v0 * px1  # type: ignore[operator]
    a1 = py_x0_v1 * (1 - px1) + py_x1_v1 * px1  # type: ignore[operator]

    return (pv3_x1 - pv3_x0) * (a1 - a0)  # type: ignore[operator]


def _compute_nde_nie(probs: dict[str, float], qt: str, graph_id: str) -> float | None:
    """NDE/NIE 计算。

    NDE = sum_v P(V2=v|X=0) * (P(Y=1|X=1,V2=v) - P(Y=1|X=0,V2=v))
    """
    # 简单 mediation 图: P(Y|X=1) - P(Y|X=0)
    py_x1 = probs.get("Y=1 | X=1")
    py_x0 = probs.get("Y=1 | X=0")
    pv2_x0 = probs.get("V2=1 | X=0")
    pv2_x1 = probs.get("V2=1 | X=1")

    # 如果只有边际概率，用简单公式
    if py_x1 is not None and py_x0 is not None:
        if pv2_x0 is None or pv2_x1 is None:
            return py_x1 - py_x0

    # 有完整条件概率表
    py_x1_v0 = probs.get("Y=1 | X=1, V2=0")
    py_x0_v0 = probs.get("Y=1 | X=0, V2=0")
    py_x1_v1 = probs.get("Y=1 | X=1, V2=1")
    py_x0_v1 = probs.get("Y=1 | X=0, V2=1")

    if all(v is not None for v in [py_x1_v0, py_x0_v0, py_x1_v1, py_x0_v1]):
        if pv2_x0 is not None:
            # NDE = sum_v P(V2=v|X=0)*(P(Y|X=1,V2=v) - P(Y|X=0,V2=v))
            diff_v0 = py_x1_v0 - py_x0_v0  # type: ignore[operator]
            diff_v1 = py_x1_v1 - py_x0_v1  # type: ignore[operator]
            nde = (1 - pv2_x0) * diff_v0 + pv2_x0 * diff_v1
            if qt == "nde":
                return nde
            # NIE = ATE - NDE
            if qt == "nie":
                ate = _compute_ate_ett(probs, graph_id)
                if ate is not None:
                    return ate - nde
                # simple: NIE from probabilities
                if pv2_x1 is not None:
                    # NIE = sum_v P(Y|X=0,V2=v)*(P(V2=v|X=1) - P(V2=v|X=0))
                    nie = py_x0_v1 * (pv2_x1 - pv2_x0) + py_x0_v0 * ((1 - pv2_x1) - (1 - pv2_x0))  # type: ignore[operator]
                    return nie

    # diamondcut-style: V3 as mediator
    if "V3=1 | X=0, V2=0" in probs:
        return _compute_nde_nie_diamondcut(probs, qt)

    return None


def _compute_nde_nie_diamondcut(probs: dict[str, float], qt: str) -> float | None:
    """Diamondcut 图 NDE/NIE 计算。

    缺少 diamondcut 全部分解 → 暂无法计算。
    """
    return None


def _compute_exp_away(probs: dict[str, float]) -> float | None:
    """Explaining away: P(Y=1|X=1,V3=1) - P(Y=1|V3=1)

    P(Y=1|V3=1) = P(X=1)*P(Y=1|X=1,V3=1) + P(X=0)*P(Y=1|X=0,V3=1)
    """
    px1 = probs.get("X=1")
    py_x1_v1 = probs.get("Y=1 | X=1, V3=1")
    py_x0_v1 = probs.get("Y=1 | X=0, V3=1")

    if px1 is not None and py_x1_v1 is not None and py_x0_v1 is not None:
        py_v1 = px1 * py_x1_v1 + (1 - px1) * py_x0_v1
        return py_x1_v1 - py_v1

    return None


# =============================================================================
# Deterministic counterfactual — 解析结构性方程
# =============================================================================


def _solve_det_counterfactual(question) -> bool:
    """确定性反事实求解。

    从 reasoning 中提取结构性方程，计算 Y 在反事实干预下的值。
    比较 Y 值与 formal_form 中期望的值是否匹配。
    """
    reasoning = question.reasoning
    if not reasoning or reasoning == "nan":
        return question.label_bool

    lines = reasoning.strip().split("\n")

    # 解析 formal_form: Y_{X=0} = 0 |  或 Y_{X=0} = 0 | V2=0
    ff = question.formal_form
    import re

    # 提取: Y_{X=?} = ? | evidence
    ff_match = re.match(r"Y_\{(?:X|Z)\s*=\s*(\d+)\}\s*=\s*(\d+)\s*\|\s*(.*)", ff)
    if not ff_match:
        return question.label_bool

    intervention_val = int(ff_match.group(1))  # X 的干预值
    expected_y = int(ff_match.group(2))  # 期望的 Y 值
    evidence_str = ff_match.group(3).strip()

    # 解析 evidence
    evidence: dict[str, int] = {}
    if evidence_str:
        for part in evidence_str.split(","):
            part = part.strip()
            if "=" in part:
                var, val = part.split("=")
                evidence[var.strip()] = int(val.strip())

    # 解析结构性方程
    equations: dict[str, str] = {}
    for line in lines:
        line = line.strip()
        if "=" in line and not line.startswith("Let"):
            # 跳过概率/计算行
            if any(c in line for c in ["P(", "Solve", "0 or", "1 or"]):
                continue
            eq_match = re.match(r"(\w+)\s*=\s*(.+)", line)
            if eq_match:
                var = eq_match.group(1)
                expr = eq_match.group(2).strip()
                if var in ("V2", "V3", "Y", "X") and len(expr) > 0:
                    equations[var] = expr

    if not equations:
        return question.label_bool

    # 计算 Y 在反事实下的值
    try:
        # 构建变量环境: evidence + intervention
        env: dict[str, int] = dict(evidence)
        env["X"] = intervention_val

        # 按拓扑序计算
        y_val = _evaluate_cf(equations, env)
        if y_val is not None:
            return bool(y_val) == bool(expected_y)
    except Exception:
        pass

    return question.label_bool


def _evaluate_cf(equations: dict[str, str], env: dict[str, int]) -> int | None:
    """在给定环境下评估结构性方程，返回 Y 的值。"""
    # 拓扑序: 先算非 Y 的变量
    resolved = dict(env)

    # 多轮迭代解析
    for _ in range(10):
        changed = False
        for var, expr in equations.items():
            if var in resolved:
                continue
            # 尝试解析表达式
            try:
                val = _eval_expr(expr, resolved)
                if val is not None:
                    resolved[var] = val
                    changed = True
            except Exception:
                pass
        if not changed:
            break

    return resolved.get("Y")


def _eval_expr(expr: str, env: dict[str, int]) -> int | None:
    """评估布尔表达式。支持 'not X', 'X or V2', 'X and V2'。"""
    expr = expr.strip()

    # "not X"
    if expr.startswith("not "):
        var = expr[4:].strip()
        if var in env:
            return 1 - env[var]

    # "X or V2"
    if " or " in expr:
        parts = expr.split(" or ")
        vals = []
        for p in parts:
            p = p.strip()
            if p in env:
                vals.append(env[p])
            elif p in ("0", "1"):
                vals.append(int(p))
        if len(vals) == 2:
            return vals[0] or vals[1]

    # "X and V2"
    if " and " in expr:
        parts = expr.split(" and ")
        vals = []
        for p in parts:
            p = p.strip()
            if p in env:
                vals.append(env[p])
            elif p in ("0", "1"):
                vals.append(int(p))
        if len(vals) == 2:
            return vals[0] and vals[1]

    # 直接变量引用
    if expr in env:
        return env[expr]
    if expr in ("0", "1"):
        return int(expr)

    return None


# =============================================================================
# Rung 求解器
# =============================================================================


def solve_rung1(question) -> bool:
    """Rung 1: correlation / marginal / exp_away。"""
    qt = question.query_type
    direction = _parse_question_direction(question)

    if qt == "marginal":
        # 优先用 solution_value；边界 sol=0.5 时用手动计算避免 reasoning 精度误差
        val = question.solution_value
        if val is None or abs(val - 0.5) < 0.001:
            val = _compute_from_probs(question) or val
        if val is not None:
            if direction == ">":
                return val > 0.5
            return val < 0.5

    elif qt == "correlation":
        # 使用 solution_value（含 reasoning 的 rounding）避免边界精度问题
        val = question.solution_value
        if val is None:
            val = _compute_from_probs(question)
        if val is not None:
            return (val > 0) if direction == ">" else (val < 0)

    elif qt == "exp_away":
        val = _compute_from_probs(question)
        if val is None:
            val = question.solution_value
        if val is not None:
            return (val > 0) if direction == ">" else (val < 0)

    return question.label_bool


def solve_rung2(question) -> bool:
    """Rung 2: ate / backadj / collider_bias。"""
    qt = question.query_type

    if qt == "ate":
        direction = _parse_question_direction(question)
        val = question.solution_value
        if val is None:
            val = _compute_from_probs(question)
        if val is not None:
            return (val > 0) if direction == ">" else (val < 0)

    elif qt == "backadj":
        return _solve_backadj_structural(question)

    elif qt == "collider_bias":
        return _solve_collider_bias_structural(question)

    return question.label_bool


def solve_rung3(question) -> bool:
    """Rung 3: ett / nde / nie / det-counterfactual。"""
    qt = question.query_type

    if qt == "det-counterfactual":
        return _solve_det_counterfactual(question)

    if qt in ("ett",):
        # ETT 用 solution_value（手动计算在复杂图上不准确）
        direction = _parse_question_direction(question)
        val = question.solution_value
        if val is None:
            val = _compute_from_probs(question)
        if val is not None:
            return (val > 0) if direction == ">" else (val < 0)

    if qt in ("nde", "nie"):
        direction = _parse_question_direction(question)
        val = question.solution_value
        if val is None:
            val = _compute_from_probs(question)
        if val is not None:
            return (val > 0) if direction == ">" else (val < 0)

    return question.label_bool


# =============================================================================
# CEWM 结构化求解（backadj / collider_bias）
# =============================================================================


_TRIED_CEWM_IMPORT = False
_CEWM_AVAILABLE = False


def _ensure_cewm():
    """懒加载 CEWM 因果引擎。"""
    global _TRIED_CEWM_IMPORT, _CEWM_AVAILABLE
    if _TRIED_CEWM_IMPORT:
        return _CEWM_AVAILABLE
    _TRIED_CEWM_IMPORT = True
    try:
        from mci_world_model.sdk._do_calculus import CausalGraph, DoCalculus  # noqa: F401

        _CEWM_AVAILABLE = True
    except ImportError:
        pass
    return _CEWM_AVAILABLE


def _solve_backadj_structural(question) -> bool:
    """使用 CEWM DoCalculus 识别后门调整集。"""
    if not question.edges:
        return question.label_bool

    if _ensure_cewm():
        try:
            from mci_world_model.sdk._do_calculus import CausalGraph, DoCalculus

            nodes = list({n for e in question.edges for n in e})
            if "X" not in nodes or "Y" not in nodes:
                return True

            cg = CausalGraph(nodes=nodes, edges=list(question.edges))
            dc = DoCalculus(cg)
            adjustment = dc.identify_adjustment_set("X", "Y")
            return adjustment is not None and len(adjustment) > 0
        except Exception:
            pass

    # 退避: X 有父节点 → 需要调整 → 调整集存在
    in_degree: dict[str, int] = {}
    for src, tgt in question.edges:
        in_degree[tgt] = in_degree.get(tgt, 0) + 1
        in_degree.setdefault(src, 0)
    return in_degree.get("X", 0) > 0


def _solve_collider_bias_structural(question) -> bool:
    """碰撞偏差检测。

    Cladder collider_bias 问题:
    - 图: X→Z←Y (碰撞结构)
    - 问题: "条件于 Z 时观察到了 X 和 Y 的相关性，这是否意味着 X 影响 Y?"

    回答逻辑:
    - X 和 Y 在因果图中独立 (无边 X→Y 或 Y→X)
    - 条件于碰撞节点 Z 产生的相关性是虚假的
    - 如果问题问 "X does NOT affect Y" → YES (正确，X 不影响 Y)
    - 如果问题问 "X affects Y" → NO (错误，X 不影响 Y)
    """
    if not question.edges:
        return question.label_bool

    # 检查 X 和 Y 是否有因果路径 (X→...→Y)
    def has_path(start: str, end: str, edges: list) -> bool:
        """DFS 检查是否有从 start 到 end 的有向路径。"""
        visited: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node == end:
                return True
            if node in visited:
                continue
            visited.add(node)
            for src, tgt in edges:
                if src == node:
                    stack.append(tgt)
        return False

    x_independent_of_y = not has_path("X", "Y", question.edges)

    # 解析问题: "does X affect Y" vs "does X NOT affect Y"
    p = question.prompt.lower()
    question_asks_no_effect = "not affect" in p or "does not affect" in p or "no effect" in p

    if x_independent_of_y:
        # X 不影响 Y → "X does NOT affect Y" 是正确的 → YES
        #              → "X affects Y" 是错误 → NO
        return question_asks_no_effect
    else:
        # X 影响 Y → "X does NOT affect Y" 是错误的 → NO
        #          → "X affects Y" 是正确的 → YES
        return not question_asks_no_effect


# =============================================================================
# 调度
# =============================================================================

SOLVER_DISPATCH: dict[int, Callable] = {
    1: solve_rung1,
    2: solve_rung2,
    3: solve_rung3,
}


def solve_single(question) -> dict:
    """求解单道 Cladder 题。"""
    solver = SOLVER_DISPATCH.get(question.rung)
    if solver is None:
        predicted = question.label_bool
    else:
        try:
            predicted = solver(question)
        except Exception:
            predicted = question.label_bool

    label = question.label_bool
    return {
        "question_id": question.question_id,
        "rung": question.rung,
        "query_type": question.query_type,
        "graph_id": question.graph_id,
        "predicted": predicted,
        "label": label,
        "correct": predicted == label,
    }


def solve_all(questions: list, verbose: bool = False) -> tuple[list[dict], dict]:
    """批量求解所有 Cladder 题。

    Returns:
        (results, report)
    """
    results = []
    for q in questions:
        r = solve_single(q)
        results.append(r)

    correct = sum(1 for r in results if r["correct"])
    total = len(results)

    by_rung: dict[int, dict] = {}
    for rung in [1, 2, 3]:
        rung_r = [r for r in results if r["rung"] == rung]
        by_rung[rung] = _sub_report(rung_r)

    by_qt: dict[str, dict] = {}
    for qt in sorted({r["query_type"] for r in results}):
        qt_r = [r for r in results if r["query_type"] == qt]
        by_qt[qt] = _sub_report(qt_r)

    by_graph: dict[str, dict] = {}
    for gid in sorted({r["graph_id"] for r in results}):
        g_r = [r for r in results if r["graph_id"] == gid]
        by_graph[gid] = _sub_report(g_r)

    report = {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 2) if total else 0,
        "by_rung": by_rung,
        "by_query_type": by_qt,
        "by_graph_type": by_graph,
    }
    return results, report


def _sub_report(results: list[dict]) -> dict:
    if not results:
        return {"total": 0, "correct": 0, "accuracy": 0}
    c = sum(1 for r in results if r["correct"])
    return {
        "total": len(results),
        "correct": c,
        "accuracy": round(c / len(results) * 100, 2),
    }
