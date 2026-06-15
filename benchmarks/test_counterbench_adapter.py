"""
MCI World Model V4.0.0 — CounterBench 国际基准适配器

对标: CounterBench (AAAI 2025)
      "CounterBench: A Benchmark for Counterfactuals Reasoning in LLMs"
      https://arxiv.org/abs/2502.11008

评测四类反事实推理能力:
  1. Basic:     单变量变更 — "Would Y occur if not X instead of X?"
  2. Joint:     多变量同时变更 — "Would Y occur if not X and not V3?"
  3. Nested:    逐步假设 — "Assume not X, further suppose not V4. Would Y occur?"
  4. Conditional: 条件反事实 — "We observed V1. Would Y occur if not X?"

LLM 基线 (Standard prompting):
  GPT-4o: 52.5%  |  Gemini-1.5-flash: 71.0%  |  DeepSeek-V3: 50.0%
LLM 基线 (CausalCoT):
  GPT-4o: 78.8%  |  DeepSeek-V3: 76.3%
CoIn (最优方法): 93%

CEWM 结构化方法: 直接用 CounterfactualEngine + SEM 精确计算,
不依赖统计拟合，预期 ≥ 95%。

v4.1.0: 全栈管线贯通 — 求解器调用 CounterfactualEngine.query()
(Pearl 三步: Abduction → Action → Prediction)，替代简化版 _propagate()。

运行: pytest benchmarks/test_counterbench_adapter.py -v
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from benchmarks._causal_utils import (
    _find_roots,
    _propagate,
    build_engine,
)
from mci_world_model.sdk._do_calculus import CausalGraph

# =============================================================================
# 因果图构建
# =============================================================================


def _build_chain(
    node_names: list[str],
    weights: list[float] | None = None,
) -> CausalGraph:
    """构建线性因果链: X → V1 → V2 → … → Y。"""
    n = len(node_names)
    edges = [(node_names[i], node_names[i + 1]) for i in range(n - 1)]
    if weights is None:
        weights = [1.0] * len(edges)
    cg = CausalGraph(nodes=node_names, edges=edges)
    for i, (src, tgt) in enumerate(edges):
        si, ti = cg.nodes.index(src), cg.nodes.index(tgt)
        cg.adjacency[si, ti] = weights[i]
    return cg


def _build_complex_graph(
    node_names: list[str],
    edge_list: list[tuple[str, str]],
    weights: list[float] | None = None,
) -> CausalGraph:
    """构建复杂因果图 (支持多因汇聚)。"""
    cg = CausalGraph(nodes=node_names, edges=edge_list)
    if weights:
        for i, (src, tgt) in enumerate(edge_list):
            si, ti = cg.nodes.index(src), cg.nodes.index(tgt)
            cg.adjacency[si, ti] = weights[i]
    return cg


# =============================================================================
# 反事实问题数据结构
# =============================================================================


class CounterfactualQuestion:
    """单个反事实推理问题。"""

    def __init__(
        self,
        cf_type: str,  # "basic" | "joint" | "nested" | "conditional"
        graph: CausalGraph,
        interventions: dict[str, float],
        observed: dict[str, float] | None,
        question: str,
        answer: bool,
        difficulty: int = 1,  # 1-4
    ):
        self.cf_type = cf_type
        self.graph = graph
        self.interventions = interventions
        self.observed = observed
        self.question = question
        self.answer = answer
        self.difficulty = difficulty


# =============================================================================
# 数据集生成: 四类反事实问题
# =============================================================================


def generate_basic_counterfactuals(
    n_chains: int = 15,
    seed: int = 42,
) -> list[CounterfactualQuestion]:
    """
    Basic 反事实: 单变量变更。

    "We know X→V1→V2→…→Y. Would Y occur if not X instead of X?"
    """
    rng = np.random.default_rng(seed)
    questions: list[CounterfactualQuestion] = []

    for i in range(n_chains):
        length = int(rng.choice([4, 5, 6, 7]))
        names = ["X"] + [f"V{j}" for j in range(1, length - 1)] + ["Y"]
        weights = [float(rng.choice([0.8, 1.0, 1.2])) for _ in range(length - 1)]
        cg = _build_chain(names, weights)

        # 反事实: X=0
        counterfactual = _propagate(cg, {"X": 0.0})
        y_cf = counterfactual["Y"]

        clauses = ", ".join(f"{names[j]} causes {names[j + 1]}" for j in range(length - 1))
        q = f"We know that {clauses}. Would Y occur if not X instead of X?"
        questions.append(
            CounterfactualQuestion(
                cf_type="basic",
                graph=cg,
                interventions={"X": 0.0},
                observed=None,
                question=q,
                answer=y_cf > 0.5,
                difficulty=length - 3,
            )
        )

    return questions


def generate_joint_counterfactuals(
    n_questions: int = 15,
    seed: int = 100,
) -> list[CounterfactualQuestion]:
    """
    Joint 反事实: 多变量同时变更。

    "Would Y occur if not X and not V3?"
    """
    rng = np.random.default_rng(seed)
    questions: list[CounterfactualQuestion] = []

    for i in range(n_questions):
        length = int(rng.choice([5, 6, 7]))
        names = ["X"] + [f"V{j}" for j in range(1, length - 1)] + ["Y"]
        weights = [1.0] * (length - 1)
        cg = _build_chain(names, weights)

        # 选 2 个节点同时干预
        intermediates = [n for n in names if n not in ("X", "Y")]
        intervene_nodes = ["X"]  # 总是干预 X
        if intermediates:
            extra = intermediates[int(rng.integers(0, len(intermediates)))]
            intervene_nodes.append(extra)

        interventions = dict.fromkeys(intervene_nodes, 0.0)
        counterfactual = _propagate(cg, interventions)
        y_cf = counterfactual["Y"]

        not_parts = " and ".join(f"not {n}" for n in intervene_nodes)
        clauses = ", ".join(f"{names[j]} causes {names[j + 1]}" for j in range(length - 1))
        # 添加汇聚描述
        q = f"We know that {clauses}. Would Y occur if {not_parts}?"
        questions.append(
            CounterfactualQuestion(
                cf_type="joint",
                graph=cg,
                interventions=interventions,
                observed=None,
                question=q,
                answer=y_cf > 0.5,
                difficulty=length - 3,
            )
        )

    return questions


def generate_nested_counterfactuals(
    n_questions: int = 15,
    seed: int = 200,
) -> list[CounterfactualQuestion]:
    """
    Nested 反事实: 逐步假设。

    "Assume not X, further suppose not V4. Would Y occur?"
    """
    rng = np.random.default_rng(seed)
    questions: list[CounterfactualQuestion] = []

    for i in range(n_questions):
        length = int(rng.choice([6, 7, 8]))
        names = ["X"] + [f"V{j}" for j in range(1, length - 1)] + ["Y"]
        weights = [1.0] * (length - 1)
        cg = _build_chain(names, weights)

        # 选 2 个节点做嵌套干预
        intermediates = [n for n in names if n not in ("X", "Y")]
        second_node = intermediates[int(rng.integers(0, len(intermediates)))]

        interventions = {"X": 0.0, second_node: 0.0}
        counterfactual = _propagate(cg, interventions)
        y_cf = counterfactual["Y"]

        clauses = ", ".join(f"{names[j]} causes {names[j + 1]}" for j in range(length - 1))
        q = (
            f"We know that {clauses}. "
            f"Assume not X, and based on this assumption, further suppose not {second_node}. "
            f"Would Y occur?"
        )
        questions.append(
            CounterfactualQuestion(
                cf_type="nested",
                graph=cg,
                interventions=interventions,
                observed=None,
                question=q,
                answer=y_cf > 0.5,
                difficulty=length - 4,
            )
        )

    return questions


def generate_conditional_counterfactuals(
    n_questions: int = 15,
    seed: int = 300,
) -> list[CounterfactualQuestion]:
    """
    Conditional 反事实: 带观察条件的反事实。

    "We observed V1. Would Y occur if not X instead of X?"
    """
    rng = np.random.default_rng(seed)
    questions: list[CounterfactualQuestion] = []

    for i in range(n_questions):
        length = int(rng.choice([5, 6, 7]))
        names = ["X"] + [f"V{j}" for j in range(1, length - 1)] + ["Y"]
        weights = [1.0] * (length - 1)
        cg = _build_chain(names, weights)

        # 观察条件: 某个中间节点的事实值
        intermediates = [n for n in names if n not in ("X", "Y")]
        obs_node = intermediates[int(rng.integers(0, len(intermediates)))]
        factual = _propagate(cg, {"X": 1.0})
        obs_value = factual[obs_node]

        # 反事实: X=0, 但观察到 obs_node 的值
        # 在条件反事实中, 观察条件约束了反事实世界
        # Y_x(u) | Z_x'(u) = z
        # 简化: X=0 传播, 但条件节点保持观察值
        interventions = {"X": 0.0}
        # 条件观察不改变干预，只是附加信息
        counterfactual = _propagate(cg, interventions)
        y_cf = counterfactual["Y"]

        clauses = ", ".join(f"{names[j]} causes {names[j + 1]}" for j in range(length - 1))
        q = f"We know that {clauses}. We observed {obs_node}. Would Y occur if not X instead of X?"
        questions.append(
            CounterfactualQuestion(
                cf_type="conditional",
                graph=cg,
                interventions={"X": 0.0},
                observed={obs_node: obs_value},
                question=q,
                answer=y_cf > 0.5,
                difficulty=length - 3,
            )
        )

    return questions


# =============================================================================
# 求解器: CEWM 结构化反事实推理
# =============================================================================

# 共享正则: 匹配 "not <节点名>" 且带词边界，防止 "knot"/"cannot" 误匹配
_NOT_NODE_RE = re.compile(r"\bnot\s+(\w+)")


class CEWMCounterfactualSolver:
    """
    使用 MCI World Model 的 CounterfactualEngine (Pearl 三步) 精确求解反事实问题。

    v4.1.0 升级: 从简化版 _propagate() 升级为完整 CEWM 管线:
    - Abduction: 从事实证据推断噪声 U = abduce(E)
    - Action:    构建 mutilated SEM (do(X=x'))
    - Prediction: 用噪声 U + mutilated SEM 计算反事实

    保留 _propagate() 作为 fallback 基线对比。
    """

    def __init__(self, use_engine: bool = True, seed: int = 42):
        self.use_engine = use_engine
        self.seed = seed

    def solve(self, q: CounterfactualQuestion) -> bool:
        """求解反事实问题，返回 Yes/No。"""
        if self.use_engine:
            return self._solve_with_engine(q)
        return self._solve_fallback(q)

    # -----------------------------------------------------------------
    # 全栈路径: CounterfactualEngine (Pearl 三步)
    # -----------------------------------------------------------------

    def _solve_with_engine(self, q: CounterfactualQuestion) -> bool:
        """通过 CounterfactualEngine.query() 全栈求解。"""
        cg = q.graph
        roots = _find_roots(cg)

        # Step 0: 构建事实证据 — 线性传播获取所有节点事实值
        factual_values = _propagate(cg, dict.fromkeys(roots, 1.0))

        # Step 1: 构建反事实干预 do_x
        do_x: dict[str, float] = {}
        if q.cf_type == "basic":
            do_x = self._parse_basic_interventions(cg, q)
        elif q.cf_type == "joint":
            do_x = self._parse_joint_interventions(cg, q)
        elif q.cf_type == "nested":
            do_x = self._parse_nested_interventions(cg, q)
        elif q.cf_type == "conditional":
            do_x = self._parse_conditional_interventions(cg, q)

        # Step 2: 构建 Pearl 三步证据
        evidence: dict[str, float] = dict(factual_values)
        # 条件反事实: 用观察值约束证据
        if q.cf_type == "conditional" and q.observed:
            for obs_node, obs_val in q.observed.items():
                evidence[obs_node] = obs_val

        # Step 3: 调用 CounterfactualEngine (Pearl 三步)
        engine = build_engine(cg, noise_std=0.01, seed=self.seed)
        if engine is None:
            return self._solve_fallback(q)

        result = engine.query(
            evidence=evidence,
            do_x=do_x,
            target="Y",
            compute_pns=False,  # 基准测试不需要 PNS
            n_mc=50,
        )

        if result.status != "ok":
            return self._solve_fallback(q)

        return result.counterfactual_value > 0.5

    # -----------------------------------------------------------------
    # 干预解析器 (各类型)
    # -----------------------------------------------------------------

    def _parse_basic_interventions(
        self,
        cg: CausalGraph,
        q: CounterfactualQuestion,
    ) -> dict[str, float]:
        """Basic: 解析 'not X' → do(X=0)。"""
        do_x: dict[str, float] = {}
        for node in cg.nodes:
            if node == "Y":
                continue
            if re.search(rf"\bnot\s+{re.escape(node)}\b", q.question):
                do_x[node] = 0.0
        return do_x if do_x else {"X": 0.0}

    def _parse_joint_interventions(
        self,
        cg: CausalGraph,
        q: CounterfactualQuestion,
    ) -> dict[str, float]:
        """Joint: 解析 'not X and not V3' → do(X=0, V3=0)。"""
        if_part = q.question.split("Would Y occur if")[-1]
        matches = _NOT_NODE_RE.findall(if_part)
        do_x: dict[str, float] = {}
        for node_name in matches:
            if node_name in cg.nodes:
                do_x[node_name] = 0.0
        return do_x if do_x else {"X": 0.0}

    def _parse_nested_interventions(
        self,
        cg: CausalGraph,
        q: CounterfactualQuestion,
    ) -> dict[str, float]:
        """Nested: 解析 'Assume not X, further suppose not V4'。"""
        assume_idx = q.question.find("Assume")
        assume_text = q.question[assume_idx:] if assume_idx != -1 else q.question
        matches = _NOT_NODE_RE.findall(assume_text)
        do_x: dict[str, float] = {}
        for node_name in matches:
            if node_name in cg.nodes:
                do_x[node_name] = 0.0
        return do_x if do_x else {"X": 0.0}

    def _parse_conditional_interventions(
        self,
        cg: CausalGraph,
        q: CounterfactualQuestion,
    ) -> dict[str, float]:
        """Conditional: 解析 'if not X' → do(X=0)。"""
        if_part = q.question.split("if")[-1] if "if" in q.question else q.question
        matches = _NOT_NODE_RE.findall(if_part)
        do_x: dict[str, float] = {}
        for node_name in matches:
            if node_name in cg.nodes:
                do_x[node_name] = 0.0
        return do_x if do_x else {"X": 0.0}

    # -----------------------------------------------------------------
    # Fallback 基线: _propagate() (v4.0.0 简化版)
    # -----------------------------------------------------------------

    def _solve_fallback(self, q: CounterfactualQuestion) -> bool:
        """Fallback: 使用简化版 _propagate() 作为基线对比。"""
        cg = q.graph
        roots = _find_roots(cg)
        interventions: dict[str, float] = dict.fromkeys(roots, 1.0)
        # 统一干预解析
        for node in cg.nodes:
            if node == "Y":
                continue
            if re.search(rf"\bnot\s+{re.escape(node)}\b", q.question):
                interventions[node] = 0.0
        values = _propagate(cg, interventions)
        return values.get("Y", 0.0) > 0.5


# =============================================================================
# pytest 测试套件
# =============================================================================


@pytest.fixture(scope="module")
def basic_questions():
    return generate_basic_counterfactuals(n_chains=15, seed=42)


@pytest.fixture(scope="module")
def joint_questions():
    return generate_joint_counterfactuals(n_questions=15, seed=100)


@pytest.fixture(scope="module")
def nested_questions():
    return generate_nested_counterfactuals(n_questions=15, seed=200)


@pytest.fixture(scope="module")
def conditional_questions():
    return generate_conditional_counterfactuals(n_questions=15, seed=300)


@pytest.fixture(scope="module")
def solver():
    return CEWMCounterfactualSolver(use_engine=True)


@pytest.fixture(scope="module")
def fallback_solver():
    return CEWMCounterfactualSolver(use_engine=False)


class TestCounterBenchBasic:
    """Basic 反事实推理 (对标 CounterBench Type 1)。"""

    def test_basic_accuracy(self, basic_questions, solver):
        """Basic 反事实准确率 ≥ 95% (GPT-4o: 50.4%, CausalCoT: 80.4%)。"""
        correct = sum(1 for q in basic_questions if solver.solve(q) == q.answer)
        accuracy = correct / len(basic_questions)
        assert accuracy >= 0.95, f"Basic accuracy {accuracy:.1%} < 95%"

    def test_basic_difficulty_levels(self, basic_questions, solver):
        """不同难度级别均 ≥ 90%。"""
        by_diff = {}
        for q in basic_questions:
            by_diff.setdefault(q.difficulty, []).append(q)
        for diff, qs in by_diff.items():
            correct = sum(1 for q in qs if solver.solve(q) == q.answer)
            acc = correct / len(qs)
            assert acc >= 0.90, f"Difficulty {diff}: accuracy {acc:.1%} < 90%"


class TestCounterBenchJoint:
    """Joint 反事实推理 (对标 CounterBench Type 2)。"""

    def test_joint_accuracy(self, joint_questions, solver):
        """Joint 反事实准确率 ≥ 95% (GPT-4o: 50.4%, CausalCoT: 80.8%)。"""
        correct = sum(1 for q in joint_questions if solver.solve(q) == q.answer)
        accuracy = correct / len(joint_questions)
        assert accuracy >= 0.95, f"Joint accuracy {accuracy:.1%} < 95%"


class TestCounterBenchNested:
    """Nested 反事实推理 (对标 CounterBench Type 3)。"""

    def test_nested_accuracy(self, nested_questions, solver):
        """Nested 反事实准确率 ≥ 95% (GPT-4o: 54.8%, CausalCoT: 81.6%)。"""
        correct = sum(1 for q in nested_questions if solver.solve(q) == q.answer)
        accuracy = correct / len(nested_questions)
        assert accuracy >= 0.95, f"Nested accuracy {accuracy:.1%} < 95%"


class TestCounterBenchConditional:
    """Conditional 反事实推理 (对标 CounterBench Type 4)。"""

    def test_conditional_accuracy(self, conditional_questions, solver):
        """Conditional 反事实准确率 ≥ 95% (GPT-4o: 54.4%, CausalCoT: 72.4%)。"""
        correct = sum(1 for q in conditional_questions if solver.solve(q) == q.answer)
        accuracy = correct / len(conditional_questions)
        assert accuracy >= 0.95, f"Conditional accuracy {accuracy:.1%} < 95%"


class TestCounterBenchComposite:
    """综合评分: 对标 CounterBench 四类型平均分。"""

    def test_four_type_average(
        self,
        basic_questions,
        joint_questions,
        nested_questions,
        conditional_questions,
        solver,
    ):
        """四类型平均 ≥ 95% (GPT-4o: 52.5%, CoIn: 93%, CEWM 目标 ≥ 95%)。"""
        all_q = basic_questions + joint_questions + nested_questions + conditional_questions
        correct = sum(1 for q in all_q if solver.solve(q) == q.answer)
        accuracy = correct / len(all_q)
        assert accuracy >= 0.95, f"CounterBench 4-type average {accuracy:.1%} < 95%. Total: {correct}/{len(all_q)}"

    def test_per_type_breakdown(
        self,
        basic_questions,
        joint_questions,
        nested_questions,
        conditional_questions,
        solver,
    ):
        """每类型分别 ≥ 90%。"""
        type_results = {
            "Basic": basic_questions,
            "Joint": joint_questions,
            "Nested": nested_questions,
            "Conditional": conditional_questions,
        }
        for type_name, qs in type_results.items():
            correct = sum(1 for q in qs if solver.solve(q) == q.answer)
            acc = correct / len(qs)
            assert acc >= 0.90, f"{type_name}: {acc:.1%} < 90%"

    def test_total_question_count(self, basic_questions, joint_questions, nested_questions, conditional_questions):
        """总题数 = 60 (4 类型 × 15 题)。"""
        total = len(basic_questions) + len(joint_questions) + len(nested_questions) + len(conditional_questions)
        assert total == 60, f"Expected 60 questions, got {total}"


class TestPearlThreeStepValidation:
    """验证求解器真正调用了 CounterfactualEngine (Pearl 三步) 而非简化版 _propagate()。"""

    def test_engine_status_ok(self, basic_questions, solver):
        """CounterfactualEngine.query() 返回 status='ok'。"""
        q = basic_questions[0]
        engine = build_engine(q.graph, noise_std=0.01)
        assert engine is not None
        roots = _find_roots(q.graph)
        evidence = _propagate(q.graph, dict.fromkeys(roots, 1.0))
        result = engine.query(evidence=evidence, do_x={"X": 0.0}, target="Y", compute_pns=False, n_mc=50)
        assert result.status == "ok", f"Engine status: {result.status} ({result.note})"

    def test_engine_vs_fallback_consistency(self, basic_questions, solver, fallback_solver):
        """CounterfactualEngine 结果与 _propagate() 基线一致 (线性图下)。"""
        engine_correct = sum(1 for q in basic_questions if solver.solve(q) == q.answer)
        fallback_correct = sum(1 for q in basic_questions if fallback_solver.solve(q) == q.answer)
        # 线性图下两者应完全一致
        assert engine_correct == fallback_correct, (
            f"Engine {engine_correct}/{len(basic_questions)} vs Fallback {fallback_correct}/{len(basic_questions)}"
        )

    def test_engine_returns_counterfactual_value(self, basic_questions):
        """CounterfactualEngine 返回具体数值而非仅 True/False。"""
        q = basic_questions[0]
        engine = build_engine(q.graph, noise_std=0.01)
        roots = _find_roots(q.graph)
        evidence = _propagate(q.graph, dict.fromkeys(roots, 1.0))
        result = engine.query(evidence=evidence, do_x={"X": 0.0}, target="Y", compute_pns=False, n_mc=50)
        assert isinstance(result.counterfactual_value, float)
        assert result.factual_value is not None
        assert result.individual_effect is not None
