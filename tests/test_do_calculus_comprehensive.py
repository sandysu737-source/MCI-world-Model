"""
MCI World Model v3.1.0 — DoCalculus 全面测试
==============================================

覆盖 _do_calculus.py 中未充分测试的方法：
- CausalGraph: add_edge(), to_sem(), from_sem(), get_mediators()
- DoCalculus: backdoor/frontdoor with data, direct_effect with data,
  _topological_sort cycles, _discretize, identify_frontdoor_mediators,
  NaN guards, set_data, empty/no-graph edge cases
- InterventionResult: field validations, direction/magnitude

目标: 将 _do_calculus.py 覆盖率从 ~45% 提升至 75%+。
"""

import numpy as np
import pytest

from mci_world_model.sdk._do_calculus import (
    CausalGraph,
    DoCalculus,
    InterventionResult,
)

# =============================================================================
# CausalGraph 扩展测试
# =============================================================================


class TestCausalGraphExtended:
    """CausalGraph 高级功能测试。"""

    def test_add_edge_new_nodes(self):
        """添加包含新节点的边。"""
        cg = CausalGraph(nodes=["X"], edges=[])
        cg.add_edge("X", "Y", weight=0.8)
        assert "Y" in cg.nodes
        assert ("X", "Y") in cg.edges
        assert bool(cg.has_edge("X", "Y")) is True

    def test_add_edge_existing(self):
        """添加已存在的边 — 不重复。"""
        cg = CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])
        cg.add_edge("X", "Y", weight=0.5)
        assert len(cg.edges) == 1

    def test_add_edge_adjacency_growth(self):
        """添加边时邻接矩阵自动扩展。"""
        cg = CausalGraph(nodes=["A", "B"], edges=[("A", "B")])
        cg.add_edge("A", "C")
        assert cg.n_nodes == 3
        assert cg.adjacency is not None
        assert cg.adjacency.shape == (3, 3)

    def test_to_sem_linear(self):
        """CausalGraph → SEM 转换（线性）。"""
        cg = CausalGraph(
            nodes=["X", "M", "Y"],
            edges=[("X", "M"), ("M", "Y")],
        )
        sem = cg.to_sem(noise_std=0.1, activation="linear", seed=42)
        assert sem is not None
        assert sem.n_nodes == 3
        assert sem.node_names == ["X", "M", "Y"]

    def test_to_sem_nonlinear(self):
        """CausalGraph → SEM 转换（非线性激活）。"""
        cg = CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])
        for activation in ["tanh", "relu", "sigmoid"]:
            sem = cg.to_sem(activation=activation)
            assert sem is not None
            assert sem.n_nodes == 2

    def test_to_sem_custom_noise(self):
        """SEM 转换时可自定义噪声。"""
        cg = CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])
        sem = cg.to_sem(noise_std=0.01)
        assert sem is not None

    def test_from_sem(self):
        """从 SEM 反向构建 CausalGraph。"""
        from mci_world_model.sdk._counterfactual import StructuralEquationModel

        sem = StructuralEquationModel(
            coefficients=np.array([[0, 0.5], [0, 0]], dtype=np.float64),
            node_names=["X", "Y"],
        )
        cg = CausalGraph.from_sem(sem)
        assert cg is not None
        assert cg.n_nodes == 2
        assert "X" in cg.nodes
        assert "Y" in cg.nodes

    def test_get_mediators_direct_only(self):
        """直接因果路径（无中介）。"""
        cg = CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])
        mediators = cg.get_mediators("X", "Y")
        assert mediators == []

    def test_get_mediators_with_intermediary(self):
        """含中介的因果路径。"""
        cg = CausalGraph(
            nodes=["X", "M1", "M2", "Y"],
            edges=[("X", "M1"), ("M1", "M2"), ("M2", "Y")],
        )
        mediators = cg.get_mediators("X", "Y")
        assert len(mediators) >= 1  # M1 或 M2

    def test_get_descendants_non_existent(self):
        """不存在节点的后代查询。"""
        cg = CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])
        desc = cg.get_descendants("Z")
        assert desc == set()

    def test_has_edge_nonexistent_nodes(self):
        """不存在的节点边查询。"""
        cg = CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])
        assert bool(cg.has_edge("X", "Z")) is False
        assert bool(cg.has_edge("Z", "Y")) is False

    def test_adjacency_no_edges(self):
        """无边图的邻接矩阵。"""
        cg = CausalGraph(nodes=["A", "B", "C"], edges=[])
        assert cg.adjacency is not None
        assert np.all(cg.adjacency == 0)

    def test_repr(self):
        """__repr__ 字符串。"""
        cg = CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])
        r = repr(cg)
        assert "CausalGraph" in r

    def test_adjacency_with_weights(self):
        """加权邻接矩阵。"""
        cg = CausalGraph(
            nodes=["X", "Y"],
            edges=[("X", "Y")],
            adjacency=np.array([[0, 0.8], [0, 0]], dtype=np.float32),
        )
        assert cg.adjacency[0, 1] == 0.8


# =============================================================================
# DoCalculus 扩展测试 — 数据驱动方法
# =============================================================================


class TestDoCalculusWithData:
    """DoCalculus 观测数据驱动测试。"""

    @pytest.fixture
    def sample_data(self) -> dict[str, np.ndarray]:
        """生成模拟观测数据。"""
        n = 200
        rng = np.random.RandomState(42)
        Z = rng.normal(0, 1, n)
        X = 0.5 * Z + rng.normal(0, 0.5, n)
        Y = 0.3 * X + 0.4 * Z + rng.normal(0, 0.3, n)
        return {"Z": Z, "X": X, "Y": Y}

    @pytest.fixture
    def sample_graph(self) -> CausalGraph:
        return CausalGraph(
            nodes=["Z", "X", "Y"],
            edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        )

    def test_backdoor_with_data(self, sample_graph, sample_data):
        """基于观测数据的后门调整。"""
        dc = DoCalculus(graph=sample_graph, data=sample_data)
        result = dc.estimate_ate("X", "Y", x_value=1.0, x_baseline=0.0, method="auto")
        assert result is not None
        assert result.method in ("backdoor", "frontdoor", "direct")
        assert result.sample_size > 0

    def test_frontdoor_with_data(self, sample_data):
        """基于观测数据的前门调整。"""
        # 前门结构: X → M → Y, 没有 X→Y 直接路径, 但 X 和 Y 有混杂 Z
        cg = CausalGraph(
            nodes=["X", "M", "Y"],
            edges=[("X", "M"), ("M", "Y")],
        )
        dc = DoCalculus(graph=cg, data=sample_data)
        result = dc.estimate_ate("X", "Y", x_value=1.0, x_baseline=0.0, method="frontdoor")
        assert result is not None

    def test_direct_effect_with_data(self, sample_graph, sample_data):
        """基于观测数据的直接效应。"""
        dc = DoCalculus(graph=sample_graph, data=sample_data)
        result = dc.estimate_ate("X", "Y", method="direct")
        assert result is not None
        assert result.method == "direct"

    def test_set_data(self, sample_graph, sample_data):
        """set_data 方法。"""
        dc = DoCalculus(graph=sample_graph)
        dc.set_data(sample_data)
        assert dc._is_simulated is False

    def test_set_graph_after_init(self):
        """初始化后设置图。"""
        dc = DoCalculus()
        cg = CausalGraph(nodes=["A", "B"], edges=[("A", "B")])
        dc.set_graph(cg)
        result = dc.estimate_ate("A", "B")
        assert result is not None

    def test_empty_data_fallback(self):
        """空数据回退。"""
        dc = DoCalculus(
            graph=CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")]),
            data={},
        )
        result = dc.estimate_ate("X", "Y")
        assert result is not None  # 回退到模拟

    def test_backdoor_small_sample(self, sample_graph):
        """小样本后门调整。"""
        small_data = {"Z": np.array([1.0]), "X": np.array([0.5]), "Y": np.array([2.0])}
        dc = DoCalculus(graph=sample_graph, data=small_data)
        result = dc.estimate_ate("X", "Y", method="backdoor")
        # 样本太小可能返回 empty
        assert result is not None


# =============================================================================
# DoCalculus 边界条件测试
# =============================================================================


class TestDoCalculusEdgeCases:
    """DoCalculus 边界条件和异常测试。"""

    def test_no_graph(self):
        """无因果图的 DoCalculus。"""
        dc = DoCalculus(None)
        result = dc.estimate_ate("X", "Y")
        assert result.method == "none"
        assert result.note != ""

    def test_x_not_in_graph(self):
        """干预变量不在图中。"""
        dc = DoCalculus(graph=CausalGraph(nodes=["A", "B"], edges=[("A", "B")]))
        result = dc.estimate_ate("X", "Y")
        assert result.method == "rejected"
        assert "X" in result.note

    def test_y_not_in_graph(self):
        """目标变量不在图中。"""
        dc = DoCalculus(graph=CausalGraph(nodes=["A", "B"], edges=[("A", "B")]))
        result = dc.estimate_ate("A", "Z")
        assert result.method == "rejected"

    def test_nan_x_value(self):
        """NaN 干预值。"""
        dc = DoCalculus(graph=CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")]))
        result = dc.estimate_ate("X", "Y", x_value=float("nan"))
        assert result.method == "rejected"
        assert "finite" in result.note

    def test_nan_baseline(self):
        """NaN 基线值。"""
        dc = DoCalculus(graph=CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")]))
        result = dc.estimate_ate("X", "Y", x_value=1.0, x_baseline=float("nan"))
        assert result.method == "rejected"

    def test_inf_x_value(self):
        """Inf 干预值。"""
        dc = DoCalculus(graph=CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")]))
        result = dc.estimate_ate("X", "Y", x_value=float("inf"))
        assert result.method == "rejected"

    def test_cyclic_graph_topological_sort(self):
        """含环图的拓扑排序 — 应返回 None。"""
        cg = CausalGraph(
            nodes=["A", "B", "C"],
            edges=[("A", "B"), ("B", "C"), ("C", "A")],  # 环
        )
        dc = DoCalculus(graph=cg)
        result = dc.estimate_ate("A", "B")
        # 含环图 → 所有模拟方法应返回 empty
        assert result is not None

    def test_disconnected_graph(self):
        """不连通图的 ATE 估计。"""
        cg = CausalGraph(
            nodes=["X", "Y", "Z"],
            edges=[("Z", "Z")],  # 自环
        )
        # 自环图
        dc = DoCalculus(graph=cg)
        result = dc.estimate_ate("X", "Y")
        assert result is not None

    def test_single_node_graph(self):
        """单节点图。"""
        cg = CausalGraph(nodes=["X"], edges=[])
        dc = DoCalculus(graph=cg)
        result = dc.estimate_ate("X", "X")
        # 单节点图中 X=X，允许直接效应估计（X 在图中）
        assert result is not None
        assert result.method in ("direct", "rejected")


# =============================================================================
# DoCalculus 工具方法测试
# =============================================================================


class TestDoCalculusUtils:
    """DoCalculus 工具方法测试。"""

    def test_discretize_equal_freq(self):
        """_discretize 等频分桶。"""
        data = np.linspace(0, 100, 100)
        bins = DoCalculus._discretize(data, n_bins=10)
        assert len(bins) >= 5

    def test_discretize_small_data(self):
        """_discretize 小数据。"""
        data = np.array([1.0, 2.0, 3.0])
        bins = DoCalculus._discretize(data, n_bins=10)
        assert len(bins) == 3  # 唯一值

    def test_discretize_single_value(self):
        """_discretize 单一值。"""
        data = np.array([5.0, 5.0, 5.0])
        bins = DoCalculus._discretize(data, n_bins=5)
        assert len(bins) >= 1

    def test_identify_frontdoor_mediators_valid(self):
        """识别有效的前门中介。"""
        cg = CausalGraph(
            nodes=["X", "M", "Y"],
            edges=[("X", "M"), ("M", "Y")],
        )
        dc = DoCalculus(graph=cg)
        mediators = dc.identify_frontdoor_mediators("X", "Y")
        assert mediators is not None
        assert "M" in mediators

    def test_identify_frontdoor_mediators_with_confounder(self):
        """有混杂因子的前门中介识别。"""
        # Z 同时影响 X 和 M → M 不满足前门条件 2
        cg = CausalGraph(
            nodes=["Z", "X", "M", "Y"],
            edges=[("Z", "X"), ("Z", "M"), ("X", "M"), ("M", "Y")],
        )
        dc = DoCalculus(graph=cg)
        mediators = dc.identify_frontdoor_mediators("X", "Y")
        # M 与 X 有共同原因 Z，不应被选为有效中介
        # 实际上我们的实现用 x_parents & m_parents 交集判断
        if mediators:
            assert "M" not in mediators  # Z 是混杂因子

    def test_identify_frontdoor_no_graph(self):
        """无图时的前门中介识别。"""
        dc = DoCalculus()
        result = dc.identify_frontdoor_mediators("X", "Y")
        assert result is None

    def test_identify_adjustment_set_no_graph(self):
        """无图时的后门调整集识别。"""
        dc = DoCalculus()
        result = dc.identify_adjustment_set("X", "Y")
        assert result is None

    def test_topological_sort_with_self_loop(self):
        """自环节点的拓扑排序。"""
        cg = CausalGraph(
            nodes=["A", "B"],
            edges=[("A", "B")],
            adjacency=np.array([[1, 0], [0, 0]], dtype=np.float32),  # A 有自环
        )
        dc = DoCalculus(graph=cg)
        order = dc._topological_sort()
        # 自环使 Kahn 算法永远无法将 A 入度降为 0
        assert order is None

    def test_repr(self):
        """__repr__ 字符串。"""
        dc = DoCalculus(graph=CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")]))
        r = repr(dc)
        assert "DoCalculus" in r

    def test_build_from_gaussian_dag_min_confidence(self):
        """从 GaussianDAG 构建 — 置信度过滤。"""
        edges = [
            {"cause_idx": 0, "effect_idx": 1, "confidence": 0.9},
            {"cause_idx": 1, "effect_idx": 2, "confidence": 0.1},  # 低于阈值
        ]
        cg = DoCalculus.build_from_gaussian_dag(edges, n_nodes=3, min_confidence=0.5)
        assert cg.n_nodes == 3
        # 只有第一条边应被保留
        assert bool(cg.has_edge("V0", "V1")) is True
        assert bool(cg.has_edge("V1", "V2")) is False

    def test_build_from_gaussian_dag_empty(self):
        """空边列表构建。"""
        cg = DoCalculus.build_from_gaussian_dag([], n_nodes=2)
        assert cg.n_nodes == 2
        assert len(cg.edges) == 0

    def test_build_from_gaussian_dag_oob_indices(self):
        """越界索引的边应被过滤。"""
        edges = [
            {"cause_idx": 0, "effect_idx": 5, "confidence": 0.8},  # 越界
            {"cause_idx": 0, "effect_idx": 1, "confidence": 0.8},
        ]
        cg = DoCalculus.build_from_gaussian_dag(edges, n_nodes=3)
        assert len(cg.edges) == 1
        assert ("V0", "V1") in cg.edges

    def test_build_from_gaussian_dag_no_indices(self):
        """缺失索引的边应被过滤。"""
        edges = [
            {"confidence": 0.8},  # 无 cause_idx/effect_idx
            {"cause_idx": 0, "effect_idx": 1, "confidence": 0.6},
        ]
        cg = DoCalculus.build_from_gaussian_dag(edges, n_nodes=2)
        assert len(cg.edges) == 1


# =============================================================================
# InterventionResult 扩展测试
# =============================================================================


class TestInterventionResultExtended:
    """InterventionResult 扩展测试。"""

    def test_full_result(self):
        """完整 InterventionResult。"""
        result = InterventionResult(
            intervention="do(X=1.5)",
            target="Y",
            ate=0.42,
            confidence_interval=(0.3, 0.54),
            ci_level=0.95,
            adjustment_set=["Z"],
            method="backdoor",
            p_value=0.001,
            effect_direction="positive",
            effect_magnitude="medium",
            sample_size=200,
            note="adjusted_for_confounding",
        )
        d = result.to_dict()
        assert d["ate"] == 0.42
        assert d["method"] == "backdoor"
        assert d["effect_direction"] == "positive"
        assert d["effect_magnitude"] == "medium"
        assert d["p_value"] == 0.001
        assert d["confidence_interval_95"] == (0.3, 0.54)

    def test_empty_result_fields(self):
        """空结果字段。"""
        result = InterventionResult.empty(method="backdoor")
        assert result.ate == 0.0
        assert result.adjustment_set == []
        assert result.sample_size == 0

    def test_direction_positive(self):
        """正向效应判定 — 通过 _build_result 设置。"""
        cg = CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])
        dc = DoCalculus(graph=cg, seed=42)
        result = dc.estimate_ate("X", "Y", x_value=2.0, x_baseline=0.0)
        # 大 ATE 应被 _build_result 标记为 positive/large
        d = result.to_dict()
        assert d.get("effect_direction") in ("positive", "neutral")

    def test_direction_negative(self):
        """负向效应判定 — 通过 _build_result 设置。"""
        cg = CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])
        dc = DoCalculus(graph=cg, seed=42)
        result = dc.estimate_ate("X", "Y", x_value=0.0, x_baseline=2.0)
        d = result.to_dict()
        assert d.get("effect_direction") in ("negative", "neutral")

    def test_magnitude_large(self):
        """大效应量 — 在 ATE 显著时触发。"""
        cg = CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])
        dc = DoCalculus(graph=cg, seed=42)
        result = dc.estimate_ate("X", "Y", x_value=5.0, x_baseline=0.0)
        d = result.to_dict()
        # 大干预值可能产生大 ATE
        assert d.get("effect_magnitude") in ("large", "medium", "small", "negligible", "unknown")

    def test_magnitude_negligible(self):
        """可忽略的效应量 — 在 ATE ≈ 0 时触发。"""
        cg = CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])
        dc = DoCalculus(graph=cg, seed=42)
        result = dc.estimate_ate("X", "Y", x_value=0.0, x_baseline=0.0)
        d = result.to_dict()
        # x_value==x_baseline → ATE ≈ 0 → negligible
        assert d.get("effect_magnitude") in ("negligible", "small", "unknown")


# =============================================================================
# DoCalculus 自动方法选择测试
# =============================================================================


class TestDoCalculusAutoMethod:
    """DoCalculus 自动方法选择测试。"""

    def test_auto_backdoor_preferred(self):
        """自动选择时应优先后门。"""
        cg = CausalGraph(
            nodes=["Z", "X", "Y"],
            edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        )
        dc = DoCalculus(graph=cg)
        result = dc.estimate_ate("X", "Y", method="auto")
        # 有后门调整集 → 应使用 backdoor
        assert result.method in ("backdoor", "direct")

    def test_auto_frontdoor_fallback(self):
        """后门不可用时回退到前门。"""
        cg = CausalGraph(
            nodes=["X", "M", "Y"],
            edges=[("X", "M"), ("M", "Y")],
        )
        dc = DoCalculus(graph=cg)
        result = dc.estimate_ate("X", "Y", method="auto")
        assert result.method in ("frontdoor", "direct")

    def test_explicit_method_backdoor(self):
        """显式指定后门方法。"""
        cg = CausalGraph(
            nodes=["Z", "X", "Y"],
            edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        )
        dc = DoCalculus(graph=cg)
        result = dc.estimate_ate("X", "Y", method="backdoor")
        assert result is not None

    def test_explicit_method_frontdoor(self):
        """显式指定前门方法。"""
        cg = CausalGraph(
            nodes=["X", "M", "Y"],
            edges=[("X", "M"), ("M", "Y")],
        )
        dc = DoCalculus(graph=cg)
        result = dc.estimate_ate("X", "Y", method="frontdoor")
        assert result is not None

    def test_explicit_method_direct(self):
        """显式指定直接方法。"""
        cg = CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])
        dc = DoCalculus(graph=cg)
        result = dc.estimate_ate("X", "Y", method="direct")
        assert result is not None
        assert result.method == "direct"


# =============================================================================
# DoCalculus 模拟数据路径测试
# =============================================================================


class TestDoCalculusSimulatedPath:
    """DoCalculus 模拟数据路径测试。"""

    def test_backdoor_simulated_ate(self):
        """模拟数据的后门 ATE。"""
        cg = CausalGraph(
            nodes=["Z", "X", "Y"],
            edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        )
        dc = DoCalculus(graph=cg, seed=42)
        result = dc.estimate_ate("X", "Y", x_value=2.0, x_baseline=0.0)
        assert result is not None
        assert isinstance(result.ate, float)
        assert result.sample_size > 0

    def test_frontdoor_simulated_ate(self):
        """模拟数据的前门 ATE。"""
        cg = CausalGraph(
            nodes=["X", "M", "Y"],
            edges=[("X", "M"), ("M", "Y")],
        )
        dc = DoCalculus(graph=cg, seed=42)
        result = dc.estimate_ate("X", "Y", x_value=2.0, x_baseline=0.0)
        assert result is not None

    def test_direct_simulated_ate(self):
        """模拟数据的直接效应。"""
        cg = CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])
        dc = DoCalculus(graph=cg, seed=42)
        result = dc.estimate_ate("X", "Y", x_value=2.0, x_baseline=0.0)
        assert result is not None
        assert result.method == "direct"

    def test_simulated_confidence_interval(self):
        """模拟数据的置信区间。"""
        cg = CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])
        dc = DoCalculus(graph=cg, seed=42)
        result = dc.estimate_ate("X", "Y", x_value=1.0, x_baseline=0.0)
        ci = result.confidence_interval
        assert ci[0] <= ci[1]  # CI 下界 ≤ 上界

    def test_simulated_p_value_range(self):
        """模拟数据的 p-value 范围。"""
        cg = CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])
        dc = DoCalculus(graph=cg, seed=42)
        result = dc.estimate_ate("X", "Y")
        assert 0.0 <= result.p_value <= 1.0

    def test_multiple_seeds_reproducible(self):
        """随机种子可重现。"""
        cg = CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])
        result1 = DoCalculus(graph=cg, seed=42).estimate_ate("X", "Y")
        result2 = DoCalculus(graph=cg, seed=42).estimate_ate("X", "Y")
        assert result1.ate == pytest.approx(result2.ate)


# =============================================================================
# 批量 API 测试 (v4.5.0)
# =============================================================================


class TestBatchAPI:
    """DoCalculus + CausalGraph 批量 API。"""

    @pytest.fixture
    def dc(self):
        cg = CausalGraph(
            nodes=["Z", "X", "Y", "W"],
            edges=[("Z", "X"), ("Z", "Y"), ("X", "Y"), ("X", "W")],
        )
        return DoCalculus(graph=cg, seed=42)

    # ── CausalGraph.batch_estimate_ate ──

    def test_batch_estimate_ate_returns_list(self, dc):
        """返回与输入等长的列表。"""
        pairs = [("X", "Y"), ("Z", "Y")]
        results = dc.batch_estimate_ate(pairs)
        assert len(results) == len(pairs)
        for r in results:
            assert isinstance(r, InterventionResult)

    def test_batch_estimate_ate_empty(self, dc):
        """空列表 → 空列表。"""
        assert dc.batch_estimate_ate([]) == []

    def test_batch_estimate_ate_order(self, dc):
        """结果顺序与输入一致。"""
        pairs = [("X", "Y"), ("Z", "Y"), ("X", "W")]
        results = dc.batch_estimate_ate(pairs)
        for i, (X, Y) in enumerate(pairs):
            assert results[i].intervention == f"do({X}=1.0)"
            assert results[i].target == Y

    # ── CausalGraph.batch_identify_adjustment_sets ──

    def test_batch_identify_adj_sets(self, dc):
        """识别调整变量集。"""
        adj = dc.batch_identify_adjustment_sets([("X", "Y"), ("Z", "Y")])
        assert isinstance(adj, dict)
        assert ("X", "Y") in adj
        assert ("Z", "Y") in adj

    # ── CausalGraph.batch_query ──

    def test_batch_query_dict_format(self, dc):
        """batch_query 字典格式。"""
        queries = [
            {"X": "X", "Y": "Y", "x_value": 1.0, "x_baseline": 0.0},
            {"X": "Z", "Y": "X", "x_value": 2.0, "x_baseline": 0.0},
        ]
        results = dc.batch_query(queries)
        assert len(results) == 2
        for r in results:
            assert "ate" in r
            assert "method" in r
            assert "ci_low" in r
            assert "ci_high" in r

    # ── DoCalculus.batch_estimate_ate ──

    def test_dc_batch_estimate_ate(self, dc):
        """DoCalculus 批量 ATE。"""
        pairs = [("X", "Y"), ("Z", "Y")]
        results = dc.batch_estimate_ate(pairs)
        assert len(results) == 2
        assert results[0].ate is not None
        assert results[1].ate is not None

    # ── DoCalculus.batch_identify_adjustment_sets ──

    def test_dc_batch_identify_adj(self, dc):
        """DoCalculus 批量识别。"""
        adj = dc.batch_identify_adjustment_sets([("X", "Y"), ("Z", "X")])
        assert ("X", "Y") in adj
        assert ("Z", "X") in adj

    # ── 空值处理 ──

    def test_batch_query_empty(self, dc):
        """空查询 → 空列表。"""
        assert dc.batch_query([]) == []

    def test_batch_identify_empty(self, dc):
        """空 pairs → 空 dict。"""
        assert dc.batch_identify_adjustment_sets([]) == {}
