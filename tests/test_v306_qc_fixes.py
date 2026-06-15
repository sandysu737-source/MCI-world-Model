"""
test_v306_qc_fixes.py — v3.0.6 QC 审计回归测试。

覆盖 P0/P1/P2 全部修复项:
- P0: _get_configurator / _get_causal_actor 惰性 getter
- P0: auto_regulate() 状态链 + converged 逻辑
- P1: pyproject.toml 版本同步
- P2: _aggregate_energy_ratios 公共聚合函数
- P2: normalize_edge 能量属性补全
- P2: health_check 含 energy_coverage 字段
"""

import os
from dataclasses import dataclass, field
from unittest.mock import MagicMock

# =============================================================================
# Helper: 最小化 MCIWorldModel 实例 (绕过 initialize)
# =============================================================================


def _make_bare_model():
    """构造未初始化的 MCIWorldModel，所有组件为 None。"""
    from mci_world_model.sdk._world_model import MCIWorldModel

    wm = MCIWorldModel.__new__(MCIWorldModel)
    wm._lite_pro = None
    wm._config = {}
    wm._initialized = False
    wm._parametric = None
    wm._energy_loss = None
    wm._cost_module = None
    wm._configurator = None
    wm._hierarchical_encoder = None
    wm._causal_actor = None
    wm._perception = None
    wm._energy_core = None
    wm._temporal_core = None
    wm._jepa_encoder = None
    wm._jepa_predictor = None
    wm._do_calculus = None
    wm._cognitive_loop = None
    wm._meta_diagnoser = None
    wm._multi_view_retriever = None
    wm._surprise_detector = None
    wm._plan_agent = None
    wm._action_conditioned_predictor = None
    wm._multi_branch_predictor = None
    wm._reflection_synthesizer = None
    wm._cognitive_diversity = None
    wm._negative_heuristic = None
    wm._parametric_memory = None
    wm._energy_flow_predictor = None

    from mci_world_model.sdk._world_model import CausalWorldModelState

    wm._state = CausalWorldModelState.empty()
    return wm


def _make_state_with_edges(causal_edges: list[dict]):
    """构造含指定 causal_edges 的 state。"""
    from mci_world_model.sdk._world_model import CausalWorldModelState

    state = CausalWorldModelState.empty()
    state.causal_edges = causal_edges
    return state


# =============================================================================
# 1) _aggregate_energy_ratios 公共函数
# =============================================================================


class TestAggregateEnergyRatios:
    """P2: 公共聚合函数正确归一化。"""

    def test_empty_returns_none(self):
        from mci_world_model.sdk._world_model import _aggregate_energy_ratios

        assert _aggregate_energy_ratios([]) is None
        assert _aggregate_energy_ratios(None) is None

    def test_no_energy_labels_returns_none(self):
        from mci_world_model.sdk._world_model import _aggregate_energy_ratios

        edges = [{"cause": "A", "effect": "B"}]
        assert _aggregate_energy_ratios(edges) is None

    def test_correct_normalization(self):
        from mci_world_model.sdk._world_model import _aggregate_energy_ratios

        edges = [
            {"cause_energy": "semantic", "effect_energy": "causal"},
            {"cause_energy": "semantic", "effect_energy": "spacetime"},
        ]
        result = _aggregate_energy_ratios(edges)
        assert result is not None
        # 总计: semantic=2, causal=1, spacetime=1 → total=4
        assert abs(result["semantic"] - 0.5) < 1e-9
        assert abs(result["causal"] - 0.25) < 1e-9
        assert abs(result["spacetime"] - 0.25) < 1e-9
        assert result["generative"] == 0.0
        assert result["trust"] == 0.0

    def test_all_five_dimensions(self):
        from mci_world_model.sdk._world_model import _aggregate_energy_ratios

        dims = ["semantic", "causal", "spacetime", "generative", "trust"]
        edges = [{"cause_energy": d, "effect_energy": d} for d in dims]
        result = _aggregate_energy_ratios(edges)
        assert result is not None
        # 每维度 2 次出现, total=10 → 每维度 0.2
        for d in dims:
            assert abs(result[d] - 0.2) < 1e-9


# =============================================================================
# 2) _extract_energy_ratios 委托到公共函数
# =============================================================================


class TestExtractEnergyRatios:
    """验证 MCIWorldModel._extract_energy_ratios 正确委托。"""

    def test_empty_state_returns_none(self):
        wm = _make_bare_model()
        assert wm._extract_energy_ratios(wm._state) is None

    def test_with_edges_delegates(self):
        wm = _make_bare_model()
        wm._state = _make_state_with_edges(
            [
                {"cause_energy": "semantic", "effect_energy": "causal"},
            ]
        )
        result = wm._extract_energy_ratios(wm._state)
        assert result is not None
        assert result["semantic"] > 0
        assert result["causal"] > 0

    def test_state_without_causal_edges_attr(self):
        wm = _make_bare_model()

        @dataclass
        class FakeState:
            pass

        assert wm._extract_energy_ratios(FakeState()) is None


# =============================================================================
# 3) _get_configurator 惰性 getter
# =============================================================================


class TestGetConfigurator:
    """P0: _get_configurator() 返回 HierarchicalConfigurator。"""

    def test_returns_instance(self):
        from mci_world_model._sys._configurator import HierarchicalConfigurator

        wm = _make_bare_model()
        result = wm._get_configurator()
        assert isinstance(result, HierarchicalConfigurator)

    def test_cached_on_second_call(self):
        wm = _make_bare_model()
        first = wm._get_configurator()
        second = wm._get_configurator()
        assert first is second

    def test_passes_energy_core(self):
        wm = _make_bare_model()
        mock_ec = MagicMock()
        wm._energy_core = mock_ec
        result = wm._get_configurator()
        assert result._energy_core is mock_ec


# =============================================================================
# 4) _get_causal_actor 惰性 getter
# =============================================================================


class TestGetCausalActor:
    """P0: _get_causal_actor() 返回 CausalActor。"""

    def test_returns_instance(self):
        from mci_world_model.sdk._causal_actor import CausalActor

        wm = _make_bare_model()
        result = wm._get_causal_actor()
        assert isinstance(result, CausalActor)

    def test_cached_on_second_call(self):
        wm = _make_bare_model()
        first = wm._get_causal_actor()
        second = wm._get_causal_actor()
        assert first is second

    def test_actor_receives_energy_core(self):
        wm = _make_bare_model()
        mock_ec = MagicMock()
        wm._energy_core = mock_ec
        actor = wm._get_causal_actor()
        assert actor._energy_core is mock_ec


# =============================================================================
# 5) auto_regulate 状态链 + converged 逻辑
# =============================================================================


class TestAutoRegulate:
    """P0: auto_regulate 状态链 + converged 逻辑验证。"""

    def test_returns_valid_structure(self):
        wm = _make_bare_model()
        # mock 所有依赖
        mock_ec = MagicMock()
        mock_ec.analyze_balance.return_value = MagicMock(status="balanced")
        wm._energy_core = mock_ec
        wm._configurator = MagicMock()
        wm._causal_actor = MagicMock()
        wm._state = _make_state_with_edges(
            [
                {"cause_energy": "semantic", "effect_energy": "causal"},
            ]
        )

        result = wm.auto_regulate()
        assert "iterations" in result
        assert "history" in result
        assert "converged" in result
        assert "no_energy_data" in result

    def test_no_energy_data(self):
        """无能量数据时 converged=False, no_energy_data=True。"""
        wm = _make_bare_model()
        mock_ec = MagicMock()
        wm._energy_core = mock_ec
        wm._configurator = MagicMock()
        wm._causal_actor = MagicMock()
        # state 无 causal_edges → ratios = None

        result = wm.auto_regulate()
        assert result["converged"] is False
        assert result["no_energy_data"] is True
        assert result["iterations"] == 0

    def test_converged_on_balanced(self):
        """已平衡时 converged=True。"""
        wm = _make_bare_model()
        wm._state = _make_state_with_edges(
            [
                {"cause_energy": "semantic", "effect_energy": "causal"},
            ]
        )

        mock_ec = MagicMock()
        mock_ec.analyze_balance.return_value = MagicMock(status="balanced")
        wm._energy_core = mock_ec
        wm._configurator = MagicMock()
        wm._causal_actor = MagicMock()

        result = wm.auto_regulate()
        assert result["converged"] is True

    def test_state_chain_via_mock(self):
        """验证 actor.apply 返回值链式传递。"""
        wm = _make_bare_model()
        wm._state = _make_state_with_edges(
            [
                {"cause_energy": "semantic", "effect_energy": "causal"},
            ]
        )

        mock_ec = MagicMock()
        # 第一次不平衡，第二次平衡 → 迭代 1 次后收敛
        call_count = {"n": 0}

        def fake_analyze(ratios):
            call_count["n"] += 1
            if call_count["n"] <= 1:
                return MagicMock(status="imbalanced", dominant="semantic")
            return MagicMock(status="balanced")

        mock_ec.analyze_balance.side_effect = fake_analyze
        wm._energy_core = mock_ec

        mock_configurator = MagicMock()
        mock_configurator.configure.return_value = []
        wm._configurator = mock_configurator

        # 创建 mock actor，apply 返回新 state
        new_state = _make_state_with_edges(
            [
                {"cause_energy": "semantic", "effect_energy": "causal"},
                {"cause_energy": "causal", "effect_energy": "trust"},
            ]
        )

        mock_actor = MagicMock()
        mock_actor.search.return_value = [MagicMock()]
        mock_actor.apply.return_value = new_state
        wm._causal_actor = mock_actor

        wm.auto_regulate(max_iterations=3)

        # 验证 apply 被调用
        assert mock_actor.apply.called
        # 最终状态被写回
        assert wm._state is new_state

    def test_exception_breaks_loop(self):
        """单次异常不会崩溃，而是安全退出。"""
        wm = _make_bare_model()
        wm._state = _make_state_with_edges(
            [
                {"cause_energy": "semantic", "effect_energy": "causal"},
            ]
        )

        mock_ec = MagicMock()
        mock_ec.analyze_balance.return_value = MagicMock(status="imbalanced", dominant="semantic")
        wm._energy_core = mock_ec

        mock_configurator = MagicMock()
        mock_configurator.configure.side_effect = RuntimeError("boom")
        wm._configurator = mock_configurator

        mock_actor = MagicMock()
        wm._causal_actor = mock_actor

        # 不应抛出异常
        result = wm.auto_regulate()
        assert result["converged"] is False
        assert result["iterations"] == 0


# =============================================================================
# 6) normalize_edge 能量属性补全
# =============================================================================


class TestNormalizeEdge:
    """P2: normalize_edge 自动补全 energy_relation/strength。"""

    def test_passthrough_when_no_energy_core(self):
        wm = _make_bare_model()
        edge = {"cause": "A", "effect": "B", "rho": 0.8}
        result = wm.normalize_edge(edge)
        # 无 energy_core → 不补全能量属性
        assert "energy_relation" not in result
        assert "energy_strength" not in result

    def test_auto_fills_with_mock_energy_core(self):
        wm = _make_bare_model()
        mock_ec = MagicMock()
        mock_ec.analyze_interaction.return_value = [MagicMock(name="enhance")]
        mock_ec.get_energy_state.return_value = MagicMock(strength=MagicMock(name="WANG"))
        wm._energy_core = mock_ec

        edge = {
            "cause": "A",
            "effect": "B",
            "cause_energy": "wood",
            "effect_energy": "fire",
        }
        result = wm.normalize_edge(edge)
        assert "energy_relation" in result
        assert "energy_strength" in result

    def test_original_edge_not_mutated(self):
        wm = _make_bare_model()
        edge = {"cause": "A", "effect": "B"}
        original_keys = set(edge.keys())
        wm.normalize_edge(edge)
        assert set(edge.keys()) == original_keys


# =============================================================================
# 7) health_check 含 energy_coverage
# =============================================================================


class TestHealthCheckEnergyCoverage:
    """P2: health_check 返回含 energy_coverage 字段。"""

    def test_energy_coverage_key_exists(self):
        wm = _make_bare_model()
        result = wm.health_check()
        assert "energy_coverage" in result

    def test_energy_coverage_structure(self):
        wm = _make_bare_model()
        result = wm.health_check()
        ec = result["energy_coverage"]
        assert "coverage_score" in ec
        assert "ratios" in ec


# =============================================================================
# 8) pyproject.toml version == __version__
# =============================================================================


class TestVersionSync:
    """P1: pyproject.toml 版本与 __init__.py __version__ 一致。"""

    def test_pyproject_matches_init(self):
        import tomllib

        import mci_world_model

        pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(mci_world_model.__file__)))
        pyproject_path = os.path.join(pkg_dir, "pyproject.toml")

        with open(pyproject_path, "rb") as fh:
            data = tomllib.load(fh)

        assert data["project"]["version"] == mci_world_model.__version__


# =============================================================================
# 9) _extract_energy_ratios_from_state 委托
# =============================================================================


class TestConfiguratorEnergyExtractor:
    """P2: _sys._configurator 中的提取函数正确委托。"""

    def test_delegates_to_aggregate(self):
        from mci_world_model._sys._configurator import (
            _extract_energy_ratios_from_state,
        )

        state = _make_state_with_edges(
            [
                {"cause_energy": "semantic", "effect_energy": "trust"},
            ]
        )
        result = _extract_energy_ratios_from_state(state)
        assert result is not None
        assert abs(result["semantic"] - 0.5) < 1e-9
        assert abs(result["trust"] - 0.5) < 1e-9

    def test_none_for_empty_state(self):
        from mci_world_model._sys._configurator import (
            _extract_energy_ratios_from_state,
        )

        @dataclass
        class EmptyState:
            causal_edges: list = field(default_factory=list)

        assert _extract_energy_ratios_from_state(EmptyState()) is None

    def test_none_for_no_attr(self):
        from mci_world_model._sys._configurator import (
            _extract_energy_ratios_from_state,
        )

        @dataclass
        class NoAttrState:
            pass

        assert _extract_energy_ratios_from_state(NoAttrState()) is None
