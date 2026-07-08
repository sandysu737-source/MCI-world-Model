"""L1 解析 oracle 测试 — d-separation 与后门调整集 (Pearl 图论)。

验证 algebra.CausalDAG 的 d-separation 和调整集判定是否符合 Pearl
(2009) Causality 的定义。这些都是图论的确定性结论, 可作解析 oracle。
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.oracle

import pytest

from mci_world_model.algebra.causal_graph import CausalDAG


# =============================================================================
# 经典图结构 — 每个的 d-separation 结论都是教科书确定答案
# =============================================================================

@pytest.fixture
def chain():
    """X → M → Y"""
    g = CausalDAG()
    g.add_edge("X", "M")
    g.add_edge("M", "Y")
    return g


@pytest.fixture
def fork():
    """X ← M → Y"""
    g = CausalDAG()
    g.add_edge("M", "X")
    g.add_edge("M", "Y")
    return g


@pytest.fixture
def collider():
    """X → Z ← Y (碰撞结构)"""
    g = CausalDAG()
    g.add_edge("X", "Z")
    g.add_edge("Y", "Z")
    return g


@pytest.fixture
def backdoor():
    """Z → X, Z → Y, X → Y (经典后门)"""
    g = CausalDAG()
    g.add_edge("Z", "X")
    g.add_edge("Z", "Y")
    g.add_edge("X", "Y")
    return g


@pytest.fixture
def frontdoor():
    """X → M → Y, X ← U → Y (U 不可观测, 前门准则场景)"""
    g = CausalDAG()
    g.add_edge("X", "M")
    g.add_edge("M", "Y")
    g.add_edge("U", "X")
    g.add_edge("U", "Y")
    return g


class TestDSeparationBasic:
    """d-separation 的三种基本模式 (chain/fork/collider)。"""

    def test_chain_unconditional_dependent(self, chain):
        assert chain.d_separated("X", "Y", set()) is False

    def test_chain_blocked_by_middle(self, chain):
        assert chain.d_separated("X", "Y", {"M"}) is True

    def test_fork_unconditional_dependent(self, fork):
        assert fork.d_separated("X", "Y", set()) is False

    def test_fork_blocked_by_common_cause(self, fork):
        assert fork.d_separated("X", "Y", {"M"}) is True

    def test_collider_unconditional_independent(self, collider):
        """碰撞结构: X⊥Y 无条件独立 (Z 未被观测)。"""
        assert collider.d_separated("X", "Y", set()) is True

    def test_collider_opened_by_conditioning(self, collider):
        """条件化碰撞子 Z 会打开路径, X⊬Y|Z。"""
        assert collider.d_separated("X", "Y", {"Z"}) is False


class TestDSeparationComplex:
    """更复杂图结构的 d-separation。"""

    def test_backdoor_x_not_separated_from_y(self, backdoor):
        """X→Y 直连, 永不 d-separated。"""
        assert backdoor.d_separated("X", "Y", set()) is False
        assert backdoor.d_separated("X", "Y", {"Z"}) is False

    def test_descendants_excludes_self(self, backdoor):
        assert backdoor.descendants("X") == {"Y"}

    def test_ancestors_chain(self, chain):
        assert chain.ancestors("Y") == {"X", "M"}


class TestBackdoorAdjustment:
    """Pearl 后门准则的调整集判定。"""

    def test_valid_adjustment_set_backdoor(self, backdoor):
        """{Z} 是 X→Y 的有效调整集 (阻断后门路径 Z)。"""
        assert backdoor.is_valid_adjustment_set("X", "Y", {"Z"}) is True

    def test_empty_adjustment_invalid_when_confounded(self, backdoor):
        """有混杂时, 空集不是有效调整集。"""
        assert backdoor.is_valid_adjustment_set("X", "Y", set()) is False

    def test_descendant_of_x_invalid_as_adjustment(self, backdoor):
        """X 的后代不能进调整集 (会阻断因果效应或引入偏倚)。"""
        # Y 是 X 的后代, 不应作为调整集
        assert backdoor.is_valid_adjustment_set("X", "Y", {"Y"}) is False

    def test_mediator_invalid_as_adjustment(self):
        """中介变量 M (X→M→Y) 是 X 的后代, 不能进调整集。"""
        g = CausalDAG()
        g.add_edge("X", "M")
        g.add_edge("M", "Y")
        assert g.is_valid_adjustment_set("X", "Y", {"M"}) is False

    def test_minimal_adjustment_set_finds_parent(self, backdoor):
        """最小调整集应找到 X 的父节点 Z。"""
        adj = backdoor.find_minimal_adjustment_set("X", "Y")
        assert adj == {"Z"}

    def test_minimal_adjustment_none_for_mediator_only(self):
        """纯中介链 X→M→Y 无后门路径, 最小调整集应为空集或 None。"""
        g = CausalDAG()
        g.add_edge("X", "M")
        g.add_edge("M", "Y")
        # 无后门路径 (无指向X的箭头开始的路径), 空集应有效
        assert g.is_valid_adjustment_set("X", "Y", set()) is True
