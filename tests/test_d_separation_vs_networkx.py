"""L2 参考库对照测试 — algebra 层 d-separation vs networkx 权威实现。

用 networkx 3.6 的 is_d_separator 作为 oracle, 验证本项目 algebra.CausalDAG
的 d_separated 在多种图结构下给出一致结论。这是发现系统性偏差的关键层。
"""

from __future__ import annotations

import itertools

import networkx as nx
import pytest

pytestmark = [pytest.mark.oracle, pytest.mark.reference]

from mci_world_model.algebra.causal_graph import CausalDAG


def _to_networkx(dag: CausalDAG) -> nx.DiGraph:
    """把 CausalDAG 转成 networkx DiGraph。"""
    g = nx.DiGraph()
    g.add_nodes_from(dag.nodes)
    for p in dag.edges:
        for c, _, _ in dag.edges[p]:
            g.add_edge(p, c)
    return g


def _assert_dsep_consistent(dag: CausalDAG, pairs, all_nodes):
    """对给定图, 枚举所有 (节点对, 条件集) 组合, 对照 networkx。

    条件集 Z 必须排除 X 和 Y 自身 (networkx 要求三者不相交)。
    """
    g = _to_networkx(dag)
    mismatches = []
    checked = 0
    for x, y in pairs:
        # 条件集候选: 排除 x, y 的其余节点
        candidates = [n for n in all_nodes if n not in (x, y)]
        for r in range(len(candidates) + 1):
            for z_tuple in itertools.combinations(candidates, r):
                z = set(z_tuple)
                ours = dag.d_separated(x, y, z)
                theirs = nx.is_d_separator(g, x, y, z)
                checked += 1
                if ours != theirs:
                    mismatches.append((x, y, z, ours, theirs))
    assert not mismatches, f"d-separation 与 networkx 不一致 ({len(mismatches)}/{checked} 处):\n" + "\n".join(
        f"  {x}⊥{y}|{z}: ours={o}, nx={t}" for x, y, z, o, t in mismatches[:5]
    )


# 各种经典图结构
@pytest.fixture
def collider():
    g = CausalDAG()
    g.add_edge("X", "Z")
    g.add_edge("Y", "Z")
    return g


@pytest.fixture
def chain():
    g = CausalDAG()
    g.add_edge("X", "M")
    g.add_edge("M", "Y")
    return g


@pytest.fixture
def fork():
    g = CausalDAG()
    g.add_edge("M", "X")
    g.add_edge("M", "Y")
    return g


@pytest.fixture
def backdoor():
    g = CausalDAG()
    g.add_edge("Z", "X")
    g.add_edge("Z", "Y")
    g.add_edge("X", "Y")
    return g


@pytest.fixture
def diamond():
    """A→B, A→C, B→D, C→D (diamond, D 是 collider)"""
    g = CausalDAG()
    for p, c in [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]:
        g.add_edge(p, c)
    return g


@pytest.fixture
def extended_chain():
    """A→B→C→D→E (4 跳链)"""
    g = CausalDAG()
    for p, c in [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]:
        g.add_edge(p, c)
    return g


@pytest.fixture
def confounded_chain():
    """U→X, U→Y, X→Z, Y→Z (两个混淆源汇聚于 collider Z)"""
    g = CausalDAG()
    for p, c in [("U", "X"), ("U", "Y"), ("X", "Z"), ("Y", "Z")]:
        g.add_edge(p, c)
    return g


class TestDSeparationVsNetworkx:
    """每种图结构上, 枚举条件集子集, 与 networkx 逐对照。"""

    def test_collider_all_subsets(self, collider):
        _assert_dsep_consistent(collider, [("X", "Y"), ("X", "Z"), ("Y", "Z")], ["X", "Y", "Z"])

    def test_chain_all_subsets(self, chain):
        _assert_dsep_consistent(chain, [("X", "Y"), ("X", "M"), ("M", "Y")], ["X", "M", "Y"])

    def test_fork_all_subsets(self, fork):
        _assert_dsep_consistent(fork, [("X", "Y"), ("M", "X"), ("M", "Y")], ["M", "X", "Y"])

    def test_backdoor_all_subsets(self, backdoor):
        _assert_dsep_consistent(backdoor, [("X", "Y"), ("X", "Z"), ("Y", "Z")], ["Z", "X", "Y"])

    def test_diamond_all_subsets(self, diamond):
        _assert_dsep_consistent(
            diamond, [("A", "D"), ("B", "C"), ("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")], ["A", "B", "C", "D"]
        )

    def test_extended_chain_all_subsets(self, extended_chain):
        _assert_dsep_consistent(
            extended_chain, [("A", "E"), ("A", "C"), ("B", "D"), ("A", "D"), ("C", "E")], ["A", "B", "C", "D", "E"]
        )

    def test_confounded_chain_all_subsets(self, confounded_chain):
        _assert_dsep_consistent(
            confounded_chain, [("X", "Y"), ("U", "Z"), ("X", "Z"), ("Y", "Z")], ["U", "X", "Y", "Z"]
        )
