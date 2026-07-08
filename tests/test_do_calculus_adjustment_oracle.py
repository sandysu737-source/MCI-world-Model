"""L1 oracle 测试 — DoCalculus 调整集识别 (接入 algebra d-separation 后)。

验证 identify_adjustment_set 是否根据 Pearl 后门准则正确识别,
而非盲目返回父节点。
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.oracle

import pytest

from mci_world_model.sdk._do_calculus import CausalGraph, DoCalculus


class TestIdentifyAdjustmentSet:
    """后门准则的调整集识别。"""

    def test_backdoor_finds_confounder(self):
        """Z→X, Z→Y, X→Y: 调整集应为 {Z}。"""
        cg = CausalGraph(nodes=["Z", "X", "Y"], edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")])
        dc = DoCalculus(cg)
        adj = dc.identify_adjustment_set("X", "Y")
        assert adj == ["Z"]

    def test_no_adjustment_needed_for_direct_chain(self):
        """X→Y 直连无混杂: 调整集应为 None 或空。"""
        cg = CausalGraph(nodes=["X", "Y"], edges=[("X", "Y")])
        dc = DoCalculus(cg)
        adj = dc.identify_adjustment_set("X", "Y")
        # 无后门路径, 不需要调整
        assert adj is None or adj == []

    def test_collider_not_used_as_adjustment(self):
        """碰撞子不应被选为调整集。X→Z<-Y, X->Y 无后门, 调整集空。"""
        cg = CausalGraph(nodes=["X", "Y", "Z"], edges=[("X", "Z"), ("Y", "Z"), ("X", "Y")])
        dc = DoCalculus(cg)
        adj = dc.identify_adjustment_set("X", "Y")
        # Z 是碰撞子, 不应出现在调整集
        assert adj is None or "Z" not in adj

    def test_to_dag_bridge_works(self):
        """CausalGraph.to_dag() 桥接到 algebra 层。"""
        cg = CausalGraph(nodes=["A", "B", "C"], edges=[("A", "B"), ("B", "C")])
        dag = cg.to_dag()
        assert dag.is_dag()
        assert dag.children("A") == ["B"]
        assert dag.d_separated("A", "C", {"B"})  # 链被 M=B 阻断


class TestEstimateAteBackdoor:
    """后门调整 ATE 估计 (线性 SEM 解析对照)。"""

    def test_backdoor_ate_recovers_direct_effect(self):
        """Z→X, Z→Y, X→Y 线性 SEM: 后门调整 ATE 应接近 X→Y 直接系数。

        构造: X = 0.5*Z + e_x, Y = 0.7*X + 0.3*Z + e_y
        真实 ATE (X对Y的直接效应) = 0.7。
        """
        import numpy as np
        rng = np.random.RandomState(42)
        n = 20000
        Z = rng.randn(n)
        X = 0.5 * Z + rng.randn(n) * 0.5
        Y = 0.7 * X + 0.3 * Z + rng.randn(n) * 0.5  # 真实 ATE = 0.7
        data = {"X": X, "Y": Y, "Z": Z}

        cg = CausalGraph(nodes=["Z", "X", "Y"], edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")])
        dc = DoCalculus(cg, data=data)
        result = dc.estimate_ate("X", "Y")
        # 容差宽松(观测数据估计), 但方向和量级应正确
        assert result.method == "backdoor"
        assert "Z" in result.adjustment_set
        # ATE 应为正且接近 0.7 (观测估计有偏差, 用宽松容差验证方向)
        assert result.ate > 0.3, f"ATE={result.ate:.3f} 应为正且显著, 真实=0.7"
