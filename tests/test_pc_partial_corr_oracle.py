"""L1 解析 oracle 测试 — PC 算法高阶偏相关正确性。

验证 PCSkeletonDiscoverer._partial_corr 在 2 阶及以上条件集下是否给出
正确的偏相关系数。当前 bug: len(cond)>1 时只处理 cond[0] 就返回,
导致高阶条件独立性检验全部退化为 1 阶。

数学 oracle: 高阶偏相关 r_{ij|S} = -P^{-1}[i,j] / sqrt(P^{-1}[i,i]*P^{-1}[j,j])
其中 P = corr[{i,j}∪S, {i,j}∪S] 的逆 (precision matrix 的归一化元素)。
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.oracle

import numpy as np
import pytest

from mci_world_model.sdk._autonomous_law_discoverer_v2 import PCSkeletonDiscoverer


def _correct_partial_corr(corr: np.ndarray, i: int, j: int, cond: list[int]) -> float:
    """高阶偏相关的正确解析实现 (precision matrix)。"""
    idx = [i, j] + list(cond)
    sub = corr[np.ix_(idx, idx)]
    P_inv = np.linalg.inv(sub)
    return -P_inv[0, 1] / np.sqrt(P_inv[0, 0] * P_inv[1, 1])


@pytest.fixture
def diamond_data():
    """Diamond 结构: Z1,Z2 都是 X,Y 的共同原因。

    X⊥Y | {Z1,Z2} (条件化两个父节点后独立)
    X⊬Y | {Z1} 或 {Z2} (只条件化一个仍相关)
    这是需要 2 阶偏相关才能正确识别的场景。
    """
    rng = np.random.RandomState(42)
    n = 5000
    Z1 = rng.randn(n)
    Z2 = rng.randn(n)
    X = 0.7 * Z1 + 0.7 * Z2 + 0.3 * rng.randn(n)
    Y = 0.7 * Z1 + 0.7 * Z2 + 0.3 * rng.randn(n)
    return np.column_stack([X, Y, Z1, Z2])  # X=0, Y=1, Z1=2, Z2=3


class TestPartialCorrHighOrder:
    """L1: 高阶偏相关必须用 precision matrix, 不能退化为 1 阶。"""

    def test_second_order_matches_precision_matrix(self, diamond_data):
        """2 阶偏相关 r_{XY|Z1,Z2} 必须接近 0(条件独立)。"""
        corr = np.corrcoef(diamond_data.T)
        # 当前实现 (bug): 只处理 cond[0]
        r_impl = PCSkeletonDiscoverer._partial_corr(corr, 0, 1, [2, 3])
        r_correct = _correct_partial_corr(corr, 0, 1, [2, 3])
        assert abs(r_impl - r_correct) < 0.05, (
            f"2阶偏相关错误: 实现={r_impl:.4f}, 正确={r_correct:.4f}。"
            f"差值{abs(r_impl-r_correct):.4f}表明实现退化成了1阶"
        )

    def test_second_order_detects_conditional_independence(self, diamond_data):
        """在 diamond 结构上, r_{XY|Z1,Z2} 应识别出条件独立(绝对值小)。"""
        corr = np.corrcoef(diamond_data.T)
        r = PCSkeletonDiscoverer._partial_corr(corr, 0, 1, [2, 3])
        assert abs(r) < 0.1, (
            f"X⊥Y|{{Z1,Z2}} 但 r={r:.4f}, 未识别出条件独立"
        )

    def test_first_order_still_correlated(self, diamond_data):
        """对照: 1 阶 r_{XY|Z1} 应仍显相关(绝对值大)。"""
        corr = np.corrcoef(diamond_data.T)
        r1 = PCSkeletonDiscoverer._partial_corr(corr, 0, 1, [2])
        assert abs(r1) > 0.5, f"X⊬Y|Z1 但 r={r1:.4f}, 误判为独立"

    def test_third_order_matches_precision_matrix(self, diamond_data):
        """3 阶偏相关也必须正确(precision matrix)。"""
        rng = np.random.RandomState(7)
        n = 5000
        # 5 节点: X,Y 共享 Z1,Z2,Z3 三个父
        Z1, Z2, Z3 = rng.randn(3, n)
        X = 0.5 * Z1 + 0.5 * Z2 + 0.5 * Z3 + 0.3 * rng.randn(n)
        Y = 0.5 * Z1 + 0.5 * Z2 + 0.5 * Z3 + 0.3 * rng.randn(n)
        data = np.column_stack([X, Y, Z1, Z2, Z3])
        corr = np.corrcoef(data.T)
        r_impl = PCSkeletonDiscoverer._partial_corr(corr, 0, 1, [2, 3, 4])
        r_correct = _correct_partial_corr(corr, 0, 1, [2, 3, 4])
        assert abs(r_impl - r_correct) < 0.05, (
            f"3阶偏相关错误: 实现={r_impl:.4f}, 正确={r_correct:.4f}"
        )

    def test_zero_and_first_order_unchanged(self, diamond_data):
        """修复不应破坏 0 阶和 1 阶(它们原本正确)。"""
        corr = np.corrcoef(diamond_data.T)
        r0 = PCSkeletonDiscoverer._partial_corr(corr, 0, 1, [])
        assert abs(r0 - corr[0, 1]) < 1e-12
        r1_impl = PCSkeletonDiscoverer._partial_corr(corr, 0, 1, [2])
        r1_correct = _correct_partial_corr(corr, 0, 1, [2])
        assert abs(r1_impl - r1_correct) < 1e-9
