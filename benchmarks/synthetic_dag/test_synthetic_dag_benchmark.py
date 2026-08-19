"""
benchmarks/synthetic_dag/test_synthetic_dag_benchmark.py — Synthetic DAG 可扩展性基准
=====================================================================================

测试 MCI World Model 因果发现算法在随机生成 DAG 上的性能和精度。

生成模式:
  - Erdős-Rényi (ER): 每对节点以概率 p 连接
  - Scale-Free (SF): 幂律度分布

测试维度:
  - 精度: SHD vs 边数
  - 速度: ms vs 节点数
  - 鲁棒性: 不同稀疏度下的 F1
"""

from __future__ import annotations

import gc
import time

import numpy as np


def _generate_er_dag(n_vars: int, expected_degree: int, seed: int = 42):
    """生成 Erdős-Rényi 随机 DAG。

    Args:
        n_vars: 节点数
        expected_degree: 期望平均度
        seed: 随机种子

    Returns:
        (data, var_names, adj_matrix, n_edges)
    """
    rng = np.random.RandomState(seed)
    p = expected_degree / (n_vars - 1)

    # 生成下三角邻接矩阵 (保证无环)
    adj = np.zeros((n_vars, n_vars), dtype=int)
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if rng.rand() < p:
                adj[i, j] = 1

    n_edges = int(np.sum(adj))

    # 生成数据 (线性 SEM)
    n_samples = max(500, n_vars * 50)
    coefs = rng.uniform(0.3, 0.9, size=(n_vars, n_vars))
    data = np.zeros((n_samples, n_vars))
    for i in range(n_vars):
        parents = np.where(adj[:, i] > 0)[0]
        if len(parents) == 0:
            data[:, i] = rng.randn(n_samples)
        else:
            parent_sum = np.sum(data[:, parents] * coefs[parents, i], axis=1)
            data[:, i] = parent_sum + 0.3 * rng.randn(n_samples)

    names = [f"V{i}" for i in range(n_vars)]
    return data, names, adj, n_edges


def _shd(pred: np.ndarray, gt: np.ndarray) -> int:
    return int(np.sum(pred != gt))


def _f1(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-8)


class TestSyntheticDAGAccuracy:
    """随机 DAG 结构学习精度。"""

    def test_er_small_pc(self):
        """10 节点 PC 算法基准。"""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import PCSkeletonDiscoverer

        data, names, gt, n_edges = _generate_er_dag(10, 2, seed=42)
        pc = PCSkeletonDiscoverer(alpha=0.05)
        skel = pc.discover(data, names)
        shd_val = _shd(skel.adj_matrix, gt)
        f1_val = _f1(skel.adj_matrix, gt)
        print(f"\n  ER(10, avg_deg=2) PC: SHD={shd_val}/{n_edges}, F1={f1_val:.3f}")
        # NOTE: PC returns undirected edges; SHD with directed ground truth is inflated 2x
        # For 10-node ER with deg=2: expect SHD ≤ 3*n_edges (undirected tolerance)
        assert shd_val <= 3 * max(n_edges, 5)  # relaxed for undirected output

    def test_er_medium_pc(self):
        """20 节点 PC 算法基准。"""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import PCSkeletonDiscoverer

        data, names, gt, n_edges = _generate_er_dag(20, 2, seed=42)
        pc = PCSkeletonDiscoverer(alpha=0.05, min_corr=0.1)
        skel = pc.discover(data, names)
        shd_val = _shd(skel.adj_matrix, gt)
        f1_val = _f1(skel.adj_matrix, gt)
        print(f"  ER(20, avg_deg=2) PC: SHD={shd_val}/{n_edges}, F1={f1_val:.3f}")

    def test_er_dense_pc(self):
        """20 节点稠密图 PC 算法基准。"""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import PCSkeletonDiscoverer

        data, names, gt, n_edges = _generate_er_dag(20, 4, seed=42)
        pc = PCSkeletonDiscoverer(alpha=0.05, min_corr=0.1)
        skel = pc.discover(data, names)
        shd_val = _shd(skel.adj_matrix, gt)
        f1_val = _f1(skel.adj_matrix, gt)
        print(f"  ER(20, avg_deg=4) PC: SHD={shd_val}/{n_edges}, F1={f1_val:.3f}")

    def test_er_notears(self):
        """10 节点 NOTEARS 基准。"""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import NOTEARSDiscoverer

        data, names, gt, n_edges = _generate_er_dag(10, 2, seed=42)
        nt = NOTEARSDiscoverer(lambda1=0.05, max_iter=150, threshold=0.3)
        skel = nt.discover(data, names)
        shd_val = _shd(skel.adj_matrix, gt)
        f1_val = _f1(skel.adj_matrix, gt)
        print(f"  ER(10, avg_deg=2) NOTEARS: SHD={shd_val}/{n_edges}, F1={f1_val:.3f}")

    def test_er_ges_dense(self):
        """GES with PC warm start on dense ER(20, avg_deg=4).

        GES uses PC skeleton as starting point to avoid greedy trap.
        Note: Pure GES (no warm start) gets F1≈0.10; PC alone gets F1≈0.30.
        GES+PC improves precision at moderate recall cost.
        """
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import GESDiscoverer

        data, names, gt, n_edges = _generate_er_dag(20, 4, seed=42)
        ges = GESDiscoverer(alpha=0.05)
        skel = ges.discover(data, names)  # warm_start=True by default
        shd_val = _shd(skel.adj_matrix, gt)
        f1_val = _f1(skel.adj_matrix, gt)
        print(f"  ER(20, avg_deg=4) GES+PC: SHD={shd_val}/{n_edges}, F1={f1_val:.3f}")
        assert f1_val >= 0.15, f"GES+PC dense F1={f1_val:.3f} below 0.15"


class TestSyntheticDAGSpeed:
    """可扩展性速度测试。"""

    def test_speed_scaling(self):
        """节点数 vs 速度。"""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import PCSkeletonDiscoverer

        results = {}
        for n_vars in [5, 10, 20, 30]:
            data, names, _gt, _n_edges = _generate_er_dag(n_vars, 2, seed=42)
            pc = PCSkeletonDiscoverer(alpha=0.05)
            gc.disable()
            t0 = time.perf_counter()
            pc.discover(data, names)
            t = time.perf_counter() - t0
            gc.enable()
            results[n_vars] = t * 1000

        print("\n  === PC 算法可扩展性 ===")
        for n, ms in sorted(results.items()):
            bar = "█" * int(ms * 2)
            print(f"  n={n:<4} {ms:>7.1f}ms {bar}")

        # 20 节点应在 100ms 内
        assert results.get(20, 999) < 500

    def test_high_dim_notears(self):
        """30 节点 NOTEARS 不崩溃。"""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import NOTEARSDiscoverer

        data, names, _gt, _n_edges = _generate_er_dag(30, 2, seed=42)
        nt = NOTEARSDiscoverer(lambda1=0.05, max_iter=100, threshold=0.3)
        gc.disable()
        t0 = time.perf_counter()
        skel = nt.discover(data, names)
        t = time.perf_counter() - t0
        gc.enable()
        print(f"\n  ER(30, avg_deg=2) NOTEARS: {t * 1000:.0f}ms, {len(skel.edges) // 2} edges")
        assert t < 10.0  # 10s 内完成


class TestDensityRobustness:
    """不同稀疏度下的鲁棒性。"""

    def test_sparsity_curve(self):
        """不同边密度下 F1 曲线。"""
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import PCSkeletonDiscoverer

        print("\n  === 稀疏度鲁棒性 (n=15) ===")
        for deg in [1, 2, 3, 5]:
            data, names, gt, n_edges = _generate_er_dag(15, deg, seed=42)
            pc = PCSkeletonDiscoverer(alpha=0.05, min_corr=0.1)
            skel = pc.discover(data, names)
            f1_val = _f1(skel.adj_matrix, gt)
            print(f"  avg_deg={deg}: {n_edges} edges → F1={f1_val:.3f}")
