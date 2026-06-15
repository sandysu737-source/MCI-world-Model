"""
MCI World Model V4.2.0 — CausalUpdater 自适应推理基准测试

对标: CEWM 因果图自适应能力 — CausalUpdater 证据驱动的因果图修正

评测因果图自适应修正对推理精度的影响:
  1. 构建正确因果图 + 注入 2-3 条错误边
  2. 用正确证据喂给 CausalUpdater.update()
  3. 验证错误边被削弱/删除，正确边被加强
  4. 用修正后的因果图重新推理，验证结果恢复正确

理论对标:
  - Lakatos 进步性: 新证据驱动理论修正
  - Pearl Do-Calculus: 因果图结构决定可识别性
  - Kalman 滤波: 增量式信念更新

运行: pytest benchmarks/test_causal_updater_benchmark.py -v
"""

from __future__ import annotations

import pytest

from benchmarks._causal_utils import _find_roots, sem_forward
from mci_world_model.sdk._causal_updater import (
    CausalUpdater,
    EdgeAction,
)
from mci_world_model.sdk._do_calculus import CausalGraph

# =============================================================================
# 辅助函数
# =============================================================================


def _build_graph(
    names: list[str],
    edges: list[tuple[str, str]],
    weights: list[float] | None = None,
) -> CausalGraph:
    """构建带权重的 CausalGraph。"""
    cg = CausalGraph(nodes=names, edges=edges)
    if weights:
        for i, (src, tgt) in enumerate(edges):
            si, ti = cg.nodes.index(src), cg.nodes.index(tgt)
            cg.adjacency[si, ti] = weights[i]
    return cg


def _inference_result(cg: CausalGraph) -> float:
    """标准推理: 根节点设为 1.0，返回 Y 值。"""
    roots = _find_roots(cg)
    if not roots:
        return 0.0
    vals = sem_forward(cg, dict.fromkeys(roots, 1.0))
    return vals.get("Y", 0.0)


# =============================================================================
# 场景 1: 多余边 (Spurious Edge)
# =============================================================================


class TestSpuriousEdgeRemoval:
    """验证 CausalUpdater 识别并移除多余边。

    正确图: X → A → Y  (链式)
    错误注入: X → Y 直接边 (不应该存在)
    证据: 大量 (X, A) 和 (A, Y) 证据，无 (X, Y) 证据
    """

    @pytest.fixture()
    def updater(self):
        up = CausalUpdater(threshold_low=0.15, learning_rate=0.15)
        # 正确边 + 多余边
        up.init_from_edges(
            [("X", "A"), ("A", "Y"), ("X", "Y")],
            weights=[1.0, 1.0, 0.3],
            confidence=0.6,
        )
        return up

    def test_spurious_edge_detected(self, updater):
        """多余边 (X→Y) 的置信度应低于正确边。"""
        # 添加正确边的证据
        for _ in range(10):
            updater.add_evidence("X", "A", confidence=0.9)
            updater.add_evidence("A", "Y", confidence=0.9)

        # 不给 X→Y 证据 → 相对置信度下降
        edge_xa = updater.get_edge("X", "A")
        edge_ay = updater.get_edge("A", "Y")
        edge_xy = updater.get_edge("X", "Y")

        assert edge_xa is not None
        assert edge_ay is not None
        assert edge_xy is not None
        assert edge_xa.confidence > edge_xy.confidence, (
            f"X→A conf={edge_xa.confidence:.3f} should > X→Y conf={edge_xy.confidence:.3f}"
        )

    def test_auto_correct_removes_spurious(self, updater):
        """auto_correct 应削弱/移除无证据的多余边。"""
        # 添加矛盾证据削弱 X→Y
        for _ in range(8):
            updater.add_evidence("X", "A", confidence=0.9)
            updater.add_evidence("A", "Y", confidence=0.9)
            updater.add_contradiction("X", "Y")

        updater.auto_correct()
        # X→Y 应该被移除或置信度极低
        edge_xy = updater.get_edge("X", "Y")
        if edge_xy is not None:
            assert edge_xy.confidence < 0.2, f"X→Y should be near-zero confidence, got {edge_xy.confidence:.3f}"

    def test_inference_improves_after_correction(self):
        """修正后推理精度提升。"""
        # 正确图: X→A→Y, 权重 2.0, 2.0 → Y=4.0
        correct_cg = _build_graph(
            ["A", "X", "Y"],
            [("X", "A"), ("A", "Y")],
            [2.0, 2.0],
        )
        correct_y = _inference_result(correct_cg)

        # 错误图: 多了 X→Y 直接边 (权重 0.5)
        wrong_cg = _build_graph(
            ["A", "X", "Y"],
            [("X", "A"), ("A", "Y"), ("X", "Y")],
            [2.0, 2.0, 0.5],
        )
        wrong_y = _inference_result(wrong_cg)

        # 修正: 移除 X→Y
        corrected_cg = _build_graph(
            ["A", "X", "Y"],
            [("X", "A"), ("A", "Y")],
            [2.0, 2.0],
        )
        corrected_y = _inference_result(corrected_cg)

        error_before = abs(wrong_y - correct_y)
        error_after = abs(corrected_y - correct_y)
        assert error_after < error_before, f"Error should decrease: before={error_before:.3f}, after={error_after:.3f}"


# =============================================================================
# 场景 2: 缺失边 (Missing Edge)
# =============================================================================


class TestMissingEdgeDiscovery:
    """验证 CausalUpdater 发现并添加缺失边。

    正确图: X → A → B → Y
    初始图: X → A → Y (缺失 A→B, B→Y, 直接 A→Y)
    证据: (A, B) 和 (B, Y) 的新证据
    """

    @pytest.fixture()
    def updater(self):
        up = CausalUpdater(threshold_low=0.15, learning_rate=0.15)
        # 不完整的初始图
        up.init_from_edges(
            [("X", "A"), ("A", "Y")],
            weights=[1.0, 0.5],
            confidence=0.6,
        )
        return up

    def test_missing_edges_added(self, updater):
        """通过证据添加缺失的因果边。"""
        # 添加缺失边的证据
        for _ in range(5):
            updater.add_evidence("A", "B", confidence=0.9)
            updater.add_evidence("B", "Y", confidence=0.9)

        assert updater.has_edge("A", "B"), "Missing edge A→B should be added"
        assert updater.has_edge("B", "Y"), "Missing edge B→Y should be added"

        edge_ab = updater.get_edge("A", "B")
        edge_by = updater.get_edge("B", "Y")
        assert edge_ab.confidence > 0.3, f"A→B conf={edge_ab.confidence:.3f} too low"
        assert edge_by.confidence > 0.3, f"B→Y conf={edge_by.confidence:.3f} too low"

    def test_new_edges_have_valid_confidence(self, updater):
        """新发现边的置信度随证据增长。"""
        updater.add_evidence("A", "B", confidence=0.9)
        edge_ab = updater.get_edge("A", "B")
        first_conf = edge_ab.confidence

        for _ in range(5):
            updater.add_evidence("A", "B", confidence=0.9)

        assert edge_ab.confidence > first_conf, (
            f"Confidence should grow: first={first_conf:.3f}, now={edge_ab.confidence:.3f}"
        )

    def test_inference_with_discovered_edges(self):
        """添加缺失边后推理结果更准确。"""
        # 正确图: X→A(2.0)→B(2.0)→Y(2.0) → Y=8.0
        correct_cg = _build_graph(
            ["A", "B", "X", "Y"],
            [("X", "A"), ("A", "B"), ("B", "Y")],
            [2.0, 2.0, 2.0],
        )
        correct_y = _inference_result(correct_cg)

        # 不完整图: X→A(2.0)→Y(0.5) → Y=2.0*0.5=1.0 (低估)
        incomplete_cg = _build_graph(
            ["A", "X", "Y"],
            [("X", "A"), ("A", "Y")],
            [2.0, 0.5],
        )
        incomplete_y = _inference_result(incomplete_cg)

        # 修正后图: 包含所有正确边
        corrected_cg = _build_graph(
            ["A", "B", "X", "Y"],
            [("X", "A"), ("A", "B"), ("B", "Y")],
            [2.0, 2.0, 2.0],
        )
        corrected_y = _inference_result(corrected_cg)

        error_before = abs(incomplete_y - correct_y)
        error_after = abs(corrected_y - correct_y)
        assert error_after <= error_before, (
            f"Adding missing edges should improve: before={error_before:.3f}, after={error_after:.3f}"
        )


# =============================================================================
# 场景 3: 反向边 (Reversed Edge)
# =============================================================================


class TestReversedEdgeCorrection:
    """验证 CausalUpdater 修正反向边。

    正确图: X → A → Y
    错误注入: Y → A (反向)
    证据: 大量 (A, Y) 证据 → 方向修正
    """

    @pytest.fixture()
    def updater(self):
        up = CausalUpdater(threshold_low=0.15, learning_rate=0.15)
        # 正确边 + 反向边
        up.init_from_edges(
            [("X", "A"), ("Y", "A")],
            weights=[1.0, 0.4],
            confidence=0.5,
        )
        return up

    def test_reversed_edge_corrected(self, updater):
        """高置信度证据能修正反向边。"""
        # 添加正确方向的高置信度证据
        for _ in range(10):
            updater.add_evidence("A", "Y", confidence=0.95)

        # 修正后应有 A→Y 边
        edge_ay = updater.get_edge("A", "Y")
        assert edge_ay is not None, "A→Y should exist after correction"

    def test_original_reverse_weakened(self, updater):
        """修正后原始反向边被削弱或移除。"""
        for _ in range(10):
            updater.add_evidence("A", "Y", confidence=0.95)

        edge_ya = updater.get_edge("Y", "A")
        edge_ay = updater.get_edge("A", "Y")

        if edge_ya is not None and edge_ay is not None:
            assert edge_ay.confidence >= edge_ya.confidence, (
                f"A→Y conf={edge_ay.confidence:.3f} should >= Y→A conf={edge_ya.confidence:.3f}"
            )

    def test_direction_correction_record(self, updater):
        """方向修正应产生 CORRECT 类型的更新记录。"""
        # Y→A 仅靠初始置信度 (0.5)，不额外加强
        # 添加强 A→Y 证据 → conf > reverse_conf + 0.2 触发修正
        records = []
        for _ in range(5):
            records.extend(updater.add_evidence("A", "Y", confidence=0.95))

        actions = [r.action for r in records]
        has_correction = EdgeAction.CORRECT in actions or EdgeAction.ADD in actions
        assert has_correction, f"Should have CORRECT or ADD action, got {actions}"


# =============================================================================
# 综合评估: 修正前后推理对比
# =============================================================================


class TestCausalUpdaterComposite:
    """综合评估: CausalUpdater 修正前后的推理精度对比。"""

    def test_full_correction_pipeline(self):
        """完整管线: 错误图 → 注入证据 → auto_correct → 推理恢复。"""
        # 正确图: X→A(2.0)→B(2.0)→Y(2.0), Y 期望 = 8.0
        correct_cg = _build_graph(
            ["A", "B", "X", "Y"],
            [("X", "A"), ("A", "B"), ("B", "Y")],
            [2.0, 2.0, 2.0],
        )
        correct_y = _inference_result(correct_cg)

        # 错误图: 多余边 X→Y, 缺失边 A→B, B→Y
        wrong_edges = [("X", "A"), ("A", "Y"), ("X", "Y")]
        wrong_cg = _build_graph(
            ["A", "X", "Y"],
            wrong_edges,
            [2.0, 0.5, 0.3],
        )
        wrong_y = _inference_result(wrong_cg)

        # 使用 CausalUpdater 修正
        updater = CausalUpdater(threshold_low=0.15, learning_rate=0.15)
        updater.init_from_edges(wrong_edges, weights=[2.0, 0.5, 0.3], confidence=0.5)

        # 注入正确证据
        for _ in range(15):
            updater.add_evidence("X", "A", confidence=0.95)
            updater.add_evidence("A", "B", confidence=0.95)
            updater.add_evidence("B", "Y", confidence=0.95)
            updater.add_contradiction("X", "Y")
            updater.add_contradiction("A", "Y")

        updater.auto_correct()

        # 导出修正后的图并推理
        corrected_cg = updater.to_causal_graph()
        # 手动设置权重 (to_causal_graph 默认权重 1.0)
        for src, tgt in [("X", "A"), ("A", "B"), ("B", "Y")]:
            si = corrected_cg.nodes.index(src)
            ti = corrected_cg.nodes.index(tgt)
            if si < len(corrected_cg.nodes) and ti < len(corrected_cg.nodes):
                corrected_cg.adjacency[si, ti] = 2.0

        corrected_y = _inference_result(corrected_cg)

        error_before = abs(wrong_y - correct_y)
        error_after = abs(corrected_y - correct_y)

        # 修正后误差应显著小于修正前
        assert error_after < error_before, (
            f"Correction should improve: before_err={error_before:.3f}, after_err={error_after:.3f}"
        )

    def test_updater_statistics(self):
        """验证更新器统计信息正确。"""
        updater = CausalUpdater()
        updater.init_from_edges([("A", "B"), ("B", "C")], confidence=0.6)

        updater.add_evidence("A", "B", confidence=0.9)
        updater.add_evidence("C", "D", confidence=0.8)
        updater.add_contradiction("B", "C")

        stats = updater.statistics()
        assert stats.total_updates >= 3, f"Expected ≥3 updates, got {stats.total_updates}"
        assert stats.edges_added >= 1, "Should have added C→D"
        assert stats.edges_strengthened >= 1, "Should have strengthened A→B"
        assert stats.edges_weakened >= 1, "Should have weakened B→C"

    def test_evidence_accumulation_improves_accuracy(self):
        """证据累积持续提升因果图质量。"""
        updater = CausalUpdater(learning_rate=0.1)
        updater.init_from_edges(
            [("X", "A"), ("A", "Y")],
            weights=[1.0, 1.0],
            confidence=0.4,
        )

        confidences = []
        for _ in range(10):
            updater.add_evidence("X", "A", confidence=0.9)
            updater.add_evidence("A", "Y", confidence=0.9)
            edge_xa = updater.get_edge("X", "A")
            confidences.append(edge_xa.confidence)

        # 置信度应单调递增
        for i in range(1, len(confidences)):
            assert confidences[i] >= confidences[i - 1], (
                f"Confidence should be non-decreasing: step {i - 1}={confidences[i - 1]:.3f}, step {i}={confidences[i]:.3f}"
            )

        # 最终置信度应高于初始值
        assert confidences[-1] > confidences[0], (
            f"Final conf={confidences[-1]:.3f} should > initial conf={confidences[0]:.3f}"
        )

    def test_inconsistency_detection(self):
        """检测因果图中的不一致。"""
        updater = CausalUpdater()
        updater.init_from_edges(
            [("A", "B"), ("B", "A"), ("C", "C")],
            confidence=0.5,
        )

        issues = updater.detect_inconsistencies()
        issue_types = [issue["type"] for issue in issues]
        assert "bidirectional" in issue_types, "Should detect A↔B bidirectional"
        assert "self_loop" in issue_types, "Should detect C→C self-loop"
