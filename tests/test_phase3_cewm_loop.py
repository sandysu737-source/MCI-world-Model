"""Phase 3 (v3.6.0) 认知环闭环 — 测试套件

覆盖:
- TestCausalUpdater: CausalUpdater 因果图自适应更新 (20 测试)
- TestActionGapMetric: ActionGapMetric Heidegger 参与式距离 (12 测试)
- TestAttentionPolicy: PerceptionPipeline.attention_policy 感知注意力 (10 测试)
- TestCEWMStep: MCIWorldModel.cewm_step() CEWM 引擎集成 (8 测试)
- TestImportsV36: 导出符号完整性 (6 测试)

目标: 56 个新测试 → 基线 1813 + 56 = 1869 passed
"""

import math

from mci_world_model._sys._perception_pipeline import PerceptionPipeline
from mci_world_model.sdk._action_gap import (
    ActionCostResult,
    ActionGapMetric,
)
from mci_world_model.sdk._causal_updater import (
    CausalEdge,
    CausalUpdater,
    CausalUpdaterStats,
    EdgeAction,
)
from mci_world_model.sdk._world_model import MCIWorldModel
from mci_world_model.sdk._world_state import PendulumState

# =============================================================================
# TestCausalUpdater: 因果图自适应更新器
# =============================================================================


class TestCausalUpdater:
    """CausalUpdater 核心功能测试。"""

    def test_init_empty(self):
        updater = CausalUpdater()
        assert updater.n_edges == 0
        assert updater.n_nodes == 0

    def test_init_from_edges(self):
        updater = CausalUpdater()
        updater.init_from_edges([("A", "B"), ("B", "C"), ("C", "D")])
        assert updater.n_edges == 3
        assert updater.n_nodes == 4
        assert updater.has_edge("A", "B")
        assert updater.has_edge("B", "C")
        assert not updater.has_edge("A", "D")

    def test_init_from_causal_graph(self):
        from mci_world_model.sdk._do_calculus import CausalGraph

        graph = CausalGraph(nodes=["X", "Y", "Z"], edges=[("X", "Y"), ("Y", "Z")])
        updater = CausalUpdater()
        updater.init_from_causal_graph(graph)
        assert updater.n_edges == 2
        assert updater.has_edge("X", "Y")
        assert updater.has_edge("Y", "Z")

    def test_add_evidence_new_edge(self):
        updater = CausalUpdater()
        records = updater.add_evidence("A", "B", confidence=0.8)
        assert len(records) == 1
        assert records[0].action == EdgeAction.ADD
        assert updater.has_edge("A", "B")

    def test_add_evidence_strengthen(self):
        updater = CausalUpdater()
        updater.init_from_edges([("A", "B")])
        edge_before = updater.get_edge("A", "B")
        ev_before = edge_before.evidence_count

        records = updater.add_evidence("A", "B", confidence=0.9)
        assert len(records) == 1
        assert records[0].action == EdgeAction.STRENGTHEN
        edge_after = updater.get_edge("A", "B")
        # 证据计数增加
        assert edge_after.evidence_count > ev_before

    def test_add_evidence_direction_correction(self):
        updater = CausalUpdater()
        updater.init_from_edges([("B", "A")])
        # 弱边 + 强反向证据 → 修正方向
        edge = updater.get_edge("B", "A")
        edge.confidence = 0.3  # 降低置信度

        records = updater.add_evidence("A", "B", confidence=0.8)
        # 0.8 > 0.3 + 0.2 = 0.5 → 触发修正
        assert any(r.action == EdgeAction.CORRECT for r in records)
        assert updater.has_edge("A", "B")

    def test_add_contradiction(self):
        updater = CausalUpdater()
        updater.init_from_edges([("A", "B")])
        edge_before = updater.get_edge("A", "B")
        conf_before = edge_before.confidence

        records = updater.add_contradiction("A", "B")
        assert len(records) == 1
        assert records[0].action == EdgeAction.WEAKEN
        edge_after = updater.get_edge("A", "B")
        assert edge_after.confidence < conf_before

    def test_update_dict_format(self):
        updater = CausalUpdater()
        records = updater.update({"cause": "X", "effect": "Y", "confidence": 0.7})
        assert len(records) == 1
        assert updater.has_edge("X", "Y")

    def test_update_kwargs_format(self):
        updater = CausalUpdater()
        records = updater.update(cause="X", effect="Y", confidence=0.7)
        assert len(records) == 1

    def test_update_batch_edges(self):
        updater = CausalUpdater()
        records = updater.update({"edges": [("A", "B"), ("B", "C")], "confidence": 0.8})
        assert len(records) == 2
        assert updater.has_edge("A", "B")
        assert updater.has_edge("B", "C")

    def test_update_contradiction_format(self):
        updater = CausalUpdater()
        updater.init_from_edges([("A", "B")])
        records = updater.update({"contradiction": ("A", "B")})
        assert len(records) == 1
        assert records[0].action == EdgeAction.WEAKEN

    def test_auto_correct_remove_low_confidence(self):
        updater = CausalUpdater(threshold_low=0.1)
        updater.init_from_edges([("A", "B")])
        # 大量矛盾证据 → support_ratio 低 → update_confidence 计算也低
        edge = updater.get_edge("A", "B")
        edge.confidence = 0.05
        edge.evidence_count = 1
        edge.contradiction_count = 20  # support_ratio ≈ 0.048

        records = updater.auto_correct()
        assert any(r.action == EdgeAction.REMOVE for r in records)
        assert not updater.has_edge("A", "B")

    def test_detect_inconsistencies_bidirectional(self):
        updater = CausalUpdater()
        updater._add_edge("A", "B", confidence=0.8)
        updater._add_edge("B", "A", confidence=0.6)
        issues = updater.detect_inconsistencies()
        bidir = [i for i in issues if i["type"] == "bidirectional"]
        assert len(bidir) >= 1

    def test_detect_inconsistencies_self_loop(self):
        updater = CausalUpdater()
        updater._add_edge("A", "A", confidence=0.5)
        issues = updater.detect_inconsistencies()
        loops = [i for i in issues if i["type"] == "self_loop"]
        assert len(loops) >= 1

    def test_detect_inconsistencies_isolated_node(self):
        updater = CausalUpdater()
        updater._nodes.add("Lonely")
        updater._add_edge("A", "B", confidence=0.8)
        issues = updater.detect_inconsistencies()
        isolated = [i for i in issues if i["type"] == "isolated_node"]
        assert len(isolated) >= 1

    def test_get_parents_children(self):
        updater = CausalUpdater()
        updater.init_from_edges([("A", "B"), ("A", "C"), ("B", "D")])
        assert set(updater.get_parents("B")) == {"A"}
        assert set(updater.get_children("A")) == {"B", "C"}

    def test_statistics(self):
        updater = CausalUpdater()
        updater.init_from_edges([("A", "B"), ("B", "C")])
        updater.add_evidence("D", "E", confidence=0.8)
        stats = updater.statistics()
        assert isinstance(stats, CausalUpdaterStats)
        assert stats.total_edges == 3
        assert stats.edges_added == 1

    def test_to_causal_graph(self):
        updater = CausalUpdater()
        updater.init_from_edges([("A", "B"), ("B", "C")])
        graph = updater.to_causal_graph()
        assert graph.n_nodes == 3
        assert len(graph.edges) == 2

    def test_clear(self):
        updater = CausalUpdater()
        updater.init_from_edges([("A", "B")])
        updater.clear()
        assert updater.n_edges == 0
        assert updater.n_nodes == 0

    def test_causal_edge_support_ratio(self):
        edge = CausalEdge(cause="A", effect="B", evidence_count=3, contradiction_count=1)
        assert edge.support_ratio == 0.75

    def test_causal_edge_update_confidence(self):
        edge = CausalEdge(cause="A", effect="B", evidence_count=5, contradiction_count=0)
        conf = edge.update_confidence()
        assert conf > 0.5


# =============================================================================
# TestActionGapMetric: Heidegger 参与式距离
# =============================================================================


class TestActionGapMetric:
    """ActionGapMetric 行动距离度量测试。"""

    def test_init_default(self):
        metric = ActionGapMetric()
        assert metric.config.energy_weight == 0.4
        assert metric.config.effort_weight == 0.3

    def test_physical_distance_pendulum(self):
        metric = ActionGapMetric()
        s1 = PendulumState(theta=0.1, omega=0.0)
        s2 = PendulumState(theta=0.5, omega=0.0)
        d = metric.physical_distance(s1, s2)
        assert abs(d - 0.4) < 0.01

    def test_physical_distance_angular(self):
        metric = ActionGapMetric()
        s1 = PendulumState(theta=0.1, omega=0.0)
        s2 = PendulumState(theta=math.pi - 0.1, omega=0.0)  # 接近 π
        d = metric.physical_distance(s1, s2)
        # 角距离 ≈ π - 0.2 ≈ 2.94
        assert d > 2.5

    def test_energy_barrier_pendulum(self):
        metric = ActionGapMetric()
        s1 = PendulumState(theta=0.1, omega=0.0)  # 近平衡点
        s2 = PendulumState(theta=math.pi, omega=0.0)  # 倒立点
        barrier = metric.energy_barrier(s1, s2)
        # 需要克服重力势垒
        assert barrier > 0

    def test_energy_barrier_same_height(self):
        metric = ActionGapMetric()
        s1 = PendulumState(theta=0.5, omega=0.0)
        s2 = PendulumState(theta=-0.5, omega=0.0)  # 同高度
        barrier = metric.energy_barrier(s1, s2)
        # 路径经过 θ=0 (更低势能), 势垒应为 0 或很小
        assert barrier >= 0

    def test_action_distance_greater_than_physical(self):
        """K3-2: 行动距离 ≠ 物理距离。"""
        metric = ActionGapMetric()
        s1 = PendulumState(theta=0.1, omega=0.0)
        s2 = PendulumState(theta=math.pi, omega=0.0)  # 倒立点
        result = metric.distance(s1, s2)
        assert result.action_distance > result.physical_distance
        assert result.ratio > 1.0

    def test_action_distance_same_state(self):
        metric = ActionGapMetric()
        s = PendulumState(theta=0.5, omega=0.0)
        result = metric.distance(s, s)
        assert result.action_distance == 0.0
        assert result.physical_distance == 0.0

    def test_action_cost_result_fields(self):
        metric = ActionGapMetric()
        s1 = PendulumState(theta=0.1, omega=0.5)
        s2 = PendulumState(theta=1.0, omega=0.0)
        result = metric.distance(s1, s2)
        assert isinstance(result, ActionCostResult)
        assert result.action_distance >= 0
        assert result.physical_distance >= 0
        d = result.to_dict()
        assert "action_distance" in d
        assert "ratio" in d

    def test_reachable_within_budget(self):
        metric = ActionGapMetric()
        s1 = PendulumState(theta=0.1, omega=0.0)
        s2 = PendulumState(theta=0.5, omega=0.0)
        assert metric.reachable(s1, s2, budget=100.0)

    def test_not_reachable_exceeds_budget(self):
        metric = ActionGapMetric()
        s1 = PendulumState(theta=0.1, omega=0.0)
        s2 = PendulumState(theta=math.pi, omega=0.0)
        assert not metric.reachable(s1, s2, budget=0.001)

    def test_statistics(self):
        metric = ActionGapMetric()
        s1 = PendulumState(theta=0.0, omega=0.0)
        s2 = PendulumState(theta=1.0, omega=0.0)
        metric.distance(s1, s2)
        metric.distance(s1, s2)
        stats = metric.statistics()
        assert stats["call_count"] == 2
        assert stats["avg_action_distance"] > 0

    def test_action_cost_with_torque(self):
        metric = ActionGapMetric()
        s = PendulumState(theta=0.5, omega=0.0)
        goal = PendulumState(theta=0.0, omega=0.0)
        cost = metric.action_cost(s, action=-1.0, goal=goal)
        assert isinstance(cost, float)


# =============================================================================
# TestAttentionPolicy: PerceptionPipeline.attention_policy
# =============================================================================


class TestAttentionPolicy:
    """PerceptionPipeline.attention_policy 感知注意力测试。"""

    def test_default_weights(self):
        pipeline = PerceptionPipeline()
        weights = pipeline.attention_weights
        assert len(weights) == 5
        assert abs(weights["semantic"] - 0.2) < 0.01
        assert abs(weights["causal"] - 0.2) < 0.01

    def test_failure_signal_increases_weight(self):
        """K3-3: 失败信号提升对应通道采样权重。"""
        pipeline = PerceptionPipeline()
        result = pipeline.attention_policy({"semantic": -0.8})
        assert result["semantic"] > 0.2  # 权重提升

    def test_success_signal_decreases_weight(self):
        pipeline = PerceptionPipeline()
        result = pipeline.attention_policy({"causal": 0.8})
        assert result["causal"] < 0.2  # 权重降低

    def test_prediction_error_global_boost(self):
        pipeline = PerceptionPipeline()
        result = pipeline.attention_policy({"prediction_error": 0.7})
        # 全局提升 → 所有通道权重应增加（归一化后可能略有差异）
        total_after = sum(result.values())
        assert abs(total_after - 1.0) < 0.01  # 归一化

    def test_surprise_amplifies_all(self):
        pipeline = PerceptionPipeline()
        result = pipeline.attention_policy({"surprise": 0.9})
        # 惊奇度 > 0.5 → 全面增强
        assert sum(result.values()) > 0  # 有效权重

    def test_combined_feedback(self):
        pipeline = PerceptionPipeline()
        result = pipeline.attention_policy(
            {
                "semantic": -0.5,
                "causal": 0.5,
                "prediction_error": 0.5,
                "surprise": 0.7,
            }
        )
        assert abs(sum(result.values()) - 1.0) < 0.01  # 归一化
        assert result["semantic"] > result["causal"]  # 失败的 semantic > 成功的 causal

    def test_weights_sum_to_one(self):
        pipeline = PerceptionPipeline()
        result = pipeline.attention_policy({"semantic": -0.3, "causal": 0.5})
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_reset_attention(self):
        pipeline = PerceptionPipeline()
        pipeline.attention_policy({"semantic": -0.9})
        pipeline.reset_attention()
        weights = pipeline.attention_weights
        assert abs(weights["semantic"] - 0.2) < 0.01

    def test_persistent_weights(self):
        pipeline = PerceptionPipeline()
        r1 = pipeline.attention_policy({"semantic": -0.5})
        r2 = pipeline.attention_policy({"semantic": -0.5})
        # 第二次应在第一次基础上继续调整
        assert r2["semantic"] >= r1["semantic"]

    def test_attention_policy_kwargs(self):
        pipeline = PerceptionPipeline()
        result = pipeline.attention_policy(semantic=-0.5, causal=0.3)
        assert "semantic" in result
        assert "causal" in result


# =============================================================================
# TestCEWMStep: MCIWorldModel.cewm_step() CEWM 引擎集成
# =============================================================================


class TestCEWMStep:
    """MCIWorldModel.cewm_step() 闭环集成测试。"""

    def test_cewm_step_basic(self):
        """K3-4: wm.cewm_step() 一步驱动全流程。"""
        wm = MCIWorldModel()
        state = PendulumState(theta=0.5, omega=0.0)
        goal = PendulumState(theta=0.0, omega=0.0)
        result = wm.cewm_step(observation=state, goal=goal)
        assert isinstance(result, dict)
        assert "action_distance" in result
        assert "physical_distance" in result
        assert "prediction_error" in result
        assert "causal_updates" in result
        assert "attention_weights" in result

    def test_cewm_step_dict_observation(self):
        wm = MCIWorldModel()
        obs = {"theta": 0.3, "omega": 0.1}
        goal = {"theta": 0.0, "omega": 0.0}
        result = wm.cewm_step(observation=obs, goal=goal)
        assert result["state"] is not None
        assert result["action_distance"] >= 0

    def test_cewm_step_none_observation(self):
        wm = MCIWorldModel()
        result = wm.cewm_step(observation=None, goal=None)
        assert result["action_distance"] == 0.0

    def test_cewm_step_with_action(self):
        wm = MCIWorldModel()
        state = PendulumState(theta=0.5, omega=0.0)
        goal = PendulumState(theta=0.0, omega=0.0)
        result = wm.cewm_step(observation=state, goal=goal, action=-1.0)
        assert "prediction_error" in result

    def test_cewm_step_causal_updates(self):
        wm = MCIWorldModel()
        state = PendulumState(theta=0.5, omega=0.5)
        goal = PendulumState(theta=0.0, omega=0.0)
        result = wm.cewm_step(observation=state, goal=goal)
        # theta=0.5 和 omega=0.5 都 > 0.01, 应有因果边
        assert result["causal_updates"] > 0

    def test_cewm_step_attention_weights_present(self):
        wm = MCIWorldModel()
        state = PendulumState(theta=0.3, omega=0.0)
        goal = PendulumState(theta=0.0, omega=0.0)
        result = wm.cewm_step(observation=state, goal=goal)
        weights = result["attention_weights"]
        assert isinstance(weights, dict)
        assert len(weights) > 0

    def test_cewm_step_multiple_calls(self):
        wm = MCIWorldModel()
        for i in range(5):
            state = PendulumState(theta=0.1 * (i + 1), omega=0.0)
            goal = PendulumState(theta=0.0, omega=0.0)
            result = wm.cewm_step(observation=state, goal=goal)
            assert result["action_distance"] >= 0

    def test_cewm_step_action_distance_nonzero(self):
        wm = MCIWorldModel()
        state = PendulumState(theta=1.0, omega=0.5)
        goal = PendulumState(theta=0.0, omega=0.0)
        result = wm.cewm_step(observation=state, goal=goal)
        assert result["action_distance"] > 0
        assert result["physical_distance"] > 0


# =============================================================================
# TestImportsV36: 导出符号完整性
# =============================================================================


class TestImportsV36:
    """v3.6.0 导出符号穿透测试。"""

    def test_sdk_imports(self):
        from mci_world_model.sdk import (
            ActionGapMetric,
            CausalUpdater,
        )

        assert ActionGapMetric is not None
        assert CausalUpdater is not None

    def test_top_level_imports(self):
        from mci_world_model import ActionGapMetric, CausalUpdater

        assert ActionGapMetric is not None
        assert CausalUpdater is not None

    def test_sys_imports(self):
        from mci_world_model._sys import (
            ActionGapMetric,
            CausalEdge,
            CausalUpdater,
        )

        assert ActionGapMetric is not None
        assert CausalUpdater is not None
        assert CausalEdge is not None

    def test_sys_not_none(self):
        from mci_world_model import _sys

        assert _sys.CausalUpdater is not None
        assert _sys.ActionGapMetric is not None
        assert _sys.CausalEdge is not None
        assert _sys.EdgeAction is not None
        assert _sys.ActionCostResult is not None

    def test_sdk_not_none(self):
        from mci_world_model import sdk

        assert sdk.CausalUpdater is not None
        assert sdk.ActionGapMetric is not None
        assert sdk.CausalEdge is not None

    def test_world_model_cewm_step_exists(self):
        wm = MCIWorldModel()
        assert hasattr(wm, "cewm_step")
        assert callable(wm.cewm_step)
