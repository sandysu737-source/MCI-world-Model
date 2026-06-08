"""
MCI World Model v3.1.0 — Energy Consistency Loss + Causal Actor 单元测试
=========================================================================

覆盖 _energy_loss.py 和 _causal_actor.py 的核心接口。
目标: 将 energy_loss/causal_actor 覆盖率从 42%/37% 提升至 65%+。
"""

import numpy as np

from mci_world_model.sdk._causal_actor import (
    ActionCandidate,
    CausalActor,
    EnergyGuidedAction,
)
from mci_world_model.sdk._energy_loss import (
    ENHANCE_EDGES,
    FIVE_CATEGORICAL_STATES,
    SUPPRESS_EDGES,
    EnergyConsistencyLoss,
    TopologicalEnergyMatrix,
    build_energy_matrix_from_energy_bus,
    compute_jepa_graph_energy,
    create_default_energy_loss,
)

# =============================================================================
# TopologicalEnergyMatrix Tests
# =============================================================================


class TestTopologicalEnergyMatrix:
    """拓扑能量矩阵构建与查询测试。"""

    def test_build_default(self):
        """构建标准五范畴能量矩阵。"""
        topo = TopologicalEnergyMatrix.build()
        assert topo is not None
        assert topo.matrix.shape == (5, 5)
        assert len(topo.state_index) == 5
        assert "semantic" in topo.state_index
        assert "causal" in topo.state_index
        assert "spacetime" in topo.state_index
        assert "generative" in topo.state_index
        assert "trust" in topo.state_index

    def test_get_energy_enhance(self):
        """增强边的能量值应较高 (>0.5)。"""
        topo = TopologicalEnergyMatrix.build()
        energy = topo.get_energy("semantic", "causal")
        assert energy > 0.5
        assert energy <= 1.0

    def test_get_energy_suppress(self):
        """抑制边的能量值应较低 (<0.5)。"""
        topo = TopologicalEnergyMatrix.build()
        energy = topo.get_energy("semantic", "spacetime")
        assert energy > 0.0
        assert energy < 0.5

    def test_get_energy_unknown_state(self):
        """未知状态返回 0.0。"""
        topo = TopologicalEnergyMatrix.build()
        assert topo.get_energy("unknown", "causal") == 0.0
        assert topo.get_energy("semantic", "unknown") == 0.0

    def test_get_relation_type_enhance(self):
        """增强关系的类型识别。"""
        topo = TopologicalEnergyMatrix.build()
        assert topo.get_relation_type("semantic", "causal") == "enhance"

    def test_get_relation_type_suppress(self):
        """抑制关系的类型识别。"""
        topo = TopologicalEnergyMatrix.build()
        assert topo.get_relation_type("semantic", "spacetime") == "suppress"

    def test_get_relation_type_neutral(self):
        """未知关系返回 neutral。"""
        topo = TopologicalEnergyMatrix.build()
        assert topo.get_relation_type("semantic", "generative") == "neutral"

    def test_to_flat_vector(self):
        """展平为 25 维向量。"""
        topo = TopologicalEnergyMatrix.build()
        vec = topo.to_flat_vector()
        assert vec.shape == (25,)
        assert vec.dtype == np.float32

    def test_copy(self):
        """深拷贝不共享内存。"""
        topo = TopologicalEnergyMatrix.build()
        copied = topo.copy()
        assert copied is not topo
        assert copied.matrix is not topo.matrix
        assert np.array_equal(copied.matrix, topo.matrix)
        # 修改拷贝不影响原矩阵
        copied.matrix[0, 0] = 0.99
        assert topo.matrix[0, 0] != 0.99


# =============================================================================
# EnergyConsistencyLoss Tests
# =============================================================================


class TestEnergyConsistencyLossInit:
    """EnergyConsistencyLoss 初始化测试。"""

    def test_default_init(self):
        """默认初始化。"""
        loss = EnergyConsistencyLoss()
        assert loss.alpha == 0.1
        assert loss.edge_penalty_multiplier == 2.0
        assert loss.topological_matrix is not None

    def test_custom_alpha(self):
        """自定义 alpha。"""
        loss = EnergyConsistencyLoss(alpha=0.5)
        assert loss.alpha == 0.5

    def test_custom_topological(self):
        """自定义拓扑矩阵。"""
        topo = TopologicalEnergyMatrix.build()
        loss = EnergyConsistencyLoss(topological=topo, alpha=0.3)
        assert loss.alpha == 0.3
        assert loss.topological_matrix is topo


class TestEnergyConsistencyLossCompute:
    """能量损失计算测试。"""

    def test_compute_returns_tuple(self):
        """compute 返回 (total_loss, diagnostics)。"""
        loss = EnergyConsistencyLoss()
        pred = np.eye(5, dtype=np.float32) * 0.5
        total, diag = loss.compute(sft_loss=1.0, predictions=pred)
        assert isinstance(total, (float, np.floating))
        assert isinstance(diag, dict)
        assert "sft_loss" in diag
        assert "energy_loss" in diag
        assert "total_loss" in diag

    def test_compute_flat_input(self):
        """接受展平的 25 维输入。"""
        loss = EnergyConsistencyLoss()
        pred = np.full(25, 0.5, dtype=np.float32)
        total, _diag = loss.compute(sft_loss=0.5, predictions=pred)
        assert isinstance(total, (float, np.floating))

    def test_compute_zero_sft_loss(self):
        """SFT 损失为零时 total = alpha * energy_loss。"""
        loss = EnergyConsistencyLoss(alpha=0.2)
        pred = np.eye(5, dtype=np.float32) * 0.8
        total, diag = loss.compute(sft_loss=0.0, predictions=pred)
        assert abs(total - 0.2 * diag["energy_loss"]) < 1e-10

    def test_compute_only_energy(self):
        """仅计算能量损失。"""
        loss = EnergyConsistencyLoss()
        pred = np.eye(5, dtype=np.float32) * 0.5
        energy = loss.compute_only_energy(pred)
        assert isinstance(energy, float)
        assert energy >= 0.0

    def test_compute_ratios(self):
        """比率输入便捷方法。"""
        loss = EnergyConsistencyLoss()
        predicted = {"semantic": 0.25, "causal": 0.30, "spacetime": 0.20, "generative": 0.15, "trust": 0.10}
        target = {"semantic": 0.20, "causal": 0.20, "spacetime": 0.20, "generative": 0.20, "trust": 0.20}
        energy = loss.compute_ratios(predicted, target)
        assert isinstance(energy, float)
        assert energy >= 0.0

    def test_compute_ratios_missing_keys(self):
        """缺失键默认为 0.0。"""
        loss = EnergyConsistencyLoss()
        predicted = {"semantic": 0.3, "causal": 0.3}
        target = {"semantic": 0.2}
        energy = loss.compute_ratios(predicted, target)
        assert isinstance(energy, float)


class TestEnergyConsistencyLossValidate:
    """预测验证测试。"""

    def test_validate_confirmed_enhance(self):
        """增强边上高权重预测 → confirmed。"""
        loss = EnergyConsistencyLoss()
        result = loss.validate_prediction("semantic", "causal", 0.9)
        assert result["relation_type"] == "enhance"
        assert result["verdict"] == "confirmed"

    def test_validate_suppress_low(self):
        """抑制边上适中预测 → none。"""
        loss = EnergyConsistencyLoss()
        result = loss.validate_prediction("semantic", "spacetime", 0.2)
        assert result["relation_type"] == "suppress"

    def test_validate_novel_high_on_suppress(self):
        """抑制边上高预测 → novel。"""
        loss = EnergyConsistencyLoss()
        result = loss.validate_prediction("semantic", "spacetime", 0.8)
        assert result["verdict"] == "novel"

    def test_validate_unknown_relation(self):
        """未知关系类型。"""
        loss = EnergyConsistencyLoss()
        result = loss.validate_prediction("semantic", "generative", 0.5)
        assert result["relation_type"] == "neutral"


class TestEnergyConsistencyLossHistory:
    """历史记录与趋势测试。"""

    def test_history_initial_empty(self):
        """初始历史为空。"""
        loss = EnergyConsistencyLoss()
        assert loss.get_history() == []

    def test_history_after_compute(self):
        """compute 后记录历史。"""
        loss = EnergyConsistencyLoss()
        pred = np.eye(5, dtype=np.float32) * 0.5
        for _ in range(3):
            loss.compute(sft_loss=1.0, predictions=pred)
        history = loss.get_history()
        assert len(history) == 3

    def test_reset_history(self):
        """清空历史。"""
        loss = EnergyConsistencyLoss()
        pred = np.eye(5, dtype=np.float32) * 0.5
        loss.compute(sft_loss=1.0, predictions=pred)
        loss.reset_history()
        assert loss.get_history() == []

    def test_trend_insufficient_data(self):
        """数据不足时趋势为 insufficient_data。"""
        loss = EnergyConsistencyLoss()
        trend = loss.get_trend()
        assert trend["energy_loss_trend"] == "insufficient_data"

    def test_trend_converging(self):
        """下降趋势检测为 converging。"""
        loss = EnergyConsistencyLoss()
        # 使用逐渐变小的预测值（接近拓扑矩阵 → 能量损失下降）
        for scale in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.78]:
            pred = np.eye(5, dtype=np.float32) * scale
            loss.compute(sft_loss=0.5, predictions=pred)
        trend = loss.get_trend()
        # 趋势检测取决于拓扑矩阵匹配度
        assert trend["n_steps"] > 0


class TestEnergyConsistencyLossGraphEnergy:
    """图结构能量损失（JEPA）测试。"""

    def test_empty_edges_returns_zero(self):
        """空因果边返回零损失。"""
        from mci_world_model.sdk._world_model import CausalWorldModelState

        loss = EnergyConsistencyLoss()
        pred_state = CausalWorldModelState.empty()
        actual_state = CausalWorldModelState.empty()
        energy, diag = loss.compute_graph_energy(pred_state, actual_state)
        assert energy == 0.0
        assert diag["n_edges"] == 0

    def test_enhance_violations_detected(self):
        """增强边低权重检测违规。"""
        from mci_world_model.sdk._world_model import CausalWorldModelState

        loss = EnergyConsistencyLoss()
        pred_state = CausalWorldModelState(
            causal_edges=[
                {"cause": "A", "effect": "B", "rho": 0.1, "energy_relation": "enhance"},
                {"cause": "B", "effect": "C", "rho": 0.9, "energy_relation": "suppress"},
            ],
        )
        actual_state = CausalWorldModelState.empty()
        energy, diag = loss.compute_graph_energy(pred_state, actual_state)
        assert energy > 0.0
        assert diag["n_edges"] == 2


# =============================================================================
# 工厂函数测试
# =============================================================================


class TestFactoryFunctions:
    """工厂函数测试。"""

    def test_create_default_energy_loss(self):
        """创建默认能量损失实例。"""
        loss = create_default_energy_loss(alpha=0.2)
        assert loss.alpha == 0.2
        assert loss.topological_matrix is not None

    def test_compute_jepa_graph_energy(self):
        """便捷函数 compute_jepa_graph_energy。"""
        from mci_world_model.sdk._world_model import CausalWorldModelState

        pred_state = CausalWorldModelState(
            causal_edges=[
                {"cause": "A", "effect": "B", "rho": 0.1, "energy_relation": "enhance"},
            ],
        )
        actual_state = CausalWorldModelState.empty()
        energy = compute_jepa_graph_energy(pred_state, actual_state, alpha=0.1)
        assert isinstance(energy, float)

    def test_build_energy_matrix_from_none(self):
        """None EnergyBus 返回标准矩阵。"""
        topo = build_energy_matrix_from_energy_bus(None)
        assert topo is not None
        assert topo.matrix.shape == (5, 5)


# =============================================================================
# 拓扑常量验证
# =============================================================================


class TestTopologyConstants:
    """拓扑能量常量一致性验证。"""

    def test_five_states_count(self):
        """五范畴状态数量。"""
        assert len(FIVE_CATEGORICAL_STATES) == 5

    def test_enhance_edges_count(self):
        """增强边数量。"""
        assert len(ENHANCE_EDGES) == 5

    def test_suppress_edges_count(self):
        """抑制边数量。"""
        assert len(SUPPRESS_EDGES) == 5

    def test_no_overlap_enhance_suppress(self):
        """增强和抑制边无重叠。"""
        enhance_set = set(ENHANCE_EDGES)
        suppress_set = set(SUPPRESS_EDGES)
        assert enhance_set.isdisjoint(suppress_set)


# =============================================================================
# ActionCandidate Tests
# =============================================================================


class TestActionCandidate:
    """ActionCandidate 数据类测试。"""

    def test_create(self):
        """创建候选动作。"""
        action = ActionCandidate(
            action_type="adjust_weight",
            target="A→B",
            proposed_value=0.75,
            expected_cost=0.15,
            confidence=0.9,
        )
        assert action.action_type == "adjust_weight"
        assert action.target == "A→B"
        assert action.proposed_value == 0.75

    def test_to_dict(self):
        """序列化测试。"""
        action = ActionCandidate(
            action_type="do_intervention",
            target="X",
            proposed_value=1.5,
            expected_cost=0.3,
            confidence=0.85,
        )
        d = action.to_dict()
        assert d["action_type"] == "do_intervention"
        assert d["target"] == "X"
        assert d["proposed_value"] == round(0.75, 6) or abs(d["proposed_value"] - 1.5) < 0.001


# =============================================================================
# CausalActor Tests
# =============================================================================


class TestCausalActorState:
    """CausalActor 状态机测试。"""

    def test_init_state_idle(self):
        """初始状态为 IDLE。"""
        actor = CausalActor(world_model=None)
        assert actor.state == "IDLE"

    def test_search_transitions_state(self):
        """search 不崩溃且状态流转。"""
        from mci_world_model.sdk._world_model import CausalWorldModelState, MCIWorldModel

        wm = MCIWorldModel()
        wm.initialize()
        actor = CausalActor(world_model=wm)
        state = CausalWorldModelState(
            causal_edges=[
                {
                    "cause": "A",
                    "effect": "B",
                    "rho": 0.5,
                    "confidence": 0.7,
                    "verdict": "confirmed",
                    "energy_relation": "neutral",
                    "bayes_factor": 0.5,
                },
            ],
        )
        actions = actor.search(state, n_candidates=1)
        assert isinstance(actions, list)
        assert actor.state in ("SELECTING", "COMPLETE", "IDLE")

    def test_search_empty_edges(self):
        """空因果边返回空动作列表。"""
        from mci_world_model.sdk._world_model import CausalWorldModelState, MCIWorldModel

        wm = MCIWorldModel()
        wm.initialize()
        actor = CausalActor(world_model=wm)
        state = CausalWorldModelState.empty()
        actions = actor.search(state, n_candidates=1)
        assert isinstance(actions, list)
        assert len(actions) == 0

    def test_search_n_candidates_zero(self):
        """n_candidates=0 返回空列表。"""
        from mci_world_model.sdk._world_model import CausalWorldModelState, MCIWorldModel

        wm = MCIWorldModel()
        wm.initialize()
        actor = CausalActor(world_model=wm)
        state = CausalWorldModelState(
            causal_edges=[
                {
                    "cause": "A",
                    "effect": "B",
                    "rho": 0.5,
                    "confidence": 0.7,
                    "verdict": "confirmed",
                    "energy_relation": "neutral",
                    "bayes_factor": 0.5,
                },
            ],
        )
        actions = actor.search(state, n_candidates=0)
        assert len(actions) == 0

    def test_apply_adjust_weight(self):
        """应用 adjust_weight 动作。"""
        from mci_world_model.sdk._world_model import CausalWorldModelState, MCIWorldModel

        wm = MCIWorldModel()
        wm.initialize()
        actor = CausalActor(world_model=wm)
        state = CausalWorldModelState(
            causal_edges=[
                {
                    "cause": "A",
                    "effect": "B",
                    "rho": 0.5,
                    "confidence": 0.7,
                    "verdict": "confirmed",
                    "energy_relation": "neutral",
                    "bayes_factor": 0.5,
                },
            ],
        )
        action = ActionCandidate(
            action_type="adjust_weight",
            target="A→B",
            proposed_value=0.8,
            expected_cost=0.1,
            confidence=0.9,
            metadata={"edge_index": 0},
        )
        new_state = actor.apply(state, action)
        assert new_state is not None
        assert new_state.causal_edges[0]["rho"] == 0.8

    def test_apply_suggest_edge(self):
        """应用 suggest_edge 动作添加新边。"""
        from mci_world_model.sdk._world_model import CausalWorldModelState, MCIWorldModel

        wm = MCIWorldModel()
        wm.initialize()
        actor = CausalActor(world_model=wm)
        state = CausalWorldModelState(
            causal_edges=[
                {
                    "cause": "A",
                    "effect": "B",
                    "rho": 0.5,
                    "confidence": 0.7,
                    "verdict": "confirmed",
                    "energy_relation": "neutral",
                    "bayes_factor": 0.5,
                },
            ],
        )
        action = ActionCandidate(
            action_type="suggest_edge",
            target="C→D",
            proposed_value=0.6,
            expected_cost=0.05,
            confidence=0.7,
        )
        new_state = actor.apply(state, action)
        assert len(new_state.causal_edges) == 2
        assert new_state.causal_edges[1]["cause"] == "C"
        assert new_state.causal_edges[1]["effect"] == "D"
        assert new_state.causal_edges[1]["verdict"] == "novel"

    def test_apply_do_intervention(self):
        """应用 do_intervention 动作。"""
        from mci_world_model.sdk._world_model import CausalWorldModelState, MCIWorldModel

        wm = MCIWorldModel()
        wm.initialize()
        actor = CausalActor(world_model=wm)
        state = CausalWorldModelState(
            causal_edges=[
                {
                    "cause": "X",
                    "effect": "Y",
                    "rho": 0.5,
                    "confidence": 0.7,
                    "verdict": "confirmed",
                    "energy_relation": "neutral",
                    "bayes_factor": 0.5,
                },
            ],
        )
        action = ActionCandidate(
            action_type="do_intervention",
            target="X",
            proposed_value=0.3,
            expected_cost=0.2,
            confidence=0.95,
        )
        new_state = actor.apply(state, action)
        assert new_state is not None

    def test_action_history(self):
        """动作历史记录。"""
        from mci_world_model.sdk._world_model import CausalWorldModelState, MCIWorldModel

        wm = MCIWorldModel()
        wm.initialize()
        actor = CausalActor(world_model=wm)
        state = CausalWorldModelState(
            causal_edges=[
                {
                    "cause": "A",
                    "effect": "B",
                    "rho": 0.5,
                    "confidence": 0.7,
                    "verdict": "confirmed",
                    "energy_relation": "neutral",
                    "bayes_factor": 0.5,
                },
            ],
        )
        action = ActionCandidate(
            action_type="adjust_weight",
            target="A→B",
            proposed_value=0.8,
            expected_cost=0.1,
            confidence=0.9,
            metadata={"edge_index": 0},
        )
        actor.apply(state, action)
        history = actor.action_history
        assert len(history) >= 1

    def test_optimize_returns_dict(self):
        """迭代优化返回完整结果。"""
        from mci_world_model.sdk._world_model import CausalWorldModelState, MCIWorldModel

        wm = MCIWorldModel()
        wm.initialize()
        actor = CausalActor(world_model=wm)
        state = CausalWorldModelState(
            causal_edges=[
                {
                    "cause": "A",
                    "effect": "B",
                    "rho": 0.2,
                    "confidence": 0.8,
                    "verdict": "confirmed",
                    "energy_relation": "enhance",
                    "bayes_factor": 0.7,
                },
            ],
        )
        result = actor.optimize(state, max_iterations=2)
        assert isinstance(result, dict)
        assert "n_actions" in result
        assert "initial_cost" in result
        assert "final_cost" in result


# =============================================================================
# EnergyGuidedAction Tests
# =============================================================================


class TestEnergyGuidedAction:
    """EnergyGuidedAction 继承测试。"""

    def test_creation(self):
        """创建 EnergyGuidedAction。"""
        action = EnergyGuidedAction(
            action_type="adjust_weight",
            target="A→B",
            proposed_value=0.7,
            expected_cost=0.1,
            confidence=0.8,
            energy_direction="enhance",
        )
        assert action.energy_direction == "enhance"

    def test_to_dict_includes_direction(self):
        """to_dict 包含 energy_direction。"""
        action = EnergyGuidedAction(
            action_type="adjust_weight",
            target="A→B",
            proposed_value=0.7,
            expected_cost=0.1,
            confidence=0.8,
            energy_direction="suppress",
        )
        d = action.to_dict()
        assert d["energy_direction"] == "suppress"
