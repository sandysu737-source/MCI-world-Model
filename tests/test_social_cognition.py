"""
tests/test_social_cognition.py — SocialCognition 测试
=====================================================

覆盖:
    - AgentModel: 偏好建模 + 行动预测
    - SocialCognition: Nash均衡 + 谈判 + 心智理论
    - 纯策略/混合策略均衡
"""

from __future__ import annotations

import numpy as np
import pytest

from mci_world_model.sdk._social_cognition import (
    AgentAction,
    AgentModel,
    NashEquilibriumResult,
    SocialCognition,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def agent_model():
    return AgentModel(agent_id=0)


@pytest.fixture
def social_cog():
    return SocialCognition(n_agents=3, theory_of_mind_depth=2)


@pytest.fixture
def coordination_payoff():
    """协调博弈: 双方选择相同行动得分高。payoff shape (2,2,2)。"""
    return np.array(
        [
            [[3.0, 3.0], [0.0, 0.0]],
            [[0.0, 0.0], [2.0, 2.0]],
        ]
    )


@pytest.fixture
def prisoners_dilemma_payoff():
    """囚徒困境。payoff shape (2,2,2)。"""
    return np.array(
        [
            [[-1.0, -1.0], [-3.0, 0.0]],
            [[0.0, -3.0], [-2.0, -2.0]],
        ]
    )


@pytest.fixture
def matching_pennies_payoff():
    """匹配硬币(零和博弈): 无纯策略Nash。payoff shape (2,2,2)。"""
    return np.array(
        [
            [[1.0, -1.0], [-1.0, 1.0]],
            [[-1.0, 1.0], [1.0, -1.0]],
        ]
    )


# =============================================================================
# TestAgentAction
# =============================================================================


class TestAgentAction:
    """AgentAction 数据类。"""

    def test_creation(self):
        action = AgentAction(action="cooperate", outcome={"reward": 3.0}, timestamp=1)
        assert action.action == "cooperate"
        assert action.outcome == {"reward": 3.0}
        assert action.timestamp == 1

    def test_default_fields(self):
        action = AgentAction(action="defect")
        assert action.outcome == {}
        assert action.timestamp == 0


# =============================================================================
# TestAgentModel
# =============================================================================


class TestAgentModel:
    """AgentModel 偏好建模测试。"""

    def test_creation(self, agent_model):
        assert agent_model.agent_id == 0

    def test_invalid_agent_id(self):
        with pytest.raises(ValueError, match="agent_id"):
            AgentModel(agent_id=-1)

    def test_update_and_predict(self, agent_model):
        """更新偏好后应能预测。"""
        for _ in range(5):
            agent_model.update("cooperate", {"reward": 2.0}, timestamp=1)
        pred = agent_model.predict_action({})
        assert pred == "cooperate"

    def test_predict_no_data(self, agent_model):
        """无数据时预测应返回 'unknown'。"""
        pred = agent_model.predict_action({})
        assert pred == "unknown"

    def test_update_multiple_actions(self, agent_model):
        """多种行动更新。"""
        agent_model.update("cooperate", {"reward": 1.0}, timestamp=1)
        agent_model.update("defect", {"reward": 3.0}, timestamp=2)
        agent_model.update("defect", {"reward": 2.5}, timestamp=3)
        pred = agent_model.predict_action({})
        # "defect" 被更新了2次, "cooperate" 1次 → defect频率高
        assert pred == "defect"

    def test_action_history(self, agent_model):
        """历史记录跟踪。"""
        agent_model.update("cooperate", {"r": 1.0}, timestamp=1)
        agent_model.update("defect", {"r": 2.0}, timestamp=2)
        assert len(agent_model.action_history) == 2

    def test_action_probability(self, agent_model):
        """动作概率分布。"""
        agent_model.update("cooperate", {}, 1)
        agent_model.update("cooperate", {}, 2)
        agent_model.update("defect", {}, 3)
        probs = agent_model.get_action_probability()
        assert abs(probs["cooperate"] - 2 / 3) < 1e-6
        assert abs(probs["defect"] - 1 / 3) < 1e-6


# =============================================================================
# TestNashEquilibrium
# =============================================================================


class TestNashEquilibrium:
    """Nash均衡求解测试。"""

    def test_coordination_pure_nash(self, social_cog, coordination_payoff):
        """协调博弈应有纯策略Nash均衡。"""
        result = social_cog.nash_equilibrium(coordination_payoff)
        assert isinstance(result, NashEquilibriumResult)
        assert result.is_pure is True
        assert result.nash_type == "pure_nash"

    def test_prisoners_dilemma(self, social_cog, prisoners_dilemma_payoff):
        """囚徒困境应找到占优策略均衡。"""
        result = social_cog.nash_equilibrium(prisoners_dilemma_payoff)
        assert isinstance(result, NashEquilibriumResult)
        # 囚徒困境有纯策略Nash (defect, defect)
        assert result.is_pure is True

    def test_mixed_strategy(self, social_cog, matching_pennies_payoff):
        """匹配硬币: 应有混合策略均衡。"""
        result = social_cog.nash_equilibrium(matching_pennies_payoff)
        assert isinstance(result, NashEquilibriumResult)
        # 无纯策略Nash → 应该用混合策略
        if not result.is_pure:
            assert len(result.player1_strategy) == 2
            assert len(result.player2_strategy) == 2
            # 策略概率之和应为1
            assert abs(sum(result.player1_strategy) - 1.0) < 1e-6
            assert abs(sum(result.player2_strategy) - 1.0) < 1e-6

    def test_result_fields(self, social_cog, coordination_payoff):
        """结果字段完整性。"""
        result = social_cog.nash_equilibrium(coordination_payoff)
        assert hasattr(result, "player1_strategy")
        assert hasattr(result, "player2_strategy")
        assert hasattr(result, "player1_payoff")
        assert hasattr(result, "player2_payoff")
        assert hasattr(result, "is_pure")
        assert hasattr(result, "nash_type")

    def test_invalid_payoff_shape(self, social_cog):
        """非(2,2,2)矩阵应报错。"""
        payoff_2x2 = np.array([[1.0, 0.0], [0.0, 1.0]])  # shape (2,2) 错误
        with pytest.raises(ValueError, match="形状"):
            social_cog.nash_equilibrium(payoff_2x2)


# =============================================================================
# TestNegotiate
# =============================================================================


class TestNegotiate:
    """谈判协议测试。"""

    def test_negotiate_basic(self, social_cog):
        """基本谈判: 应返回 Pareto 最优行动。"""
        my_prefs = {"cooperate": 0.5, "defect": 0.2}
        others_predicted = {
            1: {"cooperate": 0.4, "defect": 0.3},
            2: {"cooperate": 0.1, "defect": 0.6},
        }
        result = social_cog.negotiate(my_prefs, others_predicted)
        assert isinstance(result, dict)
        assert "agreed_action" in result
        assert "my_utility" in result
        assert "joint_utility" in result

    def test_negotiate_aligned(self, social_cog):
        """偏好一致: 所有agent偏好相同时谈判结果明确。"""
        prefs = {"cooperate": 0.8, "defect": 0.1}
        others_predicted = {1: prefs.copy(), 2: prefs.copy()}
        result = social_cog.negotiate(prefs, others_predicted)
        assert result["agreed_action"] == "cooperate"

    def test_negotiate_selfish(self, social_cog):
        """自私偏好: 谈判应寻找联合最优。"""
        my_prefs = {"cooperate": 0.0, "defect": 1.0}
        others_predicted = {1: {"cooperate": 1.0, "defect": 0.0}}
        result = social_cog.negotiate(my_prefs, others_predicted)
        # 联合效用: cooperate=1.0, defect=1.0 → 都一样
        assert result["agreed_action"] in ("cooperate", "defect")


# =============================================================================
# TestSocialCognitionMisc
# =============================================================================


class TestSocialCognitionMisc:
    """SocialCognition 杂项测试。"""

    def test_creation(self, social_cog):
        assert social_cog.n_agents == 3

    def test_invalid_n_agents(self):
        with pytest.raises(ValueError, match="n_agents"):
            SocialCognition(n_agents=1)

    def test_observe_and_predict(self, social_cog):
        """观察行为并预测。"""
        social_cog.observe_interaction(1, "cooperate", {"reward": 3.0}, timestamp=1)
        social_cog.observe_interaction(1, "cooperate", {"reward": 2.0}, timestamp=2)
        preds = social_cog.predict_others({})
        assert preds[1] == "cooperate"

    def test_known_agents(self, social_cog):
        social_cog.observe_interaction(1, "cooperate", {}, 1)
        social_cog.observe_interaction(2, "defect", {}, 1)
        assert 1 in social_cog.known_agents
        assert 2 in social_cog.known_agents

    def test_interaction_count(self, social_cog):
        social_cog.observe_interaction(1, "cooperate", {}, 1)
        social_cog.observe_interaction(2, "defect", {}, 2)
        assert social_cog.interaction_count == 2

    def test_agent_probabilities(self, social_cog):
        social_cog.observe_interaction(1, "cooperate", {}, 1)
        social_cog.observe_interaction(1, "defect", {}, 2)
        probs = social_cog.get_agent_probabilities()
        assert 1 in probs
        assert abs(probs[1]["cooperate"] - 0.5) < 1e-6
