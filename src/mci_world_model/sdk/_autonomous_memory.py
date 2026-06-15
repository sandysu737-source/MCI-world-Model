"""自主记忆管理 — TASK-C3。

使用强化学习策略决策哪些经验应该巩固、哪些应该遗忘，
替代 PersistentExperienceMemory 中硬编码的 forget(decay_factor) 时间衰减策略。

核心设计:
    记忆管理策略 π(action | memory_state):
        action ∈ {RETAIN, CONSOLIDATE, FORGET, ARCHIVE}
        memory_state = (age, access_count, relevance_score, importance_score, decay_utility)

    奖励函数:
        R = α · future_retrieval_hit_rate - β · storage_cost + γ · importance_preservation

    训练: 简化 REINFORCE 算法 (无策略网络, 使用评分函数)
        1. 对每条记忆计算 retain_score = f(features)
        2. retain_score > θ_consolidate → CONSOLIDATE (提升优先级)
        3. retain_score > θ_retain → RETAIN (保持)
        4. retain_score > θ_archive → ARCHIVE (压缩存储)
        5. retain_score ≤ θ_archive → FORGET (删除)

    评分函数:
        retain_score = w1·importance + w2·(1 - age_norm) + w3·access_freq + w4·relevance

方法签名:
    AutonomousMemoryManager.__init__(config=AMMConfig())
    AutonomousMemoryManager.decide(memories) → list[MemoryDecision]
    AutonomousMemoryManager.execute(decisions, memory_store) → dict
    AutonomousMemoryManager.update_policy(feedback) → None
    AutonomousMemoryManager.train(episodes) → dict

数据结构:
    MemoryFeatures: 记忆特征向量
    MemoryDecision: 管理决策
    AMMConfig: 配置
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 枚举与核心数据结构
# =============================================================================


class MemoryAction(Enum):
    """记忆管理动作。"""

    RETAIN = "retain"  # 保持
    CONSOLIDATE = "consolidate"  # 巩固 (提升优先级)
    ARCHIVE = "archive"  # 归档 (压缩存储)
    FORGET = "forget"  # 遗忘 (删除)


@dataclass
class MemoryFeatures:
    """记忆特征向量。

    Attributes:
        memory_id: 记忆唯一标识
        age: 年龄 (距创建的步数)
        access_count: 被检索访问次数
        relevance_score: 与当前任务的相关性 [0, 1]
        importance_score: 重要性评分 [0, 1]
        decay_utility: 衰减效用值
        tags: 标签列表
    """

    memory_id: str = ""
    age: int = 0
    access_count: int = 0
    relevance_score: float = 0.5
    importance_score: float = 0.5
    decay_utility: float = 1.0
    tags: list[str] = field(default_factory=list)

    def to_vector(self) -> np.ndarray:
        """转为特征向量。"""
        return np.array(
            [
                min(self.age / 1000.0, 1.0),  # 归一化年龄
                min(self.access_count / 100.0, 1.0),  # 归一化访问频次
                self.relevance_score,
                self.importance_score,
                self.decay_utility,
            ],
            dtype=np.float64,
        )


@dataclass
class MemoryDecision:
    """记忆管理决策。

    Attributes:
        memory_id: 记忆标识
        action: 选择的动作
        retain_score: 保留评分
        confidence: 决策置信度
        reason: 决策理由
    """

    memory_id: str = ""
    action: MemoryAction = MemoryAction.RETAIN
    retain_score: float = 0.5
    confidence: float = 0.5
    reason: str = ""


@dataclass
class AMMConfig:
    """自主记忆管理配置。

    Attributes:
        w_importance: 重要性权重
        w_age: 年龄权重 (负, 越老越倾向遗忘)
        w_access: 访问频次权重
        w_relevance: 相关性权重
        theta_consolidate: 巩固阈值
        theta_retain: 保留阈值
        theta_archive: 归档阈值
        learning_rate: 策略学习率
        max_storage_ratio: 最大存储比例 (超过则触发遗忘)
        seed: 随机种子
    """

    w_importance: float = 0.35
    w_age: float = -0.15
    w_access: float = 0.25
    w_relevance: float = 0.25
    theta_consolidate: float = 0.8
    theta_retain: float = 0.5
    theta_archive: float = 0.3
    learning_rate: float = 0.01
    max_storage_ratio: float = 0.9
    seed: int = 42


# =============================================================================
# AutonomousMemoryManager — 自主记忆管理器
# =============================================================================


class AutonomousMemoryManager:
    """自主记忆管理器。

    用强化学习策略替代硬编码 forget(decay_factor)。

    用法:
        >>> amm = AutonomousMemoryManager(AMMConfig())
        >>> features = [MemoryFeatures("m1", age=100, access_count=5, ...)]
        >>> decisions = amm.decide(features)
        >>> amm.execute(decisions, memory_store)
    """

    def __init__(self, config: AMMConfig | None = None):
        self._config = config or AMMConfig()
        self._rng = np.random.RandomState(self._config.seed)
        self._decision_history: list[list[MemoryDecision]] = []
        self._reward_history: list[float] = []

    # -----------------------------------------------------------------
    # 公开 API
    # -----------------------------------------------------------------

    def decide(self, memories: list[MemoryFeatures]) -> list[MemoryDecision]:
        """对所有记忆做管理决策。

        评分公式:
            retain_score = w1·importance + w2·(1 - age_norm) + w3·access_freq + w4·relevance

        决策规则:
            retain_score ≥ θ_consolidate → CONSOLIDATE
            retain_score ≥ θ_retain → RETAIN
            retain_score ≥ θ_archive → ARCHIVE
            retain_score < θ_archive → FORGET

        Args:
            memories: 记忆特征列表

        Returns:
            决策列表
        """
        decisions: list[MemoryDecision] = []

        for mem in memories:
            score = self._compute_retain_score(mem)

            if score >= self._config.theta_consolidate:
                action = MemoryAction.CONSOLIDATE
                reason = f"高保留评分 ({score:.3f} ≥ {self._config.theta_consolidate})"
            elif score >= self._config.theta_retain:
                action = MemoryAction.RETAIN
                reason = f"中等保留评分 ({score:.3f} ≥ {self._config.theta_retain})"
            elif score >= self._config.theta_archive:
                action = MemoryAction.ARCHIVE
                reason = f"低保留评分 ({score:.3f} ≥ {self._config.theta_archive})"
            else:
                action = MemoryAction.FORGET
                reason = f"极低保留评分 ({score:.3f} < {self._config.theta_archive})"

            confidence = min(abs(score - 0.5) * 2, 1.0)

            decisions.append(
                MemoryDecision(
                    memory_id=mem.memory_id,
                    action=action,
                    retain_score=score,
                    confidence=confidence,
                    reason=reason,
                )
            )

        self._decision_history.append(decisions)
        return decisions

    def execute(self, decisions: list[MemoryDecision], memory_store: dict[str, Any]) -> dict[str, Any]:
        """执行管理决策。

        Args:
            decisions: 决策列表
            memory_store: 记忆存储 {memory_id: data}

        Returns:
            {"n_consolidated": int, "n_retained": int, "n_archived": int, "n_forgotten": int}
        """
        counts = {action.value: 0 for action in MemoryAction}

        for decision in decisions:
            mid = decision.memory_id
            action = decision.action

            if action == MemoryAction.FORGET:
                memory_store.pop(mid, None)
            elif action == MemoryAction.CONSOLIDATE:
                if mid in memory_store and isinstance(memory_store[mid], dict):
                    memory_store[mid]["_priority"] = "high"
            elif action == MemoryAction.ARCHIVE:
                if mid in memory_store and isinstance(memory_store[mid], dict):
                    memory_store[mid]["_archived"] = True
            # RETAIN: 不做操作

            counts[action.value] += 1

        return counts

    def update_policy(self, feedback: dict[str, float]) -> None:
        """根据反馈更新策略权重。

        REINFORCE 风格更新:
            w_i += lr × (R - baseline) × ∂score/∂w_i

        Args:
            feedback: {metric: value}
                - "retrieval_hit_rate": 检索命中率
                - "storage_usage": 存储使用率
                - "importance_preservation": 重要记忆保留率
        """
        retrieval_rate = feedback.get("retrieval_hit_rate", 0.5)
        storage_usage = feedback.get("storage_usage", 0.5)
        importance_preservation = feedback.get("importance_preservation", 0.5)

        # 简化 REINFORCE: 奖励信号
        reward = 0.4 * retrieval_rate + 0.3 * (1.0 - storage_usage) + 0.3 * importance_preservation
        self._reward_history.append(reward)

        lr = self._config.learning_rate
        baseline = np.mean(self._reward_history) if len(self._reward_history) > 1 else 0.5
        advantage = reward - baseline

        # 更新权重
        self._config.w_importance += lr * advantage * 0.1
        self._config.w_access += lr * advantage * 0.1
        self._config.w_relevance += lr * advantage * 0.1

        # 年龄权重保持负 (越老越倾向遗忘)
        self._config.w_age = min(self._config.w_age + lr * advantage * 0.05, -0.01)

    def train(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        """训练策略。

        Args:
            episodes: 训练场景列表 [{"memories": [...], "feedback": {...}}]

        Returns:
            {"n_episodes": int, "avg_reward": float}
        """
        if not episodes:
            return {"n_episodes": 0, "avg_reward": 0.0}

        rewards = []
        for episode in episodes:
            memories = episode.get("memories", [])
            feedback = episode.get("feedback", {})

            self.decide(memories)
            self.update_policy(feedback)

            r = (
                0.4 * feedback.get("retrieval_hit_rate", 0.5)
                + 0.3 * (1 - feedback.get("storage_usage", 0.5))
                + 0.3 * feedback.get("importance_preservation", 0.5)
            )
            rewards.append(r)

        return {
            "n_episodes": len(episodes),
            "avg_reward": float(np.mean(rewards)),
        }

    @property
    def config(self) -> AMMConfig:
        return self._config

    @property
    def decision_history(self) -> list[list[MemoryDecision]]:
        return list(self._decision_history)

    @property
    def reward_history(self) -> list[float]:
        return list(self._reward_history)

    # -----------------------------------------------------------------
    # 内部方法
    # -----------------------------------------------------------------

    def _compute_retain_score(self, mem: MemoryFeatures) -> float:
        """计算保留评分。

        retain_score = w1·importance + w2·(1 - age_norm) + w3·access_freq + w4·relevance
        """
        age_norm = min(mem.age / 1000.0, 1.0)
        access_norm = min(mem.access_count / 100.0, 1.0)

        score = (
            self._config.w_importance * mem.importance_score
            + self._config.w_age * age_norm  # w_age < 0
            + self._config.w_access * access_norm
            + self._config.w_relevance * mem.relevance_score
        )

        # 加入 decay_utility 修正
        score *= mem.decay_utility

        return float(np.clip(score, 0.0, 1.0))
