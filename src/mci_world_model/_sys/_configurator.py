"""
MCI World Model v3.0.1 — Meta Configurator
============================================

LeCun 六模块架构中的 Configurator 模块：根据认知空洞检测结果，
动态配置 World Model 各子模块的参数和行为模式。

职责：
- 接收 CognitiveGap 列表，输出配置指令
- 不修改 World Model 内部状态，仅通过公开 API 触发配置
- 基于规则决策（v3.0.1 单层规则，v3.0.3 升级为分层决策）

状态机：IDLE → ANALYZING → CONFIGURING → MONITORING → IDLE

用法:
    from mci_world_model._sys._configurator import MetaConfigurator

    cfg = MetaConfigurator()
    actions = cfg.configure(world_model, gaps)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# ConfigAction — 配置动作
# =============================================================================


@dataclass
class ConfigAction:
    """
    单次配置动作记录。

    Attributes:
        action_type: 动作类型 (enable_m3 / adjust_weights / suggest_retrain / noop)
        reason: 触发原因
        gap_id: 关联的认知空洞 ID
        timestamp: 执行时间戳
    """

    action_type: str  # enable_m3 | adjust_weights | suggest_retrain | noop
    reason: str
    gap_id: str | None = None
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# MetaConfigurator — 元级配置器
# =============================================================================


class MetaConfigurator:
    """
    基于元认知检测结果，动态配置世界模型各模块。

    六态流转：IDLE → ANALYZING → CONFIGURING → MONITORING → IDLE

    v3.0.1 规则集（单层，v3.0.3 升级为分层决策）:
        1. causal 空洞 + severity > 0.5 → 启用 M3 端到端可微模式
        2. temporal 空洞 + severity > 0.6 → 建议重新训练时序模型
        3. domain 空洞 → 调整 energy loss 权重
        4. 低置信度 causal_edges 过多 → 启用 M2 GNN 训练
        5. 能量失衡 → 提高 alpha_energy 权重
    """

    # 决策阈值
    M3_CAUSAL_SEVERITY_THRESHOLD = 0.5  # 因果空洞严重度 > 此值启用 M3
    M3_CONSECUTIVE_COUNT = 3  # 连续 N 次因果空洞则强制启用 M3
    RETRAIN_TEMPORAL_SEVERITY = 0.6  # 时序空洞严重度 > 此值建议重训
    GNN_LOW_CONFIDENCE_RATIO = 0.3  # 低置信度边占比 > 此值启用 M2

    def __init__(self):
        self._state: str = "IDLE"  # IDLE → ANALYZING → CONFIGURING → MONITORING → IDLE
        self._config_history: list[ConfigAction] = []
        self._consecutive_causal_gaps: int = 0

    # -----------------------------------------------------------------
    # 核心接口
    # -----------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def config_history(self) -> list[dict]:
        return [
            {"type": a.action_type, "reason": a.reason, "gap_id": a.gap_id, "timestamp": a.timestamp}
            for a in self._config_history[-20:]  # 最近 20 条
        ]

    def configure(self, world_model, gaps: list | None = None) -> list[ConfigAction]:
        """
        根据认知空洞自适应调整世界模型配置。

        Args:
            world_model: MCIWorldModel 实例
            gaps: CognitiveGap 列表（可选，若为 None 则空转返回 noop）

        Returns:
            执行的配置动作列表
        """
        self._state = "ANALYZING"

        if not gaps:
            self._state = "IDLE"
            return [ConfigAction(action_type="noop", reason="no_gaps_provided")]

        actions: list[ConfigAction] = []

        # ── 规则 1: causal 空洞 → 启用 M3 ──
        actions.extend(self._check_causal_gap(world_model, gaps))

        # ── 规则 2: temporal 空洞 → 建议重训 ──
        actions.extend(self._check_temporal_gap(gaps))

        # ── 规则 3: domain 空洞 → 调整权重 ──
        actions.extend(self._check_domain_gap(gaps))

        # ── 规则 4: 低置信度边过多 → 启用 M2 ──
        actions.extend(self._check_low_confidence_edges(world_model))

        # ── 记录历史 ──
        self._state = "CONFIGURING"
        self._config_history.extend(actions)
        self._state = "MONITORING"

        logger.info("MetaConfigurator 完成 %d 个配置动作", len(actions))
        return actions

    # -----------------------------------------------------------------
    # 规则方法（私有）
    # -----------------------------------------------------------------

    def _check_causal_gap(self, world_model, gaps: list) -> list[ConfigAction]:
        """规则 1: 因果空洞检测 → 启用 M3"""
        actions: list[ConfigAction] = []

        causal_gaps = [g for g in gaps if g.gap_type == "causal"]
        high_gaps = [g for g in causal_gaps if g.severity >= self.M3_CAUSAL_SEVERITY_THRESHOLD]

        if high_gaps:
            self._consecutive_causal_gaps += 1
            if self._consecutive_causal_gaps >= self.M3_CONSECUTIVE_COUNT:
                # 连续因果空洞 → 强制升级训练模式
                try:
                    world_model.enable_m3()
                    actions.append(
                        ConfigAction(
                            action_type="enable_m3",
                            reason=f"连续 {self._consecutive_causal_gaps} 次因果空洞 (severity>{self.M3_CAUSAL_SEVERITY_THRESHOLD})",
                            gap_id=high_gaps[0].gap_id,
                        )
                    )
                    logger.info(
                        "Configurator 触发 M3 (因果空洞 ×%d, severity=%.2f)",
                        self._consecutive_causal_gaps,
                        high_gaps[0].severity,
                    )
                    self._consecutive_causal_gaps = 0  # 重置计数器
                except Exception:
                    logger.warning("Configurator 启用 M3 失败", exc_info=True)
        else:
            self._consecutive_causal_gaps = 0  # 无因果空洞则重置

        return actions

    def _check_temporal_gap(self, gaps: list) -> list[ConfigAction]:
        """规则 2: 时序空洞检测 → 建议重训"""
        actions: list[ConfigAction] = []

        temporal_gaps = [g for g in gaps if g.gap_type == "temporal"]
        high_temporal = [g for g in temporal_gaps if g.severity >= self.RETRAIN_TEMPORAL_SEVERITY]

        for gap in high_temporal:
            actions.append(
                ConfigAction(
                    action_type="suggest_retrain",
                    reason=f"时序空洞检出 (severity={gap.severity:.2f}): {gap.description}",
                    gap_id=gap.gap_id,
                )
            )
            logger.info("Configurator 建议重训 (时序空洞 severity=%.2f)", gap.severity)

        return actions

    def _check_domain_gap(self, gaps: list) -> list[ConfigAction]:
        """规则 3: 域覆盖空洞 → 调整权重"""
        actions: list[ConfigAction] = []

        domain_gaps = [g for g in gaps if g.gap_type == "domain"]
        high_domain = [g for g in domain_gaps if g.severity >= 0.6]

        if high_domain:
            actions.append(
                ConfigAction(
                    action_type="adjust_weights",
                    reason=f"域覆盖空洞 (n={len(high_domain)}, max_severity={max(g.severity for g in high_domain):.2f})",
                    gap_id=high_domain[0].gap_id,
                )
            )
            logger.info("Configurator 建议调整权重 (域空洞 ×%d)", len(high_domain))

        return actions

    def _check_low_confidence_edges(self, world_model) -> list[ConfigAction]:
        """规则 4: 低置信度边过多 → 启用 M2 GNN 训练"""
        actions: list[ConfigAction] = []

        state = world_model._state
        if not state.causal_edges:
            return actions

        n_edges = len(state.causal_edges)
        low_conf_count = sum(
            1 for e in state.causal_edges if e.get("confidence", 1.0) < 0.5
        )
        ratio = low_conf_count / max(n_edges, 1)

        if ratio >= self.GNN_LOW_CONFIDENCE_RATIO and n_edges >= 5:
            # 检查是否已经是 GNN 模式
            if not hasattr(world_model._jepa_predictor, "training_predict") and world_model._jepa_predictor is not None:
                try:
                    world_model.enable_m3()
                    actions.append(
                        ConfigAction(
                            action_type="enable_m3",
                            reason=f"低置信度边占比 {ratio:.1%} (n={n_edges}, low={low_conf_count})",
                        )
                    )
                    logger.info("Configurator 触发 M3 (低置信度边占比 %.1%%)", ratio * 100)
                except Exception:
                    logger.warning("Configurator 触发 M3 失败 (低置信度边)", exc_info=True)
            elif world_model._jepa_predictor is None:
                actions.append(
                    ConfigAction(
                        action_type="suggest_retrain",
                        reason=f"低置信度边占比 {ratio:.1%}，但无 JEPA 预测器可用",
                    )
                )

        return actions


# =============================================================================
# v3.0.3: HierarchicalConfigurator — 分层决策配置器
# =============================================================================


class HierarchicalConfigurator:
    """
    分层决策配置器，内嵌 MetaConfigurator 并叠加协调层+自适应层。

    三层决策架构:
        L1 (单层规则): 委托 MetaConfigurator 的因果/时序/域/低置信度规则
        L2 (协调层): 多 gap 综合分析，优先级排序，冲突消解
        L3 (自适应层): 基于配置效果反馈动态调整阈值

    六态流转：IDLE → ANALYZING_L1 → COORDINATING_L2 → ADAPTING_L3 →
              CONFIGURING → MONITORING → IDLE
    """

    MIN_SEVERITY = 0.3
    DECAY_FACTOR = 0.9
    BOOST_FACTOR = 1.1

    def __init__(self):
        self._base = MetaConfigurator()
        self._state: str = "IDLE"
        self._action_history: list[dict] = []
        self._adaptive_thresholds: dict[str, float] = {
            "causal": MetaConfigurator.M3_CAUSAL_SEVERITY_THRESHOLD,
            "temporal": MetaConfigurator.RETRAIN_TEMPORAL_SEVERITY,
        }
        self._feedback_scores: list[float] = []

    @property
    def state(self) -> str:
        return self._state

    @property
    def config_history(self) -> list[dict]:
        return self._action_history[-30:]

    @property
    def adaptive_thresholds(self) -> dict:
        return dict(self._adaptive_thresholds)

    def configure(self, world_model, gaps: list | None = None) -> list[dict]:
        """分层决策：L1 单层规则 → L2 协调排序 → L3 自适应。"""
        self._state = "ANALYZING_L1"

        raw_actions = self._base.configure(world_model, gaps)
        meaningful = [a for a in raw_actions if a.action_type != "noop"]

        if not meaningful:
            self._state = "IDLE"
            return []

        # L2: 优先级排序
        self._state = "COORDINATING_L2"
        scored = self._score_actions(meaningful, gaps or [])
        resolved = self._resolve_conflicts(scored)

        # L3: 自适应阈值
        self._state = "ADAPTING_L3"
        self._adapt_thresholds(scored)

        self._state = "CONFIGURING"
        actions_dict = [
            {"type": a.action_type, "priority": p, "reason": a.reason, "gap_id": a.gap_id}
            for a, p in resolved
        ]
        self._action_history.extend(actions_dict)
        self._state = "MONITORING"
        logger.info("HierarchicalConfigurator: L1=%d → L2=%d → L3 完成", len(meaningful), len(resolved))
        return actions_dict

    def _score_actions(self, actions: list, gaps: list) -> list[tuple]:
        """多维度评分排序：severity × impact - frequency_penalty。"""
        scored: list[tuple] = []
        impact_map = {"enable_m3": 3.0, "suggest_retrain": 2.0, "adjust_weights": 1.0}

        for action in actions:
            score = 0.0
            if action.gap_id:
                for g in gaps:
                    if g.gap_id == action.gap_id:
                        score += g.severity * 5.0
                        break
            score += impact_map.get(action.action_type, 1.0)

            freq = sum(1 for h in self._action_history[-10:] if h.get("type") == action.action_type)
            if freq >= 3:
                score *= 0.5

            scored.append((action, round(score, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _resolve_conflicts(self, scored: list[tuple]) -> list[tuple]:
        """冲突消解：enable_m3 已涵盖 suggest_retrain。"""
        if len(scored) <= 1:
            return scored
        resolved: list[tuple] = []
        has_m3 = False
        for action, score in scored:
            if action.action_type == "enable_m3":
                if not has_m3:
                    resolved.append((action, score))
                    has_m3 = True
            elif action.action_type == "suggest_retrain":
                if not has_m3:
                    resolved.append((action, score))
            else:
                resolved.append((action, score))
        return resolved

    def _adapt_thresholds(self, scored: list[tuple]) -> None:
        """自适应阈值：高频触发→BOOST，长期无触发→DECAY。"""
        if not scored:
            return
        action_types: dict[str, int] = {}
        for h in self._action_history[-10:]:
            t = h.get("type", "")
            action_types[t] = action_types.get(t, 0) + 1

        for atype, count in action_types.items():
            if count >= 3 and atype in ("enable_m3", "suggest_retrain"):
                self._adaptive_thresholds["causal"] = min(
                    self._adaptive_thresholds["causal"] * self.BOOST_FACTOR, 0.9
                )

        if len(self._action_history) >= 5:
            recent = sum(1 for h in self._action_history[-5:] if h.get("type") != "noop")
            if recent == 0:
                for k in ("causal", "temporal"):
                    self._adaptive_thresholds[k] = max(
                        self._adaptive_thresholds[k] * self.DECAY_FACTOR, 0.3
                    )

    def feedback(self, was_effective: bool) -> None:
        """外部反馈：上一次配置是否有效。"""
        self._feedback_scores.append(1.0 if was_effective else 0.0)
        if len(self._feedback_scores) > 10:
            self._feedback_scores.pop(0)
