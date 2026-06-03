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
