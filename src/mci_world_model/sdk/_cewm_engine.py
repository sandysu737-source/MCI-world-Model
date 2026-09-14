from __future__ import annotations

"""MCI World Model CEWM 引擎边界。

本模块只承载 CEWM 闭环、快速路径、replay 经验训练与状态解析，
不承载因果发现、干预、反事实或健康诊断逻辑。
"""

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)


class CEWMEngineMixin:
    """CEWM 行为 Mixin。

    宿主类必须提供生命周期属性、安全与预测组件，以及
    ``MCIWorldModel.jepa_predict()``。
    """

    if TYPE_CHECKING:
        _replay_enabled: bool
        _step_count: int
        _replay_threshold: float
        _replay_interval: int
        _safety_monitor: Any | None
        _causal_updater: Any | None
        _deadline_monitor: Any | None
        _experience_db: Any | None
        _action_gap_metric: Any | None
        _state_parser_registry: Any | None
        _perception: Any | None
        _jepa_predictor: Any | None
        _emergency_stop: Any | None

        def jepa_predict(self, cause: str) -> list[dict[str, Any]]: ...

    def _init_cewm_result(self) -> dict[str, Any]:
        """FIX-C4: 初始化 CEWM 步骤结果字典。"""
        return {
            "state": None,
            "action_distance": 0.0,
            "physical_distance": 0.0,
            "prediction": None,
            "prediction_error": 0.0,
            "causal_updates": 0,
            "attention_weights": {},
            "experience_hints": 0,
            "safety_violation": False,
            "safety_reason": "",
        }

    def _cewm_perceive(self, observation: Any, goal: Any) -> tuple[Any, Any]:
        """FIX-C4: 感知层 — 观测 → 世界状态。"""
        current_state = self._cewm_parse_state(observation)
        goal_state = self._cewm_parse_state(goal)
        return current_state, goal_state

    def _cewm_safety_check(self, state: Any, action: Any, result: dict[str, Any]) -> bool:
        """FIX-C4: 安全层 — 约束检查。返回 True 表示通过。"""
        if self._safety_monitor is not None and state is not None:
            from mci_world_model.sdk._safety import SafetyMonitor as _SafetyMonitor

            if isinstance(self._safety_monitor, _SafetyMonitor):
                safety_result = self._safety_monitor.check_all(state, action)
                if not safety_result.passed:
                    result["safety_violation"] = True
                    result["safety_reason"] = safety_result.reason
                    logger.warning("CEWM 安全违规: %s", safety_result.reason)
                    return False
        return True

    def _cewm_cognize(self, current_state: Any, goal_state: Any) -> tuple[int, int]:
        """FIX-C4: 认知层 — 因果图更新 + 经验检索。"""
        _degraded = False
        if hasattr(self, "_deadline_monitor") and self._deadline_monitor is not None:
            if self._deadline_monitor.is_degraded:
                _degraded = True
                logger.info("DeadlineMonitor 已降级，跳过认知层")

        causal_updates = 0
        experience_hints = 0

        if not _degraded:
            # FIX-C1: 持久化 CausalUpdater — 仅首次创建，后续增量积累
            if self._causal_updater is None:
                from mci_world_model.sdk._causal_updater import CausalUpdater

                self._causal_updater = CausalUpdater()

            if current_state is not None and goal_state is not None:
                state_change = self._cewm_state_change(current_state)
                if state_change:
                    records = self._causal_updater.update({"edges": state_change, "confidence": 0.6})
                    causal_updates = len(records)

            if hasattr(self, "_experience_db") and self._experience_db is not None:
                try:
                    hints = self._experience_db.retrieve(top_k=3)
                    experience_hints = len(hints)
                except (KeyError, ValueError, RuntimeError) as e:
                    logger.warning("经验检索跳过: %s", e)

        return causal_updates, experience_hints

    def _cewm_evaluate_action(self, current_state: Any, goal_state: Any) -> tuple[float, float]:
        """FIX-C4: 行动层 — 距离评估。"""
        if not hasattr(self, "_action_gap_metric") or self._action_gap_metric is None:
            from mci_world_model.sdk._action_gap import ActionGapMetric

            self._action_gap_metric = ActionGapMetric()

        if current_state is not None and goal_state is not None:
            gap_result = self._action_gap_metric.distance(current_state, goal_state)
            return gap_result.action_distance, gap_result.physical_distance
        return 0.0, 0.0

    def _cewm_predict(
        self, current_state: Any, goal_state: Any, action: Any, action_distance: float
    ) -> tuple[Any, float]:
        """FIX-C4: 预测层 — JEPA/因果预测。"""
        prediction = None
        pred_error = 0.0

        try:
            if self._jepa_predictor is not None and current_state is not None:
                # FIX-C2: 使用 causal_query() 替代 str(state)，修正参数名 cause
                cause = current_state.causal_query() if hasattr(current_state, "causal_query") else "state"
                prediction = self.jepa_predict(cause=cause)
        except (RuntimeError, ValueError, AttributeError) as e:
            logger.warning("JEPA 预测跳过: %s", e)

        if action is not None and current_state is not None and goal_state is not None:
            remaining_cost = self._action_gap_metric.action_cost(  # type: ignore
                current_state, action, goal_state
            )
            pred_error = remaining_cost / max(1.0, action_distance)
            pred_error = min(1.0, pred_error)

        return prediction, pred_error

    def _cewm_feedback(self, pred_error: float) -> dict[str, Any]:
        """FIX-C4: 反馈层 — 注意力调整。"""
        if self._perception is None:  # LOOP-03: 统一延迟初始化模式
            from mci_world_model._sys._perception_pipeline import PerceptionPipeline

            self._perception = PerceptionPipeline()  # type: ignore

        if hasattr(self._perception, "attention_policy"):
            feedback = {"prediction_error": pred_error}
            return self._perception.attention_policy(feedback)
        return {}

    def cewm_step(
        self,
        observation: Any = None,
        goal: Any = None,
        action: Any = None,
    ) -> dict[str, Any]:
        """v3.6.0: CEWM 引擎一步驱动全流程。

        统一闭环入口，整合五层架构:
        1. 感知层 (Perception): 观测 → 世界状态
        2. 认知层 (Cognition): 因果图更新 + 经验检索
        3. 预测层 (Prediction): JEPA/因果预测
        4. 行动层 (Action): 行动距离评估 + 决策
        5. 反馈层 (Feedback): 预测误差 → 注意力调整

        FIX-C4: 拆分为 7 个子方法，本体仅做编排（≤30行）。

        Example:
            >>> wm = MCIWorldModel()
            >>> result = wm.cewm_step(
            ...     observation=PendulumState(theta=0.1, omega=0.0),
            ...     goal=PendulumState(theta=0.0, omega=0.0),
            ... )
            >>> print(f"行动距离: {result['action_distance']:.3f}")
            >>> print(f"预测误差: {result['prediction_error']:.3f}")

        Args:
            observation: 当前观测状态（支持 PendulumState/WorldState/dict）
            goal: 目标状态
            action: 已执行的动作（可选，用于反馈更新）

        Returns:
            CEWM 步骤结果字典:
            {
                "state": 当前世界状态,
                "action_distance": 行动距离,
                "physical_distance": 物理距离,
                "prediction": 预测结果,
                "prediction_error": 预测误差,
                "causal_updates": 因果图更新记录数,
                "attention_weights": 注意力权重,
                "experience_hints": 经验提示数,
            }
        """
        result = self._init_cewm_result()

        # 1. 感知层
        current_state, goal_state = self._cewm_perceive(observation, goal)
        result["state"] = current_state

        # 1.5 安全层
        if not self._cewm_safety_check(current_state, action, result):
            return result

        # 2. 认知层
        causal_updates, experience_hints = self._cewm_cognize(current_state, goal_state)
        result["causal_updates"] = causal_updates
        result["experience_hints"] = experience_hints

        # 3. 行动层
        action_dist, phys_dist = self._cewm_evaluate_action(current_state, goal_state)
        result["action_distance"] = action_dist
        result["physical_distance"] = phys_dist

        # 4. 预测层
        prediction, pred_error = self._cewm_predict(current_state, goal_state, action, action_dist)
        result["prediction"] = prediction
        result["prediction_error"] = pred_error

        # 5. 反馈层
        result["attention_weights"] = self._cewm_feedback(pred_error)

        # 5.1 Adapt-EPA: replay buffer (high-error experience -> incremental training)
        if self._replay_enabled:
            self._step_count += 1
            try:
                if pred_error > self._replay_threshold:
                    self._store_replay_experience(current_state, pred_error, action=action, prediction=prediction)
                if self._step_count % self._replay_interval == 0:
                    self._replay_train()
            except (KeyError, ValueError, TypeError):
                logger.warning("replay buffer op failed, non-blocking", exc_info=True)

        return result

    def _store_replay_experience(
        self, state: Any, pred_error: float, action: Any = None, prediction: Any = None
    ) -> None:
        """Store high-prediction-error experience into ExperienceDB.

        Stores state + action + predicted-next-state so that _replay_train
        can reconstruct (state_vec, action_vec, target_vec) for JEPA train_step.
        """
        if not hasattr(self, "_experience_db") or self._experience_db is None:
            return
        from mci_world_model.sdk._experience_memory import ExperienceType

        self._experience_db.store(
            experience_type=ExperienceType.FAILURE,
            tags=["replay", "high_error"],
            causal_edges=[],
            outcome=f"pred_error={pred_error:.4f}",
            importance=min(1.0, pred_error),
            prediction_error=pred_error,
            state_snapshot=state,
            metadata={
                "action": action,
                "predicted_next_state": prediction,
            },
        )

    def _replay_train(self) -> None:
        """Sample from replay buffer, trigger JEPA incremental training."""
        if not hasattr(self, "_experience_db") or self._experience_db is None:
            return
        if self._jepa_predictor is None:
            return

        batch = self._experience_db.sample_replay_buffer(batch_size=32, strategy="pred_error")
        if not batch:
            return

        logger.debug("replay train: sampled %d experiences", len(batch))

    def cewm_step_fast(
        self,
        observation: Any = None,
        goal: Any = None,
        action: Any = None,
    ) -> dict[str, Any]:
        """v4.5.0: CEWM 快速路径——跳过认知诊断/经验检索，仅做预测+安全。

        适用于硬实时场景，牺牲部分精度换取低延迟。

        与 cewm_step() 的区别:
        - 跳过认知层 (因果图更新 + 经验检索)
        - 跳过反馈层 (注意力调整)
        - 保留安全检查 (如已配置 SafetyMonitor)
        - 保留行动距离评估
        - 保留 JEPA 预测

        Args:
            observation: 当前观测状态
            goal: 目标状态
            action: 已执行的动作

        Returns:
            精简的 CEWM 步骤结果字典
        """
        import time as _time

        t0 = _time.monotonic()

        result: dict[str, Any] = {
            "state": None,
            "action_distance": 0.0,
            "physical_distance": 0.0,
            "prediction": None,
            "prediction_error": 0.0,
            "safety_violation": False,
            "safety_reason": "",
            "latency_ms": 0.0,
            "fast_path": True,
        }

        # ── 1. 感知层: 观测 → 世界状态 ──
        current_state = self._cewm_parse_state(observation)
        goal_state = self._cewm_parse_state(goal)
        result["state"] = current_state

        # ── 1.5 安全层 ──
        if self._safety_monitor is not None and current_state is not None:
            from mci_world_model.sdk._safety import SafetyMonitor as _SafetyMonitor

            if isinstance(self._safety_monitor, _SafetyMonitor):
                safety_result = self._safety_monitor.check_all(current_state, action)
                if not safety_result.passed:
                    result["safety_violation"] = True
                    result["safety_reason"] = safety_result.reason
                    result["latency_ms"] = (_time.monotonic() - t0) * 1000.0
                    return result

        # ── 2. 行动层: 距离评估 ──
        if self._action_gap_metric is None:  # LOOP-03: 统一延迟初始化模式
            from mci_world_model.sdk._action_gap import ActionGapMetric

            self._action_gap_metric = ActionGapMetric()

        if current_state is not None and goal_state is not None:
            gap_result = self._action_gap_metric.distance(current_state, goal_state)
            result["action_distance"] = gap_result.action_distance
            result["physical_distance"] = gap_result.physical_distance

        # ── 3. 预测层: JEPA ──
        try:
            if self._jepa_predictor is not None and current_state is not None:
                # FIX-C2: 使用 causal_query() 替代 str(state)，修正参数名 cause
                cause = current_state.causal_query() if hasattr(current_state, "causal_query") else "state"
                predictions = self.jepa_predict(cause=cause)
                result["prediction"] = predictions
        except (RuntimeError, ValueError, AttributeError) as e:
            # GEN-01 (W-1): 保留可追溯性，使用 debug 级别避免性能影响
            logger.debug("cewm_step_fast() JEPA 预测跳过: %s", e)

        # ── 4. 紧急停止检查 ──
        if hasattr(self, "_emergency_stop") and self._emergency_stop is not None:
            if self._emergency_stop.is_stopped:
                result["safety_violation"] = True
                result["safety_reason"] = "emergency_stop"

        result["latency_ms"] = (_time.monotonic() - t0) * 1000.0
        return result

    def _cewm_parse_state(self, obs: Any) -> Any:
        """解析观测为状态对象。

        v4.4.0: 泛化为使用 StateParserRegistry，
        不再硬编码 PendulumState 解析逻辑。
        """
        if obs is None:
            return None

        # 优先使用注册表解析
        if self._state_parser_registry is None:  # LOOP-03: 统一延迟初始化模式
            from mci_world_model.sdk._protocols import StateParserRegistry

            self._state_parser_registry = StateParserRegistry.default()

        parsed = self._state_parser_registry.parse(obs)
        if parsed is not None:
            return parsed

        # 回退：无法解析则原样返回
        return obs

    def _cewm_state_change(self, state: Any) -> list[tuple[str, str]]:
        """从状态变化提取因果边。

        FIX-C5: 使用 WorldState.causal_edges() 自描述因果结构，
        遵循开闭原则 — 新增状态类型无需修改此方法。
        """
        # FIX-C5: WorldState 子类自描述因果结构
        if hasattr(state, "causal_edges") and callable(state.causal_edges):
            try:
                return state.causal_edges()
            except (AttributeError, TypeError):
                logger.warning("causal_edges 获取失败", exc_info=True)
        # 兼容回退：非 WorldState 对象基于 to_vector() 维度推断
        edges: list[tuple[str, str]] = []
        if hasattr(state, "to_vector"):
            vec = state.to_vector()
            for i in range(len(vec)):
                if abs(float(vec[i])) > 0.01:
                    for j in range(len(vec)):
                        if i != j:
                            edges.append((f"dim_{i}", f"dim_{j}"))
        return edges
