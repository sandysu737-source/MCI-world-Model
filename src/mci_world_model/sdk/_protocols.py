"""MCI World Model v4.4.0 — 协议定义层
=========================================

CEWM 架构泛化的基石——用 Protocol 定义预测器和状态解析器的接口契约，
将 Pendulum 从"唯一实现"降级为"一个实现"。

核心协议:
    PredictorProtocol   — 预测器接口（任意物理/学习预测器可插入）
    StateParserProtocol — 状态解析器接口（任意观测类型可转换为 WorldState）

设计原则:
    - 协议优于实现: 通过 Protocol/ABC 定义接口契约
    - 结构化子类型: @runtime_checkable + isinstance 检查，无需显式继承
    - 零回归: 所有现有类自动满足协议，无需修改代码即可通过 isinstance 检查
    - 渐进式解耦: 每次替换一条硬编码路径，验证后再推进
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from mci_world_model.sdk._world_state import Action, WorldState


# =============================================================================
# PredictorProtocol — 预测器接口契约
# =============================================================================


@runtime_checkable
class PredictorProtocol(Protocol):
    """世界模型预测器接口契约。

    任何预测器实现此协议即可插入 PlanAgent / MultiBranchPredictor / cewm_step()。

    协议方法:
        name:       预测器名称标识
        predict():  动作条件化多步预测
        evaluate(): 在测试数据集上评估预测精度

    现有实现:
        - PendulumPhysicsPredictor  ✅ 自动满足
        - PendulumJEPAPredictor     ✅ 自动满足
        - CartPhysicsPredictor      (Phase 0 新增)
        - GeneralizedPhysicsPredictor (Phase 1 新增)

    验证:
        >>> from mci_world_model.sdk._action_conditioned_predictor import PendulumPhysicsPredictor
        >>> isinstance(PendulumPhysicsPredictor(), PredictorProtocol)
        True
    """

    @property
    def name(self) -> str:
        """预测器名称标识。"""
        ...

    def predict(
        self,
        state: WorldState,
        action: Action | None,
        n_steps: int = 1,
    ) -> list[WorldState]:
        """动作条件化多步预测。

        Args:
            state: 当前世界状态
            action: 施加的动作 (None = 无外力自然演化)
            n_steps: 预测步数

        Returns:
            预测的未来状态序列
        """
        ...

    def evaluate(
        self,
        dataset: list,
    ) -> dict:
        """在测试数据集上评估预测精度。

        Args:
            dataset: [(state, action, ground_truth), ...] 列表

        Returns:
            评估统计字典
        """
        ...


# =============================================================================
# StateParserProtocol — 状态解析器接口契约
# =============================================================================


@runtime_checkable
class StateParserProtocol(Protocol):
    """状态解析器接口契约。

    将任意观测类型转换为 WorldState 实例。
    用于 _cewm_parse_state() 的泛化——不再硬编码 PendulumState 解析逻辑。

    协议方法:
        can_parse(): 能否解析给定观测
        parse():     将观测转换为 WorldState

    现有实现:
        - PendulumStateParser  (Phase 0: 解析 PendulumState/dict 含 theta/omega)
        - CartStateParser      (Phase 0: 解析 CartState/dict 含 x/v)
        - GenericStateParser   (Phase 0: 通用 WorldState 解析)

    验证:
        >>> parser = PendulumStateParser()
        >>> isinstance(parser, StateParserProtocol)
        True
    """

    def can_parse(self, obs: object) -> bool:
        """判断是否能解析给定观测。

        Args:
            obs: 待解析的观测对象

        Returns:
            True 如果此解析器可以处理该观测类型
        """
        ...

    def parse(self, obs: object) -> WorldState:
        """将观测转换为 WorldState。

        Args:
            obs: 待解析的观测对象

        Returns:
            解析后的 WorldState 实例

        Raises:
            TypeError: 如果观测类型不受支持
        """
        ...


# =============================================================================
# 内置解析器实现
# =============================================================================


class PendulumStateParser:
    """单摆状态解析器——解析 PendulumState 和含 theta/omega 的 dict。"""

    def can_parse(self, obs: object) -> bool:
        """检查观测是否为 PendulumState 或含 theta 的 dict。"""
        if obs is None:
            return False
        if hasattr(obs, "theta") and hasattr(obs, "omega"):
            return True
        if isinstance(obs, dict) and "theta" in obs:
            return True
        return False

    def parse(self, obs: object) -> WorldState:
        """解析为 PendulumState。"""
        from mci_world_model.sdk._world_state import PendulumState

        if hasattr(obs, "theta") and hasattr(obs, "omega"):
            if isinstance(obs, PendulumState):
                return obs
            # 其他含 theta/omega 属性的对象，构建 PendulumState
            return PendulumState(
                theta=float(getattr(obs, "theta", 0.0)),
                omega=float(getattr(obs, "omega", 0.0)),
            )
        if isinstance(obs, dict) and "theta" in obs:
            return PendulumState(
                theta=float(obs.get("theta", 0.0)),
                omega=float(obs.get("omega", 0.0)),
            )
        raise TypeError(f"PendulumStateParser 无法解析 {type(obs).__name__}")


class CartStateParser:
    """小车状态解析器——解析 CartState 和含 x/v 的 dict。"""

    def can_parse(self, obs: object) -> bool:
        """检查观测是否为 CartState 或含 x+v 的 dict。"""
        if obs is None:
            return False
        if hasattr(obs, "x") and hasattr(obs, "v") and not hasattr(obs, "theta"):
            return True
        if isinstance(obs, dict) and "x" in obs and "v" in obs and "theta" not in obs:
            return True
        return False

    def parse(self, obs: object) -> WorldState:
        """解析为 CartState。"""
        from mci_world_model.sdk._world_state import CartState

        if hasattr(obs, "x") and hasattr(obs, "v"):
            if isinstance(obs, CartState):
                return obs
            return CartState(
                x=float(getattr(obs, "x", 0.0)),
                v=float(getattr(obs, "v", 0.0)),
            )
        if isinstance(obs, dict) and "x" in obs:
            return CartState(
                x=float(obs.get("x", 0.0)),
                v=float(obs.get("v", 0.0)),
            )
        raise TypeError(f"CartStateParser 无法解析 {type(obs).__name__}")


class GenericStateParser:
    """通用状态解析器——解析任意 WorldState 对象。"""

    def can_parse(self, obs: object) -> bool:
        """检查观测是否已是 WorldState 或有 to_vector 方法。"""
        if obs is None:
            return False
        if hasattr(obs, "to_vector") and hasattr(obs, "distance"):
            return True
        return False

    def parse(self, obs: object) -> WorldState:
        """直接返回 WorldState 对象。"""
        if hasattr(obs, "to_vector") and hasattr(obs, "distance"):
            return obs  # type: ignore[return-value]
        raise TypeError(f"GenericStateParser 无法解析 {type(obs).__name__}")


# =============================================================================
# StateParserRegistry — 解析器注册表
# =============================================================================


class StateParserRegistry:
    """状态解析器注册表——按优先级尝试多个解析器。

    用法:
        >>> registry = StateParserRegistry.default()
        >>> state = registry.parse(observation)
    """

    def __init__(self) -> None:
        self._parsers: list[StateParserProtocol] = []

    def register(self, parser: StateParserProtocol) -> None:
        """注册一个解析器（后注册的优先级更高）。"""
        self._parsers.append(parser)

    def parse(self, obs: object) -> WorldState | None:
        """尝试用所有注册的解析器解析观测。

        按注册顺序的逆序尝试（后注册的优先），返回第一个成功的解析结果。
        如果所有解析器都无法处理，返回 None。
        """
        # 逆序尝试：后注册的优先级更高
        for parser in reversed(self._parsers):
            if parser.can_parse(obs):
                try:
                    return parser.parse(obs)
                except (TypeError, ValueError, KeyError):
                    continue
        return None

    @classmethod
    def default(cls) -> StateParserRegistry:
        """创建默认解析器注册表（Pendulum → Cart → Generic 优先级）。"""
        registry = cls()
        registry.register(GenericStateParser())
        registry.register(CartStateParser())
        registry.register(PendulumStateParser())
        return registry
