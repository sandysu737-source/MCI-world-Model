from __future__ import annotations

"""MCI World Model — MCIPluginInterface 插件接口
================================================

可扩展插件体系——第三方可通过实现 PluginInterface ABC
将自定义能力注入 MCI World Model 的推理流水线。

核心能力:
    PluginHook          — 插件钩子枚举
    PluginMetadata      — 插件元信息
    PluginInterface     — 插件抽象基类
    PluginManager       — 插件管理器(注册+调度+生命周期)

设计原则:
    - 钩子驱动: on_load / on_query / on_result / on_error 四阶段
    - 热插拔: 运行时动态加载/卸载插件
    - 隔离: 插件异常不扩散到核心引擎
    - 优先级: 插件按优先级排序执行
"""


import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# PluginHook — 插件钩子枚举
# =============================================================================


class PluginHook(Enum):
    """插件钩子枚举——定义插件可挂载的执行点。"""

    ON_LOAD = "on_load"  # 插件加载时
    ON_QUERY = "on_query"  # 查询到达时
    ON_RESULT = "on_result"  # 结果返回前
    ON_ERROR = "on_error"  # 异常发生时


# =============================================================================
# PluginMetadata — 插件元信息
# =============================================================================


@dataclass
class PluginMetadata:
    """插件元信息。

    Attributes:
        name: 插件名称(唯一标识)
        version: 版本号
        description: 插件描述
        author: 作者
        hooks: 支持的钩子列表
        priority: 优先级(0最高, 数字越大越后执行)
        tags: 标签列表
    """

    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    hooks: list[PluginHook] = field(default_factory=list)
    priority: int = 100
    tags: list[str] = field(default_factory=list)


# =============================================================================
# PluginContext — 插件执行上下文
# =============================================================================


@dataclass
class PluginContext:
    """插件执行上下文。

    Attributes:
        query: 原始查询
        result: 推理结果(ON_RESULT 阶段可用)
        error: 异常信息(ON_ERROR 阶段可用)
        metadata: 附加元数据
        timestamp: 执行时间戳
    """

    query: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    error: Exception | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# PluginInterface — 插件抽象基类
# =============================================================================


class PluginInterface(ABC):
    """MCI 插件接口抽象基类。

    第三方通过实现此 ABC 将自定义能力注入推理流水线。

    最小实现:
        - metadata: 属性, 返回 PluginMetadata
        - on_load(ctx): 插件加载初始化
        - on_query(ctx): 查询预处理
        - on_result(ctx): 结果后处理
        - on_error(ctx): 异常处理

    用法:
        >>> class MyPlugin(PluginInterface):
        ...     @property
        ...     def metadata(self):
        ...         return PluginMetadata(name="my_plugin", hooks=[PluginHook.ON_QUERY])
        ...     def on_query(self, ctx):
        ...         ctx.query["enhanced"] = True
        ...         return ctx
    """

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """插件元信息。"""
        ...

    def on_load(self, ctx: PluginContext) -> PluginContext:
        """插件加载钩子——在插件注册到管理器时调用。

        Args:
            ctx: 加载上下文

        Returns:
            修改后的上下文
        """
        logger.info("插件 %s 已加载", self.metadata.name)
        return ctx

    def on_query(self, ctx: PluginContext) -> PluginContext:
        """查询预处理钩子——在推理执行前调用。

        Args:
            ctx: 查询上下文

        Returns:
            修改后的上下文
        """
        return ctx

    def on_result(self, ctx: PluginContext) -> PluginContext:
        """结果后处理钩子——在推理完成后、返回前调用。

        Args:
            ctx: 结果上下文

        Returns:
            修改后的上下文
        """
        return ctx

    def on_error(self, ctx: PluginContext) -> PluginContext:
        """异常处理钩子——推理异常时调用。

        Args:
            ctx: 异常上下文

        Returns:
            修改后的上下文
        """
        return ctx

    def on_unload(self) -> None:
        """插件卸载钩子——在插件从管理器移除时调用。"""
        logger.info("插件 %s 已卸载", self.metadata.name)


# =============================================================================
# PluginManager — 插件管理器
# =============================================================================


class PluginManager:
    """插件管理器——注册+调度+生命周期管理。

    用法:
        >>> manager = PluginManager()
        >>> manager.register(my_plugin)
        >>> ctx = manager.execute_hook(PluginHook.ON_QUERY, query={"q": "test"})
        >>> manager.unregister("my_plugin")
    """

    def __init__(self) -> None:
        self._plugins: dict[str, PluginInterface] = {}
        self._load_order: list[str] = []
        self._hook_count: dict[PluginHook, int] = dict.fromkeys(PluginHook, 0)
        self._error_count: int = 0

    @property
    def plugin_count(self) -> int:
        return len(self._plugins)

    @property
    def plugin_names(self) -> list[str]:
        return list(self._load_order)

    def register(self, plugin: PluginInterface) -> None:
        """注册插件。

        Args:
            plugin: 插件实例

        Raises:
            ValueError: 插件名已存在
        """
        name = plugin.metadata.name
        if name in self._plugins:
            raise ValueError(f"插件 '{name}' 已注册, 请先卸载或使用不同名称")

        self._plugins[name] = plugin
        # 按优先级插入排序
        self._load_order.append(name)
        self._load_order.sort(key=lambda n: self._plugins[n].metadata.priority)

        # 触发 on_load
        ctx = PluginContext(metadata={"event": "load"})
        try:
            plugin.on_load(ctx)
        except Exception as e:
            logger.warning("插件 %s on_load 异常: %s", name, e)
            self._error_count += 1

        logger.info(
            "插件管理器: 注册 %s (优先级=%d, 钩子=%s)",
            name,
            plugin.metadata.priority,
            [h.value for h in plugin.metadata.hooks],
        )

    def unregister(self, name: str) -> None:
        """卸载插件。

        Args:
            name: 插件名称

        Raises:
            KeyError: 插件不存在
        """
        if name not in self._plugins:
            raise KeyError(f"插件 '{name}' 不存在")

        plugin = self._plugins.pop(name)
        self._load_order.remove(name)

        try:
            plugin.on_unload()
        except Exception as e:
            logger.warning("插件 %s on_unload 异常: %s", name, e)
            self._error_count += 1

        logger.info("插件管理器: 卸载 %s", name)

    def get_plugin(self, name: str) -> PluginInterface | None:
        """获取插件实例。"""
        return self._plugins.get(name)

    def execute_hook(
        self,
        hook: PluginHook,
        query: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> PluginContext:
        """执行指定钩子的所有插件。

        按优先级顺序依次执行, 前一插件的输出作为后一插件的输入。
        插件异常被隔离, 不影响后续插件和核心引擎。

        Args:
            hook: 钩子类型
            query: 查询数据
            result: 结果数据
            error: 异常信息

        Returns:
            最终上下文
        """
        self._hook_count[hook] += 1

        ctx = PluginContext(
            query=query or {},
            result=result or {},
            error=error,
        )

        for name in self._load_order:
            plugin = self._plugins[name]
            if hook not in plugin.metadata.hooks:
                continue

            try:
                if hook == PluginHook.ON_LOAD:
                    ctx = plugin.on_load(ctx)
                elif hook == PluginHook.ON_QUERY:
                    ctx = plugin.on_query(ctx)
                elif hook == PluginHook.ON_RESULT:
                    ctx = plugin.on_result(ctx)
                elif hook == PluginHook.ON_ERROR:
                    ctx = plugin.on_error(ctx)
            except Exception as e:
                logger.warning("插件 %s 钩子 %s 执行异常: %s", name, hook.value, e)
                self._error_count += 1

        return ctx

    def list_plugins(self) -> list[dict[str, Any]]:
        """列出所有已注册插件的信息。"""
        result = []
        for name in self._load_order:
            plugin = self._plugins[name]
            meta = plugin.metadata
            result.append(
                {
                    "name": meta.name,
                    "version": meta.version,
                    "description": meta.description,
                    "priority": meta.priority,
                    "hooks": [h.value for h in meta.hooks],
                    "tags": meta.tags,
                }
            )
        return result

    def statistics(self) -> dict[str, Any]:
        """插件管理器统计。"""
        return {
            "plugin_count": self.plugin_count,
            "plugins": self.plugin_names,
            "hook_executions": {h.value: c for h, c in self._hook_count.items()},
            "error_count": self._error_count,
        }
