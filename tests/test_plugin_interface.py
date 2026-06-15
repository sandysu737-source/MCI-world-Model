"""
tests/test_plugin_interface.py — MCIPluginInterface 测试
=======================================================

覆盖:
    - PluginInterface: on_load/on_query/on_result/on_error 钩子
    - PluginManager: 注册/卸载/调度/隔离
    - 优先级排序/异常隔离/热插拔
"""

from __future__ import annotations

import pytest

from mci_world_model.sdk._plugin_interface import (
    PluginContext,
    PluginHook,
    PluginInterface,
    PluginManager,
    PluginMetadata,
)

# =============================================================================
# 测试用插件
# =============================================================================


class LoggingPlugin(PluginInterface):
    """日志插件——记录每次钩子调用。"""

    def __init__(self):
        self.call_log = []

    @property
    def metadata(self):
        return PluginMetadata(
            name="logging",
            version="1.0.0",
            description="日志记录插件",
            hooks=[PluginHook.ON_QUERY, PluginHook.ON_RESULT],
            priority=10,
        )

    def on_query(self, ctx):
        self.call_log.append(("on_query", ctx.query))
        return ctx

    def on_result(self, ctx):
        self.call_log.append(("on_result", ctx.result))
        return ctx


class EnhancePlugin(PluginInterface):
    """增强插件——为查询添加增强字段。"""

    @property
    def metadata(self):
        return PluginMetadata(
            name="enhance",
            version="2.0.0",
            description="查询增强插件",
            hooks=[PluginHook.ON_QUERY],
            priority=5,  # 比日志插件更早执行
        )

    def on_query(self, ctx):
        ctx.query["enhanced"] = True
        return ctx


class ErrorPlugin(PluginInterface):
    """异常插件——在钩子中抛出异常。"""

    @property
    def metadata(self):
        return PluginMetadata(
            name="error_prone",
            hooks=[PluginHook.ON_QUERY],
            priority=50,
        )

    def on_query(self, ctx):
        raise RuntimeError("插件故意报错")


class ErrorHandlingPlugin(PluginInterface):
    """异常处理插件——在 ON_ERROR 钩子中记录异常。"""

    def __init__(self):
        self.errors = []

    @property
    def metadata(self):
        return PluginMetadata(
            name="error_handler",
            hooks=[PluginHook.ON_ERROR],
            priority=1,
        )

    def on_error(self, ctx):
        self.errors.append(ctx.error)
        return ctx


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def manager():
    return PluginManager()


@pytest.fixture
def logging_plugin():
    return LoggingPlugin()


@pytest.fixture
def enhance_plugin():
    return EnhancePlugin()


# =============================================================================
# TestPluginMetadata
# =============================================================================


class TestPluginMetadata:
    """PluginMetadata 数据类。"""

    def test_creation(self):
        meta = PluginMetadata(name="test", version="1.0.0")
        assert meta.name == "test"
        assert meta.version == "1.0.0"
        assert meta.priority == 100  # default

    def test_with_hooks(self):
        meta = PluginMetadata(
            name="test",
            hooks=[PluginHook.ON_QUERY, PluginHook.ON_RESULT],
        )
        assert len(meta.hooks) == 2


# =============================================================================
# TestPluginContext
# =============================================================================


class TestPluginContext:
    """PluginContext 数据类。"""

    def test_creation(self):
        ctx = PluginContext(query={"q": "test"}, result={"a": 1})
        assert ctx.query["q"] == "test"
        assert ctx.result["a"] == 1
        assert ctx.error is None

    def test_with_error(self):
        err = RuntimeError("test error")
        ctx = PluginContext(error=err)
        assert ctx.error is not None


# =============================================================================
# TestPluginInterface
# =============================================================================


class TestPluginInterface:
    """PluginInterface 基类测试。"""

    def test_abstract(self):
        """不能直接实例化抽象类。"""
        with pytest.raises(TypeError):
            PluginInterface()

    def test_logging_plugin(self, logging_plugin):
        assert logging_plugin.metadata.name == "logging"
        ctx = PluginContext(query={"q": "test"})
        _result = logging_plugin.on_query(ctx)
        assert len(logging_plugin.call_log) == 1


# =============================================================================
# TestPluginManager
# =============================================================================


class TestPluginManager:
    """PluginManager 测试。"""

    def test_register(self, manager, logging_plugin):
        manager.register(logging_plugin)
        assert manager.plugin_count == 1
        assert "logging" in manager.plugin_names

    def test_register_duplicate(self, manager, logging_plugin):
        """重复注册同名插件应报错。"""
        manager.register(logging_plugin)
        with pytest.raises(ValueError, match="已注册"):
            manager.register(logging_plugin)

    def test_unregister(self, manager, logging_plugin):
        manager.register(logging_plugin)
        manager.unregister("logging")
        assert manager.plugin_count == 0

    def test_unregister_nonexistent(self, manager):
        """卸载不存在的插件应报错。"""
        with pytest.raises(KeyError, match="不存在"):
            manager.unregister("nonexistent")

    def test_get_plugin(self, manager, logging_plugin):
        manager.register(logging_plugin)
        plugin = manager.get_plugin("logging")
        assert plugin is logging_plugin

    def test_get_nonexistent_plugin(self, manager):
        assert manager.get_plugin("nonexistent") is None

    def test_priority_order(self, manager, logging_plugin, enhance_plugin):
        """高优先级(数字小)先执行。"""
        manager.register(logging_plugin)  # priority=10
        manager.register(enhance_plugin)  # priority=5
        # enhance 应在 logging 之前
        assert manager.plugin_names[0] == "enhance"
        assert manager.plugin_names[1] == "logging"

    def test_execute_on_query(self, manager, logging_plugin):
        manager.register(logging_plugin)
        ctx = manager.execute_hook(PluginHook.ON_QUERY, query={"q": "test"})
        assert len(logging_plugin.call_log) == 1
        assert ctx.query["q"] == "test"

    def test_execute_chaining(self, manager, logging_plugin, enhance_plugin):
        """钩子链: enhance 修改查询 → logging 收到修改后的查询。"""
        manager.register(logging_plugin)  # priority=10
        manager.register(enhance_plugin)  # priority=5 → 先执行
        manager.execute_hook(PluginHook.ON_QUERY, query={"q": "test"})
        # enhance 先执行, 添加了 enhanced=True
        assert logging_plugin.call_log[0][1].get("enhanced") is True

    def test_error_isolation(self, manager):
        """插件异常不影响其他插件。"""
        manager.register(ErrorPlugin())
        manager.register(LoggingPlugin())
        ctx = manager.execute_hook(PluginHook.ON_QUERY, query={"q": "test"})
        # 不应抛出异常, 上下文应正常返回
        assert ctx.query["q"] == "test"

    def test_execute_on_result(self, manager, logging_plugin):
        manager.register(logging_plugin)
        _ctx = manager.execute_hook(PluginHook.ON_RESULT, result={"answer": 42})
        assert len(logging_plugin.call_log) == 1

    def test_execute_on_error(self, manager):
        handler = ErrorHandlingPlugin()
        manager.register(handler)
        err = RuntimeError("test")
        _ctx = manager.execute_hook(PluginHook.ON_ERROR, error=err)
        assert len(handler.errors) == 1
        assert handler.errors[0] is err

    def test_list_plugins(self, manager, logging_plugin):
        manager.register(logging_plugin)
        plugins = manager.list_plugins()
        assert len(plugins) == 1
        assert plugins[0]["name"] == "logging"
        assert plugins[0]["version"] == "1.0.0"

    def test_statistics(self, manager, logging_plugin):
        manager.register(logging_plugin)
        manager.execute_hook(PluginHook.ON_QUERY, query={"q": "test"})
        stats = manager.statistics()
        assert stats["plugin_count"] == 1
        assert stats["hook_executions"]["on_query"] == 1

    def test_empty_manager(self, manager):
        """空管理器执行钩子不报错。"""
        ctx = manager.execute_hook(PluginHook.ON_QUERY, query={"q": "test"})
        assert ctx.query["q"] == "test"
