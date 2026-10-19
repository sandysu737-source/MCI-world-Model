"""
MCI World Model v4.0.0 — _sys/ 模块内置测试的 pytest 包装器

将 _sys/ 下 8 个模块的 print-based 自建测试迁移至 pytest，
保留原始测试逻辑不变，仅包装调用入口。

v4.0.0: 全部 xfail 已修复，所有 8 个测试均应通过。
"""

from __future__ import annotations


class TestEnergyCoreSelfTest:
    """_energy_core.py 内置测试。"""

    def test_energy_core(self):
        from mci_world_model._sys._energy_core import test_energy_core

        result = test_energy_core()
        assert result is True


class TestEnergyBusSelfTest:
    """_energy_bus.py 内置测试。"""

    def test_energy_bus(self):
        from mci_world_model._sys._energy_bus import test_energy_bus

        result = test_energy_bus()
        assert result is True


class TestTemporalCoreSelfTest:
    """_temporal_core.py 内置测试。"""

    def test_temporal_core(self):
        from mci_world_model._sys._temporal_core import _run_tests

        result = _run_tests()
        assert result is None or result is True


class TestEnergyRelationsSelfTest:
    """_energy_relations.py 内置测试。"""

    def test_energy_relations(self):
        from mci_world_model._sys._energy_relations import test_energy_relations

        # 自检必须断言返回值：只调用不断言 = 空转测试（变异测试无法捕获任何缺陷）。
        assert test_energy_relations() is True


class TestCategoryCoreSelfTest:
    """_category_core.py 内置测试。"""

    def test_category_core(self):
        from mci_world_model._sys._category_core import test_trigram_core

        assert test_trigram_core() is True


class TestUnifiedUnitSelfTest:
    """_unified_unit.py 内置测试。"""

    def test_unified_unit(self):
        from mci_world_model._sys._unified_unit import run_tests

        result = run_tests()
        assert result is True


class TestDimensionMapSelfTest:
    """_dimension_map.py 内置测试。"""

    def test_dimension_map(self):
        from mci_world_model._sys._dimension_map import _run_tests

        result = _run_tests()
        assert result is True

    def test_trigram_order_tables_exact(self):
        """先天/后天序数表是领域契约，必须逐项精确匹配。"""
        from mci_world_model._sys._dimension_map import (
            POST_TRIGRAM_ORDER,
            PRIOR_TRIGRAM_ORDER,
        )

        assert PRIOR_TRIGRAM_ORDER == {0: 1, 7: 2, 5: 3, 2: 4, 3: 5, 4: 6, 6: 7, 1: 8}
        assert POST_TRIGRAM_ORDER == {4: 1, 1: 2, 2: 3, 3: 4, 0: 6, 7: 7, 6: 8, 5: 9}


class TestCausalEngineSelfTest:
    """_causal_engine.py 内置测试。"""

    def test_causal_engine_sys(self):
        from mci_world_model._sys._causal_engine import test_causal_engine

        assert test_causal_engine() is True
