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

        test_energy_relations()


class TestCategoryCoreSelfTest:
    """_category_core.py 内置测试。"""

    def test_category_core(self):
        from mci_world_model._sys._category_core import test_trigram_core

        test_trigram_core()


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


class TestCausalEngineSelfTest:
    """_causal_engine.py 内置测试。"""

    def test_causal_engine_sys(self):
        from mci_world_model._sys._causal_engine import test_causal_engine

        test_causal_engine()
