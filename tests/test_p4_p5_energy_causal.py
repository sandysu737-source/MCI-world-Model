"""P4-P5 波级集成测试 — 能量感知 + 因果推断增强
==============================================

P4 "能量": 能量核心 + 能量Bus + 能量流预测器 + 生克干预
P5 "增强": CEWM基础设施 + 参数化记忆 + 工作记忆增强 + 自主发现
"""

from __future__ import annotations

from mci_world_model import sdk


class TestP4EnergyAwareness:
    """P4 能量感知波次集成测试。"""

    def test_energy_core_exported(self):
        assert "EnergyCore" in sdk.__all__
        assert hasattr(sdk, "EnergyCore")

    def test_cost_module_exported(self):
        assert "EnergyCostModule" in sdk.__all__

    def test_energy_flow_predictor_exported(self):
        assert "EnergyFlowPredictor" in sdk.__all__

    def test_energy_core_instantiable(self):
        ec = sdk.EnergyCore()
        assert ec is not None

    def test_energy_cost_instantiable(self):
        cm = sdk.EnergyCostModule()
        assert cm is not None


class TestP5Enhancement:
    """P5 增强波次集成测试。"""

    def test_parametric_memory_exported(self):
        assert "ParametricMemory" in sdk.__all__

    def test_incremental_learner_exported(self):
        assert "IncrementalLearningEngine" in sdk.__all__

    def test_meta_diagnoser_exported(self):
        assert "MetaDiagnoser" in sdk.__all__

    def test_parametric_memory_instantiable(self):
        pm = sdk.ParametricMemory()
        assert pm is not None

    def test_autonomous_law_discoverer(self):
        ald = sdk.AutonomousLawDiscovererV2()
        assert ald is not None


class TestP4P5KPIs:
    """P4-P5 波级 KPI 验证。"""

    def test_energy_module_count(self):
        energy_symbols = [s for s in sdk.__all__ if "energy" in s.lower() or "Energy" in s]
        assert len(energy_symbols) >= 4

    def test_enhancement_module_count(self):
        enh_symbols = [
            s
            for s in sdk.__all__
            if s
            in (
                "WorkingMemoryEnhancer",
                "ParametricMemory",
                "IncrementalLearningEngine",
                "MetaDiagnoser",
                "AutonomousLawDiscovererV2",
            )
        ]
        assert len(enh_symbols) >= 3
