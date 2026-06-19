"""P14 '太极' 波次测试: CausalUniverseTheory + UltimateCausalIntelligence"""

from mci_world_model import sdk


class TestCausalUniverseTheory:
    """测试因果宇宙统一理论"""

    def setup_method(self):
        self.cut = sdk.CausalUniverseTheory()

    def test_instantiation(self):
        assert self.cut is not None

    def test_derive_universal_causal_law(self):
        result = self.cut.derive_universal_causal_law(["physics"])
        assert isinstance(result, dict)

    def test_unify_causal_reasoning(self):
        result = self.cut.unify_causal_reasoning({"query": "test"})
        assert isinstance(result, dict)

    def test_causal_scale_enum(self):
        assert hasattr(sdk.CausalScale, "MICRO") or len(list(sdk.CausalScale)) > 0

    def test_has_unify_causal_reasoning(self):
        assert hasattr(self.cut, "unify_causal_reasoning")

    def test_export_in_all(self):
        assert "CausalUniverseTheory" in sdk.__all__
        assert "CausalScale" in sdk.__all__
        assert "ScaleResult" in sdk.__all__


class TestUltimateCausalIntelligence:
    """测试终极因果智能"""

    def setup_method(self):
        self.uci = sdk.UltimateCausalIntelligence()

    def test_instantiation(self):
        assert self.uci is not None

    def test_has_autonomous_exist(self):
        assert hasattr(self.uci, "autonomous_exist")

    def test_existence_mode_enum(self):
        assert len(list(sdk.ExistenceMode)) > 0

    def test_capability_status_enum(self):
        assert len(list(sdk.CapabilityStatus)) > 0

    def test_autonomous_action_creation(self):
        action = sdk.AutonomousAction("test_action", "target", {}, "outcome")
        assert action is not None

    def test_capability_creation(self):
        cap = sdk.Capability("test_cap", sdk.CapabilityStatus.ACTIVE)
        assert cap is not None

    def test_existence_report_creation(self):
        report = sdk.ExistenceReport(
            mode=sdk.ExistenceMode.BEING,
            n_active_capabilities=0,
            n_total_capabilities=5,
            autonomy_level=0.8,
            reflection_depth=3,
            evolution_readiness=0.7,
        )
        assert report is not None

    def test_export_in_all(self):
        assert "UltimateCausalIntelligence" in sdk.__all__
        assert "AutonomousAction" in sdk.__all__
        assert "ExistenceMode" in sdk.__all__


class TestP14Integration:
    """P14 集成测试"""

    def test_universe_theory_with_ultimate(self):
        cut = sdk.CausalUniverseTheory()
        uci = sdk.UltimateCausalIntelligence()
        assert cut is not None
        assert uci is not None

    def test_derive_universal_law(self):
        cut = sdk.CausalUniverseTheory()
        result = cut.derive_universal_causal_law(["physics"])
        assert isinstance(result, dict)
