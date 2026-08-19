"""P13 创造引擎+因果经济测试"""

from mci_world_model import sdk


class TestCausalCreationEngine:
    """测试因果创造引擎"""

    def setup_method(self):
        self.cce = sdk.CausalCreationEngine()

    def test_instantiation(self):
        assert self.cce is not None

    def test_has_create(self):
        methods = [m for m in dir(self.cce) if not m.startswith("_")]
        assert len(methods) > 0

    def test_creation_strategy_enum(self):
        assert len(list(sdk.CreationStrategy)) > 0

    def test_export_in_all(self):
        assert "CausalCreationEngine" in sdk.__all__
        assert "CreationStrategy" in sdk.__all__


class TestCausalEconomy:
    """测试因果经济"""

    def setup_method(self):
        self.ce = sdk.CausalEconomy()

    def test_instantiation(self):
        assert self.ce is not None

    def test_has_trade(self):
        assert hasattr(self.ce, "trade_knowledge")

    def test_has_get_economy_report(self):
        assert hasattr(self.ce, "get_economy_report")

    def test_has_assess_market_health(self):
        assert hasattr(self.ce, "assess_market_health")

    def test_export_in_all(self):
        assert "CausalEconomy" in sdk.__all__


class TestCreativeConsciousness:
    """测试创造性因果意识"""

    def setup_method(self):
        self.cc = sdk.CreativeCausalConsciousness()

    def test_instantiation(self):
        assert self.cc is not None

    def test_export_in_all(self):
        assert "CreativeCausalConsciousness" in sdk.__all__


class TestCounterfactualOracle:
    """测试反事实先知"""

    def setup_method(self):
        self.co = sdk.CounterfactualOracle()

    def test_instantiation(self):
        assert self.co is not None

    def test_has_query(self):
        methods = [m for m in dir(self.co) if not m.startswith("_")]
        assert len(methods) > 0

    def test_export_in_all(self):
        assert "CounterfactualOracle" in sdk.__all__
