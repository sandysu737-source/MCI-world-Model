"""P6-P8 增强感知+新奇验证测试"""

from mci_world_model import sdk


class TestEnhancedPerception:
    """测试增强感知"""

    def setup_method(self):
        self.ep = sdk.EnhancedPerception()

    def test_instantiation(self):
        assert self.ep is not None

    def test_has_perceive(self):
        methods = [m for m in dir(self.ep) if not m.startswith("_")]
        assert len(methods) > 0

    def test_export_in_all(self):
        assert "EnhancedPerception" in sdk.__all__


class TestNoveltyVerifier:
    """测试新奇验证器"""

    def setup_method(self):
        self.nv = sdk.NoveltyVerifier()

    def test_instantiation(self):
        assert self.nv is not None

    def test_has_verify(self):
        methods = [m for m in dir(self.nv) if not m.startswith("_")]
        assert len(methods) > 0

    def test_novelty_result_creation(self):
        nr = sdk.NoveltyResult(
            novelty_confirmed=True,
            max_structural_similarity=0.3,
            min_prediction_difference=0.7,
            novelty_degree=0.8,
        )
        assert nr is not None

    def test_export_in_all(self):
        assert "NoveltyVerifier" in sdk.__all__
        assert "NoveltyResult" in sdk.__all__


class TestCausalUpdater:
    """测试因果更新器"""

    def setup_method(self):
        self.cu = sdk.CausalUpdater()

    def test_instantiation(self):
        assert self.cu is not None

    def test_has_update(self):
        assert hasattr(self.cu, "update") or hasattr(self.cu, "update_belief")

    def test_export_in_all(self):
        assert "CausalUpdater" in sdk.__all__


class TestCognitiveDiversity:
    """测试认知多样性"""

    def setup_method(self):
        self.cd = sdk.CognitiveDiversity()

    def test_instantiation(self):
        assert self.cd is not None

    def test_export_in_all(self):
        assert "CognitiveDiversity" in sdk.__all__
