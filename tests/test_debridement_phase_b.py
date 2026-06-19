"""Phase B: DebridementWorldModel + ForceTissueDynamics + Safety 测试."""

import numpy as np
import pytest


@pytest.fixture
def generator():
    from mci_world_model.sdk._debridement_data import SyntheticDebridementGenerator
    return SyntheticDebridementGenerator(seed=42)


@pytest.fixture
def config():
    from mci_world_model.sdk._debridement_world_model import DebridementConfig
    return DebridementConfig.tiny()


class TestDebridementConfig:
    def test_tiny(self):
        from mci_world_model.sdk._debridement_world_model import DebridementConfig
        c = DebridementConfig.tiny()
        assert c.d_model == 128
        assert c.n_layers == 2
        assert not c.use_vision

    def test_base(self):
        from mci_world_model.sdk._debridement_world_model import DebridementConfig
        c = DebridementConfig.base()
        assert c.d_model == 512
        assert c.use_vision


class TestDebridementWorldModel:
    def test_init(self, config):
        from mci_world_model.sdk._debridement_world_model import DebridementWorldModel
        model = DebridementWorldModel(config)
        assert model.n_params > 0
        assert not model.is_trained

    def test_encode_modalities(self, generator, config):
        from mci_world_model.sdk._debridement_world_model import DebridementWorldModel
        model = DebridementWorldModel(config)
        sample = generator.generate_sample()
        fused = model.encode_modalities(sample)
        assert fused.shape == (config.d_model,)

    def test_forward(self, generator, config):
        from mci_world_model.sdk._debridement_world_model import DebridementWorldModel
        model = DebridementWorldModel(config)
        sample = generator.generate_sample()
        out = model.forward(sample)
        assert "dynamics" in out
        assert "tissue_probs" in out
        assert out["tissue_probs"].shape == (4,)
        assert np.allclose(out["tissue_probs"].sum(), 1.0)

    def test_train_smoke(self, generator, config):
        from mci_world_model.sdk._debridement_world_model import DebridementWorldModel
        model = DebridementWorldModel(config)
        samples = generator.generate_batch(40, balanced=True)
        stats = model.train(samples, n_epochs=5, lr=0.01, batch_size=8)
        assert "final_loss" in stats
        assert model.is_trained

    def test_predict_tissue(self, generator, config):
        from mci_world_model.sdk._debridement_world_model import DebridementWorldModel
        model = DebridementWorldModel(config)
        samples = generator.generate_batch(20)
        model.train(samples, n_epochs=3, lr=0.01)
        sample = generator.generate_sample(0)
        probs = model.predict_tissue(sample)
        assert probs.shape == (4,)

    def test_save_load(self, generator, config):
        from mci_world_model.sdk._debridement_world_model import DebridementWorldModel
        import tempfile, shutil, os
        model = DebridementWorldModel(config)
        samples = generator.generate_batch(10)
        model.train(samples, n_epochs=2, lr=0.01)
        d = tempfile.mkdtemp()
        try:
            path = os.path.join(d, "dwm")
            assert model.save(path)
            loaded = DebridementWorldModel.load(path, config)
            assert loaded is not None
            assert loaded.is_trained
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestForceTissueDynamics:
    def test_predict_removal_nec(self):
        from mci_world_model.sdk._force_tissue_dynamics import ForceTissueDynamics
        ftd = ForceTissueDynamics()
        pred = ftd.predict_removal(0, 2.0, 5.0)
        assert pred.tissue_type == 0
        assert pred.depth_removed_mm > 0
        assert pred.is_safe  # 2N < 3N max

    def test_predict_removal_epithelial_unsafe(self):
        from mci_world_model.sdk._force_tissue_dynamics import ForceTissueDynamics
        ftd = ForceTissueDynamics()
        pred = ftd.predict_removal(3, 1.0, 1.0)
        assert not pred.is_safe
        assert "严禁" in pred.warning

    def test_safety_check_epithelial_stops(self):
        from mci_world_model.sdk._force_tissue_dynamics import ForceTissueDynamics
        ftd = ForceTissueDynamics()
        v = ftd.safety_check(3, 0.5, 1.0)
        assert not v.passed

    def test_get_max_force(self):
        from mci_world_model.sdk._force_tissue_dynamics import ForceTissueDynamics
        assert ForceTissueDynamics.get_max_force(0) == 3.0
        assert ForceTissueDynamics.get_max_force(3) == 0.5


class TestDebridementSafety:
    def test_tissue_force_constraint(self):
        from mci_world_model.sdk._safety import TissueForceConstraint
        from mci_world_model.sdk._debridement_data import DebridementSample
        c = TissueForceConstraint(tissue_label=0)
        s = DebridementSample(tool_force_n=2.0)
        r = c.check(s)
        assert r.passed
        s2 = DebridementSample(tool_force_n=5.0)
        r2 = c.check(s2)
        assert not r2.passed

    def test_thermal_constraint(self):
        from mci_world_model.sdk._safety import ThermalSafetyConstraint
        from mci_world_model.sdk._debridement_data import DebridementSample
        c = ThermalSafetyConstraint(max_temp_c=42.0)
        s = DebridementSample()
        r = c.check(s)
        assert r.passed
        s2 = DebridementSample(thermal_image=np.full((224, 224), 45.0, dtype=np.float32))
        r2 = c.check(s2)
        assert not r2.passed

    def test_depth_constraint(self):
        from mci_world_model.sdk._safety import DepthLimitConstraint
        from mci_world_model.sdk._debridement_data import DebridementSample
        c = DepthLimitConstraint(max_depth_mm=5.0)
        s = DebridementSample(wound_depth_mm=3.0)
        r = c.check(s)
        assert r.passed
        s2 = DebridementSample(wound_depth_mm=8.0)
        r2 = c.check(s2)
        assert not r2.passed
