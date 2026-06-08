"""
test_imports.py — 发布门禁：验证顶层包 + sdk 包的关键符号全部可导入。

对标 su-memory-sdk v3.8.0，验证 mci-world-model V3.0.0 独立成仓后
所有公开 API 仍能正确导入，签名对齐。
"""

import pytest

# =============================================================================
# 1) 顶层包 import smoke
# =============================================================================


class TestTopLevelImport:
    """mci_world_model 顶层 import 不报错。"""

    def test_import_top_level(self):
        import mci_world_model

        assert mci_world_model.__file__.endswith("__init__.py")
        assert "mci_world_model" in mci_world_model.__name__

    def test_import_world_model_module(self):
        from mci_world_model import world_model

        assert hasattr(world_model, "__file__")

    def test_import_sys_subpackage(self):
        from mci_world_model import _sys

        assert hasattr(_sys, "__file__")

    def test_import_sdk_subpackage(self):
        from mci_world_model import sdk

        assert hasattr(sdk, "__file__")


# =============================================================================
# 2) sdk 公开符号
# =============================================================================


class TestSdkPublicAPI:
    """sdk 层关键 API 可从 mci_world_model.sdk 导入。"""

    EXPECTED_API = [
        # Pearl 三层
        "CausalGraph",
        "DoCalculus",
        "CounterfactualEngine",
        # JEPA
        "JEPAEncoder",
        "JEPAPredictor",
        "IdentityPredictor",
        "EnergyPropagationPredictor",
        "BeliefPropagationPredictor",
        "JEPADataset",
        "JEPATrainer",
        # 能量 + 贝叶斯
        "SIGReg",
        "BayesianNetwork",
        "BayesianReasoningSystem",
        "BayesianAugmenter",
        "EnergyConsistencyLoss",  # sdk 中是 EnergyConsistencyLoss（不是 EnergyLoss）
        "GaussianDAG",  # sdk 中频谱因果的代表性类（不是 SpectralCausal）
        "ParametricMemory",
    ]

    @pytest.mark.parametrize("name", EXPECTED_API)
    def test_sdk_api_importable(self, name):
        from mci_world_model import sdk

        assert hasattr(sdk, name), f"sdk.{name} 缺失"


# =============================================================================
# 3) JEPAPredictor 子类继承关系（发布门禁）
# =============================================================================


class TestJEPAPredictorSubclasses:
    """3 个 JEPAPredictor 具体子类必须继承 JEPAPredictor ABC。"""

    SUBCLASSES = [
        "IdentityPredictor",
        "EnergyPropagationPredictor",
        "BeliefPropagationPredictor",
    ]

    def test_jepa_predictor_subclass_inheritance(self):
        from mci_world_model.sdk import (
            BeliefPropagationPredictor,
            EnergyPropagationPredictor,
            IdentityPredictor,
            JEPAPredictor,
        )

        for cls in (IdentityPredictor, EnergyPropagationPredictor, BeliefPropagationPredictor):
            assert issubclass(cls, JEPAPredictor), f"{cls.__name__} 应继承 JEPAPredictor"

    @pytest.mark.parametrize("cls_name", SUBCLASSES)
    def test_predictor_subclass_instantiable(self, cls_name):
        from mci_world_model.sdk import JEPAPredictor

        cls = getattr(__import__("mci_world_model.sdk", fromlist=[cls_name]), cls_name)
        assert issubclass(cls, JEPAPredictor)
        inst = cls()
        assert isinstance(inst, JEPAPredictor)


# =============================================================================
# 4) 顶层 re-export 关键符号
# =============================================================================


class TestTopLevelReExport:
    """sdk 主要 API 也能从 mci_world_model 直接导入。"""

    @pytest.mark.parametrize(
        "name",
        [
            "MCIWorldModel",
            "CausalGraph",
            "DoCalculus",
            "CounterfactualEngine",
            "JEPAEncoder",
            "JEPAPredictor",
            "IdentityPredictor",
            "JEPADataset",
            "JEPATrainer",
            "SIGReg",
            "BayesianNetwork",
            "BayesianReasoningSystem",
        ],
    )
    def test_top_level_reexport(self, name):
        import mci_world_model

        assert hasattr(mci_world_model, name), f"mci_world_model.{name} 缺失（顶层未 re-export）"


# =============================================================================
# 5) _sys 基础层枚举导出
# =============================================================================


class TestSysFoundationEnums:
    """_sys 基础层关键枚举与基础类型。"""

    @pytest.mark.parametrize(
        "name",
        [
            "YinYang",
            "ThreePowers",
            "FourSymbols",
            "Season",
            "TimeStem",
            "TimeBranch",
            "BranchRelation",
            "TrigramType",
            "TrigramRelation",
            "EnergyType",
            "EnergyRelation",
            "StrengthState",
            "EnergyPattern",
            "EnergyCore",
            "TemporalCore",
            "PatternInference",
            "TrigramContext",
        ],
    )
    def test_sys_api_importable(self, name):
        from mci_world_model import _sys

        assert hasattr(_sys, name), f"_sys.{name} 缺失"


# =============================================================================
# 6) 无残留 su_memory 引用（发布门禁）
# =============================================================================


class TestNoSuMemoryResidue:
    """独立成仓后，源码中不应再出现 su_memory.* 引用。"""

    SRC_ROOT = "/Users/mac/qoder m5pro/mci-world-model/src/mci_world_model"

    def test_no_su_memory_imports(self):
        """扫描所有 .py 文件，禁止出现 'from su_memory' 或 'import su_memory'。"""
        import os
        import re

        violations = []
        pattern = re.compile(r"^\s*(?:from\s+su_memory|import\s+su_memory)\b", re.MULTILINE)
        for root, _, files in os.walk(self.SRC_ROOT):
            # 跳过 __pycache__
            if "__pycache__" in root:
                continue
            for f in files:
                if not f.endswith(".py"):
                    continue
                path = os.path.join(root, f)
                with open(path, encoding="utf-8") as fh:
                    content = fh.read()
                if pattern.search(content) and not path.endswith("_world_model.py"):
                    violations.append(path)
        assert not violations, f"仍有 {len(violations)} 个文件含 su_memory 引用: {violations}"
