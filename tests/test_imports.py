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


class TestSysToSdkPenetration:
    """v3.3.1: _sys.__all__ 每个符号必须穿透到 sdk.__all__。

    这是一个自动化门禁：当 _sys 层新增符号但忘记同步到 sdk 导出时，
    本测试会立即失败，防止能力泄露问题再次发生。
    """

    def test_sys_all_subset_of_sdk_all(self):
        """_sys.__all__ 的每个符号必须在 sdk.__all__ 中出现。"""
        from mci_world_model import _sys, sdk

        sys_all = set(getattr(_sys, "__all__", []))
        sdk_all = set(getattr(sdk, "__all__", []))
        gap = sorted(sys_all - sdk_all)
        assert not gap, (
            f"_sys→sdk 穿透断裂: {len(gap)} 个符号未导出到 SDK:\n"
            + "\n".join(f"  {s}" for s in gap)
            + f"\n\n请在 sdk/__init__.py 中补充这 {len(gap)} 个符号的导入和 __all__ 注册。"
        )

    def test_sys_symbols_actually_importable(self):
        """_sys.__all__ 每个符号必须能从 sdk 实际 getattr 到。"""
        from mci_world_model import _sys, sdk

        failed = []
        for name in getattr(_sys, "__all__", []):
            if not hasattr(sdk, name):
                failed.append(name)
        assert not failed, f"{len(failed)} 个 _sys 符号无法从 sdk 访问:\n" + "\n".join(f"  {s}" for s in failed)

    def test_penetration_rate_100(self):
        """_sys→sdk 穿透率必须达到 100%。"""
        from mci_world_model import _sys, sdk

        sys_all = set(getattr(_sys, "__all__", []))
        sdk_all = set(getattr(sdk, "__all__", []))
        covered = len(sys_all & sdk_all)
        total = len(sys_all)
        rate = covered / total if total > 0 else 0
        assert rate == 1.0, f"_sys→sdk 穿透率 {rate:.1%} ({covered}/{total}) 未达 100%。缺少 {total - covered} 个符号。"


class TestTemporalSymbolsFunctional:
    """v3.3.1: 时空/时序核心符号从 SDK 导入后功能正常。"""

    def test_temporal_system_methods(self):
        """TemporalSystem 的 8 个核心方法可调用。"""
        from datetime import date

        from mci_world_model.sdk import TemporalSystem

        ts = TemporalSystem()
        # date_to_time_code
        tc = ts.date_to_time_code(date(2024, 1, 15))
        assert tc is not None
        # get_current_time_code
        curr = ts.get_current_time_code()
        assert curr is not None
        # get_jiazi_position
        pos = ts.get_jiazi_position(date(2024, 6, 1))
        assert isinstance(pos, (int, float))
        # temporal_similarity
        sim = ts.temporal_similarity(date(2024, 1, 1), date(2024, 1, 2))
        assert 0.0 <= sim <= 1.0
        # calculate_time_decay
        decay = ts.calculate_time_decay(30, "木")
        assert 0.0 <= decay <= 1.0

    def test_temporal_info_dataclass(self):
        """TemporalInfo 可作为数据容器实例化。"""
        from mci_world_model.sdk import TemporalInfo

        info = TemporalInfo(
            tian_gan="甲",
            di_zhi="子",
            time_code="甲子",
            energy_type="木",
            yin_yang="阳",
            season="冬",
            is_birthday=False,
        )
        assert info.tian_gan == "甲"
        assert info.di_zhi == "子"

    def test_dizhi_tiangan_enums(self):
        """DiZhi/TianGan 枚举含正确数量。"""
        from mci_world_model.sdk import DiZhi, TianGan

        assert len(DiZhi.NAMES) == 12
        assert len(TianGan.NAMES) == 10

    def test_time_code_factory(self):
        """create_time_code 工厂函数可用。"""
        from mci_world_model.sdk import create_time_code

        tc = create_time_code(0, 0)  # stem_idx=0, branch_idx=0
        assert tc is not None

    def test_spatial_data_tables(self):
        """时空数据表（地支冲合刑、天干冲合）非空且类型正确。"""
        from mci_world_model.sdk import (
            BRANCH_CHONG,
            BRANCH_CHONG_MAP,
            BRANCH_HE,
            BRANCH_XING,
            STEM_CHONG,
            STEM_HE,
            STEM_HE_MAP,
        )

        # BRANCH_CHONG/HE 是 dict 而非 set（地支枚举→地支枚举映射）
        assert isinstance(BRANCH_CHONG, dict) and len(BRANCH_CHONG) > 0
        assert isinstance(BRANCH_HE, dict) and len(BRANCH_HE) > 0
        assert isinstance(BRANCH_XING, dict) and len(BRANCH_XING) > 0
        assert isinstance(STEM_CHONG, (dict, set)) and len(STEM_CHONG) > 0
        assert isinstance(STEM_HE, (dict, set)) and len(STEM_HE) > 0
        assert isinstance(BRANCH_CHONG_MAP, dict) and len(BRANCH_CHONG_MAP) > 0
        assert isinstance(STEM_HE_MAP, dict) and len(STEM_HE_MAP) > 0

    def test_bayesian_engine_from_sdk(self):
        """BayesianEngine 可从 SDK 导入并实例化。"""
        from mci_world_model.sdk import BayesianEngine

        engine = BayesianEngine()
        assert engine is not None

    def test_causal_chain_from_sdk(self):
        """CausalChain 可从 SDK 导入。"""
        from mci_world_model.sdk import CausalChain

        assert CausalChain is not None


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
