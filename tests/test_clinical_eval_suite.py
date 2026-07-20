"""UnifiedEvalSuite 单元测试 — 方向四评测基准验证。

验证统一评测框架的核心契约：
    1. SharedTestCases：确定性生成、三类用例
    2. MetricRegistry：6 类指标计算正确
    3. UnifiedEvalSuite：多 backend 注册与横向对比
    4. ReportGenerator：markdown + JSON 输出格式正确
    5. 公平对比：相同 test_cases 跑所有 backend
    6. 可复现：同 SEED 同结果
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from mci_world_model.sdk._clinical_eval_suite import (
    MetricRegistry,
    SharedTestCases,
    UnifiedEvalSuite,
    UnifiedReport,
)

SEED = 42


@pytest.fixture(scope="class")
def trained_backends():
    """训练 MLP + JEPA 两个 backend。"""
    from mci_world_model.sdk import (
        ClinicalDynamicsPredictor,
        JEPAClinicalBridge,
        JEPAClinicalConfig,
    )

    mlp = ClinicalDynamicsPredictor(seed=SEED)
    mlp.fit_from_effect_table(n_samples=200, n_epochs=50, lr=0.005)
    jepa = JEPAClinicalBridge(JEPAClinicalConfig(lr=0.005, seed=SEED))
    jepa.fit_from_effect_table(n_samples=200, n_epochs=40)
    return {"MLP": mlp, "JEPA": jepa}


# =============================================================================
# 1. SharedTestCases
# =============================================================================


class TestSharedTestCases:
    """验证共享测试集生成。"""

    def test_generate_three_categories(self):
        """生成三类用例。"""
        gen = SharedTestCases(seed=SEED)
        cases = gen.generate(n_per_category=10)
        assert len(cases) == 30
        cats = {c.category for c in cases}
        assert cats == {"drug", "natural", "abnormal"}

    def test_deterministic_generation(self):
        """相同 SEED 生成相同测试集。"""
        gen1 = SharedTestCases(seed=SEED)
        gen2 = SharedTestCases(seed=SEED)
        c1 = gen1.generate(n_per_category=5)
        c2 = gen2.generate(n_per_category=5)
        for tc1, tc2 in zip(c1, c2):
            np.testing.assert_array_equal(tc1.state.vital_signs, tc2.state.vital_signs)

    def test_drug_cases_have_actions(self):
        """药物用例有动作。"""
        cases = SharedTestCases(seed=SEED).generate(n_per_category=5)
        drug_cases = [c for c in cases if c.category == "drug"]
        for tc in drug_cases:
            assert tc.action is not None

    def test_natural_cases_have_none_action(self):
        """自然演化用例无动作。"""
        cases = SharedTestCases(seed=SEED).generate(n_per_category=5)
        natural_cases = [c for c in cases if c.category == "natural"]
        for tc in natural_cases:
            assert tc.action is None

    def test_abnormal_cases_outside_normal_range(self):
        """异常用例体征超出正常范围。"""
        from mci_world_model.sdk._clinical_world_state import VITAL_NAMES, VITAL_NORMAL_RANGES

        cases = SharedTestCases(seed=SEED).generate(n_per_category=10)
        abnormal = [c for c in cases if c.category == "abnormal"]
        assert len(abnormal) > 0
        # 至少部分体征异常
        tc = abnormal[0]
        lo, hi = VITAL_NORMAL_RANGES[VITAL_NAMES[0]]
        # 异常用例应有体征在范围外（统计上大概率）
        out_of_range = sum(
            1 for i in range(len(VITAL_NAMES)) if tc.state.vital_signs[-1][i] < lo or tc.state.vital_signs[-1][i] > hi
        )
        # 异常用例应至少有体征超出正常范围
        assert out_of_range >= 1, "异常用例应有体征在范围外"
        assert tc.state.vital_signs.shape == (1, len(VITAL_NAMES))


# =============================================================================
# 2. MetricRegistry
# =============================================================================


class TestMetricRegistry:
    """验证 6 类指标计算。"""

    def test_direction_accuracy_returns_results(self, trained_backends):
        """方向准确率返回 overall + 各类别。"""
        cases = SharedTestCases(SEED).generate(n_per_category=10)
        results = MetricRegistry.direction_accuracy(trained_backends["MLP"], cases, n_steps=1)
        assert len(results) >= 2  # overall dir_acc + mae
        dir_result = next(r for r in results if r.name == "direction_accuracy" and r.category == "overall")
        assert 0.0 <= dir_result.value <= 1.0

    def test_multistep_drift_increasing(self, trained_backends):
        """多步漂移随步数递增（预测逐步偏离首步）。"""
        cases = SharedTestCases(SEED).generate(n_per_category=15)
        results = MetricRegistry.multistep_error_accumulation(trained_backends["MLP"], cases, max_steps=4)
        assert len(results) == 4
        # step1 漂移应为 0（相对自身）
        step1 = next(r for r in results if r.detail["step"] == 1)
        assert step1.value == 0.0
        # 后续步漂移应 >= step1
        step3 = next(r for r in results if r.detail["step"] == 3)
        assert step3.value >= step1.value

    def test_safety_score_range(self, trained_backends):
        """安全性分数在 [0, 1]。"""
        from mci_world_model.sdk import ClinicalMCTSPlanner

        cases = SharedTestCases(SEED).generate(n_per_category=10)
        planner = ClinicalMCTSPlanner(predictor=trained_backends["MLP"])
        result = MetricRegistry.clinical_safety(planner, cases)
        assert result.name == "safety_score"
        assert 0.0 <= result.value <= 1.0

    def test_noise_robustness_curve(self, trained_backends):
        """抗噪曲线返回每个噪声水平一个结果。"""
        cases = SharedTestCases(SEED).generate(n_per_category=15)
        results = MetricRegistry.noise_robustness(trained_backends["MLP"], cases, noise_levels=[0.0, 2.0, 5.0])
        assert len(results) == 3
        sigmas = [r.detail["noise_sigma"] for r in results]
        assert sigmas == [0.0, 2.0, 5.0]

    def test_reconstruction_fidelity_jepa(self, trained_backends):
        """JEPA 的重建保真度可计算。"""
        cases = SharedTestCases(SEED).generate(n_per_category=10)
        result = MetricRegistry.reconstruction_fidelity(trained_backends["JEPA"], cases)
        assert result.name == "reconstruction_mse"
        assert result.detail["applicable"] is True
        assert result.value > 0

    def test_reconstruction_fidelity_mlp_not_applicable(self, trained_backends):
        """MLP 无 encode/reconstruct，返回 -1（不适用）。"""
        cases = SharedTestCases(SEED).generate(n_per_category=10)
        result = MetricRegistry.reconstruction_fidelity(trained_backends["MLP"], cases)
        assert result.value == -1.0
        assert result.detail["applicable"] is False

    def test_semantic_discrimination_jepa(self, trained_backends):
        """JEPA 的语义区分能力可量化。

        fixture 中的 JEPA 为纯数值模式（未注入 semantic_embedder），相同
        体征不同诊断的潜向量距离恒为 0，故 value=0 是预期；此处验证结构
        合法性：encode 可执行（applicable=True）且返回 float 指标，而非
        用恒真的 value>=0 形同虚设。
        """
        result = MetricRegistry.semantic_discrimination(trained_backends["JEPA"], n_pairs=10)
        assert result.name == "semantic_discrimination"
        assert result.detail["applicable"] is True
        assert isinstance(result.value, float)

    def test_semantic_discrimination_mlp_zero(self, trained_backends):
        """MLP 无 encode 方法，返回 0（无法区分）。"""
        result = MetricRegistry.semantic_discrimination(trained_backends["MLP"], n_pairs=10)
        assert result.value == 0.0


# =============================================================================
# 3. UnifiedEvalSuite
# =============================================================================


class TestUnifiedEvalSuite:
    """验证统一评测套件。"""

    def test_register_backend_requires_fitted(self):
        """注册未训练 backend 抛错。"""
        from mci_world_model.sdk import ClinicalDynamicsPredictor

        suite = UnifiedEvalSuite()
        unfitted = ClinicalDynamicsPredictor(seed=SEED)
        with pytest.raises(ValueError, match="未训练"):
            suite.register_backend("unfitted", unfitted)

    def test_run_produces_report(self, trained_backends):
        """run 生成完整报告。"""
        suite = UnifiedEvalSuite("测试报告")
        suite.register_backend("MLP", trained_backends["MLP"])
        suite.register_backend("JEPA", trained_backends["JEPA"])
        report = suite.run(n_per_category=10)
        assert isinstance(report, UnifiedReport)
        assert len(report.backend_reports) == 2
        assert report.n_test_cases == 30  # 3 类 × 10

    def test_run_reproducible(self, trained_backends):
        """相同 SEED 两次 run 结果一致。"""
        suite1 = UnifiedEvalSuite("r1", seed=SEED)
        suite1.register_backend("MLP", trained_backends["MLP"])
        r1 = suite1.run(n_per_category=10)

        suite2 = UnifiedEvalSuite("r2", seed=SEED)
        suite2.register_backend("MLP", trained_backends["MLP"])
        r2 = suite2.run(n_per_category=10)

        m1 = r1.backend_reports[0].to_dict()["metrics"]
        m2 = r2.backend_reports[0].to_dict()["metrics"]
        assert m1["direction_accuracy"]["value"] == m2["direction_accuracy"]["value"]


# =============================================================================
# 4. ReportGenerator（to_dict / to_markdown / save）
# =============================================================================


class TestReportGeneration:
    """验证报告生成。"""

    def test_to_dict_serializable(self, trained_backends):
        """to_dict 可 JSON 序列化。"""
        suite = UnifiedEvalSuite("序列化测试")
        suite.register_backend("MLP", trained_backends["MLP"])
        report = suite.run(n_per_category=8)
        d = report.to_dict()
        json_str = json.dumps(d, ensure_ascii=False)
        assert "backends" in json_str

    def test_to_markdown_has_table(self, trained_backends):
        """to_markdown 包含对比表。"""
        suite = UnifiedEvalSuite("MD测试")
        suite.register_backend("MLP", trained_backends["MLP"])
        suite.register_backend("JEPA", trained_backends["JEPA"])
        report = suite.run(n_per_category=8)
        md = report.to_markdown()
        assert "## 核心指标横向对比" in md
        assert "MLP" in md
        assert "JEPA" in md
        assert "方向准确率" in md

    def test_save_report_creates_files(self, trained_backends, tmp_path):
        """save_report 创建 JSON + MD 文件。"""
        suite = UnifiedEvalSuite("保存测试")
        suite.register_backend("MLP", trained_backends["MLP"])
        report = suite.run(n_per_category=8)
        json_path, md_path = suite.save_report(report, tmp_path)
        assert json_path.exists()
        assert md_path.exists()
        assert json_path.suffix == ".json"
        assert md_path.suffix == ".md"
        # JSON 可解析
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "backends" in data

    def test_markdown_multistep_table(self, trained_backends):
        """markdown 包含多步误差表。"""
        suite = UnifiedEvalSuite("多步测试")
        suite.register_backend("MLP", trained_backends["MLP"])
        report = suite.run(n_per_category=15)
        md = report.to_markdown()
        assert "多步预测误差累积" in md
        assert "Step1" in md


# =============================================================================
# 5. 公平对比验证
# =============================================================================


class TestFairComparison:
    """验证所有 backend 跑相同 test_cases。"""

    def test_all_backends_same_test_cases(self, trained_backends):
        """所有 backend 的测试集相同（公平对比）。"""
        suite = UnifiedEvalSuite("公平对比")
        suite.register_backend("MLP", trained_backends["MLP"])
        suite.register_backend("JEPA", trained_backends["JEPA"])
        cases = suite.prepare_test_cases(n_per_category=10)
        # 所有 backend 用同一 cases
        mlp_dir = MetricRegistry.direction_accuracy(trained_backends["MLP"], cases)
        jepa_dir = MetricRegistry.direction_accuracy(trained_backends["JEPA"], cases)
        mlp_n = mlp_dir[0].detail["n_comparisons"]
        jepa_n = jepa_dir[0].detail["n_comparisons"]
        assert mlp_n == jepa_n  # 相同比较次数
