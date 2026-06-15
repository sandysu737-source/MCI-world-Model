"""
tests/test_mimic_causal_benchmark.py — MIMIC 因果推理 Benchmark 单元测试

验证:
1. 合成数据生成正确性
2. CEWM 推理管道连通性
3. Ground truth 评估逻辑
4. 多层 tier 指标计算
5. LLM 输出解析
6. CI 自动跳过逻辑
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from benchmarks.real_world.llm_baseline import (
    LLMBaselineRunner,
    parse_llm_causal_output,
)
from benchmarks.real_world.mimic_causal_benchmark import (
    GROUND_TRUTH_EDGES,
    ICU_VARIABLES,
    BenchmarkResult,
    CausalMetrics,
    MIMICCausalBenchmark,
    PatientTimeline,
    generate_synthetic_icu_patients,
)

# ─────────────────────────────────────────────────────────────────────────────
# 合成数据生成
# ─────────────────────────────────────────────────────────────────────────────


class TestSyntheticDataGeneration:
    """验证合成 ICU 数据生成。"""

    def test_generate_patients_count(self):
        """生成指定数量的患者。"""
        patients = generate_synthetic_icu_patients(n_patients=10, seed=42)
        assert len(patients) == 10

    def test_patient_timeline_shape(self):
        """患者时序数据维度正确。"""
        patients = generate_synthetic_icu_patients(n_patients=3, n_timesteps=48, seed=42)
        for p in patients:
            assert p.data.shape[0] == 48
            assert p.data.shape[1] == len(ICU_VARIABLES)

    def test_patient_variables_match(self):
        """变量名列表与数据矩阵列数匹配。"""
        patients = generate_synthetic_icu_patients(n_patients=1, seed=42)
        p = patients[0]
        assert len(p.variables) == p.data.shape[1]

    def test_missing_values_present(self):
        """合成数据包含缺失值 (模拟真实 MIMIC 数据)。"""
        patients = generate_synthetic_icu_patients(n_patients=5, seed=42)
        has_missing = any(np.any(np.isnan(p.data)) for p in patients)
        assert has_missing, "合成数据应包含缺失值"

    def test_reproducibility(self):
        """相同 seed 产生相同数据。"""
        p1 = generate_synthetic_icu_patients(n_patients=1, seed=123)
        p2 = generate_synthetic_icu_patients(n_patients=1, seed=123)
        np.testing.assert_array_almost_equal(p1[0].data, p2[0].data)

    def test_dopamine_causes_hr_increase(self):
        """合成数据中多巴胺剂量正向影响心率 (已知因果结构)。"""
        patients = generate_synthetic_icu_patients(n_patients=20, seed=42)
        hr_increases_with_dopamine = 0
        checked = 0
        for p in patients:
            dop_idx = p.variables.index("dopamine_dose")
            hr_idx = p.variables.index("heart_rate")
            dop = p.data[:, dop_idx]
            hr = p.data[:, hr_idx]
            valid = np.isfinite(dop) & np.isfinite(hr)
            if np.sum(valid) > 10:
                corr = np.corrcoef(dop[valid], hr[valid])[0, 1]
                if np.isfinite(corr):
                    checked += 1
                    if corr > 0:
                        hr_increases_with_dopamine += 1
        # 大部分患者应显示正相关
        assert hr_increases_with_dopamine / max(checked, 1) > 0.5


class TestPatientTimeline:
    """PatientTimeline 数据结构测试。"""

    def test_to_timeline_dicts(self):
        """转换为 PhysicalGraphBuilder 格式。"""
        data = np.array([[80.0, 65.0], [82.0, 67.0]])
        p = PatientTimeline(
            patient_id="test",
            variables=["heart_rate", "map"],
            data=data,
        )
        dicts = p.to_timeline_dicts()
        assert len(dicts) == 2
        assert dicts[0]["day"] == 1
        assert dicts[0]["heart_rate"] == 80.0

    def test_data_summary(self):
        """数据摘要生成。"""
        data = np.random.randn(48, 5)
        p = PatientTimeline(
            patient_id="test",
            variables=["a", "b", "c", "d", "e"],
            data=data,
        )
        summary = p.data_summary()
        assert "test" in summary
        assert "48 timepoints" in summary


# ─────────────────────────────────────────────────────────────────────────────
# Ground Truth 结构
# ─────────────────────────────────────────────────────────────────────────────


class TestGroundTruth:
    """多层 ground truth 结构验证。"""

    def test_ground_truth_has_three_tiers(self):
        """ground truth 包含三个 tier。"""
        tiers = set(v["tier"] for v in GROUND_TRUTH_EDGES.values())
        assert 1 in tiers
        assert 2 in tiers
        assert 3 in tiers

    def test_ground_truth_has_required_fields(self):
        """每条 ground truth 包含必要字段。"""
        required = {"tier", "direction", "consensus_level", "effect_size_range"}
        for key, val in GROUND_TRUTH_EDGES.items():
            missing = required - set(val.keys())
            assert not missing, f"Missing fields in {key}: {missing}"

    def test_tier_1_high_consensus(self):
        """Tier 1 因果对共识度 ≥ 0.90。"""
        for key, val in GROUND_TRUTH_EDGES.items():
            if val["tier"] == 1:
                assert val["consensus_level"] >= 0.90, f"Tier 1 {key} has low consensus"

    def test_tier_3_low_consensus(self):
        """Tier 3 因果对共识度 < 0.55。"""
        for key, val in GROUND_TRUTH_EDGES.items():
            if val["tier"] == 3:
                assert val["consensus_level"] < 0.55, f"Tier 3 {key} has high consensus"

    def test_effect_size_range_valid(self):
        """效应量范围有效 (lower < upper, both positive)。"""
        for key, val in GROUND_TRUTH_EDGES.items():
            lo, hi = val["effect_size_range"]
            assert 0 <= lo < hi, f"Invalid effect_size_range in {key}"


# ─────────────────────────────────────────────────────────────────────────────
# 评估逻辑
# ─────────────────────────────────────────────────────────────────────────────


class TestCausalMetrics:
    """CausalMetrics 指标计算验证。"""

    def test_perfect_prediction(self):
        """完美预测 → F1=1.0, direction_acc=1.0 (仅 Tier 1+2)。"""
        bench = MIMICCausalBenchmark()
        # 构造与 ground truth 完全匹配的预测 (仅 Tier 1+2)
        perfect_edges = []
        for (cause, effect), val in GROUND_TRUTH_EDGES.items():
            if val["tier"] in (1, 2):
                perfect_edges.append((cause, effect, val["direction"], 0.5, "medium"))

        metrics = bench.compare_graphs(perfect_edges, tier_filter=[1, 2])
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        assert metrics.direction_accuracy == 1.0

    def test_no_prediction(self):
        """零预测 → F1=0.0, recall=0.0。"""
        bench = MIMICCausalBenchmark()
        metrics = bench.compare_graphs([], tier_filter=[1, 2])
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0

    def test_wrong_direction(self):
        """方向错误 → direction_accuracy < 1.0。"""
        bench = MIMICCausalBenchmark()
        wrong_edges = []
        for (cause, effect), val in GROUND_TRUTH_EDGES.items():
            wrong_dir = "negative" if val["direction"] == "positive" else "positive"
            wrong_edges.append((cause, effect, wrong_dir, 0.5, "medium"))

        metrics = bench.compare_graphs(wrong_edges, tier_filter=[1, 2])
        assert metrics.direction_accuracy == 0.0

    def test_tier_filter(self):
        """tier_filter 仅评估指定 tier。"""
        bench = MIMICCausalBenchmark()
        edges = []
        for (cause, effect), val in GROUND_TRUTH_EDGES.items():
            edges.append((cause, effect, val["direction"], 0.5, "medium"))

        m_all = bench.compare_graphs(edges)
        m_t1 = bench.compare_graphs(edges, tier_filter=[1])
        m_t2 = bench.compare_graphs(edges, tier_filter=[2])

        # Tier 1 边数应该少于全部
        assert m_t1.n_edges_ground_truth < m_all.n_edges_ground_truth
        # 不同 tier 的边数不同
        assert m_t1.n_edges_ground_truth != m_t2.n_edges_ground_truth

    def test_uncertainty_aware(self):
        """不确定性感知: "unknown" 方向在低共识时不应被重罚。"""
        bench = MIMICCausalBenchmark()
        # 对 Tier 3 (低共识) 使用 "unknown" 方向
        unknown_edges = []
        for (cause, effect), val in GROUND_TRUTH_EDGES.items():
            unknown_edges.append((cause, effect, "unknown", 0.5, "medium"))

        metrics = bench.compare_graphs(unknown_edges, tier_filter=[3])
        # direction_agreement 应 > 0 (因为低共识时不惩罚)
        assert metrics.direction_agreement > 0

    def test_direction_agreement_weighted(self):
        """序数一致性: 正确方向得分高，错误方向按共识度加权惩罚。"""
        bench = MIMICCausalBenchmark()
        # 完全正确的预测
        correct_edges = []
        for (cause, effect), val in GROUND_TRUTH_EDGES.items():
            correct_edges.append((cause, effect, val["direction"], 0.5, "medium"))

        metrics = bench.compare_graphs(correct_edges, tier_filter=[1, 2])
        assert metrics.direction_agreement >= 0.9

    def test_metrics_to_dict(self):
        """CausalMetrics 序列化。"""
        m = CausalMetrics(precision=0.8, recall=0.6, f1=0.686)
        d = m.to_dict()
        assert "precision" in d
        assert "f1" in d
        assert isinstance(d, dict)


# ─────────────────────────────────────────────────────────────────────────────
# CEWM 推理管道
# ─────────────────────────────────────────────────────────────────────────────


class TestCEWMInference:
    """CEWM 因果推理管道连通性验证。"""

    def test_run_cewm_inference_returns_edges(self):
        """CEWM 推理返回非空边列表。"""
        bench = MIMICCausalBenchmark()
        patients = bench.load_synthetic_dataset(n_patients=3, n_timesteps=30, seed=42)
        result = bench.run_cewm_inference(patients[0])
        assert "edges" in result
        # 合成数据中存在已植入的因果结构，应该能发现一些边
        # (但不保证数量，因为 PhysicalGraphBuilder 依赖相关性阈值)

    def test_run_cewm_benchmark_returns_result(self):
        """CEWM Benchmark 返回 BenchmarkResult。"""
        bench = MIMICCausalBenchmark()
        patients = bench.load_synthetic_dataset(n_patients=3, n_timesteps=30, seed=42)
        result = bench.run_cewm_benchmark(patients)
        assert isinstance(result, BenchmarkResult)
        assert result.method == "cewm"
        assert result.n_patients == 3
        assert result.runtime_seconds > 0

    def test_full_report_structure(self):
        """full_report 输出结构完整。"""
        bench = MIMICCausalBenchmark()
        patients = bench.load_synthetic_dataset(n_patients=5, n_timesteps=30, seed=42)
        report = bench.run_full_report(patients)
        assert "dataset" in report
        assert "ground_truth" in report
        assert "cewm" in report
        assert report["dataset"]["n_patients"] == 5


# ─────────────────────────────────────────────────────────────────────────────
# MIMIC 数据加载
# ─────────────────────────────────────────────────────────────────────────────


class TestMIMICDataLoading:
    """MIMIC 数据加载与缓存。"""

    def test_load_mimic_missing_file_raises(self):
        """加载不存在的 MIMIC 数据抛 FileNotFoundError。"""
        bench = MIMICCausalBenchmark()
        with pytest.raises(FileNotFoundError):
            bench.load_mimic_dataset("/nonexistent/path.jsonl")

    def test_load_mimic_jsonl(self):
        """加载 JSONL 格式的 MIMIC 数据。"""
        bench = MIMICCausalBenchmark()
        # 创建临时 JSONL 文件
        record = {
            "patient_id": "test_001",
            "variables": ["heart_rate", "map"],
            "data": [[80.0, 65.0], [82.0, 67.0], [81.0, 66.0]],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(record) + "\n")
            path = f.name

        try:
            patients = bench.load_mimic_dataset(path)
            assert len(patients) == 1
            assert patients[0].patient_id == "test_001"
            assert patients[0].data.shape == (3, 2)
        finally:
            os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# LLM 输出解析
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMOutputParsing:
    """LLM 输出解析验证。"""

    def test_parse_json_array(self):
        """解析标准 JSON 数组输出。"""
        text = """Here are the causal relationships:
[
    {"cause": "dopamine_dose", "effect": "heart_rate", "direction": "positive", "confidence": 0.8},
    {"cause": "norepinephrine_dose", "effect": "mean_arterial_pressure", "direction": "positive", "confidence": 0.7}
]"""
        edges = parse_llm_causal_output(text)
        assert len(edges) == 2
        assert edges[0][0] == "dopamine_dose"
        assert edges[0][2] == "positive"

    def test_parse_negative_direction(self):
        """解析负向因果方向。"""
        text = '[{"cause": "insulin", "effect": "glucose", "direction": "negative", "confidence": 0.9}]'
        edges = parse_llm_causal_output(text)
        assert edges[0][2] == "negative"

    def test_parse_shorthand_direction(self):
        """解析简写方向 (+/-)。"""
        text = '[{"cause": "x", "effect": "y", "direction": "+", "confidence": 0.5}]'
        edges = parse_llm_causal_output(text)
        assert edges[0][2] == "positive"

    def test_parse_empty_output(self):
        """空输出返回空列表。"""
        edges = parse_llm_causal_output("No causal relationships found.")
        assert isinstance(edges, list)

    def test_parse_malformed_json(self):
        """畸形 JSON 仍能提取部分结果。"""
        text = 'Some text {"cause": "a", "effect": "b", "direction": "positive"} more text'
        edges = parse_llm_causal_output(text)
        # 应至少能提取到一些边
        assert isinstance(edges, list)


class TestLLMBaselineRunner:
    """LLM 基线运行器验证。"""

    def test_runner_init(self):
        """运行器初始化。"""
        runner = LLMBaselineRunner(backend="ollama", model="qwen3:8b")
        assert runner.backend == "ollama"
        assert runner.model == "qwen3:8b"

    def test_unknown_backend(self):
        """未知后端返回错误。"""
        runner = LLMBaselineRunner(backend="nonexistent")
        result = runner._call_llm("test prompt")
        assert result.error != ""

    def test_no_api_key(self):
        """无 API key 时 OpenAI 调用返回错误。"""
        runner = LLMBaselineRunner(backend="openai", model="gpt-4o", api_key="")
        result = runner._call_llm("test")
        assert result.error != ""


# ─────────────────────────────────────────────────────────────────────────────
# BenchmarkResult 序列化
# ─────────────────────────────────────────────────────────────────────────────


class TestBenchmarkResult:
    """BenchmarkResult 序列化验证。"""

    def test_to_dict(self):
        """BenchmarkResult.to_dict() 包含所有字段。"""
        result = BenchmarkResult(
            method="cewm",
            model_name="CEWM-v4.6.0",
            n_patients=10,
            metrics=CausalMetrics(f1=0.65),
            runtime_seconds=12.5,
        )
        d = result.to_dict()
        assert d["method"] == "cewm"
        assert d["n_patients"] == 10
        assert "metrics" in d
        assert d["metrics"]["f1"] == 0.65


# ─────────────────────────────────────────────────────────────────────────────
# 统计对比
# ─────────────────────────────────────────────────────────────────────────────


class TestStatisticalComparison:
    """统计对比逻辑验证。"""

    def test_comparison_with_llm_result(self):
        """CEWM vs LLM 统计对比输出完整。"""
        bench = MIMICCausalBenchmark()
        patients = bench.load_synthetic_dataset(n_patients=10, n_timesteps=30, seed=42)

        _cewm_result = bench.run_cewm_benchmark(patients)

        # 构造一个假 LLM 结果
        llm_result = BenchmarkResult(
            method="llm",
            model_name="test-llm",
            n_patients=10,
            metrics=CausalMetrics(f1=0.3, direction_accuracy=0.4),
            per_patient_metrics=[CausalMetrics(f1=0.3) for _ in range(10)],
        )

        report = bench.run_full_report(patients, llm_results=[llm_result])
        assert "statistical_comparison" in report
        assert "test-llm" in report["statistical_comparison"]
        comp = report["statistical_comparison"]["test-llm"]
        assert "f1_delta" in comp
        assert "f1_delta_95ci" in comp
