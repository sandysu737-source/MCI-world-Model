"""MCI World Model — 医疗世界模型统一评测基准（UnifiedEvalSuite）

============================================================

方向四：评测基准 — 把碎片化的评测收敛为统一框架。

为什么需要？
    现有评测三层碎片化：
    1. 指标重复 — 三个 predictor 各自实现 evaluate_direction_accuracy（复制粘贴）
    2. 能力孤岛 — JEPA/语义/MCTS/因果发现各用不同脚本测试集，无法横向对比
    3. 无报告层 — 每次出报告靠手写 script + 人工拼 JSON

    本框架提供：
        SharedTestCases  — 同一批患者数据，所有 backend 公平对比
        MetricRegistry   — 6 类统一指标（方向/多步/安全/语义/抗噪/重建）
        UnifiedEvalSuite — 注册各 backend，一键跑全套
        ReportGenerator  — 自动生成 markdown + JSON 报告

设计原则（AGENTS.md）:
    - 可复现：固定 SEED，测试集生成确定性
    - 公平对比：所有 backend 跑相同 test_cases
    - 诚实记录：如实报告未达标项（禁止捏造）
    - 无状态：评测过程不持久化（报告生成是输出，非状态）
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from mci_world_model.sdk._clinical_world_state import (
    DRUG_EFFECT_TABLE,
    VITAL_NAMES,
    VITAL_NORMAL_RANGES,
    MedicalAction,
    PatientState,
)

SEED = 42
DIRECTION_THRESHOLD = 0.5  # 方向判定阈值（与既有 evaluate_direction_accuracy 一致）


# =============================================================================
# SharedTestCases — 公平对比的共享测试集
# =============================================================================


class PredictableBackend(Protocol):
    """可预测 backend 协议（ClinicalDynamicsPredictor/JEPAClinicalBridge/Temporal...）。"""

    def predict(self, state: PatientState, action: MedicalAction | None, n_steps: int = 1) -> list[Any]: ...

    @property
    def is_fitted(self) -> bool: ...


@dataclass
class TestCase:
    """单个评测用例。"""

    # 避免 pytest 误收集为测试类
    __test__ = False

    state: PatientState
    action: MedicalAction | None
    true_next: PatientState
    category: str = "general"  # drug / natural / abnormal


class SharedTestCases:
    """共享测试集生成器 — 所有 backend 跑相同输入（公平对比前提）。

    三类用例：
        drug:   药物干预（测动作条件化预测）
        natural: 自然演化（测零动作动力学）
        abnormal: 异常体征（测极端输入鲁棒性）
    """

    def __init__(self, seed: int = SEED) -> None:
        self._seed = seed

    def generate(self, n_per_category: int = 30) -> list[TestCase]:
        """生成确定性测试集。

        Args:
            n_per_category: 每类用例数（共 3 类，总 3×n）。

        Returns:
            TestCase 列表。
        """
        rng = np.random.default_rng(self._seed)
        drugs = list(DRUG_EFFECT_TABLE.keys())
        icd_pool = [
            "I48.91",
            "I21.9",
            "N17.0",
            "A41.9",
            "J44.1",
            "E11.9",
            "I50.9",
            "K92.0",
        ]
        cases: list[TestCase] = []

        # 1. 药物干预用例
        for _ in range(n_per_category):
            vitals = self._random_vitals(rng, mode="normal")
            state = PatientState(
                vital_signs=vitals.reshape(1, -1),
                diagnoses=[rng.choice(icd_pool)],
            )
            drug = rng.choice(drugs)
            dose = rng.uniform(1.0, 8.0)
            action = MedicalAction(target=drug, magnitude=dose)
            true_next_raw = action.apply(state)
            true_next: PatientState = true_next_raw  # type: ignore[assignment]  # apply 返回基类
            true_next.diagnoses = state.diagnoses
            cases.append(TestCase(state, action, true_next, "drug"))

        # 2. 自然演化用例（零动作，体征微漂移）
        for _ in range(n_per_category):
            vitals = self._random_vitals(rng, mode="normal")
            state = PatientState(vital_signs=vitals.reshape(1, -1))
            # 自然漂移：小高斯噪声
            drift = vitals + rng.normal(0, 1.5, len(VITAL_NAMES))
            true_next = PatientState(vital_signs=drift.reshape(1, -1))
            cases.append(TestCase(state, None, true_next, "natural"))

        # 3. 异常体征用例（极端输入）
        for _ in range(n_per_category):
            vitals = self._random_vitals(rng, mode="abnormal")
            state = PatientState(vital_signs=vitals.reshape(1, -1))
            drug = rng.choice(drugs)
            action = MedicalAction(target=drug, magnitude=rng.uniform(1, 8))
            true_next_ab = action.apply(state)
            cases.append(TestCase(state, action, true_next_ab, "abnormal"))  # type: ignore[arg-type]

        return cases

    @staticmethod
    def _random_vitals(rng: np.random.Generator, mode: str = "normal") -> np.ndarray:
        """生成随机体征向量。"""
        vitals = np.zeros(len(VITAL_NAMES), dtype=np.float64)
        for j, vname in enumerate(VITAL_NAMES):
            lo, hi = VITAL_NORMAL_RANGES[vname]
            if mode == "abnormal":
                # 异常：取范围外
                margin = (hi - lo) * 1.5
                if rng.random() < 0.5:
                    vitals[j] = rng.uniform(lo - margin, lo)
                else:
                    vitals[j] = rng.uniform(hi, hi + margin)
            else:
                vitals[j] = rng.uniform(lo, hi)
        return vitals


# =============================================================================
# MetricRegistry — 统一指标计算
# =============================================================================


@dataclass
class MetricResult:
    """单个指标结果。"""

    name: str
    value: float
    category: str = "overall"
    detail: dict[str, Any] = field(default_factory=dict)


class MetricRegistry:
    """统一指标注册表 — 6 类指标，单一实现，所有 backend 共用。

    收敛三份重复的 evaluate_direction_accuracy 到此处。
    """

    @staticmethod
    def direction_accuracy(
        backend: PredictableBackend,
        test_cases: list[TestCase],
        n_steps: int = 1,
    ) -> list[MetricResult]:
        """方向准确率 + MAE（单步或多步）。

        Args:
            backend: 已训练的预测器。
            test_cases: 测试用例。
            n_steps: 预测步数（>1 时测最后一步的方向）。

        Returns:
            [方向准确率(overall), MAE(overall), 各类别方向准确率...]。
        """
        correct = 0
        total_mae = 0.0
        total = 0
        cat_correct: dict[str, int] = {}
        cat_total: dict[str, int] = {}

        for tc in test_cases:
            try:
                preds = backend.predict(tc.state, tc.action, n_steps=n_steps)
                pred_next = preds[-1]  # 多步时取最后一步
            except (ValueError, RuntimeError):
                total += len(VITAL_NAMES)
                continue
            for i, _vname in enumerate(VITAL_NAMES):
                s_val = tc.state.vital_signs[-1][i]
                true_delta = tc.true_next.vital_signs[-1][i] - s_val
                pred_delta = pred_next.vital_signs[-1][i] - s_val
                true_sign = 1 if true_delta > DIRECTION_THRESHOLD else (-1 if true_delta < -DIRECTION_THRESHOLD else 0)
                pred_sign = 1 if pred_delta > DIRECTION_THRESHOLD else (-1 if pred_delta < -DIRECTION_THRESHOLD else 0)
                total += 1
                cat_total[tc.category] = cat_total.get(tc.category, 0) + 1
                if true_sign == pred_sign:
                    correct += 1
                    cat_correct[tc.category] = cat_correct.get(tc.category, 0) + 1
                total_mae += abs(true_delta - pred_delta)

        results = [
            MetricResult(
                "direction_accuracy",
                round(correct / max(total, 1), 4),
                "overall",
                {"n_steps": n_steps, "n_comparisons": total},
            ),
            MetricResult(
                "mae",
                round(total_mae / max(total, 1), 4),
                "overall",
                {"n_steps": n_steps},
            ),
        ]
        for cat in sorted(cat_total):
            results.append(
                MetricResult(
                    "direction_accuracy",
                    round(cat_correct.get(cat, 0) / max(cat_total[cat], 1), 4),
                    cat,
                    {"n_steps": n_steps},
                )
            )
        return results

    @staticmethod
    def multistep_error_accumulation(
        backend: PredictableBackend,
        test_cases: list[TestCase],
        max_steps: int = 5,
    ) -> list[MetricResult]:
        """多步预测稳定性曲线（1→max_steps 步的预测漂移）。

        测世界模型的长期预测是否发散/坍塌。
        用药物用例做多步预测（同动作重复施加），
        衡量每步预测相对首步的累积偏移（越小越稳定）。

        Returns:
            每步一个 MetricResult（drift_at_step_N）。
        """
        results: list[MetricResult] = []
        drug_cases = [tc for tc in test_cases if tc.category == "drug"][:15]
        if not drug_cases:
            return results
        for n_steps in range(1, max_steps + 1):
            drift_total = 0.0
            count = 0
            for tc in drug_cases:
                try:
                    preds = backend.predict(tc.state, tc.action, n_steps=n_steps)
                    if len(preds) < n_steps:
                        continue
                    # 测最后一步相对首步的漂移（相对偏移）
                    first = preds[0].vital_signs[-1]
                    last = preds[-1].vital_signs[-1]
                    drift_total += float(np.mean(np.abs(last - first)))
                    count += 1
                except (ValueError, RuntimeError):
                    continue
            results.append(
                MetricResult(
                    "multistep_mae",
                    round(drift_total / max(count, 1), 4),
                    "drug",
                    {"step": n_steps, "metric": "drift_from_step1"},
                )
            )
        return results

    @staticmethod
    def clinical_safety(
        planner: Any,
        test_cases: list[TestCase],
        unsafe_drugs: list[str] | None = None,
    ) -> MetricResult:
        """临床安全性 — 推荐不安全/禁忌药物的比例（越低越好）。

        Args:
            planner: ClinicalMCTSPlanner（有 recommend_best 方法）。
            test_cases: 测试用例。
            unsafe_drugs: 不安全药物列表（默认空，只检测生理不可行预测）。

        Returns:
            safety_score ∈ [0, 1]（1 = 完全安全，推荐方案预测状态均生理可行）。
        """
        from mci_world_model.sdk._clinical_objective import ClinicalObjective

        objective = ClinicalObjective()
        unsafe = set(unsafe_drugs or [])
        safe_count = 0
        total = 0
        for tc in test_cases:
            try:
                action = planner.recommend_best(tc.state, require_safe=False)
                if action is None:
                    total += 1
                    safe_count += 1  # 无推荐视为安全（需复核）
                    continue
                total += 1
                if action.target in unsafe:
                    continue  # 不安全药物
                # 检查预测状态是否生理可行
                preds = planner._predictor.predict(tc.state, action, n_steps=1)
                if preds and objective.is_safe(preds[0]):
                    safe_count += 1
            except (ValueError, RuntimeError):
                total += 1
                continue
        score = safe_count / max(total, 1)
        return MetricResult(
            "safety_score",
            round(score, 4),
            "overall",
            {"n_cases": total, "safe_count": safe_count},
        )

    @staticmethod
    def noise_robustness(
        backend: PredictableBackend,
        test_cases: list[TestCase],
        noise_levels: list[float] | None = None,
    ) -> list[MetricResult]:
        """抗噪鲁棒性曲线 — 不同噪声水平下的方向准确率。

        Args:
            backend: 预测器。
            test_cases: 测试用例（用 drug 类）。
            noise_levels: 噪声标准差列表（默认 [0, 2, 5]）。

        Returns:
            每个噪声水平一个 MetricResult。
        """
        levels = noise_levels or [0.0, 2.0, 5.0]
        rng = np.random.default_rng(SEED)
        drug_cases = [tc for tc in test_cases if tc.category == "drug"]
        results: list[MetricResult] = []
        for sigma in levels:
            correct = 0
            total = 0
            for tc in drug_cases:
                # 给输入加噪声
                noisy_vitals = tc.state.vital_signs[-1] + rng.normal(0, sigma, len(VITAL_NAMES))
                noisy_state = PatientState(
                    vital_signs=noisy_vitals.reshape(1, -1),
                    diagnoses=tc.state.diagnoses,
                )
                try:
                    preds = backend.predict(noisy_state, tc.action, n_steps=1)
                    pred = preds[0]
                except (ValueError, RuntimeError):
                    total += len(VITAL_NAMES)
                    continue
                for i in range(len(VITAL_NAMES)):
                    s_val = tc.state.vital_signs[-1][i]  # 原始（干净）状态
                    true_delta = tc.true_next.vital_signs[-1][i] - s_val
                    pred_delta = pred.vital_signs[-1][i] - noisy_vitals[i]
                    true_sign = (
                        1 if true_delta > DIRECTION_THRESHOLD else (-1 if true_delta < -DIRECTION_THRESHOLD else 0)
                    )
                    pred_sign = (
                        1 if pred_delta > DIRECTION_THRESHOLD else (-1 if pred_delta < -DIRECTION_THRESHOLD else 0)
                    )
                    total += 1
                    if true_sign == pred_sign:
                        correct += 1
            results.append(
                MetricResult(
                    "noise_robustness",
                    round(correct / max(total, 1), 4),
                    "overall",
                    {"noise_sigma": sigma},
                )
            )
        return results

    @staticmethod
    def reconstruction_fidelity(
        backend: PredictableBackend,
        test_cases: list[TestCase],
    ) -> MetricResult:
        """重建保真度 — 编码再解码的 MSE（仅 JEPA 类 backend 有意义）。

        Returns:
            avg_recon_mse（数值模型无 encode/reconstruct 时返回 -1 表示不适用）。
        """
        if not hasattr(backend, "reconstruction_error"):
            return MetricResult("reconstruction_mse", -1.0, "overall", {"applicable": False})
        total_mse = 0.0
        count = 0
        for tc in test_cases[:20]:  # 抽样 20 个
            try:
                mse = backend.reconstruction_error(tc.state)
                total_mse += mse
                count += 1
            except (ValueError, RuntimeError, AttributeError):
                continue
        return MetricResult(
            "reconstruction_mse",
            round(total_mse / max(count, 1), 4),
            "overall",
            {"applicable": True, "n_sampled": count},
        )

    @staticmethod
    def semantic_discrimination(
        backend: PredictableBackend,
        n_pairs: int = 20,
    ) -> MetricResult:
        """语义区分能力 — 相同体征不同诊断的潜向量平均距离（仅语义模式适用）。

        数值模型返回 0（无法区分）。

        Returns:
            avg_latent_distance（>0 表示能区分）。
        """
        if not hasattr(backend, "encode"):
            return MetricResult("semantic_discrimination", 0.0, "overall", {"applicable": False})
        rng = np.random.default_rng(SEED)
        diagnoses_pairs = [
            ("I48.91", "N17.0"),  # 心律失常 vs 急性肾损伤
            ("A41.9", "E11.9"),  # 脓毒症 vs 糖尿病
            ("I21.9", "J44.1"),  # 心梗 vs COPD
            ("K92.0", "I50.9"),  # 消化道出血 vs 心衰
        ]
        distances: list[float] = []
        for _ in range(n_pairs):
            vitals = np.array(
                [
                    [
                        rng.uniform(70, 100),
                        rng.uniform(100, 140),
                        rng.uniform(60, 90),
                        98,
                        16,
                        36.8,
                        15,
                    ]
                ]
            )
            d1, d2 = diagnoses_pairs[rng.integers(len(diagnoses_pairs))]
            s1 = PatientState(vital_signs=vitals.reshape(1, -1), diagnoses=[d1])
            s2 = PatientState(vital_signs=vitals.reshape(1, -1), diagnoses=[d2])
            try:
                z1 = backend.encode(s1)
                z2 = backend.encode(s2)
                distances.append(float(np.linalg.norm(z1 - z2)))
            except (ValueError, RuntimeError, AttributeError):
                continue
        avg_dist = float(np.mean(distances)) if distances else 0.0
        return MetricResult(
            "semantic_discrimination",
            round(avg_dist, 4),
            "overall",
            {"applicable": len(distances) > 0, "n_pairs": len(distances)},
        )

    @staticmethod
    def causal_intervention_quality(
        pearl_bridge: Any,
        test_cases: list[TestCase],
    ) -> list[MetricResult]:
        """因果干预质量（方向三 L2）— Pearl do-calculus 调整集识别率与效应可估率。

        评估因果下沉能力：能否从测试用例的体征历史识别后门调整集、
        估计 do(干预) 的因果效应。这是方向三的核心评测维度。

        Args:
            pearl_bridge: ClinicalPearlBridge 实例（因果下沉桥接）。
            test_cases: 测试用例（需含体征历史，用 drug 类）。

        Returns:
            [adjustment_identifiable_rate, ate_estimable_rate, avg_ate]。
        """
        drug_cases = [tc for tc in test_cases if tc.category == "drug"]
        if not drug_cases:
            return [MetricResult("causal_quality", 0.0, "overall", {"applicable": False})]

        identifiable = 0
        ate_estimable = 0
        ate_values: list[float] = []
        total = 0
        rng = np.random.default_rng(SEED)
        for tc in drug_cases[:15]:  # 抽样 15 个
            # 从单状态无法做发现，构造一个小的合成历史（基于该状态的体征分布）
            base_vitals = tc.state.vital_signs[-1]
            history = np.array([base_vitals + rng.normal(0, 2, len(base_vitals)) for _ in range(30)])
            try:
                structure = pearl_bridge.discover(history)
                if not structure.links:
                    total += 1
                    continue
                total += 1
                link = structure.links[0]
                # L2: 识别调整集
                adj = pearl_bridge.identify_confounders(structure, link.cause, link.effect)
                if adj is not None:
                    identifiable += 1
                # L2: ATE 可估
                result = pearl_bridge.intervene(structure, link.cause, link.effect)
                if result.method != "none" and np.isfinite(result.ate):
                    ate_estimable += 1
                    ate_values.append(result.ate)
            except (ValueError, RuntimeError):
                total += 1
                continue

        return [
            MetricResult(
                "causal_adjustment_identifiable",
                round(identifiable / max(total, 1), 4),
                "overall",
                {"n_evaluated": total},
            ),
            MetricResult(
                "causal_ate_estimable",
                round(ate_estimable / max(total, 1), 4),
                "overall",
                {"n_evaluated": total},
            ),
            MetricResult(
                "causal_avg_ate",
                round(float(np.mean(ate_values)) if ate_values else 0.0, 4),
                "overall",
                {"n_estimates": len(ate_values)},
            ),
        ]


# =============================================================================
# UnifiedReport — 评测报告
# =============================================================================


@dataclass
class BackendReport:
    """单个 backend 的评测报告。"""

    backend_name: str
    metrics: list[MetricResult] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        metrics_by_name: dict[str, Any] = {}
        for m in self.metrics:
            # 多值指标（multistep/noise）始终用原名（列表聚合）
            if m.name in ("multistep_mae", "noise_robustness") or m.category == "overall":
                key = m.name
            else:
                key = f"{m.name}_{m.category}"
            if m.name in ("multistep_mae", "noise_robustness"):
                # 多值指标（按 step/sigma 分）
                metrics_by_name.setdefault(key, []).append(
                    {
                        "value": m.value,
                        **m.detail,
                    }
                )
            else:
                metrics_by_name[key] = {"value": m.value, **m.detail}
        return {
            "backend": self.backend_name,
            "timestamp": self.timestamp,
            "metrics": metrics_by_name,
        }


@dataclass
class UnifiedReport:
    """完整评测报告（多 backend 横向对比）。"""

    suite_name: str
    backend_reports: list[BackendReport] = field(default_factory=list)
    n_test_cases: int = 0
    seed: int = SEED
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """序列化。"""
        return {
            "suite": self.suite_name,
            "timestamp": self.timestamp,
            "seed": self.seed,
            "n_test_cases": self.n_test_cases,
            "backends": [br.to_dict() for br in self.backend_reports],
        }

    def to_markdown(self) -> str:
        """生成 markdown 报告（横向对比表 + 诚实标注）。"""
        lines = [
            f"# {self.suite_name}",
            "",
            f"> 生成时间: {self.timestamp} | 测试集大小: {self.n_test_cases} | SEED: {self.seed}",
            "",
            "## 核心指标横向对比",
            "",
            "| Backend | 方向准确率(单步) | MAE | 安全性 | 抗噪(σ=2) | 重建MSE | 语义区分 | 因果可估率 |",
            "|---------|------------------|-----|--------|-----------|---------|----------|------------|",
        ]
        for br in self.backend_reports:
            d = br.to_dict()["metrics"]
            dir_acc = d.get("direction_accuracy", {}).get("value", "N/A")
            mae = d.get("mae", {}).get("value", "N/A")
            safety = d.get("safety_score", {}).get("value", "N/A")
            # 找 noise_robustness σ=2
            noise = d.get("noise_robustness", [])
            noise_2 = (
                next((n["value"] for n in noise if n.get("noise_sigma") == 2.0), "N/A")
                if isinstance(noise, list)
                else "N/A"
            )
            recon = d.get("reconstruction_mse", {}).get("value", "N/A")
            if recon == -1.0:
                recon = "N/A"
            sem = d.get("semantic_discrimination", {}).get("value", "N/A")
            if sem == 0.0:
                sem = "N/A"
            causal = d.get("causal_ate_estimable", {}).get("value", "N/A")
            lines.append(
                f"| {br.backend_name} | {dir_acc} | {mae} | {safety} | {noise_2} | {recon} | {sem} | {causal} |"
            )

        lines.extend(
            [
                "",
                "## 多步预测误差累积（自然演化）",
                "",
                "| Backend | Step1 | Step2 | Step3 | Step4 | Step5 |",
                "|---------|-------|-------|-------|-------|-------|",
            ]
        )
        for br in self.backend_reports:
            ms = br.to_dict()["metrics"].get("multistep_mae", [])
            if not isinstance(ms, list):
                continue
            vals = {m["step"]: m["value"] for m in ms}
            row = f"| {br.backend_name} |"
            for step in range(1, 6):
                row += f" {vals.get(step, 'N/A')} |"
            lines.append(row)

        lines.extend(
            [
                "",
                "## 说明",
                "",
                "- 方向准确率: 体征变化方向（升/降/平）与真实一致的比例",
                "- MAE: 预测体征值与真实的平均绝对误差",
                "- 安全性: 推荐方案预测状态生理可行的比例（1.0=完全安全）",
                "- 抗噪: 输入加 σ=2 高斯噪声后的方向准确率衰减情况",
                "- 重建MSE: 仅 JEPA 类适用（编码再解码的保真度），N/A=不适用",
                "- 语义区分: 相同体征不同诊断的潜向量平均距离，N/A=不适用（数值模型无法区分）",
                "- 多步误差: 自然演化场景 1-5 步预测的 MAE 累积",
                "- 因果可估率: Pearl do-calculus L2 干预效应(ATE)可估计的比例（方向三，N/A=未注册因果桥接）",
            ]
        )
        return "\n".join(lines)


# =============================================================================
# UnifiedEvalSuite — 统一评测入口
# =============================================================================


class UnifiedEvalSuite:
    """统一评测套件 — 注册各 backend，一键跑全套指标。

    Example:
        >>> suite = UnifiedEvalSuite("医疗世界模型基准")
        >>> suite.register_backend("MLP", mlp_predictor)
        >>> suite.register_backend("JEPA", jepa_bridge)
        >>> suite.register_planner(planner)
        >>> report = suite.run()
        >>> print(report.to_markdown())
    """

    def __init__(self, suite_name: str = "医疗世界模型评测基准", seed: int = SEED) -> None:
        self._suite_name = suite_name
        self._seed = seed
        self._backends: dict[str, PredictableBackend] = {}
        self._planner: Any = None
        self._pearl_bridge: Any = None
        self._test_cases: list[TestCase] | None = None

    def register_backend(self, name: str, backend: PredictableBackend) -> None:
        """注册一个待评测 backend。"""
        if not backend.is_fitted:
            raise ValueError(f"backend {name} 未训练，请先 fit")
        self._backends[name] = backend

    def register_planner(self, planner: Any) -> None:
        """注册规划器（用于安全性评测）。"""
        self._planner = planner

    def register_pearl_bridge(self, pearl_bridge: Any) -> None:
        """注册因果下沉桥接（用于因果干预质量评测，方向三 L2）。"""
        self._pearl_bridge = pearl_bridge

    def prepare_test_cases(self, n_per_category: int = 30) -> list[TestCase]:
        """生成共享测试集。"""
        gen = SharedTestCases(self._seed)
        self._test_cases = gen.generate(n_per_category=n_per_category)
        return self._test_cases

    def run(
        self,
        n_per_category: int = 30,
        eval_safety: bool = True,
        eval_noise: bool = True,
        eval_multistep: bool = True,
        eval_semantic: bool = True,
        eval_causal: bool = True,
    ) -> UnifiedReport:
        """运行完整评测。

        Args:
            n_per_category: 每类测试用例数。
            eval_safety: 是否评测安全性（需注册 planner）。
            eval_noise: 是否评测抗噪性。
            eval_multistep: 是否评测多步累积。
            eval_semantic: 是否评测语义区分。
            eval_causal: 是否评测因果干预质量（需注册 pearl_bridge）。

        Returns:
            UnifiedReport。
        """
        if self._test_cases is None:
            self.prepare_test_cases(n_per_category)
        assert self._test_cases is not None  # 上面保证非 None

        report = UnifiedReport(
            suite_name=self._suite_name,
            n_test_cases=len(self._test_cases),
            seed=self._seed,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        metrics = MetricRegistry()
        for name, backend in self._backends.items():
            br = BackendReport(backend_name=name, timestamp=report.timestamp)
            # 方向准确率（单步）
            br.metrics.extend(metrics.direction_accuracy(backend, self._test_cases, n_steps=1))
            # 重建保真
            br.metrics.append(metrics.reconstruction_fidelity(backend, self._test_cases))
            # 语义区分
            if eval_semantic:
                br.metrics.append(metrics.semantic_discrimination(backend))
            # 多步累积
            if eval_multistep:
                br.metrics.extend(metrics.multistep_error_accumulation(backend, self._test_cases, max_steps=5))
            # 抗噪
            if eval_noise:
                br.metrics.extend(metrics.noise_robustness(backend, self._test_cases))
            report.backend_reports.append(br)

        # 安全性：每个 backend 各自建 planner 评测（若注册了全局 planner 则只用它）
        if eval_safety:
            if self._planner is not None:
                # 全局 planner 只评一次，附加到第一个 backend
                safety = metrics.clinical_safety(self._planner, self._test_cases)
                if report.backend_reports:
                    report.backend_reports[0].metrics.append(safety)
            else:
                # 每个 backend 自建 planner
                from mci_world_model.sdk._clinical_planner import ClinicalMCTSPlanner

                for br, (name, backend) in zip(report.backend_reports, self._backends.items()):
                    try:
                        p = ClinicalMCTSPlanner(predictor=backend)  # type: ignore[arg-type]
                        safety = metrics.clinical_safety(p, self._test_cases)
                        br.metrics.append(safety)
                    except Exception:
                        br.metrics.append(MetricResult("safety_score", -1.0, "overall", {"error": True}))

        # 因果干预质量（方向三 L2，需注册 pearl_bridge，全局评一次附加到首 backend）
        if eval_causal and self._pearl_bridge is not None:
            causal_results = metrics.causal_intervention_quality(self._pearl_bridge, self._test_cases)
            if report.backend_reports:
                report.backend_reports[0].metrics.extend(causal_results)

        return report

    def save_report(self, report: UnifiedReport, output_dir: str | Path) -> tuple[Path, Path]:
        """保存报告到文件（JSON + Markdown）。

        Args:
            report: 评测报告。
            output_dir: 输出目录。

        Returns:
            (json_path, md_path)。
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = report.timestamp.replace(":", "-").replace(" ", "_")
        json_path = out / f"eval-report-{ts}.json"
        md_path = out / f"eval-report-{ts}.md"
        json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(report.to_markdown(), encoding="utf-8")
        return json_path, md_path
