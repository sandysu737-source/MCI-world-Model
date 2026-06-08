"""
Benchmark 共享 fixtures 与 CI 跳过逻辑。

CI 环境中自动跳过 benchmark 测试（--benchmark-skip），
本地开发运行: pytest benchmarks/ -m benchmark --benchmark-only
"""

from __future__ import annotations

import numpy as np
import pytest


def generate_synthetic_patient(seed: int = 42, n_days: int = 30) -> list[dict]:
    """生成合成患者时序数据。"""
    rng = np.random.default_rng(seed)
    alb_base = rng.uniform(30, 40)
    prealb_base = rng.uniform(200, 300)
    cal_base = rng.uniform(1200, 1800)
    prot_base = rng.uniform(50, 80)
    med_base = rng.uniform(100, 300)
    nrs_base = rng.uniform(2, 4)

    alb = max(25, min(45, alb_base + rng.normal(0, 1.5)))
    prealb = max(100, min(400, prealb_base + rng.normal(0, 15)))
    calorie = cal_base
    protein = prot_base
    med = med_base
    nrs = nrs_base
    weight = rng.uniform(50, 90)

    cal_hist: list[float] = [cal_base] * 7
    prot_hist: list[float] = [prot_base] * 7
    med_hist: list[float] = [med_base] * 7
    alb_hist: list[float] = [alb] * 7

    timeline: list[dict] = []
    for day in range(1, n_days + 1):
        calorie = max(800, min(2500, cal_base + rng.normal(0, 150)))
        protein = max(30, min(120, prot_base + rng.normal(0, 10)))
        med = max(0, min(500, med_base + rng.normal(0, 40)))

        cal_hist.append(calorie)
        cal_hist.pop(0)
        prot_hist.append(protein)
        prot_hist.pop(0)
        med_hist.append(med)
        med_hist.pop(0)
        alb_hist.append(alb)
        alb_hist.pop(0)

        cal_effect = 0.6 * (cal_hist[-4] - cal_base) / 150
        med_effect = 0.4 * (med_hist[-2] - med_base) / 40
        prot_effect_prealb = 0.5 * (prot_hist[-3] - prot_base) / 10
        nrs_effect = -0.3 * (alb_hist[0] - alb_base) / 1.5 if day > 7 else 0

        alb = max(25, min(45, alb_base + cal_effect * 1.5 + med_effect * 1.5 + rng.normal(0, 0.8)))
        prealb = max(100, min(400, prealb_base + prot_effect_prealb * 15 + rng.normal(0, 8)))
        nrs = max(0, min(7, nrs_base + nrs_effect + rng.normal(0, 0.3)))
        weight += rng.normal(0, 0.3) + 0.05 * (calorie - cal_base) / 150

        timeline.append({
            "day": day,
            "albumin": round(alb, 1),
            "prealbumin": round(prealb, 1),
            "calorie_intake": round(calorie, 0),
            "protein_intake": round(protein, 1),
            "medication_dose": round(med, 1),
            "nrs2002_score": round(nrs, 1),
            "body_weight": round(weight, 1),
        })

    return timeline


@pytest.fixture(scope="module")
def patient_timeline_30d():
    """30 天合成患者时序数据。"""
    return generate_synthetic_patient(seed=42, n_days=30)


@pytest.fixture(scope="module")
def counterfactual_engine():
    """预构建反事实引擎。"""
    from mci_world_model.sdk._counterfactual import CounterfactualEngine
    from mci_world_model.sdk._do_calculus import CausalGraph

    cg = CausalGraph()
    cg.add_edge("calorie_intake", "albumin", weight=0.6)
    cg.add_edge("medication_dose", "albumin", weight=0.4)
    cg.add_edge("protein_intake", "prealbumin", weight=0.5)
    cg.add_edge("albumin", "nrs2002_score", weight=-0.3)
    cg.add_edge("calorie_intake", "body_weight", weight=0.35)

    sem = cg.to_sem(noise_std=0.2, activation="linear", seed=42)
    return CounterfactualEngine(sem, list(sem.node_names))


@pytest.fixture(scope="module")
def multimodal_signals():
    """预构建 MultimodalSignal 列表。"""
    from mci_world_model._sys._perception_pipeline import MultimodalSignal, SignalType

    rng = np.random.default_rng(42)
    signals = []
    for day in range(1, 31):
        for name in ("albumin", "calorie_intake", "nrs2002_score", "prealbumin"):
            signals.append(MultimodalSignal(
                signal_type=SignalType.NUMERICAL,
                value=float(rng.uniform(25, 45)) if name == "albumin" else float(rng.uniform(800, 2500)),
                timestamp=f"day_{day}",
                source="lab_report",
                metadata={"name": name},
            ))
    return signals
