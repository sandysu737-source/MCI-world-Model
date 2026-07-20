"""医疗世界模型端到端 ICU 演示。

展示完整的五要素闭环在三个临床场景中的应用：
    场景1: 心动过速 — 规划器推荐β受体阻滞剂
    场景2: 低血压休克 — 规划器推荐升压药
    场景3: 正常患者 — 规划器不推荐干预

用法: PYTHONPATH=src:. python scripts/demo_icu_world_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from mci_world_model.sdk import ClinicalDecisionEngine
from mci_world_model.sdk._clinical_world_state import VITAL_NAMES

SEED = 42


def print_decision(decision, scenario_name: str) -> None:
    """格式化打印决策结果。"""
    print(f"\n{'─' * 60}")
    print(f"📋 场景: {scenario_name}")
    print(f"{'─' * 60}")

    # 患者状态
    state = decision.patient_state
    if state:
        latest = state.vital_signs[-1]
        print("  患者状态:")
        for i, vname in enumerate(VITAL_NAMES):
            val = latest[i]
            print(f"    {vname:20s}: {val:6.1f}")
        print(f"    SOFA 评分: {state.sofa_score():.1f}")
        print(f"    安全状态: {'✅ 正常' if state.is_safe() else '⚠️ 异常'}")

    # 路由
    print(f"\n  路由决策: {decision.route_type.value} (置信度 {decision.route_confidence:.2f})")

    # 知识检索（semantic 路径）
    if decision.knowledge_answer and decision.knowledge_answer.get("term"):
        ka = decision.knowledge_answer
        print(
            f"  📚 知识检索: {ka['term']} (方法={ka.get('retrieval_method', '?')}, "
            f"相似度={ka.get('retrieval_score', 0):.3f})"
        )
        print(f"     定义: {ka.get('definition', '')[:60]}...")
        print(f"     指南: {ka.get('guideline', '')[:60]}...")

    # 推荐
    print(f"\n  评估: 当前 reward={decision.current_reward:.3f}")
    if decision.uncertainty_score > 0:
        print(f"  不确定性: {decision.uncertainty_score:.4f} (阈值 0.3)")
    if decision.recommended_action:
        a = decision.recommended_action
        print(f"  推荐: {a.target} {a.magnitude}{a.unit} ({a.route})")
        print(
            f"  预测: 治疗后 reward={decision.predicted_reward:.3f} "
            f"(改善 {decision.predicted_reward - decision.current_reward:+.3f})"
        )

        # Top 3 方案对比
        if decision.treatment_plan:
            print("\n  方案对比 (Top 3):")
            for c in decision.treatment_plan.to_dict()["top_3_actions"]:
                safe = "✅" if c["is_safe"] else "❌"
                print(f"    {safe} {c['action']:30s} reward={c['predicted_reward']:.3f} (Δ{c['reward_delta']:+.3f})")
    else:
        print("  推荐: 无需干预")

    # 安全
    review = "⚠️ 需医师复核" if decision.need_review else "✅ 可信输出"
    print(f"\n  安全等级: {decision.safety_level.value} — {review}")

    # 审计
    print(f"  审计步数: {len(decision.audit_trail)}")


def main() -> None:
    print("=" * 60)
    print("🏥 医疗世界模型 — ICU 端到端决策演示")
    print("=" * 60)

    # 初始化并训练
    print("\n初始化决策引擎...")
    engine = ClinicalDecisionEngine()
    print("训练世界模型转移 T（药效基线表，2000样本×500轮）...")
    info = engine.fit(n_samples=2000, n_epochs=500, lr=0.005, seed=SEED)
    print(f"  训练完成: loss={info['final_loss']:.4f}")

    # 三个临床场景
    scenarios = [
        {
            "name": "心动过速（HR=130）",
            "vitals": [
                {
                    "heart_rate": 130,
                    "systolic_bp": 140,
                    "diastolic_bp": 95,
                    "oxygen_saturation": 96,
                    "respiratory_rate": 20,
                    "temperature": 37.0,
                    "gcs": 15,
                }
            ],
            "query": "患者心动过速，需要什么药物治疗？",
            "patient_id": "ICU-001",
        },
        {
            "name": "低血压休克（SBP=75）",
            "vitals": [
                {
                    "heart_rate": 110,
                    "systolic_bp": 75,
                    "diastolic_bp": 45,
                    "oxygen_saturation": 92,
                    "respiratory_rate": 22,
                    "temperature": 36.5,
                    "gcs": 14,
                }
            ],
            "query": "患者血压偏低，如何升压处理？",
            "patient_id": "ICU-002",
        },
        {
            "name": "正常体征（对照组）",
            "vitals": [
                {
                    "heart_rate": 75,
                    "systolic_bp": 120,
                    "diastolic_bp": 80,
                    "oxygen_saturation": 98,
                    "respiratory_rate": 16,
                    "temperature": 36.8,
                    "gcs": 15,
                }
            ],
            "query": "患者生命体征监测",
            "patient_id": "ICU-003",
        },
        {
            "name": "知识问答（去甲肾上腺素）",
            "vitals": [
                {
                    "heart_rate": 85,
                    "systolic_bp": 105,
                    "diastolic_bp": 65,
                    "oxygen_saturation": 97,
                    "respiratory_rate": 18,
                    "temperature": 36.6,
                    "gcs": 15,
                }
            ],
            "query": "去甲肾上腺素的适应症和用药原则是什么",
            "patient_id": "ICU-004",
        },
    ]

    results = []
    for scenario in scenarios:
        decision = engine.decide_from_vitals(
            vital_records=scenario["vitals"],
            query=scenario["query"],
            patient_id=scenario["patient_id"],
        )
        print_decision(decision, scenario["name"])
        results.append({"scenario": scenario["name"], "decision": decision.to_dict()})

    # 汇总
    print(f"\n{'=' * 60}")
    print("📊 汇总")
    print(f"{'=' * 60}")
    print(f"{'场景':<22} {'路由':<10} {'推荐/知识':<22} {'Δreward':>8} {'不确定':>8} {'安全':>12}")
    print("-" * 82)
    for r in results:
        d = r["decision"]
        if d["recommended_action"]:
            action = d["recommended_action"]["target"][:20]
        elif d.get("knowledge_answer", {}) and d["knowledge_answer"].get("term"):
            action = "📚" + d["knowledge_answer"]["term"][:18]
        else:
            action = "无"
        delta = d["evaluation"]["improvement"]
        unc = d["evaluation"].get("uncertainty_score", 0)
        safe = d["route"]["safety"]
        print(f"{r['scenario']:<22} {d['route']['type']:<10} {action:<22} {delta:>+7.3f} {unc:>7.3f} {safe:>12}")

    # 保存结果
    out_path = Path("docs/demo-icu-world-model-result.json")
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
