# Ch13 最终评估总表 — 改进规划书

## 1. 章节概述

原报告第十三章提供了 V4.5.0 → V5.0.0 → 目标的综合评分表。本章职责是**建立自动化评估框架**，使每个维度的评分可自动计算、持续追踪、版本对比。

## 2. 改进目标

| # | 目标 | 量化指标 | 优先级 |
|---|---|---|---|
| G1 | 自动化评分框架 | 8 个维度各有自动评分函数 | P1 |
| G2 | 版本间对比报告 | v5.0.0 vs v5.1.0 vs v5.2.0 diff | P1 |
| G3 | 评分数据持久化 | JSON 存储 + 趋势图 | P2 |

## 3. 实施方案

### 3.1 自动化评分引擎

```python
class MCIScoreEngine:
    """MCI World Model 综合评分引擎"""
    DIMENSIONS = [
        "causal_reasoning",    # 因果推理
        "physical_understanding",  # 物理世界理解
        "multimodal_perception",   # 多模态感知
        "open_domain_reasoning",   # 开放域推理
        "safety_constraints",      # 安全约束
        "continual_learning",      # 持续学习
        "knowledge_scale",         # 知识规模
        "economic_efficiency",     # 经济效率
    ]
    
    def score_all(self, wm) -> dict[str, float]:
        scores = {}
        scores["causal_reasoning"] = self._score_causal(wm)
        scores["physical_understanding"] = self._score_physical(wm)
        scores["multimodal_perception"] = self._score_multimodal(wm)
        scores["open_domain_reasoning"] = self._score_open_domain(wm)
        scores["safety_constraints"] = self._score_safety(wm)
        scores["continual_learning"] = self._score_continual(wm)
        scores["knowledge_scale"] = self._score_knowledge(wm)
        scores["economic_efficiency"] = self._score_economics(wm)
        scores["composite"] = np.mean(list(scores.values()))
        return scores
    
    def _score_causal(self, wm) -> float:
        """因果推理评分 (0-10)"""
        score = 0.0
        # Pearl L1/L2/L3 完备性 (各 2 分)
        if hasattr(wm, '_do_calculus'): score += 2.0
        if hasattr(wm, '_counterfactual'): score += 2.0
        # PearlChain 串联 (2 分)
        if hasattr(wm, '_pearl_chain'): score += 2.0
        # 因果发现质量 (2 分)
        score += min(2.0, len(wm._causal_graph.get("edges", [])) / 10)
        # 因果图规模 (2 分)
        score += min(2.0, len(wm._causal_graph.get("edges", [])) / 50)
        return min(10.0, score)
    
    def _score_multimodal(self, wm) -> float:
        """多模态感知评分"""
        score = 0.0
        encoders = getattr(wm, '_modality_encoders', {})
        for name, enc in encoders.items():
            dim = getattr(enc, 'output_dim', 0)
            n_params = getattr(enc, 'n_params', 0)
            # 维度得分 (最高 2.5/编码器)
            score += min(2.5, dim / 128 * 2.5)
            # 可学习得分 (2.5 if 有参数)
            if n_params > 0: score += 2.5
        return min(10.0, score / max(len(encoders), 1))
```

### 3.2 版本对比报告

```python
def generate_version_diff(scores_history: dict) -> str:
    """生成版本间对比报告"""
    report = "# MCI World Model 版本评分对比\n\n"
    report += "| 维度 | " + " | ".join(scores_history.keys()) + " | Δ |\n"
    report += "|---|" + "---|" * len(scores_history) + "---|\n"
    
    versions = list(scores_history.keys())
    for dim in MCIScoreEngine.DIMENSIONS:
        row = f"| {dim} |"
        for v in versions:
            row += f" {scores_history[v].get(dim, 'N/A')} |"
        # 计算最新 vs 最旧的变化
        if len(versions) >= 2:
            delta = scores_history[versions[-1]].get(dim, 0) - scores_history[versions[0]].get(dim, 0)
            row += f" {delta:+.1f} |"
        report += row + "\n"
    return report
```

### 3.3 评分持久化

```python
class ScorePersistence:
    """评分数据持久化"""
    def __init__(self, db_path="scores_history.json"):
        self._db_path = db_path
    
    def save(self, version: str, scores: dict):
        data = self._load()
        data[version] = {**scores, "timestamp": time.time()}
        with open(self._db_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def trend(self, dimension: str) -> list:
        """获取某维度的历史趋势"""
        data = self._load()
        return [(v, d.get(dimension, 0)) for v, d in sorted(data.items())]
```

## 4. 时间计划

| 周 | 任务 | 交付物 |
|---|---|---|
| W4-6 | 评分引擎核心 | `MCIScoreEngine` |
| W7-9 | 8 维度评分函数 | 各维度评分器 |
| W10-12 | 版本对比 + 持久化 | diff 报告 + JSON |
| W13-18 | CI 集成 | 每次发布自动评分 |
| W19-27 | 趋势图 + 仪表板 | 可视化 |

## 5. 资源配置

| 资源 | 数量 | 成本 |
|---|---|---|
| 后端工程师 | 1人 × 8周 | 28 人天 |

## 6. KPI 指标

| KPI | 基线 | 目标 |
|---|---|---|
| 自动化评分覆盖 | 0/8 维度 | 8/8 维度 |
| 评分一致性 | N/A | 人工评分偏差 <0.5 |
| 版本对比可用 | N/A | ≥3 个版本对比 |

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|
| 自动评分与人工评估不一致 | 中 | 中 | 定期校准 + 人工修正权重 |
| 评分维度定义争议 | 低 | 中 | 文档化评分标准 |

## 8. 成本预算

| 项目 | 人天 | 硬件/软件 |
|---|---|---|
| 评分引擎 | 12 | $0 |
| 版本对比 | 6 | $0 |
| CI 集成 | 5 | $0 |
| 可视化 | 5 | $0 |
| **小计** | **28** | **$0** |

## 9. 验收标准

- [ ] 8 个维度各有自动评分函数
- [ ] 评分引擎可在 CI 中运行
- [ ] 版本对比报告自动生成
- [ ] v5.0.0 基线评分已存储
- [ ] 评分与人工评估偏差 <0.5/10

## 依赖关系

- **前置**: Ch05 (形式化不变量 → 评分输入), Ch08 (WMMM → 成熟度评分)
- **被依赖**: Ch14 (战略结论), Ch12 (门禁检查)
