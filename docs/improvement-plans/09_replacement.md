# Ch09 替代性目标评估 — 改进规划书

## 1. 章节概述

原报告第九章五视角统一结论：**V5.0.0 不可独立替代 sub-10B LLM**，但作为"可验证因果增强层"有 100-500x 运行成本优势。最佳架构是混合部署。

**核心差距**:
- 语言生成能力缺失 (0%)
- 开放域知识缺失 (无预训练权重)
- 通用动作-预测模型缺失 (仅玩具物理)

## 2. 改进目标

| # | 目标 | 量化指标 | 优先级 |
|---|---|---|---|
| G1 | 构建 MCI-LLM 混合推理网关 | 80% 查询走 MCI, 20% 走 LLM | P2 |
| G2 | 定义替代性基准测试 | 在因果密集任务上 MCI ≥ LLM | P2 |
| G3 | 领域级替代验证 | 医疗/法律场景 MCI 准确率 ≥ LLM 90% | P3 |
| G4 | 明确不可替代边界 | 文档化 MCI 不适用场景清单 | P3 |

## 3. 实施方案

### 3.1 混合推理网关

```python
class HybridInferenceGateway:
    """MCI + LLM 混合推理网关"""
    def __init__(self, mci: MCIWorldModel, llm_client, threshold=0.6):
        self._mci = mci
        self._llm = llm_client
        self._routing_threshold = threshold
    
    def route(self, query: str) -> str:
        """智能路由: 因果密集 → MCI, 开放域 → LLM"""
        causal_score = self._estimate_causal_density(query)
        if causal_score > self._routing_threshold:
            return self._mci.predict_effect(query)
        else:
            return self._llm.generate(query)
    
    def _estimate_causal_density(self, query: str) -> float:
        """评估查询的因果密度"""
        causal_keywords = {"因为", "导致", "影响", "因果", "干预", "反事实", ...}
        matches = sum(1 for kw in causal_keywords if kw in query)
        return min(1.0, matches / 3.0)
```

**文件**: `_hybrid_gateway.py` (~300行)

### 3.2 替代性基准测试

```python
class ReplacementBenchmark:
    """MCI vs LLM 替代性基准"""
    TASKS = {
        "causal_qa": "因果问答 (20题)",
        "counterfactual": "反事实分析 (10题)",
        "safety_check": "安全约束验证 (20题)",
        "prediction": "物理预测 (15题)",
        "planning": "规划任务 (10题)",
    }
    
    def compare(self, mci, llm, task_name: str) -> dict:
        results = {"mci": [], "llm": []}
        for query, ground_truth in self.load_task(task_name):
            mci_ans = mci.predict_effect(query)
            llm_ans = llm.generate(query)
            results["mci"].append(self.evaluate(mci_ans, ground_truth))
            results["llm"].append(self.evaluate(llm_ans, ground_truth))
        return self.summarize(results)
```

### 3.3 不可替代边界文档

| 场景 | MCI 适用? | 原因 |
|---|---|---|
| 医疗因果推理 | ✅ | 因果关系稳定，Pearl 可验证 |
| 工业安全约束 | ✅ | 物理规则明确 |
| 金融反事实分析 | ✅ | 反事实引擎直接支持 |
| 开放域客服 | ❌ | 需要语言生成 |
| 内容创作 | ❌ | 需要创意和广度 |
| 教育辅导 | ❌ | 需要多轮对话 |
| 代码生成 | ❌ | 需要语法理解 |

## 4. 时间计划

| 周 | 任务 | 交付物 |
|---|---|---|
| W12-16 | 混合推理网关 | `_hybrid_gateway.py` |
| W17-22 | 替代性基准测试 | `benchmarks/replacement/` |
| W23-30 | 医疗/法律领域验证 | 领域基准数据 + 对比报告 |
| W31-40 | 持续优化路由准确率 | 路由准确率日志 |
| W41-48 | 不可替代边界文档 | 文档 + 决策树 |

## 5. 资源配置

| 资源 | 数量 | 成本 |
|---|---|---|
| 后端工程师 | 1人 × 12周 | 24 人天 |
| LLM API 调用 | 按需 | $500 |
| 领域专家 (医疗) | 0.5人 × 4周 | 10 人天 |
| 领域专家 (法律) | 0.5人 × 4周 | 6 人天 |

## 6. KPI 指标

| KPI | 基线 | 目标 |
|---|---|---|
| 混合路由准确率 | N/A | ≥85% (正确识别因果/开放域) |
| 因果任务 MCI ≥ LLM | N/A | ≥80% (4/5 类任务) |
| MCI 推理延迟 | ~15ms | <5ms |
| LLM 调用比例 | 100% (全部走LLM) | ≤20% |

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|
| 路由误判 (因果查询走了 LLM) | 中 | 中 | 回退机制 + 二次验证 |
| LLM API 成本超预算 | 低 | 中 | 缓存 + 本地小模型 |
| 领域验证数据难以获取 | 高 | 中 | 用合成数据 + 专家标注 |

## 8. 成本预算

| 项目 | 人天 | 硬件/软件 |
|---|---|---|
| 混合网关 | 10 | $0 |
| 基准测试 | 8 | $500 |
| 领域验证 | 12 | $0 |
| 路由优化 | 6 | $0 |
| 文档 | 4 | $0 |
| **小计** | **40** | **$500** |

## 9. 验收标准

- [ ] 混合网关: 100 个测试查询中 ≥85 个正确路由
- [ ] 因果任务: MCI 在 4/5 类任务上准确率 ≥ LLM
- [ ] LLM 调用比例 ≤ 20%
- [ ] 不可替代边界文档完成 + 决策树可视化

## 依赖关系

- **前置**: Ch04 (致命缺陷修复), Ch06 (认知架构), Ch07 (经济成本)
- **被依赖**: Ch14 (战略结论)
