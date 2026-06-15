# Ch06 认知架构缺口 (五子系统) — 改进规划书

## 1. 章节概述

原报告第六章对标 LLM 五大认知子系统，发现 V5.0.0 平均覆盖率仅 50.6%：
- **工作记忆** 65% — AutonomousMemory/PEM 存在，缺跨会话沉淀
- **长期记忆** 35% — PEM 有 SQLite+FAISS，但检索语义弱 (F2)
- **推理引擎** 55% — DoCalculus 强，但无 CoT/ReAct/ToT
- **世界模型** 70% — JEPA+因果图架构领先，但实现落后
- **自我意识/元认知** 28% — MetaDiagnoser 存在，缺反思式学习

## 2. 改进目标

| # | 目标 | 基线 → 目标 | 优先级 |
|---|---|---|---|
| G1 | 工作记忆覆盖率 | 65% → 85% | P1 |
| G2 | 长期记忆覆盖率 | 35% → 65% | P1 |
| G3 | 推理引擎覆盖率 | 55% → 75% | P2 |
| G4 | 世界模型覆盖率 | 70% → 85% | P1 |
| G5 | 元认知覆盖率 | 28% → 55% | P2 |

## 3. 实施方案

### 3.1 工作记忆增强 (G1: 65%→85%)

**缺口**: 缺跨会话知识自动沉淀

```python
class WorkingMemoryEnhancer:
    """跨会话知识自动沉淀器"""
    def __init__(self, pem: PersistentMemory, threshold=0.7):
        self._pem = pem
        self._session_buffer: list[dict] = []
        self._consolidation_threshold = threshold
    
    def add_experience(self, experience: dict):
        self._session_buffer.append(experience)
    
    def consolidate(self) -> int:
        """会话结束时沉淀高价值经验到 PEM"""
        consolidated = 0
        for exp in self._session_buffer:
            if self._is_high_value(exp):
                self._pem.store(experience=exp)
                consolidated += 1
        self._session_buffer.clear()
        return consolidated
    
    def _is_high_value(self, exp: dict) -> bool:
        """基于 SurpriseSignal 判断经验价值"""
        surprise = exp.get("surprise_score", 0.0)
        return surprise > self._consolidation_threshold
```

**文件**: `_working_memory_enhancer.py` (~250行)

### 3.2 长期记忆增强 (G2: 35%→65%)

**缺口**: PEM 存储容量和检索质量

**Phase A**: 索引优化 (P0, Ch04 F2)
**Phase B**: 记忆压缩与遗忘

```python
class MemoryConsolidator:
    """记忆压缩与遗忘 — 模拟人类睡眠"""
    def __init__(self, pem: PersistentMemory):
        self._pem = pem
        self._compression_ratio = 0.3  # 保留 30% 高价值记忆
        self._forgetting_curve = ExponentialDecay(half_life_days=30)
    
    def sleep_consolidation(self):
        """定期压缩：合并相似记忆、遗忘低价值记忆"""
        memories = self._pem.retrieve_all()
        clusters = self._cluster_memories(memories)
        for cluster in clusters:
            representative = self._select_representative(cluster)
            self._pem.store(representative)
            for m in cluster[1:]:
                self._pem.archive(m)
```

**文件**: `_memory_consolidator.py` (~300行)

### 3.3 推理引擎增强 (G3: 55%→75%)

**缺口**: 无 CoT (Chain-of-Thought) 风格推理

```python
class CausalChainOfThought:
    """因果链式推理 — Pearl 版 CoT"""
    def __init__(self, do_calculus: DoCalculus, causal_graph: dict):
        self._do = do_calculus
        self._graph = causal_graph
    
    def reason(self, question: str, evidence: dict) -> ReasoningChain:
        """逐步推理:
        1. 解析问题为因果查询 (interventional / counterfactual)
        2. 确定混杂变量
        3. 选择调整方法 (backdoor / frontdoor)
        4. 执行因果推断
        5. 生成推理链
        """
        steps = []
        steps.append(ReasoningStep("parse", self._parse_question(question)))
        steps.append(ReasoningStep("identify_confounders", self._find_confounders(question, evidence)))
        steps.append(ReasoningStep("select_method", self._choose_method(steps[-1])))
        steps.append(ReasoningStep("estimate", self._do_estimate(steps[-1], evidence)))
        steps.append(ReasoningStep("conclude", self._generate_conclusion(steps)))
        return ReasoningChain(steps=steps)
```

**文件**: `_causal_cot.py` (~400行)

### 3.4 世界模型增强 (G4: 70%→85%)

**缺口**: 预测器泛化能力弱

复用 Ch02 G3 (通用 ActionConditionedPredictor) + Ch03 G1 (多物理系统)

### 3.5 元认知增强 (G5: 28%→55%)

**缺口**: MetaDiagnoser 仅有规则式诊断，缺学习型反思

```python
class ReflectiveMetacognition:
    """反思式元认知 — 从失败中学习"""
    def __init__(self, meta_diagnoser: MetaDiagnoser):
        self._diagnoser = meta_diagnoser
        self._failure_patterns: list[dict] = []
        self._success_patterns: list[dict] = []
    
    def observe_outcome(self, action: str, outcome: dict, expected: dict):
        """观察结果，更新模式库"""
        surprise = outcome.get("surprise_score", 0.0)
        if surprise > 0.5:
            self._failure_patterns.append({
                "action": action,
                "surprise": surprise,
                "diagnosis": self._diagnoser.diagnose(outcome),
            })
    
    def suggest_improvement(self) -> list[str]:
        """基于失败模式生成改进建议"""
        suggestions = []
        for pattern in self._failure_patterns[-10:]:  # 最近10个失败
            suggestion = self._pattern_to_suggestion(pattern)
            suggestions.append(suggestion)
        return suggestions
```

**文件**: `_reflective_metacognition.py` (~350行)

## 4. 时间计划

| 周 | 任务 | 交付物 |
|---|---|---|
| W4-5 | WorkingMemoryEnhancer | 跨会话沉淀器 + 测试 |
| W6-8 | MemoryConsolidator | 记忆压缩+遗忘 + 测试 |
| W9-12 | CausalChainOfThought | 因果 CoT 推理器 + 测试 |
| W13-16 | 世界模型增强 (复用 Ch02/Ch03) | 通用预测器集成 |
| W17-22 | ReflectiveMetacognition | 反思式元认知 + 测试 |

## 5. 资源配置

| 资源 | 数量 | 成本 |
|---|---|---|
| 后端工程师 | 1人 × 19周 | 65 人天 |

## 6. KPI 指标

| KPI | 基线 | 目标 |
|---|---|---|
| 工作记忆覆盖率 | 65% | 85% |
| 长期记忆覆盖率 | 35% | 65% |
| 推理引擎覆盖率 | 55% | 75% |
| 世界模型覆盖率 | 70% | 85% |
| 元认知覆盖率 | 28% | 55% |
| 综合覆盖率 | 50.6% | ≥73% |

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|
| 记忆沉淀引入噪声 | 中 | 高 | 高阈值 + 人工审核标记 |
| CoT 推理链过长影响延迟 | 中 | 中 | 限制 max_steps + 缓存 |
| 元认知建议质量差 | 高 | 低 | 仅作为辅助，不自动执行 |

## 8. 成本预算

| 项目 | 人天 | 硬件/软件 |
|---|---|---|
| WorkingMemoryEnhancer | 10 | $0 |
| MemoryConsolidator | 12 | $0 |
| CausalChainOfThought | 15 | $0 |
| 世界模型集成 | 8 | $0 |
| ReflectiveMetacognition | 15 | $0 |
| 测试 | 5 | $0 |
| **小计** | **65** | **$0** |

## 9. 验收标准

- [ ] 工作记忆: 跨会话沉淀器通过 100 次会话测试
- [ ] 长期记忆: PEM 语义检索准确率 ≥0.80
- [ ] 推理引擎: CausalCoT 在 20 个因果问题上生成有效推理链
- [ ] 元认知: ReflectiveMetacognition 从 10 个失败中生成 ≥5 个有用建议
- [ ] 综合覆盖率 ≥73%

## 依赖关系

- **前置**: Ch04 (F2 PEM 修复), Ch02 (TrueJEPA, PearlChain)
- **被依赖**: Ch09 (替代性目标), Ch08 (WMMM L4 反思式)
