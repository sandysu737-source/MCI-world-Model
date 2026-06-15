# Ch15 高级认知与自主发现 — 改进规划书

## 1. 章节概述

本章节覆盖 P6-P8 波次中**高级认知能力**的全面升级，填补原 Ch06（五子系统覆盖率 50.6%）和 Ch11（未解决领域 V1）在以下方向的空白：

- **社会认知**: 多智能体博弈 + 心智理论 — 原系统仅单智能体，无社会交互能力
- **自修复认知**: 推理链异常检测 + 自动修复 — 原系统无推理链自我修复机制
- **元认知 2.0**: 快慢双系统 — 原元认知 (28%) 仅有诊断能力，无策略选择
- **因果想象引擎**: 潜空间反事实模拟 — 原系统仅支持单步反事实，无多场景想象
- **自主规律发现 2.0**: 从单方程到完整因果结构 — Ch11 V1 仅发现单方程，V2 升级到 DAG + PC 算法
- **自主科学发现管线**: 观测→规律→假设→实验设计闭环 — Ch11 V1 仅有发现，无假设生成和实验设计

> **新增定位**: Ch06 覆盖"五子系统基础能力"，Ch15 覆盖"高级认知与自主发现"——是从 50.6% → 85%+ 的关键跃迁。

## 2. 改进目标

| # | 目标 | 量化指标 | 波次 | 优先级 |
|---|---|---|---|---|
| G1 | SocialCognition 多智能体社会认知 | 3 智能体博弈预测准确率 ≥60% | P6 | 中 |
| G2 | SelfRepairCognition 自修复认知 | 10 次推理异常 ≥7 次自修复 | P6 | 中 |
| G3 | MetacognitionV2 快慢系统 | 策略切换准确率 ≥80% | P6 | 中 |
| G4 | CausalImaginationEngine 因果想象 | 10 题中 ≥7 题合理 | P6 | 中 |
| G5 | AutonomousLawDiscovererV2 | 5 变量因果系统完整结构发现 | P6 | 中 |
| G6 | ScientificDiscoveryPipeline | 自主发现 + ≥2 假设 + ≥1 实验设计 | P7 | 中 |
| G7 | 科学发现闭环深化 | ≥1 完整发现→假设→验证→修正循环 | P8 | 中 |

## 3. 实施方案

### 3.1 SocialCognition 多智能体社会认知 (G1)

**缺口**: 原系统零社会认知能力，无法理解其他智能体的意图和行为

```python
class SocialCognition:
    """多智能体社会认知 — 博弈论 + 心智理论"""
    def __init__(self, n_agents=3, theory_of_mind_depth=2):
        self._n_agents = n_agents
        self._tom_depth = theory_of_mind_depth
        self._agent_models: dict[int, AgentModel] = {}
    
    def observe_interaction(self, agent_id: int, action: str, outcome: dict):
        """观察其他智能体的行为"""
        if agent_id not in self._agent_models:
            self._agent_models[agent_id] = AgentModel(agent_id)
        self._agent_models[agent_id].update(action, outcome)
    
    def predict_others(self, context: dict) -> dict:
        """心智理论: 预测其他智能体的行为"""
        predictions = {}
        for aid, model in self._agent_models.items():
            predictions[aid] = model.predict_action(context)
        return predictions
    
    def nash_equilibrium(self, payoff_matrix: np.ndarray) -> dict:
        """博弈论: 计算纳什均衡"""
        return self._solve_2x2_nash(payoff_matrix)
    
    def negotiate(self, my_preferences: dict, others_predicted: dict) -> dict:
        """社会协商: 基于心智理论的决策"""
        return self._pareto_optimize(my_preferences, others_predicted)


class AgentModel:
    """其他智能体的内部模型"""
    def __init__(self, agent_id: int):
        self._agent_id = agent_id
        self._action_history: list[str] = []
        self._preference_model = {}
    
    def update(self, action: str, outcome: dict):
        self._action_history.append(action)
        self._update_preferences(action, outcome)
    
    def predict_action(self, context: dict) -> str:
        """基于历史行为预测下一步"""
        return self._most_likely_action(context)
```

**文件**: `_social_cognition.py` (~400 行)

**理论依据**:
- 心智理论 (Theory of Mind): Premack & Woodruff (1978)
- 纳什均衡: Nash (1951)
- 多智能体博弈: Shoham & Leyton-Brown (2008)

### 3.2 SelfRepairCognition 自修复认知 (G2)

**缺口**: 原系统无推理链自我修复能力，异常时直接失败

```python
class SelfRepairCognition:
    """自修复认知 — 检测并修复推理链断裂"""
    def __init__(self, world_model, meta_diagnoser):
        self._wm = world_model
        self._diagnoser = meta_diagnoser
        self._repair_history: list[dict] = []
    
    def detect_anomaly(self, prediction, actual) -> dict:
        """检测推理异常"""
        error = np.linalg.norm(prediction - actual)
        if error > self._anomaly_threshold:
            diagnosis = self._diagnoser.diagnose(prediction, actual)
            return {"is_anomaly": True, "error": error, "diagnosis": diagnosis}
        return {"is_anomaly": False}
    
    def repair(self, anomaly_report: dict) -> dict:
        """自修复: 根据诊断调整推理链"""
        diagnosis = anomaly_report["diagnosis"]
        repair_action = self._select_repair_action(diagnosis)
        self._apply_repair(repair_action)
        self._repair_history.append({
            "anomaly": anomaly_report,
            "repair": repair_action,
            "timestamp": time.time(),
        })
        return {"repaired": True, "action": repair_action}
    
    def _select_repair_action(self, diagnosis) -> str:
        """选择修复策略"""
        if diagnosis["layer"] == "perception":
            return "recalibrate_encoder"
        elif diagnosis["layer"] == "prediction":
            return "increase_prediction_steps"
        elif diagnosis["layer"] == "causal":
            return "relearn_causal_structure"
        return "fallback_to_safe_state"
```

**文件**: `_self_repair_cognition.py` (~250 行)

**修复策略映射**:

| 诊断层 | 修复策略 | 成本 |
|---|---|---|
| 感知层 | 重新校准编码器 | 低 |
| 预测层 | 增加预测步数 | 中 |
| 因果层 | 重学因果结构 | 高 |
| 未知层 | 回退安全状态 | 极高 |

### 3.3 MetacognitionV2 元认知 2.0 (G3)

**缺口**: 原元认知 (Ch06 G5: 28%→55%) 仅有诊断能力，无推理策略选择

```python
class MetacognitionV2:
    """元认知 2.0 — 快慢双系统 (Kahneman)"""
    def __init__(self):
        self._strategies = {
            "fast": FastIntuitiveStrategy(),     # 快系统: 模式匹配
            "slow": SlowDeliberateStrategy(),   # 慢系统: 深度因果推理
            "repair": SelfRepairStrategy(),      # 修复策略: 异常修复
        }
        self._strategy_stats: dict[str, dict] = {}
    
    def select_strategy(self, query: dict) -> str:
        """根据查询特征选择推理策略"""
        complexity = self._estimate_complexity(query)
        if complexity < 0.3:
            return "fast"   # 简单因果: 快直觉
        elif complexity < 0.7:
            return "slow"   # 复杂因果: 慢审慎
        else:
            return "repair" # 可能出错: 自修复模式
    
    def reflect_on_outcome(self, strategy: str, outcome: dict):
        """反思推理结果，更新策略偏好"""
        success = outcome.get("accuracy", 0) > 0.7
        if strategy not in self._strategy_stats:
            self._strategy_stats[strategy] = {"success": 0, "total": 0}
        self._strategy_stats[strategy]["total"] += 1
        if success:
            self._strategy_stats[strategy]["success"] += 1
```

**文件**: `_metacognition_v2.py` (~200 行)

**理论依据**: Kahneman 双系统理论 (System 1 快 / System 2 慢)

**与 Ch06 G5 元认知的区别**:
- Ch06 V1: 仅诊断推理错误 → 元认知覆盖率 28%→55%
- Ch15 V2: 诊断 + 策略选择 + 反思 → 覆盖率 55%→80%+

### 3.4 CausalImaginationEngine 因果想象引擎 (G4)

**缺口**: 原系统仅支持单步反事实推理 (Pearl 三步法)，无法在潜空间中模拟多场景

```python
class CausalImaginationEngine:
    """因果想象引擎 — 在潜空间中模拟反事实场景"""
    def __init__(self, unified_encoder, do_calculus, counterfactual_engine):
        self._encoder = unified_encoder
        self._do = do_calculus
        self._cf = counterfactual_engine
    
    def imagine(self, factual: dict, intervention: dict, n_scenarios: int = 5) -> list[dict]:
        """
        因果想象: 生成多种可能的反事实场景
          1. 编码事实观测 → 潜空间 z_factual
          2. 施加干预 → 潜空间 z_counterfactual
          3. 解码反事实场景 → 多种可能结果
        """
        z_factual = self._encoder.encode(factual)
        scenarios = []
        for i in range(n_scenarios):
            z_cf = self._apply_intervention_in_latent(z_factual, intervention, seed=i)
            imagined = self._encoder.decode(z_cf)
            scenarios.append({
                "scenario_id": i,
                "intervention": intervention,
                "imagined_outcome": imagined,
                "plausibility": self._estimate_plausibility(z_factual, z_cf),
            })
        return sorted(scenarios, key=lambda s: -s["plausibility"])
    
    def imagine_causal_chain(self, initial_state, interventions: list[dict]) -> list[dict]:
        """因果链想象: 多步干预的时间展开"""
        chain = []
        state = initial_state
        for step, intervention in enumerate(interventions):
            scenarios = self.imagine(state, intervention, n_scenarios=3)
            best = scenarios[0]  # 取最合理的场景
            chain.append(best)
            state = best["imagined_outcome"]
        return chain
```

**文件**: `_causal_imagination.py` (~300 行)

**与 Ch05 反事实推理的关系**:
- Ch05 (形式化): Pearl 三步法反事实 → 单步精确推理
- Ch15 (想象): 潜空间多场景模拟 → 多步近似推理 + 合理性排序

### 3.5 AutonomousLawDiscovererV2 自主规律发现 2.0 (G5)

**缺口**: Ch11 V1 仅发现单方程，V2 升级到 PC 算法骨架发现 + 多方程系统

```python
class AutonomousLawDiscovererV2:
    """自主因果发现 2.0 — 从简单方程到复杂因果结构"""
    def __init__(self, sigreg_instance, do_calculus, pc_discovery):
        self._pysr = sigreg_instance      # V1 已有
        self._do = do_calculus             # V1 已有
        self._pc = pc_discovery            # V2 新增: PC 算法
        self._discovered_laws: list[dict] = []
        self._causal_structure: dict = {}
    
    def discover_causal_structure(self, data: np.ndarray, var_names: list[str]):
        """
        V2 增强发现管线:
          1. PC 算法学习因果骨架 (V2 新增)
          2. 符号回归生成每个因果边的方程
          3. 守恒验证 + 因果方向验证
          4. 组合验证: 多方程系统一致性
        """
        # V2 新增: 从数据学习因果骨架
        skeleton = self._pc.discover(data, var_names)
        
        # 对每个因果边做符号回归
        for edge in skeleton["edges"]:
            cause_idx = var_names.index(edge["from"])
            effect_idx = var_names.index(edge["to"])
            x = data[:, cause_idx:cause_idx+1]
            y = data[:, effect_idx:effect_idx+1]
            candidates = self._pysr.fit(
                np.hstack([x, y]),
                var_names=[edge["from"], edge["to"]],
                n_equations=5
            )
            for eq in candidates:
                if self._verify_conservation(eq, data):
                    self._discovered_laws.append({
                        "edge": edge,
                        "equation": eq.equation,
                        "r_squared": eq.r_squared,
                    })
        
        self._causal_structure = skeleton
        return self._build_system_report()
    
    def _build_system_report(self) -> dict:
        """构建多方程系统一致性报告"""
        return {
            "n_variables": len(self._causal_structure.get("nodes", [])),
            "n_edges": len(self._discovered_laws),
            "conservation_score": self._check_system_conservation(),
            "causal_dag": self._causal_structure,
            "laws": self._discovered_laws,
        }
```

**文件**: `_autonomous_law_discoverer_v2.py` (~500 行)

**V1→V2 升级对比**:

| 维度 | V1 (Ch11) | V2 (Ch15) |
|---|---|---|
| 发现范围 | 单方程 | 完整 DAG + 多方程 |
| 骨架发现 | 无 | PC 算法 |
| 变量数 | ≤3 | ≤5 (可扩展到 ≤10) |
| 一致性 | 单方程守恒 | 多方程系统一致性 |
| 波次 | P3-P4 | P6 |

### 3.6 ScientificDiscoveryPipeline 自主科学发现管线 (G6)

**缺口**: Ch11 V1 仅有规律发现，无假设生成和实验设计

```python
class ScientificDiscoveryPipeline:
    """自主科学发现管线 — 从数据到假设到验证"""
    def __init__(self, law_discoverer, hypothesis_generator, experiment_designer):
        self._discoverer = law_discoverer          # LawDiscovererV2
        self._hypo_gen = hypothesis_generator
        self._exp_designer = experiment_designer
    
    def discover(self, data: np.ndarray, var_names: list[str]) -> dict:
        """完整科学发现流程"""
        # 阶段1: 观测 → 规律发现
        laws = self._discoverer.discover_causal_structure(data, var_names)
        
        # 阶段2: 规律 → 假设生成
        hypotheses = self._hypo_gen.generate(laws)
        
        # 阶段3: 假设 → 实验设计
        experiments = []
        for hyp in hypotheses:
            exp = self._exp_designer.design(hyp)
            experiments.append(exp)
        
        return {
            "discovered_laws": laws,
            "hypotheses": hypotheses,
            "proposed_experiments": experiments,
            "confidence_score": self._overall_confidence(laws),
        }


class HypothesisGenerator:
    """假设生成器 — 从发现的规律中生成可测试假设"""
    def __init__(self, world_model):
        self._wm = world_model
    
    def generate(self, discovered_laws: dict) -> list[dict]:
        """从发现的规律中生成假设"""
        hypotheses = []
        for law in discovered_laws.get("laws", []):
            # 生成边界假设
            hyp = {
                "source_law": law["equation"],
                "hypothesis": f"If {law['edge']['from']} increases by 10%, "
                              f"then {law['edge']['to']} changes by Δ",
                "testable": True,
                "priority": law.get("r_squared", 0),
            }
            hypotheses.append(hyp)
        return sorted(hypotheses, key=lambda h: -h["priority"])


class ExperimentDesigner:
    """实验设计器 — 为假设设计验证实验"""
    def design(self, hypothesis: dict) -> dict:
        """设计对照实验"""
        return {
            "hypothesis": hypothesis,
            "control_group": {"intervention": "none"},
            "treatment_group": {"intervention": hypothesis["source_law"]},
            "sample_size": 100,
            "metrics": ["ate", "confidence_interval"],
            "expected_result": "ate显著非零",
        }
```

**文件**: `_scientific_discovery.py` (~400 行), `_hypothesis_generator.py` (~300 行), `_experiment_designer.py` (~250 行)

**闭环深化 (P8 G7)**:
```
P7: 单向流程 — 观测 → 规律 → 假设 → 实验设计
P8: 完整闭环 — 观测 → 规律 → 假设 → (虚拟)实验 → 结果 → 假设修正 → 新发现
     新增: 虚拟实验执行器 + 结果一致性分析器 + 假设修正器
```

## 4. 时间计划

| 周 | 任务 | 交付物 | 波次 |
|---|---|---|---|
| W53-55 | LawDiscovererV2 核心架构 | PC骨架 + 符号回归 | P6 |
| W56-58 | LawDiscovererV2 5变量验证 | 5变量因果系统报告 | P6 |
| W59-62 | SocialCognition 核心实现 | 多智能体博弈 + 心智理论 | P6 |
| W57-62 | SelfRepairCognition 实现 | 异常检测 + 修复循环 | P6 |
| W63-68 | CausalImaginationEngine | 反事实场景模拟 | P6 |
| W65-68 | DifferentiableCausalInference (参见 Ch16) | 可微分 ATE | P6 |
| W69-70 | MetacognitionV2 实现 | 快慢系统 + 策略选择 | P6 |
| W71-76 | ScientificDiscoveryPipeline 核心 | 发现→假设→实验设计 | P7 |
| W77-84 | 假设生成器 + 实验设计器验证 | 假设生成基准 + 实验方案 | P7 |
| W97-104 | 科学发现闭环深化 | 发现→验证→修正闭环 | P8 |

## 5. 资源配置

| 资源 | 角色 | 人天 | 说明 |
|---|---|---|---|
| 研究工程师 A | LawDiscovererV2 + SocialCognition + 可微分因果 | 30 | P6 核心 |
| 工程师 B | SelfRepairCognition + CausalImaginationEngine | 20 | P6 认知深化 |
| 研究工程师 (P7) | ScientificDiscoveryPipeline + 假设生成 + 实验设计 | 15 | P7 科学发现 |
| 研究工程师 (P8) | 科学发现闭环深化 | 5 | P8 闭环 |
| Tech Lead | MetacognitionV2 + 门禁 | 5 | P6 元认知 |
| **合计** | | **75** | |

## 6. KPI 指标

| KPI | 基线 | P6 目标 | P7 目标 | P8 目标 |
|---|---|---|---|---|
| 多智能体预测准确率 | 0% | ≥60% | — | — |
| 自修复成功率 | 0% | ≥70% | — | — |
| 快慢系统切换准确率 | 0% | ≥80% | — | — |
| 因果想象合理性 | 0% | ≥70% | — | — |
| 因果发现系统数 | 2 (V1) | ≥3 (含5变量) | 完整流程 | 完整闭环 |
| 发现方程 R² | ≥0.90 | ≥0.85 (复杂) | — | — |
| 可测试假设 | 0 | — | ≥2 | — |
| 发现闭环循环 | 0 | — | — | ≥1 |

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 | 应急方案 |
|---|---|---|---|---|
| LawDiscovererV2 高维失败 | 高 | 中 | 限制变量 ≤10 | 退回 V1 仅做方程发现 |
| SocialCognition 过于简化 | 高 | 低 | 定位为研究原型 | 标注局限性 |
| 自修复修复策略不够 | 中 | 中 | 4 层策略 + 安全回退 | 标注为半自动 |
| 因果想象结果不合理 | 中 | 中 | 增加守恒约束 | 仅做单步反事实 |
| 科学发现管线假阳性 | 高 | 中 | 交叉验证 + 守恒检查 | 标注置信度 |
| 18 周不够 (P6) | 中 | 高 | 社会认知可推迟到 P7 | 元认知 2.0 推迟 |

## 8. 成本预算

| 项目 | 人天 | 硬件/软件 | 波次 |
|---|---|---|---|
| SocialCognition | 10 | $0 | P6 |
| SelfRepairCognition | 5 | $0 | P6 |
| MetacognitionV2 | 3 | $0 | P6 |
| CausalImaginationEngine | 5 | $0 | P6 |
| LawDiscovererV2 | 12 | $500 (GPU) | P6 |
| ScientificDiscoveryPipeline | 15 | $500 (GPU) | P7 |
| 科学发现闭环深化 | 5 | $500 (GPU) | P8 |
| **合计** | **55** | **$1,500** | |

## 9. 验收标准

- [ ] SocialCognition: 3 智能体博弈预测准确率 ≥60%
- [ ] SelfRepairCognition: 10 次异常 ≥7 次自修复
- [ ] MetacognitionV2: 快慢系统切换准确率 ≥80%
- [ ] CausalImaginationEngine: 10 题中 ≥7 题合理 (与符号推理一致)
- [ ] LawDiscovererV2: 5 变量因果系统完整结构发现 + ≥3 条边
- [ ] ScientificDiscoveryPipeline: 物理数据自主发现 + ≥2 可测试假设 + ≥1 实验方案
- [ ] 科学发现闭环: ≥1 完整发现→假设→验证→修正循环

## 依赖关系

- **前置**: Ch04 (致命缺陷), Ch06 (五子系统基础), Ch11 (V1 规律发现), Ch08 (WMMM 基准)
- **被依赖**: Ch16 (可微分因果是神经符号融合前置), Ch18 (SocialCognition 是 SDK 社会协商前置)
