# Ch08 WMMM 六层成熟度评定 — 改进规划书

## 1. 章节概述

原报告第八章使用杨立昆 WMMM (World Model Maturity Model) 六层框架评估：
- L0 反应式: 100% ✅ (已超越)
- L1 预测式: 92% ✅ (已实现)
- **L2 生成式: 68%** ⚠️ (潜空间动力学部分实现)
- **L3 因果式: 55%** ⚠️ (结构性实现，缺层级串联)
- **L4 反思式: 22%** ❌ (仅原型)
- **L5 自主式: 0%** ❌ (完全缺失)

**综合 WMMM**: L2.5 (56%) → 目标 L3.5 (75%)

## 2. 改进目标

| # | 目标 | 基线 → 目标 | 优先级 |
|---|---|---|---|
| G1 | L2 生成式达标 | 68% → 90% | P1 |
| G2 | L3 因果式达标 | 55% → 80% | P1 |
| G3 | L4 反思式突破 | 22% → 50% | P2 |
| G4 | L5 自主式探索 | 0% → 15% | P3 |

## 3. 实施方案

### 3.1 L2 生成式提升 (68%→90%)

**差距**:
- 潜空间预测器 (JEPA) 输出因果图而非潜向量 → Ch02 TrueJEPA
- 学习型预测器仅 6 参数线性 → Ch04 F10 MLP 升级
- 缺少多步预测验证基准

**新增**: 多步预测基准套件

```python
class GenerativeCapabilityBenchmark:
    """L2 生成式能力基准"""
    def benchmark(self, predictor, ground_truth_predictor, n_steps=20):
        errors = []
        state = initial_state()
        for step in range(n_steps):
            predicted = predictor.predict(state, action, n_steps=1)[0]
            actual = ground_truth_predictor.predict(state, action, n_steps=1)[0]
            errors.append(predicted.distance(actual))
            state = actual  # 用真实状态推进
        return {
            "avg_error": np.mean(errors),
            "max_error": np.max(errors),
            "error_growth_rate": self._fit_growth(errors),  # 误差增长率
            "l2_score": self._score(errors),  # L2 得分 [0,1]
        }
```

### 3.2 L3 因果式提升 (55%→80%)

**差距**:
- Pearl L1/L2/L3 层级断裂 → Ch02 PearlChain
- 因果图发现基于相关性 → 需引入 PC algorithm
- 无因果图自适应更新

**新增**: PC 因果发现算法

```python
class PCCausalDiscovery:
    """PC 算法因果发现 — 从数据学习因果结构"""
    def __init__(self, alpha=0.05):
        self._alpha = alpha
    
    def discover(self, data: np.ndarray, var_names: list[str]) -> dict:
        """
        1. 条件独立性检验 → 骨架图
        2. V-structure 定向
        3. Meek 规则传播
        返回: {edges: [...], v_structures: [...], cpdag: {...}}
        """
```

**文件**: `_pc_causal_discovery.py` (~400行)

### 3.3 L4 反思式突破 (22%→50%)

**差距**: MetaDiagnoser 仅规则式，无学习型反思

**复用**: Ch06 ReflectiveMetacognition

**新增**: 自动参数调整

```python
class AutoTuner:
    """基于反思的自动参数调整"""
    def __init__(self, wm: MCIWorldModel):
        self._wm = wm
        self._param_history: list[dict] = []
    
    def tune_after_failure(self, failure_report: dict) -> dict:
        """根据失败报告自动调整参数"""
        diagnosis = failure_report.get("root_cause_layer")
        if diagnosis == "prediction":
            # 预测层问题: 增大 JEPA 学习率
            self._wm._config["jepa_learning_rate"] *= 1.2
        elif diagnosis == "perception":
            # 感知层问题: 降低噪声阈值
            self._wm._config["surprise_threshold"] *= 0.8
        return self._wm._config
```

### 3.4 L5 自主式探索 (0%→15%)

**目标**: 概念验证级别的物理规律自主发现

```python
class SimpleLawDiscoverer:
    """简单物理规律自主发现 — 概念验证"""
    def __init__(self):
        self._candidate_equations = []
    
    def observe(self, data: list[tuple[float, float]]):
        """观察 (x, y) 数据点"""
        # 尝试拟合: y = ax, y = ax², y = a/x, y = a·sin(bx)
        for model in [Linear, Quadratic, Inverse, Sinusoidal]:
            fit = model.fit(data)
            if fit.r_squared > 0.95:
                self._candidate_equations.append(fit)
    
    def discover(self) -> str:
        """返回最佳拟合方程"""
        best = max(self._candidate_equations, key=lambda f: f.r_squared)
        return best.equation_string
```

## 4. 时间计划

| 周 | 任务 | WMMM层级 | 里程碑 |
|---|---|---|---|
| W4-7 | L2 多步预测基准 + TrueJEPA 集成 | L2 | M1: L2 ≥ 85% |
| W8-11 | PC 因果发现 + PearlChain 集成 | L3 | M2: L3 ≥ 70% |
| W12-17 | AutoTuner + ReflectiveMeta | L4 | M3: L4 ≥ 40% |
| W18-22 | 全层级基准套件 | 全部 | M4: WMMM 报告自动生成 |
| W23-27 | SimpleLawDiscoverer | L5 | M5: 发现 y=ax 规律 |

## 5. 资源配置

| 资源 | 数量 | 成本 |
|---|---|---|
| 后端工程师 | 1人 × 18周 | 55 人天 |

## 6. KPI 指标

| KPI | 基线 | 目标 |
|---|---|---|
| L2 生成式得分 | 68% | ≥90% |
| L3 因果式得分 | 55% | ≥80% |
| L4 反思式得分 | 22% | ≥50% |
| L5 自主式得分 | 0% | ≥15% |
| **WMMM 综合** | **L2.5 (56%)** | **≥L3.5 (75%)** |

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|
| PC 算法在高维数据上过慢 | 中 | 高 | 限制变量数 ≤20 |
| L5 概念验证无实际价值 | 高 | 低 | 定位为研究探索，不影响主线 |
| AutoTuner 调参不稳定 | 中 | 中 | 参数变化范围限制 ±30% |

## 8. 成本预算

| 项目 | 人天 | 硬件/软件 |
|---|---|---|
| L2 基准 + 集成 | 12 | $0 |
| PC 因果发现 | 12 | $0 |
| L4 AutoTuner | 10 | $0 |
| WMMM 基准套件 | 8 | $0 |
| L5 概念验证 | 8 | $0 |
| 测试 | 5 | $0 |
| **小计** | **55** | **$0** |

## 9. 验收标准

- [ ] L2: 20 步多步预测平均误差 <0.1
- [ ] L3: PC 算法在 5 变量数据集上发现正确因果结构 ≥80%
- [ ] L4: AutoTuner 在 3 次失败后自动调参恢复 ≥2 次
- [ ] L5: SimpleLawDiscoverer 在 y=3x+2 数据上发现正确方程
- [ ] WMMM 综合得分 ≥75%

## 依赖关系

- **前置**: Ch02 (TrueJEPA, PearlChain), Ch04 (致命缺陷修复)
- **被依赖**: Ch13 (评估总表更新)
