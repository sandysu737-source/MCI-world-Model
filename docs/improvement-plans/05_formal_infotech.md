# Ch05 形式化可验证性与信息论评估 — 改进规划书

## 1. 章节概述

原报告第五章分析了两个维度：
- **形式化可验证性**: 60+ 文件无任何 formal proof/guarantee/convergence 声明；DoCalculus/Safety 数学正确但无形式化验证；EWC/GAT 无收敛保证
- **信息论瓶颈**: VisionEncoder 信息损失 58%；AudioEncoder 70%；DoCalculus ATE 75%；SafetyMonitor 83%

**当前评分**: 形式化 2/10，信息保留 4/10

## 2. 改进目标

| # | 目标 | 量化指标 | 优先级 |
|---|---|---|---|
| G1 | 核心组件添加形式化不变量 | DoCalculus/Safety/EWC 各有 invariant 文档 | P1 |
| G2 | 降低多模态信息瓶颈 | 各编码器瓶颈比 ≥0.60 (当前 0.30) | P1 |
| G3 | 降低因果推理信息损失 | DoCalculus ATE 瓶颈比 0.25→0.50 | P2 |
| G4 | 添加收敛性文档 | EWC/GAT/MLP 各有收敛条件说明 | P2 |

## 3. 实施方案

### 3.1 形式化不变量文档 (G1, G4)

为每个核心模块添加 `## Formal Guarantees` docstring 段：

```python
class DoCalculus:
    """
    ## Formal Guarantees
    - **Soundness**: backdoor_adjustment() returns correct ATE when
      all confounders are observed (no unmeasured confounding).
    - **Completeness**: frontdoor_adjustment() handles unmeasured confounders
      when a valid mediator exists.
    - **Invariants**:
      - I1: DAG has no cycles (verified by _topological_sort)
      - I2: NaN/Inf inputs are rejected (line 822-836)
      - I3: Output ATE ∈ [-∞, +∞] with CI
    
    ## Known Limitations
    - Assumes linear SEM (non-linear effects not modeled)
    - Gaussian CI only (non-parametric bootstrap not implemented)
    """
```

**文件**: 每个核心模块顶部添加，约 10-20 行/模块

### 3.2 信息瓶颈优化 (G2)

**VisionEncoder 瓶颈优化**:
```python
# 当前: 32D 统计特征 → 瓶颈比 0.42
# 目标: 128D 可学习编码 → 瓶颈比 0.70+
class ImprovedVisionEncoder:
    def __init__(self, output_dim=128):
        # 保留统计特征 (32D) + 新增可学习特征 (96D)
        self._stat_features = StatFeatureExtractor(32)  # 已有
        self._learned_features = MLP(32, 64, 96)         # 新增
        self.output_dim = 32 + 96  # = 128
```

**AudioEncoder 瓶颈优化**:
```python
# 当前: 16D FFT → 瓶颈比 0.30
# 目标: 64D (16D FFT + 48D learned) → 瓶颈比 0.60+
```

### 3.3 因果推理信息保留 (G3)

**DoCalculus ATE 信息保留优化**:
```python
# 当前: estimate_ate() 返回 {ate, ci_lower, ci_upper, p_value}
# 信息损失在: 丢弃了中间计算 (权重、分层效应、样本分布)
# 优化: 返回完整 ATEResult 对象
@dataclass
class ATEResult:
    ate: float
    ci_lower: float
    ci_upper: float
    p_value: float
    strata_effects: list[float]      # 新增: 每层效应
    weights: np.ndarray              # 新增: IPW 权重
    sample_distribution: dict        # 新增: 样本分布统计
    bottleneck_ratio: float          # 新增: 实际信息保留比
```

### 3.4 收敛性条件文档 (G4)

```python
class IncrementalLearningEngine:
    """
    ## Convergence Conditions
    - EWC converges when: λ ∈ [10, 1000] and n_tasks ≤ 10
    - Online EWC converges when: learning_rate < 0.01/λ
    - **NOT guaranteed**: catastrophic forgetting < 10% for >20 tasks
    
    ## Empirical Observations
    - Diagonal Fisher: 5-task recall ≥ 80% with λ=100
    - Full Fisher: 5-task recall ≥ 90% but O(N²) memory
    """
```

## 4. 时间计划

| 周 | 任务 | 交付物 | 里程碑 |
|---|---|---|---|
| W6-7 | 形式化不变量文档 (10 个核心模块) | docstring 更新 | M1: 10 模块有 invariant |
| W8-9 | VisionEncoder 信息瓶颈优化 | 128D 可学习编码 | M2: 瓶颈比 ≥0.65 |
| W10-11 | AudioEncoder + ThermalEncoder 优化 | 64D/32D 编码 | M3: 音频/热成像瓶颈 ≥0.55 |
| W12-13 | DoCalculus ATEResult + 信息保留 | 完整 ATEResult 对象 | M4: ATE 瓶颈比 ≥0.45 |
| W14-16 | 收敛性文档 + 自动化不变量测试 | `test_invariants.py` | M5: 不变量自动检查 |
| W17-18 | 信息论基准套件 | `benchmarks/info_theory/` | M6: 瓶颈比一键评估 |

## 5. 资源配置

| 资源 | 数量 | 成本 |
|---|---|---|
| 后端工程师 | 1人 × 13周 | 42 人天 |

## 6. KPI 指标

| KPI | 基线 | 目标 |
|---|---|---|
| 有不变量文档的模块 | 0/10 | 10/10 |
| VisionEncoder 瓶颈比 | 0.42 | ≥0.65 |
| AudioEncoder 瓶颈比 | 0.30 | ≥0.55 |
| DoCalculus ATE 瓶颈比 | 0.25 | ≥0.45 |
| 自动化不变量测试覆盖 | 0 | 10 模块 |

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|
| 瓶颈比提升但引入新 bug | 中 | 中 | 保留旧编码器作为 fallback |
| 形式化文档与实际行为不一致 | 中 | 高 | 添加自动化不变量测试验证 |

## 8. 成本预算

| 项目 | 人天 | 硬件/软件 |
|---|---|---|
| 不变量文档 | 8 | $0 |
| VisionEncoder 优化 | 10 | $0 |
| Audio/Thermal 优化 | 8 | $0 |
| ATEResult 重构 | 6 | $0 |
| 不变量测试套件 | 10 | $0 |
| **小计** | **42** | **$0** |

## 9. 验收标准

- [ ] 10 个核心模块各有 `## Formal Guarantees` docstring
- [ ] VisionEncoder 输出 128D，不同图像 L2 > 0.5
- [ ] AudioEncoder 输出 64D
- [ ] `ATEResult` 包含 strata_effects / weights / bottleneck_ratio
- [ ] `test_invariants.py` 自动检查 10 个模块的不变量

## 依赖关系

- **前置**: Ch04 (F5 维度修复), Ch02 (TrueJEPA)
- **被依赖**: Ch13 (评估总表更新)
