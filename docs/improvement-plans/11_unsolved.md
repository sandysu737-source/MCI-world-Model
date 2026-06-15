# Ch11 未解决领域探索 — 改进规划书

## 1. 章节概述

原报告第十一章探索了三个前沿方向：
- **自主物理规律发现**: `_sigreg.py` 符号回归存在但未集成
- **跨场景零样本迁移**: 无 MAML/ProgressiveNN
- **真正的持续学习**: EWC 有 50% 遗忘 (F4)

这三个方向都属于 P3 探索阶段，定位为研究原型。

## 2. 改进目标

| # | 目标 | 量化指标 | 优先级 |
|---|---|---|---|
| G1 | 自主物理规律发现管线 | 在 ≥2 个物理系统上自主发现正确方程 | P3 |
| G2 | MAML 元学习器原型 | 在 ≥2 个任务上零样本准确率 ≥60% | P3 |
| G3 | 在线持续学习框架 | 10 任务后遗忘率 <15% | P3 |
| G4 | SurpriseSignal 驱动自主学习 | 高惊奇样本自动进入训练队列 | P2 |

## 3. 实施方案

### 3.1 自主物理规律发现 (G1)

**管线**: 观测数据 → 候选方程 → 验证 → 因果图更新

```python
class AutonomousLawDiscoverer:
    """自主物理规律发现器"""
    def __init__(self, sigreg_instance, do_calculus):
        self._pysr = sigreg_instance
        self._do = do_calculus
        self._discovered_laws: list[dict] = []
    
    def discover_from_observations(self, data: np.ndarray, var_names: list[str]):
        """
        1. PySR 符号回归生成候选方程
        2. 物理守恒验证 (能量/动量)
        3. 因果图验证 (do-calculus)
        4. 置信度校准
        """
        candidates = self._pysr.fit(data, var_names=var_names, n_equations=10)
        for eq in candidates:
            if self._verify_conservation(eq, data):
                if self._verify_causal_structure(eq):
                    self._discovered_laws.append({
                        "equation": eq.equation,
                        "r_squared": eq.r_squared,
                        "conservation_verified": True,
                        "causal_verified": True,
                    })
```

**文件**: `_autonomous_law_discoverer.py` (~400行)

### 3.2 MAML 元学习器 (G2)

```python
class SimpleMAML:
    """简化版 MAML — 模型无关元学习"""
    def __init__(self, model_factory, inner_lr=0.01, outer_lr=0.001):
        self._model_factory = model_factory
        self._inner_lr = inner_lr
        self._outer_lr = outer_lr
    
    def meta_train(self, task_batch: list[dict], n_inner_steps=5):
        """
        对每个任务:
          1. 内循环: 用任务数据做 n_inner_steps 步梯度下降
          2. 外循环: 收集所有内循环后的 loss，更新初始参数
        """
        meta_loss = 0.0
        for task in task_batch:
            model = self._model_factory()
            # 内循环
            for _ in range(n_inner_steps):
                loss = model.compute_loss(task["train"])
                model.step(loss, self._inner_lr)
            # 外循环
            meta_loss += model.compute_loss(task["test"])
        # 更新初始参数
        self._update_initial_params(meta_loss / len(task_batch))
```

### 3.3 SurpriseSignal 驱动学习 (G4)

```python
class SurpriseDrivenLearner:
    """惊奇驱动的自主学习"""
    def __init__(self, surprise_detector, predictor, pem):
        self._detector = surprise_detector
        self._predictor = predictor
        self._pem = pem
        self._training_queue: list[dict] = []
    
    def observe(self, predicted, actual, context: dict):
        """观察预测-实际对"""
        signal = self._detector.compute_surprise(predicted, actual)
        if signal.is_anomaly:
            self._training_queue.append({
                "predicted": predicted,
                "actual": actual,
                "surprise_score": signal.score,
                "context": context,
                "timestamp": time.time(),
            })
    
    def learn_from_surprises(self, max_samples=100):
        """从高惊奇样本中学习"""
        # 按惊奇度排序
        self._training_queue.sort(key=lambda x: -x["surprise_score"])
        samples = self._training_queue[:max_samples]
        # 用高惊奇样本更新预测器
        for sample in samples:
            self._predictor.update(sample["predicted"], sample["actual"])
        self._training_queue.clear()
```

### 3.4 Online EWC (G3)

```python
class OnlineEWC:
    """在线 EWC — 替代标准 EWC 解决遗忘问题"""
    def __init__(self, model, damping=0.1):
        self._model = model
        self._damping = damping
        self._fisher_diag = {}  # 每个任务一个对角 Fisher
        self._star_params = {}
    
    def consolidate(self, task_id: str):
        """任务完成后固化参数"""
        self._fisher_diag[task_id] = self._compute_diagonal_fisher()
        self._star_params[task_id] = self._model.get_params()
    
    def ewc_loss(self) -> float:
        """所有已固化任务的 EWC 损失之和"""
        total = 0.0
        for task_id in self._fisher_diag:
            for p, p_star, f in zip(
                self._model.get_params(),
                self._star_params[task_id],
                self._fisher_diag[task_id],
            ):
                total += np.sum((f + self._damping) * (p - p_star) ** 2)
        return total
```

## 4. 时间计划

| 周 | 任务 | 交付物 |
|---|---|---|
| W28-30 | SurpriseDrivenLearner | 惊奇驱动学习器 + 测试 |
| W31-34 | OnlineEWC | 在线 EWC + 10 任务基准 |
| W35-38 | SimpleMAML | 元学习器原型 + 2 任务基准 |
| W39-44 | AutonomousLawDiscoverer | 物理规律发现 + 2 系统验证 |
| W45-48 | 集成测试 + 研究论文草稿 | 全量测试 + 论文 |

## 5. 资源配置

| 资源 | 数量 | 成本 |
|---|---|---|
| 研究工程师 | 1人 × 21周 | 75 人天 |
| GPU (PySR/MAML) | 按需 | $1,000 |

## 6. KPI 指标

| KPI | 基线 | 目标 |
|---|---|---|
| 物理规律自主发现 | 0 | ≥2 个系统 |
| 零样本迁移准确率 | N/A | ≥60% |
| 持续学习遗忘率 | 50% (EWC) | <15% (Online EWC) |
| 惊奇驱动训练样本利用率 | N/A | ≥80% 高惊奇样本有用 |

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|
| PySR 在高维数据上过慢 | 高 | 中 | 限制变量 ≤5 |
| MAML 不稳定 | 高 | 中 | 使用 Reptile 作为替代 |
| Online EWC 仍有过拟合 | 中 | 中 | 增加 damping 系数 |
| 研究结果无法复现 | 中 | 高 | 固定随机种子 + 详细日志 |

## 8. 成本预算

| 项目 | 人天 | 硬件/软件 |
|---|---|---|
| SurpriseDrivenLearner | 10 | $0 |
| OnlineEWC | 12 | $0 |
| SimpleMAML | 15 | $500 |
| AutonomousLawDiscoverer | 20 | $500 |
| 集成测试 | 10 | $0 |
| 论文草稿 | 8 | $0 |
| **小计** | **75** | **$1,000** |

## 9. 验收标准

- [ ] 自主发现: 在 Pendulum 数据上发现 `d²θ/dt² = -(g/L)sin(θ)` 的近似形式
- [ ] 零样本: MAML 在 Cart 任务上 (仅用 Pendulum 元训练) 准确率 ≥60%
- [ ] 持续学习: 10 任务后任务1 准确率 ≥80%
- [ ] 惊奇学习: 高惊奇样本自动进入训练队列，学习后惊奇度下降 ≥30%

## 依赖关系

- **前置**: Ch04 (F4 EWC 修复), Ch02 (TrueJEPA), Ch06 (元认知)
- **被依赖**: 无 (研究探索方向)
