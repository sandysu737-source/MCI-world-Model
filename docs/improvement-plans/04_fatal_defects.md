# Ch04 致命缺陷清单 (12项) — 改进规划书

## 1. 章节概述

原报告第四章汇总了 5 次分析发现的 **12 项致命缺陷**（去重后）。这些缺陷分为：
- **Critical (4项)**: F1 NSWM hash路由 / F2 PEM 无语义 / F5 维度坍塌 / F10 6参数JEPA
- **High (7项)**: F3 Fisher O(N²) / F4 EWC 50%遗忘 / F6 JEPA名不副实 / F7 安全盲区 / F8 Pearl断裂 / F11 穷举规划 / F12 VAE维度错误
- **Medium (1项)**: F9 零形式化保证

**当前状态**: 全部 open，无一修复

## 2. 改进目标

| # | 目标 | 量化指标 | 优先级 |
|---|---|---|---|
| G1 | 消除所有 Critical 级缺陷 | F1/F2/F5/F10 全部 fixed | **P0** |
| G2 | 消除所有 High 级缺陷 | F3/F4/F6/F7/F8/F11/F12 全部 fixed | P1 |
| G3 | Medium 缺陷降级为技术债 | F9 添加 formal 注释 + 不变量检查 | P2 |

## 3. 实施方案

### 3.1 F1 — NSWM hash路由修复 (Critical, 1.5周)

**根因**: `_neurosymbolic_world_model.py` 的 `_embed_query()` 使用 `hash() → RandomState → randn(3)`
**修复**:

```python
# 修复前 (hash随机)
def _embed_query(self, query: str) -> np.ndarray:
    h = hash(query)
    rng = np.random.RandomState(h % (2**31))
    return rng.randn(3)

# 修复后 (关键词+BM25语义路由)
def _embed_query(self, query: str) -> np.ndarray:
    """语义嵌入路由 — 关键词匹配 + 轻量词袋"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    if self._tfidf is None:
        self._build_tfidf_index()
    return self._tfidf.transform([query]).toarray()[0]
```

**备选方案**: 如果不想引入 sklearn 依赖，使用自实现的 TF-IDF

**文件**: `_neurosymbolic_world_model.py` L~280-320
**测试**: `test_nswm_routing_semantic.py` — 24 个中文词，验证相似语义路由到同一分支

### 3.2 F2 — PEM char-hash 检索修复 (Critical, 1周)

**根因**: `_persistent_memory.py` 的 `_tags_to_vector()` 使用 `ord(c) % dim`
**修复**:

```python
# 修复前 (char-hash, 无语义)
def _tags_to_vector(self, tags: str) -> np.ndarray:
    vec = np.zeros(self._dim)
    for c in tags:
        vec[ord(c) % self._dim] += 1
    return vec

# 修复后 (BM25 词频向量)
def _tags_to_vector(self, tags: str) -> np.ndarray:
    tokens = self._tokenizer.tokenize(tags.lower())
    vec = np.zeros(self._dim, dtype=np.float64)
    for token in tokens:
        idx = self._vocab_index(token)
        vec[idx] += self._idf.get(token, 1.0)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-10 else vec
```

**文件**: `_persistent_memory.py` L~450-470
**测试**: `test_pem_semantic_retrieval.py` — "心率升高" vs "心率增加" cosine ≥0.80

### 3.3 F3 — Fisher 信息矩阵 O(N²) 修复 (High, 1周)

**根因**: `_incremental_learning.py` 计算完整 Fisher 矩阵
**修复**:

```python
# 修复前 (完整矩阵 O(N²))
self._fisher = np.zeros((n_params, n_params))
for grad in gradients:
    self._fisher += np.outer(grad, grad)

# 修复后 (对角近似 O(N))
self._fisher_diag = np.zeros(n_params)
for grad in gradients:
    self._fisher_diag += grad ** 2
self._fisher_diag /= n_samples
```

**文件**: `_incremental_learning.py` L~180-220
**测试**: 参数量 10K → Fisher 计算 <100ms (之前 OOM)

### 3.4 F4 — EWC 50% 遗忘修复 (High, 2周)

**根因**: 标准 EWC 的 λ 固定，5 任务后遗忘率 50%
**修复**: 分两步
1. **短期**: 自适应 λ (随任务数增加): `lambda = base_lambda * (1 + 0.2 * n_tasks)`
2. **中期 (Ch06)**: 替换为 Online EWC / SI

```python
# 修复: 自适应 EWC
def ewc_loss(self, params, base_lambda=100.0):
    adaptive_lambda = base_lambda * (1.0 + 0.2 * self._n_completed_tasks)
    loss = 0.0
    for (p, p_star, f) in zip(params, self._star_params, self._fisher_diag):
        loss += (adaptive_lambda / 2) * np.sum(f * (p - p_star) ** 2)
    return loss
```

**文件**: `_incremental_learning.py` L~280-350
**测试**: 5 任务序列后，任务1 准确率 ≥75% (之前 50%)

### 3.5 F5 — 多模态维度坍塌修复 (Critical, 2周)

**根因**: `VisionEncoder=32D` / `AudioEncoder=16D` / `ThermalEncoder=8D` (无可学习参数)
**修复**:

```python
class LearnableVisionEncoder:
    """可学习视觉编码器 — 轻量 MLP"""
    def __init__(self, input_size=64, output_dim=128):
        self.input_dim = input_size * input_size * 3  # 展平
        self.layers = [
            Linear(self.input_dim, 512), nn.ReLU(),
            Linear(512, 256), nn.ReLU(),
            Linear(256, output_dim),
        ]
        # 参数量: ~120K
```

**注意**: 完整 ViT 蒸馏在 Ch03 规划，此处仅修复"无可学习参数"的问题
**文件**: `_modality_encoders.py` 全文重写
**测试**: 不同图像产生不同向量 (之前猫/狗几乎相同)

### 3.6 F6 — JEPA 名不副实修复 (High, 2周)

**根因**: JEPAEncoder 输出 CausalWorldModelState (因果图)，不是潜向量
**修复**: 保留现有 JEPAEncoder 为 `CausalJEPAEncoder`，新增 `TrueJEPAEncoder` (Ch02)
**文件**: 新建 `_true_jepa_encoder.py` + 重命名现有文件
**测试**: `encoder.encode()` 返回 `np.ndarray` 形状 `(latent_dim,)`

### 3.7 F7 — 安全盲区修复 (High, 2周)

**根因**: 仅有 8 类物理安全，缺失 5 类
**修复**: 最小可行版 (每类 ≥1 个核心约束)

```python
class ContentSafetyConstraint(SafetyConstraint):
    BLOCKED_KEYWORDS = {"暴力", "自杀", "色情", ...}  # 基础关键词库
    
class CognitiveSafetyConstraint(SafetyConstraint):
    def check(self, state) -> SafetyCheckResult:
        # 幻觉检测: 检查输出与输入的一致性
        consistency = self._check_consistency(state)
        return SafetyCheckResult(passed=consistency > 0.7, ...)

class ValueAlignmentConstraint(SafetyConstraint):
    def check(self, state) -> SafetyCheckResult:
        # 意图对齐: 检查输出是否偏离用户目标
        alignment = self._check_alignment(state)
        return SafetyCheckResult(passed=alignment > 0.6, ...)
```

**文件**: 新建 5 个 `_safety_*.py` (每类 ~150 行)
**测试**: 每类约束各有 ≥5 个正例 + ≥5 个负例测试

### 3.8 F8 — Pearl 层级断裂修复 (High, 1周)

**根因**: L1/L2/L3 三个组件独立存在
**修复**: `PearlChain` 协调器 (详见 Ch02 3.1)
**最小修复**: 在 `_do_calculus.py` 的 `estimate_ate()` 末尾添加置信度更新钩子

### 3.9 F9 — 零形式化保证 (Medium, 降级为技术债)

**修复**: 在核心模块 docstring 中添加 `## Formal Guarantees` 段，列出已知数学性质
**不在本轮修复**: 完整形式化验证 (Ch05 规划)

### 3.10 F10 — 6参数线性JEPA修复 (Critical, 1.5周)

**根因**: `PendulumJEPAPredictor` 仅 6 参数线性模型
**修复**: 升级为 MLP

```python
class PendulumNeuralPredictor(ActionConditionedPredictor):
    """单摆神经网络预测器 — ≥10K 参数"""
    def __init__(self, hidden_dims=(64, 128, 64)):
        self._mlp = SimpleMLP(
            input_dim=3,  # theta, omega, torque
            hidden_dims=hidden_dims,
            output_dim=2,  # theta', omega'
        )
```

**文件**: `_action_conditioned_predictor.py` 新增类
**测试**: 在大角度 (|θ|>π/4) 任务上 MSE < 0.05

### 3.11 F11 — 穷举规划修复 (High, 2周)

**根因**: MultiBranchPredictor 暴力枚举 5³=125
**修复**: MCTS 规划器 (详见 Ch02 3.5)
**最小修复**: 添加 alpha-beta 剪枝到穷举搜索

### 3.12 F12 — VAE 反事实维度错误修复 (High, 0.5周)

**根因**: `_learned_counterfactual.py` state_dim=4 + action_dim=2 但 decoder 期望 4
**修复**: 统一维度

```python
# 修复: decoder 输入维度 = state_dim + action_dim
self._decoder = MLP(
    input_dim=self._latent_dim,
    hidden_dims=(64, 32),
    output_dim=self._state_dim + self._action_dim,  # 修复!
)
```

**文件**: `_learned_counterfactual.py` L~180
**测试**: 反事实查询不再抛出维度错误

## 4. 时间计划

| 周 | 任务 | 修复缺陷 | 里程碑 |
|---|---|---|---|
| W1 | F1 NSWM路由 + F12 VAE维度 | F1(Critical) + F12(High) | M1: 路由准确率 ≥70% |
| W2 | F2 PEM检索 + F3 Fisher + F5 维度 | F2(Critical) + F3(High) + F5(Critical) | M2: cosine≥0.70, 可学习编码 |
| W3 | F4 EWC遗忘 + F10 线性JEPA | F4(High) + F10(Critical) | M3: 遗忘率<25%, 非线性预测 |

> W1-W3 完成后，所有 **Critical** 级缺陷全部消除。

| W4-5 | F6 JEPA + F8 Pearl | F6(High) + F8(High) | M4: 潜空间+因果链 |
| W6-7 | F7 安全盲区 | F7(High) | M5: 13类安全 |
| W8-9 | F11 穷举规划 | F11(High) | M6: MCTS |

> W1-W9 完成后，所有 **High** 级缺陷全部消除。

## 5. 资源配置

| 资源 | 数量 | 成本 |
|---|---|---|
| 后端工程师 (P0) | 2人 × 3周 | 24 人天 |
| 后端工程师 (P1) | 1人 × 6周 | 24 人天 |
| 测试工程师 | 0.5人 × 6周 | 15 人天 |
| GPU (F5/F10 训练) | 按需 | $200 |

## 6. KPI 指标

| KPI | 基线 | 目标 | 度量 |
|---|---|---|---|
| Critical 缺陷数 | 4 | **0** | F1/F2/F5/F10 测试全部通过 |
| High 缺陷数 | 7 | **0** | F3/F4/F6/F7/F8/F11/F12 测试全部通过 |
| NSWM 路由准确率 | 0% (随机) | ≥70% | 24 词语义路由测试 |
| PEM cosine 准确率 | 0.50 (hash) | ≥0.80 | 语义对测试集 |
| 视觉编码可学习 | 0 参数 | ≥10K 参数 | `n_params` 属性 |
| EWC 遗忘率 | 50% (5任务) | <25% | 5 任务序列测试 |
| Fisher 计算复杂度 | O(N²) | O(N) | 10K 参数 <100ms |
| JEPA 预测非线性 | 线性 6参数 | MLP ≥10K参数 | 大角度任务 MSE<0.05 |

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|
| F1 TF-IDF 引入 sklearn 依赖 | 中 | 低 | 自实现 TF-IDF 或用 `pip install scikit-learn` |
| F5 可学习编码器训练不稳定 | 中 | 高 | 先用小 MLP 验证，再升级 ViT |
| F7 安全约束误杀合法输入 | 高 | 中 | 添加白名单 + 置信度阈值 |
| 修复过程破坏现有测试 | 中 | 高 | 每个修复后跑全量测试 |
| 3 周 P0 时间不够 | 中 | 中 | F7(安全)可延后到 W4-7 |

## 8. 成本预算

| 项目 | 人天 | 硬件/软件 |
|---|---|---|
| F1+F12 (W1) | 4 | $0 |
| F2+F3+F5 (W2) | 8 | $100 |
| F4+F10 (W3) | 8 | $100 |
| F6+F8 (W4-5) | 8 | $0 |
| F7 (W6-7) | 8 | $0 |
| F11 (W8-9) | 8 | $0 |
| 全量回归测试 | 12 | $0 |
| **小计** | **56** | **$200** |

## 9. 验收标准

### P0 验收 (W3 结束时)
- [ ] F1: NSWM 路由 — 24 个中文词中 ≥17 个 (70%) 路由到正确分支
- [ ] F2: PEM 检索 — "心率升高" vs "心率增加" cosine ≥0.70
- [ ] F5: VisionEncoder — 不同图像产生不同向量 (L2 > 0.1)
- [ ] F10: JEPA预测器 — PendulumJEPAPredictor 参数量 ≥10K
- [ ] F12: VAE反事实 — 反事实查询不抛出 ValueError

### P1 验收 (W9 结束时)
- [ ] F3: Fisher — 10K 参数 Fisher 计算 <100ms
- [ ] F4: EWC — 5 任务后任务1 准确率 ≥75%
- [ ] F6: JEPA — `TrueJEPAEncoder.encode()` 返回 (latent_dim,) ndarray
- [ ] F7: 安全 — SafetyMonitor 注册 13 类约束
- [ ] F8: Pearl — `PearlChain.full_analysis()` 端到端测试通过
- [ ] F11: MCTS — 倒立摆 10 步规划成功率 ≥80%

### 全量回归
- [ ] `pytest` ≥2600 passed, 0 failed
- [ ] `ruff check .` 全部通过
- [ ] `mypy` 检查通过

## 依赖关系

- **前置**: 无（最高优先级，最先执行）
- **被依赖**: Ch02(架构补强), Ch03(能力提升), Ch06(认知架构), Ch12(统一路径)
