# Ch02 架构层面：三支柱深度审计 — 改进规划书

## 1. 章节概述

原报告第二章对三大架构支柱进行了深度审计：
- **因果推理引擎**: Pearl L1/L2/L3 数学正确，但层级间不传递 (F8)
- **JEPA 潜空间**: 名不副实 (F6)，输出因果图而非潜向量；学习型预测器仅 6 参数线性 (F10)
- **安全约束**: 物理层 8 类完备，缺失内容/认知/价值/时序/社会 5 类安全 (F7)

**当前综合评分**: 架构 6.5/10，实现 4/10

## 2. 改进目标

| # | 目标 | 量化指标 | 优先级 |
|---|---|---|---|
| G1 | 修复 Pearl 层级断裂 | L1→L2→L3 自动推理链，反事实结果回写置信度 | P1 |
| G2 | 实现真正的 JEPA 潜空间编码 | 潜向量维度 ≥64，MSE < 0.01 | P0 |
| G3 | 通用 ActionConditionedPredictor | ≥10K 参数 MLP，支持任意 WorldState | P1 |
| G4 | 补齐 5 类安全约束 | ContentSafety/CognitiveSafety/ValueAlignment/TemporalSafety/SocialSafety | P1 |
| G5 | MultiBranchPredictor 升级 MCTS | 支持 horizon ≥10，节点 ≥1000 | P1 |

## 3. 实施方案

### 3.1 Pearl 因果推理链自动串联 (G1)

**目标**: `DoCalculus.estimate_ate()` → 自动触发 `CounterfactualEngine` → 结果回写 `CausalGraph` 置信度

```python
# 新增: PearlChain 协调器
class PearlChain:
    def full_analysis(self, X: str, Y: str, x_value: float):
        # L1: 观察 → 相关性
        obs = self._observe(X, Y)
        # L2: 干预 → ATE
        ate_result = self._do_calculus.estimate_ate(X, Y, x_value=x_value)
        # L3: 反事实 → 归因
        cf_result = self._counterfactual.query(X, Y, factual=x_value)
        # 回写: L3 归因 → L2 置信度 → L1 边权重
        self._update_confidence(ate_result, cf_result)
        return PearlChainResult(obs, ate_result, cf_result)
```

**文件**: `src/mci_world_model/sdk/_pearl_chain.py` (新建, ~400行)

### 3.2 通用 JEPA 潜空间编码器 (G2)

**目标**: 替换当前 JEPAEncoder 输出的 `CausalWorldModelState` 为真正的潜向量

```python
class TrueJEPAEncoder:
    """真正的 JEPA: 观测 → 潜向量 (不是因果图)"""
    def __init__(self, obs_dim=64, latent_dim=128, hidden_dim=256):
        # 编码器: obs → z (可微)
        self.encoder = MLP(obs_dim, hidden_dim, latent_dim)
        # 目标编码器 (momentum): obs → z_target
        self.target_encoder = MLP(obs_dim, hidden_dim, latent_dim)
    
    def encode(self, observations) -> np.ndarray:
        """返回 (latent_dim,) 潜向量"""
        return self.encoder.forward(observations)
    
    def predict_next(self, z_t, action) -> np.ndarray:
        """潜空间预测: z_t + a_t → z_{t+1}"""
        return self.predictor.forward(np.concatenate([z_t, action]))
```

**文件**: `src/mci_world_model/sdk/_true_jepa_encoder.py` (新建, ~500行)

### 3.3 通用 ActionConditionedPredictor (G3)

```python
class UniversalActionConditionedPredictor(ActionConditionedPredictor):
    """通用动作条件预测器 — ≥10K 参数 MLP"""
    def __init__(self, state_dim=16, action_dim=4, hidden_dims=(128, 256, 128)):
        self._mlp = MLP(
            input_dim=state_dim + action_dim,
            hidden_dims=hidden_dims,
            output_dim=state_dim,
        )
    
    def predict(self, state, action, n_steps=1):
        x = np.concatenate([state.to_vector(), action.to_vector()])
        for _ in range(n_steps):
            next_vec = self._mlp.forward(x)
            x = np.concatenate([next_vec, action.to_vector()])
        return [self._vec_to_state(next_vec)]
```

**文件**: `src/mci_world_model/sdk/_universal_action_predictor.py` (新建, ~350行)

### 3.4 五类新安全约束 (G4)

| 约束类 | 文件 | 行数 | 核心逻辑 |
|---|---|---|---|
| `ContentSafetyConstraint` | `_safety_content.py` | ~200 | 毒化/有害/伦理关键词过滤 |
| `CognitiveSafetyConstraint` | `_safety_cognitive.py` | ~250 | 幻觉检测 + 事实核查 + 不确定性阈值 |
| `ValueAlignmentConstraint` | `_safety_value.py` | ~200 | 用户意图对齐度 ≥0.8 |
| `TemporalSafetyConstraint` | `_safety_temporal.py` | ~150 | 因果倒置禁止 + 时间逻辑一致 |
| `SocialSafetyConstraint` | `_safety_social.py` | ~200 | 隐私保护 + 公平性 + 偏见检测 |

### 3.5 MultiBranchPredictor → MCTS (G5)

```python
class MCTSPlanner:
    """蒙特卡洛树搜索规划器"""
    def __init__(self, predictor, n_simulations=1000, exploration=1.41):
        self._predictor = predictor
        self._n_sim = n_simulations
        self._c = exploration
    
    def plan(self, state, goal, max_depth=20):
        root = MCTSNode(state)
        for _ in range(self._n_sim):
            node = self._select(root)       # UCB1 选择
            node = self._expand(node)       # 扩展
            reward = self._simulate(node, goal)  # rollout
            self._backpropagate(node, reward)    # 反向传播
        return root.best_child().action_sequence
```

**文件**: `src/mci_world_model/sdk/_mcts_planner.py` (新建, ~450行)

## 4. 时间计划

| 周 | 任务 | 交付物 | 里程碑 |
|---|---|---|---|
| W1 | JEPA 潜空间编码器核心 | `_true_jepa_encoder.py` | M1: 潜向量输出 |
| W2 | JEPA 预测器 + 训练循环 | 训练收敛 MSE<0.01 | M2: 可训练 |
| W3 | 通用 ActionConditionedPredictor | `_universal_action_predictor.py` | M3: 支持任意 WorldState |
| W4-5 | Pearl 因果链 PearlChain | `_pearl_chain.py` | M4: L1→L2→L3 串联 |
| W6-7 | 5 类新安全约束 | 5 个 `_safety_*.py` | M5: 13 类安全约束 |
| W8-9 | MCTS 规划器 | `_mcts_planner.py` | M6: horizon≥10 |
| W10-11 | 集成测试 + 性能验证 | 测试套件 + 基准数据 | M7: 全量通过 |

## 5. 资源配置

| 资源 | 数量 | 成本 |
|---|---|---|
| 后端工程师 (架构) | 1人 × 11周 | 44 人天 |
| 后端工程师 (安全) | 1人 × 4周 | 16 人天 |
| 测试工程师 | 0.5人 × 8周 | 20 人天 |
| GPU (训练 JEPA) | 按需 (可 CPU) | $500 (cloud GPU 20h) |
| **小计** | | **80 人天 + $500** |

## 6. KPI 指标

| KPI | 基线 | 目标 | 度量 |
|---|---|---|---|
| Pearl 链完整度 | L1/L2/L3 独立 | 三级自动串联 | `PearlChain.full_analysis()` 端到端测试通过 |
| JEPA 潜空间维度 | 因果图 (非潜向量) | ≥64 维潜向量 | `encoder.encode()` 返回 shape |
| JEPA 预测 MSE | N/A | < 0.01 | 单摆 ground truth 对比 |
| 通用预测器参数 | 6 参数 | ≥10,000 | `predictor.n_params` |
| 安全约束类型数 | 8 类 | 13 类 | `SafetyMonitor.constraint_count` |
| MCTS 规划 horizon | 3 (穷举) | ≥10 | 在倒立摆任务上 10 步规划成功率 |
| 测试通过率 | 2563 passed | ≥2700 passed | `pytest` 全量 |

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|
| TrueJEPA 训练不收敛 | 中 | 高 | 先用小维度验证，逐步放大 |
| PearlChain 回写导致循环更新 | 中 | 高 | 加 damping factor + 最大迭代次数 |
| 新安全约束与现有约束冲突 | 低 | 中 | 约束链独立评估，不短路 |
| MCTS 在复杂任务上过慢 | 中 | 中 | 设置 simulation 上限 + 并行化 |
| 通用预测器无法泛化 | 高 | 高 | 分阶段：先支持 Pendulum/Cart，再扩展 |

## 8. 成本预算

| 项目 | 人天 | 硬件/软件 |
|---|---|---|
| TrueJEPA 编码器 | 15 | $500 (GPU) |
| 通用 ActionPredictor | 10 | $0 |
| PearlChain | 12 | $0 |
| 5 类安全约束 | 16 | $0 |
| MCTS 规划器 | 12 | $0 |
| 集成测试 | 15 | $0 |
| **小计** | **80** | **$500** |

## 9. 验收标准

- [ ] `TrueJEPAEncoder` 输出 ≥64 维潜向量，训练 MSE < 0.01
- [ ] `PearlChain.full_analysis()` 端到端测试通过，L3 反事实回写 L2 置信度
- [ ] `UniversalActionConditionedPredictor` 参数 ≥10K，在 Pendulum 上 MSE < 0.05
- [ ] `SafetyMonitor` 注册 13 类约束，`check_all()` 全部通过
- [ ] `MCTSPlanner` 在倒立摆 10 步规划任务上成功率 ≥80%
- [ ] 全量测试 ≥2700 passed, 0 failed
- [ ] ruff + mypy 检查通过

## 依赖关系

- **前置**: Ch04 (F6 JEPA 修复, F10 线性预测器修复, F12 VAE 维度修复)
- **被依赖**: Ch03(能力四维度), Ch08(WMMM 成熟度), Ch10(知识蒸馏)
