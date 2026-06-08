# MCI World Model V3.0.7 → V3.0.8 → V3.1.0 实施方案

> **版本映射**：V3.7.0→V3.0.7 / V3.8.0→V3.0.8 / V4.0.0→V3.1.0
> **基线版本**：V3.0.6（能量-因果统一世界模型，382 tests passed）
> **架构原则**：纯 CPU / numpy + scipy + MLX（零 transformers / PyTorch 硬依赖）

---

## 全局版本引用变更清单

| 文件 | 行号 | 当前 | 改为 |
|------|------|------|------|
| `_world_model.py` | L14 | `v3.7.0 L2` | `v3.0.7 L2` |
| `_world_model.py` | L16 | `v3.8.0 L3` | `v3.0.8 L3` |
| `_world_model.py` | L120 | `v3.8.0 L3` | `v3.0.8 L3` |
| `_world_model.py` | L124 | `v4.0.0 JEPA` | `v3.1.0 JEPA` |
| `_world_model.py` | L169 | `v4.0.0 JEPA` | `v3.1.0 JEPA` |
| `_world_model.py` | L295 | `v4.0.0` | `v3.1.0` |
| `_world_model.py` | L363 | `v4.0.0` | `v3.1.0` |
| `_world_model.py` | L375 | `v4.0.0` | `v3.1.0` |
| `_world_model.py` | L600 | `v3.7.0 L2` | `v3.0.7 L2` |
| `_world_model.py` | L606 | `v3.8.0 L3` | `v3.0.8 L3` |
| `_world_model.py` | L630 | `v4.0.0` | `v3.1.0` |
| `_world_model.py` | L643 | `v4.0.0 JEPA` | `v3.1.0 JEPA` |
| `_world_model.py` | L647 | `v3.7.0` | `v3.0.7` |
| `_world_model.py` | L737 | `v4.0.0 JEPA` | `v3.1.0 JEPA` |
| `_world_model.py` | L745 | `v4.0.0 JEPA` | `v3.1.0 JEPA` |
| `_world_model.py` | L1979-1981 | `v3.6.0/v3.7.0/v3.8.0` | `v3.0.7/v3.0.8/v3.1.0` |
| `_jepa_trainer.py` | L2 | `su-memory v4.0.0` | `MCI World Model v3.1.0` |
| `_jepa_gnn.py` | L2 | `su-memory v4.0.0` | `MCI World Model v3.1.0` |
| `_jepa_gat_encoder.py` | L2 | `su-memory v4.0.0` | `MCI World Model v3.1.0` |
| `_jepa_encoder.py` | L2 | `su-memory v4.0.0` | `MCI World Model v3.1.0` |
| `_jepa_predictor.py` | L2 | `su-memory v4.0.0` | `MCI World Model v3.1.0` |
| `_parametric_memory.py` | L2 | `su-memory v3.6.0` | `MCI World Model v3.0.7` |
| `_counterfactual.py` | L2 | `su-memory v3.8.0` | `MCI World Model v3.0.8` |
| `_cost_module.py` | L2 | `su-memory v3.0.1` | `MCI World Model v3.0.7` |
| `pyproject.toml` | L7 | `3.0.6` | `3.0.7` |
| `README.md` | L219-222 | V4.0.0 规划 | V3.1.0 规划 |

---

## V3.0.7 — 参数化记忆觉醒（MLX Native）

### 现状诊断（`_parametric_memory.py` 860行）

| 行号 | 当前状态 | 问题 |
|------|----------|------|
| L2 | `su-memory v3.6.0` | 旧版本标识 |
| L190-208 | `_load_mlx_model()` 加载 Qwen via `mlx_lm.load` | 仍加载 1.5B 大模型 |
| L210-247 | `_load_torch_model()` + `BitsAndBytesConfig` | torch/transformers 硬依赖 |
| L393-466 | `_train_mlx()` → L401 `from mlx_lm import lora, tuner` | QLoRA 特定导入 |
| **L441** | **`self._simulated_training_step(batch_texts, step, n_steps)`** | **训练是桩实现！返回 `0.5*0.99^step + noise`** |
| L468-532 | `_train_torch()` + `LoraConfig`/`get_peft_model` | peft 硬依赖 |
| L641-658 | `_save_adapter_mlx()` 写 `"version": "3.6.0"` | 版本标识过期 |
| **L773-795** | **`_predict_mlx()`/`_predict_torch()` 返回 `[MLX 参数化预测 #i]`** | **推理是桩实现！** |
| L548-605 | `_validate_adapter_path()` 路径白名单 | ✅ 安全机制完善，保留 |
| L257-335 | `prepare_training_data()` 多格式兼容 | ✅ 逻辑完整，保留 |

### 技术路线

```
┌─────────────────────────────────────────────────────────┐
│                   V3.0.7 MLX-Native 训练管线               │
├─────────────────────────────────────────────────────────┤
│  Reflection QA Pairs (M5 管线)                            │
│       │                                                   │
│       ▼                                                   │
│  prepare_training_data() ─── TrainingSample[] (保留)      │
│       │                                                   │
│       ▼                                                   │
│  CausalMLP (小型, ~15K params, 纯 mlx.core)               │
│  · 替代 Qwen2.5-1.5B + LoRA                              │
│  · 复用 JEPA 手写梯度范式 (mlx.value_and_grad)            │
│       │                                                   │
│       ▼                                                   │
│  L_total = L_category + α·L_energy + β·L_rho             │
│       │                                                   │
│       ▼                                                   │
│  mlx.nn.value_and_grad → mx.optimizers.SGD.step          │
│       │                                                   │
│       ▼                                                   │
│  保存 adapter (mx.save_safetensors → .safetensors)        │
│  加载 adapter → predict() → 真实的五维因果概率             │
└─────────────────────────────────────────────────────────┘
```

**核心设计决策：放弃 Qwen2.5-1.5B + QLoRA 路线，改为自研小型 CausalMLP。**

原因：
1. Qwen 需要 transformers/PyTorch/peft 依赖链，违反"CPU-first"架构哲学
2. JEPA 已有完整的参数化训练范式（`_jepa_trainer.py` 手写梯度 GNN/GAT，纯 numpy），复用同一范式是工程最优解
3. 因果推断任务不需要 1.5B 参数——结构化的因果边预测（五范畴分类 + rho 回归）可以被小型 MLP 高质量完成
4. `_train_mlx()` 当前是桩实现（`_simulated_training_step`），全链路等待替换

### 关键实现步骤

#### T1: 重构 `_parametric_memory.py` — 移除 transformers/torch 依赖链

**文件**: `src/mci_world_model/sdk/_parametric_memory.py` (860行 → ~600行)

**变更**:
1. 删除 `_load_torch_model()` (L210-247, 38行) + `_train_torch()` (L468-532, 65行) + `_save_adapter_torch()` (L661-677, 17行)
2. 删除 `import torch` / `from transformers import` / `from peft import` 三个 import 块
3. `_load_mlx_model()` → 重命名为 `_load_model()`，删除 `from mlx_lm import lora, tuner`（不再需要 QLoRA）
4. `_train_mlx()` (L393-466) → 重写为 `_train()`，`_simulated_training_step` 替换为真正的 MLX 训练循环
5. `ParametricMemoryConfig` 清理：删除 `lora_*` / `quant_*` / `use_bfloat16` / `base_model` 等 QLoRA 专属字段
6. `_predict_mlx()`/`_predict_torch()` → 合并为 `_predict()`，调用 `CausalMLP.forward()`
7. `_save_adapter_mlx()` 中的 `"version": "3.6.0"` → `"3.0.7"`
8. 保留 `_validate_adapter_path()` (安全机制)、`prepare_training_data()` (数据管道)

#### T2: 实现 `CausalMLP` — 小型因果预测网络

**新建文件**: `src/mci_world_model/sdk/_causal_mlp.py` (~300行)

```python
class CausalMLP:
    """
    V3.0.7: 因果推断小型 MLP — MLX Native 实现。

    架构:
        Input(D=128) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(5)
           ↑                                                      ↓
       cause_text_embed                              [P(semantic|cause),
                                                       P(causal|cause),
                                                       P(spacetime|cause),
                                                       P(generative|cause),
                                                       P(trust|cause)]

    训练目标:
        L_category = CrossEntropy(predicted_category, true_energy_category)
        L_rho = MSE(predicted_rho, true_rho)
        L_total = L_category + 0.1 * L_rho

    参数量: ~15K (embedding_layer: 5000×128 + 3 层 Linear)
    训练时间: ~1min / 3000 samples (M5 Pro, 纯 MLX)
    """
    
    def __init__(self, input_dim=128, hidden_dims=(64, 32), vocab_size=5000):
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 5)  # 五范畴输出

    def forward(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        return nn.softmax(self.fc3(x))  # (B, 5) 因果类别概率
```

**关键**: 复用 `mlx.core` 的计算图，与 JEPA 训练范式（`mlx.value_and_grad` + `SGD`）完全一致。

#### T3: 实现 MLX 原生训练循环（替代桩实现）

**在 `_parametric_memory.py` 的 `_train()` 方法中**：

```python
def _train(self, energy_loss_fn=None) -> dict:
    """MLX 原生训练循环（替代 L441 的 _simulated_training_step 桩实现）。"""
    import mlx.core as mx
    import mlx.nn as nn
    from mci_world_model.sdk._causal_mlp import CausalMLP

    if self._model is None:
        self._model = CausalMLP(input_dim=128, hidden_dims=[64, 32])

    # 嵌入训练数据
    embeddings, y_category, y_rho = self._embed_training_data()

    def loss_fn(model, x, y_cat, y_rho):
        logits = model.forward(x)  # (B, 5)
        l_cat = mx.mean(nn.losses.cross_entropy(logits, y_cat))
        # rho 回归: 取最大概率类别的索引作为连续值代理
        pred_rho = mx.argmax(logits, axis=1).astype(mx.float32) / 4.0
        l_rho = mx.mean((pred_rho - y_rho) ** 2)
        return l_cat + 0.1 * l_rho

    loss_and_grad = mx.value_and_grad(loss_fn)
    optimizer = mx.optimizers.SGD(learning_rate=self.config.learning_rate)

    n_steps = min(len(embeddings) // self.config.batch_size, 500)
    losses = []
    for step in range(n_steps):
        batch = get_batch(embeddings, y_category, y_rho, step, self.config.batch_size)
        loss, grads = loss_and_grad(self._model, *batch)
        optimizer.update(self._model, grads)
        mx.eval(self._model.parameters(), optimizer.state)
        losses.append(float(loss))

    self._is_trained = True
    return {"backend": "mlx", "n_steps": n_steps, 
            "final_loss": round(float(np.mean(losses)), 6)}
```

#### T4: 与 `JEPATrainer` 集成

`_jepa_trainer.py` 新增 `_train_parametric_step()`（~40行）：

```python
def _train_parametric_step(self, s_t, s_t1, learning_rate=0.01) -> float:
    """
    使用 JEPA 编码器输出作为参数化记忆的训练信号。
    流程: encoder.encode(mem_t) → s_t 的因果边 → 训练标签
          parametric_memory.predict(cause_text) → 预测的因果分布 → 损失
    """
    if not hasattr(self, '_parametric_memory') or self._parametric_memory is None:
        return 0.0
    edges = s_t.causal_edges
    if not edges:
        return 0.0
    # 从因果边提取训练信号
    cause_texts = [e.get('cause', '') for e in edges if e.get('cause')]
    true_categories = [self._energy_to_category(e.get('energy_relation', 'neutral')) for e in edges]
    if not cause_texts:
        return 0.0
    # 调用参数化记忆训练
    stats = self._parametric_memory.train_on_signals(cause_texts, true_categories)
    return stats.get('final_loss', 0.0)
```

### 接口关系

```
V3.0.6 (现有)                                     V3.0.7 (变更)
─────────────────                                 ─────────────────
MCIWorldModel.__init__()                          _parametric 字段保留
  self._parametric = None                         → ParametricMemory(CausalMLP)
ParametricMemoryConfig                            清理 QLoRA 专属字段
  · lora_rank=64, lora_alpha=128                  → 删除，新增 input_dim/hidden_dims
  · base_model="Qwen/Qwen2.5-1.5B-Instruct"       → 删除
  · quant_bits=4, use_bfloat16=True               → 删除
ParametricMemory.load_base_model()                ParametricMemory._load_model()
  → mlx_lm.load(Qwen) 或 torch+BitsAndBytes       → CausalMLP() 直接构造（无需加载大模型）
ParametricMemory._train_mlx()                     ParametricMemory._train()
  → _simulated_training_step (桩)                  → mlx.value_and_grad + SGD (真实)
ParametricMemory._predict_mlx()                   ParametricMemory._predict()
  → "[MLX 参数化预测 #i]" (桩)                     → CausalMLP.forward() → 五维概率
ParametricMemory._save_adapter_mlx()              ParametricMemory._save_adapter()
  → 写 "version": "3.6.0"                         → 写 "version": "3.0.7"
JEPATrainer.train()                               JEPATrainer._train_parametric_step()
  → GNN 训练循环                                   → MLP 训练循环 (复用同一训练器)
```

### 测试验证方案

| 测试 | 描述 | 验收标准 |
|------|------|----------|
| `test_mlp_forward` | CausalMLP 前向传播 | 输入 (128,) → 输出 (5,)，概率和 ≈ 1.0 |
| `test_mlp_training_converges` | 合成数据上损失递减 | 100 epochs 后 loss < 初始的 50% |
| `test_no_torch_import` | grep torch/transformers/peft `_parametric_memory.py` | 零匹配 |
| `test_config_no_qlora_fields` | ParametricMemoryConfig 属性列表 | 不含 lora_rank/base_model/quant_bits |
| `test_prepare_training_data` | QA pairs → TrainingSample | 至少 50% 的有效对通过置信度过滤 |
| `test_predict_returns_valid` | predict("物价上涨") | 返回 3 条候选，每条含 energy_relation ≠ "neutral" |
| `test_save_load_adapter` | save → load 往返 | 预测结果完全一致 |
| `test_jepa_integration` | JEPATrainer 调用 parametric_step | loss 可计算，梯度可传播 |
| `test_adapter_path_security` | 传入 `/etc/passwd` / `../escape` | 抛出 ValueError |

---

## V3.0.8 — 反事实推理增强

### 现状诊断（`_counterfactual.py` 796行）

| 行号 | 当前状态 | 问题/评估 |
|------|----------|-----------|
| L2 | `su-memory v3.8.0` | 旧版本标识 |
| L136-227 | `StructuralEquationModel` — 线性 SEM | ✅ 架构清晰，Kahn 拓扑排序，但仅支持线性 V_i = Σβ·V_j + U_i |
| L231-259 | `simulate()` — 逐样本循环 | ⚠️ 可向量化：`for node_i in topo` 内嵌 `np.zeros((n_samples, n_nodes))` |
| L265-320 | `abduce()` — 溯因推断 | ✅ 逻辑完整：观测节点 → 噪声回算，未观测 → 随机采样+回填 |
| L326-356 | `intervene()` — mutilated SEM | ✅ 正确实现 do-calculus 图手术：入边置零 |
| L362-401 | `simulate_with_intervention()` | ✅ 干预节点固定值，非干预节点用溯因噪声 |
| L450-481 | `from_causal_graph()` — CausalGraph→SEM | ✅ 单向转换已实现 |
| L499-643 | `query()` — Pearl 三步端到端 | ✅ 完整实现：值消毒 → 溯因 → 干预 → 预测 → CI |
| L649-758 | `_compute_pns()` — PN/PS/PNS | ⚠️ 两次独立 MC 模拟（factual + counterfactual），可共享噪声样本减少一半计算 |
| **L764-788** | **`batch_query()` — for 循环** | **❌ 不是真正批量！逐场景串行调用 query()，无矩阵化加速** |
| — | NonlinearSEM | ❌ 缺失：无 tanh/ReLU/sigmoid 支持 |
| — | CausalGraph.to_sem() | ❌ 缺失：CausalGraph 无反向导出 SEM 方法 |

### 技术路线

```
┌───────────────────────────────────────────────────────────┐
│                 V3.0.8 反事实推理增强                       │
├───────────────────────────────────────────────────────────┤
│                                                             │
│  现有能力 (v3.0.6, _counterfactual.py 796行):               │
│  ✅ StructuralEquationModel (线性 SEM, Kahn 拓扑排序)       │
│  ✅ abduce() → 噪声后验 (观测/未观测分区处理)               │
│  ✅ query() → Pearl 三步端到端 + 95% CI                    │
│  ✅ PN/PS/PNS → Monte Carlo 必要性/充分性 (300样本)        │
│  ✅ from_causal_graph() → CausalGraph→SEM                  │
│                                                             │
│  V3.0.8 增强:                                               │
│  ✅ NonlinearSEM → tanh/ReLU/sigmoid 变换                   │
│  ✅ BatchCounterfactualEngine → 矩阵化 O(N) 批量查询        │
│  ✅ CausalGraph.to_sem() → 双向转换闭环                     │
│  ✅ PN/PS/PNS 共享噪声 → 计算量减半                         │
│  ✅ 反事实基准测试套件 (5 个合成基准 × 已知真值)            │
│  ─────────────────────────────────────────                   │
│                                                             │
└───────────────────────────────────────────────────────────┘
```

### 关键实现步骤

#### T1: 扩展 SEM 为非线性版本

**文件**: `src/mci_world_model/sdk/_counterfactual.py`（修改 ~150行）

当前 `StructuralEquationModel` 仅支持线性变换：
```
V_i = Σ β_{ji}·V_j + U_i
```

扩展为支持非线性激活（在 `__init__` 新增 `activation` 参数）：

**涉及修改的方法**（均在 parent_sum 计算后插入 `_apply_activation`）：
- `simulate()` (L231-259)：`data[:, node_i] = self._apply_activation(parent_sum) + noise`
- `abduce()` (L265-320)：观测节点噪声回算需考虑激活函数的逆
- `simulate_with_intervention()` (L362-401)：同 simulate

```python
class StructuralEquationModel:
    def __init__(self, coefficients, node_names, noise_std=0.5,
                 activation: str = "linear", seed=None):
        self.activation = activation
        self._activation_fn = {
            "linear": lambda x: x,
            "relu": lambda x: np.maximum(0, x),
            "tanh": lambda x: np.tanh(x),
            "sigmoid": lambda x: 1 / (1 + np.exp(-np.clip(x, -50, 50))),
        }[activation]

    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        return self._activation_fn(x)
```

**关键**: `abduce()` 中观测节点的噪声回算需考虑非线性。对 linear/relu 可解析逆，对 tanh/sigmoid 需用 `arctanh`/`logit` 逆函数。

#### T2: 批量反事实引擎（替代 L764-788 的 for 循环）

**新建文件**: `src/mci_world_model/sdk/_batch_counterfactual.py` (~400行)

```python
class BatchCounterfactualEngine:
    """
    V3.0.8: 向量化批量反事实查询引擎。

    当前 batch_query() (L764-788) 只是 for 循环调用 query()，
    时间复杂度 O(N·M·D) (N=场景数, M=MC样本, D=节点数)。

    向量化版本将 evidence/do_x 矩阵化后单次大矩阵模拟:
    - 噪声采样: (N, M, D) 矩阵 — 单次 numpy 调用
    - SEM 模拟: 按拓扑排序广播 — 利用 numpy 向量化
    - 目标切片 + 统计: 沿 axis=1 聚合

    时间复杂度 O(M·D) — 与 N 无关！

    典型用例: 药物剂量-效果反事实分析
        scenarios = [
            {"evidence": {"dose": 50, "albumin": 30}, "do_x": {"dose": 100}, "target": "albumin"},
            {"evidence": {"dose": 75, "albumin": 32}, "do_x": {"dose": 50},  "target": "albumin"},
            ...  # 100+ 场景
        ]
        results = engine.batch_query(scenarios)  # < 1s
    """

    def batch_query(self, scenarios: list[dict], n_mc: int = 200
                    ) -> list[CounterfactualResult]:
        # 1. 收集所有场景的 evidence/do_x → 构建 (N, D) 矩阵
        # 2. 单次 abduce: noise[N, M, D] — 利用 numpy 广播
        # 3. 单次 SEM 模拟: (N, M, D) → 按拓扑排序逐层广播
        # 4. 按 target_idx 切片 → np.mean(axis=1) → CounterfactualResult
        ...
```

#### T3: CausalGraph ↔ SEM 双向转换

**在 `_do_calculus.py` 中新增**（~50行）：

```python
class CausalGraph:
    def to_sem(self, noise_std: float = 0.5,
               activation: str = "linear") -> StructuralEquationModel:
        """
        将 CausalGraph 的邻接矩阵转为 SEM 系数矩阵。

        CausalGraph.adjacency[i, j] → SEM.coefficients[i, j]
        非零边保留权重，零边保持零。
        """
        from mci_world_model.sdk._counterfactual import StructuralEquationModel
        return StructuralEquationModel(
            coefficients=np.array(self.adjacency, dtype=np.float64),
            node_names=list(self.nodes),
            noise_std=noise_std,
            activation=activation,
        )

    @staticmethod
    def from_sem(sem: StructuralEquationModel) -> CausalGraph:
        """从 SEM 系数矩阵反向构建 CausalGraph。非零系数 → 边存在。"""
        cg = CausalGraph(nodes=sem.node_names)
        for i in range(sem.n_nodes):
            for j in range(sem.n_nodes):
                if sem.coefficients[i, j] != 0:
                    cg.add_edge(sem.node_names[i], sem.node_names[j],
                                weight=sem.coefficients[i, j])
        return cg
```

#### T4: 反事实基准测试套件

**新建文件**: `tests/test_v308_counterfactual_benchmark.py` (~350行)

| 基准 | 因果结构 | 已知真值 | 评估指标 |
|------|----------|----------|----------|
| `Frontdoor` | Z→X→Y, Z confounding | `Y_{x'} = β*(α*Z + U_Z) + U_Y` | ITE 绝对误差 < 0.1 |
| `M_graph` | X→M→Y, X→Y (mediation) | NDE=β_XY, NIE=β_XM*β_MY | 分解误差 < 0.05 |
| `Collider` | X→Z←Y (no causal X→Y) | `Y_{x'} = Y_x` | ITE ≈ 0 ± 0.05 |
| `Chain3` | A→B→C→D (3-hop) | 线性可逆解析解 | 因果方向正确 |
| **`Nonlinear`** | **X→Y with tanh** | **`Y = tanh(0.5*X) + N(0,0.1)`** | **非线性拟合 R² > 0.9** |

### 接口关系

```
V3.0.6 (现有)                                     V3.0.8 (变更)
─────────────────                                 ─────────────────
CounterfactualEngine.__init__(sem, node_names)    CounterfactualEngine.__init__(sem, node_names)
  → sem: StructuralEquationModel(linear only)      → sem: StructuralEquationModel(activation="tanh")
CounterfactualEngine.query()                      CounterfactualEngine.query()
  → Pearl 三步 (L499-643)                          → 不变（已完整实现）
CounterfactualEngine.batch_query()                BatchCounterfactualEngine.batch_query()
  → for s in scenarios: query(s) (L764-788)        → 矩阵化 (N,M,D) 单次模拟
CounterfactualEngine._compute_pns()               CounterfactualEngine._compute_pns()
  → 两次独立 MC (factual + cf, L649-758)           → 共享 noise_samples，计算量减半
CausalGraph (无 to_sem)                           CausalGraph.to_sem() + from_sem()
  → CounterfactualEngine.from_causal_graph()        → 双向转换闭环
```

### 测试验证方案

| 测试 | 验收标准 |
|------|----------|
| `test_nonlinear_sem_simulate` | tanh SEM 的 simulate() 输出在 [-1, 1] 范围 |
| `test_nonlinear_abduce_inverse` | tanh SEM abduce + simulate 往返误差 < 1e-6 |
| `test_batch_query_performance` | 100 场景 × 200 MC < 2s（当前 for 循环 ~5s） |
| `test_batch_vs_serial_consistency` | 批量查询结果与串行一致（Δ < 1e-4） |
| `test_cg_to_sem_roundtrip` | CausalGraph → SEM → CausalGraph 边不丢失 |
| `test_sem_to_cg_roundtrip` | SEM → CausalGraph → SEM 系数一致 |
| `test_frontdoor_truth` | ITE 偏差 < 0.1（与解析解对比） |
| `test_collider_no_effect` | Collider 结构下 ITE ≈ 0 |
| `test_pns_monte_carlo` | 已知 SEM 的 PNS 理论值 vs MC 估计偏差 < 0.05 |
| `test_nonlinear_fit_r2` | Nonlinear 基准的 R² > 0.9 |

---

## V3.1.0 — 物理世界应用

### 现状诊断

| 文件 | 行号 | 当前状态 | 问题/评估 |
|------|------|----------|-----------|
| `_jepa_encoder.py` | L2 | `su-memory v4.0.0` | 旧版本标识 |
| `_jepa_encoder.py` | L75-109 | `encode(memories, use_parametric)` | ✅ 仅接受文本记忆，无物理信号入口 |
| `_jepa_encoder.py` | L269-280 | `_build_edge_features()` | ✅ 已提取 rho/confidence/bayes_factor 特征 |
| `_jepa_encoder.py` | L282-337 | `from_graph_tensors()` | ✅ 已支持 `metadata` (含 temporal_info/cognitive_gaps) |
| `_jepa_predictor.py` | L2 | `su-memory v4.0.0` | 旧版本标识 |
| `_jepa_predictor.py` | L62 | `predict(state: CausalWorldModelState)` | ✅ 接口清晰，但无物理信号支持 |
| `_jepa_predictor.py` | L74-104 | `evaluate(dataset)` | ✅ 已支持 distance-based 评估 |
| `_jepa_trainer.py` | L2 | `su-memory v4.0.0 M2` | 旧版本标识 |
| `_jepa_trainer.py` | L140-212 | `train()` M1/M2/M3 三模式 | ✅ 架构完整，支持 baseline/GNN/E2E |
| `_jepa_trainer.py` | L108-110 | `EnergyCostModule` 集成 | ✅ 已集成能量守恒损失 |
| `_perception.py` | — | `process()` | ❌ 仅文本，无 MultimodalSignal 支持 |
| — | — | PhysicalGraphBuilder | ❌ 缺失 |

### 技术路线

```
┌───────────────────────────────────────────────────────────────┐
│                    V3.1.0 物理世界因果建模                       │
├───────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: 多模态信号感知 (新增)                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ PerceptionPipeline.process_multimodal()                   │  │
│  │   + MultimodalSignal: numerical/temporal_series/lab      │  │
│  │   + 物理量 → 五范畴映射器 (ENERGY_PHYSICAL_MAP)           │  │
│  └─────────────────────────────────────────────────────────┘  │
│       │                                                         │
│       ▼                                                         │
│  Phase 2: 物理量 → 因果图 (新增)                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ PhysicalGraphBuilder.build_graph(timeline)                 │
│  │   → 时序物理量 → CausalWorldModelState.causal_edges       │
│  │   → 复用现有 JEPA 编码器 to_graph_tensors()                │
│  └─────────────────────────────────────────────────────────┘  │
│       │                                                         │
│       ▼                                                         │
│  Phase 3: JEPA 物理预测 (适配现有)                              │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ JEPAEncoder.encode(signals=[...])  ← 新增 signals 参数    │
│  │   → CausalWorldModelState (含 physical edges)             │
│  │ JEPAPredictor.predict(s_t)  ← 现有接口不变                 │
│  │   → s_{t+1} (预测下一时刻的物理量分布)                     │
│  │ JEPATrainer.train()  ← 复用 M1/M2/M3 模式                  │
│  └─────────────────────────────────────────────────────────┘  │
│       │                                                         │
│       ▼                                                         │
│  Phase 4: 临床营养场景基准 (新增)                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ 模板因果图: 营养干预 → 生化指标 → 临床结局               │  │
│  │ 合成数据: 100 患者 × 30 天时序 (已知真值因果结构)        │  │
│  │ 反事实查询: "如果第 5 天增加 500kcal 摄入..."             │  │
│  │ 端到端: PerceptionPipeline → PhysicalGraphBuilder        │  │
│  │         → JEPAEncoder → JEPAPredictor → Counterfactual   │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└───────────────────────────────────────────────────────────────┘
```

### 关键实现步骤

#### T1: 扩展 `PerceptionPipeline` 支持多模态信号

**文件**: `src/mci_world_model/_sys/_perception.py`（修改 ~150行）

```python
from dataclasses import dataclass
from enum import Enum

class SignalType(Enum):
    TEXT = "text"
    NUMERICAL = "numerical"          # 单值 (血糖 5.6)
    TEMPORAL_SERIES = "temporal"     # 时序 (每日白蛋白: [30,32,31,33])
    LAB_STRUCTURED = "lab"           # 实验室检查 (多项指标)
    CATEGORICAL = "categorical"      # 类别 (NRS2002 评分: 4)

@dataclass
class MultimodalSignal:
    signal_type: SignalType
    value: any                       # 原始值
    timestamp: str                   # ISO datetime
    source: str                      # "lab_report" | "nursing_note" | ...
    metadata: dict = field(default_factory=dict)

class PerceptionPipeline:
    def process_multimodal(self, signals: list[MultimodalSignal]) -> list[dict]:
        """
        将多模态信号统一转换为因果发现可用的结构化特征。

        数值信号 → 五范畴映射:
            numerical → 离散化为 categorical bins
            temporal_series → FourierCausal 频域特征
            lab_structured → {"feature_name": value} dict
        """
```

#### T2: 物理量 → 因果边转换器

**新建文件**: `src/mci_world_model/sdk/_physical_graph_builder.py` (~300行)

```python
class PhysicalGraphBuilder:
    """
    V3.1.0: 物理量 → 因果边转换器。

    将数值时序数据转换为 CausalWorldModelState.causal_edges 格式，
    使现有 JEPA 编码器（to_graph_tensors）可处理物理世界信号。

    核心映射:
        物理量 (albumin: 35 → 32) → 因果边:
            {cause: "albumin_t", effect: "albumin_t+1",
             rho: -0.3, energy_relation: "suppress",
             cause_energy: "generative", effect_energy: "trust"}
    """

    # 预定义的五范畴-物理量映射
    ENERGY_PHYSICAL_MAP = {
        "semantic":   ["diagnosis_code", "chief_complaint"],
        "causal":     ["medication_dose", "intervention_type"],
        "spacetime":  ["timestamp", "los_days", "season"],
        "generative": ["albumin", "prealbumin", "calorie_intake", "protein_intake"],
        "trust":      ["nrs2002", "apache_ii", "evidence_level"],
    }

    def build_graph(self, patient_timeline: list[dict]) -> list[dict]:
        """
        将患者时序数据转换为因果边列表。

        Args:
            patient_timeline: [
                {"day": 1, "albumin": 30, "calorie_intake": 1200, ...},
                {"day": 2, "albumin": 32, "calorie_intake": 1400, ...},
                ...
            ]

        Returns:
            causal_edges — JEPA 编码器 to_graph_tensors() 可直接消费
        """
```

#### T3: JEPA 编码器适配物理状态

**文件**: `src/mci_world_model/sdk/_jepa_encoder.py`（修改 ~60行）

`JEPAEncoder.encode()` 新增 `signals` 参数（保持 `memories` 向后兼容）：

```python
def encode(self, memories=None, signals=None,
           use_parametric=False) -> CausalWorldModelState:
    """
    Args:
        memories: 文本记忆列表 (原有路径，保持兼容)
        signals: 多模态信号列表 (V3.1.0 新增物理世界路径)
    """
    if signals is not None:
        from mci_world_model.sdk._physical_graph_builder import PhysicalGraphBuilder
        builder = PhysicalGraphBuilder()
        edges = builder.build_graph(signals)
        return CausalWorldModelState(
            causal_edges=edges,
            n_novel=len(edges),
            timestamp=signals[0].timestamp if signals else "",
        )
    if memories is not None:
        return self._wm.discover(memories, use_parametric=use_parametric)
    return CausalWorldModelState.empty()
```

**关键**: 现有 `to_graph_tensors()` 和 `evaluate()` 接口保持不变——物理边通过 `PhysicalGraphBuilder` 转换为标准 `causal_edges` 格式后，全链路兼容。

#### T4: 与 `JEPATrainer` 集成

`JEPATrainer.train()` 已支持 M1/M2/M3 三种模式（L140-212），物理世界数据通过以下路径零修改接入：

```
PhysicalGraphBuilder.build_graph() → causal_edges
  → CausalWorldModelState
    → JEPAEncoder.to_graph_tensors() → (adj, node_feat, edge_feat)
      → JEPAPredictor.predict() / GNNPredictor.training_predict()
        → JEPATrainer._compute_loss() / _train_gnn_step() / _train_e2e_step()
```

**无需修改 `JEPATrainer`** ——物理世界信号通过标准 `CausalWorldModelState` 接口接入，训练循环透明。

#### T5: 临床营养场景合成数据集 + 基准

**新建文件**: `tests/test_v310_clinical_benchmark.py` (~350行)

```python
def generate_synthetic_patient(seed=42, n_days=30):
    """
    生成合成患者时序数据 (已知真值因果结构)。

    因果结构:
        calorie_intake(t) → albumin(t+3)      [延迟效应, β=0.6]
        protein_intake(t) → prealbumin(t+2)    [延迟效应, β=0.5]
        albumin(t) → nrs2002_score(t+7)        [长延迟, β=-0.3]
        medication_dose(t) → albumin(t+1)      [短延迟, β=0.4]
    """
    ...

def test_clinical_counterfactual():
    """
    验证: 反事实干预的因果一致性。

    Scenario: 患者第 5 天增加 500kcal 摄入
        factual:   calorie_intake[5]=1200, albumin[15]=32
        do_x:      calorie_intake[5]=1700
        预测:      albumin[15] 应 > 32 (β≈0.6 的累积效应)
    """
    ...
```

### 接口关系

```
V3.0.6 (现有)                                     V3.1.0 (新增/变更)
─────────────────                                 ─────────────────
PerceptionPipeline.process()                      PerceptionPipeline.process_multimodal()
  → text → features                               → MultimodalSignal → features
JEPAEncoder.encode(memories)                      JEPAEncoder.encode(memories=None, signals=[...])
  → discover() → state                            → PhysicalGraphBuilder → state (物理路径)
JEPAEncoder.to_graph_tensors(state)               JEPAEncoder.to_graph_tensors(state)  ← 不变
  → (adj, node_feat, edge_feat)                    → 物理边同样转为标准图张量
JEPAPredictor.predict(state)                      JEPAPredictor.predict(state)  ← 不变
  → causal edges 预测                              → physical edges 预测 (数值 rho)
JEPATrainer.train()                               JEPATrainer.train()  ← 零修改！
  → M1/M2/M3 训练循环                              → 物理世界数据透明接入
CounterfactualEngine.query()                       CounterfactualEngine.query()  ← 不变
  → {"counterfactual_value": 3.5}                  → {"counterfactual_value": 35.2} (物理量)
```

### 测试验证方案

| 测试 | 验收标准 |
|------|----------|
| `test_signal_type_mapping` | numerical/temporal/lab → 正确的 `SignalType` 枚举 |
| `test_physical_graph_builder` | 30 天时序 (5 物理量) → ≥ 25 条因果边 (含 rho/energy_relation) |
| `test_jepa_encodes_physical` | `JEPAEncoder.encode(signals=...).causal_edges` 非空且格式正确 |
| `test_jepa_predict_physical` | `JEPAPredictor.predict(physical_state)` 返回状态无崩溃 |
| `test_jepa_train_physical` | `JEPATrainer.train()` 对物理数据集正常收敛 (loss 递减) |
| `test_clinical_forward_prediction` | `jepa_predict(s_t)` 的 albumin 预测 vs 真实值 MAE < 5 |
| `test_clinical_counterfactual` | 500kcal 增量干预 → albumin 反事实值 > 基线 |
| `test_batch_counterfactual_clinical` | 100 患者 × 反事实查询 < 5s |
| `test_energy_conservation_physical` | 营养摄入 vs 生化指标的能量守恒违反度 < 0.1 |
| `test_end_to_end_clinical` | PerceptionPipeline → PhysicalGraphBuilder → JEPA → CF 全链路无异常 |

---

## 执行顺序与依赖

```
V3.0.6 (基线, 382 tests, ruff clean)
  │
  ├── V3.0.7 (参数化记忆觉醒)                             预计工期: 3-5天
  │     ├── 依赖: mlx.core ≥ 0.5.0 (已有), numpy (已有)
  │     ├── 修改: _parametric_memory.py (~500行→~600行 重构)
  │     ├── 新建: _causal_mlp.py (~300行)
  │     ├── 产出: CausalMLP (~15K params), MLX 原生训练循环,
  │     │         移除 torch/transformers/peft 三个硬依赖
  │     └── 门禁: 400+ tests, ruff clean, grep torch=0
  │
  ├── V3.0.8 (反事实推理增强)                             预计工期: 4-6天
  │     ├── 依赖: numpy, scipy (已有)
  │     │         可与 V3.0.7 并发开发 (独立模块)
  │     ├── 修改: _counterfactual.py (~150行 新增非线性)
  │     ├── 新建: _batch_counterfactual.py (~400行)
  │     ├── 新建: tests/test_v308_counterfactual_benchmark.py (~350行)
  │     ├── 产出: NonlinearSEM (tanh/ReLU/sigmoid),
  │     │         BatchCounterfactualEngine (O(N)→O(1)),
  │     │         CausalGraph↔SEM 双向转换, 5个基准
  │     └── 门禁: 420+ tests, ruff clean, 5/5 基准通过
  │
  └── V3.1.0 (物理世界应用)                               预计工期: 5-7天
        ├── 依赖: V3.0.7 (JEPA 训练器) + V3.0.8 (反事实引擎)
        ├── 修改: _perception.py (~150行), _jepa_encoder.py (~60行)
        ├── 新建: _physical_graph_builder.py (~300行)
        ├── 新建: tests/test_v310_clinical_benchmark.py (~350行)
        ├── 产出: MultimodalSignal (5种信号类型),
        │         PhysicalGraphBuilder (物理量→因果边),
        │         JEPAEncoder 物理信号路径,
        │         临床营养合成数据集 + 基准
        └── 门禁: 450+ tests, ruff clean, 端到端临床场景通过
```

## 版本号变更执行清单

以下 30 处版本引用（含注释/文档字符串/配置文件/测试）需要在实施方案批准后统一变更。

### 按文件分类

| # | 文件 | 位置 | 当前值 | 目标值 |
|---|------|------|--------|--------|
| 1 | `_world_model.py` | L14 (docstring) | `v3.7.0 L2` | `v3.0.7 L2` |
| 2 | `_world_model.py` | L16 (docstring) | `v3.8.0 L3` | `v3.0.8 L3` |
| 3 | `_world_model.py` | L120 (注释) | `v3.8.0 L3` | `v3.0.8 L3` |
| 4 | `_world_model.py` | L124 (注释) | `v4.0.0 JEPA` | `v3.1.0 JEPA` |
| 5 | `_world_model.py` | L169 (注释) | `v4.0.0 JEPA` | `v3.1.0 JEPA` |
| 6 | `_world_model.py` | L295 (注释) | `v4.0.0` | `v3.1.0` |
| 7 | `_world_model.py` | L363 (注释) | `v4.0.0` | `v3.1.0` |
| 8 | `_world_model.py` | L375 (注释) | `v4.0.0` | `v3.1.0` |
| 9 | `_world_model.py` | L600 (注释) | `v3.7.0 L2` | `v3.0.7 L2` |
| 10 | `_world_model.py` | L606 (注释) | `v3.8.0 L3` | `v3.0.8 L3` |
| 11 | `_world_model.py` | L630 (注释) | `v4.0.0` | `v3.1.0` |
| 12 | `_world_model.py` | L643 (注释) | `v4.0.0 JEPA` | `v3.1.0 JEPA` |
| 13 | `_world_model.py` | L647 (注释) | `v3.7.0` | `v3.0.7` |
| 14 | `_world_model.py` | L737 (注释) | `v4.0.0 JEPA` | `v3.1.0 JEPA` |
| 15 | `_world_model.py` | L745 (注释) | `v4.0.0 JEPA` | `v3.1.0 JEPA` |
| 16 | `_world_model.py` | L1979-2007 (roadmap) | `v3.6.0/v3.7.0/v3.8.0/v4.0.0` | `v3.0.7/v3.0.8/v3.1.0` |
| 17 | `_jepa_trainer.py` | L2 | `su-memory v4.0.0 M2` | `MCI World Model v3.1.0` |
| 18 | `_jepa_gnn.py` | L2 | `su-memory v4.0.0` | `MCI World Model v3.1.0` |
| 19 | `_jepa_gat_encoder.py` | L2 | `su-memory v4.0.0` | `MCI World Model v3.1.0` |
| 20 | `_jepa_encoder.py` | L2 | `su-memory v4.0.0` | `MCI World Model v3.1.0` |
| 21 | `_jepa_predictor.py` | L2 | `su-memory v4.0.0` | `MCI World Model v3.1.0` |
| 22 | `_jepa_encoder.py` | L334 | `v4.0.0` | `v3.1.0` |
| 23 | `_parametric_memory.py` | L2 | `su-memory v3.6.0` | `MCI World Model v3.0.7` |
| 24 | `_parametric_memory.py` | L641,L656 | `"version": "3.6.0"` | `"version": "3.0.7"` |
| 25 | `_counterfactual.py` | L2 | `su-memory v3.8.0` | `MCI World Model v3.0.8` |
| 26 | `_cost_module.py` | L2 | `su-memory v3.0.1` | `MCI World Model v3.0.7` |
| 27 | `pyproject.toml` | L7 | `3.0.6` | `3.0.7` |
| 28 | `README.md` | ~L219-222 | V4.0.0 规划 | V3.1.0 规划 |
| 29 | `_world_model.py` | L1979-2007 | `v3.0.0/v3.0.0-m2/v3.0.0-m3/v3.0.1/v3.0.2/v3.0.5/v3.0.4/v3.0.3` | 保持不变（这些是历史版本，已正确） |
| 30 | `_world_model.py` | L2013 | `v3.0.2` | 保持不变（历史版本） |

### 分批执行策略

- **第一批** (伴随 V3.0.7 代码变更): #1-#16, #23-#27 (核心版本引用 + pyproject.toml)
- **第二批** (伴随 V3.0.8 代码变更): #25 (counterfactual 注释)
- **第三批** (伴随 V3.1.0 代码变更): #17-#22 (JEPA 模块注释)

---

> **本文档已完成基于 v3.0.6 实际代码状态的深度审查和精细化增强。**
> 每个版本方案均包含：现状诊断表（精确到行号）、技术路线图、关键实现步骤（含代码骨架）、接口关系图（变更前后对比）、测试验证方案。
> 待人工确认后进入详细实施阶段。
