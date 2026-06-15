# Ch16 神经符号融合与可微分因果 — 改进规划书

## 1. 章节概述

本章节覆盖 P6-P8 波次中**神经符号融合**这一全新范式的规划，填补原 14 章在以下方向的空白：

- **可微分因果推理**: 将因果推理从纯符号推进到可微分，使梯度可回传 — 原系统因果推理不可微
- **神经符号融合 2.0**: 符号推理 (高置信) 与神经预测 (高速度) 的深度融合 — 原系统两条路径独立运行
- **因果梯度传播**: 沿因果图路径回传梯度 — 原系统无因果参数梯度更新机制
- **符号接地学习**: 将因果符号与感知数据关联 — 原系统符号推理与感知数据无连接

> **新增定位**: 这是 MCI 从"纯符号因果推理器"到"可微分因果增强层"的关键跃迁。Ch05 (形式化验证) 保证了符号推理的数学完备性，Ch16 则让符号推理获得可微分和端到端学习能力，为 P8 终局融合奠定基础。

## 2. 改进目标

| # | 目标 | 量化指标 | 波次 | 优先级 |
|---|---|---|---|---|
| G1 | DifferentiableCausalInference | 可微分 ATE 与符号 ATE 误差 <10% | P6 | 中 |
| G2 | NeuralSymbolicFusionV2 | 融合推理准确率 ≥85%，端到端学习 loss <0.01 | P8 | 中 |
| G3 | CausalGradientPropagation | 因果梯度 ≥5 步路径传播 | P8 | 中 |
| G4 | SymbolGroundingLearning | 10 因果概念检索准确率 ≥70% | P8 | 中 |

## 3. 实施方案

### 3.1 DifferentiableCausalInference 可微分因果推理 (G1)

**缺口**: 原因果推理 (DoCalculus) 完全符号化，无法参与端到端学习

```python
class DifferentiableCausalInference:
    """可微分因果推理 — 神经符号融合 2.0 前置"""
    def __init__(self, causal_graph, n_intervention_samples=1000):
        self._graph = causal_graph
        self._n_samples = n_intervention_samples
    
    def differentiable_ate(self, X: str, Y: str, x_value: float) -> dict:
        """
        可微分 ATE 估计:
          1. 用 do-calculus 图结构约束
          2. 用神经网络参数化条件分布 P(Y|do(X=x))
          3. 梯度可回传 → 端到端学习因果参数
        """
        # 干预采样
        intervened = self._sample_intervention(X, x_value)
        # 观测采样
        observed = self._sample_observational(X, x_value)
        # ATE = E[Y|do(X=x)] - E[Y|X=x]
        ate = np.mean(intervened) - np.mean(observed)
        # 梯度信息 (可微分)
        ate_gradient = self._compute_ate_gradient(X, Y, x_value)
        return {
            "ate": ate,
            "gradient": ate_gradient,
            "intervention_mean": np.mean(intervened),
            "observational_mean": np.mean(observed),
        }
    
    def _compute_ate_gradient(self, X, Y, x_value):
        """计算 ATE 对模型参数的梯度"""
        eps = 1e-4
        ate_plus = self._ate_at(x_value + eps)
        ate_minus = self._ate_at(x_value - eps)
        return (ate_plus - ate_minus) / (2 * eps)
```

**文件**: `_differentiable_causal.py` (~300 行)

**技术路线**: 符号约束 (do-calculus 图结构) + 神经参数化 (P(Y|do(X=x)) 用神经网络建模) → 可微分 ATE

**与 Ch15 可微分因果的关系**: Ch15 G1 (DifferentiableCausalInference) 是本章节 G2 (NeuralSymbolicFusionV2) 的前置。

### 3.2 NeuralSymbolicFusionV2 神经符号融合 2.0 (G2)

**缺口**: 原系统符号推理 (DoCalculus) 和神经预测 (TrueJEPA+ActionPredictor) 独立运行，无融合

```python
class NeuralSymbolicFusionV2:
    """神经符号融合 2.0 — 可微分因果推理与符号推理的深度融合"""
    def __init__(self, symbolic_engine, neural_predictor, fusion_config=None):
        self._symbolic = symbolic_engine   # DoCalculus + PearlChain
        self._neural = neural_predictor    # TrueJEPA + ActionPredictor
        self._config = fusion_config or FusionConfig()
        self._fusion_weights = {
            "symbolic": 0.6,   # 符号推理权重 (可验证性优先)
            "neural": 0.4,     # 神经预测权重 (速度优先)
        }
    
    def fused_reasoning(self, query: dict) -> dict:
        """
        深度融合推理:
          1. 符号推理: 因果图 + do-calculus (高置信)
          2. 神经预测: 潜空间预测 (高速度)
          3. 融合层: 加权合并 + 一致性校验
          4. 梯度回传: 从结果到因果参数
        """
        # 符号路径
        symbolic_result = self._symbolic.reason(query)
        
        # 神经路径
        neural_result = self._neural.predict(query)
        
        # 一致性校验
        consistency = self._check_consistency(symbolic_result, neural_result)
        
        # 自适应融合
        if consistency["consistent"]:
            fused = self._weighted_fuse(symbolic_result, neural_result)
        else:
            # 不一致时优先符号推理 (可验证性)
            fused = self._symbolic_priority_fuse(symbolic_result, neural_result, consistency)
        
        # 梯度信息 (端到端可微分)
        fused["gradient"] = self._compute_fusion_gradient(symbolic_result, neural_result)
        
        return fused
    
    def _check_consistency(self, symbolic, neural) -> dict:
        """一致性校验: 符号推理和神经预测是否一致"""
        sym_direction = np.sign(symbolic.get("ate", 0))
        neu_direction = np.sign(neural.get("prediction", 0))
        return {
            "consistent": sym_direction == neu_direction,
            "symbolic_ate": symbolic.get("ate", 0),
            "neural_prediction": neural.get("prediction", 0),
        }
    
    def _weighted_fuse(self, symbolic, neural) -> dict:
        """加权融合"""
        w_s = self._fusion_weights["symbolic"]
        w_n = self._fusion_weights["neural"]
        return {
            "fused_ate": w_s * symbolic.get("ate", 0) + w_n * neural.get("prediction", 0),
            "symbolic_confidence": symbolic.get("confidence", 0),
            "neural_confidence": neural.get("confidence", 0),
            "fusion_method": "weighted",
        }
    
    def end_to_end_learn(self, query, ground_truth, lr=0.001):
        """端到端学习: 从推理结果到因果参数的梯度回传"""
        result = self.fused_reasoning(query)
        loss = (result["fused_ate"] - ground_truth) ** 2
        # 回传梯度到因果图参数
        self._symbolic.update_from_gradient(result["gradient"], lr)
        return {"loss": loss, "updated": True}
```

**文件**: `_neural_symbolic_fusion_v2.py` (~500 行)

**融合策略**:

| 场景 | 策略 | 权重 |
|---|---|---|
| 符号+神经一致 | 加权融合 | 符号0.6 + 神经0.4 |
| 符号+神经不一致 | 符号优先 (可验证性) | 符号1.0 |
| 符号不可用 | 纯神经预测 | 神经1.0 |
| 神经不可用 | 纯符号推理 | 符号1.0 |

**理论依据**:
- 神经符号融合: Garcez et al. (2019) "Neural-Symbolic Cognitive Reasoning"
- 可微分因果: Parascandolo et al. (2018) "Learning Causal Models"
- 端到端学习: 引入可微分 ATE (G1) 作为桥梁

### 3.3 CausalGradientPropagation 因果梯度传播 (G3)

**缺口**: 原系统无因果参数梯度更新机制，因果图参数静态

```python
class CausalGradientPropagation:
    """因果梯度传播 — 从推理结果到因果图结构的梯度回传"""
    def __init__(self, causal_graph, neural_components):
        self._graph = causal_graph
        self._neural = neural_components
    
    def propagate(self, loss_gradient: np.ndarray, path: list[str]) -> dict:
        """
        沿因果路径传播梯度:
          1. 从损失函数出发
          2. 沿因果图边反向传播
          3. 更新因果参数 (边权重、条件概率)
        """
        gradients = {}
        for i in range(len(path) - 1):
            edge = (path[i], path[i+1])
            edge_gradient = self._compute_edge_gradient(edge, loss_gradient)
            gradients[edge] = edge_gradient
        
        # 更新因果图参数
        self._update_graph_parameters(gradients)
        
        return {
            "propagated_edges": len(gradients),
            "total_gradient_norm": sum(np.linalg.norm(g) for g in gradients.values()),
            "updated_parameters": True,
        }
    
    def _compute_edge_gradient(self, edge, loss_grad):
        """计算因果边的梯度"""
        return loss_grad * self._graph.get_edge_weight(edge)
```

**文件**: `_causal_gradient.py` (~300 行)

**与标准反向传播的区别**:
- 标准反向传播: 沿计算图 (computation graph) 传播
- 因果梯度传播: 沿因果图 (causal graph) 传播，受因果方向约束

### 3.4 SymbolGroundingLearning 符号接地学习 (G4)

**缺口**: 原系统因果符号 (如 "gravity"、"mass") 与感知数据无连接

```python
class SymbolGroundingLearning:
    """符号接地学习 — 将符号因果概念与感知数据关联"""
    def __init__(self, unified_encoder, causal_graph):
        self._encoder = unified_encoder
        self._graph = causal_graph
        self._grounding_map: dict[str, np.ndarray] = {}
    
    def ground_symbol(self, symbol: str, observations: list[dict]) -> dict:
        """将因果符号接地到感知数据"""
        vectors = [self._encoder.encode(obs) for obs in observations]
        centroid = np.mean(vectors, axis=0)
        self._grounding_map[symbol] = centroid
        return {
            "symbol": symbol,
            "grounding_vector": centroid,
            "n_observations": len(observations),
            "grounding_confidence": self._compute_confidence(vectors),
        }
    
    def retrieve_symbol(self, observation: dict) -> str:
        """从感知数据检索最匹配的因果符号"""
        z = self._encoder.encode(observation)
        best_symbol = None
        best_sim = -1
        for symbol, centroid in self._grounding_map.items():
            sim = np.dot(z, centroid) / (np.linalg.norm(z) * np.linalg.norm(centroid))
            if sim > best_sim:
                best_sim = sim
                best_symbol = symbol
        return best_symbol
```

**文件**: `_symbol_grounding.py` (~250 行)

**理论依据**: Harnad (1990) "The Symbol Grounding Problem"

**接地映射示例**:
```
"gravity" → [0.23, 0.87, 0.12, ...] (物体下落观测的潜向量中心)
"collision" → [0.45, 0.31, 0.89, ...] (碰撞观测的潜向量中心)
"spring" → [0.67, 0.12, 0.55, ...] (弹簧振动观测的潜向量中心)
```

## 4. 时间计划

| 周 | 任务 | 交付物 | 波次 |
|---|---|---|---|
| W65-68 | DifferentiableCausalInference | 可微分 ATE 实现 | P6 |
| W91-93 | NeuralSymbolicFusionV2 核心 | 融合推理 + 一致性校验 | P8 |
| W94-96 | NeuralSymbolicFusionV2 验证 | 融合基准 + 端到端学习 | P8 |
| W97-100 | CausalGradientPropagation | 因果梯度传播实现 | P8 |
| W101-102 | SymbolGroundingLearning | 符号接地实现 | P8 |

## 5. 资源配置

| 资源 | 角色 | 人天 | 说明 |
|---|---|---|---|
| 研究工程师 (P6) | 可微分因果推理 | 8 | P6 Stage3 |
| 研究工程师 (P8) | 神经符号融合 + 梯度传播 + 符号接地 | 20 | P8 核心 |
| 工程师 (P8) | 融合验证 + 基准 | 5 | P8 辅助 |
| **合计** | | **33** | |

## 6. KPI 指标

| KPI | 基线 | P6 目标 | P8 目标 |
|---|---|---|---|
| 可微分 ATE 误差 vs 符号 ATE | N/A | <10% | <5% |
| 融合推理准确率 | N/A | — | ≥85% |
| 端到端学习收敛 | N/A | — | loss <0.01 |
| 因果梯度传播步数 | N/A | — | ≥5 步 |
| 符号接地检索准确率 | N/A | — | ≥70% (10概念) |

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 | 应急方案 |
|---|---|---|---|---|
| 可微分 ATE 梯度不稳定 | 中 | 中 | 梯度裁剪 + 学习率 warmup | 退回符号推理 |
| 神经符号融合推理不稳定 | 中 | 高 | 渐进融合 (先符号→后神经→再融合) | 保留独立推理模式 |
| 端到端学习不收敛 | 高 | 中 | 降低学习率 + 增加正则 | 仅做前向推理不回传 |
| 符号接地准确率不达标 | 中 | 中 | 增加训练数据 + 增强编码器 | 退回手动符号映射 |
| 因果梯度传播路径爆炸 | 中 | 中 | 限制传播深度 ≤5 步 | 仅做单步梯度 |

## 8. 成本预算

| 项目 | 人天 | 硬件/软件 | 波次 |
|---|---|---|---|
| DifferentiableCausalInference | 8 | $500 (GPU) | P6 |
| NeuralSymbolicFusionV2 | 10 | $500 (GPU) | P8 |
| CausalGradientPropagation | 5 | $0 | P8 |
| SymbolGroundingLearning | 5 | $0 | P8 |
| 融合验证 + 基准 | 5 | $0 | P8 |
| **合计** | **33** | **$1,000** | |

## 9. 验收标准

- [ ] DifferentiableCausalInference: 可微分 ATE 与符号 ATE 误差 <10%
- [ ] NeuralSymbolicFusionV2: 融合推理准确率 ≥85%
- [ ] NeuralSymbolicFusionV2: 端到端学习收敛 loss <0.01
- [ ] CausalGradientPropagation: ≥5 步因果路径梯度传播
- [ ] SymbolGroundingLearning: 10 个因果概念检索准确率 ≥70%

## 依赖关系

- **前置**: Ch05 (形式化验证/DoCalculus), Ch02 (TrueJEPA/架构), Ch15 (可微分因果前置), Ch17 (UnifiedModalEncoder 符号接地前置)
- **被依赖**: Ch18 (AGIIntegrationProtocol 需要融合推理能力), WMMM (L5/L6 需要因果推理能力提升)
