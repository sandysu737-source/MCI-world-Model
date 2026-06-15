# Ch17 多模态统一与跨模态因果推理 — 改进规划书

## 1. 章节概述

本章节覆盖 P6 波次中**多模态统一表征与跨模态因果推理**的规划，填补原 Ch03 (四维度量化审计) 在以下方向的空白：

- **统一多模态编码器**: 视觉+语言+动作 3 模态共享 256D 潜空间 — 原系统各模态独立编码，无统一表征
- **跨模态因果推理**: 在统一潜空间中进行跨模态因果推断 — 原系统因果推理仅支持单一文本模态
- **跨模态对比学习**: InfoNCE 损失拉近匹配模态、推远非匹配模态 — 原系统无跨模态对齐机制
- **模态接地验证**: 验证统一表征的语义一致性 — 原系统无模态间语义验证

> **新增定位**: Ch03 覆盖"四维度量化审计" (推理/预测/安全/泛化)，Ch17 覆盖"多模态统一与跨模态因果"——是从单模态因果推理到多模态统一因果推理的关键跃迁。

## 2. 改进目标

| # | 目标 | 量化指标 | 波次 | 优先级 |
|---|---|---|---|---|
| G1 | UnifiedModalEncoder | 3 模态共享 256D 潜空间，跨模态检索 top-5 ≥65% | P6 | 中 |
| G2 | CrossModalCausalReasoner | 20 个跨模态因果查询准确率 ≥70% | P6 | 中 |
| G3 | 跨模态对比学习训练 | 3 模态对齐收敛，top-5 检索准确率 ≥65% | P6 | 中 |
| G4 | 模态接地验证 | 统一表征语义一致性验证通过 | P6 | 中 |

## 3. 实施方案

### 3.1 UnifiedModalEncoder 统一多模态编码器 (G1)

**缺口**: 原系统视觉/语言/动作三模态各自独立编码，无共享潜空间

```python
class UnifiedModalEncoder:
    """统一多模态表征 — 视觉+语言+动作共享潜空间"""
    def __init__(self, shared_dim=256, n_modalities=3):
        self._shared_dim = shared_dim
        # 各模态编码器 (P0-P2已有)
        self._visual = LearnableVisualEncoder(output_dim=shared_dim)
        self._textual = TextEncoder(output_dim=shared_dim)  # SBERT
        self._action = ActionEncoder(output_dim=shared_dim)
        # 跨模态对齐投影头
        self._alignment_heads = {
            "visual_text": AlignmentHead(shared_dim),
            "text_action": AlignmentHead(shared_dim),
            "visual_action": AlignmentHead(shared_dim),
        }
    
    def encode(self, observation: dict) -> np.ndarray:
        """统一编码: 多模态观测 → 共享潜向量"""
        vectors = []
        if "image" in observation:
            vectors.append(self._visual.encode(observation["image"]))
        if "text" in observation:
            vectors.append(self._textual.encode(observation["text"]))
        if "action" in observation:
            vectors.append(self._action.encode(observation["action"]))
        # 加权融合
        weights = np.array([1.0 / len(vectors)] * len(vectors))
        return np.average(vectors, axis=0, weights=weights)
    
    def cross_modal_retrieve(self, query_modality, query_data, target_modality, k=5):
        """跨模态检索: 用一种模态查询另一种模态"""
        query_vec = self._encode_single(query_modality, query_data)
        aligned_vec = self._alignment_heads[f"{query_modality}_{target_modality}"].project(query_vec)
        return self._retrieve_from_index(target_modality, aligned_vec, k)
    
    def decode(self, latent_vector: np.ndarray, target_modality: str = "text") -> dict:
        """从潜向量解码到目标模态 (因果想象引擎依赖)"""
        decoder = self._get_decoder(target_modality)
        return decoder.decode(latent_vector)
```

**文件**: `_unified_modal_encoder.py` (~400 行)

**架构设计**:

```
Visual ──┐
          ├── UnifiedModalEncoder ── 256D 共享潜空间 ──┬── CrossModalReasoner
Text ─────┤                                              ├── CausalImagination
          │                                              └── SymbolGrounding
Action ───┘
```

**与 Ch03 能力四维度的关系**:
- Ch03 G2 (视觉能力): 仅有独立 VisualEncoder，无跨模态
- Ch17: 在 Ch03 基础上统一 3 模态到共享潜空间

### 3.2 CrossModalCausalReasoner 跨模态因果推理器 (G2)

**缺口**: 原因果推理 (DoCalculus) 仅处理单一文本模态的因果查询

```python
class CrossModalCausalReasoner:
    """跨模态因果推理 — 在统一潜空间中做因果推断"""
    def __init__(self, unified_encoder, do_calculus):
        self._encoder = unified_encoder
        self._do = do_calculus
    
    def reason_cross_modal(self, observation: dict, query: str) -> dict:
        """
        跨模态因果推理流程:
          1. 多模态观测 → 统一潜向量
          2. 潜空间中定位因果图节点
          3. DoCalculus 因果推断
          4. 结果投射回目标模态
        """
        z = self._encoder.encode(observation)
        causal_query = self._parse_cross_modal_query(query, observation)
        result = self._do.estimate_ate(
            causal_query["cause"], causal_query["effect"],
            x_value=causal_query.get("x_value")
        )
        return {
            "cause_modality": causal_query["cause_modality"],
            "effect_modality": causal_query["effect_modality"],
            "ate": result.ate,
            "confidence": result.confidence,
            "latent_representation": z,
        }
    
    def _parse_cross_modal_query(self, query: str, observation: dict) -> dict:
        """解析跨模态因果查询"""
        # 识别查询中涉及的模态
        modalities = set()
        if "see" in query or "look" in query or "image" in observation:
            modalities.add("visual")
        if "cause" in query or "effect" in query:
            modalities.add("text")
        if "do" in query or "action" in observation:
            modalities.add("action")
        
        return {
            "cause_modality": list(modalities)[0] if len(modalities) > 0 else "text",
            "effect_modality": list(modalities)[-1] if len(modalities) > 1 else "text",
            "cause": query.split("cause")[0].strip() if "cause" in query else query,
            "effect": query.split("effect")[-1].strip() if "effect" in query else "",
        }
```

**文件**: `_cross_modal_causal.py` (~350 行)

**跨模态因果查询示例**:
```
Q1: "看到红色警报时，执行紧急制动会产生什么效果？" (视觉→动作)
Q2: "文字描述'温度升高'导致物体膨胀的因果关系" (文本→视觉)
Q3: "旋转动作对物体位置的因果影响" (动作→视觉)
```

### 3.3 跨模态对比学习训练 (G3)

```python
class CrossModalContrastiveLoss:
    """跨模态对比学习损失 — 统一潜空间"""
    def __init__(self, temperature=0.07):
        self._temperature = temperature
    
    def compute(self, anchor, positive, negatives):
        """InfoNCE: 拉近匹配模态，推远非匹配模态"""
        pos_sim = np.dot(anchor, positive) / self._temperature
        neg_sims = [np.dot(anchor, neg) / self._temperature for neg in negatives]
        logits = np.array([pos_sim] + neg_sims)
        softmax = np.exp(logits) / np.sum(np.exp(logits))
        return -np.log(softmax[0])


class CrossModalTrainer:
    """跨模态对齐训练器"""
    def __init__(self, encoder, loss_fn, lr=1e-4):
        self._encoder = encoder
        self._loss_fn = loss_fn
        self._lr = lr
    
    def train_step(self, batch: list[dict]) -> float:
        """一个训练步"""
        total_loss = 0.0
        for sample in batch:
            # 编码配对模态
            anchor = self._encoder.encode_single(sample["anchor_modality"], sample["anchor"])
            positive = self._encoder.encode_single(sample["positive_modality"], sample["positive"])
            negatives = [
                self._encoder.encode_single(sample["anchor_modality"], neg)
                for neg in sample.get("negatives", [])
            ]
            loss = self._loss_fn.compute(anchor, positive, negatives)
            total_loss += loss
        return total_loss / len(batch)
```

**训练数据**:
- COCO (图像-文本配对): 视觉-语言对齐
- ConceptNet (概念-动作配对): 语言-动作对齐
- 自建 (图像-动作配对): 视觉-动作对齐

### 3.4 模态接地验证 (G4)

```python
class ModalGroundingValidator:
    """模态接地验证 — 验证统一表征的语义一致性"""
    def __init__(self, unified_encoder):
        self._encoder = unified_encoder
    
    def validate_consistency(self, multimodal_data: list[dict]) -> dict:
        """验证同一概念在不同模态中的表征一致性"""
        results = []
        for sample in multimodal_data:
            # 同一概念的不同模态编码
            vecs = {}
            for modality in ["visual", "text", "action"]:
                if modality in sample:
                    vecs[modality] = self._encoder.encode_single(modality, sample[modality])
            
            # 计算模态间余弦相似度
            similarities = {}
            modality_list = list(vecs.keys())
            for i in range(len(modality_list)):
                for j in range(i+1, len(modality_list)):
                    m1, m2 = modality_list[i], modality_list[j]
                    sim = np.dot(vecs[m1], vecs[m2]) / (
                        np.linalg.norm(vecs[m1]) * np.linalg.norm(vecs[m2])
                    )
                    similarities[f"{m1}_{m2}"] = sim
            
            results.append({
                "concept": sample.get("concept", "unknown"),
                "similarities": similarities,
                "consistent": all(s > 0.5 for s in similarities.values()),
            })
        
        return {
            "total_samples": len(results),
            "consistent_samples": sum(1 for r in results if r["consistent"]),
            "consistency_rate": sum(1 for r in results if r["consistent"]) / len(results),
        }
```

## 4. 时间计划

| 周 | 任务 | 交付物 | 波次 |
|---|---|---|---|
| W53-55 | UnifiedModalEncoder 核心架构 | 3模态编码器 + 共享潜空间 | P6 |
| W56-58 | 跨模态对比学习训练 | InfoNCE + 对齐收敛 | P6 |
| W59-62 | CrossModalCausalReasoner | 跨模态因果推理 + 20题基准 | P6 |
| W63-64 | 模态接地验证 | 接地基准 + 一致性验证 | P6 |

## 5. 资源配置

| 资源 | 角色 | 人天 | 说明 |
|---|---|---|---|
| 工程师 B | 多模态统一 + 跨模态推理 + 接地验证 | 18 | P6 核心 |
| 领域专家 (多模态验证) | 0.3 人 × 8 周 | 5 人天 | 多模态+因果审核 |
| **合计** | | **23** | |

## 6. KPI 指标

| KPI | 基线 | P6 目标 | 度量 |
|---|---|---|---|
| 统一潜空间维度 | 独立编码 | 256D 共享 | UnifiedModalEncoder |
| 跨模态检索 top-5 | N/A | ≥65% | 对比学习基准 |
| 跨模态因果推理 | N/A | ≥70% 准确率 | 20 题基准 |
| 模态接地一致性 | N/A | 通过 | 接地基准 |

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 | 应急方案 |
|---|---|---|---|---|
| 多模态统一训练不收敛 | 中 | 高 | 渐进式对齐 (先2模态→3模态) | 保留独立编码器 |
| 跨模态因果推理准确率低 | 中 | 中 | 增加训练数据 + 优化查询解析 | 降级为单模态推理 |
| 对比学习负样本质量差 | 低 | 中 | hard negative mining | 随机负样本 |
| GPU 成本超预算 | 中 | 中 | 减少训练 epoch + 混合精度 | 缩小模型规模 |

## 8. 成本预算

| 项目 | 人天 | 硬件/软件 | 波次 |
|---|---|---|---|
| UnifiedModalEncoder | 10 | $500 (GPU) | P6 |
| CrossModalCausalReasoner | 8 | $0 | P6 |
| 模态接地验证 | 5 | $500 (API+专家) | P6 |
| **合计** | **23** | **$1,000** | |

## 9. 验收标准

- [ ] UnifiedModalEncoder: 3 模态共享 256D 潜空间
- [ ] 跨模态检索 top-5 准确率 ≥65%
- [ ] 跨模态因果推理 20 题准确率 ≥70%
- [ ] 模态接地验证基准通过 (一致性 ≥50%)

## 依赖关系

- **前置**: Ch02 (架构三支柱: VisualEncoder/TextEncoder/ActionEncoder), Ch03 (能力四维度), Ch10 (知识蒸馏: SBERT 文本编码器)
- **被依赖**: Ch15 (CausalImaginationEngine 需要 UnifiedModalEncoder 的 encode/decode), Ch16 (SymbolGroundingLearning 需要 UnifiedModalEncoder)
