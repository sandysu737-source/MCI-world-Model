# Ch03 能力层面：四维度量化审计 — 改进规划书

## 1. 章节概述

原报告第三章量化审计了四大能力维度：
- **物理世界理解**: 6.5/10 — 物理公式正确，但学习型预测器仅 6 参数 (F10)
- **多模态视觉感知**: 2.0/10 — 32D/16D/8D 统计特征，无可学习参数 (F5)
- **开放域知识推理**: 3.5/10 — 无自回归预训练
- **大规模知识处理**: 4.0/10 — PEM 存储好，但检索语义弱 (F1/F2)

**综合评分**: 4.0/10 → 目标 7.5/10

## 2. 改进目标

| # | 目标 | 量化指标 | 优先级 |
|---|---|---|---|
| G1 | 物理理解从玩具级到工程级 | 学习型预测器支持 ≥3 种物理系统 | P1 |
| G2 | 视觉感知从统计级到语义级 | 编码维度 32D → 512D，可学习 | P1 |
| G3 | 知识检索从 hash 到语义 | PEM cosine 准确率 0.50 → 0.85+ | P0 |
| G4 | 建立多系统物理动力学库 | 支持 Pendulum/Cart/弹簧/流体 | P2 |

## 3. 实施方案

### 3.1 物理动力学扩展 (G1, G4)

**现状**: 仅 PendulumPhysicsPredictor + CartPhysicsPredictor

**新增**:
```
src/mci_world_model/sdk/_physics/
├── __init__.py
├── _pendulum.py           # 已有
├── _cart.py               # 已有
├── _spring.py             # 新增: 弹簧-质量系统
├── _double_pendulum.py    # 新增: 双摆 (混沌系统)
├── _projectile.py         # 新增: 抛体运动 (空气阻力)
└── _fluid.py              # 新增: 简化流体 (伯努利方程)
```

每个物理系统包含:
- `WorldState` 子类 (状态定义)
- `Action` 子类 (动作定义)
- `PhysicsPredictor` (Euler/RK4 积分)
- 默认候选动作生成器

### 3.2 可学习视觉编码器 (G2)

**现状**: `VisionEncoder` = 32D Sobel+直方图统计 (无参数)

**方案**: CLIP-style 蒸馏 → 轻量 ViT

```python
class LearnableVisualEncoder:
    """可学习视觉编码器 — 轻量 ViT"""
    def __init__(self, image_size=64, patch_size=8, embed_dim=256,
                 num_heads=4, num_layers=4, output_dim=512):
        # Patch embedding
        self.patch_embed = PatchEmbed(image_size, patch_size, embed_dim)
        # Transformer layers
        self.blocks = [TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        # Projection head
        self.head = nn.Linear(embed_dim, output_dim)
    
    def encode(self, image: np.ndarray) -> np.ndarray:
        """image (H,W,C) → (output_dim,)"""
        patches = self.patch_embed(image)
        for block in self.blocks:
            patches = block(patches)
        return self.head(patches.mean(axis=0))  # global avg pool
```

**参数量**: ~200K (轻量级，CPU 可运行)
**训练**: 用 CLIP 做 teacher，蒸馏到 student

### 3.3 语义检索升级 (G3)

**现状**: `_tags_to_vector()` 使用 char-hash (DeepSeek F2)

**方案**: 分两步升级

**Phase A (P0, 1周)**: 临时修复 — BM25 关键词匹配
```python
def _tags_to_vector_bm25(self, tags: str) -> np.ndarray:
    """临时方案: 词频+TF-IDF 代替 char-hash"""
    from collections import Counter
    words = tags.lower().split()
    tf = Counter(words)
    # 使用预计算的 IDF
    vec = np.zeros(self._dim, dtype=np.float64)
    for word, count in tf.items():
        idx = hash(word) % self._dim
        vec[idx] += count * self._idf.get(word, 1.0)
    return vec
```

**Phase B (P1, 3周)**: 永久修复 — Sentence-BERT 嵌入
```python
def _tags_to_vector_sbert(self, tags: str) -> np.ndarray:
    """永久方案: 预训练句子嵌入模型"""
    # 使用本地小模型 (all-MiniLM-L6-v2, 22M参数, CPU 可运行)
    embedding = self._sbert_model.encode(tags)
    return embedding  # (384,)
```

## 4. 时间计划

| 周 | 任务 | 交付物 | 里程碑 |
|---|---|---|---|
| W4 | PEM BM25 临时修复 | 修改 `_persistent_memory.py` | M1: cosine 准确率 ≥0.70 |
| W5-6 | 4 个新物理系统 | `_physics/*.py` | M2: 4 种物理系统 |
| W7 | PEM SBERT 嵌入集成 | 集成 all-MiniLM-L6-v2 | M3: cosine 准确率 ≥0.85 |
| W8-10 | LearnableVisualEncoder | `_learnable_visual.py` | M4: 512D 可学习 |
| W11-12 | 视觉编码器 CLIP 蒸馏 | 训练脚本 + checkpoint | M5: 蒸馏完成 |
| W13-18 | 物理系统基准测试 | `benchmarks/physics/` | M6: ≥3 种系统基准达标 |
| W19-22 | 四维度集成回归测试 | 全量测试 | M7: 四维度评分 ≥7.0 |

## 5. 资源配置

| 资源 | 数量 | 成本 |
|---|---|---|
| 后端工程师 | 1人 × 18周 | 72 人天 |
| 模型训练 (CLIP 蒸馏) | 按需 GPU | $800 |
| SBERT 模型 | all-MiniLM-L6-v2 | $0 (开源) |

## 6. KPI 指标

| KPI | 基线 | 目标 | 度量 |
|---|---|---|---|
| 物理系统数 | 2 (Pendulum/Cart) | ≥6 | `PhysicsPredictor` 子类数 |
| 视觉编码维度 | 32D (统计) | 512D (可学习) | `encoder.output_dim` |
| 视觉编码参数 | 0 | ≥100K | `encoder.n_params` |
| PEM 检索准确率 | cosine=0.50 | ≥0.85 | 语义相似度测试集 |
| 物理理解评分 | 6.5/10 | ≥8.0/10 | 基准测试套件 |
| 四维度综合评分 | 4.0/10 | ≥7.5/10 | 综合评估脚本 |

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|
| CLIP 蒸馏 CPU 训练太慢 | 中 | 中 | 降采样到 64×64 + 减少层数 |
| SBERT 增加推理延迟 | 中 | 低 | 异步编码 + 缓存 |
| 新物理系统缺乏 ground truth | 低 | 中 | 用 SymPy 生成解析解 |
| BM25 临时方案被遗忘 | 高 | 中 | 在代码中标记 `# TODO: replace with SBERT` |

## 8. 成本预算

| 项目 | 人天 | 硬件/软件 |
|---|---|---|
| PEM 语义修复 (BM25+SBERT) | 8 | $0 |
| 4 个新物理系统 | 16 | $0 |
| LearnableVisualEncoder 开发 | 15 | $0 |
| CLIP 蒸馏训练 | 10 | $800 |
| 基准测试 | 15 | $0 |
| 集成回归测试 | 14 | $0 |
| **小计** | **78** | **$800** |

## 9. 验收标准

- [ ] PEM 语义检索: "心率升高" vs "心率增加" cosine ≥0.85
- [ ] 至少 6 种物理系统 (Pendulum/Cart/Spring/DoublePendulum/Projectile/Fluid)
- [ ] LearnableVisualEncoder: 512D 可学习，CPU 推理 <50ms/张
- [ ] 四维度综合评分 ≥7.5/10 (Ch01 评估套件)
- [ ] 全量测试 ≥2800 passed, 0 failed

## 依赖关系

- **前置**: Ch02 (TrueJEPA, 通用 ActionPredictor), Ch04 (F1/F2 修复)
- **被依赖**: Ch10 (知识蒸馏需要视觉编码器), Ch08 (WMMM 评分)
