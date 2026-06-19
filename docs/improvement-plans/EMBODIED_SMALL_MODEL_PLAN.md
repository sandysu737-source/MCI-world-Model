# 小模型训练 × 具身智能 — AI 智能清创机器人实施计划书 (V3)

> **代号**: "清创" (Debridement)  
> **应用场景**: AI 智能清创机器人 — 多模态感知 + 因果推理 + 安全力控  
> **基线版本**: MCI World Model v4.3.3 (3,568 tests, WMMM ~80%)  
> **规划周期**: 2026-06-20 → 2026-08-15 (共 8 周)  
> **人力**: 1 人全职  
> **硬件**: Apple M5 Pro / 18 核 / 48 GB / MLX 0.31.2  
> **参数预算**: 文本轨 15K-420K / 具身轨 10M → 50M → 200M → 500M  
> **终极目标**: 为 AI 智能清创机器人构建生产级多模态因果世界模型

---

## 1. 场景分析：清创机器人需要什么

### 1.1 清创工作流

```
术前评估              术中实时                 术后验证
────────              ────────                 ────────
RGB 拍照              实时 RGB 视频流           创面愈合评估
深度相机 (3D)         力传感器 (6轴)            组织分类对比
热成像 (炎症)         关节状态 (7-DOF)          手术记录生成
患者临床数据          工具-组织交互建模          因果可审计轨迹
    │                      │                       │
    └──────────┬───────────┴───────────┬───────────┘
               ▼                       ▼
        CEWM 世界模型          安全力控闭环
        · 组织分类              · 力限制 (≤5N)
        · 坏死/存活判定          · 深度限制
        · 清创策略规划           · 温度限制
        · 因果推理              · 紧急停止
```

### 1.2 清创机器人必需的 6 种模态

| # | 模态 | 传感器 | 数据类型 | 编码维度 | 已有关键模块 |
|---|------|--------|---------|---------|------------|
| M1 | **RGB 视觉** | 内窥镜/术野相机 | 224×224×3 图像 | 256D | `VisualEncoder`, `VisionEncoder` |
| M2 | **深度/3D** | 结构光/ToF | 224×224 深度图 | 128D | ❌ 需新建 |
| M3 | **热成像** | 红外热像仪 | 224×224 温度矩阵 | 64D | `ThermalEncoder` |
| M4 | **力/触觉** | 6轴力传感器 | (fx,fy,fz,tx,ty,tz) 6D | 32D | ❌ 需新建 |
| M5 | **本体感知** | 关节编码器 | 7-DOF (pos+vel+effort) 21D | 128D | `RobotWorldState` |
| M6 | **临床元数据** | EMR/手工录入 | 伤口类型/阶段/患者数据 | 64D | `MedicalCausalSDK` |

### 1.3 清创特有能力需求

| 能力 | 说明 | 当前状态 |
|------|------|---------|
| **组织分类** | 坏死/腐肉/肉芽/上皮 4分类 | ❌ 需新建 `TissueClassifier` |
| **清创深度预测** | 预测给定力下的组织去除深度 | ❌ 需学习型动力学模型 |
| **力-组织响应模型** | 不同组织类型的力-位移曲线 | ❌ 核心缺口 |
| **安全力控** | 组织特异性力限制 (坏死≤2N, 健康≤0.5N) | `ToolForceConstraint` 存在但未分组织 |
| **热安全** | 清创产热监控 (≤42°C) | `ThermalEncoder` 存在但无温度安全约束 |
| **手术相识别** | 自动识别清创阶段 (探查→清创→止血→验证) | ❌ 需新建 |
| **因果审计** | 每步操作可追溯因果链 | `MedicalCausalSDK` + `AuditableCausal` 已有基础 |

---

## 2. 多模态具身模型架构

### 2.1 DebridementWorldModel 架构

```
┌──────────────────────────────────────────────────────────────────┐
│                   DebridementWorldModel                           │
│                                                                  │
│  ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌────────┐ ┌───────────┐ │
│  │ RGB     │ │ Depth   │ │ Thermal  │ │ Force  │ │ Proprio   │ │
│  │ Encoder │ │ Encoder │ │ Encoder  │ │Encoder │ │ Encoder   │ │
│  │ (ViT)   │ │ (CNN)   │ │ (MLP)    │ │ (MLP)  │ │ (MLP)     │ │
│  │ 224→256 │ │ 224→128 │ │ 224→64   │ │ 6→32   │ │ 21→128    │ │
│  └────┬────┘ └────┬────┘ └────┬─────┘ └───┬────┘ └─────┬─────┘ │
│       │           │           │            │            │        │
│       └───────────┴───────────┴────────────┴────────────┘        │
│                            │                                      │
│                   ┌────────▼────────┐                            │
│                   │ Cross-Modal     │  ← 跨模态注意力融合         │
│                   │ Fusion          │                             │
│                   │ (CrossAttn)     │                             │
│                   └────────┬────────┘                            │
│                            │                                      │
│                   ┌────────▼────────┐                            │
│                   │ Temporal        │  ← 时序 Transformer         │
│                   │ Transformer     │     (因果掩码)              │
│                   │ N_layers        │                             │
│                   └───┬────────┬───┘                            │
│                       │        │                                  │
│              ┌────────▼──┐ ┌───▼──────────┐                     │
│              │ Dynamics  │ │ Tissue       │                     │
│              │ Head      │ │ Classifier   │                     │
│              │ (next     │ │ Head         │                     │
│              │  state)   │ │ (4-class)    │                     │
│              └───────────┘ └──────────────┘                     │
│                                                                  │
│  ┌──────────────────────────────────────────────────┐           │
│  │ Clinical Metadata Encoder (EMR → 64D)            │           │
│  │ · 伤口类型 · 患者年龄 · 合并症 · 感染标志物        │           │
│  └──────────────────────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 六级参数配置 (清创专用)

| 配置 | d_model | layers | heads | 视觉 | 参数量 | 内存 | 用途 |
|------|---------|--------|-------|------|--------|------|------|
| **Tiny** | 128 | 2 | 4 | ❌ | ~5M | ~3GB | Pendulum + Force 单模态验证 |
| **Small** | 256 | 4 | 8 | CNN | ~30M | ~8GB | RGB+Force+Prop 三模态融合 |
| **Base** | 512 | 6 | 12 | ViT-T | ~80M | ~15GB | 全模态 (6模态) + 组织分类 |
| **Large** | 768 | 10 | 16 | ViT-S | ~250M | ~30GB | 生产级清创模型 |
| **XL** | 1024 | 14 | 16 | ViT-B | ~500M | ~45GB | 高精度清创 (含时序长上下文) |
| **XXL** | 1536 | 18 | 24 | ViT-L | ~1B | ~48GB | 极限精度 (需 fp16 + G.C.) |

### 2.3 训练时间估算 (M5 Pro MLX, 10K 样本)

| 配置 | 参数量 | 10K 样本/10epoch | 100K 样本/10epoch |
|------|--------|-----------------|------------------|
| Tiny | 5M | ~30s | ~5min |
| Small | 30M | ~2min | ~20min |
| Base | 80M | ~5min | ~50min |
| Large | 250M | ~15min | ~2.5h |
| XL | 500M | ~35min | ~6h |
| XXL | 1B | ~1.5h (fp16) | ~15h (隔夜) |

**推荐训练配置**: 日常迭代用 **Base (80M)**，正式训练用 **Large (250M)**，隔夜精调用 **XL (500M)**。

---

## 3. 分阶段实施计划

### Phase A: 基础数据与单模态验证 (Week 1-2, 8天)

**目标**: 清创数据管线 + 各模态独立编码器验证。

#### A1: 清创数据管线设计 (2天)

**文件**: `sdk/_debridement_data.py` (~300行)

```python
@dataclass
class DebridementSample:
    """单帧清创数据样本。"""
    # 视觉
    rgb_image: np.ndarray       # (224, 224, 3) uint8
    depth_image: np.ndarray     # (224, 224) float32 (mm)
    thermal_image: np.ndarray   # (224, 224) float32 (°C)
    
    # 力触觉
    force_torque: np.ndarray    # (6,) float32 (fx,fy,fz,tx,ty,tz)
    
    # 本体感知
    joint_positions: np.ndarray # (7,) float32
    joint_velocities: np.ndarray
    joint_efforts: np.ndarray
    
    # 临床
    tissue_label: int           # 0=坏死 1=腐肉 2=肉芽 3=上皮
    wound_depth_mm: float
    surgical_phase: int         # 0=探查 1=清创 2=止血 3=验证
    
    # 动作
    tool_force_n: float         # 施加的力 (N)
    tool_velocity: float        # 工具速度 (mm/s)

class DebridementDataPipeline:
    """清创数据采集与增强管线。
    
    数据源:
    1. 公开伤口数据集 (WoundDB, Medetec, AZH)
    2. 物理模拟器 (有限元组织模型)
    3. 专家标注 (合成数据)
    
    增强:
    - 光照变化 (±30%)
    - 旋转/平移
    - 力传感器噪声 (σ=0.1N)
    - 组织标注边界抖动
    """
    def load_public_dataset(self, name: str) -> list[DebridementSample]: ...
    def generate_synthetic(self, n_samples: int) -> list[DebridementSample]: ...
    def augment(self, samples) -> list[DebridementSample]: ...
    def to_training_batch(self, samples) -> dict: ...
```

#### A2: 新增 Depth Encoder (1天)

**文件**: `sdk/_modality_encoders.py` 新增 `DepthEncoder` (~100行)

```python
class DepthEncoder(LearnableMixin):
    """深度图编码器 — 3D 创面重建。
    
    输入: (H, W) 深度图 (mm)
    统计特征: 均值、方差、梯度幅值、曲率
    可学习投影: 8D → 32D
    """
    def __init__(self, feature_dim=8, learnable_dim=32): ...
    def encode(self, depth_map: np.ndarray) -> np.ndarray: ...
```

#### A3: 新增 ForceEncoder (1天)

**文件**: `sdk/_modality_encoders.py` 新增 `ForceEncoder` (~80行)

```python
class ForceEncoder:
    """力触觉编码器 — 工具-组织交互。
    
    输入: (6,) 力/力矩
    编码: 统计特征 (均值/方差/峰值) + 频域特征
    输出: (32,) 
    """
    def __init__(self, feature_dim=32): ...
    def encode(self, ft_signal: np.ndarray) -> np.ndarray: ...
    def encode_history(self, ft_window: np.ndarray) -> np.ndarray: ...
```

#### A4: 新增 TissueClassifier (2天)

**文件**: `sdk/_tissue_classifier.py` (~400行)

```python
class TissueClassifier:
    """清创组织分类器 — 4 分类 + 置信度。
    
    输入: 融合后的多模态特征 (视觉+深度+热+临床)
    输出: P(坏死), P(腐肉), P(肉芽), P(上皮) + confidence
    
    架构: MLP (fused_dim → 256 → 128 → 4) + softmax
    训练: CrossEntropy + 类别加权 (坏死×2, 健康组织×0.5)
    
    安全约束:
    - 置信度 < 0.7: 标记为 "不确定"，需要人工确认
    - 坏死 vs 健康混淆: 触发安全停止
    """
    def __init__(self, input_dim=256, hidden_dims=(256, 128)): ...
    def classify(self, fused_features: np.ndarray) -> TissueResult: ...
    def is_safe_to_debride(self, result: TissueResult) -> bool: ...
```

#### A5: 单模态独立验证 (2天)

对每个编码器做独立验证：
- DepthEncoder: NYU Depth V2 子集上的深度估计精度
- ForceEncoder: 合成力曲线的编码-重建保真度
- TissueClassifier: 公开伤口数据集上的 4 分类准确率

#### Phase A KPI

| KPI | 基线 | 目标 |
|-----|------|------|
| DepthEncoder 深度误差 | N/A | ≤5mm |
| ForceEncoder 编码保真度 | N/A | 重建误差 ≤5% |
| TissueClassifier 4分类准确率 | N/A | ≥75% (平衡准确率) |
| 合成数据生成能力 | 0 | ≥500 帧/领域 |
| 新增编码器 3 个 | 0 | ✅ |

---

### Phase B: 多模态融合与组织动力学 (Week 3-4, 9天)

**目标**: 6 模态融合 + 组织动力学学习 + 清创策略规划。

#### B1: DebridementWorldModel (4天)

**文件**: `sdk/_debridement_world_model.py` (~700行)

```python
class DebridementWorldModel:
    """清创多模态世界模型。
    
    输入 6 模态 → 跨模态融合 → 时序推理 → 双头输出
    
    核心能力:
    - predict_next_state(): RGB→RGB, Depth→Depth, Force→Force 等
    - classify_tissue(): 4 分类 + 置信度
    - predict_force_response(): 给定动作 → 预测力响应
    - estimate_wound_depth(): 深度估计
    - detect_phase(): 手术相识别
    
    通过 DebridementConfig 切换 6 级参数规模。
    """
    
    def __init__(self, config: DebridementConfig):
        # 编码器组
        self.rgb_encoder = self._build_rgb_encoder(config)
        self.depth_encoder = DepthEncoder()
        self.thermal_encoder = ThermalEncoder()
        self.force_encoder = ForceEncoder()
        self.proprio_encoder = MLP(21, config.d_model)
        self.clinical_encoder = ClinicalMetadataEncoder()
        
        # 跨模态融合
        self.cross_modal_fusion = CrossModalFusion(
            n_modalities=6, d_model=config.d_model, n_heads=config.n_heads
        )
        
        # 时序 Transformer
        self.temporal_transformer = TransformerEncoder(
            config.n_layers, config.d_model, config.n_heads,
            mlp_dims=config.d_model * config.mlp_ratio
        )
        
        # 双头输出
        self.dynamics_head = MLP(config.d_model, state_dim)
        self.tissue_head = TissueClassifier(config.d_model)
    
    def forward(self, sample: DebridementSample) -> WorldPrediction:
        """单帧前向: 多模态编码 → 融合 → Transformer → 双头输出。"""
        ...
    
    def predict_rollout(self, state, actions, n_steps=20) -> list:
        """20步 rollout: 自回归预测，含不确定性估计。"""
        ...

class CrossModalFusion:
    """跨模态注意力融合。
    
    6 种模态 token → Cross-Attention → 统一表示。
    
    关键: 力 token 与视觉 token 的交叉注意力，
    让模型学习 "看到坏死组织 → 预期低阻力" 的因果关联。
    """
    def __init__(self, n_modalities, d_model, n_heads): ...
    def fuse(self, modal_tokens: dict[str, mx.array]) -> mx.array: ...
```

#### B2: Force-Tissue Dynamics (2天)

**文件**: `sdk/_force_tissue_dynamics.py` (~300行)

```python
class ForceTissueDynamics:
    """力-组织响应动力学模型。
    
    清创的核心物理约束:
    - 坏死组织: 低阻力, 小力即可去除 (0.5-2N)
    - 腐肉: 中等阻力 (1-3N)
    - 肉芽组织: 高阻力, 需保护 (力 >3N 会损伤)
    - 上皮/健康: 极高阻力, 严格禁止切割
    
    学习目标: 
    f(tissue_type, tool_force, tool_velocity) → depth_removed, force_feedback
    
    安全门禁:
    - 力超限: 坏死 >3N 或 肉芽 >1N → 紧急停止
    - 深度超限: 去除深度 > 坏死层厚度 → 停止
    - 温度超限: 热成像 >42°C → 冷却暂停
    """
    
    def __init__(self): ...
    def predict_removal(self, tissue_type, force, velocity) -> RemovalPrediction: ...
    def safety_check(self, prediction, current_state) -> SafetyVerdict: ...
```

#### B3: Debridement Safety Constraints (1.5天)

**文件**: `sdk/_safety.py` 新增 2 个清创专用约束 (~150行)

```python
class TissueForceConstraint(SafetyConstraint):
    """组织特异性力约束。
    
    | 组织类型 | 最大力 (N) | 最大速度 (mm/s) |
    |---------|-----------|----------------|
    | 坏死     | 3.0       | 10             |
    | 腐肉     | 2.0       | 5              |
    | 肉芽     | 1.0       | 3              |
    | 上皮     | 0.5       | 1              |
    """
    def __init__(self, tissue_classifier: TissueClassifier): ...
    def check(self, state, action) -> SafetyCheckResult: ...

class ThermalSafetyConstraint(SafetyConstraint):
    """热安全约束 — 清创产热 ≤42°C。"""
    def __init__(self, max_temp_c=42.0): ...
    def check(self, state, action) -> SafetyCheckResult: ...

class DepthLimitConstraint(SafetyConstraint):
    """清创深度约束 — 不超过坏死层厚度。"""
    def __init__(self, max_depth_mm=5.0): ...
    def check(self, state, action) -> SafetyCheckResult: ...
```

#### B4: 多模态训练器 (1.5天)

**文件**: `sdk/_debridement_trainer.py` (~400行)

```python
class DebridementTrainer:
    """清创多模态模型训练器 — MLX Native。
    
    多任务损失:
    L_total = L_dynamics (MSE) 
            + λ1 × L_tissue (CrossEntropy, 类别加权)
            + λ2 × L_force (Huber, 力预测)
            + λ3 × L_phase (CrossEntropy, 手术相)
            + λ4 × L_reconstruction (MSE, 深度图重建)
    
    训练策略:
    1. 单模态预训练 (各编码器独立)
    2. 冻结编码器 → 训练融合层
    3. 全模型联合微调
    """
    def pretrain_encoders(self, data) -> dict: ...
    def train_fusion(self, data, freeze_encoders=True) -> dict: ...
    def finetune_all(self, data) -> dict: ...
    def evaluate(self, test_data) -> DebridementMetrics: ...

@dataclass
class DebridementMetrics:
    """清创模型评估指标。"""
    tissue_accuracy: float       # 4分类准确率
    force_mse: float             # 力预测 MSE
    depth_mse: float             # 深度预测 MSE
    phase_accuracy: float        # 手术相识别准确率
    safety_violation_rate: float # 安全违规率 (越小越好)
    rollout_stability_20: float  # 20步 rollout 不发散率
```

#### Phase B KPI

| KPI | Tiny | Small | Base | Large |
|-----|------|-------|------|-------|
| Tissue 4分类准确率 | ≥60% | ≥70% | ≥80% | ≥85% |
| Force 预测 MSE | ≤0.5N² | ≤0.3N² | ≤0.15N² | ≤0.1N² |
| Depth 预测误差 | ≤10mm | ≤5mm | ≤3mm | ≤2mm |
| 20步 rollout 稳定 | ✅ | ✅ | ✅ | ✅ |
| 安全违规率 | <5% | <2% | <1% | <0.5% |

---

### Phase C: 文本因果推理轨 (Week 5, 4天)

**目标**: TinyTransformer 文本推理 + 临床知识图谱对接。

#### C1: CharTokenizer + TinyTransformer (2天)

**文件**: `sdk/_char_tokenizer.py` (150行) + `sdk/_tiny_transformer.py` (300行)

清创场景的文本因果推理示例：
```
输入: "坏死组织覆盖创面，伴有轻度感染"
输出: P(需清创)=0.92, P(需抗生素)=0.78, P(需植皮)=0.15
```

#### C2: 临床 QA 数据集 (1天)

**文件**: `data/debridement_qa_baseline.jsonl` (200对清创专用 QA)

```
{"cause": "坏死组织未清除", "effect": "感染风险增加", "relation": "enhance"}
{"cause": "过度清创", "effect": "健康组织损伤", "relation": "enhance"}  
{"cause": "清创后湿性敷料", "effect": "肉芽生长加速", "relation": "enhance"}
```

#### C3: TinyTransformer 与 CausalMLP 对比 (1天)

#### Phase C KPI

| KPI | 目标 |
|-----|------|
| TinyTransformer 5类别准确率 | ≥0.70 |
| CausalMLP (TF-IDF) 准确率 | ≥0.65 |
| 清创 QA 数据集 | ≥200 对 |

---

### Phase D: CEWM 清创闭环 (Week 6-7, 9天)

**目标**: 清创模型接入 CEWM 闭环 + 安全力控 + MCTS 规划。

#### D1: CEWM 清创闭环集成 (3天)

```python
def cewm_debridement_step(self, sample: DebridementSample) -> DebridementResult:
    """CEWM 清创单步闭环。
    
    1. 感知: DebridementWorldModel.encode(sample) → fused_state
    2. 分类: TissueClassifier.classify(fused) → tissue_type, confidence
    3. 安全: TissueForceConstraint.check(tissue_type, proposed_action)
    4. 预测: Dynamics.predict(state, action) → next_state
    5. 规划: MCTS.search(state, goal="最小化剩余坏死组织")
    6. 执行: 安全过滤后的 action → ROS2
    7. 学习: prediction_error → online_update()
    """
    ...
```

#### D2: Surgical Workflow State Machine (2天)

**文件**: `sdk/_surgical_workflow.py` (~250行)

```python
class SurgicalWorkflowSM:
    """清创手术工作流状态机。
    
    状态: EXPLORE → DEBRIDE → HEMOSTASIS → VERIFY → COMPLETE
    
    转换条件:
    - EXPLORE → DEBRIDE: 坏死组织面积 >10%
    - DEBRIDE → HEMOSTASIS: 出血检测 (RGB 红色区域 >5%)
    - DEBRIDE → VERIFY: 坏死组织面积 <2%
    - VERIFY → COMPLETE: 无残留坏死 + 止血完成
    - 任意状态 → EMERGENCY: 力超限 / 温度超限 / 深度超限
    """
    ...
```

#### D3: 安全力控 MCTS (2天)

```python
class SafeDebridementMCTS:
    """清创安全 MCTS 规划器。
    
    区别于通用 MCTSPlanner:
    - 组织特异性力约束作为硬约束（不可违反）
    - 安全代价 > 效率代价
    - 每步规划必须通过 TissueForceConstraint
    - 规划失败 → 切换到保守策略 (最小力, 低速度)
    """
    def search(self, wound_state, surgical_phase) -> DebridementPlan: ...
    def emergency_fallback(self) -> DebridementPlan: ...
```

#### D4: 模拟清创闭环验证 (2天)

用物理模拟器 + 合成伤口数据验证：
1. 坏死组织识别 → 清创动作 → 力反馈 → 组织分类更新 → 清创完成判定
2. 安全约束触发测试：力超限 / 深度超限 / 温度超限
3. 100 episode 在线学习：清创策略收敛

#### Phase D KPI

| KPI | 目标 |
|-----|------|
| CEWM 清创闭环 | ✅ |
| 手术工作流状态机 | ✅ |
| 安全约束触达率 | 100% (应触必触) |
| 误触发率 | <1% |
| 100 episode 清创成功率 | ≥85% (模拟) |

---

### Phase E: 生产级评估 (Week 8, 5天)

**目标**: Base(80M) 完整训练 + 评估 + 后续路线图。

#### E1: Base(80M) 全量训练 (2天)

- 数据: 5K 合成清创帧 (6模态) + 数据增强 → 50K 训练集
- 训练: 3 阶段 (编码器预训练 → 融合层 → 联合微调)
- 验证: 1K 测试集

#### E2: 完整指标评估 (2天)

| 维度 | 指标 | 目标 |
|------|------|------|
| 组织分类 | 4分类 F1 | ≥0.80 |
| 力预测 | MSE | ≤0.15 N² |
| 深度预测 | MAE | ≤3mm |
| 手术相识别 | 4分类准确率 | ≥85% |
| 安全违规率 | 模拟清创 100ep | <1% |
| Rollout 稳定性 | 20步 | 100% |
| 推理延迟 | 单帧 (M5 Pro) | <50ms |
| 训练时间 | 50K 样本 (M5 Pro) | <1h |

#### E3: 后续扩展路线图 (1天)

#### Phase E KPI

| KPI | 目标 |
|-----|------|
| Base(80M) 全量训练完成 | ✅ |
| 6 模态全融合 | ✅ |
| 全量测试 ≥3,850 | ✅ |
| 完整评估报告 | ✅ |

---

## 4. 新增/修改文件清单

| 文件 | 操作 | 行数 | Phase |
|------|------|------|-------|
| `sdk/_debridement_data.py` | 🆕 | ~300 | A |
| `sdk/_modality_encoders.py` | ✏️ +Depth +Force | +180 | A |
| `sdk/_tissue_classifier.py` | 🆕 | ~400 | A |
| `sdk/_debridement_world_model.py` | 🆕 | ~700 | B |
| `sdk/_force_tissue_dynamics.py` | 🆕 | ~300 | B |
| `sdk/_debridement_trainer.py` | 🆕 | ~400 | B |
| `sdk/_safety.py` | ✏️ +3 清创约束 | +150 | B |
| `sdk/_surgical_workflow.py` | 🆕 | ~250 | D |
| `sdk/_char_tokenizer.py` | 🆕 | ~150 | C |
| `sdk/_tiny_transformer.py` | 🆕 | ~300 | C |
| `sdk/_tiny_trainer.py` | 🆕 | ~200 | C |
| `sdk/_causal_mlp.py` | ✏️ TF-IDF | +80 | C |
| `sdk/_mcts_planner.py` | ✏️ SafeDebridementMCTS | +120 | D |
| `sdk/_world_model.py` | ✏️ cewm_debridement_step | +150 | D |
| `sdk/__init__.py` | ✏️ 导出 | +50 | B |
| `pyproject.toml` | ✏️ mlx_model | +3 | C |
| `data/debridement_qa_baseline.jsonl` | 🆕 | 200对 | C |
| 测试文件 × 8 | 🆕 | ~1,500 | 各Phase |
| **合计** | | **~5,200** | |

---

## 5. 工时总览

| Phase | 周 | 工作日 | 新建 | 修改 | 测试 | 核心交付 |
|-------|-----|--------|------|------|------|---------|
| A | W1-2 | 8 | +880 | +180 | +400 | 6模态编码器 + TissueClassifier |
| B | W3-4 | 9 | +1400 | +150 | +500 | DebridementWorldModel 多模态融合 |
| C | W5 | 4 | +650 | +83 | +250 | TinyTransformer 文本推理 |
| D | W6-7 | 9 | +250 | +270 | +250 | CEWM 清创闭环 + 安全力控 |
| E | W8 | 5 | — | — | — | 评估报告 + 路线图 |
| **总计** | | **35** | **~3,180** | **~683** | **~1,400** | |

---

## 6. 清创安全性设计 (贯穿全 Phase)

```
┌──────────────────────────────────────────┐
│          清创安全五层防护                   │
├──────────────────────────────────────────┤
│ L1 传感器层: 力/温度/深度 硬件冗余         │
│ L2 约束层: TissueForce + Thermal + Depth  │
│ L3 模型层: 分类置信度 <0.7 → 人工确认      │
│ L4 规划层: MCTS 安全代价 > 效率代价         │
│ L5 系统层: EmergencyStop + 手术相守护      │
└──────────────────────────────────────────┘
```

**每层独立触发、独立回退，任一层触发 → 安全停止。**

---

## 7. 风险登记册

| ID | 风险 | 概率 | 影响 | 缓解 |
|----|------|------|------|------|
| R1 | 公开伤口数据集质量不足 | 高 | 高 | 物理模拟器 + 专家合成数据补充 |
| R2 | 力-组织响应建模误差导致安全违规 | 中 | 极高 | L1-L5 五层防护, 保守力限制 |
| R3 | 多模态训练不收敛 (6 模态梯度冲突) | 中 | 中 | 渐进式训练策略 (单模态→融合→联合) |
| R4 | 500M 模型 M5 Pro 训练 OOM | 低 | 中 | Gradient Checkpointing + fp16 + batch=1 |
| R5 | 临床验证数据不可及 | 高 | 中 | Phase A-E 用合成数据, 临床数据延后 |

---

## 8. 验收检查点

### DC-1 (W2 结束): Phase A
- [ ] 6 种模态编码器全部可用
- [ ] TissueClassifier 4分类 ≥75%
- [ ] 合成数据 ≥500 帧/领域

### DC-2 (W4 结束): Phase B
- [ ] DebridementWorldModel Base(80M) 可训练
- [ ] Force-Tissue Dynamics 预测 MSE ≤0.15
- [ ] 组织特异性安全约束 3 个新增

### DC-3 (W5 结束): Phase C
- [ ] TinyTransformer 可用
- [ ] 清创 QA ≥200 对

### DC-4 (W7 结束): Phase D
- [ ] CEWM 清创闭环可用
- [ ] 手术工作流状态机
- [ ] 安全约束 100% 触达, <1% 误触发

### DC-5 (W8 结束): Phase E
- [ ] Base(80M) 全量训练完成
- [ ] 模拟清创成功率 ≥85%
- [ ] 全量测试 ≥3,850
- [ ] 完整评估报告

---

*文档版本: V3*  
*生成时间: 2026-06-19*  
*应用场景: AI 智能清创机器人*  
*6 模态: RGB + Depth + Thermal + Force + Proprioception + Clinical Metadata*  
*参数预算: 5M-1B (6级可伸缩), 推荐训练配置 Base(80M)/Large(250M)*
