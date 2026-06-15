# Ch10 知识获取策略 (LLM 蒸馏) — 改进规划书

## 1. 章节概述

原报告第十章整合了五视角的蒸馏建议：不蒸馏"知识内容"，而是蒸馏"知识结构"。V5.0.0 的核心优势是因果可解释性，蒸馏目标是将 LLM 的隐式知识转化为 MCI 的显式因果图/安全规则/动力学模型。

## 2. 改进目标

| # | 目标 | 量化指标 | 优先级 |
|---|---|---|---|
| G1 | 因果结构蒸馏管线 | LLM CoT → MCI CausalGraph edges | P1 |
| G2 | 视觉蒸馏管线 | CLIP → MCI VisualEncoder | P1 |
| G3 | 安全规则蒸馏 | Constitutional AI → SafetyConstraint 规则 | P2 |
| G4 | 动力学蒸馏管线 | MuZero/Dreamer → ActionPredictor | P2 |
| G5 | 持续学习蒸馏 | 在线微调 + 遗忘控制 | P3 |

## 3. 实施方案

### 3.1 因果结构蒸馏 (G1)

**管线**: LLM CoT → 因果边提取 → 因果图构建

```python
class CausalDistillationPipeline:
    """LLM CoT → MCI 因果图蒸馏管线"""
    def __init__(self, llm_client, causal_graph: dict):
        self._llm = llm_client
        self._graph = causal_graph
    
    def distill_from_cot(self, domain_text: str) -> list[dict]:
        """
        1. LLM 生成领域 CoT (Chain-of-Thought)
        2. 从 CoT 中提取因果关系
        3. 验证 + 添加到因果图
        """
        cot = self._llm.generate(
            f"请分析以下领域的因果关系:\n{domain_text}\n"
            f"用'因为X所以Y'的格式列出所有因果链。"
        )
        edges = self._extract_causal_edges(cot)
        verified = []
        for edge in edges:
            # 用 DoCalculus 验证
            ate = self._verify_edge(edge)
            if ate is not None and abs(ate) > 0.05:
                edge["ate"] = ate
                edge["source"] = "llm_distillation"
                verified.append(edge)
        return verified
    
    def _extract_causal_edges(self, cot: str) -> list[dict]:
        """从 CoT 文本中提取 '因为X所以Y' 模式"""
        import re
        pattern = r"因为(.+?)所以(.+?)[。\n]"
        matches = re.findall(pattern, cot)
        return [{"cause": m[0].strip(), "effect": m[1].strip()} for m in matches]
```

**文件**: `_distillation/causal_pipeline.py` (~350行)

### 3.2 视觉蒸馏 (G2)

**管线**: CLIP teacher → MCI student (轻量 ViT)

```python
class VisualDistillationTrainer:
    """CLIP → MCI 视觉蒸馏训练器"""
    def __init__(self, teacher_model, student_encoder):
        self._teacher = teacher_model  # CLIP (frozen)
        self._student = student_encoder  # LearnableVisualEncoder
    
    def train(self, images: list, n_epochs=50, lr=1e-4):
        """知识蒸馏训练"""
        for epoch in range(n_epochs):
            for img in images:
                # Teacher embeddings (no gradient)
                z_teacher = self._teacher.encode(img)  # (512,)
                # Student embeddings
                z_student = self._student.encode(img)  # (512,)
                # L2 distillation loss
                loss = np.mean((z_student - z_teacher) ** 2)
                # Backward + update
                self._student.update(loss, lr)
```

**训练数据**: COCO-1K (1000 张代表性图像)
**硬件**: 可选 GPU ($500) 或 CPU (慢 10x)

### 3.3 安全规则蒸馏 (G3)

**管线**: Constitutional AI 原则 → MCI SafetyConstraint 规则

```python
class SafetyDistillationPipeline:
    """Constitutional AI → MCI 安全规则蒸馏"""
    SAFETY_PRINCIPLES = [
        "不生成有害内容",
        "不产生幻觉（事实核查）",
        "保护用户隐私",
        "保持价值中立",
        "承认不确定性",
    ]
    
    def distill(self, llm_client) -> list[SafetyConstraint]:
        constraints = []
        for principle in self.SAFETY_PRINCIPLES:
            # 让 LLM 生成具体规则
            rules = llm_client.generate(
                f"将原则'{principle}'转化为 5 条可检查的规则:"
            )
            for rule in self._parse_rules(rules):
                constraints.append(SafetyConstraint(
                    name=rule["name"],
                    check_fn=self._compile_rule(rule),
                    source="llm_distillation",
                ))
        return constraints
```

### 3.4 动力学蒸馏 (G4)

**管线**: MuZero/Dreamer 预训练 → MCI ActionPredictor

**方案**: 不在 V5.0.0 内复现 MuZero，而是：
1. 用 MuZero 在 Pendulum/Cart 上预训练
2. 提取策略网络权重
3. 初始化 `UniversalActionConditionedPredictor`

## 4. 时间计划

| 周 | 任务 | 交付物 | 里程碑 |
|---|---|---|---|
| W4-6 | 因果结构蒸馏管线 | `causal_pipeline.py` | M1: CoT→边提取 |
| W7-9 | 视觉蒸馏训练器 | 训练脚本 + checkpoint | M2: CLIP→MCI |
| W10-12 | 安全规则蒸馏 | 安全规则集 | M3: 5 原则→规则 |
| W13-16 | 动力学蒸馏 | ActionPredictor 初始化 | M4: MuZero→MCI |
| W17-20 | 领域知识库构建 | 医疗/法律/工程因果图 | M5: 领域知识 |
| W21-24 | 蒸馏质量评估 | 蒸馏效果基准 | M6: 质量验证 |

## 5. 资源配置

| 资源 | 数量 | 成本 |
|---|---|---|
| 后端工程师 | 1人 × 15周 | 45 人天 |
| LLM API (蒸馏数据) | 按需 | $1,000 |
| GPU (视觉蒸馏) | 按需 | $800 |
| 领域专家标注 | 0.5人 × 4周 | 10 人天 |

## 6. KPI 指标

| KPI | 基线 | 目标 |
|---|---|---|
| 因果边蒸馏准确率 | N/A | ≥70% (人工评估) |
| 视觉蒸馏 MSE (CLIP) | N/A | <0.1 |
| 安全规则覆盖 | 0 条 LLM 蒸馏 | ≥25 条 |
| 领域因果图规模 | 0 条 | 医疗≥50条/法律≥30条 |
| 蒸馏后 MCI 因果准确率提升 | 基线 | +15% |

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|
| LLM CoT 提取的因果边不正确 | 高 | 高 | 人工审核 + DoCalculus 验证 |
| CLIP 蒸馏 MSE 过高 | 中 | 中 | 增加训练数据 + 更多 epoch |
| LLM API 成本超预算 | 中 | 中 | 分批蒸馏 + 本地小模型 |
| 蒸馏知识过拟合特定领域 | 中 | 中 | 跨领域验证 |

## 8. 成本预算

| 项目 | 人天 | 硬件/软件 |
|---|---|---|
| 因果蒸馏管线 | 12 | $200 (API) |
| 视觉蒸馏 | 10 | $800 (GPU) |
| 安全蒸馏 | 8 | $200 (API) |
| 动力学蒸馏 | 8 | $0 |
| 领域知识库 | 12 | $600 (API+标注) |
| 质量评估 | 5 | $0 |
| **小计** | **55** | **$1,800** |

## 9. 验收标准

- [ ] 因果蒸馏: 100 个医疗 CoT 中提取 ≥70 条有效因果边
- [ ] 视觉蒸馏: student MSE < 0.1 vs CLIP teacher
- [ ] 安全蒸馏: 5 个原则各生成 ≥5 条可检查规则
- [ ] 领域知识: 医疗因果图 ≥50 条边，法律 ≥30 条边
- [ ] 蒸馏后 MCI 在因果任务上准确率提升 ≥15%

## 依赖关系

- **前置**: Ch03 (LearnableVisualEncoder), Ch04 (F1/F2 修复), Ch02 (TrueJEPA)
- **被依赖**: Ch09 (替代性基准), Ch08 (WMMM L3 因果式)
