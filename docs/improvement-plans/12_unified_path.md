# Ch12 统一改进路径 (P0-P3) — 改进规划书

## 1. 章节概述

原报告第十二章提出了五视角融合的四阶段改进路径：P0(2-3周)→P1(4-8周)→P2(8-16周)→P3(16-24周)。本章的职责是**监控整体执行进度**，确保各阶段按计划推进，处理跨章节依赖冲突。

## 2. 改进目标

| # | 目标 | 量化指标 | 优先级 |
|---|---|---|---|
| G1 | P0 阶段按时交付 | W3 结束: 4 Critical 缺陷全部 fixed | **P0** |
| G2 | P1 阶段按时交付 | W11 结束: 7 High 缺陷全部 fixed | P1 |
| G3 | P2 阶段按时交付 | W27 结束: 蒸馏管线 + 形式化验证完成 | P2 |
| G4 | P3 阶段按时交付 | W48 结束: 混合架构 + 自主探索原型 | P3 |
| G5 | 构建项目管理看板 | 12 缺陷 + 30+ 改进项全部追踪 | 贯穿 |

## 3. 实施方案

### 3.1 项目看板数据结构

```python
@dataclass
class Milestone:
    phase: str        # P0/P1/P2/P3
    week_start: int
    week_end: int
    items: list[str]  # 关联的改进项 ID
    gate_checks: list[str]  # 门禁检查项
    status: str       # pending/in_progress/completed/blocked
    
class ProjectTracker:
    """统一项目追踪器"""
    def __init__(self):
        self._milestones: list[Milestone] = []
        self._blocking_issues: list[str] = []
    
    def check_gate(self, phase: str) -> dict:
        """门禁检查: 该阶段是否满足进入下一阶段的条件"""
        return {
            "phase": phase,
            "items_completed": self._count_completed(phase),
            "items_total": self._count_total(phase),
            "gate_passed": self._all_gates_passed(phase),
            "blocking": self._get_blocking(phase),
        }
```

### 3.2 跨章节依赖解析

```python
DEPENDENCY_GRAPH = {
    "Ch04_F1": [],           # 无前置
    "Ch04_F2": [],           # 无前置
    "Ch04_F5": [],           # 无前置
    "Ch04_F10": [],          # 无前置
    "Ch04_F12": [],          # 无前置
    "Ch02_TrueJEPA": ["Ch04_F10"],   # 需要 F10 修复
    "Ch02_PearlChain": ["Ch04_F12"],  # 需要 VAE 修复
    "Ch03_SBERT": ["Ch04_F2"],       # 需要 PEM 修复
    "Ch03_VisualEncoder": ["Ch04_F5"], # 需要维度修复
    "Ch06_CausalCoT": ["Ch02_PearlChain"],
    "Ch10_CausalDistill": ["Ch02_PearlChain", "Ch03_SBERT"],
    "Ch09_HybridGW": ["Ch06_CausalCoT"],
}
```

### 3.3 门禁检查清单

**P0 门禁 (W3)**:
- [ ] F1 NSWM 路由准确率 ≥70%
- [ ] F2 PEM cosine 准确率 ≥0.70
- [ ] F5 VisionEncoder 可学习
- [ ] F10 JEPA 预测器 ≥10K 参数
- [ ] F12 VAE 维度修复
- [ ] `pytest` 全量通过，0 failed

**P1 门禁 (W11)**:
- [ ] F3 Fisher O(N)
- [ ] F4 EWC 遗忘 <25%
- [ ] F6 TrueJEPA 潜向量
- [ ] F7 13 类安全
- [ ] F8 PearlChain 串联
- [ ] F11 MCTS 规划
- [ ] WMMM 得分 ≥65%

**P2 门禁 (W27)**:
- [ ] 因果蒸馏管线工作
- [ ] 视觉蒸馏完成
- [ ] 形式化不变量文档
- [ ] WMMM 得分 ≥70%
- [ ] 综合评分 ≥6.5/10

**P3 门禁 (W48)**:
- [ ] 混合推理网关工作
- [ ] 物理规律自主发现原型
- [ ] 零样本迁移原型
- [ ] WMMM 得分 ≥75%
- [ ] 综合评分 ≥7.0/10

### 3.4 进度可视化脚本

```python
def generate_progress_report(tracker: ProjectTracker) -> str:
    """生成进度报告"""
    report = "# MCI World Model v5.x Progress Report\n\n"
    for phase in ["P0", "P1", "P2", "P3"]:
        gate = tracker.check_gate(phase)
        report += f"## {phase}: {gate['status']}\n"
        report += f"- Completed: {gate['items_completed']}/{gate['items_total']}\n"
        if gate['blocking']:
            report += f"- BLOCKING: {gate['blocking']}\n"
    return report
```

## 4. 时间计划

| 周 | 任务 | 产出 |
|---|---|---|
| W1-3 | P0 执行 + 监控 | 日报 + 周会 |
| W4-11 | P1 执行 + 监控 | 周报 + 门禁检查 |
| W12-27 | P2 执行 + 监控 | 双周报 + 里程碑评审 |
| W28-48 | P3 执行 + 监控 | 月报 + 研究评审 |
| 全程 | 看板维护 + 依赖解析 | 实时看板 |

## 5. 资源配置

| 资源 | 数量 | 成本 |
|---|---|---|
| 项目经理 / Tech Lead | 0.3人 × 48周 | 35 人天 |
| GitHub Projects | 现有 | $0 |

## 6. KPI 指标

| KPI | 基线 | 目标 |
|---|---|---|
| P0 按时交付率 | N/A | 100% |
| P1 按时交付率 | N/A | ≥90% |
| P2 按时交付率 | N/A | ≥80% |
| 门禁检查通过率 | N/A | 100% |
| 阻塞问题平均解决时间 | N/A | <3 工作日 |

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|
| P0 延期导致全线推迟 | 中 | 极高 | 提前 buffer (实际 P0 仅需 2.5 周) |
| 跨章节依赖形成瓶颈 | 高 | 高 | 拓扑排序 + 关键路径优先 |
| 人员变动 | 低 | 高 | 文档 + 知识交接 |

## 8. 成本预算

| 项目 | 人天 | 硬件/软件 |
|---|---|---|
| 项目管理 | 35 | $0 |
| **小计** | **35** | **$0** |

## 9. 验收标准

- [ ] 4 个阶段各有门禁检查清单
- [ ] 依赖关系图覆盖全部改进项
- [ ] P0 在 W3 结束前交付
- [ ] 每个阶段门禁 100% 通过后才进入下一阶段
- [ ] 最终综合评分 ≥7.0/10

## 依赖关系

- **前置**: 所有其他章节
- **被依赖**: 无 (管理协调层)
