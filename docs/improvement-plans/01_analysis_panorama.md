# Ch01 分析视角全景与核心发现矩阵 — 改进规划书

## 1. 章节概述

原报告第一章汇总了 5 次跃迁升级分析（R1基础评估/R2-GLM/R3-DeepSeek/R4-MiniMax/R5-Kimi）的核心发现矩阵。当前问题：
- 5 次分析使用不同框架（杨立昆5维/Pearl三支柱/WMMM六层/形式化-涌现-经济），结论难以直接对比
- 缺少统一的回归测试基准验证各视角发现的持久性
- 缺少"发现→修复→验证"的闭环追踪机制

**当前评分**: 信息整合度 60% — 有矩阵但无闭环

## 2. 改进目标

| # | 目标 | 量化指标 | 优先级 |
|---|---|---|---|
| G1 | 建立统一分析基准测试套件 | 覆盖 5 视角 × 12 致命缺陷的回归测试 | P1 |
| G2 | 构建发现追踪看板 | 每个发现有 owner/status/验证日期 | P1 |
| G3 | 自动化多视角评估流水线 | 一键运行 5 类评估脚本 | P1 |

## 3. 实施方案

### 3.1 基准测试套件 (`benchmarks/multi_perspective/`)

```
benchmarks/multi_perspective/
├── run_all.py              # 统一入口
├── pearl_causal_suite.py   # R2 GLM 因果三支柱测试
├── quantitative_stress.py  # R3 DeepSeek 压力测试
├── wmmm_level_check.py     # R4 MiniMax WMMM 层级测试
├── formal_verify.py        # R5 Kimi 形式化检查
└── baseline_v5.json        # v5.0.0 基线数据
```

### 3.2 发现追踪数据结构

```python
@dataclass
class Finding:
    id: str               # F1-F12
    source: str           # R1/R2/R3/R4/R5
    severity: str         # Critical/High/Medium
    component: str        # 源码文件
    status: str           # open/in_progress/fixed/verified
    owner: str            # 责任人
    fix_pr: str           # 修复 PR 链接
    verified_date: str    # 验证日期
```

### 3.3 自动化评估脚本

每次 CI/CD 流水线运行时自动执行：
1. 跑 12 项致命缺陷的回归测试
2. 计算 WMMM 层级得分变化
3. 输出与 `baseline_v5.json` 的 diff 报告

## 4. 时间计划

| 周 | 任务 | 交付物 |
|---|---|---|
| W4 | 设计基准测试套件架构 | `run_all.py` + 接口定义 |
| W5 | 实现 5 个测试子套件 | 5 个 `.py` 文件 |
| W6 | 集成 CI + 发现追踪看板 | CI 配置 + JSON 看板 |

## 5. 资源配置

| 资源 | 数量 | 成本 |
|---|---|---|
| 后端工程师 | 1人 × 3周 | 12 人天 |
| CI/CD runner | 现有 | $0 |
| 看板工具 | GitHub Issues | $0 |

## 6. KPI 指标

| KPI | 基线 | 目标 | 度量方式 |
|---|---|---|---|
| 基准测试覆盖率 | 0% | 100% (12/12 缺陷) | 测试用例数 / 缺陷数 |
| 发现追踪完成率 | 0% | 100% | 有 owner 的发现 / 总发现 |
| CI 执行时间 | N/A | < 5 min | CI pipeline duration |
| 回归检测灵敏度 | N/A | 100% (已知缺陷可复现) | 故意引入 → 测试失败 |

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|
| 测试套件与代码耦合过紧 | 中 | 中 | 通过接口抽象，测试只调 public API |
| CI 运行时间过长 | 低 | 低 | 分 fast/slow 两个 tier |
| 基线数据过时 | 中 | 中 | 每次发版自动更新 baseline |

## 8. 成本预算

| 项目 | 人天 | 硬件/软件 | 合计 |
|---|---|---|---|
| 基准测试开发 | 8 | $0 | 8 人天 |
| 发现追踪系统 | 2 | $0 | 2 人天 |
| CI 集成 | 2 | $0 | 2 人天 |
| **小计** | **12** | **$0** | **12 人天** |

## 9. 验收标准

- [ ] `benchmarks/multi_perspective/run_all.py` 可一键运行
- [ ] 12 项致命缺陷各有 ≥1 个回归测试
- [ ] 故意回退 F1(NSWM hash路由) → 测试失败
- [ ] 故意回退 F2(PEM char-hash) → 测试失败
- [ ] CI pipeline 5 分钟内完成
- [ ] `baseline_v5.json` 包含 v5.0.0 所有关键数值

## 依赖关系

- **前置**: 无（可独立开发）
- **被依赖**: Ch04(致命缺陷修复后需此套件验证) → Ch12(统一路径)
