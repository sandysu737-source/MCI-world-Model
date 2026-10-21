# Governance 叠加层（AI 代码生成规范 v2）

> 本文件说明 `_ai-eng-kit/governance/` 规则包如何叠加到现有四阶段流水线之上。
> 所有项目经 `install.sh`（新项目）或 `sync-governance.sh`（已接入项目）自动注入。

## 一句话定位
现有 SOP 管「流程顺序」（需求→架构→编码→测试→质检），governance 管「分级路由 + 量化门禁」——告诉流水线每段代码该走多严的卡。

## 三个脚本（落地在 `scripts/ai-verify/`）
1. `risk-classify.sh [files...]` — 对改动逐文件输出 L0/L1/L2 + 依据，退出码=最高等级。**保守兜底：默认 L2**，仅纯工具/测试路径且无危险 import 才降 L0。
2. `quality-gate.sh <L0|L1|L2> [files...]` — 按等级跑差异化门禁（覆盖率/圈复杂度/SAST/lint），报告归档 `.ai-governance/reports/`。
3. `ai-guard.sh`（governance 增强版）— 串联：超大diff拦截 → 定级 → 门禁 → 路由：
   - L0 全过 → 放行
   - L1 全过 → 放行 + 标记需架构/用例评审
   - L2 全过 → **阻断提交**，待双人源码Review（`AI_REVIEW_CONFIRMED=1` 可放行，仍需后续灰度）
   - 任一门禁失败 → 阻断

## 与四阶段的映射
| 流水线阶段 | governance 增强 |
| --- | --- |
| 需求规格化 | 灰区时 AI 输出 `confidence: low` 回退人工，不无限重试 |
| 架构约束 | 质量门禁含架构 lint（跨层/违规依赖） |
| 编码 | 编码 Agent 先看 `risk-classify` 结果决定本段等级 |
| 质检 | 质检 Agent 跑 `quality-gate.sh <等级>` 拿量化报告 |

## 关键修正（相对原规范）
- 默认保守定级 L2，AI 不自评（解决"谁来定级"盲点）
- 圈复杂度 ≤15（原 ≤10 过严），规则引擎/状态机豁免 ≤20
- 覆盖率差异化（L0/L1/L2 不同阈值），变异分数 ≥80% 可豁免 5% 缺口
- SAST 误报走「修复/豁免」闭环，禁止静默压制
- 门禁失败 `MAX_RETRY=3` 后转人工，防烧 token
- 删除"禁止 L0/L1 申请人工审核"——人工主动介入是正向行为

## 升级 governance 规则
改 `_ai-eng-kit/governance/` 源头后，全量重推：
```bash
bash "/Users/mac/qoder m5pro/_ai-eng-kit/sync-governance.sh"
```
单项目：
```bash
bash "/Users/mac/qoder m5pro/_ai-eng-kit/sync-governance.sh" "/项目路径"
```

### 治理资产单源与同步纪律（M6）
- **单源**：编辑源是 `_ai-eng-kit/governance/` + `base/`；项目副本是同步产物，不得作为唯一真源私自演进。
- **唯一通道**：只走 `sync-governance.sh` 三态非破坏语义；禁止 `cp` 覆盖式同步。
- **本地改动可回灌**：项目副本的本地修复必须登记台账并回灌 kit；采用 kit 版必须显式 `--adopt-kit`。项目级超集须在 `.ai-governance/sync-exceptions.tsv` 登记指纹与期限，漂移或逾期即回落冲突。
- **parity 校验**：`test-guard-parity.sh` 本地必须绿；CI 无 kit 只能证明副本未被静默改动。
- **漂移核查**：每次治理合并后或每周至少一次跑 `sync-governance.sh --check`；漂移未清账前不开新的治理环。

## 工单合同与执行前验证口径（跨项目通用）

- 构建、部署、合并、跨 PR 或对外交付的工单，必须先写合同卡：原始验收条款、合同要求与可选加固、依赖与部署影响、唯一验证入口、回滚责任人、T6+ 采样时间。
- **相关不等于依赖**：仅相关不得作为前置条件；一个生产窗口只归因一个主要变更。
- **不得擅自加严验收**：新增 no-cache、额外构建或额外门禁，必须标为可选加固；升级为阻塞项须负责人确认并记录成本与弃选方案。
- **单一验证入口**：必须复用项目脚本或 workflow；裸命令只能排查，不得作为关账证据。
- **T6+ 通用口径**：1-min load 小于本项目 CPU 核数，项目有更严阈值取更严值；无其他项目全量测试；runner `online/idle`；无 `in_progress` run；涉及容器时 Docker 正常；磁盘足够。不适用项显式记为 `N/A`。
- **复核前置**：AI 提交负责人复核前，必须先完成独立自审或对抗性审计，给出问题、修复、残余风险和明确建议；自审不替代独立复核，也不构成自批。

## 给 AI 助手的开场补充提示
```
本项目已注入 governance 规则包（.ai-governance/AI-CODE-SPEC.md）。
写业务代码前先跑 scripts/ai-verify/risk-classify.sh 确认风险等级；
按等级对应阈值实现，提交前 scripts/ai-verify/ai-guard.sh 必须通过。
L2 代码禁止跳过双人 Review 与灰度。
```
