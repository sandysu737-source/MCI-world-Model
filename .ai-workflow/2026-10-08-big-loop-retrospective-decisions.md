# 大环复盘产出 · 治理脚本「同名 ≠ 同源」双载体分叉

> 日期：2026-10-08 ｜ 层级：**大环**（事件触发） ｜ 主持人：Codex（负责人 2026-10-08 委托） ｜ 裁决：负责人本人
> 登记：`OODA-20261007-054`（2026-10-07 开环） ｜ 登记材料：`.ai-workflow/2026-10-07-big-loop-retrospective-registration.md:1`
> 授权：负责人 2026-10-08 明确「委托主持 + 授权共享 kit」

## 1. 议程执行情况

| 议程 | 状态 | 说明 |
|---|---|---|
| 1 验旧账 | ✅ 完成 | 见 §2 |
| 2 观察 | ✅ 完成 | 登记文档 §3 材料 + 本轮新增两项事实（见 §3） |
| 3 判断 | ✅ 完成 | 见 §3 |
| 4 决策 | ✅ 完成 | 见 §4（M1–M6，含载体/责任人/期限/验收） |
| 过程缺陷（如实记录） | ⚠️ | 登记当日（10-07）未产出决策条目，跨日至 10-08 完成 → 计入下次复盘的「复盘按期落地率」 |

## 2. 议程 1 · 验旧账结果

### 2.1 上次大环（`OODA-20260917-032`）条目

| 条目 | 要求 | 现状 | 判定 |
|---|---|---|---|
| P0-3 数据前提 | 数据授权确认前不承诺真实数据结果 | 本机仍无 MIMIC 授权数据与本地 manifest | **维持阻塞**（`P0-3c` 不开） |
| 治理改动收口 | `main` 上治理门禁改动须收口 | PR #31/#32/#33/#34/#35 已合并，`main` = `eb4500a`，工作树干净 | 已落地 |
| 未跟踪材料裁决 | `FIRST_PRINCIPLES_PLAN.md` 需裁决 | 仍未跟踪 | **本次裁决：明确搁置**（本地参考材料，不入库；不再计为欠账） |

### 2.2 本轮未闭合条目逐条处置

| 条目 | 原状态 | 处置 | 依据 |
|---|---|---|---|
| `OODA-20261006-045`（kit 副本缺 F-20 阈值注入） | `open` | 由 **M1c** 回灌 kit + **M2** 同步后复验（本机合法提交不再需要 `SKIP=ai-guard`） | 本环执行 |
| `OODA-20261006-048`（`mutation-check.sh` 双载体分叉） | `writing` | 由 **M1a**（F-23 回灌）+ **M2**（同步）+ **M3**（parity 入 CI）闭合 | 本环执行 |
| `OODA-20261007-052`（本地默认跑 kit 副本 → 假绿） | `written` | 由 **M4**（本地/CI 同向 + 不一致告警）闭合；回写文书已落地 | 本环执行 |
| `OODA-20261007-053`（`MUTATION_SCORE=N/A` 空转） | `open` | **M5** 独立开环（先量化影响面再定判据） | 本环决策 |
| `OODA-20261006-044`（`benchmarks/**` 端到端证据） | `written` | **维持待补**（首个触及 `benchmarks/**` 的 PR，预计 `P0-3c`） | 维持 |
| `OODA-20261007-054`（本次复盘） | `open` | 本文件产出决策条目后置 `written` | 本文件 |

## 3. 议程 3 · 判断（结论）

**新增两项事实（本轮核实，改变了决策前提）**

1. **kit 已是独立版本仓**：`_ai-eng-kit` 自带 `.git`（`7ffc3bb`，2 次提交），且已包含**其它窗口**的修复（`risk-classify.sh` 的「删除即放行」盲区修复、`hook-integrity`/`ledger-lint`/`commit-audit` 三个钩子、4 套治理回归套件）。
2. **漂移是全局的、且双向**：另一位窗口的只读核查（`_ai-eng-kit/governance-rollout/20261007-governance-drift-check/REPORT.md`）显示 8 个注册项目的副本 **8/8 漂移**、`ledger-lint.sh`/`commit-audit.sh` **8/8 缺失**；而本仓独有的修复（F-23、`RISK_BRANCH`、F-20 阈值注入）**只在本仓**，kit 没有 → 「上游落后于下游」与「下游落后于上游」同时存在。

**根因（收敛为一条）**

> 治理引擎以「同名文件 + 固定路径」分发，**没有版本契约**：单源未被 CI 校验，副本各自演化，本地与 CI 又各取一份不同载体 → 任何一处修复都可能在另一侧静默丢失。

**系统性偏差（4 项，全部保留有效）**

| # | 偏差 | 证据 |
|---|---|---|
| B1 | 同名不同源在规范层无显式约束 | 045/048/052 三次命中同一根因 |
| B2 | 本地解析（优先 kit）与 CI（优先仓库）方向相反 | `scripts/ai-verify/ai-guard.sh:11-19` |
| B3 | 双向漂移无强制同步/校验流程 | 8/8 漂移 + kit 缺本仓三项修复 |
| B4 | 「跨项目同步需授权」导致债务长期挂账 | 045 挂账 1 天；048 挂账 2 天 |

**有效规范（维持，不弱化）**

- fail-closed 契约测试（`test-anti-evasion.sh` 用例 3/7 拦下为绿灯放宽）；
- 审批指纹绑定「PR 号 + 北京时间当日」；
- 单一权威源 + 期望 hash 断言（kit `test-guard-parity.sh`）；
- 高风险动作先方案后执行、留痕可追溯。

## 4. 议程 4 · 决策条目（M1–M6，全部含载体/责任人/期限/验收）

> 优先级顺序 = 先打通「单源 + 同步 + 校验」闭环（M1→M2→M3），再改语义（M4/M5），最后固化规范（M6）。
> 授权状态：负责人 2026-10-08 已授权动共享 kit；M3 涉及 CI workflow 改动，需负责人当日指纹（L2）。

| 编号 | 决策 | 载体 | 责任人 | 期限 | 验收标准 |
|---|---|---|---|---|---|
| **M1** | 本仓三项修复**上游回灌 kit 源**（保留 kit 已有修复，纯增量）：<br>a) `mutation-check.sh` 补 F-23（f-string 整段排除 + 语法破坏型不计分）；<br>b) `risk-classify.sh` 补 `RISK_BRANCH` 覆盖 + detached 警告；<br>c) `quality-gate.sh` 补 `COV_CORE_THRESHOLD`/`COV_BR_THRESHOLD` env 覆盖（`.ai-coverage-threshold` 机制保留，env 优先） | `_ai-eng-kit/governance/{mutation-check,risk-classify,quality-gate}.sh` | Codex | 2026-10-08 | 三项修复位于 kit；本仓 `bash scripts/check.sh` 与本仓回归套件在**同步后**全绿；kit 侧改动逐项有 diff 说明 |
| **M2** | **本仓同步 kit 并复验**：`sync-governance.sh "<本仓>"`；补齐 `ledger-lint.sh`/`commit-audit.sh`；按回归栈复验 | `scripts/ai-verify/**`、`.ai-governance/**` | Codex | 2026-10-08 | 副本 hash == kit hash（除本仓专属脚本）；`test-anti-evasion` 7/7、`test-mutation-skip`、`test-risk-branch` 7/7、`test-mutation-map` 5/5、`test-owner-approval-marker` 10/10 全绿；`quality-gate.sh L2 benchmarks/real_world/p03_metrics.py` 变异 100% |
| **M3** | **parity 入 CI + 期望 hash 治理流程**：把 `test-guard-parity.sh` 纳入本仓 `scripts/ai-verify/tests/` 并在 governance workflow 的对抗回归步骤执行；同步 kit 侧期望 hash 常量；流程固化为「改 kit → sync → 更新期望 hash → CI 校验」 | `scripts/ai-verify/tests/test-guard-parity.sh`、`.github/workflows/governance.yml` | Codex 实施；负责人授权 L2 | 2026-10-09 | CI 中 parity 通过；仓库副本被静默改动时 CI 变红（反例验证）；期望 hash 与 kit 一致 |
| **M4** | **本地/CI 同向**：`ai-guard.sh` 默认使用**仓库副本**；仅显式 `AI_GUARD_USE_KIT=1` 用 kit；检测到双载体不一致时打印同步指令与告警 | `_ai-eng-kit/governance/ai-guard.sh`（+ 同步至本仓） | Codex | 2026-10-09 | 不带环境变量时本地与 CI 结论一致；不一致场景产出显式告警（留痕）；`OODA-20261007-052` 同场景不再假绿 |
| **M5** | **`MUTATION_SCORE=N/A` 语义开环**：先量化影响面（多少受约束文件会落 N/A、其中多少会新增红灯），再定「严格模式显式告警+人工确认」或「fail-closed」 | `scripts/ai-verify/{mutation-check,quality-gate}.sh`、本仓台账新条目 | Codex 起草；负责人裁决 | 2026-10-09 | 影响面量化报告 + 判据选定并落地；`N/A` 路径有可追溯判定 |
| **M6** | **规范固化**：把「同名 ≠ 同源」的四件套（单源=kit、唯一同步通道、parity 校验、漂移核查）写入规范与检查单 | `_ai-eng-kit/governance/AI-CODE-SPEC.md`、`_ai-eng-kit/base/governance-overlay.md`、本仓 `.ai-rules.md`、全局记忆 | Codex 起草；负责人批准 | 2026-10-11（下次复盘前） | 下一次同类评审可据规范直接判分叉；新项目接入默认带 parity 校验 |

**弃选（写明理由）**

| 弃选 | 理由 |
|---|---|
| 直接 `cp` kit→本仓（或反向） | 会静默丢掉另一侧修复：kit 有「删除即放行」修复与 3 个钩子，本仓有 F-23/`RISK_BRANCH`/F-20 → 必须先回灌再同步 |
| 取消双载体、只保留 kit 一份 | 各项目 CI 需在无 kit 的 runner 上运行 → 副本是必需的；正确姿势是「单源 + 校验」而非「只剩一份」 |
| 让 8 个项目统一由本窗口代跑 `sync-governance.sh` | 违反「不跨项目混改」（用户口径与 `OODA-20261007-GOVDRIFT-001` 一致）；改由各窗口按同一流程自证 |
| 把「本地默认用 kit」直接改成硬失败 | 会让 8 个未同步项目本地门禁立即不可用；先告警 + 同向，等 M2/M3 铺开后再评估收紧 |
| 放宽 `N/A` 或给映射加白名单以换绿灯 | 属「为绿灯调口径」，红线 |

## 5. 回写计划（知行闭合）

| 动作 | 载体 | 期限 |
|---|---|---|
| 本文件产出决策条目 | `.ai-workflow/2026-10-08-big-loop-retrospective-decisions.md` | 2026-10-08 |
| 台账 `OODA-20261007-054` → `written`（引用本文件） | `.ai-ooda/ledger.md` | 2026-10-08 |
| M1/M2 执行后回写 `045`/`048`/`052` → `verified` | `.ai-ooda/ledger.md` | 2026-10-08 |
| M3 执行后回写（含 CI 反例证据） | `.ai-ooda/ledger.md` | 2026-10-09 |
| M5/M6 完成后各自条目验证 | `.ai-ooda/ledger.md`、kit `AI-CODE-SPEC.md` | 2026-10-11 前 |
| 跨项目条目（8 个项目的同步与复验） | 全局 `~/qoder m5pro/OODA回写台账.md`（`OODA-20261007-GOVDRIFT-001` 已登记） | 各窗口自定 |

## 6. 复盘闭环门禁自检

- [x] 议程 1 的 6 条未闭合条目全部处置（执行 / 裁决 / 维持，均写明依据）
- [x] 议程 4 产出可执行决策条目（M1–M6 + 载体/责任人/期限/验收）
- [x] 弃选项与理由写明
- [ ] M1–M6 回写并经独立复核达 `verified`（进行中：M1/M2 本轮执行）
- [x] 未授权事项不先行（M3 的 workflow 改动标注需负责人当日指纹；跨项目同步不代跑）
