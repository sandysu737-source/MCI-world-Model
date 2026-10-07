# 大环复盘登记 · 治理脚本「同名 ≠ 同源」双载体分叉

> 日期：2026-10-07 ｜ 层级：**大环**（事件触发） ｜ 主持人：负责人本人 ｜ 执行人：Codex
> 关联台账：`OODA-20261006-045`、`OODA-20261006-048`、`OODA-20261007-052`、`OODA-20261007-053`、`OODA-20261007-054`（本登记）
> 关联材料：`.ai-workflow/2026-10-07-governance-l-testmap-structured-requirement.md` §14.3、`~/qoder m5pro/_ai-eng-kit/governance/tests/test-guard-parity.sh`

## 1. 触发依据（事件触发，当日开环）

| 项 | 事实 | 证据 |
|---|---|---|
| 触发类型 | 事件触发 · ① 同类问题第二次出现 | OODA 规则 §3「大环固定复盘节奏」 |
| 同类问题第 1 次 | 治理 I：pre-commit 调用的 kit 副本缺 F-20 阈值注入，导致本地"合法提交必红" | `OODA-20261006-045`（`open`） |
| 同类问题第 2 次 | 治理 J3：`mutation-check.sh` kit 副本与仓库副本口径分叉 | `OODA-20261006-048`（`writing`） |
| 同类问题第 3 次 | 本机默认用 kit 副本跑对抗套件 → 本轮「本地 7/7 绿 / CI 2 项红」 | `OODA-20261007-052`（`written`）；CI run `37520226504` |
| 开环时限 | 当日内（2026-10-07） | 规则：事件触发优先级高于一切新任务 |
| 阻塞动作 | 本次复盘产出决策条目并回写前，**不开新功能环**（P0-3c 顺延） | 规则：「不清账，不开新环」 |

## 2. 议程 1 · 验旧账

| 旧账 | 上次大环（`OODA-20260917-032`）要求 | 现状 | 判定 |
|---|---|---|---|
| P0-3 数据前提 | 数据权限确认前不承诺真实数据结果 | 本机仍无 MIMIC 授权数据与本地 manifest → 只能 `not_run`/`degraded` | **未落地**（阻塞 P0-3c） |
| 治理改动收口 | `main` 上治理门禁改动须收口 | PR #31/#32/#33/#34 已合并，`main` = `0800bb5` 工作树干净 | 已落地 |
| 未跟踪材料裁决 | `FIRST_PRINCIPLES_PLAN.md` 等需裁决 | 仍未跟踪（按约定不入库，也未明确搁置） | **未落地** |

当前未闭合台账条目（本次复盘必须逐条处置）：

| 条目 | 状态 | 未闭合原因 | 本次需裁决 |
|---|---|---|---|
| `OODA-20261006-045` | `open` | kit 副本 F-20 差异未同步；本机仍以 `SKIP=ai-guard` 提交 | 同步方向与载体 |
| `OODA-20261006-048` | `writing` | 双载体分叉且**双方各有对方没有的修复**（见议程 2） | 单一权威源与回灌顺序 |
| `OODA-20261007-052` | `written` | 回写已落地；缺负责人独立复核 | 置 `verified` 或补验证 |
| `OODA-20261007-053` | `open` | `MUTATION_SCORE=N/A` 空转未处置 | 是否开环 + 判据 |
| `OODA-20261006-044` | `written` | 端到端证据待首个触及 `benchmarks/**` 的 PR | 维持待补或改口径 |

## 3. 议程 2 · 观察（实测证据，2026-10-07）

**A. 单一权威源已由 kit 侧建立，但双载体仍分叉**

| 项 | 实测 |
|---|---|
| kit parity 套件 | `_ai-eng-kit/governance/tests/test-guard-parity.sh` 把 `mutation-check.sh` 钉为单源，期望 hash `870f6217658419be327baa4ec6f7d3fb1011388c6e485544074591133b21f66f`（= kit 现值，kit 于 2026-10-07 14:54 更新） |
| 仓库副本 hash | `800804e15b67e1b09d4f37adececd33feb783d620faa608761d77fb35b699b15`（259 行） ≠ 期望值 |
| kit 独有 | `git diff HEAD` 行号口径（OODA-20261007-103）、索引/工作区不一致告警、纯删除变更判 `N/A`（OODA-20261007-124） |
| 仓库独有 | F-23：f-string 整段排除 + 语法破坏型不计分；`risk-classify.sh` 的 `RISK_BRANCH`（治理 L）；`quality-gate.sh` 内置阈值默认注入（F-20） |
| 直接 `cp` 覆盖的后果 | 退掉 F-23 → 本机 Python 3.13 变异计分失真（同一文件在 3.11/3.13 落点不同） |
| 仓库内 parity 覆盖 | `scripts/ai-verify/tests/` **无** `test-guard-parity.sh`，仓库内无任何引用 → CI 无法自证双载体一致 |

**B. 文件集合差（kit `governance/` ↔ 仓库 `scripts/ai-verify/`，排除 `rollout/` 与 `__pycache__/`）**

- 仅仓库有（10）：`ai-guard.sh.legacy.20260728`、`owner-approval-marker.sh`、`resolve-mutation-test.sh`、`sensitive-terms.toml`、`sensitive_scan.py`、`test-map-gen.py`、`tests/test-mutation-map.sh`、`tests/test-mutation-skip.sh`、`tests/test-owner-approval-marker.sh`、`tests/test-risk-branch.sh`
- 仅 kit 有（8）：`ACTIVATE.md`、`AI-CODE-SPEC.md`、`ONBOARDING.md`、`eslint.governance.config.mjs`、`governance-ci.yml`、`tests/test-guard-parity.sh`、`tests/test-mutation-check-annotations.sh`、`tests/test-quality-gate-project-context.sh`

**C. 同名文件差异行数（`diff | grep -c '^[<>]'`）**

| 文件 | kit | 仓库 | 差异行 |
|---|---|---|---|
| `quality-gate.sh` | 709 | 643 | 224 |
| `mutation-check.sh` | 254 | 259 | 79 |
| `ai-guard.sh` | 184 | 184 | 40 |
| `risk-classify.sh` | 126 | 137 | 13 |
| `hook-integrity.sh` | 41 | 41 | 0 |
| `review-token.sh` | 83 | 83 | 0 |

**D. 意外事件清单（本阶段中环）**

1. CI 回环：把变异门禁 fail-closed 收敛为"仅代码根目录"的放宽被既有契约测试 `test-anti-evasion.sh` 用例 3/7 判红（本地因跑 kit 副本误绿）→ 已撤销放宽，改手工映射区 + 真实单测。
2. `MUTATION_SCORE=N/A` 空转：映射到纯声明/类型文件时变异门禁静默放行（`OODA-20261007-053`）。
3. 本机假绿：`ai-guard.sh:11-19` 在本机优先 kit 副本、CI 才回退仓库副本（`OODA-20261007-052`）。

**E. 影响面（定量）**

| 指标 | 值 |
|---|---|
| 已命中的同类问题次数 | 3（045 → 048 → 052） |
| 受影响载体 | pre-commit、CI `gate`、变异门禁、覆盖率门禁、对抗套件（5 条链路） |
| 受影响项目 | 接入 kit 的全部 8 个项目（本仓为其中之一） |
| 本轮误判代价 | 1 次 CI 回环 + 1 轮返工（`37520226504` 红 → 撤销 → `37520916049` 绿） |

## 4. 议程 3 · 判断（待负责人裁决）

**有效规范（维持）**

| 规范 | 有效性证据 |
|---|---|
| fail-closed 契约测试 | `test-anti-evasion.sh` 用例 3/7 拦下"为绿灯放宽门禁" |
| 指纹绑定「PR 号 + 北京时间当日」 | 旧指纹 `OWNER-APPROVED:2026-10-04:PR22` 在新 PR 失效（`OODA-20261006-049`） |
| 单一权威源 + 期望 hash 断言 | kit `test-guard-parity.sh` 可拦住副本被静默改动 |
| 「验证证据须同源」 | 本轮 `GITHUB_ACTIONS=true` 复现 CI 失败后才找到真根因 |

**系统性偏差（待裁决）**

1. 治理脚本以「文件名 + 固定路径」定位，**同名不同源**的分叉在规范层无显式约束；
2. 本地解析策略（优先 kit）与 CI 策略（优先仓库）**方向相反**，且差异只写在脚本注释与台账里，未进检查单；
3. 共享 kit 与 8 个仓库副本长期双向漂移，缺少「同步即更新期望 hash」的强制流程与 CI 校验；
4. 台账中"跨项目同步需授权"与实际工程需要冲突 → 债务长期挂 `open`（045 已挂 1 天）。

## 5. 议程 4 · 决策（待复盘产出，草案候选）

| 编号 | 候选修订条目 | 载体 | 验收 | 前置授权 |
|---|---|---|---|---|
| D1 | kit 为唯一权威源；`sync-governance.sh` 为唯一同步通道；仓库内引入 parity 套件并纳入 CI | kit、本仓 `scripts/ai-verify/tests/` | 仓库副本 hash == 期望值；CI 运行 parity | 需授权（动 kit + 新增 CI 步骤） |
| D2 | 先把 F-23 两项（f-string 整段排除、语法破坏型不计分）回灌 kit 源，再升级期望 hash | kit `mutation-check.sh` | kit 与仓库副本字节一致且 3.11/3.13 分数一致 | 需授权（动 kit） |
| D3 | 本地验证默认走**仓库副本**，仅显式开关才用 kit（与 CI 同向） | `ai-guard.sh`（两侧） | 不加环境变量即与 CI 同结论 | 需授权 |
| D4 | kit 补齐 F-20 阈值注入与 `RISK_BRANCH`（治理 L 回灌） | kit `quality-gate.sh` / `risk-classify.sh` | 本地合法提交无需 `SKIP=ai-guard` | 需授权（动 kit） |
| D5 | `MUTATION_SCORE=N/A` 语义：严格模式产出显式警告 + 人工确认（先量化红灯数再决定是否 fail-closed） | `mutation-check.sh` / `quality-gate.sh` | `N/A` 路径有留痕 | 可先在本仓开环 |
| D6 | 把「同名 ≠ 同源」写入规范与检查单（含同步流程：改 kit → sync → 更新期望 hash → CI 校验） | `.ai-rules.md`、AGENTS.md、记忆 | 下一次同类问题不再复发 | 需授权（跨项目载体） |

> 决策条目必须在本次复盘内产出（无决策条目的大环视为空转）；产出后逐条写回对应载体，并在下次复盘前达到 `verified`。

## 6. 议程 5 · 复现命令（材料可核查）

```bash
# 双载体 hash 与行数
shasum -a 256 _ai-eng-kit/governance/mutation-check.sh \
  mci-world-model/scripts/ai-verify/mutation-check.sh
wc -l _ai-eng-kit/governance/mutation-check.sh mci-world-model/scripts/ai-verify/mutation-check.sh
# 文件集合差
comm -23 <(cd mci-world-model/scripts/ai-verify && find . -type f -not -path "./__pycache__/*" | sort) \
         <(cd _ai-eng-kit/governance && find . -type f -not -path "./rollout/*" | sort)
comm -13 <(cd mci-world-model/scripts/ai-verify && find . -type f -not -path "./__pycache__/*" | sort) \
         <(cd _ai-eng-kit/governance && find . -type f -not -path "./rollout/*" | sort)
# 同名文件差异行数
for f in ai-guard.sh quality-gate.sh mutation-check.sh risk-classify.sh; do
  diff "_ai-eng-kit/governance/$f" "mci-world-model/scripts/ai-verify/$f" | grep -c '^[<>]'
done
# 本地假绿口径（必须 GITHUB_ACTIONS=true 才是仓库副本）
GITHUB_ACTIONS=true bash scripts/ai-verify/tests/test-anti-evasion.sh
```

## 7. 闭环门禁

- [ ] 议程 1 的 5 条未闭合台账条目全部处置（写回或明确维持 + 理由）
- [ ] 议程 4 产出可执行决策条目（载体 / 责任人 / 期限 / 验收）
- [ ] 决策条目回写并经独立复核，状态达 `written` → `verified`
- [ ] 未获授权的动 kit 条目先出方案报批，不先行改动共享套件
