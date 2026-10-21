# 项目收口报告（2026-10-21）

## 结论

MCI World Model 达到可收口状态：`main` 基线为 `968bd60`，无开放 PR、无开放 Issue、远端仅 `main` 一条分支、无 `codex/*` 残留；M6d/M6e 已合并并取证，M6f 治理文档已按单源通道同步，`sync-governance.sh --check` 收敛为「0 写入 / 0 冲突 / 1 登记例外」，本地全量与 CI 均绿。

收口动作本身仍走 PR：取得负责人当日 `OWNER-APPROVED` 指纹、PR gate 绿后 squash 合并，台账条目由 `written` 转 `verified`。

## 仓库基线

| 项 | 读数 |
|---|---|
| `main` HEAD | `968bd60`（`docs(ooda): M6e 合并证据回写并转 verified (#48)`） |
| `origin/main` | `968bd60`，与本地一致 |
| 远端分支 | 仅 `main`；`refs/heads/codex/*` 计数为 0 |
| 开放 PR | 0 |
| 开放 Issue | 0 |
| 未跟踪件 | `.serena/`、`FIRST_PRINCIPLES_PLAN.md`；按仓库纪律不提交 |

## 合并与 CI

| 对象 | 结果 |
|---|---|
| PR #47 | MERGED，squash `18ff9a7`；`gate` success；4-Gate 全 pass（Gate 4 首跑平台 cancelled，重跑 43s pass） |
| PR #48 | MERGED，squash `968bd60`；`gate` success；4-Gate 全 pass |
| main push governance `39533326664` | success |
| main push 4-Gate `39533326803` | success（2026-10-21T02:58:21Z 完成；3.10/3.11/3.12/3.13、ruff、mypy、benchmarks 全绿） |

历史红灯 `39478652664`（`c1874f6`）的唯一失败作业是 `Gate 4 · SIGReg Bench Smoke` 被平台 `cancelled`，不是断言失败；该提交已被 `18ff9a7` / `968bd60` 的绿跑覆盖。

3.11 全量测试作业本次耗时约 26 分钟，与上一轮 `39530788967` 的 25m42s 一致；已排除挂死，不以「比 3.10/3.13 慢」判失败。

## M6f 同步

kit 单源 HEAD 为 `7e91854`（`feat(governance): 固化工单合同与执行前验证口径（M6f）`），该提交已推送至 kit 分支 `codex/governance-workorder-v1`。本仓通过 `sync-governance.sh` 同步，未使用 `cp` 或整文件覆盖。

| 文件 | 变化 | 同步后 hash |
|---|---|---|
| `.ai-governance/AI-CODE-SPEC.md` | 新增 §9「工单合同与执行前验证口径」（+26 行） | `c52748796cc89a7d` |
| `.ai-governance/README.md` | 新增工单合同速览（+9 行） | `1e6fbfe85d43cf01` |
| `.ai-governance/sync-manifest.tsv` | 基线更新 2 行 | 由工具维护 |

复验：`sync-governance.sh --check` → 写入=0、冲突=0、例外=1、rc=0。

## 本地验证

| 检查 | 结果 |
|---|---|
| `bash scripts/check.sh` | `4841 passed, 1 skipped, 12 warnings in 581.63s`，rc=0 |
| `test-guard-parity.sh`（仓库模式） | PASS=7 FAIL=0 |
| kit `test-guard-parity.sh`（自检模式） | PASS=8 FAIL=0 |
| `test-anti-evasion.sh` | PASS=38 FAIL=0 WAIVE=0 |
| `test-mutation-check-annotations.sh` | PASS=22 FAIL=0 SKIP=1 |
| `ledger-lint.sh .ai-ooda/ledger.md` | 59 条合法，rc=0 |
| `sensitive_scan.py` 改动文件 | status=pass，findings=0 |
| `git diff --check` | rc=0 |

## 已知例外与边界

- 唯一登记例外：`.github/workflows/governance.yml`，模式 `repo-superset`，指纹 `0e56df099e756067`，期限 `2027-10-31`。本仓 CI 超集包含 benchmarks 硬门禁、`RISK_BRANCH` / `GATE_DIFF_BASE` 波次基线、单人 owner 指纹与项目专属覆盖率阈值，kit 模板不携带这些上下文。
- 根仓全局台账 `OODA回写台账.md` 当前只含其它窗口的 `OODA-20261021-WORKORDER-001` 未提交行；本窗口不代改、不混提。M6e 属项目内闭环，已在项目台账 `OODA-20261015-058` 与 `m6e-sync-convergence-20261021.md` 取证，无需另立全局行。
- kit 工作树仍有其它窗口的 `governance/kit-exceptions.tsv` 修改；本窗口未触碰。

## 收口判定

功能、治理、同步、CI、台账五项均已具备证据。剩余动作只有本次收口 PR 的负责人指纹、PR gate 复跑与 squash 合并；完成后本报告即为项目收口基线。
