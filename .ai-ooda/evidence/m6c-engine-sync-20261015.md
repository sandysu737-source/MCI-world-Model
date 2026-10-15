# M6c 受控同步报告（2026-10-15）

- **同步基线**：kit 提交 `d4bb804`（**已提交状态**；kit 工作树当时被其它窗口改到一半，故不取工作树）
- **通道**：`sync-governance.sh` 的三态语义（本次由人工按引擎逐个 diff 审核后 `--adopt-kit` 等价执行）
- **原则**：先审 diff 再落地；每个引擎落地后立刻跑本仓对抗套件

## 引擎 hash（同步后）

| 引擎 | 同步后 hash（前 12） |
|---|---|
| mutation-check.sh | `1b81e3da23c6` |
| risk-classify.sh | `76c3217dc12e` |
| hook-integrity.sh | `77872d5110b9` |
| ai-guard.sh | `b9d1aef2ddcf` |
| quality-gate.sh | `ccf8e55f9988` |
| review-token.sh | `819d4787…` |

同步后 `test-guard-parity.sh` 的「kit ↔ 仓库副本字节一致」断言为 **PASS** —— 即本仓副本与 kit 源已真正同源（这是 M6 的单源验收项）。

## 引入的 kit 侧能力（按引擎）

- **mutation-check**：多测试目标（逗号分隔）、波次基线 `MUTATION_DIFF_BASE`（干净检出下按「基线..HEAD」取改动行，不再回退全文件把历史盲区记成本次分数）、存活/杀死明细带行号。
- **risk-classify**：F-26 路径枚举 NUL 分隔（修非 ASCII 路径被判 `SKIP not-a-file` 的漏判/绕过面）、H-05-17b 已提交删除回看父提交内容、`.xlsx/.csv` 归 L1（`.sql` 仍 L2）。
- **hook-integrity**：枚举受管辖钩子（pre-commit + commit-msg）、尊重 `core.hooksPath`（修 `.githooks/` 下的死文件比对假安全感）。
- **ai-guard**：`mark_l2_pass`、`QGATE_TMP_DIR` 隔离、载体快照 + `bash -n` 预检、§3.5 提示词↔active 漂移预检。
- **quality-gate**：波次基线 `GATE_DIFF_BASE`、`pytest_diag` 失败诊断、`QGATE_TMP_DIR`（默认仍 `/tmp`）。
- **review-token**：空暂存（CI 干净检出）跳过绑定校验且**显式告警**，非静默降级。

## 回归（全部实测）

| 套件 | 结果 |
|---|---|
| test-guard-parity.sh | **8/0**（含 kit↔副本字节一致） |
| test-anti-evasion.sh | **22/0**（16 → 22 用例；含 9a–9d 载体、10a–10c 覆盖率口径、11 非 ASCII 路径、12a/12b 暂存绑定） |
| test-mutation-check-annotations.sh | **22/0/1**（16 → 22；含 T11 波次基线） |
| test-mutation-skip.sh / test-mutation-map.sh / test-risk-branch.sh / test-owner-approval-marker.sh | 全绿（5/0、7/0、10/0） |
| hook-integrity.sh | rc=0 |

## 遗留（已登记，另行处理）

1. **kit 侧 parity 期望常量陈旧**：kit 源仍钉 `8a15d9f401dc`（≠ 其自身引擎 `1b81e3da`），本仓同步时按「期望值 = 实际采用引擎」重钉并加注；kit 侧待回灌修正。
2. **`ledger-lint.sh` 试用即抓到本仓台账 10 处异常**：行 8–12 列数 11（缺一列）、行 31/45/48/57/59 含未转义竖线（列数 13–15）。工具尚未落地。
3. **`commit-audit.sh` 试用对本次 4 个 `SKIP=ai-guard` 提交报「无放行证据」**（符合设计：本地无 owner-explicit 授权时，放行证据由 CI `gate` + 当日指纹给出）。落地需配套 `l2-trailer-inject.sh` 与 commit-msg 钩子。
