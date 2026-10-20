# M6e 治理资产同步收敛报告（2026-10-21）

## 结论

本仓 M6e 验收完成：`sync-governance.sh --check` 从「4 INSTALL / 3 CONFLICT」收敛为
「0 写入 / 0 CONFLICT / 1 登记例外」，rc=0。唯一例外是本仓有意保留的 CI 超集
`.github/workflows/governance.yml`，已在 `.ai-governance/sync-exceptions.tsv` 绑定内容指纹与期限。

## 收敛前读数

复现命令：

```bash
bash "/Users/mac/qoder m5pro/_ai-eng-kit/sync-governance.sh" --check "/Users/mac/qoder m5pro/mci-world-model"
```

收敛前：

- INSTALL 4：`eslint.governance.config.mjs`、`ledger-lint.sh`、`commit-audit.sh`、`test-quality-gate-project-context.sh`
- CONFLICT 3：`.ai-governance/AI-CODE-SPEC.md`、`.ai-governance/README.md`、`.github/workflows/governance.yml`

## 处置

| 对象 | 处置 | 依据 |
|---|---|---|
| `AI-CODE-SPEC.md` | `--adopt-kit` 采用 kit 版 | 本仓版本停留在 2026-09-18，kit 已含证据例外、§3.5 变异裁定、§8 单源纪律 |
| `README.md` | `--adopt-kit` 采用 kit 版 | 仅缺 M6 同步纪律小节 |
| `eslint.governance.config.mjs` | 安装 kit 版 | 缺失的通用治理配置 |
| `ledger-lint.sh` | 安装 kit 版 | 缺失的台账结构门禁 |
| `commit-audit.sh` | 安装 kit 版 | 缺失的提交后审计工具 |
| `test-quality-gate-project-context.sh` | 不安装 | 该套件绑定 MCI-BMI 的 56/32 阈值、`backend/requirements-ci.txt` 与项目专属 L2 文案，跨仓安装必红；已从 kit 通用载体列表移除 |
| `governance.yml` | 登记 `repo-superset` 例外 | 本仓含 benchmarks 硬门禁、`RISK_BRANCH` / `GATE_DIFF_BASE` 波次基线、单人 owner 指纹、项目专属覆盖率阈值；kit 模板不携带这些项目上下文 |

例外文件：`.ai-governance/sync-exceptions.tsv`。模式为 `repo-superset`，指纹绑定
`0e56df099e756067`，期限 `2027-10-31`；指纹漂移或逾期会自动回落为 CONFLICT。

## 工具侧改动

kit 提交 `afb6993`：

- `sync-governance.sh` 新增项目级例外通道：只接受 `repo-superset` / `mixed-merge`，必须匹配 16 位内容指纹且未逾期；否则回落 CONFLICT。
- `test-sync-nondestructive.sh` 新增 T7–T9：匹配例外不写入、指纹失配失败关闭、逾期失败关闭。
- 通用载体列表移除 `test-quality-gate-project-context.sh`，避免项目专属套件跨仓安装即红。
- `AI-CODE-SPEC.md` §8、`base/governance-overlay.md`、`CHANGELOG.md` 同步记录例外纪律。

## 验证

| 检查 | 结果 |
|---|---|
| `sync-governance.sh --check` | 写入=0、冲突=0、例外=1，rc=0 |
| `governance/tests/test-sync-nondestructive.sh` | PASS=9 FAIL=0 |
| 本仓 `test-guard-parity.sh`（仓库模式） | PASS=7 FAIL=0 |
| kit `test-guard-parity.sh`（自检模式） | PASS=8 FAIL=0 |
| 本仓 `test-anti-evasion.sh` | PASS=38 FAIL=0 WAIVE=0 |
| 本仓 `test-mutation-check-annotations.sh` | PASS=22 FAIL=0 SKIP=1 |
| `ledger-lint.sh .ai-ooda/ledger.md` | 59 条合法，rc=0 |
| `sensitive_scan.py` 改动文档 | status=pass，findings=0 |
| kit pre-push | 对抗 38/0、治理覆盖率 4/0 |

## 口径校正

历史台账中的「parity 8/0」指 **kit 自检模式**；本仓仓库模式当前是 **7/0**。
两者差一条「kit 载体与基线一致」断言，不是本仓少跑一条行为回归。

## 残留与期限

- `.github/workflows/governance.yml` 的 `repo-superset` 例外期限为 `2027-10-31`。到期前应评估把通用部分上游化到 kit，或重新登记并说明理由。
- `ai-weight-management-platform` 与 `clinical-nutrition-finance-prod` 已各自持有项目专属的 `test-quality-gate-project-context.sh`；本次只停止向其它仓继续安装，不代改这两个项目。
- 本报告不替代 CI：PR 合并前仍以 governance gate 与 4-Gate 结果为准。
