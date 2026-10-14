# M6a 治理资产漂移核查报告（2026-10-15）

- 工具：`_ai-eng-kit/sync-governance.sh --check`（非破坏性同步通道，kit 提交 `77e7211`）
- 复现：`bash "/Users/mac/qoder m5pro/_ai-eng-kit/sync-governance.sh" --check`
- 语义：`INSTALL`=项目缺失该文件（可直接安装）；`CONFLICT`=项目副本与 kit 不一致且**无同步基线记录**（首轮必然出现，需人工判定「本地修复 vs kit 新版」）

## 汇总（9 个项目）

| 指标 | 读数 |
|---|---|
| 项目数 | 9 |
| 缺失可安装（INSTALL） | 33 |
| 冲突（CONFLICT，无基线） | 87 |
| 实际写入 | 0（`--check` 只报告） |

## 各项目分布

| 项目 | INSTALL | CONFLICT |
|---|---|---|
| ai-weight-management-platform | 2 | 13 |
| MCI-SDK | 5 | 10 |
| **mci-world-model（本仓）** | **4** | **11** |
| mci-huan | 6 | 9 |
| ai-tumor-nutrition-platform | 6 | 9 |
| MCI 儿童生长发育智能平台 | 6 | 9 |
| 工作区宿主仓 `/Users/mac/qoder m5pro` | 0 | 13 |
| nutrition-system-hospital | 4 | 7 |
| clinical-nutrition-finance-prod | 0 | 6 |

## 解读

1. **87 个冲突全部源于「无基线」**：旧同步通道（`cp` 覆盖式）不记录同步基线，无法区分「项目本地修复」与「kit 新版滞后」，故一律保守判冲突。建立基线需要一次人工审核后的受控同步（`--adopt-kit` 或人工合并），之后冲突才会收敛为真实本地改动。
2. **本仓 11 处冲突 + 4 处缺失**与已登记的 `MUTATION_STRICT` 降级开关、覆盖率口径下沉、载体策略、workflow K/L 增量一致（`OODA-20261014-057`）。
3. **本次未写入任何项目**（`--check` 只读，已复核 `git status` 干净）。

## 待办（M6b–M6e）

- M6b 回灌 kit：`MUTATION_STRICT` 降级开关（quality-gate + CI 模板 + T10d 回归）。
- M6c 逐引擎受控同步本仓副本，每个引擎同步后跑齐对抗套件。
- M6d 规范固化（本报告与 `.ai-rules.md` §8 为其中一步）。
- M6e 验收：本仓 `test-guard-parity.sh` 本地 8/0 且 parity 常量 = kit 源 hash。
