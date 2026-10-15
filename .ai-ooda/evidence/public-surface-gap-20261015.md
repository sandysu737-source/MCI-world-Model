# 公开面缺口登记：敏感术语门禁口径与发布面不一致（2026-10-15）

> 台账：`OODA-20261015-PUBSURFACE-016`（本仓为**公开仓**，故登记件不含类别名与词条原文——
> 类别名本身即敏感面；本文件只记计数、路径与复现命令）

## 1. 事实（可复现）

| 口径 | 命令 | 覆盖文件 | 命中 | 结论 |
|---|---|---|---|---|
| 交付路径（现行门禁） | `python scripts/ai-verify/sensitive_scan.py src README.md CHANGELOG.md docs` | 316 | 0 | pass |
| 全仓口径（本登记） | `python scripts/ai-verify/sensitive_scan.py --tracked` | 922 | **52** | **fail** |

词表版本 `1.1.0`；词表文件自身已按 `self_excluded` 排除（不计命中，留痕于报告）。

## 2. 根因

门禁口径 `src README.md CHANGELOG.md docs` 编码的是**私有仓假设**——「发布面 ＝ 交付路径」，
只有会被交付出去的路径才需要脱敏。本仓为**公开仓**（MIT，创建日 2026-06-03 起 public），
入库即发布，发布面 ＝ 全部受版本控制文件。两者相差 **606 个文件**，期间的 `status=pass`
是**假绿**：它证明的是「已扫描的部分干净」，而不是「已发布的部分干净」。

## 3. 命中分布（按文件）

| 文件 | 命中数 |
|---|---|
| `tests/test_sys_coverage_fill.py` | 17 |
| `tests/test_temporal_core.py` | 11 |
| `tests/test_imports.py` | 6 |
| `tests/test_foundation_types.py` | 5 |
| `benchmarks/test_causal_bench.py` | 3 |
| `tests/test_energy_bridge_propagation.py` | 3 |
| `tests/test_energy_counterfactual_bridge.py` | 3 |
| `tests/adversarial/test_sensitive_scan.py` | 2 |
| `tests/test_untested_modules_smoke.py` | 2 |

## 4. 已做的处置（不改变公开策略）

1. 扫描器修正自指缺陷：词表文件自身不再计入命中，且在报告 `self_excluded` 留痕
   （否则一旦扩口径，扫描器必然先在自己的词表上失败，扩口径从第一步就不可行）。
2. 新增全仓口径 `--tracked`（发布面 ＝ 受版本控制文件），并接入 `scripts/check.sh` 与
   `ci.yml`；**当前非阻断**（`continue-on-error`），但状态与命中数必须可见。
3. 明细只落 `.ai-governance/reports/`（已 gitignore）：文件名与类别名不得进入公开日志或提交。

## 5. 待负责人裁定（未自裁）

| 选项 | 含义 | 代价 |
|---|---|---|
| A 脱敏 | 改写 `tests/`、`benchmarks/` 中的领域术语为安全别名 | 需改测试语义/断言文本，工作量中等 |
| B 白名单 | 对测试/基准路径显式登记豁免（逐条 + 期限 + 审批人） | 保留公开暴露，但豁免可见可审计 |
| C 收窄公开面 | 公开仓只保留 `src`/`docs`/`papers` 等，测试与基准迁至私有仓 | 结构性调整，影响 CI 与开发流 |

**未做**：把仓库转为 private。依据：`LICENSE`(MIT) 与 README 徽章表明公开是创建日即定的有意安排，
转私密属「对外可见」动作且会破坏既定发布策略，须负责人裁定。
