# 结构化开发需求：J 环 · 变异门禁计分口径与边界测试

- 日期：2026-10-06
- 风险等级：L2（改动 CI 门禁脚本与门禁编排）
- 环：治理环 J（J1 + J2），台账 `OODA-20261006-046`（J2）/`047`（J1）/`048`（J3，未开）
- 触发：`e0d942e` 合并后 main push `governance` = failure（run `37443625783`，变异分数 44% < 80%）

## 1. 业务目标

1. 让变异门禁的"有效变异"只统计**可能改变可观察运行行为**的代码位点；
2. 让同一份源文件在 Python 3.11 / 3.13 / 3.14 下得到**相同的落点与分数**（口径不随解释器版本漂移）；
3. 让 `benchmarks/real_world/p03_metrics.py` 的变异分数**可达标**（≥80%），从而清掉 main push 常态红灯；
4. **不通过**降低阈值、放宽风险等级、跳过门禁等方式换绿灯。

## 2. 输入与参数

| 参数 | 说明 |
|---|---|
| 位置参数 1 | 源文件路径（纯文本，UTF-8） |
| 位置参数 2 | 测试目标（传给 pytest） |
| 位置参数 3 | pytest 命令前缀（可含 `-m pytest` 等） |
| `MUTATION_MIN` | 达标阈值，默认 80 |
| `MUTATION_SCOPE` | `changed`（默认，仅变异改动行）/ `all`（全文件） |
| `PYTHON_BIN` | 应用变异所用的解释器，默认 `python3` |

## 3. 输出与计分口径

- 输出：`杀死 K / 存活 S / 有效变异 A / 语法破坏跳过 X` + `MUTATION_SCORE=<K/A*100>`；退出码 `0` 达标、`1` 不达标、`2` 环境错误。
- **不计分位点（skip）**：注释、字符串、f-string 整段、类型注解区域（参数/返回/变量注解）、返回箭头 `->`。
- **不计分变异**：变异后无法 `ast.parse` 的"语法破坏型"变异（单独计数，不进分母）。
- 计分不变式：`A + X ≤ 规则数`，且 `X` 中的每一项都不可能是"测试有效性"证据。

## 4. 已知根因（本轮实测）

1. 仓库副本偏移量重复计换行：`sum(len(l)+1 for l in src.splitlines(True)…)`，而 `splitlines(True)` 已保留 `\n` → 排除区间整体后移 `(row-1)` 字符，注释与 f-string 位点实际未被排除（**与 Python 版本无关**，CI 3.11 同样命中）。
2. Python ≥3.12 把 f-string 拆成 `FSTRING_START/MIDDLE/END`，仅匹配 `STRING` 会漏排除 f-string 字面量。
3. `->` 被 `>` 规则命中后变成 `-<`，语法不成立却被计为"杀死"，虚高分数。

## 4b. F-23b：测试文件不参与"同名测试"配对（同一环内一并修复）

- 现象（实测）：PR #27 的 `governance` run `37449296630` = failure，日志为
  `❌ 改动文件 test_p03_metrics 无对应测试,变异门禁无法验证(L0/L2 fail-closed)`。
- 根因：`quality-gate.sh` 把 `tests/test_*.py` 也当作"需要同名测试的 py 文件"，
  而它们的同名测试永远是 `tests/test_test_*.py`（不存在）→ 在 PR 事件（`MUTATION_STRICT=1`，
  即真正的合入拦截点）必然 fail-closed → "只改测试"这一最常见合法改动必然红灯。
- 修复：`test_*`/`*_test` 基名不参与配对，且 fail-closed 只在**存在非测试 py 文件**时触发；
  改动的 py 全是测试文件时打印说明后跳过变异门禁（不空转、不误报）。
- 反削弱控制：源码文件缺同名测试仍然 fail-closed（对抗套件用例7 双向断言，PASS=7）。

## 5. 边界条件

- 源文件不在 git 仓库内 / 无改动行 → 回退全文件（不得空转成 `N/A`）。
- 文件不含任何可变异位点 → `MUTATION_SCORE=N/A`（退出码 0，由上游按需判读）。
- `PYTHON_BIN` 指向的解释器只需标准库（不需 numpy/pytest）。

## 6. 异常场景

- 基准测试不通过 → 退出码 2（先修测试）。
- 找不到 `mutation-check.sh` → 由 `quality-gate.sh` 按等级 fail-closed / warn。
- f-string 跨行、嵌套引号 → 以 `FSTRING_START…FSTRING_END` 配对区间整段排除。

## 7. 安全与治理约束

- 不得改动 `risk-classify.sh` 的定级口径；不得改 push 事件的硬门禁口径来换绿灯。
- 计分口径变更必须附回归测试（`scripts/ai-verify/tests/test-mutation-skip.sh`）并接入 CI。
- 仓库副本与 kit 副本的口径分叉属 J3，需负责人授权后单独同步（本环不动共享 kit）。

## 8. 必须覆盖的测试用例

| 用例 | 期望 |
|---|---|
| 只含注释/文档串/f-string/注解位点的样本 | 有效变异 = 1（仅 `value + 1` 的数值变异） |
| 含真实比较/布尔位点的样本 | 有效变异 = 3 |
| `a << 2` 样本（`<` 变异破坏语法） | 有效变异 = 1 且 语法破坏跳过 = 1 |
| `p03_metrics.py` + `tests/test_p03_metrics.py` | ≥80%（本环补 4 条边界/契约测试后实测 100%） |
| 3.11 / 3.13 变异器 | 同一文件同分数（本环实测一致） |
| 只改 `tests/test_*.py`（MUTATION_STRICT=1） | 不再 fail-closed；源码文件缺同名测试仍 fail-closed（对抗套件用例7） |

## 9. 验证命令

```bash
cd "/Users/mac/qoder m5pro/mci-world-model"
bash scripts/ai-verify/tests/test-mutation-skip.sh
bash scripts/ai-verify/tests/test-anti-evasion.sh
PYTHONPATH="$PWD:$PWD/src:../su-memory-sdk/src" .venv/bin/python -m pytest tests/test_p03_metrics.py -q
PYTHONPATH="$PWD:$PWD/src:../su-memory-sdk/src" PYTHON_BIN=.venv/bin/python \
  bash scripts/ai-verify/mutation-check.sh benchmarks/real_world/p03_metrics.py \
  tests/test_p03_metrics.py ".venv/bin/python -m pytest --override-ini=addopts="
```

## 10. 弃选方案

| 方案 | 弃选原因 |
|---|---|
| 把 `MUTATION_MIN` 从 80 降到 44 | 用放松阈值换绿灯，掩盖真实盲区（4 处） |
| 只在 push 事件把变异门禁降级为警告 | F-22 已把 push 定位为"合并后审计"，但分数本身可修；降级会永久失去信号 |
| 给 `-> float \\| None` 注解写断言凑分 | 注解在 `from __future__ import annotations` 下不参与运行，属凑分而非覆盖真实行为 |
| 直接删除 f-string 里的 `>=` 文案 | 掩盖工具缺陷，且下次仍会复发（红线：不修根因） |

## 11. 台账回写

| 条目 | 内容 | 验证证据（本地实测） |
|---|---|---|
| `OODA-20261006-046`（J2 工具口径） | 本 PR 的 `mutation-check.sh` + `test-mutation-skip.sh` + CI 接入 | skip_only 样本：旧副本有效变异 4 → 新副本 1；3.11 与 3.13 分数一致（均 1）；3 项回归断言通过；`p03_metrics.py` 10/10 = 100% |
| `OODA-20261006-047`（J1 边界测试） | 本 PR 的 4 条边界/契约测试 | `tests/test_p03_metrics.py` 24 passed；变异 10/10 = 100% ≥ 80% |
| `OODA-20261006-048`（J3 双副本分叉） | 本环只同步"仓库副本"，共享 kit 未改（需负责人授权） | 两份脚本仍不同源：kit 缺 ① f-string 整段排除 ② 语法破坏型不计分 |

> 台账 046/047 已在 PR #26 的提交 `34dfca4` 中以 `open` 建条；本环的验证证据在 PR #26 合并后回填为 `written/verified`
> （避免与 #26 冲突而在两处分叉编辑同一行）。
