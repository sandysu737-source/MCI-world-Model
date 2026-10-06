# 结构化开发需求：治理加固环 M2（L：PR 侧定级口径 + 变异测试映射）

> 来源：台账 `OODA-20261006-051`（治理 L）；M1 需求文档 §13（影响面证据）。
> 风险等级：L2（改动 `.github/workflows/` + `scripts/ai-verify/`）。

## 1. 业务目标
① 修复 PR 事件下 `risk-classify` 分支探测失真（业务代码 L2→L1，PR 侧拦截弱于 push 侧）；
② 把变异门禁的"同名测试"启发式替换为**覆盖率上下文推导的测试映射**，使 L2 严格模式在 PR 侧既不可绕过、也不再结构性误报。

## 2. 输入参数定义

| 字段 | 类型 | 必填 | 默认 | 校验规则 |
|---|---|---|---|---|
| RISK_BRANCH | string | 否 | 空 | 非空时覆盖 `git rev-parse --abbrev-ref HEAD`；CI PR 事件传 `github.base_ref` |
| `.ai-governance/test-map.tsv` | TSV | 否（缺失即回退） | 无 | 每行 `<repo 相对源文件>\t<测试目标>\t<覆盖行数>`；`#` 开头为注释 |
| CHANGED_FILES | 文件列表 | 是 | — | 由 workflow 生成（同 M1） |

## 3. 输出数据结构
- `risk-classify` 输出格式不变（`LEVEL\tpath\treason` + `MAX_LEVEL`）；detached HEAD 且未设 `RISK_BRANCH` 时向 stderr 打警告（可追溯）
- 变异门禁新增解析日志：`[mut] 测试解析: <源文件> -> <测试目标>（来源: map|同名|无）`

## 4. 依赖资源
`coverage` ≥7（`--cov-context=test` + `coverage json --show-contexts`）、pytest、`gh`、GitHub Actions。

## 5. 业务流程步骤
1. 生成映射（三步，可复现）：
   ```bash
   PYTHONPATH="$PWD:$PWD/src:../su-memory-sdk/src" .venv/bin/pytest -m "not slow" \
     --cov=mci_world_model --cov=benchmarks --cov=adapters --cov-context=test -q --cov-report=
   .venv/bin/python -m coverage json --show-contexts -o /tmp/cov_ctx.json
   .venv/bin/python scripts/ai-verify/test-map-gen.py /tmp/cov_ctx.json .ai-governance/test-map.tsv \
     --root . --sha "$(git rev-parse --short HEAD)"
   ```
   注：额外加 `--cov=benchmarks --cov=adapters` 只为让研究/适配层文件进入映射，**不改变** CI 覆盖率阈值口径（`source=mci_world_model`），不得据此判断覆盖率（该次运行总覆盖率 69.82% 属预期，见 §13）。
2. PR 事件：workflow 传 `RISK_BRANCH=<base_ref>` → 目标为 `main` 的 PR 与 push 同口径（业务代码 L2）。
3. 变异门禁：解析顺序 = 映射表 → 同名启发式 → **fail-closed**（L0/L2 严格模式）。

## 6. 边界条件
- [x] 空值：映射缺失/空 → 回退同名启发式（不静默放行）
- [x] 上下限：映射条目数不做上下限，但生成器只保留"覆盖该文件行数最多"的 top-1 测试，控制变异耗时
- [x] 重复：同一源文件多条映射 → 取第一条并在生成时去重（生成器保证唯一）
- [x] 过期：映射与代码漂移 → 生成命令写入文件头（日期 + 提交 sha），漂移可查

## 7. 异常场景
| 场景 | 期望行为 |
|---|---|
| 改动文件既无映射又无同名测试 | L0/L2 严格模式 fail-closed，提示"补映射或补测试" |
| 映射中的测试目标不存在 | 视为无映射（回退同名），并在日志中 `来源: 无` |
| PR 事件未传 `RISK_BRANCH` 且 detached | stderr 警告 + 按非生产分支定级（不静默） |
| 映射文件被篡改以放行 | 映射只影响"用哪个测试做变异"，不放行任何门禁；未覆盖文件仍 fail-closed |

## 8. 性能约束
- 映射仅按文件解析（awk 单次遍历）；变异门禁单文件耗时 ≈ 单测试文件耗时（≤1 min）

## 9. 安全约束
- 门禁强度不得因映射而下降：映射只替换"测试选择"，阈值（`MUTATION_MIN=80`、覆盖率 77/61）不变
- 不引入新依赖、不访问网络

## 10. 必须覆盖测试用例
1. 正向：`RISK_BRANCH=main` + `benchmarks/**/*.py` → `MAX_LEVEL=L2`
2. 边界：`RISK_BRANCH` 未设 + PR 事件 detached HEAD → L1 + stderr 警告
3. 异常：映射表命中 → 用映射测试；映射缺失 → 同名回退；两者皆无 → fail-closed
4. 异常：映射目标文件不存在 → 按"无映射"处理

## 11. 弃选方案与理由
| 方案 | 弃选理由 |
|---|---|
| 手工维护 275 条映射 | 不可持续、易漂移；覆盖率上下文推导可复现 |
| 目录级粗粒度映射（`src/**` → `tests/`） | 单次变异要跑全量测试套件，耗时不可接受 |
| 直接把"无同名测试"降级为警告 | 违背 fail-closed；等于把 L2 严格模式变成形式 |
| 只在 push 侧保留 L2 | 与 F-2 设计意图相悖：审批是合入前拦截点 |

## 12. 验收标准
1. PR 改动 `benchmarks/**` 或 `src/**` 时 `MAX_LEVEL=L2`，且「L2 双人 approve 校验」步骤在 PR 事件执行。
2. 改动文件有映射 → 变异门禁按映射执行并通过（≥80%）；无映射且无同名测试 → fail-closed。
3. 映射生成命令可复现，文件头含日期与提交 sha。
4. `bash scripts/ai-verify/tests/*.sh` 全通过；CI 4-Gate + governance 全绿。

## 13. 本轮实测结果（2026-10-07）

| 项 | 实测 |
|---|---|
| 覆盖率运行（映射生成用） | `pytest -m "not slow"` → `4804 passed, 1 skipped`（790.99s）；`source=mci_world_model` 口径总覆盖率 **81.06%** ≥ 77 ✓ |
| 映射产出 | `.ai-governance/test-map.tsv`：**215 条映射**（src 205 / benchmarks 10 / adapters 0） |
| 受门禁约束的 py 文件 | 242（排除 `__init__.py` 与 `test_*.py`） |
| L 生效后的 fail-closed 集合 | **27**（src 10 / benchmarks 15 / adapters 2）——修 L 前为 275，降幅 89% |
| 回归测试 | `test-risk-branch.sh` 7/7；`test-mutation-map.sh` 5/5 |

**已知后果（不视为缺陷）**：27 个"无任何测试上下文且无同名测试"的文件在新口径下会 `fail-closed`（含 `adapters/mci_huan_bridge.py`、`adapters/zvec_bridge.py` —— 台账 `OODA-20261004-040` 已记录为未接线死代码）。正确处置是补测试或按死代码清理，**不得**通过放宽门禁或加豁免白名单绕过。
