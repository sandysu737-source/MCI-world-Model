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

## 14. 本轮补充：变异门禁的适用范围与一处新发现

### 14.1 适用范围收敛为"代码根目录"（本轮实测后补充）

`pytest testpaths=tests`，`scripts/**` 等治理工具不可能在同名/映射口径下提供 pytest 测试（本环新增的 `scripts/ai-verify/test-map-gen.py` 即为实例：它没有也无法有 `tests/test_test-map-gen.py`）。若不收敛范围，L 修好后本环自身与所有治理工具 PR 都会结构性 `fail-closed`。

故 `quality-gate.sh` 的变异门禁新增一条显式边界：**只有 `src/`、`adapters/`、`benchmarks/` 下的改动文件要求"可解析测试"**；其余（`scripts/**`、仓库根等）跳过并留痕：

```
[gate] 改动的 py 文件均非代码根目录(src/adapters/benchmarks),变异门禁不适用(治理工具由对抗套件验证)
```

治理工具的正确性由 `scripts/ai-verify/tests/test-*.sh` 对抗套件承担（本环新增 2 个测试文件、12 项断言）。这不是放宽代码根目录的 `fail-closed`：`src|adapters|benchmarks` 下"既无映射又无同名测试"的文件仍然红灯（本轮 27 个）。

### 14.2 新发现：无有效变异时门禁空转（登记为待开环，本环不修）

本地实测（本环 dry-run）：`quality-gate.sh L0 src/mci_world_model/_sys/_c1.py` → `[mut] 未产生有效变异` / `MUTATION_SCORE=N/A` → 退出码 0 → 门禁 **通过**。即"映射命中的文件若没有可变异的语法位点，变异门禁会静默空转"。

- 这是 `mutation-check.sh` 既有语义（`MUTATION_SCORE=N/A` 视为 0 个有效变异→放行），**不是** M2 引入；
- 与 M2 的组合效应：映射到"纯声明/类型文件"时门禁形同虚设；
- 处置建议（待开环评估，不在本环修，避免两个根因混在一个环里）：`N/A` 在 L0/L2 严格模式下应至少产出**显式警告并要求人工确认**，或改判 `fail-closed`（需先量化会新增多少红灯）。

## 15. 本环本地验证（dry-run 证据）

| 命令（本地、PATH 前置 .venv/bin） | 结果 |
|---|---|
| `quality-gate.sh L0 scripts/ai-verify/test-map-gen.py` | `改动的 py 文件均非代码根目录…变异门禁不适用` → 通过（治理工具不误红） |
| `quality-gate.sh L0 src/mci_world_model/_sys/_c1.py` | `测试解析: src/mci_world_model/_sys/_c1.py -> tests/test_foundation_types.py`；变异 `N/A`（见 §14.2） |
| `quality-gate.sh L2 benchmarks/real_world/p03_metrics.py` | `测试解析: … -> tests/test_p03_metrics.py`；`变异测试通过(p03_metrics: 100% ≥ 80%)`；bandit SAST 通过 |
