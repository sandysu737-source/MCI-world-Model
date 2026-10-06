# 结构化开发需求：治理加固环 M1（H + K）

> 来源：`.ai-workflow/2026-10-06-p0-3b-adversarial-audit.md` §5/§8 未闭环项 H、K；台账 `OODA-20261006-044`（H）、`OODA-20261006-049`（K）。
> 风险等级：L2（改动 `.github/workflows/` + `scripts/ai-verify/`，命中 L2_CONFIG_RE）。

## 1. 业务目标
把「CI 对 `benchmarks/**` 改动只警告不硬拦」与「L2 审批指纹可跨 PR 重放」两个治理缺口闭环，使 PR 侧门禁强度与 F-2 设计意图一致，且审批留痕不可复用。

## 2. 输入参数定义（变更面）

| 字段 | 类型 | 必填 | 默认 | 校验规则 |
|---|---|---|---|---|
| CHANGED_FILES | 文件列表 | 是 | 由 workflow 生成 | `git diff origin/<base_ref>...HEAD`（PR）或 `HEAD~1 HEAD`（push） |
| PYTEST_SCOPE | `full` / `changed` | 是 | `full` | 改动命中 `^(src\|adapters\|benchmarks)/.*\.py$` → `full` |
| COVERAGE_HARD_GATE | `0` / `1` | 是 | `1` | 同上命中 → `1`（硬阻断） |
| GOV_OWNER_APPROVAL_PREFIX | string | 是 | `OWNER-APPROVED` | 固定前缀，不含 PR 号与日期 |
| EXPECTED_MARKER | string | 仅 PR+L2 | 运行时拼接 | `OWNER-APPROVED:PR<PR号>:<Asia/Shanghai 当日 YYYY-MM-DD>` |

## 3. 输出数据结构
- gate 步骤：`scope`、`hard`、`mutation_strict` 输出不变（H 只扩判定范围）
- 审批步骤：`Owner explicit approval marker: true|false`；失败信息只打印**格式模板**，不打印期望值本体

## 4. 依赖资源
GitHub Actions（ubuntu-latest）、`gh` CLI、tzdata（`Asia/Shanghai`）、分支保护 status check `governance`。

## 5. 业务流程步骤
1. PR 改动含 `benchmarks/**/*.py` → `scope=full`、`hard=1` → quality-gate 跑全量 pytest + 覆盖率硬门禁（原为 warn）。
2. L2 PR → 审批步骤读取 PR 正文，要求存在**整行等于**当日动态指纹的内容。
3. 旧静态指纹（`OWNER-APPROVED:2026-10-04:PR22`）不再匹配任何 PR → 步骤失败。

## 6. 边界条件
- [x] 空值：`changed-files.txt` 为空 → 保持原有 else 分支，不误判
- [x] 时区：日期以 `TZ=Asia/Shanghai` 计算，避免 UTC 跨日导致的当日指纹失效
- [x] 重复：同一指纹在其它 PR（PR 号不同）或其它日期（日期不同）使用 → 不通过
- [x] 过期：昨日指纹 → 不通过
- [ ] 禁用：无

## 7. 异常场景
| 场景 | 期望行为 |
|---|---|
| 正文无指纹 | 审批步骤失败，打印格式模板（不打印期望值） |
| 正文残留旧静态指纹 | 失败；须替换为当日指纹 |
| push 事件 | 审批步骤不执行（F-22 既有语义，合并后审计留痕） |
| benchmarks 改动但全量 pytest 失败 | 硬阻断（H 的目标行为） |

## 8. 性能约束
- `benchmarks/**` PR 的 CI 时长上升到"全量 pytest + 覆盖率"量级（≈15 min），与 `src/**` PR 一致；不叠加额外门禁。

## 9. 安全约束
- 权限：审批指纹仅在 PR 事件、且 `github.event.pull_request.user.login == github.repository_owner` 时接受
- 防泄漏：期望指纹**不得**出现在日志、注释或文档中（避免"读日志→自满足"）
- 残留风险（显式记录）：能编辑 PR 正文者仍可写入指纹；GitHub 禁止作者自审，单人模式无更强机制，故保留为已知限制而非"技术上只有所有者可确认"
- 不涉及敏感数据与生产数据

## 10. 必须覆盖测试用例
1. 正向：当日正确指纹（PR 号 + 当日北京时间）→ `marker=true`
2. 边界：跨日（昨日）与跨 PR（他号）指纹 → `false`
3. 异常：无指纹 → 步骤失败；旧静态指纹 → 步骤失败
4. H：改动 `benchmarks/real_world/p03_metrics.py` → gate 输出 `scope=full`、`hard=1`

## 11. 弃选方案与理由
| 方案 | 弃选理由 |
|---|---|
| K：仅把固定字面量改成另一个固定字面量 | 不消除重放，只是换了个可复用字符串 |
| K：用 `GOV_REVIEW_SECRET` 做 HMAC 指纹 | 需要所有者持有并在本机生成；secret 在本地工具链可达时防伪性退化，且增加人工步骤；当前威胁模型（单人 + AI 辅助）下 PR 号+日期已能消除跨 PR/跨日重放 |
| K：依赖 GitHub Environment 必选审阅人 | 单人模式无第二人，等于死锁（F-2 已记录该原因） |
| H：只改 PR 模板强制标注"含警告通过" | 依赖人工自觉，非门禁；与"服务端不可绕过"原则冲突 |
| H：为 `benchmarks/**` 单独造一套覆盖率门禁 | 重复实现、口径分叉风险（同 J3 教训） |

## 12. 验收标准
1. 改动 `benchmarks/**` 的 PR：gate 输出 `scope=full`、`hard=1`，且 CI 实测通过（同一 commit 的 CI 运行 id 为证）。
2. 旧静态指纹在任一 PR 上不再使审批步骤通过（保留 run id 对照）。
3. 当日动态指纹可使 L2 审批步骤通过；跨 PR/跨日指纹不通过。
4. `bash scripts/ai-verify/tests/*.sh` 全通过；`scripts/check.sh` 全绿。
5. 合并后 main push governance = success。

## 13. 本次不纳入：治理 L（下一环 M2）与变异门禁映射

台账 `OODA-20261006-051`：PR 事件下 `scripts/ai-verify/risk-classify.sh:29` 取 detached `HEAD` → `PROD_BRANCH=0`，业务代码定级 L2→L1。

**为何本环不一起修（证据）**：修复 L 后，PR 事件下改动 `src/**` 的 PR 将升为 L2，而 L2 在 PR 事件会以 `MUTATION_STRICT=1` 进入变异门禁严格模式，其前置条件是"每个改动 py 文件存在同名测试 `tests/**/test_<模块>.py`"，否则 `fail-closed`。本仓实测（`find src adapters benchmarks -maxdepth 3 -name '*.py'`，排除 `__init__.py`）：

| 目录 | 文件数 | 缺同名测试 |
|---|---|---|
| src | 215 | 214 |
| adapters | 2 | 2 |
| benchmarks | 61 | 59 |
| 合计 | 278 | 275 |

即当前口径下 275/278 文件必然 `fail-closed`。直接修 L 会让"改动业务源码的 PR"变成结构性红灯（与 F-22 要消除的"红灯常态噪声"同源）。

**M2 建议方案**（待单独开环评审）：把"同名测试"启发式替换为**覆盖率上下文推导的测试映射**（`pytest --cov --cov-context=test` → `coverage json --show-contexts` 生成 `.ai-governance/test-map.tsv`：源文件 → 覆盖它的测试文件），仍然对"无任何测试覆盖的改动文件"`fail-closed`，但不再是不可满足的口径；同时修 L 的 `RISK_BRANCH`。验收：改动 `benchmarks/**` 或 `src/**` 的 PR 在 PR 事件下 `MAX_LEVEL=L2`，且变异门禁按映射执行而非一刀切红灯。
