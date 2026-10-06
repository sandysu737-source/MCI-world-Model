# 架构评审：治理加固环 M2（L：PR 侧定级 + 变异映射）

> 评审对象：`.ai-workflow/2026-10-07-governance-l-testmap-structured-requirement.md`
> 风险等级：L2 ｜ 评审人：负责人（用户本人，单人模式由所有者显式授权替代第二审阅人）

## 1. 现状（证据）
- L：`scripts/ai-verify/risk-classify.sh:29` 用 `git rev-parse --abbrev-ref HEAD`；PR 事件 checkout 为 detached HEAD → `PROD_BRANCH=0`。同一提交两次 run 对照：PR run `37482929311` = `L1 ... 普通业务代码默认定级` ↔ push run `37487653909` = `L2 ... 生产分支默认保守定级`。
- 变异门禁前置条件：`scripts/ai-verify/quality-gate.sh:520-545` 要求每个改动 py 文件存在 `tests/**/test_<模块>.py`，否则 L0/L2 严格模式 `fail-closed`。
- 本仓实测（M1 需求 §13）：`src/adapters/benchmarks` 278 个 py 文件，275 个无同名测试 → 直接修 L 会让业务源码 PR 结构性红灯。

## 2. 方案
1. `risk-classify.sh` 支持 `RISK_BRANCH` 覆盖；detached 且未设时打警告（可追溯）。
2. workflow PR 事件传 `RISK_BRANCH=${{ github.base_ref }}`（push 传 `ref_name`）。
3. 新增 `test-map-gen.py`：从覆盖率上下文推导"源文件 → 覆盖它的 top-1 测试文件"，产出 `.ai-governance/test-map.tsv` 入库。
4. `quality-gate.sh` 变异门禁解析顺序：映射 → 同名 → fail-closed；日志标注来源。

## 3. 影响面
| 面 | 影响 |
|---|---|
| CI 定级 | 目标为 main 的 PR 由 L1 → L2：覆盖率仍硬门禁（原本已 full），变异门禁由"跳过"变为"按映射执行"，并新增 L2 审批步骤（需所有者当日指纹） |
| 本地 | 无行为变化（本地非 detached，且 `RISK_BRANCH` 可选） |
| 运行时 | 无影响 |
| 回滚 | 三处单文件 revert；映射文件可删除（缺省回退同名启发式） |
| 兼容性 | `risk-classify` 输出格式不变；阈值不变 |

## 4. 风险与缓解
| 风险 | 等级 | 缓解 |
|---|---|---|
| L2 生效后 L2 审批步骤成为每个业务 PR 的必经点（人工成本↑） | 中 | 这正是 F-2 的设计意图（合入前审批）；指纹当日有效、可复制；失败信息给出格式模板 |
| 未覆盖文件在新口径下 fail-closed，阻塞 PR | 中 | 生成器输出"未覆盖文件清单"，本环先量化；对确无测试的改动文件，正确处置是补测试（fail-closed 的价值所在），不得为绿灯放宽 |
| 映射漂移（测试改名/拆分） | 中 | 文件头记录生成日期 + commit sha；CI 里映射目标不存在 → 视为无映射并回退同名，且日志可见 |
| 变异门禁耗时上升（每个改动文件都要跑一次变异） | 中 | 单文件的 top-1 测试 + `-x -q`；实测耗时写入验收证据 |

## 5. 结论
方案把"不可满足的严格口径"换成"可满足且不可绕过的严格口径"，符合失败关闭原则；弃选方案（手工映射/目录级映射/降级为警告）均已在需求 §11 记录。同意按 §12 验收。
