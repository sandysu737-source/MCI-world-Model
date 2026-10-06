# 架构评审：治理加固环 M1（H + K）

> 评审对象：`.ai-workflow/2026-10-06-governance-hk-structured-requirement.md`
> 风险等级：L2 ｜ 评审人：负责人（用户本人，单人模式由所有者显式授权替代第二审阅人）

## 1. 现状（证据）
- H：`.github/workflows/governance.yml:66` 仅 `^(src|adapters)/.*\.py$` 命中才 `scope=full, hard=1`；`benchmarks/**` 落入 `scope=changed, hard=0`（warn 通道）。台账 `OODA-20261006-044`。
- K：`.github/workflows/governance.yml:25` 为固定字面量 `OWNER-APPROVED:2026-10-04:PR22`，`:129` 仅做 `grep -Fqx` 整行匹配，不校验 PR 号与日期 → 可跨 PR 重放。台账 `OODA-20261006-049`。

## 2. 方案
1. H：判定正则扩为 `^(src|adapters|benchmarks)/.*\.py$`，不改其它步骤与阈值口径。
2. K：删除固定字面量；新增 `GOV_OWNER_APPROVAL_PREFIX`，期望指纹在步骤内由 `PR 号 + TZ=Asia/Shanghai 当日日期` 拼接；失败信息只打印格式模板。

## 3. 影响面
| 面 | 影响 |
|---|---|
| CI 时长 | `benchmarks/**` PR 由 ≈3 min 升至 ≈15 min（等同 src PR） |
| 合并流程 | L2 PR 必须由所有者填写当日指纹；旧指纹立即失效 |
| 运行时 | 无影响（workflow 与本地脚本不改业务代码） |
| 回滚 | 单文件 revert 即可；无数据/状态迁移 |
| 兼容性 | 不改变 `classify.txt` / gate 输出格式，不触碰 threshold 值 |

## 4. 风险与缓解
| 风险 | 等级 | 缓解 |
|---|---|---|
| benchmarks PR 因全量覆盖率硬门禁变红 | 中 | 本环 CI 实测覆盖率基线（77 行 / 61 分支）已在 F-20 校准；先在本环 PR 上实测 |
| 所有者忘记填当日指纹导致 L2 PR 卡住 | 中 | 失败信息打印格式模板；台账与 PR 模板提示当日字符串生成方式（`TZ=Asia/Shanghai date +%F`） |
| 期望指纹泄漏进日志 → 自满足 | 低 | 明确禁止打印期望值；失败信息只给模板 |
| 时区跨日导致"当天填的指纹第二天失效" | 低 | 固定 `Asia/Shanghai` 口径；跨日需重新填写（符合"当日授权"语义） |

## 5. 结论
方案最小改动、可回滚、与既有 F-2/F-20/F-22 语义一致；残留风险（能编辑 PR 正文者可写指纹）属单人模式固有限制，显式记录为已知限制。同意按结构化需求 §12 验收。
