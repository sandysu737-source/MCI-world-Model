# 架构设计：治理门禁 Stage 2

> 日期：2026-09-17 ｜ 风险等级：L2 ｜ 审核人：负责人本人

## 1. 模块职责
本阶段只收口 AI 工程化守卫本身：防止本地环境劫持、伪造工具、伪造 CI、token 偷换、危险代码样例绕过和 `nosec` 无痕豁免。

## 2. 模块拆分
| 组件 | 职责 |
|---|---|
| `ai-guard.sh` | 暂存清单收集、风险路由、L2 确认与重试控制 |
| `quality-gate.sh` | 静态门禁、工具完整性、SQL/会话、`nosec`、覆盖率、复杂度与变异测试 |
| `review-token.sh` | L2 token 签发与暂存 hash 绑定校验 |
| `risk-classify.sh` | 按路径、符号、文档和配置语义定级 |
| `hook-integrity.sh` | 校验入库 hook 副本与 hash 记录一致 |
| `tests/test-anti-evasion.sh` | 对抗回归套件，锁定六个攻击面 |
| `.github/workflows/governance.yml` | 服务端强制对抗测试、hook 完整性与审计留痕 |

## 3. 依赖关系
```text
ai-guard.sh → risk-classify.sh → quality-gate.sh
                 ↓
review-token.sh（L2 可选凭证）
CI governance → test-anti-evasion.sh → hook-integrity.sh → quality-gate.sh
```
- 不依赖项目业务模块、数据库、缓存或外部服务。
- 禁止业务代码反向导入或调用治理脚本内部函数。

## 4. 接口设计
| 接口 | 输入 | 输出 |
|---|---|---|
| `ai-guard.sh` | git 暂存区 | 退出码 + `.ai-governance/reports/` |
| `quality-gate.sh <L0|L1|L2> [files...]` | 等级与暂存文件 | 退出码 + 报告 |
| `review-token.sh issue ...` | reviewer 与暂存状态 | token |
| `review-token.sh verify ...` | token 与当前暂存状态 | 有效 / invalid |
| `test-anti-evasion.sh` | 无 | `PASS/FAIL/SKIP` 汇总 |

## 5. 数据层设计
- 门禁报告留在 `.ai-governance/reports/`，不进入运行时数据面。
- 重试计数迁移到用户目录，避免项目内状态被篡改后重置。
- `nosec` 豁免登记在 `.ai-governance/nosec-allowlist.tsv`，字段为工单、路径、登记人和原因。

## 6. 安全设计
1. `AI_ENG_KIT` 本地一律失效，防止假 kit 劫持。
2. 所有外部工具二进制必须可解析且不得是 `true` / `false`。
3. `AI_REVIEW_CI=1` 必须同时存在真实 `GITHUB_ACTIONS=true`。
4. L2 token 绑定分支、commit、暂存 hash 与 reviewer。
5. 裸 `nosec` 拦截，合法豁免必须登记且路径匹配。
6. SQL 拼接、Session 泄漏、空 catch 与危险 import 采用 AST/模式双检。

## 7. 可扩展预留点
1. 后续可在对抗套件中新增新攻击样例，不改变主门禁接口。
2. 后续可将 L0/L2 变异测试阈值放入项目配置。
3. 后续可扩展多项目共享治理 kit，但必须保留服务端 hash 校验。

## 8. 行为保持策略
- L1 文档仍按描述性文本定级，不因提及安全术语误升 L2。
- 文档疑似真实密钥仍强制 L2。
- 本地全量测试与 CI fallback 行为一致；显式降低强度只用于诊断。

## 9. 风险与规避
| 风险 | 规避 |
|---|---|
| 本地改环境变量绕过 | 固化 kit 路径并校验 CI 指纹 |
| 工具替换导致假通过 | 工具路径与真实二进制完整性校验 |
| token 被偷换 | token 绑定暂存 hash |
| 豁免滥用 | `nosec` 必须绑定工单与白名单 |
| 门禁脚本自身被绕过 | 对抗套件随 CI 强制运行 |

## 10. 验收标准
1. `test-anti-evasion.sh` 六项攻击全部拦截。
2. `hook-integrity.sh` 通过或按契约跳过。
3. `bash scripts/check.sh` 全绿。
4. PR Gate 1–6 与 governance 检查全绿。
5. 台账记录 PR 证据并经负责人复核。

## 审核结论
- [x] 职责单一
- [x] 无跨层
- [x] 依赖方向单向
- [x] 兼容策略完整
- [x] 安全点齐全
- 审核人：负责人本人
- 批准：已批准收口为独立 PR
