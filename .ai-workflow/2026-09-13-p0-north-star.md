# P0-1 North Star Benchmark + CI 分层结构化需求

## 1. 业务目标
建立统一可追溯的 North Star 基准报告，并用 CI 分层区分 PR 快检与夜间全量，防止结构发现、因果方向与证据链质量长期只靠主观判断。

## 2. 输入参数定义
| 字段 | 类型 | 必填 | 默认 | 校验规则 |
| --- | --- | --- | --- | --- |
| suite | str | 否 | smoke | 只允许 smoke/full |
| output | Path | 否 | docs/north-star-result.json | 可写路径 |

## 3. 输出数据结构
`BenchmarkReport` 输出 `schema_version`、`suite`、`commit`、`status` 和 `metrics`。每个 metric 包含 benchmark、metric、value、threshold、comparison、passed、samples、source、seed、config。

## 4. 依赖资源
Tübingen-like、BNLearn-style、MIMIC-like 三个现有真实基准入口；不新增外部网络依赖。

## 5. 业务流程步骤
1. CLI 按声明式规格延迟执行指标。
2. 聚合统一 JSON 报告并逐项判定阈值。
3. PR 运行 smoke，nightly 运行 full。
4. governance 对非 Python 生产代码使用 changed pytest scope。

## 6. 边界条件
- [x] suite 只允许声明值
- [x] 禁止重复 metric key
- [x] 禁止空指标集合
- [x] 非有限指标值直接失败

## 7. 异常场景
未知 suite、重复 key、空规格、报告写入失败、指标阈值不达标均以可观察错误退出，不伪装通过。

## 8. 性能约束
PR smoke 目标控制在 20 分钟内；当前实测 smoke 约 3 秒。

## 9. 安全约束
报告只包含指标、配置、来源、seed 与 commit，不输出训练数据原文或敏感术语。

## 10. 必须覆盖测试用例
1. 正向：smoke 报告通过且 schema 完整。
2. 边界：非有限值不通过，报告可序列化。
3. 异常：重复 key、空规格、未知 suite 拒绝执行。
