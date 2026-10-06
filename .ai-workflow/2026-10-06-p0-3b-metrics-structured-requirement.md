# 结构化开发需求：P0-3b 评估器（p03_metrics）

> 日期：2026-10-06 ｜ 风险等级：L1 ｜ 负责人：sandysu737-source ｜ 执行人：Codex
> 上游文档：`.ai-workflow/2026-09-17-p0-3-clinical-jepa-validation-{structured-requirement,architecture}.md`
> 增量定位：P0-3a（manifest + 时序管道）已合并 `555518e`；本次只做 §2 表中的“评估器”一片。

## 1. 业务目标
对 `P03WindowSplit` 上的预测输出计算可复现的聚合指标与配对显著性检验，使 P0-3 的“JEPA 是否优于基线”结论有量化依据；数据不可用时只能产出 `not_run`，禁止捏造结论。

## 2. 输入参数定义
| 字段 | 类型 | 必填 | 默认 | 校验规则 |
|---|---|---|---|---|
| model / jepa / baseline | 对象 | 是 | 无 | 必须提供可调用 `predict(inputs)`；输入 `(N,H,7)` → 输出 `(N,F,7)` |
| split | `P03WindowSplit` | 是 | 无 | 三阶数组、第 3 维 =7、测试输入与目标窗口数一致、全为有限值 |
| name | `str` | 否 | `"model"` | 非空 |
| seed | `int` | 否 | `42` | 与 manifest.seed 对齐；bootstrap 唯一随机源 |
| n_resamples | `int` | 否 | `2000` | `>= 2` |
| alpha | `float` | 否 | `0.05` | `0 < alpha < 1` |
| source | `str` | 否 | `None` | `mimic` / `degraded_synthetic`；`degraded_synthetic` 时报告状态降级 |

## 3. 输出数据结构
- `P03ModelMetrics.to_dict()`：`model_name`、`status`、`failure_reason`、`n_windows`、`history_window`、`forecast_horizon`、`vital_keys`、`mae`、`mae_per_step`、`relative_mae`、`relative_mae_per_step`、`direction_accuracy`、`direction_accuracy_per_step`、`constraint_violation_rate`、`constraint_violation_rate_per_step`、`ci95`、`seed`、`limits`。
- `P03Report.to_dict()`：`status`、`metric`、`source`、`split`（患者/窗口计数）、`models`、`statistical_test`、`seed`、`limitations`。
- 指标键一律使用 7 个短名：`hr/sbp/dbp/spo2/rr/temp/gcs`（顺序 = `VITAL_KEYS`）。
- 未定义值（NaN）序列化为 `null`，不得回落成 0。

## 4. 指标口径（本次固化）
| 指标 | 定义 |
|---|---|
| MAE | 逐（步长, 体征）平均绝对误差；另给跨步聚合值 |
| relative MAE | `MAE / mean(|target|)`；分母 `< 1e-9` 视为未定义 → `null` |
| direction accuracy | 以输入窗口末步为参照的升降方向命中率；`|Δtarget| <= 1e-9` 的样本剔除；预测零变化对非零变化记为未命中；无有效样本 → `null` |
| constraint violation rate | 预测值超出生理可行范围（`VITAL_FEASIBLE_RANGES`）的比例，基于 clip 之前的原始预测；非有限值计为违反 |
| 95% CI | 以窗口为重采样单位的配对 bootstrap 百分位法（`seed` 固定） |
| 显著性 | `mean_difference = baseline_mae - jepa_mae`（>0 表示 JEPA 更优）；95% CI 不跨 0 判显著；p 值为百分位近似 |

## 5. 业务流程步骤
1. 校验 split 与选项 → 2. 调用 `predict()` 并校验输出形状/有限性 → 3. 计算四类指标 → 4. bootstrap CI →
5. `compare_models` 在同一 split 上评估两个模型，按窗口配对求差并做 bootstrap → 6. 汇合 `P03Report`。

## 6. 边界条件
- [x] 空值：目标含 NaN/Inf → 契约错误；预测含 NaN/Inf → `status=failed`（fail-closed，不产出达标结论）
- [x] 上下限：超出 `VITAL_FEASIBLE_RANGES` 计入违反率，不在评估器内 clip
- [x] 重复：同一窗口重复计权（配对 bootstrap 已按窗口重采样）
- [x] 空集：测试窗口为 0 → `status=not_run`，全部指标为 `null`
- [x] 短序列/泄漏：由 P0-3a 管道拦截，本模块只校验形状

## 7. 异常场景
1. 模型无 `predict()` 或输出形状不符 → `P03MetricsError`（编程错误，直接抛）
2. `n_resamples < 2` 或 `alpha` 越界 → `P03MetricsError`
3. 预测含非有限值 → `status=failed` + `failure_reason`，不抛异常
4. `source=degraded_synthetic` 且 JEPA 显著更优 → 报告状态 `degraded`（不得写 `passed`）
5. `source` 未声明为 `mimic` 时即使 JEPA 显著更优也不判定 `passed`（防越权结论）

## 8. 性能约束
- 单次评估不得对 `(N, H, 7)` 之外的数组做全量物化；bootstrap 索引分块生成（内存上限约 5e6 索引/块）
- 默认 2000 次重采样在 1e5 窗口量级下应 < 5s

## 9. 安全约束
- 输出只含聚合指标与计数；禁止 `subject_id`、时间戳原文、原始观测进入 `to_dict()`
- 不新增依赖（仅 numpy）；不读取文件、不访问网络、不直连数据库

## 10. 必须覆盖测试用例
1. 正向：完美预测 → MAE=0、方向命中率=1、违反率=0；固定偏移 → MAE 等于解析值
2. 边界：空测试集 → `not_run`；目标无变化样本被剔除；rel MAE 分母过小 → `null`
3. 异常：形状不符/无 `predict` → 抛错；NaN 预测 → `failed`；CI 与 bootstrap 可复现
4. 安全：`to_dict()` 无患者标识；`degraded_synthetic` 状态降级；配对 bootstrap 方向正确

## 11. 验证命令
```bash
# 定向回归
PYTHONPATH="../su-memory-sdk/src:$PWD/src" .venv/bin/python -m pytest tests/test_p03_metrics.py tests/test_p03_data_foundation.py -q
# 仓库级门禁（ruff + format + mypy + pytest + ai-guard）
bash scripts/check.sh
```
本地 pre-commit 的 L1 覆盖率默认阈值（90%）高于仓库实测基线（78%），与 CI 校准值（`COV_CORE_THRESHOLD=77` / `COV_BR_THRESHOLD=61`）不一致；
本地提交按 CI 同源阈值注入，最终口径以 CI governance 为准。
