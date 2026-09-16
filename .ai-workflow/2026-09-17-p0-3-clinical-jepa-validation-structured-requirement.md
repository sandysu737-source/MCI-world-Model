# 结构化开发需求：P0-3 真实生理时序 JEPA 验证

> 日期：2026-09-17 ｜ 负责人：sandysu737-source ｜ 执行人：Codex

## 1. 业务目标
在授权可用的真实生理时序数据上，验证 `JEPAClinicalBridge` 相对基线预测器是否具有可复现的预测优势；数据不可用时，只能输出 `not_run` 或降级实验，不得声称 P0-3 通过。

## 2. 输入参数定义
| 名称 | 类型 | 必填 | 默认 | 校验规则 |
|---|---|---|---|---|
| 数据清单 | `Path` | 是 | 无 | 指向仓库外本地 JSON manifest，包含来源、版本、授权状态、文件 SHA-256、时间范围、患者数 |
| 数据文件 | `Path` | 条件 | 无 | 仅 `source=mimic` 时必填；支持 `.csv` / `.csv.gz` / `.parquet` |
| 数据源 | `str` | 是 | 无 | 枚举：`mimic`、`degraded_synthetic`；默认不启用降级 |
| 变量映射 | `dict[str, str]` | 是 | 无 | 必须覆盖 `hr`、`sbp`、`dbp`、`spo2`、`rr`、`temp`、`gcs` |
| 历史窗口 | `int` | 是 | `12` | `>= 2`，对应 1 小时历史 |
| 预测步长 | `int` | 是 | `3` | `>= 1`，对应 15 分钟前瞻 |
| 切分策略 | `str` | 是 | `patient_holdout` | 只允许按患者/受试者切分，禁止随机窗口切分 |
| 随机种子 | `int` | 是 | `42` | 全流程唯一；重复实验不得改 seed |
| 输出目录 | `Path` | 是 | 无 | 必须为仓库外结果目录，禁止写入患者级数据 |

## 3. 输出数据结构
`report.json` 只允许聚合结果：
```json
{
  "status": "passed|failed|not_run|degraded",
  "source": "mimic|degraded_synthetic",
  "dataset_manifest": {"version": "", "sha256": "", "n_subjects": 0, "n_windows": 0},
  "split": {"strategy": "patient_holdout", "n_train_subjects": 0, "n_test_subjects": 0},
  "models": {
    "jepa_clinical_bridge": {"mae": {}, "relative_mae": {}, "direction_accuracy": {}, "constraint_violation_rate": {}, "ci95": {}},
    "clinical_dynamics_baseline": {"mae": {}, "relative_mae": {}, "direction_accuracy": {}, "constraint_violation_rate": {}, "ci95": {}}
  },
  "statistical_test": {"method": "paired_bootstrap", "p_value": 0.0, "significant": false},
  "seed": 42,
  "limitations": []
}
```
`report.md` 只呈现聚合表格与失败模式类别，不得包含 `subject_id`、时间戳原文或原始观测值。

## 4. 依赖资源
| 类型 | 内容 |
|---|---|
| 模型 | `JEPAClinicalBridge` |
| 基线 | `ClinicalDynamicsPredictor`；可选 `LinearPhysicsPredictor` 作为弱基线 |
| 数据 | MIMIC-III Chartevents 派生宽表；仓库内不保存患者数据 |
| 降级数据 | 仅允许明确标注为 `degraded_synthetic` 的生理仿真数据 |

## 5. 业务流程步骤
1. 数据准入审计：确认授权、文件路径、schema、SHA-256、时间范围和变量覆盖。
2. 清洗与窗口化：处理缺失、重复、乱序和异常值，生成 `(N, 12, 7)` 输入与 `(N, 3, 7)` 目标。
3. 患者级切分：先按 `subject_id` 分训练/测试，再做窗口切片，防止跨窗泄漏。
4. 同预算训练：JEPA 与基线使用相同样本、相同 epoch 预算、相同 seed。
5. 多步评估：输出 1/3 步 MAE、relative MAE、direction accuracy、constraint violation rate 与 95% CI。
6. 显著性检验：按患者或窗口配对 bootstrap；无法配对时必须声明 `not_applicable`。
7. 报告落盘：`report.json` 与 `report.md` 只含聚合结果与 limitations。

## 6. 边界条件
- [x] 空值：缺失值必须显式填补或剔除，并统计填补比例。
- [x] 上下限：超出生理可行范围的输入记录数量，预测输出 clip 并统计违反率。
- [x] 重复：同一患者同一时间的重复观测必须去重或聚合。
- [x] 乱序：时间戳必须排序；乱序比例超过阈值时失败。
- [x] 患者泄漏：同一 `subject_id` 不得同时出现在训练和测试。
- [x] 短序列：不足窗口长度的患者剔除并计数。

## 7. 异常场景
1. manifest 缺失、授权字段为空或 SHA-256 不匹配：返回 `not_run`。
2. 文件 schema 缺列、时间戳不可解析或变量覆盖率不足：返回 `failed`。
3. 训练样本不足、结果出现 NaN/Inf 或模型未收敛：返回 `failed`，不得写达标结论。
4. MIMIC 权限不可用：不得自动转成 Sachs/Tuebingen 冒充 P0-3；只能另开 `degraded_synthetic` 实验。

## 8. 性能约束
- 单次验证目标不超过 60 分钟。
- 大文件必须分块或按患者流式读取，禁止一次性加载全部 Chartevents。
- 训练预算在 JEPA 与基线间保持一致。

## 9. 安全约束
- 禁止提交、缓存或打印患者级数据。
- 禁止在代码或日志中出现数据库密码、令牌和连接串。
- 数据文件只允许通过本地路径读取；不新增网络数据拉取。
- 报告只保存聚合指标、哈希和样本计数。

## 10. 必须覆盖测试用例
1. 正向：合成小样本能完成窗口化、训练、评估和聚合报告。
2. 边界：缺失值、重复时间、乱序时间、超范围值和短序列均被正确统计。
3. 异常：manifest 缺失、schema 缺列、患者泄漏和 NaN 指标均 fail-closed。
4. 兼容：旧 `scripts/jepa_mimic_validation.py` 不得被继续用作 P0-3 通过证据。
