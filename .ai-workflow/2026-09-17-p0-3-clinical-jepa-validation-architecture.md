# 架构设计：P0-3 真实生理时序 JEPA 验证

> 日期：2026-09-17 ｜ 风险等级：L1 ｜ 审核人：负责人本人

## 1. 模块职责
P0-3 只建立“数据准入 → 患者级窗口切分 → 同预算训练 → 聚合评估 → 可复现报告”的验证闭环。模型能力仍由 `JEPAClinicalBridge` 承担；本阶段不改 SDK 接口。

## 2. 模块拆分
| 组件 | 建议路径 | 职责 |
|---|---|---|
| 数据清单契约 | `benchmarks/real_world/p03_manifest.py` | 校验来源、授权、版本、SHA-256、变量映射 |
| 时序管道 | `benchmarks/real_world/p03_timeseries.py` | 读取宽表、清洗、排序、窗口化、患者级切分 |
| 评估器 | `benchmarks/real_world/p03_metrics.py` | 计算 MAE、relative MAE、direction accuracy、constraint violation 与 CI |
| 验证入口 | `scripts/p03_clinical_jepa_validation.py` | 编排模型、基线、统计检验与报告输出 |
| 报告 | 仓库外 `p03-report/report.json|.md` | 只保存聚合结果和 limitations |

旧 `scripts/jepa_mimic_validation.py` 当前使用 `ClinicalDynamicsPredictor` 与合成/半合成数据，不能继续作为 P0-3 的 JEPA/真实数据证据；后续可改名或归档，但不在 P0-3 首个 PR 混入。

## 3. 依赖关系（内部 + 外部）
```text
manifest contract → timeseries pipeline → split windows
                                          ├→ JEPAClinicalBridge
                                          ├→ ClinicalDynamicsPredictor baseline
                                          └→ evaluator → paired bootstrap → report
```
- 允许依赖：`numpy`、现有 SDK/benchmark 模块。
- 禁止依赖：数据库直连、网络下载、云存储 SDK。
- 禁止反向依赖：验证脚本不得修改 `JEPAClinicalBridge` 的模型契约。

## 4. 接口设计
```python
def load_manifest(path: Path) -> P03Manifest: ...
def load_windows(manifest: P03Manifest) -> P03WindowSplit: ...
def evaluate_model(model, split: P03WindowSplit) -> P03ModelMetrics: ...
def compare_models(jepa, baseline, split: P03WindowSplit) -> P03Report: ...
```
CLI 只接受 `--manifest`、`--output-dir`、`--seed`、`--dry-run` 四个稳定参数；其余实验配置写入 manifest，避免命令行漂移。

## 5. 数据层设计
| 层 | 规则 |
|---|---|
| 原始数据 | 只存在仓库外；不入 Git、不进缓存 |
| manifest | 记录本地路径、SHA-256、来源、授权状态、时间范围 |
| 宽表 schema | `subject_id, charttime, hr, sbp, dbp, spo2, rr, temp, gcs` |
| 窗口 | 输入 `(N, 12, 7)`，目标 `(N, 3, 7)` |
| 切分 | 先按 `subject_id`，再做窗口；禁止随机窗口切分 |
| 输出 | 仅聚合指标、哈希、样本数和 limitations |

## 6. 安全设计
1. MIMIC 属受限数据：必须确认授权状态后才能运行。
2. 仓库只保存 schema、manifest 模板和聚合报告。
3. 日志禁止输出患者 ID、原始观测、时间戳原文。
4. 结果文件默认写入仓库外目录；若写入仓库，必须先经过脱敏审查。

## 7. 可扩展预留点
1. 后续可增加 eICU / HiRID / PhysioNet 数据源，但必须新增独立 source adapter。
2. 后续可加入不确定性校准指标，不改变本次主指标。
3. 后续可把数据管道抽象为 `ClinicalTimeseriesDataset`，但本次不过度设计。

## 8. 行为保持策略
- 不修改 `JEPAClinicalBridge` 的预测契约。
- 不修改现有测试。
- 不删除旧验证脚本；先新增清晰命名的 P0-3 入口。
- 不把合成或半合成结果混入真实数据报告。

## 9. 风险与规避
| 风险 | 规避 |
|---|---|
| 把旧合成结果误称为 MIMIC 结果 | 报告必须带 `source` 与 manifest SHA-256 |
| 患者级数据泄漏 | 按 `subject_id` 切分并加结构测试 |
| JEPA 与基线预算不公平 | 使用相同窗口、epoch、seed 并记录训练信息 |
| 缺失值处理掩盖失败 | 报告缺失率、剔除率、填补方法 |
| 指标偶然达标 | 配对 bootstrap 输出 95% CI 与 p 值 |
| 数据权限不明 | 无授权时输出 `not_run`，不启动训练 |

## 10. 验收标准
1. 数据准入通过且 manifest 校验全绿后才能训练。
2. JEPA 与基线在同一 split 上评估。
3. `report.json` / `report.md` 无患者级信息。
4. 全部指标为真实运行结果，NaN/Inf/未收敛即 `failed`。
5. 本地 Ruff、format、mypy、pytest、敏感扫描与 ai-guard 全绿。
6. PR Gate 1–6 与 governance 10 项必需检查全绿后合并。

## 审核结论
- [ ] 职责单一
- [ ] 无跨层
- [ ] 依赖方向单向
- [ ] 兼容策略完整
- [ ] 安全点齐全
- 审核人：负责人本人
- 批准：待负责人批准
