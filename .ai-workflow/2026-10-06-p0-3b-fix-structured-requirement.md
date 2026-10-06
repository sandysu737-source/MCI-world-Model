# 结构化开发需求：P0-3b-fix 环（评估器统计口径整改）

> 日期：2026-10-06 ｜ 风险等级：L2 ｜ 负责人：sandysu737-source ｜ 执行人：Codex
> 上游：`.ai-workflow/2026-10-06-p0-3b-adversarial-audit.md`（两轮对抗性审计）§5 整改清单
> 范围：P0-A / P0-B / P1-C / P2-D / P2-E / P2-F / P3-G（J1/J2 已在 PR #27 落地；H/I/J3/K 各自单独开环）

## 1. 业务目标

在"JEPA 是否优于基线"的结论链路上，消除四类可被证伪的失真：统计单元错配（P0-A）、
结论不可追溯（P0-B）、主判据被大尺度体征主导（P1-C）、以及若干边界/性能缺陷（P2/P3）。
本环**不放宽任何门禁、不删除测试**；所有绿灯由修根因达成。

## 2. 契约变更（对外可见）

| 项 | 变更 | 兼容性 |
|---|---|---|
| `P03WindowSplit` | 新增 `train_window_subjects` / `test_window_subjects`（window→patient 映射） | 追加字段、默认空元组；有窗口且缺映射即 `P03MetricsError`（P0-A fail-closed） |
| `compare_models` | 新增 keyword `dataset_manifest`；主判据改尺度归一化 MAE | 缺 manifest 时 `passed` 不可达（状态 `not_run`），其余状态不变 |
| `P03DatasetManifest` | 新增冻结 dataclass：`version` / `sha256` / `authorization` / `n_subjects` / `source` | 新类型，仅用于绑定 |
| `P03Report` | 新增 `dataset_manifest` 字段、`n_deteriorated_vitals` 属性；`metric` 默认改 `scale_normalized_mae` | `to_dict()` 新增两个键 |
| `P03StatisticalTest` | 新增 `n_deteriorated_vitals` / `deteriorated_vitals`；`per_vital[*]` 新增 `normalized_mae_jepa` / `normalized_mae_baseline` / `worse` | 追加键，原键保留 |
| `evaluate_model` | `n_resamples` 接受 `numbers.Integral`（含 `np.int64`）；bool 仍非法 | 放宽入参、失败信息不变（P3-G） |
| `relative_mae` | 分母下限由统一 `1e-9` 改为按体征临床尺度（可行范围跨度 × 1%） | 目标量级远低于临床尺度时由"伪精度"变为 `null`（P2-F） |

## 3. 口径定义（本环固化）

1. **统计单元**：独立单元是患者；bootstrap 以患者为整簇重采样（block bootstrap），
   实现上先按簇聚合 sum/计数再做簇抽样合并，内存与簇数相关（P0-A + P2-D）。
2. **主判据**：`scale_normalized_mae = mean(|error| / 体征可行范围跨度)`，
   配对差 `baseline_normalized - jepa_normalized`；>0 表示 JEPA 更优（P1-C）。
3. **劣化守卫**：7 个体征中 `>= 4` 个尺度归一化 MAE 相对劣化 → `passed` 不可达（P1-C）。
4. **样本门槛**：测试窗口 `>= 20` 且测试患者 `>= 5` 才允许 `passed`（P2-E）。
5. **清单绑定**：`dataset_manifest` 必须字段合法、`source` 与报告一致、`n_subjects >= 测试患者数`；
   否则 `passed` 不可达（P0-B）。
6. **分块预算**：bootstrap 每个分块的浮点元素总量 `<= 2e6`（`_BOOTSTRAP_GATHER_ELEMENTS`），
   不再只约束索引矩阵（P2-D）。
7. **相对 MAE 下限**：`VITAL_ERROR_SCALES` = 各体征可行范围跨度；分母 `< 跨度 × 1%` → `null`（P2-F）。

`passed` 语义（最终）：**同时**满足 `source="mimic"` + JEPA 显著更优 + manifest 绑定通过 +
样本达门槛 + 无多数体征劣化；任一不满足即 `not_run`/`failed`/`degraded`，并在 `limitations` 写明原因。

## 4. 步骤与验收（含实测证据）

| 编号 | 验收标准 | 证据 | 结果 |
|---|---|---|---|
| P0-A | 聚类场景误报 ≤10%、覆盖 ≥90%；映射与切分一致 | `p0a-verify.py`：200 次 H0 → 8.00% / 92.00%（对照窗口级 36.50% / 63.50%，参考患者级 block 8.00% / 92.00%）；探针 a 探针2：7.50% / 92.50% | ✅ |
| P0-A | 缺映射/长度不一致 fail-closed | `tests/test_p03_metrics.py::test_split_without_window_patient_mapping_fails_closed`、`::test_split_with_mismatched_mapping_fails_closed`、`tests/test_p03_data_foundation.py` 映射一致性断言 | ✅ |
| P0-B | 无绑定时任何输入不可 `passed`；报告含 manifest | `::test_manifest_binding_is_required_for_passed`、`::test_invalid_manifest_blocks_passed`（4 组破坏用例）、`::test_valid_manifest_binding_allows_passed`；探针 a 探针4：`source='mimic'` → `not_run` | ✅ |
| P1-C | 5/7 体征劣化反例不再 `passed` | `::test_majority_vital_deterioration_blocks_passed`（5/7 劣化 + 整体归一化更优 → `not_run`）；探针 a 探针3：`status=failed`（旧口径 `passed`） | ✅ |
| P2-D | N=100k、2000 次 → <5s 且峰值 <120MB | `p2d-verify.py`：**0.69s / 100.1MB**（旧口径 8.2–8.9s / 360MB）；探针 a 探针6：0.10s / 105.0MB | ✅ |
| P2-E | 低于门槛 `passed` 不可达 | `::test_small_sample_cannot_pass`（3×4=12 窗口 / 2×15=30 窗口但仅 2 患者，双向）；探针 b 探针7：N=5 误报 19.33% → 门槛必要性 | ✅ |
| P2-F | 目标量级 1e-8 → `relative_mae=null` | `::test_relative_mae_uses_clinical_scale_floor`；探针 a 探针5：`hr=None` | ✅ |
| P3-G | `np.int64` 与 `int` 等价 | `::test_n_resamples_accepts_numpy_integer`；探针 a 探针5：`n_resamples=np.int64` → ok | ✅ |

## 5. 影响面与风险

- **结论口径变化**：旧报告里"原始单位 MAE 显著更优"不再等价于 `passed`；历史报告不可与新报告直接比较
  （`P03Report.metric` 已改为 `scale_normalized_mae`，可按 `metric` 字段区分）。
- **调用方需要提供映射与 manifest**：真实结论路径（P0-3c）必须由 P0-3a 管道产出映射，
  并由负责人提供授权 manifest；本机无 MIMIC 授权数据与本地 manifest，故真实结论只能 `not_run`/`degraded`。
- **fail-closed 风险**：门槛与守卫会让短期更难拿到 `passed` —— 这是设计目标（宁可不判，不可误判）。
- **回滚**：本环为单一 commit 系列，`git revert` 即可回到旧口径；不涉及数据迁移与生产配置。

## 6. 弃选方案

- 用 `p` 值做多重比较校正替代多数体征守卫：校正只影响显著性判定，无法阻止"整体显著但多数体征更差"，弃。
- 用 target 均值做相对 MAE 分母（原审计建议之一）：分母随样本波动、可被极小目标放大，改为固定临床尺度，弃。
- 直接放宽变异门禁阈值或删除测试换绿灯：违反红线，弃。

## 7. 验证命令

```bash
PYTHONPATH="$PWD:$PWD/src:../su-memory-sdk/src" .venv/bin/python -m pytest tests/test_p03_metrics.py tests/test_p03_data_foundation.py -q --override-ini=addopts=
PYTHONPATH="$PWD:$PWD/src:../su-memory-sdk/src" .venv/bin/python .ai-workflow/2026-10-06-p0-3b-fix-p0a-verify.py
PYTHONPATH="$PWD:$PWD/src:../su-memory-sdk/src" .venv/bin/python .ai-workflow/2026-10-06-p0-3b-fix-p2d-verify.py
PYTHONPATH="$PWD:$PWD/src:../su-memory-sdk/src" .venv/bin/python .ai-workflow/2026-10-06-p0-3b-adversarial-probes-a.py
PYTHONPATH="$PWD:$PWD/src:../su-memory-sdk/src" .venv/bin/python .ai-workflow/2026-10-06-p0-3b-adversarial-probes-b.py
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

## 8. 不在本环范围（另行开环）

- H：`benchmarks/**` 纳入 CI `scope=full`（现走 warn 通道）。
- I：本地钩子覆盖率阈值注入（kit 副本 F-20 缺阈值）。
- J3：仓库副本与 kit 副本 `mutation-check.sh` 同源化。
- K：L2 审批指纹改为按 PR 号 + 日期动态生成（消除重放）。
- P0-3c：真实数据结论路径（须在 P0-A/B/C 全部落地后才允许开启）。
