# 架构设计：P0-2f 世界模型预测域拆分

> 日期：2026-09-16 ｜ 风险等级：L1 ｜ 审核人：负责人本人

## 1. 模块职责
`src/mci_world_model/sdk/_prediction_components.py` 只承载因果检索增强预测、JEPA 潜空间预测、记忆预测、融合预测、JEPA 训练与预测器接入/升级。`_world_model.py` 保留世界模型初始化、因果发现、能量流、健康诊断、编排与参数化记忆接口。

## 2. 模块拆分
新增 `PredictionComponentsMixin`，迁移 10 个方法：

| 方法 | 行数 | 归属理由 |
|---|---:|---|
| `attach_true_jepa()` | 10 | 接入 JEPA 编码器与预测器 |
| `predict_effect()` | 35 | 检索增强因果预测入口 |
| `jepa_predict()` | 106 | JEPA 潜空间主预测路径 |
| `parametric_predict()` | 21 | 参数化预测分流 |
| `predict_from_memories_m3()` | 59 | M3 记忆预测路径 |
| `fused_predict()` | 59 | 检索与 JEPA 融合预测 |
| `train_jepa()` | 75 | JEPA 编码器/预测器训练 |
| `_is_gnn_predictor()` | 5 | GNN 预测器判定 |
| `_is_e2e_mode()` | 5 | 端到端模式判定 |
| `enable_m3()` | 45 | M3 预测组件升级 |

合计迁移 420 行；预计 `_world_model.py` 收敛到约 1,420 行。

## 3. 依赖关系（内部 + 外部）
`PredictionComponentsMixin` 允许延迟导入：
- `_jepa_encoder.JEPAEncoder`
- `_jepa_training.JEPADataset`、`JEPATrainer`
- `_jepa_gnn.GNNPredictor`

允许宿主契约：
- `_state`
- `_lite_pro`
- `_energy_core`
- `_jepa_encoder`
- `_jepa_predictor`
- `_jepa_mode`
- `_get_memories_from_lite_pro()`
- `_extract_energy_ratios()`

禁止导入 `_world_model.py`，避免主类/Mixin 回环。

## 4. 接口设计
所有公开方法签名、参数名、默认值、返回键、日志级别、异常类型与提前返回顺序保持不变。`MCIWorldModel` 继续作为唯一对外门面，Mixins 只作为内部组织边界。

## 5. 数据层设计
不新增持久化、缓存、数据库或状态字段。组件仍由宿主 `__init__()` 持有并更新，Mixin 只读写宿主槽位。

## 6. 安全设计
无鉴权、脱敏或隔离策略变化。不新增外部输入处理，仅迁移既有方法边界。

## 7. 可扩展预留点
后续可将预测器接口契约抽成 protocol，但本次不做；避免先扩边界再拆职责。

## 8. 行为保持策略
- 逐字迁移实现，仅调整模块边界与宿主契约声明。
- 不合并懒加载初始化，不缓存新状态。
- 不改变 `PlanAgent`、能量流、CEWM、因果发现依赖注入顺序。
- 不复制双实现；删除宿主原方法。

## 9. 风险与规避
| 风险 | 规避 |
|---|---|
| JEPA 语义漂移 | AST 对比与 WP-10/JEPA 回归 |
| 组件生命周期变化 | 保留宿主 `__init__()` 持有状态 |
| 跨 Mixin 隐藏依赖 | TYPE_CHECKING 显式声明契约 |
| 模块回环 | 源码断言禁止导入 `_world_model.py` |
| 大补丁 | 方案、代码、台账分批提交，单提交 ≤800 行 |

## 10. 弃选方案
- 一并迁移 `_get_memories_from_lite_pro()`：它同时服务 `discover()`，不是纯预测域，拒绝。
- 一并迁移 `predict_causal_category()`：属于参数化记忆域，后续单独处理，拒绝。
- 迁移健康诊断或六模块编排：依赖面宽、回归半径大，拒绝。
- 用独立预测器类替换 Mixin：改变组合关系与生命周期，超出兼容目标，拒绝。

## 11. 验收标准
1. 10 个方法 AST 级别与原实现一致。
2. `MCIWorldModel` MRO 包含 `PredictionComponentsMixin`。
3. 新增结构测试通过；定向预测/JEPA 回归通过。
4. 本地 Ruff、Ruff format、mypy、全量 pytest、敏感扫描与 ai-guard 全绿。
5. PR Gate 1–6 与 governance 全绿后按 PR 流程合并。
6. `_world_model.py` 行数不超过 1,500。

## 审核结论
- [x] 职责单一
- [x] 无跨层
- [x] 依赖方向单向
- [x] 兼容策略完整
- [x] 安全点齐全
- 审核人：负责人本人
- 批准：2026-09-16 负责人下达“继续”，授权按既定 P0-2 序列执行 P0-2f
