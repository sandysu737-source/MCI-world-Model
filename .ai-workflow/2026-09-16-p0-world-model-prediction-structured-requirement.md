# 结构化开发需求：P0-2f 世界模型预测域拆分

> 日期：2026-09-16 ｜ 负责人：sandysu737-source ｜ 执行人：Codex

## 1. 业务目标
在不改变公开接口与运行语义的前提下，把 `_world_model.py` 中的预测与 JEPA 预测器域拆出，使主编排文件收敛到 1,500 行以内。

## 2. 输入参数定义
| 名称 | 类型 | 必填 | 默认 | 校验规则 |
|---|---|---|---|---|
| 宿主实例 | `MCIWorldModel` | 是 | 无 | 提供 `_state`、`_lite_pro`、`_energy_core`、`_jepa_encoder`、`_jepa_predictor`、`_jepa_mode`、`_get_memories_from_lite_pro()` |
| 迁移方法 | 方法定义集合 | 是 | 见架构方案 | AST 级逐字一致 |

## 3. 输出数据结构
固定输出结构保持不变：
- `predict_effect()` / `jepa_predict()` / `parametric_predict()` / `predict_from_memories_m3()` / `fused_predict()` 继续返回原有 `list[dict[str, Any]]`。
- `train_jepa()`、`attach_true_jepa()`、`enable_m3()`、`_is_gnn_predictor()`、`_is_e2e_mode()` 继续保持原有返回值。

## 4. 依赖资源
| 类型 | 内容 |
|---|---|
| 宿主状态 | `_state`、`_lite_pro`、`_energy_core`、`_jepa_encoder`、`_jepa_predictor`、`_jepa_mode` |
| 宿主行为 | `_get_memories_from_lite_pro()`、`_extract_energy_ratios()`、`_is_gnn_predictor()`、`_is_e2e_mode()` |
| 延迟导入 | `JEPAEncoder`、`JEPADataset`、`JEPATrainer`、`GNNPredictor` |

## 5. 业务流程步骤
1. 以 `6f23bee` 为基线建立 `codex/world-model-prediction-split` 分支。
2. 新增 `PredictionComponentsMixin`，逐字迁移预测/JEPA 方法定义。
3. 主类挂载 Mixin，删除原实现并补齐跨 Mixin 宿主契约。
4. 增加方法归属、MRO、导入方向、公开入口兼容与懒加载语义结构测试。
5. 运行定向回归、全量验证与治理门禁。

## 6. 边界条件
- [x] `memories=None` 且 `_lite_pro=None` / 非 None：保留原路径。
- [x] `_jepa_encoder` / `_jepa_predictor` 为 `None`：保留原异常与返回路径。
- [x] `top_k` 由调用方传入：保留原校验与排序语义。
- [x] 同一 `cause` 重复调用：不得新增缓存或改变去重结果。

## 7. 异常场景
保留原异常：
- `predict_effect()` 的 `ValueError` / `RuntimeError` / `KeyError` / 兜底 `RuntimeError`。
- `jepa_predict()` / `predict_from_memories_m3()` 的 `ValueError` / `RuntimeError` / `AttributeError` / `KeyError`。
- `train_jepa()` 的 `ImportError` / `ValueError` / `RuntimeError`。
- `enable_m3()` 的 `TypeError` / `ValueError` / `ImportError`。

## 8. 性能约束
- 不引入新缓存。
- 不改变循环上限。
- 不提前导入重型组件。
- 公开方法调用栈至多新增一层 Mixin 方法查找。

## 9. 安全约束
- 不新增网络、文件写入或外部命令。
- 不改变日志中可能携带的因果文本。
- 禁止在 Mixin 中反向导入 `_world_model.py`。

## 10. 必须覆盖测试用例
1. `MCIWorldModel` MRO 包含 `PredictionComponentsMixin`。
2. 10 个目标方法只在新 Mixin 定义，不在主类重复定义。
3. 新模块源码不包含 `_world_model` 导入。
4. `predict_effect()` / `jepa_predict()` / `parametric_predict()` / `fused_predict()` 关键入口可调用且行为不变。
5. `_is_gnn_predictor()` / `_is_e2e_mode()` / `enable_m3()` / `attach_true_jepa()` 保留原宿主语义。
