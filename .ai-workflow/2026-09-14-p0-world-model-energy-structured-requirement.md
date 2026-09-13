# P0-2b 世界模型能量流拆分结构化需求

## 1. 业务目标
把 `_world_model.py` 中能量比率聚合、能量覆盖度与能量流预测迁入独立能量边界；EnergyBus 外部适配保留在主文件，继续收敛可独立测试的能量逻辑。

## 2. 输入参数定义
| 字段 | 类型 | 必填 | 默认 | 校验规则 |
| --- | --- | --- | --- | --- |
| 外部导入路径 | str | 是 | 无 | `_world_model._aggregate_energy_ratios` 与 `MCIWorldModel` 能量方法必须保持可导入、可调用 |
| 方法签名 | API | 是 | 无 | 名称、参数、默认值、返回结构与异常行为零变更 |
| 宿主依赖 | API | 是 | 无 | Mixin 仅依赖 `_state`、`_energy_core`、`_energy_flow_predictor` 与 `_get_energy_core()` |
| 验证范围 | str | 是 | full | 至少覆盖定向能量回归、ruff、mypy strict、全量 pytest |

## 3. 输出数据结构
新增 `src/mci_world_model/sdk/_energy_flow.py`：`_aggregate_energy_ratios()` 与 `EnergyFlowMixin`。迁移 `_extract_energy_ratios()`、`_compute_energy_coverage()`、`predict_energy_flow()`。`_build_energy_bus()` 与 `_propagate_energy()` 因 `TestNoSuMemoryResidue` 禁止扩散 `su_memory` 导入，保留在 `_world_model.py` 作为外部总线适配层。`_world_model.py` 保持旧私有函数 re-export，并让 `MCIWorldModel` 继承 Mixin。

## 4. 依赖资源
依赖现有 `typing`、`_energy_flow_predictor.EnergyFlowPredictor` 与状态模块。`_energy_flow.py` 禁止导入 `_world_model.py` 和直接导入 `su_memory`，避免回环与适配层扩散。

## 5. 业务流程步骤
1. 迁移能量纯函数与 3 个方法到 `_energy_flow.py`。
2. `_world_model.py` re-export `_aggregate_energy_ratios` 并继承 `EnergyFlowMixin`。
3. 新增定向兼容测试保护 Mixin 来源、方法行为与旧导入。
4. 运行定向回归、`check.sh` 与全量 pytest。
5. 走 PR、CI 与 governance 合并。

## 6. 边界条件
- [x] 空 causal_edges、无能量标签、缺失状态属性均返回原结果
- [x] EnergyBus 缺失 API 的异常处理路径不变
- [x] `steps=0`、默认 steps、空能量比率回退行为不变
- [x] predictor 懒加载与复用行为不变

## 7. 异常场景
外部依赖缺失、API 差异、非法状态与能量预测异常的类型、来源和时序必须保持不变；禁止新增静默吞错。

## 8. 性能约束
Mixin 不增加运行时中转层；除方法查找外，调用路径保持相同。

## 9. 安全约束
不新增网络、持久化、密钥或外部输入处理；医疗语义、脱敏和异常消息不变。

## 10. 必须覆盖测试用例
1. 正向：旧导入路径仍可导入 `_aggregate_energy_ratios` 与 3 个方法。
2. 边界：空边、无能量标签、无 predictor、空 ratios 默认分布。
3. 异常：EnergyBus 缺少 `propagate()` 或 `get_bus_state()` 时按原行为处理。
4. 结构：`EnergyFlowMixin` 不反向导入主模块，`MCIWorldModel` MRO 包含 Mixin。
