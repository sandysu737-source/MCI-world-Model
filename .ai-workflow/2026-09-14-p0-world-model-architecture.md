# P0-2 世界模型拆分架构方案

## 1. 观察结论
- `_world_model.py` 实测 3,834 行、79 个实例方法。
- `from ..._world_model import` 直接引用共 149 处，外部兼容是硬约束。
- 既有设计文档已提出数据类、能量流、CEWM Mixin 三阶段拆分，但仓库未执行。
- 定向基线 `tests/test_world_state.py`、`tests/test_world_model_e2e.py`、`tests/test_jepa_encoder_predictor.py`、`tests/test_cost_module_comprehensive.py` 共 143 passed。

## 2. 阶段边界
| 阶段 | 范围 | 风险 | 目标 |
| --- | --- | --- | --- |
| P0-2a | 迁出 `CausalWorldModelState`、`TrajectoryStep`、`WorkingMemory` | 低中 | 建立无环状态边界 |
| P0-2b | 能量流函数与方法迁出 | 中 | 独立能量边界 |
| P0-2c | CEWM 12 个方法迁出为 Mixin | 中高 | 主文件 ≤1,500 行 |

本轮只执行 P0-2a；P0-2b/P0-2c 必须在 P0-2a 验证后再单独审批。

## 3. 模块职责
### `_world_model_state.py`
只承载世界模型的纯状态契约：`CausalWorldModelState`、`TrajectoryStep`、`WorkingMemory` 及其构造、序列化、兼容升级。

### `_world_model.py`
保留编排、发现、预测、因果推理、CEWM 调用入口，并 re-export 三个旧公开状态符号。

### 禁止事项
- 状态模块禁止导入 `MCIWorldModel` 或 `_world_model.py`。
- 主模块不复制状态实现。
- 不改变字段名、默认值、私有字段、线程约束或异常类型。

## 4. 依赖关系
`_world_model_state.py` 向下依赖 `typing`、`dataclasses`、NumPy 与 `_sys` 契约；`_world_model.py` 依赖 `_world_model_state.py`。依赖方向单向，禁止回环。

## 5. 接口设计
公开 API 保持：
```python
from mci_world_model.sdk._world_model import CausalWorldModelState
from mci_world_model.sdk._world_model import TrajectoryStep
from mci_world_model.sdk._world_model import WorkingMemory
```
新增导入路径仅作为内部实现推荐：
```python
from mci_world_model.sdk._world_model_state import CausalWorldModelState
```

## 6. 数据层设计
无数据库、缓存和网络。`WorkingMemory` 的内存结构、`CausalWorldModelState` 的因果边、`from_dict()` 兼容升级逻辑原样迁移。

## 7. 安全设计
无鉴权边界新增；不落盘；不改医疗语义与字段脱敏行为。

## 8. 风险与规避
| 风险 | 规避 |
| --- | --- |
| 149 处直接导入断裂 | `_world_model.py` re-export 并保留定向导入兼容测试 |
| 循环导入 | 状态模块不反向导入主模块 |
| 序列化回归 | 迁移后运行全量与定向状态/反序列化测试 |
| 混合大补丁 | P0-2a 单独 PR，预计代码差异低于 800 行 |

## 9. 弃选方案
- 一次性抽取全部 79 个方法：冲突半径过大，拒绝。
- 把状态放入通用 `helpers.py`：降低边界清晰度，拒绝。
- 只复制类不删原实现：造成双实现漂移，拒绝。

## 10. 验收标准
1. `_world_model.py` 行数 ≤3,300，状态模块无反向导入。
2. 定向测试 143 passed；`check.sh` 等效门禁全绿。
3. 旧导入路径和类行为零变更。
4. PR Gate 1–6、governance、nightly 配置全部通过。

## 审核结论
- [x] 职责单一
- [x] 无跨层
- [x] 无耦合混乱
- [x] 无过度设计
- [x] 安全点齐全
- 审核人：负责人本人
- 日期：2026-09-14
