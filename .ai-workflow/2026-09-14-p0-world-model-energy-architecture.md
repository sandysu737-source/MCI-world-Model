# P0-2b 世界模型能量流拆分架构方案

## 1. 观察结论
- P0-2a 合并后 `_world_model.py` 实测 3,255 行。
- 当前能量流相关代码包括顶层 `_aggregate_energy_ratios()` 与 `_extract_energy_ratios()`、`_build_energy_bus()`、`_propagate_energy()`、`_compute_energy_coverage()`、`predict_energy_flow()`。
- `_sys/_configurator.py` 懒加载导入 `_world_model._aggregate_energy_ratios`，现有测试也直接调用该私有符号，因此 re-export 是兼容硬约束。
- 既有定向测试覆盖 EnergyBus、能量提取、能量流预测与公共聚合函数。

## 2. 模块职责
### `_energy_flow.py`
只承载可独立测试的能量边界：五维比率聚合、状态能量提取、能量覆盖度计算、能量流预测。禁止承载因果发现、CEWM 或编排逻辑，也禁止直接导入 `su_memory`。

### `_world_model.py`
保留生命周期、发现、因果推理、CEWM、JEPA 训练与健康诊断编排；通过继承 `EnergyFlowMixin` 暴露纯能量方法，并 re-export `_aggregate_energy_ratios`。EnergyBus 构建与传播依赖 `su_memory`，受 `TestNoSuMemoryResidue` 白名单约束，必须保留在 `_world_model.py` 作为适配层。

## 3. 接口设计
`EnergyFlowMixin` 仅依赖宿主提供的：
- `_state`
- `_energy_core`
- `_energy_flow_predictor`
- `_get_energy_core()`

公开与私有调用入口保持：
```python
from mci_world_model.sdk._world_model import _aggregate_energy_ratios
wm._extract_energy_ratios(state)
wm._build_energy_bus()
wm._propagate_energy(steps=3)
wm._compute_energy_coverage()
wm.predict_energy_flow(steps=5)
```

## 4. 依赖关系
`_energy_flow.py -> _energy_flow_predictor.py`；`_world_model.py -> _energy_flow.py -> _world_model_state.py`。`_energy_flow.py` 禁止反向导入 `_world_model.py`，也禁止直接导入 `su_memory`。`_configurator.py` 的旧懒加载导入不改，仍由 `_world_model.py` re-export 满足。

## 5. 行为保持策略
- 逐字迁移实现，仅调整宿主依赖引用。
- 不改默认参数、返回 dict 键、注释语义、异常类型与懒加载时序。
- `_aggregate_energy_ratios` 在 `_world_model.py` re-export，防止私有外部导入断裂。
- 不复制双实现。

## 6. 风险与规避
| 风险 | 规避 |
| --- | --- |
| 私有导入断裂 | `_world_model.py` re-export 并运行既有定向导入测试 |
| Mixin 隐式契约扩大 | 文档固化宿主依赖并新增 MRO/来源测试 |
| EnergyBus 适配误扩散 | `TestNoSuMemoryResidue` 证明其必须留在 `_world_model.py`，架构已回环修正 |
| EnergyBus API 差异回归 | 定向运行 `TestMCIWorldModelEnergyBus` |
| predictor 懒加载回归 | 定向运行 `TestEnergyFlowPredictor` |
| 大补丁 | 预计差异低于 800 行，单独 P0-2b PR |

## 7. 弃选方案
- 一次性迁移全部能量引用：会触及 discover、intervene、health 等非能量域，扩大回归面，拒绝。
- 把 Mixin 放入通用 helpers：边界不清晰，拒绝。
- 只复制不删除原实现：双实现漂移，拒绝。
- 修改 `_configurator.py` 旧导入：超出兼容目标，拒绝。

## 8. 验收标准
1. `_energy_flow.py` 不导入 `_world_model.py`。
2. `_world_model.py` 仍导出 `_aggregate_energy_ratios`，MRO 包含 `EnergyFlowMixin`；EnergyBus 两个方法保留在主类。
3. 新增定向能量拆分测试通过；既有 `TestMCIWorldModelEnergyBus`、`TestEnergyFlowPredictor`、`TestExtractEnergyRatios` 通过。
4. `ruff`、`ruff format --check`、`mypy`、全量 pytest、`ai-guard` 全绿。
5. PR Gate 1–6 与 governance 全绿后按 rebase 合并。

## 审核结论
- [x] 职责单一
- [x] 无跨层
- [x] 依赖方向单向
- [x] 兼容策略完整
- [x] 安全点齐全
- 审核人：负责人本人
- 批准：2026-09-14 明确批准“继续推进”
