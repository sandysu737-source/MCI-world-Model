# P0-2c 世界模型 CEWM Mixin 拆分架构方案

## 1. 观察结论
- P0-2b 合并后 `_world_model.py` 实测 3,143 行。
- AST 确认 CEWM 域实际为 13 个方法，共约 389 行；旧设计文档估算 12 个方法，本次按实际代码收敛。
- 定向测试覆盖 `test_cewm_submethods.py`、`test_cewm_e2e_closed_loop.py`、`test_phase3_cewm_loop.py`、`test_replay_buffer.py`。
- Mixin 不能导入主模块；JEPA 预测仍留在主类，由 CEWM Mixin 通过宿主契约调用。

## 2. 模块职责
### `_cewm_engine.py`
只承载 CEWM 四环闭环、快速路径、replay 经验写入、JEPA 增量训练触发、状态解析与状态变化提取。禁止承载发现、干预、反事实或健康诊断。

### `_world_model.py`
保留生命周期、发现、预测、因果推理、能量编排与健康诊断；继承 `CEWMEngineMixin` 并通过既有 `jepa_predict()` 提供预测能力。

## 3. 接口设计
`CEWMEngineMixin` 仅依赖宿主提供的：
- `logger`
- `_replay_enabled`、`_step_count`、`_replay_threshold`、`_replay_interval`
- `_safety_monitor`、`_causal_updater`、`_deadline_monitor`
- `_experience_db`、`_action_gap_metric`、`_state_parser_registry`
- `_perception`、`_jepa_predictor`、`_emergency_stop`
- `jepa_predict()`

公开与私有调用入口保持不变：
```python
wm.cewm_step(observation=..., goal=..., action=...)
wm.cewm_step_fast(observation=..., goal=..., action=...)
wm._init_cewm_result()
wm._cewm_parse_state(obs)
wm._cewm_state_change(state)
```

## 4. 依赖关系
`_cewm_engine.py -> SDK 懒加载组件 / _world_model_state`；`_world_model.py -> _cewm_engine.py -> _world_model_state.py`。`_cewm_engine.py` 禁止反向导入 `_world_model.py`。

## 5. 行为保持策略
- 逐字迁移实现，仅调整宿主契约声明。
- 不改默认参数、返回 dict 键、日志级别、异常类型与提前返回顺序。
- `_cewm_parse_state` 与 `_cewm_state_change` 一并迁出，避免 CEWM 内部跨文件跳转。
- 不复制双实现。

## 6. 风险与规避
| 风险 | 规避 |
| --- | --- |
| Mixin 隐式契约扩大 | 固化宿主契约并新增 MRO/归属测试 |
| CEWM 行为漂移 | 定向运行 4 个既有 CEWM/replay 测试文件 |
| replay 时序回归 | 保持 `_step_count`、阈值、间隔与异常捕获顺序 |
| 状态解析回环 | `_cewm_engine.py` 禁止反向导入主模块并做源码断言 |
| 大补丁 | 预计差异低于 800 行，单独 P0-2c PR |

## 7. 弃选方案
- 一次性迁移认知循环、诊断、规划等相邻方法：超出 P0-2c 边界，扩大回归面，拒绝。
- 把 CEWM 做成独立引擎类：会改变组合关系与生命周期，超出兼容目标，拒绝。
- 只复制不删除原实现：双实现漂移，拒绝。
- 将 `jepa_predict()` 迁入 Mixin：预测属于世界模型主路径，会造成职责反转，拒绝。

## 8. 验收标准
1. `_cewm_engine.py` 不导入 `_world_model.py`。
2. `MCIWorldModel` MRO 包含 `CEWMEngineMixin`，13 个方法归属新 Mixin。
3. 定向 CEWM/replay 测试通过；新增结构测试通过。
4. `ruff`、`ruff format --check`、`mypy`、全量 pytest、`ai-guard` 全绿。
5. PR Gate 1–6 与 governance 全绿后按 rebase 合并。

## 审核结论
- [x] 职责单一
- [x] 无跨层
- [x] 依赖方向单向
- [x] 兼容策略完整
- [x] 安全点齐全
- 审核人：负责人本人
- 批准：2026-09-14 明确批准开始 P0-2c
