# P0-2e 世界模型认知组件拆分架构方案

## 1. 观察结论
- P0-2d 后 `_world_model.py` 为 2,246 行、56 个实例方法。
- AST 确认 v4.3.0~v4.3.2 认知组件域为 8 个方法、396 行。
- 该域只依赖 12 个组件属性和 `_cewm_parse_state()`，不依赖因果图构建、DoCalculus 或 JEPA 训练。
- 方法块连续分布于 `run_cognitive_loop()` 至 `check_admissibility()`，边界清晰。
- `health_check()` 几乎依赖全部模块，`six_module_pipeline()` 是编排入口，均不放入本 Mixin。

## 2. 模块职责
### `_cognitive_components.py`
只承载认知闭环传播、失败诊断、经验检索、惊奇检测、行动规划、反思数据合成、多样性评估与启发式可接受性检查。禁止承载健康诊断、因果推理、JEPA 或世界模型编排。

### `_world_model.py`
保留生命周期、初始化、发现、预测、因果推理、健康诊断与编排；继承 `CognitiveComponentsMixin` 并持有组件状态。

## 3. 接口设计
公开调用入口保持不变：
```python
wm.run_cognitive_loop(layer_errors=..., n_rounds=...)
wm.diagnose_failure(surprise_signals=..., context=...)
wm.retrieve_experiences(query=..., top_k=..., strategy=...)
wm.detect_surprise(predicted=..., actual=..., threshold=...)
wm.plan_action(current=..., goal=..., max_horizon=..., n_branches=...)
wm.synthesize_training_data(memories=...)
wm.assess_diversity(memories=...)
wm.check_admissibility(target=..., action=..., description=...)
```

## 4. 宿主契约
`CognitiveComponentsMixin` 在 `TYPE_CHECKING` 中声明 12 个组件属性和 `_cewm_parse_state()` 方法。所有组件属性使用 `Any | None`，保持现有动态组件语义。

## 5. 依赖关系
`_cognitive_components.py` 允许懒加载：
- `_cognitive_loop`
- `_meta_diagnoser`
- `_experience_memory` / `_multi_view_retriever`
- `_surprise_detector`
- `_action_conditioned_predictor` / `_plan_agent` / `_multi_branch_predictor`
- `_reflection_synthesizer`
- `_cognitive_diversity`
- `_negative_heuristic`

禁止导入 `_world_model.py`，避免主类/Mixin 回环。

## 6. 行为保持策略
- 逐字迁移实现，仅调整模块边界与宿主契约声明。
- 不改方法签名、默认参数、返回键、日志级别、异常类型与提前返回顺序。
- 不合并懒加载初始化，不缓存新状态，不改 PlanAgent 依赖注入顺序。
- 不复制双实现；删除宿主原方法。

## 7. 风险与规避
| 风险 | 规避 |
| --- | --- |
| 认知闭环语义漂移 | 逐字迁移并运行 cart closed loop / v430 回归 |
| 懒加载状态双源 | 仍由宿主持有并更新组件属性 |
| Mixin 契约膨胀 | 只声明 13 项宿主契约并新增归属测试 |
| 模块回环 | 源码断言禁止导入 `_world_model.py` |
| 大补丁 | 方案、代码、台账分批提交，单提交 ≤800 行 |

## 8. 弃选方案
- 一并拆出健康诊断：依赖面横跨全部模块，回归半径过大，拒绝。
- 一并拆出六模块编排：编排属于世界模型主路径，拒绝。
- 把 `_cewm_parse_state()` 复制进本模块：状态解析双实现会漂移，拒绝。
- 用独立引擎类替换 Mixin：改变组合关系与生命周期，超出兼容目标，拒绝。

## 9. 验收标准
1. 8 个方法 AST 级别与原实现一致。
2. `MCIWorldModel` MRO 包含 `CognitiveComponentsMixin`。
3. 新增结构测试通过；定向认知回归通过。
4. 本地 Ruff、Ruff format、mypy、全量 pytest、敏感扫描与 ai-guard 全绿。
5. PR Gate 1–6 与 governance 全绿后按 PR 流程合并。

## 审核结论
- [x] 职责单一
- [x] 无跨层
- [x] 依赖方向单向
- [x] 兼容策略完整
- [x] 安全点齐全
- 审核人：负责人本人
- 批准：2026-09-15 明确批准 P0-2e
