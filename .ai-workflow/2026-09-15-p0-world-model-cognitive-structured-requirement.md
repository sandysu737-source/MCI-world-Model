# P0-2e 世界模型认知组件拆分结构化需求

## 1. 目标
在 P0-2d 完成后继续拆分 `_world_model.py`，把 v4.3.0~v4.3.2 认知组件集成域迁入独立 Mixin，保持公开 API、返回结构、懒加载和错误语义不变。

## 2. 背景与证据
- 当前 `main` 位于 `19cb5d5`；P0-2d 已合并。
- `_world_model.py` 实测 2,246 行，仍高于 P0-2 的 ≤1,500 行目标。
- AST 确认认知组件域为 8 个方法，共 396 行：
  1. `run_cognitive_loop`
  2. `diagnose_failure`
  3. `retrieve_experiences`
  4. `detect_surprise`
  5. `plan_action`
  6. `synthesize_training_data`
  7. `assess_diversity`
  8. `check_admissibility`
- 该域与 P0-2c 架构中明确暂缓的“认知循环、诊断、规划等相邻方法”一致，是下一最小边界。
- 既有定向覆盖包括 `tests/test_cart_closed_loop.py` 与 `tests/test_world_model_v430.py`。

## 3. 范围
新增 `src/mci_world_model/sdk/_cognitive_components.py`，承载 `CognitiveComponentsMixin`。

`_world_model.py` 继承 `CognitiveComponentsMixin` 并删除原 8 个方法。组件初始化、阈值更新、可选依赖降级和返回 dict 结构保持不变。

## 4. 行为不变要求
- 方法签名、默认参数、返回 dict 键和输出顺序语义不变。
- 懒加载组件的创建条件、参数、异常处理和阈值更新逻辑不变。
- 认知闭环传播轮数、早停参数、误差注入顺序不变。
- 惊奇信号、失败诊断、经验检索、规划、反思合成、多样性与启发式检查逻辑不变。
- 不新增异常捕获、日志、网络、持久化或重试。

## 5. 宿主契约
Mixin 只允许通过以下宿主成员执行：
- `_cognitive_loop`
- `_meta_diagnoser`
- `_surprise_detector`
- `_experience_db`
- `_multi_view_retriever`
- `_action_conditioned_predictor`
- `_cost_module`
- `_multi_branch_predictor`
- `_plan_agent`
- `_reflection_synthesizer`
- `_cognitive_diversity`
- `_negative_heuristic`
- `_cewm_parse_state()`

以上组件属性均为 `Any | None`，由宿主生命周期持有并由 Mixin 懒加载更新。

## 6. 依赖方向
`_world_model.py -> _cognitive_components.py -> 各认知组件模块`。

禁止 `_cognitive_components.py` 导入 `_world_model.py`；禁止复制组件状态或业务实现。

## 7. 异常与安全
保持现有可选依赖 `ImportError` 行为、无输入提前返回和组件缺失分支。不新增静默吞错，不改变医疗语义与数据边界。

## 8. 验收标准
1. MRO 包含 `CognitiveComponentsMixin`，8 个方法归属新 Mixin。
2. 新模块不导入 `_world_model.py`。
3. 定向认知闭环测试与结构测试通过。
4. Ruff、Ruff format、mypy、全量 pytest、敏感扫描与 ai-guard 全绿。
5. PR Gate 1–6 与 governance 全绿后按 rebase 合并。

## 9. 授权
负责人本人于 2026-09-15 批准 P0-2e 方案与执行。
