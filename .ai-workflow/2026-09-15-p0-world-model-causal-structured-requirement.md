# P0-2d 世界模型因果推理拆分结构化需求

## 1. 目标
在 P0-2c 完成后继续拆分 `_world_model.py`，把 Pearl L2/L3 因果推理与解释链路迁入独立 Mixin，保持所有旧调用入口与返回结构不变。

## 2. 背景与证据
- 当前 `main` 位于 `ebee7c2`；P0-2c 已合并。
- `_world_model.py` 实测 2,762 行，仍高于 P0-2 的 ≤1,500 行目标。
- AST 确认因果推理域为 7 个方法，共 490 行：
  1. `_build_causal_graph_from_state`
  2. `intervene`
  3. `decompose_effect`
  4. `query_counterfactual`
  5. `explain`
  6. `_trace_causal_chains`
  7. `_generate_explanation_summary`
- 既有定向覆盖包括 `tests/adversarial/test_wp09_data_source_intervention.py`、`tests/test_clinical_pearl_bridge.py`、`tests/test_world_model_e2e.py`、`tests/test_world_model_v430.py`。

## 3. 范围
新增 `src/mci_world_model/sdk/_causal_inference.py`，承载 `CausalInferenceMixin`。

`_world_model.py` 继承 `CausalInferenceMixin` 并删除原 7 个方法。`DoCalculus`、`CounterfactualEngine`、`CausalGraph` 的懒加载与异常语义保持不变。

## 4. 行为不变要求
- 方法签名、默认参数、返回 dict 键不变。
- 参数校验、状态码、错误消息、日志级别和提前返回顺序不变。
- 单变量干预、有限数值校验、多变量拒绝语义不变。
- `_state.do_interventions`、`counterfactual_graph` 和 `_intervention_history` 更新时序不变。
- 因果链 BFS、排序、置信度与解释摘要格式不变。

## 5. 宿主契约
Mixin 仅允许通过以下宿主成员执行：
- `_state`
- `_do_calculus`
- `_do_calculus_lock`
- `_intervention_history`
- `_intervention_history_lock`

模块级 `logger` 与 NumPy 由 `_causal_inference.py` 自持。

## 6. 依赖方向
`_world_model.py -> _causal_inference.py -> _do_calculus.py/_counterfactual.py`。

禁止 `_causal_inference.py` 导入 `_world_model.py`；禁止复制宿主状态或业务实现。

## 7. 异常与安全
保持 `ImportError`、`ValueError`、`KeyError`、`TypeError`、`RuntimeError` 的现有捕获范围。无观测数据时继续 fail-closed，不生成因果性结论。

## 8. 验收标准
1. MRO 包含 `CausalInferenceMixin`，7 个方法归属新 Mixin。
2. 新模块不导入 `_world_model.py`。
3. 定向因果测试与结构测试通过。
4. Ruff、Ruff format、mypy、全量 pytest、敏感扫描与 ai-guard 全绿。
5. PR Gate 1–6 与 governance 全绿后按 rebase 合并。

## 9. 授权
负责人本人于 2026-09-15 批准 P0-2d 方案与执行。
