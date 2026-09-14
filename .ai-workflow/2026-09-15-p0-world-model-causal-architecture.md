# P0-2d 世界模型因果推理拆分架构方案

## 1. 观察结论
- P0-2c 后 `_world_model.py` 为 2,762 行，仍承担编排、因果推理、JEPA、健康诊断与认知组件。
- AST 确认因果推理域为 7 个方法、490 行，不是按旧估算的新范围。
- 该域依赖宿主的 `_state`、`_do_calculus`、锁与干预历史；不依赖 JEPA 预测。
- 定向测试已覆盖 WP-09 数据源干预、临床 Pearl bridge、World Model E2E 和 v430 回归。

## 2. 模块职责
### `_causal_inference.py`
只承载 Pearl do-operator 干预、因果效应分解、L3 反事实推理、因果链解释与因果图构建。禁止承载发现、JEPA、CEWM、健康诊断或编排。

### `_world_model.py`
保留生命周期、发现、预测、认知组件与健康诊断；继承 `CausalInferenceMixin`，通过既有 `discover()` 维护 `_state.causal_edges`。

## 3. 接口设计
公开与私有调用入口保持不变：
```python
wm.intervene(state=..., do_x=..., target=..., method=...)
wm.decompose_effect(cause=..., effect=..., mediator=...)
wm.query_counterfactual(evidence=..., do_x=..., target=..., compute_pns=...)
wm.explain(query=..., max_depth=...)
wm._build_causal_graph_from_state()
wm._trace_causal_chains(query=..., max_depth=...)
wm._generate_explanation_summary(chains=..., query=...)
```

## 4. 宿主契约
`CausalInferenceMixin` 在 `TYPE_CHECKING` 中声明：
- `_state: CausalWorldModelState`
- `_do_calculus: Any | None`
- `_do_calculus_lock: threading.Lock`
- `_intervention_history: list[dict[str, Any]]`
- `_intervention_history_lock: threading.Lock`

`_causal_inference.py` 顶部自持 `logging`、`numpy`、`typing` 和模块级 `logger`。

## 5. 依赖关系
`_causal_inference.py` 只允许懒加载：
- `mci_world_model.sdk._do_calculus`
- `mci_world_model.sdk._counterfactual`

禁止导入 `_world_model.py`，避免主类/Mixin 回环。

## 6. 行为保持策略
- 逐字迁移实现，仅调整模块导入与宿主契约声明。
- 不改默认参数、返回键、日志级别、异常类型与提前返回顺序。
- 不合并锁、不改时间戳生成、不调整干预历史写入顺序。
- 不复制双实现；删除宿主原方法。

## 7. 风险与规避
| 风险 | 规避 |
| --- | --- |
| L2/L3 语义漂移 | 逐字迁移并保留 WP-09/clinical Pearl 回归 |
| 干预历史时序变化 | 保持锁、append 和 history_count 逻辑 |
| Mixin 契约膨胀 | 只声明 5 个宿主成员并新增归属测试 |
| 模块回环 | 源码断言禁止导入 `_world_model.py` |
| 大补丁 | 方案、代码、台账分批提交，单提交 ≤800 行 |

## 8. 弃选方案
- 一次拆出 JEPA、健康诊断与认知组件：范围过大，拒绝。
- 将 `discover()` 迁入因果 Mixin：发现与推理职责不同，拒绝。
- 把 `_state` 或锁复制进 Mixin：状态双源会漂移，拒绝。
- 只复制不删除原实现：导致双实现漂移，拒绝。

## 9. 验收标准
1. 7 个方法 AST 级别与原实现一致。
2. `MCIWorldModel` MRO 包含 `CausalInferenceMixin`。
3. 新增结构测试通过；定向因果回归通过。
4. 本地 Ruff、Ruff format、mypy、全量 pytest、敏感扫描与 ai-guard 全绿。
5. PR Gate 1–6 与 governance 全绿后按 PR 流程合并。

## 审核结论
- [x] 职责单一
- [x] 无跨层
- [x] 依赖方向单向
- [x] 兼容策略完整
- [x] 安全点齐全
- 审核人：负责人本人
- 批准：2026-09-15 明确批准 P0-2d
