# P0-2c 世界模型 CEWM Mixin 拆分结构化需求

## 1. 业务目标
把 `_world_model.py` 中 CEWM 闭环、快速路径、replay 与状态解析行为迁入独立 Mixin，减少主编排文件的集中变更面。

## 2. 输入参数定义
| 字段 | 类型 | 必填 | 默认 | 校验规则 |
| --- | --- | --- | --- | --- |
| 方法签名 | API | 是 | 无 | 13 个 CEWM 方法名称、参数、默认值、返回结构与异常行为零变更 |
| 宿主契约 | API | 是 | 无 | Mixin 仅依赖文档列出的宿主属性与 `jepa_predict()` |
| 调用入口 | API | 是 | 无 | `MCIWorldModel.cewm_step()`、`cewm_step_fast()` 及私有 CEWM 方法保持可调用 |
| 验证范围 | str | 是 | full | 至少覆盖定向 CEWM 回归、ruff、mypy strict、全量 pytest |

## 3. 输出数据结构
新增 `src/mci_world_model/sdk/_cewm_engine.py`：`CEWMEngineMixin`。迁移 `_init_cewm_result`、`_cewm_perceive`、`_cewm_safety_check`、`_cewm_cognize`、`_cewm_evaluate_action`、`_cewm_predict`、`_cewm_feedback`、`cewm_step`、`_store_replay_experience`、`_replay_train`、`cewm_step_fast`、`_cewm_parse_state`、`_cewm_state_change`。`_world_model.py` 保留编排入口并继承 Mixin。

## 4. 依赖资源
依赖现有 `typing`、`logger`、状态模块、SDK 懒加载组件和 `MCIWorldModel.jepa_predict()`。`_cewm_engine.py` 禁止导入 `_world_model.py`，避免回环。

## 5. 业务流程步骤
1. 迁移 13 个 CEWM 方法到 `_cewm_engine.py`。
2. `_world_model.py` 继承 `CEWMEngineMixin` 并删除原实现。
3. 新增定向结构测试保护 Mixin 来源、MRO、无反向导入和旧入口。
4. 运行定向 CEWM 回归、静态门禁与全量 pytest。
5. 走 PR、CI 与 governance 合并。

## 6. 边界条件
- [x] `observation=None`、`goal=None`、`action=None` 行为不变
- [x] 安全违规、紧急停止、降级模式提前返回行为不变
- [x] replay 阈值、步数、间隔与异常处理时序不变
- [x] 状态解析回退和因果边提取行为不变

## 7. 异常场景
`SafetyMonitor`、`CausalUpdater`、经验库、JEPA 预测、ActionGap 和状态解析的异常类型、捕获范围与日志级别不变；禁止新增静默吞错。

## 8. 性能约束
Mixin 不新增运行时间接调用；除方法查找外，CEWM 主路径保持相同。

## 9. 安全约束
不改变 SafetyMonitor、紧急停止、降级模式与状态快照语义；不新增网络、持久化或密钥。

## 10. 必须覆盖测试用例
1. 正向：13 个方法仍可通过 `MCIWorldModel` 调用。
2. 结构：MRO 包含 `CEWMEngineMixin`，方法归属新模块。
3. 行为：`cewm_step` 与 `cewm_step_fast` 的既有 E2E / 子方法 / phase3 / replay 测试通过。
4. 边界：状态解析、安全检查、replay 与降级路径保持不变。
