# P0-2 世界模型拆分结构化需求

## 1. 业务目标
把 3,834 行的 `_world_model.py` 收敛为可独立验证、可维护的编排层，避免后续 JEPA、CEWM、因果推理变更继续集中碰撞。

## 2. 输入参数定义
| 字段 | 类型 | 必填 | 默认 | 校验规则 |
| --- | --- | --- | --- | --- |
| 外部导入路径 | str | 是 | 无 | 保持 `_world_model` 现有符号导入不变 |
| 类字段与行为 | API | 是 | 无 | 数据类字段、方法签名、序列化结果零变更 |
| 验证范围 | str | 是 | full | 至少覆盖定向回归、mypy strict、全量 pytest |

## 3. 输出数据结构
P0-2a 新建 `_world_model_state.py`，迁出 `CausalWorldModelState`、`TrajectoryStep`、`WorkingMemory`；`_world_model.py` re-export 全部三个符号。文件行数从 3,834 降至约 3,250。

## 4. 依赖资源
依赖现有 NumPy、dataclasses、typing 与 `_sys` 层类型；状态模块禁止反向导入 `_world_model.py`，避免循环依赖。

## 5. 业务流程步骤
1. 迁移三个状态类到 `_world_model_state.py`。
2. `_world_model.py` 保持旧导入路径 re-export。
3. `_world_model.py` 内部方法改从状态模块导入。
4. 运行定向回归、静态门禁与全量测试。
5. 以 PR 提交，等待 CI 与 governance。

## 6. 边界条件
- [x] 空值构造、`empty()`、`from_dict()`、序列化行为不变
- [x] 字段默认值与私有字段不变
- [x] 类型标注保持 strict 可通过
- [x] 线程语义不改变

## 7. 异常场景
数据类校验错误、反序列化旧格式升级、缺省字段补齐行为必须保持不变；禁止静默吞掉新异常。

## 8. 性能约束
拆分只移动代码，不新增运行时间接调用；导入初始化延迟不应显著增加。

## 9. 安全约束
不引入密钥、网络访问或数据持久化；医疗与敏感字段语义不变。

## 10. 必须覆盖测试用例
1. 正向：三个类可继续从 `_world_model` 导入，`isinstance` 与字段访问不变。
2. 边界：`empty()`、最小字段、默认值、旧 dict 兼容。
3. 异常：非法字段与反序列化失败行为不变。
