# Nightly zvec 兼容修复结构化需求

## 1. 业务目标
修复 Python 3.11 Nightly Full Quality Gate 中 `_zvec_store.py` 的 mypy 失败，并确保可选依赖 `zvec 0.7` 的存储与查询行为兼容。

## 2. 输入参数定义
| 字段 | 类型 | 必填 | 默认 | 校验规则 |
| --- | --- | --- | --- | --- |
| 可选依赖 | dependency | 是 | 无 | `full`、`small_model`、`zvec` extra 声明支持当前代码使用的 zvec API |
| 存储路径 | str | 是 | 无 | 只创建父目录，`zvec.create_and_open()` 自己创建存储路径 |
| 向量查询 | API | 是 | 无 | 使用 `Query(vector=...)`，`topk` 通过 `Collection.query()` 传递 |
| 验证范围 | str | 是 | full | Python 3.11 mypy + zvec 定向测试 + 本地 `check.sh` |

## 3. 输出数据结构
`ZvecEmbeddingStore._collection` 显式标注为 `Any | None`；`docs` 标注为 `list[Any]`；新增 `tests/test_zvec_store.py` 覆盖当前 API。运行行为仍保持“zvec 可用则用 zvec，异常则 fallback”的既有策略。

## 4. 依赖资源
依赖 `zvec>=0.7.0`。`_dense_retriever.py` 已使用 0.7 API，本次将 `_zvec_store.py` 与该权威用法对齐。

## 5. 业务流程步骤
1. 在隔离 Python 3.11 + zvec 0.7 环境复现 mypy 错误。
2. 修复 `_collection` 类型、旧 Query 语法与存储路径预创建。
3. 新增 zvec 插入/向量检索回归测试。
4. 运行静态检查、3.11 mypy、定向测试与全量 `check.sh`。
5. 以 PR 提交，等待 CI 与 governance。

## 6. 边界条件
- [x] 未安装 zvec 时测试跳过，运行时继续 numpy fallback
- [x] zvec 初始化异常时 `_collection=None`，不改变异常处理策略
- [x] `topk` 上限 `500` 保持不变
- [x] 插入失败仍仅使用 fallback 并保留原日志语义

## 7. 异常场景
`create_and_open`、插入、查询异常的类型与捕获时序不变；不新增静默吞错。

## 8. 性能约束
仅修正 API 调用与类型，不引入额外缓存、持久化或运行时间接层。

## 9. 安全约束
不新增网络、密钥或数据外发；测试数据存放在 pytest 临时目录。

## 10. 必须覆盖测试用例
1. 插入 2 个 QA 对后 `n_docs=2`。
2. 当前 zvec API 向量检索返回 1 条且字段完整。
3. `_dense_retriever.py` 既有 8 个 zvec 测试继续通过。
4. Python 3.11 + zvec 0.7 下相关 mypy 通过。
