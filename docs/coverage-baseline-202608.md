# 覆盖率基线报告（2026-08-20）

> **基线版本**: v4.6.0 + T2 配置（`main`）
> **测量命令**: `.venv/bin/python -m pytest -q --cov=mci_world_model --cov-branch --cov-report=term`
> **结果**: 4628 passed / 2 skipped（802s）

## 基线数据

| 指标 | 值 |
|------|-----|
| 行覆盖 | **77.57%**（36,309 语句 / 6,972 未覆盖） |
| 分支覆盖 | **85.1%**（10,220 分支 / 1,524 未覆盖） |
| 旧阈值（pyproject） | 53（历史遗留，过低） |
| 旧阈值（CI） | 45（命令行覆盖，更低） |

## 门禁升级路线

| 阶段 | fail_under | 目标 | 状态 |
|------|-----------|------|------|
| 阶段 0 | 53/45 | 历史遗留 | 已废弃 |
| 阶段 1 | 77 | 当前实测基线，立即生效 | ✅ 2026-08-20 生效 |
| 阶段 2 | 80 | 补测 `_world_model.py`(68%)/`server`(69%) 等缺口 | 待 T2 补测 |
| 阶段 3 | 85 | 分支对齐 + 核心模块 85%+ | 待定 |
| 阶段 4 | 90 | ai-guard L2 目标（核心≥95/分支≥90） | 待定 |

## 低覆盖模块清单（阶段 2 补测目标）

| 模块 | 覆盖 | 缺口特征 | 处置 |
|------|------|----------|------|
| `sdk/_world_model.py` | 68% | 3,771 行大文件，421 行未覆盖 | 随 T3 拆分同步补测 |
| `server/app.py` | 71%（+2pp） | HTTP 安全路径 | 2026-08-20 已补认证 401×2/限流 429/404 用例（`tests/test_server_api.py::TestSecurityEndpoints`） |
| `sdk/_zvec_store.py` | 21% | **zvec 未安装**（可选依赖，非真实缺口） | 装 zvec 后复测 |
| `sdk/_persistent_memory.py` | 需查 | 持久化路径 | 补测 |
| 其余 200+ 模块 | ≥78% | — | 维持 |

## 验证命令（可复现）

```bash
.venv/bin/python -m pytest -q --cov=mci_world_model --cov-branch --cov-report=term
# 末尾应显示 "Required test coverage of 77.0% reached. Total coverage: 77.57%"
```
