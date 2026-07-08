# AGENTS.md — MCI World Model · AI 协作宪法

> 本文件是所有 AI 编程代理在本仓库工作的**硬性约束**。优先级高于个人偏好与通用最佳实践。
> 人类直接指令 > 本文件 > 通用约定。冲突时以人类指令为准。

## 1. 项目概览

MCI World Model — 因果世界模型引擎（Pearl 三层 + JEPA 世界建模 + 能量中心）。
独立运行的通用 AI 系统"CPU"，不依赖 Transformer/GPU（后者为可选加速器）。

| 维度 | 技术 |
|------|------|
| 语言 | Python 3.10–3.13 |
| 计算 | NumPy / PyTorch（纯可微世界预测器） |
| 依赖管理 | pyproject.toml |
| 测试 | pytest |
| Lint | ruff + mypy |

**领域**: 因果推断（关联/干预/反事实）、世界建模、研究型项目。**可复现性敏感，禁止捏造实验结果**。

## 2. 硬性规则（违反即返工）

### 2.1 代码改动
- **禁止臆造/捏造结果**: 不确定的逻辑先 `rg` 查证；实验结果必须真实可复现，禁止 silently 改 seed 或数据以"跑通"。
- **可复现性**: 所有随机源必须设 seed（random/numpy/torch）；数据/权重记录版本与来源。
- **最小改动**: 只改任务要求的部分，不顺手重构。
- **配置与代码分离**: 学习率/批次/维度走 config，禁止魔法值。
- **风格一致**: 遵循 ruff/mypy，不擅自引依赖。

### 2.2 质量红线
- **禁止删测试以通过 CI**。
- **数值健壮性**: 关键计算用 assert 验证形状/范围/有限性（防 NaN/Inf）。
- **Notebook 探索 → 模块沉淀**: 正式逻辑必须从 notebook 沉淀为 .py 模块 + 测试。

## 3. 验证命令（改动后必须跑通）

```bash
ruff check . && ruff format --check .
mypy .
pytest
bash scripts/ai-verify/ai-guard.sh
```

## 4. 架构约定

- **Pearl 三层**: 关联/干预/反事实因果推断须完整覆盖。
- **JEPA**: 纯 NumPy/PyTorch 可微世界预测器。
- **与 su-memory-sdk 关系**: 本引擎是 CPU，记忆引擎是独立组件，接口清晰分离。

## 5. 工作流约定

- 写正式模块前填 `.ai-templates/structured-requirement.md`。
- 提交前 ai-guard 全绿；单次提交 ≤800 行。
- 不自动 commit/push，改完汇报由人类决定。

## 6. 目录关键路径

```
src/ 或 包名/    核心模块（因果/世界模型/能量中心）
adapters/        外部模型/数据源适配
benchmarks/      基准测试
scripts/         可运行入口（训练/评估）
tests/           pytest
notebooks/       探索用（不进生产路径）
scripts/ai-verify/ AI 工程化守卫
```

详细规范见 `.ai-rules.md`。
