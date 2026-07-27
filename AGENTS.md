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

---

## Git 工作流与收尾纪律（硬性约束）

> 本节由全项目 Git 治理统一制定，适用于所有 AI 代理与人类开发者。

### 分支规范

- **主分支**：`main`（唯一长期分支，保持可发布状态）。
- **功能分支**：`feat/<简述>`、`fix/<简述>`、`refactor/<简述>`。
- **AI 代理分支**：`codex/<任务简述>`，**任务结束即删，不得长期留存**。
- 禁止在 `main` 上直接提交大段改动；禁止推 `-temp`、`-backup`、`-test` 等无意义分支名。

### 提交规范（Conventional Commits）

- 格式：`<type>(<scope>): <简述>`，type ∈ `feat | fix | chore | docs | refactor | test | style | perf | ci`。
- 中文简述即可，一行 ≤72 字；一个逻辑变更一次 commit，禁止「大杂烩」提交。
- 提交前确认 `user.name` / `user.email` 为真实身份（本仓库已统一为 `sandysu737-source` / `sandysu737@gmail.com`）。

### 严禁入库的内容（体积/安全红线）

以下内容**绝对禁止** `git add`，违反将导致 `.git` 膨胀或密钥泄露：

| 类别 | 示例 | 处置 |
|------|------|------|
| 模型权重 | `*.safetensors` `*.gguf` `*.bin` `*.pt` `*.pth` `*.onnx` | 用对象存储 / 模型仓库托管 |
| 虚拟环境 | `.venv/` `node_modules/` `venv/` | 本地生成，不入库 |
| 数据库 | `*.db` `*.sqlite3` `data_persist/` | 备份目录单独管理 |
| 构建产物 | `dist/` `build/` `__pycache__/` `.mypy_cache/` | gitignore 兜底 |
| 密钥凭据 | `.env` `*.pem` `*.key` `secrets/` | 仅提交 `*.example` 模板 |
| 大二进制 | `*.whl` `ollama-models/` `vendor/models/` | 外部依赖管理 |

> 已由全局 `~/.gitignore_global` 兜底排除；若误提交，须 `git rm --cached` 后从历史清除。

### AI 代理任务收尾清单（每次任务结束必须执行）

任何 AI 代理（含 Codex/Claude/Qoder）在完成任务、准备返回前，**必须**依次完成：

1. **切回主分支**：`git checkout main`（不得把 HEAD 遗留在 `codex/*` 分支上）。
2. **合并或删除工作分支**：有价值的工作合并进 `main`（优先 `git merge --ff-only` 或 `--rebase`）；无价值的 `codex/*` 分支立即 `git branch -D` 删除。
3. **清理工作树**：`git status` 必须为 clean；临时文件、调试产物、`.DS_Store` 全部移除。
4. **确认无大文件误提交**：提交前自查 `git status` 中无模型/数据库/环境目录。
5. **留下可读提交**：最后一次 commit 信息须能说明本次任务成果，不得用 `wip` / `tmp` / `update`。

> **违规判定**：若发现 `codex/*` 残留分支、HEAD 停在非 main 分支、工作树脏、或大文件入库，视为任务未完成，触发 ai-guard 拒绝合入。

### 同步与历史卫生

- 开工先 `git pull --rebase`；收工前 push。
- 历史保持线性（已全局配置 `pull.rebase=true`）。
- 每月一次 `git gc --prune=now` 回收对象（项目维护者执行）。
