# Changelog

所有本项目显著变更都记录在此文件。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [3.0.0] - 2026-06-03

### 🎉 重大里程碑：从 su-memory-sdk 独立

MCI World Model **V3.0.0** 是世界模型引擎**正式独立成仓**的第一个发布版本。
原 su-memory-sdk V3.5.x 内部的 World Model 模块（Pearl 三层 + JEPA + 能量中心）整体迁移至本仓库。

### ✨ 新增

- **顶层统一入口** `mci_world_model`：导出 `_sys` / `sdk` / `world_model` 全套关键符号
- **品牌门面**：从「**MCI World Model SDK**」去 SDK 后缀，对外统一为「**MCI World Model**」
- **包名规范化**：`su_memory.*` → `mci_world_model.*`（发布门禁预处理）
- **SIGReg 基准套件**：`benchmarks/sigreg/{prepare,back_projection,lambda_sweep}.py`
- **V3.0.0 路线图**：`docs/ROADMAP_V3.0.0.md`
- **CI 4-Gate**：`.github/workflows/ci.yml`（ruff / mypy / pytest / sigreg-bench）

### 🧬 模块清单

| 子包 | 角色 | 关键符号 |
|------|------|---------|
| `mci_world_model._sys` | 系统层（能量中心） | YinYang / ThreePowers / EnergyCore / CausalEngine |
| `mci_world_model.sdk` | 推理层（Pearl + JEPA） | MCIWorldModel / DoCalculus / CounterfactualEngine / JEPA* |
| `mci_world_model.world_model` | 三才合一统一入口 | WorldModel / 状态合并 |
| `mci_world_model.benchmarks` | SIGReg 基准 | _noise_generator / sigreg/* |

### ⚠️ 破坏性变更（Breaking Changes）

- 包名 `su_memory` → `mci_world_model`（所有 `import su_memory.sdk._world_model` 必须改为 `import mci_world_model.sdk`）
- 顶层符号路径 `su_memory.world_model.MCIWorldModel` → `mci_world_model.sdk.MCIWorldModel`
- Python ≥ 3.10（沿用 su-memory-sdk 既有基线）

### 📦 版本映射

| 项目 | 版本 | 角色 |
|------|------|------|
| su-memory-sdk | V3.5.1 | 记忆引擎（保持原状，独立仓库） |
| MCI World Model | V3.0.0 | 世界模型（本仓库，原 V4.0.0 内部迭代） |

### 🗑️ 清理

- V3.5.0 / V3.4.0 论文主文件及对应 `.tex` / `.html` 资产（属于 su-memory-sdk 记忆引擎遗产，不纳入本项目）
- v3.1 / v3.2 / v3.3 规划书（属于早期迭代，不纳入本项目）
- `arxiv/` / `jmlr/` / `uai/` 论文子目录（提交材料遗留）
- `codalab/` / `codalab_submission/` 提交材料
- `node_modules/` 前端构件（不属本项目）
- `build/` / `dist/` / `__pycache__/` / `.mypy_cache/` / `.ruff_cache/` / `.pytest_cache/`（构建与缓存产物）

### 📊 核心代码指标

- 32 个 Python 源文件，~9000 行业务代码
- `world_model.py`（350 行，三才合一统一入口）
- `sdk/_world_model.py`（1739 行，MCIWorldModel 核心类）
- `sdk/_do_calculus.py`（1001 行，Pearl 干预层）
- JEPA 全套（6 个文件，2165 行）
- `sdk/_spectral_causal.py`（1464 行，频谱因果引擎）
- `sdk/bayesian_augmenter.py`（1146 行，贝叶斯增强器）
- `_sys/` 系统层（15 个文件，30+ 系统符号）

[3.0.0]: https://github.com/sandysu737-source/mci-world-model/releases/tag/v3.0.0
