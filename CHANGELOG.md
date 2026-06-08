# Changelog

所有本项目显著变更都记录在此文件。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [3.2.0] - 2026-06-03

### Added — WorldState 通用抽象 + Action-Conditioned JEPA + PhysicalSignal 感知通路

- **WorldState ABC** (`sdk/_world_state.py`): 四方法契约 `to_vector/from_vector/distance/copy`，独立于推理元数据
- **PendulumState**: 单摆物理世界最简验证器 — `(theta, omega)` 2D 状态 + Euler 积分 + `from_signals()` 感知桥接
- **Action ABC + PendulumAction**: 抽象动作基类 + 单摆力矩实现（函数式语义，不修改原状态）
- **ActionConditionedPredictor** (`sdk/_action_conditioned_predictor.py`): `predict(state, action, n_steps) -> list[WorldState]` 动作条件化多步预测
- **PendulumPhysicsPredictor**: Euler 积分物理金标准（零误差）
- **PendulumJEPAPredictor**: 线性 MLP + 最小二乘训练（8 参数）
- **PhysicalSignal 感知通路**: `SensorModality` 六感 + `SignalSubType` 17 种 + `PerceptionPipeline.process_physical()`
- **四环闭环验证**: 感知→认知→预测→行动端到端闭环，A1-A4 验收标准全过
- **BatchCounterfactualEngine**: 向量化批量反事实推理
- **EnhancedPerception**: LLM 增强感知管道（文本→信号→状态）
- **EnergyFlowPredictor**: 能量流预测器
- **_sys/ 覆盖率补全**: 97% causal.py / 100% awareness.py / 93% evidence.py / 90% configurator.py / 79% states.py / 75% energy_core.py
- **Benchmark 扩展**: JEPA 训练基准 + 因果推理基准 + 性能基线基准（46 benchmarks）
- **cov_fix.py**: 修复 coverage.py 7.x + numpy 2.x C 扩展双加载冲突

### Changed

- `CausalWorldModelState` 新增 `world_state` 桥接字段，连接新旧架构
- 测试套件从 905 个扩展到 1462 个

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

## [3.0.7] - 2026-06-03

### ✨ 新增

- **CausalMLP**: 小型因果推断 MLP (~15K params)，纯 numpy+scipy 实现，彻底移除 torch/transformers/peft 硬依赖
- **ParametricMemory**: 基于 CausalMLP 的参数化记忆训练引擎，支持 prepare/train/predict/save/load
- **SimpleTextEmbedder**: 基于字符 n-gram 哈希的轻量文本嵌入器（零外部依赖）

### 🗑️ 清理

- 移除 Qwen2.5-1.5B + QLoRA 桩实现（`_parametric_memory.py` 中的 torch 路径）
- 移除 peft 导入残留

## [3.0.8] - 2026-06-03

### ✨ 新增

- **CounterfactualEngine**: Pearl L3 反事实推理引擎（结构化方程模型 SEM）
- **BatchCounterfactualEngine**: 批量反事实查询（蒙特卡洛采样）
- **CG↔SEM 双向转换**: [to_sem()](file:///Users/mac/qoder m5pro/mci-world-model/src/mci_world_model/sdk/_do_calculus.py#L207) + [from_sem()](file:///Users/mac/qoder m5pro/mci-world-model/src/mci_world_model/sdk/_do_calculus.py#L242)

### 🔧 改进

- `CausalWorldModelState` 扩展：`node_names` 字段支持 SEM 节点名传递

## [3.1.0] - 2026-06-03

### ✨ 新增

- **多模态信号体系**: `SignalType` 枚举 (5 种) + `MultimodalSignal` 数据结构
- **PhysicalGraphBuilder**: 物理量→因果边转换器，含滞后相关检测 (1-7 天窗口)
- **JEPAEncoder 物理路径**: `encode(signals=...)` 支持多模态信号输入
- **临床营养基准测试**: 100 患者 × 30 天的合成数据 + 20 项因果推理测试

### 🔧 改进

- `PerceptionPipeline.process_multimodal()`: 5 种信号分派处理器
- `ENERGY_PHYSICAL_MAP`: 五范畴→物理量名称映射

### 🛡️ QC 审计 (V3.0.7-V3.1.0 三迭代)

- **综合评级**: B+ (3.83/5) → 修复后 A (4.5+)
- **P1 修复**: 异常静默吞噬 (5 处 logger.debug→logger.warning)
- **P2 修复**: 测试覆盖 (1220 行核心代码 0→23 项测试)
- **P2 修复**: 版本号语义混淆 (v3.7.0/v3.8.0→Pearl L2/L3)
- **新增**: coverage 门禁 (fail_under=45%), pytest marker 扩展
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
