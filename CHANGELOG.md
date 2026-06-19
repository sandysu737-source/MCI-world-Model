# Changelog

所有本项目显著变更都记录在此文件。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [4.0.0] - 2026-06-03

### 🎉 CEWM (Cognitive-Enhanced World Model) 完整迭代交付

**v4.0.0 是 CEWM 五阶段迭代的最终交付版本**，在 v3.3.0 基础上完成 Phase 1-5 全部特性。

### Added — Phase 1-5 全量特性集

**Phase 1 (v3.4.0) — 四元融合因果建模体系:**
- 能量中心因果发现 (CausalDiscovery) + 证据累积 (EvidenceAccumulator)
- 四元融合量化: 能量×因果×时间×语义四维统一
- 测试: 1765 passed

**Phase 2 (v3.5.0) — _sys→sdk 能力释放:**
- 能量中心能力从 `_sys/` 释放至 `sdk/` 层公开 API
- EnergyCenterOrchestrator + EnergyFlowOptimizer
- 测试: 1813 passed

**Phase 3 (v3.6.0) — 认知环闭环:**
- Perception → WorldModel → Configurator → Cost → Actor → STM 六模块闭环
- CognitiveScorecard 六维评分 + CognitiveHealthMonitor
- 测试: 1870 passed

**Phase 4 (v3.7.0) — 认知诊断系统:**
- **MetaDiagnoser** (`sdk/_meta_diagnoser.py`): 学习型认知诊断，10 种 FailurePattern + SurpriseSignal + 六维健康度
- **NegativeHeuristic** (`sdk/_negative_heuristic.py`): Lakatos 负面启发法，7 条硬核规则 (HC-1~HC-7)
- **HierarchicalConfigurator 升级**: multi_objective_optimize() 三目标优化 + diagnose_and_configure()
- 测试: 1916 passed

**Phase 5 (v4.0.0) — 基准验证体系:**
- **CEWM 六维认知基准** (`benchmarks/cognitive/test_cewm_cognitive.py`): D1-D6 30 个测试
- **噪声鲁棒性基准** (`benchmarks/test_noise_robustness.py`): N1-N4 + MSE 13 个测试
- **临床营养验证** (`benchmarks/clinical/test_clinical_nutrition.py`): C1-C4 12 个测试，10 个临床案例
- 基准测试: 76 passed

### Fixed — _sys/ 模块历史技术债

- **EnergyBus 测试**: 节点 ID 从 `wuxing_*` 修正为 `element_*`
- **TemporalCore.get_cycle_name()**: 支持取模回绕 (index 60 → 0)
- **CategoryCore/DimensionMap 测试**: energy_type 断言对齐 CEWM 认知能量类型
- 全部 4 个 xfail 移除，0 xfailed

### 📊 核心代码指标

- 测试套件: **1920 passed**, 0 xfailed, 0 failed
- 基准测试: **76 passed**
- 新增组件: MetaDiagnoser / NegativeHeuristic / SurpriseSignal + 15 个导出符号
- 三层导出: `_sys/__init__.py` → `sdk/__init__.py` → `__init__.py`

## [3.3.0] - 2026-06-03

### Added — 多模态因果世界模型 + Phase 1 P0 三大特性

**Phase 1 P0 — 三大新特性:**
- **MultiBranchPredictor** (`sdk/_multi_branch_predictor.py`): 多分支未来推演引擎，复用 `rollout()` + `BatchCounterfactualEngine`
- **SurpriseDetector** (`sdk/_surprise_detector.py`): 惊奇误差三维度量化 (state_distance 0.4 + vector_deviation 0.3 + direction_error 0.3)
- **PlanAgent** (`sdk/_plan_agent.py`): “先模拟后执行”因果决策前置化，包裹 CausalActor + MultiBranchPredictor + SurpriseDetector

**v3.3.0 — 多模态感知:**
- **VisionEncoder / AudioEncoder / ThermalEncoder** (`sdk/_modality_encoders.py`): 纯 numpy 模态特征提取器（零外部依赖）
- **MultimodalFusion** (`sdk/_multimodal_fusion.py`): 三种融合策略 (attention / weighted / concat)
- **MultimodalGraphBuilder** (`sdk/_multimodal_graph_builder.py`): 多模态特征时序 → 跨模态因果图
- **MultimodalWorldState** (`sdk/_world_state.py`): 多模态世界状态子类（5 个可选模态字段）
- **感知管道多模态扩展**: `process_multimodal_fused()` + IMAGE / AUDIO_FEATURES 信号分派
- **SignalType 枚举扩展**: IMAGE = "image" / AUDIO_FEATURES = "audio_features"
- **_infer_state_class() 注册表模式**: 支持多模态状态自动推断

### Changed

- `to_multimodal()` 映射补全: ENCODER_POSITION / RGB_FRAME / DEPTH_FRAME / THERMAL_FRAME
- 测试套件从 1537 个扩展到 1612 个

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

## [4.3.3] - 2026-06-03

### 🔧 CEWM v4.0.0 代码审查修复 — 21 个问题全量修复

基于 `docs/CEWM-v4.3.3-代码审查修复实施计划书.md` 的系统化代码审查修复，
覆盖 5 Critical + 7 Warning + 9 Suggestion 共 21 个问题。

测试: **3568+ passed**（从 3505 增长 +63 个新测试）

---

#### Phase 0: 止血修复 — Critical Issues（5 个）

- **C-1 因果图重建 Bug**: `cewm_step()` 每次 `CausalUpdater()` 重建 → 改为 `__init__` 初始化 + 增量更新。
  修复后因果图跨步骤积累，`causal_updates` 连续递增。
- **C-2 JEPA 字符串查询 Bug**: `str(state)` 作为 JEPA 查询 → 改为 `state.to_vector()` 嵌入查询。
  修复后 JEPA 预测非空且非全零。
- **C-3 后端推断维度碰撞**: `_infer_backend()` 仅按维度推断 → 引入 `state_cls` 精确匹配 + 维度回退。
  PendulumState(2D) → "pendulum"，CartState(2D) → "cart"，不混淆。
- **C-4 函数过长**: `cewm_step()` 158 行 → 拆分为 6 个子方法（≤30 行/个）。
- **C-5 硬编码类型分支**: `_cewm_state_change()` `hasattr(state, "theta")` → `state.causal_edges()` 多态。

#### Phase 1: 泛化修复 — Warnings + 泛化任务（6 个）

- **W-1 异常静默**: `cewm_step_fast()` 裸 `except: pass` → `logger.debug()` 记录异常。
- **W-2 Pendulum 硬编码回退**: `_action_gap.py` PendulumState 物理模拟回退 → 移除硬编码，返回原状态 + warning。
- **W-3 双摆耦合项**: 两个独立单摆近似 → 完整拉格朗日方程耦合项。
- **W-4 dataclass 可变状态**: `MetaDiagnoser` 裸 `list` 默认值 → `field(default_factory=list)`。
- **GEN-05 CartState 解析器**: 新增 `CartStateParser`，StateParserRegistry 支持 CartState 自动解析。
- **GEN-06 DoublePendulumState**: 新增 `DoublePendulumState` 类 + 双摆耦合动力学测试。

#### Phase 2: 闭环验证 — Warnings + Suggestions（7 个）

- **W-5 StateParserRegistry 优先级**: 逆序优先级歧义 → 显式 `priority: int` 数字排序。
  Pendulum(30) > Cart(20) > Generic(10)。
- **W-7 NegativeHeuristic 内容检查**: 仅检查 `change_type` → 新增内容级双重检测
  （关键词意图识别 + 参数归零检测）。
- **S-1 延迟初始化统一**: `hasattr` 检查 → `__init__` 声明 `None` + `is None` 标准模式。
  补全 `_action_gap_metric` + `_state_parser_registry` 声明。
- **S-3 to_dict() 序列化完整**: 新增三维索引状态（semantic/causal/temporal）+ 内部计数器
  （consolidation_count/forget_count/id_counter）。
- **S-6 CausalUpdater.reset()**: 新增 `reset()` 完全重置方法（清空 edges/nodes/history/stats），
  区别于已有的 `reset_stats()`。
- **S-7 子方法单元测试**: 新增 `test_cewm_submethods.py`（27 测试覆盖全部 8 个 `_cewm_*` 子方法）。
- **LOOP-07 E2E 闭环测试**: 新增 `test_cewm_e2e_closed_loop.py`（18 测试覆盖五层全链路闭环）。

#### Phase 3: 质量提升 — Suggestions + 文档（5 个）

- **S-4 Ashby 阈值量化**: `CognitiveDiversity` 新增 `is_sufficient_diversity()` 布尔门禁方法 +
  `ashby_ratio` 属性暴露。阈值 `ASHBY_SUFFICIENCY_THRESHOLD = 1.0`。
- **S-5 Pendulum 硬编码 CI 追踪**: 新增 `scripts/track_pendulum_hardcode.py`，
  支持 `--ci`（超阈值阻断）和 `--json` 输出模式，白名单豁免物理引擎。
- **S-8 模块拆分设计文档**: 新增 `docs/CEWM-_world_model.py模块拆分设计文档.md`，
  定义 4 模块拆分方案（Mixin 模式）+ 迁移步骤 + 向后兼容策略。
- **S-9 Wiener 四环可视化**: 新增 `docs/CEWM-Wiener四环嵌套反馈闭环架构.md`，
  含 3 张 Mermaid 图 + 跨层误差传播详解 + 代码对应关系。
- **QUAL-05 变更日志**: 本条目。

### 修复统计

| 阶段 | 问题数 | 新增测试 | 状态 |
|------|--------|----------|------|
| Phase 0: 止血 | 5 (C-1~C-5) | 0 | ✅ 完成 |
| Phase 1: 泛化 | 6 (W-1~W-4, GEN-05~06) | 0 | ✅ 完成 |
| Phase 2: 闭环 | 7 (W-5, W-7, S-1, S-3, S-6, S-7, LOOP-07) | +45 | ✅ 完成 |
| Phase 3: 质量 | 5 (S-4, S-5, S-8, S-9, QUAL-05) | 0 | ✅ 完成 |
| **合计** | **23** | **+45** | **3568+ passed** |

[4.3.3]: https://github.com/sandysu737-source/mci-world-model/releases/tag/v4.3.3
[3.0.0]: https://github.com/sandysu737-source/mci-world-model/releases/tag/v3.0.0
[3.3.0]: https://github.com/sandysu737-source/mci-world-model/releases/tag/v3.3.0
[4.0.0]: https://github.com/sandysu737-source/mci-world-model/releases/tag/v4.0.0
