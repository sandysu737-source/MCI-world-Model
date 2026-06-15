# CEWM 架构加固与泛化 — KPI 实施规划书

**版本**: V1.0
**编制日期**: 2026-06-03
**编制依据**: CEWM v4.3.3 逐模块能力审计 + Pendulum 硬编码架构债务诊断 + LLM-CEWM 集成深度评估
**前置规划**: CEWM v4.0.0 迭代计划书 (已完成) + MCI 医疗世界模型发展规划书 v3.2.0
**目标基线**: CEWM v4.3.3 → v4.5.0 = 从 Pendulum 验证器升级为泛化世界模型引擎

---

## 目录

1. [总体目标与 KPI 体系](#一总体目标与-kpi-体系)
2. [KPI 指标定义与基线](#二kpi-指标定义与基线)
3. [四阶段实施路线图](#三四阶段实施路线图)
4. [Phase 0: 止血 — 抽象 Pendulum 硬编码依赖 (第 1-2 周)](#四phase-0-止血--抽象-pendulum-硬编码依赖)
5. [Phase 1: 泛化 — 通用物理预测器 + 多模态管线集成 (第 3-5 周)](#五phase-1-泛化--通用物理预测器--多模态管线集成)
6. [Phase 2: 闭环 — LLM↔CEWM 反馈 + 安全约束 (第 6-8 周)](#六phase-2-闭环--llmcewm-反馈--安全约束)
7. [Phase 3: 就绪 — 手术机器人桥接 + 硬实时保证 (第 9-12 周)](#七phase-3-就绪--手术机器人桥接--硬实时保证)
8. [门禁检查机制](#八门禁检查机制)
9. [角色与分工](#九角色与分工)
10. [风险管理与应急预案](#十风险管理与应急预案)

---

## 一、总体目标与 KPI 体系

### 1.1 愿景

> 将 CEWM 从"在单摆上验证的世界模型理论框架"升级为"可泛化到任意物理/机器人场景的因果增强引擎"，解除 Pendulum 硬编码这一系统性架构债务，让 Pearl 因果推理能力真正服务于通用智能体。

### 1.2 与既有规划的关系

```
CEWM v4.0.0 迭代计划 (已完成)        本规划书 (v4.3.3 → v4.5.0)
═══════════════════════════          ════════════════════════════════
聚焦：认知能力（因果/记忆/诊断）       聚焦：架构泛化（解除 Pendulum 硬编码）
产出：CognitiveLoopBus,             产出：PredictorProtocol, 
      ExperienceDB,                       GeneralizedPhysicsPredictor,
      MetaDiagnoser,                      LLM↔CEWM Feedback Loop,
      ActionGapMetric,                    ROS2 Bridge 预研
      PlanAgent, ...
状态：已交付 v4.3.3                    状态：待实施

MCI 医疗世界模型规划 v3.2.0           本规划书
═══════════════════════               ════════
聚焦：医疗模拟器（ClinicalWorldState） 聚焦：底层引擎泛化（让医疗模拟器有
产出：ClinicalWorldState,             可替换的物理后端）
      ActionConditionedPredictor,
      SignalParser, ...
状态：远期规划                         状态：架构前置条件
```

**本规划书是 CEWM 从"可以工作"到"可以泛化"的关键桥梁。它不重复 v4.0.0 的认知能力建设，不替代 v3.2.0 的医疗领域建模，而是填补两者之下的架构地基。**

### 1.3 核心问题诊断

基于 v4.3.3 逐模块审计，识别出以下系统性架构债务：

| 问题 | 严重度 | 影响范围 | 当前状态 |
|------|--------|----------|----------|
| **Pendulum 硬编码** | 🔴 P0 | 整个 `sdk/` 预测-规划-行动链路 | `PendulumPhysicsPredictor` / `PendulumAction` / `_cewm_parse_state()` 均只接受 PendulumState |
| **WorldState 多态缺失** | 🔴 P0 | 状态解析、距离计算、动作模拟 | `_cewm_parse_state()` 只解析 `theta/omega`，非 Pendulum 状态无法参与闭环 |
| **LLM→CEWM 单向** | 🟡 P1 | Agent 决策质量 | `OrchestratorBridge` 仅做 intent→CEWM 调用，无反事实结果回注 LLM |
| **多模态融合未集成** | 🟡 P1 | 感知管线 | `MultimodalFusion` 为独立工具类，`from_vector()` 丢失模态结构 |
| **手术机器人零代码** | 🟡 P2 | 机器人应用场景 | 无 ROS 桥接、无 6-DOF 状态空间、无安全约束层 |
| **硬实时缺失** | 🟡 P2 | 安全关键系统 | 无 WCET 分析、无 deadline 监控、无紧急停止集成 |

### 1.4 KPI 体系架构

```
                  ┌──────────────────────────────────────────┐
                  │      CEWM 架构泛化成熟度 L0 → L3          │
                  │    （终极目标：任意 WorldState 可插入）     │
                  └──────────────────┬───────────────────────┘
                                     │
       ┌─────────────────────────────┼─────────────────────────────┐
       │                             │                             │
┌──────▼──────┐              ┌──────▼──────┐              ┌───────▼──────┐
│  泛化 KPI   │              │  集成 KPI   │              │  就绪 KPI    │
│             │              │             │              │              │
│ 状态多态率  │              │ LLM反馈闭环 │              │ ROS桥接度    │
│ 预测器可替换│              │ 融合管线集成│              │ 安全约束覆盖  │
│ 动作空间多样性│            │ 消融可追溯  │              │ 实时性保证   │
└─────────────┘              └─────────────┘              └──────────────┘
```

### 1.5 核心原则

| 原则 | 说明 |
|------|------|
| **零回归** | 所有 Phase 改动不破坏现有 1,950+ 测试，Pendulum 闭环持续可用 |
| **协议优于实现** | 通过 Protocol/ABC 定义接口契约，Pendulum 降级为一个实现 |
| **消融可追溯** | 每个 feature 有独立 CLI flag，可对比 Pendulum vs 泛化模式 |
| **渐进式解耦** | 不一次性重写，每 Phase 替换一条硬编码路径，验证→再推进 |
| **手术机器人诚实定位** | Phase 3 标注为"桥接预研"，不宣称生产就绪 |

---

## 二、KPI 指标定义与基线

### 2.1 泛化类 KPI

| KPI ID | 指标名称 | 当前基线 (v4.3.3) | Phase 0 目标 | Phase 1 目标 | Phase 2 目标 | Phase 3 目标 | 测量方法 |
|--------|----------|-------------------|-------------|-------------|-------------|-------------|----------|
| **G-1** | WorldState 多态支持率 | 1/1 (仅 PendulumState) | **≥ 2 种** State 可插入 cewm_step() | **≥ 3 种** | ≥ 3 种 | **≥ 5 种** | `test_state_polymorphism.py` 参数化测试 |
| **G-2** | 预测器可替换率 | 0% (硬编码 PendulumPhysics) | **Pendulum 可显式替换为 CartPhysics** | **任意 PredictorProtocol 实现可插入** | 继承 | **3+ 种预测器后端** | `PlanAgent` 接受非 Pendulum predictor |
| **G-3** | 动作空间多样性 | 1 维 (torque: float) | **≥ 2 维** 动作空间可定义 | **≥ 4 维** | ≥ 4 维 | **≥ 6 维** (含工具动作) | `Action` 抽象基类的子类数量 |
| **G-4** | `_cewm_parse_state()` 泛化度 | 仅解析 `theta/omega` | **支持任意 `to_vector()` 的 WorldState** | 继承 | 继承 | 支持多模态状态 | `test_cewm_parse_generic.py` |
| **G-5** | 非 Pendulum 闭环测试覆盖 | 0 个测试 | **≥ 5 个测试** (CartState) | **≥ 10 个测试** | ≥ 15 个测试 | **≥ 20 个测试** | `pytest -k "cart"` 测试数 |

### 2.2 集成类 KPI

| KPI ID | 指标名称 | 当前基线 (v4.3.3) | Phase 0 目标 | Phase 1 目标 | Phase 2 目标 | Phase 3 目标 | 测量方法 |
|--------|----------|-------------------|-------------|-------------|-------------|-------------|----------|
| **I-1** | LLM↔CEWM 反馈闭环 | 单向 (intent→CEWM) | — | — | **双向闭环** (LLM 提问→CEWM 反事实→LLM 再推理) | 继承 | `test_llm_cewm_feedback.py` E2E |
| **I-2** | 多模态融合管线集成度 | 融合类独立，不集成默认路径 | — | **PerceptionPipeline 默认启用融合** | 继承 | 继承 | `process_multimodal()` 默认使用融合 |
| **I-3** | from_vector() 保真度 | 0% (丢失模态结构) | — | **≥ 80%** 往返保真 | ≥ 90% | ≥ 90% | `test_from_vector_roundtrip.py` |
| **I-4** | 消融追踪完整性 (泛化) | 无 | — | **--predictor-backend flag** | 继承 Phase 1 | 全链路 flag | CLI `--ablation` 覆盖泛化维度 |
| **I-5** | 惊奇检测非 Pendulum 兼容 | 仅 PendulumState | — | **≥ 2 种 State 兼容** | 继承 | ≥ 3 种 State | `test_surprise_multistate.py` |

### 2.3 就绪类 KPI

| KPI ID | 指标名称 | 当前基线 (v4.3.3) | Phase 2 目标 | Phase 3 目标 | 测量方法 |
|--------|----------|-------------------|-------------|-------------|----------|
| **R-1** | ROS2 桥接原型 | 无 | — | **ROS2 node 可收发 WorldState** | `test_ros2_bridge.py` |
| **R-2** | 6-DOF 机械臂状态空间 | 无 | — | **JointState(6 angles) 可参与 cewm_step()** | `test_robot_state.py` |
| **R-3** | 安全约束覆盖率 | 0 条约束 | **≥ 3 条** (力/位置/速度) | **≥ 8 条** | `test_safety_constraints.py` |
| **R-4** | 单次闭环 WCET | 未测量 | **≤ 100ms** (Pendulum) | **≤ 10ms** (泛化) | `test_wcet.py` 百分位统计 |
| **R-5** | 紧急停止集成 | 无 | — | **信号注入→50ms 内停止** | `test_emergency_stop.py` |

### 2.4 工程类 KPI

| KPI ID | 指标名称 | 当前基线 (v4.3.3) | Phase 0 目标 | Phase 1 目标 | Phase 2 目标 | Phase 3 目标 | 测量方法 |
|--------|----------|-------------------|-------------|-------------|-------------|-------------|----------|
| **E-1** | 测试通过率 | ~1,950 passed | **≥ 1,950 passed (零回归)** | ≥ 2,000 passed | ≥ 2,050 passed | ≥ 2,100 passed | `pytest -q` |
| **E-2** | xfailed 数量 | ≤ 2 | ≤ 2 | ≤ 2 | ≤ 1 | **0** | `pytest --xfail-strict` |
| **E-3** | ruff/mypy 合规 | 0 error | 0 error | 0 error | 0 error | 0 error | `pre-commit run --all-files` |
| **E-4** | 测试覆盖率 | ~55% (估) | ≥ 58% | ≥ 62% | ≥ 65% | ≥ 68% | `pytest --cov` |
| **E-5** | Pendulum 硬编码引用数 | 12+ 处 | **≤ 6 处** | **≤ 2 处** (仅测试) | 0 处 (仅测试) | 0 处 (仅测试) | `grep "Pendulum" sdk/ --include="*.py" \| grep -v test` |
| **E-6** | 新增 Protocol/ABC 数 | 0 | **≥ 2** (PredictorProtocol, StateParser) | ≥ 3 | ≥ 4 | ≥ 5 | 代码统计 |
| **E-7** | 向后兼容性 | 基准 | 100% | 100% | 100% | 100% | `test_imports.py` 全量符号校验 |

### 2.5 架构泛化成熟度目标

| 里程碑 | 当前 (v4.3.3) | Phase 0 | Phase 1 | Phase 2 | Phase 3 |
|--------|-------------|---------|---------|---------|---------|
| **成熟度等级** | L0 (硬编码验证器) | L0 → L1 | L1 → L2 | L2 → L3 | L3 (泛化引擎) |
| **WorldState 支持** | 仅 PendulumState (2D) | Pendulum + Cart (4D) | + Multimodal 任意 | 继承 | + JointState (6D) + ToolState |
| **预测器后端** | PendulumPhysics | Pendulum + Cart (可替换) | Generalized Euler/Lagrange | 继承 | + MuJoCo/Isaac 桥接 |
| **动作空间** | torque: float (1D) | torque + force (2D) | 任意 N 维 | 继承 | 6-DOF + 工具开关 |
| **LLM 集成** | 单向 intent→CEWM | 不变 | 不变 | 双向反馈闭环 | + CoT/Tree-of-Thought oracle |
| **机器人就绪度** | 无 | 无 | 无 | 安全约束层 | ROS2 桥接 + 紧急停止 |

---

## 三、四阶段实施路线图

```
Week 1-2         Week 3-5         Week 6-8         Week 9-12
Phase 0           Phase 1           Phase 2           Phase 3
止血              泛化              闭环              就绪

 P0-1 Predictor   P1-1 通用物理    P2-1 LLM↔CEWM    P3-1 ROS2 桥接
     Protocol          预测器           反馈闭环
 P0-2 泛化         P1-2 动作空间    P2-2 安全约束    P3-2 机器人状态
     _cewm_parse      泛化             层                  空间
 P0-3 CartState    P1-3 多模态      P2-3 消融追踪    P3-3 硬实时
     替代验证          融合集成           泛化链路          保证

 移除 6 处硬编码   新增 3 种 State   LLM 双向推理     桥接原型可用
────────────────────────────────────────────────────────────────►
          Phase 0 完成门禁      Phase 1 完成门禁      Phase 2 完成门禁
          ── Pendulum闭环不破 ──  ── 泛化闭环全通 ──  ── LLM反馈验证 ──
```

---

## 四、Phase 0: 止血 — 抽象 Pendulum 硬编码依赖 (第 1-2 周)

### 4.1 目标

| KPI | 当前 → 目标 |
|-----|------------|
| G-1 WorldState 多态支持率 | 1/1 → **≥ 2 种** |
| G-4 `_cewm_parse_state()` 泛化度 | 仅 theta/omega → **任意 to_vector()** |
| E-5 Pendulum 硬编码引用数 | 12+ 处 → **≤ 6 处** |
| E-6 新增 Protocol/ABC 数 | 0 → **≥ 2** |

**预期收益**: 架构地基建立——Pendulum 从"唯一世界"降级为"一个实现"，为 Phase 1 泛化铺路。**不引入新功能，纯重构**。

### 4.2 任务清单

#### 4.2.1 P0-1: 定义 PredictorProtocol + StateParser Protocol 🔴 最高优先级

| 编号 | 任务 | 关联 KPI | 文件 | 验证方法 | 优先级 | 工时 |
|------|------|----------|------|----------|--------|------|
| **PRO-01** | 定义 `PredictorProtocol` — 预测器接口契约 | E-6 | `sdk/_protocols.py` (新建) | `isinstance(PendulumPhysicsPredictor(), PredictorProtocol)` → True | 🔴 Day 1 | 2h |
| **PRO-02** | 定义 `StateParserProtocol` — 状态解析器接口 | E-6, G-4 | `sdk/_protocols.py` (新建) | `isinstance(lambda obs: ..., StateParserProtocol)` → True (structural) | 🔴 Day 1 | 2h |
| **PRO-03** | `PendulumPhysicsPredictor` 显式声明实现 `PredictorProtocol` | E-5 | `sdk/_action_conditioned_predictor.py` | 现有 Pendulum 测试全部通过 | 🔴 Day 1 | 1h |
| **PRO-04** | `PendulumJEPAPredictor` 显式声明实现 `PredictorProtocol` | E-5 | `sdk/_action_conditioned_predictor.py` | 现有 JEPA 测试全部通过 | 🔴 Day 1 | 1h |
| **PRO-05** | `PlanAgent.__init__()` 接受 `PredictorProtocol` 而非硬编码 PendulumPhysics | G-2, E-5 | `sdk/_plan_agent.py` | 类型注解变更，现有测试不变 | 🔴 Day 2 | 3h |
| **PRO-06** | `MultiBranchPredictor.__init__()` 接受 `PredictorProtocol` | E-5 | `sdk/_multi_branch_predictor.py` | 类型注解变更，现有测试不变 | 🔴 Day 2 | 2h |
| **PRO-07** | `ActionGapMetric._action_effort()` 改为通过 WorldState.distance() 而非硬编码 omega | E-5 | `sdk/_action_gap.py` | `omega_factor` 逻辑移到 `hasattr(state, "omega")` 守卫内 | 🔴 Day 3 | 2h |

**验证标准**: `from mci_world_model.sdk._protocols import PredictorProtocol` 可导入，PendulumPhysicsPredictor 实现该协议，PlanAgent 接受任意 PredictorProtocol 实例。

**PredictorProtocol 定义**:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class PredictorProtocol(Protocol):
    """世界模型预测器接口契约。
    
    任何预测器实现此协议即可插入 PlanAgent / MultiBranchPredictor / cewm_step()。
    """
    name: str
    
    def predict(
        self,
        state: WorldState,
        action: Action | None,
        n_steps: int = 1,
    ) -> list[WorldState]: ...
    
    def evaluate(
        self,
        test_pairs: list[tuple[WorldState, Action | None, WorldState]],
    ) -> dict: ...
```

#### 4.2.2 P0-2: 泛化 `_cewm_parse_state()` + MCIWorldModel 入口

| 编号 | 任务 | 关联 KPI | 文件 | 验证方法 | 优先级 | 工时 |
|------|------|----------|------|----------|--------|------|
| **PRS-01** | 重写 `_cewm_parse_state()` — 支持任意 `to_vector()` 对象 | G-4, E-5 | `sdk/_world_model.py` | 传入 PendulumState/CartState/dict → 均返回非 None WorldState | 🔴 Day 2 | 3h |
| **PRS-02** | 新增 `_resolve_state_class()` — 根据输入类型推断 State 类 | G-4 | `sdk/_world_model.py` | `isinstance(obs, PendulumState)` → PendulumState, `dict with "x"` → CartState | 🔴 Day 3 | 2h |
| **PRS-03** | `plan_action()` 使用 `self._predictor` 而非硬编码 `PendulumPhysicsPredictor()` | G-2, E-5 | `sdk/_world_model.py::plan_action()` L3012-3014 | 现有 plan_action 测试通过，且可注入 mock predictor | 🔴 Day 3 | 3h |
| **PRS-04** | `cewm_step()` 预测器可配置（默认 PendulumPhysics，可覆盖） | G-2 | `sdk/_world_model.py::cewm_step()` | `wm.cewm_step(obs, goal, predictor_backend="cart")` 使用 CartPhysics | 🔴 Day 4 | 3h |
| **PRS-05** | 回归：全量 Pendulum 测试（闭环/PlanAgent/cewm_step） | E-1 | — | `pytest tests/test_pendulum*.py tests/test_plan*.py tests/test_world_model_v430.py -q` | 🔴 Day 5 | 2h |

**验证标准**: `cewm_step()` 接受 PendulumState 的行为与重构前完全一致（零回归），同时接受其他 WorldState 子类不崩溃。

#### 4.2.3 P0-3: CartState 替代验证 — 证明抽象正确

| 编号 | 任务 | 关联 KPI | 文件 | 验证方法 | 优先级 | 工时 |
|------|------|----------|------|----------|--------|------|
| **CRT-01** | 定义 `CartState` — 2D 移动机器人状态 (x, v) | G-1, G-5 | `sdk/_world_state.py` (新增 dataclass) | `CartState` 实现 WorldState ABC 四个核心方法 | 🔴 Day 4 | 2h |
| **CRT-02** | 定义 `CartAction` — 力/加速度 (force: float) | G-3 | `sdk/_world_state.py` | `CartAction(force=2.0).apply(state)` 正确演化 | 🔴 Day 4 | 1h |
| **CRT-03** | 实现 `CartPhysicsPredictor` — Euler 积分 (x' = x + v*dt, v' = v + force*dt) | G-2 | `sdk/_action_conditioned_predictor.py` (新增) | 与手算 ground truth 零误差 | 🔴 Day 5 | 2h |
| **CRT-04** | CartState 闭环测试: Perception → Predict → Act → Feedback | G-5 | `tests/test_cart_closed_loop.py` (新建) | 5 个端到端测试全部通过 | 🔴 Day 6 | 4h |
| **CRT-05** | CartState 插入 cewm_step() E2E | G-1, G-5 | `tests/test_cart_closed_loop.py` | `wm.cewm_step(CartState(x=0,v=0), CartState(x=10,v=0))` 返回预期结果 | 🔴 Day 7 | 3h |
| **CRT-06** | PlanAgent 使用 CartPhysics 规划路径 | G-2 | `tests/test_cart_closed_loop.py` | `wm.plan_action(current=CartState, goal=CartState)` 生成有效 action 序列 | 🔴 Day 8 | 3h |

**验证标准**: CartState 能在 CEWM 的感知→预测→行动→反馈五层闭环中完整运行，与 PendulumState 的 API 体验一致。**这证明 Protocol 抽象是正确的。**

### 4.3 Phase 0 完成门禁

> 以下条件**全部**满足，Phase 0 才算完成：

- [ ] `PredictorProtocol` + `StateParserProtocol` 定义完成，可导入
- [ ] `PendulumPhysicsPredictor` / `PendulumJEPAPredictor` 显式声明实现 PredictorProtocol
- [ ] `PlanAgent` / `MultiBranchPredictor` / `ActionGapMetric` 解除 Pendulum 硬编码
- [ ] `_cewm_parse_state()` 支持任意 WorldState 子类
- [ ] `CartState` + `CartPhysicsPredictor` 完整闭环测试通过（≥ 5 个测试）
- [ ] 全量 Pendulum 测试零回归（≥ 1,950 passed）
- [ ] Pendulum 硬编码引用数从 12+ 降至 ≤ 6（仅在 `_world_model.py` 默认值和测试中）
- [ ] 代码通过 `ruff` + `mypy` 检查
- [ ] `grep "Pendulum" sdk/ --include="*.py" | grep -v test | grep -v "PendulumState"` 仅剩合理的默认初始化

---

## 五、Phase 1: 泛化 — 通用物理预测器 + 多模态管线集成 (第 3-5 周)

### 5.1 目标

| KPI | Phase 0 → Phase 1 目标 |
|-----|------------------------|
| G-1 WorldState 多态支持率 | ≥ 2 种 → **≥ 3 种** |
| G-3 动作空间多样性 | ≥ 2 维 → **≥ 4 维** |
| I-2 多模态融合管线集成度 | 独立工具类 → **默认管线集成** |
| I-3 from_vector() 保真度 | 0% → **≥ 80%** |
| E-5 Pendulum 硬编码引用数 | ≤ 6 处 → **≤ 2 处** |

**预期收益**: CEWM 从"支持两种玩具状态"升级为"支持任意 WorldState + N 维动作"的通用引擎。多模态融合从独立工具变为默认路径。

### 5.2 任务清单

#### 5.2.1 P1-1: 通用物理预测器 🔴

| 编号 | 任务 | 关联 KPI | 文件 | 验证方法 | 优先级 | 工时 | 预计收益 |
|------|------|----------|------|----------|--------|------|----------|
| **GPH-01** | 定义 `GeneralizedPhysicsPredictor` — 基于 Euler/Lagrange 积分的通用预测器 | G-2 | `sdk/_generalized_physics.py` (新建) | 接受任意 `(state_dim, action_dim)` 配置 | 🔴 Week 1 | 8h | — |
| **GPH-02** | 实现 `_euler_step(state_vec, action_vec, dynamics_fn)` 通用步进 | G-2 | `sdk/_generalized_physics.py` | 线性动力学的数值精度 < 1e-10 | 🔴 Week 1 | 4h | — |
| **GPH-03** | `dynamics_fn` 支持注册自定义动力学函数 | G-2 | `sdk/_generalized_physics.py` | `register_dynamics("my_robot", my_fn)` 后可使用 | 🟡 Week 2 | 3h | — |
| **GPH-04** | 用 CartPhysics 验证 GeneralizedPhysicsPredictor 等价性 | G-2 | `tests/test_generalized_physics.py` | 与手写 CartPhysicsPredictor 零误差 | 🔴 Week 2 | 3h | — |
| **GPH-05** | 用 PendulumPhysics 验证 GeneralizedPhysicsPredictor 等价性 | G-2 | `tests/test_generalized_physics.py` | 与 PendulumPhysicsPredictor 零误差（已知动力学函数） | 🔴 Week 2 | 2h | — |
| **GPH-06** | `PlanAgent` 默认使用 `GeneralizedPhysicsPredictor`（Pendulum 动力学函数） | G-2, E-5 | `sdk/_world_model.py` + `sdk/_plan_agent.py` | 现有 PlanAgent 测试全部通过 | 🔴 Week 3 | 3h | — |
| **GPH-07** | 新增 `DoublePendulumState` (4D) 验证泛化能力 | G-1, G-3 | `sdk/_world_state.py` + `tests/test_double_pendulum.py` | 双摆闭环测试通过，动作空间为 2D (torque1, torque2) | 🟡 Week 3 | 6h | — |

**验证标准**: GeneralizedPhysicsPredictor 能通过注册动力学函数支持任意维度的物理系统，Pendulum/Cart/DoublePendulum 三个系统共用同一个预测器类。

#### 5.2.2 P1-2: 动作空间泛化 🔴

| 编号 | 任务 | 关联 KPI | 文件 | 验证方法 | 优先级 | 工时 |
|------|------|----------|------|----------|--------|------|
| **ACT-01** | 定义 `Action` ABC — `apply(state: WorldState) → WorldState` | G-3 | `sdk/_world_state.py` (增强) | `CartAction` / `PendulumAction` / `DoublePendulumAction` 均实现 | 🔴 Week 2 | 3h |
| **ACT-02** | `Action.to_vector() → np.ndarray` — 动作编码为向量 | G-3 | `sdk/_world_state.py` | `PendulumAction(torque=3).to_vector()` → `np.array([3.0])` | 🔴 Week 2 | 2h |
| **ACT-03** | `Action.from_vector(vec) → Action` — 向量解码为动作 | G-3 | `sdk/_world_state.py` | 往返: `Action.from_vector(action.to_vector()) == action` | 🔴 Week 2 | 2h |
| **ACT-04** | `ActionGapMetric` 通过 `Action.to_vector()` 而非硬编码 torque | E-5 | `sdk/_action_gap.py` | 传入 CartAction 不崩溃，距离计算正确 | 🔴 Week 3 | 3h |
| **ACT-05** | `PlanAgent._generate_candidates()` 通过 Action 维度而非硬编码 | G-3 | `sdk/_plan_agent.py` | 自动适配 1D/2D/ND 动作空间 | 🔴 Week 3 | 4h |

**验证标准**: 任何实现 `Action` ABC 的动作类可以插入 PlanAgent/ActionGapMetric，无需修改框架代码。

#### 5.2.3 P1-3: 多模态融合管线集成 🟡

| 编号 | 任务 | 关联 KPI | 文件 | 验证方法 | 优先级 | 工时 |
|------|------|----------|------|----------|--------|------|
| **FUS-01** | `PerceptionPipeline.process_multimodal()` 默认启用融合（而非手动调用 `process_multimodal_fused()`） | I-2 | `_sys/_perception_pipeline.py` | `process_multimodal(signals)` 返回的 `PerceivedFeatures.world_state` 为融合后 MultimodalWorldState | 🟡 Week 3 | 4h |
| **FUS-02** | 修复 `MultimodalWorldState.from_vector()` — 保留模态结构 | I-3 | `sdk/_world_state.py` | 从 `to_vector()` 结果调用 `from_vector()` 后，`active_modalities()` 返回相同列表 | 🟡 Week 4 | 4h |
| **FUS-03** | 修复 `distance()` 维度不一致问题 — 使用 padding/投影 而非截断 | I-3 | `sdk/_world_state.py` | 不同模态组合的状态距离计算不再截断 | 🟡 Week 4 | 3h |
| **FUS-04** | 添加 `modality_alignment()` — 跨模态时间对齐 | I-2 | `sdk/_multimodal_fusion.py` | 不同时间戳的信号对齐到统一时间轴 | 🟡 Week 5 | 4h |
| **FUS-05** | 集成测试: 多模态信号 → 融合 → WorldState → 闭环 | I-2 | `tests/test_multimodal_closed_loop.py` | 4 模态 (proprioception/vision/audio/thermal) 端到端 | 🟡 Week 5 | 4h |

**验证标准**: 默认感知管线产出融合后的 MultimodalWorldState，`from_vector()` 往返保真度 ≥ 80%。

### 5.3 Phase 1 完成门禁

> 以下条件**全部**满足，Phase 1 才算完成：

- [ ] `GeneralizedPhysicsPredictor` 通过注册动力学函数支持 ≥ 3 种物理系统
- [ ] `Action` ABC 定义完成，≥ 3 种子类（Pendulum/Cart/DoublePendulum）
- [ ] `PlanAgent._generate_candidates()` 自动适配 N 维动作空间
- [ ] `PerceptionPipeline` 默认启用多模态融合
- [ ] `MultimodalWorldState.from_vector()` 往返保真度 ≥ 80%
- [ ] 全量测试零回归（≥ 2,000 passed）
- [ ] Pendulum 硬编码引用数 ≤ 2 处（仅在测试中）
- [ ] 新增 `--predictor-backend pendulum|cart|generalized` CLI flag（消融可追溯）
- [ ] `DoublePendulumState` 闭环测试通过（证明 4D 状态 + 2D 动作泛化成功）

---

## 六、Phase 2: 闭环 — LLM↔CEWM 反馈 + 安全约束 (第 6-8 周)

### 6.1 目标

| KPI | Phase 1 → Phase 2 目标 |
|-----|------------------------|
| I-1 LLM↔CEWM 反馈闭环 | 单向 → **双向闭环** |
| R-3 安全约束覆盖率 | 0 条 → **≥ 3 条** |
| I-4 消融追踪完整性 | 无 → **全链路泛化 flag** |
| E-5 Pendulum 硬编码引用数 | ≤ 2 处 → **0 处** (仅测试) |

**预期收益**: LLM 能利用 CEWM 反事实推演结果优化自身推理决策，CEWM 具备安全关键系统的基础约束能力。

### 6.2 任务清单

#### 6.2.1 P2-1: LLM↔CEWM 双向反馈闭环 🔴

| 编号 | 任务 | 关联 KPI | 文件 | 验证方法 | 优先级 | 工时 | 预计收益 |
|------|------|----------|------|----------|--------|------|----------|
| **FB-01** | 实现 `CounterfactualOracle` — LLM 查询 → CEWM 批量反事实 → 排序回注 | I-1 | `sdk/_counterfactual_oracle.py` (新建) | LLM 生成 3 个假设，每个经 CEWM 反事实推演后排序，最优假设排第一 | 🔴 Week 1 | 8h | — |
| **FB-02** | `CounterfactualOracle.batch_what_if(scenarios)` — 并行反事实推演 | I-1 | `sdk/_counterfactual_oracle.py` | N 个场景在 < 1s 内完成推演 | 🔴 Week 1 | 4h | — |
| **FB-03** | 增强 `OrchestratorBridge` — 新增 `CF_QUERY` intent + 反馈回注路径 | I-1 | `sdk/_orchestrator_bridge.py` | `bridge.route({"intent": "CF_QUERY", "hypotheses": [...]})` 返回排序结果 | 🔴 Week 2 | 6h | — |
| **FB-04** | `MultiLLMAdapter` 增加 `reason_with_cf()` — 接收反事实结果做再推理 | I-1 | `sdk/_multillm_adapter.py` | LLM 接收到 CEWM 反事实结果后，输出包含 "基于世界模型推演..." 的推理 | 🔴 Week 2 | 6h | — |
| **FB-05** | E2E 测试: LLM 提问 → CEWM 反事实 → LLM 再推理 → 决策改变 | I-1 | `tests/test_llm_cewm_feedback.py` | 模拟"患者营养方案选择"场景，LLM 初始选择 A，经 CEWM 反事实后改选 B | 🔴 Week 3 | 6h | — |
| **FB-06** | 性能基准: 批量反事实推演延迟 (N=10 场景) | I-1 | `benchmarks/cf_oracle_latency.py` | 10 场景 < 500ms | 🟡 Week 3 | 3h | — |

**验证标准**: LLM 可以通过 `CounterfactualOracle` 查询 CEWM 反事实推演结果，并将结果融入后续推理，实现"如果选 A 会怎样 → CEWM 推演 → A 的代价是 X → 改为选 B"的决策闭环。

**架构示意**:

```
LLM 生成假设              CEWM 反事实推演              LLM 再推理
═══════════              ══════════════              ══════════
"如果给患者方案A..."  →   query_counterfactual()  →  "方案A导致营养指标↓5%"
"如果给患者方案B..."  →   query_counterfactual()  →  "方案B导致营养指标↑8%"
"如果给患者方案C..."  →   query_counterfactual()  →  "方案C导致营养指标↑3%"
                                                    ↓
                                              "推荐方案B，因为..."
```

#### 6.2.2 P2-2: 安全约束层 🔴

| 编号 | 任务 | 关联 KPI | 文件 | 验证方法 | 优先级 | 工时 |
|------|------|----------|------|----------|--------|------|
| **SAF-01** | 定义 `SafetyConstraint` ABC — `check(state, action) → (bool, str)` | R-3 | `sdk/_safety.py` (新建) | 约束检查返回 (通过/不通过, 原因) | 🔴 Week 2 | 3h |
| **SAF-02** | 实现 `ForceLimitConstraint` — 力矩不超过 max_torque | R-3 | `sdk/_safety.py` | `ForceLimitConstraint(max_torque=10).check(state, PendulumAction(15))` → (False, "力矩超限") | 🔴 Week 2 | 2h |
| **SAF-03** | 实现 `PositionBoundConstraint` — 状态空间不越界 | R-3 | `sdk/_safety.py` | `PositionBoundConstraint(theta_range=(-π,π)).check(PendulumState(4.0), ...)` → (False, "角度越界") | 🔴 Week 3 | 2h |
| **SAF-04** | 实现 `VelocityLimitConstraint` — 速度不超过 max_velocity | R-3 | `sdk/_safety.py` | 角速度越界被拦截 | 🔴 Week 3 | 2h |
| **SAF-05** | `SafetyMonitor` — 约束链注册 + 执行前检查 | R-3 | `sdk/_safety.py` | `monitor.check_all(state, action)` → 全部通过才放行 | 🔴 Week 3 | 3h |
| **SAF-06** | 集成到 `cewm_step()` — 每个 action 执行前过安全门 | R-3 | `sdk/_world_model.py::cewm_step()` | 越界 action 被拦截，cewm_step 返回 `"safety_violation"` | 🔴 Week 4 | 4h |
| **SAF-07** | 安全约束集成测试 | R-3 | `tests/test_safety.py` | 超力矩/越界/超速三个场景均被拦截 | 🔴 Week 4 | 3h |

**验证标准**: ≥ 3 条安全约束可通过 SafetyMonitor 链式注册，cewm_step() 在约束违反时返回安全违规信号而非崩溃。

#### 6.2.3 P2-3: 消融追踪泛化链路 + 文档 🟡

| 编号 | 任务 | 关联 KPI | 文件 | 验证方法 | 优先级 | 工时 |
|------|------|----------|------|----------|--------|------|
| **ABL-01** | CLI 增加 `--predictor-backend pendulum|cart|generalized` | I-4 | `benchmarks/cladder/__main__.py` + `benchmarks/cognitive/` | 不同 backend 可切换，Pendulum 保持 99.54% | 🟡 Week 4 | 3h |
| **ABL-02** | CLI 增加 `--safety on|off` | I-4 | `benchmarks/` | 安全约束可独立开关 | 🟡 Week 4 | 1h |
| **ABL-03** | CLI 增加 `--fusion-strategy attention|weighted|concat|off` | I-4 | `benchmarks/` | 多模态融合策略可独立开关 | 🟡 Week 5 | 2h |
| **ABL-04** | 生成 Phase 0-2 消融对比报告模板 | I-4 | `docs/ablation_report_template.md` | 每个维度可独立对比 | 🟡 Week 5 | 3h |
| **ABL-05** | `sdk/__init__.py` 更新导出符号列表（新增 Protocol/类） | E-7 | `sdk/__init__.py` | `test_imports.py` 全量符号校验通过 | 🟡 Week 5 | 2h |

### 6.3 Phase 2 完成门禁

> 以下条件**全部**满足，Phase 2 才算完成：

- [ ] `CounterfactualOracle` 可被 LLM 调用，完成"假设→推演→排序→反馈"闭环
- [ ] `OrchestratorBridge` 支持 `CF_QUERY` intent，返回排序后的反事实结果
- [ ] ≥ 3 条安全约束（力限/位置界/速度限）通过 SafetyMonitor 链式注册
- [ ] `cewm_step()` 在约束违反时返回安全违规信号
- [ ] CLI 支持 `--predictor-backend` / `--safety` / `--fusion-strategy` 独立开关
- [ ] 全量测试零回归（≥ 2,050 passed）
- [ ] Pendulum 硬编码引用数 = 0（仅在测试中出现 PendulumState 是正常的）
- [ ] 消融对比报告模板就绪
- [ ] `sdk/__init__.py` 导出符号完整更新

---

## 七、Phase 3: 就绪 — 手术机器人桥接 + 硬实时保证 (第 9-12 周)

### 7.1 目标

| KPI | Phase 2 → Phase 3 目标 |
|-----|------------------------|
| R-1 ROS2 桥接原型 | 无 → **可收发 WorldState** |
| R-2 6-DOF 机械臂状态空间 | 无 → **JointState 可参与闭环** |
| R-3 安全约束覆盖率 | ≥ 3 条 → **≥ 8 条** |
| R-4 单次闭环 WCET | 未测量 → **≤ 10ms** |
| R-5 紧急停止集成 | 无 → **信号注入→50ms 内停止** |

**预期收益**: CEWM 具备与物理机器人系统对接的**原型能力**（非生产就绪），完成从"纯软件世界模型"到"软硬结合因果引擎"的架构验证。

**重要声明**: Phase 3 交付的是**桥接原型和架构验证**，不是生产级手术机器人控制系统。生产部署需要硬件在环测试、FDA/CFDA 认证、临床验证等额外阶段。

### 7.2 任务清单

#### 7.2.1 P3-1: ROS2 桥接原型 🟡

| 编号 | 任务 | 关联 KPI | 文件 | 验证方法 | 优先级 | 工时 |
|------|------|----------|------|----------|--------|------|
| **ROS-01** | 定义 `RobotWorldState` — 继承 WorldState，增加 `joint_positions/velocities/efforts` | R-2 | `sdk/_robot_state.py` (新建) | `to_vector()` 返回 (N_joints*3,) 向量 | 🟡 Week 1 | 4h |
| **ROS-02** | 定义 `RobotAction` — 继承 Action ABC，N-DOF 关节目标 | R-2, G-3 | `sdk/_robot_state.py` (新建) | `RobotAction(target_positions=[...]).apply(state)` 模拟一步 | 🟡 Week 1 | 3h |
| **ROS-03** | 实现 `ROS2Bridge` — ROS2 node 收发 `JointState` ↔ `RobotWorldState` | R-1 | `sdk/_ros2_bridge.py` (新建) | `ros2 topic echo /joint_states` 可被 bridge 接收并转为 RobotWorldState | 🟡 Week 2-3 | 12h |
| **ROS-04** | Bridge → `cewm_step()` 集成 — 从 ROS2 接收状态 → CEWM 推演 → ROS2 发布预测 | R-1 | `sdk/_ros2_bridge.py` | E2E: ROS2 topic → WorldState → cewm_step() → 预测状态 → ROS2 topic | 🟡 Week 3 | 6h |
| **ROS-05** | 仿真环境验证: 使用 `joint_state_publisher_gui` 模拟机械臂 | R-1 | `tests/test_ros2_bridge.py` | 手动拖动关节 → CEWM 推演轨迹 → 验证物理一致性 | 🟡 Week 4 | 6h |

**验证标准**: ROS2Bridge 能从 `/joint_states` topic 接收数据，转换为 RobotWorldState，推演后发布到 `/cewm_prediction` topic。

#### 7.2.2 P3-2: 安全约束扩展 + 紧急停止 🟡

| 编号 | 任务 | 关联 KPI | 文件 | 验证方法 | 优先级 | 工时 |
|------|------|----------|------|----------|--------|------|
| **SAF2-01** | `JointLimitConstraint` — 各关节角度不超过硬件限位 | R-3 | `sdk/_safety.py` | 6 个关节各自的 min/max 独立检查 | 🟡 Week 4 | 3h |
| **SAF2-02** | `SelfCollisionConstraint` — 基于简化包围盒的自碰撞检测 | R-3 | `sdk/_safety.py` | 两连杆接近时触发碰撞警告 | 🟡 Week 5 | 6h |
| **SAF2-03** | `WorkspaceBoundConstraint` — 末端执行器不超出工作空间 | R-3 | `sdk/_safety.py` | 正运动学计算末端位置，检查是否在边界内 | 🟡 Week 5 | 4h |
| **SAF2-04** | `ToolForceConstraint` — 手术工具接触力不超过安全阈值 | R-3 | `sdk/_safety.py` | 力传感器读数 > max_force → 拦截 | 🟡 Week 6 | 3h |
| **SAF2-05** | `EmergencyStop` — 独立线程监听停止信号，50ms 内设置全局停止标志 | R-5 | `sdk/_emergency_stop.py` (新建) | 发送 SIGUSR1 → 50ms 内 `cewm_step()` 返回停止 | 🟡 Week 6 | 6h |
| **SAF2-06** | 紧急停止集成测试 | R-5 | `tests/test_emergency_stop.py` | 信号注入后 50ms 内完成停止 | 🟡 Week 7 | 4h |

**验证标准**: ≥ 8 条安全约束可注册，紧急停止信号在 50ms 内生效。

#### 7.2.3 P3-3: 硬实时保证预研 🟡

| 编号 | 任务 | 关联 KPI | 文件 | 验证方法 | 优先级 | 工时 |
|------|------|----------|------|----------|--------|------|
| **RT-01** | `cewm_step()` 各阶段延迟测量: 感知/认知/预测/行动/反馈 分阶段计时 | R-4 | `sdk/_world_model.py` | `--profile` flag 输出各阶段 p50/p95/p99 耗时 | 🟡 Week 7 | 4h |
| **RT-02** | WCET 估算: 各阶段最大耗时统计（1,000 次运行取 p99.9） | R-4 | `benchmarks/wcet_analysis.py` | p99.9 < 10ms (Pendulum) / < 50ms (Multimodal) | 🟡 Week 7 | 4h |
| **RT-03** | `DeadlineMonitor` — 超时检测 + 降级策略 | R-4 | `sdk/_deadline_monitor.py` (新建) | 配置 10ms deadline，超时后自动降级到简化预测器 | 🟡 Week 8 | 5h |
| **RT-04** | 快速路径: `cewm_step_fast()` — 跳过认知诊断/经验检索，仅做预测+安全 | R-4 | `sdk/_world_model.py` | `cewm_step_fast()` 延迟 ≤ `cewm_step()` 的 30% | 🟡 Week 8 | 3h |
| **RT-05** | 实时性集成测试 | R-4 | `tests/test_realtime.py` | 100 次运行 95% 在 deadline 内完成 | 🟡 Week 8 | 3h |

**验证标准**: `cewm_step_fast()` 在 Pendulum 场景下 p99 < 1ms，GeneralizedPhysics 场景下 p99 < 5ms。

### 7.3 Phase 3 完成门禁

> Phase 3 完成 = CEWM 架构泛化 **正式进入 L3（泛化引擎）**

- [ ] ROS2Bridge 原型可从 ROS2 topic 收发 WorldState
- [ ] RobotWorldState (6-DOF) 可参与 cewm_step() 闭环
- [ ] ≥ 8 条安全约束可注册，紧急停止 50ms 内生效
- [ ] `cewm_step_fast()` 延迟 ≤ 通用路径的 30%
- [ ] p99 WCET < 10ms (Pendulum 场景)
- [ ] 全量测试零回归（≥ 2,100 passed），xfailed = 0
- [ ] 所有 Phase 0-2 feature 可组合开关
- [ ] `sdk/__init__.py` 导出符号完整，`test_imports.py` 通过
- [ ] 手术机器人能力诚实标注为"原型验证阶段"

---

## 八、门禁检查机制

### 8.1 每次改动的验证流水线

```
代码改动 ──► ruff lint + mypy (本地)
               │
               ▼
         冒烟测试: 10 个关键 Pendulum E2E 测试
               │
               ├── 检查: 零回归、Pendulum 闭环不破
               │
               ▼
         全量 Pendulum 回归 (≥ 1,950 tests)
               │
               ├── 检查: passed 数不降、xfailed 数不增
               │
               ▼
         非 Pendulum 泛化测试 (Cart/DoublePendulum/...)
               │
               ├── 检查: 泛化路径功能正确
               │
               ▼
         每个 Phase 完成: 完整 KPI 报告
               │
               ├── 泛化 KPI (G-1 ~ G-5)
               ├── 集成 KPI (I-1 ~ I-5)
               ├── 就绪 KPI (R-1 ~ R-5)
               ├── 工程 KPI (E-1 ~ E-7)
               └── 与上一 Phase 对比报告
```

### 8.2 消融追踪规范

每个 Phase 的 feature 必须有独立开关：

```python
# 在 benchmarks/cladder/__main__.py 或 benchmarks/cognitive/ 中
parser.add_argument("--predictor-backend", 
    choices=["pendulum", "cart", "generalized", "double_pendulum"],
    default="pendulum",
    help="P0-1/P1-1: 预测器后端")
parser.add_argument("--fusion-strategy",
    choices=["attention", "weighted", "concat", "off"],
    default="off",
    help="P1-3: 多模态融合策略")
parser.add_argument("--enable-cf-oracle", action="store_true",
    help="P2-1: 启用 CounterfactualOracle")
parser.add_argument("--safety", choices=["on", "off"], default="off",
    help="P2-2: 启用安全约束层")
parser.add_argument("--deadline-ms", type=int, default=0,
    help="P3-3: 设置 WCET deadline (0=不限制)")
```

### 8.3 反回归检测

每个 Phase 末尾运行防回归检查：

```bash
# 1. Pendulum 硬编码残留检查
grep -r "PendulumPhysics\|PendulumAction\|PendulumJEPA" \
  mci-world-model/src/mci_world_model/sdk/ \
  --include="*.py" | grep -v test | grep -v __pycache__

# 2. 全量符号导入
python -c "from mci_world_model import __all__; print(len(__all__))"

# 3. 全量测试
pytest tests/ -q --tb=short

# 4. Cladder 基准（如有改动影响因果求解器）
python -m benchmarks.cladder --rung all
```

---

## 九、角色与分工

| 角色 | 职责 | Phase 0 工作 | Phase 1 工作 | Phase 2 工作 | Phase 3 工作 |
|------|------|-------------|-------------|-------------|-------------|
| **架构负责人** | 协议设计 + Pendulum 解耦决策 | P0-1 Protocol 设计 + P0-2 泛化 parse | P1-1 通用预测器架构评审 | P2-1 LLM 反馈架构评审 | P3 机器人集成架构决策 |
| **SDK 工程师** | 核心 SDK 实现 | P0-1/P0-2 全部代码 + P0-3 CartState | P1-1 通用预测器 + P1-2 动作泛化 | P2-2 安全约束层 | P3-1 ROS2 桥接 + P3-2 安全扩展 |
| **感知工程师** | 多模态 + 融合管线 | — | P1-3 融合集成 + from_vector 修复 | — | — |
| **LLM 集成工程师** | LLM↔CEWM 反馈 | — | — | P2-1 CounterfactualOracle + OrchestratorBridge | — |
| **实时系统工程师** | WCET + 紧急停止 | — | — | — | P3-3 硬实时保证 |
| **质量保证** | 回归测试 + 消融追踪 | Phase 0 门禁检查 | Phase 1 门禁检查 | Phase 2 门禁检查 | Phase 3 门禁检查 |

---

## 十、风险管理与应急预案

### 10.1 风险识别

| 风险 | 影响 | 概率 | 应对 |
|------|------|------|------|
| **Protocol 抽象后 Pendulum 性能下降** | G-2 达标但效率倒退 | 低 | Protocol 使用 `@runtime_checkable` + `isinstance` 而非每次调用时检查；热路径直接调用 |
| **GeneralizedPhysicsPredictor 无法覆盖 Pendulum 的 sin(g/L·θ) 非线性** | P1-1 无法替代 PendulumPhysics | 中 | dynamics_fn 支持注册任意非线性函数；Pendulum 保留为特殊的 `dynamics_fn` 而非独立类 |
| **CartState 引入后发现更多硬编码点** | Phase 0 延期 | 中 | 每次发现新硬编码点，标记但不立即修复（Phase 1 统一处理），Phase 0 聚焦 6 个已知点 |
| **LLM 反事实结果质量差导致决策退化** | LLM 决策反而不如不用 CEWM | 中 | CounterfactualOracle 返回置信度；低置信度结果标注 "uncertain"，LLM 可自主选择是否采信 |
| **ROS2 桥接原型引入复杂系统依赖** | Phase 3 部署困难 | 高 | ROS2Bridge 作为可选 extras: `pip install mci-world-model[ros2]`；CI 不运行 ROS2 测试 |
| **硬实时无法在 Python 层面保证** | R-4 WCET 无法达标 | 高 | 诚实标注 Python GIL 限制；`cewm_step_fast()` 仅做 best-effort；真正硬实时预留 C++/Rust 扩展接口 |
| **安全约束层与现有 CircuitBreaker 功能重叠** | 重复逻辑、维护混乱 | 低 | SafetyMonitor 聚焦物理约束（力/位置/速度）；CircuitBreaker 聚焦服务降级（API 失败/超时），职责清晰 |

### 10.2 应急回滚方案

| 变更类型 | 回滚方案 |
|----------|----------|
| PredictorProtocol 引入 | 移除 Protocol 声明，恢复硬编码类型注解（Phase 0 可回滚） |
| GeneralizedPhysicsPredictor | `--predictor-backend pendulum` 恢复 PendulumPhysics（P1 可回滚） |
| Action ABC 泛化 | 移除 ABC 继承，恢复 PendulumAction 硬编码（P1 可回滚） |
| 多模态融合默认启用 | `--fusion-strategy off` 关闭融合（P1 可回滚） |
| CounterfactualOracle | `--no-cf-oracle` 关闭反馈闭环（P2 可回滚） |
| 安全约束层 | `--safety off` 关闭约束检查（P2 可回滚） |
| ROS2 桥接 | 不安装 `[ros2]` extras 即无影响（P3 可选） |

### 10.3 关键决策检查点

#### DC-1: Phase 0 完成后

- **通过条件**: G-1/G-4/E-5/E-6 全部达标，Pendulum 全量测试零回归
- **决策**: Protocol 抽象是否正确？CartState 验证是否充分？是否继续 Phase 1？

#### DC-2: Phase 1 完成后

- **通过条件**: G-1/G-3/I-2/I-3 全部达标，DoublePendulum 泛化验证通过
- **决策**: GeneralizedPhysicsPredictor 性能是否可接受？是否需要在 Phase 1.5 做性能优化？

#### DC-3: Phase 2 完成后

- **通过条件**: I-1/R-3 全部达标，LLM↔CEWM 反馈闭环 E2E 通过
- **决策**: CounterfactualOracle 的 LLM 推理质量是否达到预期？安全约束是否足够？

#### DC-4: Phase 3 完成后

- **通过条件**: R-1 ~ R-5 全部达标
- **决策**: 是否进入生产级机器人控制系统开发？或需要额外的硬件在环验证阶段？

---

## 附录 A: Pendulum 硬编码追踪矩阵 (Phase 0 输入)

| 序号 | 文件 | 行号区域 | 硬编码内容 | Phase 0 修复 | 修复后状态 |
|------|------|----------|-----------|-------------|-----------|
| 1 | `_world_model.py` | L3012-3014 | `PendulumPhysicsPredictor()` 硬编码到 PlanAgent 初始化 | PRS-03 | 接受 `PredictorProtocol` |
| 2 | `_world_model.py` | L3287-3293 | `_cewm_parse_state()` 仅解析 `theta/omega` | PRS-01/02 | 支持任意 `to_vector()` |
| 3 | `_action_conditioned_predictor.py` | L72 | `PendulumPhysicsPredictor.predict()` 只接受 PendulumState | PRO-03 | 实现 PredictorProtocol |
| 4 | `_action_conditioned_predictor.py` | L246 | `PendulumJEPAPredictor.predict()` 只接受 PendulumState | PRO-04 | 实现 PredictorProtocol |
| 5 | `_multi_branch_predictor.py` | ~L45 | 包装的 predictor 被假定为 pendulum | PRO-06 | 接受 PredictorProtocol |
| 6 | `_action_gap.py` | L131-146 | `_action_effort()` 硬编码 `state.omega` | PRO-07 | hasattr 守卫 |
| 7 | `_action_gap.py` | L148-177 | `_simulate_action()` 硬编码 Pendulum 物理 | PRO-07 | 委托给 `state.step_physics()` |
| 8 | `_plan_agent.py` | ~L100+ | `_generate_candidates()` 基于 torque 生成候选 | ACT-05 (Phase 1) | 基于 Action 维度泛化 |
| 9 | `_world_state.py` | L309-323 | `PendulumAction.apply()` 只作用于 PendulumState | — (保留) | 合理：Action 作用于特定 State |
| 10 | `_world_model.py` | L2679-2680 | `cewm_step()` 状态解析仅 Pendulum | PRS-04 | 泛化解析 |
| 11 | `_world_model.py` | L2693-2702 | `_cewm_state_change()` 仅处理 theta/omega | PRS-02 | hasattr 守卫 |
| 12 | `_action_gap.py` | L97 | `distance()` 调用 `state.omega` | PRO-07 | hasattr 守卫 |

## 附录 B: 新增文件清单

| 编号 | 文件 | Phase | 预估行数 | 说明 |
|------|------|-------|---------|------|
| N1 | `sdk/_protocols.py` | Phase 0 | ~80 | PredictorProtocol + StateParserProtocol |
| N2 | `tests/test_cart_closed_loop.py` | Phase 0 | ~200 | CartState E2E 闭环测试 |
| N3 | `sdk/_generalized_physics.py` | Phase 1 | ~300 | 通用物理预测器 + 动力学注册 |
| N4 | `tests/test_generalized_physics.py` | Phase 1 | ~200 | 通用预测器测试（3 种物理系统） |
| N5 | `tests/test_double_pendulum.py` | Phase 1 | ~150 | 双摆闭环测试 |
| N6 | `tests/test_multimodal_closed_loop.py` | Phase 1 | ~150 | 多模态融合闭环测试 |
| N7 | `sdk/_counterfactual_oracle.py` | Phase 2 | ~250 | LLM↔CEWM 反事实 Oracle |
| N8 | `tests/test_llm_cewm_feedback.py` | Phase 2 | ~200 | LLM 反馈闭环 E2E |
| N9 | `sdk/_safety.py` | Phase 2 | ~300 | 安全约束层 (ABC + 3 种约束 + Monitor) |
| N10 | `tests/test_safety.py` | Phase 2 | ~150 | 安全约束测试 |
| N11 | `sdk/_robot_state.py` | Phase 3 | ~200 | RobotWorldState + RobotAction |
| N12 | `sdk/_ros2_bridge.py` | Phase 3 | ~300 | ROS2 桥接原型 |
| N13 | `sdk/_emergency_stop.py` | Phase 3 | ~150 | 紧急停止模块 |
| N14 | `sdk/_deadline_monitor.py` | Phase 3 | ~150 | 硬实时监控 |
| N15 | `benchmarks/wcet_analysis.py` | Phase 3 | ~100 | WCET 分析脚本 |

## 附录 C: 工时汇总

| Phase | 版本 | 周数 | 设计工时 | 编码工时 | 测试工时 | 总工时 | 人力（1人） |
|-------|------|------|---------|---------|---------|--------|-----------|
| Phase 0 | v4.4.0 | 2 | 8h | 22h | 14h | **44h** | 5.5 人日 |
| Phase 1 | v4.4.1 | 3 | 10h | 35h | 20h | **65h** | 8.1 人日 |
| Phase 2 | v4.4.2 | 3 | 8h | 38h | 18h | **64h** | 8.0 人日 |
| Phase 3 | v4.5.0 | 4 | 12h | 42h | 22h | **76h** | 9.5 人日 |
| **合计** | **v4.5.0** | **12** | **38h** | **137h** | **74h** | **249h** | **31.1 人日** |

**并行优化可能性**: Phase 1 的 P1-1 和 P1-2 可部分并行（动作 ABC 定义可先行），Phase 2 的 P2-1 和 P2-2 可完全并行。并行后总工时压缩至 **~210h / 26 人日**。

---

*本规划书基于 CEWM v4.3.3 逐模块能力审计 (2026-06-03) 编制*
*与 CEWM v4.0.0 迭代计划书 + MCI 医疗世界模型发展规划书 v3.2.0 形成三件套互补关系*
*下次 review：Phase 0 完成后*
