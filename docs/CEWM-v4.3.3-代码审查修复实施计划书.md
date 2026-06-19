# CEWM v4.3.3 — 代码审查修复实施计划书

> **文档编号**: REVIEW-FIX-CEWM-20260603
> **制定日期**: 2026年6月3日
> **编制依据**: CEWM v4.0.0 六大维度深度代码审查报告
> **关联文档**:
>   - `docs/CEWM-架构加固与泛化-KPI实施规划书.md`（架构泛化路线图）
>   - `docs/修复实施计划书_20260603.md`（测试/风格/类型修复计划）
>   - `docs/P0-P11_COMPREHENSIVE_TEST_REPORT.md`（全面测试报告）
> **实施周期**: 8 周（2026.06.03 — 2026.07.28）
> **目标**: 修复代码审查发现的 5 个 Critical Issues、7 个 Warnings、9 个 Suggestions，将代码质量评分从 ⭐⭐⭐⭐☆ (4.5/5) 提升至 ⭐⭐⭐⭐⭐ (5.0/5)，确保 CEWM 闭环功能真正生效。

---

## 执行摘要

### 审查背景

本次计划书基于对 CEWM v4.3.3 代码库的六大维度深度审查，审查范围覆盖以下核心组件（共 ~6,000 行代码，10 个核心模块）：

| 组件 | 文件 | 行数 | 审查维度 |
|------|------|------|----------|
| CEWM 闭环引擎 | `_world_model.py` (cewm_step/cewm_step_fast) | 158+86 | 闭环功能验证、函数长度 |
| 通用物理预测器 | `_generalized_physics.py` | 389 | 架构泛化、维度碰撞 |
| 协议抽象层 | `_protocols.py` | 276 | 接口契约、解析器优先级 |
| 因果更新器 | `_causal_updater.py` | 610 | 因果图积累、增量更新 |
| 元诊断器 | `_meta_diagnoser.py` | 752 | 失败模式匹配、可变状态 |
| 负面启发式 | `_negative_heuristic.py` | 632 | 硬核规则、变更检查粒度 |
| 经验记忆库 | `_experience_memory.py` | 742 | 三维索引、序列化完整性 |
| 认知环总线 | `_cognitive_loop.py` | 807 | Wiener 四环传播 |
| 行动差距度量 | `_action_gap.py` | 368 | Pendulum 回退债务 |
| 认知多样性 | `_cognitive_diversity.py` | 574 | Shannon 熵、Ashby 阈值 |

### 关键发现

审查发现了 **21 个问题**，按严重度分三级：

| 级别 | 数量 | 核心影响 | 修复紧迫度 |
|------|------|----------|-----------|
| 🔴 Critical Issues | 5 | 闭环功能失效、因果图断裂、预测偏差 | 立即修复 |
| 🟡 Warnings | 7 | 功能降级、维护困难、隐患积累 | 应当修复 |
| 🟢 Suggestions | 9 | 架构优化、代码规范、技术债务 | 可考虑修复 |

**最严重发现**: C-1（因果图重建）和 C-2（JEPA 字符串查询）直接导致 CEWM 闭环的两个核心子系统——因果积累和嵌入预测——**功能性失效**。当前 `cewm_step()` 能跑通但不产生有效的因果推理和预测，这解释了 v4.3.3 在复杂场景下表现退化的根因。

### 修复优先级与总体目标

```
Phase 0 (止血)  →  Phase 1 (泛化)  →  Phase 2 (闭环)  →  Phase 3 (就绪)
  3天               2周               2周               3周
  C-1~C-5          W-1~W-4          W-5~W-7+S-1~S-3   S-4~S-9
  闭环功能恢复      硬编码清零        架构规范化         质量提升
```

| 里程碑 | 时间 | 核心交付 | 验收指标 |
|--------|------|---------|---------|
| M0: 止血完成 | Day 3 | C-1~C-5 全部修复 | cewm_step 因果图可积累、JEPA 嵌入查询正常 |
| M1: 泛化达标 | Week 2 末 | W-1~W-4 修复 | Pendulum 硬编码 ≤ 2 处、双摆耦合修正 |
| M2: 闭环验证 | Week 4 末 | W-5~W-7 + S-1~S-3 | cewm_step E2E 全流程验证通过 |
| M3: 质量提升 | Week 8 末 | S-4~S-9 全部完成 | 覆盖率 ≥ 60%、mypy errors ≤ 200 |

---

## 问题分类与修复计划

### 一、Critical Issues（必须立即修复）

> 🔴 **MUST FIX** — 这 5 个问题直接导致 CEWM 闭环核心子系统功能失效，必须在 Phase 0（3天内）全部修复。

---

#### C-1: 因果图重建 Bug — 每次 cewm_step() 调用创建新 CausalUpdater

| 属性 | 内容 |
|------|------|
| **严重度** | 🔴 Critical |
| **文件** | `_world_model.py` |
| **行号** | L2733-L2735 |
| **影响范围** | CEWM 闭环认知层 — 因果图无法积累，跨步骤因果推理完全失效 |
| **理论对标** | Pearl Do-Calculus 要求因果图随观测增量更新；Wiener COGNITION 层 (Layer 1) 依赖因果图作为跨层误差传播基础 |

**问题描述**:

`cewm_step()` 在认知层每次调用都创建新的 `CausalUpdater()` 实例：

```python
# _world_model.py L2733-L2735
from mci_world_model.sdk._causal_updater import CausalUpdater
self._causal_updater = CausalUpdater()  # ← 每次重建！
```

**后果**: 因果图在每次 `cewm_step()` 后被丢弃，无法在时间步之间积累因果证据。CausalUpdater 的 `update()` 方法设计为增量积累模式，但重建使所有历史因果发现归零。这导致：

1. `_causal_updater.update()` 返回的 records 长度永远为单步证据数，无法构建跨步骤因果链
2. Wiener 四环中 COGNITION→PREDICTION 的误差传播缺少因果图支撑
3. MetaDiagnoser 的根因链追溯因因果图过浅而无法有效定位失败根因

**修复方案**:

```python
# 方案: 在 __init__ 中初始化，cewm_step() 中仅做增量更新

# __init__ 中添加:
self._causal_updater: CausalUpdater | None = None

# cewm_step() 中修改 L2732-L2735:
if self._causal_updater is None:
    from mci_world_model.sdk._causal_updater import CausalUpdater
    self._causal_updater = CausalUpdater()
# 不再重新创建！直接使用 self._causal_updater 做增量更新
```

| 修复项 | 详情 |
|--------|------|
| **预估工时** | 0.5h |
| **修改文件** | `_world_model.py` |
| **回归风险** | 极低（行为变更：从"每次重置"变为"持续积累"，符合预期设计） |
| **验证方法** | 连续调用 `cewm_step()` 3+ 次，检查 `result["causal_updates"]` 累积递增 |

---

#### C-2: JEPA 字符串查询 Bug — str(state) 作为嵌入预测查询

| 属性 | 内容 |
|------|------|
| **严重度** | 🔴 Critical |
| **文件** | `_world_model.py` |
| **行号** | L2778-L2780 (cewm_step), L2881-L2883 (cewm_step_fast) |
| **影响范围** | CEWM 闭环预测层 — JEPA 预测完全失效 |
| **理论对标** | LeCun JEPA 要求在嵌入空间进行预测；当前实现退化为字符串匹配 |

**问题描述**:

`cewm_step()` 和 `cewm_step_fast()` 均使用 `str(current_state)` 作为 JEPA 预测查询：

```python
# _world_model.py L2778-L2780
predictions = self.jepa_predict(
    query=str(current_state) if hasattr(current_state, "__str__") else "state"
)
```

**后果**: JEPA (Joint Embedding Predictive Architecture) 的核心设计是在**嵌入空间**进行预测，而非在原始状态字符串空间。`str(PendulumState)` 产生的字符串（如 `"PendulumState(theta=0.1, omega=0.0)"`）与 JEPA 编码器期望的嵌入向量完全不匹配，导致：

1. JEPA 预测器内部的嵌入查询永远无法匹配训练数据
2. 预测结果退化为全零或随机值，`prediction_error` 无实际意义
3. 反馈层的注意力调整因预测误差失真而产生错误的权重分配

**修复方案**:

```python
# 方案: 优先使用 JEPA 编码器进行嵌入查询，回退到 to_vector()

# cewm_step() 中修改 L2776-L2780:
if self._jepa_predictor is not None and current_state is not None:
    # 优先使用嵌入查询
    if hasattr(current_state, "to_vector"):
        query_vec = current_state.to_vector()
    else:
        query_vec = None

    if query_vec is not None:
        predictions = self.jepa_predict(query=query_vec)
    else:
        logger.debug("JEPA 查询跳过: 状态无 to_vector()")
```

| 修复项 | 详情 |
|--------|------|
| **预估工时** | 1h |
| **修改文件** | `_world_model.py` (cewm_step + cewm_step_fast) |
| **回归风险** | 低（修复后 JEPA 预测将真正工作，可能影响依赖当前行为的测试） |
| **验证方法** | `cewm_step()` 后 `result["prediction"]` 非空且非全零向量 |

---

#### C-3: _infer_backend 维度碰撞 — 2D/4D 系统推断歧义

| 属性 | 内容 |
|------|------|
| **严重度** | 🔴 Critical |
| **文件** | `_generalized_physics.py` |
| **行号** | L351-L364 |
| **影响范围** | 通用物理预测器 — 后端推断错误导致物理演化方向错误 |
| **理论对标** | PredictorProtocol 要求预测器能根据状态类型正确选择动力学后端 |

**问题描述**:

`_infer_backend()` 仅根据状态向量维度推断后端，但存在维度碰撞：

```python
# _generalized_physics.py L357-L364
state_dim = len(state.to_vector())
for name, dim in self._state_dims.items():
    if dim == state_dim:
        return name  # ← 维度碰撞！
```

**维度碰撞矩阵**:

| 状态维度 | 注册的系统 | 碰撞数 | 推断结果（取决于字典遍历顺序） |
|----------|-----------|--------|------------------------------|
| 2D | pendulum, cart, spring_mass | 3 | 不确定（dict 顺序） |
| 3D | fluid_flow | 1 | 确定但脆弱 |
| 4D | double_pendulum, projectile | 2 | 不确定 |

**后果**: 当 `self._backend` 未显式设置时，2D 状态可能被错误推断为 `spring_mass` 而非 `pendulum`，导致完全错误的物理演化（弹簧-质量系统的动力学方程与单摆完全不同）。

**修复方案**:

```python
# 方案: 引入显式后端标记 + 状态类型注册

# 1. 在 register_dynamics 时注册关联的状态类型
def register_dynamics(self, name, fn, state_dim, action_dim, state_cls=None):
    ...
    if state_cls is not None:
        self._state_cls_map[state_cls] = name

# 2. _infer_backend 优先按状态类型推断
def _infer_backend(self, state):
    # 显式设置优先
    if self._backend in self._dynamics_registry:
        return self._backend

    # 按状态类型推断（精确匹配）
    state_cls = type(state)
    if state_cls in self._state_cls_map:
        return self._state_cls_map[state_cls]

    # 维度推断仅作最后回退，且需用户确认
    state_dim = len(state.to_vector())
    candidates = [name for name, dim in self._state_dims.items() if dim == state_dim]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        logger.warning("维度碰撞: %dD 有 %s，使用默认 %s", state_dim, candidates, self._backend)
    return self._backend
```

| 修复项 | 详情 |
|--------|------|
| **预估工时** | 1.5h |
| **修改文件** | `_generalized_physics.py` |
| **回归风险** | 中（需更新现有 register_dynamics 调用，添加 state_cls 参数） |
| **验证方法** | PendulumState(2D) → "pendulum"；CartState(2D) → "cart"；不混淆 |

---

#### C-4: cewm_step() 函数过长 — 158 行（超规范 3.16 倍）

| 属性 | 内容 |
|------|------|
| **严重度** | 🔴 Critical |
| **文件** | `_world_model.py` |
| **行号** | L2649-L2807 |
| **影响范围** | 代码可维护性、可测试性 |
| **规范** | 项目规范：单函数 ≤ 50 行 |

**问题描述**:

`cewm_step()` 包含感知、安全、认知、行动、预测、反馈共 6 层逻辑，合计 158 行，远超 50 行规范。这导致：

1. 难以对单个层进行独立测试
2. 修改某一层逻辑时容易影响其他层
3. 无法对单层进行性能 profiling

**修复方案**: 将 cewm_step() 拆分为 6 个子方法：

```python
def cewm_step(self, observation=None, goal=None, action=None):
    """CEWM 引擎一步驱动全流程（编排层）。"""
    result = self._init_cewm_result()
    current_state, goal_state = self._cewm_perceive(observation, goal)
    result["state"] = current_state

    # 安全层
    if self._cewm_safety_check(current_state, action, result):
        return result

    # 认知层
    self._cewm_cognize(current_state, goal_state, result)

    # 行动层
    self._cewm_evaluate_action(current_state, goal_state, result)

    # 预测层
    self._cewm_predict(current_state, goal_state, action, result)

    # 反馈层
    self._cewm_feedback(result)

    return result

# 每个子方法 ≤ 30 行
```

| 修复项 | 详情 |
|--------|------|
| **预估工时** | 2h |
| **修改文件** | `_world_model.py` |
| **回归风险** | 低（纯重构，行为不变） |
| **验证方法** | 拆分前后行为等价性测试 + cewm_step() 本体 ≤ 30 行 |

---

#### C-5: _cewm_state_change() 硬编码类型分支

| 属性 | 内容 |
|------|------|
| **严重度** | 🔴 Critical |
| **文件** | `_world_model.py` |
| **行号** | L3456-L3500 |
| **影响范围** | CEWM 闭环认知层 — 因果边提取依赖硬编码属性名 |
| **KPI 对标** | E-5 Pendulum 硬编码引用数 ≤ 2（当前此处为第 2 处残留） |

**问题描述**:

`_cewm_state_change()` 使用 `hasattr` 硬编码检查特定状态类型的属性名：

```python
# _world_model.py L3464-L3478
if hasattr(state, "theta") and hasattr(state, "omega"):  # Pendulum
    edges.append(("theta", "omega"))
if hasattr(state, "x") and hasattr(state, "v"):  # Cart
    edges.append(("x", "v"))
```

**后果**: 每新增一种 WorldState 子类，就需要在此处添加新的 hasattr 分支，违反开闭原则。当前已有 3 个分支（Pendulum/Cart/Robot），如果新增 5 种状态类，此处将膨胀为不可维护的分支瀑布。

**修复方案**: 让 WorldState 子类自行声明因果边：

```python
# 在 WorldState ABC 中添加:
def causal_edges(self) -> list[tuple[str, str]]:
    """声明此状态类型的因果边。子类可覆盖以提供特定因果结构。"""
    # 默认: 基于 to_vector() 维度的相邻项
    vec = self.to_vector()
    return [(f"dim_{i}", f"dim_{i+1}") for i in range(len(vec) - 1)]

# _cewm_state_change() 简化为:
def _cewm_state_change(self, state):
    if hasattr(state, "causal_edges"):
        return state.causal_edges()
    return []
```

| 修复项 | 详情 |
|--------|------|
| **预估工时** | 1.5h |
| **修改文件** | `_world_model.py`, `_world_state.py`（PendulumState/CartState/RobotWorldState） |
| **回归风险** | 低（行为保持一致，架构更优） |
| **验证方法** | PendulumState.causal_edges() 返回 [("theta","omega"),...] |

---

### 二、Warnings（应当修复）

> 🟡 **SHOULD FIX** — 这 7 个问题不直接导致功能失效但会造成功能降级、维护困难和隐患积累，应在 Phase 1-2 修复。

---

#### W-1: cewm_step_fast() 异常完全静默

| 属性 | 内容 |
|------|------|
| **严重度** | 🟡 Warning |
| **文件** | `_world_model.py` |
| **行号** | L2885-L2886 |
| **影响范围** | 快速路径调试 — 预测失败无任何日志 |

**问题描述**:

```python
# _world_model.py L2885-L2886
except Exception:
    pass  # 快速路径: 预测失败不阻塞
```

**修复方案**: 替换为 `logger.debug()` 记录异常信息（不影响性能，但保留可追溯性）。

| 修复项 | 详情 |
|--------|------|
| **预估工时** | 0.5h |
| **优先级** | Phase 1 |

---

#### W-2: _simulate_action() Pendulum 硬编码回退路径

| 属性 | 内容 |
|------|------|
| **严重度** | 🟡 Warning |
| **文件** | `_action_gap.py` |
| **行号** | L332-L354 |
| **影响范围** | 行动距离计算 — 非 Pendulum 状态可能落入硬编码物理模拟 |
| **KPI 对标** | E-5 Pendulum 硬编码引用数 ≤ 2（当前此处为第 1 处残留） |

**问题描述**:

`_simulate_action()` 在 action.apply() 和 state.step_physics() 均失败时，回退到 PendulumState 硬编码物理模拟（L332-L354）。

**修复方案**: 移除 Pendulum 硬编码回退，改为返回 `state` 原样（无模拟）并记录 warning。如果需要物理模拟，应通过注入的 PredictorProtocol 实现。

| 修复项 | 详情 |
|--------|------|
| **预估工时** | 1h |
| **优先级** | Phase 1 |

---

#### W-3: 双摆动力学缺少耦合项

| 属性 | 内容 |
|------|------|
| **严重度** | 🟡 Warning |
| **文件** | `_generalized_physics.py` |
| **行号** | L93-L119 |
| **影响范围** | 双摆物理预测精度 |

**问题描述**:

```python
# _generalized_physics.py L113-L117
# 简化动力学（小角度近似）—— 两个摆被当作独立单摆！
d_omega1 = -(g / L1) * np.sin(theta1) + torque1 / (m1 * L1**2)
d_omega2 = -(g / L2) * np.sin(theta2) + torque2 / (m2 * L2**2)
```

第二摆的角加速度 `d_omega2` 仅依赖 `theta2` 本身，未包含来自第一摆的耦合力。标准双摆方程中，两个摆通过约束力相互作用，`d_omega2` 应包含 `theta1` 和 `omega1` 的耦合项。

**修复方案**: 补充拉格朗日方程的耦合项：

```python
# 双摆拉格朗日方程（完整耦合）
delta = theta1 - theta2
den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) ** 2
d_omega1 = (m2 * L1 * omega2**2 * np.sin(delta) * np.cos(delta)
            + m2 * g * np.sin(theta2) * np.cos(delta)
            + m2 * L2 * omega2**2 * np.sin(delta)
            - (m1 + m2) * g * np.sin(theta1)) / den1

den2 = (L2 / L1) * den1
d_omega2 = (-m2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta)
            + (m1 + m2) * g * np.sin(theta1) * np.cos(delta)
            - (m1 + m2) * L1 * omega1**2 * np.sin(delta)
            - (m1 + m2) * g * np.sin(theta2)) / den2
```

| 修复项 | 详情 |
|--------|------|
| **预估工时** | 2h |
| **优先级** | Phase 1 |

---

#### W-4: MetaDiagnoser @dataclass 包含可变状态

| 属性 | 内容 |
|------|------|
| **严重度** | 🟡 Warning |
| **文件** | `_meta_diagnoser.py` |
| **影响范围** | 诊断器线程安全性 |

**问题描述**: `MetaDiagnoser` 使用 `@dataclass` 但包含可变 `_history` 列表，dataclass 默认不安全地共享可变默认值。

**修复方案**: 使用 `field(default_factory=list)` 并添加 `eq=False` 避免比较时的性能问题。

| 修复项 | 详情 |
|--------|------|
| **预估工时** | 0.5h |
| **优先级** | Phase 2 |

---

#### W-5: StateParserRegistry 优先级歧义

| 属性 | 内容 |
|------|------|
| **严重度** | 🟡 Warning |
| **文件** | `_protocols.py` |
| **影响范围** | 状态解析 — 多解析器可解析同一输入时优先级不确定 |

**问题描述**: `StateParserRegistry` 使用逆序优先级注册，但 `GenericStateParser` 作为最通用解析器可能抢先解析特殊状态。

**修复方案**: 引入显式优先级数字（`priority: int`），数字越大优先级越高。

| 修复项 | 详情 |
|--------|------|
| **预估工时** | 1h |
| **优先级** | Phase 2 |

---

#### W-6: _world_model.py 文件过大 (3579 行)

| 属性 | 内容 |
|------|------|
| **严重度** | 🟡 Warning |
| **文件** | `_world_model.py` |
| **影响范围** | 代码可维护性 |
| **规范** | 项目规范：单文件 ≤ 1000 行（理想 ≤ 500 行） |

**问题描述**: `_world_model.py` 有 3579 行，包含 MCIWorldModel 主类 + CEWM 闭环 + 能量流 + 安全 + ROS2 桥接等 10+ 个职责域。

**修复方案**: 将 CEWM 闭环逻辑提取为 `_cewm_engine.py`，能量流提取为 `_energy_flow.py`。

| 修复项 | 详情 |
|--------|------|
| **预估工时** | 4h |
| **优先级** | Phase 3 |

---

#### W-7: NegativeHeuristic._check_rule() 变更检查粒度过粗

| 属性 | 内容 |
|------|------|
| **严重度** | 🟡 Warning |
| **文件** | `_negative_heuristic.py` |
| **影响范围** | 硬核规则保护 — 仅检查变更类型不检查变更内容 |

**问题描述**: `_check_rule()` 通过检查 `change.change_type ∈ forbidden_types` 判断违规，但不检查变更的具体内容是否真正违反硬核规则的精神。

**修复方案**: 增加内容检查，如能量守恒规则检查变更是否新增了无能量补偿的状态转移。

| 修复项 | 详情 |
|--------|------|
| **预估工时** | 2h |
| **优先级** | Phase 2 |

---

### 三、Suggestions（可考虑修复）

> 🟢 **CONSIDER** — 这 9 个问题为架构优化和技术债务管理建议，在 Phase 3 统一处理。

| # | 编号 | 问题 | 文件 | 工时 | 优先级 |
|---|------|------|------|------|--------|
| 1 | S-1 | 延迟初始化模式不统一 | `_world_model.py` | 1.5h | Phase 3 |
| 2 | S-2 | JEPA 嵌入查询测试补充 | `tests/` | 2h | Phase 1 |
| 3 | S-3 | ExperienceDB.to_dict() 序列化不完整 | `_experience_memory.py` | 2h | Phase 3 |
| 4 | S-4 | Ashby 必要多样性阈值缺少量化定义 | `_cognitive_diversity.py` | 1.5h | Phase 3 |
| 5 | S-5 | Pendulum 硬编码追踪矩阵需持续维护 | `docs/` | 1h | Phase 3 |
| 6 | S-6 | CausalUpdater 增量更新 API 缺少 reset() 方法 | `_causal_updater.py` | 1h | Phase 2 |
| 7 | S-7 | cewm_step() 子方法添加独立单元测试 | `tests/` | 3h | Phase 2 |
| 8 | S-8 | _world_model.py 模块拆分设计文档 | `docs/` | 2h | Phase 3 |
| 9 | S-9 | 补充 Wiener 四环跨层误差传播的可视化文档 | `docs/` | 1.5h | Phase 3 |

---

## S-1 ~ S-9 详情

#### S-1: 延迟初始化模式不统一

**问题**: `cewm_step()` 中 `_action_gap_metric`、`_perception`、`_state_parser_registry` 使用 `hasattr` 检查延迟初始化，模式不统一。

**修复**: 统一为 `__init__` 中声明 `None` + 方法中首次访问初始化的标准模式。

#### S-2: JEPA 嵌入查询测试补充

**问题**: C-2 修复后需补充测试验证嵌入查询正确性。

**修复**: 新增 `test_jepa_embedding_query.py`，验证 `to_vector()` → jepa_predict → 非零预测。

#### S-3: ExperienceDB.to_dict() 序列化不完整

**问题**: `to_dict()` 不输出三维索引（_SemanticIndex/_CausalIndex/_TemporalIndex）状态，导致持久化后恢复时丢失索引。

**修复**: 扩展 `to_dict()` 包含索引状态，或使用 `pickle` 作为备选序列化方案。

#### S-4: Ashby 必要多样性阈值缺少量化定义

**问题**: `CognitiveDiversity` 计算 Shannon 熵但缺少 Ashby 定律的量化阈值（H(C) ≥ H(S)）。

**修复**: 添加 `ashby_ratio` 属性和 `is_sufficient_diversity(threshold)` 方法。

#### S-5: Pendulum 硬编码追踪矩阵持续维护

**问题**: 架构泛化 KPI 要求持续追踪 Pendulum 硬编码引用数。

**修复**: 在 CI 中添加自动化追踪脚本。

#### S-6: CausalUpdater 增量更新 API 缺少 reset()

**问题**: C-1 修复后 CausalUpdater 持续积累，但缺少显式重置方法。

**修复**: 添加 `reset()` 方法用于会话边界清理。

#### S-7: cewm_step() 子方法独立单元测试

**问题**: C-4 拆分后的 6 个子方法需要独立测试覆盖。

**修复**: 为每个 `_cewm_*` 子方法编写测试。

#### S-8: _world_model.py 模块拆分设计文档

**问题**: W-6 拆分需要设计文档指导。

**修复**: 编写拆分方案文档，明确各模块边界。

#### S-9: Wiener 四环跨层误差传播可视化

**问题**: 缺少跨层误差传播 Δθ_l(t) = −α·∇‖e‖² + β·e_{l−1}·γ 的可视化。

**修复**: 生成 Mermaid 流程图 + 代码注释标注各层误差传播点。

---

## 实施阶段规划

### Phase 0: 紧急修复 — 止血阶段（Day 1-3）

> **目标**: 修复 C-1~C-5 全部 5 个 Critical Issues，恢复 CEWM 闭环核心功能。
> **原则**: 最小改动、零回归、逐个验证。

#### 时间线

| 天 | 任务 | 编号 | 工时 | 交付物 |
|----|------|------|------|--------|
| Day 1 上午 | C-1 因果图重建修复 | FIX-C1 | 0.5h | `_world_model.py` __init__ 初始化 CausalUpdater |
| Day 1 下午 | C-2 JEPA 字符串查询修复 | FIX-C2 | 1h | `_world_model.py` cewm_step + cewm_step_fast 嵌入查询 |
| Day 2 上午 | C-3 _infer_backend 维度碰撞修复 | FIX-C3 | 1.5h | `_generalized_physics.py` 状态类型注册 + 碰撞警告 |
| Day 2 下午 | C-5 _cewm_state_change 硬编码修复 | FIX-C5 | 1.5h | `_world_state.py` causal_edges() + `_world_model.py` 简化 |
| Day 3 上午 | C-4 cewm_step 拆分重构 | FIX-C4 | 2h | `_world_model.py` 6 个子方法 |
| Day 3 下午 | 回归验证 + 修复验证测试 | VERIFY | 2h | 全量测试通过 + 审查项验证报告 |

#### 详细任务卡

##### FIX-C1: 因果图重建修复

```
变更范围:  _world_model.py L2732-L2735 + __init__
变更类型:  Bug Fix（逻辑修复）
风险等级:  极低

Before:
  L2733: from mci_world_model.sdk._causal_updater import CausalUpdater
  L2735: self._causal_updater = CausalUpdater()  # 每次重建！

After:
  __init__: self._causal_updater: CausalUpdater | None = None
  L2732:   if self._causal_updater is None:
  L2733:       from ..._causal_updater import CausalUpdater
  L2734:       self._causal_updater = CausalUpdater()
  # 不再每次重建

验证:
  1. 连续调用 cewm_step() 3次，检查 causal_updates 累积
  2. 全量 Pendulum 测试零回归
```

##### FIX-C2: JEPA 字符串查询修复

```
变更范围:  _world_model.py L2776-L2780 + L2878-L2886
变更类型:  Bug Fix（逻辑修复）
风险等级:  低

Before:
  L2778: query=str(current_state) if hasattr(current_state, "__str__") else "state"

After:
  if hasattr(current_state, "to_vector"):
      query_vec = current_state.to_vector()
      predictions = self.jepa_predict(query=query_vec)
  else:
      logger.debug("JEPA 跳过: 状态无 to_vector()")

验证:
  1. cewm_step() 后 result["prediction"] 非空且非全零
  2. PendulumState → to_vector() → jepa_predict 查询正常
```

##### FIX-C3: _infer_backend 维度碰撞修复

```
变更范围:  _generalized_physics.py L266-L273 + L351-L364
变更类型:  Bug Fix + 架构增强
风险等级:  中

变更内容:
  1. register_dynamics 增加 state_cls 可选参数
  2. _infer_backend 优先按 type(state) 精确匹配
  3. 维度碰撞时记录 warning 而非静默选第一个

验证:
  1. PendulumState(2D) → "pendulum"（不误选 cart/spring_mass）
  2. CartState(2D) → "cart"
  3. 未注册类型 → 维度碰撞 warning + 默认 backend
```

##### FIX-C4: cewm_step() 拆分重构

```
变更范围:  _world_model.py L2649-L2807
变更类型:  Refactor（纯重构，行为不变）
风险等级:  低

拆分为:
  _init_cewm_result()        → 初始化结果字典
  _cewm_perceive(obs, goal)  → 感知层: 状态解析
  _cewm_safety_check(...)    → 安全层: 约束检查
  _cewm_cognize(...)         → 认知层: 因果更新 + 经验检索
  _cewm_evaluate_action(...) → 行动层: 距离评估
  _cewm_predict(...)         → 预测层: JEPA 预测
  _cewm_feedback(...)        → 反馈层: 注意力调整

  cewm_step() 本体仅做编排（≤30行）

验证:
  1. 拆分前后的 cewm_step() 输出完全一致
  2. cewm_step() 本体 ≤ 30 行
  3. 每个子方法可独立测试
```

##### FIX-C5: _cewm_state_change 硬编码修复

```
变更范围:  _world_model.py L3456-L3500 + _world_state.py
变更类型:  Architecture（开闭原则修正）
风险等级:  低

变更内容:
  1. WorldState ABC 新增 causal_edges() 方法
  2. PendulumState/CartState/RobotWorldState 各自实现
  3. _cewm_state_change() 简化为调用 state.causal_edges()

验证:
  1. PendulumState.causal_edges() 返回 [("theta","omega"),("omega","theta")]
  2. CartState.causal_edges() 返回 [("x","v"),("v","x")]
  3. _cewm_state_change() 代码行数从 ~45 降至 ~5
```

#### Phase 0 完成门禁

> 以下条件**全部**满足，Phase 0 才算完成：

- [ ] C-1: 连续 3 次 cewm_step() 调用，`causal_updates` 累积递增
- [ ] C-2: cewm_step() 的 `result["prediction"]` 非空且非全零向量
- [ ] C-3: PendulumState(2D) 和 CartState(2D) 分别正确推断为各自后端
- [ ] C-4: cewm_step() 本体 ≤ 30 行，6 个子方法各 ≤ 35 行
- [ ] C-5: `_cewm_state_change()` 不含 hasattr(theta) 等硬编码属性检查
- [ ] 全量 Pendulum 测试零回归（`pytest tests/ -q` 通过率 ≥ 99.4%）
- [ ] 新增 `test_causal_accumulation.py` 验证因果图跨步骤积累
- [ ] 新增 `test_jepa_embedding_query.py` 验证嵌入查询

---

### Phase 1: 架构泛化 — 泛化阶段（Week 2）

> **目标**: 修复 W-1~W-4 + S-2，清除 Pendulum 硬编码残留，确保物理预测正确性。
> **KPI 对标**: E-5 Pendulum 硬编码引用数 12+ → ≤ 2

#### 任务清单

| 编号 | 任务 | 关联问题 | 文件 | 工时 | 验证方法 |
|------|------|----------|------|------|----------|
| **GEN-01** | cewm_step_fast() 异常日志化 | W-1 | `_world_model.py` | 0.5h | `except Exception: pass` → `logger.debug()` |
| **GEN-02** | _simulate_action() 移除 Pendulum 回退 | W-2 | `_action_gap.py` | 1h | L332-L354 Pendulum 硬编码移除 |
| **GEN-03** | 双摆动力学耦合项补全 | W-3 | `_generalized_physics.py` | 2h | `double_pendulum_dynamics()` 含耦合项 |
| **GEN-04** | MetaDiagnoser 可变状态修复 | W-4 | `_meta_diagnoser.py` | 0.5h | `field(default_factory=list)` |
| **GEN-05** | JEPA 嵌入查询测试 | S-2 | `tests/` | 2h | 新增 `test_jepa_embedding_query.py` |
| **GEN-06** | Pendulum 硬编码残留审计 | — | 全模块 | 1h | `grep -r "Pendulum" sdk/` 确认 ≤ 2 处 |

#### GEN-03 双摆耦合项验证

```python
# 验证: 双摆混沌行为——初始角度差 0.001 rad 的两条轨迹应在 ~10s 内发散
import numpy as np
from mci_world_model.sdk._generalized_physics import double_pendulum_dynamics

# 两条微小差异的初始状态
s1 = np.array([np.pi/4, 0.0, np.pi/4, 0.0])
s2 = np.array([np.pi/4 + 0.001, 0.0, np.pi/4, 0.0])

dt = 0.01
for _ in range(1000):  # 10 秒
    s1 = s1 + double_pendulum_dynamics(s1, np.zeros(2)) * dt
    s2 = s2 + double_pendulum_dynamics(s2, np.zeros(2)) * dt

divergence = np.linalg.norm(s1 - s2)
assert divergence > 0.5, f"双摆应表现混沌行为，实际发散度: {divergence}"
```

#### Phase 1 完成门禁

- [ ] W-1: cewm_step_fast() 无裸 `except: pass`
- [ ] W-2: _action_gap.py 无 PendulumState 硬编码引用（仅 import 除外）
- [ ] W-3: 双摆动力学含完整拉格朗日耦合项，混沌行为验证通过
- [ ] W-4: MetaDiagnoser 使用 `field(default_factory=list)`
- [ ] Pendulum 硬编码引用 ≤ 2 处（仅测试文件中合理引用）
- [ ] 全量测试零回归
- [ ] 新增 JEPA 嵌入查询测试通过

---

### Phase 2: 闭环验证 — 闭环阶段（Week 3-4）

> **目标**: 修复 W-5~W-7 + S-1 + S-3 + S-6 + S-7，验证 cewm_step() 全流程功能。
> **KPI 对标**: cewm_step() E2E 全流程验证

#### 任务清单

| 编号 | 任务 | 关联问题 | 文件 | 工时 | 验证方法 |
|------|------|----------|------|------|----------|
| **LOOP-01** | StateParserRegistry 优先级数字化 | W-5 | `_protocols.py` | 1h | 解析器按 priority 排序 |
| **LOOP-02** | NegativeHeuristic 内容级检查 | W-7 | `_negative_heuristic.py` | 2h | `_check_rule()` 检查变更内容 |
| **LOOP-03** | 延迟初始化模式统一 | S-1 | `_world_model.py` | 1.5h | 所有延迟属性在 __init__ 声明 None |
| **LOOP-04** | ExperienceDB 序列化完整性 | S-3 | `_experience_memory.py` | 2h | `to_dict()` 含三维索引状态 |
| **LOOP-05** | CausalUpdater reset() 方法 | S-6 | `_causal_updater.py` | 1h | `reset()` 清空因果图 |
| **LOOP-06** | cewm_step 子方法单元测试 | S-7 | `tests/` | 3h | 每个 `_cewm_*` 方法有测试 |
| **LOOP-07** | cewm_step E2E 集成测试 | — | `tests/` | 2h | 感知→认知→预测→行动→反馈全链路 |

#### LOOP-07 cewm_step E2E 测试设计

```python
# test_cewm_e2e_closed_loop.py
class TestCEWMClosedLoop:
    """验证 cewm_step() 五层闭环全流程。"""

    def test_perception_layer(self):
        """感知层: PendulumState → 解析为 WorldState"""
        ...

    def test_cognition_layer_accumulation(self):
        """认知层: 多步因果图积累"""
        wm = MCIWorldModel()
        for i in range(5):
            result = wm.cewm_step(
                observation=PendulumState(theta=0.1*i, omega=0.01),
                goal=PendulumState(theta=0.0, omega=0.0),
            )
        # 第5步的 causal_updates 应 > 第1步
        assert result["causal_updates"] > 0

    def test_prediction_layer_embedding(self):
        """预测层: JEPA 嵌入查询返回非零预测"""
        ...

    def test_action_layer_distance(self):
        """行动层: 行动距离计算正确"""
        ...

    def test_feedback_layer_attention(self):
        """反馈层: 注意力权重基于预测误差调整"""
        ...

    def test_full_closed_loop_5_steps(self):
        """完整闭环: 连续 5 步 cewm_step()"""
        ...
```

#### Phase 2 完成门禁

- [ ] W-5: StateParserRegistry 使用显式 priority 排序
- [ ] W-7: NegativeHeuristic 检查变更内容而非仅类型
- [ ] S-1: 所有延迟初始化属性在 `__init__` 中声明 `None`
- [ ] S-3: ExperienceDB `to_dict()` 可完整恢复三维索引
- [ ] S-6: CausalUpdater `reset()` 方法可用
- [ ] S-7: cewm_step 6 个子方法各有 ≥ 2 个测试
- [ ] cewm_step E2E 5 步闭环测试通过
- [ ] 全量测试零回归

---

### Phase 3: 质量提升 — 就绪阶段（Week 5-8）

> **目标**: 修复 W-6 + S-4 + S-5 + S-8 + S-9，完成模块拆分和文档完善。
> **KPI 对标**: _world_model.py ≤ 1500 行、覆盖率 ≥ 60%

#### 任务清单

| 编号 | 任务 | 关联问题 | 文件 | 工时 | 验证方法 |
|------|------|----------|------|------|----------|
| **QUAL-01** | _world_model.py 模块拆分 | W-6, S-8 | 多文件 | 4h | 主文件 ≤ 1500 行 |
| **QUAL-02** | Ashby 阈值量化 | S-4 | `_cognitive_diversity.py` | 1.5h | `ashby_ratio` 属性 |
| **QUAL-03** | Pendulum 硬编码 CI 追踪 | S-5 | `scripts/` | 1h | CI 脚本自动统计 |
| **QUAL-04** | Wiener 四环可视化文档 | S-9 | `docs/` | 1.5h | Mermaid 图 + 代码注释 |
| **QUAL-05** | 审查修复变更日志 | — | `CHANGELOG.md` | 1h | 记录 C-1~C-5 + W-1~W-7 修复 |

#### QUAL-01 _world_model.py 拆分方案

```
当前: _world_model.py (3579 行)
        ├── MCIWorldModel 主类 (~2000 行)
        ├── cewm_step / cewm_step_fast (~250 行)
        ├── EnergyFlow 相关 (~300 行)
        ├── ROS2 桥接接口 (~200 行)
        └── 辅助方法 (~800 行)

拆分后:
        _world_model.py     → MCIWorldModel 主类 (≤1200 行)
        _cewm_engine.py     → cewm_step + 子方法 (≤400 行)
        _energy_flow.py     → EnergyFlowPredictor (≤400 行)
        _world_model_helpers.py → 辅助方法 (≤500 行)
```

#### Phase 3 完成门禁

- [ ] W-6: `_world_model.py` ≤ 1500 行（或拆分方案文档已完成）
- [ ] S-4: CognitiveDiversity 含 `ashby_ratio` 和 `is_sufficient_diversity()`
- [ ] S-5: CI 中有 Pendulum 硬编码自动追踪脚本
- [ ] S-8: 模块拆分设计文档已完成
- [ ] S-9: Wiener 四环可视化文档已完成
- [ ] CHANGELOG 记录全部 21 个问题修复
- [ ] 代码质量评分 ⭐⭐⭐⭐⭐ (5.0/5)

---

## 资源分配与时间安排

### 总体时间线

```
Week 1              Week 2              Week 3-4            Week 5-8
Phase 0 (止血)      Phase 1 (泛化)      Phase 2 (闭环)      Phase 3 (就绪)
3天                 5天                 10天                20天

C-1 C-2             W-1 W-2             W-5 W-7             W-6 S-8
C-3 C-5             W-3 W-4             S-1 S-3             S-4 S-5
C-4                 S-2                 S-6 S-7             S-9
                    GEN-06              LOOP-07             QUAL-05

═══════════         ═══════════         ═══════════         ═══════════
M0: 止血完成         M1: 泛化达标         M2: 闭环验证         M3: 质量提升
Day 3               Week 2 末            Week 4 末            Week 8 末
```

### 工时汇总

| 阶段 | 任务数 | 设计工时 | 编码工时 | 测试工时 | 总工时 | 人日（1人） |
|------|--------|---------|---------|---------|--------|-----------|
| Phase 0 | 5 | 1h | 6.5h | 2h | **9.5h** | 1.2 |
| Phase 1 | 6 | 1h | 5h | 3h | **9h** | 1.1 |
| Phase 2 | 7 | 2h | 10.5h | 5h | **17.5h** | 2.2 |
| Phase 3 | 5 | 3h | 6h | 3h | **12h** | 1.5 |
| **合计** | **23** | **7h** | **28h** | **13h** | **48h** | **6.0** |

### 与现有计划的关系

| 计划书 | 聚焦 | 时间 | 关系 |
|--------|------|------|------|
| **本计划书** | 代码审查发现的逻辑 Bug + 架构问题 | Week 1-8 | 修复 C/W/S 级问题 |
| `修复实施计划书_20260603.md` | 测试失败 + ruff + mypy | Week 1-10 | 修复工程规范问题 |
| `CEWM-架构加固与泛化-KPI实施规划书.md` | Pendulum 硬编码解耦 + 泛化路线 | Week 1-12 | 架构演进路线图 |

**并行策略**: 本计划书的 Phase 0-1 可与 `修复实施计划书` 的 Phase 1 并行执行（不冲突）。Phase 2-3 可与 `KPI实施规划书` 的 Phase 0-1 并行（部分任务重叠，如硬编码清理）。

---

## 风险评估与应对措施

### 风险矩阵

| # | 风险 | 概率 | 影响 | 风险等级 | 应对措施 |
|---|------|------|------|----------|----------|
| R1 | **C-2 修复后 JEPA 预测行为变化导致测试失败** | 中 | 中 | 🟡 | 修复前先运行 `pytest -k jepa` 确认基线；修复后逐个排查失败测试 |
| R2 | **C-4 拆分引入隐藏的行为差异** | 低 | 高 | 🟡 | 拆分前生成 cewm_step() 输出快照；拆分后逐字段对比 |
| R3 | **C-3 状态类型注册遗漏导致推断失败** | 中 | 低 | 🟢 | 在 `_infer_backend` 中保留维度回退 + warning |
| R4 | **W-3 双摆耦合项引入数值不稳定** | 低 | 中 | 🟢 | 使用小步长 dt=0.001 验证；添加能量守恒检查 |
| R5 | **Phase 0 与现有测试计划并行产生合并冲突** | 中 | 低 | 🟢 | Phase 0 仅修改 `_world_model.py` 和 `_generalized_physics.py`，与 ruff/mypy 修复不冲突 |
| R6 | **C-1 修复后因果图无限增长导致内存泄漏** | 低 | 中 | 🟡 | 添加 CausalUpdater 最大边数限制（S-6 reset 方法辅助） |
| R7 | **_world_model.py 拆分破坏导入路径** | 中 | 高 | 🟡 | 拆分后保留 `_world_model.py` 中的 re-export，确保 `from mci_world_model.sdk._world_model import MCIWorldModel` 不变 |

### 应急回滚方案

| 变更 | 回滚方式 | 回滚成本 |
|------|----------|----------|
| C-1 因果图修复 | 恢复 `self._causal_updater = CausalUpdater()` | 1 min（git revert） |
| C-2 JEPA 修复 | 恢复 `query=str(current_state)` | 1 min |
| C-3 后端推断 | 恢复维度遍历逻辑 | 5 min |
| C-4 函数拆分 | 恢复内联实现 | 10 min（git revert） |
| C-5 因果边方法 | 恢复 hasattr 分支 | 5 min |
| W-3 双摆耦合 | 恢复独立摆近似 | 1 min |
| W-6 文件拆分 | 合并回 `_world_model.py` | 30 min |

---

## 验收标准

### M0: 止血完成（Phase 0，Day 3）

| 验收项 | 标准 | 验证命令/方法 |
|--------|------|--------------|
| C-1 因果图积累 | 连续 3 步 cewm_step，causal_updates 累积递增 | `pytest tests/test_causal_accumulation.py -v` |
| C-2 JEPA 嵌入查询 | cewm_step prediction 非空且非全零 | `pytest tests/test_jepa_embedding_query.py -v` |
| C-3 后端推断 | PendulumState → "pendulum"，CartState → "cart" | 单元测试 |
| C-4 函数长度 | cewm_step() ≤ 30 行 | `wc -l` 或 AST 分析 |
| C-5 硬编码清除 | `_cewm_state_change()` 无 hasattr(theta/omega/x/v) | `grep "hasattr.*theta" _world_model.py` 为空 |
| 零回归 | 全量测试通过率 ≥ 99.4% | `.venv/bin/pytest tests/ -q` |

### M1: 泛化达标（Phase 1，Week 2 末）

| 验收项 | 标准 | 验证方法 |
|--------|------|----------|
| W-1 异常日志 | cewm_step_fast 无裸 `except: pass` | `grep "except.*pass" _world_model.py` |
| W-2 Pendulum 回退移除 | _action_gap.py 无 PendulumState 硬编码 | `grep "PendulumState" _action_gap.py` 仅剩 import |
| W-3 双摆耦合 | 双摆混沌行为验证通过 | `pytest tests/test_double_pendulum_chaos.py` |
| W-4 dataclass 安全 | MetaDiagnoser 使用 field(default_factory) | 代码审查 |
| Pendulum 硬编码 | ≤ 2 处（仅测试中） | `grep -r "Pendulum" sdk/ --include="*.py" \| grep -v test \| wc -l` ≤ 2 |
| 零回归 | 全量测试通过 | `.venv/bin/pytest tests/ -q` |

### M2: 闭环验证（Phase 2，Week 4 末）

| 验收项 | 标准 | 验证方法 |
|--------|------|----------|
| W-5 解析器优先级 | StateParserRegistry 按 priority 排序 | 单元测试 |
| W-7 规则内容检查 | NegativeHeuristic 检查变更内容 | 单元测试 |
| cewm_step E2E | 5 步闭环测试通过 | `pytest tests/test_cewm_e2e_closed_loop.py -v` |
| 子方法覆盖 | 每个 _cewm_* 方法 ≥ 2 个测试 | 测试计数 |
| 序列化完整性 | ExperienceDB to_dict → from_dict 等价 | 单元测试 |
| 零回归 | 全量测试通过 | `.venv/bin/pytest tests/ -q` |

### M3: 质量提升（Phase 3，Week 8 末）

| 验收项 | 标准 | 验证方法 |
|--------|------|----------|
| W-6 文件大小 | _world_model.py ≤ 1500 行（或拆分方案文档完成） | `wc -l _world_model.py` |
| S-4 Ashby 量化 | CognitiveDiversity 含 ashby_ratio | 单元测试 |
| CI 硬编码追踪 | scripts/ 中有自动追踪脚本 | 脚本执行 |
| 代码质量评分 | ⭐⭐⭐⭐⭐ (5.0/5) | 综合评估 |
| CHANGELOG | 全部 21 个问题修复记录 | 文档审查 |

### 最终验收 — 基线对比

| 指标 | 审查时基线 | M0 目标 | M1 目标 | M2 目标 | M3 目标 |
|------|-----------|---------|---------|---------|---------|
| Critical Issues | 5 | **0** | 0 | 0 | 0 |
| Warnings | 7 | 7 | **3** (W-1~W-4 修复) | **0** | 0 |
| Suggestions | 9 | 9 | 8 (S-2) | **3** (S-4,5,8,9) | **0** |
| cewm_step 行数 | 158 | **≤30** (编排) | ≤30 | ≤30 | ≤30 |
| Pendulum 硬编码 | 12+ | 12+ | **≤2** | ≤2 | ≤2 |
| 因果图积累 | ❌ 失效 | ✅ **生效** | ✅ | ✅ | ✅ |
| JEPA 嵌入查询 | ❌ 字符串 | ✅ **嵌入** | ✅ | ✅ | ✅ |
| _world_model.py 行数 | 3579 | 3579 | 3579 | 3579 | **≤1500** |
| 代码质量评分 | ⭐⭐⭐⭐☆ (4.5) | ⭐⭐⭐⭐☆ (4.7) | ⭐⭐⭐⭐☆ (4.8) | ⭐⭐⭐⭐⭐ (4.9) | ⭐⭐⭐⭐⭐ (5.0) |

---

## 附录

### A. 审查发现代码位置索引

| 编号 | 文件 | 行号 | 问题描述 |
|------|------|------|----------|
| C-1 | `_world_model.py` | L2733-L2735 | `self._causal_updater = CausalUpdater()` 每次重建 |
| C-2 | `_world_model.py` | L2778-L2780, L2881-L2883 | `query=str(current_state)` JEPA 字符串查询 |
| C-3 | `_generalized_physics.py` | L351-L364 | `_infer_backend()` 维度碰撞 |
| C-4 | `_world_model.py` | L2649-L2807 | `cewm_step()` 158 行 |
| C-5 | `_world_model.py` | L3456-L3500 | `_cewm_state_change()` 硬编码 hasattr |
| W-1 | `_world_model.py` | L2885-L2886 | `except Exception: pass` 静默 |
| W-2 | `_action_gap.py` | L332-L354 | Pendulum 硬编码物理回退 |
| W-3 | `_generalized_physics.py` | L93-L119 | 双摆缺耦合项 |
| W-4 | `_meta_diagnoser.py` | 类定义 | @dataclass 可变状态 |
| W-5 | `_protocols.py` | StateParserRegistry | 优先级歧义 |
| W-6 | `_world_model.py` | 全文件 | 3579 行 |
| W-7 | `_negative_heuristic.py` | `_check_rule()` | 变更检查粒度过粗 |

### B. 新增测试文件清单

| 文件 | Phase | 测试数 | 说明 |
|------|-------|--------|------|
| `tests/test_causal_accumulation.py` | Phase 0 | ~3 | 验证因果图跨步骤积累（C-1） |
| `tests/test_jepa_embedding_query.py` | Phase 0/1 | ~3 | 验证 JEPA 嵌入查询（C-2） |
| `tests/test_backend_inference.py` | Phase 0 | ~4 | 验证后端推断无碰撞（C-3） |
| `tests/test_cewm_submethod_units.py` | Phase 2 | ~12 | cewm_step 子方法单元测试（S-7） |
| `tests/test_cewm_e2e_closed_loop.py` | Phase 2 | ~6 | CEWM 五层闭环 E2E（LOOP-07） |
| `tests/test_double_pendulum_chaos.py` | Phase 1 | ~2 | 双摆混沌行为验证（W-3） |

### C. 关键决策记录

| 决策 | 选项 | 选择 | 理由 |
|------|------|------|------|
| C-2 JEPA 查询修复 | (a) 嵌入向量 (b) 字符串哈希 (c) TF-IDF | (a) 嵌入向量 | 符合 LeCun JEPA 原始设计；to_vector() 已有实现 |
| C-3 后端推断 | (a) 状态类型注册 (b) 仅维度+警告 (c) 强制显式设置 | (a) + (b) 混合 | 类型注册精确，维度回退兼容旧代码 |
| C-4 拆分粒度 | (a) 6 个子方法 (b) 3 个子方法 (c) 不拆分 | (a) 6 个子方法 | 每层独立可测试，符合五层架构 |
| C-5 因果边设计 | (a) WorldState.causal_edges() (b) 外部注册表 (c) 配置文件 | (a) causal_edges() | 开闭原则最优；状态类自描述因果结构 |
| W-3 双摆方程 | (a) 完整拉格朗日 (b) 小角度线性耦合 (c) 数值积分查表 | (a) 完整拉格朗日 | 物理正确性优先；混沌行为是双摆本质特征 |

### D. 工具链

| 用途 | 工具 | 命令 |
|------|------|------|
| 测试执行 | pytest | `.venv/bin/pytest tests/ -q` |
| 类型检查 | mypy | `.venv/bin/mypy src/mci_world_model/sdk/_world_model.py` |
| 代码风格 | ruff | `.venv/bin/ruff check` |
| 硬编码追踪 | grep | `grep -r "Pendulum" src/mci_world_model/sdk/ --include="*.py" \| grep -v test` |
| 函数行数统计 | Python AST | `python -c "import ast; ..."` |
| 覆盖率 | pytest-cov | `.venv/bin/pytest --cov=mci_world_model tests/` |

---

**文档版本**: v1.0
**制定人**: Qoder AI 代码审查团队
**审批状态**: 待确认
**下一步**: 用户确认后开始执行 Phase 0 紧急修复（Day 1-3）
