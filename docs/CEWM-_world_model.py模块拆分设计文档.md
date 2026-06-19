# _world_model.py 模块拆分设计文档

> **QUAL-01 (W-6/S-8)** — CEWM v4.0.0 代码审查修复 Phase 3 交付物
> 
> 本文档定义 `_world_model.py`（3604 行）的模块拆分方案，
> 确保拆分后主文件 ≤ 1500 行，且导入路径完全向后兼容。

---

## 1. 现状分析

### 1.1 当前文件结构

```
_world_model.py (3604 行)
├── _aggregate_energy_ratios()      L79-L98       (20 行)  — 顶层辅助函数
├── CausalWorldModelState           L107-L446     (340 行) — 因果世界状态
├── TrajectoryStep                  L455-L474     (20 行)  — 轨迹步
├── WorkingMemory                   L478-L678     (201 行) — 工作记忆
└── MCIWorldModel                   L686-L3604    (2919 行, 65 方法) — 主类
```

### 1.2 MCIWorldModel 方法分类（65 方法）

| 功能域 | 方法数 | 行数估计 | 代表方法 |
|--------|--------|----------|----------|
| **生命周期** | 5 | ~400 | `__init__`, `initialize`, `discover`, `close`, `__repr__` |
| **CEWM 引擎** | 12 | ~500 | `cewm_step`, `cewm_step_fast`, `_cewm_*` (8个子方法), `_init_cewm_result` |
| **JEPA 预测** | 4 | ~250 | `jepa_predict`, `train_jepa`, `_get_jepa_*` |
| **因果推理** | 6 | ~500 | `intervene`, `decompose_effect`, `query_counterfactual`, `_trace_causal_chains` |
| **健康诊断** | 4 | ~300 | `health_check`, `auto_regulate`, `_compute_*` |
| **规划** | 3 | ~150 | `plan_action`, `_plan_*` |
| **管线编排** | 2 | ~200 | `six_module_pipeline`, `_execute_pipeline` |
| **能量流** | 3 | ~150 | `_get_energy_*`, `_aggregate_energy_*` |
| **辅助/序列化** | 8 | ~200 | `to_dict`, `from_dict`, `get_status`, `_validate_*` |
| **ROS2 桥接** | 4 | ~100 | `_ros2_*` |
| **属性/委托** | 14 | ~100 | `@property` 委托方法 |

---

## 2. 拆分方案

### 2.1 目标模块结构

```
sdk/
├── _world_model.py          ← MCIWorldModel 主类骨架 (≤1200 行)
├── _cewm_engine.py          ← CEWM 闭环引擎 (≤400 行)        ★ 新建
├── _energy_flow.py          ← EnergyFlowPredictor (≤200 行)   ★ 新建
├── _world_model_helpers.py  ← 辅助函数和数据类 (≤500 行)      ★ 新建
└── (现有模块保持不变)
```

### 2.2 模块边界定义

#### 模块 A: `_cewm_engine.py` — CEWM 闭环引擎

**职责**: Wiener 四环嵌套反馈闭环的执行逻辑

**迁出方法** (12 个):
```python
# 从 MCIWorldModel 迁出，转为独立函数或 Mixin
cewm_step()                    # L2780-L2852  (73 行)  — 编排层
cewm_step_fast()               # L2854-L2945  (92 行)  — 快速路径
_init_cewm_result()            # 编排辅助
_cewm_perceive()               # Layer 0
_cewm_safety_check()           # 安全层
_cewm_cognize()                # Layer 1
_cewm_evaluate_action()        # Layer 3
_cewm_predict()                # Layer 2
_cewm_feedback()               # 反馈层
_cewm_parse_state()            # 状态解析
_cewm_state_change()           # 因果边提取
_cewm_get_causal_updater()     # 因果图访问
```

**设计模式**: Mixin 类
```python
# _cewm_engine.py
class CEWMEngineMixin:
    """CEWM 闭环引擎 Mixin —— 由 MCIWorldModel 继承。
    
    依赖宿主类提供:
    - self._causal_updater
    - self._action_gap_metric  
    - self._state_parser_registry
    - self._perception
    - self._jepa_predictor
    - self._safety_monitor
    """
    
    def cewm_step(self, observation=None, goal=None, action=None): ...
    def cewm_step_fast(self, observation=None, goal=None, action=None): ...
    # ... 其他 _cewm_* 方法
```

**合并方式**: 多重继承
```python
# _world_model.py
from ._cewm_engine import CEWMEngineMixin

class MCIWorldModel(CEWMEngineMixin, ...):
    ...
```

#### 模块 B: `_energy_flow.py` — 能量流预测器

**职责**: 能量守恒验证和能量流预测

**迁出方法** (3 个):
```python
_get_energy_ratios()           # 能量比率计算
_aggregate_energy_ratios()     # L79-L98  — 能量聚合（已为顶层函数）
_get_energy_balance()          # 能量平衡验证
```

**设计模式**: 独立工具模块
```python
# _energy_flow.py
def compute_energy_ratios(state: WorldState) -> dict: ...
def aggregate_energy_ratios(ratios: list[dict]) -> dict: ...
def check_energy_balance(state: WorldState, prediction: Any) -> float: ...
```

#### 模块 C: `_world_model_helpers.py` — 辅助函数和数据类

**职责**: 不依赖 MCIWorldModel 实例状态的辅助函数和数据结构

**迁出内容**:
```python
# 数据类（已是独立类，只需移动）
CausalWorldModelState    # L107-L446  (340 行)
TrajectoryStep           # L455-L474  (20 行)
WorkingMemory            # L478-L678  (201 行)

# 辅助函数
_validate_state()        # 状态验证
_serialize_result()      # 结果序列化
_compute_status_score()  # 健康分数计算
```

### 2.3 主文件 `_world_model.py` 保留内容

**保留在主文件的方法** (~1200 行):
```python
class MCIWorldModel(CEWMEngineMixin):
    # 生命周期 (~400 行)
    __init__()          # L721-L813
    initialize()        # L819-L957
    discover()          # L1112-L1246
    close()
    
    # JEPA 预测 (~250 行)
    jepa_predict()      # L1288-L1386
    train_jepa()        # L2014-L2088
    
    # 因果推理 (~500 行)
    intervene()         # L1555-L1680
    decompose_effect()  # L1686-L1787
    query_counterfactual()  # L1793-L1873
    _trace_causal_chains()  # L1917-L1989
    
    # 健康诊断 (~300 行)
    health_check()      # L2094-L2210
    auto_regulate()     # L2262-L2329
    
    # 管线编排 (~200 行)
    six_module_pipeline()   # L2335-L2472
    
    # 规划 (~150 行)
    plan_action()       # L3172-L3240
    
    # 属性委托 (~100 行)
    # @property methods
```

---

## 3. 向后兼容策略

### 3.1 Re-export 保证

拆分后 `_world_model.py` 保持所有现有导入路径不变：

```python
# _world_model.py — 文件头部 re-export
from ._world_model_helpers import (
    CausalWorldModelState,
    TrajectoryStep,
    WorkingMemory,
)
from ._cewm_engine import CEWMEngineMixin
from ._energy_flow import compute_energy_ratios

__all__ = [
    "MCIWorldModel",
    "CausalWorldModelState",
    "TrajectoryStep",
    "WorkingMemory",
    # ... 其他原有导出
]
```

### 3.2 外部导入不变

```python
# 以下所有导入路径拆分前后完全一致：
from mci_world_model.sdk._world_model import MCIWorldModel
from mci_world_model.sdk._world_model import CausalWorldModelState
from mci_world_model.sdk._world_model import WorkingMemory
# 全部正常工作 ✓
```

### 3.3 `isinstance` 检查不变

```python
# 拆分前
wm = MCIWorldModel()
isinstance(wm, MCIWorldModel)  # True

# 拆分后（多重继承）
wm = MCIWorldModel()
isinstance(wm, MCIWorldModel)  # True ✓
isinstance(wm, CEWMEngineMixin)  # True（新增，不影响现有代码）
```

---

## 4. 迁移步骤

### Phase 3a: 数据类提取（低风险）

1. 将 `CausalWorldModelState`、`TrajectoryStep`、`WorkingMemory` 移至 `_world_model_helpers.py`
2. 在 `_world_model.py` 中添加 re-export
3. 运行全量测试验证零回归

### Phase 3b: 能量流提取（低风险）

1. 将能量流相关方法提取为 `_energy_flow.py` 中的独立函数
2. MCIWorldModel 中保留委托调用
3. 运行全量测试

### Phase 3c: CEWM 引擎提取（中风险）

1. 创建 `_cewm_engine.py`，定义 `CEWMEngineMixin`
2. 将 `cewm_step()`、`cewm_step_fast()` 和 8 个 `_cewm_*` 子方法移入 Mixin
3. MCIWorldModel 继承 `CEWMEngineMixin`
4. 生成 `cewm_step()` 输出快照，拆分后逐字段对比
5. 运行全量测试 + E2E 闭环测试

### Phase 3d: 主文件精简验证

1. 验证主文件行数 ≤ 1500 行
2. 运行全量测试（目标 3568+ passed）
3. 运行 CI 脚本验证 Pendulum 硬编码引用

---

## 5. 风险评估

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|----------|
| Mixin 方法间 `self` 依赖混乱 | 低 | 中 | 明确 Mixin 依赖接口，使用 `TYPE_CHECKING` 声明类型约束 |
| 数据类循环导入 | 中 | 高 | `_world_model_helpers.py` 不导入 `_world_model.py`；使用 `TYPE_CHECKING` 前向引用 |
| JEPA/Causal 引用断裂 | 低 | 高 | 保持方法签名不变，仅改变物理位置 |
| 反序列化兼容性 | 低 | 中 | `from_dict()` 保持兼容，旧格式自动升级 |

---

## 6. 验收标准

| 验收项 | 标准 | 验证方法 |
|--------|------|----------|
| 主文件行数 | `_world_model.py` ≤ 1500 行 | `wc -l` |
| 导入兼容 | 所有现有 `from ... import` 路径不变 | `grep -r "from.*_world_model import" tests/ src/` 无变更 |
| 测试零回归 | 3568+ 测试全部通过 | `pytest tests/ -q` |
| E2E 闭环 | `test_cewm_e2e_closed_loop.py` 18/18 通过 | `pytest tests/test_cewm_e2e_closed_loop.py -v` |
| 模块边界清晰 | 无循环导入 | `python -c "from mci_world_model.sdk._world_model import MCIWorldModel"` |

---

*本文档为 CEWM v4.0.0 代码审查修复 Phase 3 交付物，关联 W-6 + S-8 问题修复。*
