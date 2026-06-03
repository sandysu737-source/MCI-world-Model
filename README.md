# MCI World Model

[![CI](https://github.com/sandysu737-source/mci-world-model/actions/workflows/ci.yml/badge.svg)](https://github.com/sandysu737-source/mci-world-model/actions)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![Version](https://img.shields.io/badge/version-3.0.0-orange)](https://github.com/sandysu737-source/mci-world-model/releases)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Memor-Engine](https://img.shields.io/badge/su--memory--sdk-%3E%3D3.5.1-blue)](https://github.com/sandysu737-source/su-memory-sdk)

> **因果世界模型引擎** — Pearl 三层 + JEPA 世界建模 + 能量中心

---

## 🎯 项目定位

MCI World Model 是一款**独立运行的因果世界模型引擎**，定位为通用 AI 系统的「**CPU**」：

- ✅ **独立运行** — 不依赖 Transformer / GPU
- ✅ **可插拔加速器** — Transformer 可作为可选的「GPU」插在后端，随时更换或移除
- ✅ **可解释因果** — 完整覆盖 Pearl 三层（关联 / 干预 / 反事实）
- ✅ **能量中心** — 三才合一（天 / 地 / 人）统一信息建模
- ✅ **JEPA 世界建模** — 纯 NumPy / PyTorch 可微世界预测器

**与记忆引擎（su-memory-sdk V3.5.1）的关系**：

| 项目 | 角色 | 版本 |
|------|------|------|
| **su-memory-sdk** | 记忆引擎（短期 / 长期 / 检索） | V3.5.1 |
| **MCI World Model** | 世界模型（因果推理 / 干预 / 反事实） | V3.0.0 |

> 📌 本项目是 **su-memory-sdk V3.6.0+** 起分离出来的独立仓库
> 📌 项目内的工程内部版本仍标记为 V4.0.0，对外品牌为 **MCI World Model V3.0.0**

---

## 🧬 三大支柱

### 1️⃣ Pearl Causal Hierarchy — 因果三层

| 层 | 名称 | 引擎 |
|----|------|------|
| **P1** | 关联 (Association) | `FourierCausal` / `BayesianCausal` / `GaussianDAG` |
| **P2** | 干预 (Intervention) | `DoCalculus` / `CausalGraph` |
| **P3** | 反事实 (Counterfactual) | `CounterfactualEngine` |

```python
from mci_world_model.sdk import CausalGraph, DoCalculus

# 构建因果图
cg = CausalGraph(
    nodes=["Z", "X", "Y"],
    edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
)

# 干预推理：do(X = 1)
dc = DoCalculus(cg)
effect = dc.intervene("X", value=1, target="Y")
print(f"ATE = {effect.ate:.4f}")
```

### 2️⃣ JEPA World Modeling — 世界建模

```python
from mci_world_model.sdk import (
    JEPADataset, JEPAEncoder, JEPAPredictor, JEPATrainer,
)

# 数据 → 编码器 → 预测器
dataset = JEPADataset(X, edges, num_nodes)
encoder = JEPAEncoder(in_dim=64, hidden_dim=128, out_dim=32)
predictor = JEPAPredictor(in_dim=32, hidden_dim=64, out_dim=32)

trainer = JEPATrainer(encoder, predictor, lr=1e-3)
trainer.fit(dataset, epochs=100)
```

### 3️⃣ Energy Center — 能量中心三才合一

```
┌────────────────────────────────────────────┐
│  UnifiedInfoUnit — 天+地+人 三才合一         │
├──────────┬──────────────┬───────────────────┤
│ 天 (Sky) │ 地 (Earth)   │ 人 (Human)        │
│ 时空感知 │ 64卦三层推断  │ 五行能量 + 加权因果 │
│ 六十甲子 │ TrigramCore  │ EnergyCore        │
│ Temporal │ PatternInfer │ CausalEngine      │
└──────────┴──────────────┴───────────────────┘
```

---

## 📦 安装

### 基础安装
```bash
pip install mci-world-model
```

### 完整安装（含 JEPA + 基准）
```bash
pip install mci-world-model[full]
```

### 源码安装（开发）
```bash
git clone https://github.com/sandysu737-source/mci-world-model.git
cd mci-world-model
pip install -e ".[dev]"
```

---

## 🚀 5 分钟上手

```python
from mci_world_model.sdk import (
    MCIWorldModel, CausalWorldModelState,
    CausalGraph, DoCalculus, CounterfactualEngine,
    JEPAEncoder, GNNPredictor, JEPATrainer,
    SIGReg,
)

# 1. 创建世界模型实例
wm = MCIWorldModel()

# 2. 加载因果图
cg = CausalGraph(
    nodes=["天气", "出行", "心情", "效率"],
    edges=[("天气", "心情"), ("天气", "出行"), ("心情", "效率"), ("出行", "效率")],
)

# 3. 干预推理：天气 = 阴天
dc = DoCalculus(cg)
effect = dc.intervene("天气", value="阴天", target="效率")
print(f"do(天气=阴天) → E[效率] = {effect.ate:.3f}")

# 4. 反事实推理：如果当时是晴天，效率会怎样
cf = CounterfactualEngine(cg)
counterfactual = cf.imagine(
    factual={"天气": "雨天"},
    intervention={"天气": "晴天"},
    target="效率",
)
print(f"反事实: P(效率 | 天气=晴天, 观察=雨天) = {counterfactual.probability:.3f}")
```

---

## 🏗️ 架构

```
mci-world-model/
├── src/mci_world_model/
│   ├── __init__.py              # 顶层统一入口
│   ├── world_model.py           # 能量中心三才合一统一入口
│   ├── _sys/                    # 系统层（15 个模块）
│   │   ├── _energy_core.py      # 能量核心
│   │   ├── _temporal_core.py    # 时空感知
│   │   ├── _causal_engine.py    # 因果引擎
│   │   ├── bayesian*.py         # 贝叶斯系统
│   │   └── ...
│   └── sdk/                     # SDK 层（16 个核心模块）
│       ├── _world_model.py      # MCIWorldModel 核心
│       ├── _do_calculus.py      # Pearl Do-Calculus
│       ├── _counterfactual.py   # 反事实推理
│       ├── _jepa_*.py           # JEPA 全套（6 个）
│       ├── _spectral_causal.py  # 频谱因果
│       ├── _energy_loss.py      # 能量一致性
│       ├── _parametric_memory.py # 参数化记忆
│       └── bayesian_augmenter.py
├── benchmarks/
│   ├── sigreg/                  # SIGReg 嵌入正则化基准
│   └── _noise_generator.py
├── docs/
│   ├── 本地优先训练路线重评估.md
│   └── ROADMAP_V3.0.0.md
├── tests/                       # 单元测试
└── assets/                      # 资源文件
```

---

## 🧪 测试

```bash
# 全部测试
pytest tests/ -v

# 带覆盖率
pytest tests/ --cov=mci_world_model --cov-report=term-missing

# 跳过慢测试
pytest tests/ -v -m "not slow"
```

---

## 🛠️ 工具链

| 工具 | 用途 | 最低版本 |
|------|------|---------|
| **Ruff** | Lint / Format | 0.6.0+ |
| **Mypy** | 类型检查（strict） | 1.10+ |
| **Pytest** | 单元测试 | 7.4+ |
| **PyTorch**（可选） | JEPA GNN 训练 | 2.0+ |

---

## 📊 路线图

| 版本 | 主题 | 状态 |
|------|------|------|
| **V3.0.0**（更名前 V4.0.0） | Pearl Do-Calculus + JEPA 全套 + 能量中心 | 🚧 当前版本 |
| V3.1.0 | Pearl P3 反事实推理强化 + Causal Counterfactual | 📋 规划中 |
| V3.2.0 | JEPA GAT 大规模图训练 + LongMemEval | 📋 规划中 |
| V4.0.0 | 多模态因果世界模型 + 真实环境部署 | 📋 规划中 |

详见 [docs/ROADMAP_V3.0.0.md](docs/ROADMAP_V3.0.0.md)

---

## 🔗 关联项目

| 项目 | 关系 |
|------|------|
| [su-memory-sdk](https://github.com/sandysu737-source/su-memory-sdk) | 记忆引擎（上游依赖） |
| [mci-huan](https://github.com/sandysu737-source/mci-huan) | MCI·焕（皮肤医美应用） |
| [mci-kernel-integrations](https://github.com/sandysu737-source/mci-kernel-integrations) | mci-kernel 集成层 |

---

## 📄 许可证

MIT License — 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- 世界模型能力来自 [su-memory-sdk](https://github.com/sandysu737-source/su-memory-sdk) 项目（V3.6.0+ 分离）
- 底层依赖：[NumPy](https://numpy.org/) + [SciPy](https://scipy.org/)（核心 CPU）
- 可选依赖：[PyTorch](https://pytorch.org/) + [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)（JEPA GNN 加速）

---

## 📮 反馈

- 🐛 Bug 报告：[GitHub Issues](https://github.com/sandysu737-source/mci-world-model/issues)
- 💡 功能建议：[GitHub Discussions](https://github.com/sandysu737-source/mci-world-model/discussions)
- 📧 Email: team@mci-world-model.ai
