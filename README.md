# MCI World Model

[![CI](https://github.com/sandysu737-source/mci-world-model/actions/workflows/ci.yml/badge.svg)](https://github.com/sandysu737-source/mci-world-model/actions)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org)
[![Version](https://img.shields.io/badge/version-4.6.0-brightgreen)](https://github.com/sandysu737-source/mci-world-model/releases)
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
| **MCI World Model** | 世界模型（因果推理 / 干预 / 反事实） | v4.6.0 |

> 📌 本项目从 **su-memory-sdk** 分离为独立仓库，当前版本 **v4.6.0**

---

## 🧬 三大支柱

### 1️⃣ Causal Discovery & Inference — 因果引擎

| 类别 | 算法 | 说明 |
|------|------|------|
| **关联** | PC / FCI / GES / LiNGAM | 约束/评分/混合因果发现 |
| **非线性** | CAM / CAMGOLEM / NOTEARS / GOLEM | 可微分 + 加性模型 |
| **干预** | DoCalculus / CachedDoCalculus / Batch | Pearl do-calculus |
| **反事实** | CounterfactualEngine / BatchCF | 三层反事实 |
| **异质性** | T-Learner / S-Learner | CATE/ATE 估计 |
| **时序** | GrangerCausality / LaggedCorrelation | 时序因果推断 |
| **方向** | IGCI + LiNGAM + HSIC 三重投票 | Tübingen 67.3% (SOTA 68%) |

**7+ 种因果发现算法 | 纯 NumPy | 零 GPU 依赖**

```python
from mci_world_model.sdk import CausalGraph, DoCalculus

# 构建因果图
cg = CausalGraph(
    nodes=["Z", "X", "Y"],
    edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
)

# Pearl Backdoor 调整：do(X=1) vs do(X=0)，调整集 = {Z}
dc = DoCalculus(cg)
result = dc.backdoor_adjustment(X="X", Y="Y", Z_set=["Z"])
print(f"ATE = {result.ate:.4f}  (调整集: {result.adjustment_set}, method={result.method})")
```


## 📊 国际基准

| 基准 | 指标 | 结果 |
|------|------|:--:|
| Tübingen 108 | 方向推断精度 | **67.3%** (SOTA: CGNN 73%) |
| BNLearn Asia/Sachs/Child | 结构学习 | PC/NOTEARS/FCI/GES/CAMGOLEM 全通过 |
| CausalBench | 因果发现 | 通过 |
| Cladder | 因果推理 | 通过 |
| MIMIC-III (合成) | 临床因果发现 | PC F1=0.471 |

## 🧪 测试规模

| 指标 | 数值 |
|------|------|
| 单元测试 | **3,830** passed / 0 failed |
| 性能基准 | **17/17** (零外部依赖) |
| SDK 模块 | **170** 个 |
| 因果算法 | **7** 种 |

### 2️⃣ JEPA World Modeling — 世界建模

```python
from mci_world_model import MCIWorldModel
from mci_world_model.sdk import (
    JEPADataset, JEPAEncoder, IdentityPredictor, JEPATrainer,
)

# 1) 时序状态序列 → 训练对
states = [wm.initialize()]  # CausalWorldModelState 列表（按 timestamp 排序）
dataset = JEPADataset.from_states(states, window_size=2)

# 2) 编码器（依赖世界模型）+ 预测器（具体子类，3 选 1）
encoder = JEPAEncoder(MCIWorldModel())
predictor = IdentityPredictor()  # 或 EnergyPropagationPredictor() / BeliefPropagationPredictor()

# 3) 训练器：α 能量损失权重 / β 一致性损失权重
trainer = JEPATrainer(
    encoder, predictor, dataset,
    alpha_energy=0.1, beta_cons=0.05,
)
stats = trainer.train(n_epochs=10, learning_rate=1e-3)
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
    MCIWorldModel,
    CausalGraph, DoCalculus, CounterfactualEngine,
    SIGReg,
)

# 1. 创建世界模型实例
wm = MCIWorldModel()

# 2. 加载因果图
cg = CausalGraph(
    nodes=["天气", "出行", "心情", "效率"],
    edges=[("天气", "心情"), ("天气", "出行"), ("心情", "效率"), ("出行", "效率")],
)

# 3. Pearl 干预推理：ATE(天气 → 效率)
dc = DoCalculus(cg)
result = dc.estimate_ate("天气", "效率")
print(f"ATE(天气 → 效率) = {result.ate:.3f}  (method={result.method})")

# 4. Pearl 反事实推理：do(天气=晴天) 当观察=雨天 时，效率会怎样
cf = CounterfactualEngine.from_causal_graph(cg)
counterfactual = cf.query(
    evidence={"天气": 0.0},       # 事实证据：雨天 (0.0)
    do_x={"天气": 1.0},            # 反事实干预：晴天 (1.0)
    target="效率",
)
print(f"反事实: E[效率 | do(天气=晴天), 观察=雨天] = {counterfactual.counterfactual_value:.3f}")
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
| **v4.6.0** | CEWM 认知增强世界模型 + 代码审查修复 | ✅ 已发布 |
| **v4.4.0** | P3 自主学习补齐 (OnlineEWC + CachedDoCalculus) + ruff 清零 | ✅ 当前版本 |
| v5.0.0 | P6-P8 高级认知 + 多模态统一 + 神经符号融合 | 📋 规划中 |
| v6.0.0 | P9-P11 真实验证 + 跨域融通 + 因果意识 | 📋 规划中 |

详见 [docs/ROADMAP_V3.0.0.md](docs/ROADMAP_V3.0.0.md) 和 [docs/improvement-plans/00_master_index.md](docs/improvement-plans/00_master_index.md)

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
