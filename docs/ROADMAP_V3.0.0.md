# MCI World Model V3.0.0 路线图

> 锚定 V3.0.0 独立发布窗口（2026 年 6 月），并向前规划至 V4.0.0 大一统世界模型。

## 🎯 V3.0.0 — 独立成仓（✅ 已发布 · 2026-06-03）

- [x] 从 su-memory-sdk 分离 World Model 核心模块（32 个 .py 文件 / ~9000 行）
- [x] SDK 去名化：`su_memory.*` → `mci_world_model.*`
- [x] 品牌去后缀：「MCI World Model SDK」→「MCI World Model」
- [x] 清理 V3.5.0 / V3.4.0 过时论文与规划（不纳入本项目）
- [x] pyproject.toml / README / CHANGELOG / LICENSE / CONTRIBUTING 完备
- [x] CI 4-Gate：ruff / mypy / pytest / sigreg-bench
- [x] 顶层 `mci_world_model` 统一入口（_sys / sdk / world_model）

## 🚀 V3.1.0 — Pearl 三层强化（计划：2026 Q3）

- [ ] P1 频谱因果：`FourierCausal` 在线更新接口
- [ ] P2 Do-Calculus：批量干预 API + 缓存层
- [ ] P3 反事实引擎：与 `EnergyCore` 双向耦合
- [ ] Pearl 三层一致性断言（do-calculus + counterfactual 一致性证明）
- [ ] 基准数据集（半合成 + 公开）接入

## 🧠 V3.2.0 — JEPA 工业级训练（计划：2026 Q4）

- [ ] `JEPATrainer` 多机分布式支持（PyTorch DDP / FSDP）
- [ ] `JEPADataset` 标准化接口（与 su-memory-sdk 记忆引擎 V3.5.1 对接）
- [ ] SIGReg 嵌入正则化收敛性证明
- [ ] Reflection QA 合成器与记忆引擎双向流
- [ ] JEPA 在 OGB / Planetoid 公开数据集上的基线报告

## 🔮 V3.3.0 — 能量中心正式化（计划：2027 Q1）

- [ ] YinYang / ThreePowers 数学公理化（公理 → 定理 → 推论 三段式）
- [ ] `EnergyConsistencyLoss` 形式化梯度推导
- [ ] 64 卦 → 因果结构归纳证明
- [ ] 能量中心可视化面板（与 SIGReg 基准统一）

## 🏆 V4.0.0 — 大一统世界模型（计划：2027 Q2）

- [ ] 「CPU + GPU」分层接口稳定（`CPU = MCIWorldModel` 因果引擎；`GPU = Transformer` 可选加速器）
- [ ] Transformer 仅作为可选加速器（接口层而非实现层，可随时更换或移除）
- [ ] 与 mci-huan / mci-kernel-integrations 全链路打通
- [ ] 第一篇正式发表论文（Causal World Model with Energy Center · JMLR / UAI 投稿）

## 📊 关键指标

| 版本 | 代码行数 | 文档完整度 | 测试覆盖率 | 论文状态 |
|------|---------|-----------|-----------|---------|
| **V3.0.0** | ~9,000 | 80% | ≥ 70% | — |
| V3.1.0 | ~12,000 | 85% | ≥ 80% | arXiv 预印本 |
| V3.2.0 | ~15,000 | 90% | ≥ 85% | arXiv 完整版 |
| V3.3.0 | ~18,000 | 95% | ≥ 88% | 期刊投稿中 |
| V4.0.0 | ~25,000 | 100% | ≥ 90% | JMLR / UAI |

## 🧭 依赖与版本映射

```
su-memory-sdk V3.5.1（记忆引擎，独立仓库）
        ↓
MCI World Model V3.0.0（独立成仓）──→ V3.1.0 ──→ V3.2.0 ──→ V3.3.0 ──→ V4.0.0
        │                                │            │            │            │
        └──── 上游：记忆引擎              │            │            │            │
        └──── 下游：mci-huan              └─ 下游：mci-kernel-integrations ────┘
```

## 🛡️ 兼容性承诺

- **V3.0.x** → **V3.1.x** → **V3.2.x**：次版本升级保持向后兼容（仅新增、不破坏 API）
- **V3.x** → **V4.0**：主版本允许破坏性变更，但会提供 `mci_world_model.v3_compat` 兼容层（至少维护 6 个月）

## 🤝 协作原则

- **Harness Engineering 4-Gate**：每条 PR 必须通过 ruff / mypy / pytest / sigreg-bench
- **Do-calculus 优先**：新增能力必须先在 Pearl 三层（P1/P2/P3）找到落点，不引入「黑盒」捷径
- **能量一致性**：所有数值模块必须经过 `EnergyConsistencyLoss` 校验
- **公开透明**：所有设计决策在 Discussions 公开讨论，不在私下 review 中私自裁决

---

🤖 Generated with [Qoder](https://qoder.com)
