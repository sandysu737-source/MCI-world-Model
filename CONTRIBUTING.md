# 贡献指南

欢迎参与 **MCI World Model** 项目的建设！本项目是「世界模型引擎」独立仓库，聚焦 Pearl 因果推理 + JEPA 世界建模 + 能量中心三才合一。

## 🧭 仓库定位

| 仓库 | 版本 | 角色 |
|------|------|------|
| [su-memory-sdk](https://github.com/sandysu737-source/su-memory-sdk) | V3.5.1 | 记忆引擎（独立仓库） |
| **mci-world-model**（本仓库） | V3.0.0 | 因果世界模型引擎 |
| [mci-huan](https://github.com/sandysu737-source/mci-huan) | - | 上层产品 |
| [mci-kernel-integrations](https://github.com/sandysu737-source/mci-kernel-integrations) | - | 内核集成 |

## 🛠️ 开发环境

```bash
git clone git@github.com:sandysu737-source/mci-world-model.git
cd mci-world-model
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,full]"
```

可选安装组：

- `.[jepa]` — JEPA 训练（PyTorch + PyG）
- `.[bench]` — SIGReg 基准（matplotlib + tqdm）
- `.[full]` — 全量依赖（含 networkx）
- `.[dev]` — 开发者工具（pytest + ruff + mypy + black）

## 🧪 测试与质量门禁（CI 4-Gate）

```bash
# Gate 1 — Lint
ruff check src/ tests/ benchmarks/

# Gate 2 — Type Check
mypy src/mci_world_model

# Gate 3 — Unit Test
pytest -m "not slow" --cov=mci_world_model

# Gate 4 — Bench Smoke
python -c "from mci_world_model.benchmarks._noise_generator import generate_spectral_noise; print('OK')"
```

## 📐 编码规范

- **命名**：`PascalCase` 类、`snake_case` 函数与变量、`UPPER_SNAKE_CASE` 常量
- **包名**：`mci_world_model.*`（**不再使用** `su_memory.*`，发布门禁已硬性约束）
- **品牌名**：对外统一「**MCI World Model**」（**不带 SDK 后缀**）
- **类型注解**：公共 API 必须有 `->` 返回类型，`mypy strict` 通过
- **测试覆盖**：新增模块需配套单元测试，覆盖率 ≥ 80%
- **导入顺序**：标准库 → 第三方 → 本地（`from mci_world_model.*`），用 `ruff --select I` 自动修复

## 🌿 分支与提交

- **主分支**：`main`（受保护，仅通过 PR 合并）
- **功能分支**：
  - `feat/<scope>-<short-desc>`（新增能力）
  - `fix/<scope>-<short-desc>`（缺陷修复）
  - `docs/<short-desc>`（文档/注释）
  - `refactor/<scope>`（重构，不变更行为）
- **提交信息格式**（Conventional Commits + Qoder 签名）：
  ```
  fix: <简洁描述（祈使句，不超过 50 字）>

  - 修复点 1
  - 修复点 2
  - 验证点 1

  🤖 Generated with [Qoder](https://qoder.com)
  Co-Authored-By: Qoder <noreply@qoder.com>
  ```

## 🧩 模块边界

新增能力请明确归属：

| 子包 | 角色 | 典型符号 |
|------|------|---------|
| `mci_world_model._sys` | 系统层（能量中心） | YinYang / ThreePowers / EnergyCore / CausalEngine |
| `mci_world_model.sdk` | 推理层（Pearl + JEPA） | MCIWorldModel / DoCalculus / CounterfactualEngine / JEPA* |
| `mci_world_model.world_model` | 三才合一协调器 | WorldModel / 状态合并 |
| `mci_world_model.benchmarks` | SIGReg 基准 | _noise_generator / sigreg/* |

**禁止**：
- 在本仓库内引用 `su_memory.*`（必须用本地子包）
- 在新代码中使用「MCI World Model SDK」品牌名（已发布门禁）
- 把 su-memory-sdk 记忆引擎专属能力（如 MEMO 检索）迁入本项目

## 📬 反馈渠道

- **Issue**：[github.com/sandysu737-source/mci-world-model/issues](https://github.com/sandysu737-source/mci-world-model/issues)
- **Discussion**：[github.com/sandysu737-source/mci-world-model/discussions](https://github.com/sandysu737-source/mci-world-model/discussions)
- **Security**：见 [SECURITY.md](SECURITY.md)

---

🤖 Generated with [Qoder](https://qoder.com)
