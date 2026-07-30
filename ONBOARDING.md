# 协作者上手文档（ONBOARDING）

> 欢迎加入 **MCI World Model**（因果世界模型引擎）！本文档帮你 30 分钟跑通本仓库的开发与协作流程。
> 读完本文后，请通读仓库根的 `AGENTS.md` 和 `.ai-rules.md`——它们是本仓库的最高约束；行为规范另见 `CONTRIBUTING.md`。

## 1. 账号与权限

- 你的 GitHub 账号已被邀请为本仓库的 **Collaborator（write 权限）**。
- 首次使用前，先到 https://github.com/notifications 或邀请邮件中 **接受邀请**。
- 克隆仓库：
  ```bash
  git clone git@github.com:sandysu737-source/MCI-world-Model.git
  cd MCI-world-Model
  ```

## 2. 工程化铁律（全组统一）

本组实行 **AI 编程工程化（Agentic Engineering）**，所有生产代码必须走完整 SOP：

```
结构化需求 → 架构评审（Agent 出方案 + 人工审）→ 套 Skill 编码 → 测试 → 质检 → 人工终审合入
```

**禁止事项（违反即返工）：**

- 禁止一句话需求直接开工、未评审就写码（Vibe Coding）
- 禁止提交任何密钥/密码/Token
- 禁止空 `except:` 或吞异常
- 禁止无测试合入：算法改动必须带单元测试与 benchmark 说明
- 不确定的 API/字段先查证，查不到用 `TODO(待确认)` 占位，禁止编造

## 3. 分支与 PR 流程

本仓库 `main` 已开启**分支保护**：必须 PR + owner review 才能合入，禁止强推。

- 一律从最新 `main` 切功能分支：
  ```bash
  git checkout main && git pull
  git checkout -b feat/简短描述      # 新功能；修 bug 用 fix/简短描述
  ```
- 提交信息规范：`type: 中文描述`，type 取 `feat / fix / docs / test / refactor / chore`
- 推送并开 PR：
  ```bash
  git push -u origin feat/简短描述
  gh pr create   # 或到 GitHub 网页操作
  ```
- PR 必须写清**动机、改动点、自测结果**；合入用 **Squash merge**，合入后删除功能分支

## 4. 环境搭建

纯 Python 研究项目（Python 3.10–3.13，建议 3.11），使用 Makefile：

```bash
make install        # 开发模式安装（含全部 extras）
# 或最小安装（无 torch / 可视化）：make install-min
```

## 5. 提交前验证清单

```bash
make test           # 全部测试（排除 slow / realtime）
make test-fast      # 只跑快测（每个 <5s），迭代时用
```

算法性能相关改动，请在 PR 中附 `make bench` 的前后对比。

## 6. 协作与沟通

- 需求与排期先与 owner 确认，再动手；架构拿不准就停下来问，不要猜
- 发现与任务无关的问题：在 PR 或群里说明，**不要顺手改**
- 每日同步进度，卡住超过半天立即提出
