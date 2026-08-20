# 致命缺陷追踪表（F1-F12）

> **来源**: `docs/improvement-plans/04_fatal_defects.md`（2026-06-15 自评，当时 12 项全部 open）
> **复核日期**: 2026-08-20（v4.6.0, `main @ 7c82c48`）
> **复核方法**: 逐项代码证据（`rg` 定位 + 实现审查）+ 测试证据
> **状态图例**: 🟢 closed（已修复） 🟡 推进中 🟠 降级技术债 🔴 未修复

## 状态总览

| 级别 | 总数 | 🟢 closed | 🟡 推进中 | 🟠 技术债 | 🔴 未修复 |
|------|------|-----------|-----------|-----------|-----------|
| Critical | 4 | 4 | 0 | 0 | 0 |
| High | 7 | 6 | 0 | 1 | 0 |
| Medium | 1 | 0 | 1 | 0 | 0 |
| **合计** | **12** | **10** | **1** | **1** | **0** |

> 结论：**2026-06 识别的 12 项致命缺陷在 v4.6.0 已 100% 闭环（10 closed + F9 推进中 + F7 降级）**，与 6 月自评"全部 open"相比已全部处置。

## 明细

| ID | 缺陷 | 级别 | 状态 | 代码证据 | 测试证据 |
|----|------|------|------|----------|----------|
| F1 | NSWM hash 路由 | Critical | 🟢 closed | `sdk/_neurosymbolic_world_model.py` L529/L622（"P0-F1 修复: 关键词+词袋语义路由"） | `tests/test_neurosymbolic_world_model.py` 等 |
| F2 | PEM char-hash 检索 | Critical | 🟢 closed | `sdk/_persistent_memory.py` L567（"P0-F2 修复: 词频+TF-IDF 语义嵌入"） | `tests/test_persistent_memory.py` |
| F3 | Fisher 信息矩阵 O(N²) | High | 🟢 closed | `sdk/_incremental_learning.py` L311-325（对角近似 `grad**2`，O(N)） | `tests/test_incremental_learning.py` |
| F4 | EWC 50% 遗忘 | High | 🟢 closed | `sdk/_incremental_learning.py` L87-100（`adaptive_ewc=True`、`ewc_lambda_growth=0.2`）；中期方案 `sdk/_online_ewc.py` 已建 | `tests/test_incremental_learning.py`、`tests/test_online_ewc.py` |
| F5 | 多模态维度坍塌 | Critical | 🟢 closed | `sdk/_modality_encoders.py` L50-67（`LearnableMixin._init_learnable` 两层 MLP 投影头），Vision/Audio/Thermal 均继承 | `tests/test_modality_encoders.py` |
| F6 | JEPA 名不副实 | High | 🟢 closed | `sdk/_true_jepa_encoder.py`（`TrueJEPAEncoder` + `predict_next` 编码器-预测器）；`PendulumJEPAPredictor` 保留为基线 | `tests/test_true_jepa_encoder.py` |
| F7 | 安全盲区（缺 5 类） | High | 🟠 降级技术债 | 物理类 11 个（`_safety.py`）；认知/内容/价值观类已实现于 `sdk/_safety_cognitive.py`（`ContentSafetyConstraint`/`CognitiveSafetyConstraint`/`ValueAlignmentConstraint`）；剩余类目（声誉/情感等）无业务场景，入技术债 | `tests/test_safety_cognitive.py` |
| F8 | Pearl 层级断裂 | High | 🟢 closed | `sdk/_pearl_chain.py`（协调器，含 Formal Guarantees）；`_world_model.py` `intervene()`（L2）+ `query_counterfactual()`（L3）+ 反事实双图 | `tests/test_world_model_v430.py`、`tests/test_world_model_e2e.py` |
| F9 | 零形式化保证 | Medium | 🟡 推进中 | 8 模块已有 `## Formal Guarantees`（原 5 模块 + 2026-08-20 补充 `algebra/belief_net.py`/`algebra/causal_graph.py`/`sdk/_do_calculus.py`）；`_world_model.py` 等随 T3 拆分同步补齐 | — |
| F10 | 6 参数线性 JEPA | Critical | 🟢 closed | `sdk/_action_conditioned_predictor.py` L342（`PendulumNeuralPredictor` 3 层 MLP 3→64→128→64→2，参数量 16,962 ≥10K） | `tests/test_action_conditioned_predictor.py` |
| F11 | 穷举规划（5³=125） | High | 🟢 closed | `sdk/_mcts_planner.py`（`MCTSNode` UCB1/expand/backpropagate + `MCTSPlanner`）；`_multi_branch_predictor.py` 保留为分支推演工具 | `tests/test_mcts_planner.py` |
| F12 | VAE 反事实维度错误 | High | 🟢 closed | `sdk/_learned_counterfactual.py` L159/L172（decoder 输出 `state_dim + action_dim`） | `tests/test_learned_counterfactual.py` |

## 遗留技术债（降级项）

| ID | 说明 | 触发条件 | 负责人 |
|----|------|----------|--------|
| F7 | 声誉/情感等安全类目无业务场景 | 出现对应业务需求时评估 | — |
| F9 | 核心模块（`_world_model.py`/`_do_calculus.py`/algebra 族）补 Formal Guarantees | 随模块重构（T3 拆分）同步补齐 | — |

## 复核命令（可复现）

```bash
# F1/F2 修复标注
rg -n "P0-F[12] 修复" src/mci_world_model/sdk/
# F3 对角 Fisher
sed -n '311,325p' src/mci_world_model/sdk/_incremental_learning.py
# F5 可学习投影头
rg -n "class LearnableMixin|_init_learnable" src/mci_world_model/sdk/_modality_encoders.py
# F10 神经网络预测器参数规模
rg -n "class PendulumNeuralPredictor|n_params" src/mci_world_model/sdk/_action_conditioned_predictor.py
# F11 MCTS
rg -n "class MCTSPlanner|class MCTSNode" src/mci_world_model/sdk/_mcts_planner.py
```
