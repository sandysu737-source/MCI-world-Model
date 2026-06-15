"""MCI World Model — Cladder 因果推理基准测试 (NeurIPS 2023).

CLadder: Assessing Causal Reasoning in Language Models
https://arxiv.org/abs/2312.04350

10K 道 yes/no 题覆盖 Pearl 三层因果阶梯 (Rung 1→2→3):
  Rung 1 (Association):       correlation, marginal, exp_away
  Rung 2 (Intervention):      ate, backadj, collider_bias
  Rung 3 (Counterfactual):    ett, nde, nie, det-counterfactual

CEWM 结构化求解: 用 CausalGraph + DoCalculus + CounterfactualEngine
代替 LLM 的统计拟合，预期 ≥95% 准确率。
"""
