"""MCI World Model — BNLearn 标准因果发现基准

国际标准 DAG 数据集:
  - Asia (8 nodes, 8 edges) — 肺癌诊断
  - Sachs (11 nodes, 17 edges) — 蛋白质信号网络
  - Child (20 nodes, 25 edges) — 先天性疾病诊断
  - Alarm (37 nodes, 46 edges) — 重症监护报警
  - Insurance (27 nodes, 52 edges) — 保险风险评估

指标: SHD (Structural Hamming Distance), F1-score, Precision, Recall
验收: SHD ≤ ground_truth_edges * 0.5 (50% 容差)
"""
