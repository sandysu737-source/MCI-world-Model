"""因果推理标准 Benchmark 适配器包 — TASK-B4。

提供两个学术社区标准数据集的适配器:
    - CausalBench (CLeAR 因果推理标准集)
    - Tübingen Cause-Effect Pairs

验收标准:
    - CausalBench: 因果方向判断准确率 ≥ 0.70
    - Tübingen pairs: 因果方向判断准确率 ≥ 0.65
    - 对比表含 ≥ 3 个已有方法的引用分数
"""
