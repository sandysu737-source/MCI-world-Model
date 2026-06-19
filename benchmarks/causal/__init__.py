"""Causal Benchmark Suite — MCI World Model 因果推理能力外部评测。

标准数据集:
    - LinearGaussian: 线性高斯 SCM
    - NonlinearSCM: 非线性结构因果模型
    - IHDP-style: 半合成异质处理效应
    - Backdoor: 后门调整图

评测维度:
    - ATE 估计误差 (|ATÊ - ATE|)
    - 反事实预测 MSE
    - 因果发现 F1
    - 调整集正确率
"""
