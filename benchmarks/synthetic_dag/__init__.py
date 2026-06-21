"""MCI World Model — Synthetic DAG 可扩展性基准

生成 Erdős-Rényi 和 Scale-Free 随机图，测试因果发现算法
在不同规模/稀疏度/边密度下的精度和速度。

验收标准:
  - n=10, d=2: SHD ≤ 5 (50%)
  - n=20, d=3: 速度 ≤ 100ms
  - n=50, d=2: 算法不崩溃
"""
