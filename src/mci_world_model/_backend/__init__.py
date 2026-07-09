"""计算后端抽象层 — 自动选择 NumPy (CPU) 或 PyTorch (GPU)。

核心思想: 代码中统一用 backend.tensor/backend.matmul/backend.solve 等,
运行时根据 GPU 可用性自动切换实现, 源文件零改动。

用法:
    from mci_world_model._backend import B
    x = B.array([1.0, 2.0])
    y = B.matmul(A, x)
"""
from mci_world_model._backend.selector import get_backend

B = get_backend()

__all__ = ["B", "get_backend"]
