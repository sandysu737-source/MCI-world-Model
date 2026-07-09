"""PyTorch 后端 — GPU 加速 (可选, 需安装 torch)。

当 torch 可用且有 CUDA GPU 时自动启用。
提供与 NumPyBackend 相同的接口, 底层用 torch.tensor + GPU。
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import torch
    _TORCH_AVAILABLE = True
    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _TORCH_AVAILABLE = False
    _CUDA_AVAILABLE = False

from mci_world_model._backend.interface import BackendInterface


class TorchBackend(BackendInterface):
    """PyTorch GPU 后端。"""

    name = "torch"
    is_gpu = True

    def __init__(self) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch 未安装, 无法使用 GPU 后端")
        self._device = torch.device("cuda" if _CUDA_AVAILABLE else "cpu")
        logger.info("TorchBackend 初始化: device=%s", self._device)

    @property
    def device(self):
        return self._device

    def array(self, data, dtype=None):
        dtype_map = {None: torch.float64, "float64": torch.float64, "float32": torch.float32}
        t = torch.as_tensor(data, dtype=dtype_map.get(dtype, torch.float64))
        return t.to(self._device)

    def matmul(self, a, b):
        return torch.matmul(a, b)

    def dot(self, a, b):
        return torch.dot(a.flatten(), b.flatten())

    def norm(self, x, axis=None):
        return torch.norm(x, dim=axis)

    def inv(self, a):
        return torch.linalg.inv(a)

    def solve(self, a, b):
        return torch.linalg.solve(a, b)

    def lstsq(self, a, b):
        result = torch.linalg.lstsq(a, b)
        return result.solution

    def eigh(self, a):
        return torch.linalg.eigh(a)

    def eigvalsh(self, a):
        return torch.linalg.eigvalsh(a)

    def zeros(self, shape):
        return torch.zeros(shape, dtype=torch.float64, device=self._device)

    def eye(self, n):
        return torch.eye(n, dtype=torch.float64, device=self._device)

    def to_numpy(self, x):
        return x.detach().cpu().numpy()
