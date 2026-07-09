"""NumPy 后端 — CPU 实现 (默认)。"""
from __future__ import annotations

import numpy as np

from mci_world_model._backend.interface import BackendInterface


class NumPyBackend(BackendInterface):
    """NumPy CPU 后端。"""

    name = "numpy"
    is_gpu = False

    def array(self, data, dtype=None):
        return np.array(data, dtype=dtype or np.float64)

    def matmul(self, a, b):
        return a @ b

    def dot(self, a, b):
        return np.dot(a, b)

    def norm(self, x, axis=None):
        return np.linalg.norm(x, axis=axis)

    def inv(self, a):
        return np.linalg.inv(a)

    def solve(self, a, b):
        return np.linalg.solve(a, b)

    def lstsq(self, a, b):
        result, *_ = np.linalg.lstsq(a, b, rcond=None)
        return result

    def eigh(self, a):
        return np.linalg.eigh(a)

    def eigvalsh(self, a):
        return np.linalg.eigvalsh(a)

    def zeros(self, shape):
        return np.zeros(shape, dtype=np.float64)

    def eye(self, n):
        return np.eye(n, dtype=np.float64)

    def to_numpy(self, x):
        return np.asarray(x)
