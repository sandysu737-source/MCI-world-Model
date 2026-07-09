"""后端抽象接口 — 7 种核心线性代数操作。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BackendInterface(ABC):
    """计算后端抽象接口。

    覆盖项目中使用的 7 种核心操作:
    array, matmul, dot, norm, inv, solve, lstsq, eigh, eigvalsh
    """

    name: str = "abstract"
    is_gpu: bool = False

    @abstractmethod
    def array(self, data: Any, dtype: Any = None) -> Any:
        """创建数组。"""

    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any:
        """矩阵乘法。"""

    @abstractmethod
    def dot(self, a: Any, b: Any) -> Any:
        """向量/矩阵点积。"""

    @abstractmethod
    def norm(self, x: Any, axis: int | None = None) -> Any:
        """范数。"""

    @abstractmethod
    def inv(self, a: Any) -> Any:
        """矩阵求逆。"""

    @abstractmethod
    def solve(self, a: Any, b: Any) -> Any:
        """解线性方程组 Ax=b。"""

    @abstractmethod
    def lstsq(self, a: Any, b: Any) -> Any:
        """最小二乘解。"""

    @abstractmethod
    def eigh(self, a: Any) -> tuple[Any, Any]:
        """对称矩阵特征分解, 返回 (eigenvalues, eigenvectors)。"""

    @abstractmethod
    def eigvalsh(self, a: Any) -> Any:
        """对称矩阵特征值。"""

    @abstractmethod
    def zeros(self, shape: tuple[int, ...]) -> Any:
        """零矩阵。"""

    @abstractmethod
    def eye(self, n: int) -> Any:
        """单位矩阵。"""

    @abstractmethod
    def to_numpy(self, x: Any) -> Any:
        """转换为本后端的 NumPy 表示 (用于 I/O)。"""
