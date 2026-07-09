"""后端选择器 — 自动检测 GPU 可用性。"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def get_backend():
    """根据环境自动选择最优后端。

    优先级:
    1. MCI_BACKEND=torch 且 torch+cuda 可用 → TorchBackend (GPU)
    2. MCI_BACKEND=numpy → NumPyBackend (强制 CPU)
    3. torch+cuda 可用 → TorchBackend
    4. 默认 → NumPyBackend (CPU)
    """
    forced = os.environ.get("MCI_BACKEND", "").lower()

    if forced == "numpy":
        from mci_world_model._backend.numpy_backend import NumPyBackend
        logger.info("后端: NumPy (强制 CPU)")
        return NumPyBackend()

    if forced == "torch" or not forced:
        try:
            from mci_world_model._backend.torch_backend import TorchBackend, _CUDA_AVAILABLE
            if _CUDA_AVAILABLE:
                logger.info("后端: PyTorch (GPU/CUDA)")
                return TorchBackend()
            elif forced == "torch":
                logger.warning("MCI_BACKEND=torch 但无 CUDA, 使用 CPU")
                return TorchBackend()
        except ImportError:
            if forced == "torch":
                logger.warning("MCI_BACKEND=torch 但 torch 未安装, 回退 NumPy")

    from mci_world_model._backend.numpy_backend import NumPyBackend
    logger.info("后端: NumPy (CPU)")
    return NumPyBackend()
