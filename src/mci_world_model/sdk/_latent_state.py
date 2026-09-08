from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LatentState:
    """JEPA 潜空间状态的稳定输出契约。"""

    latent: np.ndarray
    source: str
    encoder_id: str
    is_trained: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    graph_state: Any | None = None

    def __post_init__(self) -> None:
        self.latent = np.asarray(self.latent, dtype=np.float64)
        if not np.all(np.isfinite(self.latent)):
            raise ValueError("潜向量包含非有限值")
        if self.source not in {"not_ready", "true_jepa", "learnable", "legacy_graph"}:
            raise ValueError(f"未知潜状态来源: {self.source}")
        if self.source == "true_jepa" and not self.is_trained:
            self.status = "not_ready"

    @property
    def is_ready(self) -> bool:
        return self.status == "ok" and self.latent.size > 0

    def to_array(self) -> np.ndarray:
        return self.latent.copy()
