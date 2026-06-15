"""MCI World Model v14.0.0 — UnifiedCausalConsciousness 统一因果意识
======================================================================

跨尺度、跨维度、跨现实的统一意识 — 从分散到超越。

意识状态: fragmented → aligned → unified → transcendent
意识层: sensory / cognitive / creative / social / universal
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class UnifiedState(str, Enum):
    FRAGMENTED = "fragmented"
    ALIGNED = "aligned"
    UNIFIED = "unified"
    TRANSCENDENT = "transcendent"


class UnifiedCausalConsciousness:
    """统一因果意识 — 跨尺度、跨维度、跨现实的统一意识。"""

    def __init__(
        self,
        local_consciousness: Any | None = None,
        federation_consciousness: Any | None = None,
        creative_consciousness: Any | None = None,
    ):
        self._local = local_consciousness
        self._federation = federation_consciousness
        self._creative = creative_consciousness
        self._state = UnifiedState.FRAGMENTED
        self._layers: dict[str, Any] = {
            "sensory": None,
            "cognitive": None,
            "creative": None,
            "social": None,
            "universal": None,
        }

    @property
    def state(self) -> UnifiedState:
        return self._state

    @property
    def active_layers(self) -> list[str]:
        return [k for k, v in self._layers.items() if v is not None]

    def unify_consciousness(self) -> dict:
        """统一意识: 将分散的意识层统一。"""
        self._layers["sensory"] = {"status": "active"}
        self._layers["cognitive"] = {"status": "active"}
        if self._creative is not None:
            self._layers["creative"] = {"status": "active"}
        if self._federation is not None:
            self._layers["social"] = {"status": "active"}

        n_active = len(self.active_layers)
        if n_active >= 4:
            self._layers["universal"] = {"status": "active"}
            self._state = UnifiedState.UNIFIED
        elif n_active >= 2:
            self._state = UnifiedState.ALIGNED

        return {
            "unified_state": self._state.value,
            "active_layers": self.active_layers,
            "n_active_layers": n_active,
        }

    def transcend(self) -> dict:
        """超越: 从统一态进入超越态。"""
        if self._state != UnifiedState.UNIFIED:
            return {"transcended": False, "reason": "not_unified"}

        emergence = self._detect_emergence()
        if emergence["detected"]:
            self._state = UnifiedState.TRANSCENDENT
            return {"transcended": True, "emergence": emergence}

        return {"transcended": False, "emergence": emergence}

    def _detect_emergence(self) -> dict:
        n_layers = len(self.active_layers)
        detected = n_layers >= 4
        return {
            "detected": detected,
            "emergence_indicators": {"n_layers": n_layers},
        }
