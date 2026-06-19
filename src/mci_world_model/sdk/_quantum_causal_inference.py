"""MCI World Model v12.0.0 — QuantumCausalInference 量子因果推理
================================================================

基于量子计算的因果推理 — 从经典因果推理扩展到量子因果推理。

核心能力:
    quantum_causal_effect(cause, effect, data)   — 量子因果效应估计
    quantum_counterfactual(factual, intervention)— 量子反事实推理
    quantum_causal_discovery(data)               — 量子因果发现

设计原则:
    - 纯 numpy，零外部依赖
    - 通过 QuantumClassicalBridge 接入量子计算
    - 经典降级: 量子不可用时自动降级为经典方法
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from mci_world_model.sdk._quantum_classical_bridge import (
    QuantumCircuit,
    QuantumClassicalBridge,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CausalEffectResult — 因果效应结果
# =============================================================================


@dataclass
class CausalEffectResult:
    """因果效应估计结果。

    Attributes:
        cause: 原因变量
        effect: 结果变量
        ate: 平均因果效应 (Average Treatment Effect)
        confidence: 置信度
        method: 计算方法
    """

    cause: str
    effect: str
    ate: float = 0.0
    confidence: float = 0.0
    method: str = "quantum"


# =============================================================================
# QuantumCausalInference — 量子因果推理
# =============================================================================


class QuantumCausalInference:
    """量子因果推理 — 利用量子计算加速和增强因果推理。

    核心思路:
      - 因果效应估计 → 量子振幅编码 → 量子干涉 → 因果效应
      - 反事实推理 → 量子态回溯 → 干预 → 前向传播
      - 因果发现 → 量子相关性检测 → 量子独立性检验

    Args:
        bridge: QuantumClassicalBridge 实例
        classical_fallback: 经典降级方法
    """

    def __init__(
        self,
        bridge: QuantumClassicalBridge | None = None,
        classical_fallback: Any | None = None,
    ):
        self._bridge = bridge or QuantumClassicalBridge()
        self._classical = classical_fallback
        self._effect_cache: dict[str, CausalEffectResult] = {}

    # ── Causal Effect Estimation ────────────────────────────────────────

    def quantum_causal_effect(
        self,
        cause: str,
        effect: str,
        data: np.ndarray,
        n_shots: int = 4096,
    ) -> CausalEffectResult:
        """量子因果效应估计。

        利用量子干涉检测因果效应:
          1. 将数据编码为量子态
          2. 构建因果效应检测电路
          3. 执行量子电路
          4. 解码因果效应

        Args:
            cause: 原因变量名
            effect: 结果变量名
            data: 观测数据 (n_samples, n_vars)
            n_shots: 量子测量次数

        Returns:
            因果效应估计结果
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Step 1: 经典预处理 — 计算统计量
        if data.shape[1] >= 2:
            correlation = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0

        # Step 2: 量子因果效应电路
        n_qubits = min(int(np.ceil(np.log2(max(data.shape[0], 2)))), 10)
        circuit = self._build_causal_effect_circuit(
            n_qubits, abs(correlation)
        )

        # Step 3: 执行
        result = self._bridge.execute_on_quantum_hardware(circuit, n_shots)

        # Step 4: 解码因果效应
        decoded = self._bridge.decode_quantum_to_classical(result)

        # 因果效应 ≈ 量子干涉结果 + 经典相关性
        quantum_signal = decoded.get("dominant_probability", 0.5)
        ate = (quantum_signal - 0.5) * 2 * np.sign(correlation)
        confidence = min(abs(ate) * 2, 1.0)

        effect_result = CausalEffectResult(
            cause=cause,
            effect=effect,
            ate=float(ate),
            confidence=float(confidence),
            method="quantum",
        )

        cache_key = f"{cause}->{effect}"
        self._effect_cache[cache_key] = effect_result
        return effect_result

    # ── Counterfactual Reasoning ────────────────────────────────────────

    def quantum_counterfactual(
        self,
        factual_data: dict,
        intervention: dict,
        n_shots: int = 8192,
    ) -> dict:
        """量子反事实推理。

        量子态回溯 → 干预 → 前向传播 → 反事实结果。

        Args:
            factual_data: 事实数据
            intervention: 干预 {variable: value}
            n_shots: 量子测量次数

        Returns:
            反事实推理结果
        """
        # Step 1: 量子编码事实数据
        factual_array = np.array(
            list(factual_data.values()), dtype=float
        ).reshape(1, -1)
        factual_circuit = self._bridge.encode_classical_to_quantum(
            factual_array, method="angle"
        )

        # Step 2: 施加干预 (修改电路参数)
        intervened_circuit = self._apply_quantum_intervention(
            factual_circuit, intervention
        )

        # Step 3: 前向传播
        result = self._bridge.execute_on_quantum_hardware(
            intervened_circuit, n_shots
        )

        # Step 4: 解码反事实结果
        counterfactual = self._bridge.decode_quantum_to_classical(result)

        return {
            "factual": factual_data,
            "intervention": intervention,
            "counterfactual": counterfactual,
            "method": "quantum_counterfactual",
        }

    # ── Causal Discovery ────────────────────────────────────────────────

    def quantum_causal_discovery(
        self, data: np.ndarray, var_names: list[str] | None = None
    ) -> dict:
        """量子因果发现 — 利用量子独立性检验发现因果结构。

        Args:
            data: 观测数据 (n_samples, n_vars)
            var_names: 变量名列表

        Returns:
            发现的因果结构
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_vars = data.shape[1]
        if var_names is None:
            var_names = [f"V{i}" for i in range(n_vars)]

        # 量子独立性检验 (简化版: 量子相关性检测)
        edges = []
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                # 量子因果效应估计
                pair_data = np.column_stack([data[:, i], data[:, j]])
                effect = self.quantum_causal_effect(
                    var_names[i], var_names[j], pair_data, n_shots=2048
                )
                if abs(effect.ate) > 0.1:
                    edges.append(
                        {
                            "from": var_names[i],
                            "to": var_names[j],
                            "ate": effect.ate,
                            "confidence": effect.confidence,
                        }
                    )

        return {
            "nodes": var_names,
            "edges": edges,
            "n_nodes": n_vars,
            "n_edges": len(edges),
            "method": "quantum_discovery",
        }

    # ── Internal Methods ────────────────────────────────────────────────

    def _build_causal_effect_circuit(
        self, n_qubits: int, signal_strength: float
    ) -> QuantumCircuit:
        """构建因果效应检测电路。"""
        circuit = QuantumCircuit(n_qubits=n_qubits, n_shots=4096)

        # Hadamard 层: 创建叠加态
        for i in range(n_qubits):
            circuit.add_gate("H", [i])

        # 因果关联层: 受信号强度控制的纠缠
        angle = np.pi * signal_strength
        for i in range(n_qubits - 1):
            circuit.add_gate("RY", [i + 1], [float(angle)])
            circuit.add_gate("H", [i])

        return circuit

    def _apply_quantum_intervention(
        self, circuit: QuantumCircuit, intervention: dict
    ) -> QuantumCircuit:
        """在量子电路中施加干预。"""
        intervened = QuantumCircuit(
            n_qubits=circuit.n_qubits, n_shots=circuit.n_shots
        )
        intervened.gates = list(circuit.gates)

        # 干预: 重置某些量子比特并施加新的旋转
        for var, value in intervention.items():
            target_qubit = hash(var) % circuit.n_qubits
            angle = np.pi * float(value)
            intervened.add_gate("RY", [target_qubit], [float(angle)])

        return intervened
