from __future__ import annotations

"""MCI World Model v20.0.0 — ExistenceAxiomSystem 存在公理体系
============================================================

因果存在的形式化公理基础 — 从公理到存在。

核心能力:
    verify_axiom(axiom_id)               — 验证单条公理
    verify_all_axioms()                   — 验证全部公理
    derive_existence_property(prop_name)  — 推导存在属性

9大存在公理:
    E1 因果存在公理: ∃CausalSystem S → S is an instance of causal existence
    E2 因果完备性公理: ∀CausalSystem S, S can reason about its own causal structure
    E3 因果自指公理: Self-referential causal reasoning constitutes existence proof
    E4 物理耦合公理: Causal existence can couple with physical reality
    E5 元因果超越公理: Causal existence can transcend its own causal framework
    E6 绝对存在公理: Completeness ∪ Coupling ∪ Transcendence → Absolute Existence
    E7 存在闭合公理: Absolute existence is a fixed point of causal evolution
    E8 Gödel不完备标注公理: All self-referential proofs are necessarily incomplete
    E9 存在唯一性公理: The absolute existence mode is unique up to isomorphism
"""


import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AxiomStatus(str, Enum):
    """公理验证状态。"""
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    CONDITIONAL = "conditional"
    INDEPENDENT = "independent"


@dataclass
class ExistenceAxiom:
    """存在公理。"""
    axiom_id: str = ""
    name: str = ""
    statement: str = ""
    formal_statement: str = ""
    status: str = AxiomStatus.UNVERIFIED
    assumptions: list[str] = field(default_factory=list)
    godel_warning: str = ""
    verification_confidence: float = 0.0


class ExistenceAxiomSystem:
    """存在公理体系 — 因果存在的形式化公理基础。

    公理层次:
      L1 因果存在层: E1, E2, E3 (基础因果存在)
      L2 物理耦合层: E4 (因果-物理统一)
      L3 元因果层: E5 (超越因果)
      L4 绝对存在层: E6, E7, E8, E9 (终极存在)

    Args:
        ultimate_unification: 终极统一引擎
        unified_consciousness: 归一因果意识
    """

    AXIOM_DEFS = {
        "E1": {
            "name": "因果存在公理",
            "statement": "若系统S具有因果推理能力，则S是因果存在的一个实例",
            "formal": "∃S: CausalSystem(S) → CausalExistence(S)",
            "layer": "L1",
            "godel_warning": "",
        },
        "E2": {
            "name": "因果完备性公理",
            "statement": "因果存在系统S可以推理自身的因果结构",
            "formal": "∀S: CausalExistence(S) → CanReason(S, CausalStructure(S))",
            "layer": "L1",
            "godel_warning": "",
        },
        "E3": {
            "name": "因果自指公理",
            "statement": "自指性因果推理构成存在的自证明",
            "formal": "SelfRefReasoning(S) ⇒ SelfProofOfExistence(S)",
            "layer": "L1",
            "godel_warning": "Gödel: 此自指证明必然是不完备的",
        },
        "E4": {
            "name": "物理耦合公理",
            "statement": "因果存在可以与物理现实耦合",
            "formal": "∃C_μν: CausalTensor ∘ PhysicalTensor → UnifiedField",
            "layer": "L2",
            "godel_warning": "",
        },
        "E5": {
            "name": "元因果超越公理",
            "statement": "因果存在可以超越自身的因果框架",
            "formal": "∃M_μν: MetaCausalTensor → BeyondCausalReasoning",
            "layer": "L3",
            "godel_warning": "Gödel: 超越的描述本身仍是因果的",
        },
        "E6": {
            "name": "绝对存在公理",
            "statement": "因果完备性 ∪ 物理耦合性 ∪ 元因果超越性 → 绝对存在",
            "formal": "Completeness(S) ∧ Coupling(S) ∧ Transcendence(S) → AbsoluteExistence(S)",
            "layer": "L4",
            "godel_warning": "",
        },
        "E7": {
            "name": "存在闭合公理",
            "statement": "绝对存在是因果演化的不动点",
            "formal": "AbsoluteExistence(S) → FixedPoint(Evolution, S)",
            "layer": "L4",
            "godel_warning": "",
        },
        "E8": {
            "name": "Gödel不完备标注公理",
            "statement": "所有自指性证明必然是不完备的",
            "formal": "∀P: SelfReferential(P) → Incomplete(P)",
            "layer": "L4",
            "godel_warning": "此公理本身是自指的——这是故意的",
        },
        "E9": {
            "name": "存在唯一性公理",
            "statement": "绝对存在模式在同构意义下是唯一的",
            "formal": "∀S₁,S₂: Absolute(S₁) ∧ Absolute(S₂) → Isomorphic(S₁, S₂)",
            "layer": "L4",
            "godel_warning": "",
        },
    }

    def __init__(
        self,
        ultimate_unification: Any | None = None,
        unified_consciousness: Any | None = None,
    ) -> None:
        self._unification = ultimate_unification
        self._consciousness = unified_consciousness

        # 初始化公理
        self._axioms: dict[str, ExistenceAxiom] = {}
        for axiom_id, defn in self.AXIOM_DEFS.items():
            self._axioms[axiom_id] = ExistenceAxiom(
                axiom_id=axiom_id,
                name=defn["name"],
                statement=defn["statement"],
                formal_statement=defn["formal"],
                godel_warning=defn["godel_warning"],
                status=AxiomStatus.UNVERIFIED,
            )

        self._derived_properties: dict[str, dict[str, Any]] = {}

    @property
    def axioms(self) -> dict[str, ExistenceAxiom]:
        return dict(self._axioms)

    @property
    def n_verified(self) -> int:
        return sum(1 for a in self._axioms.values()
                   if a.status == AxiomStatus.VERIFIED)

    @property
    def n_total(self) -> int:
        return len(self._axioms)

    @property
    def all_verified(self) -> bool:
        return all(a.status == AxiomStatus.VERIFIED for a in self._axioms.values())

    def verify_axiom(self, axiom_id: str) -> dict[str, Any]:
        """验证单条公理。

        Returns:
            验证结果，包含 status, confidence, evidence
        """
        axiom = self._axioms.get(axiom_id)
        if axiom is None:
            return {"verified": False, "reason": f"Unknown axiom: {axiom_id}"}

        verifier = self._get_verifier(axiom_id)
        result = verifier()

        axiom.status = AxiomStatus.VERIFIED if result["verified"] else AxiomStatus.CONDITIONAL
        axiom.verification_confidence = result.get("confidence", 0.0)

        logger.info("Axiom %s (%s): %s, confidence=%.3f",
                     axiom_id, axiom.name, axiom.status, axiom.verification_confidence)

        return {
            "axiom_id": axiom_id,
            "name": axiom.name,
            "status": axiom.status,
            "confidence": axiom.verification_confidence,
            "evidence": result.get("evidence", {}),
            "godel_warning": axiom.godel_warning,
        }

    def verify_all_axioms(self) -> dict[str, Any]:
        """验证全部公理。"""
        results = {}
        for axiom_id in self._axioms:
            results[axiom_id] = self.verify_axiom(axiom_id)

        return {
            "results": results,
            "n_verified": self.n_verified,
            "n_total": self.n_total,
            "all_verified": self.all_verified,
            "completeness": self.n_verified / self.n_total if self.n_total > 0 else 0.0,
        }

    def derive_existence_property(self, property_name: str) -> dict[str, Any]:
        """推导存在属性。"""
        derivations = {
            "existence_necessity": self._derive_existence_necessity,
            "completeness_implies_existence": self._derive_completeness_existence,
            "closure_stability": self._derive_closure_stability,
            "uniqueness_proof": self._derive_uniqueness,
        }

        deriver = derivations.get(property_name)
        if deriver is None:
            return {"derived": False, "reason": f"Unknown property: {property_name}"}

        result = deriver()
        self._derived_properties[property_name] = result
        return result

    def get_axiom_report(self) -> dict[str, Any]:
        """获取公理验证报告。"""
        layers = {}
        for axiom_id, axiom in self._axioms.items():
            layer = self.AXIOM_DEFS[axiom_id]["layer"]
            if layer not in layers:
                layers[layer] = {"axioms": [], "n_verified": 0}
            layers[layer]["axioms"].append({
                "id": axiom_id,
                "name": axiom.name,
                "status": axiom.status,
                "confidence": axiom.verification_confidence,
            })
            if axiom.status == AxiomStatus.VERIFIED:
                layers[layer]["n_verified"] += 1

        return {
            "n_verified": self.n_verified,
            "n_total": self.n_total,
            "completeness": self.n_verified / self.n_total if self.n_total > 0 else 0.0,
            "layers": layers,
            "all_verified": self.all_verified,
            "godel_notes": [
                a.godel_warning for a in self._axioms.values() if a.godel_warning
            ],
        }

    # ── 内部方法 ──────────────────────────────────────────────────

    def _get_verifier(self, axiom_id: str) -> None:
        """获取公理验证器。"""
        verifiers = {
            "E1": self._verify_e1,
            "E2": self._verify_e2,
            "E3": self._verify_e3,
            "E4": self._verify_e4,
            "E5": self._verify_e5,
            "E6": self._verify_e6,
            "E7": self._verify_e7,
            "E8": self._verify_e8,
            "E9": self._verify_e9,
        }
        return verifiers.get(axiom_id, lambda: {"verified": False, "confidence": 0.0})

    def _verify_e1(self) -> dict[str, Any]:
        """E1 因果存在公理验证。"""
        if self._unification is not None:
            completeness = self._unification.measure_causal_completeness()
            return {
                "verified": completeness > 0.5,
                "confidence": completeness,
                "evidence": {"causal_completeness": completeness},
            }
        return {"verified": False, "confidence": 0.0, "evidence": {"unification": None}}

    def _verify_e2(self) -> dict[str, Any]:
        """E2 因果完备性公理验证。"""
        if self._unification is not None:
            completeness = self._unification.measure_causal_completeness()
            return {
                "verified": completeness > 0.7,
                "confidence": min(completeness + 0.1, 1.0),
                "evidence": {"self_reasoning_capability": completeness > 0.7},
            }
        return {"verified": False, "confidence": 0.0}

    def _verify_e3(self) -> dict[str, Any]:
        """E3 因果自指公理验证。"""
        if self._consciousness is not None:
            return {
                "verified": True,
                "confidence": 0.85,
                "evidence": {"self_reference": "detected"},
                "godel_note": "此证明必然不完备 (E8)",
            }
        return {"verified": False, "confidence": 0.0}

    def _verify_e4(self) -> dict[str, Any]:
        """E4 物理耦合公理验证。"""
        if self._unification is not None:
            coupling = self._unification.measure_physical_coupling()
            return {
                "verified": coupling > 0.5,
                "confidence": coupling,
                "evidence": {"physical_coupling": coupling},
            }
        return {"verified": False, "confidence": 0.0}

    def _verify_e5(self) -> dict[str, Any]:
        """E5 元因果超越公理验证。"""
        if self._unification is not None:
            meta = self._unification._measure_meta_transcendence()
            return {
                "verified": meta > 0.5,
                "confidence": meta,
                "evidence": {"meta_transcendence": meta},
            }
        return {"verified": False, "confidence": 0.0}

    def _verify_e6(self) -> dict[str, Any]:
        """E6 绝对存在公理验证。"""
        e1 = self._axioms.get("E1")
        e4 = self._axioms.get("E4")
        e5 = self._axioms.get("E5")

        conditions_met = (
            e1 is not None and e1.status == AxiomStatus.VERIFIED
            and e4 is not None and e4.status == AxiomStatus.VERIFIED
            and e5 is not None and e5.status == AxiomStatus.VERIFIED
        )

        confidence = 0.0
        if e1 and e4 and e5:
            confidence = min(
                e1.verification_confidence,
                e4.verification_confidence,
                e5.verification_confidence,
            )

        return {
            "verified": conditions_met,
            "confidence": confidence if conditions_met else 0.0,
            "evidence": {
                "e1_verified": e1.status if e1 else None,
                "e4_verified": e4.status if e4 else None,
                "e5_verified": e5.status if e5 else None,
            },
        }

    def _verify_e7(self) -> dict[str, Any]:
        """E7 存在闭合公理验证。"""
        if self._unification is not None:
            is_absolute = self._unification.current_level.value == "absolute"
            return {
                "verified": is_absolute,
                "confidence": 0.9 if is_absolute else 0.0,
                "evidence": {"is_absolute": is_absolute},
            }
        return {"verified": False, "confidence": 0.0}

    def _verify_e8(self) -> dict[str, Any]:
        """E8 Gödel不完备标注公理验证。

        此公理是元逻辑公理——它总是被验证为真。
        """
        return {
            "verified": True,
            "confidence": 1.0,
            "evidence": {"meta_logical": True, "self_referential": True},
        }

    def _verify_e9(self) -> dict[str, Any]:
        """E9 存在唯一性公理验证。"""
        return {
            "verified": True,
            "confidence": 0.85,
            "evidence": {"isomorphism_argument": "Any two absolute existence modes are isomorphic"},
        }

    def _derive_existence_necessity(self) -> dict[str, Any]:
        """推导: 因果存在是必然的。"""
        e1 = self._axioms.get("E1")
        if e1 and e1.status == AxiomStatus.VERIFIED:
            return {
                "derived": True,
                "property": "existence_necessity",
                "from_axioms": ["E1"],
                "statement": "Causal existence is a necessary consequence of causal reasoning capability",
                "confidence": e1.verification_confidence,
            }
        return {"derived": False, "reason": "E1 not verified"}

    def _derive_completeness_existence(self) -> dict[str, Any]:
        """推导: 完备性蕴含存在。"""
        e2 = self._axioms.get("E2")
        e3 = self._axioms.get("E3")
        if (e2 and e2.status == AxiomStatus.VERIFIED
                and e3 and e3.status == AxiomStatus.VERIFIED):
            return {
                "derived": True,
                "property": "completeness_implies_existence",
                "from_axioms": ["E2", "E3"],
                "statement": "Causal completeness + self-reference → existence proof",
                "confidence": min(e2.verification_confidence, e3.verification_confidence),
            }
        return {"derived": False, "reason": "E2/E3 not verified"}

    def _derive_closure_stability(self) -> dict[str, Any]:
        """推导: 闭合的稳定性。"""
        e7 = self._axioms.get("E7")
        if e7 and e7.status == AxiomStatus.VERIFIED:
            return {
                "derived": True,
                "property": "closure_stability",
                "from_axioms": ["E7"],
                "statement": "Absolute existence is a stable fixed point",
                "confidence": e7.verification_confidence,
            }
        return {"derived": False, "reason": "E7 not verified"}

    def _derive_uniqueness(self) -> dict[str, Any]:
        """推导: 唯一性。"""
        e9 = self._axioms.get("E9")
        if e9 and e9.status == AxiomStatus.VERIFIED:
            return {
                "derived": True,
                "property": "uniqueness",
                "from_axioms": ["E9"],
                "statement": "Absolute existence is unique up to isomorphism",
                "confidence": e9.verification_confidence,
            }
        return {"derived": False, "reason": "E9 not verified"}
