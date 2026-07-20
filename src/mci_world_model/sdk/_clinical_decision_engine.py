"""MCI World Model — 临床决策引擎（ClinicalDecisionEngine）

============================================================

Phase 2-3 模块：医疗世界模型的编排层。

将世界模型五要素（S, A, T, R, π）编排为完整的临床决策闭环：

    临床输入 → E(编码) → S(状态) → π(规划) → T(预测) → R(评估) → 临床决策

这是世界模型从"组件"到"系统"的最后一层：把 PatientState、MedicalAction、
ClinicalDynamicsPredictor、ClinicalObjective、ClinicalMCTSPlanner 组装成一个
可调用的决策引擎。

设计原则：
    - 无状态编排层：每次 decide() 调用独立，不持有跨调用状态
    - 全程审计：每步推理记录 audit_trail（合规要求）
    - 安全降级：低置信度/不安全时输出"需医师复核"
    - 与 su-memory-sdk 严格区隔：零持久化，零记忆检索
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mci_world_model.sdk._clinical_causal_discovery import ClinicalCausalDiscovery
from mci_world_model.sdk._clinical_dynamics import ClinicalDynamicsPredictor
from mci_world_model.sdk._clinical_objective import ClinicalObjective
from mci_world_model.sdk._clinical_pearl_bridge import ClinicalPearlBridge
from mci_world_model.sdk._clinical_planner import ClinicalMCTSPlanner, TreatmentPlan
from mci_world_model.sdk._clinical_state_encoder import ClinicalStateEncoder
from mci_world_model.sdk._clinical_tri_router import (
    ClinicalQuery,
    ClinicalTriRouter,
    KnowledgeEntry,
    RouteType,
    SafetyLevel,
    SemanticKnowledgeBase,
)
from mci_world_model.sdk._clinical_world_state import (
    VITAL_NAMES,
    MedicalAction,
    PatientState,
)
from mci_world_model.sdk._compliance_engine import ComplianceRuleEngine
from mci_world_model.sdk._jepa_clinical_bridge import JEPAClinicalBridge, JEPAClinicalConfig
from mci_world_model.sdk._medical_causal_sdk import ClinicalEvidence, MedicalCausalSDK

# =============================================================================
# ClinicalDecision — 统一临床决策输出
# =============================================================================


@dataclass
class ClinicalDecision:
    """统一临床决策输出（含全程审计轨迹）。

    Attributes:
        patient_state: 编码后的患者状态 S
        route_type: 路由类型（physical/causal/semantic/fused）
        route_confidence: 路由置信度
        safety_level: 安全等级
        need_review: 是否需医师复核
        recommended_action: 推荐的临床动作 A（可能为 None）
        treatment_plan: 治疗方案（含预测轨迹和评估）
        current_reward: 当前状态评分 R(s)
        predicted_reward: 预测状态评分 R(s')
        reasoning: 决策推理说明
        audit_trail: 全程审计轨迹
    """

    patient_state: PatientState | None = None
    route_type: RouteType = RouteType.FUSED
    route_confidence: float = 0.0
    safety_level: SafetyLevel = SafetyLevel.NEEDS_REVIEW
    need_review: bool = True
    recommended_action: MedicalAction | None = None
    treatment_plan: TreatmentPlan | None = None
    knowledge_answer: dict[str, Any] | None = None
    current_reward: float = 0.0
    predicted_reward: float = 0.0
    uncertainty_score: float = 0.0
    compliance_report: dict[str, Any] | None = None
    causal_diagnosis: dict[str, Any] | None = None
    reasoning: str = ""
    audit_trail: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典（审计日志/前端展示用）。"""
        return {
            "patient_state": self.patient_state.to_dict() if self.patient_state else None,
            "route": {
                "type": self.route_type.value,
                "confidence": round(self.route_confidence, 4),
                "safety": self.safety_level.value,
                "need_review": self.need_review,
            },
            "recommended_action": (self.recommended_action.to_dict() if self.recommended_action else None),
            "knowledge_answer": self.knowledge_answer,
            "treatment_plan": self.treatment_plan.to_dict() if self.treatment_plan else None,
            "evaluation": {
                "current_reward": round(self.current_reward, 4),
                "predicted_reward": round(self.predicted_reward, 4),
                "improvement": round(self.predicted_reward - self.current_reward, 4),
                "uncertainty_score": round(self.uncertainty_score, 4),
            },
            "compliance": self.compliance_report,
            "causal_diagnosis": self.causal_diagnosis,
            "reasoning": self.reasoning,
            "audit_steps": len(self.audit_trail),
            "timestamp": self.timestamp,
        }

    def __repr__(self) -> str:
        action = (
            f"{self.recommended_action.target} {self.recommended_action.magnitude}"
            if self.recommended_action
            else "no_action"
        )
        return (
            f"ClinicalDecision(route={self.route_type.value}, action={action}, "
            f"reward {self.current_reward:.2f}→{self.predicted_reward:.2f}, "
            f"review={self.need_review})"
        )


# =============================================================================
# ClinicalDecisionEngine — 临床决策编排层
# =============================================================================


class ClinicalDecisionEngine:
    """临床决策引擎 — 医疗世界模型五要素编排层。

    无状态编排：每次 decide() 调用独立完成完整的决策流程，不维护跨调用状态。

    流程：
        1. 编码 E：临床输入 → PatientState
        2. 路由：判断查询类型（物理/因果/语义）
        3. 规划 π：搜索最优治疗方案
        4. 预测 T：预测治疗后的状态变化
        5. 评估 R：量化状态改善程度
        6. 安全校验：置信度/安全性检查
        7. 输出 ClinicalDecision（含审计轨迹）

    Example:
        >>> engine = ClinicalDecisionEngine()
        >>> engine.fit(n_samples=1000, n_epochs=300)  # 训练世界模型
        >>> decision = engine.decide_from_vitals(
        ...     vital_records=[{"heart_rate": 130, "systolic_bp": 140, ...}],
        ...     query="患者心动过速如何处理？",
        ... )

    与 su-memory-sdk 的边界：
        本引擎是无状态编排层，不持久化决策结果。
        如需跨调用记忆（经验复用/案例库），通过 adapters/su_memory_bridge.py 对接。
    """

    def __init__(
        self,
        predictor: ClinicalDynamicsPredictor | None = None,
        objective: ClinicalObjective | None = None,
        router: ClinicalTriRouter | None = None,
        knowledge_base: SemanticKnowledgeBase | None = None,
        confidence_threshold: float = 0.6,
        uncertainty_threshold: float = 0.3,
    ) -> None:
        """初始化决策引擎。

        Args:
            predictor: 转移模型 T（未提供则延迟初始化）。
            objective: 评估函数 R（默认 ClinicalObjective）。
            router: 三元路由器（可选，用于查询类型判断）。
            knowledge_base: 语义知识库（可选，用于 semantic 路径检索）。
                未提供时自动创建带临床指南条目的默认知识库。
            confidence_threshold: 置信度阈值，低于此值标记需复核。
            uncertainty_threshold: 不确定性分数阈值，高于此值标记需复核。
        """
        self._predictor = predictor
        self._objective = objective or ClinicalObjective()
        self._router = router or ClinicalTriRouter(seed=42)
        self._confidence_threshold = confidence_threshold
        self._uncertainty_threshold = uncertainty_threshold
        self._planner: ClinicalMCTSPlanner | None = None
        self._encoder = ClinicalStateEncoder()
        self._knowledge_base = knowledge_base or self._build_default_knowledge_base()
        self._medical_sdk = MedicalCausalSDK(strict_mode=False)
        self._compliance_engine = ComplianceRuleEngine(auto_register_defaults=True)
        # D4: 数据驱动的临床因果发现器（从本次推理的状态数据学习，非持久化）
        self._causal_discovery = ClinicalCausalDiscovery(significance=0.05, min_samples=10)
        # 方向三: 因果下沉桥接（L1发现+L2干预+L3反事实）
        self._pearl_bridge = ClinicalPearlBridge(seed=42)
        self._fitted = predictor is not None and predictor.is_fitted

    @property
    def is_fitted(self) -> bool:
        """世界模型是否已训练。"""
        return self._fitted

    def fit(
        self,
        n_samples: int = 2000,
        n_epochs: int = 500,
        lr: float = 0.005,
        seed: int = 42,
    ) -> dict[str, Any]:
        """训练世界模型转移 T（从药效基线表）。

        Args:
            n_samples: 训练样本数。
            n_epochs: 训练轮数。
            lr: 学习率。
            seed: 随机种子。

        Returns:
            训练信息。
        """
        if self._predictor is None:
            self._predictor = ClinicalDynamicsPredictor(seed=seed)
        info = self._predictor.fit_from_effect_table(n_samples=n_samples, n_epochs=n_epochs, lr=lr)
        self._planner = ClinicalMCTSPlanner(predictor=self._predictor, objective=self._objective)
        self._fitted = True
        return info

    def fit_with_jepa(
        self,
        n_samples: int = 2000,
        n_epochs: int = 500,
        latent_dim: int = 64,
        lr: float = 0.005,
        seed: int = 42,
        use_semantic: bool = False,
    ) -> dict[str, Any]:
        """方向一/二：用 JEPA 潜空间 backend 训练转移模型 T。

        与 fit()（原始空间 MLP）的区别：
            - fit(): ClinicalDynamicsPredictor 在 R¹³ 观测空间直接预测
            - fit_with_jepa(): JEPAClinicalBridge 在潜空间预测（EMA+VICReg+重建）

        方向二（use_semantic=True）额外启用：
            - 临床语义嵌入：诊断/用药编码为语义向量融入状态空间
            - 让世界模型能区分"相同体征、不同病因"的临床情境

        Args:
            n_samples: 训练样本数。
            n_epochs: 训练轮数。
            latent_dim: 潜空间维度。
            lr: 学习率。
            seed: 随机种子。
            use_semantic: 是否启用临床语义嵌入（方向二）。

        Returns:
            训练信息（含 backend="jepa"）。
        """
        from mci_world_model.sdk._clinical_semantic_embedding import ClinicalSemanticEmbedding

        cfg = JEPAClinicalConfig(latent_dim=latent_dim, lr=lr, seed=seed)
        sem = ClinicalSemanticEmbedding() if use_semantic else None
        bridge = JEPAClinicalBridge(cfg, semantic_embedder=sem)
        info = bridge.fit_from_effect_table(n_samples=n_samples, n_epochs=n_epochs)
        self._predictor = bridge
        self._planner = ClinicalMCTSPlanner(predictor=bridge, objective=self._objective)
        self._fitted = True
        return info

    def attach_jepa_bridge(self, bridge: JEPAClinicalBridge) -> None:
        """挂载已训练的 JEPA 桥接作为转移模型 backend。

        Args:
            bridge: 已训练的 JEPAClinicalBridge。
        """
        if not bridge.is_fitted:
            raise ValueError("JEPA 桥接未训练，无法挂载")
        self._predictor = bridge
        self._planner = ClinicalMCTSPlanner(predictor=bridge, objective=self._objective)
        self._fitted = True

    def get_backend_type(self) -> str:
        """获取当前转移模型 backend 类型。

        Returns:
            "jepa" / "mlp" / "none"。
        """
        if self._predictor is None:
            return "none"
        if isinstance(self._predictor, JEPAClinicalBridge):
            return "jepa"
        return "mlp"

    def decide_from_vitals(
        self,
        vital_records: list[dict[str, float]],
        query: str = "",
        patient_id: str = "",
        age: int = 0,
        gender: str = "",
    ) -> ClinicalDecision:
        """从体征数据执行端到端临床决策。

        Args:
            vital_records: 体征记录列表（每条一个时间窗）。
            query: 临床自然语言查询（可选）。
            patient_id: 患者ID。
            age: 年龄。
            gender: 性别。

        Returns:
            ClinicalDecision（含推荐动作、预测轨迹、审计链）。
        """
        audit: list[dict[str, Any]] = []
        decision = ClinicalDecision(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))

        # Step 1: 编码 E — 临床输入 → PatientState
        t0 = time.time()
        state = self._encoder.encode_from_dicts(vital_records, patient_id=patient_id, age=age, gender=gender)
        audit.append(
            {
                "step": "encode",
                "input": f"{len(vital_records)} 条体征记录",
                "output": repr(state),
                "duration_ms": round((time.time() - t0) * 1000, 1),
            }
        )
        decision.patient_state = state

        # Step 2: 路由判断（简单规则版，无需 router）
        t0 = time.time()
        route_type, route_confidence = self._simple_route(query, state)
        audit.append(
            {
                "step": "route",
                "input": query[:50] if query else "(no query)",
                "output": route_type.value,
                "duration_ms": round((time.time() - t0) * 1000, 1),
            }
        )
        decision.route_type = route_type
        decision.route_confidence = route_confidence

        # Step 3: 语义路径 — 知识库检索（BGE-M3 语义匹配）
        if route_type == RouteType.SEMANTIC and query:
            t0 = time.time()
            entry, score, method = self._knowledge_base.query_with_score(query)
            audit.append(
                {
                    "step": "knowledge_retrieval",
                    "input": query[:50],
                    "output": entry.term if entry else "无匹配",
                    "similarity": round(score, 4),
                    "method": method,
                    "duration_ms": round((time.time() - t0) * 1000, 1),
                }
            )
            if entry is not None:
                decision.knowledge_answer = {
                    "term": entry.term,
                    "definition": entry.definition,
                    "guideline": entry.guideline,
                    "retrieval_score": round(score, 4),
                    "retrieval_method": method,
                }
                decision.reasoning = f"知识库检索: {entry.term} (相似度={score:.3f})"
            else:
                decision.knowledge_answer = {"term": None, "reason": "知识库无匹配"}
                decision.reasoning = "知识库无匹配条目"

        # Step 4: 评估 R — 当前状态评分
        decision.current_reward = self._objective.reward(state)

        # Step 4: 规划 π — 搜索治疗方案
        if self._planner is not None and self._fitted:
            t0 = time.time()
            plan = self._planner.plan(state)
            audit.append(
                {
                    "step": "plan",
                    "input": repr(state),
                    "output": repr(plan),
                    "duration_ms": round((time.time() - t0) * 1000, 1),
                }
            )
            decision.treatment_plan = plan
            decision.recommended_action = plan.best_action
            decision.predicted_reward = plan.best_predicted_reward

            # Step 5a: 因果诊断 + 合规校验（仅 CAUSAL 路径）
            if route_type == RouteType.CAUSAL and decision.recommended_action:
                t0 = time.time()
                action = decision.recommended_action
                # 用 MedicalCausalSDK 做因果诊断
                try:
                    self._medical_sdk.clear_evidence()
                    # 从体征构造证据
                    self._medical_sdk.add_evidence(
                        ClinicalEvidence(
                            evidence_id="vital_observation",
                            evidence_type="vital_sign",
                            description=f"患者体征: HR={state.vital_signs[-1][0]:.0f}",
                            confidence=0.8,
                        )
                    )
                    diagnosis = self._medical_sdk.diagnose(
                        cause=action.target,
                        effect="clinical_state",
                        prior_strength=0.5,
                    )
                    decision.causal_diagnosis = {
                        "is_conclusive": diagnosis.is_conclusive,
                        "confidence": round(diagnosis.confidence, 4),
                        "warnings": diagnosis.warnings,
                        "evidence_count": len(diagnosis.evidence_ids),
                    }
                except Exception:
                    decision.causal_diagnosis = {"error": "diagnosis_failed"}

                # 合规校验
                try:
                    compliance = self._compliance_engine.check(
                        {
                            "domain": "medical",
                            "action": action.target,
                            "confidence": decision.route_confidence,
                            "has_audit_trail": len(audit) > 0,
                        }
                    )
                    decision.compliance_report = {
                        "is_compliant": compliance.is_compliant,
                        "summary": compliance.summary,
                        "n_rules_checked": len(compliance.reports),
                    }
                except Exception:
                    decision.compliance_report = {"error": "compliance_check_failed"}

                audit.append(
                    {
                        "step": "compliance_check",
                        "input": action.target,
                        "output": "compliant"
                        if (decision.compliance_report and decision.compliance_report.get("is_compliant"))
                        else "non_compliant",
                        "duration_ms": round((time.time() - t0) * 1000, 1),
                    }
                )

            # Step 5b: 不确定性量化（贝叶斯 bootstrap CI）
            t0 = time.time()
            if decision.recommended_action is not None:
                try:
                    uq = self._predictor.predict_with_uncertainty(
                        state,
                        decision.recommended_action,
                        n_steps=1,
                        n_bootstrap=20,
                    )
                    decision.uncertainty_score = uq.uncertainty_score()
                    audit.append(
                        {
                            "step": "uncertainty",
                            "input": repr(decision.recommended_action),
                            "output": f"score={decision.uncertainty_score:.4f}",
                            "duration_ms": round((time.time() - t0) * 1000, 1),
                        }
                    )
                except (ValueError, RuntimeError):
                    decision.uncertainty_score = 0.5

            # Step 6: 三级安全降级
            reward_worse = decision.predicted_reward < decision.current_reward
            low_confidence = route_confidence < self._confidence_threshold
            high_uncertainty = decision.uncertainty_score > self._uncertainty_threshold
            non_compliant = decision.compliance_report is not None and not decision.compliance_report.get(
                "is_compliant", True
            )

            if non_compliant or (low_confidence and high_uncertainty):
                # 低置信 + 高不确定 → 拒绝输出
                decision.safety_level = SafetyLevel.REFUSED
                decision.need_review = True
                if non_compliant:
                    decision.reasoning = f"拒绝输出：合规校验失败 ({decision.compliance_report.get('summary', '')})"
                else:
                    decision.reasoning = (
                        f"拒绝输出：路由置信度低({route_confidence:.2f}) + 不确定性高({decision.uncertainty_score:.3f})"
                    )
            elif reward_worse or low_confidence or high_uncertainty:
                # 任一条件触发 → 需复核
                decision.safety_level = SafetyLevel.NEEDS_REVIEW
                decision.need_review = True
                reasons = []
                if non_compliant:
                    reasons.append("合规校验未通过")
                if reward_worse:
                    reasons.append(f"reward 下降({decision.predicted_reward - decision.current_reward:+.3f})")
                if low_confidence:
                    reasons.append(f"路由置信度低({route_confidence:.2f})")
                if high_uncertainty:
                    reasons.append(f"不确定性高({decision.uncertainty_score:.3f})")
                decision.reasoning = "需复核：" + "；".join(reasons)
            else:
                decision.safety_level = SafetyLevel.TRUSTED
                decision.need_review = False
                decision.reasoning = (
                    f"{plan.reasoning}；不确定性 {decision.uncertainty_score:.3f} < 阈值 {self._uncertainty_threshold}"
                )
        else:
            decision.need_review = True
            decision.safety_level = SafetyLevel.REFUSED
            decision.reasoning = "世界模型未训练，无法提供决策建议"

        decision.audit_trail = audit
        return decision

    def discover_causal_structure(
        self,
        vitals_history: np.ndarray,
        max_conditioning_size: int = 1,
    ) -> dict[str, Any]:
        """D4: 从患者体征时序数据发现因果结构（数据驱动）。

        与 MedicalCausalSDK（规则因果诊断）互补：本方法从数据中学习
        体征间因果结构，提供数据驱动的因果归因。

        输入是本次推理的患者多时间窗体征数据（非持久化记忆，
        不违反 su-memory-sdk 边界）。

        Args:
            vitals_history: 体征矩阵 shape (T, N_VITALS)。
            max_conditioning_size: PC 算法 conditioning set 最大规模。

        Returns:
            因果结构字典（含 links、adjacency、n_samples）。
        """
        structure = self._causal_discovery.discover(vitals_history, max_conditioning_size=max_conditioning_size)
        return structure.to_dict()

    def causal_intervention_effect(
        self,
        patient_history: np.ndarray,
        treatment_vital: str,
        outcome_vital: str,
    ) -> dict[str, Any]:
        """方向三 L2: 估计 do(treatment_vital) 对 outcome_vital 的因果效应。

        用 do-calculus（后门调整）估计干预效应，而非朴素关联。
        需要从患者历史数据先发现因果结构（L1），再下沉到干预层（L2）。

        Args:
            patient_history: 体征时序矩阵 (T, N_VITALS)。
            treatment_vital: 干预体征名（如 "heart_rate"）。
            outcome_vital: 结果体征名（如 "systolic_bp"）。

        Returns:
            干预效应字典（含 ATE/CI/调整集/方法）。
        """
        structure = self._pearl_bridge.discover(patient_history)
        # 构造观测数据字典
        data_dict = {name: patient_history[:, i] for i, name in enumerate(VITAL_NAMES) if i < patient_history.shape[1]}
        result = self._pearl_bridge.intervene(structure, treatment_vital, outcome_vital, data_dict)
        return result.to_dict()

    def counterfactual_evaluation(
        self,
        patient_history: np.ndarray,
        observed_state: PatientState,
        factual_action: MedicalAction,
        alternative_action: MedicalAction,
        target_vital: str = "systolic_bp",
    ) -> dict[str, Any]:
        """方向三 L3: 反事实治疗评估。

        回答"若当初选 alternative_action 而非 factual_action，
        target_vital 会怎样"。用 Pearl 反事实推理（噪声对齐+因果图调整）。

        Args:
            patient_history: 体征历史（用于发现因果结构）。
            observed_state: 观测到的患者状态。
            factual_action: 实际施加的动作。
            alternative_action: 反事实假设的动作。
            target_vital: 评估目标体征。

        Returns:
            反事实评估字典。
        """
        structure = self._pearl_bridge.discover(patient_history)
        return self._pearl_bridge.counterfactual_treatment_eval(
            structure, observed_state, factual_action, alternative_action, target_vital
        )

    @staticmethod
    def _build_default_knowledge_base() -> SemanticKnowledgeBase:
        """构建默认临床知识库（含常见 ICU 指南条目）。"""
        kb = SemanticKnowledgeBase(semantic_threshold=0.5)
        entries = [
            KnowledgeEntry(
                term="感染性休克",
                definition="感染导致的全身性炎症反应综合征，引起循环衰竭和组织灌注不足",
                guideline="早期目标导向治疗：6小时内 CVP 8-12mmHg，MAP≥65mmHg，尿量≥0.5ml/kg/h",
            ),
            KnowledgeEntry(
                term="心源性休克",
                definition="心脏泵功能衰竭导致心输出量急剧下降，组织缺血缺氧",
                guideline="血管活性药物首选去甲肾上腺素，必要时联合多巴酚丁胺",
            ),
            KnowledgeEntry(
                term="急性呼吸窘迫综合征",
                definition="肺毛细血管渗漏导致难治性低氧血症，即 ARDS",
                guideline="小潮气量通气 6ml/kg 理想体重，平台压<30cmH2O，俯卧位通气",
            ),
            KnowledgeEntry(
                term="急性肾损伤",
                definition="肾功能在48小时内急剧下降，血肌酐升高或尿量减少",
                guideline="KDIGO 分级，避免肾毒性药物，必要时肾脏替代治疗",
            ),
            KnowledgeEntry(
                term="多巴胺",
                definition="儿茶酚胺类血管活性药物，剂量依赖性兴奋α/β/多巴胺受体",
                guideline="低剂量(1-3μg/kg/min)扩肾血管；中剂量(3-10)强心；高剂量(>10)升压",
            ),
            KnowledgeEntry(
                term="去甲肾上腺素",
                definition="强效α受体激动剂，一线升压药，收缩外周血管升高血压",
                guideline="感染性休克首选升压药，目标 MAP≥65mmHg，中心静脉给药",
            ),
            # ── 扩充：常见 ICU 综合征 ──
            KnowledgeEntry(
                term="脓毒症",
                definition="感染导致的器官功能障碍综合征（Sepsis-3 定义），SOFA 评分急性增加≥2",
                guideline="1小时集束化治疗：测乳酸、血培养、广谱抗生素、晶体液30ml/kg、升压药",
            ),
            KnowledgeEntry(
                term="多器官功能障碍综合征",
                definition="MODS，全身性炎症反应导致两个或以上器官同时或序贯性功能障碍",
                guideline="支持性治疗为主，各器官功能支持（呼吸/循环/肾脏/肝脏），消除诱因",
            ),
            KnowledgeEntry(
                term="糖尿病酮症酸中毒",
                definition="DKA，胰岛素绝对缺乏致高血糖、酮症、代谢性酸中毒",
                guideline="小剂量胰岛素 0.1U/kg/h，生理盐水补液，监测钾离子防低钾",
            ),
            KnowledgeEntry(
                term="急性肝衰竭",
                definition="ALF，无既往肝病者出现肝功能急剧恶化伴凝血障碍/脑病",
                guideline="N-乙酰半胱氨酸，监测颅内压，肝移植评估，纠正凝血",
            ),
            KnowledgeEntry(
                term="上消化道出血",
                definition="Treitz 韧带以上消化道出血，表现为呕血/黑便",
                guideline="内镜止血（24小时内），质子泵抑制剂，风险评估（Glasgow-Blatchford）",
            ),
            # ── 扩充：常用药物 ──
            KnowledgeEntry(
                term="美托洛尔",
                definition="选择性β1受体阻滞剂，减慢心率、降低心肌耗氧",
                guideline="室上速/房颤控心率，目标心率<110，哮喘慎用",
            ),
            KnowledgeEntry(
                term="肾上腺素",
                definition="非选择性α/β受体激动剂，强心+升压+扩支气管",
                guideline="过敏反应首选（0.3-0.5mg IM），心脏骤停（1mg IV）",
            ),
            KnowledgeEntry(
                term="呋塞米",
                definition="襻利尿剂，抑制 Henle 襻升支钠钾氯重吸收",
                guideline="容量过负荷/急性肺水肿，监测电解质防低钾",
            ),
            KnowledgeEntry(
                term="丙泊酚",
                definition="静脉全身麻醉药，起效快、苏醒快，ICU 镇静常用",
                guideline="镇静剂量 0.5-3mg/kg/h，监测低血压、丙泊酚输注综合征",
            ),
            # ── 扩充：常见操作与监测 ──
            KnowledgeEntry(
                term="机械通气",
                definition="呼吸机提供正压通气支持，用于呼吸衰竭患者",
                guideline="ARDS 用小潮气量（6ml/kg），COPD 用低频率长呼气，定期评估撤机",
            ),
            KnowledgeEntry(
                term="中心静脉压",
                definition="CVP，反映右心前负荷和容量状态，正常 8-12mmHg",
                guideline="容量管理参考指标，补液试验观察 CVP 变化评估液体反应性",
            ),
            KnowledgeEntry(
                term="SOFA评分",
                definition="序贯器官衰竭评估，6器官系统各0-4分，总分反映器官功能",
                guideline="用于脓毒症诊断（≥2分急性增加），动态监测评估病情进展",
            ),
        ]
        for entry in entries:
            kb.add(entry)
        return kb

    def _route_with_tri_router(self, query: str, state: PatientState) -> tuple[RouteType, float]:
        """使用 ClinicalTriRouter 做三元融合路由。

        将患者状态 + 查询构造为 ClinicalQuery，调用 ClinicalTriRouter.route()。
        ClinicalTriRouter 使用 LSA 语义嵌入 + 原型向量匹配做路由决策。

        Args:
            query: 临床查询字符串。
            state: 患者状态。

        Returns:
            (路由类型, 置信度)。
        """
        if not query:
            return RouteType.CAUSAL, 0.7

        # 构造 ClinicalQuery
        cq = ClinicalQuery(
            query_text=query,
            patient_state=state.vital_signs,
            evidence_count=3,
        )

        try:
            decision = self._router.route(cq)
            return decision.route_type, decision.confidence
        except Exception:
            # 降级到简单关键词路由
            return self._simple_route(query, state)

    def _simple_route(self, query: str, state: PatientState) -> tuple[RouteType, float]:
        """简单路由判断（基于关键词 + 状态）。

        Args:
            query: 临床查询字符串。
            state: 患者状态。

        Returns:
            (路由类型, 置信度)。
        """
        if not query:
            # 无查询时默认 causal（治疗决策）
            return RouteType.CAUSAL, 0.7

        # 生理预测类
        if any(kw in query for kw in ["趋势", "预测", "走势", "未来", "推演"]):
            return RouteType.PHYSICAL, 0.85

        # 知识问答类
        if any(
            kw in query
            for kw in [
                "是什么",
                "定义",
                "机制",
                "指南",
                "原理",
                "概念",
                "原则",
                "适应症",
                "禁忌",
                "适应",
                "指征",
                "标准",
            ]
        ):
            return RouteType.SEMANTIC, 0.80

        # 因果干预类（默认）
        if any(kw in query for kw in ["给药", "剂量", "副作用", "影响", "效应", "处理", "治疗"]):
            return RouteType.CAUSAL, 0.85

        return RouteType.FUSED, 0.6
