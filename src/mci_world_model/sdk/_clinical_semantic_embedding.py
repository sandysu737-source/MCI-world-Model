"""MCI World Model — 临床语义嵌入（ClinicalSemanticEmbedding）

============================================================

方向二：真实嵌入 — 把诊断/用药的语义信息融入世界模型状态空间。

为什么需要？
    方向一接通 JEPA 后，PatientState 仍是纯数值向量 R¹³（7 体征 + 6 检验），
    诊断（ICD-10 代码）和用药（药物名+剂量）的语义**完全丢失**。
    这导致世界模型无法区分"心率 130 + 诊断房颤"vs"心率 130 + 诊断甲亢"
    —— 两者体征相同但病因、治疗方案截然不同。

    方向二补齐这一缺口：把诊断/用药编码为语义向量，与体征数值融合，
    让世界模型状态空间携带临床语义。

设计原则（AGENTS.md 边界）:
    - 无状态计算：嵌入由 embedder 现算，不缓存到磁盘（存储归 su-memory）
    - 轻量无重依赖：不强制 BGE-M3（4.2GB），用临床概念哈希投影，
      可选注入外部嵌入器（如 BGEM3Embedder）做增强
    - 可降级：无外部嵌入器时退化为哈希投影（定性正确，精度略低）
    - 可复现：哈希种子固定

嵌入方案（三层，精度递进）:
    Layer 1: ICD-10 → 临床概念桶（如 I48 → "心律失常"）
    Layer 2: 概念桶 → 哈希投影向量（固定维，语义相近的桶向量相近）
    Layer 3: （可选）外部嵌入器（BGE-M3）做文本语义增强

用药嵌入:
    药物名 → 药理学类别桶（如 "metoprolol" → "β受体阻滞剂"）
    类别桶 → 哈希投影 + 剂量归一化
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from mci_world_model.sdk._clinical_world_state import (
    Medication,
)

# =============================================================================
# 临床概念映射表（ICD-10 → 概念桶 / 药物 → 药理学类别）
# =============================================================================

#: ICD-10 代码前缀 → 临床概念桶（用于语义聚合）
#: 来源：ICD-10 WHO 标准编码范围的中文临床共识映射（研究级近似）
ICD10_CONCEPT_MAP: dict[str, str] = {
    # 循环系统（I00-I99）
    "I48": "心律失常",
    "I47": "心律失常",
    "I49": "心律失常",
    "I21": "急性冠脉综合征",
    "I22": "急性冠脉综合征",
    "I50": "心力衰竭",
    "I11": "高血压性心脏病",
    "I10": "高血压",
    "I63": "脑血管病",
    "I64": "脑血管病",
    "I80": "静脉血栓",
    # 呼吸系统（J00-J99）
    "J44": "慢性阻塞性肺病",
    "J45": "哮喘",
    "J96": "呼吸衰竭",
    "J81": "急性肺水肿",
    "J15": "肺炎",
    "J18": "肺炎",
    # 消化系统（K00-K93）
    "K92": "消化道出血",
    "K72": "肝衰竭",
    "K85": "急性胰腺炎",
    # 内分泌/代谢（E00-E90）
    "E11": "糖尿病",
    "E10": "糖尿病",
    "E87": "酸碱平衡紊乱",
    "E83": "电解质紊乱",
    # 泌尿系统（N00-N99）
    "N17": "急性肾损伤",
    "N18": "慢性肾脏病",
    "N19": "肾脏替代治疗",
    # 感染/脓毒症（A00-B99, 部分系统性）
    "A40": "脓毒症",
    "A41": "脓毒症",
    "A49": "细菌感染",
    "B99": "感染",
    # 神经系统（G00-G99）
    "G40": "癫痫",
    "G45": "短暂性脑缺血",
    "G93": "脑病",
    # 损伤/中毒（S00-T98）
    "T78": "过敏反应",
    "T50": "药物中毒",
}

#: 药物名 → 药理学类别桶
DRUG_CLASS_MAP: dict[str, str] = {
    "dopamine": "儿茶酚胺类血管活性药",
    "norepinephrine": "儿茶酚胺类血管活性药",
    "epinephrine": "儿茶酚胺类血管活性药",
    "metoprolol": "β受体阻滞剂",
    "furosemide": "襻利尿剂",
    "fluid_resuscitation": "容量复苏",
}

#: 所有临床概念桶的有序列表（用于 one-hot 索引）
CLINICAL_CONCEPTS: list[str] = sorted(set(ICD10_CONCEPT_MAP.values()) | set(DRUG_CLASS_MAP.values()))
CONCEPT_TO_ID: dict[str, int] = {c: i for i, c in enumerate(CLINICAL_CONCEPTS)}
N_CONCEPTS: int = len(CLINICAL_CONCEPTS)

#: 诊断嵌入默认维度（哈希投影目标维，> N_CONCEPTS 以分散）
DIAG_EMBED_DIM: int = 32
#: 用药嵌入默认维度
MED_EMBED_DIM: int = 16


def icd10_to_concept(code: str) -> str:
    """ICD-10 代码 → 临床概念桶。

    匹配逻辑：取代码前 3 位（如 "I48.0" → "I48"）查映射表，
    未命中则按首字母归类为"其他{系统}"。

    Args:
        code: ICD-10 代码（如 "I48.91", "N17.0"）。

    Returns:
        临床概念桶名称（如 "心律失常"）。
    """
    code = code.strip().upper().replace(".", "")
    prefix3 = code[:3]
    if prefix3 in ICD10_CONCEPT_MAP:
        return ICD10_CONCEPT_MAP[prefix3]
    # 按首字母归类（ICD-10 章节）
    chapter_map = {
        "I": "循环系统疾病",
        "J": "呼吸系统疾病",
        "K": "消化系统疾病",
        "E": "内分泌代谢疾病",
        "N": "泌尿系统疾病",
        "A": "感染性疾病",
        "B": "感染性疾病",
        "G": "神经系统疾病",
        "S": "损伤",
        "T": "损伤中毒",
    }
    return chapter_map.get(code[0] if code else "", "未分类")


def drug_to_class(drug_name: str) -> str:
    """药物名 → 药理学类别桶。

    Args:
        drug_name: 药物名（如 "metoprolol", "多巴胺"）。

    Returns:
        药理学类别名称。
    """
    name = drug_name.strip().lower()
    if name in DRUG_CLASS_MAP:
        return DRUG_CLASS_MAP[name]
    # 中文名回查（反向）
    for en, cn_class in DRUG_CLASS_MAP.items():
        # 简单包含判断（如 "多巴胺" 匹配）
        cn_drug_map = {
            "dopamine": "多巴胺",
            "norepinephrine": "去甲肾上腺素",
            "epinephrine": "肾上腺素",
            "metoprolol": "美托洛尔",
            "furosemide": "呋塞米",
            "fluid_resuscitation": "液体复苏",
        }
        if en in cn_drug_map and cn_drug_map[en] in drug_name:
            return cn_class
    return "其他药物"


# =============================================================================
# 哈希投影嵌入器（轻量、无依赖、可复现）
# =============================================================================


class _HashProjectionEmbedder:
    """概念桶 → 哈希投影向量（语义相近的桶向量相近）。

    用确定性哈希把概念名映射到固定维向量。
    语义相近的概念（如"心律失常"vs"急性冠脉综合征"都属循环系统）
    通过共享字符/词素获得相近的哈希签名。
    """

    def __init__(self, embed_dim: int = DIAG_EMBED_DIM, seed: int = 42) -> None:
        self._embed_dim = embed_dim
        self._seed = seed
        # 预计算所有概念的嵌入（缓存，无持久化）
        self._cache: dict[str, np.ndarray] = {}

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def embed_concept(self, concept: str) -> np.ndarray:
        """单个概念桶 → 嵌入向量。"""
        if concept in self._cache:
            return self._cache[concept].copy()
        # 字符级特征哈希：每个字符贡献一个哈希位
        vec = np.zeros(self._embed_dim, dtype=np.float64)
        # 为每个字符位置生成确定性投影
        for i, ch in enumerate(concept):
            # 字符码 + 位置做种子，生成 embed_dim 向量后叠加
            char_seed = (self._seed * 31 + ord(ch) * 17 + i * 7) % (2**31)
            char_rng = np.random.default_rng(char_seed)
            vec += char_rng.standard_normal(self._embed_dim) * 0.1
        # L2 归一化（语义相近的概念因共享字符而向量相近）
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm
        self._cache[concept] = vec.copy()
        return vec

    def embed_concepts(self, concepts: list[str]) -> np.ndarray:
        """多个概念 → 聚合嵌入向量（均值池化）。

        Args:
            concepts: 概念桶列表。

        Returns:
            聚合嵌入 (embed_dim,)。空列表返回零向量。
        """
        if not concepts:
            return np.zeros(self._embed_dim, dtype=np.float64)
        vecs = [self.embed_concept(c) for c in concepts]
        agg = np.mean(vecs, axis=0)
        norm = np.linalg.norm(agg)
        if norm > 1e-10:
            agg = agg / norm
        return agg


# =============================================================================
# 外部嵌入器协议（可选增强，如 BGE-M3）
# =============================================================================


class TextEmbedderProtocol(Protocol):
    """外部文本嵌入器协议（如 BGEM3Embedder）。"""

    def embed(self, text: str) -> np.ndarray:
        """文本 → 嵌入向量。"""
        ...


# =============================================================================
# ClinicalSemanticEmbedding — 主嵌入器
# =============================================================================


@dataclass
class SemanticStateVector:
    """患者语义状态向量（体征数值 ⊕ 诊断嵌入 ⊕ 用药嵌入）。

    Attributes:
        numeric: 体征+检验数值向量 R^13（原 to_vector）
        diagnosis_embedding: 诊断语义向量 R^diag_dim
        medication_embedding: 用药语义向量 R^med_dim
        concepts: 解析出的临床概念列表（审计用）
    """

    numeric: np.ndarray
    diagnosis_embedding: np.ndarray
    medication_embedding: np.ndarray
    concepts: list[str]

    @property
    def full_vector(self) -> np.ndarray:
        """拼接的完整语义向量 [numeric | diag | med]。"""
        return np.concatenate(
            [
                self.numeric,
                self.diagnosis_embedding,
                self.medication_embedding,
            ]
        )

    @property
    def semantic_dim(self) -> int:
        """语义部分维度（diag + med）。"""
        return len(self.diagnosis_embedding) + len(self.medication_embedding)

    @property
    def full_dim(self) -> int:
        """完整维度（numeric + semantic）。"""
        return len(self.numeric) + self.semantic_dim

    def to_dict(self) -> dict[str, Any]:
        """序列化（审计用，嵌入向量取前 4 位精度）。"""
        return {
            "numeric_dim": len(self.numeric),
            "diagnosis_embedding_dim": len(self.diagnosis_embedding),
            "medication_embedding_dim": len(self.medication_embedding),
            "full_dim": self.full_dim,
            "concepts": self.concepts,
            "diagnosis_embedding_preview": [round(float(v), 4) for v in self.diagnosis_embedding[:4]],
        }


class ClinicalSemanticEmbedding:
    """临床语义嵌入器 — 把诊断/用药编码为语义向量。

    三层嵌入（精度递进，可降级）:
        Layer 1: ICD-10 → 临床概念桶（icd10_to_concept）
        Layer 2: 概念桶 → 哈希投影向量（_HashProjectionEmbedder）
        Layer 3: （可选）外部文本嵌入器增强（TextEmbedderProtocol）

    Example:
        >>> from mci_world_model.sdk import PatientState, Medication
        >>> embedder = ClinicalSemanticEmbedding()
        >>> state = PatientState(
        ...     vital_signs=np.array([[130, 140, 90, 98, 20, 37, 15]]),
        ...     diagnoses=["I48.91"],
        ...     medications=[Medication(name="metoprolol", dose=5.0)],
        ... )
        >>> sem = embedder.embed(state)
        >>> print(sem.full_dim)  # 13 + 32 + 16 = 61
    """

    def __init__(
        self,
        diag_embed_dim: int = DIAG_EMBED_DIM,
        med_embed_dim: int = MED_EMBED_DIM,
        text_embedder: TextEmbedderProtocol | None = None,
        seed: int = 42,
    ) -> None:
        """初始化临床语义嵌入器。

        Args:
            diag_embed_dim: 诊断嵌入维度（默认 32）。
            med_embed_dim: 用药嵌入维度（默认 16）。
            text_embedder: 可选外部文本嵌入器（如 BGEM3Embedder），
                提供时用其做语义增强（Layer 3），否则退化为哈希投影。
            seed: 哈希随机种子。
        """
        self._diag_embedder = _HashProjectionEmbedder(diag_embed_dim, seed)
        self._med_embedder = _HashProjectionEmbedder(med_embed_dim, seed + 1)
        self._text_embedder = text_embedder
        self._diag_dim = diag_embed_dim
        self._med_dim = med_embed_dim
        # 若外部嵌入器维度不同，调整（外部优先）
        if text_embedder is not None:
            try:
                probe = text_embedder.embed("test")
                self._ext_dim = len(probe)
            except Exception:
                self._text_embedder = None
                self._ext_dim = 0
        else:
            self._ext_dim = 0

    @property
    def diag_embed_dim(self) -> int:
        return self._diag_dim if self._text_embedder is None else self._ext_dim

    @property
    def med_embed_dim(self) -> int:
        return self._med_dim if self._text_embedder is None else self._ext_dim

    @property
    def semantic_dim(self) -> int:
        """语义部分总维度（diag + med）。"""
        return self.diag_embed_dim + self.med_embed_dim

    @property
    def has_text_embedder(self) -> bool:
        """是否启用了外部文本嵌入器。"""
        return self._text_embedder is not None

    def embed_diagnoses(self, diagnoses: list[str]) -> tuple[np.ndarray, list[str]]:
        """诊断列表 → 语义嵌入向量。

        Args:
            diagnoses: ICD-10 代码列表（如 ["I48.91", "N17.0"]）。

        Returns:
            (嵌入向量, 概念桶列表)。
        """
        if not diagnoses:
            if self._text_embedder is not None:
                return np.zeros(self._ext_dim), []
            return np.zeros(self._diag_dim), []

        concepts = [icd10_to_concept(d) for d in diagnoses]
        if self._text_embedder is not None:
            # Layer 3: 外部嵌入器（诊断文本拼接）
            text = " ".join(concepts + diagnoses)
            try:
                vec = self._text_embedder.embed(text)
                return np.asarray(vec, dtype=np.float64), concepts
            except Exception:
                pass  # 降级到哈希
        # Layer 2: 哈希投影
        return self._diag_embedder.embed_concepts(concepts), concepts

    def embed_medications(self, medications: list[Medication]) -> np.ndarray:
        """用药列表 → 语义嵌入向量。

        编码：药物类别桶哈希投影 + 剂量归一化加权。

        Args:
            medications: Medication 列表。

        Returns:
            用药嵌入向量。
        """
        if not medications:
            return np.zeros(self.med_embed_dim)

        # 类别桶 + 剂量
        classes = [drug_to_class(m.name) for m in medications]
        if self._text_embedder is not None:
            text = " ".join(classes + [m.name for m in medications])
            try:
                return np.asarray(self._text_embedder.embed(text), dtype=np.float64)
            except Exception:
                pass
        # 哈希投影 + 剂量加权
        # 剂量调制幅度（不二次归一化），使不同剂量产生可区分的嵌入
        base = self._med_embedder.embed_concepts(classes)
        doses = [max(min(m.dose / 10.0, 1.0), 0.0) for m in medications]
        if doses:
            avg_dose = float(np.mean(doses))
            # 剂量作为幅度缩放（保留方向，幅度 = 0.5 + dose）
            # 这样同药不同剂量方向相同但范数不同，可被模型区分
            base = base * (0.5 + avg_dose)
        return base

    def embed(self, state: Any) -> SemanticStateVector:
        """编码 PatientState 为完整语义向量。

        把体征数值（R¹³）+ 诊断嵌入 + 用药嵌入组合成 SemanticStateVector。

        Args:
            state: PatientState（需有 to_vector/diagnoses/medications 属性）。

        Returns:
            SemanticStateVector。
        """
        # 延迟导入避免循环
        from mci_world_model.sdk._clinical_world_state import PatientState

        if not isinstance(state, PatientState):
            raise TypeError(f"需要 PatientState，收到 {type(state)}")

        numeric = state.to_vector()
        diag_vec, concepts = self.embed_diagnoses(state.diagnoses)
        med_vec = self.embed_medications(state.medications)
        return SemanticStateVector(
            numeric=numeric,
            diagnosis_embedding=diag_vec,
            medication_embedding=med_vec,
            concepts=concepts,
        )

    def embed_numeric_only(self, state: Any) -> np.ndarray:
        """仅编码数值部分（向后兼容 to_vector）。"""
        from mci_world_model.sdk._clinical_world_state import PatientState

        if not isinstance(state, PatientState):
            raise TypeError(f"需要 PatientState，收到 {type(state)}")
        return state.to_vector()
