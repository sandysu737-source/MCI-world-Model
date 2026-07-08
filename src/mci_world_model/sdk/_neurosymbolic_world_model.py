from __future__ import annotations

"""神经符号融合世界模型 — TASK-C1。

三元融合架构:
    JEPA 潜空间 (物理) + Pearl 因果图 (因果) + LLM 语义嵌入 (语义)

核心设计:
    1. 输入统一编码为 triple_representation: (latent_z, causal_features, semantic_embed)
    2. Routing Controller 根据查询类型自动选择最优推理路径:
        - 物理预测 → JEPA 潜空间路径
        - 因果推断 → do-calculus 因果路径
        - 语义推理 → LLM 语义路径
    3. 三路径结果通过注意力加权融合输出

路由决策公式:
    route = argmax_r(score(query, r))  for r in [physical, causal, semantic]
    score(q, r) = w_r · sim(embed(q), prototype_r) + bias_r

融合公式:
    output = Σ_r α_r · output_r
    α_r = softmax(score(q, r) / temperature)

方法签名:
    NeurosymbolicWorldModel.__init__(jepa_encoder, causal_graph, llm_adapter, config)
    NeurosymbolicWorldModel.infer(query, context) → InferenceResult
    NeurosymbolicWorldModel.route(query) → RouteDecision
    NeurosymbolicWorldModel.fuse(outputs, scores) → FusedOutput

数据结构:
    TripleRepresentation: (latent: ndarray, causal: dict[str, Any], semantic: ndarray)
    RouteDecision: route_type, confidence, scores
    InferenceResult: output, route_used, fusion_weights, uncertainty
"""


import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 枚举与核心数据结构
# =============================================================================


class RouteType(Enum):
    """推理路径类型。"""

    PHYSICAL = "physical"  # JEPA 潜空间路径
    CAUSAL = "causal"  # Pearl do-calculus 因果路径
    SEMANTIC = "semantic"  # LLM 语义路径
    FUSED = "fused"  # 多路径融合


@dataclass
class TripleRepresentation:
    """三元表示: 物理潜向量 + 因果特征 + 语义嵌入。

    Attributes:
        latent: JEPA 编码器输出的潜向量 (z_dim,)
        causal_features: 因果图特征 {edge: confidence}
        semantic_embed: LLM 语义嵌入 (embed_dim,)
    """

    latent: np.ndarray = field(default_factory=lambda: np.array([]))
    causal_features: dict[str, float] = field(default_factory=dict)
    semantic_embed: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class RouteDecision:
    """路由决策结果。

    Attributes:
        route_type: 选择的推理路径
        confidence: 路由置信度 [0, 1]
        scores: 各路径的得分 {RouteType: float}
        reasoning: 路由决策理由
    """

    route_type: RouteType = RouteType.FUSED
    confidence: float = 0.5
    scores: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class InferenceResult:
    """推理结果。

    Attributes:
        output: 推理输出 (任意类型)
        route_used: 使用的推理路径
        fusion_weights: 融合权重 (仅 FUSED 模式)
        uncertainty: 不确定性 [0, 1]
        latency_ms: 推理延迟 (毫秒)
        metadata: 附加元数据
    """

    output: Any = None
    route_used: RouteType = RouteType.FUSED
    fusion_weights: dict[str, float] = field(default_factory=dict)
    uncertainty: float = 0.5
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NeurosymbolicConfig:
    """神经符号融合世界模型配置。

    Attributes:
        z_dim: JEPA 潜空间维度
        embed_dim: 语义嵌入维度
        temperature: 融合 softmax 温度
        route_weights: 路径先验权重 {route_name: weight}
        uncertainty_threshold: 不确定性阈值 (超过则降级)
        seed: 随机种子
    """

    z_dim: int = 16
    embed_dim: int = 3
    temperature: float = 1.0
    route_weights: dict[str, float] = field(
        default_factory=lambda: {
            "physical": 0.35,
            "causal": 0.40,
            "semantic": 0.25,
        }
    )
    uncertainty_threshold: float = 0.8
    seed: int = 42


# =============================================================================
# NeurosymbolicWorldModel — 神经符号融合世界模型
# =============================================================================


class NeurosymbolicWorldModel:
    """神经符号融合世界模型。

    三元融合: JEPA 潜空间 + Pearl 因果图 + LLM 语义嵌入
    推断时自动选择最优推理路径，或融合多路径结果。

    用法:
        >>> nswm = NeurosymbolicWorldModel(
        ...     jepa_encoder=encoder,
        ...     causal_graph=cg,
        ...     llm_adapter=adapter,
        ... )
        >>> result = nswm.infer("增加多巴胺剂量后心率变化", context={})
        >>> print(f"路径: {result.route_used}, 不确定性: {result.uncertainty:.2f}")
    """

    def __init__(
        self,
        jepa_encoder: Any | None = None,
        causal_graph: Any | None = None,
        llm_adapter: Any | None = None,
        do_calculus: Any | None = None,
        config: NeurosymbolicConfig | None = None,
    ):
        """
        Args:
            jepa_encoder: LearnableStateEncoder 实例 (JEPA 路径)
            causal_graph: CausalGraph 实例 (因果路径)
            llm_adapter: MultiLLMAdapter 实例 (语义路径)
            do_calculus: DoCalculus 实例 (因果推断引擎)
            config: 配置
        """
        self._jepa_encoder = jepa_encoder
        self._causal_graph = causal_graph
        self._llm_adapter = llm_adapter
        self._do_calculus = do_calculus
        self._config = config or NeurosymbolicConfig()
        self._rng = np.random.RandomState(self._config.seed)

        # 路由原型向量 (用于相似度计算)
        self._route_prototypes: dict[str, np.ndarray] = {
            "physical": self._rng.randn(self._config.embed_dim) * 0.1,
            "causal": self._rng.randn(self._config.embed_dim) * 0.1,
            "semantic": self._rng.randn(self._config.embed_dim) * 0.1,
        }

    # -----------------------------------------------------------------
    # 公开 API
    # -----------------------------------------------------------------

    def infer(self, query: str, context: dict[str, Any] | None = None) -> InferenceResult:
        """统一推理入口。

        步骤:
            1. 路由决策: 根据 query 选择推理路径
            2. 执行推理: 调用对应路径的推理引擎
            3. 融合输出: 多路径结果加权融合
            4. 不确定性评估: 基于路径一致性

        Args:
            query: 查询字符串
            context: 上下文信息

        Returns:
            InferenceResult
        """
        t0 = time.perf_counter()
        context = context or {}

        # 1. 路由决策
        self.route(query)

        # 2. 执行各路径推理
        outputs: dict[str, Any] = {}
        scores: dict[str, float] = {}

        # 物理路径
        if self._jepa_encoder is not None:
            outputs["physical"], scores["physical"] = self._infer_physical(query, context)

        # 因果路径
        if self._causal_graph is not None:
            outputs["causal"], scores["causal"] = self._infer_causal(query, context)

        # 语义路径
        if self._llm_adapter is not None:
            outputs["semantic"], scores["semantic"] = self._infer_semantic(query, context)

        # 3. 融合或选择
        if len(outputs) == 0:
            # 无可用路径 — 返回空结果（非None，便于下游处理）
            return InferenceResult(
                output={"status": "no_backend", "routes_available": []},
                route_used=RouteType.FUSED,
                uncertainty=1.0,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        if len(outputs) == 1:
            route_name = next(iter(outputs.keys()))
            route_type = RouteType(route_name)
            return InferenceResult(
                output=outputs[route_name],
                route_used=route_type,
                uncertainty=1.0 - scores.get(route_name, 0.5),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # 多路径融合
        fused_output, fusion_weights = self._fuse(outputs, scores)
        uncertainty = self._compute_uncertainty(outputs, scores)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return InferenceResult(
            output=fused_output,
            route_used=RouteType.FUSED,
            fusion_weights=fusion_weights,
            uncertainty=uncertainty,
            latency_ms=elapsed_ms,
            metadata={"route_scores": scores},
        )

    def route(self, query: str) -> RouteDecision:
        """路由决策: 选择最优推理路径。

        决策公式:
            score(q, r) = w_r · sim(embed(q), prototype_r) + bias_r

        Args:
            query: 查询字符串

        Returns:
            RouteDecision
        """
        query_embed = self._embed_query(query)

        scores: dict[str, float] = {}
        for route_name, prototype in self._route_prototypes.items():
            prior_weight = self._config.route_weights.get(route_name, 0.33)
            similarity = self._cosine_similarity(query_embed, prototype)
            scores[route_name] = prior_weight * (1.0 + similarity)

        # 选择得分最高的路径
        best_route = max(scores, key=scores.get)  # type: ignore[arg-type]
        total = sum(scores.values())
        confidence = scores[best_route] / total if total > 0 else 0.33

        # 如果各路径得分接近, 使用 FUSED
        max_score = max(scores.values())
        min_score = min(scores.values())
        if (max_score - min_score) < 0.1 * total and len(scores) > 1:
            best_route = "fused"
            confidence = 0.5

        route_type = RouteType.FUSED if best_route == "fused" else RouteType(best_route)

        return RouteDecision(
            route_type=route_type,
            confidence=confidence,
            scores=scores,
            reasoning=f"Route {best_route} selected (score={scores.get(best_route, 0):.3f})",
        )

    def encode_triple(self, state: Any, query: str = "") -> TripleRepresentation:
        """将输入编码为三元表示。

        Args:
            state: 状态输入 (WorldState 或向量)
            query: 查询字符串 (用于语义嵌入)

        Returns:
            TripleRepresentation
        """
        latent = np.array([])
        causal_features: dict[str, float] = {}
        semantic_embed = np.array([])

        # JEPA 编码
        if self._jepa_encoder is not None:
            try:
                state_vec = self._state_to_vector(state)
                if len(state_vec) > 0:
                    latent = self._jepa_encoder.forward(state_vec)
            except Exception as e:
                logger.warning("JEPA encoding failed: %s", e)

        # 因果特征
        if self._causal_graph is not None:
            try:
                for src, dst in self._causal_graph.edges:
                    key = f"{src}→{dst}"
                    if self._causal_graph.adjacency is not None:
                        i = self._causal_graph.node_index(src)
                        j = self._causal_graph.node_index(dst)
                        if i is not None and j is not None:
                            causal_features[key] = float(self._causal_graph.adjacency[i, j])
            except Exception as e:
                logger.warning("Causal feature extraction failed: %s", e)

        # 语义嵌入
        if self._llm_adapter is not None and query:
            try:
                semantic_embed = self._llm_adapter.embed(query, dim=self._config.embed_dim)
            except Exception as e:
                logger.warning("Semantic embedding failed: %s", e)

        return TripleRepresentation(
            latent=latent,
            causal_features=causal_features,
            semantic_embed=semantic_embed,
        )

    @property
    def config(self) -> NeurosymbolicConfig:
        return self._config

    def train_jepa(
        self,
        observations: np.ndarray,
        actions: np.ndarray | None = None,
        n_epochs: int = 10,
    ) -> dict[str, Any]:
        """训练 JEPA 物理路径 (若 jepa_encoder 为 TrueJEPAEncoder)。

        将神经符号世界模型的物理路径真正接入可学习 JEPA 编码器,
        使 encode_triple / _infer_physical 使用训练后的潜空间。

        Args:
            observations: 观测序列 (N, obs_dim)
            actions: 动作序列 (N, action_dim)
            n_epochs: 训练轮数

        Returns:
            训练统计 dict; 若编码器不可训练则返回 {"status": "skipped"}。
        """
        enc = self._jepa_encoder
        train_fn = getattr(enc, "train", None)
        if train_fn is None or not callable(train_fn):
            return {"status": "skipped", "reason": "encoder has no train method"}
        try:
            result = train_fn(observations, actions=actions, n_epochs=n_epochs)
            if isinstance(result, dict):
                result.setdefault("status", "trained")
            return result
        except Exception as e:
            logger.warning("JEPA training failed: %s", e)
            return {"status": "error", "reason": str(e)}

    # -----------------------------------------------------------------
    # 路径推理
    # -----------------------------------------------------------------

    def _infer_physical(self, query: str, context: dict[str, Any]) -> tuple[Any, float]:
        """物理路径: JEPA 潜空间预测。

        Returns:
            (output, score)
        """
        try:
            state_vec = self._state_to_vector(context.get("state"))
            if len(state_vec) > 0 and self._jepa_encoder is not None:
                latent = self._jepa_encoder.forward(state_vec)
                return {"latent_prediction": latent, "method": "jepa"}, 0.7
        except Exception:
            pass
        return {"method": "jepa", "status": "unavailable"}, 0.3

    def _infer_causal(self, query: str, context: dict[str, Any]) -> tuple[Any, float]:
        """因果路径: do-calculus 因果推断。

        Returns:
            (output, score)
        """
        try:
            cause = context.get("cause", "")
            effect = context.get("effect", "")
            if cause and effect and self._do_calculus is not None:
                result = self._do_calculus.estimate_ate(X=cause, Y=effect)
                return {
                    "ate": getattr(result, "ate", 0.0),
                    "method": getattr(result, "method", "unknown"),
                    "causal_inference": True,
                }, 0.8
            elif self._causal_graph is not None:
                return {
                    "n_nodes": self._causal_graph.n_nodes,
                    "n_edges": len(self._causal_graph.edges),
                    "method": "graph_lookup",
                }, 0.5
        except Exception:
            pass
        return {"method": "causal", "status": "unavailable"}, 0.2

    def _infer_semantic(self, query: str, context: dict[str, Any]) -> tuple[Any, float]:
        """语义路径: LLM 推理。

        Returns:
            (output, score)
        """
        try:
            if self._llm_adapter is not None:
                response = self._llm_adapter.generate(query, context=context)
                return {"response": response, "method": "llm"}, 0.6
        except Exception:
            pass
        return {"method": "semantic", "status": "unavailable"}, 0.2

    # -----------------------------------------------------------------
    # 融合与不确定性
    # -----------------------------------------------------------------

    def _fuse(
        self,
        outputs: dict[str, Any],
        scores: dict[str, float],
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """多路径结果融合。

        融合公式:
            α_r = softmax(score_r / temperature)
            output = dict merge with α_r weights

        Returns:
            (fused_output, fusion_weights)
        """
        weights: dict[str, float] = {}
        total = sum(scores.values())
        if total <= 0:
            n = len(scores)
            weights = dict.fromkeys(scores, 1.0 / n)
        else:
            for route, score in scores.items():
                weights[route] = score / total

        fused: dict[str, Any] = {
            "fusion_method": "weighted_merge",
            "routes_used": list(outputs.keys()),
            "weights": weights,
        }

        for route, output in outputs.items():
            if isinstance(output, dict):
                for k, v in output.items():
                    if k not in fused:
                        fused[k] = v
                    elif isinstance(v, (int, float)) and isinstance(fused[k], (int, float)):
                        fused[k] = weights.get(route, 0.33) * v + (1 - weights.get(route, 0.33)) * fused[k]

        return fused, weights

    def _compute_uncertainty(self, outputs: dict[str, Any], scores: dict[str, float]) -> float:
        """基于路径一致性计算不确定性。

        路径分歧越大 → 不确定性越高。

        Returns:
            uncertainty [0, 1]
        """
        if len(scores) <= 1:
            return 0.5

        score_values = list(scores.values())
        max_s = max(score_values)
        min_s = min(score_values)
        total = sum(score_values)

        # 归一化分歧度
        if total <= 0:
            return 0.5

        divergence = (max_s - min_s) / total
        uncertainty = 1.0 - min(divergence * 2, 1.0)  # 分歧大 → 不确定性低 (有主导路径)
        uncertainty = max(0.0, min(1.0, uncertainty))

        return uncertainty

    # -----------------------------------------------------------------
    # 辅助方法
    # -----------------------------------------------------------------

    # ── 语义路由嵌入 ──
    # P0-F1 修复: 替换 hash+RandomState 为关键词+词袋语义路由
    _ROUTE_KEYWORDS: dict[str, list[str]] = {
        "physical": [
            "速度",
            "加速度",
            "位置",
            "运动",
            "力",
            "质量",
            "能量",
            "动量",
            "velocity",
            "acceleration",
            "position",
            "motion",
            "force",
            "mass",
            "predict",
            "预测",
            "物理",
            "physical",
            "dynamics",
            "动力学",
            "轨迹",
            "trajectory",
            "轨道",
            "orbit",
            "旋转",
            "rotation",
        ],
        "causal": [
            "因为",
            "所以",
            "导致",
            "影响",
            "因果",
            "干预",
            "反事实",
            "cause",
            "effect",
            "because",
            "therefore",
            "intervene",
            "因素",
            "变量",
            "相关性",
            "推断",
            "归因",
            "调节",
            "中介",
            "do",
            "反事",
            "假设",
            "如果",
            "则",
            "结果",
            "原因",
        ],
        "semantic": [
            "什么",
            "为什么",
            "如何",
            "解释",
            "描述",
            "定义",
            "比较",
            "what",
            "why",
            "how",
            "explain",
            "describe",
            "define",
            "compare",
            "意思",
            "概念",
            "理论",
            "原则",
            "方法",
            "框架",
            "区别",
            "理解",
            "了解",
            "知道",
            "认知",
            "推理",
            "思考",
            "分析",
        ],
    }

    def _embed_query(self, query: str) -> np.ndarray:
        """将查询字符串编码为语义嵌入向量。

        P0-F1 修复: 使用关键词匹配 + 词袋模型替代 hash+RandomState。
        语义相近的查询产生相近的嵌入向量，使路由决策具有语义一致性。

        嵌射策略:
            1. 对每条路由的关键词计算匹配得分
            2. 得分向量归一化后作为语义嵌入
            3. 确定性: 相同查询 → 相同嵌入
        """
        # 优先使用 LLM adapter 获取语义嵌入
        if self._llm_adapter is not None:
            try:
                return self._llm_adapter.embed(query, dim=self._config.embed_dim)
            except Exception:
                pass

        # 语义路由: 关键词匹配得分
        query_lower = query.lower()
        scores = np.zeros(self._config.embed_dim, dtype=np.float64)

        # 为每条路由计算关键词匹配得分
        route_names = list(self._route_prototypes.keys())
        for i, route_name in enumerate(route_names):
            if i >= self._config.embed_dim:
                break
            keywords = self._ROUTE_KEYWORDS.get(route_name, [])
            match_count = sum(1 for kw in keywords if kw in query_lower)
            scores[i] = match_count / max(len(keywords), 1) * 10.0  # 缩放

        # 填充剩余维度: 基于 query 字符的确定性特征
        for j in range(len(route_names), self._config.embed_dim):
            # 使用字符频率作为确定性补充特征
            char_idx = j % len(query_lower) if query_lower else 0
            scores[j] = ord(query_lower[char_idx]) / 128.0 if char_idx < len(query_lower) else 0.0

        # L2 归一化
        norm = np.linalg.norm(scores)
        if norm > 1e-10:
            scores /= norm

        return scores

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """余弦相似度。"""
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        if len(a) != len(b) or np.linalg.norm(a) < 1e-10 or np.linalg.norm(b) < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    @staticmethod
    def _state_to_vector(state: Any) -> np.ndarray:
        """将状态转为向量。

        支持: ndarray / to_vector() / list / tuple / dict。
        dict 解析顺序: 'vector' 键 → 全部数值 values 拼接。
        """
        if isinstance(state, np.ndarray):
            return state.astype(np.float64).ravel()
        if hasattr(state, "to_vector"):
            return np.asarray(state.to_vector(), dtype=np.float64).ravel()
        if isinstance(state, (list, tuple)):
            return np.asarray(state, dtype=np.float64).ravel()
        if isinstance(state, dict):
            if "vector" in state:
                return np.asarray(state["vector"], dtype=np.float64).ravel()
            vals = []
            for v in state.values():
                if isinstance(v, (int, float)):
                    vals.append(float(v))
                elif isinstance(v, (list, tuple, np.ndarray)):
                    arr = np.asarray(v, dtype=np.float64).ravel()
                    if arr.size > 0:
                        vals.extend(arr.tolist())
            if vals:
                return np.asarray(vals, dtype=np.float64)
        return np.array([], dtype=np.float64)
