"""世界模型预测与 JEPA 预测器组件边界。

本模块只承载因果检索增强预测、JEPA 潜空间预测、M3 记忆预测、
融合预测、JEPA 训练与预测器接入/升级，不承载健康诊断或世界模型编排。
"""

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)


class PredictionComponentsMixin:
    """预测与 JEPA 预测器行为 Mixin。

    宿主类必须提供世界模型状态、记忆源、能量流解析与预测组件槽位。
    """

    if TYPE_CHECKING:
        _state: Any
        _lite_pro: Any
        _energy_core: Any | None
        _jepa_encoder: Any | None
        _jepa_predictor: Any | None
        _jepa_mode: str

        def _get_memories_from_lite_pro(self) -> list[dict[str, Any]]: ...
        def _extract_energy_ratios(self, state: Any) -> dict[str, float] | None: ...

    def attach_true_jepa(self, encoder: Any, predictor: Any | None = None) -> None:
        """显式注入 TrueJEPA 编码器，接入潜空间主闭环。"""
        self._jepa_mode = "latent"
        from mci_world_model.sdk._jepa_encoder import JEPAEncoder

        if self._jepa_encoder is None:
            self._jepa_encoder = JEPAEncoder(self, jepa_mode="latent")
        self._jepa_encoder.attach_true_jepa(encoder)
        if predictor is not None:
            self._jepa_predictor = predictor

    def predict_effect(
        self,
        cause: str,
        memories: list[dict[str, Any]] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        检索路径因果预测（v3.5.0 能力）。

        基于 CausalEngine 关键词 + 偏相关统计，
        不依赖参数化模型。

        Args:
            cause: 原因文本
            memories: 记忆列表
            top_k: 返回前 K 个效应

        Returns:
            [{"effect": str, "confidence": float, "causal_type": str}, ...]
        """
        if memories is None and self._lite_pro is not None:
            memories = self._get_memories_from_lite_pro()

        if not memories:
            return []

        try:
            from mci_world_model.sdk._causal import CausalEngine

            engine = CausalEngine(min_confidence=0.4)
            effects = engine.predict_effects(cause, memories, top_k=top_k)
            return effects
        except (KeyError, ValueError, RuntimeError) as e:
            logger.error("检索预测失败: %s", e)
            return []

    def jepa_predict(
        self,
        cause: str,
        target_category: str | None = None,
        top_k: int = 3,
        memories: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        JEPA 潜空间因果预测（v3.1.0）。

        流程: 编码器(记忆 → 因果图状态) → 预测器(状态 → 下一状态) →
              差分分析(原因 → 效应)

        Args:
            cause: 原因文本
            target_category: 目标状态类别（可选）
            top_k: 返回前 K 个预测
            memories: 记忆列表（None 时从 lite_pro 获取）

        Returns:
            [{"effect": str, "confidence": float, "energy_relation": str}, ...]
        """
        if self._jepa_encoder is None or self._jepa_predictor is None:
            logger.warning("JEPA 编码器/预测器未初始化，回退到检索路径")
            return self.predict_effect(cause, top_k=top_k)

        # ── 获取记忆并编码 ──
        if memories is None and self._lite_pro is not None:
            memories = self._get_memories_from_lite_pro()

        if not memories or len(memories) < 3:
            logger.warning("记忆不足 JEPA 预测（需要 ≥ 3 条）")
            return self.predict_effect(cause, top_k=top_k)

        try:
            # 0. 检测 M3 模式
            is_m3 = (
                self._jepa_encoder is not None
                and hasattr(self._jepa_encoder, "_differentiable")
                and self._jepa_encoder._differentiable
            )

            # 1. 编码: latent 模式只接受潜向量，legacy_graph 是显式过渡。
            encoded = self._jepa_encoder.encode(memories)
            if getattr(encoded, "source", None) == "legacy_graph":
                state = encoded.graph_state
            else:
                if not getattr(encoded, "is_ready", False):
                    logger.warning("JEPA latent 编码未就绪: %s", encoded.diagnostics)
                    return self.predict_effect(cause, top_k=top_k)
                return []

            # 2. 预测: 状态 → 下一状态 (GNN)
            next_state = self._jepa_predictor.predict(state)

            # 3. 差分: 找出新增/增强的因果边
            predictions = []
            if state.causal_edges and next_state.causal_edges:
                current_edge_keys = {(e.get("cause", ""), e.get("effect", "")) for e in state.causal_edges}
                for edge in next_state.causal_edges:
                    ee = edge.get("effect", "")
                    ec = edge.get("cause", "")
                    # 只返回与 cause 相关的新增/变化边
                    if cause.lower() in ec.lower() or cause.lower() in ee.lower():
                        key = (ec, ee)
                        is_new = key not in current_edge_keys
                        predictions.append(
                            {
                                "effect": ee,
                                "confidence": edge.get("confidence", 0.5) * (1.1 if is_new else 0.9),
                                "energy_relation": edge.get("energy_relation", "neutral"),
                                "cause": ec,
                                "verdict": edge.get("verdict", "predicted"),
                                "_mode": "m3_gat_gnn" if is_m3 else "jepa_baseline",
                            }
                        )

            # 按置信度排序
            predictions.sort(key=lambda x: x["confidence"], reverse=True)

            # ── v3.0.5: 预测后能量守恒验证 ──
            if self._energy_core is not None:
                try:
                    energy_before = self._extract_energy_ratios(state)
                    energy_after = self._extract_energy_ratios(next_state)
                    if energy_before and energy_after:
                        simulated = self._energy_core.simulate_energy_flow(energy_after, steps=3)
                        # 检测能量是否收敛到合理范围（每维偏离 0.2 上限不超过 0.3）
                        final = simulated[-1]
                        max_deviation = max(abs(final.get(k, 0) - 0.2) for k in final)
                        if max_deviation > 0.3:
                            # 能量不守恒 → 降低全部预测置信度
                            for p in predictions:
                                p["confidence"] *= 0.7
                            logger.debug(
                                "JEPA 预测能量不守恒 (max_deviation=%.3f)，置信度降权",
                                max_deviation,
                            )
                except (ValueError, AttributeError) as e:
                    logger.warning("能量守恒验证跳过: %s", e)

            return predictions[:top_k] if predictions else self.predict_effect(cause, top_k=top_k)

        except (RuntimeError, ValueError, KeyError) as e:
            logger.error("JEPA 预测失败: %s，回退到检索路径", e)
            return self.predict_effect(cause, top_k=top_k)

    def parametric_predict(
        self,
        cause: str,
        target_category: str | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        参数化路径因果预测（v3.6.0 — v3.1.0 降级为 jepa_predict 别名）。

        v3.1.0: 重路由到 JEPA 潜空间预测。
        保留接口兼容性，内部调用 jepa_predict()。

        Args:
            cause: 原因文本
            target_category: 目标状态类别（可选）
            top_k: 返回前 K 个预测

        Returns:
            [{"effect": str, "confidence": float, "energy_relation": str}, ...]
        """
        return self.jepa_predict(cause, target_category, top_k)

    def predict_from_memories_m3(
        self,
        memories: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        M3 专用推理：从记忆列表直接预测因果边（GAT + GNN）。

        与 jepa_predict() 的区别:
        - 不依赖 cause 文本过滤，返回所有预测的因果边
        - 直接输出 GNN 预测的 (cause, effect, rho) 三元组
        - 可用于端到端评估和可视化

        Args:
            memories: 记忆列表（至少 3 条）
            top_k: 返回前 K 条最显著的因果边

        Returns:
            [{"cause": str, "effect": str, "rho": float, "confidence": float}, ...]
        """
        if self._jepa_encoder is None or self._jepa_predictor is None:
            logger.warning("JEPA 编码器/预测器未初始化")
            return []

        if not memories or len(memories) < 3:
            logger.warning("记忆不足 M3 预测（需要 ≥ 3 条）")
            return []

        try:
            # 1. GAT 编码
            state = self._jepa_encoder.encode_graph(memories)

            if not state.causal_edges:
                logger.info("GAT 编码未发现因果边")
                return []

            # 2. GNN 预测下一状态
            next_state = self._jepa_predictor.predict(state)

            # 3. 提取预测边
            predictions = []
            for edge in next_state.causal_edges:
                predictions.append(
                    {
                        "cause": edge.get("cause", ""),
                        "effect": edge.get("effect", ""),
                        "rho": edge.get("rho", 0.0),
                        "confidence": edge.get("confidence", 0.5),
                        "verdict": edge.get("verdict", "predicted"),
                        "energy_relation": edge.get("energy_relation", "neutral"),
                    }
                )

            predictions.sort(key=lambda x: abs(x["rho"]), reverse=True)
            return predictions[:top_k]

        except (RuntimeError, ValueError, AttributeError) as e:
            logger.error("M3 预测失败: %s", e)
            return []

    def fused_predict(
        self,
        cause: str,
        memories: list[dict[str, Any]] | None = None,
        top_k: int = 5,
        retrieval_weight: float = 0.4,
        parametric_weight: float = 0.6,
    ) -> list[dict[str, Any]]:
        """
        融合预测（v3.1.0: 检索 + JEPA 加权）。

        v3.1.0: 将"参数化"路径替换为 JEPA 潜空间预测。
        parametric_weight 参数保留但语义变为 JEPA 预测权重。

        Args:
            cause: 原因文本
            memories: 记忆列表
            top_k: 返回数量
            retrieval_weight: 检索路径权重
            parametric_weight: JEPA 预测路径权重

        Returns:
            加权融合后的预测列表
        """
        retrieval_results = self.predict_effect(cause, memories, top_k=top_k)
        jepa_results = self.jepa_predict(cause, top_k=top_k, memories=memories)

        # 融合策略: JEPA 结果在前，检索结果补充
        fused = []
        seen_effects: set[str] = set()

        for r in jepa_results:
            effect_key = r.get("effect", "")
            if effect_key not in seen_effects:
                seen_effects.add(effect_key)
                fused.append(
                    {
                        "effect": effect_key,
                        "confidence": r.get("confidence", 0.5) * parametric_weight,
                        "source": "jepa",
                        "energy_relation": r.get("energy_relation", "neutral"),
                    }
                )

        for r in retrieval_results:
            content = r.get("content", "")
            if content not in seen_effects:
                seen_effects.add(content)
                fused.append(
                    {
                        "effect": content,
                        "confidence": r.get("confidence", 0.5) * retrieval_weight,
                        "source": "retrieval",
                        "causal_type": r.get("causal_type", ""),
                    }
                )

        fused.sort(key=lambda x: x["confidence"], reverse=True)
        return fused[:top_k]

    def train_jepa(
        self,
        dataset: object | None = None,
        qa_pairs: list | None = None,  # type: ignore
        output_dir: str = "./checkpoints/mci-world-model",
        n_epochs: int = 10,
        learning_rate: float = 0.01,
    ) -> dict[str, Any]:
        """
        JEPA 端到端训练（v3.1.0，替代 train_parametric）。

        Args:
            dataset: JEPADataset 实例（优先）
            qa_pairs: Reflection QA 对列表（备选，自动转 JEPADataset）
            output_dir: checkpoint 输出目录
            n_epochs: 训练轮数
            learning_rate: 学习率

        Returns:
            训练统计
        """
        # ── 构造 JEPADataset ──
        if dataset is None and qa_pairs is not None:
            try:
                from mci_world_model.sdk._jepa_dataset import JEPADataset

                # 从 QA 对提取记忆并构造数据集
                memories = []
                for qa in qa_pairs:
                    if isinstance(qa, dict):
                        memories.append(
                            {
                                "content": qa.get("question", ""),
                                "answer": qa.get("answer", ""),
                            }
                        )
                if memories:
                    dataset = JEPADataset.from_memories(memories, self)  # type: ignore
            except ImportError as e:
                return {"error": f"jepa_dataset_unavailable: {e}"}

        if dataset is None:
            return {
                "error": "no_training_data",
                "message": "需要 JEPADataset 或 qa_pairs",
            }

        # ── 构造训练器并训练 ──
        try:
            from mci_world_model.sdk._jepa_trainer import JEPATrainer

            trainer = JEPATrainer(
                encoder=self._jepa_encoder,
                predictor=self._jepa_predictor,
                dataset=dataset,
            )
            stats = trainer.train(
                n_epochs=n_epochs,
                learning_rate=learning_rate,
            )

            self._state.n_qa_pairs = len(dataset.pairs) if hasattr(dataset, "pairs") else 0

            return {
                "n_pairs": len(dataset.pairs) if hasattr(dataset, "pairs") else 0,
                "n_epochs": n_epochs,
                "training_stats": stats.to_dict() if hasattr(stats, "to_dict") else stats,
                "adapter_path": output_dir,
                "mode": "e2e" if trainer._is_e2e else ("gnn" if trainer._is_gnn else "baseline"),
            }
        except ImportError as e:
            return {"error": f"jepa_trainer_unavailable: {e}"}
        except (RuntimeError, ValueError, KeyError) as e:
            logger.error("JEPA 训练失败: %s", e)
            return {"error": str(e)}

    def _is_gnn_predictor(self) -> bool:
        """检测是否使用可微 GNN 预测器。"""
        if self._jepa_predictor is None:
            return False
        return hasattr(self._jepa_predictor, "training_predict")

    def _is_e2e_mode(self) -> bool:
        """检测是否启用 M3 端到端可微模式。"""
        if self._jepa_encoder is None or self._jepa_predictor is None:
            return False
        return hasattr(self._jepa_encoder, "training_encode") and self._is_gnn_predictor()

    def enable_m3(
        self,
        encoder_key_dim: int = 16,
        predictor_hidden_dim: int = 16,
    ) -> dict[str, Any]:
        """
        启用 M3 端到端可微训练模式。

        安装 GAT 编码器 + GNN 预测器，替代基线预测器。
        调用后，train_jepa() 走 e2e 训练路径。

        Args:
            encoder_key_dim: GAT 注意力键维度
            predictor_hidden_dim: GNN 隐层维度

        Returns:
            状态报告
        """
        report = {"encoder": "unchanged", "predictor": "unchanged"}

        # ── GAT 编码器 ──
        if self._jepa_encoder is not None:
            try:
                from mci_world_model.sdk._jepa_encoder import JEPAEncoder

                encoder = JEPAEncoder(self, differentiable=True, gat_key_dim=encoder_key_dim)
                self._jepa_encoder = encoder
                report["encoder"] = "gat_encoder_initialized"
                logger.info("M3 GAT 编码器已安装 (key_dim=%d)", encoder_key_dim)
            except (TypeError, ValueError, ImportError) as e:
                report["encoder"] = f"failed: {e}"
                logger.warning("M3 GAT 编码器失败: %s", e)

        # ── GNN 预测器 ──
        try:
            from mci_world_model.sdk._jepa_gnn import GNNPredictor

            self._jepa_predictor = GNNPredictor(hidden_dim=predictor_hidden_dim)
            report["predictor"] = "gnn_predictor_installed"
            logger.info("M3 GNN 预测器已安装 (hidden_dim=%d)", predictor_hidden_dim)
        except (TypeError, ValueError, ImportError) as e:
            report["predictor"] = f"failed: {e}"
            logger.warning("M3 GNN 预测器失败: %s", e)

        return report
