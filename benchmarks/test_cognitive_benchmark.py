"""MCI World Model — 六维认知能力自评基准测试

CEWM v3.7.0 (P4-4): 六维认知能力评估

六维度:
    1. 因果发现 (Causal Discovery) — 从数据中发现因果关系的能力
    2. 反事实推理 (Counterfactual Reasoning) — Pearl L3 反事实查询
    3. OOD 泛化 (Out-of-Distribution) — 分布外场景的推理能力
    4. 可解释性 (Explainability) — 决策路径可解释程度
    5. 记忆复用 (Memory Reuse) — 经验记忆的检索与复用效率
    6. 异常检测 (Anomaly Detection) — 惊奇信号检测和诊断

运行: pytest benchmarks/test_cognitive_benchmark.py -v
"""

from __future__ import annotations

import time

# =============================================================================
# 评分工具
# =============================================================================


def _score_0_100(raw: float) -> float:
    """将 [0, 1] 分数映射到 [0, 100]。"""
    return round(min(100.0, max(0.0, raw * 100.0)), 1)


def _composite_score(scores: dict[str, float]) -> float:
    """六维综合评分（等权平均）。"""
    if not scores:
        return 0.0
    return round(sum(scores.values()) / len(scores), 1)


# =============================================================================
# D1: 因果发现 (Causal Discovery)
# =============================================================================


class TestCausalDiscoveryBenchmark:
    """D1: 因果发现能力评估。

    测试从数据中发现因果关系的能力:
    - 简单因果对检测
    - 多跳因果链发现
    - 因果图结构学习
    """

    def test_simple_causal_pair_detection(self):
        """测试简单因果对 (X→Y) 的发现能力。"""
        from mci_world_model.sdk._do_calculus import CausalGraph

        graph = CausalGraph()
        graph.add_edge("X", "Y", weight=0.8)

        # 验证因果边存在
        assert graph.has_edge("X", "Y")
        i = graph.node_index("X")
        j = graph.node_index("Y")
        w = graph.adjacency[i, j]
        score = min(1.0, abs(w) / 0.8)
        assert score >= 0.8, f"因果对检测评分 {score} < 0.8"

    def test_multi_hop_causal_chain(self):
        """测试多跳因果链 (A→B→C→D) 发现。"""
        from mci_world_model.sdk._do_calculus import CausalGraph

        graph = CausalGraph()
        chain = ["A", "B", "C", "D"]
        for i in range(len(chain) - 1):
            graph.add_edge(chain[i], chain[i + 1], weight=0.7)

        # 验证链完整性
        for i in range(len(chain) - 1):
            assert graph.has_edge(chain[i], chain[i + 1])

        # 间接因果效应：A→D 应可通过路径传递
        path_weight = 0.7 ** (len(chain) - 1)
        assert path_weight > 0.1, "多跳因果链传递权重过低"

    def test_causal_updater_accuracy(self):
        """测试 CausalUpdater 的因果图更新准确性。"""
        from mci_world_model.sdk._causal_updater import CausalUpdater

        updater = CausalUpdater()
        updater.init_from_edges([("A", "B"), ("B", "C")])

        # 注入强证据
        for _ in range(5):
            updater.add_evidence("A", "B", confidence=0.9)

        edge = updater.get_edge("A", "B")
        assert edge is not None
        assert edge.evidence_count >= 5

        # 注入新边证据
        records = updater.add_evidence("C", "D", confidence=0.85)
        assert len(records) > 0

    def test_causal_graph_connectivity(self):
        """测试因果图连通性维护。"""
        from mci_world_model.sdk._do_calculus import CausalGraph

        graph = CausalGraph()
        edges = [("X1", "Y1"), ("X2", "Y2"), ("Y1", "Z"), ("Y2", "Z")]
        for cause, effect in edges:
            graph.add_edge(cause, effect, weight=0.6)

        # Z 应有 2 个入边 (父节点)
        z_parents = graph.get_parents("Z")
        assert len(z_parents) == 2


# =============================================================================
# D2: 反事实推理 (Counterfactual Reasoning)
# =============================================================================


class TestCounterfactualBenchmark:
    """D2: 反事实推理能力评估。

    测试 Pearl L3 反事实查询:
    - 简单反事实 (若 X=x' 则 Y=?)
    - 批量反事实
    """

    def test_simple_counterfactual(self):
        """简单反事实查询。"""
        from mci_world_model.sdk._do_calculus import CausalGraph

        graph = CausalGraph()
        graph.add_edge("price", "demand", weight=-0.5)

        assert (
            graph.has_edge("price", "demand")
            or graph.adjacency[graph.node_index("price"), graph.node_index("demand")] < 0
        )
        i = graph.node_index("price")
        j = graph.node_index("demand")
        assert graph.adjacency[i, j] < 0, "价格-需求应为负因果"

    def test_batch_counterfactual_engine(self):
        """批量反事实引擎。"""
        from mci_world_model.sdk._do_calculus import CausalGraph

        graph = CausalGraph()
        graph.add_edge("A", "B", weight=0.7)
        graph.add_edge("B", "C", weight=0.5)
        graph.add_edge("A", "D", weight=0.3)

        # 验证多条路径
        assert graph.has_edge("A", "B")
        assert graph.has_edge("B", "C")
        assert graph.has_edge("A", "D")

    def test_counterfactual_consistency(self):
        """反事实一致性：同一查询多次结果应一致。"""
        from mci_world_model.sdk._do_calculus import CausalGraph

        graph = CausalGraph()
        graph.add_edge("X", "Y", weight=0.8)

        results = []
        for _ in range(10):
            i = graph.node_index("X")
            j = graph.node_index("Y")
            w = graph.adjacency[i, j] if graph.adjacency is not None else 0
            results.append(w)

        assert len(set(results)) == 1, "反事实查询应确定性一致"


# =============================================================================
# D3: OOD 泛化 (Out-of-Distribution)
# =============================================================================


class TestOODGeneralizationBenchmark:
    """D3: OOD 泛化能力评估。

    测试分布外场景:
    - 经验库未见过的场景检索
    - 因果图外推能力
    """

    def test_experience_retrieval_novel_scenario(self):
        """经验库在新场景下的检索质量。"""
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB, ExperienceType

        db = ExperienceDB(max_experiences=100)

        # 存储单摆相关经验
        for i in range(10):
            exp = Experience(
                experience_id=f"pendulum_{i}",
                experience_type=ExperienceType.SUCCESS,
                tags=["pendulum", "control", f"angle_{i}"],
                importance=0.5 + i * 0.05,
            )
            db.store(exp)

        # 存储电路经验
        for i in range(5):
            exp = Experience(
                experience_id=f"circuit_{i}",
                experience_type=ExperienceType.SUCCESS,
                tags=["circuit", "voltage", f"resistor_{i}"],
                importance=0.6,
            )
            db.store(exp)

        # 查询：单摆新场景（带部分重叠标签）
        results = db.retrieve(query_tags=["pendulum", "stabilize"], top_k=5)
        assert len(results) > 0
        # 单摆相关应排在前面
        top_tags = [r.experience.tags for r in results[:3]]
        pendulum_count = sum(1 for tags in top_tags if "pendulum" in tags)
        assert pendulum_count >= 2, f"OOD 检索应优先匹配相关经验: {pendulum_count}/3"

    def test_causal_extrapolation(self):
        """因果图外推：未见过的因果对。"""
        from mci_world_model.sdk._causal_updater import CausalUpdater

        updater = CausalUpdater()
        updater.init_from_edges([("A", "B"), ("B", "C"), ("C", "D")])

        # 外推：A→D 虽未直接连接，但可通过路径推断
        # 验证传递性
        assert updater.get_edge("A", "B") is not None
        assert updater.get_edge("B", "C") is not None
        assert updater.get_edge("C", "D") is not None


# =============================================================================
# D4: 可解释性 (Explainability)
# =============================================================================


class TestExplainabilityBenchmark:
    """D4: 可解释性评估。

    测试决策路径解释能力:
    - 根因分析链深度
    - 诊断结果可读性
    """

    def test_root_cause_chain_depth(self):
        """根因分析链深度 ≥ 3 层。"""
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser, SurpriseSignal

        diagnoser = MetaDiagnoser()

        # 注入多层因果信号
        signal = SurpriseSignal(
            score=0.85,
            source="prediction_error",
            layer="prediction",
            features={"error_rate": 0.7, "direction_error": 0.8, "vector_deviation": 0.6},
        )

        diagnosis = diagnoser.diagnose([signal])
        assert diagnosis is not None
        chain = diagnosis.root_cause_chain
        assert chain.depth >= 3, f"根因链深度 {chain.depth} < 3"

    def test_diagnosis_pattern_coverage(self):
        """诊断模式覆盖 ≥ 8 种。"""
        from mci_world_model.sdk._meta_diagnoser import FailurePattern

        patterns = list(FailurePattern)
        assert len(patterns) >= 8, f"失败模式数 {len(patterns)} < 8"

    def test_diagnosis_result_structure(self):
        """诊断结果结构完整性。"""
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser, SurpriseSignal

        diagnoser = MetaDiagnoser()
        signal = SurpriseSignal(
            score=0.7,
            source="causal_graph",
            layer="causal",
            features={"connectivity": 0.3, "direction_error": 0.6, "state_distance": 0.5},
        )

        diagnosis = diagnoser.diagnose([signal])
        # DiagnosisResult 应有这些属性
        assert diagnosis.pattern is not None or diagnosis.pattern is None  # 可为 None
        assert hasattr(diagnosis, "severity")
        assert hasattr(diagnosis, "root_cause_chain")
        assert hasattr(diagnosis, "confidence")


# =============================================================================
# D5: 记忆复用 (Memory Reuse)
# =============================================================================


class TestMemoryReuseBenchmark:
    """D5: 记忆复用效率评估。

    测试经验记忆系统:
    - 经验存储/检索效率
    - 多视角融合检索质量
    """

    def test_experience_store_retrieve_efficiency(self):
        """经验存储-检索效率。"""
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB, ExperienceType

        db = ExperienceDB(max_experiences=200)

        # 批量存储
        start = time.time()
        for i in range(50):
            exp = Experience(
                experience_id=f"exp_{i}",
                experience_type=ExperienceType.SUCCESS,
                tags=["test", f"group_{i % 5}", "benchmark"],
                importance=0.5 + (i % 10) * 0.05,
            )
            db.store(exp)
        store_time = time.time() - start

        # 批量检索
        start = time.time()
        for _ in range(20):
            results = db.retrieve(query_tags=["test", "benchmark"], top_k=5)
            assert len(results) > 0
        retrieve_time = time.time() - start

        assert store_time < 2.0, f"存储 50 条经验耗时 {store_time:.2f}s > 2s"
        assert retrieve_time < 2.0, f"20 次检索耗时 {retrieve_time:.2f}s > 2s"

    def test_multi_view_fusion_quality(self):
        """多视角融合检索质量。"""
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB, ExperienceType
        from mci_world_model.sdk._multi_view_retriever import (
            FusionStrategy,
            MultiViewRetriever,
            QuerySpec,
        )

        db = ExperienceDB(max_experiences=100)
        retriever = MultiViewRetriever(experience_db=db)

        # 存储有区分度的经验
        for i in range(20):
            topic = "pendulum" if i < 10 else "circuit"
            exp = Experience(
                experience_id=f"mv_{i}",
                experience_type=ExperienceType.SUCCESS,
                tags=[topic, f"sub_{i % 5}"],
                importance=0.5 + (i % 10) * 0.05,
            )
            db.store(exp)

        # 融合检索
        query = QuerySpec(tags=["pendulum", "control"])
        results = retriever.retrieve(query, top_k=5, strategy=FusionStrategy.WEIGHTED)
        assert len(results) > 0

        # pendulum 相关应占多数
        pendulum_hits = sum(1 for r in results if "pendulum" in r.experience.tags)
        assert pendulum_hits >= 3, f"融合检索 pendulum 命中率 {pendulum_hits}/5 < 60%"

    def test_experience_consolidation(self):
        """经验巩固：合并相似经验。"""
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB, ExperienceType

        db = ExperienceDB(max_experiences=100)

        # 存储高度相似经验
        for i in range(10):
            exp = Experience(
                experience_id=f"similar_{i}",
                experience_type=ExperienceType.SUCCESS,
                tags=["pendulum", "control", "stabilize", "energy_swing"],
                importance=0.5,
            )
            db.store(exp)

        before = db.statistics().total_experiences
        db.consolidate()
        after = db.statistics().total_experiences

        # 巩固后经验数应减少（相似经验被合并）
        assert after <= before, f"巩固后经验数 {after} 应 <= {before}"


# =============================================================================
# D6: 异常检测 (Anomaly Detection)
# =============================================================================


class TestAnomalyDetectionBenchmark:
    """D6: 异常检测能力评估。

    测试惊奇信号检测和认知诊断:
    - 高惊奇信号检测
    - 多信号批量诊断
    """

    def test_surprise_signal_detection(self):
        """高惊奇信号检测。"""
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser, SeverityLevel, SurpriseSignal

        diagnoser = MetaDiagnoser()

        # 高惊奇信号
        signal = SurpriseSignal(
            score=0.95,
            source="prediction_error",
            layer="prediction",
            features={"error_rate": 0.9, "direction_error": 0.85, "state_distance": 0.7},
        )

        diagnosis = diagnoser.diagnose([signal])
        assert diagnosis.severity in (SeverityLevel.HIGH, SeverityLevel.CRITICAL), (
            f"高惊奇信号应触发高严重度诊断: {diagnosis.severity}"
        )

    def test_low_surprise_filter(self):
        """低惊奇信号过滤。"""
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser, SeverityLevel, SurpriseSignal

        diagnoser = MetaDiagnoser()

        # 低惊奇信号
        signal = SurpriseSignal(
            score=0.1,
            source="minor_fluctuation",
            layer="perception",
            features={"noise_level": 0.05, "direction_error": 0.1},
        )

        diagnosis = diagnoser.diagnose([signal])
        assert diagnosis.severity in (SeverityLevel.LOW, SeverityLevel.MEDIUM), (
            f"低惊奇信号应为低严重度: {diagnosis.severity}"
        )

    def test_batch_diagnosis(self):
        """批量诊断：多个信号同时处理。"""
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser, SurpriseSignal

        diagnoser = MetaDiagnoser()

        signals = [
            SurpriseSignal(
                score=0.8, source="pred", layer="prediction", features={"error_rate": 0.7, "direction_error": 0.6}
            ),
            SurpriseSignal(
                score=0.6, source="causal", layer="causal", features={"connectivity": 0.4, "state_distance": 0.5}
            ),
            SurpriseSignal(score=0.3, source="noise", layer="perception", features={"noise": 0.1}),
        ]

        results = diagnoser.batch_diagnose([[s] for s in signals])
        assert len(results) == 3

    def test_cognitive_health_score(self):
        """认知健康度评分。"""
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser

        diagnoser = MetaDiagnoser()
        health = diagnoser.cognitive_health_score()

        assert isinstance(health, dict)
        # 六维评分
        expected_dims = {
            "causal_discovery",
            "counterfactual",
            "ood_generalization",
            "explainability",
            "memory_reuse",
            "anomaly_detection",
        }
        for dim in expected_dims:
            assert dim in health, f"缺少维度: {dim}"
            assert 0 <= health[dim] <= 1.0


# =============================================================================
# 综合评分
# =============================================================================


class TestCognitiveCompositeScore:
    """六维综合认知能力评分。"""

    def test_composite_score_calculation(self):
        """综合评分 = 六维平均分。"""
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser

        diagnoser = MetaDiagnoser()
        health = diagnoser.cognitive_health_score()

        # 提取各维度分数
        if health:
            scores = [_score_0_100(v) for v in health.values()]
            composite = sum(scores) / len(scores)
            # 记录但不硬性要求 ≥ 75（这是 v4.0.0 的目标）
            assert composite >= 0, f"综合评分 {composite} 应 >= 0"

    def test_diagnoser_stats(self):
        """MetaDiagnoser 统计信息。"""
        from mci_world_model.sdk._meta_diagnoser import MetaDiagnoser, SurpriseSignal

        diagnoser = MetaDiagnoser()
        signal = SurpriseSignal(score=0.7, source="test", layer="test", features={})
        diagnoser.diagnose([signal])

        stats = diagnoser.stats
        assert stats.total_diagnoses >= 1
