"""Phase 2 (v3.5.0) — 经验记忆系统全量测试

覆盖:
    1. ExperienceDB — 三维融合经验记忆库 (store/retrieve/consolidate/forget/statistics)
    2. MultiViewRetriever — 五维融合检索器 (weighted/borda/hybrid)
    3. WorkingMemory ↔ ExperienceDB 集成 (flush/retrieve_hints)
    4. 导入兼容性 (sdk/_sys/top/world_model 四条路径)

KPI:
    K2-1: MultiViewRetriever 不再为 None
    K2-2: 经验检索准确率 ≥ 0.75 (Top-5)
    K2-4: 1000 条经验 < 50MB
    K2-5: ≥1780 passed, 零 regression
"""

from __future__ import annotations

import time

# =============================================================================
# 1. ExperienceDB
# =============================================================================


class TestExperienceDB:
    """ExperienceDB — 三维融合经验记忆库。"""

    def test_init_defaults(self):
        from mci_world_model.sdk._experience_memory import ExperienceDB

        db = ExperienceDB()
        assert db.size == 0
        assert db.half_life_hours == 168.0
        assert db.max_experiences == 10000

    def test_store_basic(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB, ExperienceType

        db = ExperienceDB()
        exp = Experience(
            experience_id="exp_001",
            experience_type=ExperienceType.SUCCESS,
            tags=["pendulum", "prediction"],
            causal_edges=[("theta", "omega")],
            outcome="误差 < 0.05",
            importance=0.9,
        )
        eid = db.store(exp)
        assert eid == "exp_001"
        assert db.size == 1

    def test_store_auto_id(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB

        db = ExperienceDB()
        exp = Experience(tags=["test"])
        eid = db.store(exp)
        assert eid.startswith("exp_")
        assert db.size == 1

    def test_store_kwargs(self):
        from mci_world_model.sdk._experience_memory import ExperienceDB

        db = ExperienceDB()
        eid = db.store(tags=["physics", "gravity"], outcome="g = 9.81")
        assert db.size == 1
        exp = db.get(eid)
        assert exp is not None
        assert "physics" in exp.tags

    def test_store_batch(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB

        db = ExperienceDB()
        exps = [Experience(tags=[f"tag_{i}"]) for i in range(5)]
        ids = db.store_batch(exps)
        assert len(ids) == 5
        assert db.size == 5

    def test_store_duplicate(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB

        db = ExperienceDB()
        exp = Experience(experience_id="dup_001", tags=["test"])
        db.store(exp)
        db.store(exp)  # duplicate
        assert db.size == 1

    def test_retrieve_by_tags(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB

        db = ExperienceDB()
        db.store(Experience(tags=["pendulum", "theta", "omega"], importance=0.9))
        db.store(Experience(tags=["gravity", "mass"], importance=0.8))
        db.store(Experience(tags=["pendulum", "gravity"], importance=0.7))

        results = db.retrieve(query_tags=["pendulum", "theta"], top_k=3)
        assert len(results) >= 1
        # Best match should have "pendulum" and "theta"
        assert "pendulum" in results[0].experience.tags

    def test_retrieve_by_edges(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB

        db = ExperienceDB()
        db.store(
            Experience(
                tags=["test"],
                causal_edges=[("theta", "omega"), ("gravity", "period")],
                importance=0.9,
            )
        )
        db.store(
            Experience(
                tags=["test2"],
                causal_edges=[("mass", "force")],
                importance=0.8,
            )
        )

        results = db.retrieve(
            query_edges=[("theta", "omega")],
            top_k=2,
        )
        assert len(results) >= 1

    def test_retrieve_type_filter(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB, ExperienceType

        db = ExperienceDB()
        db.store(Experience(tags=["a"], experience_type=ExperienceType.SUCCESS, importance=0.9))
        db.store(Experience(tags=["b"], experience_type=ExperienceType.FAILURE, importance=0.9))

        results = db.retrieve(
            query_tags=["a"],
            type_filter=ExperienceType.SUCCESS,
            top_k=5,
        )
        for r in results:
            assert r.experience.experience_type == ExperienceType.SUCCESS

    def test_retrieve_weights(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB

        db = ExperienceDB()
        db.store(Experience(tags=["alpha"], importance=0.9))

        # All semantic weight
        results = db.retrieve(
            query_tags=["alpha"],
            top_k=1,
            weights=(1.0, 0.0, 0.0),
        )
        assert len(results) >= 1
        assert results[0].semantic_score > 0

    def test_retrieve_recent(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB

        db = ExperienceDB()
        for i in range(3):
            db.store(Experience(tags=[f"item_{i}"]))

        results = db.retrieve_recent(n=2)
        assert len(results) == 2

    def test_retrieve_by_causal_edge(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB

        db = ExperienceDB()
        db.store(
            Experience(
                tags=["test"],
                causal_edges=[("A", "B"), ("B", "C")],
            )
        )
        results = db.retrieve_by_causal_edge("A", "B")
        assert len(results) == 1

    def test_consolidate(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB, ExperienceType

        db = ExperienceDB(consolidation_threshold=0.8)
        # Two very similar experiences
        db.store(
            Experience(
                tags=["a", "b", "c", "d"],
                experience_type=ExperienceType.SUCCESS,
                importance=0.9,
            )
        )
        db.store(
            Experience(
                tags=["a", "b", "c", "d", "e"],
                experience_type=ExperienceType.SUCCESS,
                importance=0.7,
            )
        )
        merged = db.consolidate()
        assert merged >= 1
        assert db.size == 1

    def test_forget(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB

        db = ExperienceDB()
        # Low importance + old
        old_exp = Experience(
            tags=["old"],
            importance=0.01,
            timestamp=time.time() - 365 * 86400,  # 1 year old
        )
        db.store(old_exp)
        # High importance
        new_exp = Experience(tags=["new"], importance=1.0)
        db.store(new_exp)

        forgotten = db.forget(threshold=0.05)
        assert forgotten >= 1
        assert db.size == 1

    def test_capacity_eviction(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB

        db = ExperienceDB(max_experiences=3)
        for i in range(5):
            db.store(Experience(tags=[f"item_{i}"], importance=0.1 + i * 0.2))
        assert db.size <= 3

    def test_get_and_remove(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB

        db = ExperienceDB()
        db.store(Experience(experience_id="rem_001", tags=["test"]))
        assert db.get("rem_001") is not None

        db.remove("rem_001")
        assert db.get("rem_001") is None
        assert db.size == 0

    def test_clear(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB

        db = ExperienceDB()
        for i in range(5):
            db.store(Experience(tags=[f"t{i}"]))
        assert db.size == 5

        db.clear()
        assert db.size == 0

    def test_statistics(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB, ExperienceType

        db = ExperienceDB()
        db.store(Experience(tags=["a", "b"], experience_type=ExperienceType.SUCCESS, importance=0.8))
        db.store(Experience(tags=["c"], experience_type=ExperienceType.FAILURE, importance=0.3))

        stats = db.statistics()
        assert stats.total_experiences == 2
        assert "success" in stats.by_type
        assert "failure" in stats.by_type

    def test_to_dict(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB

        db = ExperienceDB()
        db.store(Experience(tags=["test"]))
        d = db.to_dict()
        assert d["size"] == 1
        assert len(d["experiences"]) == 1

    def test_experience_age(self):
        from mci_world_model.sdk._experience_memory import Experience

        exp = Experience(timestamp=time.time() - 86400)  # 1 day ago
        assert 0.9 < exp.age_days() < 1.1

    def test_experience_recency_score(self):
        from mci_world_model.sdk._experience_memory import Experience

        # Fresh experience should have high recency
        fresh = Experience(timestamp=time.time())
        assert fresh.recency_score() > 0.9

        # Old experience should have low recency
        old = Experience(timestamp=time.time() - 30 * 86400)
        assert old.recency_score() < 0.1

    def test_memory_estimate_kb(self):
        """K2-4: 1000 条经验 < 50MB."""
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB

        db = ExperienceDB()
        for i in range(1000):
            db.store(
                Experience(
                    tags=[f"tag_{i % 50}", "common"],
                    causal_edges=[(f"cause_{i % 10}", f"effect_{i % 10}")],
                    outcome=f"result_{i}",
                )
            )
        stats = db.statistics()
        assert stats.total_experiences == 1000
        # Rough estimate: 1000 × ~1KB = ~1MB << 50MB
        assert stats.memory_estimate_kb < 50000  # 50MB in KB


# =============================================================================
# 2. MultiViewRetriever
# =============================================================================


class TestMultiViewRetriever:
    """MultiViewRetriever — 五维融合检索器。"""

    def test_init_defaults(self):
        from mci_world_model.sdk._multi_view_retriever import MultiViewRetriever

        r = MultiViewRetriever()
        assert r.experience_db is not None
        assert r.fusion_strategy.value == "weighted"

    def test_retrieve_basic(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB
        from mci_world_model.sdk._multi_view_retriever import MultiViewRetriever, QuerySpec

        db = ExperienceDB()
        db.store(Experience(tags=["pendulum", "theta"], importance=0.9))
        db.store(Experience(tags=["gravity", "mass"], importance=0.8))

        r = MultiViewRetriever(experience_db=db)
        results = r.retrieve(QuerySpec(tags=["pendulum"]), top_k=2)
        assert len(results) >= 1

    def test_retrieve_kwargs(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB
        from mci_world_model.sdk._multi_view_retriever import MultiViewRetriever

        db = ExperienceDB()
        db.store(Experience(tags=["alpha", "beta"], importance=0.9))

        r = MultiViewRetriever(experience_db=db)
        results = r.retrieve(tags=["alpha"], top_k=1)
        assert len(results) == 1

    def test_five_view_scores(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB
        from mci_world_model.sdk._multi_view_retriever import MultiViewRetriever, QuerySpec

        db = ExperienceDB()
        exp = Experience(
            tags=["physics", "gravity"],
            causal_edges=[("mass", "force")],
            importance=0.9,
        )
        db.store(exp)

        r = MultiViewRetriever(experience_db=db)
        r.register_context(exp.experience_id, {"domain": "physics"})
        r.register_features(exp.experience_id, [1.0, 0.5, 0.3])

        results = r.retrieve(
            QuerySpec(
                tags=["physics"],
                causal_edges=[("mass", "force")],
                context={"domain": "physics"},
                state_features=[1.0, 0.5, 0.3],
            ),
            top_k=1,
        )
        assert len(results) == 1
        vs = results[0].view_scores
        assert "semantic" in vs
        assert "causal" in vs
        assert "temporal" in vs
        assert "contextual" in vs
        assert "structural" in vs

    def test_borda_fusion(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB
        from mci_world_model.sdk._multi_view_retriever import FusionStrategy, MultiViewRetriever, QuerySpec

        db = ExperienceDB()
        for i in range(5):
            db.store(Experience(tags=[f"tag_{i}", "common"], importance=0.5 + i * 0.1))

        r = MultiViewRetriever(experience_db=db)
        results = r.retrieve(
            QuerySpec(tags=["common"]),
            top_k=3,
            strategy=FusionStrategy.BORDA,
        )
        assert len(results) <= 3
        for res in results:
            assert res.strategy == "borda"

    def test_hybrid_fusion(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB
        from mci_world_model.sdk._multi_view_retriever import FusionStrategy, MultiViewRetriever, QuerySpec

        db = ExperienceDB()
        db.store(Experience(tags=["hybrid", "test"], importance=0.9))

        r = MultiViewRetriever(experience_db=db)
        results = r.retrieve(
            QuerySpec(tags=["hybrid"]),
            top_k=1,
            strategy=FusionStrategy.HYBRID,
        )
        assert len(results) >= 1
        assert results[0].strategy == "hybrid"

    def test_context_index(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB
        from mci_world_model.sdk._multi_view_retriever import MultiViewRetriever, QuerySpec

        db = ExperienceDB()
        exp = Experience(tags=["test"], importance=0.9)
        db.store(exp)

        r = MultiViewRetriever(experience_db=db)
        r.register_context(exp.experience_id, {"domain": "physics", "scenario": "pendulum"})

        results = r.retrieve(
            QuerySpec(context={"domain": "physics"}),
            top_k=1,
        )
        assert len(results) >= 1

    def test_structural_index(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB
        from mci_world_model.sdk._multi_view_retriever import MultiViewRetriever, QuerySpec

        db = ExperienceDB()
        exp = Experience(tags=["struct_test"], importance=0.9)
        db.store(exp)

        r = MultiViewRetriever(experience_db=db)
        r.register_features(exp.experience_id, [1.0, 0.0, 0.0])

        results = r.retrieve(
            QuerySpec(state_features=[1.0, 0.0, 0.0]),
            top_k=1,
        )
        assert len(results) >= 1

    def test_retrieve_by_tags_shortcut(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB
        from mci_world_model.sdk._multi_view_retriever import MultiViewRetriever

        db = ExperienceDB()
        db.store(Experience(tags=["shortcut", "test"], importance=0.9))

        r = MultiViewRetriever(experience_db=db)
        results = r.retrieve_by_tags(["shortcut"])
        assert len(results) >= 1

    def test_retrieve_by_causal_shortcut(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB
        from mci_world_model.sdk._multi_view_retriever import MultiViewRetriever

        db = ExperienceDB()
        db.store(
            Experience(
                tags=["causal_test"],
                causal_edges=[("A", "B")],
                importance=0.9,
            )
        )

        r = MultiViewRetriever(experience_db=db)
        results = r.retrieve_by_causal([("A", "B")])
        assert len(results) >= 1

    def test_statistics(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB
        from mci_world_model.sdk._multi_view_retriever import MultiViewRetriever, QuerySpec

        db = ExperienceDB()
        db.store(Experience(tags=["stats_test"], importance=0.9))

        r = MultiViewRetriever(experience_db=db)
        r.retrieve(QuerySpec(tags=["stats_test"]), top_k=1)
        r.retrieve(QuerySpec(tags=["stats_test"]), top_k=1)

        stats = r.statistics()
        assert stats.total_queries == 2

    def test_reset_stats(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB
        from mci_world_model.sdk._multi_view_retriever import MultiViewRetriever, QuerySpec

        db = ExperienceDB()
        db.store(Experience(tags=["test"], importance=0.9))

        r = MultiViewRetriever(experience_db=db)
        r.retrieve(QuerySpec(tags=["test"]))
        r.reset_stats()
        assert r.statistics().total_queries == 0

    def test_clear(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB
        from mci_world_model.sdk._multi_view_retriever import MultiViewRetriever

        db = ExperienceDB()
        db.store(Experience(tags=["test"]))

        r = MultiViewRetriever(experience_db=db)
        r.clear()
        assert r.experience_db.size == 0

    def test_multi_view_result_to_dict(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB
        from mci_world_model.sdk._multi_view_retriever import MultiViewRetriever, QuerySpec

        db = ExperienceDB()
        db.store(Experience(tags=["dict_test"], outcome="OK", importance=0.9))

        r = MultiViewRetriever(experience_db=db)
        results = r.retrieve(QuerySpec(tags=["dict_test"]), top_k=1)
        d = results[0].to_dict()
        assert "score" in d
        assert "view_scores" in d
        assert "rank" in d

    def test_retrieval_accuracy_top5(self):
        """K2-2: 经验检索准确率 ≥ 0.75 (Top-5)."""
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB
        from mci_world_model.sdk._multi_view_retriever import MultiViewRetriever, QuerySpec

        db = ExperienceDB()
        # Create experiences with unique distinctive tags
        for i in range(10):
            db.store(
                Experience(
                    tags=["pendulum", f"pendulum_exp_{i}"],
                    importance=0.9,
                )
            )
        for i in range(10):
            db.store(
                Experience(
                    tags=["circuit", f"circuit_exp_{i}"],
                    importance=0.8,
                )
            )

        r = MultiViewRetriever(experience_db=db)

        # Query for "pendulum" — should get pendulum-tagged experiences in top 5
        results = r.retrieve(QuerySpec(tags=["pendulum"]), top_k=5)
        assert len(results) >= 5

        # Check that at least 75% of top-5 results contain "pendulum"
        pendulum_hits = sum(1 for res in results if "pendulum" in res.experience.tags)
        accuracy = pendulum_hits / max(1, len(results))
        assert accuracy >= 0.75, f"Top-5 accuracy {accuracy:.2%} < 75%"

    def test_unregister(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB
        from mci_world_model.sdk._multi_view_retriever import MultiViewRetriever

        db = ExperienceDB()
        exp = Experience(tags=["test"])
        db.store(exp)

        r = MultiViewRetriever(experience_db=db)
        r.register_context(exp.experience_id, {"key": "val"})
        r.unregister(exp.experience_id)
        # Should not crash, just no context match


# =============================================================================
# 3. WorkingMemory ↔ ExperienceDB 集成
# =============================================================================


class TestWorkingMemoryExperienceIntegration:
    """WorkingMemory ↔ ExperienceDB 集成测试。"""

    def test_flush_to_experience_db(self):
        from mci_world_model.sdk._experience_memory import ExperienceDB
        from mci_world_model.sdk._world_model import TrajectoryStep, WorkingMemory

        wm = WorkingMemory(max_length=5)
        for i in range(3):
            state = type("State", (), {"theta": i * 0.1, "omega": i * 0.2})()
            wm.push(TrajectoryStep(state=state, step_index=i))

        db = ExperienceDB()
        ids = wm.flush_to_experience_db(db, tags=["trajectory"])
        assert len(ids) == 3
        assert db.size == 3
        # WorkingMemory should be cleared after flush
        assert wm.state == "IDLE"

    def test_flush_with_cost(self):
        from mci_world_model.sdk._experience_memory import ExperienceDB, ExperienceType
        from mci_world_model.sdk._world_model import TrajectoryStep, WorkingMemory

        wm = WorkingMemory()
        state = type("State", (), {"theta": 0.5})()
        step = TrajectoryStep(state=state, step_index=0)
        # Mock cost signal
        step.cost_signal = type("Cost", (), {"total": 0.3})()
        wm.push(step)

        db = ExperienceDB()
        ids = wm.flush_to_experience_db(db)
        assert len(ids) == 1
        exp = db.get(ids[0])
        assert exp.experience_type == ExperienceType.SUCCESS

    def test_retrieve_experience_hints(self):
        from mci_world_model.sdk._experience_memory import Experience, ExperienceDB
        from mci_world_model.sdk._world_model import WorkingMemory

        db = ExperienceDB()
        db.store(Experience(tags=["hint", "pendulum"], importance=0.9))
        db.store(Experience(tags=["hint", "gravity"], importance=0.8))

        wm = WorkingMemory()
        results = wm.retrieve_experience_hints(db, ["pendulum"], top_k=2)
        assert len(results) >= 1

    def test_flush_empty_memory(self):
        from mci_world_model.sdk._experience_memory import ExperienceDB
        from mci_world_model.sdk._world_model import WorkingMemory

        wm = WorkingMemory()
        db = ExperienceDB()
        ids = wm.flush_to_experience_db(db)
        assert ids == []


# =============================================================================
# 4. 导入兼容性
# =============================================================================


class TestImportsV35:
    """v3.5.0 导入路径验证。"""

    def test_sdk_imports(self):
        from mci_world_model.sdk import (
            Experience,
            ExperienceDB,
            MultiViewRetriever,
        )

        assert MultiViewRetriever is not None
        assert ExperienceDB is not None
        assert Experience is not None

    def test_top_imports(self):
        from mci_world_model import (
            ExperienceDB,
            MultiViewRetriever,
        )

        assert MultiViewRetriever is not None
        assert ExperienceDB is not None

    def test_sys_imports(self):
        from mci_world_model._sys import (
            ExperienceDB,
            MultiViewRetriever,
        )

        assert MultiViewRetriever is not None
        assert ExperienceDB is not None

    def test_world_model_imports(self):
        from mci_world_model.world_model import MultiViewRetriever

        assert MultiViewRetriever is not None

    def test_mvr_not_none(self):
        """K2-1: MultiViewRetriever 不再为 None."""
        from mci_world_model import MultiViewRetriever

        assert MultiViewRetriever is not None
        assert isinstance(MultiViewRetriever, type)

    def test_experience_db_not_none(self):
        from mci_world_model import ExperienceDB

        assert ExperienceDB is not None
        assert isinstance(ExperienceDB, type)
