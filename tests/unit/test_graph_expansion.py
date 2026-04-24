"""Unit tests for Generator graph-aware expansion (STORY-R096)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from memorus.core.config import GraphExpansionConfig, RetrievalConfig
from memorus.core.engines.generator.engine import BulletForSearch, GeneratorEngine
from memorus.core.engines.generator.graph import (
    EdgeStoreProvider,
    GraphNeighbor,
    adamic_adar_weight,
    apply_graph_scores,
    expand_neighbors,
    scored_bullet_for_new_neighbor,
)
from memorus.core.engines.generator.metadata_matcher import MetadataInfo
from memorus.core.engines.generator.score_merger import ScoredBullet
from memorus.core.engines.generator.vector_searcher import VectorSearcher
from memorus.core.storage.edges import (
    EdgeType,
    SqliteEdgeStore,
    write_supersedes_edge,
)


_NOW = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class InMemoryEdges:
    """Lightweight EdgeProvider stub for isolated graph tests."""

    _out: dict[str, list[tuple[str, float]]]
    _deg: dict[str, int]

    @classmethod
    def new(cls) -> "InMemoryEdges":
        return cls(_out={}, _deg={})

    def add(self, frm: str, to: str, weight: float) -> None:
        self._out.setdefault(frm, []).append((to, weight))
        self._deg[frm] = self._deg.get(frm, 0) + 1
        self._deg[to] = self._deg.get(to, 0) + 1

    # Protocol methods
    def out_edges(self, from_id: str) -> list[tuple[str, float]]:
        return list(self._out.get(from_id, []))

    def degree(self, bullet_id: str) -> int:
        return self._deg.get(bullet_id, 0)


def _make_bullet(bid: str, content: str = "hello world") -> BulletForSearch:
    return BulletForSearch(
        bullet_id=bid,
        content=content,
        metadata=MetadataInfo(),
        created_at=_NOW - timedelta(days=30),
        decay_weight=1.0,
    )


def _make_scored(bid: str, final: float, content: str = "x") -> ScoredBullet:
    return ScoredBullet(
        bullet_id=bid,
        content=content,
        final_score=final,
        keyword_score=0.0,
        semantic_score=0.0,
        decay_weight=1.0,
        recency_boost=1.0,
        metadata={},
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestAdamicAdar:
    def test_handles_zero_degree(self) -> None:
        w = adamic_adar_weight(1.0, 0)
        assert w > 0.0
        assert math.isfinite(w)

    def test_known_value(self) -> None:
        # edge_weight=1.0, degree=1 -> 1 / ln(3)
        assert abs(adamic_adar_weight(1.0, 1) - 1.0 / math.log(3.0)) < 1e-12

    def test_decreases_with_degree(self) -> None:
        assert adamic_adar_weight(1.0, 1) > adamic_adar_weight(1.0, 100)


# ---------------------------------------------------------------------------
# expand_neighbors
# ---------------------------------------------------------------------------


class TestExpandNeighbors:
    def test_empty_seeds(self) -> None:
        assert expand_neighbors([], InMemoryEdges.new(), 5) == []

    def test_isolated_seed_returns_empty(self) -> None:
        out = expand_neighbors([("a", 1.0)], InMemoryEdges.new(), 5)
        assert out == []

    def test_prefers_low_degree_neighbor(self) -> None:
        edges = InMemoryEdges.new()
        edges.add("a", "low", 1.0)
        edges.add("a", "high", 1.0)
        for i in range(10):
            edges.add(f"x{i}", "high", 0.1)
        out = expand_neighbors([("a", 1.0)], edges, 5)
        scores = {n.bullet_id: n.graph_score for n in out}
        assert scores["low"] > scores["high"]

    def test_k_expand_caps_new_neighbors(self) -> None:
        edges = InMemoryEdges.new()
        for i in range(10):
            edges.add("a", f"n{i}", 1.0)
        out = expand_neighbors([("a", 1.0)], edges, 3)
        assert len(out) == 3

    def test_excluded_seeds_survive_beyond_k_cap(self) -> None:
        edges = InMemoryEdges.new()
        edges.add("a", "b", 1.0)
        edges.add("a", "c", 1.0)
        edges.add("a", "d", 1.0)
        # Exclude b and c (they represent seeds already in top-K);
        # only 1 new neighbor allowed.
        out = expand_neighbors(
            [("a", 1.0)], edges, 1, exclude_ids={"b", "c"},
        )
        ids = {n.bullet_id for n in out}
        # b + c must appear (excluded means "already in top-K") plus 1 new.
        assert ids == {"b", "c", "d"}

    def test_sorted_descending_by_score(self) -> None:
        edges = InMemoryEdges.new()
        edges.add("a", "high", 1.0)  # degree = 1 -> high score
        edges.add("a", "low", 0.01)  # degree = 1 -> low score
        out = expand_neighbors([("a", 1.0)], edges, 5)
        assert out[0].graph_score >= out[-1].graph_score

    def test_self_loops_defensive_skip(self) -> None:
        edges = InMemoryEdges.new()
        # Deliberately register a self-loop on "a".
        edges._out["a"] = [("a", 1.0)]
        edges._deg["a"] = 1
        out = expand_neighbors([("a", 1.0)], edges, 5)
        assert out == []


# ---------------------------------------------------------------------------
# apply_graph_scores
# ---------------------------------------------------------------------------


class TestApplyGraphScores:
    def test_weight_zero_passes_through(self) -> None:
        base = [_make_scored("a", 1.0)]
        out = apply_graph_scores(base, [GraphNeighbor("a", 10.0)], 0.0)
        assert out[0].final_score == 1.0

    def test_no_neighbors_passes_through(self) -> None:
        base = [_make_scored("a", 1.0)]
        assert apply_graph_scores(base, [], 0.3) == base

    def test_boost_applied_and_sorted(self) -> None:
        base = [_make_scored("a", 1.0), _make_scored("b", 2.0)]
        neighbors = [GraphNeighbor("a", 1.0)]  # 1.0 + 0.3*1.0 = 1.3 < 2.0
        out = apply_graph_scores(base, neighbors, 0.3)
        assert out[0].bullet_id == "b"
        assert out[1].bullet_id == "a"
        assert abs(out[1].final_score - 1.3) < 1e-9

    def test_original_list_not_mutated(self) -> None:
        base = [_make_scored("a", 1.0)]
        apply_graph_scores(base, [GraphNeighbor("a", 0.5)], 0.3)
        assert base[0].final_score == 1.0

    def test_neighbor_not_in_base_is_ignored(self) -> None:
        base = [_make_scored("a", 1.0)]
        out = apply_graph_scores(base, [GraphNeighbor("unknown", 1.0)], 0.3)
        assert out[0].final_score == 1.0


class TestScoredBulletForNewNeighbor:
    def test_score_is_weighted_graph_score(self) -> None:
        sb = scored_bullet_for_new_neighbor(
            bullet_id="x", content="c", graph_score=0.5, graph_score_weight=0.3,
        )
        assert abs(sb.final_score - 0.15) < 1e-9
        assert sb.keyword_score == 0.0
        assert sb.semantic_score == 0.0


# ---------------------------------------------------------------------------
# GeneratorEngine.search_with_graph
# ---------------------------------------------------------------------------


def _vs_empty() -> VectorSearcher:
    """Degraded-mode vector searcher so tests are deterministic."""
    return VectorSearcher(search_fn=None)


class TestSearchWithGraph:
    def test_disabled_is_byte_identical_to_search(self) -> None:
        engine = GeneratorEngine(
            config=RetrievalConfig(), vector_searcher=_vs_empty(),
        )
        bullets = [_make_bullet("a", "pnpm install fix"), _make_bullet("b", "other")]
        edges = InMemoryEdges.new()
        edges.add("a", "b", 1.0)

        base = engine.search("pnpm install", bullets, limit=20, now=_NOW)
        with_graph = engine.search_with_graph(
            "pnpm install", bullets, edges,
            graph_config=GraphExpansionConfig(enabled=False),
            limit=20, now=_NOW,
        )

        assert len(base) == len(with_graph)
        for b, g in zip(base, with_graph):
            assert b.bullet_id == g.bullet_id
            assert b.final_score == g.final_score

    def test_empty_base_returns_empty(self) -> None:
        engine = GeneratorEngine(
            config=RetrievalConfig(), vector_searcher=_vs_empty(),
        )
        edges = InMemoryEdges.new()
        edges.add("a", "b", 1.0)
        # Empty bullets -> base is empty -> expansion is skipped.
        out = engine.search_with_graph(
            "pnpm install", [], edges,
            graph_config=GraphExpansionConfig(enabled=True),
            limit=20, now=_NOW,
        )
        assert out == []

    def test_enabled_pulls_in_new_neighbor(self) -> None:
        engine = GeneratorEngine(
            config=RetrievalConfig(), vector_searcher=_vs_empty(),
        )
        # Query matches "a" on exact keyword; "b" is off-topic but linked.
        bullets = [
            _make_bullet("a", "pnpm install frozen lockfile"),
            _make_bullet("b", "node 20 compatibility"),
        ]
        edges = InMemoryEdges.new()
        edges.add("a", "b", 1.0)

        base_ids = {
            r.bullet_id for r in engine.search("pnpm install", bullets, limit=20, now=_NOW)
        }
        assert "a" in base_ids
        # Note: "b" may or may not score above zero in keyword search alone.
        # The point is expansion should surface it if missing.
        out = engine.search_with_graph(
            "pnpm install", bullets, edges,
            graph_config=GraphExpansionConfig(
                enabled=True, k_expand=5, graph_score_weight=0.3,
            ),
            limit=20, now=_NOW,
        )
        result_ids = {r.bullet_id for r in out}
        assert "a" in result_ids
        assert "b" in result_ids

    def test_enabled_boosts_existing_neighbor(self) -> None:
        engine = GeneratorEngine(
            config=RetrievalConfig(), vector_searcher=_vs_empty(),
        )
        bullets = [
            _make_bullet("a", "pnpm install frozen lockfile"),
            _make_bullet("b", "pnpm workspace lockfile workaround"),
        ]
        edges = InMemoryEdges.new()
        edges.add("a", "b", 1.0)
        edges.add("b", "a", 1.0)

        baseline = engine.search("pnpm install", bullets, limit=20, now=_NOW)
        baseline_by_id = {r.bullet_id: r.final_score for r in baseline}

        out = engine.search_with_graph(
            "pnpm install", bullets, edges,
            graph_config=GraphExpansionConfig(
                enabled=True, k_expand=0, graph_score_weight=0.3,
            ),
            limit=20, now=_NOW,
        )
        for r in out:
            # Everything already in baseline must have score >= baseline.
            if r.bullet_id in baseline_by_id:
                assert r.final_score >= baseline_by_id[r.bullet_id]

    def test_respects_limit(self) -> None:
        engine = GeneratorEngine(
            config=RetrievalConfig(), vector_searcher=_vs_empty(),
        )
        bullets = [_make_bullet(f"b{i}", f"pnpm workspace {i}") for i in range(10)]
        bullets[0] = _make_bullet("b0", "pnpm install primary")
        edges = InMemoryEdges.new()
        for i in range(1, 10):
            edges.add("b0", f"b{i}", 1.0)

        out = engine.search_with_graph(
            "pnpm install", bullets, edges,
            graph_config=GraphExpansionConfig(enabled=True, k_expand=5),
            limit=3, now=_NOW,
        )
        assert len(out) <= 3


# ---------------------------------------------------------------------------
# EdgeStoreProvider integration (Python-Rust shared data path)
# ---------------------------------------------------------------------------


class TestEdgeStoreProvider:
    def test_wraps_sqlite_store(self) -> None:
        store = SqliteEdgeStore("")
        try:
            store.upsert_edge("a", "b", EdgeType.SUPERSEDES, 0.9)
            store.upsert_edge("a", "c", EdgeType.CO_RECALLED_WITH, 0.4)
            provider = EdgeStoreProvider(store)
            out = provider.out_edges("a")
            assert {oid for oid, _ in out} == {"b", "c"}
            assert provider.degree("a") == 2
        finally:
            store.close()

    def test_falls_back_gracefully_on_error(self) -> None:
        class Broken:
            def out_edges(self, from_id: str):
                raise RuntimeError("boom")

            def degree(self, bullet_id: str):
                raise RuntimeError("boom")

        p = EdgeStoreProvider(Broken())
        assert p.out_edges("a") == []
        assert p.degree("a") == 0


# ---------------------------------------------------------------------------
# Supersede hook integration (Curator -> edge store helper)
# ---------------------------------------------------------------------------


class TestSupersedeHook:
    def test_write_supersedes_edge_weight_is_one(self) -> None:
        store = SqliteEdgeStore("")
        try:
            edge = write_supersedes_edge(store, "old", "new")
            assert edge is not None
            assert edge.edge_type == EdgeType.SUPERSEDES
            assert edge.weight == 1.0
        finally:
            store.close()

    def test_supersedes_from_merges_drops_self_loops(self) -> None:
        from memorus.core.engines.curator.engine import (
            ExistingBullet,
            MergeCandidate,
            SupersedeRecord,
            supersedes_from_merges,
        )
        from memorus.core.types import CandidateBullet

        existing = ExistingBullet(bullet_id="old1", content="x")
        existing_self = ExistingBullet(bullet_id="same", content="y")
        candidate = CandidateBullet(content="z")
        merges = [
            MergeCandidate(candidate=candidate, existing=existing, similarity=0.9),
            MergeCandidate(candidate=candidate, existing=existing_self, similarity=0.9),
        ]
        records = supersedes_from_merges(
            merges,
            new_ids={"old1": "new1", "same": "same"},  # second is self-loop
        )
        assert records == [SupersedeRecord(existing_id="old1", new_id="new1")]

    def test_persist_supersedes_writes_edges(self) -> None:
        from memorus.core.engines.curator.engine import (
            SupersedeRecord,
            persist_supersedes,
        )

        store = SqliteEdgeStore("")
        try:
            records = [
                SupersedeRecord(existing_id="old1", new_id="new1"),
                SupersedeRecord(existing_id="old2", new_id="new2"),
                SupersedeRecord(existing_id="self", new_id="self"),  # no-op
            ]
            written = persist_supersedes(store, records)
            assert len(written) == 2
            assert store.get_weight("old1", "new1", EdgeType.SUPERSEDES) == 1.0
            assert store.get_weight("old2", "new2", EdgeType.SUPERSEDES) == 1.0
        finally:
            store.close()
