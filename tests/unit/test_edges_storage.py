"""Unit tests for ``memorus.core.storage.edges`` (STORY-R096)."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from memorus.core.storage.edges import (
    BulletEdge,
    EdgeType,
    SqliteEdgeStore,
    adamic_adar_weight,
    clamp_weight,
    compute_sources_jaccard,
    record_co_recall,
    recompute_derived_from_edges,
    write_supersedes_edge,
)
from memorus.core.types import SourceRef


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source(conv_id: str, offset: int) -> SourceRef:
    return SourceRef(
        conversation_id=conv_id,
        turn_hash=SourceRef.compute_turn_hash(f"{conv_id}:{offset}"),
        turn_offset=offset,
        timestamp=datetime.now(timezone.utc),
        role="assistant",
    )


@pytest.fixture
def store() -> SqliteEdgeStore:
    """Fresh in-memory edge store."""
    s = SqliteEdgeStore("")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestPureHelpers:
    def test_clamp_weight_in_range(self) -> None:
        assert clamp_weight(0.5) == 0.5

    def test_clamp_weight_below_zero(self) -> None:
        assert clamp_weight(-1.0) == 0.0

    def test_clamp_weight_above_one(self) -> None:
        assert clamp_weight(2.5) == 1.0

    def test_adamic_adar_handles_zero_degree(self) -> None:
        w = adamic_adar_weight(1.0, 0)
        assert w > 0.0

    def test_adamic_adar_decreases_with_degree(self) -> None:
        assert adamic_adar_weight(1.0, 1) > adamic_adar_weight(1.0, 100)

    def test_jaccard_basic(self) -> None:
        a = [_make_source("c1", 0), _make_source("c1", 1)]
        b = [_make_source("c1", 0), _make_source("c2", 0)]
        # intersection = 1 (c1,0); union = 3
        assert abs(compute_sources_jaccard(a, b) - 1.0 / 3.0) < 1e-9

    def test_jaccard_empty_sides(self) -> None:
        a = [_make_source("c1", 0)]
        assert compute_sources_jaccard([], a) == 0.0
        assert compute_sources_jaccard(a, []) == 0.0


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


class TestUpsert:
    def test_insert_then_read(self, store: SqliteEdgeStore) -> None:
        edge = store.upsert_edge("a", "b", EdgeType.SUPERSEDES, 0.8)
        assert isinstance(edge, BulletEdge)
        assert edge.weight == 0.8
        assert store.get_weight("a", "b", EdgeType.SUPERSEDES) == 0.8

    def test_upsert_replaces(self, store: SqliteEdgeStore) -> None:
        store.upsert_edge("a", "b", EdgeType.SUPERSEDES, 0.3)
        store.upsert_edge("a", "b", EdgeType.SUPERSEDES, 0.9)
        assert len(store.all_edges()) == 1
        assert store.get_weight("a", "b", EdgeType.SUPERSEDES) == 0.9

    def test_upsert_clamps_weight(self, store: SqliteEdgeStore) -> None:
        store.upsert_edge("a", "b", EdgeType.SUPERSEDES, 2.5)
        assert store.get_weight("a", "b", EdgeType.SUPERSEDES) == 1.0
        store.upsert_edge("a", "b", EdgeType.SUPERSEDES, -0.5)
        assert store.get_weight("a", "b", EdgeType.SUPERSEDES) == 0.0

    def test_upsert_rejects_self_loop(self, store: SqliteEdgeStore) -> None:
        with pytest.raises(ValueError):
            store.upsert_edge("a", "a", EdgeType.SUPERSEDES, 1.0)

    def test_upsert_rejects_empty_id(self, store: SqliteEdgeStore) -> None:
        with pytest.raises(ValueError):
            store.upsert_edge("", "b", EdgeType.SUPERSEDES, 1.0)
        with pytest.raises(ValueError):
            store.upsert_edge("a", "", EdgeType.SUPERSEDES, 1.0)

    def test_different_types_coexist(self, store: SqliteEdgeStore) -> None:
        store.upsert_edge("a", "b", EdgeType.SUPERSEDES, 1.0)
        store.upsert_edge("a", "b", EdgeType.CO_RECALLED_WITH, 0.5)
        assert len(store.all_edges()) == 2


class TestIncrementDecay:
    def test_increment_accumulates_and_caps(self, store: SqliteEdgeStore) -> None:
        for _ in range(15):
            store.increment_weight("a", "b", EdgeType.CO_RECALLED_WITH, 0.1, 1.0)
        w = store.get_weight("a", "b", EdgeType.CO_RECALLED_WITH)
        assert abs(w - 1.0) < 1e-9

    def test_increment_respects_max_weight(self, store: SqliteEdgeStore) -> None:
        for _ in range(10):
            store.increment_weight("a", "b", EdgeType.CO_RECALLED_WITH, 0.2, 0.5)
        assert store.get_weight("a", "b", EdgeType.CO_RECALLED_WITH) == 0.5

    def test_decay_all_multiplies(self, store: SqliteEdgeStore) -> None:
        store.upsert_edge("a", "b", EdgeType.CO_RECALLED_WITH, 1.0)
        store.upsert_edge("c", "d", EdgeType.SUPERSEDES, 0.8)
        n = store.decay_all(0.5)
        assert n == 2
        assert abs(store.get_weight("a", "b", EdgeType.CO_RECALLED_WITH) - 0.5) < 1e-9

    def test_decay_filters_by_type(self, store: SqliteEdgeStore) -> None:
        store.upsert_edge("a", "b", EdgeType.CO_RECALLED_WITH, 0.8)
        store.upsert_edge("c", "d", EdgeType.SUPERSEDES, 0.8)
        store.decay_all(0.5, edge_type=EdgeType.CO_RECALLED_WITH)
        assert abs(store.get_weight("a", "b", EdgeType.CO_RECALLED_WITH) - 0.4) < 1e-9
        assert store.get_weight("c", "d", EdgeType.SUPERSEDES) == 0.8

    def test_decay_rejects_negative_factor(self, store: SqliteEdgeStore) -> None:
        with pytest.raises(ValueError):
            store.decay_all(-0.1)


class TestQueries:
    def test_out_and_in_edges(self, store: SqliteEdgeStore) -> None:
        store.upsert_edge("a", "b", EdgeType.SUPERSEDES, 1.0)
        store.upsert_edge("a", "c", EdgeType.DERIVED_FROM, 0.5)
        store.upsert_edge("d", "a", EdgeType.CO_RECALLED_WITH, 0.3)

        out_ids = {e.to_id for e in store.out_edges("a")}
        in_ids = {e.from_id for e in store.in_edges("a")}
        assert out_ids == {"b", "c"}
        assert in_ids == {"d"}

    def test_degree_counts_both_directions(self, store: SqliteEdgeStore) -> None:
        store.upsert_edge("a", "b", EdgeType.SUPERSEDES, 1.0)
        store.upsert_edge("c", "a", EdgeType.SUPERSEDES, 1.0)
        assert store.degree("a") == 2

    def test_degree_zero_for_unknown(self, store: SqliteEdgeStore) -> None:
        assert store.degree("ghost") == 0

    def test_get_weight_absent_is_zero(self, store: SqliteEdgeStore) -> None:
        assert store.get_weight("a", "b", EdgeType.SUPERSEDES) == 0.0


class TestCascadeDelete:
    def test_delete_by_bullet_cascades(self, store: SqliteEdgeStore) -> None:
        store.upsert_edge("a", "b", EdgeType.SUPERSEDES, 1.0)
        store.upsert_edge("b", "c", EdgeType.DERIVED_FROM, 0.5)
        store.upsert_edge("a", "c", EdgeType.CO_RECALLED_WITH, 0.3)
        n = store.delete_by_bullet("b")
        assert n == 2
        remaining = [(e.from_id, e.to_id) for e in store.all_edges()]
        assert remaining == [("a", "c")]

    def test_delete_unknown_is_noop(self, store: SqliteEdgeStore) -> None:
        store.upsert_edge("a", "b", EdgeType.SUPERSEDES, 1.0)
        assert store.delete_by_bullet("ghost") == 0
        assert len(store.all_edges()) == 1

    def test_delete_empty_id_is_noop(self, store: SqliteEdgeStore) -> None:
        store.upsert_edge("a", "b", EdgeType.SUPERSEDES, 1.0)
        assert store.delete_by_bullet("") == 0
        assert len(store.all_edges()) == 1


# ---------------------------------------------------------------------------
# Edge-writing helpers
# ---------------------------------------------------------------------------


class TestWriteSupersedesEdge:
    def test_basic(self, store: SqliteEdgeStore) -> None:
        edge = write_supersedes_edge(store, "old", "new")
        assert edge is not None
        assert edge.edge_type == EdgeType.SUPERSEDES
        assert edge.weight == 1.0
        assert edge.from_id == "old"
        assert edge.to_id == "new"

    def test_noop_on_self_loop(self, store: SqliteEdgeStore) -> None:
        assert write_supersedes_edge(store, "a", "a") is None
        assert store.all_edges() == []

    def test_noop_on_empty_id(self, store: SqliteEdgeStore) -> None:
        assert write_supersedes_edge(store, "", "b") is None
        assert write_supersedes_edge(store, "a", "") is None
        assert store.all_edges() == []


class TestRecordCoRecall:
    def test_pairs_written_both_directions(self, store: SqliteEdgeStore) -> None:
        n = record_co_recall(store, ["a", "b", "c"], delta=0.1, max_weight=1.0)
        # 3 IDs -> 3 unordered pairs * 2 directions = 6
        assert n == 6
        assert store.get_weight("a", "b", EdgeType.CO_RECALLED_WITH) == 0.1
        assert store.get_weight("b", "a", EdgeType.CO_RECALLED_WITH) == 0.1

    def test_singleton_is_noop(self, store: SqliteEdgeStore) -> None:
        assert record_co_recall(store, ["a"]) == 0
        assert store.all_edges() == []

    def test_dedup_within_batch(self, store: SqliteEdgeStore) -> None:
        # Duplicates collapse to a single id before pairing.
        record_co_recall(store, ["a", "a", "b"])
        ids = {(e.from_id, e.to_id) for e in store.all_edges()}
        assert ids == {("a", "b"), ("b", "a")}

    def test_accumulates_across_calls(self, store: SqliteEdgeStore) -> None:
        for _ in range(5):
            record_co_recall(store, ["a", "b"], delta=0.1, max_weight=1.0)
        assert abs(store.get_weight("a", "b", EdgeType.CO_RECALLED_WITH) - 0.5) < 1e-9


class TestRecomputeDerivedFromEdges:
    def test_jaccard_on_shared_sources(self, store: SqliteEdgeStore) -> None:
        bullets = [
            ("b1", [_make_source("c1", 0), _make_source("c1", 1)]),
            ("b2", [_make_source("c1", 0), _make_source("c2", 0)]),
            ("b3", [_make_source("c9", 0)]),
        ]
        n = recompute_derived_from_edges(store, bullets)
        # Only b1 ↔ b2 overlaps; 2 edges (both directions).
        assert n == 2
        w = store.get_weight("b1", "b2", EdgeType.DERIVED_FROM)
        assert abs(w - 1.0 / 3.0) < 1e-9

    def test_skips_empty_sources(self, store: SqliteEdgeStore) -> None:
        bullets = [("b1", []), ("b2", [_make_source("c1", 0)])]
        assert recompute_derived_from_edges(store, bullets) == 0

    def test_min_weight_filters(self, store: SqliteEdgeStore) -> None:
        bullets = [
            ("b1", [_make_source("c1", 0), _make_source("c1", 1)]),
            ("b2", [_make_source("c1", 0), _make_source("c2", 0)]),
        ]
        # Jaccard = 1/3; set min above that -> no edges.
        n = recompute_derived_from_edges(store, bullets, min_weight=0.5)
        assert n == 0


# ---------------------------------------------------------------------------
# Schema & persistence
# ---------------------------------------------------------------------------


class TestSchemaPersistence:
    def test_migration_idempotent(self, tmp_path) -> None:
        path = str(tmp_path / "edges.db")
        s1 = SqliteEdgeStore(path)
        s1.upsert_edge("a", "b", EdgeType.SUPERSEDES, 1.0)
        s1.close()
        # Second open must not re-error on schema.
        s2 = SqliteEdgeStore(path)
        assert s2.get_weight("a", "b", EdgeType.SUPERSEDES) == 1.0
        s2.close()

    def test_shares_schema_with_rust_layout(self, tmp_path) -> None:
        """Confirm the schema columns exactly match the Rust migration."""
        path = str(tmp_path / "edges.db")
        store = SqliteEdgeStore(path)
        try:
            import sqlite3

            conn = sqlite3.connect(path)
            cols = [
                row[1]
                for row in conn.execute("PRAGMA table_info(bullet_edges)").fetchall()
            ]
            assert cols == [
                "from_id", "to_id", "edge_type", "weight", "updated_at",
            ]
            idxs = {
                row[1]
                for row in conn.execute(
                    "SELECT * FROM sqlite_master WHERE type='index' "
                    "AND tbl_name='bullet_edges'"
                ).fetchall()
            }
            assert "idx_edges_from" in idxs
            assert "idx_edges_to" in idxs
            conn.close()
        finally:
            store.close()

    def test_unknown_edge_type_in_db_decodes_as_supersedes(
        self, tmp_path
    ) -> None:
        """Forward-compat: unknown type strings must not crash reads."""
        path = str(tmp_path / "edges.db")
        store = SqliteEdgeStore(path)
        try:
            import sqlite3

            conn = sqlite3.connect(path)
            conn.execute(
                "INSERT INTO bullet_edges (from_id, to_id, edge_type, weight, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("a", "b", "future_type", 0.5, "2026-04-23T00:00:00+00:00"),
            )
            conn.commit()
            conn.close()
            edges = store.all_edges()
            assert len(edges) == 1
            assert edges[0].edge_type == EdgeType.SUPERSEDES  # fallback
        finally:
            store.close()
