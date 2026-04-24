"""SQLite-backed bullet-edge storage for graph-aware retrieval (STORY-R096).

Stores directed, typed, weighted edges between bullets. The schema is
identical to the Rust implementation
(``memorus-r/memorus-providers/src/edges.rs``) so the two can share a
single SQLite file:

.. code:: sql

    CREATE TABLE bullet_edges (
        from_id TEXT NOT NULL,
        to_id TEXT NOT NULL,
        edge_type TEXT NOT NULL,   -- 'supersedes' | 'co_recalled_with' | 'derived_from'
        weight REAL NOT NULL,      -- clamped to [0.0, 1.0]
        updated_at TEXT NOT NULL,  -- ISO-8601 UTC
        PRIMARY KEY (from_id, to_id, edge_type)
    );
    CREATE INDEX idx_edges_from ON bullet_edges(from_id);
    CREATE INDEX idx_edges_to   ON bullet_edges(to_id);
"""

from __future__ import annotations

import logging
import math
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, Optional

from memorus.core.types import SourceRef

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS bullet_edges (
    from_id TEXT NOT NULL,
    to_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    weight REAL NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (from_id, to_id, edge_type)
);
CREATE INDEX IF NOT EXISTS idx_edges_from ON bullet_edges(from_id);
CREATE INDEX IF NOT EXISTS idx_edges_to   ON bullet_edges(to_id);
"""


# ---------------------------------------------------------------------------
# Edge type + data
# ---------------------------------------------------------------------------


class EdgeType(str, Enum):
    """Type of bullet-to-bullet edge.

    The string values are the canonical SQLite ``edge_type`` column values;
    they must stay in sync with the Rust ``EdgeType::as_str`` mapping.
    """

    SUPERSEDES = "supersedes"
    CO_RECALLED_WITH = "co_recalled_with"
    DERIVED_FROM = "derived_from"

    @classmethod
    def parse(cls, value: str) -> Optional["EdgeType"]:
        """Return the matching enum variant or ``None`` for unknown strings."""
        try:
            return cls(value)
        except ValueError:
            return None


@dataclass(frozen=True)
class BulletEdge:
    """A single directed edge between two bullets."""

    from_id: str
    to_id: str
    edge_type: EdgeType
    weight: float
    updated_at: datetime


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def clamp_weight(value: float) -> float:
    """Clamp *value* to the ``[0.0, 1.0]`` interval."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def adamic_adar_weight(edge_weight: float, neighbor_degree: int) -> float:
    """Weighted Adamic-Adar term: ``edge_weight / ln(degree + 2)``.

    Uses natural log; safe for ``degree == 0`` because ``ln(2) > 0``.
    """
    denom = math.log(neighbor_degree + 2.0)
    if denom <= 0.0:
        return 0.0
    return edge_weight / denom


def compute_sources_jaccard(a: Iterable[SourceRef], b: Iterable[SourceRef]) -> float:
    """Jaccard similarity over ``(conversation_id, turn_offset)`` keys.

    Returns ``0.0`` for empty inputs, matching the Rust implementation.
    """
    set_a: set[tuple[str, int]] = {(s.conversation_id, s.turn_offset) for s in a}
    set_b: set[tuple[str, int]] = {(s.conversation_id, s.turn_offset) for s in b}
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return inter / union


# ---------------------------------------------------------------------------
# SQLite store
# ---------------------------------------------------------------------------


class SqliteEdgeStore:
    """SQLite-backed storage for :class:`BulletEdge`.

    The schema is initialised eagerly on construction — callers never need to
    run migrations separately. When *path* is empty (or ``":memory:"``) the
    store is in-memory.

    Because :mod:`sqlite3` connections are per-thread, this class creates a
    new connection on demand for each thread that calls into it. A single
    lock serialises writes to avoid ``sqlite3.OperationalError: database is
    locked`` under concurrent access.
    """

    # Thread-local connection cache.

    def __init__(self, path: str = "") -> None:
        self._path = path if path else ":memory:"
        self._is_memory = self._path == ":memory:"
        self._write_lock = threading.Lock()
        self._local = threading.local()

        # Memory databases cannot be shared across connections — keep one.
        if self._is_memory:
            self._memory_conn: Optional[sqlite3.Connection] = sqlite3.connect(
                self._path,
                detect_types=0,
                check_same_thread=False,
            )
        else:
            self._memory_conn = None

        # Run migration up front.
        conn = self._conn()
        conn.executescript(CREATE_TABLE_SQL)
        conn.commit()

    # -- connection management -----------------------------------------

    def _conn(self) -> sqlite3.Connection:
        """Return a thread-bound connection (or the single in-memory one)."""
        if self._memory_conn is not None:
            return self._memory_conn
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path)
            self._local.conn = conn
        return conn

    def close(self) -> None:
        """Close any connections held by this store."""
        if self._memory_conn is not None:
            try:
                self._memory_conn.close()
            except Exception:  # noqa: BLE001 — best-effort close
                logger.debug("Failed to close in-memory edge-store connection", exc_info=True)
            self._memory_conn = None
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 — best-effort close
                logger.debug("Failed to close edge-store connection", exc_info=True)
            self._local.conn = None

    def __enter__(self) -> "SqliteEdgeStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()

    # -- writes ---------------------------------------------------------

    def upsert_edge(
        self,
        from_id: str,
        to_id: str,
        edge_type: EdgeType,
        weight: float,
    ) -> BulletEdge:
        """Insert or replace an edge (clamps *weight* to ``[0, 1]``).

        Raises ``ValueError`` for empty IDs or self-loops.
        """
        if not from_id or not to_id:
            raise ValueError("from_id/to_id must be non-empty")
        if from_id == to_id:
            raise ValueError("self-loops are not allowed")

        w = clamp_weight(weight)
        now = datetime.now(timezone.utc)
        now_iso = _to_rfc3339(now)
        with self._write_lock:
            conn = self._conn()
            conn.execute(
                "INSERT OR REPLACE INTO bullet_edges "
                "(from_id, to_id, edge_type, weight, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (from_id, to_id, edge_type.value, w, now_iso),
            )
            conn.commit()
        return BulletEdge(
            from_id=from_id,
            to_id=to_id,
            edge_type=edge_type,
            weight=w,
            updated_at=now,
        )

    def increment_weight(
        self,
        from_id: str,
        to_id: str,
        edge_type: EdgeType,
        delta: float,
        max_weight: float = 1.0,
    ) -> BulletEdge:
        """Increment the edge's weight by *delta*, capped at *max_weight*.

        Creates the edge when it is absent.
        """
        current = self.get_weight(from_id, to_id, edge_type)
        new_weight = clamp_weight(min(current + delta, max_weight))
        return self.upsert_edge(from_id, to_id, edge_type, new_weight)

    def decay_all(
        self,
        factor: float,
        edge_type: Optional[EdgeType] = None,
    ) -> int:
        """Multiply every stored weight by *factor* (clamped to ``[0, 1]``).

        When *edge_type* is given, only edges of that type are affected.
        Returns the number of rows updated. Exposed so R094's IdleOrchestrator
        can run monthly decay without this module scheduling anything.
        """
        if factor < 0.0:
            raise ValueError("decay factor must be >= 0")
        now_iso = _to_rfc3339(datetime.now(timezone.utc))
        with self._write_lock:
            conn = self._conn()
            if edge_type is None:
                cur = conn.execute(
                    "UPDATE bullet_edges SET "
                    "weight = MIN(1.0, MAX(0.0, weight * ?)), updated_at = ?",
                    (factor, now_iso),
                )
            else:
                cur = conn.execute(
                    "UPDATE bullet_edges SET "
                    "weight = MIN(1.0, MAX(0.0, weight * ?)), updated_at = ? "
                    "WHERE edge_type = ?",
                    (factor, now_iso, edge_type.value),
                )
            conn.commit()
            return cur.rowcount

    def delete_by_bullet(self, bullet_id: str) -> int:
        """Cascade-delete every edge incident to *bullet_id*.

        Returns the number of rows removed.
        """
        if not bullet_id:
            return 0
        with self._write_lock:
            conn = self._conn()
            cur = conn.execute(
                "DELETE FROM bullet_edges WHERE from_id = ? OR to_id = ?",
                (bullet_id, bullet_id),
            )
            conn.commit()
            return cur.rowcount

    # -- reads ----------------------------------------------------------

    def get_weight(
        self,
        from_id: str,
        to_id: str,
        edge_type: EdgeType,
    ) -> float:
        """Return the stored weight (or ``0.0`` if the edge is absent)."""
        conn = self._conn()
        row = conn.execute(
            "SELECT weight FROM bullet_edges "
            "WHERE from_id = ? AND to_id = ? AND edge_type = ?",
            (from_id, to_id, edge_type.value),
        ).fetchone()
        if row is None:
            return 0.0
        return float(row[0])

    def out_edges(self, from_id: str) -> list[BulletEdge]:
        """Return every edge leaving *from_id*."""
        return self._query_edges(
            "SELECT from_id, to_id, edge_type, weight, updated_at "
            "FROM bullet_edges WHERE from_id = ?",
            (from_id,),
        )

    def in_edges(self, to_id: str) -> list[BulletEdge]:
        """Return every edge entering *to_id*."""
        return self._query_edges(
            "SELECT from_id, to_id, edge_type, weight, updated_at "
            "FROM bullet_edges WHERE to_id = ?",
            (to_id,),
        )

    def degree(self, bullet_id: str) -> int:
        """Total number of incident edges (in + out)."""
        conn = self._conn()
        row = conn.execute(
            "SELECT COUNT(*) FROM bullet_edges "
            "WHERE from_id = ? OR to_id = ?",
            (bullet_id, bullet_id),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def all_edges(self) -> list[BulletEdge]:
        """Return every edge in the store (diagnostic helper)."""
        return self._query_edges(
            "SELECT from_id, to_id, edge_type, weight, updated_at "
            "FROM bullet_edges",
            (),
        )

    # -- internals ------------------------------------------------------

    def _query_edges(self, sql: str, params: tuple) -> list[BulletEdge]:
        conn = self._conn()
        rows = conn.execute(sql, params).fetchall()
        out: list[BulletEdge] = []
        for row in rows:
            from_id, to_id, type_str, weight, updated_str = row
            edge_type = EdgeType.parse(type_str) or EdgeType.SUPERSEDES
            out.append(
                BulletEdge(
                    from_id=from_id,
                    to_id=to_id,
                    edge_type=edge_type,
                    weight=float(weight),
                    updated_at=_from_rfc3339(updated_str),
                )
            )
        return out


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def _to_rfc3339(dt: datetime) -> str:
    """Format *dt* as the RFC-3339 string the Rust side writes."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    # Match Rust ``chrono::Utc::now().to_rfc3339()`` which uses +00:00, not Z.
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "+00:00")


def _from_rfc3339(value: str) -> datetime:
    """Parse an RFC-3339 timestamp; tolerant to ``Z`` suffix."""
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        logger.debug("Failed to parse edge timestamp %r — falling back to now", value)
        return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Edge-writing helpers (pure glue; callable from multiple writer paths)
# ---------------------------------------------------------------------------


def write_supersedes_edge(
    store: SqliteEdgeStore,
    old_id: str,
    new_id: str,
) -> Optional[BulletEdge]:
    """Record an ``old → new`` Supersede edge with weight 1.0.

    Returns ``None`` when the IDs are equal or empty (no edge written).

    Exposed as a free function so it can be invoked both from the inline
    Curator flow AND from R094's Idle Orchestrator Curator executor without
    coupling the two implementations.
    """
    if not old_id or not new_id or old_id == new_id:
        return None
    return store.upsert_edge(old_id, new_id, EdgeType.SUPERSEDES, 1.0)


def record_co_recall(
    store: SqliteEdgeStore,
    bullet_ids: list[str],
    delta: float = 0.1,
    max_weight: float = 1.0,
) -> int:
    """Increment ``co_recalled_with`` edges for every unordered pair.

    Both directions (a→b and b→a) are written so the graph is undirected in
    effect. Duplicate IDs are deduped (first occurrence kept). Returns the
    total number of pair writes; singleton input is a no-op.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for bid in bullet_ids:
        if not bid or bid in seen:
            continue
        seen.add(bid)
        unique.append(bid)
    if len(unique) < 2:
        return 0
    count = 0
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            store.increment_weight(
                unique[i], unique[j], EdgeType.CO_RECALLED_WITH, delta, max_weight,
            )
            store.increment_weight(
                unique[j], unique[i], EdgeType.CO_RECALLED_WITH, delta, max_weight,
            )
            count += 2
    return count


def recompute_derived_from_edges(
    store: SqliteEdgeStore,
    bullets: list[tuple[str, list[SourceRef]]],
    min_weight: float = 0.0,
) -> int:
    """Recompute ``derived_from`` edges from SourceRef Jaccard overlap.

    Pure function — the caller (typically R094's IdleOrchestrator) decides
    when to run it. Returns the number of edges written. Bullets with empty
    ``sources`` are skipped.
    """
    keys: list[tuple[str, set[tuple[str, int]]]] = []
    for bid, sources in bullets:
        if not bid:
            continue
        keyset: set[tuple[str, int]] = {
            (s.conversation_id, s.turn_offset) for s in sources
        }
        keys.append((bid, keyset))

    count = 0
    for i in range(len(keys)):
        a_id, a_keys = keys[i]
        if not a_keys:
            continue
        for j in range(i + 1, len(keys)):
            b_id, b_keys = keys[j]
            if not b_keys or a_id == b_id:
                continue
            inter = len(a_keys & b_keys)
            if inter == 0:
                continue
            union = len(a_keys | b_keys)
            if union == 0:
                continue
            w = inter / union
            if w <= min_weight:
                continue
            store.upsert_edge(a_id, b_id, EdgeType.DERIVED_FROM, w)
            store.upsert_edge(b_id, a_id, EdgeType.DERIVED_FROM, w)
            count += 2
    return count
