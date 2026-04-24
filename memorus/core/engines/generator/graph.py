"""Graph-aware retrieval expansion (STORY-R096).

This module is the pure-compute counterpart of
:mod:`memorus.core.storage.edges`: it consumes the seed bullets produced by
the base Generator pipeline plus a minimal :class:`EdgeProvider` and emits
1-hop Adamic-Adar neighbors. It mirrors
``memorus-r/memorus-ace/src/generator/graph.rs`` exactly so Python and Rust
produce identical scores over the same data.

Design:
    - :class:`EdgeProvider` is a tiny Protocol so the Generator never takes
      a hard dependency on SQLite (tests inject an in-memory stub).
    - :func:`expand_neighbors` is deterministic given the same input.
    - :func:`apply_graph_scores` adds ``graph_score_weight * graph_score``
      to every :class:`ScoredBullet` that appears in *neighbors* without
      mutating the original list.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable, Protocol

from memorus.core.engines.generator.score_merger import ScoredBullet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EdgeProvider protocol
# ---------------------------------------------------------------------------


class EdgeProvider(Protocol):
    """Minimal read-only view of a bullet-edge graph for expansion.

    Implementations should return empty lists / zero rather than raising on
    transient backend failures, so graph expansion degrades gracefully.
    """

    def out_edges(self, from_id: str) -> list[tuple[str, float]]:
        """Return every ``(to_id, edge_weight)`` leaving *from_id*."""
        ...

    def degree(self, bullet_id: str) -> int:
        """Total number of incident edges for *bullet_id* (in + out)."""
        ...


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def adamic_adar_weight(edge_weight: float, neighbor_degree: int) -> float:
    """Weighted Adamic-Adar term: ``edge_weight / ln(degree + 2)``.

    Natural log. Safe for ``degree == 0`` because ``ln(2) > 0``.
    """
    denom = math.log(neighbor_degree + 2.0)
    if denom <= 0.0:
        return 0.0
    return edge_weight / denom


@dataclass(frozen=True)
class GraphNeighbor:
    """Output of 1-hop expansion."""

    bullet_id: str
    graph_score: float


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------


def expand_neighbors(
    seeds: list[tuple[str, float]],
    provider: EdgeProvider,
    k_expand: int,
    exclude_ids: Iterable[str] | None = None,
) -> list[GraphNeighbor]:
    """Run 1-hop graph expansion over *seeds* using *provider*.

    *seeds* are ``(bullet_id, seed_score)`` pairs (typically the top-K
    :class:`ScoredBullet` ids + ``final_score``). Returns up to *k_expand*
    new neighbors **plus** any seeds/excluded IDs that accumulated
    ``graph_score > 0`` (so callers can re-rank the existing results).

    Determinism: when multiple neighbors tie on score, Python's stable sort
    preserves the insertion order, which itself depends on the order of
    ``seeds`` and the provider's ``out_edges`` iteration.
    """
    if not seeds:
        return []

    excluded: set[str] = {sid for sid, _ in seeds}
    if exclude_ids:
        excluded.update(exclude_ids)

    if k_expand <= 0 and not excluded:
        return []

    accum: dict[str, float] = {}

    for seed_id, seed_score in seeds:
        out = provider.out_edges(seed_id)
        for n_id, edge_weight in out:
            if n_id == seed_id:
                # Defensive: skip self-loops.
                continue
            degree = provider.degree(n_id)
            term = seed_score * adamic_adar_weight(edge_weight, degree)
            if term <= 0.0:
                continue
            accum[n_id] = accum.get(n_id, 0.0) + term

    if not accum:
        return []

    seed_hits: list[tuple[str, float]] = []
    new_hits: list[tuple[str, float]] = []
    for bid, score in accum.items():
        if bid in excluded:
            seed_hits.append((bid, score))
        else:
            new_hits.append((bid, score))

    # Cap *new* hits; excluded (seed) hits always survive so callers can
    # re-rank top-K even when k_expand is small.
    new_hits.sort(key=lambda p: p[1], reverse=True)
    new_hits = new_hits[:k_expand] if k_expand > 0 else []

    combined = seed_hits + new_hits
    combined.sort(key=lambda p: p[1], reverse=True)

    return [GraphNeighbor(bullet_id=bid, graph_score=score) for bid, score in combined]


# ---------------------------------------------------------------------------
# Score merge
# ---------------------------------------------------------------------------


def apply_graph_scores(
    results: list[ScoredBullet],
    neighbors: list[GraphNeighbor],
    graph_score_weight: float,
) -> list[ScoredBullet]:
    """Return a new list with ``final_score += weight * graph_score`` applied.

    The input *results* list is NOT mutated. Neighbors whose ``bullet_id``
    is not present in *results* are ignored (the caller is responsible for
    synthesising :class:`ScoredBullet` stubs for genuinely new bullets).

    When ``graph_score_weight <= 0`` or *neighbors* is empty, returns a
    shallow copy of *results* with no modifications — which is how the
    "disabled" code-path stays byte-identical to the pre-R096 Generator.
    """
    if graph_score_weight <= 0.0 or not neighbors:
        return list(results)

    boosts: dict[str, float] = {
        n.bullet_id: graph_score_weight * n.graph_score for n in neighbors
    }

    updated: list[ScoredBullet] = []
    for sb in results:
        boost = boosts.get(sb.bullet_id, 0.0)
        if boost == 0.0:
            updated.append(sb)
        else:
            # Dataclass instances are mutable — build a new one to avoid
            # surprising callers that hold references to the originals.
            updated.append(
                ScoredBullet(
                    bullet_id=sb.bullet_id,
                    content=sb.content,
                    final_score=sb.final_score + boost,
                    keyword_score=sb.keyword_score,
                    semantic_score=sb.semantic_score,
                    decay_weight=sb.decay_weight,
                    recency_boost=sb.recency_boost,
                    metadata=dict(sb.metadata),
                )
            )

    updated.sort(key=lambda sb: sb.final_score, reverse=True)
    return updated


def scored_bullet_for_new_neighbor(
    bullet_id: str,
    content: str,
    graph_score: float,
    graph_score_weight: float,
    *,
    decay_weight: float = 1.0,
    metadata: dict | None = None,
) -> ScoredBullet:
    """Build a :class:`ScoredBullet` stub for a *new* neighbor pulled in by
    graph expansion. The caller must supply the underlying content/metadata.

    ``final_score = graph_score_weight * graph_score`` (no base score).
    """
    return ScoredBullet(
        bullet_id=bullet_id,
        content=content,
        final_score=graph_score_weight * graph_score,
        keyword_score=0.0,
        semantic_score=0.0,
        decay_weight=decay_weight,
        recency_boost=1.0,
        metadata=dict(metadata or {}),
    )


# ---------------------------------------------------------------------------
# EdgeStore adapter
# ---------------------------------------------------------------------------


class EdgeStoreProvider:
    """Thin adapter turning :class:`SqliteEdgeStore` into an :class:`EdgeProvider`.

    Defined here (not in ``storage.edges``) so ``EdgeProvider`` stays a pure
    protocol and the storage module has no dependency on Generator types.
    """

    def __init__(self, store: "object") -> None:
        # Typed as object to avoid importing SqliteEdgeStore (optional dep).
        self._store = store

    def out_edges(self, from_id: str) -> list[tuple[str, float]]:
        try:
            edges = self._store.out_edges(from_id)  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001 — degrade gracefully
            logger.warning("EdgeStoreProvider.out_edges failed for %r: %s", from_id, e)
            return []
        return [(e.to_id, e.weight) for e in edges]

    def degree(self, bullet_id: str) -> int:
        try:
            return int(self._store.degree(bullet_id))  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001 — degrade gracefully
            logger.warning("EdgeStoreProvider.degree failed for %r: %s", bullet_id, e)
            return 0
