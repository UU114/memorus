"""Topic clusterer: build similarity graph + Louvain community detection.

Given a list of active bullets (weight > 0.1, not archived) this module

1. builds an undirected similarity graph using R096's :class:`SqliteEdgeStore`
   when available, falling back to cosine similarity on pre-computed
   embeddings,
2. runs a deterministic Louvain community detection over the adjacency
   dict (standard modularity optimisation, ~160 LOC),
3. filters communities by ``min_cluster_size``.

Determinism is achieved by (a) iterating nodes in sorted-id order and
(b) breaking ties on the partition move using sorted-neighbour order,
so both Python and Rust implementations produce the same partition
given identical edges. Louvain is not globally deterministic under
arbitrary rng, but given a fixed adjacency dict and sorted iteration
order, the output clusters are stable across runs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from memorus.core.storage.edges import EdgeType, SqliteEdgeStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass
class BulletNode:
    """Minimal view of a bullet required for clustering."""

    bullet_id: str
    content: str = ""
    decay_weight: float = 1.0
    archived: bool = False
    embedding: Optional[Sequence[float]] = None


@dataclass
class Cluster:
    """A Louvain community that survived the min-size filter."""

    bullet_ids: list[str] = field(default_factory=list)
    internal_edge_weight: float = 0.0


# ---------------------------------------------------------------------------
# Cosine helper (used for the embedding fallback path)
# ---------------------------------------------------------------------------


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity clamped to ``[0, 1]`` (negatives pulled to 0)."""
    if not a or not b:
        return 0.0
    # Iterate up to min length — defensive against mismatched dims.
    n = min(len(a), len(b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        x = float(a[i])
        y = float(b[i])
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    sim = dot / math.sqrt(na * nb)
    if sim < 0.0:
        return 0.0
    if sim > 1.0:
        return 1.0
    return sim


# ---------------------------------------------------------------------------
# Graph build
# ---------------------------------------------------------------------------


def build_adjacency_from_edges(
    nodes: list[BulletNode],
    edge_store: SqliteEdgeStore,
    threshold: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Build undirected adjacency dict from an :class:`SqliteEdgeStore`.

    Edges are aggregated across every :class:`EdgeType`; weights are summed
    and then clamped to ``[0, 1]``. Self-loops are discarded. Only edges
    whose endpoints are both in *nodes* are kept.
    """
    id_set = {n.bullet_id for n in nodes}
    adj: dict[str, dict[str, float]] = {n.bullet_id: {} for n in nodes}

    for node in nodes:
        try:
            for edge in edge_store.out_edges(node.bullet_id):
                if edge.to_id == node.bullet_id:
                    continue
                if edge.to_id not in id_set:
                    continue
                w = float(edge.weight)
                if w <= threshold:
                    continue
                cur = adj[node.bullet_id].get(edge.to_id, 0.0)
                adj[node.bullet_id][edge.to_id] = min(1.0, cur + w)
        except Exception as e:  # defensive — edge store IO should never kill cluster
            logger.warning("edge_store.out_edges(%s) failed: %s", node.bullet_id, e)

    # Symmetrise: take max(a→b, b→a) as the undirected weight.
    sym: dict[str, dict[str, float]] = {n.bullet_id: {} for n in nodes}
    for u, nbrs in adj.items():
        for v, w in nbrs.items():
            cur = max(w, adj.get(v, {}).get(u, 0.0))
            sym[u][v] = cur
            sym[v][u] = cur
    return sym


def build_adjacency_from_embeddings(
    nodes: list[BulletNode],
    threshold: float = 0.6,
) -> dict[str, dict[str, float]]:
    """Fallback path: pairwise cosine similarity over embeddings.

    Only edges with ``sim > threshold`` are kept. Runs in O(n^2) — good
    up to the documented 1k-bullet benchmark (<30s).
    """
    with_emb = [n for n in nodes if n.embedding]
    adj: dict[str, dict[str, float]] = {n.bullet_id: {} for n in nodes}
    for i in range(len(with_emb)):
        a = with_emb[i]
        for j in range(i + 1, len(with_emb)):
            b = with_emb[j]
            sim = cosine_similarity(a.embedding or (), b.embedding or ())
            if sim <= threshold:
                continue
            adj[a.bullet_id][b.bullet_id] = sim
            adj[b.bullet_id][a.bullet_id] = sim
    return adj


# ---------------------------------------------------------------------------
# Louvain community detection
# ---------------------------------------------------------------------------


def _modularity_gain(
    node: str,
    community: frozenset[str],
    adj: dict[str, dict[str, float]],
    degree: dict[str, float],
    community_weight: dict[int, float],
    comm_id_of: dict[str, int],
    m2: float,
) -> float:
    """Δmodularity if *node* joins the community identified by *community*.

    Uses the standard Louvain formula — this function is called in the
    inner loop so is kept allocation-light.
    """
    if m2 <= 0.0:
        return 0.0
    k_i = degree[node]
    # Sum of edge weights from node to the community members.
    k_i_in = 0.0
    for nbr, w in adj[node].items():
        if nbr in community and nbr != node:
            k_i_in += w
    # Σ_tot for the community *excluding* node if node already in it.
    if community and next(iter(community)) in comm_id_of:
        cid = comm_id_of[next(iter(community))]
        sigma_tot = community_weight.get(cid, 0.0)
        if node in community:
            sigma_tot -= degree[node]
    else:
        sigma_tot = sum(degree[v] for v in community)
        if node in community:
            sigma_tot -= degree[node]
    return (k_i_in / m2) - (sigma_tot * k_i) / (m2 * m2)


def louvain_communities(
    adj: dict[str, dict[str, float]],
    *,
    max_passes: int = 20,
    tolerance: float = 1.0e-7,
) -> list[list[str]]:
    """Run Louvain modularity optimisation on an undirected weighted graph.

    Returns clusters as lists of node ids, sorted ascending inside each
    cluster. Clusters themselves are sorted by (a) size desc, (b) first
    member asc — ensuring determinism for tests and cross-language parity.

    Implementation: single-phase modularity optimisation (aka the "first
    phase" of Louvain). For node counts up to ~1000 that is sufficient —
    multi-level collapse would only refine things further and is not
    required by the acceptance criteria. The loop iterates until no node
    moves improve modularity (classical guarantee of local convergence).
    """
    if not adj:
        return []

    # Canonical node order — determinism.
    nodes: list[str] = sorted(adj.keys())
    node_set = set(nodes)

    # Guard against isolated / missing neighbours.
    safe_adj: dict[str, dict[str, float]] = {}
    for n in nodes:
        nbrs = {k: float(v) for k, v in adj.get(n, {}).items() if k in node_set and k != n}
        safe_adj[n] = nbrs

    degree: dict[str, float] = {
        n: sum(w for w in safe_adj[n].values()) for n in nodes
    }
    m2 = sum(degree.values())  # 2m
    if m2 <= 0.0:
        # Nothing connected — every node is its own cluster. Return only
        # singletons, caller will filter min_cluster_size.
        return [[n] for n in nodes]

    # Initial partition — each node in its own community, id = sorted index.
    comm_id_of: dict[str, int] = {n: i for i, n in enumerate(nodes)}
    members_of: dict[int, set[str]] = {i: {n} for i, n in enumerate(nodes)}
    community_weight: dict[int, float] = {i: degree[n] for i, n in enumerate(nodes)}

    for _pass in range(max_passes):
        moved = False
        for n in nodes:
            current_cid = comm_id_of[n]
            # Candidate communities = node's own + those of its neighbours.
            candidates: dict[int, float] = {}
            for nbr, w in safe_adj[n].items():
                cid = comm_id_of[nbr]
                candidates[cid] = candidates.get(cid, 0.0) + w
            # Include current community so "stay" is evaluated.
            candidates.setdefault(current_cid, 0.0)

            k_i = degree[n]
            best_cid = current_cid
            best_gain = 0.0

            for cid in sorted(candidates.keys()):  # deterministic tie-break
                k_i_in = candidates[cid]
                # Σ_tot excluding n if it is already a member.
                sigma_tot = community_weight.get(cid, 0.0)
                if comm_id_of[n] == cid:
                    sigma_tot -= k_i
                gain = (k_i_in / m2) - (sigma_tot * k_i) / (m2 * m2)
                if gain > best_gain + tolerance:
                    best_gain = gain
                    best_cid = cid

            if best_cid != current_cid:
                moved = True
                # Remove n from current community.
                members_of[current_cid].discard(n)
                community_weight[current_cid] -= k_i
                if not members_of[current_cid]:
                    del members_of[current_cid]
                    del community_weight[current_cid]
                # Add n to new community.
                members_of.setdefault(best_cid, set()).add(n)
                community_weight[best_cid] = community_weight.get(best_cid, 0.0) + k_i
                comm_id_of[n] = best_cid
        if not moved:
            break

    clusters: list[list[str]] = []
    for cid, mem in members_of.items():
        clusters.append(sorted(mem))

    # Stable sort: size desc, then first-id asc.
    clusters.sort(key=lambda c: (-len(c), c[0] if c else ""))
    return clusters


# ---------------------------------------------------------------------------
# TopicClusterer facade
# ---------------------------------------------------------------------------


class TopicClusterer:
    """Wraps graph-build + Louvain + min-size filter into one callable."""

    def __init__(
        self,
        *,
        min_cluster_size: int = 3,
        similarity_edge_threshold: float = 0.6,
        edge_store: Optional[SqliteEdgeStore] = None,
    ) -> None:
        if min_cluster_size < 1:
            raise ValueError("min_cluster_size must be >= 1")
        self._min_cluster_size = min_cluster_size
        self._threshold = similarity_edge_threshold
        self._edge_store = edge_store

    def cluster(self, bullets: Iterable[BulletNode]) -> list[Cluster]:
        """Filter bullets → build graph → Louvain → filter by min size."""
        active = [
            b for b in bullets
            if not b.archived and b.decay_weight > 0.1
        ]
        if not active:
            return []

        # Prefer the edge store when available; fall back to embeddings.
        if self._edge_store is not None:
            adj = build_adjacency_from_edges(active, self._edge_store, threshold=0.0)
            # If the edge store yields nothing usable, try embeddings.
            if not any(adj.values()):
                adj = build_adjacency_from_embeddings(active, threshold=self._threshold)
        else:
            adj = build_adjacency_from_embeddings(active, threshold=self._threshold)

        if not adj:
            return []
        # When there are no edges at all, Louvain would emit every node as
        # its own singleton community — callers filter by min_cluster_size
        # later, so still hand off to Louvain (it handles empty-edge graphs).

        communities = louvain_communities(adj)

        clusters: list[Cluster] = []
        for ids in communities:
            if len(ids) < self._min_cluster_size:
                continue
            internal = 0.0
            for i, u in enumerate(ids):
                for v in ids[i + 1:]:
                    internal += adj.get(u, {}).get(v, 0.0)
            clusters.append(Cluster(bullet_ids=ids, internal_edge_weight=internal))
        return clusters


__all__ = [
    "BulletNode",
    "Cluster",
    "TopicClusterer",
    "build_adjacency_from_edges",
    "build_adjacency_from_embeddings",
    "cosine_similarity",
    "louvain_communities",
]


# Re-export for convenience.
EdgeType = EdgeType
