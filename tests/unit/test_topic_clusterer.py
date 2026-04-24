"""Unit tests for the TopicClusterer + Louvain helpers (STORY-R097)."""

from __future__ import annotations

from memorus.core.engines.topic.clusterer import (
    BulletNode,
    TopicClusterer,
    build_adjacency_from_edges,
    build_adjacency_from_embeddings,
    cosine_similarity,
    louvain_communities,
)
from memorus.core.storage.edges import EdgeType, SqliteEdgeStore


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_cosine_similarity_basic():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([], [1.0]) == 0.0
    # negative cosine clamped to 0.0
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == 0.0


def test_louvain_empty_graph():
    assert louvain_communities({}) == []


def test_louvain_singletons_only():
    # Three disconnected nodes — each is its own community.
    adj = {"a": {}, "b": {}, "c": {}}
    comms = louvain_communities(adj)
    sizes = [len(c) for c in comms]
    assert sizes == [1, 1, 1]


def test_louvain_two_clique_graph():
    # Two triangles connected by a single bridge edge.
    adj = {
        "a": {"b": 1.0, "c": 1.0},
        "b": {"a": 1.0, "c": 1.0},
        "c": {"a": 1.0, "b": 1.0, "d": 0.1},
        "d": {"c": 0.1, "e": 1.0, "f": 1.0},
        "e": {"d": 1.0, "f": 1.0},
        "f": {"d": 1.0, "e": 1.0},
    }
    comms = louvain_communities(adj)
    # Louvain should produce two communities of size 3 each.
    sizes = sorted(len(c) for c in comms)
    assert sizes == [3, 3]
    left = set(comms[0])
    right = set(comms[1])
    assert ({"a", "b", "c"} == left or {"a", "b", "c"} == right)
    assert ({"d", "e", "f"} == left or {"d", "e", "f"} == right)


def test_louvain_is_deterministic():
    """Same adjacency → same ordered communities on repeated runs."""
    adj = {
        "n1": {"n2": 1.0, "n3": 1.0},
        "n2": {"n1": 1.0, "n3": 1.0},
        "n3": {"n1": 1.0, "n2": 1.0},
        "n4": {"n5": 1.0},
        "n5": {"n4": 1.0},
    }
    a = louvain_communities(adj)
    b = louvain_communities(adj)
    assert a == b


# ---------------------------------------------------------------------------
# Graph build paths
# ---------------------------------------------------------------------------


def test_build_adjacency_from_embeddings_respects_threshold():
    nodes = [
        BulletNode(bullet_id="a", embedding=[1.0, 0.0]),
        BulletNode(bullet_id="b", embedding=[0.95, 0.1]),  # highly similar
        BulletNode(bullet_id="c", embedding=[0.0, 1.0]),   # orthogonal
    ]
    adj = build_adjacency_from_embeddings(nodes, threshold=0.6)
    assert "b" in adj["a"]
    assert "a" in adj["b"]
    # c should not be connected to either.
    assert "c" not in adj["a"]
    assert "c" not in adj["b"]


def test_build_adjacency_from_edges_via_sqlite():
    store = SqliteEdgeStore("")
    store.upsert_edge("a", "b", EdgeType.DERIVED_FROM, 0.8)
    store.upsert_edge("b", "c", EdgeType.CO_RECALLED_WITH, 0.5)
    nodes = [BulletNode(bullet_id=x) for x in ("a", "b", "c", "d")]
    adj = build_adjacency_from_edges(nodes, store, threshold=0.0)
    assert "b" in adj["a"]
    assert "a" in adj["b"]  # symmetrised
    assert "c" in adj["b"]
    assert "b" in adj["c"]
    assert adj["d"] == {}
    store.close()


# ---------------------------------------------------------------------------
# TopicClusterer facade
# ---------------------------------------------------------------------------


def _nodes(names: list[str]) -> list[BulletNode]:
    # Embeddings are near-identical so clustering collapses them all together.
    return [
        BulletNode(bullet_id=n, content=f"content-{n}", embedding=[1.0, 0.0])
        for n in names
    ]


def test_clusterer_filters_archived_and_low_weight():
    nodes = [
        BulletNode(
            bullet_id="a",
            content="a",
            embedding=[1.0, 0.0],
            archived=True,
        ),
        BulletNode(
            bullet_id="b",
            content="b",
            embedding=[1.0, 0.0],
            decay_weight=0.05,
        ),
        BulletNode(bullet_id="c", content="c", embedding=[1.0, 0.0]),
    ]
    clusterer = TopicClusterer(min_cluster_size=1)
    # Only "c" is active; singleton so no cluster produced when min=2.
    clusters = clusterer.cluster(nodes)
    # With min_cluster_size=1 the single active node becomes a cluster.
    assert len(clusters) == 1
    assert clusters[0].bullet_ids == ["c"]


def test_clusterer_min_size_filter():
    clusterer = TopicClusterer(min_cluster_size=3)
    nodes = _nodes(["a", "b"])
    clusters = clusterer.cluster(nodes)
    # Only two connected nodes — below the min-size threshold.
    assert clusters == []


def test_clusterer_produces_one_cluster_for_connected_group():
    clusterer = TopicClusterer(min_cluster_size=3)
    nodes = _nodes(["a", "b", "c", "d"])
    clusters = clusterer.cluster(nodes)
    assert len(clusters) == 1
    assert sorted(clusters[0].bullet_ids) == ["a", "b", "c", "d"]


def test_clusterer_uses_edge_store_when_provided():
    store = SqliteEdgeStore("")
    store.upsert_edge("x1", "x2", EdgeType.DERIVED_FROM, 0.9)
    store.upsert_edge("x2", "x3", EdgeType.DERIVED_FROM, 0.9)
    store.upsert_edge("x1", "x3", EdgeType.DERIVED_FROM, 0.9)
    nodes = [BulletNode(bullet_id=x) for x in ("x1", "x2", "x3")]
    clusterer = TopicClusterer(min_cluster_size=3, edge_store=store)
    clusters = clusterer.cluster(nodes)
    assert len(clusters) == 1
    assert sorted(clusters[0].bullet_ids) == ["x1", "x2", "x3"]
    store.close()


def test_clusterer_performance_budget():
    """Hard upper bound: 1000 nodes complete clustering in under 30s."""
    import time

    # Build two dense clusters of 500 each — realistic load.
    nodes: list[BulletNode] = []
    for i in range(500):
        nodes.append(BulletNode(bullet_id=f"g1_{i}", embedding=[1.0, 0.0]))
    for i in range(500):
        nodes.append(BulletNode(bullet_id=f"g2_{i}", embedding=[0.0, 1.0]))

    start = time.monotonic()
    clusterer = TopicClusterer(min_cluster_size=3)
    clusters = clusterer.cluster(nodes)
    elapsed = time.monotonic() - start

    assert elapsed < 30.0, f"clustering took {elapsed:.1f}s (>30s budget)"
    # Two big groups → at least two clusters.
    assert len(clusters) >= 2
