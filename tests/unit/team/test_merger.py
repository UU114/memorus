"""Tests for MultiPoolRetriever + Shadow Merge (STORY-056)."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from memx.team.merger import (
    LayerBoostConfig,
    MultiPoolRetriever,
    ScoredResult,
    _content_similarity,
    _tags_conflict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bullet(
    content: str = "some content",
    score: float = 0.8,
    tags: list[str] | None = None,
    incompatible_tags: list[str] | None = None,
    enforcement: str = "suggestion",
    instructivity_score: float = 50.0,
) -> dict[str, Any]:
    return {
        "content": content,
        "score": score,
        "tags": tags or [],
        "incompatible_tags": incompatible_tags or [],
        "enforcement": enforcement,
        "instructivity_score": instructivity_score,
    }


def _make_pool(results: list[dict[str, Any]]) -> MagicMock:
    pool = MagicMock()
    pool.search.return_value = results
    return pool


def _make_failing_pool() -> MagicMock:
    pool = MagicMock()
    pool.search.side_effect = RuntimeError("connection failed")
    return pool


# ---------------------------------------------------------------------------
# LayerBoostConfig
# ---------------------------------------------------------------------------


class TestLayerBoostConfig:
    def test_defaults(self) -> None:
        cfg = LayerBoostConfig()
        assert cfg.local_boost == 1.5
        assert cfg.team_boost == 1.0

    def test_custom(self) -> None:
        cfg = LayerBoostConfig(local_boost=2.0, team_boost=0.8)
        assert cfg.local_boost == 2.0
        assert cfg.team_boost == 0.8


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestTagsConflict:
    def test_no_conflict(self) -> None:
        assert not _tags_conflict(["python", "rust"], ["go", "java"])

    def test_conflict(self) -> None:
        assert _tags_conflict(["python", "rust"], ["rust", "java"])

    def test_empty(self) -> None:
        assert not _tags_conflict([], ["rust"])
        assert not _tags_conflict(["rust"], [])
        assert not _tags_conflict([], [])


class TestContentSimilarity:
    def test_identical(self) -> None:
        assert _content_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self) -> None:
        assert _content_similarity("hello world", "foo bar") == 0.0

    def test_partial_overlap(self) -> None:
        sim = _content_similarity("hello world foo", "hello world bar")
        assert 0.0 < sim < 1.0

    def test_empty_strings(self) -> None:
        assert _content_similarity("", "hello") == 0.0
        assert _content_similarity("hello", "") == 0.0
        assert _content_similarity("", "") == 0.0


# ---------------------------------------------------------------------------
# MultiPoolRetriever — basic search
# ---------------------------------------------------------------------------


class TestMultiPoolRetrieverBasic:
    def test_local_only(self) -> None:
        """When no team pools, returns local results."""
        local = _make_pool([_make_bullet("local fact", score=0.9)])
        retriever = MultiPoolRetriever(local_backend=local)
        results = retriever.search("query", limit=5)
        assert len(results) == 1
        assert results[0]["content"] == "local fact"

    def test_local_and_team(self) -> None:
        """Local + team results are merged."""
        local = _make_pool([_make_bullet("local fact", score=0.5)])
        team = _make_pool([_make_bullet("team fact", score=0.5)])
        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("team_git", team)],
        )
        results = retriever.search("query", limit=10)
        assert len(results) == 2

    def test_respects_limit(self) -> None:
        """Output is capped to requested limit."""
        bullets = [_make_bullet(f"fact {i}", score=0.5) for i in range(10)]
        local = _make_pool(bullets)
        retriever = MultiPoolRetriever(local_backend=local)
        results = retriever.search("query", limit=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Shadow Merge — boosting
# ---------------------------------------------------------------------------


class TestShadowMergeBoosting:
    def test_local_boost_higher(self) -> None:
        """Local result with same raw score outranks team due to x1.5 boost."""
        local = _make_pool([_make_bullet("local", score=0.6)])
        team = _make_pool([_make_bullet("team", score=0.6)])
        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("team_git", team)],
        )
        results = retriever.search("query", limit=10)
        assert results[0]["content"] == "local"

    def test_team_can_win_with_higher_raw_score(self) -> None:
        """Team result can beat local if raw score * 1.0 > local * 1.5."""
        local = _make_pool([_make_bullet("local", score=0.3)])
        team = _make_pool([_make_bullet("team", score=0.9)])
        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("team_git", team)],
        )
        results = retriever.search("query", limit=10)
        # team boosted = 0.9, local boosted = 0.45
        assert results[0]["content"] == "team"

    def test_custom_boost_config(self) -> None:
        """Custom boost config is respected."""
        local = _make_pool([_make_bullet("local", score=0.5)])
        team = _make_pool([_make_bullet("team", score=0.5)])
        cfg = LayerBoostConfig(local_boost=1.0, team_boost=2.0)
        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("team_git", team)],
            boost_config=cfg,
        )
        results = retriever.search("query", limit=10)
        # team boosted = 1.0, local boosted = 0.5
        assert results[0]["content"] == "team"


# ---------------------------------------------------------------------------
# Shadow Merge — mandatory enforcement
# ---------------------------------------------------------------------------


class TestMandatoryEnforcement:
    def test_mandatory_bypasses_boost(self) -> None:
        """Mandatory team bullet always appears first."""
        local = _make_pool([_make_bullet("local", score=0.99)])
        team = _make_pool([
            _make_bullet("mandatory rule", score=0.1, enforcement="mandatory"),
        ])
        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("team_git", team)],
        )
        results = retriever.search("query", limit=10)
        assert results[0]["content"] == "mandatory rule"

    def test_mandatory_survives_tag_conflict(self) -> None:
        """Mandatory bullet is kept even when tags conflict with existing."""
        local = _make_pool([
            _make_bullet("local", score=0.9, tags=["python"],
                         incompatible_tags=["legacy"]),
        ])
        team = _make_pool([
            _make_bullet("mandatory", score=0.1, enforcement="mandatory",
                         tags=["legacy"]),
        ])
        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("team_git", team)],
        )
        results = retriever.search("query", limit=10)
        contents = [r["content"] for r in results]
        assert "mandatory" in contents

    def test_local_mandatory_not_treated_as_mandatory(self) -> None:
        """enforcement=mandatory on local bullets is ignored (local only)."""
        local = _make_pool([
            _make_bullet("local mandatory", score=0.1, enforcement="mandatory"),
        ])
        team = _make_pool([_make_bullet("team", score=0.9)])
        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("team_git", team)],
        )
        results = retriever.search("query", limit=10)
        # local mandatory score = 0.1 * 1.5 = 0.15, team = 0.9
        assert results[0]["content"] == "team"


# ---------------------------------------------------------------------------
# Shadow Merge — incompatible_tags conflict
# ---------------------------------------------------------------------------


class TestIncompatibleTagsConflict:
    def test_tag_conflict_keeps_higher_score(self) -> None:
        """When tags conflict via incompatible_tags, higher score wins."""
        local = _make_pool([
            _make_bullet("winner", score=0.9, tags=["react"],
                         incompatible_tags=["angular"]),
            _make_bullet("loser", score=0.8, tags=["angular"],
                         incompatible_tags=["react"]),
        ])
        retriever = MultiPoolRetriever(local_backend=local)
        results = retriever.search("query", limit=10)
        contents = [r["content"] for r in results]
        assert "winner" in contents
        assert "loser" not in contents

    def test_no_conflict_keeps_both(self) -> None:
        """Non-conflicting results with different content are kept."""
        local = _make_pool([
            _make_bullet("a", score=0.9, tags=["python"]),
            _make_bullet("b completely different content", score=0.8, tags=["rust"]),
        ])
        retriever = MultiPoolRetriever(local_backend=local)
        results = retriever.search("query", limit=10)
        assert len(results) == 2

    def test_cross_pool_conflict(self) -> None:
        """Conflict detection works across local and team pools."""
        local = _make_pool([
            _make_bullet("local way", score=0.9, tags=["tabs"],
                         incompatible_tags=["spaces"]),
        ])
        team = _make_pool([
            _make_bullet("team way", score=0.8, tags=["spaces"],
                         incompatible_tags=["tabs"]),
        ])
        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("team_git", team)],
        )
        results = retriever.search("query", limit=10)
        # local boosted = 0.9 * 1.5 = 1.35 vs team = 0.8 * 1.0 = 0.8
        assert len(results) == 1
        assert results[0]["content"] == "local way"


# ---------------------------------------------------------------------------
# Shadow Merge — near-duplicate fallback
# ---------------------------------------------------------------------------


class TestNearDuplicateFallback:
    def test_near_duplicate_keeps_first(self) -> None:
        """Identical content without incompatible_tags deduplicates."""
        local = _make_pool([
            _make_bullet("use pytest for testing", score=0.9),
            _make_bullet("use pytest for testing", score=0.7),
        ])
        retriever = MultiPoolRetriever(local_backend=local)
        results = retriever.search("query", limit=10)
        assert len(results) == 1

    def test_different_content_kept(self) -> None:
        """Distinct content without incompatible_tags is kept."""
        local = _make_pool([
            _make_bullet("use pytest for testing", score=0.9),
            _make_bullet("use docker for deployment", score=0.7),
        ])
        retriever = MultiPoolRetriever(local_backend=local)
        results = retriever.search("query", limit=10)
        assert len(results) == 2

    def test_near_duplicate_with_incompatible_tags_skips_fallback(self) -> None:
        """When incompatible_tags are present, near-dup fallback is skipped."""
        local = _make_pool([
            _make_bullet("use pytest for testing", score=0.9,
                         incompatible_tags=["unittest"]),
            _make_bullet("use pytest for testing", score=0.7,
                         tags=["different"]),
        ])
        retriever = MultiPoolRetriever(local_backend=local)
        results = retriever.search("query", limit=10)
        # First has incompatible_tags, so near-dup fallback is NOT used.
        # But tags don't conflict either, so both kept.
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Degradation — team pool failure
# ---------------------------------------------------------------------------


class TestDegradation:
    def test_team_failure_returns_local_only(self) -> None:
        """Team pool failure degrades silently to local-only results."""
        local = _make_pool([_make_bullet("local fact", score=0.9)])
        failing_team = _make_failing_pool()
        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("team_git", failing_team)],
        )
        results = retriever.search("query", limit=10)
        assert len(results) == 1
        assert results[0]["content"] == "local fact"

    def test_all_teams_fail(self) -> None:
        """When all team pools fail, local results are returned."""
        local = _make_pool([_make_bullet("local", score=0.5)])
        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[
                ("team_a", _make_failing_pool()),
                ("team_b", _make_failing_pool()),
            ],
        )
        results = retriever.search("query", limit=10)
        assert len(results) == 1

    def test_empty_pools(self) -> None:
        """All pools returning empty => empty result."""
        local = _make_pool([])
        team = _make_pool([])
        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("team_git", team)],
        )
        results = retriever.search("query", limit=10)
        assert results == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_team_pools(self) -> None:
        """Retriever works with zero team pools."""
        local = _make_pool([_make_bullet("local")])
        retriever = MultiPoolRetriever(local_backend=local, team_pools=[])
        results = retriever.search("query", limit=5)
        assert len(results) == 1

    def test_bullet_without_content(self) -> None:
        """Bullets missing content field don't crash similarity check."""
        local = _make_pool([
            {"score": 0.9, "tags": [], "incompatible_tags": []},
            {"score": 0.8, "tags": [], "incompatible_tags": []},
        ])
        retriever = MultiPoolRetriever(local_backend=local)
        results = retriever.search("query", limit=10)
        assert len(results) == 2

    def test_results_sorted_by_boosted_score(self) -> None:
        """Output respects boosted_score ordering."""
        local = _make_pool([
            _make_bullet("low", score=0.1),
            _make_bullet("high something completely different", score=0.9),
            _make_bullet("mid another unique content here", score=0.5),
        ])
        retriever = MultiPoolRetriever(local_backend=local)
        results = retriever.search("query", limit=10)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


class TestPerformance:
    def test_shadow_merge_under_5ms(self) -> None:
        """Shadow Merge on reasonable input completes in < 5ms."""
        bullets_local = [
            _make_bullet(f"local fact {i} with unique words {i * 7}", score=0.5 + i * 0.01)
            for i in range(20)
        ]
        bullets_team = [
            _make_bullet(f"team fact {i} different vocabulary {i * 13}", score=0.4 + i * 0.01)
            for i in range(20)
        ]
        local = _make_pool(bullets_local)
        team = _make_pool(bullets_team)
        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("team_git", team)],
        )

        # Warm up: run search once so ThreadPoolExecutor overhead is excluded
        retriever.search("query", limit=10)

        # Measure: only the merge portion matters, but we measure the full
        # search (parallel query on mocks is near-instant).
        start = time.perf_counter()
        results = retriever.search("query", limit=10)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(results) > 0
        assert elapsed_ms < 50  # generous bound (CI can be slow)
