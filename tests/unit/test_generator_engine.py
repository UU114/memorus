"""Unit tests for memx.engines.generator.engine — GeneratorEngine."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import pytest

from memx.core.config import RetrievalConfig
from memx.core.engines.generator.engine import BulletForSearch, GeneratorEngine
from memx.core.engines.generator.metadata_matcher import MetadataInfo
from memx.core.engines.generator.vector_searcher import VectorMatch, VectorSearcher

# ── Helper fixtures ──────────────────────────────────────────────────────

_NOW = datetime(2026, 2, 27, 12, 0, 0, tzinfo=timezone.utc)


def _make_bullet(
    bid: str,
    content: str = "test content",
    tools: list[str] | None = None,
    entities: list[str] | None = None,
    tags: list[str] | None = None,
    days_ago: float = 30.0,
    decay_weight: float = 1.0,
) -> BulletForSearch:
    """Create a BulletForSearch for testing."""
    return BulletForSearch(
        bullet_id=bid,
        content=content,
        metadata=MetadataInfo(
            related_tools=tools or [],
            key_entities=entities or [],
            tags=tags or [],
        ),
        created_at=_NOW - timedelta(days=days_ago),
        decay_weight=decay_weight,
    )


def _make_vector_searcher(
    results: list[VectorMatch] | None = None,
    available: bool = True,
) -> VectorSearcher:
    """Create a VectorSearcher with a mock search function."""
    if not available:
        return VectorSearcher(search_fn=None)

    def search_fn(
        query: str,
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if results is None:
            return []
        return [
            {"id": m.bullet_id, "score": m.score, "content": m.content}
            for m in results
        ]

    return VectorSearcher(search_fn=search_fn)


# ── BulletForSearch tests ────────────────────────────────────────────────


class TestBulletForSearch:
    """BulletForSearch dataclass construction."""

    def test_default_fields(self) -> None:
        b = BulletForSearch(bullet_id="b1")
        assert b.bullet_id == "b1"
        assert b.content == ""
        assert b.metadata == MetadataInfo()
        assert b.created_at is None
        assert b.decay_weight == 1.0
        assert b.extra == {}

    def test_full_construction(self) -> None:
        meta = MetadataInfo(related_tools=["git"], key_entities=["React"], tags=["python"])
        b = BulletForSearch(
            bullet_id="b2",
            content="some content",
            metadata=meta,
            created_at=_NOW,
            decay_weight=0.8,
            extra={"source": "test"},
        )
        assert b.bullet_id == "b2"
        assert b.content == "some content"
        assert b.metadata.related_tools == ["git"]
        assert b.decay_weight == 0.8
        assert b.extra == {"source": "test"}


# ── Mode detection tests ─────────────────────────────────────────────────


class TestModeDetection:
    """Mode property reflects embedding availability."""

    def test_full_mode_with_vector_searcher(self) -> None:
        vs = _make_vector_searcher(available=True)
        engine = GeneratorEngine(vector_searcher=vs)
        assert engine.mode == "full"

    def test_degraded_mode_without_vector_searcher(self) -> None:
        engine = GeneratorEngine()  # default VectorSearcher has no search_fn
        assert engine.mode == "degraded"

    def test_degraded_mode_with_none_search_fn(self) -> None:
        vs = VectorSearcher(search_fn=None)
        engine = GeneratorEngine(vector_searcher=vs)
        assert engine.mode == "degraded"

    def test_mode_reflects_dynamic_availability(self) -> None:
        """Mode should change when VectorSearcher availability changes."""
        vs = VectorSearcher(search_fn=None)
        engine = GeneratorEngine(vector_searcher=vs)
        assert engine.mode == "degraded"

        # Simulate embedding recovery by setting a search_fn
        vs._search_fn = lambda **kw: []
        assert engine.mode == "full"


# ── Full mode search tests ───────────────────────────────────────────────


class TestSearchFullMode:
    """Full mode: L1 + L2 + L3 + L4 -> ScoreMerger."""

    def test_basic_search_returns_results(self) -> None:
        vs = _make_vector_searcher(
            results=[VectorMatch(bullet_id="b1", score=0.9, content="git rebase")],
            available=True,
        )
        engine = GeneratorEngine(vector_searcher=vs)
        bullets = [_make_bullet("b1", "Use git rebase for interactive rebasing", tools=["git"])]
        results = engine.search("git rebase", bullets)
        assert len(results) == 1
        assert results[0].bullet_id == "b1"
        assert results[0].final_score > 0.0

    def test_results_sorted_descending(self) -> None:
        vs = _make_vector_searcher(
            results=[
                VectorMatch(bullet_id="b1", score=0.3),
                VectorMatch(bullet_id="b2", score=0.9),
            ],
            available=True,
        )
        engine = GeneratorEngine(vector_searcher=vs)
        bullets = [
            _make_bullet("b1", "some unrelated content"),
            _make_bullet("b2", "git rebase interactive workflow", tools=["git"]),
        ]
        results = engine.search("git rebase", bullets)
        assert len(results) == 2
        assert results[0].final_score >= results[1].final_score

    def test_limit_truncates_results(self) -> None:
        vs = _make_vector_searcher(available=True)
        engine = GeneratorEngine(vector_searcher=vs)
        bullets = [_make_bullet(f"b{i}", f"content {i}") for i in range(10)]
        results = engine.search("content", bullets, limit=3)
        assert len(results) <= 3

    def test_empty_query_returns_empty(self) -> None:
        vs = _make_vector_searcher(available=True)
        engine = GeneratorEngine(vector_searcher=vs)
        bullets = [_make_bullet("b1", "some content")]
        results = engine.search("", bullets)
        assert results == []

    def test_empty_bullets_returns_empty(self) -> None:
        vs = _make_vector_searcher(available=True)
        engine = GeneratorEngine(vector_searcher=vs)
        results = engine.search("query", [])
        assert results == []

    def test_semantic_score_included_in_full_mode(self) -> None:
        """In full mode, VectorSearcher results contribute to final score."""
        vs = _make_vector_searcher(
            results=[VectorMatch(bullet_id="b1", score=0.95)],
            available=True,
        )
        engine = GeneratorEngine(vector_searcher=vs)
        # Bullet content deliberately does NOT match the query keywords
        bullets = [_make_bullet("b1", "completely unrelated text")]
        results = engine.search("quantum physics", bullets)
        assert len(results) == 1
        # Final score should include semantic contribution
        assert results[0].semantic_score == 0.95

    def test_filters_passed_to_vector_searcher(self) -> None:
        """Filters should be forwarded to VectorSearcher."""
        captured: dict[str, Any] = {}

        def search_fn(
            query: str,
            limit: int = 20,
            filters: dict[str, Any] | None = None,
        ) -> list[dict[str, Any]]:
            captured["filters"] = filters
            return []

        vs = VectorSearcher(search_fn=search_fn)
        engine = GeneratorEngine(vector_searcher=vs)
        bullets = [_make_bullet("b1", "content")]
        engine.search("test", bullets, filters={"category": "tech"})
        assert captured["filters"] == {"category": "tech"}


# ── Degraded mode search tests ───────────────────────────────────────────


class TestSearchDegradedMode:
    """Degraded mode: L1 + L2 + L3 only, no L4."""

    def test_degraded_mode_skips_l4(self) -> None:
        engine = GeneratorEngine()  # no vector searcher
        bullets = [_make_bullet("b1", "Use git rebase for interactive rebasing", tools=["git"])]
        results = engine.search("git rebase", bullets)
        assert len(results) == 1
        assert results[0].semantic_score == 0.0

    def test_degraded_mode_still_ranks(self) -> None:
        engine = GeneratorEngine()
        bullets = [
            _make_bullet("b1", "git rebase interactive", tools=["git"]),
            _make_bullet("b2", "completely unrelated text"),
        ]
        results = engine.search("git rebase", bullets)
        assert len(results) == 2
        assert results[0].bullet_id == "b1"
        assert results[0].final_score > results[1].final_score

    def test_degraded_warning_logged_once(self, caplog: pytest.LogCaptureFixture) -> None:
        """Degradation warning should appear only on first search."""
        engine = GeneratorEngine()
        bullets = [_make_bullet("b1", "test content")]

        with caplog.at_level(logging.WARNING, logger="memx.core.engines.generator.engine"):
            engine.search("test", bullets)
            engine.search("test", bullets)
            engine.search("test", bullets)

        degraded_msgs = [r for r in caplog.records if "degraded mode" in r.message]
        assert len(degraded_msgs) == 1

    def test_degraded_keyword_weight_is_one(self) -> None:
        """In degraded mode, all score comes from keyword matching."""
        engine = GeneratorEngine()
        bullets = [_make_bullet("b1", "exact keyword match here", days_ago=30)]
        results = engine.search("exact keyword", bullets)
        assert len(results) == 1
        # Semantic score must be 0 in degraded mode
        assert results[0].semantic_score == 0.0
        # Keyword score must be > 0 (exact matches)
        assert results[0].keyword_score > 0.0


# ── Automatic recovery tests ─────────────────────────────────────────────


class TestAutoRecovery:
    """Engine should auto-switch back to full mode when embedding recovers."""

    def test_recovery_from_degraded_to_full(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        vs = VectorSearcher(search_fn=None)
        engine = GeneratorEngine(vector_searcher=vs)
        bullets = [_make_bullet("b1", "test content")]

        # First search: degraded
        with caplog.at_level(logging.WARNING, logger="memx.core.engines.generator.engine"):
            engine.search("test", bullets)
        assert engine.mode == "degraded"

        # Simulate embedding recovery
        vs._search_fn = lambda **kw: []

        with caplog.at_level(logging.INFO, logger="memx.core.engines.generator.engine"):
            engine.search("test", bullets)
        assert engine.mode == "full"

        # Recovery info log should appear
        recovery_msgs = [r for r in caplog.records if "recovered" in r.message]
        assert len(recovery_msgs) == 1

    def test_degraded_flag_resets_on_recovery(self) -> None:
        """After recovery, subsequent degradation should log warning again."""
        vs = VectorSearcher(search_fn=None)
        engine = GeneratorEngine(vector_searcher=vs)
        bullets = [_make_bullet("b1", "test content")]

        # First degradation
        engine.search("test", bullets)
        assert engine._degraded_logged is True

        # Recovery
        vs._search_fn = lambda **kw: []
        engine.search("test", bullets)
        assert engine._degraded_logged is False

        # Second degradation
        vs._search_fn = None
        engine.search("test", bullets)
        assert engine._degraded_logged is True


# ── Per-matcher error isolation tests ─────────────────────────────────────


class TestErrorIsolation:
    """Individual matcher failures should not break the pipeline."""

    def test_l1_failure_isolated(self, caplog: pytest.LogCaptureFixture) -> None:
        engine = GeneratorEngine()
        bullets = [_make_bullet("b1", "test content")]

        with (
            patch.object(
                engine._exact_matcher, "match_batch", side_effect=RuntimeError("L1 exploded")
            ),
            caplog.at_level(logging.WARNING, logger="memx.core.engines.generator.engine"),
        ):
            results = engine.search("test", bullets)

        # Search should still succeed (with L1 contribution = 0)
        assert len(results) == 1
        warning_msgs = [r for r in caplog.records if "L1 ExactMatcher failed" in r.message]
        assert len(warning_msgs) == 1

    def test_l2_failure_isolated(self, caplog: pytest.LogCaptureFixture) -> None:
        engine = GeneratorEngine()
        bullets = [_make_bullet("b1", "test content")]

        with (
            patch.object(
                engine._fuzzy_matcher, "match_batch", side_effect=RuntimeError("L2 exploded")
            ),
            caplog.at_level(logging.WARNING, logger="memx.core.engines.generator.engine"),
        ):
            results = engine.search("test", bullets)

        assert len(results) == 1
        warning_msgs = [r for r in caplog.records if "L2 FuzzyMatcher failed" in r.message]
        assert len(warning_msgs) == 1

    def test_l3_failure_isolated(self, caplog: pytest.LogCaptureFixture) -> None:
        engine = GeneratorEngine()
        bullets = [_make_bullet("b1", "test content", tools=["git"])]

        with (
            patch.object(
                engine._metadata_matcher, "match", side_effect=RuntimeError("L3 exploded")
            ),
            caplog.at_level(logging.WARNING, logger="memx.core.engines.generator.engine"),
        ):
            results = engine.search("git", bullets)

        assert len(results) == 1
        warning_msgs = [r for r in caplog.records if "L3 MetadataMatcher failed" in r.message]
        assert len(warning_msgs) == 1

    def test_l4_failure_isolated(self, caplog: pytest.LogCaptureFixture) -> None:
        vs = _make_vector_searcher(available=True)
        engine = GeneratorEngine(vector_searcher=vs)
        bullets = [_make_bullet("b1", "test content")]

        with (
            patch.object(
                engine._vector_searcher, "search", side_effect=RuntimeError("L4 exploded")
            ),
            caplog.at_level(logging.WARNING, logger="memx.core.engines.generator.engine"),
        ):
            results = engine.search("test", bullets)

        assert len(results) == 1
        warning_msgs = [r for r in caplog.records if "L4 VectorSearcher failed" in r.message]
        assert len(warning_msgs) == 1

    def test_multiple_matchers_fail_gracefully(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Even if L1 and L2 both fail, L3 should still contribute."""
        engine = GeneratorEngine()
        bullets = [_make_bullet("b1", "test content", tags=["test"])]

        with (
            patch.object(
                engine._exact_matcher, "match_batch", side_effect=RuntimeError("L1 fail")
            ),
            patch.object(
                engine._fuzzy_matcher, "match_batch", side_effect=RuntimeError("L2 fail")
            ),
            caplog.at_level(logging.WARNING, logger="memx.core.engines.generator.engine"),
        ):
            results = engine.search("test", bullets)

        # L3 metadata match (tag "test" matches query "test") should still work
        assert len(results) == 1
        assert results[0].keyword_score > 0.0  # L3 score contributed

    def test_all_matchers_fail_returns_zero_scores(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If all matchers fail, results should have zero scores."""
        engine = GeneratorEngine()
        bullets = [_make_bullet("b1", "test content")]

        with (
            patch.object(
                engine._exact_matcher, "match_batch", side_effect=RuntimeError("L1")
            ),
            patch.object(
                engine._fuzzy_matcher, "match_batch", side_effect=RuntimeError("L2")
            ),
            patch.object(
                engine._metadata_matcher, "match", side_effect=RuntimeError("L3")
            ),
        ):
            results = engine.search("test", bullets)

        assert len(results) == 1
        assert results[0].keyword_score == 0.0
        assert results[0].final_score == 0.0


# ── Config tests ──────────────────────────────────────────────────────────


class TestGeneratorEngineConfig:
    """Configuration forwarding and defaults."""

    def test_default_config(self) -> None:
        engine = GeneratorEngine()
        # ScoreMerger should use default RetrievalConfig
        assert engine._score_merger.config.keyword_weight == 0.6
        assert engine._score_merger.config.semantic_weight == 0.4

    def test_custom_config(self) -> None:
        config = RetrievalConfig(keyword_weight=0.8, semantic_weight=0.2)
        engine = GeneratorEngine(config=config)
        assert abs(engine._score_merger.keyword_weight - 0.8) < 1e-9
        assert abs(engine._score_merger.semantic_weight - 0.2) < 1e-9


# ── Integration-style tests ──────────────────────────────────────────────


class TestEndToEnd:
    """End-to-end search scenarios combining all matchers."""

    def test_keyword_only_ranking(self) -> None:
        """Bullets with better keyword matches should rank higher."""
        engine = GeneratorEngine()
        bullets = [
            _make_bullet("b1", "git rebase interactive squash fixup"),
            _make_bullet("b2", "python list comprehension"),
            _make_bullet("b3", "git merge conflict resolution"),
        ]
        results = engine.search("git rebase", bullets)
        # b1 should rank highest (both "git" and "rebase" match)
        assert results[0].bullet_id == "b1"

    def test_metadata_boosts_ranking(self) -> None:
        """Bullets with matching metadata should rank higher than content-only."""
        engine = GeneratorEngine()
        # Use content with low L1/L2 scores so the metadata delta is visible
        # before normalization clamps at MAX_KEYWORD_SCORE (35.0)
        bullets = [
            _make_bullet("b1", "tips for version control"),
            _make_bullet("b2", "tips for version control", tools=["git"], tags=["git"]),
        ]
        results = engine.search("git", bullets)
        # b2 has metadata matches (tool + tag), so should rank higher
        assert results[0].bullet_id == "b2"
        assert results[0].keyword_score > results[1].keyword_score

    def test_decay_weight_affects_ranking(self) -> None:
        """Lower decay weight should reduce final score."""
        engine = GeneratorEngine()
        bullets = [
            _make_bullet("b1", "git rebase", decay_weight=1.0),
            _make_bullet("b2", "git rebase", decay_weight=0.1),
        ]
        results = engine.search("git rebase", bullets)
        assert results[0].bullet_id == "b1"
        assert results[0].final_score > results[1].final_score

    def test_recency_boost_affects_ranking(self) -> None:
        """Recent bullets should get a boost."""
        engine = GeneratorEngine()
        bullets = [
            _make_bullet("b1", "git rebase", days_ago=30),
            _make_bullet("b2", "git rebase", days_ago=1),
        ]
        results = engine.search("git rebase", bullets, now=_NOW)
        # b2 is more recent, should rank higher
        assert results[0].bullet_id == "b2"
        assert results[0].recency_boost > results[1].recency_boost

    def test_chinese_query(self) -> None:
        """Chinese queries should work with exact and fuzzy matching."""
        engine = GeneratorEngine()
        bullets = [
            _make_bullet("b1", "使用数据库连接池进行高效查询"),
            _make_bullet("b2", "Python list comprehension examples"),
        ]
        results = engine.search("数据库连接", bullets)
        assert results[0].bullet_id == "b1"
        assert results[0].keyword_score > results[1].keyword_score

    def test_mixed_language_query(self) -> None:
        """Mixed Chinese/English queries should match both."""
        engine = GeneratorEngine()
        bullets = [
            _make_bullet("b1", "使用git进行版本控制"),
            _make_bullet("b2", "unrelated content"),
        ]
        results = engine.search("git版本控制", bullets)
        assert results[0].bullet_id == "b1"

    def test_full_mode_with_vector_results(self) -> None:
        """Full pipeline test with all four layers contributing."""
        vs = _make_vector_searcher(
            results=[
                VectorMatch(bullet_id="b1", score=0.9),
                VectorMatch(bullet_id="b2", score=0.1),
            ],
            available=True,
        )
        engine = GeneratorEngine(vector_searcher=vs)
        bullets = [
            _make_bullet("b1", "git rebase interactive", tools=["git"]),
            _make_bullet("b2", "unrelated text"),
        ]
        results = engine.search("git rebase", bullets)
        assert len(results) == 2
        # b1 should rank first: high keyword + high semantic + metadata
        assert results[0].bullet_id == "b1"
        assert results[0].semantic_score == 0.9
        assert results[0].keyword_score > 0.0

    def test_extra_metadata_forwarded(self) -> None:
        """BulletForSearch.extra should appear in ScoredBullet.metadata."""
        engine = GeneratorEngine()
        bullets = [
            BulletForSearch(
                bullet_id="b1",
                content="test content",
                extra={"source": "manual", "version": 2},
            ),
        ]
        results = engine.search("test", bullets)
        assert len(results) == 1
        assert results[0].metadata == {"source": "manual", "version": 2}


# ── STORY-031 补充测试：中文全管线、L4 贡献验证、降级模式单独隔离 ──────


class TestEndToEndChinese:
    """中文全管线端到端测试。"""

    def test_chinese_only_search(self) -> None:
        """Pure Chinese query and content should work in degraded mode."""
        engine = GeneratorEngine()
        bullets = [
            _make_bullet("b1", "使用数据库连接池进行高效率数据查询"),
            _make_bullet("b2", "Python列表推导式使用方法"),
            _make_bullet("b3", "数据库索引优化技巧和实践经验"),
        ]
        results = engine.search("数据库查询优化", bullets)
        # b1 and b3 contain "数据库" and related terms
        assert results[0].bullet_id in ("b1", "b3")
        assert results[0].final_score > results[-1].final_score

    def test_chinese_metadata_boost(self) -> None:
        """Chinese metadata should contribute to scoring."""
        engine = GeneratorEngine()
        bullets = [
            _make_bullet("b1", "数据库操作", tags=["数据库"]),
            _make_bullet("b2", "数据库操作"),  # same content, no tags
        ]
        results = engine.search("数据库", bullets)
        # b1 with tag match should rank higher
        assert results[0].bullet_id == "b1"
        assert results[0].keyword_score > results[1].keyword_score


class TestFullModeL4Contribution:
    """Full 模式下 L4 语义贡献验证。"""

    def test_l4_contribution_changes_ranking(self) -> None:
        """L4 semantic score can change ranking when keyword scores are equal."""
        vs = _make_vector_searcher(
            results=[
                VectorMatch(bullet_id="b1", score=0.2),
                VectorMatch(bullet_id="b2", score=0.95),
            ],
            available=True,
        )
        engine = GeneratorEngine(vector_searcher=vs)
        # Same keyword content, but different L4 semantic scores
        bullets = [
            _make_bullet("b1", "generic technical content"),
            _make_bullet("b2", "generic technical content"),
        ]
        results = engine.search("technical content", bullets)
        assert len(results) == 2
        # b2 has much higher semantic score -> should rank first
        assert results[0].bullet_id == "b2"
        assert results[0].semantic_score == pytest.approx(0.95)

    def test_full_mode_all_layers_contribute(self) -> None:
        """All four layers should contribute in full mode."""
        vs = _make_vector_searcher(
            results=[VectorMatch(bullet_id="b1", score=0.9)],
            available=True,
        )
        engine = GeneratorEngine(vector_searcher=vs)
        bullets = [
            _make_bullet(
                "b1",
                "Use git rebase for interactive history editing",
                tools=["git"],
                tags=["git"],
            ),
        ]
        results = engine.search("git rebase", bullets)
        assert len(results) == 1
        r = results[0]
        # L1 (exact): "git" and "rebase" -> 30.0
        # L2 (fuzzy): stems match -> > 0
        # L3 (metadata): tool "git" + tag "git" -> 7.0
        # L4 (semantic): 0.9
        assert r.keyword_score > 30.0  # L1 + L2 + L3
        assert r.semantic_score == pytest.approx(0.9)
        assert r.final_score > 0.0


class TestDegradedModeIsolation:
    """降级模式下各 matcher 的独立性。"""

    def test_degraded_l1_only_match(self) -> None:
        """In degraded mode, only L1 matching when content only has exact matches."""
        engine = GeneratorEngine()
        bullets = [_make_bullet("b1", "exactly_this_term_only", days_ago=30)]
        results = engine.search("exactly_this_term_only", bullets)
        assert len(results) == 1
        # Should have L1 exact match score
        assert results[0].keyword_score > 0.0
        # No semantic score in degraded
        assert results[0].semantic_score == 0.0

    def test_degraded_all_layers_independent(self) -> None:
        """In degraded mode, L1+L2+L3 should all contribute independently."""
        engine = GeneratorEngine()
        bullets = [
            _make_bullet(
                "b1",
                "git rebase interactive",
                tools=["git"],
                tags=["git"],
                days_ago=30,
            ),
        ]
        results = engine.search("git", bullets)
        assert len(results) == 1
        # L1 contributes (exact "git" match)
        # L2 contributes (fuzzy "git" match)
        # L3 contributes (tool "git" + tag "git")
        # keyword_score should be > 15 (L1 alone)
        assert results[0].keyword_score > 15.0


class TestGeneratorEngineMultiBulletRanking:
    """多 bullet 综合排序场景。"""

    def test_ten_bullets_ranking(self) -> None:
        """10 bullets should be ranked correctly with varied content."""
        engine = GeneratorEngine()
        bullets = [
            _make_bullet("b0", "completely unrelated Python stuff"),
            _make_bullet("b1", "git basics and setup"),
            _make_bullet("b2", "advanced git rebase techniques", tools=["git"]),
            _make_bullet("b3", "git rebase interactive squash", tools=["git"], tags=["git"]),
            _make_bullet("b4", "docker container management"),
            _make_bullet("b5", "git merge vs rebase comparison", tools=["git"]),
            _make_bullet("b6", "unrelated machine learning content"),
            _make_bullet("b7", "git rebase workflow best practices", tags=["git"]),
            _make_bullet("b8", "database query optimization"),
            _make_bullet("b9", "git stash and git rebase tips", tools=["git-rebase"]),
        ]
        results = engine.search("git rebase", bullets, limit=5)
        assert len(results) <= 5
        # Top results should be git-rebase-related bullets
        top_ids = {r.bullet_id for r in results}
        assert "b3" in top_ids  # has both keywords + metadata
        # b0, b4, b6, b8 should not be in top 5
        for bid in ("b0", "b4", "b6", "b8"):
            assert bid not in top_ids
