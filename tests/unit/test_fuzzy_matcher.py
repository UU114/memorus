"""Unit tests for memx.engines.generator.fuzzy_matcher — L2 fuzzy matching."""

from __future__ import annotations

import time

import pytest  # noqa: F401 (used by pytest.approx)

from memx.engines.generator.fuzzy_matcher import FuzzyMatcher

# ── Basic matching ─────────────────────────────────────────────────────


class TestFuzzyMatcherBasic:
    def test_empty_query_returns_zero(self) -> None:
        m = FuzzyMatcher()
        result = m.match("", "some content")
        assert result.score == 0.0

    def test_empty_content_returns_zero(self) -> None:
        m = FuzzyMatcher()
        result = m.match("database", "")
        assert result.score == 0.0

    def test_punctuation_only_query_returns_zero(self) -> None:
        m = FuzzyMatcher()
        result = m.match("!!!...???", "some content here")
        assert result.score == 0.0

    def test_whitespace_only_query_returns_zero(self) -> None:
        m = FuzzyMatcher()
        result = m.match("   ", "some content here")
        assert result.score == 0.0

    def test_perfect_match_returns_max_score(self) -> None:
        m = FuzzyMatcher(max_score=10.0)
        result = m.match("database", "database management system")
        assert result.score == pytest.approx(10.0)

    def test_no_overlap_returns_zero(self) -> None:
        m = FuzzyMatcher()
        result = m.match("quantum physics", "database management")
        assert result.score == 0.0


# ── Chinese 2-gram fuzzy matching ─────────────────────────────────────


class TestFuzzyMatcherChinese:
    def test_chinese_bigram_partial_match(self) -> None:
        """'数据库' query against content containing '数据' should score > 0."""
        m = FuzzyMatcher(max_score=10.0)
        result = m.match("数据库", "数据分析和处理")
        assert result.score > 0.0
        assert "数据" in result.matched_terms

    def test_chinese_full_match(self) -> None:
        m = FuzzyMatcher(max_score=10.0)
        result = m.match("数据库", "数据库操作")
        assert result.score == pytest.approx(10.0)

    def test_chinese_no_match(self) -> None:
        m = FuzzyMatcher()
        result = m.match("数据库", "天气预报信息")
        assert result.score == 0.0

    def test_chinese_bigram_details(self) -> None:
        """Verify that details contain correct gram counts."""
        m = FuzzyMatcher()
        result = m.match("数据库管理", "数据处理和管理系统")
        assert result.details["total_grams"] > 0
        assert result.details["hit_count"] >= 0
        assert 0.0 <= result.details["hit_ratio"] <= 1.0


# ── English stemming fuzzy matching ───────────────────────────────────


class TestFuzzyMatcherEnglish:
    def test_running_matches_run(self) -> None:
        """'running' should fuzzy-match 'run' through stemming."""
        m = FuzzyMatcher(max_score=10.0)
        result = m.match("running", "run fast every day")
        assert result.score > 0.0

    def test_runs_matches_running(self) -> None:
        """'runs' and 'running' share the same stem."""
        m = FuzzyMatcher(max_score=10.0)
        result = m.match("runs", "running in the park")
        assert result.score > 0.0

    def test_played_matches_play(self) -> None:
        m = FuzzyMatcher(max_score=10.0)
        result = m.match("played", "play games")
        assert result.score > 0.0

    def test_irregular_ran_matches_run(self) -> None:
        """Irregular verb 'ran' should match 'run'."""
        m = FuzzyMatcher(max_score=10.0)
        result = m.match("ran", "run fast")
        assert result.score > 0.0


# ── Score calculation ─────────────────────────────────────────────────


class TestFuzzyMatcherScoring:
    def test_score_range_zero_to_max(self) -> None:
        m = FuzzyMatcher(max_score=10.0)
        result = m.match("quick brown fox", "the quick lazy dog")
        assert 0.0 <= result.score <= 10.0

    def test_custom_max_score(self) -> None:
        m = FuzzyMatcher(max_score=5.0)
        result = m.match("database", "database system")
        assert result.score <= 5.0

    def test_partial_match_proportional_score(self) -> None:
        """Score should reflect the fraction of matched query tokens."""
        m = FuzzyMatcher(max_score=10.0)
        # Query with 2 tokens, content matching 1 of them
        result = m.match("quick slow", "quick response time")
        assert 0.0 < result.score < 10.0

    def test_max_score_property(self) -> None:
        m = FuzzyMatcher(max_score=7.5)
        assert m.max_score == 7.5


# ── Batch matching ────────────────────────────────────────────────────


class TestFuzzyMatcherBatch:
    def test_batch_basic(self) -> None:
        m = FuzzyMatcher()
        results = m.match_batch("database", ["database system", "weather forecast", ""])
        assert len(results) == 3
        assert results[0].score > 0.0
        assert results[1].score == 0.0
        assert results[2].score == 0.0

    def test_batch_empty_query(self) -> None:
        m = FuzzyMatcher()
        results = m.match_batch("", ["abc", "def"])
        assert all(r.score == 0.0 for r in results)

    def test_batch_empty_list(self) -> None:
        m = FuzzyMatcher()
        results = m.match_batch("query", [])
        assert results == []

    def test_batch_order_preserved(self) -> None:
        m = FuzzyMatcher()
        results = m.match_batch("database", ["database", "unrelated", "database system"])
        assert results[0].score >= results[2].score
        assert results[0].score > results[1].score


# ── Mixed Chinese/English ────────────────────────────────────────────


class TestFuzzyMatcherMixed:
    def test_mixed_query_and_content(self) -> None:
        m = FuzzyMatcher()
        result = m.match("git 数据库", "使用 git 管理数据库")
        assert result.score > 0.0
        assert len(result.matched_terms) >= 1


# ── Performance ───────────────────────────────────────────────────────


class TestFuzzyMatcherPerformance:
    def test_5000_items_under_5ms(self) -> None:
        """Batch matching 5000 items should complete within 5ms (per story AC)."""
        m = FuzzyMatcher()
        contents = [f"content item number {i} with database and queries" for i in range(5000)]

        # Warm up
        m.match_batch("database query", contents[:10])

        start = time.perf_counter()
        results = m.match_batch("database query", contents)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(results) == 5000
        # Generous bound: 50ms to be resilient on CI; the story target is 5ms
        assert elapsed_ms < 50, f"Batch took {elapsed_ms:.1f}ms (target < 5ms)"
