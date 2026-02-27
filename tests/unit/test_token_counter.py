"""Unit tests for memx.utils.token_counter — TokenBudgetTrimmer."""

from __future__ import annotations

import pytest

from memx.engines.generator.score_merger import ScoredBullet
from memx.utils.token_counter import TokenBudgetTrimmer


# ── Helper factories ────────────────────────────────────────────────────


def _bullet(
    bid: str,
    content: str = "",
    final_score: float = 1.0,
) -> ScoredBullet:
    """Create a minimal ScoredBullet for testing."""
    return ScoredBullet(
        bullet_id=bid,
        content=content,
        final_score=final_score,
        keyword_score=0.0,
        semantic_score=0.0,
        decay_weight=1.0,
        recency_boost=1.0,
    )


# ── Token estimation tests ─────────────────────────────────────────────


class TestEstimateTokens:
    """Token estimation via character-count heuristic."""

    def test_empty_string(self) -> None:
        trimmer = TokenBudgetTrimmer()
        assert trimmer.estimate_tokens("") == 0

    def test_pure_ascii(self) -> None:
        trimmer = TokenBudgetTrimmer(chars_per_token=4.0)
        # 20 chars / 4 = 5 tokens
        assert trimmer.estimate_tokens("a" * 20) == 5

    def test_pure_ascii_rounding(self) -> None:
        trimmer = TokenBudgetTrimmer(chars_per_token=4.0)
        # 7 chars / 4 = 1.75 -> int(1.75) = 1
        assert trimmer.estimate_tokens("abcdefg") == 1

    def test_pure_cjk(self) -> None:
        trimmer = TokenBudgetTrimmer()
        # 6 CJK chars / 1.5 = 4.0 tokens
        assert trimmer.estimate_tokens("数据库管理系") == 4

    def test_mixed_cjk_and_ascii(self) -> None:
        trimmer = TokenBudgetTrimmer(chars_per_token=4.0)
        # "数据库" = 3 CJK chars -> 3/1.5 = 2.0
        # "test" = 4 ASCII chars -> 4/4.0 = 1.0
        # Total = 3.0 -> int(3.0) = 3
        assert trimmer.estimate_tokens("数据库test") == 3

    def test_custom_chars_per_token(self) -> None:
        trimmer = TokenBudgetTrimmer(chars_per_token=2.0)
        # 10 chars / 2 = 5
        assert trimmer.estimate_tokens("abcdefghij") == 5

    def test_single_char(self) -> None:
        trimmer = TokenBudgetTrimmer(chars_per_token=4.0)
        # 1 char / 4 = 0.25 -> int(0.25) = 0
        assert trimmer.estimate_tokens("a") == 0

    def test_whitespace_counted(self) -> None:
        trimmer = TokenBudgetTrimmer(chars_per_token=4.0)
        # "hello world" = 11 chars / 4 = 2.75 -> int = 2
        assert trimmer.estimate_tokens("hello world") == 2


# ── Trim basic behaviour tests ──────────────────────────────────────────


class TestTrimBasic:
    """Core trimming logic: budget and max_results."""

    def test_empty_input(self) -> None:
        trimmer = TokenBudgetTrimmer()
        assert trimmer.trim([]) == []

    def test_single_result_within_budget(self) -> None:
        trimmer = TokenBudgetTrimmer(token_budget=100)
        results = [_bullet("b1", "short text", final_score=1.0)]
        trimmed = trimmer.trim(results)
        assert len(trimmed) == 1
        assert trimmed[0].bullet_id == "b1"

    def test_all_results_fit(self) -> None:
        trimmer = TokenBudgetTrimmer(token_budget=10000, max_results=10)
        results = [
            _bullet("b1", "alpha", final_score=3.0),
            _bullet("b2", "beta", final_score=2.0),
            _bullet("b3", "gamma", final_score=1.0),
        ]
        trimmed = trimmer.trim(results)
        assert len(trimmed) == 3

    def test_max_results_cap(self) -> None:
        trimmer = TokenBudgetTrimmer(token_budget=10000, max_results=2)
        results = [
            _bullet("b1", "x", final_score=3.0),
            _bullet("b2", "y", final_score=2.0),
            _bullet("b3", "z", final_score=1.0),
        ]
        trimmed = trimmer.trim(results)
        assert len(trimmed) == 2
        assert trimmed[0].bullet_id == "b1"
        assert trimmed[1].bullet_id == "b2"

    def test_budget_limits_results(self) -> None:
        # Each content = 40 chars -> 40/4 = 10 tokens
        content = "a" * 40
        trimmer = TokenBudgetTrimmer(token_budget=25, max_results=10)
        results = [
            _bullet("b1", content, final_score=3.0),
            _bullet("b2", content, final_score=2.0),
            _bullet("b3", content, final_score=1.0),
        ]
        trimmed = trimmer.trim(results)
        # b1 = 10 tokens (fits, cumulative=10)
        # b2 = 10 tokens (fits, cumulative=20)
        # b3 = 10 tokens (cumulative would be 30 > 25, stop)
        assert len(trimmed) == 2
        assert trimmed[0].bullet_id == "b1"
        assert trimmed[1].bullet_id == "b2"

    def test_exact_budget_boundary(self) -> None:
        # 20 chars / 4 = 5 tokens per bullet
        content = "a" * 20
        trimmer = TokenBudgetTrimmer(token_budget=10, max_results=10)
        results = [
            _bullet("b1", content, final_score=3.0),
            _bullet("b2", content, final_score=2.0),
            _bullet("b3", content, final_score=1.0),
        ]
        trimmed = trimmer.trim(results)
        # b1 = 5 tokens (fits, cumulative=5)
        # b2 = 5 tokens (cumulative=10, exactly at budget, fits)
        # b3 = 5 tokens (cumulative would be 15 > 10, stop)
        assert len(trimmed) == 2


# ── Guarantee: at least 1 result ────────────────────────────────────────


class TestGuaranteeAtLeastOne:
    """Even if the first result exceeds budget, at least 1 must be returned."""

    def test_single_result_exceeds_budget(self) -> None:
        # Content = 400 chars -> 100 tokens, budget = 10
        trimmer = TokenBudgetTrimmer(token_budget=10)
        results = [_bullet("b1", "x" * 400, final_score=1.0)]
        trimmed = trimmer.trim(results)
        assert len(trimmed) == 1
        assert trimmed[0].bullet_id == "b1"

    def test_all_results_exceed_budget(self) -> None:
        trimmer = TokenBudgetTrimmer(token_budget=1, max_results=5)
        results = [
            _bullet("b1", "x" * 400, final_score=3.0),
            _bullet("b2", "y" * 400, final_score=2.0),
            _bullet("b3", "z" * 400, final_score=1.0),
        ]
        trimmed = trimmer.trim(results)
        # Only the highest-scoring result is returned
        assert len(trimmed) == 1
        assert trimmed[0].bullet_id == "b1"

    def test_zero_budget_still_returns_one(self) -> None:
        trimmer = TokenBudgetTrimmer(token_budget=0)
        results = [_bullet("b1", "some content", final_score=1.0)]
        trimmed = trimmer.trim(results)
        assert len(trimmed) == 1


# ── Sorting / ordering tests ───────────────────────────────────────────


class TestTrimOrdering:
    """Trimmer should process results by final_score descending."""

    def test_preserves_descending_order(self) -> None:
        trimmer = TokenBudgetTrimmer(token_budget=10000, max_results=10)
        results = [
            _bullet("b1", "aaa", final_score=3.0),
            _bullet("b2", "bbb", final_score=1.0),
            _bullet("b3", "ccc", final_score=2.0),
        ]
        trimmed = trimmer.trim(results)
        assert len(trimmed) == 3
        assert trimmed[0].bullet_id == "b1"
        assert trimmed[1].bullet_id == "b3"
        assert trimmed[2].bullet_id == "b2"

    def test_unsorted_input_handled(self) -> None:
        """Even if input is not pre-sorted, trimmer sorts internally."""
        trimmer = TokenBudgetTrimmer(token_budget=10000, max_results=2)
        results = [
            _bullet("low", "x", final_score=0.1),
            _bullet("high", "y", final_score=0.9),
            _bullet("mid", "z", final_score=0.5),
        ]
        trimmed = trimmer.trim(results)
        assert len(trimmed) == 2
        assert trimmed[0].bullet_id == "high"
        assert trimmed[1].bullet_id == "mid"


# ── Property access tests ──────────────────────────────────────────────


class TestTrimmerProperties:
    """Verify read-only property access."""

    def test_default_properties(self) -> None:
        trimmer = TokenBudgetTrimmer()
        assert trimmer.token_budget == 2000
        assert trimmer.max_results == 5
        assert trimmer.chars_per_token == 4.0

    def test_custom_properties(self) -> None:
        trimmer = TokenBudgetTrimmer(
            token_budget=500,
            max_results=3,
            chars_per_token=2.0,
        )
        assert trimmer.token_budget == 500
        assert trimmer.max_results == 3
        assert trimmer.chars_per_token == 2.0


# ── CJK-specific budget tests ──────────────────────────────────────────


class TestCJKBudget:
    """CJK content uses a higher token density (fewer chars per token)."""

    def test_cjk_consumes_more_budget(self) -> None:
        trimmer = TokenBudgetTrimmer(chars_per_token=4.0)
        # 6 CJK chars -> 6/1.5 = 4 tokens
        # 6 ASCII chars -> 6/4.0 = 1 token (int)
        assert trimmer.estimate_tokens("数据库管理系") == 4
        assert trimmer.estimate_tokens("abcdef") == 1

    def test_budget_reached_faster_with_cjk(self) -> None:
        # Each CJK content: 15 chars -> 15/1.5 = 10 tokens
        cjk_content = "数据库管理系统测试用例关系型索引优化"  # 15 CJK chars
        trimmer = TokenBudgetTrimmer(token_budget=15, max_results=10)
        results = [
            _bullet("b1", cjk_content, final_score=3.0),
            _bullet("b2", cjk_content, final_score=2.0),
        ]
        trimmed = trimmer.trim(results)
        # b1 = 10 tokens (fits), b2 = 10 tokens (cumulative 20 > 15, stop)
        assert len(trimmed) == 1


# ── Integration-style test ──────────────────────────────────────────────


class TestTrimIntegration:
    """Realistic scenario with varied content lengths and scores."""

    def test_realistic_scenario(self) -> None:
        trimmer = TokenBudgetTrimmer(token_budget=50, max_results=5)
        results = [
            _bullet("b1", "a" * 80, final_score=5.0),   # 20 tokens
            _bullet("b2", "b" * 60, final_score=4.0),   # 15 tokens
            _bullet("b3", "c" * 40, final_score=3.0),   # 10 tokens
            _bullet("b4", "d" * 20, final_score=2.0),   # 5 tokens
            _bullet("b5", "e" * 200, final_score=1.0),  # 50 tokens
        ]
        trimmed = trimmer.trim(results)
        # b1: 20 tokens (cumulative=20, fits)
        # b2: 15 tokens (cumulative=35, fits)
        # b3: 10 tokens (cumulative=45, fits)
        # b4: 5 tokens  (cumulative=50, fits — exactly at budget)
        # b5: 50 tokens (cumulative=100, exceeds budget, stop)
        assert len(trimmed) == 4
        assert [b.bullet_id for b in trimmed] == ["b1", "b2", "b3", "b4"]

    def test_mixed_content_scenario(self) -> None:
        trimmer = TokenBudgetTrimmer(token_budget=30, max_results=5)
        results = [
            _bullet("b1", "数据库管理系统测试用例关系型索引优化", final_score=5.0),  # 15 CJK -> 10 tokens
            _bullet("b2", "a" * 40, final_score=4.0),   # 40/4 = 10 tokens
            _bullet("b3", "a" * 60, final_score=3.0),   # 60/4 = 15 tokens
        ]
        trimmed = trimmer.trim(results)
        # b1: 10 tokens (cumulative=10, fits)
        # b2: 10 tokens (cumulative=20, fits)
        # b3: 15 tokens (cumulative=35 > 30, stop)
        assert len(trimmed) == 2
        assert trimmed[0].bullet_id == "b1"
        assert trimmed[1].bullet_id == "b2"
