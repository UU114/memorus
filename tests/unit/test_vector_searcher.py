"""Unit tests for memx.engines.generator.vector_searcher — VectorSearcher."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from memx.core.engines.generator.vector_searcher import (
    VectorMatch,
    VectorSearcher,
    _clamp,
    _normalize_score,
)

# -- Helper factories -------------------------------------------------------


def _make_raw_results(
    items: list[dict[str, Any]],
    wrapper: str | None = "results",
) -> Any:
    """Build a raw result payload similar to mem0 search output."""
    if wrapper:
        return {wrapper: items}
    return items


def _mock_search_fn(
    return_value: Any = None,
    side_effect: Any = None,
) -> MagicMock:
    """Create a mock search function with configurable behavior."""
    fn = MagicMock()
    if side_effect is not None:
        fn.side_effect = side_effect
    elif return_value is not None:
        fn.return_value = return_value
    else:
        fn.return_value = {"results": []}
    return fn


# -- Test 1: available property ---------------------------------------------


class TestAvailableProperty:
    def test_available_with_search_fn(self) -> None:
        """VectorSearcher with a search_fn is available."""
        searcher = VectorSearcher(search_fn=lambda **kw: {"results": []})
        assert searcher.available is True

    def test_unavailable_without_search_fn(self) -> None:
        """VectorSearcher without a search_fn is not available."""
        searcher = VectorSearcher()
        assert searcher.available is False

    def test_unavailable_with_none(self) -> None:
        """VectorSearcher(search_fn=None) is not available."""
        searcher = VectorSearcher(search_fn=None)
        assert searcher.available is False


# -- Test 2: graceful degradation when unavailable ---------------------------


class TestGracefulDegradation:
    def test_search_returns_empty_when_unavailable(self) -> None:
        """search() returns empty list when search_fn is None."""
        searcher = VectorSearcher()
        result = searcher.search("test query")
        assert result == []

    def test_search_returns_empty_when_none_fn(self) -> None:
        """search() returns empty list when explicitly None."""
        searcher = VectorSearcher(search_fn=None)
        result = searcher.search("some query", limit=10, filters={"user": "abc"})
        assert result == []


# -- Test 3: search_fn exception handling ------------------------------------


class TestExceptionHandling:
    def test_search_fn_exception_returns_empty(self) -> None:
        """search_fn raises -> empty list, no exception propagated."""
        fn = _mock_search_fn(side_effect=RuntimeError("connection lost"))
        searcher = VectorSearcher(search_fn=fn)
        result = searcher.search("test")
        assert result == []

    def test_search_fn_exception_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """search_fn raises -> WARNING is logged."""
        fn = _mock_search_fn(side_effect=ValueError("bad embedding"))
        searcher = VectorSearcher(search_fn=fn)
        with caplog.at_level(logging.WARNING):
            searcher.search("test")
        assert "VectorSearcher.search() failed" in caplog.text
        assert "bad embedding" in caplog.text

    def test_search_fn_timeout_returns_empty(self) -> None:
        """search_fn raises TimeoutError -> empty list."""
        fn = _mock_search_fn(side_effect=TimeoutError("timed out"))
        searcher = VectorSearcher(search_fn=fn)
        result = searcher.search("test")
        assert result == []


# -- Test 4: normal search path ---------------------------------------------


class TestNormalSearch:
    def test_basic_search_results(self) -> None:
        """Normal search returns VectorMatch objects with correct fields."""
        raw = _make_raw_results([
            {"id": "b1", "score": 0.95, "memory": "Use git rebase carefully"},
            {"id": "b2", "score": 0.80, "memory": "Always run tests first"},
        ])
        fn = _mock_search_fn(return_value=raw)
        searcher = VectorSearcher(search_fn=fn)

        results = searcher.search("git workflow")
        assert len(results) == 2
        assert all(isinstance(r, VectorMatch) for r in results)

        assert results[0].bullet_id == "b1"
        assert results[0].score == pytest.approx(0.95)
        assert results[0].content == "Use git rebase carefully"

        assert results[1].bullet_id == "b2"
        assert results[1].score == pytest.approx(0.80)
        assert results[1].content == "Always run tests first"

    def test_search_passes_parameters(self) -> None:
        """search() passes query, limit, and filters to search_fn."""
        fn = _mock_search_fn(return_value={"results": []})
        searcher = VectorSearcher(search_fn=fn)

        searcher.search("my query", limit=5, filters={"user_id": "u1"})
        fn.assert_called_once_with(
            query="my query", limit=5, filters={"user_id": "u1"}
        )

    def test_search_default_limit(self) -> None:
        """Default limit is 20."""
        fn = _mock_search_fn(return_value={"results": []})
        searcher = VectorSearcher(search_fn=fn)

        searcher.search("query")
        fn.assert_called_once_with(query="query", limit=20, filters=None)

    def test_search_with_list_format(self) -> None:
        """search_fn returning a plain list is handled correctly."""
        raw = [
            {"id": "b1", "score": 0.7, "content": "some content"},
        ]
        fn = _mock_search_fn(return_value=raw)
        searcher = VectorSearcher(search_fn=fn)

        results = searcher.search("query")
        assert len(results) == 1
        assert results[0].bullet_id == "b1"
        assert results[0].content == "some content"

    def test_search_with_memories_key(self) -> None:
        """search_fn returning dict with 'memories' key."""
        raw = {"memories": [
            {"id": "m1", "score": 0.6, "memory": "remember this"},
        ]}
        fn = _mock_search_fn(return_value=raw)
        searcher = VectorSearcher(search_fn=fn)

        results = searcher.search("query")
        assert len(results) == 1
        assert results[0].bullet_id == "m1"
        assert results[0].content == "remember this"


# -- Test 5: score normalization --------------------------------------------


class TestScoreNormalization:
    def test_score_in_range(self) -> None:
        """Scores already in [0, 1] are preserved."""
        assert _normalize_score(0.0) == pytest.approx(0.0)
        assert _normalize_score(0.5) == pytest.approx(0.5)
        assert _normalize_score(1.0) == pytest.approx(1.0)

    def test_negative_cosine_similarity(self) -> None:
        """Negative score (cosine [-1, 1]) is mapped to [0, 1]."""
        # -1 -> 0.0
        assert _normalize_score(-1.0) == pytest.approx(0.0)
        # 0 (from cosine) -> 0.5
        assert _normalize_score(-0.0) == pytest.approx(0.0)  # -0.0 is not < 0

    def test_negative_half(self) -> None:
        """Negative -0.5 -> (−0.5 + 1) / 2 = 0.25."""
        assert _normalize_score(-0.5) == pytest.approx(0.25)

    def test_score_above_one(self) -> None:
        """Scores > 1.0 are clamped to 1.0."""
        assert _normalize_score(1.5) == pytest.approx(1.0)
        assert _normalize_score(100.0) == pytest.approx(1.0)

    def test_clamp_helper(self) -> None:
        """_clamp works correctly."""
        assert _clamp(-0.5) == 0.0
        assert _clamp(0.5) == 0.5
        assert _clamp(1.5) == 1.0
        assert _clamp(50.0, lo=0.0, hi=100.0) == 50.0

    def test_normalized_scores_in_results(self) -> None:
        """Search results have scores normalized to [0, 1]."""
        raw = _make_raw_results([
            {"id": "b1", "score": -0.5},   # negative cosine -> 0.25
            {"id": "b2", "score": 0.8},     # normal -> 0.8
            {"id": "b3", "score": 1.5},     # overflow -> 1.0
        ])
        fn = _mock_search_fn(return_value=raw)
        searcher = VectorSearcher(search_fn=fn)
        results = searcher.search("q")

        assert len(results) == 3
        assert results[0].score == pytest.approx(0.25)
        assert results[1].score == pytest.approx(0.8)
        assert results[2].score == pytest.approx(1.0)


# -- Test 6: filters passthrough -------------------------------------------


class TestFiltersPassthrough:
    def test_filters_none(self) -> None:
        """filters=None is passed as-is."""
        fn = _mock_search_fn(return_value={"results": []})
        searcher = VectorSearcher(search_fn=fn)
        searcher.search("q", filters=None)
        fn.assert_called_once_with(query="q", limit=20, filters=None)

    def test_filters_dict(self) -> None:
        """Arbitrary filter dict is passed through unchanged."""
        filters = {"user_id": "u1", "section": "debugging"}
        fn = _mock_search_fn(return_value={"results": []})
        searcher = VectorSearcher(search_fn=fn)
        searcher.search("q", filters=filters)
        fn.assert_called_once_with(query="q", limit=20, filters=filters)


# -- Test 7: limit parameter -----------------------------------------------


class TestLimitParameter:
    def test_custom_limit(self) -> None:
        """Custom limit is passed to search_fn."""
        fn = _mock_search_fn(return_value={"results": []})
        searcher = VectorSearcher(search_fn=fn)
        searcher.search("q", limit=5)
        fn.assert_called_once_with(query="q", limit=5, filters=None)


# -- Test 8: metadata extraction -------------------------------------------


class TestMetadataExtraction:
    def test_metadata_from_metadata_key(self) -> None:
        """Metadata dict in result item is preserved."""
        raw = _make_raw_results([
            {
                "id": "b1",
                "score": 0.9,
                "memory": "content",
                "metadata": {"section": "debugging", "tags": ["git"]},
            },
        ])
        fn = _mock_search_fn(return_value=raw)
        searcher = VectorSearcher(search_fn=fn)
        results = searcher.search("q")

        assert len(results) == 1
        assert results[0].metadata == {"section": "debugging", "tags": ["git"]}

    def test_extra_keys_as_metadata(self) -> None:
        """Extra keys (not id/score/content) are collected as metadata."""
        raw = _make_raw_results([
            {
                "id": "b1",
                "score": 0.8,
                "memory": "content",
                "user_id": "u1",
                "agent": "bot",
            },
        ])
        fn = _mock_search_fn(return_value=raw)
        searcher = VectorSearcher(search_fn=fn)
        results = searcher.search("q")

        assert len(results) == 1
        assert results[0].metadata == {"user_id": "u1", "agent": "bot"}


# -- Test 9: edge cases ----------------------------------------------------


class TestEdgeCases:
    def test_empty_results(self) -> None:
        """search_fn returning empty results -> empty list."""
        fn = _mock_search_fn(return_value={"results": []})
        searcher = VectorSearcher(search_fn=fn)
        assert searcher.search("q") == []

    def test_items_without_id_skipped(self) -> None:
        """Items missing 'id' field are silently skipped."""
        raw = _make_raw_results([
            {"score": 0.9, "memory": "no id here"},
            {"id": "b1", "score": 0.8, "memory": "has id"},
        ])
        fn = _mock_search_fn(return_value=raw)
        searcher = VectorSearcher(search_fn=fn)
        results = searcher.search("q")

        assert len(results) == 1
        assert results[0].bullet_id == "b1"

    def test_non_dict_items_skipped(self) -> None:
        """Non-dict items in result list are silently skipped."""
        raw = _make_raw_results([
            "not a dict",
            42,
            {"id": "b1", "score": 0.7, "memory": "ok"},
        ])
        fn = _mock_search_fn(return_value=raw)
        searcher = VectorSearcher(search_fn=fn)
        results = searcher.search("q")

        assert len(results) == 1
        assert results[0].bullet_id == "b1"

    def test_unexpected_return_type_returns_empty(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """search_fn returning unexpected type -> empty list + warning."""
        fn = _mock_search_fn(return_value="not a dict or list")
        searcher = VectorSearcher(search_fn=fn)
        with caplog.at_level(logging.WARNING):
            results = searcher.search("q")
        assert results == []
        assert "unexpected result type" in caplog.text

    def test_bullet_id_from_bullet_id_key(self) -> None:
        """Result item with 'bullet_id' key instead of 'id'."""
        raw = _make_raw_results([
            {"bullet_id": "x1", "score": 0.9, "memory": "alt key"},
        ])
        fn = _mock_search_fn(return_value=raw)
        searcher = VectorSearcher(search_fn=fn)
        results = searcher.search("q")

        assert len(results) == 1
        assert results[0].bullet_id == "x1"

    def test_similarity_key_for_score(self) -> None:
        """Result with 'similarity' key instead of 'score'."""
        raw = _make_raw_results([
            {"id": "b1", "similarity": 0.75, "memory": "sim key"},
        ])
        fn = _mock_search_fn(return_value=raw)
        searcher = VectorSearcher(search_fn=fn)
        results = searcher.search("q")

        assert len(results) == 1
        assert results[0].score == pytest.approx(0.75)

    def test_distance_key_for_score(self) -> None:
        """Result with 'distance' key instead of 'score'."""
        raw = _make_raw_results([
            {"id": "b1", "distance": 0.3, "memory": "dist key"},
        ])
        fn = _mock_search_fn(return_value=raw)
        searcher = VectorSearcher(search_fn=fn)
        results = searcher.search("q")

        assert len(results) == 1
        assert results[0].score == pytest.approx(0.3)

    def test_content_key_for_content(self) -> None:
        """Result with 'content' key instead of 'memory'."""
        raw = _make_raw_results([
            {"id": "b1", "score": 0.5, "content": "via content key"},
        ])
        fn = _mock_search_fn(return_value=raw)
        searcher = VectorSearcher(search_fn=fn)
        results = searcher.search("q")

        assert len(results) == 1
        assert results[0].content == "via content key"


# -- Test 10: VectorMatch dataclass -----------------------------------------


class TestVectorMatchDataclass:
    def test_default_values(self) -> None:
        """VectorMatch default values are correct."""
        m = VectorMatch(bullet_id="b1", score=0.5)
        assert m.bullet_id == "b1"
        assert m.score == 0.5
        assert m.content == ""
        assert m.metadata == {}

    def test_full_construction(self) -> None:
        """VectorMatch with all fields."""
        m = VectorMatch(
            bullet_id="b1",
            score=0.9,
            content="test content",
            metadata={"key": "val"},
        )
        assert m.bullet_id == "b1"
        assert m.score == 0.9
        assert m.content == "test content"
        assert m.metadata == {"key": "val"}

    def test_metadata_isolation(self) -> None:
        """Default metadata dict is independent per instance."""
        m1 = VectorMatch(bullet_id="a", score=0.1)
        m2 = VectorMatch(bullet_id="b", score=0.2)
        m1.metadata["x"] = 1
        assert "x" not in m2.metadata


# ── STORY-031 补充测试：异常路径、两种 mem0 输出格式、分数类型兜底 ─────


class TestVectorSearcherInvalidScoreTypes:
    """分数字段类型异常时的兜底行为。"""

    def test_score_as_string_numeric(self) -> None:
        """Score provided as numeric string should be parsed correctly."""
        raw = _make_raw_results([
            {"id": "b1", "score": "0.85", "memory": "test"},
        ])
        fn = _mock_search_fn(return_value=raw)
        searcher = VectorSearcher(search_fn=fn)
        results = searcher.search("q")
        assert len(results) == 1
        assert results[0].score == pytest.approx(0.85)

    def test_score_as_invalid_string(self) -> None:
        """Score as non-numeric string should fallback to 0.0."""
        raw = _make_raw_results([
            {"id": "b1", "score": "not_a_number", "memory": "test"},
        ])
        fn = _mock_search_fn(return_value=raw)
        searcher = VectorSearcher(search_fn=fn)
        results = searcher.search("q")
        assert len(results) == 1
        # All score keys fail to parse -> raw_score stays 0.0
        assert results[0].score == pytest.approx(0.0)

    def test_score_as_none(self) -> None:
        """Score key exists but value is None -> fallback to 0.0."""
        raw = _make_raw_results([
            {"id": "b1", "score": None, "memory": "test"},
        ])
        fn = _mock_search_fn(return_value=raw)
        searcher = VectorSearcher(search_fn=fn)
        results = searcher.search("q")
        assert len(results) == 1
        assert results[0].score == pytest.approx(0.0)

    def test_no_score_key_at_all(self) -> None:
        """Result with no score/similarity/distance key -> score 0.0."""
        raw = _make_raw_results([
            {"id": "b1", "memory": "no score here"},
        ])
        fn = _mock_search_fn(return_value=raw)
        searcher = VectorSearcher(search_fn=fn)
        results = searcher.search("q")
        assert len(results) == 1
        assert results[0].score == pytest.approx(0.0)


class TestVectorSearcherEmptyIdFallback:
    """Empty or missing ID 字段的兜底行为。"""

    def test_empty_string_id_skipped(self) -> None:
        """Item with id='' should be skipped."""
        raw = _make_raw_results([
            {"id": "", "score": 0.5, "memory": "empty id"},
            {"id": "b2", "score": 0.8, "memory": "good id"},
        ])
        fn = _mock_search_fn(return_value=raw)
        searcher = VectorSearcher(search_fn=fn)
        results = searcher.search("q")
        assert len(results) == 1
        assert results[0].bullet_id == "b2"

    def test_numeric_id_converted_to_string(self) -> None:
        """Numeric id should be converted to string."""
        raw = _make_raw_results([
            {"id": 42, "score": 0.5, "memory": "numeric id"},
        ])
        fn = _mock_search_fn(return_value=raw)
        searcher = VectorSearcher(search_fn=fn)
        results = searcher.search("q")
        assert len(results) == 1
        assert results[0].bullet_id == "42"


class TestVectorSearcherMem0Formats:
    """两种 mem0 输出格式的兼容性。"""

    def test_dict_with_results_key(self) -> None:
        """Standard format: dict with 'results' key."""
        raw = {"results": [
            {"id": "r1", "score": 0.9, "memory": "result 1"},
        ]}
        fn = _mock_search_fn(return_value=raw)
        searcher = VectorSearcher(search_fn=fn)
        results = searcher.search("q")
        assert len(results) == 1
        assert results[0].bullet_id == "r1"

    def test_dict_with_memories_key(self) -> None:
        """Alternate format: dict with 'memories' key."""
        raw = {"memories": [
            {"id": "m1", "score": 0.8, "memory": "memory 1"},
        ]}
        fn = _mock_search_fn(return_value=raw)
        searcher = VectorSearcher(search_fn=fn)
        results = searcher.search("q")
        assert len(results) == 1
        assert results[0].bullet_id == "m1"

    def test_plain_list_format(self) -> None:
        """Plain list format without wrapper dict."""
        raw = [
            {"id": "l1", "score": 0.7, "content": "list item"},
        ]
        fn = _mock_search_fn(return_value=raw)
        searcher = VectorSearcher(search_fn=fn)
        results = searcher.search("q")
        assert len(results) == 1
        assert results[0].bullet_id == "l1"
        assert results[0].content == "list item"

    def test_dict_with_neither_results_nor_memories(self) -> None:
        """Dict without 'results' or 'memories' key -> empty results."""
        raw = {"data": [{"id": "x", "score": 0.5}]}
        fn = _mock_search_fn(return_value=raw)
        searcher = VectorSearcher(search_fn=fn)
        results = searcher.search("q")
        assert results == []

    def test_integer_return_value_handled(self) -> None:
        """Integer return value (neither list nor dict) -> empty list."""
        fn = _mock_search_fn(return_value=42)
        searcher = VectorSearcher(search_fn=fn)
        results = searcher.search("q")
        assert results == []


class TestNormalizeScoreEdge:
    """Score normalization edge cases."""

    def test_exactly_zero(self) -> None:
        """Score of exactly 0.0 should pass through unchanged."""
        assert _normalize_score(0.0) == pytest.approx(0.0)

    def test_exactly_one(self) -> None:
        """Score of exactly 1.0 should pass through unchanged."""
        assert _normalize_score(1.0) == pytest.approx(1.0)

    def test_very_small_negative(self) -> None:
        """Very small negative -> mapped via (s+1)/2."""
        assert _normalize_score(-0.01) == pytest.approx((-0.01 + 1.0) / 2.0)

    def test_large_negative(self) -> None:
        """Large negative value clamped to 0.0."""
        assert _normalize_score(-100.0) == pytest.approx(0.0)
