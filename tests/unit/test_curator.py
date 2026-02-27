"""Comprehensive unit tests for the Curator subsystem (STORY-019).

Covers:
- CuratorEngine.cosine_similarity precision
- CuratorEngine.text_similarity basics
- CuratorEngine.curate deduplication logic with threshold boundaries
- KeepBestStrategy merge results
- MergeContentStrategy merge results
- Empty / edge-case inputs
- Embedding fallback mode (no embedding -> text_similarity)
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from unittest.mock import patch

import pytest

from memx.config import CuratorConfig
from memx.engines.curator.engine import (
    CurateResult,
    CuratorEngine,
    ExistingBullet,
    MergeCandidate,
)
from memx.engines.curator.merger import (
    KeepBestStrategy,
    MergeContentStrategy,
    MergeResult,
    _split_sentences,
    _union_list,
    get_merge_strategy,
)
from memx.types import CandidateBullet

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _candidate(
    content: str = "test content",
    score: float = 50.0,
    related_tools: list[str] | None = None,
    key_entities: list[str] | None = None,
    tags: list[str] | None = None,
) -> CandidateBullet:
    """Create a CandidateBullet with configurable fields."""
    return CandidateBullet(
        content=content,
        instructivity_score=score,
        related_tools=related_tools or [],
        key_entities=key_entities or [],
        tags=tags or [],
    )


def _existing(
    bullet_id: str = "b1",
    content: str = "existing content",
    embedding: list[float] | None = None,
    score: float = 50.0,
    recall_count: int = 0,
    related_tools: list[str] | None = None,
    key_entities: list[str] | None = None,
    tags: list[str] | None = None,
) -> ExistingBullet:
    """Create an ExistingBullet with metadata populated like a real bullet."""
    return ExistingBullet(
        bullet_id=bullet_id,
        content=content,
        embedding=embedding,
        metadata={
            "instructivity_score": score,
            "recall_count": recall_count,
            "related_tools": related_tools or [],
            "key_entities": key_entities or [],
            "tags": tags or [],
        },
    )


# ===========================================================================
# TestCosineSimilarity
# ===========================================================================


class TestCosineSimilarity:
    """Validate cosine_similarity numeric precision and edge cases."""

    def test_identical_vectors_return_one(self) -> None:
        """Identical vectors -> 1.0."""
        a = [1.0, 0.0, 0.0]
        assert CuratorEngine.cosine_similarity(a, a) == 1.0

    def test_orthogonal_vectors_return_zero(self) -> None:
        """Orthogonal (perpendicular) vectors -> 0.0."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert CuratorEngine.cosine_similarity(a, b) == 0.0

    def test_opposite_vectors_return_neg_one(self) -> None:
        """Opposite vectors -> -1.0 (anti-parallel)."""
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert CuratorEngine.cosine_similarity(a, b) == -1.0

    def test_negative_similarity_clamps_to_neg_one(self) -> None:
        """Anti-parallel higher-dim vectors clamp correctly."""
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        result = CuratorEngine.cosine_similarity(a, b)
        assert result == -1.0

    def test_known_similarity_value(self) -> None:
        """Verify a known analytic value: dot=32, |a|=sqrt(14), |b|=sqrt(77)."""
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        expected = 32.0 / (math.sqrt(14) * math.sqrt(77))
        result = CuratorEngine.cosine_similarity(a, b)
        assert abs(result - expected) < 1e-9

    def test_different_length_returns_zero(self) -> None:
        """Dimension mismatch -> 0.0."""
        assert CuratorEngine.cosine_similarity([1.0, 2.0], [1.0]) == 0.0

    def test_empty_vectors_return_zero(self) -> None:
        """Both empty -> 0.0 (graceful)."""
        assert CuratorEngine.cosine_similarity([], []) == 0.0

    def test_zero_vector_a_returns_zero(self) -> None:
        """Zero-magnitude vector a -> 0.0."""
        assert CuratorEngine.cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_zero_vector_b_returns_zero(self) -> None:
        """Zero-magnitude vector b -> 0.0."""
        assert CuratorEngine.cosine_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0

    def test_both_zero_vectors_return_zero(self) -> None:
        """Both zero-magnitude -> 0.0."""
        assert CuratorEngine.cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_clamping_near_one(self) -> None:
        """Large identical vectors should not exceed 1.0 due to FP rounding."""
        a = [1e10, 1e10, 1e10]
        result = CuratorEngine.cosine_similarity(a, a)
        assert result <= 1.0

    def test_single_dimension(self) -> None:
        """Single-element vectors work correctly."""
        assert CuratorEngine.cosine_similarity([3.0], [3.0]) == 1.0
        assert CuratorEngine.cosine_similarity([3.0], [-3.0]) == -1.0

    def test_unit_vectors(self) -> None:
        """Unit vectors at 45 degrees -> cos(45) ~ 0.707."""
        a = [1.0, 0.0]
        b = [math.sqrt(2) / 2, math.sqrt(2) / 2]
        expected = math.sqrt(2) / 2
        result = CuratorEngine.cosine_similarity(a, b)
        assert abs(result - expected) < 1e-9


# ===========================================================================
# TestTextSimilarity
# ===========================================================================


class TestTextSimilarity:
    """Validate text_similarity (Jaccard over tokens)."""

    def test_identical_strings_return_one(self) -> None:
        """Identical strings -> 1.0."""
        assert CuratorEngine.text_similarity("hello world", "hello world") == 1.0

    def test_no_overlap_returns_zero(self) -> None:
        """Completely disjoint tokens -> 0.0."""
        assert CuratorEngine.text_similarity("hello world", "foo bar") == 0.0

    def test_partial_overlap(self) -> None:
        """Partial overlap: Jaccard({use,cargo,check}, {use,cargo,build}) = 2/4 = 0.5."""
        result = CuratorEngine.text_similarity("use cargo check", "use cargo build")
        assert abs(result - 0.5) < 1e-9

    def test_case_insensitive(self) -> None:
        """Token comparison is case-insensitive."""
        assert CuratorEngine.text_similarity("Hello World", "hello world") == 1.0

    def test_empty_first_string_returns_zero(self) -> None:
        """Empty string a -> 0.0."""
        assert CuratorEngine.text_similarity("", "hello") == 0.0

    def test_empty_second_string_returns_zero(self) -> None:
        """Empty string b -> 0.0."""
        assert CuratorEngine.text_similarity("hello", "") == 0.0

    def test_both_empty_returns_zero(self) -> None:
        """Both empty -> 0.0."""
        assert CuratorEngine.text_similarity("", "") == 0.0

    def test_single_word_match(self) -> None:
        """Single identical word -> 1.0."""
        assert CuratorEngine.text_similarity("hello", "hello") == 1.0

    def test_single_word_no_match(self) -> None:
        """Single different words -> 0.0."""
        assert CuratorEngine.text_similarity("hello", "world") == 0.0

    def test_superset_subset(self) -> None:
        """Subset/superset: {a,b} vs {a,b,c} -> Jaccard = 2/3."""
        result = CuratorEngine.text_similarity("a b", "a b c")
        assert abs(result - 2.0 / 3.0) < 1e-9

    def test_whitespace_only_returns_zero(self) -> None:
        """Whitespace-only strings -> 0.0 (split yields empty set)."""
        assert CuratorEngine.text_similarity("   ", "hello") == 0.0


# ===========================================================================
# TestCuratorEngine -- curate() dedup logic
# ===========================================================================


class TestCuratorEngine:
    """Validate CuratorEngine.curate() deduplication decisions."""

    # -- Empty / edge cases --

    def test_empty_candidates_returns_empty_result(self) -> None:
        """Empty candidates list -> empty CurateResult."""
        engine = CuratorEngine()
        result = engine.curate([], [_existing()])
        assert result.to_add == []
        assert result.to_merge == []
        assert result.to_skip == []

    def test_empty_existing_all_insert(self) -> None:
        """No existing bullets -> every candidate is Insert."""
        engine = CuratorEngine()
        c1 = _candidate("use dark mode")
        c2 = _candidate("prefer vim keybindings")
        result = engine.curate([c1, c2], [])
        assert len(result.to_add) == 2
        assert c1 in result.to_add
        assert c2 in result.to_add
        assert result.to_merge == []
        assert result.to_skip == []

    def test_both_lists_empty(self) -> None:
        """Both empty -> empty result."""
        engine = CuratorEngine()
        result = engine.curate([], [])
        assert result == CurateResult()

    def test_empty_content_is_skipped(self) -> None:
        """Candidate with empty/whitespace content -> Skip."""
        engine = CuratorEngine()
        c_empty = _candidate("")
        c_whitespace = _candidate("   ")
        result = engine.curate([c_empty, c_whitespace], [_existing()])
        assert len(result.to_skip) == 2
        assert result.to_add == []
        assert result.to_merge == []

    # -- Threshold boundary: 0.79 vs 0.80 (acceptance criteria) --

    def test_threshold_079_is_insert(self) -> None:
        """AC: similarity=0.79 with default threshold=0.80 -> Insert.

        Craft Jaccard = 0.79 is hard with small token sets, so we use a
        custom threshold of 0.5 and craft similarity just below (0.49).
        For the exact 0.79/0.80 boundary, we mock _compare.
        """
        engine = CuratorEngine(CuratorConfig(similarity_threshold=0.80))
        c = _candidate("candidate content")
        ex = _existing(content="existing content")

        # Mock _compare to return exactly 0.79
        with patch.object(engine, "_compare", return_value=0.79):
            result = engine.curate([c], [ex])
        assert len(result.to_add) == 1
        assert result.to_merge == []

    def test_threshold_080_is_merge(self) -> None:
        """AC: similarity=0.80 with default threshold=0.80 -> Merge."""
        engine = CuratorEngine(CuratorConfig(similarity_threshold=0.80))
        c = _candidate("candidate content")
        ex = _existing(content="existing content")

        # Mock _compare to return exactly 0.80
        with patch.object(engine, "_compare", return_value=0.80):
            result = engine.curate([c], [ex])
        assert len(result.to_merge) == 1
        assert abs(result.to_merge[0].similarity - 0.80) < 1e-9

    def test_threshold_boundary_just_above(self) -> None:
        """Similarity just above threshold -> Merge."""
        engine = CuratorEngine(CuratorConfig(similarity_threshold=0.80))
        c = _candidate("content")
        ex = _existing(content="content")
        with patch.object(engine, "_compare", return_value=0.81):
            result = engine.curate([c], [ex])
        assert len(result.to_merge) == 1

    # -- Identical / different content via text fallback --

    def test_identical_content_triggers_merge(self) -> None:
        """Identical text content -> text_similarity=1.0 -> Merge."""
        engine = CuratorEngine()
        c = _candidate("use cargo check for fast feedback")
        ex = _existing(content="use cargo check for fast feedback")
        result = engine.curate([c], [ex])
        assert len(result.to_merge) == 1
        assert result.to_merge[0].similarity == 1.0

    def test_completely_different_triggers_insert(self) -> None:
        """Completely different text -> Jaccard ~ 0.0 -> Insert."""
        engine = CuratorEngine()
        c = _candidate("prefer dark mode in vscode")
        ex = _existing(content="use cargo check for fast feedback")
        result = engine.curate([c], [ex])
        assert len(result.to_add) == 1
        assert result.to_merge == []

    # -- Multiple existing: best match wins --

    def test_multiple_existing_picks_best_match(self) -> None:
        """When multiple existing, the one with highest similarity is picked."""
        engine = CuratorEngine(CuratorConfig(similarity_threshold=0.5))
        c = _candidate("use cargo check for fast feedback")
        ex1 = _existing(bullet_id="b1", content="something completely unrelated")
        ex2 = _existing(bullet_id="b2", content="use cargo check for fast feedback")
        result = engine.curate([c], [ex1, ex2])
        assert len(result.to_merge) == 1
        assert result.to_merge[0].existing.bullet_id == "b2"

    # -- Multiple candidates can match same existing --

    def test_multiple_candidates_same_existing(self) -> None:
        """Multiple candidates can independently match the same existing bullet."""
        engine = CuratorEngine()
        c1 = _candidate("use cargo check for feedback")
        c2 = _candidate("use cargo check for feedback")
        ex = _existing(content="use cargo check for feedback")
        result = engine.curate([c1, c2], [ex])
        assert len(result.to_merge) == 2

    # -- Mixed outcomes --

    def test_mixed_add_merge_skip(self) -> None:
        """A batch with mixed outcomes: skip, add, and merge."""
        engine = CuratorEngine()
        c_skip = _candidate("")
        c_add = _candidate("completely new unique content xyz abc")
        c_merge = _candidate("prefer dark mode in the editor")
        ex = _existing(content="prefer dark mode in the editor")
        result = engine.curate([c_skip, c_add, c_merge], [ex])
        assert len(result.to_skip) == 1
        assert len(result.to_add) == 1
        assert len(result.to_merge) == 1

    # -- Custom threshold --

    def test_low_threshold_merges_more(self) -> None:
        """Very low threshold -> even slight overlap triggers Merge."""
        config = CuratorConfig(similarity_threshold=0.1)
        engine = CuratorEngine(config)
        c = _candidate("use cargo check")
        ex = _existing(content="use npm test")
        # Jaccard({use,cargo,check}, {use,npm,test}) = 1/5 = 0.2 >= 0.1
        result = engine.curate([c], [ex])
        assert len(result.to_merge) == 1

    def test_high_threshold_inserts_more(self) -> None:
        """Very high threshold -> near-identical still gets Insert."""
        config = CuratorConfig(similarity_threshold=0.99)
        engine = CuratorEngine(config)
        c = _candidate("use cargo check for fast feedback")
        ex = _existing(content="use cargo check for quick feedback")
        result = engine.curate([c], [ex])
        assert len(result.to_add) == 1
        assert result.to_merge == []

    # -- Fallback to text_similarity when embeddings are None (AC) --

    def test_fallback_to_text_when_no_embedding(self) -> None:
        """AC: embedding=None -> automatically fall back to text_similarity.

        CandidateBullet has no embedding field, so _get_candidate_embedding
        always returns None. Even if existing has an embedding, fallback
        should trigger.
        """
        engine = CuratorEngine()
        c = _candidate("use cargo check for fast feedback")
        # Existing has embedding, but candidate does not
        ex = _existing(
            content="use cargo check for fast feedback",
            embedding=[0.1, 0.2, 0.3],
        )
        result = engine.curate([c], [ex])
        # text_similarity on identical strings = 1.0, so should merge
        assert len(result.to_merge) == 1
        assert result.to_merge[0].similarity == 1.0

    def test_fallback_when_existing_has_no_embedding(self) -> None:
        """When existing embedding is None, text fallback is used."""
        engine = CuratorEngine()
        c = _candidate("use cargo check for fast feedback")
        ex = _existing(
            content="use cargo check for fast feedback",
            embedding=None,
        )
        result = engine.curate([c], [ex])
        assert len(result.to_merge) == 1

    def test_fallback_different_content_below_threshold(self) -> None:
        """Fallback text similarity on different content -> Insert."""
        engine = CuratorEngine()
        c = _candidate("alpha bravo charlie")
        ex = _existing(content="delta echo foxtrot", embedding=[0.1, 0.2])
        result = engine.curate([c], [ex])
        assert len(result.to_add) == 1


# ===========================================================================
# TestCuratorEngine -- properties and config
# ===========================================================================


class TestCuratorProperties:
    """Validate engine property accessors and config integration."""

    def test_default_threshold(self) -> None:
        engine = CuratorEngine()
        assert engine.threshold == 0.8

    def test_custom_threshold(self) -> None:
        engine = CuratorEngine(CuratorConfig(similarity_threshold=0.5))
        assert engine.threshold == 0.5

    def test_config_none_uses_defaults(self) -> None:
        engine = CuratorEngine(config=None)
        assert engine.threshold == 0.8


# ===========================================================================
# TestKeepBestStrategy
# ===========================================================================


class TestKeepBestStrategy:
    """Validate KeepBestStrategy merge results."""

    def test_candidate_higher_score_wins(self) -> None:
        """AC: higher instructivity_score wins."""
        strategy = KeepBestStrategy()
        c = _candidate(content="candidate text", score=90.0)
        ex = _existing(content="existing text", score=60.0)
        result = strategy.merge(c, ex)
        assert result.merged_content == "candidate text"
        assert result.merged_metadata["instructivity_score"] == 90.0
        assert result.source_id == "b1"
        assert result.strategy_used == "keep_best"

    def test_existing_higher_score_wins(self) -> None:
        """Existing with higher score is retained."""
        strategy = KeepBestStrategy()
        c = _candidate(content="candidate text", score=40.0)
        ex = _existing(content="existing text", score=80.0)
        result = strategy.merge(c, ex)
        assert result.merged_content == "existing text"
        assert result.merged_metadata["instructivity_score"] == 80.0

    def test_equal_score_longer_content_wins(self) -> None:
        """AC: same score -> longer content wins."""
        strategy = KeepBestStrategy()
        c = _candidate(content="a much longer candidate content string", score=70.0)
        ex = _existing(content="short", score=70.0)
        result = strategy.merge(c, ex)
        assert result.merged_content == "a much longer candidate content string"

    def test_equal_score_equal_length_keeps_existing(self) -> None:
        """Same score, same length -> keep existing (tie-break)."""
        strategy = KeepBestStrategy()
        c = _candidate(content="abcd", score=70.0)
        ex = _existing(content="efgh", score=70.0)
        result = strategy.merge(c, ex)
        assert result.merged_content == "efgh"

    def test_metadata_union_related_tools(self) -> None:
        """AC: related_tools from both sources are unioned, no duplicates."""
        strategy = KeepBestStrategy()
        c = _candidate(content="c", score=90.0, related_tools=["cargo", "rustc"])
        ex = _existing(content="e", score=60.0, related_tools=["cargo", "clippy"])
        result = strategy.merge(c, ex)
        tools = result.merged_metadata["related_tools"]
        assert set(tools) == {"cargo", "rustc", "clippy"}
        assert len(tools) == 3  # no duplicates

    def test_metadata_union_key_entities(self) -> None:
        """key_entities from both sources are unioned."""
        strategy = KeepBestStrategy()
        c = _candidate(content="c", score=90.0, key_entities=["Rust", "WASM"])
        ex = _existing(content="e", score=60.0, key_entities=["Rust", "LLVM"])
        result = strategy.merge(c, ex)
        entities = result.merged_metadata["key_entities"]
        assert set(entities) == {"Rust", "WASM", "LLVM"}

    def test_metadata_union_tags(self) -> None:
        """tags from both sources are unioned."""
        strategy = KeepBestStrategy()
        c = _candidate(content="c", score=90.0, tags=["perf", "build"])
        ex = _existing(content="e", score=60.0, tags=["build", "ci"])
        result = strategy.merge(c, ex)
        tags = result.merged_metadata["tags"]
        assert set(tags) == {"perf", "build", "ci"}

    def test_updated_at_is_valid_timestamp(self) -> None:
        """updated_at is set to a recent ISO timestamp."""
        strategy = KeepBestStrategy()
        c = _candidate(content="c", score=90.0)
        ex = _existing(content="e", score=60.0)
        result = strategy.merge(c, ex)
        ts = result.merged_metadata["updated_at"]
        parsed = datetime.fromisoformat(ts)
        assert parsed.year >= 2026

    def test_recall_count_preserved(self) -> None:
        """recall_count from existing bullet is always preserved."""
        strategy = KeepBestStrategy()
        c = _candidate(content="c", score=90.0)
        ex = _existing(content="e", score=60.0, recall_count=42)
        result = strategy.merge(c, ex)
        assert result.merged_metadata["recall_count"] == 42

    def test_empty_candidate_content_keeps_existing(self) -> None:
        """Empty candidate -> existing wins regardless of score."""
        strategy = KeepBestStrategy()
        c = _candidate(content="", score=100.0)
        ex = _existing(content="real content", score=30.0)
        result = strategy.merge(c, ex)
        assert result.merged_content == "real content"

    def test_empty_existing_content_candidate_wins(self) -> None:
        """Empty existing -> candidate wins."""
        strategy = KeepBestStrategy()
        c = _candidate(content="real content", score=50.0)
        ex = _existing(content="", score=50.0)
        result = strategy.merge(c, ex)
        assert result.merged_content == "real content"

    def test_whitespace_candidate_keeps_existing(self) -> None:
        """Whitespace-only candidate -> existing wins."""
        strategy = KeepBestStrategy()
        c = _candidate(content="   ", score=100.0)
        ex = _existing(content="valid content", score=30.0)
        result = strategy.merge(c, ex)
        assert result.merged_content == "valid content"

    def test_existing_wins_metadata_order(self) -> None:
        """When existing wins, its tools come first in the union order."""
        strategy = KeepBestStrategy()
        c = _candidate(content="c", score=40.0, related_tools=["tool_c"])
        ex = _existing(content="e", score=80.0, related_tools=["tool_e"])
        result = strategy.merge(c, ex)
        tools = result.merged_metadata["related_tools"]
        # existing tools come first when existing wins
        assert tools[0] == "tool_e"


# ===========================================================================
# TestMergeContentStrategy
# ===========================================================================


class TestMergeContentStrategy:
    """Validate MergeContentStrategy merge results."""

    def test_basic_content_concatenation(self) -> None:
        """AC: content from both sources is concatenated."""
        strategy = MergeContentStrategy()
        c = _candidate(content="Use cargo check", score=70.0)
        ex = _existing(content="Run clippy for lints", score=60.0)
        result = strategy.merge(c, ex)
        assert "cargo check" in result.merged_content
        assert "clippy" in result.merged_content
        assert result.strategy_used == "merge_content"

    def test_identical_content_no_duplication(self) -> None:
        """Identical content is not repeated."""
        strategy = MergeContentStrategy()
        c = _candidate(content="Use cargo check", score=70.0)
        ex = _existing(content="Use cargo check", score=60.0)
        result = strategy.merge(c, ex)
        assert result.merged_content == "Use cargo check"

    def test_field_union_related_tools(self) -> None:
        """AC: field union -- related_tools from both sources are unioned."""
        strategy = MergeContentStrategy()
        c = _candidate(content="A", score=50.0, related_tools=["npm", "webpack"])
        ex = _existing(content="B", score=50.0, related_tools=["npm", "vite"])
        result = strategy.merge(c, ex)
        tools = result.merged_metadata["related_tools"]
        assert set(tools) == {"npm", "webpack", "vite"}

    def test_field_union_key_entities(self) -> None:
        """key_entities from both sources are unioned."""
        strategy = MergeContentStrategy()
        c = _candidate(content="A", score=50.0, key_entities=["React", "JSX"])
        ex = _existing(content="B", score=50.0, key_entities=["React", "Hooks"])
        result = strategy.merge(c, ex)
        entities = result.merged_metadata["key_entities"]
        assert set(entities) == {"React", "JSX", "Hooks"}

    def test_field_union_tags(self) -> None:
        """tags from both sources are unioned."""
        strategy = MergeContentStrategy()
        c = _candidate(content="A", score=50.0, tags=["frontend", "perf"])
        ex = _existing(content="B", score=50.0, tags=["frontend", "ux"])
        result = strategy.merge(c, ex)
        tags = result.merged_metadata["tags"]
        assert set(tags) == {"frontend", "perf", "ux"}

    def test_higher_score_retained(self) -> None:
        """The max instructivity_score is kept."""
        strategy = MergeContentStrategy()
        c = _candidate(content="A", score=90.0)
        ex = _existing(content="B", score=60.0)
        result = strategy.merge(c, ex)
        assert result.merged_metadata["instructivity_score"] == 90.0

    def test_existing_higher_score_retained(self) -> None:
        """When existing has higher score, it is kept."""
        strategy = MergeContentStrategy()
        c = _candidate(content="A", score=30.0)
        ex = _existing(content="B", score=80.0)
        result = strategy.merge(c, ex)
        assert result.merged_metadata["instructivity_score"] == 80.0

    def test_updated_at_is_refreshed(self) -> None:
        """AC: updated_at is set to a current timestamp."""
        strategy = MergeContentStrategy()
        c = _candidate(content="A", score=50.0)
        ex = _existing(content="B", score=50.0)
        result = strategy.merge(c, ex)
        ts = result.merged_metadata["updated_at"]
        parsed = datetime.fromisoformat(ts)
        assert parsed.year >= 2026

    def test_recall_count_preserved(self) -> None:
        """recall_count from existing is preserved."""
        strategy = MergeContentStrategy()
        c = _candidate(content="A", score=50.0)
        ex = _existing(content="B", score=50.0, recall_count=17)
        result = strategy.merge(c, ex)
        assert result.merged_metadata["recall_count"] == 17

    def test_source_id_from_existing(self) -> None:
        """source_id is the existing bullet's ID."""
        strategy = MergeContentStrategy()
        c = _candidate(content="A", score=50.0)
        ex = _existing(bullet_id="b99", content="B", score=50.0)
        result = strategy.merge(c, ex)
        assert result.source_id == "b99"

    def test_empty_candidate_returns_existing(self) -> None:
        """Empty candidate content -> return existing content only."""
        strategy = MergeContentStrategy()
        c = _candidate(content="", score=50.0)
        ex = _existing(content="Keep this", score=50.0)
        result = strategy.merge(c, ex)
        assert result.merged_content == "Keep this"

    def test_empty_existing_returns_candidate(self) -> None:
        """Empty existing content -> return candidate content only."""
        strategy = MergeContentStrategy()
        c = _candidate(content="New stuff", score=50.0)
        ex = _existing(content="", score=50.0)
        result = strategy.merge(c, ex)
        assert result.merged_content == "New stuff"

    def test_sentence_level_dedup(self) -> None:
        """Overlapping sentences are not duplicated in merged output."""
        strategy = MergeContentStrategy()
        c = _candidate(
            content="Use cargo check. Run tests often.",
            score=50.0,
        )
        ex = _existing(
            content="Use cargo check. Prefer debug builds.",
            score=50.0,
        )
        result = strategy.merge(c, ex)
        # "Use cargo check" should appear only once
        assert result.merged_content.lower().count("use cargo check") == 1
        assert "tests often" in result.merged_content.lower()
        assert "debug builds" in result.merged_content.lower()

    def test_whitespace_candidate_returns_existing(self) -> None:
        """Whitespace-only candidate -> return existing."""
        strategy = MergeContentStrategy()
        c = _candidate(content="   ", score=50.0)
        ex = _existing(content="Keep this", score=50.0)
        result = strategy.merge(c, ex)
        assert result.merged_content == "Keep this"


# ===========================================================================
# TestGetMergeStrategy -- factory
# ===========================================================================


class TestGetMergeStrategy:
    """Validate the get_merge_strategy factory."""

    def test_keep_best(self) -> None:
        strategy = get_merge_strategy("keep_best")
        assert isinstance(strategy, KeepBestStrategy)

    def test_merge_content(self) -> None:
        strategy = get_merge_strategy("merge_content")
        assert isinstance(strategy, MergeContentStrategy)

    def test_unknown_falls_back_to_keep_best(self) -> None:
        strategy = get_merge_strategy("nonexistent_strategy")
        assert isinstance(strategy, KeepBestStrategy)

    def test_unknown_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            get_merge_strategy("bad_name")
        assert "Unknown merge strategy" in caplog.text

    def test_empty_string_falls_back(self) -> None:
        strategy = get_merge_strategy("")
        assert isinstance(strategy, KeepBestStrategy)

    def test_config_default_is_keep_best(self) -> None:
        config = CuratorConfig()
        strategy = get_merge_strategy(config.merge_strategy)
        assert isinstance(strategy, KeepBestStrategy)

    def test_config_merge_content(self) -> None:
        config = CuratorConfig(merge_strategy="merge_content")
        strategy = get_merge_strategy(config.merge_strategy)
        assert isinstance(strategy, MergeContentStrategy)


# ===========================================================================
# TestDataclasses -- CurateResult, ExistingBullet, MergeCandidate, MergeResult
# ===========================================================================


class TestCurateResult:
    """Validate CurateResult dataclass defaults."""

    def test_default_empty_lists(self) -> None:
        r = CurateResult()
        assert r.to_add == []
        assert r.to_merge == []
        assert r.to_skip == []

    def test_independent_instances(self) -> None:
        """Dataclass field factory creates independent lists."""
        r1 = CurateResult()
        r2 = CurateResult()
        r1.to_add.append(_candidate("x"))
        assert r2.to_add == []


class TestExistingBullet:
    """Validate ExistingBullet dataclass."""

    def test_defaults(self) -> None:
        eb = ExistingBullet(bullet_id="b1", content="hello")
        assert eb.embedding is None
        assert eb.metadata == {}

    def test_with_embedding(self) -> None:
        eb = ExistingBullet(bullet_id="b1", content="hi", embedding=[1.0, 2.0])
        assert eb.embedding == [1.0, 2.0]


class TestMergeCandidate:
    """Validate MergeCandidate dataclass."""

    def test_fields(self) -> None:
        c = _candidate("hello")
        ex = _existing()
        mc = MergeCandidate(candidate=c, existing=ex, similarity=0.85)
        assert mc.candidate is c
        assert mc.existing is ex
        assert mc.similarity == 0.85


class TestMergeResult:
    """Validate MergeResult dataclass defaults."""

    def test_defaults(self) -> None:
        r = MergeResult(merged_content="hello")
        assert r.merged_metadata == {}
        assert r.source_id == ""
        assert r.strategy_used == ""


# ===========================================================================
# Test internal helpers -- _split_sentences, _union_list
# ===========================================================================


class TestHelperFunctions:
    """Validate internal utility functions for coverage."""

    def test_split_sentences_basic(self) -> None:
        result = _split_sentences("Hello world. Foo bar.")
        assert "Hello world" in result
        assert "Foo bar" in result

    def test_split_sentences_newlines(self) -> None:
        result = _split_sentences("Line one\nLine two")
        assert "Line one" in result
        assert "Line two" in result

    def test_split_sentences_empty(self) -> None:
        result = _split_sentences("")
        assert result == []

    def test_union_list_dedup(self) -> None:
        result = _union_list(["a", "b", "c"], ["b", "c", "d"])
        assert result == ["a", "b", "c", "d"]

    def test_union_list_preserves_order(self) -> None:
        result = _union_list(["x", "y"], ["y", "z"])
        assert result == ["x", "y", "z"]

    def test_union_list_empty_inputs(self) -> None:
        assert _union_list([], []) == []
        assert _union_list(["a"], []) == ["a"]
        assert _union_list([], ["b"]) == ["b"]
