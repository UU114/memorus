"""Unit tests for memx.engines.curator.engine — CuratorEngine."""

from __future__ import annotations

import math

from memx.config import CuratorConfig
from memx.engines.curator.engine import (
    CurateResult,
    CuratorEngine,
    ExistingBullet,
    MergeCandidate,
)
from memx.types import CandidateBullet

# ── Helper factories ─────────────────────────────────────────────────


def _candidate(content: str = "test content") -> CandidateBullet:
    """Create a minimal CandidateBullet for testing."""
    return CandidateBullet(content=content)


def _existing(
    bullet_id: str = "b1",
    content: str = "test content",
    embedding: list[float] | None = None,
) -> ExistingBullet:
    """Create a minimal ExistingBullet for testing."""
    return ExistingBullet(bullet_id=bullet_id, content=content, embedding=embedding)


# ── CuratorEngine.cosine_similarity ──────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors_return_one(self) -> None:
        a = [1.0, 0.0, 0.0]
        assert CuratorEngine.cosine_similarity(a, a) == 1.0

    def test_orthogonal_vectors_return_zero(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert CuratorEngine.cosine_similarity(a, b) == 0.0

    def test_opposite_vectors_return_neg_one(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert CuratorEngine.cosine_similarity(a, b) == -1.0

    def test_known_similarity(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        # manual: dot=32, |a|=sqrt(14), |b|=sqrt(77)
        expected = 32.0 / (math.sqrt(14) * math.sqrt(77))
        result = CuratorEngine.cosine_similarity(a, b)
        assert abs(result - expected) < 1e-9

    def test_different_length_returns_zero(self) -> None:
        assert CuratorEngine.cosine_similarity([1.0, 2.0], [1.0]) == 0.0

    def test_empty_vectors_return_zero(self) -> None:
        assert CuratorEngine.cosine_similarity([], []) == 0.0

    def test_zero_vector_returns_zero(self) -> None:
        assert CuratorEngine.cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_clamping_near_one(self) -> None:
        # Large identical vectors should not exceed 1.0 due to FP rounding
        a = [1e10, 1e10, 1e10]
        result = CuratorEngine.cosine_similarity(a, a)
        assert result <= 1.0


# ── CuratorEngine.text_similarity ────────────────────────────────────


class TestTextSimilarity:
    def test_identical_strings(self) -> None:
        assert CuratorEngine.text_similarity("hello world", "hello world") == 1.0

    def test_no_overlap(self) -> None:
        assert CuratorEngine.text_similarity("hello world", "foo bar") == 0.0

    def test_partial_overlap(self) -> None:
        # tokens a: {use, cargo, check}, b: {use, cargo, build}
        # intersection: {use, cargo}, union: {use, cargo, check, build}
        result = CuratorEngine.text_similarity("use cargo check", "use cargo build")
        assert abs(result - 0.5) < 1e-9

    def test_case_insensitive(self) -> None:
        assert CuratorEngine.text_similarity("Hello World", "hello world") == 1.0

    def test_empty_string_returns_zero(self) -> None:
        assert CuratorEngine.text_similarity("", "hello") == 0.0
        assert CuratorEngine.text_similarity("hello", "") == 0.0
        assert CuratorEngine.text_similarity("", "") == 0.0

    def test_single_word_match(self) -> None:
        assert CuratorEngine.text_similarity("hello", "hello") == 1.0


# ── CuratorEngine.curate — empty/edge cases ─────────────────────────


class TestCurateEdgeCases:
    def test_empty_candidates_returns_empty_result(self) -> None:
        engine = CuratorEngine()
        result = engine.curate([], [_existing()])
        assert result.to_add == []
        assert result.to_merge == []
        assert result.to_skip == []

    def test_empty_existing_all_insert(self) -> None:
        engine = CuratorEngine()
        c1 = _candidate("use dark mode")
        c2 = _candidate("prefer vim keybindings")
        result = engine.curate([c1, c2], [])
        assert len(result.to_add) == 2
        assert c1 in result.to_add
        assert c2 in result.to_add
        assert result.to_merge == []
        assert result.to_skip == []

    def test_empty_content_is_skipped(self) -> None:
        engine = CuratorEngine()
        c_empty = _candidate("")
        c_whitespace = _candidate("   ")
        result = engine.curate([c_empty, c_whitespace], [_existing()])
        assert len(result.to_skip) == 2
        assert result.to_add == []
        assert result.to_merge == []

    def test_both_empty(self) -> None:
        engine = CuratorEngine()
        result = engine.curate([], [])
        assert result.to_add == []
        assert result.to_merge == []
        assert result.to_skip == []


# ── CuratorEngine.curate — text similarity dedup ────────────────────


class TestCurateTextSimilarity:
    def test_identical_content_triggers_merge(self) -> None:
        engine = CuratorEngine()
        c = _candidate("use cargo check for fast feedback")
        ex = _existing(content="use cargo check for fast feedback")
        result = engine.curate([c], [ex])
        assert len(result.to_merge) == 1
        assert result.to_merge[0].candidate is c
        assert result.to_merge[0].existing is ex
        assert result.to_merge[0].similarity == 1.0
        assert result.to_add == []

    def test_completely_different_content_triggers_insert(self) -> None:
        engine = CuratorEngine()
        c = _candidate("prefer dark mode in vscode")
        ex = _existing(content="use cargo check for fast feedback")
        result = engine.curate([c], [ex])
        assert len(result.to_add) == 1
        assert result.to_merge == []

    def test_threshold_boundary_exactly_equal_is_merge(self) -> None:
        """When similarity == threshold, should be treated as Merge."""
        # Use a custom threshold and craft text to match exactly
        config = CuratorConfig(similarity_threshold=0.5)
        engine = CuratorEngine(config)
        # tokens: a={use, cargo, check}, b={use, cargo, build}
        # Jaccard = 2/4 = 0.5
        c = _candidate("use cargo check")
        ex = _existing(content="use cargo build")
        result = engine.curate([c], [ex])
        assert len(result.to_merge) == 1
        assert abs(result.to_merge[0].similarity - 0.5) < 1e-9

    def test_threshold_boundary_just_below_is_insert(self) -> None:
        """When similarity < threshold, should be Insert."""
        config = CuratorConfig(similarity_threshold=0.6)
        engine = CuratorEngine(config)
        # Jaccard = 2/4 = 0.5, which is < 0.6
        c = _candidate("use cargo check")
        ex = _existing(content="use cargo build")
        result = engine.curate([c], [ex])
        assert len(result.to_add) == 1
        assert result.to_merge == []

    def test_multiple_existing_picks_best_match(self) -> None:
        engine = CuratorEngine(CuratorConfig(similarity_threshold=0.5))
        c = _candidate("use cargo check for fast feedback")
        ex1 = _existing(bullet_id="b1", content="something completely unrelated")
        ex2 = _existing(bullet_id="b2", content="use cargo check for fast feedback")
        result = engine.curate([c], [ex1, ex2])
        assert len(result.to_merge) == 1
        assert result.to_merge[0].existing.bullet_id == "b2"

    def test_multiple_candidates_same_existing(self) -> None:
        """Multiple candidates can independently match the same existing bullet."""
        engine = CuratorEngine()
        c1 = _candidate("use cargo check for feedback")
        c2 = _candidate("use cargo check for feedback")
        ex = _existing(content="use cargo check for feedback")
        result = engine.curate([c1, c2], [ex])
        assert len(result.to_merge) == 2
        assert result.to_merge[0].existing is ex
        assert result.to_merge[1].existing is ex

    def test_mixed_add_merge_skip(self) -> None:
        """A batch with mixed outcomes: some add, some merge, some skip."""
        engine = CuratorEngine()
        c_skip = _candidate("")  # empty -> skip
        c_add = _candidate("completely new and unique content here")
        c_merge = _candidate("prefer dark mode in the editor")
        ex = _existing(content="prefer dark mode in the editor")
        result = engine.curate([c_skip, c_add, c_merge], [ex])
        assert len(result.to_skip) == 1
        assert len(result.to_add) == 1
        assert len(result.to_merge) == 1


# ── CuratorEngine.curate — custom threshold ──────────────────────────


class TestCurateCustomThreshold:
    def test_low_threshold_merges_more(self) -> None:
        config = CuratorConfig(similarity_threshold=0.1)
        engine = CuratorEngine(config)
        c = _candidate("use cargo check")
        ex = _existing(content="use npm test")
        # Jaccard({use,cargo,check}, {use,npm,test}) = 1/5 = 0.2 >= 0.1
        result = engine.curate([c], [ex])
        assert len(result.to_merge) == 1

    def test_high_threshold_inserts_more(self) -> None:
        config = CuratorConfig(similarity_threshold=0.99)
        engine = CuratorEngine(config)
        c = _candidate("use cargo check for fast feedback")
        ex = _existing(content="use cargo check for quick feedback")
        # Not identical, Jaccard < 0.99
        result = engine.curate([c], [ex])
        assert len(result.to_add) == 1
        assert result.to_merge == []


# ── CuratorEngine — config / properties ──────────────────────────────


class TestCuratorProperties:
    def test_default_threshold(self) -> None:
        engine = CuratorEngine()
        assert engine.threshold == 0.8

    def test_custom_threshold(self) -> None:
        engine = CuratorEngine(CuratorConfig(similarity_threshold=0.5))
        assert engine.threshold == 0.5


# ── CurateResult dataclass defaults ──────────────────────────────────


class TestCurateResult:
    def test_default_empty_lists(self) -> None:
        r = CurateResult()
        assert r.to_add == []
        assert r.to_merge == []
        assert r.to_skip == []

    def test_independent_instances(self) -> None:
        r1 = CurateResult()
        r2 = CurateResult()
        r1.to_add.append(_candidate("x"))
        assert r2.to_add == []


# ── ExistingBullet dataclass ─────────────────────────────────────────


class TestExistingBullet:
    def test_defaults(self) -> None:
        eb = ExistingBullet(bullet_id="b1", content="hello")
        assert eb.bullet_id == "b1"
        assert eb.content == "hello"
        assert eb.embedding is None
        assert eb.metadata == {}

    def test_with_embedding(self) -> None:
        eb = ExistingBullet(bullet_id="b1", content="hi", embedding=[1.0, 2.0])
        assert eb.embedding == [1.0, 2.0]


# ── MergeCandidate dataclass ────────────────────────────────────────


class TestMergeCandidate:
    def test_fields(self) -> None:
        c = _candidate("hello")
        ex = _existing()
        mc = MergeCandidate(candidate=c, existing=ex, similarity=0.85)
        assert mc.candidate is c
        assert mc.existing is ex
        assert mc.similarity == 0.85
