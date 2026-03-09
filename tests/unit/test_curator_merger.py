"""Unit tests for memx.engines.curator.merger — MergeStrategy implementations."""

from __future__ import annotations

import logging
from datetime import datetime

import pytest

from memx.core.engines.curator.engine import ExistingBullet
from memx.core.engines.curator.merger import (
    KeepBestStrategy,
    MergeContentStrategy,
    MergeResult,
    get_merge_strategy,
)
from memx.core.types import CandidateBullet


# ── Helper factories ─────────────────────────────────────────────────


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
        metadata={
            "instructivity_score": score,
            "recall_count": recall_count,
            "related_tools": related_tools or [],
            "key_entities": key_entities or [],
            "tags": tags or [],
        },
    )


# ── MergeResult dataclass ────────────────────────────────────────────


class TestMergeResult:
    def test_defaults(self) -> None:
        r = MergeResult(merged_content="hello")
        assert r.merged_content == "hello"
        assert r.merged_metadata == {}
        assert r.source_id == ""
        assert r.strategy_used == ""

    def test_full_construction(self) -> None:
        r = MergeResult(
            merged_content="content",
            merged_metadata={"score": 80.0},
            source_id="b42",
            strategy_used="keep_best",
        )
        assert r.source_id == "b42"
        assert r.strategy_used == "keep_best"


# ── KeepBestStrategy ─────────────────────────────────────────────────


class TestKeepBestStrategy:
    def test_candidate_higher_score_wins(self) -> None:
        """Candidate with higher instructivity_score is kept."""
        strategy = KeepBestStrategy()
        c = _candidate(content="candidate text", score=90.0)
        ex = _existing(content="existing text", score=60.0)
        result = strategy.merge(c, ex)

        assert result.merged_content == "candidate text"
        assert result.merged_metadata["instructivity_score"] == 90.0
        assert result.source_id == "b1"
        assert result.strategy_used == "keep_best"

    def test_existing_higher_score_wins(self) -> None:
        """Existing with higher instructivity_score is kept."""
        strategy = KeepBestStrategy()
        c = _candidate(content="candidate text", score=40.0)
        ex = _existing(content="existing text", score=80.0)
        result = strategy.merge(c, ex)

        assert result.merged_content == "existing text"
        assert result.merged_metadata["instructivity_score"] == 80.0

    def test_equal_score_longer_content_wins(self) -> None:
        """When scores are equal, the longer content wins."""
        strategy = KeepBestStrategy()
        c = _candidate(content="a much longer candidate content string", score=70.0)
        ex = _existing(content="short", score=70.0)
        result = strategy.merge(c, ex)

        assert result.merged_content == "a much longer candidate content string"

    def test_equal_score_equal_length_keeps_existing(self) -> None:
        """When scores and lengths are equal, existing is kept."""
        strategy = KeepBestStrategy()
        c = _candidate(content="abcd", score=70.0)
        ex = _existing(content="efgh", score=70.0)
        result = strategy.merge(c, ex)

        assert result.merged_content == "efgh"

    def test_identical_content_keeps_existing(self) -> None:
        """Identical content with equal scores keeps existing."""
        strategy = KeepBestStrategy()
        c = _candidate(content="same content", score=50.0)
        ex = _existing(content="same content", score=50.0)
        result = strategy.merge(c, ex)

        assert result.merged_content == "same content"

    def test_related_tools_union(self) -> None:
        """related_tools from both sources are unioned."""
        strategy = KeepBestStrategy()
        c = _candidate(
            content="c", score=90.0,
            related_tools=["cargo", "rustc"],
        )
        ex = _existing(
            content="e", score=60.0,
            related_tools=["cargo", "clippy"],
        )
        result = strategy.merge(c, ex)
        tools = result.merged_metadata["related_tools"]

        assert set(tools) == {"cargo", "rustc", "clippy"}
        # No duplicates
        assert len(tools) == 3

    def test_key_entities_union(self) -> None:
        """key_entities from both sources are unioned."""
        strategy = KeepBestStrategy()
        c = _candidate(content="c", score=90.0, key_entities=["Rust", "WASM"])
        ex = _existing(content="e", score=60.0, key_entities=["Rust", "LLVM"])
        result = strategy.merge(c, ex)

        entities = result.merged_metadata["key_entities"]
        assert set(entities) == {"Rust", "WASM", "LLVM"}

    def test_tags_union(self) -> None:
        """tags from both sources are unioned."""
        strategy = KeepBestStrategy()
        c = _candidate(content="c", score=90.0, tags=["perf", "build"])
        ex = _existing(content="e", score=60.0, tags=["build", "ci"])
        result = strategy.merge(c, ex)

        tags = result.merged_metadata["tags"]
        assert set(tags) == {"perf", "build", "ci"}

    def test_updated_at_is_set(self) -> None:
        """updated_at is set to a recent timestamp."""
        strategy = KeepBestStrategy()
        c = _candidate(content="c", score=90.0)
        ex = _existing(content="e", score=60.0)
        result = strategy.merge(c, ex)

        ts = result.merged_metadata["updated_at"]
        # Should be a valid ISO timestamp
        parsed = datetime.fromisoformat(ts)
        assert parsed.year >= 2026

    def test_recall_count_preserved(self) -> None:
        """recall_count from existing is always preserved."""
        strategy = KeepBestStrategy()
        c = _candidate(content="c", score=90.0)
        ex = _existing(content="e", score=60.0, recall_count=42)
        result = strategy.merge(c, ex)

        assert result.merged_metadata["recall_count"] == 42

    def test_empty_candidate_content_keeps_existing(self) -> None:
        """When candidate content is empty, existing wins regardless of score."""
        strategy = KeepBestStrategy()
        # Candidate has higher score but empty content
        c = _candidate(content="", score=100.0)
        ex = _existing(content="real content", score=30.0)
        result = strategy.merge(c, ex)

        # existing wins because candidate content length is 0
        assert result.merged_content == "real content"

    def test_empty_existing_content_candidate_wins(self) -> None:
        """When existing content is empty, candidate should win on length."""
        strategy = KeepBestStrategy()
        c = _candidate(content="real content", score=50.0)
        ex = _existing(content="", score=50.0)
        result = strategy.merge(c, ex)

        assert result.merged_content == "real content"


# ── MergeContentStrategy ─────────────────────────────────────────────


class TestMergeContentStrategy:
    def test_basic_merge(self) -> None:
        """Two different content strings are concatenated."""
        strategy = MergeContentStrategy()
        c = _candidate(content="Use cargo check", score=70.0)
        ex = _existing(content="Run clippy for lints", score=60.0)
        result = strategy.merge(c, ex)

        assert "cargo check" in result.merged_content
        assert "clippy" in result.merged_content
        assert result.strategy_used == "merge_content"

    def test_identical_content_no_duplication(self) -> None:
        """Identical content is not duplicated."""
        strategy = MergeContentStrategy()
        c = _candidate(content="Use cargo check", score=70.0)
        ex = _existing(content="Use cargo check", score=60.0)
        result = strategy.merge(c, ex)

        assert result.merged_content == "Use cargo check"

    def test_higher_score_retained(self) -> None:
        """The higher instructivity_score is kept."""
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

    def test_related_tools_union(self) -> None:
        """related_tools from both sources are unioned."""
        strategy = MergeContentStrategy()
        c = _candidate(
            content="A", score=50.0,
            related_tools=["npm", "webpack"],
        )
        ex = _existing(
            content="B", score=50.0,
            related_tools=["npm", "vite"],
        )
        result = strategy.merge(c, ex)

        tools = result.merged_metadata["related_tools"]
        assert set(tools) == {"npm", "webpack", "vite"}

    def test_key_entities_union(self) -> None:
        """key_entities from both sources are unioned."""
        strategy = MergeContentStrategy()
        c = _candidate(content="A", score=50.0, key_entities=["React", "JSX"])
        ex = _existing(content="B", score=50.0, key_entities=["React", "Hooks"])
        result = strategy.merge(c, ex)

        entities = result.merged_metadata["key_entities"]
        assert set(entities) == {"React", "JSX", "Hooks"}

    def test_tags_union(self) -> None:
        """tags from both sources are unioned."""
        strategy = MergeContentStrategy()
        c = _candidate(content="A", score=50.0, tags=["frontend", "perf"])
        ex = _existing(content="B", score=50.0, tags=["frontend", "ux"])
        result = strategy.merge(c, ex)

        tags = result.merged_metadata["tags"]
        assert set(tags) == {"frontend", "perf", "ux"}

    def test_recall_count_preserved(self) -> None:
        """recall_count from existing is preserved."""
        strategy = MergeContentStrategy()
        c = _candidate(content="A", score=50.0)
        ex = _existing(content="B", score=50.0, recall_count=17)
        result = strategy.merge(c, ex)

        assert result.merged_metadata["recall_count"] == 17

    def test_updated_at_is_set(self) -> None:
        """updated_at is set to a recent timestamp."""
        strategy = MergeContentStrategy()
        c = _candidate(content="A", score=50.0)
        ex = _existing(content="B", score=50.0)
        result = strategy.merge(c, ex)

        ts = result.merged_metadata["updated_at"]
        parsed = datetime.fromisoformat(ts)
        assert parsed.year >= 2026

    def test_source_id_from_existing(self) -> None:
        """source_id is the existing bullet's ID."""
        strategy = MergeContentStrategy()
        c = _candidate(content="A", score=50.0)
        ex = _existing(bullet_id="b99", content="B", score=50.0)
        result = strategy.merge(c, ex)

        assert result.source_id == "b99"

    def test_empty_candidate_content(self) -> None:
        """Empty candidate content returns existing content only."""
        strategy = MergeContentStrategy()
        c = _candidate(content="", score=50.0)
        ex = _existing(content="Keep this", score=50.0)
        result = strategy.merge(c, ex)

        assert result.merged_content == "Keep this"

    def test_empty_existing_content(self) -> None:
        """Empty existing content returns candidate content only."""
        strategy = MergeContentStrategy()
        c = _candidate(content="New stuff", score=50.0)
        ex = _existing(content="", score=50.0)
        result = strategy.merge(c, ex)

        assert result.merged_content == "New stuff"

    def test_sentence_level_dedup(self) -> None:
        """Overlapping sentences are not duplicated."""
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


# ── get_merge_strategy factory ────────────────────────────────────────


class TestGetMergeStrategy:
    def test_keep_best(self) -> None:
        strategy = get_merge_strategy("keep_best")
        assert isinstance(strategy, KeepBestStrategy)

    def test_merge_content(self) -> None:
        strategy = get_merge_strategy("merge_content")
        assert isinstance(strategy, MergeContentStrategy)

    def test_unknown_falls_back_to_keep_best(self) -> None:
        """Unknown strategy name falls back to keep_best with a warning."""
        strategy = get_merge_strategy("nonexistent_strategy")
        assert isinstance(strategy, KeepBestStrategy)

    def test_unknown_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify warning is logged for unknown strategy."""
        with caplog.at_level(logging.WARNING):
            strategy = get_merge_strategy("bad_name")
        assert isinstance(strategy, KeepBestStrategy)
        assert "Unknown merge strategy" in caplog.text

    def test_empty_string_falls_back(self) -> None:
        """Empty string falls back to keep_best."""
        strategy = get_merge_strategy("")
        assert isinstance(strategy, KeepBestStrategy)


# ── Integration: Strategy via CuratorConfig ──────────────────────────


class TestStrategyFromConfig:
    def test_config_default_is_keep_best(self) -> None:
        from memx.core.config import CuratorConfig

        config = CuratorConfig()
        strategy = get_merge_strategy(config.merge_strategy)
        assert isinstance(strategy, KeepBestStrategy)

    def test_config_merge_content(self) -> None:
        from memx.core.config import CuratorConfig

        config = CuratorConfig(merge_strategy="merge_content")
        strategy = get_merge_strategy(config.merge_strategy)
        assert isinstance(strategy, MergeContentStrategy)
