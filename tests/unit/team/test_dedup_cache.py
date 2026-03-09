"""Unit tests for read-time deduplication + playbook.cache (STORY-057)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from memx.team.git_storage import GitFallbackStorage, TeamBulletRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HEADER = {"_header": True, "model": "all-MiniLM-L6-v2", "dim": 384, "version": "1.0"}


def _make_bullet(content: str, score: float = 50.0, **kwargs: object) -> dict:
    """Create a bullet dict for writing to JSONL."""
    base = {
        "content": content,
        "section": "general",
        "knowledge_type": "Knowledge",
        "instructivity_score": score,
        "schema_version": 2,
        "enforcement": "suggestion",
        "tags": [],
    }
    base.update(kwargs)
    return base


def _write_playbook(path: Path, lines: list[dict | str]) -> Path:
    """Write a playbook.jsonl to the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            if isinstance(line, str):
                f.write(line + "\n")
            else:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# Tests: Deduplication algorithm
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_exact_duplicates_removed(self, tmp_path: Path) -> None:
        """Exact duplicate content should be deduplicated."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [
                HEADER,
                _make_bullet("Always use --locked with cargo build", score=80),
                _make_bullet("Always use --locked with cargo build", score=90),
                _make_bullet("Never commit .env files", score=70),
            ],
        )
        storage = GitFallbackStorage(playbook_path=pb)
        assert storage.bullet_count == 2

    def test_near_duplicates_removed(self, tmp_path: Path) -> None:
        """Content with Jaccard similarity >= 0.90 should be deduplicated."""
        # 10 shared words out of 11 total unique => Jaccard = 10/11 ≈ 0.909
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [
                HEADER,
                _make_bullet(
                    "a b c d e f g h i j", score=80
                ),
                _make_bullet(
                    "a b c d e f g h i j k",
                    score=90,
                ),
                _make_bullet("never commit env files to git", score=70),
            ],
        )
        storage = GitFallbackStorage(playbook_path=pb)
        # The two near-duplicates should collapse to one
        assert storage.bullet_count == 2

    def test_keeps_highest_score(self, tmp_path: Path) -> None:
        """When deduplicating, the entry with highest score survives."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [
                HEADER,
                _make_bullet("Always use --locked with cargo build", score=60),
                _make_bullet("Always use --locked with cargo build", score=95),
            ],
        )
        storage = GitFallbackStorage(playbook_path=pb)
        assert storage.bullet_count == 1
        results = storage.search("locked")
        assert results[0]["instructivity_score"] == 95

    def test_no_dedup_for_dissimilar(self, tmp_path: Path) -> None:
        """Dissimilar content should not be deduplicated."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [
                HEADER,
                _make_bullet("Always use cargo build locked", score=80),
                _make_bullet("Never commit .env files to git", score=70),
                _make_bullet("Run pytest with -x flag for fast fail", score=60),
            ],
        )
        storage = GitFallbackStorage(playbook_path=pb)
        assert storage.bullet_count == 3

    def test_single_bullet_no_dedup(self, tmp_path: Path) -> None:
        """A single bullet should pass through without error."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, _make_bullet("Single rule", score=80)],
        )
        storage = GitFallbackStorage(playbook_path=pb)
        assert storage.bullet_count == 1

    def test_empty_playbook_no_cache(self, tmp_path: Path) -> None:
        """Empty playbook should not create a cache file."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER],
        )
        storage = GitFallbackStorage(playbook_path=pb)
        assert storage.bullet_count == 0
        cache_path = tmp_path / ".ace" / "playbook.cache"
        assert not cache_path.exists()

    def test_all_duplicates_keep_one(self, tmp_path: Path) -> None:
        """If all bullets are duplicates, keep the one with highest score."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [
                HEADER,
                _make_bullet("use locked cargo build", score=60),
                _make_bullet("use locked cargo build", score=80),
                _make_bullet("use locked cargo build", score=70),
            ],
        )
        storage = GitFallbackStorage(playbook_path=pb)
        assert storage.bullet_count == 1
        results = storage.search("locked")
        assert results[0]["instructivity_score"] == 80


# ---------------------------------------------------------------------------
# Tests: Cache creation and loading
# ---------------------------------------------------------------------------


class TestCacheCreation:
    def test_cache_created_on_first_load(self, tmp_path: Path) -> None:
        """First load should create playbook.cache."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [
                HEADER,
                _make_bullet("rule one", score=80),
                _make_bullet("rule one", score=60),
                _make_bullet("rule two", score=70),
            ],
        )
        storage = GitFallbackStorage(playbook_path=pb)
        _ = storage.bullet_count  # trigger load

        cache_path = tmp_path / ".ace" / "playbook.cache"
        assert cache_path.exists()

        cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
        assert cache_data["original_count"] == 3
        assert cache_data["deduped_count"] == 2
        assert "source_mtime" in cache_data
        assert "deduped_indices" in cache_data

    def test_cache_hit_on_second_load(self, tmp_path: Path) -> None:
        """Second load should use cache (no re-dedup)."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [
                HEADER,
                _make_bullet("rule alpha", score=80),
                _make_bullet("rule alpha", score=60),
                _make_bullet("rule beta", score=70),
            ],
        )
        # First load - creates cache
        s1 = GitFallbackStorage(playbook_path=pb)
        assert s1.bullet_count == 2

        # Second load - should use cache
        s2 = GitFallbackStorage(playbook_path=pb)
        assert s2.bullet_count == 2

    def test_cache_format(self, tmp_path: Path) -> None:
        """Cache file should have the expected JSON structure."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [
                HEADER,
                _make_bullet("rule A", score=90),
                _make_bullet("rule B", score=70),
            ],
        )
        storage = GitFallbackStorage(playbook_path=pb)
        _ = storage.bullet_count

        cache_path = tmp_path / ".ace" / "playbook.cache"
        cache_data = json.loads(cache_path.read_text(encoding="utf-8"))

        assert isinstance(cache_data["source_mtime"], float)
        assert isinstance(cache_data["original_count"], int)
        assert isinstance(cache_data["deduped_count"], int)
        assert isinstance(cache_data["deduped_indices"], list)


# ---------------------------------------------------------------------------
# Tests: Cache expiration
# ---------------------------------------------------------------------------


class TestCacheExpiration:
    def test_cache_invalidated_on_mtime_change(self, tmp_path: Path) -> None:
        """Cache should be rebuilt when playbook.jsonl mtime changes."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [
                HEADER,
                _make_bullet("rule one", score=80),
                _make_bullet("rule one", score=60),
            ],
        )
        # First load
        s1 = GitFallbackStorage(playbook_path=pb)
        assert s1.bullet_count == 1

        # Modify playbook (change mtime)
        time.sleep(0.05)
        _write_playbook(
            pb,
            [
                HEADER,
                _make_bullet("rule one", score=80),
                _make_bullet("rule two", score=70),
                _make_bullet("rule three", score=60),
            ],
        )

        # Second load - cache should be invalidated
        s2 = GitFallbackStorage(playbook_path=pb)
        assert s2.bullet_count == 3

    def test_corrupt_cache_triggers_rebuild(self, tmp_path: Path) -> None:
        """Corrupt cache file should be deleted and rebuilt."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [
                HEADER,
                _make_bullet("rule one", score=80),
                _make_bullet("rule one", score=60),
            ],
        )

        # Write corrupt cache
        cache_path = tmp_path / ".ace" / "playbook.cache"
        cache_path.write_text("NOT VALID JSON {{{", encoding="utf-8")

        storage = GitFallbackStorage(playbook_path=pb)
        assert storage.bullet_count == 1

        # Cache should be rebuilt (valid JSON now)
        cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
        assert "source_mtime" in cache_data

    def test_cache_count_mismatch_triggers_rebuild(self, tmp_path: Path) -> None:
        """Cache with wrong original_count should be rebuilt."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [
                HEADER,
                _make_bullet("rule one", score=80),
                _make_bullet("rule two", score=70),
            ],
        )

        # Write cache with wrong count but correct mtime
        source_mtime = os.path.getmtime(pb)
        cache_path = tmp_path / ".ace" / "playbook.cache"
        cache_data = {
            "source_mtime": source_mtime,
            "original_count": 999,
            "deduped_count": 1,
            "deduped_indices": [0],
        }
        cache_path.write_text(json.dumps(cache_data), encoding="utf-8")

        storage = GitFallbackStorage(playbook_path=pb)
        assert storage.bullet_count == 2  # rebuilt, no actual duplicates


# ---------------------------------------------------------------------------
# Tests: Gitignore management
# ---------------------------------------------------------------------------


class TestGitignore:
    def test_gitignore_created(self, tmp_path: Path) -> None:
        """playbook.cache should be added to .ace/.gitignore."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [
                HEADER,
                _make_bullet("rule one", score=80),
                _make_bullet("rule one", score=60),
            ],
        )
        storage = GitFallbackStorage(playbook_path=pb)
        _ = storage.bullet_count

        gitignore = tmp_path / ".ace" / ".gitignore"
        assert gitignore.exists()
        assert "playbook.cache" in gitignore.read_text(encoding="utf-8")

    def test_gitignore_not_duplicated(self, tmp_path: Path) -> None:
        """If playbook.cache is already in .gitignore, don't add again."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir(parents=True)
        gitignore = ace_dir / ".gitignore"
        gitignore.write_text("playbook.cache\n", encoding="utf-8")

        pb = _write_playbook(
            ace_dir / "playbook.jsonl",
            [
                HEADER,
                _make_bullet("rule one", score=80),
                _make_bullet("rule one", score=60),
            ],
        )
        storage = GitFallbackStorage(playbook_path=pb)
        _ = storage.bullet_count

        content = gitignore.read_text(encoding="utf-8")
        assert content.count("playbook.cache") == 1


# ---------------------------------------------------------------------------
# Tests: Text similarity
# ---------------------------------------------------------------------------


class TestTextSimilarity:
    def test_identical_strings(self) -> None:
        sim = GitFallbackStorage._text_similarity("hello world", "hello world")
        assert sim == 1.0

    def test_completely_different(self) -> None:
        sim = GitFallbackStorage._text_similarity("hello world", "foo bar")
        assert sim == 0.0

    def test_partial_overlap(self) -> None:
        sim = GitFallbackStorage._text_similarity(
            "use cargo build", "use cargo test"
        )
        # Jaccard: {use, cargo} / {use, cargo, build, test} = 2/4 = 0.5
        assert abs(sim - 0.5) < 0.01

    def test_empty_strings(self) -> None:
        assert GitFallbackStorage._text_similarity("", "") == 0.0
        assert GitFallbackStorage._text_similarity("hello", "") == 0.0

    def test_case_insensitive(self) -> None:
        sim = GitFallbackStorage._text_similarity("Hello World", "hello world")
        assert sim == 1.0

    def test_high_similarity_threshold(self) -> None:
        """9 out of 10 words shared => 0.9 Jaccard."""
        a = "a b c d e f g h i j"
        b = "a b c d e f g h i k"
        sim = GitFallbackStorage._text_similarity(a, b)
        # Jaccard: 9 / 11 ≈ 0.818
        assert sim < 0.90  # not enough for dedup threshold


# ---------------------------------------------------------------------------
# Tests: Zero overhead on search
# ---------------------------------------------------------------------------


class TestZeroOverhead:
    def test_search_after_dedup(self, tmp_path: Path) -> None:
        """Search should work correctly after deduplication."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [
                HEADER,
                _make_bullet("use cargo build locked", score=80),
                _make_bullet("use cargo build locked", score=60),
                _make_bullet("never commit env files", score=90),
            ],
        )
        storage = GitFallbackStorage(playbook_path=pb)

        results = storage.search("cargo")
        assert len(results) == 1
        assert results[0]["instructivity_score"] == 80

        results = storage.search("env")
        assert len(results) == 1
        assert results[0]["instructivity_score"] == 90

    def test_search_after_cache_load(self, tmp_path: Path) -> None:
        """Search should work correctly when loading from cache."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [
                HEADER,
                _make_bullet("use cargo build locked", score=80),
                _make_bullet("use cargo build locked", score=60),
                _make_bullet("never commit env files", score=90),
            ],
        )
        # First load (creates cache)
        s1 = GitFallbackStorage(playbook_path=pb)
        assert s1.bullet_count == 2

        # Second load (from cache)
        s2 = GitFallbackStorage(playbook_path=pb)
        results = s2.search("cargo")
        assert len(results) == 1
        assert results[0]["instructivity_score"] == 80
