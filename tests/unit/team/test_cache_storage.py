"""Tests for memx.team.cache_storage — TeamCacheStorage."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from memx.team.cache_storage import TeamCacheStorage, _sanitize_team_id
from memx.team.config import TeamConfig
from memx.team.types import TeamBullet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def team_config(tmp_path: Path) -> TeamConfig:
    """TeamConfig pointing to a temp directory for cache."""
    return TeamConfig(
        enabled=True,
        team_id="test-team-001",
        cache_max_bullets=100,
    )


@pytest.fixture
def storage(team_config: TeamConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TeamCacheStorage:
    """TeamCacheStorage with cache dir redirected to tmp_path."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return TeamCacheStorage(team_config)


def _make_bullet(
    content: str = "Use pytest for testing",
    score: float = 50.0,
    tags: list[str] | None = None,
    enforcement: str = "suggestion",
    origin_id: str | None = None,
    status: str = "approved",
    upvotes: int = 0,
    downvotes: int = 0,
    context_summary: str | None = None,
) -> TeamBullet:
    """Helper to construct a TeamBullet with sensible defaults."""
    return TeamBullet(
        content=content,
        instructivity_score=score,
        tags=tags or [],
        enforcement=enforcement,
        origin_id=origin_id,
        status=status,
        upvotes=upvotes,
        downvotes=downvotes,
        context_summary=context_summary,
    )


# ---------------------------------------------------------------------------
# StorageBackend Protocol compliance
# ---------------------------------------------------------------------------


class TestStorageBackendProtocol:
    """Verify TeamCacheStorage satisfies the StorageBackend protocol."""

    def test_isinstance_check(self, storage: TeamCacheStorage) -> None:
        """TeamCacheStorage must be runtime-checkable as StorageBackend."""
        from memx.team.merger import StorageBackend

        assert isinstance(storage, StorageBackend)

    def test_search_signature(self, storage: TeamCacheStorage) -> None:
        """search() accepts query, limit keyword, and **kwargs."""
        result = storage.search("anything", limit=5, extra_param="ignored")
        assert isinstance(result, list)

    def test_search_returns_list_of_dicts(self, storage: TeamCacheStorage) -> None:
        storage.add_bullets([_make_bullet(content="Python typing best practices")])
        results = storage.search("Python", limit=5)
        assert isinstance(results, list)
        if results:
            assert isinstance(results[0], dict)


# ---------------------------------------------------------------------------
# Empty cache behavior
# ---------------------------------------------------------------------------


class TestEmptyCache:
    """Empty cache should return empty results, never error."""

    def test_search_empty(self, storage: TeamCacheStorage) -> None:
        assert storage.search("anything") == []

    def test_bullet_count_empty(self, storage: TeamCacheStorage) -> None:
        assert storage.bullet_count == 0

    def test_last_sync_time_none(self, storage: TeamCacheStorage) -> None:
        assert storage.last_sync_time is None

    def test_get_bullet_empty(self, storage: TeamCacheStorage) -> None:
        assert storage.get_bullet("nonexistent") is None


# ---------------------------------------------------------------------------
# Keyword search
# ---------------------------------------------------------------------------


class TestKeywordSearch:
    """Test keyword-based search functionality."""

    def test_exact_substring_match(self, storage: TeamCacheStorage) -> None:
        storage.add_bullets([
            _make_bullet(content="Always use black for code formatting"),
            _make_bullet(content="Prefer ruff over flake8"),
        ])
        results = storage.search("black", limit=10)
        assert len(results) >= 1
        assert results[0]["content"] == "Always use black for code formatting"
        assert results[0]["source"] == "team_cache"

    def test_tag_match(self, storage: TeamCacheStorage) -> None:
        storage.add_bullets([
            _make_bullet(content="Some unrelated content", tags=["python", "testing"]),
        ])
        results = storage.search("python", limit=10)
        assert len(results) >= 1

    def test_case_insensitive(self, storage: TeamCacheStorage) -> None:
        storage.add_bullets([
            _make_bullet(content="Use PYTEST for testing"),
        ])
        results = storage.search("pytest", limit=10)
        assert len(results) >= 1

    def test_no_match_returns_empty(self, storage: TeamCacheStorage) -> None:
        storage.add_bullets([
            _make_bullet(content="Something about python"),
        ])
        results = storage.search("zzzzxyznonexistent", limit=10)
        assert results == []

    def test_context_summary_match(self, storage: TeamCacheStorage) -> None:
        storage.add_bullets([
            _make_bullet(
                content="Use structured logging",
                context_summary="Logging practices for microservices",
            ),
        ])
        results = storage.search("microservices", limit=10)
        assert len(results) >= 1

    def test_limit_respected(self, storage: TeamCacheStorage) -> None:
        bullets = [
            _make_bullet(content=f"Python tip number {i}", origin_id=f"tip-{i}")
            for i in range(20)
        ]
        storage.add_bullets(bullets)
        results = storage.search("Python", limit=3)
        assert len(results) <= 3

    def test_inactive_bullets_excluded(self, storage: TeamCacheStorage) -> None:
        storage.add_bullets([
            _make_bullet(content="Active bullet about python", status="approved"),
            _make_bullet(content="Deprecated bullet about python", status="deprecated"),
        ])
        results = storage.search("python", limit=10)
        # Only active bullet should appear
        assert all(r["content"] != "Deprecated bullet about python" for r in results)

    def test_result_dict_format(self, storage: TeamCacheStorage) -> None:
        storage.add_bullets([
            _make_bullet(
                content="Use type hints everywhere",
                score=80.0,
                tags=["python", "typing"],
                enforcement="recommended",
            ),
        ])
        results = storage.search("type hints", limit=1)
        assert len(results) == 1
        r = results[0]
        assert "content" in r
        assert "section" in r
        assert "knowledge_type" in r
        assert "instructivity_score" in r
        assert "tags" in r
        assert "enforcement" in r
        assert "score" in r
        assert r["source"] == "team_cache"
        assert r["enforcement"] == "recommended"


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------


class TestVectorSearch:
    """Test vector search with mocked ONNXEmbedder."""

    def _make_storage_with_mock_embedder(
        self, tmp_path: Path, config: TeamConfig
    ) -> tuple[TeamCacheStorage, MagicMock]:
        """Create storage with a mocked embedder that returns deterministic vectors."""
        import numpy as np

        mock_embedder = MagicMock()

        # Generate deterministic embeddings based on content hash
        def fake_embed(text: str) -> list[float]:
            vec = [0.0] * 8
            for i, ch in enumerate(text[:8]):
                vec[i] = ord(ch) / 255.0
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            return vec

        def fake_embed_batch(texts: list[str]) -> list[list[float]]:
            return [fake_embed(t) for t in texts]

        mock_embedder.embed = fake_embed
        mock_embedder.embed_batch = fake_embed_batch

        with patch.object(Path, "home", return_value=tmp_path):
            store = TeamCacheStorage(config)
            store._embedder = mock_embedder

        return store, mock_embedder

    def test_vector_search_basic(self, tmp_path: Path, team_config: TeamConfig) -> None:
        store, embedder = self._make_storage_with_mock_embedder(tmp_path, team_config)
        store.add_bullets([
            _make_bullet(content="Use pytest fixtures", origin_id="v1"),
            _make_bullet(content="Use pytest parametrize", origin_id="v2"),
            _make_bullet(content="Deploy with docker compose", origin_id="v3"),
        ])
        # Should find pytest-related bullets
        results = store.search("Use pytest", limit=2)
        assert len(results) >= 1

    def test_vector_search_falls_back_on_no_embedder(self, storage: TeamCacheStorage) -> None:
        """When embedder is None, should fall back to keyword search."""
        storage.add_bullets([
            _make_bullet(content="Keyword searchable content"),
        ])
        storage._vectors = None
        storage._embedder = None
        results = storage.search("Keyword", limit=5)
        assert len(results) >= 1

    def test_vector_ids_alignment(self, tmp_path: Path, team_config: TeamConfig) -> None:
        """Vector index IDs must align with bullets dict."""
        store, _ = self._make_storage_with_mock_embedder(tmp_path, team_config)
        bullets = [
            _make_bullet(content=f"Bullet {i}", origin_id=f"b-{i}")
            for i in range(5)
        ]
        store.add_bullets(bullets)
        assert len(store._vector_ids) == 5
        for vid in store._vector_ids:
            assert store.get_bullet(vid) is not None


# ---------------------------------------------------------------------------
# Capacity limit eviction
# ---------------------------------------------------------------------------


class TestCapacityLimit:
    """Test cache_max_bullets enforcement."""

    def test_eviction_triggers_at_limit(self, tmp_path: Path) -> None:
        config = TeamConfig(enabled=True, team_id="cap-test", cache_max_bullets=5)
        with patch.object(Path, "home", return_value=tmp_path):
            store = TeamCacheStorage(config)

        bullets = [
            _make_bullet(
                content=f"Bullet {i}",
                score=float(i * 10),
                origin_id=f"cap-{i}",
            )
            for i in range(10)
        ]
        store.add_bullets(bullets)
        assert store.bullet_count == 5

    def test_keeps_highest_effective_score(self, tmp_path: Path) -> None:
        config = TeamConfig(enabled=True, team_id="cap-test", cache_max_bullets=3)
        with patch.object(Path, "home", return_value=tmp_path):
            store = TeamCacheStorage(config)

        bullets = [
            _make_bullet(content="Low score", score=10.0, origin_id="low"),
            _make_bullet(content="Mid score", score=50.0, origin_id="mid"),
            _make_bullet(content="High score", score=90.0, origin_id="high"),
            _make_bullet(content="Very high", score=95.0, origin_id="vhigh"),
            _make_bullet(content="Mega high", score=99.0, origin_id="mega"),
        ]
        store.add_bullets(bullets)
        assert store.bullet_count == 3
        # Low and mid should be evicted
        assert store.get_bullet("low") is None
        assert store.get_bullet("mid") is None
        assert store.get_bullet("high") is not None
        assert store.get_bullet("vhigh") is not None
        assert store.get_bullet("mega") is not None

    def test_under_limit_no_eviction(self, storage: TeamCacheStorage) -> None:
        bullets = [
            _make_bullet(content=f"Bullet {i}", origin_id=f"ok-{i}")
            for i in range(5)
        ]
        storage.add_bullets(bullets)
        assert storage.bullet_count == 5


# ---------------------------------------------------------------------------
# Add / Remove / Get
# ---------------------------------------------------------------------------


class TestMutations:
    """Test add_bullets, remove_bullets, get_bullet."""

    def test_add_and_get(self, storage: TeamCacheStorage) -> None:
        b = _make_bullet(content="Test bullet", origin_id="get-1")
        storage.add_bullets([b])
        retrieved = storage.get_bullet("get-1")
        assert retrieved is not None
        assert retrieved.content == "Test bullet"

    def test_add_updates_existing(self, storage: TeamCacheStorage) -> None:
        b1 = _make_bullet(content="Version 1", origin_id="upd-1", score=50.0)
        storage.add_bullets([b1])
        b2 = _make_bullet(content="Version 2", origin_id="upd-1", score=80.0)
        storage.add_bullets([b2])
        assert storage.bullet_count == 1
        assert storage.get_bullet("upd-1").content == "Version 2"

    def test_remove_bullets(self, storage: TeamCacheStorage) -> None:
        storage.add_bullets([
            _make_bullet(content="A", origin_id="rm-1"),
            _make_bullet(content="B", origin_id="rm-2"),
        ])
        storage.remove_bullets(["rm-1"])
        assert storage.bullet_count == 1
        assert storage.get_bullet("rm-1") is None
        assert storage.get_bullet("rm-2") is not None

    def test_remove_nonexistent_id(self, storage: TeamCacheStorage) -> None:
        """Removing a nonexistent ID should not error."""
        storage.add_bullets([_make_bullet(content="A", origin_id="keep")])
        storage.remove_bullets(["nonexistent"])
        assert storage.bullet_count == 1

    def test_add_empty_list(self, storage: TeamCacheStorage) -> None:
        storage.add_bullets([])
        assert storage.bullet_count == 0

    def test_remove_empty_list(self, storage: TeamCacheStorage) -> None:
        storage.add_bullets([_make_bullet(content="A", origin_id="keep")])
        storage.remove_bullets([])
        assert storage.bullet_count == 1

    def test_bullet_without_origin_id_gets_uuid(self, storage: TeamCacheStorage) -> None:
        b = _make_bullet(content="No origin ID", origin_id=None)
        storage.add_bullets([b])
        assert storage.bullet_count == 1


# ---------------------------------------------------------------------------
# Cache persistence and reload
# ---------------------------------------------------------------------------


class TestPersistence:
    """Test saving to and loading from disk."""

    def test_persist_and_reload(self, tmp_path: Path) -> None:
        config = TeamConfig(enabled=True, team_id="persist-test")

        with patch.object(Path, "home", return_value=tmp_path):
            store1 = TeamCacheStorage(config)
        store1.add_bullets([
            _make_bullet(content="Persisted bullet", origin_id="p-1", score=75.0),
            _make_bullet(content="Another persisted", origin_id="p-2", score=60.0),
        ])

        # Create a new storage instance — should load from disk
        with patch.object(Path, "home", return_value=tmp_path):
            store2 = TeamCacheStorage(config)
        assert store2.bullet_count == 2
        b = store2.get_bullet("p-1")
        assert b is not None
        assert b.content == "Persisted bullet"
        assert b.instructivity_score == 75.0

    def test_persist_creates_directory(self, tmp_path: Path) -> None:
        config = TeamConfig(enabled=True, team_id="mkdir-test")
        with patch.object(Path, "home", return_value=tmp_path):
            store = TeamCacheStorage(config)
        store.add_bullets([_make_bullet(content="Hello", origin_id="dir-1")])
        cache_dir = tmp_path / ".ace" / "team_cache" / "mkdir-test"
        assert cache_dir.exists()
        assert (cache_dir / "bullets.json").exists()

    def test_last_sync_time_updated(self, storage: TeamCacheStorage) -> None:
        assert storage.last_sync_time is None
        storage.add_bullets([_make_bullet(content="Sync test", origin_id="sync-1")])
        assert storage.last_sync_time is not None

    def test_cache_file_is_valid_json(self, tmp_path: Path) -> None:
        config = TeamConfig(enabled=True, team_id="json-test")
        with patch.object(Path, "home", return_value=tmp_path):
            store = TeamCacheStorage(config)
        store.add_bullets([_make_bullet(content="JSON valid", origin_id="j-1")])

        cache_file = tmp_path / ".ace" / "team_cache" / "json-test" / "bullets.json"
        with cache_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert "bullets" in data
        assert "j-1" in data["bullets"]


# ---------------------------------------------------------------------------
# Corrupt cache recovery
# ---------------------------------------------------------------------------


class TestCorruptCacheRecovery:
    """Test recovery from corrupt or invalid cache files."""

    def test_corrupt_json(self, tmp_path: Path) -> None:
        config = TeamConfig(enabled=True, team_id="corrupt-test")
        cache_dir = tmp_path / ".ace" / "team_cache" / "corrupt-test"
        cache_dir.mkdir(parents=True)
        cache_file = cache_dir / "bullets.json"
        cache_file.write_text("{invalid json!!!}", encoding="utf-8")

        with patch.object(Path, "home", return_value=tmp_path):
            store = TeamCacheStorage(config)
        # Should recover gracefully with empty cache
        assert store.bullet_count == 0
        assert store.search("anything") == []

    def test_invalid_format(self, tmp_path: Path) -> None:
        config = TeamConfig(enabled=True, team_id="invalid-test")
        cache_dir = tmp_path / ".ace" / "team_cache" / "invalid-test"
        cache_dir.mkdir(parents=True)
        cache_file = cache_dir / "bullets.json"
        cache_file.write_text('"just a string"', encoding="utf-8")

        with patch.object(Path, "home", return_value=tmp_path):
            store = TeamCacheStorage(config)
        assert store.bullet_count == 0

    def test_invalid_bullets_field(self, tmp_path: Path) -> None:
        config = TeamConfig(enabled=True, team_id="badbullets-test")
        cache_dir = tmp_path / ".ace" / "team_cache" / "badbullets-test"
        cache_dir.mkdir(parents=True)
        cache_file = cache_dir / "bullets.json"
        cache_file.write_text('{"bullets": "not a dict"}', encoding="utf-8")

        with patch.object(Path, "home", return_value=tmp_path):
            store = TeamCacheStorage(config)
        assert store.bullet_count == 0

    def test_partial_corrupt_bullets(self, tmp_path: Path) -> None:
        """One corrupt bullet entry should not break loading of valid ones."""
        config = TeamConfig(enabled=True, team_id="partial-test")
        cache_dir = tmp_path / ".ace" / "team_cache" / "partial-test"
        cache_dir.mkdir(parents=True)

        data = {
            "team_id": "partial-test",
            "last_sync": "2026-01-01T00:00:00+00:00",
            "bullets": {
                "good-1": {
                    "content": "Good bullet",
                    "instructivity_score": 70.0,
                    "section": "general",
                    "knowledge_type": "knowledge",
                    "schema_version": 2,
                    "author_id": "",
                    "enforcement": "suggestion",
                    "upvotes": 0,
                    "downvotes": 0,
                    "status": "approved",
                },
                "bad-1": {
                    "instructivity_score": "not_a_number_but_pydantic_may_coerce",
                },
            },
        }
        cache_file = cache_dir / "bullets.json"
        with cache_file.open("w", encoding="utf-8") as f:
            json.dump(data, f)

        with patch.object(Path, "home", return_value=tmp_path):
            store = TeamCacheStorage(config)
        # At least the good bullet should load
        assert store.get_bullet("good-1") is not None


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Basic thread safety tests."""

    def test_concurrent_adds(self, storage: TeamCacheStorage) -> None:
        """Multiple threads adding bullets should not corrupt state."""
        errors: list[Exception] = []

        def add_batch(batch_id: int) -> None:
            try:
                bullets = [
                    _make_bullet(
                        content=f"Thread {batch_id} bullet {i}",
                        origin_id=f"t{batch_id}-{i}",
                    )
                    for i in range(10)
                ]
                storage.add_bullets(bullets)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_batch, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        # All 50 bullets should be present (capacity is 100)
        assert storage.bullet_count == 50

    def test_concurrent_search_and_add(self, storage: TeamCacheStorage) -> None:
        """Searching while adding should not crash."""
        storage.add_bullets([
            _make_bullet(content="Initial bullet", origin_id="init-1"),
        ])
        errors: list[Exception] = []

        def searcher() -> None:
            try:
                for _ in range(20):
                    storage.search("bullet", limit=5)
            except Exception as e:
                errors.append(e)

        def adder() -> None:
            try:
                for i in range(10):
                    storage.add_bullets([
                        _make_bullet(content=f"New {i}", origin_id=f"new-{i}"),
                    ])
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=searcher)
        t2 = threading.Thread(target=adder)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Thread errors: {errors}"


# ---------------------------------------------------------------------------
# team_id sanitization
# ---------------------------------------------------------------------------


class TestTeamIdSanitization:
    """Test path safety for team_id."""

    def test_sanitize_safe_id(self) -> None:
        assert _sanitize_team_id("my-team-01") == "my-team-01"

    def test_sanitize_slashes(self) -> None:
        assert "/" not in _sanitize_team_id("my/team")
        assert "\\" not in _sanitize_team_id("my\\team")

    def test_sanitize_special_chars(self) -> None:
        result = _sanitize_team_id("team@org:name")
        assert "@" not in result
        assert ":" not in result

    def test_sanitize_dots_preserved(self) -> None:
        assert _sanitize_team_id("team.v2") == "team.v2"

    def test_default_team_id(self, tmp_path: Path) -> None:
        config = TeamConfig(enabled=True, team_id=None)
        with patch.object(Path, "home", return_value=tmp_path):
            store = TeamCacheStorage(config)
        assert store._team_id == "default"
