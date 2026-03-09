"""Federation Mode end-to-end integration tests.

Tests the full sync/search/nomination flow using mocked components.
No real network calls or external dependencies required.

STORY-067: Federation MVP End-to-End Integration Tests
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memx.team.cache_storage import TeamCacheStorage
from memx.team.config import AutoNominateConfig, RedactorConfig, TeamConfig
from memx.team.merger import LayerBoostConfig, MultiPoolRetriever
from memx.team.nominator import Nominator
from memx.team.redactor import Redactor
from memx.team.sync_client import (
    BulletIndexEntry,
    IndexResponse,
    NominateResponse,
    SyncConnectionError,
)
from memx.team.sync_manager import SyncManager
from memx.team.tombstone import TombstoneManager
from memx.team.types import TeamBullet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_team_bullet(
    content: str,
    *,
    origin_id: str | None = None,
    tags: list[str] | None = None,
    enforcement: str = "suggestion",
    instructivity_score: float = 70.0,
    status: str = "approved",
    recall_count: int = 0,
) -> TeamBullet:
    """Create a TeamBullet with sensible defaults for testing."""
    return TeamBullet(
        content=content,
        origin_id=origin_id,
        tags=tags or [],
        enforcement=enforcement,
        instructivity_score=instructivity_score,
        status=status,
        recall_count=recall_count,
    )


class FakeLocalStorage:
    """Minimal local backend returning canned results."""

    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        self._results = results or []

    def search(
        self, query: str, *, limit: int = 10, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return self._results[:limit]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def team_config(tmp_path: Path) -> TeamConfig:
    return TeamConfig(
        enabled=True,
        server_url="https://ace.example.com",
        team_id="test-team",
        cache_max_bullets=100,
    )


@pytest.fixture()
def team_cache(team_config: TeamConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TeamCacheStorage:
    """TeamCacheStorage with temp directory for cache files."""
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    return TeamCacheStorage(team_config)


@pytest.fixture()
def mock_sync_client() -> AsyncMock:
    """Mock AceSyncClient with controllable responses."""
    client = AsyncMock()
    client.pull_index = AsyncMock(
        return_value=IndexResponse(bullets=[], cursor=None)
    )
    client.fetch_bullets = AsyncMock(return_value=[])
    client.nominate_bullet = AsyncMock(
        return_value=NominateResponse(id="nom-001", status="staging")
    )
    return client


# ---------------------------------------------------------------------------
# TestIncrementalSync
# ---------------------------------------------------------------------------


class TestIncrementalSync:
    """Test incremental sync pulls new bullets correctly."""

    def test_full_sync_first_time(
        self,
        team_cache: TeamCacheStorage,
        mock_sync_client: AsyncMock,
        team_config: TeamConfig,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """First sync pulls all bullets from server index."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        # Server returns 3 bullets in the index
        now = datetime.now(timezone.utc)
        mock_sync_client.pull_index.return_value = IndexResponse(
            bullets=[
                BulletIndexEntry(id="b-1", updated_at=now, status="approved"),
                BulletIndexEntry(id="b-2", updated_at=now, status="approved"),
                BulletIndexEntry(id="b-3", updated_at=now, status="approved"),
            ],
            cursor=None,
        )
        # fetch_bullets returns full bullet data
        mock_sync_client.fetch_bullets.return_value = [
            {"content": "Use pytest fixtures", "origin_id": "b-1", "tags": ["python"]},
            {"content": "Always lint before commit", "origin_id": "b-2", "tags": ["ci"]},
            {"content": "Pin dependency versions", "origin_id": "b-3", "tags": ["deps"]},
        ]

        manager = SyncManager(team_cache, mock_sync_client, team_config)
        manager.sync_now()

        assert team_cache.bullet_count == 3
        assert manager.last_sync_status == "success"
        mock_sync_client.pull_index.assert_called_once()
        mock_sync_client.fetch_bullets.assert_called_once_with(["b-1", "b-2", "b-3"])

    def test_incremental_sync_with_timestamp(
        self,
        team_cache: TeamCacheStorage,
        mock_sync_client: AsyncMock,
        team_config: TeamConfig,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Incremental sync passes 'since' timestamp and pulls only new bullets."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        # Pre-populate cache with 1 bullet
        team_cache.add_bullets([_make_team_bullet("Existing rule", origin_id="b-1")])
        assert team_cache.bullet_count == 1

        now = datetime.now(timezone.utc)
        # First sync: no updates
        mock_sync_client.pull_index.return_value = IndexResponse(bullets=[], cursor=None)
        manager = SyncManager(team_cache, mock_sync_client, team_config)
        manager.sync_now()

        # Second sync: server returns 1 new bullet
        mock_sync_client.pull_index.return_value = IndexResponse(
            bullets=[
                BulletIndexEntry(id="b-2", updated_at=now, status="approved"),
            ],
            cursor=None,
        )
        mock_sync_client.fetch_bullets.return_value = [
            {"content": "New team rule", "origin_id": "b-2", "tags": ["new"]},
        ]
        manager.sync_now()

        assert team_cache.bullet_count == 2
        assert manager.last_sync_status == "success"
        assert manager.sync_count == 2


# ---------------------------------------------------------------------------
# TestTombstoneIntegration
# ---------------------------------------------------------------------------


class TestTombstoneIntegration:
    """Test tombstone mechanism cleans deleted bullets."""

    def test_tombstone_removes_from_cache(
        self,
        team_cache: TeamCacheStorage,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Tombstone entries remove corresponding bullets from cache."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        # Add bullets
        team_cache.add_bullets([
            _make_team_bullet("Rule A", origin_id="b-1"),
            _make_team_bullet("Rule B", origin_id="b-2"),
            _make_team_bullet("Rule C", origin_id="b-3"),
        ])
        assert team_cache.bullet_count == 3

        # Process tombstone for b-2
        tombstone_mgr = TombstoneManager(team_cache)
        now = datetime.now(timezone.utc)
        removed = tombstone_mgr.process_tombstones([
            BulletIndexEntry(id="b-2", updated_at=now, status="tombstone"),
        ])

        assert removed == ["b-2"]
        assert team_cache.bullet_count == 2
        assert team_cache.get_bullet("b-2") is None
        assert team_cache.get_bullet("b-1") is not None
        assert team_cache.get_bullet("b-3") is not None

    def test_full_sync_check_removes_stale(
        self,
        team_cache: TeamCacheStorage,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Full sync check removes locally cached bullets absent from server."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        team_cache.add_bullets([
            _make_team_bullet("Rule A", origin_id="b-1"),
            _make_team_bullet("Rule B", origin_id="b-2"),
            _make_team_bullet("Rule C", origin_id="b-3"),
        ])

        tombstone_mgr = TombstoneManager(team_cache)
        # Server only has b-1 and b-3
        stale = tombstone_mgr.full_sync_check({"b-1", "b-3"})

        assert "b-2" in stale
        assert team_cache.bullet_count == 2

    def test_tombstone_ignores_non_tombstone_entries(
        self,
        team_cache: TeamCacheStorage,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Only entries with status='tombstone' are processed."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        team_cache.add_bullets([
            _make_team_bullet("Rule A", origin_id="b-1"),
        ])

        tombstone_mgr = TombstoneManager(team_cache)
        now = datetime.now(timezone.utc)
        removed = tombstone_mgr.process_tombstones([
            BulletIndexEntry(id="b-1", updated_at=now, status="approved"),
        ])

        assert removed == []
        assert team_cache.bullet_count == 1


# ---------------------------------------------------------------------------
# TestMergedSearch
# ---------------------------------------------------------------------------


class TestMergedSearch:
    """Test Local + Team Cache merged search results."""

    def test_local_and_team_results_merged(
        self,
        team_cache: TeamCacheStorage,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Results from both local and team pools are merged."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        # Set up team cache with a searchable bullet
        team_cache.add_bullets([
            _make_team_bullet(
                "Use pytest fixtures for test isolation",
                origin_id="t-1",
                tags=["testing"],
            ),
        ])

        # Local results
        local = FakeLocalStorage(results=[
            {"content": "Always write unit tests first", "tags": ["testing"], "score": 0.9},
        ])

        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("team_cache", team_cache)],
        )

        results = retriever.search("testing")
        assert len(results) >= 2
        contents = [r["content"] for r in results]
        assert "Always write unit tests first" in contents
        assert any("pytest fixtures" in c for c in contents)

    def test_local_only_when_team_empty(self) -> None:
        """When team cache is empty, only local results are returned."""
        local = FakeLocalStorage(results=[
            {"content": "Local rule about deployment", "tags": ["deploy"], "score": 0.8},
        ])
        # Empty team pool
        empty_team = FakeLocalStorage(results=[])

        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("team_cache", empty_team)],
        )

        results = retriever.search("deployment")
        assert len(results) == 1
        assert results[0]["content"] == "Local rule about deployment"

    def test_team_only_when_local_empty(
        self,
        team_cache: TeamCacheStorage,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """When local returns nothing, team results are still returned."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        team_cache.add_bullets([
            _make_team_bullet(
                "Team rule about database migrations",
                origin_id="t-1",
                tags=["database"],
            ),
        ])

        local = FakeLocalStorage(results=[])
        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("team_cache", team_cache)],
        )

        results = retriever.search("database migrations")
        assert len(results) >= 1
        assert any("database migrations" in r["content"] for r in results)

    def test_shadow_merge_dedup(self) -> None:
        """Near-duplicate content across local and team is deduplicated."""
        local = FakeLocalStorage(results=[
            {"content": "always run linter before pushing code changes", "tags": ["ci"], "score": 0.8},
        ])
        # Team has near-duplicate
        team = FakeLocalStorage(results=[
            {"content": "always run linter before pushing code changes", "tags": ["ci"], "score": 0.6},
        ])

        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("team_cache", team)],
        )

        results = retriever.search("linter")
        # Exact duplicate should be merged — only 1 result
        assert len(results) == 1

    def test_mandatory_enforcement_survives_merge(
        self,
        team_cache: TeamCacheStorage,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Mandatory enforcement bullets bypass dedup and always appear."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        team_cache.add_bullets([
            _make_team_bullet(
                "MANDATORY: All PRs require two approvals",
                origin_id="t-m1",
                tags=["policy"],
                enforcement="mandatory",
            ),
        ])

        local = FakeLocalStorage(results=[
            {"content": "All PRs require two approvals", "tags": ["policy"], "score": 0.9},
        ])

        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("team_cache", team_cache)],
        )

        results = retriever.search("approvals")
        # Mandatory team bullet should survive even with near-duplicate local
        mandatory_found = any(
            r.get("enforcement") == "mandatory" for r in results
        )
        assert mandatory_found


# ---------------------------------------------------------------------------
# TestNominationE2E
# ---------------------------------------------------------------------------


class TestNominationE2E:
    """Test nomination pipeline end-to-end: detect -> redact -> upload."""

    def test_detect_redact_upload(
        self,
        mock_sync_client: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Full pipeline: scan candidates, redact PII, upload to server."""
        config = AutoNominateConfig(
            min_recall_count=2,
            min_score=60.0,
        )
        redactor = Redactor(config=RedactorConfig())

        nominator = Nominator(
            config=config,
            redactor=redactor,
            sync_client=mock_sync_client,
            state_dir=tmp_path,
        )

        # Local bullets — one qualifies, one does not
        bullets = [
            {
                "id": "local-1",
                "content": "Use connection pooling for postgres://admin:pass@10.0.0.5/mydb",
                "recall_count": 5,
                "instructivity_score": 80.0,
            },
            {
                "id": "local-2",
                "content": "Minor formatting note",
                "recall_count": 0,
                "instructivity_score": 20.0,
            },
        ]

        # Scan candidates
        candidates = nominator.scan_candidates(bullets)
        assert len(candidates) == 1
        assert candidates[0]["id"] == "local-1"

        # Nominate (redact + upload)
        result = asyncio.run(nominator.nominate(candidates[0]))
        assert result.success is True
        assert result.bullet_id == "nom-001"

        # Verify the uploaded bullet was redacted
        call_args = mock_sync_client.nominate_bullet.call_args
        uploaded = call_args[0][0]
        # PII (DB connection string) should be redacted
        assert "admin:pass" not in uploaded.get("content", "")

    def test_nomination_upload_failure_preserves_candidate(
        self,
        tmp_path: Path,
    ) -> None:
        """When upload fails, the bullet remains a candidate for retry."""
        config = AutoNominateConfig(min_recall_count=1, min_score=50.0)
        redactor = Redactor(config=RedactorConfig())

        failing_client = AsyncMock()
        failing_client.nominate_bullet = AsyncMock(
            side_effect=SyncConnectionError("Server down")
        )

        nominator = Nominator(
            config=config,
            redactor=redactor,
            sync_client=failing_client,
            state_dir=tmp_path,
        )

        bullet = {
            "id": "local-fail",
            "content": "Important rule about testing",
            "recall_count": 3,
            "instructivity_score": 75.0,
        }

        candidates = nominator.scan_candidates([bullet])
        assert len(candidates) == 1

        result = asyncio.run(nominator.nominate(bullet))
        assert result.success is False
        assert result.error is not None

        # Bullet should still be scannable (not in nominated_ids)
        new_candidates = nominator.scan_candidates([bullet])
        assert len(new_candidates) == 1

    def test_nomination_without_redactor_fails_gracefully(
        self,
        mock_sync_client: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Nomination without a redactor returns error, not crash."""
        config = AutoNominateConfig(min_recall_count=1, min_score=50.0)
        nominator = Nominator(
            config=config,
            redactor=None,
            sync_client=mock_sync_client,
            state_dir=tmp_path,
        )

        bullet = {
            "id": "no-redactor",
            "content": "Some content",
            "recall_count": 5,
            "instructivity_score": 80.0,
        }

        result = asyncio.run(nominator.nominate(bullet))
        assert result.success is False
        assert "redactor" in result.error.lower()


# ---------------------------------------------------------------------------
# TestServerDegradation
# ---------------------------------------------------------------------------


class TestServerDegradation:
    """Test fallback when server is unreachable."""

    def test_sync_failure_keeps_cache(
        self,
        team_cache: TeamCacheStorage,
        team_config: TeamConfig,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """When sync fails, existing cache data is preserved."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        # Pre-populate cache
        team_cache.add_bullets([
            _make_team_bullet("Cached rule A", origin_id="cached-1"),
            _make_team_bullet("Cached rule B", origin_id="cached-2"),
        ])
        assert team_cache.bullet_count == 2

        # Sync client that always fails
        failing_client = AsyncMock()
        failing_client.pull_index = AsyncMock(
            side_effect=SyncConnectionError("Connection refused")
        )

        manager = SyncManager(team_cache, failing_client, team_config)
        manager.sync_now()

        # Cache should be intact
        assert team_cache.bullet_count == 2
        assert manager.last_sync_status == "failed"

    def test_search_works_during_server_outage(
        self,
        team_cache: TeamCacheStorage,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Search returns local + cached team results even when server is down."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        # Team cache has data from a previous sync
        team_cache.add_bullets([
            _make_team_bullet(
                "Team best practice for error handling",
                origin_id="t-cached",
                tags=["errors"],
            ),
        ])

        local = FakeLocalStorage(results=[
            {"content": "Local error handling rule", "tags": ["errors"], "score": 0.7},
        ])

        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("team_cache", team_cache)],
        )

        # Search should work purely from local + cached team data
        results = retriever.search("error handling")
        assert len(results) >= 2
        sources_content = [r["content"] for r in results]
        assert "Local error handling rule" in sources_content
        assert any("error handling" in c for c in sources_content)

    def test_team_pool_failure_degrades_gracefully(self) -> None:
        """If a team pool raises during search, local results still returned."""

        class FailingPool:
            def search(self, query: str, *, limit: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
                raise RuntimeError("Pool crashed")

        local = FakeLocalStorage(results=[
            {"content": "Local result survives", "tags": [], "score": 0.8},
        ])

        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("broken", FailingPool())],
        )

        results = retriever.search("anything")
        assert len(results) == 1
        assert results[0]["content"] == "Local result survives"


# ---------------------------------------------------------------------------
# TestSyncManagerLifecycle
# ---------------------------------------------------------------------------


class TestSyncManagerLifecycle:
    """Additional SyncManager lifecycle and state tests."""

    def test_sync_manager_state_persistence(
        self,
        team_cache: TeamCacheStorage,
        mock_sync_client: AsyncMock,
        team_config: TeamConfig,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """SyncManager persists state to sync_state.json after sync."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        manager = SyncManager(team_cache, mock_sync_client, team_config)
        manager.sync_now()

        state_file = tmp_path / ".ace" / "team_cache" / "test-team" / "sync_state.json"
        assert state_file.exists()

    def test_sync_count_increments(
        self,
        team_cache: TeamCacheStorage,
        mock_sync_client: AsyncMock,
        team_config: TeamConfig,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Each successful sync increments the sync counter."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        manager = SyncManager(team_cache, mock_sync_client, team_config)
        assert manager.sync_count == 0

        manager.sync_now()
        assert manager.sync_count == 1

        manager.sync_now()
        assert manager.sync_count == 2
