"""Tests for TombstoneManager — tombstone processing and full sync checks."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, call

import pytest

from memx.team.sync_client import BulletIndexEntry
from memx.team.tombstone import TombstoneManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cache_mock(**overrides: Any) -> MagicMock:
    """Create a mock TeamCacheStorage with sensible defaults."""
    mock = MagicMock()
    mock._bullets = overrides.get("bullets", {})
    mock.remove_bullets = MagicMock()
    return mock


def _make_entry(
    id: str,
    status: str = "active",
    updated_at: datetime | None = None,
) -> BulletIndexEntry:
    """Create a BulletIndexEntry with test defaults."""
    return BulletIndexEntry(
        id=id,
        status=status,
        updated_at=updated_at or datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Tests: process_tombstones
# ---------------------------------------------------------------------------


class TestProcessTombstones:
    """Verify tombstone processing removes bullets from cache."""

    def test_tombstone_entries_removed_from_cache(self) -> None:
        """Tombstone entries trigger remove_bullets on the cache."""
        cache = _make_cache_mock()
        mgr = TombstoneManager(cache, retention_days=90)

        entries = [
            _make_entry("b-1", status="tombstone"),
            _make_entry("b-2", status="tombstone"),
        ]
        removed = mgr.process_tombstones(entries)

        assert removed == ["b-1", "b-2"]
        cache.remove_bullets.assert_called_once_with(["b-1", "b-2"])
        assert mgr.tombstone_count == 2

    def test_non_tombstone_entries_ignored(self) -> None:
        """Entries with status != 'tombstone' are not processed."""
        cache = _make_cache_mock()
        mgr = TombstoneManager(cache, retention_days=90)

        entries = [
            _make_entry("b-1", status="active"),
            _make_entry("b-2", status="updated"),
        ]
        removed = mgr.process_tombstones(entries)

        assert removed == []
        cache.remove_bullets.assert_not_called()
        assert mgr.tombstone_count == 0

    def test_mixed_entries(self) -> None:
        """Only tombstone entries are processed in a mixed list."""
        cache = _make_cache_mock()
        mgr = TombstoneManager(cache, retention_days=90)

        entries = [
            _make_entry("b-1", status="active"),
            _make_entry("b-2", status="tombstone"),
            _make_entry("b-3", status="active"),
        ]
        removed = mgr.process_tombstones(entries)

        assert removed == ["b-2"]
        cache.remove_bullets.assert_called_once_with(["b-2"])

    def test_empty_entries(self) -> None:
        """Empty entry list is a no-op."""
        cache = _make_cache_mock()
        mgr = TombstoneManager(cache, retention_days=90)

        removed = mgr.process_tombstones([])

        assert removed == []
        cache.remove_bullets.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: cleanup_expired
# ---------------------------------------------------------------------------


class TestCleanupExpired:
    """Verify tombstone retention and cleanup logic."""

    def test_cleanup_after_retention_period(self) -> None:
        """Tombstones older than retention_days are cleaned up."""
        cache = _make_cache_mock()
        mgr = TombstoneManager(cache, retention_days=90)

        now = datetime(2026, 3, 8, tzinfo=timezone.utc)
        old_time = now - timedelta(days=91)

        entries = [_make_entry("b-1", status="tombstone", updated_at=old_time)]
        mgr.process_tombstones(entries)
        assert mgr.tombstone_count == 1

        cleaned = mgr.cleanup_expired(now=now)

        assert cleaned == 1
        assert mgr.tombstone_count == 0

    def test_no_cleanup_before_retention_period(self) -> None:
        """Tombstones younger than retention_days are kept."""
        cache = _make_cache_mock()
        mgr = TombstoneManager(cache, retention_days=90)

        now = datetime(2026, 3, 8, tzinfo=timezone.utc)
        recent_time = now - timedelta(days=30)

        entries = [_make_entry("b-1", status="tombstone", updated_at=recent_time)]
        mgr.process_tombstones(entries)

        cleaned = mgr.cleanup_expired(now=now)

        assert cleaned == 0
        assert mgr.tombstone_count == 1

    def test_cleanup_mixed_ages(self) -> None:
        """Only expired tombstones are cleaned; recent ones remain."""
        cache = _make_cache_mock()
        mgr = TombstoneManager(cache, retention_days=90)

        now = datetime(2026, 3, 8, tzinfo=timezone.utc)

        entries = [
            _make_entry("old", status="tombstone", updated_at=now - timedelta(days=100)),
            _make_entry("new", status="tombstone", updated_at=now - timedelta(days=10)),
        ]
        mgr.process_tombstones(entries)

        cleaned = mgr.cleanup_expired(now=now)

        assert cleaned == 1
        assert mgr.tombstone_count == 1

    def test_cleanup_does_not_affect_cache(self) -> None:
        """Cleanup only removes tombstone tracking, not cache bullets."""
        cache = _make_cache_mock()
        mgr = TombstoneManager(cache, retention_days=90)

        now = datetime(2026, 3, 8, tzinfo=timezone.utc)
        old_time = now - timedelta(days=91)

        entries = [_make_entry("b-1", status="tombstone", updated_at=old_time)]
        mgr.process_tombstones(entries)

        # Reset call tracking after process_tombstones
        cache.remove_bullets.reset_mock()

        mgr.cleanup_expired(now=now)

        # Cleanup should NOT call remove_bullets again
        cache.remove_bullets.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: needs_full_sync
# ---------------------------------------------------------------------------


class TestNeedsFullSync:
    """Verify full sync detection logic."""

    def test_needs_full_sync_when_never_synced(self) -> None:
        """First sync (last_sync=None) always requires full sync."""
        cache = _make_cache_mock()
        mgr = TombstoneManager(cache, retention_days=90)

        assert mgr.needs_full_sync(last_sync=None) is True

    def test_needs_full_sync_when_too_old(self) -> None:
        """Sync older than retention window triggers full sync."""
        cache = _make_cache_mock()
        mgr = TombstoneManager(cache, retention_days=90)

        now = datetime(2026, 3, 8, tzinfo=timezone.utc)
        old_sync = now - timedelta(days=100)

        result = mgr.needs_full_sync(
            last_sync=old_sync,
            server_tombstone_cutoff=now - timedelta(days=90),
        )

        assert result is True

    def test_no_full_sync_for_recent_sync(self) -> None:
        """Recent sync does not trigger full sync."""
        cache = _make_cache_mock()
        mgr = TombstoneManager(cache, retention_days=90)

        now = datetime(2026, 3, 8, tzinfo=timezone.utc)
        recent_sync = now - timedelta(days=10)

        result = mgr.needs_full_sync(
            last_sync=recent_sync,
            server_tombstone_cutoff=now - timedelta(days=90),
        )

        assert result is False

    def test_needs_full_sync_custom_cutoff(self) -> None:
        """Server-provided cutoff overrides default calculation."""
        cache = _make_cache_mock()
        mgr = TombstoneManager(cache, retention_days=90)

        now = datetime(2026, 3, 8, tzinfo=timezone.utc)
        # Sync 10 days ago, but server cutoff is only 5 days
        last_sync = now - timedelta(days=10)
        server_cutoff = now - timedelta(days=5)

        result = mgr.needs_full_sync(
            last_sync=last_sync,
            server_tombstone_cutoff=server_cutoff,
        )

        assert result is True


# ---------------------------------------------------------------------------
# Tests: full_sync_check
# ---------------------------------------------------------------------------


class TestFullSyncCheck:
    """Verify full sync reconciliation removes stale local bullets."""

    def test_removes_extra_local_ids(self) -> None:
        """Local bullets not on server are removed."""
        cache = _make_cache_mock(bullets={"b-1": "x", "b-2": "y", "b-3": "z"})
        mgr = TombstoneManager(cache, retention_days=90)

        server_ids = {"b-1", "b-3"}
        removed = mgr.full_sync_check(server_ids)

        assert removed == ["b-2"]
        cache.remove_bullets.assert_called_once_with(["b-2"])

    def test_keeps_server_matching_ids(self) -> None:
        """All local IDs present on server — nothing removed."""
        cache = _make_cache_mock(bullets={"b-1": "x", "b-2": "y"})
        mgr = TombstoneManager(cache, retention_days=90)

        server_ids = {"b-1", "b-2", "b-3"}
        removed = mgr.full_sync_check(server_ids)

        assert removed == []
        cache.remove_bullets.assert_not_called()

    def test_empty_local_cache(self) -> None:
        """Empty local cache — nothing to remove."""
        cache = _make_cache_mock(bullets={})
        mgr = TombstoneManager(cache, retention_days=90)

        removed = mgr.full_sync_check({"b-1", "b-2"})

        assert removed == []
        cache.remove_bullets.assert_not_called()

    def test_empty_server_ids_removes_all(self) -> None:
        """Empty server set removes everything from local cache."""
        cache = _make_cache_mock(bullets={"b-1": "x", "b-2": "y"})
        mgr = TombstoneManager(cache, retention_days=90)

        removed = mgr.full_sync_check(set())

        assert set(removed) == {"b-1", "b-2"}
        cache.remove_bullets.assert_called_once()
        assert set(cache.remove_bullets.call_args[0][0]) == {"b-1", "b-2"}


# ---------------------------------------------------------------------------
# Tests: state persistence
# ---------------------------------------------------------------------------


class TestStatePersistence:
    """Verify save/load round-trip for tombstone state."""

    def test_save_load_roundtrip(self) -> None:
        """State survives a save/load cycle."""
        cache = _make_cache_mock()
        mgr = TombstoneManager(cache, retention_days=90)

        ts = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
        entries = [_make_entry("b-1", status="tombstone", updated_at=ts)]
        mgr.process_tombstones(entries)

        state = mgr.save_state()

        # Load into a fresh manager
        mgr2 = TombstoneManager(_make_cache_mock(), retention_days=90)
        mgr2.load_state(state)

        assert mgr2.tombstone_count == 1
        assert mgr2._tombstones["b-1"] == ts

    def test_load_invalid_timestamps_skipped(self) -> None:
        """Invalid timestamp strings are silently skipped."""
        cache = _make_cache_mock()
        mgr = TombstoneManager(cache, retention_days=90)

        mgr.load_state({"tombstones": {"b-1": "not-a-date", "b-2": None}})

        assert mgr.tombstone_count == 0

    def test_load_empty_data(self) -> None:
        """Loading empty data is a no-op."""
        cache = _make_cache_mock()
        mgr = TombstoneManager(cache, retention_days=90)

        mgr.load_state({})

        assert mgr.tombstone_count == 0

    def test_save_empty_state(self) -> None:
        """Empty manager produces clean state dict."""
        cache = _make_cache_mock()
        mgr = TombstoneManager(cache, retention_days=90)

        state = mgr.save_state()

        assert state == {"tombstones": {}}


# ---------------------------------------------------------------------------
# Tests: capacity isolation
# ---------------------------------------------------------------------------


class TestCapacityIsolation:
    """Verify tombstone operations don't affect capacity calculation."""

    def test_tombstone_count_independent_of_cache(self) -> None:
        """Tombstone tracking is separate from cache bullet count."""
        cache = _make_cache_mock(bullets={"b-1": "x", "b-2": "y"})
        mgr = TombstoneManager(cache, retention_days=90)

        entries = [
            _make_entry("b-3", status="tombstone"),
            _make_entry("b-4", status="tombstone"),
        ]
        mgr.process_tombstones(entries)

        # Tombstone count reflects only tombstone records
        assert mgr.tombstone_count == 2
        # remove_bullets was called (affecting cache), not add
        cache.remove_bullets.assert_called_once_with(["b-3", "b-4"])
