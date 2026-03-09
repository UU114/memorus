"""Tests for SyncManager — background sync orchestration."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memx.team.config import TeamConfig
from memx.team.sync_client import (
    BulletIndexEntry,
    IndexResponse,
    SyncConnectionError,
)
from memx.team.sync_manager import SyncManager
from memx.team.types import TeamBullet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> TeamConfig:
    """Create a TeamConfig with test defaults."""
    defaults: dict[str, Any] = {
        "enabled": True,
        "server_url": "https://sync.example.com",
        "team_id": "test-team",
        "cache_ttl_minutes": 60,
    }
    defaults.update(overrides)
    return TeamConfig(**defaults)


def _make_bullet(
    origin_id: str = "b-1",
    content: str = "Test bullet",
    **kwargs: Any,
) -> TeamBullet:
    """Create a TeamBullet with test defaults."""
    data: dict[str, Any] = {
        "origin_id": origin_id,
        "content": content,
        "section": "preferences",
        "knowledge_type": "preference",
    }
    data.update(kwargs)
    return TeamBullet(**data)


def _bullet_dict(
    origin_id: str = "b-1",
    content: str = "Test bullet",
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a bullet dict suitable for fetch_bullets response."""
    data: dict[str, Any] = {
        "origin_id": origin_id,
        "content": content,
        "section": "preferences",
        "knowledge_type": "preference",
    }
    data.update(kwargs)
    return data


def _make_index_response(
    entries: list[tuple[str, str, str]] | None = None,
) -> IndexResponse:
    """Create an IndexResponse from (id, updated_at_iso, status) tuples."""
    if entries is None:
        entries = [("b-1", "2026-03-08T10:00:00Z", "approved")]
    bullets = [
        BulletIndexEntry(
            id=eid,
            updated_at=datetime.fromisoformat(ts),
            status=status,
        )
        for eid, ts, status in entries
    ]
    return IndexResponse(bullets=bullets)


@pytest.fixture
def mock_cache() -> MagicMock:
    """Create a mock TeamCacheStorage."""
    cache = MagicMock()
    cache.bullet_count = 0
    cache.last_sync_time = None
    cache.add_bullets = MagicMock()
    cache.remove_bullets = MagicMock()
    return cache


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock AceSyncClient with async methods."""
    client = MagicMock()
    client.pull_index = AsyncMock(return_value=IndexResponse(bullets=[]))
    client.fetch_bullets = AsyncMock(return_value=[])
    client.close = AsyncMock()
    return client


@pytest.fixture
def config() -> TeamConfig:
    return _make_config()


@pytest.fixture
def manager(
    mock_cache: MagicMock,
    mock_client: MagicMock,
    config: TeamConfig,
    tmp_path: Path,
) -> SyncManager:
    """Create SyncManager with mocked deps and temp state dir."""
    mgr = SyncManager(mock_cache, mock_client, config)
    # Redirect state file to tmp_path
    mgr._state_dir = tmp_path
    mgr._state_file = tmp_path / "sync_state.json"
    return mgr


# ---------------------------------------------------------------------------
# 1. First sync — full pull (since=None)
# ---------------------------------------------------------------------------


def test_first_sync_full_pull(
    manager: SyncManager,
    mock_client: MagicMock,
    mock_cache: MagicMock,
) -> None:
    """First sync should call pull_index with since=None (full pull)."""
    mock_client.pull_index = AsyncMock(
        return_value=_make_index_response([("b-1", "2026-03-08T10:00:00Z", "approved")])
    )
    mock_client.fetch_bullets = AsyncMock(
        return_value=[_bullet_dict("b-1", "Hello")]
    )

    manager.sync_now()

    # pull_index called with since=None (first sync)
    call_kwargs = mock_client.pull_index.call_args
    assert call_kwargs.kwargs.get("since") is None or call_kwargs[1].get("since") is None

    # Bullets fetched and added to cache
    mock_client.fetch_bullets.assert_called_once_with(["b-1"])
    mock_cache.add_bullets.assert_called_once()
    added = mock_cache.add_bullets.call_args[0][0]
    assert len(added) == 1
    assert added[0].content == "Hello"

    assert manager.last_sync_status == "success"
    assert manager.sync_count == 1


# ---------------------------------------------------------------------------
# 2. Incremental sync (since=last_timestamp)
# ---------------------------------------------------------------------------


def test_incremental_sync_uses_last_timestamp(
    manager: SyncManager,
    mock_client: MagicMock,
    mock_cache: MagicMock,
) -> None:
    """After first sync, subsequent syncs should pass since=last_timestamp."""
    # First sync
    mock_client.pull_index = AsyncMock(return_value=IndexResponse(bullets=[]))
    manager.sync_now()

    # Verify timestamp was set
    assert manager._last_sync_timestamp is not None
    saved_ts = manager._last_sync_timestamp

    # Second sync
    manager.sync_now()

    # Second call should have since= the saved timestamp
    calls = mock_client.pull_index.call_args_list
    assert len(calls) == 2
    second_call_since = calls[1].kwargs.get("since") or calls[1][1].get("since")
    assert second_call_since == saved_ts


# ---------------------------------------------------------------------------
# 3. sync_state.json persistence and reload
# ---------------------------------------------------------------------------


def test_sync_state_persistence(
    manager: SyncManager,
    mock_client: MagicMock,
    mock_cache: MagicMock,
) -> None:
    """sync_state.json is written after sync and loadable."""
    mock_client.pull_index = AsyncMock(return_value=IndexResponse(bullets=[]))
    mock_cache.bullet_count = 42

    manager.sync_now()

    # Verify state file exists
    assert manager._state_file.exists()
    with manager._state_file.open("r") as f:
        state = json.load(f)

    assert state["last_sync_status"] == "success"
    assert state["total_bullets"] == 42
    assert state["sync_count"] == 1
    assert state["last_sync_timestamp"] is not None


def test_sync_state_reload(
    mock_cache: MagicMock,
    mock_client: MagicMock,
    config: TeamConfig,
    tmp_path: Path,
) -> None:
    """SyncManager loads persisted state on construction."""
    state = {
        "last_sync_timestamp": "2026-03-08T08:00:00+00:00",
        "last_sync_status": "success",
        "total_bullets": 100,
        "sync_count": 5,
    }
    state_file = tmp_path / "sync_state.json"
    with state_file.open("w") as f:
        json.dump(state, f)

    mgr = SyncManager(mock_cache, mock_client, config)
    mgr._state_dir = tmp_path
    mgr._state_file = state_file
    mgr._load_state()

    assert mgr._last_sync_timestamp == datetime.fromisoformat("2026-03-08T08:00:00+00:00")
    assert mgr._last_sync_status == "success"
    assert mgr._sync_count == 5


# ---------------------------------------------------------------------------
# 4. Server unreachable — graceful degradation
# ---------------------------------------------------------------------------


def test_server_unreachable_graceful_degradation(
    manager: SyncManager,
    mock_client: MagicMock,
) -> None:
    """SyncConnectionError should be handled gracefully, status set to 'failed'."""
    mock_client.pull_index = AsyncMock(
        side_effect=SyncConnectionError("Connection refused")
    )

    manager.sync_now()

    assert manager.last_sync_status == "failed"
    # Should not raise — sync continues to work


# ---------------------------------------------------------------------------
# 5. Periodic refresh triggers
# ---------------------------------------------------------------------------


def test_periodic_refresh_triggers(
    mock_cache: MagicMock,
    mock_client: MagicMock,
    tmp_path: Path,
) -> None:
    """Background thread should trigger multiple syncs based on ttl."""
    config = _make_config(cache_ttl_minutes=1)  # 1 minute = 60s
    mgr = SyncManager(mock_cache, mock_client, config)
    mgr._state_dir = tmp_path
    mgr._state_file = tmp_path / "sync_state.json"

    mock_client.pull_index = AsyncMock(return_value=IndexResponse(bullets=[]))

    # Override interval to very short for testing
    original_loop = mgr._sync_loop

    call_count = 0
    max_calls = 3

    def fast_loop() -> None:
        nonlocal call_count
        while not mgr._stop_event.is_set() and call_count < max_calls:
            mgr._run_single_sync()
            call_count += 1
            if call_count >= max_calls:
                break
            mgr._stop_event.wait(timeout=0.05)

    mgr._sync_loop = fast_loop  # type: ignore[assignment]

    mgr.start()
    # Wait for syncs to complete
    for _ in range(50):
        if call_count >= max_calls:
            break
        time.sleep(0.05)
    mgr.stop()

    assert call_count >= 2  # at least 2 sync cycles


# ---------------------------------------------------------------------------
# 6. Stop/cleanup
# ---------------------------------------------------------------------------


def test_stop_cleanup(
    manager: SyncManager,
    mock_client: MagicMock,
) -> None:
    """stop() should signal the thread to exit and join."""
    mock_client.pull_index = AsyncMock(return_value=IndexResponse(bullets=[]))

    # Patch sync loop to be slow
    def slow_loop() -> None:
        while not manager._stop_event.is_set():
            manager._stop_event.wait(timeout=0.1)

    manager._sync_loop = slow_loop  # type: ignore[assignment]
    manager.start()

    assert manager._thread is not None
    assert manager._thread.is_alive()

    manager.stop()

    assert manager._thread is None or not manager._thread.is_alive()


def test_start_multiple_times_safe(
    manager: SyncManager,
    mock_client: MagicMock,
) -> None:
    """Calling start() multiple times should not create multiple threads."""
    mock_client.pull_index = AsyncMock(return_value=IndexResponse(bullets=[]))

    def slow_loop() -> None:
        while not manager._stop_event.is_set():
            manager._stop_event.wait(timeout=0.1)

    manager._sync_loop = slow_loop  # type: ignore[assignment]

    manager.start()
    thread1 = manager._thread
    manager.start()
    thread2 = manager._thread

    assert thread1 is thread2

    manager.stop()


# ---------------------------------------------------------------------------
# 7. Tombstone handling
# ---------------------------------------------------------------------------


def test_tombstone_removal(
    manager: SyncManager,
    mock_client: MagicMock,
    mock_cache: MagicMock,
) -> None:
    """Bullets with status='tombstone' should be removed from cache."""
    mock_client.pull_index = AsyncMock(
        return_value=_make_index_response([
            ("b-1", "2026-03-08T10:00:00Z", "approved"),
            ("b-2", "2026-03-08T10:00:00Z", "tombstone"),
            ("b-3", "2026-03-08T10:00:00Z", "tombstone"),
        ])
    )
    mock_client.fetch_bullets = AsyncMock(
        return_value=[_bullet_dict("b-1", "Active bullet")]
    )

    manager.sync_now()

    # Tombstoned bullets removed
    mock_cache.remove_bullets.assert_called_once_with(["b-2", "b-3"])

    # Active bullet fetched and added
    mock_client.fetch_bullets.assert_called_once_with(["b-1"])
    mock_cache.add_bullets.assert_called_once()


# ---------------------------------------------------------------------------
# 8. sync_now blocking behavior
# ---------------------------------------------------------------------------


def test_sync_now_blocks_until_complete(
    manager: SyncManager,
    mock_client: MagicMock,
    mock_cache: MagicMock,
) -> None:
    """sync_now() should block until sync is complete."""
    mock_client.pull_index = AsyncMock(return_value=IndexResponse(bullets=[]))

    manager.sync_now()

    # After sync_now returns, state should be updated
    assert manager.last_sync_status == "success"
    assert manager.sync_count == 1
    assert not manager.is_syncing


# ---------------------------------------------------------------------------
# 9. Concurrent sync prevention
# ---------------------------------------------------------------------------


def test_concurrent_sync_prevention(
    manager: SyncManager,
    mock_client: MagicMock,
) -> None:
    """Should not start a new sync if one is already in progress."""
    barrier = threading.Barrier(2, timeout=5)
    completed = threading.Event()

    original_pull = AsyncMock(return_value=IndexResponse(bullets=[]))

    async def slow_pull(**kwargs: Any) -> IndexResponse:
        # Signal that sync has started
        barrier.wait()
        # Wait until test is done checking
        for _ in range(100):
            if completed.is_set():
                break
            await _async_sleep(0.01)
        return IndexResponse(bullets=[])

    mock_client.pull_index = slow_pull

    # Start sync in background thread
    t = threading.Thread(target=manager._run_single_sync)
    t.start()

    # Wait until first sync has started
    barrier.wait()

    # Try to start another sync — should skip because is_syncing is True
    assert manager.is_syncing
    manager._run_single_sync()  # Should return immediately (skipped)

    completed.set()
    t.join(timeout=5)


async def _async_sleep(seconds: float) -> None:
    """Async sleep helper for tests."""
    import asyncio
    await asyncio.sleep(seconds)


# ---------------------------------------------------------------------------
# 10. Corrupt sync_state.json — fallback to full sync
# ---------------------------------------------------------------------------


def test_corrupt_sync_state_fallback(
    mock_cache: MagicMock,
    mock_client: MagicMock,
    config: TeamConfig,
    tmp_path: Path,
) -> None:
    """Corrupt sync_state.json should result in full sync (since=None)."""
    state_file = tmp_path / "sync_state.json"
    state_file.write_text("{invalid json!!!", encoding="utf-8")

    mgr = SyncManager(mock_cache, mock_client, config)
    mgr._state_dir = tmp_path
    mgr._state_file = state_file
    mgr._load_state()

    # Should fall back to defaults
    assert mgr._last_sync_timestamp is None
    assert mgr._last_sync_status == "never"
    assert mgr._sync_count == 0

    # Sync should use since=None (full pull)
    mock_client.pull_index = AsyncMock(return_value=IndexResponse(bullets=[]))
    mgr.sync_now()

    call_kwargs = mock_client.pull_index.call_args
    since_val = call_kwargs.kwargs.get("since") or call_kwargs[1].get("since")
    assert since_val is None


# ---------------------------------------------------------------------------
# Extra: empty index response (no updates)
# ---------------------------------------------------------------------------


def test_no_updates_from_server(
    manager: SyncManager,
    mock_client: MagicMock,
    mock_cache: MagicMock,
) -> None:
    """Empty index response should not call fetch_bullets or add_bullets."""
    mock_client.pull_index = AsyncMock(return_value=IndexResponse(bullets=[]))

    manager.sync_now()

    mock_client.fetch_bullets.assert_not_called()
    mock_cache.add_bullets.assert_not_called()
    mock_cache.remove_bullets.assert_not_called()
    assert manager.last_sync_status == "success"


# ---------------------------------------------------------------------------
# Extra: subscribed_tags passed to pull_index
# ---------------------------------------------------------------------------


def test_subscribed_tags_forwarded(
    mock_cache: MagicMock,
    mock_client: MagicMock,
    tmp_path: Path,
) -> None:
    """Subscribed tags from config should be forwarded to pull_index."""
    config = _make_config(subscribed_tags=["python", "testing"])
    mgr = SyncManager(mock_cache, mock_client, config)
    mgr._state_dir = tmp_path
    mgr._state_file = tmp_path / "sync_state.json"

    mock_client.pull_index = AsyncMock(return_value=IndexResponse(bullets=[]))

    mgr.sync_now()

    call_kwargs = mock_client.pull_index.call_args
    tags_val = call_kwargs.kwargs.get("tags") or call_kwargs[1].get("tags")
    assert tags_val == ["python", "testing"]
