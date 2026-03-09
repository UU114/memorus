"""SyncManager — background sync orchestration for TeamCacheStorage.

Handles session-start async pull, periodic refresh, and sync state persistence.
Sync runs in a daemon thread so it never blocks user operations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memx.team.cache_storage import TeamCacheStorage
from memx.team.config import TeamConfig
from memx.team.sync_client import AceSyncClient, SyncConnectionError
from memx.team.taxonomy import TaxonomyResolver
from memx.team.types import TeamBullet

logger = logging.getLogger(__name__)


def _sanitize_team_id(team_id: str) -> str:
    """Remove unsafe characters from team_id for use as directory name."""
    return re.sub(r"[^\w\-.]", "_", team_id)


class SyncManager:
    """Background sync orchestration for TeamCacheStorage.

    Manages incremental pulls from ACE Sync Server, periodic refresh,
    and sync state persistence to ``sync_state.json``.

    Thread safety:
      - Sync writes go through TeamCacheStorage (already lock-protected).
      - Internal state guarded by ``_state_lock``.
    """

    def __init__(
        self,
        cache: TeamCacheStorage,
        client: AceSyncClient,
        config: TeamConfig,
    ) -> None:
        self._cache = cache
        self._client = client
        self._config = config

        self._team_id: str = config.team_id or "default"
        safe_id = _sanitize_team_id(self._team_id)
        self._state_dir: Path = Path.home() / ".ace" / "team_cache" / safe_id
        self._state_file: Path = self._state_dir / "sync_state.json"

        # Sync state
        self._last_sync_timestamp: datetime | None = None
        self._last_sync_status: str = "never"
        self._sync_count: int = 0
        self._is_syncing: bool = False

        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Taxonomy resolver for tag normalization
        self._taxonomy_resolver = TaxonomyResolver(team_id=self._team_id)

        # Load persisted sync state
        self._load_state()

    # -- Public API -----------------------------------------------------------

    def start(self) -> None:
        """Start background sync. Non-blocking. Safe to call multiple times."""
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                logger.debug("Sync thread already running, ignoring start()")
                return

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._sync_loop,
                name="memx-team-sync",
                daemon=True,
            )
            self._thread.start()
            logger.info("Background sync started (ttl=%dm)", self._config.cache_ttl_minutes)

    def stop(self) -> None:
        """Stop background sync and cleanup."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10)
        with self._state_lock:
            self._thread = None
        logger.info("Background sync stopped")

    def sync_now(self) -> None:
        """Trigger immediate sync (blocking). For CLI 'ace team sync' use."""
        self._run_single_sync()

    @property
    def is_syncing(self) -> bool:
        """Whether a sync operation is currently in progress."""
        with self._state_lock:
            return self._is_syncing

    @property
    def last_sync_status(self) -> str:
        """Last sync result: 'success', 'failed', or 'never'."""
        with self._state_lock:
            return self._last_sync_status

    @property
    def sync_count(self) -> int:
        """Total number of completed sync cycles."""
        with self._state_lock:
            return self._sync_count

    @property
    def taxonomy_resolver(self) -> TaxonomyResolver:
        """Return the taxonomy resolver for tag normalization."""
        return self._taxonomy_resolver

    # -- Background loop ------------------------------------------------------

    def _sync_loop(self) -> None:
        """Background sync loop: sync once, sleep, repeat."""
        interval_seconds = self._config.cache_ttl_minutes * 60

        while not self._stop_event.is_set():
            self._run_single_sync()
            # Wait for interval or until stop is signaled
            self._stop_event.wait(timeout=interval_seconds)

    def _run_single_sync(self) -> None:
        """Execute one sync cycle (pull index, fetch new bullets, remove tombstones)."""
        with self._state_lock:
            if self._is_syncing:
                logger.debug("Sync already in progress, skipping")
                return
            self._is_syncing = True

        try:
            self._do_sync()
            with self._state_lock:
                self._last_sync_status = "success"
                self._sync_count += 1
        except SyncConnectionError as exc:
            logger.warning(
                "Server unreachable during sync, using cached snapshot: %s", exc
            )
            with self._state_lock:
                self._last_sync_status = "failed"
        except Exception as exc:
            logger.error("Sync failed unexpectedly: %s", exc)
            with self._state_lock:
                self._last_sync_status = "failed"
        finally:
            with self._state_lock:
                self._is_syncing = False
            self._save_state()

    def _do_sync(self) -> None:
        """Core sync logic — runs async operations in a new event loop."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._async_sync())
        finally:
            loop.close()

    async def _async_sync(self) -> None:
        """Async sync: pull index, fetch new/updated bullets, remove tombstones."""
        since = self._last_sync_timestamp
        tags = self._config.subscribed_tags or None

        logger.info(
            "Starting %s sync (since=%s)",
            "incremental" if since else "full",
            since.isoformat() if since else "None",
        )

        # Pull index
        index_response = await self._client.pull_index(since=since, tags=tags)
        entries = index_response.bullets

        if not entries:
            logger.info("No updates from server")
            self._last_sync_timestamp = datetime.now(timezone.utc)
            return

        # Separate active vs tombstoned bullets
        tombstone_ids: list[str] = []
        fetch_ids: list[str] = []

        for entry in entries:
            if entry.status == "tombstone":
                tombstone_ids.append(entry.id)
            else:
                fetch_ids.append(entry.id)

        # Remove tombstoned bullets
        if tombstone_ids:
            self._cache.remove_bullets(tombstone_ids)
            logger.info("Removed %d tombstoned bullets", len(tombstone_ids))

        # Fetch and store new/updated bullets
        if fetch_ids:
            bullet_dicts = await self._client.fetch_bullets(fetch_ids)
            bullets: list[TeamBullet] = []
            for data in bullet_dicts:
                try:
                    bullets.append(TeamBullet.model_validate(data))
                except Exception as exc:
                    logger.warning("Skipping invalid bullet data: %s", exc)

            if bullets:
                self._cache.add_bullets(bullets)
                logger.info("Added/updated %d bullets", len(bullets))

        # Pull taxonomy (best-effort, non-blocking on failure)
        try:
            taxonomy_response = await self._client.pull_taxonomy()
            self._taxonomy_resolver.update_from_server(taxonomy_response)
            logger.info("Taxonomy updated from server")
        except Exception as exc:
            logger.warning("Failed to pull taxonomy, using cached/preset: %s", exc)

        self._last_sync_timestamp = datetime.now(timezone.utc)
        logger.info("Sync complete: %d fetched, %d tombstoned", len(fetch_ids), len(tombstone_ids))

    # -- State persistence ----------------------------------------------------

    def _load_state(self) -> None:
        """Load sync state from sync_state.json."""
        if not self._state_file.exists():
            return

        try:
            with self._state_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Corrupt sync_state.json at %s, starting fresh: %s",
                self._state_file,
                exc,
            )
            return

        if not isinstance(data, dict):
            logger.warning("Invalid sync_state.json format, starting fresh")
            return

        # Restore last_sync_timestamp
        raw_ts = data.get("last_sync_timestamp")
        if raw_ts:
            try:
                self._last_sync_timestamp = datetime.fromisoformat(raw_ts)
            except (ValueError, TypeError):
                logger.warning("Invalid last_sync_timestamp in sync_state.json, ignoring")

        self._last_sync_status = data.get("last_sync_status", "never")
        self._sync_count = data.get("sync_count", 0)

        logger.debug(
            "Loaded sync state: last_sync=%s, status=%s, count=%d",
            self._last_sync_timestamp,
            self._last_sync_status,
            self._sync_count,
        )

    def _save_state(self) -> None:
        """Persist sync state to sync_state.json."""
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Failed to create state dir %s: %s", self._state_dir, exc)
            return

        state: dict[str, Any] = {
            "last_sync_timestamp": (
                self._last_sync_timestamp.isoformat()
                if self._last_sync_timestamp
                else None
            ),
            "last_sync_status": self._last_sync_status,
            "total_bullets": self._cache.bullet_count,
            "sync_count": self._sync_count,
        }

        try:
            with self._state_file.open("w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.warning("Failed to write sync state to %s: %s", self._state_file, exc)
