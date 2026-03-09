"""TombstoneManager — manages tombstone records and full sync checks.

Handles server-side deletions propagated via tombstone status entries,
retention-based cleanup, and full sync reconciliation when incremental
sync is no longer possible.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from memx.team.cache_storage import TeamCacheStorage
from memx.team.sync_client import BulletIndexEntry

logger = logging.getLogger(__name__)


class TombstoneManager:
    """Manages tombstone records and full sync checks.

    Tombstone records track server-side deletions so that local caches
    can remove bullets that no longer exist on the server.  Records are
    retained for ``retention_days`` (default 90) to allow incremental
    sync; after that window a full sync check is required.
    """

    def __init__(self, cache: TeamCacheStorage, retention_days: int = 90) -> None:
        self._cache = cache
        self._retention_days = retention_days
        self._tombstones: dict[str, datetime] = {}  # id -> deleted_at

    # -- Tombstone processing ------------------------------------------------

    def process_tombstones(self, entries: list[BulletIndexEntry]) -> list[str]:
        """Process tombstone entries from sync index.

        Only entries with ``status == "tombstone"`` are processed; all
        others are silently ignored.

        Returns:
            List of removed bullet IDs.
        """
        removed: list[str] = []
        for entry in entries:
            if entry.status == "tombstone":
                self._tombstones[entry.id] = entry.updated_at
                removed.append(entry.id)

        if removed:
            self._cache.remove_bullets(removed)
            logger.info("Processed %d tombstone(s): %s", len(removed), removed)

        return removed

    # -- Cleanup -------------------------------------------------------------

    def cleanup_expired(self, now: datetime | None = None) -> int:
        """Remove tombstone records older than the retention period.

        This only affects the in-memory tombstone tracking; it does NOT
        modify the cache's capacity calculation.

        Returns:
            Count of cleaned-up tombstone records.
        """
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self._retention_days)
        expired = [tid for tid, ts in self._tombstones.items() if ts < cutoff]

        for tid in expired:
            del self._tombstones[tid]

        if expired:
            logger.info(
                "Cleaned up %d expired tombstone(s) (cutoff=%s)",
                len(expired),
                cutoff.isoformat(),
            )

        return len(expired)

    # -- Full sync decision --------------------------------------------------

    def needs_full_sync(
        self,
        last_sync: datetime | None,
        server_tombstone_cutoff: datetime | None = None,
    ) -> bool:
        """Check if a full sync is needed because incremental is unreliable.

        A full sync is required when ``last_sync`` is ``None`` (first sync)
        or when it falls before the tombstone retention cutoff, meaning
        some deletions may have been purged and missed.
        """
        if last_sync is None:
            return True
        cutoff = server_tombstone_cutoff or (
            datetime.now(timezone.utc) - timedelta(days=self._retention_days)
        )
        return last_sync < cutoff

    # -- Full sync check -----------------------------------------------------

    def full_sync_check(self, server_ids: set[str]) -> list[str]:
        """Compare local cache with server IDs and remove extras.

        Any bullet present locally but absent from ``server_ids`` is
        considered stale and removed from the cache.

        Returns:
            List of removed (stale) bullet IDs.
        """
        local_ids = set(self._cache._bullets.keys())
        stale_ids = list(local_ids - server_ids)

        if stale_ids:
            self._cache.remove_bullets(stale_ids)
            logger.info(
                "Full sync check removed %d stale bullet(s): %s",
                len(stale_ids),
                stale_ids,
            )

        return stale_ids

    # -- Properties ----------------------------------------------------------

    @property
    def tombstone_count(self) -> int:
        """Number of tombstone records currently tracked."""
        return len(self._tombstones)

    # -- State persistence ---------------------------------------------------

    def save_state(self) -> dict[str, Any]:
        """Serialize tombstone state for persistence."""
        return {
            "tombstones": {
                tid: ts.isoformat() for tid, ts in self._tombstones.items()
            }
        }

    def load_state(self, data: dict[str, Any]) -> None:
        """Restore tombstone state from persisted data."""
        raw = data.get("tombstones", {})
        for tid, ts_str in raw.items():
            try:
                self._tombstones[tid] = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                logger.warning("Skipping invalid tombstone timestamp for %s", tid)
