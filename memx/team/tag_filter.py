"""TagSubscriptionManager — tag-based subscription filtering for team sync.

Filters team bullets by subscribed tags during sync and retrieval.
Detects tag configuration changes to trigger cache cleanup and re-sync.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from memx.team.config import TeamConfig

logger = logging.getLogger(__name__)


def _sanitize_team_id(team_id: str) -> str:
    """Remove unsafe characters from team_id for use as directory name."""
    return re.sub(r"[^\w\-.]", "_", team_id)


class TagSubscriptionManager:
    """Manages tag-based subscription filtering for team sync.

    When ``subscribed_tags`` is non-empty, only bullets whose tags overlap
    with the subscription list are kept.  An empty subscription means
    "pull everything" (up to ``cache_max_bullets``).
    """

    def __init__(
        self,
        config: TeamConfig,
        state_dir: Path | None = None,
    ) -> None:
        self._subscribed_tags: list[str] = [
            t.lower().strip() for t in config.subscribed_tags
        ]
        self._team_id: str = config.team_id or "default"
        self._state_dir: Path = state_dir or (
            Path.home() / ".ace" / "team_cache" / _sanitize_team_id(self._team_id)
        )

    # -- public API ---------------------------------------------------------

    def get_sync_tags(self) -> list[str] | None:
        """Return tags to pass to pull_index, or None for all."""
        return self._subscribed_tags if self._subscribed_tags else None

    def has_tags_changed(self, previous_tags: list[str] | None) -> bool:
        """Check if subscribed tags have changed since last sync."""
        prev = set(t.lower().strip() for t in (previous_tags or []))
        current = set(self._subscribed_tags)
        return prev != current

    def filter_bullets(
        self, bullets: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter bullets by subscribed tags (client-side filtering).

        If no tags are subscribed, all bullets pass through.
        """
        if not self._subscribed_tags:
            return bullets
        return [b for b in bullets if self._matches_tags(b)]

    def get_stale_bullets(
        self,
        cache_bullets: dict[str, Any],
        new_tags: list[str],
    ) -> list[str]:
        """Find cached bullets that no longer match the new tag subscription.

        Returns list of bullet IDs to remove from cache.
        """
        new_tags_set = set(t.lower() for t in new_tags) if new_tags else set()
        if not new_tags_set:
            return []  # no tags = keep everything

        stale: list[str] = []
        for bid, bullet in cache_bullets.items():
            # Support both dict and object with .tags attribute
            if isinstance(bullet, dict):
                raw_tags = bullet.get("tags", [])
            else:
                raw_tags = getattr(bullet, "tags", [])
            bullet_tags = set(t.lower() for t in raw_tags)
            if not bullet_tags & new_tags_set:
                stale.append(bid)
        return stale

    # -- internals ----------------------------------------------------------

    def _matches_tags(self, bullet: dict[str, Any]) -> bool:
        """Check if bullet's tags overlap with subscribed tags."""
        bullet_tags = set(t.lower() for t in bullet.get("tags", []))
        return bool(bullet_tags & set(self._subscribed_tags))
