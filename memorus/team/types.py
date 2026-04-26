"""TeamBullet data model — extends BulletMetadata with team governance fields."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field

from memorus.core.types import BulletMetadata


class GovernanceTier(str, Enum):
    """Three-tier governance classification for team bullets."""

    AUTO_APPROVE = "auto_approve"
    P2P_REVIEW = "p2p_review"
    CURATOR_REQUIRED = "curator_required"


class TeamBullet(BulletMetadata):
    """Extended Bullet with team governance metadata.

    Adds author tracking, enforcement levels, community voting,
    lifecycle status, and provenance fields on top of core BulletMetadata.

    Compatibility:
      - v1 -> v2: missing team fields are filled with defaults.
      - v2 -> v1: extra="allow" preserves unknown fields during round-trip.
    """

    model_config = {"extra": "allow"}  # forward compatibility (v2 -> v1)

    # Team-specific fields
    author_id: str = ""
    enforcement: str = "suggestion"  # suggestion | recommended | mandatory
    upvotes: int = 0
    downvotes: int = 0
    status: str = "approved"  # staging | approved | deprecated | tombstone
    deleted_at: datetime | None = None
    origin_id: str | None = None
    context_summary: str | None = None

    # Governance fields (STORY-069)
    governance_tier: str = GovernanceTier.P2P_REVIEW.value
    nominated_at: datetime | None = None

    # Verified-status mirror (STORY-R104)
    # Display-only metadata propagated from the local pool when packaging a
    # bullet for sync. Governance never reads these fields. ``None`` means the
    # local pool never reported a verification status (treated as "-" in CLI).
    last_verified_status: str | None = None
    last_verified_at: datetime | None = None

    def __init__(self, **data: Any) -> None:
        """Auto-set schema_version to 2 for TeamBullet instances."""
        data.setdefault("schema_version", 2)
        super().__init__(**data)

    @property
    def effective_score(self) -> float:
        """Score adjusted by community votes, bounded to [0, 100]."""
        base = self.instructivity_score + self.upvotes - self.downvotes
        return max(0.0, min(100.0, base))

    @property
    def is_active(self) -> bool:
        """Whether the bullet is in an active lifecycle state."""
        return self.status in ("staging", "approved")
