"""TeamCacheStorage — local cache storage for Federation Mode team knowledge.

Stores TeamBullet objects in a local JSON cache under ~/.ace/team_cache/{team_id}/.
Provides vector search (cosine similarity) with keyword search fallback.
Thread-safe writes via threading.Lock.

Includes governance vote tracking, backlog alerts, and timeout rejection (STORY-069).

Also provides update notification logic: when a team bullet is updated in cache,
check_update_notifications() detects local bullets still holding older versions
and returns SupersedeNotification objects to inform the user.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from memx.team.config import TeamConfig
from memx.team.types import GovernanceTier, TeamBullet

logger = logging.getLogger(__name__)

# Vote score adjustments
_UPVOTE_DELTA: int = 5
_DOWNVOTE_DELTA: int = 10

# Auto-approve effective_score multiplier when entering pool
_AUTO_APPROVE_WEIGHT: float = 0.5

# Backlog thresholds
_BACKLOG_MAX_STAGING: int = 50
_BACKLOG_MAX_PENDING_DAYS: int = 7

# Timeout threshold — reject after N days without review
_TIMEOUT_REJECT_DAYS: int = 30

# Optional numpy — vector search disabled when unavailable
try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

# Minimum cosine similarity threshold for vector search results
_MIN_SIMILARITY: float = 0.3


def _sanitize_team_id(team_id: str) -> str:
    """Remove unsafe characters from team_id for use as directory name."""
    return re.sub(r"[^\w\-.]", "_", team_id)


class TeamCacheStorage:
    """Local cache storage for team knowledge bullets.

    Implements the ``StorageBackend`` protocol (search method) so it can be
    plugged into MultiPoolRetriever as a team pool.

    Storage layout:
        ~/.ace/team_cache/{team_id}/bullets.json

    Thread safety: all mutations go through ``_lock``.
    """

    def __init__(self, config: TeamConfig) -> None:
        self._config = config
        self._team_id: str = config.team_id or "default"
        self._max_bullets: int = config.cache_max_bullets

        safe_id = _sanitize_team_id(self._team_id)
        self._cache_dir: Path = Path.home() / ".ace" / "team_cache" / safe_id
        self._cache_file: Path = self._cache_dir / "bullets.json"

        self._bullets: dict[str, TeamBullet] = {}  # keyed by bullet ID
        self._lock = threading.Lock()
        self._last_sync: datetime | None = None

        # Vector index state
        self._vectors: Any | None = None  # np.ndarray (N, dim) or None
        self._vector_ids: list[str] = []  # bullet IDs aligned with _vectors rows
        self._embedder: Any | None = None

        # Track updated bullets for notification (content changes during sync)
        self._updated_bullet_ids: set[str] = set()

        # Load existing cache from disk
        self._load()

    # -- StorageBackend Protocol -----------------------------------------------

    def search(
        self, query: str, *, limit: int = 10, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Search cached team bullets by vector similarity or keyword fallback.

        Returns a list of result dicts compatible with StorageBackend Protocol.
        Empty cache returns empty list without error.
        """
        if not self._bullets:
            return []

        if self._vectors is not None and _HAS_NUMPY:
            results = self._vector_search(query, limit)
            if results:
                return results

        return self._keyword_search(query, limit)

    # -- Public mutation methods -----------------------------------------------

    def add_bullets(self, bullets: list[TeamBullet]) -> None:
        """Add or update bullets in the cache.

        After adding, enforces capacity limit and persists to disk.
        Tracks which bullets were updated (content changed) for notification.
        """
        if not bullets:
            return

        with self._lock:
            for bullet in bullets:
                bid = self._bullet_id(bullet)
                old = self._bullets.get(bid)
                if old is not None and old.content != bullet.content:
                    # Track that this bullet's content changed
                    self._updated_bullet_ids.add(bid)
                self._bullets[bid] = bullet

            self._enforce_capacity()
            self._rebuild_vectors()
            self._persist()

    def remove_bullets(self, ids: list[str]) -> None:
        """Remove bullets by ID."""
        if not ids:
            return

        with self._lock:
            for bid in ids:
                self._bullets.pop(bid, None)
            self._rebuild_vectors()
            self._persist()

    def get_bullet(self, id: str) -> TeamBullet | None:
        """Return a single bullet by ID, or None if not found."""
        return self._bullets.get(id)

    # -- Governance: voting (STORY-069) ----------------------------------------

    def vote_bullet(self, bullet_id: str, *, upvote: bool) -> TeamBullet | None:
        """Record an upvote or downvote on a bullet.

        Upvote adds +5, downvote adds +10 penalty (subtracted from score).
        Returns the updated bullet, or None if not found.
        """
        with self._lock:
            bullet = self._bullets.get(bullet_id)
            if bullet is None:
                return None

            if upvote:
                bullet.upvotes += _UPVOTE_DELTA
            else:
                bullet.downvotes += _DOWNVOTE_DELTA

            self._persist()
            logger.info(
                "Vote recorded on %s: %s (effective_score=%.1f)",
                bullet_id,
                "upvote" if upvote else "downvote",
                bullet.effective_score,
            )
            return bullet

    def apply_auto_approve_weight(self, bullet_id: str) -> TeamBullet | None:
        """Apply auto_approve weight multiplier to a bullet entering the pool.

        For auto_approve bullets, effective_score is halved by adjusting
        downvotes to simulate a 0.5x multiplier on the base score.
        Returns the updated bullet, or None if not found.
        """
        with self._lock:
            bullet = self._bullets.get(bullet_id)
            if bullet is None:
                return None

            if bullet.governance_tier == GovernanceTier.AUTO_APPROVE.value:
                # Halve effective score by adding artificial downvote penalty
                # effective = base + upvotes - downvotes, we want effective * 0.5
                current = bullet.effective_score
                target = current * _AUTO_APPROVE_WEIGHT
                # Add penalty as downvotes to reach target score
                penalty = current - target
                bullet.downvotes += int(penalty)
                logger.info(
                    "Applied auto_approve weight to %s: effective_score=%.1f",
                    bullet_id,
                    bullet.effective_score,
                )

            self._persist()
            return bullet

    # -- Governance: backlog monitoring (STORY-069) ----------------------------

    def check_backlog(self) -> dict[str, Any]:
        """Check staging backlog for threshold violations.

        Returns a dict with alert info:
          - staging_count: number of staging bullets
          - staging_overflow: True if > 50
          - oldest_pending_days: age of oldest pending staging bullet
          - oldest_pending_overflow: True if > 7 days
          - needs_attention: True if any threshold exceeded
        """
        now = datetime.now(timezone.utc)
        staging = [
            b for b in self._bullets.values() if b.status == "staging"
        ]
        staging_count = len(staging)

        oldest_days = 0.0
        for b in staging:
            if b.nominated_at:
                age = (now - b.nominated_at).total_seconds() / 86400
                oldest_days = max(oldest_days, age)

        staging_overflow = staging_count > _BACKLOG_MAX_STAGING
        oldest_overflow = oldest_days > _BACKLOG_MAX_PENDING_DAYS

        result = {
            "staging_count": staging_count,
            "staging_overflow": staging_overflow,
            "oldest_pending_days": round(oldest_days, 1),
            "oldest_pending_overflow": oldest_overflow,
            "needs_attention": staging_overflow or oldest_overflow,
        }

        if result["needs_attention"]:
            logger.warning(
                "Backlog alert: %d staging bullets, oldest=%.1f days",
                staging_count,
                oldest_days,
            )

        return result

    def reject_timed_out(self, *, timeout_days: int = _TIMEOUT_REJECT_DAYS) -> list[str]:
        """Reject staging bullets that exceed the timeout threshold.

        Bullets in 'staging' status with nominated_at older than timeout_days
        are moved to 'rejected' status.

        Returns list of rejected bullet IDs.
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=timeout_days)
        rejected_ids: list[str] = []

        with self._lock:
            for bid, bullet in self._bullets.items():
                if bullet.status != "staging":
                    continue
                if bullet.nominated_at and bullet.nominated_at < cutoff:
                    bullet.status = "rejected"
                    rejected_ids.append(bid)

            if rejected_ids:
                self._persist()
                logger.info(
                    "Rejected %d timed-out staging bullets (>%d days)",
                    len(rejected_ids),
                    timeout_days,
                )

        return rejected_ids

    # -- Update notification (STORY-070) ---------------------------------------

    def check_update_notifications(
        self,
        local_bullets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Check if any recently updated team bullets conflict with local versions.

        Compares recently updated cached team bullets against local bullets.
        When a local bullet's content overlaps significantly with the *old*
        content of a team bullet that was just updated, a notification is
        generated so the user knows the team version has changed.

        Args:
            local_bullets: Current local bullet pool (list of dicts with id, content).

        Returns:
            List of notification dicts with keys:
              - team_bullet_id, new_content, local_bullet_id, message
        """
        if not self._updated_bullet_ids:
            return []

        notifications: list[dict[str, Any]] = []
        for bid in list(self._updated_bullet_ids):
            team_bullet = self._bullets.get(bid)
            if team_bullet is None:
                continue

            team_content = team_bullet.content or ""
            if not team_content:
                continue

            for local in local_bullets:
                local_content = local.get("content", "")
                local_id = local.get("id", local.get("origin_id", ""))
                if not local_content or not local_id:
                    continue

                # Word-overlap similarity check
                sim = _word_similarity(local_content, team_content)
                if sim >= 0.5:
                    notifications.append({
                        "team_bullet_id": bid,
                        "new_content": team_content,
                        "local_bullet_id": local_id,
                        "message": (
                            f"Team bullet '{bid}' was updated. "
                            f"Your local bullet '{local_id}' may hold an older version. "
                            f"(similarity={sim:.2f})"
                        ),
                    })

        logger.debug("Generated %d update notification(s)", len(notifications))
        return notifications

    def clear_update_notifications(self) -> None:
        """Clear the set of updated bullet IDs after notifications are consumed."""
        self._updated_bullet_ids.clear()

    @property
    def pending_update_ids(self) -> set[str]:
        """IDs of bullets whose content was changed during the last sync."""
        return set(self._updated_bullet_ids)

    # -- Properties ------------------------------------------------------------

    @property
    def bullet_count(self) -> int:
        """Number of bullets currently in cache."""
        return len(self._bullets)

    @property
    def last_sync_time(self) -> datetime | None:
        """Timestamp of the last sync operation, or None if never synced."""
        return self._last_sync

    # -- Internal: persistence -------------------------------------------------

    def _persist(self) -> None:
        """Save cache to disk as JSON. Caller must hold _lock."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Failed to create cache dir %s: %s", self._cache_dir, exc)
            return

        self._last_sync = datetime.now(timezone.utc)

        data: dict[str, Any] = {
            "team_id": self._team_id,
            "last_sync": self._last_sync.isoformat(),
            "bullets": {},
        }
        for bid, bullet in self._bullets.items():
            data["bullets"][bid] = bullet.model_dump(mode="json")

        try:
            with self._cache_file.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except OSError as exc:
            logger.warning("Failed to write cache to %s: %s", self._cache_file, exc)

    def _load(self) -> None:
        """Load cache from disk. Rebuilds in-memory index after load."""
        if not self._cache_file.exists():
            return

        try:
            with self._cache_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Corrupt cache at %s, starting fresh: %s", self._cache_file, exc
            )
            self._bullets.clear()
            return

        if not isinstance(data, dict):
            logger.warning("Invalid cache format at %s, starting fresh", self._cache_file)
            self._bullets.clear()
            return

        raw_bullets = data.get("bullets", {})
        if not isinstance(raw_bullets, dict):
            logger.warning("Invalid bullets format in cache, starting fresh")
            self._bullets.clear()
            return

        loaded = 0
        for bid, bdata in raw_bullets.items():
            try:
                bullet = TeamBullet.model_validate(bdata)
                self._bullets[bid] = bullet
                loaded += 1
            except Exception as exc:
                logger.warning("Skipping corrupt bullet %s: %s", bid, exc)

        # Restore last_sync
        raw_sync = data.get("last_sync")
        if raw_sync:
            try:
                self._last_sync = datetime.fromisoformat(raw_sync)
            except (ValueError, TypeError):
                pass

        if loaded > 0:
            logger.info("Loaded %d bullets from cache %s", loaded, self._cache_file)
            self._rebuild_vectors()

    # -- Internal: capacity management -----------------------------------------

    def _enforce_capacity(self) -> None:
        """Evict lowest effective_score bullets when over capacity. Caller must hold _lock."""
        if len(self._bullets) <= self._max_bullets:
            return

        # Sort by effective_score descending, keep top N
        sorted_items = sorted(
            self._bullets.items(),
            key=lambda item: item[1].effective_score,
            reverse=True,
        )
        evict_count = len(sorted_items) - self._max_bullets
        self._bullets = dict(sorted_items[: self._max_bullets])
        logger.info("Evicted %d low-score bullets (capacity: %d)", evict_count, self._max_bullets)

    # -- Internal: search implementations --------------------------------------

    def _keyword_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Keyword-based search against cached bullets."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        scored: list[tuple[str, TeamBullet, float]] = []

        for bid, bullet in self._bullets.items():
            if not bullet.is_active:
                continue

            content_lower = bullet.content.lower() if bullet.content else ""
            score = 0.0

            # Exact substring match
            if query_lower in content_lower:
                score = 1.0
            else:
                # Word overlap scoring
                content_words = set(content_lower.split())
                overlap = query_words & content_words
                if overlap:
                    score = len(overlap) / len(query_words) * 0.7

                # Tag match bonus
                if any(query_lower in t.lower() for t in bullet.tags):
                    score = max(score, 0.5)

                # Context summary match
                if bullet.context_summary:
                    if query_lower in bullet.context_summary.lower():
                        score = max(score, 0.6)

            if score > 0:
                scored.append((bid, bullet, score))

        scored.sort(key=lambda x: (-x[2], -x[1].effective_score))
        return [self._to_result_dict(b, s) for _, b, s in scored[:limit]]

    def _vector_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Cosine similarity search against in-memory vector index."""
        if not _HAS_NUMPY or self._vectors is None or len(self._vector_ids) == 0:
            return []

        embedder = self._get_embedder()
        if embedder is None:
            return []

        try:
            query_vec = np.array(embedder.embed(query), dtype=np.float32)
        except Exception as exc:
            logger.warning("Failed to embed query, falling back to keyword: %s", exc)
            return []

        # Cosine similarity
        norms = np.linalg.norm(self._vectors, axis=1)
        query_norm = float(np.linalg.norm(query_vec))
        safe_norms = np.where(norms > 1e-9, norms, 1.0)
        safe_query_norm = max(query_norm, 1e-9)

        sims = np.dot(self._vectors, query_vec) / (safe_norms * safe_query_norm)
        top_indices = np.argsort(sims)[::-1][:limit]

        results: list[dict[str, Any]] = []
        for idx in top_indices:
            sim = float(sims[idx])
            if sim <= _MIN_SIMILARITY:
                break
            bid = self._vector_ids[idx]
            bullet = self._bullets.get(bid)
            if bullet is None or not bullet.is_active:
                continue
            results.append(self._to_result_dict(bullet, sim))

        return results

    # -- Internal: vector index ------------------------------------------------

    def _rebuild_vectors(self) -> None:
        """Rebuild in-memory vector index from current bullets."""
        if not _HAS_NUMPY:
            return

        active_bullets = [
            (bid, b) for bid, b in self._bullets.items() if b.is_active and b.content
        ]
        if not active_bullets:
            self._vectors = None
            self._vector_ids = []
            return

        embedder = self._get_embedder()
        if embedder is None:
            self._vectors = None
            self._vector_ids = []
            return

        ids = [bid for bid, _ in active_bullets]
        texts = [b.content for _, b in active_bullets]

        try:
            raw_vectors = embedder.embed_batch(texts)
            self._vectors = np.array(raw_vectors, dtype=np.float32)
            self._vector_ids = ids
        except Exception as exc:
            logger.warning("Failed to build vector index: %s", exc)
            self._vectors = None
            self._vector_ids = []

    def _get_embedder(self) -> Any | None:
        """Try to get an ONNXEmbedder instance. Returns None on failure."""
        if self._embedder is not None:
            return self._embedder
        try:
            from memx.core.embeddings.onnx import ONNXEmbedder

            self._embedder = ONNXEmbedder()
            return self._embedder
        except (ImportError, Exception) as exc:
            logger.debug("ONNX embedder not available: %s", exc)
            return None

    # -- Internal: helpers -----------------------------------------------------

    @staticmethod
    def _bullet_id(bullet: TeamBullet) -> str:
        """Derive a stable ID for a bullet.

        Uses origin_id if available, otherwise generates a UUID.
        """
        if bullet.origin_id:
            return bullet.origin_id
        return str(uuid.uuid4())

    @staticmethod
    def _to_result_dict(bullet: TeamBullet, score: float) -> dict[str, Any]:
        """Convert a TeamBullet to the standard search result dict format."""
        return {
            "content": bullet.content or "",
            "section": str(bullet.section.value) if hasattr(bullet.section, "value") else str(bullet.section),
            "knowledge_type": str(bullet.knowledge_type.value) if hasattr(bullet.knowledge_type, "value") else str(bullet.knowledge_type),
            "instructivity_score": bullet.instructivity_score,
            "tags": list(bullet.tags),
            "enforcement": bullet.enforcement,
            "score": score,
            "source": "team_cache",
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _word_similarity(a: str, b: str) -> float:
    """Jaccard word-level similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)
