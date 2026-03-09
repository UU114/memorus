"""MultiPoolRetriever + Shadow Merge — merges Local and Team search results.

Combines one Local pool and zero-or-more Team pools (GitFallbackStorage,
federation cache, etc.) into a single ranked result list using the Shadow
Merge algorithm.

Shadow Merge rules:
  - Local results receive a configurable boost (default x1.5).
  - Team results receive a configurable boost (default x1.0).
  - TeamBullets with enforcement="mandatory" bypass boosting and get
    a fixed top-priority score.
  - Incompatible-tag conflicts: when a new result's tags intersect an
    already-accepted result's incompatible_tags (or vice-versa), the
    lower-scored entry is dropped.
  - Near-duplicate fallback: when neither result carries incompatible_tags,
    Jaccard word-similarity >= 0.95 is treated as a duplicate and the
    lower-scored entry is dropped.
  - Supersede conflict: when a local bullet has a ``supersedes`` field pointing
    to a team bullet origin_id, the local version wins and the team version
    is dropped (Shadow Merge override).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, Sequence, runtime_checkable

from memx.team.config import MandatoryOverride

logger = logging.getLogger(__name__)

# Sentinel score for mandatory bullets (must exceed any real boosted score).
_MANDATORY_SCORE: float = 999.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerBoostConfig:
    """Boost multipliers for each retrieval layer."""

    local_boost: float = 1.5
    team_boost: float = 1.0


# ---------------------------------------------------------------------------
# StorageBackend Protocol — minimal contract expected by MultiPoolRetriever
# ---------------------------------------------------------------------------


@runtime_checkable
class StorageBackend(Protocol):
    """Minimal search interface that every pool must satisfy."""

    def search(self, query: str, *, limit: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
        ...


# ---------------------------------------------------------------------------
# ScoredResult
# ---------------------------------------------------------------------------


@dataclass
class ScoredResult:
    """A search result annotated with source and boosted score."""

    bullet: dict[str, Any]
    raw_score: float
    boosted_score: float
    source: str  # "local" | "team_git" | "team_cache" | …
    is_mandatory: bool = False


@dataclass
class SupersedeConflict:
    """Detected conflict where a local bullet supersedes a team bullet."""

    local_bullet: dict[str, Any]
    team_bullet: dict[str, Any]
    similarity: float


# ---------------------------------------------------------------------------
# MultiPoolRetriever
# ---------------------------------------------------------------------------


class MultiPoolRetriever:
    """Combines Local and Team pools with Shadow Merge.

    Implements the ``StorageBackend`` protocol so it can itself be nested.
    """

    def __init__(
        self,
        local_backend: StorageBackend,
        team_pools: Sequence[tuple[str, StorageBackend]] | None = None,
        boost_config: LayerBoostConfig | None = None,
        *,
        pool_timeout: float = 0.5,
        mandatory_overrides: Sequence[MandatoryOverride] | None = None,
        audit_callback: Any | None = None,
    ) -> None:
        self._local = local_backend
        self._team_pools: Sequence[tuple[str, StorageBackend]] = team_pools or []
        self._boost = boost_config or LayerBoostConfig()
        self._pool_timeout = pool_timeout
        # Build override lookup: bullet_id -> MandatoryOverride (last wins)
        self._overrides: dict[str, MandatoryOverride] = {}
        for ov in mandatory_overrides or []:
            self._overrides[ov.bullet_id] = ov
        # Optional async callback for audit events: fn(event_dict) -> None
        self._audit_callback = audit_callback

    # -- StorageBackend Protocol -----------------------------------------------

    def search(
        self, query: str, *, limit: int = 10, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Query all pools in parallel, then Shadow Merge."""
        raw_results: list[ScoredResult] = []

        pool_count = len(self._team_pools) + 1
        with ThreadPoolExecutor(max_workers=pool_count) as executor:
            futures: dict[Any, str] = {}
            futures[
                executor.submit(self._local.search, query, limit=limit * 2)
            ] = "local"
            for name, pool in self._team_pools:
                futures[
                    executor.submit(pool.search, query, limit=limit * 2)
                ] = name

            for future in as_completed(futures):
                source = futures[future]
                try:
                    pool_results = future.result(timeout=self._pool_timeout)
                    raw_results.extend(self._score_results(pool_results, source))
                except Exception:
                    # Silent degradation — team pool failure never breaks search.
                    logger.warning("Pool %s query failed, skipping", source)

        merged = self._shadow_merge(raw_results)
        # Apply mandatory override logic + hint injection
        merged = self._apply_overrides(merged)
        return merged[:limit]

    # -- Override logic --------------------------------------------------------

    def _apply_overrides(
        self, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Check mandatory overrides: skip mandatory enforcement for active overrides,
        inject deviation hints, and fire audit events."""
        if not self._overrides:
            return results

        now = datetime.now(timezone.utc)
        output: list[dict[str, Any]] = []

        for bullet in results:
            bid = bullet.get("id", "")
            enforcement = bullet.get("enforcement", "")
            override = self._overrides.get(bid)

            if enforcement == "mandatory" and override is not None:
                if override.is_active(now=now):
                    # Active override: downgrade mandatory to normal scoring
                    # and inject deviation hint
                    hint = (
                        f"[OVERRIDE] 你的项目已覆盖团队规则 [{bid}]: "
                        f"{override.reason} "
                        f"(有效期至 {override.expires.isoformat()})"
                    )
                    bullet = {**bullet, "_override_hint": hint}
                    logger.info(
                        "Mandatory bullet %s overridden: %s (expires %s)",
                        bid, override.reason, override.expires.isoformat(),
                    )
                    # Fire audit event (non-blocking)
                    self._fire_audit(bid, override)
                # else: override expired — mandatory behavior restored automatically

            output.append(bullet)
        return output

    def _fire_audit(self, bullet_id: str, override: MandatoryOverride) -> None:
        """Fire an audit event for a mandatory override deviation.
        Failures are logged but never block retrieval."""
        if self._audit_callback is None:
            return
        event = {
            "type": "mandatory_override_deviation",
            "bullet_id": bullet_id,
            "reason": override.reason,
            "expires": override.expires.isoformat(),
        }
        try:
            self._audit_callback(event)
        except Exception:
            logger.warning(
                "Audit callback failed for bullet %s, ignoring", bullet_id
            )

    # -- Internal helpers ------------------------------------------------------

    def _score_results(
        self, results: list[dict[str, Any]], source: str
    ) -> list[ScoredResult]:
        """Apply layer boost to raw search results.

        When a bullet has an active MandatoryOverride, its mandatory
        enforcement is downgraded to normal team-boost scoring.
        """
        scored: list[ScoredResult] = []
        now = datetime.now(timezone.utc)
        for r in results:
            raw_score: float = float(
                r.get("score", r.get("instructivity_score", 50.0) / 100.0)
            )
            bid = r.get("id", "")
            is_mandatory = (
                r.get("enforcement") == "mandatory" and source != "local"
            )

            # Check if an active override suppresses mandatory enforcement
            override = self._overrides.get(bid)
            if is_mandatory and override is not None and override.is_active(now=now):
                is_mandatory = False  # downgrade to normal scoring

            boost = (
                self._boost.local_boost if source == "local"
                else self._boost.team_boost
            )
            boosted = _MANDATORY_SCORE if is_mandatory else raw_score * boost

            scored.append(
                ScoredResult(
                    bullet=r,
                    raw_score=raw_score,
                    boosted_score=boosted,
                    source=source,
                    is_mandatory=is_mandatory,
                )
            )
        return scored

    def detect_supersede_conflicts(
        self,
        local_results: list[dict[str, Any]],
        team_results: list[dict[str, Any]],
        *,
        similarity_threshold: float = 0.8,
    ) -> list[SupersedeConflict]:
        """Detect supersede conflicts between local and team results.

        A supersede conflict occurs when:
          - A local bullet has ``supersedes`` field matching a team bullet ID, OR
          - Content similarity >= threshold but content is not identical
            (indicating a local correction of team knowledge).

        Args:
            local_results: Bullets from the local pool.
            team_results: Bullets from team pools.
            similarity_threshold: Min Jaccard similarity for implicit detection.

        Returns:
            List of SupersedeConflict instances.
        """
        conflicts: list[SupersedeConflict] = []

        # Build team ID -> bullet lookup
        team_by_id: dict[str, dict[str, Any]] = {}
        for t in team_results:
            tid = t.get("id", t.get("origin_id", ""))
            if tid:
                team_by_id[tid] = t

        for local in local_results:
            # Explicit supersede reference
            supersedes_id = local.get("supersedes")
            if supersedes_id and supersedes_id in team_by_id:
                team = team_by_id[supersedes_id]
                sim = _content_similarity(
                    local.get("content", ""), team.get("content", "")
                )
                conflicts.append(
                    SupersedeConflict(
                        local_bullet=local,
                        team_bullet=team,
                        similarity=sim,
                    )
                )
                continue

            # Implicit detection: high similarity but not identical
            local_content = local.get("content", "")
            if not local_content:
                continue
            for team in team_results:
                team_content = team.get("content", "")
                if not team_content:
                    continue
                sim = _content_similarity(local_content, team_content)
                if sim >= similarity_threshold and sim < 1.0:
                    conflicts.append(
                        SupersedeConflict(
                            local_bullet=local,
                            team_bullet=team,
                            similarity=sim,
                        )
                    )

        logger.debug("Detected %d supersede conflict(s)", len(conflicts))
        return conflicts

    def _shadow_merge(
        self, results: list[ScoredResult]
    ) -> list[dict[str, Any]]:
        """Merge results with conflict detection.

        Invariants:
          - Output is sorted by boosted_score descending.
          - Tag-incompatible entries: keep higher-scored only.
          - Near-duplicate entries (no incompatible_tags, Jaccard >= 0.95):
            keep higher-scored only.
          - Supersede override: local bullet with ``supersedes`` field drops
            the referenced team bullet.
          - Mandatory entries always survive conflict checks.
        """
        results.sort(key=lambda r: -r.boosted_score)

        # Collect supersede overrides: team IDs that should be dropped
        superseded_ids: set[str] = set()
        for r in results:
            if r.source == "local":
                sid = r.bullet.get("supersedes")
                if sid:
                    superseded_ids.add(sid)

        merged: list[ScoredResult] = []

        for r in results:
            # Drop team bullets that have been superseded by local
            if superseded_ids and r.source != "local" and not r.is_mandatory:
                bullet_id = r.bullet.get("id", r.bullet.get("origin_id", ""))
                if bullet_id in superseded_ids:
                    logger.debug(
                        "Dropping superseded team bullet %s in shadow merge",
                        bullet_id,
                    )
                    continue
            tags = _get_tags(r.bullet)
            incomp = _get_incompatible_tags(r.bullet)

            conflict = False
            for existing in merged:
                existing_tags = _get_tags(existing.bullet)
                existing_incomp = _get_incompatible_tags(existing.bullet)

                # Mutual incompatibility check
                if _tags_conflict(tags, existing_incomp) or _tags_conflict(
                    existing_tags, incomp
                ):
                    conflict = True
                    break

                # Fallback: near-duplicate detection when neither has incompatible_tags
                if not incomp and not existing_incomp:
                    content_a = r.bullet.get("content", "")
                    content_b = existing.bullet.get("content", "")
                    if content_a and content_b:
                        sim = _content_similarity(content_a, content_b)
                        if sim >= 0.95:
                            conflict = True
                            break

            if not conflict or r.is_mandatory:
                merged.append(r)

        return [r.bullet for r in merged]


# ---------------------------------------------------------------------------
# Pure helper functions (module-level for easy testing)
# ---------------------------------------------------------------------------


def _get_tags(bullet: dict[str, Any]) -> list[str]:
    """Extract tags list from a bullet dict."""
    val = bullet.get("tags", [])
    return list(val) if val else []


def _get_incompatible_tags(bullet: dict[str, Any]) -> list[str]:
    """Extract incompatible_tags list from a bullet dict."""
    val = bullet.get("incompatible_tags", [])
    return list(val) if val else []


def _tags_conflict(tags_a: list[str], incompatible_b: list[str]) -> bool:
    """Return True when any tag in *tags_a* appears in *incompatible_b*."""
    return bool(set(tags_a) & set(incompatible_b))


def _content_similarity(a: str, b: str) -> float:
    """Quick Jaccard word-similarity for near-duplicate fallback."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)
