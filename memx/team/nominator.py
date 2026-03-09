"""Nominator — automatic nomination pipeline for local bullets to team pool.

Auto-detects high-value local bullets based on recall count and instructivity
score, applies redaction, and uploads via AceSyncClient. Supports silent mode,
rate limiting, session-scoped skipping, and permanent ignore lists.

Includes GovernanceClassifier for three-tier review classification (STORY-069).

Also provides SupersedeDetector for detecting when a local bullet corrects
team-sourced knowledge, and a SupersedeProposal dataclass for the submission.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memx.team.config import AutoNominateConfig
from memx.team.types import GovernanceTier

logger = logging.getLogger(__name__)

# Tags that require curator review
_SENSITIVE_TAGS: frozenset[str] = frozenset({"security", "architecture", "mandatory"})

# Score threshold for auto-approval (combined with non-sensitive tags)
_AUTO_APPROVE_SCORE_THRESHOLD: float = 90.0

# Default file for permanently ignored supersede detections
_IGNORED_SUPERSEDES_FILENAME = "ignored_supersedes.json"


# ---------------------------------------------------------------------------
# Result / summary data classes
# ---------------------------------------------------------------------------


@dataclass
class NominationResult:
    """Outcome of a single nomination attempt."""

    success: bool
    bullet_id: str | None = None
    error: str | None = None
    was_ignored: bool = False


@dataclass
class NominationSummary:
    """End-of-session summary of nomination activity."""

    total_candidates: int = 0
    nominated_count: int = 0
    ignored_count: int = 0
    pending_count: int = 0
    pending_bullets: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SupersedeProposal:
    """Proposal to correct a team bullet with updated local knowledge.

    Attributes:
        origin_id: ID of the original team bullet being corrected.
        new_content: Corrected content from local bullet.
        local_bullet_id: ID of the local bullet that triggered detection.
        priority: "normal" or "urgent".
        reason: Human-readable reason for the correction.
        similarity: Semantic similarity score between local and team content.
    """

    origin_id: str
    new_content: str
    local_bullet_id: str = ""
    priority: str = "normal"
    reason: str = ""
    similarity: float = 0.0


@dataclass
class SupersedeNotification:
    """Notification that a team bullet was updated and may override local version.

    Generated when a cached team bullet is updated and the local pool still
    holds an older version.
    """

    team_bullet_id: str
    old_content: str
    new_content: str
    local_bullet_id: str = ""
    message: str = ""


# ---------------------------------------------------------------------------
# State file name
# ---------------------------------------------------------------------------

_STATE_FILENAME = "nomination_state.json"


# ---------------------------------------------------------------------------
# GovernanceClassifier
# ---------------------------------------------------------------------------


class GovernanceClassifier:
    """Classify bullets into governance tiers based on score and tags.

    Rules:
      - score >= 90 AND no sensitive tags -> auto_approve
      - any sensitive tag (security, architecture, mandatory) -> curator_required
      - everything else -> p2p_review

    Sensitive tags and score threshold can be overridden at init time.
    """

    def __init__(
        self,
        *,
        sensitive_tags: frozenset[str] | None = None,
        auto_approve_threshold: float = _AUTO_APPROVE_SCORE_THRESHOLD,
    ) -> None:
        self._sensitive_tags = sensitive_tags or _SENSITIVE_TAGS
        self._auto_approve_threshold = auto_approve_threshold

    def classify(self, score: float, tags: list[str] | tuple[str, ...]) -> GovernanceTier:
        """Return the governance tier for a bullet.

        Args:
            score: Instructivity score (0-100).
            tags: List of tags attached to the bullet.

        Returns:
            GovernanceTier enum value.
        """
        tag_set = {t.lower() for t in tags}

        # Sensitive tags always require curator
        if tag_set & {t.lower() for t in self._sensitive_tags}:
            return GovernanceTier.CURATOR_REQUIRED

        # High score with no sensitive tags -> auto approve
        if score >= self._auto_approve_threshold:
            return GovernanceTier.AUTO_APPROVE

        return GovernanceTier.P2P_REVIEW


# ---------------------------------------------------------------------------
# Nominator
# ---------------------------------------------------------------------------


class Nominator:
    """Automatic nomination pipeline for local bullets.

    Scans local bullets for candidates meeting recall-count and score
    thresholds, applies redaction via Redactor, and uploads via
    AceSyncClient.nominate_bullet.

    Args:
        config: AutoNominateConfig with thresholds and limits.
        redactor: Optional Redactor for sanitisation before upload.
        sync_client: Optional AceSyncClient for uploading nominations.
        state_dir: Directory for persisting nominated/ignored IDs.
    """

    def __init__(
        self,
        config: AutoNominateConfig,
        redactor: Any | None = None,
        sync_client: Any | None = None,
        *,
        state_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._redactor = redactor
        self._sync_client = sync_client
        self._state_dir = state_dir

        # Session state
        self._session_prompt_count: int = 0
        self._candidates: list[dict[str, Any]] = []
        self._nominated_ids: set[str] = set()
        self._ignored_ids: set[str] = set()
        self._session_skipped_ids: set[str] = set()

        # Load persisted state (nominated_ids, ignored_ids)
        self._load_state()

    # -- candidate scanning -------------------------------------------------

    def scan_candidates(self, bullets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Find nomination candidates from a local bullet pool.

        Filters by:
          - recall_count >= min_recall_count
          - instructivity_score >= min_score
          - not already nominated
          - not permanently ignored
          - not session-skipped

        Args:
            bullets: List of bullet dicts with ``id``, ``recall_count``,
                     ``instructivity_score``, and ``content`` keys.

        Returns:
            List of qualifying candidate dicts.
        """
        candidates: list[dict[str, Any]] = []
        for b in bullets:
            bid = _bullet_id(b)
            if not bid:
                continue
            if bid in self._nominated_ids or bid in self._ignored_ids:
                continue
            if bid in self._session_skipped_ids:
                continue

            recall = b.get("recall_count", 0)
            score = b.get("instructivity_score", 0.0)

            if (
                recall >= self._config.min_recall_count
                and score >= self._config.min_score
            ):
                candidates.append(b)

        self._candidates = candidates
        logger.debug("Nominator found %d candidate(s)", len(candidates))
        return candidates

    # -- prompt gating ------------------------------------------------------

    def should_prompt(self) -> bool:
        """Whether to show a nomination prompt to the user.

        Returns ``False`` in silent mode or when the session prompt limit
        has been reached.
        """
        if self._config.silent:
            return False
        return (
            self._session_prompt_count < self._config.max_prompts_per_session
            and bool(self._candidates)
        )

    # -- nomination pipeline ------------------------------------------------

    async def nominate(
        self,
        bullet: dict[str, Any],
        *,
        user_approved_content: str | None = None,
    ) -> NominationResult:
        """Full nomination pipeline: redact -> (user edits) -> upload.

        Args:
            bullet: Bullet dict to nominate.
            user_approved_content: If provided, applied as user edits after
                L1 redaction.

        Returns:
            NominationResult indicating success or failure.
        """
        bid = _bullet_id(bullet)
        content = bullet.get("content", "")

        if not self._sync_client:
            return NominationResult(success=False, error="No sync client available")

        if not self._redactor:
            return NominationResult(success=False, error="No redactor available")

        # L1 redact
        redacted = self._redactor.redact_l1(content)

        # Apply user edits if provided
        if user_approved_content is not None:
            redacted = self._redactor.apply_user_edits(redacted, user_approved_content)

        # Finalize into upload-ready dict
        final = self._redactor.finalize(redacted)
        # Merge bullet metadata (except raw content which is replaced by redacted)
        for k, v in bullet.items():
            if k != "content" and k not in final:
                final[k] = v
        final["content"] = final.get("content", content)

        try:
            response = await self._sync_client.nominate_bullet(final)
            self._nominated_ids.add(bid)
            self._session_prompt_count += 1
            self._save_state()
            logger.info("Nominated bullet %s -> server id %s", bid, response.id)
            return NominationResult(success=True, bullet_id=response.id)
        except Exception as exc:
            logger.warning("Nomination failed for bullet %s: %s", bid, exc)
            return NominationResult(success=False, error=str(exc))

    # -- pending list -------------------------------------------------------

    def get_pending_nominations(self) -> list[dict[str, Any]]:
        """List candidates waiting for user action.

        Excludes bullets that have already been nominated or session-skipped.
        """
        return [
            c
            for c in self._candidates
            if _bullet_id(c) not in self._nominated_ids
            and _bullet_id(c) not in self._session_skipped_ids
        ]

    # -- ignore / skip ------------------------------------------------------

    def ignore_bullet(self, bullet_id: str) -> None:
        """Permanently ignore a bullet for nomination (persisted)."""
        self._ignored_ids.add(bullet_id)
        self._save_state()
        logger.debug("Permanently ignored bullet %s", bullet_id)

    def skip_bullet(self, bullet_id: str) -> None:
        """Skip a bullet for the current session only (not persisted)."""
        self._session_skipped_ids.add(bullet_id)
        logger.debug("Session-skipped bullet %s", bullet_id)

    # -- session summary ----------------------------------------------------

    def session_summary(self) -> NominationSummary:
        """End-of-session summary of nomination activity."""
        pending = self.get_pending_nominations()
        return NominationSummary(
            total_candidates=len(self._candidates),
            nominated_count=len(self._nominated_ids),
            ignored_count=len(self._ignored_ids),
            pending_count=len(pending),
            pending_bullets=pending,
        )

    # -- state persistence --------------------------------------------------

    def _state_path(self) -> Path | None:
        """Resolve path to the state JSON file."""
        if self._state_dir is None:
            return None
        return self._state_dir / _STATE_FILENAME

    def _load_state(self) -> None:
        """Load nominated and ignored IDs from disk."""
        path = self._state_path()
        if path is None or not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._nominated_ids = set(data.get("nominated_ids", []))
            self._ignored_ids = set(data.get("ignored_ids", []))
            logger.debug(
                "Loaded nomination state: %d nominated, %d ignored",
                len(self._nominated_ids),
                len(self._ignored_ids),
            )
        except Exception:
            logger.warning("Failed to load nomination state from %s", path, exc_info=True)

    def _save_state(self) -> None:
        """Persist nominated and ignored IDs to disk."""
        path = self._state_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "nominated_ids": sorted(self._nominated_ids),
                "ignored_ids": sorted(self._ignored_ids),
            }
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.debug("Saved nomination state to %s", path)
        except Exception:
            logger.warning("Failed to save nomination state to %s", path, exc_info=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bullet_id(bullet: dict[str, Any]) -> str:
    """Extract bullet ID, preferring 'id' over 'origin_id'."""
    return bullet.get("id", bullet.get("origin_id", ""))


def _jaccard_similarity(a: str, b: str) -> float:
    """Jaccard word-level similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


# ---------------------------------------------------------------------------
# SupersedeDetector
# ---------------------------------------------------------------------------


class SupersedeDetector:
    """Detects when a local bullet corrects team-sourced knowledge.

    A supersede is detected when:
      - Semantic similarity between local and team content >= similarity_threshold (0.8)
      - Content difference (1 - Jaccard word similarity) >= difference_threshold (0.2)

    This means the topics overlap significantly, but the actual wording differs
    enough to constitute a correction rather than a duplicate.

    Args:
        similarity_threshold: Min semantic similarity for topic overlap.
        difference_threshold: Min content difference for correction detection.
        state_dir: Directory for persisting ignored supersede IDs.
    """

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.8,
        difference_threshold: float = 0.2,
        state_dir: Path | None = None,
    ) -> None:
        self._similarity_threshold = similarity_threshold
        self._difference_threshold = difference_threshold
        self._state_dir = state_dir
        self._ignored_pairs: set[str] = set()  # "local_id::team_id"

        self._load_ignored()

    def detect(
        self,
        local_bullets: list[dict[str, Any]],
        team_bullets: list[dict[str, Any]],
        *,
        similarity_fn: Any | None = None,
    ) -> list[SupersedeProposal]:
        """Scan local bullets against team bullets for correction candidates.

        Args:
            local_bullets: List of local bullet dicts (with id, content).
            team_bullets: List of team bullet dicts (with id/origin_id, content, source).
            similarity_fn: Optional callable(str, str) -> float for semantic
                similarity. Falls back to Jaccard word similarity if not provided.

        Returns:
            List of SupersedeProposal for detected corrections.
        """
        proposals: list[SupersedeProposal] = []
        sim_fn = similarity_fn or _jaccard_similarity

        for local in local_bullets:
            local_id = _bullet_id(local)
            local_content = local.get("content", "")
            if not local_id or not local_content:
                continue

            for team in team_bullets:
                team_id = _bullet_id(team)
                team_content = team.get("content", "")
                if not team_id or not team_content:
                    continue

                # Skip if this pair is permanently ignored
                pair_key = f"{local_id}::{team_id}"
                if pair_key in self._ignored_pairs:
                    continue

                # Only consider bullets sourced from team
                source = team.get("source", "")
                if source == "local":
                    continue

                similarity = sim_fn(local_content, team_content)
                if similarity < self._similarity_threshold:
                    continue

                # Check content difference via word-level Jaccard
                word_sim = _jaccard_similarity(local_content, team_content)
                difference = 1.0 - word_sim
                if difference < self._difference_threshold:
                    continue

                proposals.append(
                    SupersedeProposal(
                        origin_id=team_id,
                        new_content=local_content,
                        local_bullet_id=local_id,
                        priority="normal",
                        reason=f"Local bullet corrects team content (similarity={similarity:.2f}, diff={difference:.2f})",
                        similarity=similarity,
                    )
                )

        logger.debug("SupersedeDetector found %d proposal(s)", len(proposals))
        return proposals

    def ignore_pair(self, local_id: str, team_id: str) -> None:
        """Permanently ignore a supersede detection pair."""
        pair_key = f"{local_id}::{team_id}"
        self._ignored_pairs.add(pair_key)
        self._save_ignored()
        logger.debug("Permanently ignored supersede pair: %s", pair_key)

    def is_ignored(self, local_id: str, team_id: str) -> bool:
        """Check if a pair is permanently ignored."""
        return f"{local_id}::{team_id}" in self._ignored_pairs

    # -- Persistence for ignored supersedes ------------------------------------

    def _ignored_path(self) -> Path | None:
        if self._state_dir is None:
            return None
        return self._state_dir / _IGNORED_SUPERSEDES_FILENAME

    def _load_ignored(self) -> None:
        """Load ignored supersede pairs from disk."""
        path = self._ignored_path()
        if path is None or not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._ignored_pairs = set(data.get("ignored_pairs", []))
            logger.debug("Loaded %d ignored supersede pair(s)", len(self._ignored_pairs))
        except Exception:
            logger.warning("Failed to load ignored supersedes from %s", path, exc_info=True)

    def _save_ignored(self) -> None:
        """Persist ignored supersede pairs to disk."""
        path = self._ignored_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {"ignored_pairs": sorted(self._ignored_pairs)}
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.debug("Saved ignored supersedes to %s", path)
        except Exception:
            logger.warning("Failed to save ignored supersedes to %s", path, exc_info=True)


async def submit_supersede(
    proposal: SupersedeProposal,
    *,
    redactor: Any | None = None,
    sync_client: Any | None = None,
) -> NominationResult:
    """Submit a supersede proposal: redact content then upload.

    Args:
        proposal: SupersedeProposal to submit.
        redactor: Optional Redactor for sanitisation before upload.
        sync_client: AceSyncClient for uploading the proposal.

    Returns:
        NominationResult indicating success or failure.
    """
    if not sync_client:
        return NominationResult(success=False, error="No sync client available")

    content = proposal.new_content

    # Apply redaction if available
    if redactor is not None:
        redacted = redactor.redact_l1(content)
        final = redactor.finalize(redacted)
        content = final.get("content", content)

    new_bullet = {
        "content": content,
        "priority": proposal.priority,
        "reason": proposal.reason,
    }

    try:
        response = await sync_client.propose_supersede(
            proposal.origin_id,
            new_bullet,
            priority=proposal.priority,
        )
        logger.info(
            "Supersede proposal submitted: %s -> %s (status=%s)",
            proposal.origin_id,
            response.id,
            response.status,
        )
        return NominationResult(success=True, bullet_id=response.id)
    except Exception as exc:
        logger.warning("Supersede proposal failed for %s: %s", proposal.origin_id, exc)
        return NominationResult(success=False, error=str(exc))
