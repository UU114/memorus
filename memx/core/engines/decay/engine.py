"""DecayEngine — computes temporal decay weights for memory lifecycle management.

Implements exponential decay with protection period, permanent retention,
and archival threshold logic.  All parameters are configurable via DecayConfig.

Extends with batch sweep() for lifecycle management and reinforce() for
recall-based weight strengthening.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from memx.core.config import DecayConfig
from memx.core.engines.decay.formulas import boosted_weight, exponential_decay

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecayResult:
    """Result of a single decay weight computation.

    Attributes:
        weight:         New decay weight, clamped to [0.0, 1.0].
        should_archive: True when weight falls below archive_threshold.
        is_permanent:   True when recall_count >= permanent_threshold.
        is_protected:   True when the memory is within its protection period.
    """

    weight: float
    should_archive: bool
    is_permanent: bool
    is_protected: bool


@dataclass
class BulletDecayInfo:
    """Input for sweep: minimal info needed for decay calculation.

    Attributes:
        bullet_id:      Unique identifier of the memory bullet.
        created_at:     When the memory was created.
        recall_count:   Number of times the memory has been recalled.
        last_recall:    Timestamp of the most recent recall.
        current_weight: Current decay weight before this sweep.
    """

    bullet_id: str
    created_at: datetime
    recall_count: int = 0
    last_recall: datetime | None = None
    current_weight: float = 1.0


@dataclass
class DecaySweepResult:
    """Aggregated result of a batch sweep operation.

    Attributes:
        updated:   Number of bullets whose weight changed.
        archived:  Number of bullets marked for archival.
        permanent: Number of bullets with permanent retention.
        unchanged: Number of bullets whose weight did not change.
        errors:    List of error descriptions for bullets that failed.
        details:   Per-bullet mapping of bullet_id -> DecayResult.
    """

    updated: int = 0
    archived: int = 0
    permanent: int = 0
    unchanged: int = 0
    errors: list[str] = field(default_factory=list)
    details: dict[str, DecayResult] = field(default_factory=dict)


class DecayEngine:
    """Orchestrates decay weight computation for memory records.

    Provides single-record ``compute_weight()``, batch ``sweep()``, and
    recall-based ``reinforce()`` operations.

    Usage::

        engine = DecayEngine()  # uses default DecayConfig
        result = engine.compute_weight(
            created_at=some_datetime,
            recall_count=3,
        )
        print(result.weight, result.should_archive)

        # Batch sweep
        sweep_result = engine.sweep(bullets)
        print(sweep_result.updated, sweep_result.archived)

        # Reinforce on recall hit
        count = engine.reinforce(["id1", "id2"], update_fn)
    """

    def __init__(self, config: DecayConfig | None = None) -> None:
        self._config = config or DecayConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_weight(
        self,
        created_at: datetime | None = None,
        recall_count: int | None = None,
        last_recall: datetime | None = None,
        now: datetime | None = None,
    ) -> DecayResult:
        """Compute current decay weight for a memory record.

        Args:
            created_at:   When the memory was created (defaults to *now*).
            recall_count: Number of times the memory has been recalled (defaults to 0).
            last_recall:  Timestamp of the most recent recall (unused in v1 formula,
                          reserved for future refinement).
            now:          Reference "current time" for testing; defaults to UTC now.

        Returns:
            DecayResult with weight, archive flag, permanent flag, and protection flag.
        """
        # Resolve defaults for None / missing inputs
        if now is None:
            now = datetime.now(timezone.utc)
        if created_at is None:
            created_at = now
        if recall_count is None:
            recall_count = 0

        cfg = self._config

        # -- Permanent retention check --
        if recall_count >= cfg.permanent_threshold:
            logger.debug(
                "DecayEngine: PERMANENT (recall_count=%d >= threshold=%d)",
                recall_count, cfg.permanent_threshold,
            )
            return DecayResult(
                weight=1.0,
                should_archive=False,
                is_permanent=True,
                is_protected=False,
            )

        # -- Age computation (handle future timestamps / clock skew) --
        age_days = self._age_in_days(created_at, now)
        logger.debug("DecayEngine: age_days=%.2f recall_count=%d", age_days, recall_count)

        # -- Protection period check --
        if age_days <= cfg.protection_days:
            logger.debug("DecayEngine: PROTECTED (age=%.2f <= protection=%d)", age_days, cfg.protection_days)
            return DecayResult(
                weight=1.0,
                should_archive=False,
                is_permanent=False,
                is_protected=True,
            )

        # -- Exponential decay with recall boost --
        base = exponential_decay(age_days, cfg.half_life_days)
        weight = boosted_weight(base, cfg.boost_factor, recall_count)

        should_archive = weight < cfg.archive_threshold
        logger.debug(
            "DecayEngine: base=%.4f boosted=%.4f archive=%s (threshold=%.4f)",
            base, weight, should_archive, cfg.archive_threshold,
        )

        return DecayResult(
            weight=weight,
            should_archive=should_archive,
            is_permanent=False,
            is_protected=False,
        )

    def sweep(
        self,
        bullets: list[BulletDecayInfo],
        now: datetime | None = None,
    ) -> DecaySweepResult:
        """Batch-compute decay weights for all provided bullets.

        Processes each bullet independently so that a single failure does not
        abort the entire sweep.  Bullets with ``created_at`` as ``None``-like
        invalid values are skipped and recorded in ``errors``.

        Args:
            bullets: List of BulletDecayInfo records to process.
            now:     Reference "current time" for testing; defaults to UTC now.

        Returns:
            DecaySweepResult with per-category counts and per-bullet details.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        result = DecaySweepResult()

        logger.debug("DecayEngine.sweep: processing %d bullets", len(bullets))
        for bullet in bullets:
            try:
                logger.debug("DecayEngine.sweep: bullet=%r age_days=%.1f recalls=%d",
                             bullet.bullet_id,
                             self._age_in_days(bullet.created_at, now),
                             bullet.recall_count)
                decay = self.compute_weight(
                    created_at=bullet.created_at,
                    recall_count=bullet.recall_count,
                    last_recall=bullet.last_recall,
                    now=now,
                )
            except Exception as exc:
                msg = f"bullet_id={bullet.bullet_id!r}: {exc}"
                logger.warning("sweep error — %s", msg)
                result.errors.append(msg)
                continue

            result.details[bullet.bullet_id] = decay

            # Categorize the outcome
            if decay.is_permanent:
                logger.debug("DecayEngine.sweep: %r -> PERMANENT", bullet.bullet_id)
                result.permanent += 1
            elif decay.should_archive:
                logger.debug("DecayEngine.sweep: %r -> ARCHIVE (weight=%.4f)", bullet.bullet_id, decay.weight)
                result.archived += 1
            elif decay.weight != bullet.current_weight:
                logger.debug("DecayEngine.sweep: %r -> UPDATED (%.4f -> %.4f)",
                             bullet.bullet_id, bullet.current_weight, decay.weight)
                result.updated += 1
            else:
                logger.debug("DecayEngine.sweep: %r -> UNCHANGED (weight=%.4f)", bullet.bullet_id, decay.weight)
                result.unchanged += 1

        logger.debug(
            "DecayEngine.sweep done: updated=%d archived=%d permanent=%d unchanged=%d errors=%d",
            result.updated, result.archived, result.permanent, result.unchanged, len(result.errors),
        )
        return result

    def reinforce(
        self,
        bullet_ids: list[str],
        update_fn: Callable[[str, dict[str, object]], None],
    ) -> int:
        """Reinforce memories on recall hit by incrementing recall_count.

        For each bullet_id, calls ``update_fn(bullet_id, payload)`` where
        payload contains ``{"recall_count_delta": 1, "last_recall": <utc_now>}``.
        If the callback raises, the error is logged and processing continues
        with the remaining IDs.

        Args:
            bullet_ids: List of bullet IDs that were recalled.
            update_fn:  Callback that persists the update. Signature:
                        ``(bullet_id: str, update: dict) -> None``.

        Returns:
            Number of successfully reinforced bullets.
        """
        reinforced = 0
        now = datetime.now(timezone.utc)

        for bid in bullet_ids:
            try:
                update_fn(bid, {
                    "recall_count_delta": 1,
                    "last_recall": now,
                })
                reinforced += 1
            except Exception as exc:
                logger.warning(
                    "reinforce error — bullet_id=%r: %s", bid, exc,
                )

        return reinforced

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> DecayConfig:
        """Current decay configuration (read-only)."""
        return self._config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _age_in_days(created_at: datetime, now: datetime) -> float:
        """Return age in fractional days, floored at 0 for future timestamps."""
        delta = now - created_at
        age = delta.total_seconds() / 86400.0
        return max(0.0, age)
