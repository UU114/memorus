"""ConsolidateExecutor — turn a ``ConsolidateResult`` into store side effects.

Given the output of :meth:`CuratorEngine.consolidate_corpus`, this module
applies three kinds of actions:

* ``merge_groups``  -> Supersede each group into the survivor bullet
  (highest ``recall_count``, tie broken by latest ``created_at``). Sources
  are merged via :func:`memorus.core.types.merge_sources` (STORY-R092).
* ``redundant_ids`` -> Soft delete only (sets ``deleted_at`` metadata;
  never physically removes the record).
* ``conflicts``     -> Three-tier triage:

    - ``confidence >= auto_supersede_min_confidence`` -> auto Supersede
    - ``review_queue_min_confidence <= confidence < auto_supersede_min``
      -> append to ``.ace/review_queue.jsonl`` with a pre-generated
      ``memorus review <id>`` hint
    - below that -> set ``conflict_with: [id]`` on both bullets' metadata

All backend operations go through a :class:`MemoryAdapter` protocol so the
executor stays decoupled from mem0 specifics and STORY-R096's graph-edge
hook can be layered on top without touching this file.

Mirrors ``memorus_ace::curator::executor::ConsolidateExecutor`` byte-for-byte.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from memorus.core.config import ConsolidateConfig
from memorus.core.engines.curator.engine import (
    ConsolidateConflictInfo,
    ConsolidateResult,
    MergeGroup,
)
from memorus.core.types import SourceRef, merge_sources

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapter protocol & patch types
# ---------------------------------------------------------------------------


@dataclass
class BulletRecord:
    """Minimal bullet shape the executor needs from the adapter."""

    bullet_id: str
    content: str
    recall_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sources: list[SourceRef] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class ConflictResolutionInfo:
    """Audit trail of an auto-resolved conflict."""

    conflict_type: str
    reason: str
    confidence: float
    resolved_at: datetime


@dataclass
class MetadataPatch:
    """Optional fields the executor may patch on a bullet."""

    sources: list[SourceRef] | None = None
    supersedes: list[str] | None = None
    superseded_by: str | None = None
    conflict_with: list[str] | None = None
    conflict_resolution: ConflictResolutionInfo | None = None
    updated_at: datetime | None = None


class MemoryAdapter(Protocol):
    """Backend-agnostic shape of the operations the executor performs.

    Implementations must be idempotent: :meth:`get_bullet` for an
    already-deleted id should return None, and the executor will skip
    without raising.
    """

    def get_bullet(self, bullet_id: str) -> BulletRecord | None:
        ...

    def update_metadata(self, bullet_id: str, patch: MetadataPatch) -> None:
        ...

    def mark_deleted(self, bullet_id: str, when: datetime) -> None:
        ...


# ---------------------------------------------------------------------------
# Report & log format
# ---------------------------------------------------------------------------


@dataclass
class ExecutionReport:
    """Summary of a single executor run. Used for `.ace/log.md`."""

    merged_groups: int = 0
    merged_bullets_total: int = 0
    soft_deleted: int = 0
    auto_supersede: int = 0
    queued_for_review: int = 0
    marked_conflict: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_log_block(self, ts: datetime) -> str:
        """Render as the standard ``.ace/log.md`` block.

        Output matches Rust's :meth:`ExecutionReport::to_log_block` exactly.
        """
        ts_str = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        return (
            f"## {ts_str} consolidate\n"
            f"merged: {self.merged_groups} groups "
            f"({self.merged_bullets_total} bullets -> {self.merged_groups})\n"
            f"soft-deleted: {self.soft_deleted}\n"
            f"auto-supersede: {self.auto_supersede}\n"
            f"queued-for-review: {self.queued_for_review}\n"
            f"marked-conflict: {self.marked_conflict}\n"
            f"duration: {self.duration_seconds:.1f}s\n"
        )


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class ConsolidateExecutor:
    """Apply a ``ConsolidateResult`` to a memory store via ``MemoryAdapter``."""

    def __init__(
        self,
        adapter: MemoryAdapter,
        config: ConsolidateConfig,
    ) -> None:
        self._adapter = adapter
        self._config = config

    def execute(self, result: ConsolidateResult) -> ExecutionReport:
        """Apply all three kinds of effects and return a summary."""
        report = ExecutionReport()

        # 1. merge_groups -> Supersede
        for group in result.merge_groups:
            report.merged_bullets_total += len(group.bullet_ids)
            try:
                if self._execute_merge(group):
                    report.merged_groups += 1
            except Exception as e:
                logger.error("executor: merge failed: %s", e, exc_info=True)
                report.errors.append(f"merge: {e}")

        # 2. redundant_ids -> soft delete
        now = datetime.now(timezone.utc)
        for bid in result.redundant_ids:
            if self._soft_delete(bid, now):
                report.soft_deleted += 1

        # 3. conflicts -> three-tier triage
        auto_min = self._config.conflict.auto_supersede_min_confidence
        review_min = self._config.conflict.review_queue_min_confidence
        for conflict in result.conflicts:
            confidence = conflict_confidence(conflict)
            if confidence >= auto_min:
                try:
                    if self._auto_supersede_conflict(conflict, confidence):
                        report.auto_supersede += 1
                except Exception as e:
                    logger.error(
                        "executor: auto_supersede failed: %s", e, exc_info=True
                    )
                    report.errors.append(f"conflict: {e}")
            elif confidence >= review_min:
                self._enqueue_review(conflict, confidence)
                report.queued_for_review += 1
            else:
                self._mark_conflict_with(conflict)
                report.marked_conflict += 1

        return report

    # ----- merge groups ------------------------------------------------

    def _execute_merge(self, group: MergeGroup) -> bool:
        """Merge a group into its survivor. Returns True on success.

        Returns False (without error) if fewer than two bullets in the group
        are still live — the executor is idempotent by design.
        """
        records: list[BulletRecord] = []
        for bid in group.bullet_ids:
            rec = self._adapter.get_bullet(bid)
            if rec is not None:
                records.append(rec)
        if len(records) < 2:
            logger.debug(
                "_execute_merge: only %d live bullets, skipping", len(records)
            )
            return False

        winner_idx = _pick_winner_idx(records)
        winner = records.pop(winner_idx)
        losers = records

        merged = list(winner.sources)
        for loser in losers:
            merged = merge_sources(merged, loser.sources)

        now = datetime.now(timezone.utc)
        loser_ids = [l.bullet_id for l in losers]

        self._adapter.update_metadata(
            winner.bullet_id,
            MetadataPatch(
                sources=merged,
                supersedes=loser_ids,
                updated_at=now,
            ),
        )
        for loser_id in loser_ids:
            self._adapter.update_metadata(
                loser_id,
                MetadataPatch(
                    superseded_by=winner.bullet_id,
                    updated_at=now,
                ),
            )
            self._adapter.mark_deleted(loser_id, now)
        return True

    # ----- soft delete -------------------------------------------------

    def _soft_delete(self, bullet_id: str, when: datetime) -> bool:
        if self._adapter.get_bullet(bullet_id) is None:
            logger.debug("_soft_delete: %s already gone", bullet_id)
            return False
        self._adapter.mark_deleted(bullet_id, when)
        return True

    # ----- conflict triage ---------------------------------------------

    def _auto_supersede_conflict(
        self,
        conflict: ConsolidateConflictInfo,
        confidence: float,
    ) -> bool:
        """Auto-Supersede the higher-risk side of a high-confidence conflict.

        Strategy: keep the bullet with the higher recall_count (tie-break
        by latest created_at); soft-delete the other.
        """
        a = self._adapter.get_bullet(conflict.a_id)
        b = self._adapter.get_bullet(conflict.b_id)

        if a is None and b is None:
            return False

        # Pick winner by recall_count then created_at
        if a is None:
            winner, loser = b, None
        elif b is None:
            winner, loser = a, None
        else:
            if a.recall_count > b.recall_count or (
                a.recall_count == b.recall_count and a.created_at >= b.created_at
            ):
                winner, loser = a, b
            else:
                winner, loser = b, a

        now = datetime.now(timezone.utc)
        resolution = ConflictResolutionInfo(
            conflict_type=conflict.conflict_type,
            reason=conflict.reason,
            confidence=confidence,
            resolved_at=now,
        )

        if loser is None:
            # Only one side is still live -> just audit-stamp the survivor.
            self._adapter.update_metadata(
                winner.bullet_id,
                MetadataPatch(
                    conflict_resolution=resolution,
                    updated_at=now,
                ),
            )
            return True

        # Loser gets superseded_by + deleted; winner gets supersedes + audit.
        self._adapter.update_metadata(
            winner.bullet_id,
            MetadataPatch(
                supersedes=[loser.bullet_id],
                conflict_resolution=resolution,
                updated_at=now,
            ),
        )
        self._adapter.update_metadata(
            loser.bullet_id,
            MetadataPatch(
                superseded_by=winner.bullet_id,
                conflict_resolution=resolution,
                updated_at=now,
            ),
        )
        self._adapter.mark_deleted(loser.bullet_id, now)
        return True

    def _enqueue_review(
        self,
        conflict: ConsolidateConflictInfo,
        confidence: float,
    ) -> None:
        """Append one review-queue row; fall back to ``_mark_conflict_with``
        if the path is not writable.
        """
        path = Path(self._config.review_queue_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(
                "_enqueue_review: cannot create %s: %s -> fallback mark",
                path.parent, e,
            )
            self._mark_conflict_with(conflict)
            return

        row = {
            "a_id": conflict.a_id,
            "b_id": conflict.b_id,
            "a_content": conflict.a_content,
            "b_content": conflict.b_content,
            "similarity": conflict.similarity,
            "conflict_type": conflict.conflict_type,
            "reason": conflict.reason,
            "confidence": confidence,
            "enqueued_at": datetime.now(timezone.utc).isoformat(),
            "hint": f"memorus review {conflict.b_id}",
        }

        try:
            line = json.dumps(row, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            logger.warning("_enqueue_review: serialize failed: %s", e)
            self._mark_conflict_with(conflict)
            return

        try:
            with open(path, "a", encoding="utf-8") as fp:
                fp.write(line + "\n")
        except OSError as e:
            logger.warning(
                "_enqueue_review: write failed %s: %s -> fallback", path, e
            )
            self._mark_conflict_with(conflict)

    def _mark_conflict_with(self, conflict: ConsolidateConflictInfo) -> None:
        """Mark both sides with ``conflict_with: [peer_id]`` where live."""
        now = datetime.now(timezone.utc)
        if self._adapter.get_bullet(conflict.a_id) is not None:
            self._adapter.update_metadata(
                conflict.a_id,
                MetadataPatch(
                    conflict_with=[conflict.b_id],
                    updated_at=now,
                ),
            )
        if self._adapter.get_bullet(conflict.b_id) is not None:
            self._adapter.update_metadata(
                conflict.b_id,
                MetadataPatch(
                    conflict_with=[conflict.a_id],
                    updated_at=now,
                ),
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_winner_idx(records: list[BulletRecord]) -> int:
    """Pick survivor index: highest recall_count, tie broken by latest
    ``created_at``. Panics if empty (caller guarantees len >= 2).
    """
    best_idx = 0
    for i in range(1, len(records)):
        rec = records[i]
        best = records[best_idx]
        if rec.recall_count > best.recall_count or (
            rec.recall_count == best.recall_count and rec.created_at > best.created_at
        ):
            best_idx = i
    return best_idx


def conflict_confidence(c: ConsolidateConflictInfo) -> float:
    """Derive a [0,1] confidence for a detected conflict.

    Mirrors Rust's :func:`conflict_confidence` byte-for-byte.
    """
    base = max(0.0, min(1.0, c.similarity))
    weight_by_type = {
        "negation": 1.0,
        "contradiction": 0.95,
        "value_difference": 0.9,
    }
    weight = weight_by_type.get(c.conflict_type, 0.9)
    return round(base * weight, 6)


def append_consolidate_log(
    log_path: str | os.PathLike[str],
    report: ExecutionReport,
    ts: datetime,
) -> None:
    """Append one consolidate block to ``.ace/log.md`` (creating parents)."""
    path = Path(log_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("append_consolidate_log: cannot create %s: %s", path.parent, e)
        return

    block = report.to_log_block(ts)
    try:
        # Separator newline if the file has content already
        need_sep = path.exists() and path.stat().st_size > 0
        with open(path, "a", encoding="utf-8") as fp:
            if need_sep:
                fp.write("\n")
            fp.write(block)
    except OSError as e:
        logger.warning("append_consolidate_log: write failed: %s", e)
