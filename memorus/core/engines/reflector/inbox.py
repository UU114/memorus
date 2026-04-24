"""Inbox — append-only JSONL queue for deferred distillation (STORY-R095).

Each inbox record represents a single conversation turn waiting to be
batch-distilled. The file lives at ``config.reflector.batch.inbox_path``
(default ``.ace/inbox.jsonl``) and uses JSON-Lines encoding so that
simultaneous appends cost O(1) and full-file scans are fast.

State machine::

    pending  ---(BatchAnalyzer claims) --->  in_progress
    in_progress  ---(success) --------------> consumed
    in_progress  ---(crash) ----------------> pending   (recovered at startup)
    pending  ---(explicit skip) ------------> skipped

``consumed`` entries are kept for ``consumed_retention_seconds`` and
then purged on the next write; ``skipped`` entries are purged
immediately by :meth:`purge_terminal`.

The file is rewritten via an atomic ``tempfile + os.replace`` when state
transitions change existing rows (append-only writes use ``O_APPEND``
text mode directly).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Record model
# ---------------------------------------------------------------------------


@dataclass
class InboxEntry:
    """One turn awaiting batch distillation."""

    id: str
    conversation_id: str
    turn_hash: str
    turn_offset: int
    role: str
    content: str
    timestamp: str  # ISO-8601 UTC
    status: str = "pending"  # pending | in_progress | consumed | skipped
    correction_detected: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    consumed_at: Optional[str] = None
    assistant_message: Optional[str] = None  # paired response, when known

    # --------------------------------------------------------------
    # Serialisation helpers
    # --------------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(self._as_dict(), ensure_ascii=False)

    def _as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "turn_hash": self.turn_hash,
            "turn_offset": self.turn_offset,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "status": self.status,
            "correction_detected": self.correction_detected,
        }
        if self.metadata:
            d["metadata"] = self.metadata
        if self.consumed_at is not None:
            d["consumed_at"] = self.consumed_at
        if self.assistant_message is not None:
            d["assistant_message"] = self.assistant_message
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "InboxEntry":
        return cls(
            id=str(raw["id"]),
            conversation_id=str(raw.get("conversation_id", "")),
            turn_hash=str(raw.get("turn_hash", "")),
            turn_offset=int(raw.get("turn_offset", 0)),
            role=str(raw.get("role", "user")),
            content=str(raw.get("content", "")),
            timestamp=str(raw.get("timestamp", "")),
            status=str(raw.get("status", "pending")),
            correction_detected=bool(raw.get("correction_detected", False)),
            metadata=dict(raw.get("metadata", {}) or {}),
            consumed_at=raw.get("consumed_at"),
            assistant_message=raw.get("assistant_message"),
        )


# ---------------------------------------------------------------------------
# Inbox store
# ---------------------------------------------------------------------------


class Inbox:
    """File-backed inbox queue."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        consumed_retention_seconds: int = 7 * 24 * 3600,
    ) -> None:
        self._path = Path(path)
        self._retention = max(0, int(consumed_retention_seconds))
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.touch()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def _read_all(self) -> list[InboxEntry]:
        if not self._path.exists():
            return []
        entries: list[InboxEntry] = []
        with self._path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Inbox: skip malformed line %d in %s: %s",
                        lineno, self._path, e,
                    )
                    continue
                try:
                    entries.append(InboxEntry.from_dict(raw))
                except (KeyError, TypeError, ValueError) as e:
                    logger.warning(
                        "Inbox: skip invalid record at line %d: %s", lineno, e,
                    )
        return entries

    def list_pending(self) -> list[InboxEntry]:
        with self._lock:
            return [e for e in self._read_all() if e.status == "pending"]

    def list_all(self) -> list[InboxEntry]:
        with self._lock:
            return self._read_all()

    def oldest_pending_age_seconds(self, now: Optional[datetime] = None) -> Optional[float]:
        """Return age in seconds of the oldest pending entry, or None if empty."""
        now = now or datetime.now(timezone.utc)
        pendings = self.list_pending()
        if not pendings:
            return None
        oldest = min(
            (_parse_ts(e.timestamp) for e in pendings),
            default=None,
        )
        if oldest is None:
            return None
        return max(0.0, (now - oldest).total_seconds())

    def counts(self, within_hour: bool = True) -> dict[str, int]:
        """Return counts of pending / in_progress / consumed-last-hour / skipped."""
        with self._lock:
            entries = self._read_all()
        pending = sum(1 for e in entries if e.status == "pending")
        in_progress = sum(1 for e in entries if e.status == "in_progress")
        consumed = 0
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        for e in entries:
            if e.status != "consumed":
                continue
            if not within_hour:
                consumed += 1
                continue
            ts = _parse_ts(e.consumed_at or e.timestamp)
            if ts >= cutoff:
                consumed += 1
        skipped = sum(1 for e in entries if e.status == "skipped")
        return {
            "pending": pending,
            "in_progress": in_progress,
            "consumed_last_hour": consumed,
            "skipped": skipped,
            "total": len(entries),
        }

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append(self, entry: InboxEntry) -> None:
        """O(1) append of a new pending entry."""
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(entry.to_json())
                f.write("\n")
        logger.debug(
            "Inbox.append: id=%s conv=%s offset=%d corr=%s",
            entry.id, entry.conversation_id, entry.turn_offset, entry.correction_detected,
        )

    # ------------------------------------------------------------------
    # State transitions (rewrite file)
    # ------------------------------------------------------------------

    def _rewrite(self, entries: Iterable[InboxEntry]) -> None:
        parent = self._path.parent
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=self._path.name + ".", suffix=".tmp", dir=str(parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for e in entries:
                    f.write(e.to_json())
                    f.write("\n")
            os.replace(tmp_path, self._path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def claim(self, batch_size: int) -> list[InboxEntry]:
        """Atomically move up to *batch_size* pending entries to in_progress."""
        if batch_size <= 0:
            return []
        with self._lock:
            entries = self._read_all()
            claimed: list[InboxEntry] = []
            mutated = False
            for e in entries:
                if e.status == "pending" and len(claimed) < batch_size:
                    e.status = "in_progress"
                    claimed.append(e)
                    mutated = True
            if mutated:
                self._rewrite(entries)
        logger.debug("Inbox.claim: %d entries marked in_progress", len(claimed))
        return claimed

    def mark_consumed(self, ids: Iterable[str]) -> int:
        id_set = set(ids)
        if not id_set:
            return 0
        now_iso = _now_iso()
        with self._lock:
            entries = self._read_all()
            count = 0
            for e in entries:
                if e.id in id_set and e.status in ("pending", "in_progress"):
                    e.status = "consumed"
                    e.consumed_at = now_iso
                    count += 1
            if count:
                entries = self._prune_expired(entries)
                self._rewrite(entries)
        return count

    def mark_skipped(self, ids: Iterable[str]) -> int:
        id_set = set(ids)
        if not id_set:
            return 0
        with self._lock:
            entries = self._read_all()
            kept: list[InboxEntry] = []
            removed = 0
            for e in entries:
                if e.id in id_set:
                    removed += 1
                    continue
                kept.append(e)
            if removed:
                self._rewrite(kept)
        return removed

    def requeue(self, ids: Iterable[str]) -> int:
        """Move the given in_progress entries back to pending."""
        id_set = set(ids)
        if not id_set:
            return 0
        with self._lock:
            entries = self._read_all()
            count = 0
            for e in entries:
                if e.id in id_set and e.status == "in_progress":
                    e.status = "pending"
                    count += 1
            if count:
                self._rewrite(entries)
        return count

    # ------------------------------------------------------------------
    # Recovery / maintenance
    # ------------------------------------------------------------------

    def recover_crashed(self) -> int:
        """Roll back any in_progress entries to pending. Called at startup."""
        with self._lock:
            entries = self._read_all()
            count = 0
            for e in entries:
                if e.status == "in_progress":
                    e.status = "pending"
                    count += 1
            if count:
                self._rewrite(entries)
        if count:
            logger.info("Inbox.recover_crashed: %d in_progress entries reverted to pending", count)
        return count

    def _prune_expired(self, entries: list[InboxEntry]) -> list[InboxEntry]:
        """Drop consumed entries older than ``consumed_retention_seconds``."""
        if self._retention <= 0:
            return entries
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self._retention)
        pruned: list[InboxEntry] = []
        dropped = 0
        for e in entries:
            if e.status == "consumed":
                ts = _parse_ts(e.consumed_at or e.timestamp)
                if ts < cutoff:
                    dropped += 1
                    continue
            pruned.append(e)
        if dropped:
            logger.debug("Inbox: pruned %d expired consumed entries", dropped)
        return pruned

    def vacuum(self) -> int:
        """Force a full prune of expired consumed entries. Returns number removed."""
        with self._lock:
            entries = self._read_all()
            before = len(entries)
            entries = self._prune_expired(entries)
            removed = before - len(entries)
            if removed:
                self._rewrite(entries)
        return removed


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_entry(
    *,
    user_message: str,
    assistant_message: str,
    conversation_id: str = "",
    turn_offset: int = 0,
    role: str = "user",
    correction_detected: bool = False,
    timestamp: Optional[datetime] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> InboxEntry:
    """Build an ``InboxEntry`` from an InteractionEvent-like payload."""
    from memorus.core.types import SourceRef

    ts = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
    turn_hash = SourceRef.compute_turn_hash(user_message + "\n" + assistant_message)
    return InboxEntry(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        turn_hash=turn_hash,
        turn_offset=int(turn_offset),
        role=role,
        content=user_message,
        timestamp=ts.isoformat(),
        correction_detected=bool(correction_detected),
        metadata=dict(metadata or {}),
        assistant_message=assistant_message,
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.fromtimestamp(0, tz=timezone.utc)
