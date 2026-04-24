"""Unit tests for Inbox state machine + crash recovery (STORY-R095)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from memorus.core.engines.reflector.inbox import Inbox, make_entry


def _seed(tmp_path: Path, n: int) -> Inbox:
    box = Inbox(tmp_path / "inbox.jsonl")
    for i in range(n):
        box.append(
            make_entry(
                user_message=f"user {i}",
                assistant_message=f"asst {i}",
                conversation_id="conv1",
                turn_offset=i,
            )
        )
    return box


def test_append_and_list(tmp_path: Path) -> None:
    box = _seed(tmp_path, 3)
    pending = box.list_pending()
    assert len(pending) == 3
    # Order preserved
    assert [e.turn_offset for e in pending] == [0, 1, 2]


def test_claim_transitions_pending_to_in_progress(tmp_path: Path) -> None:
    box = _seed(tmp_path, 5)
    claimed = box.claim(3)
    assert len(claimed) == 3
    assert all(e.status == "in_progress" for e in claimed)
    assert len(box.list_pending()) == 2
    # Persisted
    reborn = Inbox(tmp_path / "inbox.jsonl")
    assert len(reborn.list_pending()) == 2


def test_mark_consumed(tmp_path: Path) -> None:
    box = _seed(tmp_path, 3)
    claimed = box.claim(2)
    ids = [e.id for e in claimed]
    n = box.mark_consumed(ids)
    assert n == 2
    all_entries = box.list_all()
    statuses = {e.status for e in all_entries}
    assert "consumed" in statuses
    assert "pending" in statuses
    assert box.counts()["consumed_last_hour"] == 2


def test_recover_crashed_reverts_in_progress(tmp_path: Path) -> None:
    box = _seed(tmp_path, 4)
    box.claim(3)
    # Simulate crash: open fresh handle, entries still in_progress.
    crashed = Inbox(tmp_path / "inbox.jsonl")
    n = crashed.recover_crashed()
    assert n == 3
    assert len(crashed.list_pending()) == 4
    assert all(e.status == "pending" for e in crashed.list_all())


def test_recover_is_idempotent(tmp_path: Path) -> None:
    box = _seed(tmp_path, 2)
    box.claim(2)
    box.recover_crashed()
    # Running again should produce no changes.
    assert box.recover_crashed() == 0


def test_malformed_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "inbox.jsonl"
    box = _seed(tmp_path, 2)
    with path.open("a", encoding="utf-8") as f:
        f.write("{not json}\n")
        f.write("\n")
    # Fresh handle
    box2 = Inbox(path)
    assert len(box2.list_pending()) == 2


def test_vacuum_prunes_expired(tmp_path: Path) -> None:
    box = Inbox(tmp_path / "inbox.jsonl", consumed_retention_seconds=1)
    box.append(make_entry(user_message="u", assistant_message="a"))
    claimed = box.claim(1)
    # Back-date consumed_at
    box.mark_consumed([claimed[0].id])
    entries = box.list_all()
    path = tmp_path / "inbox.jsonl"
    # Rewrite with old consumed_at
    old = (datetime.now(timezone.utc) - timedelta(seconds=3600)).isoformat()
    lines = []
    for e in entries:
        d = json.loads(path.read_text(encoding="utf-8").strip().splitlines()[0])
        d["consumed_at"] = old
        lines.append(json.dumps(d))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    removed = box.vacuum()
    assert removed == 1
    assert box.list_all() == []


def test_requeue_restores_pending(tmp_path: Path) -> None:
    box = _seed(tmp_path, 2)
    claimed = box.claim(2)
    ids = [e.id for e in claimed]
    n = box.requeue(ids)
    assert n == 2
    assert len(box.list_pending()) == 2
