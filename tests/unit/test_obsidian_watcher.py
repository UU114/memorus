"""Tests for the Obsidian InboxWatcher fail-closed redaction guard."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from memorus.obsidian_adapter.watcher import InboxWatcher


def _make_inbox_file(vault_root: Path, name: str, body: str, kind: str = "team") -> Path:
    inbox = vault_root / "memorus" / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    path = inbox / name
    path.write_text(f"---\ntype: {kind}\n---\n{body}\n", encoding="utf-8")
    return path


def test_process_file_fails_closed_without_redactor(tmp_path: Path) -> None:
    """SECURITY: with no redactor configured, content must NOT be submitted —
    the file is reported as an error, and the sync client is never called."""
    _make_inbox_file(tmp_path, "note.md", "team knowledge with a secret token")
    sync = MagicMock()

    watcher = InboxWatcher(tmp_path, sync_client=sync, redactor=None)
    results = watcher.run_once()

    assert len(results) == 1
    assert results[0].state == "error"
    assert "redactor" in results[0].detail.lower()
    sync.nominate_bullet_sync.assert_not_called()


def test_process_file_submits_when_redactor_present(tmp_path: Path) -> None:
    """With a redactor, redacted content reaches the sync client."""
    _make_inbox_file(tmp_path, "note.md", "team knowledge")

    redacted = MagicMock()
    redacted.is_fully_redacted = False
    redacted.clean_content = "team knowledge"
    redactor = MagicMock()
    redactor.redact_l1.return_value = redacted

    sync = MagicMock()
    sync.nominate_bullet_sync.return_value = {"id": "b1", "status": "submitted"}

    watcher = InboxWatcher(tmp_path, sync_client=sync, redactor=redactor)
    results = watcher.run_once()

    assert len(results) == 1
    redactor.redact_l1.assert_called_once()
    sync.nominate_bullet_sync.assert_called_once()
