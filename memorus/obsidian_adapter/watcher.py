"""Inbox watcher: vault/memorus/inbox/*.md -> memorus write path.

Each inbox file is one draft. The adapter:
  1. parses frontmatter + body
  2. runs Redactor.redact_l1 locally (short-circuit obvious PII)
  3. submits via AceSyncClient.nominate_bullet_sync (team) or Memory.add (personal)
  4. writes a sidecar ``<file>.status.md`` with the outcome

The watcher never bypasses governance: enforcement / status / governance_tier
fields supplied in frontmatter are stripped before submission; only the
server may elevate enforcement.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memorus.obsidian_adapter.frontmatter import dump as fm_dump
from memorus.obsidian_adapter.frontmatter import parse as fm_parse

logger = logging.getLogger(__name__)

# Fields that the server alone is allowed to set; client submissions are
# stripped of these regardless of what the inbox frontmatter contains.
_SERVER_OWNED_FIELDS = frozenset(
    {
        "enforcement",
        "status",
        "governance_tier",
        "upvotes",
        "downvotes",
        "nominated_at",
        "verified_at",
        "verified_status",
        "trust_score",
        "deleted_at",
        "origin_id",
    }
)


@dataclass
class SubmitResult:
    file: Path
    kind: str  # "team" | "personal"
    state: str  # "submitted" | "rejected_pii" | "error" | "skipped"
    detail: str = ""
    bullet_id: str | None = None


class InboxWatcher:
    """Process inbox files. Two modes:

    * ``run_once()`` - process every existing file under inbox/ then return.
    * ``run_forever(interval=2.0)`` - poll loop based on file mtime.
    """

    def __init__(
        self,
        vault_root: Path | str,
        *,
        sync_client: Any | None = None,
        personal_memory: Any | None = None,
        redactor: Any | None = None,
        author_id: str = "",
        priority: str = "normal",
    ) -> None:
        self._root = Path(vault_root) / "memorus"
        self._inbox = self._root / "inbox"
        self._tombstones = self._root / ".tombstones"
        self._sync = sync_client
        self._personal = personal_memory
        self._redactor = redactor
        self._author_id = author_id
        self._priority = priority
        self._seen: dict[Path, float] = {}

    # -- public ------------------------------------------------------------

    def run_once(self) -> list[SubmitResult]:
        results: list[SubmitResult] = []
        if not self._inbox.exists():
            return results
        for path in sorted(self._inbox.glob("*.md")):
            if path.name.endswith(".status.md"):
                continue
            results.append(self._process_file(path))
        for path in sorted(self._tombstones.glob("*.md")) if self._tombstones.exists() else []:
            results.append(self._process_tombstone(path))
        return results

    def run_forever(self, interval: float = 2.0) -> None:
        logger.info("InboxWatcher polling %s every %.1fs", self._inbox, interval)
        while True:
            self.run_once()
            time.sleep(interval)

    # -- core --------------------------------------------------------------

    def _process_file(self, path: Path) -> SubmitResult:
        try:
            mtime = path.stat().st_mtime
        except OSError as e:
            return SubmitResult(path, "?", "error", f"stat failed: {e}")
        if self._seen.get(path) == mtime:
            return SubmitResult(path, "?", "skipped", "unchanged")
        self._seen[path] = mtime

        doc = fm_parse(path.read_text(encoding="utf-8"))
        kind = str(doc.meta.get("type", "team")).lower()
        body = doc.body.strip()
        if not body:
            return self._record(path, SubmitResult(path, kind, "error", "empty body"))

        # SECURITY (fail-closed): never submit external-vault content without a
        # redactor. A missing redactor is a hard error, not a silent bypass.
        if self._redactor is None:
            return self._record(
                path, SubmitResult(path, kind, "error", "no redactor configured")
            )
        try:
            rr = self._redactor.redact_l1(body)
        except Exception as e:
            return self._record(path, SubmitResult(path, kind, "error", f"redactor: {e}"))
        if rr.is_fully_redacted:
            return self._record(
                path,
                SubmitResult(path, kind, "rejected_pii", "content fully redacted"),
            )
        body = rr.clean_content

        if kind == "personal":
            return self._record(path, self._submit_personal(path, doc.meta, body))
        return self._record(path, self._submit_team(path, doc.meta, body))

    def _submit_team(self, path: Path, meta: dict[str, Any], body: str) -> SubmitResult:
        if self._sync is None:
            return SubmitResult(path, "team", "error", "no sync_client configured")
        bullet = {k: v for k, v in meta.items() if k not in _SERVER_OWNED_FIELDS}
        bullet.pop("type", None)
        bullet.pop("kind", None)
        bullet["content"] = body
        if self._author_id and not bullet.get("author_id"):
            bullet["author_id"] = self._author_id
        try:
            resp = self._sync.nominate_bullet_sync(bullet, priority=self._priority)
        except Exception as e:
            return SubmitResult(path, "team", "error", f"nominate failed: {e}")
        bid = getattr(resp, "id", None) or (resp.get("id") if isinstance(resp, dict) else None)
        status = getattr(resp, "status", "submitted") or "submitted"
        return SubmitResult(path, "team", "submitted", str(status), bullet_id=str(bid) if bid else None)

    def _submit_personal(self, path: Path, meta: dict[str, Any], body: str) -> SubmitResult:
        if self._personal is None:
            return SubmitResult(path, "personal", "error", "no personal Memory configured")
        scope = meta.get("scope") or None
        user_id = meta.get("user_id") or self._author_id or None
        try:
            res = self._personal.add(body, user_id=user_id, scope=scope, metadata=meta.get("metadata"))
        except Exception as e:
            return SubmitResult(path, "personal", "error", f"Memory.add failed: {e}")
        return SubmitResult(path, "personal", "submitted", str(res)[:160])

    def _process_tombstone(self, path: Path) -> SubmitResult:
        # Tombstones are not auto-applied — they require operator action via
        # the team server. We log a sidecar so the user can see the request
        # was recorded and copy the bullet_id into the upstream workflow.
        doc = fm_parse(path.read_text(encoding="utf-8"))
        bid = str(doc.meta.get("bullet_id") or doc.meta.get("hash") or "")
        if not bid:
            return self._record(path, SubmitResult(path, "tombstone", "error", "missing bullet_id"))
        return self._record(
            path,
            SubmitResult(
                path,
                "tombstone",
                "submitted",
                f"tombstone request recorded for {bid}; complete via server workflow",
                bullet_id=bid,
            ),
        )

    def _record(self, path: Path, result: SubmitResult) -> SubmitResult:
        sidecar = path.with_suffix(".status.md")
        meta = {
            "source": str(path.name),
            "kind": result.kind,
            "state": result.state,
            "detail": result.detail,
        }
        if result.bullet_id:
            meta["bullet_id"] = result.bullet_id
        sidecar.write_text(fm_dump(meta, ""), encoding="utf-8")
        return result
