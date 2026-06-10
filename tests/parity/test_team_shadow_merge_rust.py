"""Team shadow-merge READ-path integration test (Rust backend).

Validates that a ``scope="team:NAME"`` search through the Rust binding + shim
merges team-pool knowledge READ-ONLY into the results:

* A team-only bullet (present only in an offline ``.ace/playbook.jsonl`` team
  pool, NOT in the local store) surfaces in the search results, tagged with
  ``source == "git-fallback"``.
* The local store is NEVER mutated by the team-scoped search (count + contents
  unchanged before/after).
* With no team source available, a ``team:NAME`` search degrades gracefully to
  the local results instead of crashing.

The suite is SKIPPED (never failed) when the ``memorus_r`` extension is not
importable, mirroring the Rust availability gate in
``test_cross_language_parity.py``.

Scope note
----------
This exercises the offline GitFallbackStorage backend only. Online sync, the
TeamCacheStorage server path, and governance/nomination are intentionally out of
scope for this read-path test (they require server/sync infrastructure) and are
deferred to a follow-up.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Rust availability gate. The wheel may be absent in some CI lanes; in that
# case this whole module is SKIPPED (never failed) with a clear reason.
# ---------------------------------------------------------------------------
try:
    import memorus_r  # noqa: F401

    _RUST_AVAILABLE = True
    _RUST_SKIP_REASON = ""
except Exception as exc:  # pragma: no cover - exercised only without the wheel
    _RUST_AVAILABLE = False
    _RUST_SKIP_REASON = (
        f"memorus_r extension not importable ({exc!r}); "
        "build/install it to run the team shadow-merge read-path test"
    )

pytestmark = pytest.mark.skipif(not _RUST_AVAILABLE, reason=_RUST_SKIP_REASON)


def _full_bullet_jsonl_line(
    content: str,
    *,
    tags: list[str] | None = None,
    key_entities: list[str] | None = None,
) -> str:
    """Build one ``.ace/playbook.jsonl`` line as a FULL ``BulletMetadata`` JSON.

    GitFallbackStorage deserializes each line directly into the Rust
    ``BulletMetadata`` struct, whose fields are mostly NOT ``#[serde(default)]``
    — so every required field must be present (a partial / export-style
    ``JsonlRecord`` line would be silently skipped as malformed). Enum wire
    forms match the Rust serde attributes (lowercase / snake_case).
    """
    now = datetime.now(timezone.utc).isoformat()
    return json.dumps(
        {
            "id": str(uuid.uuid4()),
            "content": content,
            # Any stable string works for a team-only bullet (no local hash
            # collision is expected); use the content so it is human-readable.
            "hash": f"team-{abs(hash(content))}",
            "section": "general",
            "knowledge_type": "method",
            "instructivity_score": 0,
            "recall_count": 0,
            "last_recall": None,
            "decay_weight": 1.0,
            "related_tools": [],
            "related_files": [],
            "key_entities": key_entities or [],
            "tags": tags or [],
            "distilled_rule": None,
            "source_type": "manual",
            "scope": "global",
            "schema_version": 1,
            "incompatible_tags": [],
            "metadata": {},
            "user_id": None,
            "agent_id": None,
            "run_id": None,
            "created_at": now,
            "updated_at": now,
        }
    )


def _seed_team_pool(tmp_path: Any, lines: list[str]) -> str:
    """Write a ``.ace/playbook.jsonl`` team pool and return its path."""
    ace_dir = tmp_path / ".ace"
    ace_dir.mkdir(parents=True, exist_ok=True)
    pool = ace_dir / "playbook.jsonl"
    pool.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(pool)


def _rows(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the result rows from a search envelope."""
    return list((envelope or {}).get("results") or [])


def test_team_scope_surfaces_team_bullet_read_only(tmp_path: Any) -> None:
    """A team-only bullet surfaces under team scope; local store is untouched."""
    from memorus.core._rust_backend import RustBackedMemory

    team_content = "always run cargo clippy before committing rust changes"
    pool_path = _seed_team_pool(
        tmp_path,
        [_full_bullet_jsonl_line(team_content, tags=["rust", "clippy"], key_entities=["rust"])],
    )

    mem = RustBackedMemory()
    # Seed a DISTINCT local bullet so we can tell the two pools apart.
    local_content = "deploy the python service with gunicorn workers"
    mem.add(local_content, user_id="dev-1", infer=False)

    # Snapshot the local store before the team-scoped search.
    before = mem.get_all(user_id="dev-1")
    before_count = len(list((before or {}).get("memories") or []))

    # Team-scoped search: query targets the team bullet's keywords.
    out = mem.search(
        "cargo clippy rust",
        scope="team:demo",
        team_pool_path=pool_path,
    )
    rows = _rows(out)

    # The team bullet must appear, tagged as coming from the git-fallback pool.
    team_hits = [r for r in rows if r.get("source") == "git-fallback"]
    assert team_hits, f"expected a git-fallback team hit, got: {rows}"
    assert any(r.get("memory") == team_content for r in team_hits)
    # Source / is_mandatory keys are present on merged rows.
    assert all("is_mandatory" in r for r in rows)

    # Local store is NOT mutated by the read-only shadow merge.
    after = mem.get_all(user_id="dev-1")
    after_rows = list((after or {}).get("memories") or [])
    assert len(after_rows) == before_count
    assert any(
        (r.get("content") or r.get("memory")) == local_content for r in after_rows
    )
    # The team-only content must NOT have been written into the local store.
    assert not any(
        (r.get("content") or r.get("memory")) == team_content for r in after_rows
    )


def test_team_scope_without_pool_degrades_gracefully(tmp_path: Any) -> None:
    """A team scope with no offline pool degrades to local results, no crash."""
    from memorus.core._rust_backend import RustBackedMemory

    missing_pool = str(tmp_path / ".ace" / "playbook.jsonl")  # never created
    assert not os.path.exists(missing_pool)

    mem = RustBackedMemory()
    local_content = "prefer structured logging over print statements"
    mem.add(local_content, user_id="dev-2", infer=False)

    # Should not raise even though the team pool is absent.
    out = mem.search(
        "structured logging",
        scope="team:demo",
        team_pool_path=missing_pool,
    )
    rows = _rows(out)
    # No team source -> only local results survive the merge, each tagged local.
    assert all(r.get("source") in (None, "local") for r in rows)
    assert any(r.get("memory") == local_content for r in rows)


@pytest.mark.asyncio
async def test_team_scope_async_surfaces_team_bullet(tmp_path: Any) -> None:
    """Async binding: team-scoped search also merges the team pool read-only."""
    from memorus.core._rust_backend import RustBackedAsyncMemory

    team_content = "use tokio::select for concurrent cancellation in rust"
    pool_path = _seed_team_pool(
        tmp_path,
        [_full_bullet_jsonl_line(team_content, tags=["rust", "tokio"], key_entities=["rust"])],
    )

    mem = RustBackedAsyncMemory()
    await mem.add("local async note about asyncio gather", user_id="dev-3", infer=False)

    out = await mem.search(
        "tokio select rust",
        scope="team:demo",
        team_pool_path=pool_path,
    )
    rows = _rows(out)
    assert any(
        r.get("source") == "git-fallback" and r.get("memory") == team_content
        for r in rows
    ), f"expected async git-fallback team hit, got: {rows}"
