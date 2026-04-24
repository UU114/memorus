"""Team export bridge — the ONLY ext module for playbook/index generation.

Moved here from ``memorus.core.cli.main`` to satisfy the core↔team
decoupling invariant: ``memorus.core`` must not import ``memorus.team``.
The CLI calls ``export_team_index`` via this module instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def export_team_index(
    *,
    scope: Optional[str],
    out_dir: Optional[str],
    generator_version: str,
) -> tuple[Path, int]:
    """Generate ``.ace/index.md`` from an existing ``.ace/playbook.jsonl``.

    Returns (written_path, bullet_count).
    """
    from memorus.team.export import write_index_md
    from memorus.team.git_storage import GitFallbackStorage

    target_dir = Path(out_dir) if out_dir else Path(".ace")
    playbook_path = target_dir / "playbook.jsonl"

    storage = GitFallbackStorage(playbook_path=playbook_path)
    _ = storage.bullet_count  # force lazy load

    bullets: list[dict[str, Any]] = []
    for record in storage._bullets:  # internal: read-only use
        item: dict[str, Any] = {
            "content": record.content,
            "section": record.section,
            "tags": list(record.tags),
            "knowledge_type": record.knowledge_type,
            "decay_weight": record.extra.get("decay_weight", 1.0),
            "recall_count": record.extra.get("recall_count", 0),
            "created_at": record.extra.get("created_at", ""),
        }
        for key in ("sources", "source_refs", "sources_count"):
            if key in record.extra:
                item[key] = record.extra[key]
        bullets.append(item)

    resolved_scope = scope or (
        (storage.header or {}).get("scope", "") or "team"
    )
    header = storage.header or {}
    model = header.get("model") or "all-MiniLM-L6-v2"
    dim_val = header.get("dim")
    try:
        dim = int(dim_val) if dim_val is not None else 384
    except (TypeError, ValueError):
        dim = 384

    target = write_index_md(
        bullets,
        resolved_scope,
        target_dir,
        generator=generator_version,
        embedding_model=model,
        embedding_dim=dim,
    )
    return target, len(bullets)
