"""Team Playbook index.md generator (STORY-R093).

Produces a deterministic, human-readable Markdown index of team playbook
bullets suitable for PR review. Byte-identical output across Python/Rust
(timestamps excluded).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

# Default embedding model fingerprint (matches GitFallbackStorage).
_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_DIM = 384

# Markdown special characters that require backslash escaping inside bullets.
_MD_ESCAPE_CHARS = ("\\", "*", "`", "[", "]", "_")


def _escape_markdown(text: str) -> str:
    """Escape Markdown special characters with backslash."""
    out = []
    for ch in text:
        if ch in _MD_ESCAPE_CHARS:
            out.append("\\")
        out.append(ch)
    return "".join(out)


def _truncate_codepoints(text: str, max_points: int) -> str:
    """Truncate a string to at most `max_points` Unicode code points.

    Collapses internal whitespace runs to a single space, strips
    leading/trailing whitespace, and does NOT add an ellipsis (so the
    output is fully reversible for hashing purposes).
    """
    # Collapse whitespace to match the story's "truncate at 120 code points"
    # expectation without smuggling hidden newlines into Markdown.
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_points:
        return cleaned
    return cleaned[:max_points]


def _bullet_tag(bullet: Mapping[str, Any]) -> str:
    """Resolve the primary group tag for a bullet.

    Precedence: explicit ``tag`` field > first entry in ``tags`` list >
    ``section`` field > "general".
    """
    tag = bullet.get("tag")
    if isinstance(tag, str) and tag:
        return tag
    tags = bullet.get("tags")
    if isinstance(tags, list) and tags:
        first = tags[0]
        if isinstance(first, str) and first:
            return first
    section = bullet.get("section")
    if isinstance(section, str) and section:
        return section
    return "general"


def _bullet_recall(bullet: Mapping[str, Any]) -> int:
    val = bullet.get("recall_count", 0)
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _bullet_weight(bullet: Mapping[str, Any]) -> float:
    val = bullet.get("decay_weight", 1.0)
    try:
        return float(val)
    except (TypeError, ValueError):
        return 1.0


def _bullet_sources(bullet: Mapping[str, Any]) -> int | None:
    """Return count of SourceRef entries, or None if the field is absent.

    Tolerates R092 not being landed yet: if no sources-related field exists,
    returns None and the caller omits the suffix entirely.
    """
    # Accept both ``sources`` (list) and ``source_refs`` (list) or a
    # pre-computed ``sources_count`` integer.
    count = bullet.get("sources_count")
    if isinstance(count, int):
        return count
    for key in ("sources", "source_refs"):
        val = bullet.get(key)
        if isinstance(val, list):
            return len(val)
    return None


def _bullet_created_at(bullet: Mapping[str, Any]) -> str:
    """Return an ISO-8601 timestamp string for tie-breaking.

    Missing/invalid values sort earliest via the empty string so they are
    grouped deterministically.
    """
    val = bullet.get("created_at")
    if isinstance(val, datetime):
        # Normalise to UTC isoformat with 'Z' suffix.
        if val.tzinfo is None:
            val = val.replace(tzinfo=timezone.utc)
        return val.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(val, str):
        return val
    return ""


def _bullet_content(bullet: Mapping[str, Any]) -> str:
    val = bullet.get("content", "")
    return val if isinstance(val, str) else ""


def _format_bullet_line(bullet: Mapping[str, Any]) -> str:
    """Format one bullet as a Markdown list item."""
    recall = _bullet_recall(bullet)
    weight = _bullet_weight(bullet)
    sources = _bullet_sources(bullet)
    content = _truncate_codepoints(_bullet_content(bullet), 120)
    content_esc = _escape_markdown(content)

    extras = f"weight: {weight:.2f}"
    if sources is not None:
        extras += f", sources: {sources}"

    return f"- `[#{recall}]` {content_esc} *({extras})*"


def _format_timestamp(now: datetime) -> str:
    """Format a UTC timestamp as ISO-8601 with trailing 'Z'."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    iso = now.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    return iso.replace("+00:00", "Z")


def generate_index_md(
    bullets: Iterable[Mapping[str, Any]],
    scope: str,
    *,
    generator: str = "memorus",
    embedding_model: str = _DEFAULT_MODEL,
    embedding_dim: int = _DEFAULT_DIM,
    now: datetime | None = None,
) -> str:
    """Generate a deterministic index.md string.

    Args:
        bullets: iterable of bullet dicts/mappings (TeamBulletRecord.__dict__,
            parsed JSONL lines, etc.). Each must expose at minimum ``content``
            plus optional ``tag``/``tags``/``section``, ``recall_count``,
            ``decay_weight``, ``created_at`` and either ``sources``
            (list) or ``sources_count`` (int).
        scope: scope label, e.g. ``team:backend``.
        generator: ``"memorus <version>"`` string for the header.
        embedding_model / embedding_dim: playbook embedding fingerprint.
        now: override for the ``Last updated`` timestamp (tests only).

    Returns:
        A single Markdown string ending with a trailing newline.
    """
    bullets_list = list(bullets)
    total = len(bullets_list)
    ts = _format_timestamp(now or datetime.now(timezone.utc))

    lines: list[str] = []
    lines.append("# Team Playbook Index")
    lines.append("")
    lines.append(f"**Scope:** {scope}")
    lines.append(f"**Total bullets:** {total}")
    lines.append(f"**Last updated:** {ts}")
    lines.append(
        f"**Embedding model:** {embedding_model} (dim={embedding_dim})"
    )
    lines.append(f"**Generated by:** {generator}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Group by primary tag.
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for b in bullets_list:
        groups.setdefault(_bullet_tag(b), []).append(b)

    for tag in sorted(groups.keys()):
        members = groups[tag]
        # Intra-group sort: recall_count DESC, created_at ASC, content ASC.
        # The third key guarantees full determinism even when recall_count
        # and created_at are both identical (stable sort is not enough when
        # callers may pass bullets in arbitrary order).
        members_sorted = sorted(
            members,
            key=lambda b: (
                -_bullet_recall(b),
                _bullet_created_at(b),
                _bullet_content(b),
            ),
        )
        lines.append(f"## {tag} ({len(members_sorted)})")
        lines.append("")
        for b in members_sorted:
            lines.append(_format_bullet_line(b))
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "*Auto-generated index — do not edit manually. "
        "Regenerate via `memorus export --team`.*"
    )
    lines.append("")

    return "\n".join(lines)


def write_index_md(
    bullets: Iterable[Mapping[str, Any]],
    scope: str,
    out_dir: Path | str,
    *,
    generator: str = "memorus",
    embedding_model: str = _DEFAULT_MODEL,
    embedding_dim: int = _DEFAULT_DIM,
    now: datetime | None = None,
) -> Path:
    """Write index.md into ``out_dir`` (auto-creates the directory)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target = out_path / "index.md"
    content = generate_index_md(
        bullets,
        scope,
        generator=generator,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        now=now,
    )
    target.write_text(content, encoding="utf-8")
    return target
