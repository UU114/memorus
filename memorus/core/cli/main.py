"""Memorus CLI — Click-based command-line interface.

Provides `memorus status`, `memorus search`, `memorus learn`, `memorus list`,
`memorus forget`, and `memorus sweep` commands for inspecting and managing
the knowledge base. All commands support --json for machine-readable output.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Optional

import click

import memorus as _memorus_pkg


def _create_memory(user_id: Optional[str] = None) -> Any:
    """Create a Memory instance with error handling.

    Returns the Memory object or None if initialization fails.
    The error message is printed to stderr.
    """
    try:
        from memorus.core.memory import Memory

        return Memory(config={"ace_enabled": True})
    except Exception as e:
        click.echo(f"Error: Failed to initialize Memorus: {e}", err=True)
        return None


def _count_by(memories: list[dict[str, Any]], field: str) -> dict[str, int]:
    """Count memories grouped by a metadata field."""
    counts: dict[str, int] = {}
    for mem in memories:
        if not isinstance(mem, dict):
            continue
        meta = mem.get("metadata", {})
        value = meta.get(field, "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _avg_field(memories: list[dict[str, Any]], field: str) -> float:
    """Compute average of a numeric metadata field."""
    if not memories:
        return 0.0
    total = 0.0
    count = 0
    for mem in memories:
        if not isinstance(mem, dict):
            continue
        meta = mem.get("metadata", {})
        val = meta.get(field, 1.0)
        if val is not None:
            total += float(val)
            count += 1
    return round(total / count, 2) if count > 0 else 0.0


def _print_status(stats: dict[str, Any]) -> None:
    """Print status in human-friendly format."""
    total = stats["total"]
    if total == 0:
        click.echo("No memories yet. Use `memorus learn` to add knowledge.")
        return

    ace_label = "ON" if stats["ace_enabled"] else "OFF"

    click.echo("Memorus Knowledge Base Status")
    click.echo("\u2550" * 26)
    click.echo(f"Total memories:    {total}")
    click.echo(f"ACE mode:          {ace_label}")
    click.echo()

    # Section distribution
    sections = stats.get("sections", {})
    if sections:
        click.echo("Section distribution:")
        for name, count in sorted(sections.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            click.echo(f"  {name:<16} {count:>4} ({pct:.1f}%)")
        click.echo()

    # Knowledge type distribution
    ktypes = stats.get("knowledge_types", {})
    if ktypes:
        click.echo("Knowledge types:")
        for name, count in sorted(ktypes.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            click.echo(f"  {name:<16} {count:>4} ({pct:.1f}%)")
        click.echo()

    click.echo(f"Avg decay weight:  {stats['avg_decay_weight']}")


def _truncate(text: str, max_len: int = 80) -> str:
    """Truncate text to max_len, appending ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _print_results(query: str, results: list[dict[str, Any]]) -> None:
    """Print search results in human-friendly format."""
    if not results:
        click.echo(f'No results found for: "{query}"')
        return

    click.echo(f'Search: "{query}"  ({len(results)} result{"s" if len(results) != 1 else ""})')
    click.echo("\u2500" * 32)
    click.echo()

    for item in results:
        score = item.get("score", 0.0)
        content = item.get("memory", "")
        meta = item.get("metadata", {})
        ktype = meta.get("memorus_knowledge_type", "unknown")
        tags_raw = meta.get("memorus_tags", "[]")
        bullet_id = item.get("id", "")

        # Parse tags (may be JSON string or list)
        if isinstance(tags_raw, str):
            try:
                tags = json.loads(tags_raw)
            except (json.JSONDecodeError, TypeError):
                tags = []
        elif isinstance(tags_raw, list):
            tags = tags_raw
        else:
            tags = []

        tags_str = ", ".join(tags) if tags else ""

        click.echo(f"[{score:.2f}] {_truncate(content)}")
        tag_part = f" | tags: {tags_str}" if tags_str else ""
        click.echo(f"       type: {ktype}{tag_part}")
        click.echo(f"       id: {bullet_id}")
        click.echo()


def _parse_tags(tags_raw: Any) -> list[str]:
    """Parse tags from metadata (may be JSON string or list)."""
    if isinstance(tags_raw, str):
        try:
            tags = json.loads(tags_raw)
            return tags if isinstance(tags, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    elif isinstance(tags_raw, list):
        return tags_raw
    return []


def _print_learn_result(result: dict[str, Any]) -> None:
    """Print learn result in human-friendly format."""
    ace = result.get("ace_ingest", {})
    bullets = ace.get("bullets_added", [])
    errors = ace.get("errors", [])
    raw_fallback = ace.get("raw_fallback", False)

    if errors:
        for err in errors:
            click.echo(f"Warning: {err}", err=True)

    if raw_fallback:
        click.echo("Learned (raw fallback, Reflector skipped):")
    elif not bullets:
        click.echo("Learned new knowledge:")
    else:
        click.echo("Learned new knowledge:")

    for bullet in bullets:
        if isinstance(bullet, dict):
            ktype = bullet.get("knowledge_type", "unknown")
            rule = bullet.get("distilled_rule", bullet.get("content", ""))
            tags = bullet.get("tags", [])
            bid = bullet.get("id", "")
            click.echo(f"  Type:      {ktype}")
            click.echo(f"  Rule:      \"{rule}\"")
            if tags:
                click.echo(f"  Tags:      {', '.join(tags)}")
            if bid:
                click.echo(f"  ID:        {bid}")

    if not bullets:
        # Fallback: show raw result info
        mem_results = result.get("results", [])
        if mem_results:
            for r in mem_results:
                if isinstance(r, dict):
                    click.echo(f"  ID:        {r.get('id', 'unknown')}")
                    click.echo(f"  Content:   {_truncate(r.get('memory', ''), 60)}")
        else:
            click.echo("  (content processed)")


def _apply_filters(
    memories: list[dict[str, Any]],
    scope: Optional[str] = None,
    knowledge_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Filter memories by scope and/or knowledge type."""
    filtered = memories
    if scope is not None:
        filtered = [
            m for m in filtered
            if isinstance(m, dict)
            and m.get("metadata", {}).get("memorus_scope", "") == scope
        ]
    if knowledge_type is not None:
        filtered = [
            m for m in filtered
            if isinstance(m, dict)
            and m.get("metadata", {}).get("memorus_knowledge_type", "") == knowledge_type
        ]
    return filtered


def _print_memory_list(memories: list[dict[str, Any]], total: int) -> None:
    """Print memory list in human-friendly table format."""
    if not memories:
        click.echo("No memories found.")
        return

    showing = len(memories)
    click.echo(f"Memories ({total} total, showing {showing})")
    click.echo("\u2550" * 40)
    click.echo()

    for mem in memories:
        if not isinstance(mem, dict):
            continue
        mid = mem.get("id", "???")
        content = mem.get("memory", "")
        meta = mem.get("metadata", {})
        decay = meta.get("memorus_decay_weight", 1.0)
        ktype = meta.get("memorus_knowledge_type", "unknown")

        # Format: id  [weight] type  | content summary
        short_id = mid[:8] if len(mid) > 8 else mid
        click.echo(
            f"{short_id:<8}  [{float(decay):.2f}] {ktype:<14} | "
            f"{_truncate(content, 50)}"
        )

    click.echo()
    click.echo("Use --scope or --type to filter, --limit to show more.")


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version=_memorus_pkg.__version__, prog_name="memorus")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Memorus - Intelligent Memory Engine for AI Tools."""
    ctx.ensure_object(dict)


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--user-id", default=None, help="Filter by user ID")
@click.pass_context
def status(ctx: click.Context, as_json: bool, user_id: Optional[str]) -> None:
    """Show knowledge base statistics."""
    memory = _create_memory()
    if memory is None:
        ctx.exit(1)
        return

    try:
        stats = memory.status(user_id=user_id)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
        return

    if as_json:
        click.echo(json.dumps(stats, indent=2))
    else:
        _print_status(stats)


# ---------------------------------------------------------------------------
# search command
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=5, show_default=True, help="Max results")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--user-id", default=None, help="Filter by user ID")
@click.option("--scope", default=None, help="Filter by scope (e.g., project:myapp)")
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    limit: int,
    as_json: bool,
    user_id: Optional[str],
    scope: Optional[str],
) -> None:
    """Search knowledge base with hybrid retrieval."""
    memory = _create_memory()
    if memory is None:
        ctx.exit(1)
        return

    try:
        raw = memory.search(query, user_id=user_id, limit=limit, scope=scope)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
        return

    results = raw.get("results", []) if isinstance(raw, dict) else []

    if as_json:
        click.echo(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        _print_results(query, results)


# ---------------------------------------------------------------------------
# learn command
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("content")
@click.option("--raw", is_flag=True, help="Skip Reflector, store as-is")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--user-id", default=None, help="User ID for the memory")
@click.pass_context
def learn(
    ctx: click.Context,
    content: str,
    raw: bool,
    as_json: bool,
    user_id: Optional[str],
) -> None:
    """Teach Memorus new knowledge."""
    if not content.strip():
        click.echo("Error: Content cannot be empty.", err=True)
        ctx.exit(1)
        return

    memory = _create_memory()
    if memory is None:
        ctx.exit(1)
        return

    effective_user_id = user_id or "manual"
    try:
        if raw:
            result = memory.add(
                [{"role": "user", "content": content}],
                user_id=effective_user_id,
                metadata={"source_type": "manual"},
            )
        else:
            result = memory.add(
                [{"role": "user", "content": content}],
                user_id=effective_user_id,
            )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
        return

    if as_json:
        click.echo(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_learn_result(result)


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------


@cli.command("list")
@click.option("--scope", default=None, help="Filter by scope (e.g., project:myapp)")
@click.option("--type", "knowledge_type", default=None, help="Filter by knowledge type")
@click.option("--limit", "-n", default=20, show_default=True, help="Max results")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--user-id", default=None, help="Filter by user ID")
@click.pass_context
def list_memories(
    ctx: click.Context,
    scope: Optional[str],
    knowledge_type: Optional[str],
    limit: int,
    as_json: bool,
    user_id: Optional[str],
) -> None:
    """List all memories in the knowledge base."""
    memory = _create_memory()
    if memory is None:
        ctx.exit(1)
        return

    try:
        kwargs: dict[str, Any] = {}
        if user_id:
            kwargs["user_id"] = user_id
        raw = memory.get_all(**kwargs)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
        return

    all_memories = raw.get("memories", []) if isinstance(raw, dict) else []

    # Apply filters
    filtered = _apply_filters(all_memories, scope=scope, knowledge_type=knowledge_type)
    total = len(filtered)
    limited = filtered[:limit]

    if as_json:
        click.echo(json.dumps(limited, indent=2, ensure_ascii=False))
    else:
        _print_memory_list(limited, total)


# ---------------------------------------------------------------------------
# forget command
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("memory_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def forget(
    ctx: click.Context,
    memory_id: str,
    yes: bool,
    as_json: bool,
) -> None:
    """Delete a specific memory by ID."""
    memory = _create_memory()
    if memory is None:
        ctx.exit(1)
        return

    if not yes:
        # Show memory content for confirmation
        try:
            mem = memory.get(memory_id)
        except Exception:
            mem = None

        if not mem:
            click.echo(f"Error: Memory not found: {memory_id}", err=True)
            ctx.exit(1)
            return

        content = mem.get("memory", "") if isinstance(mem, dict) else str(mem)
        click.echo(f"Will delete: {_truncate(content, 100)}")
        if not click.confirm("Are you sure?"):
            click.echo("Cancelled.")
            return

    try:
        memory.delete(memory_id)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
        return

    result = {"deleted": memory_id}
    if as_json:
        click.echo(json.dumps(result))
    else:
        click.echo(f"Deleted memory: {memory_id}")


# ---------------------------------------------------------------------------
# sweep command
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def sweep(ctx: click.Context, as_json: bool) -> None:
    """Run manual decay sweep on all memories."""
    memory = _create_memory()
    if memory is None:
        ctx.exit(1)
        return

    try:
        result = memory.run_decay_sweep()
    except NotImplementedError:
        click.echo("Error: Decay sweep is not yet implemented.", err=True)
        ctx.exit(1)
        return
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
        return

    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo("Decay sweep complete:")
        click.echo(f"  Updated:   {result.get('updated', 0)}")
        click.echo(f"  Archived:  {result.get('archived', 0)}")
        click.echo(f"  Permanent: {result.get('permanent', 0)}")
        click.echo(f"  Unchanged: {result.get('unchanged', 0)}")


# ---------------------------------------------------------------------------
# conflicts command
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--user-id", default=None, help="Filter by user ID")
@click.pass_context
def conflicts(ctx: click.Context, as_json: bool, user_id: str | None) -> None:
    """Detect contradictory memories."""
    memory = _create_memory()
    if memory is None:
        ctx.exit(1)
        return

    try:
        found = memory.detect_conflicts(user_id=user_id)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
        return

    if as_json:
        from dataclasses import asdict

        click.echo(json.dumps([asdict(c) for c in found], indent=2, ensure_ascii=False))
    else:
        if not found:
            click.echo("No conflicts detected.")
            return

        click.echo(f"Detected {len(found)} potential conflict{'s' if len(found) != 1 else ''}:")
        click.echo("\u2500" * 40)
        click.echo()
        for i, c in enumerate(found, 1):
            click.echo(f"  [{i}] similarity: {c.similarity:.2f}  reason: {c.reason}")
            click.echo(f"      A ({c.memory_a_id[:8]}): {_truncate(c.memory_a_content, 60)}")
            click.echo(f"      B ({c.memory_b_id[:8]}): {_truncate(c.memory_b_content, 60)}")
            click.echo()


# ---------------------------------------------------------------------------
# export command
# ---------------------------------------------------------------------------


@cli.command("export")
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["json", "markdown"]),
    default="json",
    show_default=True,
    help="Export format",
)
@click.option("--scope", default=None, help="Only export memories with this scope")
@click.option("--output", "-o", default=None, help="Write to file instead of stdout")
@click.pass_context
def export_memories(
    ctx: click.Context,
    fmt: str,
    scope: Optional[str],
    output: Optional[str],
) -> None:
    """Export memories to JSON or Markdown."""
    memory = _create_memory()
    if memory is None:
        ctx.exit(1)
        return

    try:
        result = memory.export(format=fmt, scope=scope)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
        return

    # Serialize output
    if isinstance(result, dict):
        text = json.dumps(result, indent=2, ensure_ascii=False)
    else:
        text = str(result)

    if output:
        try:
            with open(output, "w", encoding="utf-8") as f:
                f.write(text)
            click.echo(f"Exported to {output}")
        except OSError as e:
            click.echo(f"Error writing file: {e}", err=True)
            ctx.exit(1)
    else:
        click.echo(text)


# ---------------------------------------------------------------------------
# import command
# ---------------------------------------------------------------------------


@cli.command("import")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Force JSON format parsing")
@click.pass_context
def import_memories(
    ctx: click.Context,
    file_path: str,
    as_json: bool,
) -> None:
    """Import memories from a JSON file."""
    memory = _create_memory()
    if memory is None:
        ctx.exit(1)
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
    except OSError as e:
        click.echo(f"Error reading file: {e}", err=True)
        ctx.exit(1)
        return

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as e:
        click.echo(f"Error: Invalid JSON in {file_path}: {e}", err=True)
        ctx.exit(1)
        return

    try:
        result = memory.import_data(data, format="json")
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
        return
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
        return

    click.echo("Import complete:")
    click.echo(f"  Imported: {result.get('imported', 0)}")
    click.echo(f"  Skipped:  {result.get('skipped', 0)}")
    click.echo(f"  Merged:   {result.get('merged', 0)}")


# ---------------------------------------------------------------------------
# purpose command group (STORY-R098)
# ---------------------------------------------------------------------------


@cli.group()
def purpose() -> None:
    """Manage pool-level Purpose (.ace/purpose.md)."""


@purpose.command("show")
@click.option(
    "--scope",
    type=click.Choice(["project", "global", "effective"]),
    default="effective",
    show_default=True,
    help="Which purpose to display.",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def purpose_show(scope: str, as_json: bool) -> None:
    """Show the current purpose (project-local, global, or merged effective)."""
    from memorus.core.purpose import (
        load_pool_purpose,
        load_purpose_file,
        resolve_purpose_paths,
    )

    project_path, global_path = resolve_purpose_paths()
    if scope == "project":
        p = load_purpose_file(project_path)
    elif scope == "global":
        p = load_purpose_file(global_path)
    else:
        p = load_pool_purpose()

    if p is None or p.is_empty():
        if as_json:
            click.echo(json.dumps({"scope": scope, "empty": True}))
        else:
            click.echo(f"No purpose defined for scope={scope}.")
            click.echo(f"Project path: {project_path}")
            click.echo(f"Global path:  {global_path}")
        return

    if as_json:
        click.echo(
            json.dumps(
                {
                    "scope": p.scope,
                    "keywords": p.keywords,
                    "excluded_topics": p.excluded_topics,
                    "intent_body": p.intent_body,
                    "source_path": str(p.source_path) if p.source_path else None,
                    "nominate_threshold": p.nominate_threshold,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    click.echo(f"Scope:              {p.scope or '(unset)'}")
    click.echo(f"Source:             {p.source_path or '(synthesized)'}")
    click.echo(f"Keywords:           {', '.join(p.keywords) or '(none)'}")
    click.echo(f"Excluded topics:    {', '.join(p.excluded_topics) or '(none)'}")
    click.echo(f"Nominate threshold: {p.nominate_threshold:.2f}")
    click.echo()
    click.echo("Intent body:")
    click.echo(p.intent_body or "(empty)")


@purpose.command("init")
@click.option(
    "--scope",
    type=click.Choice(["project", "global"]),
    default="project",
    show_default=True,
    help="Target location for the new purpose.md.",
)
@click.option("--force", is_flag=True, help="Overwrite existing file.")
@click.option(
    "--name",
    default=None,
    help="Value to write into the YAML `scope:` field (defaults to the chosen scope).",
)
@click.pass_context
def purpose_init(
    ctx: click.Context, scope: str, force: bool, name: Optional[str]
) -> None:
    """Generate a template `.ace/purpose.md` at the chosen location."""
    from memorus.core.purpose import purpose_template, resolve_purpose_paths

    project_path, global_path = resolve_purpose_paths()
    target = project_path if scope == "project" else global_path

    if target.exists() and not force:
        click.echo(f"Error: {target} already exists. Use --force to overwrite.", err=True)
        ctx.exit(1)
        return

    scope_name = name or scope
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(purpose_template(scope_name), encoding="utf-8")
    except OSError as exc:
        click.echo(f"Error writing {target}: {exc}", err=True)
        ctx.exit(1)
        return

    click.echo(f"Wrote purpose template to {target}")


@purpose.command("check")
@click.argument("bullet_id")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def purpose_check(ctx: click.Context, bullet_id: str, as_json: bool) -> None:
    """Show how the current purpose would score a given bullet."""
    from memorus.core.purpose import (
        apply_purpose,
        load_pool_purpose,
        tokenize,
    )

    memory = _create_memory()
    if memory is None:
        ctx.exit(1)
        return

    # Best-effort bullet lookup: try common public APIs in order.
    bullet: Optional[dict[str, Any]] = None
    for method_name in ("get_bullet", "get_memory", "get"):
        method = getattr(memory, method_name, None)
        if callable(method):
            try:
                b = method(bullet_id)
            except Exception:  # pragma: no cover
                b = None
            if b:
                if isinstance(b, dict):
                    bullet = b
                else:
                    bullet = {
                        "content": getattr(b, "content", ""),
                        "embedding": getattr(b, "embedding", None),
                    }
                break

    if bullet is None:
        click.echo(f"Error: Bullet {bullet_id!r} not found.", err=True)
        ctx.exit(1)
        return

    purpose = load_pool_purpose()
    content = str(bullet.get("content", ""))
    base_score = float(bullet.get("instructivity_score", 50.0))
    adjusted = apply_purpose(base_score, content, purpose)

    content_tokens = tokenize(content)
    keyword_hits = [
        kw for kw in purpose.keywords if tokenize(kw).issubset(content_tokens)
    ]
    exclusion_hits = [
        ex for ex in purpose.excluded_topics if tokenize(ex).issubset(content_tokens)
    ]

    if as_json:
        click.echo(
            json.dumps(
                {
                    "bullet_id": bullet_id,
                    "base_score": base_score,
                    "adjusted_score": adjusted,
                    "keyword_hits": keyword_hits,
                    "exclusion_hits": exclusion_hits,
                    "purpose_empty": purpose.is_empty(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    click.echo(f"Bullet: {bullet_id}")
    click.echo(f"Purpose source:  {purpose.source_path or '(none)'}")
    click.echo(f"Base score:      {base_score:.2f}")
    click.echo(f"Adjusted score:  {adjusted:.2f}")
    click.echo(f"Keyword hits:    {', '.join(keyword_hits) or '(none)'}")
    click.echo(f"Exclusion hits:  {', '.join(exclusion_hits) or '(none)'}")
