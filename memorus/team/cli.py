"""Team CLI commands — team status, sync, and nomination management."""

from __future__ import annotations

import json
import sys
from typing import Any

import click


def _ensure_team_enabled():
    """Check if team is configured. Returns (team_config, error_msg)."""
    try:
        from memorus.team.config import load_team_config

        config = load_team_config()
        if not config.enabled:
            return None, (
                "Team features not enabled. "
                "Set 'enabled: true' in team_config.yaml or MEMORUS_TEAM_ENABLED=true"
            )
        return config, None
    except Exception as e:
        return None, f"Failed to load team config: {e}"


# ---------------------------------------------------------------------------
# team group
# ---------------------------------------------------------------------------


@click.group("team")
def team_group():
    """Team memory management commands."""
    pass


@team_group.command("status")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def team_status(as_json: bool) -> None:
    """Show team memory status overview."""
    config, err = _ensure_team_enabled()
    if err:
        if as_json:
            click.echo(json.dumps({"error": err}))
        else:
            click.echo(f"Error: {err}", err=True)
        sys.exit(1)

    # Determine mode
    mode = "federation" if config.server_url else "git-fallback"

    info: dict[str, Any] = {
        "mode": mode,
        "team_id": config.team_id or "default",
        "server_url": config.server_url or "N/A",
        "cache_max_bullets": config.cache_max_bullets,
        "cache_ttl_minutes": config.cache_ttl_minutes,
        "subscribed_tags": config.subscribed_tags,
    }

    # Try to get cache stats
    try:
        from memorus.team.cache_storage import TeamCacheStorage

        cache = TeamCacheStorage(config)
        info["cached_bullets"] = cache.bullet_count
        info["last_sync"] = (
            cache.last_sync_time.isoformat() if cache.last_sync_time else "never"
        )
    except Exception:
        info["cached_bullets"] = "N/A"
        info["last_sync"] = "N/A"

    if as_json:
        click.echo(json.dumps(info, indent=2, default=str))
    else:
        click.echo(f"Mode: {info['mode']}")
        click.echo(f"Team ID: {info['team_id']}")
        click.echo(f"Server: {info['server_url']}")
        click.echo(
            f"Cached Bullets: {info['cached_bullets']} / {info['cache_max_bullets']}"
        )
        click.echo(f"Last Sync: {info['last_sync']}")
        tags = (
            ", ".join(f"#{t}" for t in info["subscribed_tags"])
            if info["subscribed_tags"]
            else "all"
        )
        click.echo(f"Subscribed Tags: {tags}")


@team_group.command("sync")
@click.option("--full", is_flag=True, help="Force full sync (not incremental)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def team_sync(full: bool, as_json: bool) -> None:
    """Sync team knowledge from server."""
    config, err = _ensure_team_enabled()
    if err:
        if as_json:
            click.echo(json.dumps({"error": err}))
        else:
            click.echo(f"Error: {err}", err=True)
        sys.exit(1)

    if not config.server_url:
        msg = "No server_url configured. Sync requires Federation Mode."
        if as_json:
            click.echo(json.dumps({"error": msg}))
        else:
            click.echo(f"Error: {msg}", err=True)
        sys.exit(1)

    try:
        from memorus.team.cache_storage import TeamCacheStorage
        from memorus.team.sync_client import AceSyncClient
        from memorus.team.sync_manager import SyncManager

        cache = TeamCacheStorage(config)
        client = AceSyncClient(
            server_url=config.server_url,
            auth_token="",  # Would come from config in real usage
            team_id=config.team_id,
        )
        manager = SyncManager(cache, client, config)

        if full:
            # Reset sync timestamp to force full pull
            manager._last_sync_timestamp = None

        if not as_json:
            click.echo("Syncing...")

        manager.sync_now()

        result = {
            "status": manager.last_sync_status,
            "cached_bullets": cache.bullet_count,
            "sync_count": manager.sync_count,
        }

        if as_json:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(
                f"Sync {result['status']}. "
                f"Cache: {result['cached_bullets']} / {config.cache_max_bullets}"
            )
    except Exception as e:
        if as_json:
            click.echo(json.dumps({"error": str(e)}))
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# nominate group
# ---------------------------------------------------------------------------


@click.group("nominate")
def nominate_group():
    """Knowledge nomination commands."""
    pass


@nominate_group.command("list")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def nominate_list(as_json: bool) -> None:
    """List pending nomination candidates."""
    config, err = _ensure_team_enabled()
    if err:
        if as_json:
            click.echo(json.dumps({"error": err}))
        else:
            click.echo(f"Error: {err}", err=True)
        sys.exit(1)

    try:
        from memorus.team.nominator import Nominator

        nominator = Nominator(config.auto_nominate)
        pending = nominator.get_pending_nominations()

        if as_json:
            click.echo(json.dumps({"candidates": pending}, indent=2, default=str))
            return

        if not pending:
            click.echo("No candidates found.")
            return

        # Table header
        click.echo(f"{'ID':<20} | {'Score':>5} | Content Preview")
        click.echo("-" * 60)
        for b in pending:
            bid = b.get("id", b.get("origin_id", "?"))[:20]
            score = b.get("instructivity_score", 0.0)
            content = b.get("content", "")[:50]
            click.echo(f"{bid:<20} | {score:>5.1f} | \"{content}...\"")
    except Exception as e:
        if as_json:
            click.echo(json.dumps({"error": str(e)}))
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@nominate_group.command("submit")
@click.argument("bullet_id")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def nominate_submit(bullet_id: str, as_json: bool) -> None:
    """Submit a bullet for team nomination."""
    config, err = _ensure_team_enabled()
    if err:
        if as_json:
            click.echo(json.dumps({"error": err}))
        else:
            click.echo(f"Error: {err}", err=True)
        sys.exit(1)

    # Placeholder — actual integration requires Memory instance and Redactor
    if as_json:
        click.echo(
            json.dumps({"error": "Bullet not found", "bullet_id": bullet_id})
        )
    else:
        click.echo(
            f"Error: Bullet '{bullet_id}' not found in local pool.", err=True
        )
    sys.exit(1)


# ---------------------------------------------------------------------------
# vote commands (STORY-069)
# ---------------------------------------------------------------------------


def _vote_command(bullet_id: str, *, upvote: bool, as_json: bool) -> None:
    """Shared logic for upvote/downvote commands."""
    config, err = _ensure_team_enabled()
    if err:
        if as_json:
            click.echo(json.dumps({"error": err}))
        else:
            click.echo(f"Error: {err}", err=True)
        sys.exit(1)

    try:
        from memorus.team.cache_storage import TeamCacheStorage

        cache = TeamCacheStorage(config)
        result = cache.vote_bullet(bullet_id, upvote=upvote)

        if result is None:
            msg = f"Bullet '{bullet_id}' not found in cache."
            if as_json:
                click.echo(json.dumps({"error": msg}))
            else:
                click.echo(f"Error: {msg}", err=True)
            sys.exit(1)

        action = "upvote" if upvote else "downvote"
        info = {
            "action": action,
            "bullet_id": bullet_id,
            "effective_score": result.effective_score,
            "upvotes": result.upvotes,
            "downvotes": result.downvotes,
        }

        if as_json:
            click.echo(json.dumps(info, indent=2))
        else:
            click.echo(
                f"Recorded {action} on '{bullet_id}'. "
                f"Score: {result.effective_score:.1f} "
                f"(+{result.upvotes}/-{result.downvotes})"
            )
    except Exception as e:
        if as_json:
            click.echo(json.dumps({"error": str(e)}))
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@team_group.command("upvote")
@click.argument("bullet_id")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def team_upvote(bullet_id: str, as_json: bool) -> None:
    """Upvote a team bullet (+5 effective score)."""
    _vote_command(bullet_id, upvote=True, as_json=as_json)


@team_group.command("downvote")
@click.argument("bullet_id")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def team_downvote(bullet_id: str, as_json: bool) -> None:
    """Downvote a team bullet (-10 effective score)."""
    _vote_command(bullet_id, upvote=False, as_json=as_json)


@team_group.command("list")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--limit", type=int, default=50, show_default=True, help="Max rows to show")
def team_list(as_json: bool, limit: int) -> None:
    """List team bullets with status and last-verified column (STORY-R104)."""
    config, err = _ensure_team_enabled()
    if err:
        if as_json:
            click.echo(json.dumps({"error": err}))
        else:
            click.echo(f"Error: {err}", err=True)
        sys.exit(1)

    try:
        from memorus.team.cache_storage import TeamCacheStorage

        cache = TeamCacheStorage(config)
        bullets = list(cache._bullets.values())[:limit]

        rows: list[dict[str, Any]] = []
        for b in bullets:
            bid = b.origin_id or ""
            rows.append(
                {
                    "id": bid,
                    "title": (b.content or "")[:60],
                    "status": b.status,
                    "verified": (b.last_verified_status or "-"),
                }
            )

        if as_json:
            click.echo(json.dumps({"bullets": rows}, indent=2, default=str))
            return

        if not rows:
            click.echo("No team bullets cached.")
            return

        # Plain table output — fixed-width columns to keep snapshot tests
        # easy to assert against.
        click.echo(
            f"{'ID':<36}  {'Title':<60}  {'Status':<10}  {'Verified':<10}"
        )
        click.echo("-" * 124)
        for r in rows:
            click.echo(
                f"{r['id']:<36}  {r['title']:<60}  {r['status']:<10}  {r['verified']:<10}"
            )
    except Exception as e:
        if as_json:
            click.echo(json.dumps({"error": str(e)}))
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@team_group.command("backlog")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def team_backlog(as_json: bool) -> None:
    """Check staging backlog status and alerts."""
    config, err = _ensure_team_enabled()
    if err:
        if as_json:
            click.echo(json.dumps({"error": err}))
        else:
            click.echo(f"Error: {err}", err=True)
        sys.exit(1)

    try:
        from memorus.team.cache_storage import TeamCacheStorage

        cache = TeamCacheStorage(config)
        info = cache.check_backlog()

        if as_json:
            click.echo(json.dumps(info, indent=2))
        else:
            click.echo(f"Staging bullets: {info['staging_count']}")
            click.echo(f"Oldest pending: {info['oldest_pending_days']} days")
            if info["needs_attention"]:
                click.echo("WARNING: Backlog needs attention!")
            else:
                click.echo("Backlog OK.")
    except Exception as e:
        if as_json:
            click.echo(json.dumps({"error": str(e)}))
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)
