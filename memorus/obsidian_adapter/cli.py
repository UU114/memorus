"""Command-line entry: ``python -m memorus.obsidian_adapter.cli``.

Subcommands:
  export   - render team + personal bullets to <vault>/memorus/{team,personal}/
  watch    - poll <vault>/memorus/inbox/ and submit drafts
  nominate - submit a single .md file (one-shot of watch)
  status   - print state from sidecar files

Each subcommand wires existing memorus components; no memorus core code is
modified by this CLI.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("memorus.obsidian_adapter")


def _build_team_components(team_id: str | None, server_url: str | None, token: str | None):
    """Return (TeamCacheStorage|None, AceSyncClient|None, Redactor|None)."""
    cache = sync = redactor = None
    try:
        from memorus.team.config import RedactorConfig, TeamConfig
        from memorus.team.cache_storage import TeamCacheStorage
        from memorus.team.redactor import Redactor
        tc = TeamConfig(team_id=team_id) if team_id else TeamConfig()
        cache = TeamCacheStorage(tc)
        redactor = Redactor(RedactorConfig())
    except Exception as e:
        logger.warning("team cache/redactor unavailable: %s", e)
    if server_url and token:
        try:
            from memorus.team.sync_client import AceSyncClient
            sync = AceSyncClient(server_url, token, team_id=team_id)
        except Exception as e:
            logger.warning("AceSyncClient unavailable: %s", e)
    return cache, sync, redactor


def _build_personal():
    try:
        from memorus import Memory
        return Memory()
    except Exception as e:
        logger.warning("personal Memory unavailable: %s", e)
        return None


def cmd_export(args: argparse.Namespace) -> int:
    from memorus.obsidian_adapter.exporter import VaultExporter
    cache, _, _ = _build_team_components(args.team_id, args.server_url, args.token)
    personal = _build_personal() if args.with_personal else None
    exp = VaultExporter(
        args.vault,
        team_cache=cache,
        personal_memory=personal,
        personal_query=args.query or "",
        personal_limit=args.limit,
    )
    summary = exp.export()
    print(f"team={summary.team_written} personal={summary.personal_written}")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    from memorus.obsidian_adapter.watcher import InboxWatcher
    _, sync, redactor = _build_team_components(args.team_id, args.server_url, args.token)
    personal = _build_personal() if args.with_personal else None
    w = InboxWatcher(
        args.vault,
        sync_client=sync,
        personal_memory=personal,
        redactor=redactor,
        author_id=args.author_id or "",
    )
    if args.once:
        for r in w.run_once():
            print(f"{r.file.name}: {r.kind}/{r.state} {r.detail}")
        return 0
    w.run_forever(interval=args.interval)
    return 0


def cmd_nominate(args: argparse.Namespace) -> int:
    from memorus.obsidian_adapter.watcher import InboxWatcher
    _, sync, redactor = _build_team_components(args.team_id, args.server_url, args.token)
    personal = _build_personal() if args.with_personal else None
    w = InboxWatcher(
        Path(args.file).parent.parent.parent,  # …/memorus/inbox/file.md -> root
        sync_client=sync,
        personal_memory=personal,
        redactor=redactor,
        author_id=args.author_id or "",
    )
    result = w._process_file(Path(args.file))  # noqa: SLF001 — single-file helper
    print(f"{result.kind}/{result.state} {result.detail}")
    return 0 if result.state == "submitted" else 2


def cmd_status(args: argparse.Namespace) -> int:
    root = Path(args.vault) / "memorus" / "inbox"
    if not root.exists():
        print("no inbox")
        return 0
    sidecars = sorted(root.glob("*.status.md"))
    for s in sidecars:
        print(f"--- {s.name} ---")
        print(s.read_text(encoding="utf-8").strip())
    return 0


def _common_team_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--vault", required=True, help="Obsidian vault root directory")
    p.add_argument("--team-id", default=None)
    p.add_argument("--server-url", default=None)
    p.add_argument("--token", default=None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="memorus-obs")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    exp = sub.add_parser("export")
    _common_team_args(exp)
    exp.add_argument("--with-personal", action="store_true")
    exp.add_argument("--query", default="")
    exp.add_argument("--limit", type=int, default=500)
    exp.set_defaults(func=cmd_export)

    wat = sub.add_parser("watch")
    _common_team_args(wat)
    wat.add_argument("--with-personal", action="store_true")
    wat.add_argument("--author-id", default="")
    wat.add_argument("--interval", type=float, default=2.0)
    wat.add_argument("--once", action="store_true")
    wat.set_defaults(func=cmd_watch)

    nom = sub.add_parser("nominate")
    nom.add_argument("--file", required=True)
    nom.add_argument("--team-id", default=None)
    nom.add_argument("--server-url", default=None)
    nom.add_argument("--token", default=None)
    nom.add_argument("--with-personal", action="store_true")
    nom.add_argument("--author-id", default="")
    nom.set_defaults(func=cmd_nominate)

    sta = sub.add_parser("status")
    sta.add_argument("--vault", required=True)
    sta.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
