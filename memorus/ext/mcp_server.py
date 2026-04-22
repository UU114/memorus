"""Memorus MCP Server — expose Memory as MCP tools for IDE integration.

Requires: pip install memorus[mcp]
Run:      memorus-mcp  (stdio transport)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    FastMCP = None  # type: ignore[assignment,misc]

_memory_singleton: Any = None


def _get_memory(config: Optional[dict[str, Any]] = None) -> Any:
    """Lazily initialize a Memory singleton."""
    global _memory_singleton
    if _memory_singleton is None:
        from memorus.core.memory import Memory

        _memory_singleton = Memory(config=config)
    return _memory_singleton


def create_mcp_server(config: Optional[dict[str, Any]] = None) -> FastMCP:
    """Create and return a configured MCP server with Memorus tools.

    Raises:
        ImportError: If the ``mcp`` package is not installed.
    """
    if FastMCP is None:
        raise ImportError(
            "MCP server requires the 'mcp' package. "
            "Install it with: pip install memorus[mcp]"
        )

    mcp = FastMCP("memorus", description="Memorus Memory Server")

    @mcp.tool()
    async def search_memory(
        query: str,
        user_id: Optional[str] = None,
        limit: int = 100,
        scope: Optional[str] = None,
    ) -> dict[str, Any]:
        """Search the knowledge base for relevant memories. Use BEFORE answering to check known context."""
        mem = _get_memory(config)
        return await asyncio.to_thread(
            mem.search, query, user_id=user_id, limit=limit, scope=scope
        )

    @mcp.tool()
    async def add_memory(
        content: str,
        user_id: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> dict[str, Any]:
        """Store a learned fact. Content is auto-deduplicated and classified."""
        mem = _get_memory(config)
        return await asyncio.to_thread(mem.add, content, user_id=user_id, scope=scope)

    @mcp.tool()
    async def list_memories(
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """List all stored memories, optionally filtered by user. Use to browse the knowledge base."""
        mem = _get_memory(config)
        kwargs: dict[str, Any] = {}
        if user_id is not None:
            kwargs["user_id"] = user_id
        if limit != 100:
            kwargs["limit"] = limit
        return await asyncio.to_thread(mem.get_all, **kwargs)

    @mcp.tool()
    async def forget_memory(memory_id: str) -> dict[str, str]:
        """Permanently delete a memory by its ID."""
        mem = _get_memory(config)
        await asyncio.to_thread(mem.delete, memory_id)
        return {"status": "deleted", "memory_id": memory_id}

    @mcp.tool()
    async def memory_status(user_id: Optional[str] = None) -> dict[str, Any]:
        """Get knowledge base statistics: total count, section distribution, decay health."""
        mem = _get_memory(config)
        return await asyncio.to_thread(mem.status, user_id=user_id)

    @mcp.tool()
    async def detect_conflicts(user_id: Optional[str] = None) -> list[dict[str, Any]]:
        """Detect contradictory or conflicting memories in the knowledge base."""
        mem = _get_memory(config)
        conflicts = await asyncio.to_thread(mem.detect_conflicts, user_id=user_id)
        return [c.model_dump() if hasattr(c, "model_dump") else str(c) for c in conflicts]

    @mcp.tool()
    async def run_decay_sweep(
        user_id: Optional[str] = None,
        archive: bool = True,
    ) -> dict[str, Any]:
        """Run temporal decay to age memories and archive stale ones. Usually runs automatically."""
        mem = _get_memory(config)
        return await asyncio.to_thread(
            mem.run_decay_sweep, user_id=user_id, archive=archive
        )

    @mcp.tool()
    async def export_memories(
        filepath: str,
        format: str = "json",
        scope: Optional[str] = None,
    ) -> dict[str, Any]:
        """Export all memories as JSON or Markdown for backup or migration."""
        import json as json_mod

        mem = _get_memory(config)
        result = await asyncio.to_thread(mem.export, format=format, scope=scope)
        path = os.path.expanduser(filepath)
        if format == "json":
            with open(path, "w", encoding="utf-8") as f:
                json_mod.dump(result, f, ensure_ascii=False, indent=2)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(result)
        total = result.get("total", 0) if isinstance(result, dict) else "unknown"
        return {"status": "exported", "filepath": path, "total": total, "format": format}

    @mcp.tool()
    async def import_memories(
        filepath: str,
        format: str = "json",
    ) -> dict[str, Any]:
        """Import memories from a JSON payload. Auto-deduplicates against existing knowledge."""
        import json as json_mod

        path = os.path.expanduser(filepath)
        with open(path, "r", encoding="utf-8") as f:
            data = json_mod.load(f)
        mem = _get_memory(config)
        return await asyncio.to_thread(mem.import_data, data, format=format)

    return mcp


def main() -> None:
    """Entry point for ``memorus-mcp`` console script."""
    import argparse
    import json as json_mod
    import sys

    parser = argparse.ArgumentParser(description="Memorus MCP Server")
    parser.add_argument(
        "--config", default=None, help="Path to Memorus config JSON file"
    )
    args = parser.parse_args()

    config = None
    if args.config:
        config_path = os.path.expanduser(args.config)
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json_mod.load(f)
        except FileNotFoundError:
            print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        except json_mod.JSONDecodeError as exc:
            print(
                f"ERROR: Invalid JSON in config file {config_path}: "
                f"line {exc.lineno}, column {exc.colno}: {exc.msg}",
                file=sys.stderr,
            )
            sys.exit(1)

    server = create_mcp_server(config)
    server.run()


if __name__ == "__main__":
    main()
