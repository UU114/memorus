"""MemX MCP Server — expose Memory as MCP tools for IDE integration.

Requires: pip install memx[mcp]
Run:      memx-mcp  (stdio transport)
"""

from __future__ import annotations

import asyncio
import logging
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
        from memx.core.memory import Memory

        _memory_singleton = Memory(config=config)
    return _memory_singleton


def create_mcp_server(config: Optional[dict[str, Any]] = None) -> FastMCP:
    """Create and return a configured MCP server with MemX tools.

    Raises:
        ImportError: If the ``mcp`` package is not installed.
    """
    if FastMCP is None:
        raise ImportError(
            "MCP server requires the 'mcp' package. "
            "Install it with: pip install memx[mcp]"
        )

    mcp = FastMCP("memx", description="MemX Memory Server")

    @mcp.tool()
    async def search_memory(
        query: str,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Search memories by semantic similarity."""
        mem = _get_memory(config)
        return await asyncio.to_thread(
            mem.search, query, user_id=user_id, limit=limit
        )

    @mcp.tool()
    async def add_memory(
        content: str,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Add a new memory entry."""
        mem = _get_memory(config)
        return await asyncio.to_thread(mem.add, content, user_id=user_id)

    @mcp.tool()
    async def list_memories(
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """List all memories, optionally filtered by user."""
        mem = _get_memory(config)
        kwargs: dict[str, Any] = {}
        if user_id is not None:
            kwargs["user_id"] = user_id
        if limit != 100:
            kwargs["limit"] = limit
        return await asyncio.to_thread(mem.get_all, **kwargs)

    @mcp.tool()
    async def forget_memory(memory_id: str) -> dict[str, str]:
        """Delete a memory by ID."""
        mem = _get_memory(config)
        await asyncio.to_thread(mem.delete, memory_id)
        return {"status": "deleted", "memory_id": memory_id}

    @mcp.tool()
    async def memory_status(user_id: Optional[str] = None) -> dict[str, Any]:
        """Get memory status and statistics."""
        mem = _get_memory(config)
        return await asyncio.to_thread(mem.status, user_id=user_id)

    return mcp


def main() -> None:
    """Entry point for ``memx-mcp`` console script."""
    server = create_mcp_server()
    server.run()


if __name__ == "__main__":
    main()
